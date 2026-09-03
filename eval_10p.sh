#!/bin/bash
#SBATCH --job-name=hcs_eval_10p
#SBATCH --account=soney0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1500M
#SBATCH --time=01:30:00
#SBATCH --array=1-8
#SBATCH --output=logs/eval_10p_%A_%a.out
#SBATCH --error=logs/eval_10p_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xiangyz@umich.edu

# Per-participant evaluation of the 10p anchor-drive fits (after
# fit_anchor_10p.sh): (1) eval-main with the fitted persona — all eight
# participants share ONE eval-main folder; (2) model gaze-lead PDFs (model
# sawtooth vs human rounds, per trial). Then aggregate:
#   EVAL_ID=$(sbatch --parsable eval_10p.sh)
#   sbatch --dependency=afterok:$EVAL_ID eval_10p_aggregate.sh
# Use the SAME RUN_TAG as the fit generation.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load python3.11-anaconda/2024.02 2>/dev/null || true
source venv/bin/activate 2>/dev/null || source .venv/bin/activate 2>/dev/null || true
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg TMPDIR=/tmp

RUN_TAG="${RUN_TAG:-}"
RESULTS_ROOT="${RESULTS_ROOT:-/home/xiangyz/ondemand/data/sys/myjobs/projects/chi-27/results}"
FIT_DIR="$RESULTS_ROOT/anchor_fitting_10p${RUN_TAG:+-$RUN_TAG}/stages/base"
export HCS_EVAL_RESULTS_DIR="$RESULTS_ROOT/eval-main-10p${RUN_TAG:+-$RUN_TAG}"
PERSONA_DIR="$RESULTS_ROOT/personas_10p${RUN_TAG:+-$RUN_TAG}"
GAZE_LEAD_DIR="$RESULTS_ROOT/gaze-lead-10p${RUN_TAG:+-$RUN_TAG}"
SEED="${SEED:-42}"
DATA_DIR="human_data/task_aligned_all"
mkdir -p "$HCS_EVAL_RESULTS_DIR" "$PERSONA_DIR" "$GAZE_LEAD_DIR" logs

SHORT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" participants_10p.txt)          # p01 ...
# run_eval keys participants by the embedded Prolific id in the session file
SESS=$(ls "$DATA_DIR/${SHORT}"_*_P*_*.json | head -1); SESS=$(basename "$SESS")
TMP="${SESS#*_P}"; PID="P${TMP%%_*}"                                     # P103405 ...
CFG="$FIT_DIR/${SHORT}_anchor_config_s${SEED}.json"
[ -f "$CFG" ] || { echo "missing fitted persona $CFG"; exit 1; }
# Stage the persona under the P-id so run_eval's --config-dir resolution
# ({pid}.json) finds it; idempotent across array tasks. Guard against
# personas saved by pre-fix fit_anchor.py with the fitting harness's
# noiseless settings baked in (add_noise false, latency cv 0).
python3 - "$CFG" "$PERSONA_DIR/${PID}.json" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))
c["add_noise"] = True
if not float(c.get("replan_latency_cv", 0) or 0):
    c["replan_latency_cv"] = 0.89
json.dump(c, open(sys.argv[2], "w"), indent=2)
EOF
MIN_RUNS="${MIN_RUNS:-0}"   # >0: extra noise realisations per condition

echo "[$(date)] eval $SHORT ($PID) persona=$CFG -> $HCS_EVAL_RESULTS_DIR"

# 1) eval-main with the fitted persona (shared results folder for all six)
python -u eval/eval-main/run_eval.py \
    --pid "$PID" \
    --config-dir "$PERSONA_DIR" \
    --seed "$SEED" \
    --fresh-sim \
    --min-runs "$MIN_RUNS" \
    --data-dir "$DATA_DIR" \
    2>&1 | tee "$HCS_EVAL_RESULTS_DIR/eval_${SHORT}_s${SEED}.log"

# 2) model gaze-lead PDFs: model sawtooth (fitted persona, its own budget,
#    noise forced on) vs human rounds, one page per trial + summary page
python -u eval/eval-gaze-lead/model_gaze_lead.py \
    --letters "$SHORT" \
    --config "$PERSONA_DIR/${PID}.json" \
    --noise on \
    --out-dir "$GAZE_LEAD_DIR/$SHORT" \
    2>&1 | tee "$GAZE_LEAD_DIR/model_gaze_lead_${SHORT}.log"

# 3) human-gaze-lead-10p-style PNGs with the model overlaid: per-task plots
#    (individual/) plus lead_by_width/ and lead_by_curvature/ grids, one
#    noisy model run per round column (human series from the committed
#    human-gaze-lead-10p/data CSVs — no gaze recomputation here)
python -u eval/eval-gaze-lead/gaze_lead_grids.py \
    --letters "$SHORT" \
    --config "$PERSONA_DIR/${PID}.json" \
    --noise on --runs 3 \
    --out-dir "$GAZE_LEAD_DIR/$SHORT" \
    2>&1 | tee "$GAZE_LEAD_DIR/gaze_lead_grids_${SHORT}.log"

echo "[$(date)] done $SHORT"

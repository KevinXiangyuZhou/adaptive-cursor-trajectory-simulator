#!/bin/bash
#SBATCH --account=soney0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1500M
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xiangyz@umich.edu

# eval-14p: one array task per NEW participant (eval-14p/participants_14p.txt),
# evaluating ONE fixed persona (RUN_DIR/personas/default.json — fitted on the
# earlier 8-participant cohort, nothing is fitted here) with the same harness
# as the main runs, eval/eval-main/run_eval.py. Never sbatch directly:
# eval-14p/submit_eval_14p.sh chains it before the aggregate and exports
# RUN_DIR / MODEL / SOURCE_PERSONA / SEED / MIN_RUNS / BUCKETS /
# PARTICIPANTS_FILE / DATA_DIR / VENV_DIR. MODEL=baseline drives the CHI-26-EA
# package (eval/chi-26-ea_baseline_pacakage) with the pooled baseline persona.
#
# The 14p data has only steering tunnels (25 conditions: 5 tunnel types x
# widths 10/20/30/40/50 mm), 3 wide-to-narrow conditions and lasso / menu
# tasks; the harness excludes lasso and menu itself, and BUCKETS=steering
# (default) leaves wide-to-narrow out as in the 8-participant evaluation.
# One participant is ~115 tunnel runs (20 conditions x 5 rounds + 5 x 3):
# ~5 min on 4 cores (2026-09-11 run), no pointing, no gaze-lead figures (no
# gaze data). The per-condition worker pool follows the allocation
# (run_eval.n_pool_workers: 4 here, not the node's 36). The baseline replans
# every step: ~58 core-min per participant (P204813, laptop), i.e. ~30 min
# on 4 cluster cores — submit it with --wall 02:00:00. Wall 1 h otherwise;
# pass submit --wall for --min-runs well above 5.

set -euo pipefail
: "${RUN_DIR:?set by submit_eval_14p.sh}" "${SEED:?}"
MODEL="${MODEL:-mpcc}"
cd "$RUN_DIR/code"
module load python3.11-anaconda/2024.02 2>/dev/null || true
source "${VENV_DIR:-$RUN_DIR/code/venv}/bin/activate"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
export TMPDIR="$RUN_DIR/tmp/job_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT

export HCS_EVAL_RESULTS_DIR="$RUN_DIR/eval"
PARTICIPANTS_FILE="${PARTICIPANTS_FILE:-eval-14p/participants_14p.txt}"
DATA_DIR="${DATA_DIR:-eval-14p/human_data/raw}"
MIN_RUNS="${MIN_RUNS:-0}"        # >0: extra noise realisations per condition
BUCKETS="${BUCKETS:-steering}"   # space-separated run_eval --buckets (add id4scs_w2n for wide-to-narrow)
# CHI-26 protocol: a trial ends within a FIXED 10 mm of the tunnel end at
# every width (the 8p web experiment ends within W/2, the harness default).
# The model must stop by the same rule or its completion times are biased.
TUNNEL_TARGET_RADIUS="${TUNNEL_TARGET_RADIUS:-0.01}"
PERSONA="$RUN_DIR/personas/default.json"
mkdir -p "$HCS_EVAL_RESULTS_DIR"

PID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PARTICIPANTS_FILE" | tr -d '[:space:]')
[ -n "$PID" ] || { echo "no participant on line ${SLURM_ARRAY_TASK_ID} of $PARTICIPANTS_FILE"; exit 1; }
if [ ! -f "$PERSONA" ]; then
    # submitted while the source fit was still running (afterok on its fit
    # job): stage the persona now. Every array task does this; the content
    # is identical, the temp + mv keeps the file whole for the others.
    : "${SOURCE_PERSONA:?no staged persona and no SOURCE_PERSONA to stage}"
    [ -f "$SOURCE_PERSONA" ] || { echo "source persona $SOURCE_PERSONA missing after the fit job"; exit 1; }
    mkdir -p "$RUN_DIR/personas"
    python3 cluster/stage_persona.py "$SOURCE_PERSONA" "$PERSONA.$$" "$MODEL" pooled8 && mv -f "$PERSONA.$$" "$PERSONA"
    cp "$SOURCE_PERSONA" "$RUN_DIR/personas/source_persona.json"
fi
[ -d "$DATA_DIR" ] || { echo "missing data dir $DATA_DIR"; exit 1; }

echo "[$(date)] eval-14p $MODEL $PID persona=$PERSONA data=$DATA_DIR buckets=[$BUCKETS] goal_radius=$TUNNEL_TARGET_RADIUS min_runs=$MIN_RUNS -> $HCS_EVAL_RESULTS_DIR (code $(cat "$RUN_DIR/COMMIT" 2>/dev/null || echo ?))"
# shellcheck disable=SC2086
python -u eval/eval-main/run_eval.py \
    --model "$MODEL" \
    --pid "$PID" \
    --config-dir "$RUN_DIR/personas" \
    --seed "$SEED" \
    --fresh-sim \
    --min-runs "$MIN_RUNS" \
    --data-dir "$DATA_DIR" \
    --buckets $BUCKETS \
    --tunnel-target-radius "$TUNNEL_TARGET_RADIUS" \
    2>&1 | tee "$HCS_EVAL_RESULTS_DIR/eval_${PID}_s${SEED}.log"

# run_eval exits 0 with "No human data found." when --pid matches nothing —
# make that a failure so the aggregate does not run on a missing participant
[ -f "$HCS_EVAL_RESULTS_DIR/sim_cache/${PID}_sim_cache.json" ] \
    || { echo "no simulator output for $PID (pid not found in $DATA_DIR?)"; exit 1; }
echo "[$(date)] done $PID"

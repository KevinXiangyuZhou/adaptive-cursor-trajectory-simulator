#!/bin/bash
#SBATCH --account=soney0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1500M
#SBATCH --time=01:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xiangyz@umich.edu

# Per-participant evaluation (one array task per participant) of one run,
# after its fit: (1) stage the fitted persona under the Prolific id;
# (2) eval-main with --model; (3) model gaze-lead figures (mpcc only — the
# baseline has no anchors). Never sbatch directly: submit_run.sh chains it
# after the fit with --dependency and exports RUN_DIR / MODEL / VARIANT /
# KIND / SEED / PARTICIPANTS_FILE / VENV_DIR / MIN_RUNS.

set -euo pipefail
: "${RUN_DIR:?set by submit_run.sh}" "${MODEL:?}" "${KIND:?}" "${SEED:?}"
cd "$RUN_DIR/code"
module load python3.11-anaconda/2024.02 2>/dev/null || true
source "${VENV_DIR:-$RUN_DIR/code/venv}/bin/activate"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
export TMPDIR="$RUN_DIR/tmp/job_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT

export HCS_EVAL_RESULTS_DIR="$RUN_DIR/eval"
PERSONA_DIR="$RUN_DIR/personas"
GAZE_LEAD_DIR="$RUN_DIR/gaze-lead"
DATA_DIR="human_data/task_aligned_all"
PARTICIPANTS_FILE="${PARTICIPANTS_FILE:-participants_10p.txt}"
MIN_RUNS="${MIN_RUNS:-0}"   # >0: extra noise realisations per condition
TAGM=$([ "$MODEL" = "mpcc" ] && echo anchor || echo baseline)
mkdir -p "$HCS_EVAL_RESULTS_DIR" "$PERSONA_DIR" "$GAZE_LEAD_DIR"

SHORT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PARTICIPANTS_FILE")          # p01 ...
# run_eval keys participants by the embedded Prolific id in the session file
SESS=$(ls "$DATA_DIR/${SHORT}"_*_P*_*.json | head -1); SESS=$(basename "$SESS")
TMP="${SESS#*_P}"; PID="P${TMP%%_*}"                                     # P103405 ...
if [ "$KIND" = "pooled8" ]; then
    CFG="$RUN_DIR/fit/stages/pooled8/pooled8_${TAGM}_config_s${SEED}.json"
else
    CFG="$RUN_DIR/fit/stages/base/${SHORT}_${TAGM}_config_s${SEED}.json"
fi
[ -f "$CFG" ] || { echo "missing fitted persona $CFG"; exit 1; }
python3 cluster/stage_persona.py "$CFG" "$PERSONA_DIR/${PID}.json" "$MODEL" "$KIND"

# SKIP_EVAL=1 (resubmit_eval.sh --gaze-lead-only): leave the eval outputs as
# they are and only (re)generate the gaze-lead outputs below.
if [ "${SKIP_EVAL:-0}" != 1 ]; then
echo "[$(date)] eval $MODEL/$KIND $SHORT ($PID) persona=$CFG -> $HCS_EVAL_RESULTS_DIR"
python -u eval/eval-main/run_eval.py \
    --model "$MODEL" \
    --pid "$PID" \
    --config-dir "$PERSONA_DIR" \
    --seed "$SEED" \
    --fresh-sim \
    --min-runs "$MIN_RUNS" \
    --data-dir "$DATA_DIR" \
    2>&1 | tee "$HCS_EVAL_RESULTS_DIR/eval_${SHORT}_s${SEED}.log"
fi

# Gaze-lead outputs after the eval. GAZE_LEAD level (default by variant):
#   full   model_gaze_lead.py (planning-event CSV + per-trial PDF) AND
#          gaze_lead_grids.py (PNG grids; 3 extra noisy runs per steering
#          trial) — the full model only, whose figures the paper shows
#   events model_gaze_lead.py only — the ablations: the event CSV is what
#          the cycle measures of the ablation table are computed from; the
#          figures are not needed and every-step variants make them slow
#   0      nothing — the baseline has no anchors
# These are diagnostics on top of the eval, not part of it: a failure is
# logged and marked, but must not fail the eval task and thereby cancel the
# aggregate (2026-09-08: it did, for every mpcc run).
if [ -z "${GAZE_LEAD:-}" ] || [ "${GAZE_LEAD}" = 1 ]; then
    if [ "$MODEL" != "mpcc" ]; then GAZE_LEAD=0
    elif [ "${VARIANT:-full}" = "full" ]; then GAZE_LEAD=full
    else GAZE_LEAD=events; fi
fi
if [ "$MODEL" = "mpcc" ] && [ "$GAZE_LEAD" != 0 ]; then
    set +e
    rm -f "$GAZE_LEAD_DIR/FAILED_model_gaze_lead_${SHORT}" "$GAZE_LEAD_DIR/FAILED_gaze_lead_grids_${SHORT}"
    python -u eval/eval-gaze-lead/model_gaze_lead.py \
        --letters "$SHORT" --config "$PERSONA_DIR/${PID}.json" --noise on \
        --out-dir "$GAZE_LEAD_DIR/$SHORT" \
        > "$GAZE_LEAD_DIR/model_gaze_lead_${SHORT}.log" 2>&1 \
        || { echo "model_gaze_lead FAILED for $SHORT (see $GAZE_LEAD_DIR/model_gaze_lead_${SHORT}.log)"; touch "$GAZE_LEAD_DIR/FAILED_model_gaze_lead_${SHORT}"; }
    if [ "$GAZE_LEAD" = full ]; then
        python -u eval/eval-gaze-lead/gaze_lead_grids.py \
            --letters "$SHORT" --config "$PERSONA_DIR/${PID}.json" --noise on --runs 3 \
            --out-dir "$GAZE_LEAD_DIR/$SHORT" \
            > "$GAZE_LEAD_DIR/gaze_lead_grids_${SHORT}.log" 2>&1 \
            || { echo "gaze_lead_grids FAILED for $SHORT (see $GAZE_LEAD_DIR/gaze_lead_grids_${SHORT}.log)"; touch "$GAZE_LEAD_DIR/FAILED_gaze_lead_grids_${SHORT}"; }
    fi
    set -e
fi
echo "[$(date)] done $SHORT"

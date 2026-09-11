#!/bin/bash
#SBATCH --account=soney0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xiangyz@umich.edu

# eval-14p aggregate: pooled Steering-Law summary across the 14 participants
# from the cached per-participant sims (no new simulation), then the
# generalisation table (eval-14p/summarize_14p.py: per tunnel type x width,
# widths shared with the fitting cohort vs new ones, per participant) and
# the DONE marker. Never sbatch directly: eval-14p/submit_eval_14p.sh chains
# it after the eval array.
#
# Sizing (2026-09-11): the first 14p aggregate ran over its 30 min on one
# core: --aggregate-only recomputes every (participant x condition) — ~4 s
# each in the point-by-segment Python loops of utils/stats.py's progress
# resampling (~23 core-min for 350 conditions) plus ~1.5 s of per-trial
# figures the eval tasks had already drawn, with os.cpu_count() (36)
# workers contending for the one core. Fixed on three fronts: the
# resampling is vectorised (bit-identical, ~150x), --no-trial-plots skips
# the redraw (the figures stay as the eval tasks made them), and the worker
# count follows the allocation (4 cores here). The whole pass is now ~1-2
# min. TRIAL_PLOTS=1 redraws the figures (~10 min on 4 cores).

set -euo pipefail
: "${RUN_DIR:?set by submit_eval_14p.sh}" "${SEED:?}"
MODEL="${MODEL:-mpcc}"
cd "$RUN_DIR/code"
module load python3.11-anaconda/2024.02 2>/dev/null || true
source "${VENV_DIR:-$RUN_DIR/code/venv}/bin/activate"
export MPLBACKEND=Agg
export TMPDIR="$RUN_DIR/tmp/job_${SLURM_JOB_ID:-local}_agg"
mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
export HCS_EVAL_RESULTS_DIR="$RUN_DIR/eval"
DATA_DIR="${DATA_DIR:-eval-14p/human_data/raw}"
BUCKETS="${BUCKETS:-steering}"
TUNNEL_TARGET_RADIUS="${TUNNEL_TARGET_RADIUS:-0.01}"   # CHI-26 fixed 10 mm goal, as in eval_job.sh

echo "[$(date)] aggregate $RUN_DIR"
# shellcheck disable=SC2086
python -u eval/eval-main/run_eval.py \
    --model "$MODEL" \
    --config-dir "$RUN_DIR/personas" \
    --seed "$SEED" \
    --aggregate-only \
    $([ "${TRIAL_PLOTS:-0}" = 1 ] || echo --no-trial-plots) \
    --data-dir "$DATA_DIR" \
    --buckets $BUCKETS \
    --tunnel-target-radius "$TUNNEL_TARGET_RADIUS" \
    2>&1 | tee "$HCS_EVAL_RESULTS_DIR/eval_aggregate_s${SEED}.log"

# generalisation summary; REFERENCE_CSV (submit --reference) = the source
# run's own Steering/steering_condition_summary.csv on the fitting cohort
REF_ARG=()
[ -n "${REFERENCE_CSV:-}" ] && [ -f "${REFERENCE_CSV}" ] && REF_ARG=(--reference "$REFERENCE_CSV")
python -u eval-14p/summarize_14p.py "$HCS_EVAL_RESULTS_DIR" --data-dir "$DATA_DIR" \
    --out "$HCS_EVAL_RESULTS_DIR/SUMMARY_14p" ${REF_ARG[@]+"${REF_ARG[@]}"} \
    2>&1 | tee "$HCS_EVAL_RESULTS_DIR/summary_14p.log"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/DONE"
echo "[$(date)] done"

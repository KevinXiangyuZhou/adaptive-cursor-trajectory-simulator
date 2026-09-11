#!/bin/bash
#SBATCH --account=soney0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xiangyz@umich.edu

# eval-14p aggregate: pooled Steering-Law summary across the 14 participants
# from the cached per-participant sims (no new simulation), then the
# generalisation table (eval-14p/summarize_14p.py: per tunnel type x width,
# widths shared with the fitting cohort vs new ones, per participant) and
# the DONE marker. Never sbatch directly: eval-14p/submit_eval_14p.sh chains
# it after the eval array.

set -euo pipefail
: "${RUN_DIR:?set by submit_eval_14p.sh}" "${SEED:?}"
cd "$RUN_DIR/code"
module load python3.11-anaconda/2024.02 2>/dev/null || true
source "${VENV_DIR:-$RUN_DIR/code/venv}/bin/activate"
export MPLBACKEND=Agg
export TMPDIR="$RUN_DIR/tmp/job_${SLURM_JOB_ID:-local}_agg"
mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
export HCS_EVAL_RESULTS_DIR="$RUN_DIR/eval"
DATA_DIR="${DATA_DIR:-eval-14p/human_data/raw}"
BUCKETS="${BUCKETS:-steering}"

echo "[$(date)] aggregate $RUN_DIR"
# shellcheck disable=SC2086
python -u eval/eval-main/run_eval.py \
    --model mpcc \
    --config-dir "$RUN_DIR/personas" \
    --seed "$SEED" \
    --aggregate-only \
    --data-dir "$DATA_DIR" \
    --buckets $BUCKETS \
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

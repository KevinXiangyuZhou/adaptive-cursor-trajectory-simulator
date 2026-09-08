#!/bin/bash
#SBATCH --account=soney0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xiangyz@umich.edu

# Aggregate pass over one run's eval folder (pooled Fitts/Steering/ID4SCS
# summaries + overview across participants) from the cached per-participant
# sims — no new simulation — then mark the run DONE. Never sbatch directly:
# submit_run.sh chains it after the eval array.

set -euo pipefail
: "${RUN_DIR:?set by submit_run.sh}" "${MODEL:?}" "${SEED:?}"
cd "$RUN_DIR/code"
module load python3.11-anaconda/2024.02 2>/dev/null || true
source "${VENV_DIR:-$RUN_DIR/code/venv}/bin/activate"
export MPLBACKEND=Agg
export TMPDIR="$RUN_DIR/tmp/job_${SLURM_JOB_ID:-local}_agg"
mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
export HCS_EVAL_RESULTS_DIR="$RUN_DIR/eval"

echo "[$(date)] aggregate $RUN_DIR"
python -u eval/eval-main/run_eval.py \
    --model "$MODEL" \
    --config-dir "$RUN_DIR/personas" \
    --seed "$SEED" \
    --aggregate-only \
    --data-dir human_data/task_aligned_all \
    2>&1 | tee "$HCS_EVAL_RESULTS_DIR/eval_aggregate_s${SEED}.log"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/DONE"
python3 eval/collect_runs.py --root "$(dirname "$(dirname "$RUN_DIR")")" --quiet || true
echo "[$(date)] done"

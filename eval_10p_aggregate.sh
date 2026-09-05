#!/bin/bash
#SBATCH --job-name=hcs_eval_10p_agg
#SBATCH --account=soney0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_10p_agg_%j.out
#SBATCH --error=logs/eval_10p_agg_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xiangyz@umich.edu

# Aggregate pass over the shared eval-main-10p folder (pooled Fitts/Steering/
# ID4SCS summaries + overview across all six participants) from the cached
# per-participant sims — no new simulation. Run after eval_10p.sh:
#   sbatch --dependency=afterok:<eval jobid> eval_10p_aggregate.sh
# Use the SAME RUN_TAG as the fit + eval generation.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load python3.11-anaconda/2024.02 2>/dev/null || true
source venv/bin/activate 2>/dev/null || source .venv/bin/activate 2>/dev/null || true
export MPLBACKEND=Agg

RUN_TAG="${RUN_TAG:-}"
RESULTS_ROOT="${RESULTS_ROOT:-/home/xiangyz/ondemand/data/sys/myjobs/projects/chi-27/results}"
export TMPDIR="${RESULTS_ROOT}/tmp/job_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT   # node /tmp is small+shared: 60064097_3 died ENOSPC
export HCS_EVAL_RESULTS_DIR="$RESULTS_ROOT/eval-main-10p${RUN_TAG:+-$RUN_TAG}"
PERSONA_DIR="$RESULTS_ROOT/personas_10p${RUN_TAG:+-$RUN_TAG}"
SEED="${SEED:-42}"

echo "[$(date)] aggregate over $HCS_EVAL_RESULTS_DIR"
python -u eval/eval-main/run_eval.py \
    --config-dir "$PERSONA_DIR" \
    --seed "$SEED" \
    --aggregate-only \
    --data-dir human_data/task_aligned_all \
    2>&1 | tee "$HCS_EVAL_RESULTS_DIR/eval_aggregate_s${SEED}.log"
echo "[$(date)] done"

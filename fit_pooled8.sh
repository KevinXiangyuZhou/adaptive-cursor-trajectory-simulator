#!/bin/bash
#SBATCH --job-name=hcs_fit_pooled8
#SBATCH --account=soney0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem-per-cpu=512M
#SBATCH --time=08:00:00
#SBATCH --output=logs/fit_pooled8_%j.out
#SBATCH --error=logs/fit_pooled8_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=xiangyz@umich.edu

# ONE pooled anchor-drive persona fitted jointly on all eight kept 10p
# sessions (eval/eval-anchor-drive/fit_anchor_pooled8.py): pooled Stage-0
# GAM + one CMA-fitted parameter set (jerk, contour, constraint, goal, D0;
# gamma/plan_vmax pinned, T0 calibrated post-fit on the pooled pointing
# loss). Single wide task, not an array: parallelism is over
# (candidate x participant) units — popsize 12 x 8 pids = 96 sims-batches
# per generation across 36 workers.
#
# Wall 8 h; CMA budget 5 h (default TIME_LIMIT=18000) — the pooled T0 scan
# (~10 min) and EIGHT per-participant held-out probes (~2 h total) run
# after the budget and must fit inside the wall.
#
#   sbatch fit_pooled8.sh
#   TIME_LIMIT=10800 sbatch fit_pooled8.sh        # shorter CMA budget
#   RUN_TAG=v2 sbatch fit_pooled8.sh              # tagged output dir

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load python3.11-anaconda/2024.02 2>/dev/null || true
source venv/bin/activate 2>/dev/null || source .venv/bin/activate 2>/dev/null || true

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
RUN_TAG="${RUN_TAG:-}"
RESULTS_ROOT="${RESULTS_ROOT:-/home/xiangyz/ondemand/data/sys/myjobs/projects/chi-27/results}"
export TMPDIR="${RESULTS_ROOT}/tmp/job_${SLURM_JOB_ID:-local}_0"
mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT   # node /tmp is small+shared
export HCS_FIT_RESULTS_DIR="$RESULTS_ROOT/anchor_fitting_pooled8${RUN_TAG:+-$RUN_TAG}"
export HCS_HUMAN_DATA_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}/human_data/task_aligned_all"
SEED="${SEED:-42}"
TIME_LIMIT="${TIME_LIMIT:-18000}"      # CMA budget < wall: T0 scan + 8 probes follow
mkdir -p "$HCS_FIT_RESULTS_DIR" logs

echo "[$(date)] pooled-8 fit -> $HCS_FIT_RESULTS_DIR (budget ${TIME_LIMIT}s, seed $SEED)"
# fit script writes to eval/eval-anchor-drive/results by default; symlink the
# tagged results dir there so cluster and local layouts match.
ln -sfn "$HCS_FIT_RESULTS_DIR" eval/eval-anchor-drive/results
python3 eval/eval-anchor-drive/fit_anchor_pooled8.py \
    --time-limit "$TIME_LIMIT" --workers "$SLURM_CPUS_PER_TASK" --seed "$SEED" \
    2>&1 | tee "$HCS_FIT_RESULTS_DIR/fit_pooled8_s${SEED}.log"
echo "[$(date)] done"

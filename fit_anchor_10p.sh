#!/bin/bash
#SBATCH --job-name=hcs_fit_10p
#SBATCH --account=soney0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=256M
#SBATCH --time=08:00:00
#SBATCH --array=1-8
#SBATCH --output=logs/fit_10p_%A_%a.out
#SBATCH --error=logs/fit_10p_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=xiangyz@umich.edu

# Anchor-drive persona fit for the 8 kept 10-participant-batch sessions
# (p01-p04, p06-p08, p10; p06/p08 recollected 2026-09-03; p05/p09 deprecated) under the FINALIZED cycle
# design (2026-09-03): GAM traversal deadline (shipped pooled artifact,
# hcs_package/models/gam_traversal_10p.pkl), no deadline rule stack, no
# free_velocity damping. Base personas: eval/model_fitting/base_configs_gaze/
# {pid}.json (10p budget priors D0=1.0, gamma=0.66; both refined by the fit).
#
# Wall 8 h; CMA budget 6.5 h — start-up, the generation in flight when the
# budget expires, the noise-on stability runs and the FULL held-out probe +
# save all run after the budget, and the 8-25 fits taught us the wall must
# clear the budget with real margin or Stage-3-equivalent results are lost.
#
#   sbatch fit_anchor_10p.sh                     # all eight
#   sbatch --array=3 fit_anchor_10p.sh           # rerun p03 only
#   RUN_TAG=v2 sbatch fit_anchor_10p.sh          # tagged output dirs

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load python3.11-anaconda/2024.02 2>/dev/null || true
source venv/bin/activate 2>/dev/null || source .venv/bin/activate 2>/dev/null || true

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
RUN_TAG="${RUN_TAG:-}"
RESULTS_ROOT="${RESULTS_ROOT:-/home/xiangyz/ondemand/data/sys/myjobs/projects/chi-27/results}"
export TMPDIR="${RESULTS_ROOT}/tmp/job_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT   # node /tmp is small+shared: 60064097_3 died ENOSPC
export HCS_FIT_RESULTS_DIR="$RESULTS_ROOT/anchor_fitting_10p${RUN_TAG:+-$RUN_TAG}"
# 10p cohort: the task-aligned batch (load_participant also finds it via the
# fallback, but being explicit keeps scan_conditions off the wrong dir).
export HCS_HUMAN_DATA_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}/human_data/task_aligned_all"
SEED="${SEED:-42}"
TIME_LIMIT="${TIME_LIMIT:-23400}"      # 6.5 h CMA budget < 8 h wall
POPSIZE="${POPSIZE:-12}"
mkdir -p "$HCS_FIT_RESULTS_DIR" logs

PARTICIPANTS_FILE="${PARTICIPANTS_FILE:-participants_10p.txt}"
PID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PARTICIPANTS_FILE")
echo "[$(date)] 10p anchor fit $PID -> $HCS_FIT_RESULTS_DIR (budget ${TIME_LIMIT}s, seed $SEED)"
# fit_anchor.py writes to eval/eval-anchor-drive/results by default; symlink
# the tagged results dir there so cluster and local layouts match.
ln -sfn "$HCS_FIT_RESULTS_DIR" eval/eval-anchor-drive/results
python3 eval/eval-anchor-drive/fit_anchor.py --pid "$PID" --time-limit "$TIME_LIMIT" \
    --popsize "$POPSIZE" --workers "$SLURM_CPUS_PER_TASK" --seed "$SEED" \
    2>&1 | tee "$HCS_FIT_RESULTS_DIR/fit_anchor_${PID}_s${SEED}.log"
echo "[$(date)] done $PID"

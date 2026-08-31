#!/bin/bash
#SBATCH --job-name=hcs_fit_anchor
#SBATCH --account=soney0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=1G
#SBATCH --time=11:00:00
#SBATCH --array=1-3
#SBATCH --output=logs/fit_anchor_%A_%a.out
#SBATCH --error=logs/fit_anchor_%A_%a.err

# Joint one-stage CMA-ES fit of the anchor-drive persona (eval/eval-anchor-drive/
# fit_anchor.py) — one array task per participant. Locally a candidate costs
# ~2-3 min (33 sims), so this is a cluster job: ~10 h of CMA budget gives
# ~200 generations at 12 workers. Same RUN_TAG / cohort conventions as
# fit_all_participants.sh. Defaults to the gaze cohort (the anchor persona needs
# a Stage G base config with the gaze budget and latency).
#
#   RUN_TAG=anchor-1 sbatch fit_anchor_all_participants.sh
#   PARTICIPANTS_FILE=participants_gaze.txt DATA_DIR=human_data/gaze_cursor_data \
#   BASE_CONFIG_DIR=eval/model_fitting/base_configs_gaze

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load python3.11-anaconda/2024.02 2>/dev/null || true
source .venv/bin/activate 2>/dev/null || true

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg TMPDIR=/tmp
RUN_TAG="${RUN_TAG:-}"
RESULTS_ROOT="${RESULTS_ROOT:-$SLURM_SUBMIT_DIR/chi-27/results}"
export HCS_FIT_RESULTS_DIR="$RESULTS_ROOT/anchor_fitting${RUN_TAG:+-$RUN_TAG}"
PARTICIPANTS_FILE="${PARTICIPANTS_FILE:-participants_gaze.txt}"
DATA_DIR="${DATA_DIR:-human_data/gaze_cursor_data}"
export HCS_HUMAN_DATA_DIR="$SLURM_SUBMIT_DIR/$DATA_DIR"
SEED="${SEED:-42}"
TIME_LIMIT="${TIME_LIMIT:-36000}"      # keep below the wall (held-out eval + save run after the budget)
POPSIZE="${POPSIZE:-12}"
mkdir -p "$HCS_FIT_RESULTS_DIR" logs

PID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PARTICIPANTS_FILE")
echo "[$(date)] anchor fit $PID -> $HCS_FIT_RESULTS_DIR (budget ${TIME_LIMIT}s, seed $SEED)"
# fit_anchor.py writes to eval/eval-anchor-drive/results by default; symlink the
# tagged results dir there so cluster and local layouts match.
mkdir -p eval/eval-anchor-drive
ln -sfn "$HCS_FIT_RESULTS_DIR" eval/eval-anchor-drive/results
python3 eval/eval-anchor-drive/fit_anchor.py --pid "$PID" --time-limit "$TIME_LIMIT" \
    --popsize "$POPSIZE" --workers "$SLURM_CPUS_PER_TASK" --seed "$SEED" \
    2>&1 | tee "$HCS_FIT_RESULTS_DIR/fit_anchor_${PID}_s${SEED}.log"
echo "[$(date)] done $PID"

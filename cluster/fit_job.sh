#!/bin/bash
#SBATCH --account=soney0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem-per-cpu=256M
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xiangyz@umich.edu

# Per-participant joint fit (one array task per participant) of one run.
# Never sbatch this directly: submit_run.sh creates the run tree, snapshots
# the code, and exports RUN_DIR / MODEL / VARIANT / KIND / SEED / TIME_LIMIT /
# POPSIZE / PARTICIPANTS_FILE / VENV_DIR. Every path below derives from
# RUN_DIR, so concurrent runs never touch each other's files.
#
# One full node (36 cores): the work unit is (candidate x trial), ~300
# units per generation, so every core stays busy and a generation ends
# after the longest single trial (2026-09-08; was 12 cores = one candidate
# per core).
#
# Wall 8 h; CMA budget 6.5 h — start-up, the generation in flight when the
# budget expires, the noise-on stability runs and the held-out probe + save
# all run after the budget and must clear the wall with margin.

set -euo pipefail
: "${RUN_DIR:?set by submit_run.sh}" "${MODEL:?}" "${VARIANT:?}" "${SEED:?}"
cd "$RUN_DIR/code"
module load python3.11-anaconda/2024.02 2>/dev/null || true
source "${VENV_DIR:-$RUN_DIR/code/venv}/bin/activate"

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
export TMPDIR="$RUN_DIR/tmp/job_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT   # node /tmp is small+shared
export HCS_FIT_RESULTS_DIR="$RUN_DIR/fit"
export HCS_HUMAN_DATA_DIR="$RUN_DIR/code/human_data/task_aligned_all"
TIME_LIMIT="${TIME_LIMIT:-23400}"      # 6.5 h CMA budget < 8 h wall
POPSIZE="${POPSIZE:-12}"
PARTICIPANTS_FILE="${PARTICIPANTS_FILE:-participants_10p.txt}"
ABLATION=$([ "$VARIANT" = "full" ] && echo none || echo "$VARIANT")
mkdir -p "$HCS_FIT_RESULTS_DIR"

PID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PARTICIPANTS_FILE")
echo "[$(date)] fit $MODEL/$VARIANT $PID -> $HCS_FIT_RESULTS_DIR (budget ${TIME_LIMIT}s, seed $SEED, code $(cat "$RUN_DIR/COMMIT" 2>/dev/null || echo ?))"
python3 eval/eval-anchor-drive/fit_anchor.py --pid "$PID" --model "$MODEL" --ablation "$ABLATION" \
    --tag base --time-limit "$TIME_LIMIT" --popsize "$POPSIZE" \
    --workers "$SLURM_CPUS_PER_TASK" --seed "$SEED" \
    2>&1 | tee "$HCS_FIT_RESULTS_DIR/fit_${PID}_s${SEED}.log"
echo "[$(date)] done $PID"

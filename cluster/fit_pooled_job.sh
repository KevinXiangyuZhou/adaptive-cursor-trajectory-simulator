#!/bin/bash
#SBATCH --account=soney0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem-per-cpu=512M
#SBATCH --time=15:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xiangyz@umich.edu

# ONE pooled persona fitted jointly on all participants of one run
# (eval/eval-anchor-drive/fit_anchor_pooled8.py). Never sbatch directly:
# submit_run.sh exports RUN_DIR / MODEL / VARIANT / KIND / SEED / TIME_LIMIT /
# POPSIZE / PARTICIPANTS_FILE / VENV_DIR / GAMMA / TRAIN_ALL. Single wide
# task, not an array: parallelism is over (candidate x participant x trial)
# units.
#
# Training set (2026-09-11): TRAIN_ALL=1 (the default) trains on EVERY
# steering width and EVERY pointing condition of all 8 participants — no
# held-out split (25 tunnel + 45 pointing + 3 stability units per
# participant, ~7000 units per generation of 12 candidates, ~3x the split
# protocol's ~2600). TRAIN_ALL=0 restores the split.
#
# Wall 15 h; CMA budget 12 h (default TIME_LIMIT=43200), leaving a 3 h
# margin for the in-flight generation (a train-all generation is ~25-30 min
# on 36 cores, extrapolated from the ~10 min split generations of the 9-10
# pooled run: 36 generations in 6 h), the pooled
# T0 scan (23 grid points x 8 pids x 45 pointing rounds), the per-participant
# train-loss pass and the 8 full anchor probes + save (~1.5 h in total).

set -euo pipefail
: "${RUN_DIR:?set by submit_run.sh}" "${MODEL:?}" "${VARIANT:?}" "${SEED:?}"
cd "$RUN_DIR/code"
module load python3.11-anaconda/2024.02 2>/dev/null || true
source "${VENV_DIR:-$RUN_DIR/code/venv}/bin/activate"

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
export TMPDIR="$RUN_DIR/tmp/job_${SLURM_JOB_ID:-local}_0"
mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
export HCS_FIT_RESULTS_DIR="$RUN_DIR/fit"
export HCS_HUMAN_DATA_DIR="$RUN_DIR/code/human_data/task_aligned_all"
TIME_LIMIT="${TIME_LIMIT:-43200}"      # 12 h CMA budget < 15 h wall (3 h pooled margin)
POPSIZE="${POPSIZE:-12}"
TRAIN_ALL="${TRAIN_ALL:-1}"            # 1 = every condition trains (no held-out split)
TRAIN_ALL_ARG=$([ "$TRAIN_ALL" = 1 ] && echo --train-all || echo "")
PARTICIPANTS_FILE="${PARTICIPANTS_FILE:-participants_10p.txt}"
ABLATION=$([ "$VARIANT" = "full" ] && echo none || echo "$VARIANT")
mkdir -p "$HCS_FIT_RESULTS_DIR"
LETTERS=$(tr '\n' ' ' < "$PARTICIPANTS_FILE")
GAMMA_ARG=""
if [ -n "${GAMMA:-}" ] && [ "$MODEL" = "mpcc" ]; then GAMMA_ARG="--gamma $GAMMA"; fi

echo "[$(date)] pooled fit $MODEL/$VARIANT over [$LETTERS] -> $HCS_FIT_RESULTS_DIR (budget ${TIME_LIMIT}s, train_all=$TRAIN_ALL, seed $SEED, code $(cat "$RUN_DIR/COMMIT" 2>/dev/null || echo ?))"
# shellcheck disable=SC2086
python3 eval/eval-anchor-drive/fit_anchor_pooled8.py --model "$MODEL" --ablation "$ABLATION" \
    --tag pooled8 --letters $LETTERS \
    --time-limit "$TIME_LIMIT" --popsize "$POPSIZE" --workers "$SLURM_CPUS_PER_TASK" --seed "$SEED" \
    $GAMMA_ARG $TRAIN_ALL_ARG \
    2>&1 | tee "$HCS_FIT_RESULTS_DIR/fit_pooled8_s${SEED}.log"
echo "[$(date)] done"

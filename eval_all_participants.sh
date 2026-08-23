#!/bin/bash
#SBATCH --job-name=hcs_eval
#SBATCH --account=soney0
#SBATCH --partition=standard
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=1g
#SBATCH --array=1-10
#SBATCH --output=/home/xiangyz/ondemand/data/sys/myjobs/projects/chi-27/logs/eval_%A_%a.out
#SBATCH --error=/home/xiangyz/ondemand/data/sys/myjobs/projects/chi-27/logs/eval_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xiangyz@umich.edu
#
# eval-main with each participant's fitted persona (after fit_all_participants.sh).
# Submit from the repo root:  sbatch eval_all_participants.sh
# Then aggregate:             sbatch --dependency=afterok:<jobid> eval_aggregate.sh

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

# All outputs go to the chi-27 project folder (override with RESULTS_ROOT)
RESULTS_ROOT="${RESULTS_ROOT:-/home/xiangyz/ondemand/data/sys/myjobs/projects/chi-27/results}"
export HCS_FIT_RESULTS_DIR="$RESULTS_ROOT/model_fitting"
export HCS_EVAL_RESULTS_DIR="$RESULTS_ROOT/eval-main"
mkdir -p "$HCS_FIT_RESULTS_DIR" "$HCS_EVAL_RESULTS_DIR"

module load python3.11-anaconda/2024.02
source venv/bin/activate
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TMPDIR=/tmp
export MPLBACKEND=Agg

PID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" participants.txt)
SEED="${SEED:-42}"
echo "Evaluating participant: $PID (array task $SLURM_ARRAY_TASK_ID, seed $SEED)"
echo "Start time: $(date)"

# Cohort selection mirrors fit_all_participants.sh:
#   PARTICIPANTS_FILE=participants_gaze.txt DATA_DIR=human_data/gaze_cursor_data sbatch --array=1-3 ...
DATA_ARG=()
if [ -n "$DATA_DIR" ]; then DATA_ARG=(--data-dir "$PROJECT_DIR/$DATA_DIR"); fi
python -u eval/eval-main/run_eval.py \
    --pid "$PID" \
    --per-participant \
    --seed "$SEED" \
    --fresh-sim \
    "${DATA_ARG[@]}" \
    2>&1 | tee "$HCS_EVAL_RESULTS_DIR/eval_${PID}_s${SEED}.log"

echo "End time: $(date)"

#!/bin/bash
#SBATCH --job-name=hcs_fit
#SBATCH --account=soney0
#SBATCH --partition=standard
#SBATCH --time=13:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=1g
#SBATCH --array=1-10
#SBATCH --output=/home/xiangyz/ondemand/data/sys/myjobs/projects/chi-27/logs/fit_%A_%a.out
#SBATCH --error=/home/xiangyz/ondemand/data/sys/myjobs/projects/chi-27/logs/fit_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=xiangyz@umich.edu
#
# Per-participant model fitting (array job, one task per line of participants.txt).
# Submit from the repo root:  sbatch fit_all_participants.sh
# Rerun one participant:      sbatch --array=N fit_all_participants.sh
# Pointing stage only (reuse tunnel fit): STAGES=pointing sbatch fit_all_participants.sh

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

# All outputs go to the chi-27 project folder (override with RESULTS_ROOT)
RESULTS_ROOT="${RESULTS_ROOT:-/home/xiangyz/ondemand/data/sys/myjobs/projects/chi-27/results}"
export HCS_FIT_RESULTS_DIR="$RESULTS_ROOT/model_fitting"
export HCS_EVAL_RESULTS_DIR="$RESULTS_ROOT/eval-main"
mkdir -p "$HCS_FIT_RESULTS_DIR" "$HCS_EVAL_RESULTS_DIR"

module load python3.11-anaconda/2024.02
source venv/bin/activate

# one BLAS thread per worker; CMA-ES workers already use all CPUs
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TMPDIR=/tmp
export MPLBACKEND=Agg

# Cohort selection (defaults = aug-26-prolific). Gaze cohort:
#   PARTICIPANTS_FILE=participants_gaze.txt \
#   DATA_DIR=human_data/gaze_cursor_data \
#   BASE_CONFIG_DIR=eval/model_fitting/base_configs_gaze \
#   sbatch --array=1-3 fit_all_participants.sh
PARTICIPANTS_FILE="${PARTICIPANTS_FILE:-participants.txt}"
PID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PARTICIPANTS_FILE")
SEED="${SEED:-42}"
STAGES="${STAGES:-all}"
if [ -n "$DATA_DIR" ]; then export HCS_HUMAN_DATA_DIR="$PROJECT_DIR/$DATA_DIR"; fi
CONFIG_ARG=()
if [ -n "$BASE_CONFIG_DIR" ]; then CONFIG_ARG=(--config "$PROJECT_DIR/$BASE_CONFIG_DIR/${PID}.json"); fi
# leave ~30 min of the 13 h wall time for start-up + final validation
TIME_LIMIT="${TIME_LIMIT:-43200}"

echo "Fitting participant: $PID (array task $SLURM_ARRAY_TASK_ID, seed $SEED, stages $STAGES)"
echo "Start time: $(date)"

python -u -m eval.model_fitting.fit_speed_model \
    --pid "$PID" \
    --time-limit "$TIME_LIMIT" \
    --seed "$SEED" \
    --popsize 12 \
    --n-workers "$SLURM_CPUS_PER_TASK" \
    --stages "$STAGES" \
    "${CONFIG_ARG[@]}" \
    2>&1 | tee "$HCS_FIT_RESULTS_DIR/fit_${PID}_s${SEED}.log"

echo "End time: $(date)"

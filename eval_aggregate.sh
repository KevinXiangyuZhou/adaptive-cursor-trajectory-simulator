#!/bin/bash
#SBATCH --job-name=hcs_agg
#SBATCH --account=soney0
#SBATCH --partition=standard
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4g
#SBATCH --output=/home/xiangyz/ondemand/data/sys/myjobs/projects/chi-27/logs/eval_agg_%j.out
#SBATCH --error=/home/xiangyz/ondemand/data/sys/myjobs/projects/chi-27/logs/eval_agg_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xiangyz@umich.edu
#
# Aggregate eval-main outputs across participants from the cached per-participant
# simulations (no new simulation). Run after eval_all_participants.sh:
#   EVAL_JOB_ID=$(sbatch --parsable eval_all_participants.sh)
#   sbatch --dependency=afterok:$EVAL_JOB_ID eval_aggregate.sh

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

# All outputs go to the chi-27 project folder (override with RESULTS_ROOT)
RESULTS_ROOT="${RESULTS_ROOT:-/home/xiangyz/ondemand/data/sys/myjobs/projects/chi-27/results}"
export HCS_FIT_RESULTS_DIR="$RESULTS_ROOT/model_fitting"
export HCS_EVAL_RESULTS_DIR="$RESULTS_ROOT/eval-main"
mkdir -p "$HCS_FIT_RESULTS_DIR" "$HCS_EVAL_RESULTS_DIR"
module load python3.11-anaconda/2024.02
source venv/bin/activate
export MPLBACKEND=Agg
SEED="${SEED:-42}"
echo "Start time: $(date)"
# Cohort selection mirrors eval_all_participants.sh:
#   DATA_DIR=human_data/gaze_cursor_data sbatch eval_aggregate.sh
DATA_ARG=()
if [ -n "$DATA_DIR" ]; then DATA_ARG=(--data-dir "$PROJECT_DIR/$DATA_DIR"); fi
python -u eval/eval-main/run_eval.py --per-participant --seed "$SEED" --aggregate-only \
    "${DATA_ARG[@]}" \
    2>&1 | tee "$HCS_EVAL_RESULTS_DIR/eval_aggregate_s${SEED}.log"
echo "End time: $(date)"

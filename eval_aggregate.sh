#!/bin/bash
#SBATCH --job-name=hcs_agg
#SBATCH --account=soney0
#SBATCH --partition=standard
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4g
#SBATCH --output=/home/xiangyz/ondemand/data/sys/myjobs/projects/uist-26/simulation-for-webGUI-evaluation/logs/eval_agg_%j.out
#SBATCH --error=/home/xiangyz/ondemand/data/sys/myjobs/projects/uist-26/simulation-for-webGUI-evaluation/logs/eval_agg_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xiangyz@umich.edu

# Aggregation job — run after all eval array tasks complete.
# Usage:
#   EVAL_JOB_ID=$(sbatch --parsable eval_all_participants.sh)
#   sbatch --dependency=afterok:$EVAL_JOB_ID eval_aggregate.sh

PROJECT_DIR="/home/xiangyz/ondemand/data/sys/myjobs/projects/uist-26/simulation-for-webGUI-evaluation"
cd "$PROJECT_DIR"

module load python3.11-anaconda/2024.02
source venv/bin/activate

echo "Aggregating results from all participants ..."
echo "Start time: $(date)"

python -u eval/experiment-main/run_eval.py \
    --per-participant \
    --include-baseline \
    --rounds 5 \
    --seed 42 \
    --aggregate-only \
    2>&1 | tee "eval/experiment-main/results/eval_aggregate.log"

echo "End time: $(date)"

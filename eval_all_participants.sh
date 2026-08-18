#!/bin/bash
#SBATCH --job-name=hcs_eval
#SBATCH --account=soney0
#SBATCH --partition=standard
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=256m
#SBATCH --array=1-14
#SBATCH --output=/home/xiangyz/ondemand/data/sys/myjobs/projects/uist-26/simulation-for-webGUI-evaluation/logs/eval_%A_%a.out
#SBATCH --error=/home/xiangyz/ondemand/data/sys/myjobs/projects/uist-26/simulation-for-webGUI-evaluation/logs/eval_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=xiangyz@umich.edu

PROJECT_DIR="/home/xiangyz/ondemand/data/sys/myjobs/projects/uist-26/simulation-for-webGUI-evaluation"
cd "$PROJECT_DIR"

# Load Python and activate virtual environment
module load python3.11-anaconda/2024.02
source venv/bin/activate

# Force 1 thread per worker to avoid CPU contention when multiple jobs share a node
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Use local disk for temp files (faster than networked /home)
export TMPDIR=/tmp

# Headless matplotlib (also set inside run_eval workers, but belt-and-suspenders)
export MPLBACKEND=Agg

# Read participant ID for this array task
PID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" participants.txt)
echo "Evaluating participant: $PID (array task $SLURM_ARRAY_TASK_ID)"
echo "Start time: $(date)"

python -u eval/experiment-main/run_eval.py \
    --pid "$PID" \
    --per-participant \
    --include-baseline \
    --rounds 5 \
    --seed 42 \
    2>&1 | tee "eval/experiment-main/results/eval_${PID}_s42.log"

echo "End time: $(date)"

#!/bin/bash
#SBATCH --job-name=hcs_bfit
#SBATCH --account=soney0
#SBATCH --partition=standard
#SBATCH --time=013:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=512m
#SBATCH --array=1-14
#SBATCH --output=/home/xiangyz/ondemand/data/sys/myjobs/projects/uist-26/simulation-for-webGUI-evaluation/logs/bfit_%A_%a.out
#SBATCH --error=/home/xiangyz/ondemand/data/sys/myjobs/projects/uist-26/simulation-for-webGUI-evaluation/logs/bfit_%A_%a.err
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

# Read participant ID for this array task
PID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" participants.txt)
echo "Baseline fitting participant: $PID (array task $SLURM_ARRAY_TASK_ID)"
echo "Start time: $(date)"

python -m eval.baseline_fitting.fit_baseline \
    --pid "$PID" \
    --time-limit 43200 \
    --seed 42 \
    --popsize 12 \
    2>&1 | tee "eval/baseline_fitting/results/bfit_${PID}_s42.log"

echo "End time: $(date)"

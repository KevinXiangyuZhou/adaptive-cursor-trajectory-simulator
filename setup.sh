#!/bin/bash
# Run once on the Great Lakes login node, from the repo root, before submitting jobs.
#   cd <repo>   (e.g. /home/xiangyz/ondemand/data/sys/myjobs/projects/uist-26/adaptive-cursor-trajectory-simulator)
#   bash setup.sh
#   sbatch fit_all_participants.sh
set -e
CHI27=/home/xiangyz/ondemand/data/sys/myjobs/projects/chi-27
mkdir -p "$CHI27/logs" "$CHI27/results/model_fitting" "$CHI27/results/eval-main"

# `module available python` lists versions; keep in sync with the #SBATCH scripts.
module load python3.11-anaconda/2024.02
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install "numpy<2.0" scipy cma pygam matplotlib pandas seaborn
pip install -e hcs_package/

echo "Setup complete. Next: sbatch fit_all_participants.sh   (then eval_all_participants.sh)"

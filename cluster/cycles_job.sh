#!/bin/bash
#SBATCH --account=soney0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1500M
#SBATCH --time=03:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xiangyz@umich.edu

# Saccade-fixate-catch-up cycle statistics of one mpcc run (paper Sec. 5.1):
# every participant's fitted persona is re-simulated on all steering trials
# (noise on, 3 runs) and the cycle measures — saccade distance and onset
# lead vs width, catch-up duration vs width and curvature, overrun, speed ->
# lead — are compared with the human ones (paper/quantify_gaze_cycles.py,
# stages model + stats). Outputs in $RUN_DIR/cycles/: cycles_model.csv,
# stats.txt. One core per participant; the rows do not depend on the worker
# count. Never sbatch directly: submit_run.sh chains it after the fit, and
# submit_cycles.sh runs it on a finished run. Exports: RUN_DIR / KIND / SEED /
# PARTICIPANTS_FILE / VENV_DIR.

set -euo pipefail
: "${RUN_DIR:?set by submit_run.sh / submit_cycles.sh}" "${KIND:?}" "${SEED:?}"
cd "$RUN_DIR/code"
module load python3.11-anaconda/2024.02 2>/dev/null || true
source "${VENV_DIR:-$RUN_DIR/code/venv}/bin/activate"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
export TMPDIR="$RUN_DIR/tmp/job_${SLURM_JOB_ID:-local}_cycles"
mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT

QUANT="$RUN_DIR/cycles"
PARTICIPANTS_FILE="${PARTICIPANTS_FILE:-participants_10p.txt}"
LETTERS=$(tr '\n' ' ' < "$PARTICIPANTS_FILE")
mkdir -p "$QUANT"
# human side of the comparison: the cycle tables behind the paper's numbers
for f in cycles_human.csv curv_human.csv; do
    if [ ! -f "$QUANT/$f" ]; then
        for q in paper/quant_refit paper/quant; do
            [ -f "$q/$f" ] && { cp "$q/$f" "$QUANT/$f"; break; }
        done
    fi
    [ -f "$QUANT/$f" ] || { echo "missing human cycle table $f (paper/quant_refit or paper/quant)"; exit 1; }
done
if [ "$KIND" = "pooled8" ]; then
    CFG_ARG=(--config "$RUN_DIR/fit/stages/pooled8/pooled8_anchor_config_s${SEED}.json")
    [ -f "${CFG_ARG[1]}" ] || { echo "missing pooled persona ${CFG_ARG[1]}"; exit 1; }
else
    CFG_ARG=(--config-dir "$RUN_DIR/fit/stages/base")
    ls "$RUN_DIR/fit/stages/base/"*_anchor_config_s"${SEED}".json >/dev/null 2>&1 || { echo "no fitted personas in $RUN_DIR/fit/stages/base"; exit 1; }
fi

echo "[$(date)] cycles ${CFG_ARG[*]} letters [$LETTERS] -> $QUANT"
# shellcheck disable=SC2086
python -u paper/quantify_gaze_cycles.py --stage model "${CFG_ARG[@]}" --quant-dir "$QUANT" \
    --letters $LETTERS --workers "${SLURM_CPUS_PER_TASK:-4}" 2>&1 | tee "$QUANT/model_stage.log"
# shellcheck disable=SC2086
python -u paper/quantify_gaze_cycles.py --stage stats --quant-dir "$QUANT" --letters $LETTERS \
    2>&1 | tee "$QUANT/stats.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$QUANT/DONE"
echo "[$(date)] done"

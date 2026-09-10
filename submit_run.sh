#!/bin/bash
# Submit one isolated run (fit -> eval -> aggregate) to Great Lakes.
#
#   submit_run.sh --model {mpcc,baseline} --variant {full,no_gaze,no_lookahead,no_pace,no_intermittent} \
#                 --kind {perpid,pooled8} [--seed 42] [--participants participants_10p.txt] \
#                 [--time-limit S] [--popsize N] [--min-runs N] [--gamma G] \
#                 [--wall HH:MM:SS] [--results-root DIR] [--data-root DIR] [--allow-dirty] [--dry-run]
#
# --time-limit is the CMA-ES budget in seconds; --wall overrides the fit job's
# SLURM time limit (default 08:00:00, per-participant and pooled). Keep
# wall >= budget + 0.5 h (per-participant) / + 2 h (pooled) for the post-fit
# T0 scan, held-out probes and save; a shorter wall backfills sooner.
#
# One run = one RUN_ID = one code snapshot = one results tree:
#   $RESULTS_ROOT/runs/<RUN_ID>/{RUN_INFO.json,COMMIT,code/,fit/,personas/,eval/,gaze-lead/,logs/,tmp/}
# RUN_ID = <model>-<variant>-<kind>-s<seed>-<yyyymmdd-HHMM>-<sha7>. The jobs
# execute from the snapshot in code/ (git archive of HEAD), so editing the
# working tree after submission cannot change what a queued job runs; every
# output path derives from RUN_DIR, so concurrent runs never collide. A run
# directory that already exists aborts submission. One line per run is
# appended to $RESULTS_ROOT/runs/INDEX.tsv.
#
# Run from the repo root on the login node (needs: venv/ set up by setup.sh,
# human_data/task_aligned_all rsynced into this checkout).

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

MODEL=""; VARIANT=""; KIND=""; SEED=42; PARTICIPANTS_FILE="participants_10p.txt"
TIME_LIMIT=""; WALL=""; POPSIZE=12; MIN_RUNS=0; GAMMA=""; ALLOW_DIRTY=0; DRY=0
RESULTS_ROOT="${RESULTS_ROOT:-/home/xiangyz/ondemand/data/sys/myjobs/projects/chi-27/results}"
DATA_ROOT="$REPO_ROOT/human_data"
DATA_DIR="$DATA_ROOT/task_aligned_all"
VENV_DIR="$REPO_ROOT/venv"
while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL="$2"; shift 2;;
        --variant) VARIANT="$2"; shift 2;;
        --kind) KIND="$2"; shift 2;;
        --seed) SEED="$2"; shift 2;;
        --participants) PARTICIPANTS_FILE="$2"; shift 2;;
        --time-limit) TIME_LIMIT="$2"; shift 2;;
        --wall) WALL="$2"; shift 2;;
        --popsize) POPSIZE="$2"; shift 2;;
        --min-runs) MIN_RUNS="$2"; shift 2;;
        --gamma) GAMMA="$2"; shift 2;;
        --results-root) RESULTS_ROOT="$2"; shift 2;;
        --data-root) DATA_ROOT="$2"; DATA_DIR="$DATA_ROOT/task_aligned_all"; shift 2;;
        --venv) VENV_DIR="$2"; shift 2;;
        --allow-dirty) ALLOW_DIRTY=1; shift;;
        --dry-run) DRY=1; shift;;
        -h|--help) sed -n 2,20p "$0"; exit 0;;
        *) echo "unknown option $1"; exit 2;;
    esac
done
case "$MODEL" in mpcc|baseline) ;; *) echo "--model must be mpcc or baseline"; exit 2;; esac
case "$VARIANT" in full|no_gaze|no_lookahead|no_pace|no_intermittent) ;; *) echo "--variant invalid"; exit 2;; esac
case "$KIND" in perpid|pooled8) ;; *) echo "--kind must be perpid or pooled8"; exit 2;; esac
if [ "$MODEL" = "baseline" ] && [ "$VARIANT" != "full" ]; then echo "baseline has no ablations (use --variant full)"; exit 2; fi
[ -f "$PARTICIPANTS_FILE" ] || { echo "missing $PARTICIPANTS_FILE"; exit 2; }
[ -d "$DATA_DIR" ] || { echo "missing data dir $DATA_DIR (rsync human_data/task_aligned_all first)"; exit 2; }
ls "$DATA_DIR"/p01_task_aligned_analysis*.csv >/dev/null 2>&1 || echo "WARNING: no task-aligned gaze CSVs under $DATA_DIR — the mpcc gaze-lead figures will fail (rsync human_data/task_aligned_all/*.csv up to enable; the eval itself is unaffected)"
[ -x "$VENV_DIR/bin/python" ] || { echo "missing venv at $VENV_DIR (bash setup.sh)"; exit 2; }
N_PIDS=$(grep -c . "$PARTICIPANTS_FILE")

SHA=$(git rev-parse --short=7 HEAD)
DIRTY=0
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    DIRTY=1
    if [ "$ALLOW_DIRTY" != 1 ]; then
        echo "working tree has uncommitted changes to tracked files; commit them or pass --allow-dirty"; exit 2
    fi
fi
STAMP=$(date +%Y%m%d-%H%M)
RUN_ID="${MODEL}-${VARIANT}-${KIND}-s${SEED}-${STAMP}-${SHA}"
RUN_DIR="$RESULTS_ROOT/runs/$RUN_ID"
[ -e "$RUN_DIR" ] && { echo "run dir exists: $RUN_DIR (wait a minute or change --seed)"; exit 2; }
echo "RUN_ID  $RUN_ID"
echo "RUN_DIR $RUN_DIR"
[ "$DRY" = 1 ] && { echo "(dry run: nothing created)"; exit 0; }

# --- run tree + code snapshot --------------------------------------------
mkdir -p "$RUN_DIR"/{code,fit,personas,eval,gaze-lead,logs,tmp}
if [ "$DIRTY" = 1 ]; then
    rsync -a --exclude .git --exclude 'results*' --exclude venv --exclude .venv \
          --exclude '__pycache__' --exclude 'human_data' --exclude 'backup' \
          --exclude 'model_fitting*' ./ "$RUN_DIR/code/"
else
    git archive HEAD | tar -x -C "$RUN_DIR/code"
fi
git rev-parse HEAD > "$RUN_DIR/COMMIT"
# Human data: git carries only part of human_data/ (the task-aligned gaze
# CSVs in gaze_cursor_data/ and the task_aligned_all sessions are rsynced,
# not committed). Point the snapshot's human_data/ at this checkout's
# directory as a whole so every consumer (fits, eval-main, gaze-lead
# figures) finds the same files the working tree has.
rm -rf "$RUN_DIR/code/human_data"
ln -s "$DATA_ROOT" "$RUN_DIR/code/human_data"
# the per-run tree's own fit dir is what the fit scripts read/write; make
# the legacy default location inside the snapshot point at it as well
rm -rf "$RUN_DIR/code/eval/eval-anchor-drive/results"; ln -s "$RUN_DIR/fit" "$RUN_DIR/code/eval/eval-anchor-drive/results"

EXPORTS="ALL,RUN_DIR=$RUN_DIR,MODEL=$MODEL,VARIANT=$VARIANT,KIND=$KIND,SEED=$SEED,POPSIZE=$POPSIZE,MIN_RUNS=$MIN_RUNS,PARTICIPANTS_FILE=$PARTICIPANTS_FILE,VENV_DIR=$VENV_DIR"
[ -n "$TIME_LIMIT" ] && EXPORTS="$EXPORTS,TIME_LIMIT=$TIME_LIMIT"
[ -n "$GAMMA" ] && EXPORTS="$EXPORTS,GAMMA=$GAMMA"

# --- submit the chain -----------------------------------------------------
WALL_ARG=(); [ -n "$WALL" ] && WALL_ARG=(--time "$WALL")
if [ "$KIND" = perpid ]; then
    FIT_JOB=$(sbatch --parsable --job-name "fit-$RUN_ID" --array="1-$N_PIDS" "${WALL_ARG[@]}" \
        --output "$RUN_DIR/logs/fit_%A_%a.out" --error "$RUN_DIR/logs/fit_%A_%a.err" \
        --export="$EXPORTS" "$RUN_DIR/code/cluster/fit_job.sh")
else
    FIT_JOB=$(sbatch --parsable --job-name "fitp-$RUN_ID" "${WALL_ARG[@]}" \
        --output "$RUN_DIR/logs/fit_pooled_%j.out" --error "$RUN_DIR/logs/fit_pooled_%j.err" \
        --export="$EXPORTS" "$RUN_DIR/code/cluster/fit_pooled_job.sh")
fi
FIT_JOB="${FIT_JOB%%;*}"
EVAL_JOB=$(sbatch --parsable --job-name "eval-$RUN_ID" --array="1-$N_PIDS" \
    --dependency="afterok:$FIT_JOB" \
    --output "$RUN_DIR/logs/eval_%A_%a.out" --error "$RUN_DIR/logs/eval_%A_%a.err" \
    --export="$EXPORTS" "$RUN_DIR/code/cluster/eval_job.sh")
EVAL_JOB="${EVAL_JOB%%;*}"
AGG_JOB=$(sbatch --parsable --job-name "agg-$RUN_ID" \
    --dependency="afterok:$EVAL_JOB" \
    --output "$RUN_DIR/logs/agg_%j.out" --error "$RUN_DIR/logs/agg_%j.err" \
    --export="$EXPORTS" "$RUN_DIR/code/cluster/aggregate_job.sh")
AGG_JOB="${AGG_JOB%%;*}"

# --- record ---------------------------------------------------------------
python3 - "$RUN_DIR" <<EOF
import json, sys, datetime, os
d = sys.argv[1]
info = {
  "run_id": "$RUN_ID", "model": "$MODEL", "variant": "$VARIANT", "kind": "$KIND", "seed": $SEED,
  "commit": open(os.path.join(d, "COMMIT")).read().strip(), "dirty": bool($DIRTY),
  "submitted": datetime.datetime.now().isoformat(timespec="seconds"),
  "participants_file": "$PARTICIPANTS_FILE", "n_participants": $N_PIDS,
  "time_limit": "${TIME_LIMIT:-default}", "wall": "${WALL:-default}", "popsize": $POPSIZE, "min_runs": $MIN_RUNS, "gamma": "${GAMMA:-default}",
  "data_root": "$DATA_ROOT", "data_dir": "$DATA_DIR", "venv": "$VENV_DIR",
  "jobs": {"fit": "$FIT_JOB", "eval": "$EVAL_JOB", "aggregate": "$AGG_JOB"},
  "cmdline": " ".join(sys.argv[1:]) or "$0 --model $MODEL --variant $VARIANT --kind $KIND --seed $SEED",
}
json.dump(info, open(os.path.join(d, "RUN_INFO.json"), "w"), indent=2)
EOF
INDEX="$RESULTS_ROOT/runs/INDEX.tsv"
[ -f "$INDEX" ] || printf 'run_id\tmodel\tvariant\tkind\tseed\tcommit\tdirty\tsubmitted\tfit_job\teval_job\tagg_job\n' > "$INDEX"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$RUN_ID" "$MODEL" "$VARIANT" "$KIND" "$SEED" "$SHA" "$DIRTY" "$STAMP" "$FIT_JOB" "$EVAL_JOB" "$AGG_JOB" >> "$INDEX"
echo "submitted: fit $FIT_JOB -> eval $EVAL_JOB -> aggregate $AGG_JOB"
echo "logs:      $RUN_DIR/logs/"

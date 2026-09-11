#!/bin/bash
# Submit one isolated eval-14p run (eval array -> aggregate) to Great Lakes:
# a persona fitted on the earlier 8-participant cohort, evaluated with the
# main harness (eval/eval-main/run_eval.py) on the 14 NEW participants in
# eval-14p/human_data/raw. No fitting: the persona is fixed, one file for
# every participant (staged as <RUN_DIR>/personas/default.json).
#
#   eval-14p/submit_eval_14p.sh [--persona FILE | --source-run RUN_ID] [--tag NAME]
#                               [--seed 42] [--min-runs N] [--buckets "steering"]
#                               [--wall HH:MM:SS] [--participants FILE]
#                               [--reference CSV] [--results-root DIR]
#                               [--allow-dirty] [--dry-run]
#
# --persona     fitted config to evaluate. Default: the tracked copy of the
#               mpcc-full-pooled8-s42-20260910-1540-991900f pooled8 fit,
#               eval-14p/personas/pooled8-991900f/default.json (see SOURCE.md).
# --source-run  alternative: stage $RESULTS_ROOT/runs/<RUN_ID>/fit/stages/
#               pooled8/pooled8_anchor_config_s<seed>.json from a cluster run
#               (its own Steering summary becomes --reference unless given).
# --tag         persona label in the RUN_ID (default: the persona's parent
#               directory name, or the source run's kind-sha, e.g. pooled8-991900f).
# --buckets     run_eval buckets, space separated (default "steering"; the 14p
#               data also has 3 wide-to-narrow conditions: add id4scs_w2n).
# --reference   the fitting cohort's Steering/steering_condition_summary.csv,
#               shown side by side in the generalisation table (SUMMARY_14p).
# --min-runs    extra noise realisations per condition (default: one model run
#               per human round, 5 for most conditions).
#
# One run = one RUN_ID = one code snapshot = one results tree:
#   $RESULTS_ROOT/runs-14p/<RUN_ID>/{RUN_INFO.json,COMMIT,code/,personas/,eval/,logs/,tmp/}
# RUN_ID = eval14p-<tag>-s<seed>-<yyyymmdd-HHMM>-<sha7>. runs-14p/ is a
# separate tree from runs/ so eval/collect_runs.py never reads these as fit
# runs. The jobs execute from the snapshot in code/ (git archive of HEAD —
# eval-14p/ must be committed, or pass --allow-dirty to rsync the working
# tree); the snapshot's eval-14p/human_data is a symlink to this checkout's.
#
# Run from the repo root on the login node (venv/ from setup.sh).

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PERSONA=""; SOURCE_RUN=""; TAG=""; SEED=42; MIN_RUNS=0; BUCKETS="steering"; WALL=""
PARTICIPANTS_FILE="eval-14p/participants_14p.txt"; REFERENCE=""; ALLOW_DIRTY=0; DRY=0
RESULTS_ROOT="${RESULTS_ROOT:-/home/xiangyz/ondemand/data/sys/myjobs/projects/chi-27/results}"
VENV_DIR="$REPO_ROOT/venv"
DATA_DIR="eval-14p/human_data/raw"
while [ $# -gt 0 ]; do
    case "$1" in
        --persona) PERSONA="$2"; shift 2;;
        --source-run) SOURCE_RUN="$2"; shift 2;;
        --tag) TAG="$2"; shift 2;;
        --seed) SEED="$2"; shift 2;;
        --min-runs) MIN_RUNS="$2"; shift 2;;
        --buckets) BUCKETS="$2"; shift 2;;
        --wall) WALL="$2"; shift 2;;
        --participants) PARTICIPANTS_FILE="$2"; shift 2;;
        --reference) REFERENCE="$2"; shift 2;;
        --results-root) RESULTS_ROOT="$2"; shift 2;;
        --venv) VENV_DIR="$2"; shift 2;;
        --allow-dirty) ALLOW_DIRTY=1; shift;;
        --dry-run) DRY=1; shift;;
        -h|--help) sed -n 2,38p "$0"; exit 0;;
        *) echo "unknown option $1"; exit 2;;
    esac
done

# --- persona ----------------------------------------------------------------
if [ -n "$SOURCE_RUN" ] && [ -n "$PERSONA" ]; then echo "give --persona or --source-run, not both"; exit 2; fi
if [ -n "$SOURCE_RUN" ]; then
    SRC_DIR="$RESULTS_ROOT/runs/$SOURCE_RUN"
    PERSONA="$SRC_DIR/fit/stages/pooled8/pooled8_anchor_config_s${SEED}.json"
    [ -f "$PERSONA" ] || { echo "no pooled8 persona in $SRC_DIR (fit/stages/pooled8/pooled8_anchor_config_s${SEED}.json)"; exit 2; }
    [ -z "$REFERENCE" ] && [ -f "$SRC_DIR/eval/Steering/steering_condition_summary.csv" ] \
        && REFERENCE="$SRC_DIR/eval/Steering/steering_condition_summary.csv"
    if [ -z "$TAG" ]; then
        # mpcc-full-pooled8-s42-20260910-1540-991900f -> pooled8-991900f
        TAG="$(echo "$SOURCE_RUN" | awk -F- '{print $3"-"$NF}')"
    fi
elif [ -z "$PERSONA" ]; then
    PERSONA="eval-14p/personas/pooled8-991900f/default.json"
    [ -z "$REFERENCE" ] && [ -f "eval-14p/personas/pooled8-991900f/reference_steering_condition_summary.csv" ] \
        && REFERENCE="eval-14p/personas/pooled8-991900f/reference_steering_condition_summary.csv"
fi
[ -f "$PERSONA" ] || { echo "missing persona $PERSONA"; exit 2; }
[ -z "$TAG" ] && TAG="$(basename "$(cd "$(dirname "$PERSONA")" && pwd)")"
TAG="${TAG//[^A-Za-z0-9_.]/-}"
[ -f "$PARTICIPANTS_FILE" ] || { echo "missing $PARTICIPANTS_FILE"; exit 2; }
N_PIDS=$(grep -c . "$PARTICIPANTS_FILE")
[ -d "$DATA_DIR" ] || { echo "missing data dir $DATA_DIR"; exit 2; }
N_FILES=$(ls "$DATA_DIR"/*.json 2>/dev/null | wc -l | tr -d ' ')
[ "$N_FILES" -ge "$N_PIDS" ] || echo "WARNING: $N_FILES data files in $DATA_DIR for $N_PIDS participants"
[ -n "$REFERENCE" ] && { [ -f "$REFERENCE" ] || { echo "missing --reference $REFERENCE"; exit 2; }; REFERENCE="$(cd "$(dirname "$REFERENCE")" && pwd)/$(basename "$REFERENCE")"; }
[ -x "$VENV_DIR/bin/python" ] || { echo "missing venv at $VENV_DIR (bash setup.sh)"; exit 2; }
for b in $BUCKETS; do case "$b" in steering|id4scs_w2n|id4scs_n2w|fitts|c2u) ;; *) echo "bad bucket $b"; exit 2;; esac; done

SHA=$(git rev-parse --short=7 HEAD)
DIRTY=0
if [ -n "$(git status --porcelain --untracked-files=no)" ] || ! git ls-files --error-unmatch eval-14p/cluster/eval_job.sh >/dev/null 2>&1; then
    DIRTY=1
    if [ "$ALLOW_DIRTY" != 1 ]; then
        echo "working tree has uncommitted changes to tracked files, or eval-14p/ is not committed; commit or pass --allow-dirty"; exit 2
    fi
fi
STAMP=$(date +%Y%m%d-%H%M)
RUN_ID="eval14p-${TAG}-s${SEED}-${STAMP}-${SHA}"
RUN_DIR="$RESULTS_ROOT/runs-14p/$RUN_ID"
[ -e "$RUN_DIR" ] && { echo "run dir exists: $RUN_DIR (wait a minute or change --tag)"; exit 2; }
echo "RUN_ID    $RUN_ID"
echo "RUN_DIR   $RUN_DIR"
echo "PERSONA   $PERSONA"
echo "DATA      $DATA_DIR ($N_FILES files, $N_PIDS participants)"
echo "BUCKETS   $BUCKETS   min_runs $MIN_RUNS   seed $SEED   code $SHA$([ "$DIRTY" = 1 ] && echo ' (dirty/rsync)')"
echo "REFERENCE ${REFERENCE:-none}"
[ "$DRY" = 1 ] && { echo "(dry run: nothing created)"; exit 0; }

# --- run tree + code snapshot -------------------------------------------------
mkdir -p "$RUN_DIR"/{code,personas,eval,logs,tmp}
if [ "$DIRTY" = 1 ]; then
    rsync -a --exclude .git --exclude 'results' --exclude 'results-*' --exclude venv --exclude .venv \
          --exclude '__pycache__' --exclude 'human_data' --exclude 'backup' \
          --exclude 'model_fitting*' --exclude 'eval-14p/experiment-main/results' \
          --exclude 'eval-14p/video' --exclude 'eval-14p/figures' --exclude 'eval-14p/unit_experiment' \
          ./ "$RUN_DIR/code/"
else
    git archive HEAD | tar -x -C "$RUN_DIR/code"
fi
for f in eval-14p/cluster/eval_job.sh eval-14p/cluster/aggregate_job.sh eval-14p/summarize_14p.py eval/eval-main/run_eval.py; do
    [ -f "$RUN_DIR/code/$f" ] || { echo "snapshot lacks $f (commit eval-14p/ or --allow-dirty)"; rm -rf "$RUN_DIR"; exit 2; }
done
git rev-parse HEAD > "$RUN_DIR/COMMIT"
# human data: point the snapshot at this checkout's copies (as submit_run.sh does)
rm -rf "$RUN_DIR/code/human_data"; ln -s "$REPO_ROOT/human_data" "$RUN_DIR/code/human_data"
rm -rf "$RUN_DIR/code/eval-14p/human_data"; ln -s "$REPO_ROOT/eval-14p/human_data" "$RUN_DIR/code/eval-14p/human_data"
cp "$PARTICIPANTS_FILE" "$RUN_DIR/code/eval-14p/participants_14p.txt"
# ONE persona for every participant: run_eval resolves default.json when no {pid}.json exists
python3 cluster/stage_persona.py "$PERSONA" "$RUN_DIR/personas/default.json" mpcc pooled8
cp "$PERSONA" "$RUN_DIR/personas/source_persona.json"

EXPORTS="ALL,RUN_DIR=$RUN_DIR,SEED=$SEED,MIN_RUNS=$MIN_RUNS,BUCKETS=$BUCKETS,PARTICIPANTS_FILE=eval-14p/participants_14p.txt,DATA_DIR=$DATA_DIR,VENV_DIR=$VENV_DIR,REFERENCE_CSV=$REFERENCE"

# --- submit the chain ---------------------------------------------------------
WALL_ARG=(); [ -n "$WALL" ] && WALL_ARG=(--time "$WALL")
EVAL_JOB=$(sbatch --parsable --job-name "$RUN_ID" --array="1-$N_PIDS" ${WALL_ARG[@]+"${WALL_ARG[@]}"} \
    --output "$RUN_DIR/logs/eval_%A_%a.out" --error "$RUN_DIR/logs/eval_%A_%a.err" \
    --export="$EXPORTS" "$RUN_DIR/code/eval-14p/cluster/eval_job.sh")
EVAL_JOB="${EVAL_JOB%%;*}"
AGG_JOB=$(sbatch --parsable --job-name "agg-$RUN_ID" \
    --dependency="afterok:$EVAL_JOB" \
    --output "$RUN_DIR/logs/agg_%j.out" --error "$RUN_DIR/logs/agg_%j.err" \
    --export="$EXPORTS" "$RUN_DIR/code/eval-14p/cluster/aggregate_job.sh")
AGG_JOB="${AGG_JOB%%;*}"

# --- record -------------------------------------------------------------------
python3 - "$RUN_DIR" <<PYEOF
import json, sys, datetime, os
d = sys.argv[1]
info = {
  "run_id": "$RUN_ID", "kind": "eval14p", "model": "mpcc", "seed": $SEED,
  "persona": "$PERSONA", "persona_tag": "$TAG", "source_run": "${SOURCE_RUN:-}",
  "persona_fit": json.load(open(os.path.join(d, "personas", "default.json"))).get("_fit", {}),
  "commit": open(os.path.join(d, "COMMIT")).read().strip(), "dirty": bool($DIRTY),
  "submitted": datetime.datetime.now().isoformat(timespec="seconds"),
  "participants_file": "$PARTICIPANTS_FILE", "n_participants": $N_PIDS,
  "data_dir": "$DATA_DIR", "buckets": "$BUCKETS", "min_runs": $MIN_RUNS,
  "wall": "${WALL:-default}", "reference_csv": "${REFERENCE:-}", "venv": "$VENV_DIR",
  "jobs": {"eval": "$EVAL_JOB", "aggregate": "$AGG_JOB"},
  "cmdline": "eval-14p/submit_eval_14p.sh --persona $PERSONA --tag $TAG --seed $SEED --min-runs $MIN_RUNS --buckets '$BUCKETS'",
}
json.dump(info, open(os.path.join(d, "RUN_INFO.json"), "w"), indent=2)
PYEOF
INDEX="$RESULTS_ROOT/runs-14p/INDEX.tsv"
[ -f "$INDEX" ] || printf 'run_id\tpersona_tag\tseed\tcommit\tdirty\tsubmitted\tn_participants\teval_job\tagg_job\n' > "$INDEX"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$RUN_ID" "$TAG" "$SEED" "$SHA" "$DIRTY" "$STAMP" "$N_PIDS" "$EVAL_JOB" "$AGG_JOB" >> "$INDEX"
echo "submitted: eval $EVAL_JOB (array 1-$N_PIDS) -> aggregate $AGG_JOB"
echo "logs:      $RUN_DIR/logs/"
echo "results:   $RUN_DIR/eval/  (Steering/, SUMMARY_14p.*, DONE when finished)"

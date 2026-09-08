#!/bin/bash
# Re-run ONLY the eval + aggregate of an existing run whose fit completed
# (e.g. after an eval-side fix), keeping its fitted personas and RUN_ID.
#
#   resubmit_eval.sh <RUN_DIR> [--min-runs N]
#
# The run's code snapshot keeps the fitting code it was fitted with; only
# the cluster job scripts (cluster/*.sh, cluster/*.py) and eval-side Python
# (eval/eval-main, eval/utils, eval/collect_runs.py, eval/eval-gaze-lead,
# eval/eval-gaze-cursor) are refreshed from THIS checkout, and the
# snapshot's human_data/ is re-pointed at this checkout's directory as a
# whole. What was refreshed and from which commit is recorded in
# RUN_INFO.json ("eval_resubmits"), so the run stays auditable.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
RUN_DIR="${1:?usage: resubmit_eval.sh <RUN_DIR> [--min-runs N]}"; shift
MIN_RUNS=0
while [ $# -gt 0 ]; do
    case "$1" in
        --min-runs) MIN_RUNS="$2"; shift 2;;
        *) echo "unknown option $1"; exit 2;;
    esac
done
[ -f "$RUN_DIR/RUN_INFO.json" ] || { echo "not a run dir: $RUN_DIR"; exit 2; }
read -r MODEL VARIANT KIND SEED PARTICIPANTS_FILE N_PIDS < <(python3 -c "
import json,sys; j=json.load(open(sys.argv[1]+'/RUN_INFO.json'))
print(j['model'], j['variant'], j['kind'], j['seed'], j.get('participants_file','participants_10p.txt'), j.get('n_participants', 8))" "$RUN_DIR")
VENV_DIR="$REPO_ROOT/venv"
SHA=$(git rev-parse --short=7 HEAD)
CODE="$RUN_DIR/code"
[ -d "$CODE" ] || { echo "missing snapshot $CODE"; exit 2; }
STAGE=$([ "$KIND" = pooled8 ] && echo pooled8 || echo base)
ls "$RUN_DIR/fit/stages/$STAGE/"*_config_s"$SEED".json >/dev/null 2>&1 || { echo "no fitted personas under $RUN_DIR/fit/stages/$STAGE"; exit 2; }

# refresh eval-side code + job scripts in the snapshot (fitting code untouched)
for d in cluster eval/eval-main eval/utils eval/eval-gaze-lead eval/eval-gaze-cursor; do
    rsync -a --delete --exclude '__pycache__' --exclude 'results' --exclude 'human-gaze-lead*' \
          --exclude 'model-gaze-lead*' --exclude 'gl-*' "$REPO_ROOT/$d/" "$CODE/$d/"
done
cp "$REPO_ROOT/eval/collect_runs.py" "$CODE/eval/collect_runs.py"
rm -rf "$CODE/human_data"; ln -s "$REPO_ROOT/human_data" "$CODE/human_data"
rm -rf "$RUN_DIR/eval" "$RUN_DIR/gaze-lead" "$RUN_DIR/personas" "$RUN_DIR/DONE"
mkdir -p "$RUN_DIR/eval" "$RUN_DIR/gaze-lead" "$RUN_DIR/personas" "$RUN_DIR/logs" "$RUN_DIR/tmp"

EXPORTS="ALL,RUN_DIR=$RUN_DIR,MODEL=$MODEL,VARIANT=$VARIANT,KIND=$KIND,SEED=$SEED,MIN_RUNS=$MIN_RUNS,PARTICIPANTS_FILE=$PARTICIPANTS_FILE,VENV_DIR=$VENV_DIR"
EVAL_JOB=$(sbatch --parsable --job-name "eval-$(basename "$RUN_DIR")" --array="1-$N_PIDS" \
    --output "$RUN_DIR/logs/eval2_%A_%a.out" --error "$RUN_DIR/logs/eval2_%A_%a.err" \
    --export="$EXPORTS" "$CODE/cluster/eval_job.sh")
EVAL_JOB="${EVAL_JOB%%;*}"
AGG_JOB=$(sbatch --parsable --job-name "agg-$(basename "$RUN_DIR")" --dependency="afterok:$EVAL_JOB" \
    --output "$RUN_DIR/logs/agg2_%j.out" --error "$RUN_DIR/logs/agg2_%j.err" \
    --export="$EXPORTS" "$CODE/cluster/aggregate_job.sh")
AGG_JOB="${AGG_JOB%%;*}"
python3 - "$RUN_DIR" "$SHA" "$EVAL_JOB" "$AGG_JOB" <<'EOF'
import json, sys, datetime
d, sha, ev, ag = sys.argv[1:]
p = d + "/RUN_INFO.json"; j = json.load(open(p))
j.setdefault("eval_resubmits", []).append({
    "when": datetime.datetime.now().isoformat(timespec="seconds"), "eval_code_commit": sha,
    "refreshed": ["cluster", "eval/eval-main", "eval/utils", "eval/eval-gaze-lead", "eval/eval-gaze-cursor", "eval/collect_runs.py", "human_data link"],
    "jobs": {"eval": ev, "aggregate": ag}})
j.pop("superseded_by", None)
json.dump(j, open(p, "w"), indent=2)
EOF
echo "resubmitted eval $EVAL_JOB -> aggregate $AGG_JOB for $(basename "$RUN_DIR") (eval code $SHA)"

#!/bin/bash
# Cycle statistics (cluster/cycles_job.sh) for an EXISTING mpcc run whose fit
# completed — e.g. the September ablations, whose gaze-lead step failed.
#
#   submit_cycles.sh <RUN_DIR> [--wall HH:MM:SS] [--dry-run]
#
# The run keeps its own simulator and fitted personas (its code snapshot);
# only the analysis script (paper/quantify_gaze_cycles.py), the job script
# and the human cycle tables are copied in from THIS checkout. Outputs go to
# <RUN_DIR>/cycles/; the submission is recorded in RUN_INFO.json
# ("cycles_jobs"). Run from the repo root on the login node.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
RUN_DIR="${1:?usage: submit_cycles.sh <RUN_DIR> [--wall HH:MM:SS] [--dry-run]}"; shift
WALL=""; DRY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --wall) WALL="$2"; shift 2;;
        --dry-run) DRY=1; shift;;
        *) echo "unknown option $1"; exit 2;;
    esac
done
[ -f "$RUN_DIR/RUN_INFO.json" ] || { echo "not a run dir: $RUN_DIR"; exit 2; }
read -r MODEL KIND SEED PARTICIPANTS_FILE < <(python3 -c "
import json,sys; j=json.load(open(sys.argv[1]+'/RUN_INFO.json'))
print(j['model'], j['kind'], j['seed'], j.get('participants_file','participants_10p.txt'))" "$RUN_DIR")
[ "$MODEL" = mpcc ] || { echo "$(basename "$RUN_DIR") is a $MODEL run: the baseline has no anchors, so no cycles"; exit 2; }
CODE="$RUN_DIR/code"
[ -d "$CODE" ] || { echo "missing snapshot $CODE"; exit 2; }
STAGE=$([ "$KIND" = pooled8 ] && echo pooled8 || echo base)
ls "$RUN_DIR/fit/stages/$STAGE/"*_anchor_config_s"$SEED".json >/dev/null 2>&1 || { echo "no fitted personas under $RUN_DIR/fit/stages/$STAGE"; exit 2; }
HUM=""; for q in paper/quant_refit paper/quant; do [ -f "$q/cycles_human.csv" ] && [ -f "$q/curv_human.csv" ] && { HUM="$q"; break; }; done
[ -n "$HUM" ] || { echo "no human cycle tables in paper/quant_refit or paper/quant"; exit 2; }
VENV_DIR="$REPO_ROOT/venv"
SHA=$(git rev-parse --short=7 HEAD)
echo "RUN     $(basename "$RUN_DIR")  ($KIND, seed $SEED)"
echo "HUMAN   $HUM/{cycles_human,curv_human}.csv   analysis code $SHA"
[ "$DRY" = 1 ] && { echo "(dry run: nothing submitted)"; exit 0; }

mkdir -p "$RUN_DIR/cycles" "$RUN_DIR/logs" "$RUN_DIR/tmp" "$CODE/paper" "$CODE/cluster"
cp "$REPO_ROOT/paper/quantify_gaze_cycles.py" "$CODE/paper/quantify_gaze_cycles.py"
cp "$REPO_ROOT/cluster/cycles_job.sh" "$CODE/cluster/cycles_job.sh"
cp "$HUM/cycles_human.csv" "$HUM/curv_human.csv" "$RUN_DIR/cycles/"
rm -rf "$CODE/human_data"; ln -s "$REPO_ROOT/human_data" "$CODE/human_data"
JOB=$(sbatch --parsable --job-name "cyc-$(basename "$RUN_DIR")" ${WALL:+--time "$WALL"} \
    --output "$RUN_DIR/logs/cycles_%j.out" --error "$RUN_DIR/logs/cycles_%j.err" \
    --export="ALL,RUN_DIR=$RUN_DIR,KIND=$KIND,SEED=$SEED,PARTICIPANTS_FILE=$PARTICIPANTS_FILE,VENV_DIR=$VENV_DIR" \
    "$CODE/cluster/cycles_job.sh")
JOB="${JOB%%;*}"
python3 - "$RUN_DIR" "$SHA" "$JOB" <<'EOF2'
import json, sys, datetime
d, sha, job = sys.argv[1:]
p = d + "/RUN_INFO.json"; j = json.load(open(p))
j.setdefault("cycles_jobs", []).append({"when": datetime.datetime.now().isoformat(timespec="seconds"),
                                         "analysis_code_commit": sha, "job": job})
json.dump(j, open(p, "w"), indent=2)
EOF2
echo "submitted cycles job $JOB -> $RUN_DIR/cycles/"

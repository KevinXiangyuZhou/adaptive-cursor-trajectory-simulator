#!/bin/bash
# Local (no SLURM) eval-14p run: the same harness, persona and data as the
# cluster job, for a smoke test or a full run on a workstation.
#
#   eval-14p/run_local.sh [--pid P...] [--persona-dir DIR] [--tag NAME] [--min-runs N]
#                         [--buckets "steering"] [--aggregate-only] [--reference CSV]
#
# --pid            one participant only (any line of eval-14p/participants_14p.txt);
#                  without it every participant runs in turn, then the aggregate
# --persona-dir    directory holding default.json (default eval-14p/personas/pooled8-991900f)
# --tag            results subfolder (default: the persona dir's name)
# --aggregate-only rebuild the pooled outputs + SUMMARY_14p from the cached sims
# Outputs: eval-14p/results/<tag>/ (gitignored via results/).

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PID=""; PERSONA_DIR="eval-14p/personas/pooled8-991900f"; TAG=""; MIN_RUNS=0; BUCKETS="steering"; AGG=0; REFERENCE=""; SEED=42
while [ $# -gt 0 ]; do
    case "$1" in
        --pid) PID="$2"; shift 2;;
        --persona-dir) PERSONA_DIR="$2"; shift 2;;
        --tag) TAG="$2"; shift 2;;
        --min-runs) MIN_RUNS="$2"; shift 2;;
        --buckets) BUCKETS="$2"; shift 2;;
        --seed) SEED="$2"; shift 2;;
        --aggregate-only) AGG=1; shift;;
        --reference) REFERENCE="$2"; shift 2;;
        -h|--help) sed -n 2,14p "$0"; exit 0;;
        *) echo "unknown option $1"; exit 2;;
    esac
done
[ -f "$PERSONA_DIR/default.json" ] || { echo "missing $PERSONA_DIR/default.json"; exit 2; }
[ -z "$TAG" ] && TAG="$(basename "$PERSONA_DIR")"
[ -z "$REFERENCE" ] && [ -f "$PERSONA_DIR/reference_steering_condition_summary.csv" ] && REFERENCE="$PERSONA_DIR/reference_steering_condition_summary.csv"
PY="${PYTHON:-$REPO_ROOT/.venv/bin/python}"; [ -x "$PY" ] || PY=python3
DATA_DIR="eval-14p/human_data/raw"
RESULTS="$REPO_ROOT/eval-14p/results/$TAG"
mkdir -p "$RESULTS"
export MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

run_eval() {  # $@ = extra run_eval args
    # shellcheck disable=SC2086
    "$PY" -u eval/eval-main/run_eval.py --model mpcc --config-dir "$PERSONA_DIR" --seed "$SEED" \
        --data-dir "$DATA_DIR" --results-dir "$RESULTS" --buckets $BUCKETS "$@"
}
if [ "$AGG" = 0 ]; then
    if [ -n "$PID" ]; then
        run_eval --pid "$PID" --fresh-sim --min-runs "$MIN_RUNS" 2>&1 | tee "$RESULTS/eval_${PID}_s${SEED}.log"
    else
        while read -r P; do
            [ -n "$P" ] || continue
            run_eval --pid "$P" --fresh-sim --min-runs "$MIN_RUNS" 2>&1 | tee "$RESULTS/eval_${P}_s${SEED}.log"
        done < eval-14p/participants_14p.txt
    fi
fi
if [ "$AGG" = 1 ] || [ -z "$PID" ]; then
    run_eval --aggregate-only 2>&1 | tee "$RESULTS/eval_aggregate_s${SEED}.log"
    REF_ARG=(); [ -n "$REFERENCE" ] && REF_ARG=(--reference "$REFERENCE")
    "$PY" -u eval-14p/summarize_14p.py "$RESULTS" --data-dir "$DATA_DIR" --out "$RESULTS/SUMMARY_14p" ${REF_ARG[@]+"${REF_ARG[@]}"}
fi
echo "results: $RESULTS"

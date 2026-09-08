#!/bin/bash
# RETIRED (2026-09-08): this script pointed a shared symlink at a results dir
# and ran from the live working tree, so concurrent/queued runs collided.
# Submit isolated runs with the wrapper instead:
#   ./submit_run.sh --model {mpcc,baseline} --variant {full,no_gaze,no_lookahead,no_pace,no_intermittent} --kind {perpid,pooled8}
# Job bodies live in cluster/{fit_job,fit_pooled_job,eval_job,aggregate_job}.sh.
echo "fit_pooled8.sh is retired; use ./submit_run.sh (see header)" >&2
exit 2

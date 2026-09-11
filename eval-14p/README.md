# eval-14p — the pooled model on 14 new participants

Generalisation test: the persona fitted on the earlier 8-participant gaze
cohort is run, unchanged, on 14 participants it never saw, with the same
harness and metrics as the main evaluation.

## What is here

| path | what |
|---|---|
| `human_data/raw/` | the 14 new participants (13 Prolific `participant_P<hex>.json` with a `sessions` list, 1 lab `steering_experiment_P204813_*.json`). `stale/`, `filtered-out/`, `backup_individual/` are excluded participants and are not read. |
| `participants_14p.txt` | the 14 participant ids, one per line (SLURM array index = line) |
| `personas/pooled8-991900f/default.json` | the persona under test: the pooled8 fit of run `mpcc-full-pooled8-s42-20260910-1540-991900f` (see `SOURCE.md`); `reference_steering_condition_summary.csv` is that run's own steering summary on the 8 fitting participants |
| `submit_eval_14p.sh` | Great Lakes submission: isolated run dir, eval array (14 tasks) then aggregate |
| `cluster/eval_job.sh`, `cluster/aggregate_job.sh` | the SLURM job bodies (never `sbatch` them directly) |
| `run_local.sh` | the same pipeline without SLURM |
| `summarize_14p.py` | generalisation tables from the harness outputs |
| `experiment-main/`, `model_fitting/`, `baseline_fitting/`, `unit_experiment/`, `utils/`, `video/`, `chi-26-ea_baseline_pacakage/` | the CHI-26 EA project's evaluation code and its per-participant GAM fits of these 14 participants. **Not used** by the pipeline above: its `run_eval.py` expects the old package API, old paths (`eval/human_data/raw`) and per-participant `*_gam_config_s42.json` personas. Kept for reference; `experiment-main/results/` holds the old baseline results. |

## The task data

The 14p sessions are the CHI-26 steering protocol: 25 constant-width
tunnels (straight, gentle sinusoidal, sinusoidal, sharp sinusoidal, corner
× widths 10 / 20 / 30 / 40 / 50 mm; 5 rounds each, 3 for straight), 3
wide-to-narrow tunnels (tids 26–28), 3 lasso and 6 cascading-menu tasks.
There is no pointing. The files are in the format `eval/eval-main/run_eval.py`
already reads (nested `sessions` or flat `trialData`; trajectories as
`{x, y}` metres, timestamps in ms), so nothing is converted.

The harness buckets each trial by its `condition` keys: the 25 tunnels are
`steering`, wide-to-narrow is `id4scs_w2n` (off by default, like the main
evaluation; `--buckets "steering id4scs_w2n"` adds it), lasso and menu have
no width/distance key and are excluded. Tunnel geometry is rebuilt from the
condition (`build_steering_task_config`) exactly as for the 8p data, whose
sinusoidal/corner/straight families share the same curvature and corner
parameters — only the widths differ: the fit cohort had 10 / 12.5 / 16.5 /
25 / 50 mm, so 10 and 50 mm are shared and 20 / 30 / 40 mm are new.

Rounds longer than 60 s are dropped (harness rule); the old CHI-26 harness
used 20 s. In the 14p steering data 17 of 1610 rounds exceed 20 s and none
exceed 60 s, so every round is kept.

## Run on Great Lakes

```bash
# once: the tracked persona copy; eval-14p/ must be committed (git archive
# snapshot) or pass --allow-dirty
eval-14p/submit_eval_14p.sh                       # defaults: persona pooled8-991900f, seed 42, steering
eval-14p/submit_eval_14p.sh --source-run mpcc-full-pooled8-s42-20260910-1540-991900f   # stage from the cluster run instead
eval-14p/submit_eval_14p.sh --min-runs 10 --wall 01:30:00                             # more noise draws per condition
eval-14p/submit_eval_14p.sh --dry-run
```

Run tree: `$RESULTS_ROOT/runs-14p/<RUN_ID>/` with `RUN_ID =
eval14p-<persona tag>-s<seed>-<stamp>-<sha7>` (`runs-14p/` is separate
from `runs/` so `eval/collect_runs.py` never treats it as a fit run):

```
RUN_INFO.json  COMMIT  code/ (snapshot; eval-14p/human_data -> this checkout)
personas/default.json        the staged persona (add_noise on), source_persona.json
eval/Steering/participant_<pid>/trial_<tid>/   per-trial plots + results_summary.json
eval/Steering/steering_results.csv, steering_condition_summary.csv, steering_law.pdf
eval/SUMMARY_14p.txt / .csv  generalisation tables (below)
eval/eval_<pid>_s42.log, eval_aggregate_s42.log, summary_14p.log
logs/eval_<job>_<n>.out|err, agg_<job>.out|err     DONE when finished
```

Each array task evaluates one participant (`run_eval.py --pid`, fresh
simulation, one model run per human round unless `--min-runs`); 25
conditions / 115 runs take a few minutes on 4 cores (about a minute on a
laptop), no pointing, no gaze-lead step. The aggregate rebuilds the pooled
Steering outputs from the cached sims and writes `SUMMARY_14p`.

## Locally

```bash
eval-14p/run_local.sh --pid P204813 --tag check     # one participant -> eval-14p/results/check/
eval-14p/run_local.sh                               # all 14 in turn, then aggregate + SUMMARY_14p
eval-14p/run_local.sh --aggregate-only --tag check
```

## Reading the result

`SUMMARY_14p.txt` lists, for the new cohort and (with `--reference`) the
fitting cohort in-sample: the Steering-Law fit (MT = a + b·L/W; one point
per condition averaged over participants) and, per tunnel type × width,
per width (marked shared / new), per type, overall, and per participant:

- `ratio` — model MT / human MT, mean over participant × condition rows
- `latRMSEmm` — lateral RMSE of the model trajectory against the human one
- `spdRMSE`, `spdCorr` — speed-profile RMSE (m/s) and correlation over progress
- `relTdiff` — |model − human| completion time relative to the human time
- `timeouts` — model runs that hit the step limit

The question the run answers is how much these degrade from the reference
(in-sample) rows to the new participants, and whether the new widths
(20 / 30 / 40 mm) degrade more than the shared ones (10 / 50 mm).

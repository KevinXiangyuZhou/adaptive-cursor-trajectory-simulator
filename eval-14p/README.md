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

### Protocol checks (2026-09-11)

Geometry: every 14p human sample of all 28 tunnel conditions lies inside
the tunnel the harness rebuilds from the condition dict (100% within
W/2 · 1.05), every round starts on the centerline start, and the arc
lengths equal the 8p ones per family (507.2 / 660.0 / 458.0 / 476.6 /
626.2 mm). The current `experiment/environment.py` reproduces the CHI-26
tunnels.

Goal rule — the one real difference: a CHI-26 trial ends when the cursor is
within a FIXED 10 mm of the tunnel end (last samples within 10.00 mm at
every width, 322 rounds per width), whereas the current web experiment ends
within W/2 (8p last samples within exactly W/2), which is also the harness's
default for the model. With the default the model would travel 5 mm farther
than the humans at W = 10 mm and stop up to 15 mm earlier at W = 50 mm. The
pipeline therefore runs the harness with `--tunnel-target-radius 0.01`
(env `TUNNEL_TARGET_RADIUS`, recorded in RUN_INFO.json and in the
`target_radius_mm` column of the summary CSV). The override changes only
the stop rule: the waypoint spacing of the reference path the model
receives stays at the width rule (W/4, what the persona was fitted with),
and without the flag the harness output is byte-identical to before. On
P204813 the correction moves the model/human time ratio by about −0.1 at
10 mm and +0.1 at 50 mm. The first cluster run (before this flag existed)
used W/2 — rerun.

## Run on Great Lakes

```bash
# once: the tracked persona copy; eval-14p/ must be committed (git archive
# snapshot) or pass --allow-dirty
eval-14p/submit_eval_14p.sh                       # defaults: persona pooled8-991900f, seed 42, steering
# evaluate other pooled runs: persona, model (mpcc / baseline) and the 8p
# reference summary come from the run itself; tag = <model>-<kind>-<sha7>
eval-14p/submit_eval_14p.sh --source-run mpcc-full-pooled8-s42-20260910-1540-991900f
eval-14p/submit_eval_14p.sh --source-run mpcc-full-pooled8-s42-20260911-0334-c70a5dd
eval-14p/submit_eval_14p.sh --source-run mpcc-full-pooled8all-s42-20260911-0335-c70a5dd
eval-14p/submit_eval_14p.sh --source-run baseline-full-pooled8all-s42-20260911-0336-c70a5dd --wall 02:00:00   # baseline: ~58 core-min per participant
eval-14p/submit_eval_14p.sh --min-runs 10 --wall 01:30:00                             # more noise draws per condition
eval-14p/submit_eval_14p.sh --dry-run
TUNNEL_TARGET_RADIUS=0.01 eval-14p/submit_eval_14p.sh   # the default; the CHI-26 goal rule (see Protocol checks)
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
conditions / 115 runs take ~5 min on 4 cores for the MPCC model (about a
minute on a laptop; the baseline replans every step, ~3x longer), no
pointing, no gaze-lead step; wall 1 h. A `--source-run` whose fit is still
in the queue is accepted: the eval array waits for the fit job (afterok)
and stages the persona itself; the source run's 8p summary is used as the
reference if it exists when the aggregate runs. The aggregate (4 cores, 30 min
wall) rebuilds the pooled Steering outputs from the cached sims with
`--no-trial-plots` (the per-trial figures were already written by the eval
tasks) and writes `SUMMARY_14p` — one to two minutes. The first aggregate
overran 30 min on one core: the progress resampling in `eval/utils/stats.py`
was a point-by-segment Python loop (~4 s per condition, ~23 core-min for
350 conditions) and the pool spawned one worker per node core. It is now
vectorised (bit-identical results, ~150x faster), and the worker pool
follows the SLURM allocation (`run_eval.n_pool_workers`).

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

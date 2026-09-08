# Plan: baseline refit (per-participant + pooled) and ablation fits

Goal: produce Steering-Law and Fitts-Law lines, trajectory metrics, and cycle
statistics for (B) the CHI-EA baseline and (C) ablations of the current
model, under exactly the protocol used for (A) the full model, so the paper
can compare them on equal footing.

## 0. Comparison matrix

| Row | Model | Horizon | Catch-up time | Replanning | Fitted params |
|-----|-------|---------|---------------|------------|---------------|
| A   | Full model (done: `anchor_fitting_10p`, `pooled8`) | budget (D0, γ=0.66) | pace-law integral | arrival + latency | jerk, contour, constraint, goal, D0 (+ post-fit T0) |
| B   | CHI-EA baseline (`eval/chi-26-ea_baseline_pacakage`) | fixed Th | n/a (tracks `v_des/(1+c·|κ|)`) | every step | jerk, progress, wall, contour, lag, desired_speed, Th, curvature_scale |
| C1  | no gaze module | fixed lead | constant T0 | every step | jerk, contour, constraint, goal, fixed_lead_m, T0 |
| C2  | no adaptive lookahead | fixed lead | pace-law integral | arrival + latency | jerk, contour, constraint, goal, fixed_lead_m |
| C3  | no pace law | budget | constant T0 | arrival + latency | jerk, contour, constraint, goal, D0, T0 |
| C4  | no intermittency | budget | pace-law integral | every step | jerk, contour, constraint, goal, D0 |

C1–C3 remove one mechanism each of the three named in the Introduction
(where to commit, how long, when to replan); C1 removes the gaze module
entirely and is the in-framework analogue of B. Every row gets at least as
many free parameters as A.

## 1. Shared protocol (all rows)

- Data: `human_data/task_aligned_all`, eight participants in
  `participants_10p.txt`; `fsm.load_participant`, `fsm.split_tunnel`,
  `fsm.split_pointing` (same train/test widths and radii as A).
- Loss: `fsm.tunnel_loss` (lateral RMSE, speed RMSE, speed corr, |log CT|)
  + `w_pt ·` `fsm.pointing_loss` + noise-on stability penalty, human-variability
  scaled — i.e. `fit_anchor._tunnel_part / _pointing_part / _noise_stability`.
- Optimiser: `fsm.run_cmaes`, popsize 12, sigma0 0.2, seed 42,
  `TIME_LIMIT=23400` (6.5 h CMA in an 8 h wall). Fits run noiseless with
  latency CV 0; the SAVED persona gets `add_noise=True` and
  `replan_latency_cv=0.89` back (as `fit_anchor` already does).
- Parity constants: identical `nc`, `forearm`, `Interval`, `dwell_s=0`,
  `acc_max` where applicable. Check the EA defaults (`nc=[0.2,0.02]`,
  `forearm=0.357`) against the 10p base personas and align to the personas.
- Outputs: `<RESULTS_ROOT>/<fit_dir>/stages/<tag>/{pid}_<model>_config_s42.json`
  + `_fit_s42.json` (fitted params, train/test loss, history). Pooled:
  `stages/pooled8/`.
- Evaluation: `eval_10p.sh` → `eval/eval-main/run_eval.py` per participant,
  then `eval_10p_aggregate.sh` → `steering_law.pdf`,
  `fitts_regression_plot.png`, `steering_condition_summary.csv`,
  `fitts_regression.json`; trajectory-metric table via the same summary
  CSVs. Model gaze-lead analysis (`model_gaze_lead.py`) runs for A and C
  (anchors exist), skipped for B.

## 2. Baseline (row B)

### 2.1 Port `eval/baseline_fitting/fit_baseline.py` to the current pipeline
1. Drop the imports from `eval.model_fitting.run_fitting` (stale 8-25
   pipeline). Import `fit_speed_model as fsm` and `run_eval as em` the way
   `eval/eval-anchor-drive/fit_anchor.py` does (`probe_anchor` sets paths).
2. Keep `_load_baseline_cls()` (sys.path / sys.modules swap: both packages
   are named `hcs_package`). Load the class ONCE in the parent before the
   worker pool forks and pass it through the shared tuple — do not re-import
   inside workers.
3. Keep `_run_baseline_sim(sim, task_config)`; it already consumes the same
   task-config dicts that `fsm.build_tunnel_tasks` and
   `em.build_fitts_bypass_config` emit (`waypoints`, `constraints`,
   screen dims, `target_radius`, `max_steps`). Verify the EA
   `generate_trajectory_with_waypoints` (a) honours `target_radius` as the
   termination (final target entry, no dwell) and (b) parses the path
   constraint block of the current task JSON. Patch the EA package only if
   one of these fails, and log the patch in this file.
4. Parameter spec (8, all log-scale except Th):
   `jerk, progress, wall, contour, lag, desired_speed, Th, curvature_scale`.
   Add `curvature_scale` with bounds log10 ∈ [0, 2] (1–100; default 10).
   Keep the existing bounds for the other seven.
5. Objective: refactor `fit_anchor._tunnel_part`, `_pointing_part`,
   `_noise_stability` to take a `make_sim(cfg)` and `run_sim(sim, tc)` pair
   (default = `fsm._make_sim` / `fsm.run_single_sim`), then call them from
   `fit_baseline` with the EA class and `_run_baseline_sim`. One loss
   implementation, two simulators.
6. Pointing: use `em.build_fitts_bypass_config` for every pointing round
   (same 10 m bypass corridor, same start point, same radius). No post-fit
   T0 scan for B (it has no free-space deadline); Th is in the CMA search.
7. Base config: EA defaults with `add_noise=False`, `nc`/`forearm` from the
   participant's 10p base persona, `random_seed` fixed. Save to
   `stages/base/{pid}_baseline_config_s42.json` with noise restored.
8. CLI mirrors `fit_anchor`: `--pid --time-limit --popsize --workers --seed
   --quick --tag --override`.

### 2.2 Pooled baseline
`eval/baseline_fitting/fit_baseline_pooled8.py`, modelled on
`eval/eval-anchor-drive/fit_anchor_pooled8.py`: one parameter set, loss =
mean over the eight participants of (tunnel + w_pt·pointing + stability),
work unit = (candidate × participant), training data loaded once in the
parent. Output `stages/pooled8/pooled_baseline_config_s42.json`.

### 2.3 Cluster scripts
- `fit_baseline_10p.sh`: copy of `fit_anchor_10p.sh` with array 1–8 calling
  `python3 eval/baseline_fitting/fit_baseline.py`; all paths from `RUN_DIR`
  (§6.4), no symlinks, no hard-coded result dirs.
- `fit_baseline_pooled8.sh`: copy of `fit_pooled8.sh` (36 CPUs, 8 h wall,
  5 h CMA budget), same conventions.
- Submitted only through `submit_run.sh --model baseline` (§6.4).

### 2.4 Evaluation hook
- `eval/eval-main/run_eval.py`: `--model {mpcc,baseline}` (default `mpcc`)
  — DONE. For `baseline` the simulator sites construct the EA class via
  `eval/utils/baseline_loader.py` (shared with fitting). Default buckets are
  now `steering fitts`: ID4SCS and constrained-to-unconstrained tasks are
  out of the paper's scope (2026-09-08) and are also dropped from the
  fitting held-out split.
- `eval_10p.sh` / `eval_10p_aggregate.sh`: `MODEL=${MODEL:-mpcc}`; when
  `baseline`, read personas from `$RESULTS_ROOT/baseline_fitting_10p/stages/base`,
  write to `eval-main-baseline-10p`, and skip the `model_gaze_lead.py` step.
  Pooled eval: persona dir with the single pooled config copied under each
  pid name (as done for `eval-main-pooled8-local`).

### 2.5 Smoke tests before submitting
```
python eval/baseline_fitting/fit_baseline.py --pid p01 --quick --time-limit 120 --popsize 4 --workers 4 --seed 7
python eval/eval-main/run_eval.py --model baseline --pid P103405 --config-dir <smoke persona dir> --buckets steering --fresh-sim
```
Expect: 0 timeouts, a saved config with `add_noise: true`, and steering
trajectories that complete. Do not leave seed-7 outputs in `results/`.

## 3. Ablations (rows C1–C4, current simulator)

### 3.1 Simulator switches (gated; default = current behaviour, bit-identical)
Implement in `hcs_package/src/hcs_package/gaze_module.py` and
`cursor_simulator.py`. Reference for the removed variants:
`git show s14-variant-graveyard:hcs_package/src/hcs_package/cursor_simulator.py`.

1. `horizon_mode: "fixed_lead"` — anchor = `min(s0 + fixed_lead_m, path_end)`,
   bypassing `DifficultyBudgetHorizon.anchor`. New top-level key
   `fixed_lead_m`. Keep the reaction-time floor (`v·T_min`) so C2/C1 differ
   from A only in the budget. Update `_reject_pruned` to accept
   `"fixed_lead"` (keep refusing the old `"fixed"`).
2. `catchup_mode: "constant"` — skip `_traversal_time`; `t_plan = max(T0,
   lead/plan_vmax, t_acc)` everywhere (the free-space rule applied in
   corridors too). Default `"pace_law"` = current behaviour.
3. `replan_mode: "every_step"` — already implemented; no change.

Add a unit test that runs the S14 probes with all new keys absent and
asserts bit-identical trajectories (same pattern as the module-split
refactor).

### 3.2 Fitting: `--ablation` flag in `fit_anchor.py` and `fit_anchor_pooled8.py`
Map each name to (persona override, CMA spec):

| `--ablation` | override | spec change |
|---|---|---|
| `no_gaze` (C1) | `horizon_mode=fixed_lead, catchup_mode=constant, replan_mode=every_step` | − D0; + `fixed_lead_m` [0.01, 0.15], + `plan_deadline_s` [0.08, 0.40]; skip T0 scan |
| `no_lookahead` (C2) | `horizon_mode=fixed_lead` | − D0; + `fixed_lead_m` |
| `no_pace` (C3) | `catchup_mode=constant` | + `plan_deadline_s`; skip T0 scan |
| `no_intermittent` (C4) | `replan_mode=every_step` | none |

Tag → `stages/<ablation>/`. `fsm.apply_params` already handles top-level and
`budget` keys; add `fixed_lead_m` as a top-level key.

### 3.3 Cluster
One `submit_run.sh --model mpcc --variant <ablation> --kind {perpid,pooled8}`
per row (§6.4): four per-participant arrays (8 tasks × 8 h each) and four
pooled jobs, each in its own RUN_ID tree with its own code snapshot, so all
eight can sit in the queue together. If the budget is tight, submit C1 and
C3 per-participant first (the two the Introduction's claim depends on most)
and all four pooled.

### 3.4 Evaluation
`RUN_TAG=<ablation> sbatch eval_10p.sh` then the aggregate; `model_gaze_lead.py`
runs as usual so the cycle laws (saccade exponent, catch-up exponent,
overrun) are available for every ablation.

## 4. Reporting targets

- Table 1 (laws): row × {steering slope, R²; Fitts slope, R², throughput;
  CT ratio train / test}.
- Table 2 (trajectory): row × {lateral RMSE, speed RMSE, speed corr} on
  held-out widths, all corridor types pooled.
- Table 3 (cycle laws, A and C only): row × {saccade-distance exponent,
  catch-up exponent, overrun fraction}.
- Figures: extend `steering_law` and `fitts_regression_plot` plotting to
  overlay multiple result directories (human, A, B, optionally C1).

## 5. Order of work

1. Run isolation (§6): `submit_run.sh`, env-driven job scripts, symlink
   removal, `RUN_INFO.json`/`INDEX.tsv`, `collect_runs.py`. Verify with a
   two-run smoke on the cluster (two `--quick` fits submitted back to back
   with a code edit in between; confirm each ran its own snapshot and wrote
   only its own tree) (½ day).
2. Ablation switches + bit-identity test (½ day).
3. `fit_baseline` port + shared-objective refactor + smoke (½ day).
4. Eval `--model baseline` hook + loader util + smoke (¼ day).
5. Commit, then submit in one sitting so every run shares one commit:
   B perpid, B pooled8, C1/C3 perpid, C1–C4 pooled8 (add C2/C4 perpid if
   time allows). Each submission chains fit → eval → aggregate.
6. `collect_runs.py` → `SUMMARY.csv`; build Tables 1–3 and overlay figures
   from named RUN_IDs; record them in `paper/RUNS_USED.md`.

## 6. Run isolation on the cluster (required before queuing anything)

Many fits and evals will sit in the queue at once and start hours apart.
Two things must hold for every job: it runs the code that was current when
it was submitted, and it writes only into a directory that no other job
writes into.

### 6.1 One run = one RUN_ID = one code snapshot = one results tree
- `RUN_ID = <model>-<variant>-<kind>-<seed>-<yyyymmdd-HHMM>-<sha7>`, e.g.
  `mpcc-no_pace-perpid-s42-20260908-1410-ced27fc`,
  `baseline-full-pooled8-s42-20260908-1412-ced27fc`. `variant` is the
  ablation name or `full`; `kind` is `perpid` or `pooled8`.
- Results tree, all under one root:
  ```
  $RESULTS_ROOT/runs/<RUN_ID>/
      RUN_INFO.json     # model, variant, kind, seed, commit, dirty flag, submit time, job ids, cmdline
      code/             # git snapshot the jobs execute from (see 6.2)
      fit/stages/...    # fit outputs (was anchor_fitting_10p/stages/...)
      personas/         # noise-restored personas consumed by eval
      eval/             # eval-main outputs incl. sim_cache
      gaze-lead/        # model_gaze_lead outputs (A and C only)
      logs/             # SLURM .out/.err for every job of this run
  ```
  Nothing else writes here; no other run reads from here. A run dir that
  already exists aborts submission instead of being reused.

### 6.2 Code snapshot per run
- The submit wrapper (6.4) does `git archive HEAD | tar -x -C
  $RUN_DIR/code` (refuse to submit with a dirty tree unless `--allow-dirty`,
  in which case copy the working tree with rsync and set `dirty: true` in
  `RUN_INFO.json`). The job scripts `cd "$RUN_DIR/code"` instead of
  `cd $SLURM_SUBMIT_DIR`; the venv stays shared (read-only) at the repo
  root and is activated by absolute path.
- Human data is not in git: the wrapper symlinks
  `$RUN_DIR/code/human_data/task_aligned_all -> <shared read-only copy>`.
  Shipped GAM artifacts (`hcs_package/.../models/*.pkl`) and base personas
  are in git and therefore inside the snapshot.
- Editing the working tree after submission is then harmless.

### 6.3 Remove every shared mutable path
- Delete the `ln -sfn "$HCS_FIT_RESULTS_DIR" eval/eval-anchor-drive/results`
  lines from `fit_anchor_10p.sh` and `fit_pooled8.sh`. In `fit_anchor.py`,
  `fit_anchor_pooled8.py`, `fit_baseline*.py` set
  `RESULTS = Path(os.environ["HCS_FIT_RESULTS_DIR"])` when the variable is
  set, falling back to `HERE/"results"` only for local use.
- `run_eval.py` already honours `HCS_EVAL_RESULTS_DIR` (sim_cache lives
  under it) — point it at `$RUN_DIR/eval`.
- `#SBATCH --output=logs/...` resolves relative to the submit directory:
  pass `--output "$RUN_DIR/logs/%x_%A_%a.out"` on the `sbatch` command line
  from the wrapper instead of the in-script header.
- `TMPDIR` is already job-scoped; keep it, but under `$RUN_DIR/tmp`.

### 6.4 Submit wrapper `submit_run.sh` (replaces hand-typed sbatch)
```
submit_run.sh --model {mpcc,baseline} --variant {full,no_gaze,no_lookahead,no_pace,no_intermittent} \
              --kind {perpid,pooled8} [--seed 42] [--participants participants_10p.txt] [--allow-dirty]
```
1. Build `RUN_ID`, create the tree, snapshot the code, write `RUN_INFO.json`.
2. `sbatch` the fit (array 1–8 or the single pooled job) with
   `--export=ALL,RUN_DIR=...,MODEL=...,VARIANT=...,KIND=...,SEED=...`.
3. `sbatch --dependency=afterok:<fit_jobid>` the persona-staging + eval array
   (`eval_10p.sh`), then `--dependency=afterok:<eval_jobid>` the aggregate.
   For `pooled8`, the staging step copies the single fitted config under
   every participant's name into `$RUN_DIR/personas/`.
4. Append job ids to `RUN_INFO.json` and one line to
   `$RESULTS_ROOT/runs/INDEX.tsv`:
   `RUN_ID  model  variant  kind  seed  commit  submitted  fit_job  eval_job  agg_job  status`.
The four scripts (`fit_*_10p.sh`, `fit_*pooled8.sh`, `eval_10p.sh`,
`eval_10p_aggregate.sh`) become variant-agnostic: they read `MODEL`,
`VARIANT`, `KIND`, `RUN_DIR` from the environment and never compute paths
themselves. `MODEL=baseline` selects `fit_baseline*.py` and adds
`--model baseline` to `run_eval.py`; `VARIANT` becomes `--ablation` for
`MODEL=mpcc` and is refused for `baseline`.

### 6.5 Collecting results without ambiguity
- `eval/collect_runs.py`: walks `$RESULTS_ROOT/runs/*/RUN_INFO.json`, reads
  each run's `eval/Steering/steering_condition_summary.csv`,
  `eval/Fitts/fitts_regression.json`, fit records, and gaze-lead summaries,
  and writes `$RESULTS_ROOT/runs/SUMMARY.csv` with one row per run
  (columns = Tables 1–3 of §4) plus `RUN_ID`, `commit`, `status`. A run whose
  aggregate has not finished is listed with `status=pending`; a run with any
  failed array task is `failed` (read from `sacct` by job id).
- Paper figures and tables are generated from `SUMMARY.csv` and from
  explicit `RUN_ID`s written into the plotting scripts, never from
  "the latest directory". The RUN_IDs used in the paper are recorded in
  `paper/RUNS_USED.md`.
- Re-running a variant (bug fix, new seed) is a new RUN_ID; the old one is
  left in place and marked `superseded_by` in its `RUN_INFO.json`.

### 6.6 Local mirror
`rsync` `$RESULTS_ROOT/runs/<RUN_ID>/{RUN_INFO.json,fit,eval/*.csv,eval/*.json,eval/*/**.pdf,eval/*/**.png,gaze-lead}`
into `results-cluster-10p/runs/<RUN_ID>/` (exclude `sim_cache`, `code`, `tmp`,
per-round trajectory dumps). The local tree keeps the same RUN_ID names so
paths in the paper scripts resolve on both machines.

## 7. Status (2026-09-08)

Evaluation parallelism: the CMA work unit is now (candidate x trial) —
`fit_anchor.run_cmaes_units` / `fit_anchor_pooled8.pooled_cmaes` map ~300
(per-participant) / ~2400 (pooled) trial simulations per generation over the
whole node, so a generation ends after the longest trial, not the slowest
candidate; `cluster/fit_job.sh` asks for 36 cores. Per-trial caps are 2x the
human completion time (floor 3 s) for tunnels and 5 s for pointing, for every
model (`fit_anchor.TUNNEL_CAP_MULT`, `POINT_CAP_STEPS`).

Implemented and smoke-tested locally: ablation switches + bit-identity
test (`hcs_package/tests/test_ablation_switches.py`), `fit_anchor.py`
`--model/--ablation` (per-participant), `fit_anchor_pooled8.py`
`--model/--ablation`, `fit_baseline*.py` aliases, `utils/baseline_loader.py`,
`run_eval.py --model`, `submit_run.sh` + `cluster/*.sh` + `stage_persona.py`,
`eval/collect_runs.py`. The four old root scripts (`fit_anchor_10p.sh`,
`fit_pooled8.sh`, `eval_10p.sh`, `eval_10p_aggregate.sh`) are retired stubs.

## 8. Pitfalls

- Today's scripts `ln -sfn` a results dir onto `eval/eval-anchor-drive/results`
  inside the working tree: two concurrent runs flip it under each other and
  write into the wrong tree. Remove it before anything is queued (§6.3).
- Today's scripts `cd $SLURM_SUBMIT_DIR`: a queued job runs whatever the
  working tree contains when it starts, not when it was submitted. Use the
  per-run code snapshot (§6.2).
- Both packages are `hcs_package`: the class swap must happen once in the
  parent; forked workers inherit it. Never import both in one process.
- EA dwell: the EA simulator had a terminal hold; force it to 0 so MT
  alignment matches A.
- The CMA objective must stay deterministic: `add_noise=False`,
  `replan_latency_cv=0`, fixed `random_seed` for the stability trials.
- `_reject_pruned` currently refuses `horizon_mode="fixed"` and
  `speed_model type="gam"`. Use the new names above; do not resurrect the
  old flags.
- Old baseline fits in `eval/baseline_fitting/results` are for the Prolific
  cohort on the stale pipeline — do not reuse or compare.

# Great Lakes Setup & Job Submission (adaptive-cursor-trajectory-simulator)

Updated 2026-09-03: section 3b is the CURRENT fitting round — the 10p gaze
batch under the finalized cycle design (GAM traversal deadline, no deadline
rule stack, no free_velocity damping). Sections 3/4 describe the legacy
aug-26-prolific GAM pipeline; personas it produced are REFUSED by the
current simulator until refit.

## 3b. CURRENT: 10p anchor-drive fit (finalized cycle design)

```bash
sbatch fit_anchor_10p.sh
```

- 8 array tasks (participants_10p.txt: p01-p04, p06-p08, p10; p06/p08
  recollected 2026-09-03; p05/p09 deprecated), 12 CPUs, 1 GB/CPU, **8 h wall**,
  CMA budget 6.5 h (`TIME_LIMIT=23400`) so the noise-on stability runs, the
  full held-out probe and the save all finish inside the wall.
- Data: `human_data/task_aligned_all` (short pXX ids are aliased to the
  embedded Prolific ids by `fit_speed_model.load_participant`).
  **The pXX_task_aligned_analysis.csv files are NOT in git** (>100 MB,
  GitHub limit; gitignored 2026-09-03) — copy them up once from the Mac:
  ```bash
  rsync -av --include='*_task_aligned_analysis.csv' --exclude='*' \
      ~/Desktop/adaptive-cursor-trajectory-simulator/human_data/task_aligned_all/ \
      xiangyz@greatlakes-xfer.arc-ts.umich.edu:<repo>/human_data/task_aligned_all/
  ```
- Base personas: `eval/model_fitting/base_configs_gaze/{pid}.json` —
  finalized-design configs (speed_model `gam_traversal` → the shipped pooled
  artifact `hcs_package/models/gam_traversal_10p.pkl`; budget priors D0=1.0,
  gamma=0.66 from the 10p cohort analysis; pooled replan latency 0.19 s /
  CV 0.89).
- CMA-ES fits FIVE params per participant (eval/eval-anchor-drive/
  fit_anchor.py): jerk, contour, constraint, goal, D0. gamma (0.66, gaze
  constant) and plan_vmax (0.66 m/s, stage0_plan_vmax.py pooled pace) are
  pinned; plan_deadline_s is calibrated post-fit by a 1-D pointing-loss
  scan (t0_scan in the fit record; grid-edge = endgame diagnostic).
  free_velocity and the turn-time/width-time deadline keys are GONE — the
  simulator refuses configs that carry them.
- Outputs → `chi-27/results/anchor_fitting_10p[-RUN_TAG]/stages/base/`:
  `{pid}_anchor_config_s42.json`, `{pid}_anchor_fit_s42.json`, fit logs.
- Rerun one participant: `sbatch --array=N fit_anchor_10p.sh` (N = line
  number in participants_10p.txt). Tagged generation: `RUN_TAG=v2 sbatch ...`.
- Before the first submit after pulling: `pip install -e hcs_package/` in the
  venv picks up the new `speed_model.py` + `models/` artifact (pygam is
  already in setup.sh).

### Pooled single-model fit (optional)

```bash
sbatch fit_pooled8.sh
```

One persona fitted jointly on all eight participants
(eval/eval-anchor-drive/fit_anchor_pooled8.py): pooled Stage-0 GAM + one
CMA-fitted parameter set. Single wide task (36 CPUs), parallel over
(candidate x participant) units; 5 h CMA budget, then pooled T0 scan and
eight per-participant held-out probes. Output:
`chi-27/results/anchor_fitting_pooled8/stages/pooled8/pooled8_anchor_config_s42.json`
(+ fit record with per-pid probes). To evaluate it with eval-main, stage it
as `personas_10p/default.json` — run_eval's --config-dir resolution falls
back to default.json for every participant.

### Evaluation (after the fits finish)

```bash
EVAL_ID=$(sbatch --parsable eval_10p.sh)
sbatch --dependency=afterok:$EVAL_ID eval_10p_aggregate.sh
```

- `eval_10p.sh` (8 array tasks, 1.5 h wall, 4 CPU/6 GB): per participant it stages the
  fitted persona as `personas_10p/{Prolific-id}.json`, runs eval-main
  (`--config-dir --fresh-sim`, all buckets) into ONE shared folder
  `chi-27/results/eval-main-10p[-RUN_TAG]/` (Steering / ID4SCS / Fitts /
  sim_cache, all participants together), then runs
  `model_gaze_lead.py --config <fitted persona>` to render the model
  sawtooth vs human rounds per trial into
  `chi-27/results/gaze-lead-10p[-RUN_TAG]/{pXX}/` (one PDF per participant
  + model_lead_events.csv) and gaze_lead_grids.py (individual/, lead_by_width/,
  lead_by_curvature/ PNGs with the model overlaid).
- `eval_10p_aggregate.sh` (1.5 h): pooled Fitts/Steering/ID4SCS summaries +
  overview across all participants, from the cached sims — no new simulation.
- Use the SAME `RUN_TAG` for fit, eval and aggregate of one generation.

## Legacy pipeline (pre-2026-09-03)

Updated 2026-08-17 for the aug-26-prolific dataset (10 participants; steering +
ID4SCS + unconstrained pointing) and the then-current model (fixed plant,
free-space LQR objective, `dwell_s`).

## 0. Where things live

- Project/results folder: `/home/xiangyz/ondemand/data/sys/myjobs/projects/chi-27/`
  - `logs/`                       SLURM stdout/stderr
  - `results/model_fitting/`      fitted personas / GAMs / fit records / fit logs
  - `results/eval-main/`          eval-main outputs (Steering, ID4SCS, Fitts, sim_cache, overview)
- Repo: clone it inside chi-27 (or anywhere) — the scripts locate the repo via
  `SLURM_SUBMIT_DIR` and send all outputs to chi-27 through `HCS_FIT_RESULTS_DIR` /
  `HCS_EVAL_RESULTS_DIR` (override the root with `RESULTS_ROOT=... sbatch ...`).

## 1. Clone / update the repo on the login node

```bash
cd /home/xiangyz/ondemand/data/sys/myjobs/projects/chi-27
git clone <repo-url> adaptive-cursor-trajectory-simulator   # or: git pull
cd adaptive-cursor-trajectory-simulator
```

Always `sbatch` from the repo root.

## 2. One-time setup

```bash
bash setup.sh        # venv, deps (numpy<2, scipy, cma, pygam, matplotlib, pandas, seaborn), pip install -e hcs_package/
```

## 3. Fitting jobs (one array task per line of participants.txt)

```bash
sbatch fit_all_participants.sh
```

- 10 tasks, 12 CPUs, 1 GB/CPU, 13 h wall; `--time-limit 43200` (12 h) inside.
- Per participant: Phase 0 ref-path (≤5 min) → GAM speed model (seconds) →
  Stage 2 tunnel MPCC weights (80 % of budget) → Stage 3 pointing LQ weights (rest).
- Outputs in `chi-27/results/model_fitting/`:
  `{PID}_gam_s42.pkl`, `{PID}_gam_config_s42.json` (persona), `{PID}_gam_fit_s42.json`
  (params, train/test losses, histories), `fit_{PID}_s42.log`.
- Rerun one participant: `sbatch --array=N fit_all_participants.sh` (N = line in participants.txt).
- Re-fit only the pointing stage, reusing a finished tunnel fit:
  `STAGES=pointing TIME_LIMIT=7200 sbatch fit_all_participants.sh`
- Different seed: `SEED=43 sbatch fit_all_participants.sh` (all outputs are suffixed `_s43`).

## 4. Evaluation with fitted personas (after fitting finishes)

```bash
EVAL_JOB_ID=$(sbatch --parsable eval_all_participants.sh)
sbatch --dependency=afterok:$EVAL_JOB_ID eval_aggregate.sh
```

- eval-main `--per-participant --fresh-sim`: steering / ID4SCS / Fitts (aligned MT_kin,
  onset & click latencies, endpoint depth) per participant, then one aggregate pass.
- Outputs: `chi-27/results/eval-main/{Steering,ID4SCS,Fitts}/…`, `.../overview/`.

## 5. Monitoring

```bash
squeue -u xiangyz
tail -f /home/xiangyz/ondemand/data/sys/myjobs/projects/chi-27/logs/fit_<jobid>_<n>.out
grep -h "Stage 2 done\|Stage 3 done\|test loss" /home/xiangyz/ondemand/data/sys/myjobs/projects/chi-27/results/model_fitting/fit_*_s42.log
```

## Notes

- Fitting is noiseless (`nc=[0,0]`); evaluation runs with the persona's noise.
- Human reaction and click latencies are not fitted: pointing MT is compared as
  onset → final target entry on both sides; the model `dwell_s` (0.25 s) stands in
  for the ~0.31 s human click latency.
- Constrained→unconstrained trials (tids 57–83) are excluded from fitting/eval.
- Cancel: `scancel <job_id>` or `scancel -u xiangyz`.
- Old CHI-26 baseline fitting (`fit_baseline_all_participants.sh`) needs the
  `eval/chi-26-ea_baseline_pacakage/` package, which is not in this repo.

## Isolated runs (2026-09-08): submit_run.sh

Every fit/eval generation is now ONE isolated run: its own code snapshot
(`git archive HEAD`), its own results tree, a chained fit -> eval ->
aggregate submission, and a line in `results/runs/INDEX.tsv`. The old
`fit_anchor_10p.sh` / `fit_pooled8.sh` / `eval_10p.sh` / `eval_10p_aggregate.sh`
are retired stubs (they shared a symlink and the live working tree).

```
cd <repo>; git pull; git status          # commit first: submit refuses a dirty tree
./submit_run.sh --model mpcc     --variant full            --kind perpid
./submit_run.sh --model mpcc     --variant full            --kind pooled8
./submit_run.sh --model baseline --variant full            --kind perpid
./submit_run.sh --model baseline --variant full            --kind pooled8
./submit_run.sh --model mpcc     --variant no_gaze         --kind perpid   # ablations: no_gaze,
./submit_run.sh --model mpcc     --variant no_pace         --kind perpid   # no_lookahead, no_pace,
./submit_run.sh --model mpcc     --variant no_lookahead    --kind pooled8  # no_intermittent
./submit_run.sh --model mpcc     --variant no_intermittent --kind pooled8
```
Options: `--seed`, `--time-limit` (CMA seconds), `--popsize`, `--min-runs`,
`--gamma`, `--participants FILE`, `--results-root DIR`, `--allow-dirty`,
`--dry-run`. Each call prints RUN_ID and the three job ids.

Layout: `results/runs/<RUN_ID>/{RUN_INFO.json,COMMIT,code/,fit/stages/{base|pooled8}/,personas/,eval/{Steering,Fitts}/,gaze-lead/,logs/,DONE}`.
Monitor: `squeue -u $USER`, `tail -f results/runs/<RUN_ID>/logs/fit_*_1.out`.
Collect: `python eval/collect_runs.py` -> `results/runs/SUMMARY.csv` (+ a
console table: steering slope/R2, Fitts slope, time ratio, lateral RMSE per
run, with status done/fit_done/pending/failed).
Mirror locally: `rsync -av --exclude sim_cache --exclude code --exclude tmp
--exclude 'participant_*' greatlakes:.../results/runs/<RUN_ID> results-cluster-10p/runs/`.

Fit jobs use one full node (36 cores) since 2026-09-08: the CMA work unit is
(candidate x trial), so a generation of 12 candidates is ~300 simulations
spread over the node. The baseline replans every step and is ~3x slower per
trial than the current model; give it a longer budget:
  ./submit_run.sh --model baseline --variant full --kind perpid  --time-limit 37800 --wall 12:00:00
  ./submit_run.sh --model baseline --variant full --kind pooled8 --time-limit 75600 --wall 24:00:00
Check the pace after the first hour: `grep "gen " <RUN_DIR>/fit/fit_p01_s42.log | tail`
(the "(NNNs gen" field is the generation time).

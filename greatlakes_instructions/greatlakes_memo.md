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

- 6 array tasks (participants_10p.txt: p01 p02 p03 p04 p07 p10 — the kept
  sessions; p05/p06/p08/p09 are deprecated), 12 CPUs, 1 GB/CPU, **8 h wall**,
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
- Fitted per participant (eval/eval-anchor-drive/fit_anchor.py): jerk,
  contour, constraint, goal, D0, gamma, plan_deadline_s (free-space T0),
  plan_vmax. free_velocity and the turn-time/width-time deadline keys are
  GONE — the simulator refuses configs that carry them.
- Outputs → `chi-27/results/anchor_fitting_10p[-RUN_TAG]/stages/base/`:
  `{pid}_anchor_config_s42.json`, `{pid}_anchor_fit_s42.json`, fit logs.
- Rerun one participant: `sbatch --array=N fit_anchor_10p.sh` (N = line
  number in participants_10p.txt). Tagged generation: `RUN_TAG=v2 sbatch ...`.
- Before the first submit after pulling: `pip install -e hcs_package/` in the
  venv picks up the new `speed_model.py` + `models/` artifact (pygam is
  already in setup.sh).

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

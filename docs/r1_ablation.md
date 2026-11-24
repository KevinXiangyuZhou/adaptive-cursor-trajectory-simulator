# R1 ablation: adaptive gaze lookahead vs. pace law

Reviewer 1: the baseline differs from our model in both the lookahead and
the speed mechanism, so the comparison does not separate them. The two new
variants complete a 2 x 2 design:

| lookahead \ catch-up time | pace law (GAM traversal time) | fixed target speed |
|---|---|---|
| adaptive gaze budget | full model (`--variant full`) | **V2** `--variant fixed_speed` |
| fixed horizon | **V1** `--variant fixed_horizon` | CHI-26-EA baseline (`--model baseline`) |

Both are ordinary `submit_run.sh` variants: same fit protocol (CMA-ES joint
loss, noise-on stability, post-fit T0 scan), same eval (`run_eval.py`
steering + Fitts, aggregate) and the same gaze-lead "events" step as the
other ablations.

## What each variant changes

**V1 `fixed_horizon`** (fixed horizon + pace law)
- `horizon_mode: fixed_lead`. Every planning event puts the anchor at a
  constant arc lead `fixed_lead_m` ahead of the cursor.
- `fixed_lead_m` is not fitted. It is set to the participant's mean human
  gaze lead at fixation onset (`eval/eval-anchor-drive/human_gaze_horizon_10p.json`,
  made by `human_gaze_horizon.py`: canonical merged fixation events, the
  25 constant-width steering conditions). A pooled fit uses the mean of
  the fitted participants' means (41.7 mm for the 8).
  - Per participant (mm): p01 34.3, p02 39.7, p03 30.2, p04 49.4,
    p06 46.5, p07 38.3, p08 38.2, p10 56.9.
  - The model's horizon is the anchor lead at a planning event, so the
    matching human quantity is the lead at fixation onset. The
    time-averaged continuous lead is about 0 for every participant, because
    the cursor catches up to and passes each anchor, so it would not work.
- `fixed_lead_speed_floor: false`: no `v*T_min` floor, so the lead is the
  same constant at every speed. The old `no_lookahead` ablation kept the
  floor.
- `fixed_lead_free_space: target`: in unconstrained space the anchor is the
  target, which is also where the budget puts it. Only the corridor
  lookahead is fixed. With a fixed lead in free space, pointing breaks into
  short hops. The old `no_lookahead` did this, and its Fitts slope rose to
  0.28 against 0.21 for the model and 0.23 for humans.
- Search: jerk, contour, constraint, goal. D0 is dropped and nothing is
  added. The pace law, replanning and the T0 scan are the same as in the
  full model.

**V2 `fixed_speed`** (adaptive gaze lookahead + fixed target speed)
- `catchup_mode: target_speed`. In a corridor the plan deadline is
  `lead / v_type`, which replaces the pace-law traversal time. The same
  `lead/v_max` and acceleration floors apply. Free space keeps the T0 rule
  and its post-fit scan.
- One constant speed per steering task type (straight, corner, sinusoidal,
  gentle and sharp sinusoidal). This is the baseline's reference-velocity
  parameterisation (`desired_speed:<type>`, same bounds 0.03–1.26 m/s),
  fitted jointly by CMA-ES on the same training data, but without the
  baseline's curvature slowdown.
- The starting value is the training rounds' mean speed per type
  (centerline length / completion time).
- Search: jerk, contour, constraint, goal, D0 and the 5 `target_speed:<type>`.
- The task type reaches the simulator through a `task_type` key that
  `run_eval.build_steering_task_config` now writes into every steering task
  config. Every other mode ignores it.

Defaults are unchanged. Personas without the new keys produce bit-identical
trajectories. This was checked on the paper's fitted p01 personas (full,
no_lookahead, no_pace; 4 tunnels + 1 pointing round, noise on), and
`hcs_package/tests/test_ablation_switches.py` has a test for it.

## Protocol parity: which code to run

The Study 1 numbers (Figure 3, Tables 1–2) come from runs fitted at
`22c349c`:
- `mpcc-full-perpid-s42-20260909-1257-22c349c`
- `baseline-full-perpid-s42-20260908-1800-c96338d`

The fit protocol changed after that commit (`45e1b97`, `6881cda`,
`74ea011`):
- train widths are now {10, 16.5, 50} mm instead of {10, 50};
- pointing trains on 3 conditions instead of R = 5 mm only;
- there are 3 stability trials instead of 2;
- the CMA budget is 7.5 h instead of 6.5 h.

So:

- **Study 1 (per participant, 8p):** run from branch `r1-ablation-s1`.
  This is `22c349c` plus one commit
  (`docs/r1_ablation_s1_22c349c.patch`) that carries the two variants and
  the `model_gaze_lead.py` GAM-path fix from `8385105`. Without that fix,
  every 22c349c ablation's gaze-lead step failed with
  `FAILED_model_gaze_lead_*`. The fix touches no fit or eval code. On this
  branch the paper personas still reproduce bit-identically.
- **Study 2 (pooled train-all, 14p):** run from `precision` with these
  changes. The pooled fitting code there is the same as `c70a5dd`, the
  commit behind the paper's pooled runs.

## Cluster commands (Great Lakes login node)

```bash
cd /home/xiangyz/ondemand/data/sys/myjobs/projects/chi-27/adaptive-cursor-trajectory-simulator

# --- Study 1: per-participant fits, 22c349c protocol -------------------------
git show precision:docs/r1_ablation_s1_22c349c.patch > /tmp/r1_s1.patch   # after pushing precision
git checkout -b r1-ablation-s1 22c349c && git am /tmp/r1_s1.patch
./submit_run.sh --model mpcc --variant fixed_horizon --kind perpid --dry-run
./submit_run.sh --model mpcc --variant fixed_horizon --kind perpid
./submit_run.sh --model mpcc --variant fixed_speed   --kind perpid
# 8 x (8 h fit) -> 8 x eval (+ gaze-lead events) -> aggregate, per variant

# --- Study 2: pooled train-all fits on 8p, evaluated on the 14 new participants
git checkout precision
./submit_run.sh --model mpcc --variant fixed_horizon --kind pooled8   # train-all, 12 h budget
./submit_run.sh --model mpcc --variant fixed_speed   --kind pooled8
eval-14p/submit_eval_14p.sh --source-run <mpcc-fixed_horizon-pooled8all-...>   # waits on the fit job
eval-14p/submit_eval_14p.sh --source-run <mpcc-fixed_speed-pooled8all-...>
```

## Analysis (after mirroring the runs into results-cluster-10p/)

```bash
# Study 1: laws, trajectory metrics, time ratio by width; paper aggregation
python paper/r1_ablation_table.py --train-widths 10,50 \
  --run Ours=results-cluster-10p/runs/mpcc-full-perpid-s42-20260909-1257-22c349c \
  --run Baseline=results-cluster-10p/runs/baseline-full-perpid-s42-20260908-1800-c96338d \
  --run V1_fixed_horizon=results-cluster-10p/runs/<mpcc-fixed_horizon-perpid-...> \
  --run V2_fixed_speed=results-cluster-10p/runs/<mpcc-fixed_speed-perpid-...>

# Study 1: saccade-fixate-catch-up cycle statistics (Sec. eval_cycle), the
# human side reused from the paper's quant_refit
for v in fixed_horizon fixed_speed; do
  mkdir -p paper/quant_r1_$v && cp paper/quant_refit/{cycles_human,curv_human}.csv paper/quant_r1_$v/
  python paper/quantify_gaze_cycles.py --stage model --config-dir results-cluster-10p/runs/<mpcc-$v-perpid-...>/fit/stages/base --quant-dir paper/quant_r1_$v
  python paper/quantify_gaze_cycles.py --stage stats --quant-dir paper/quant_r1_$v
done

# Study 2: Section 5.4 numbers with the ablation in the model slot
python paper/stats_study2.py --s2-model results-cluster-10p/runs-14p/<eval14p-mpcc-fixed_speed-pooled8all-...> \
  --out paper/quant_study2_fixed_speed
```

The cycle statistics are where V1 and the full model should differ:
- saccade amplitude vs width exponent (human b = 0.55, model 0.68);
- catch-up duration vs width;
- overrun.

A fixed horizon predicts no width scaling of the anchor lead, so the cycle
statistics measure what the gaze mechanism explains beyond the pace law.

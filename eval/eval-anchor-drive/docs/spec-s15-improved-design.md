# S15 spec: the improved design in full (keep / remove / add), 09-01

Reference for revising hcs_package. Read against HEAD 0816e0a (GazeModule/MotorModule split,
variant graveyard pruned) plus the uncommitted `chance_z` edit in `mpcc_model.py`.

## 0. The design in one paragraph

Two modules and one shared quantity. The gaze module decides WHERE to look (anchor from the
width-only difficulty budget, floored at v*T_min and at cursor + local room) and WHEN to look again
(distance arrival + lognormal latency, deviation interrupt, exhaustion backstop). It hands the motor
module an anchor and one time constant T0, the straight-segment crossing time. It says nothing about
curvature. The motor module solves one MPCC with jerk effort, damping, a physiological |a| cap,
lateral contour + stiff along-path coupling, a soft via-point at the deadline with a pace tail, and
corridor hinges tightened by the lateral scatter the plant's own signal-dependent noise will produce
over the open-loop window. Slowing in bends is what the tightened corridor forces; slowing at the
goal is what the path end forces; width scaling of cruise is lead/T0. No speed formula anywhere. The
cursor feeds back to gaze once per cycle through v_now (lead floor) and the deviation trigger, and
every fixation onset is the observation that resets execution uncertainty.

## 1. Gaze module (`gaze_module.py`, `intermittent.py`)

Keep:
- `DifficultyBudgetHorizon` width-only density (W_ref/W)^gamma / W_ref with D0, gamma calibrated
  on human onset leads (Stage G) and FIXED in cursor fits; T_min floor; cap at path end.
- Anchor lead floor (anchor >= cursor + local room), always on — delete the `anchor_lead_floor`
  switch.
- t_plan = max(T0, lead / v_max). T0 = `plan_deadline_s`, fitted on cursor within the gaze range
  (0.10-0.25). v_max is a physiological cap (0.8 m/s), NOT fitted — a fitted v_max is a prescribed
  speed by another name (S6/S7 fits used it exactly that way).
- n_solve = n_base + 2*tau_steps padding; `horizon_max_steps` clamp.
- `ReplanScheduler`: distance-or-passed arrival, lognormal latency (median, CV, cap), deviation
  interrupt (0.3 of local width), refractory `min_open_loop_s`, exhaustion backstop.
- `w_solve` / `dev_scale` / `arrival_tol` bookkeeping.

Remove:
- `plan_turn_time_s`, `plan_turn_width_exp` and the whole turning-time block in `plan_fixation`
  (the calibrated slow-down rule). With it goes `solve_anchor_s` and the 3-node horizon-floor
  branch it needed; `Fixation.solve_anchor_s` collapses into `anchor_s`.
- The `t_acc` floor (`acc_max` inside GazeModule). The motor module owns the acceleration bound;
  an unreachable deadline must produce lateness, which is the emergent slowing. Drop `acc_max`
  from GazeModule's constructor.
- `budget.lam`, `budget.beta` (additive curvature toll). The gaze evidence says the lead is
  width-set and crosses corners; the toll was the gaze-side half of the slow-down rule.
- `arrival_mode` switch: keep only distance-or-passed (progress arrival stalls in corner wedges).
- `replan_mode: every_step` and `horizon_min_steps` (now dead: the binding floor is T_min).
- `Th`, `Tp`, `pred_horizon`, `TH_SCALE`, `BumpParams.Tp` — fixed-horizon leftovers, unused in
  budget mode.

Add:
- GazeModule output gains `t_ol = t_plan + replan_latency_s` (expected open-loop window) and
  passes it to the motor module; the chance constraint uses it instead of a separate
  `chance_t_ol` weight.
- Optional, gated, later: uncertainty anchor (first s where z*sigma_lat(s) at v_now reaches the
  room), combined as min(budget anchor, uncertainty anchor). Not part of the S15 fit.

## 2. Motor module (`motor_module.py`, `model.py`, `mpcc_model.py`, `params.py`)

The S15 objective, per plan (all stage sums multiplied by dt/dt_ref, dt_ref = 0.05):

  J = jerk   * sum |j_k|^2
    + free_velocity * sum |v_k|^2
    + acc_weight * sum hinge(|a_k| - acc_max)^2                       [acc_max 4, fixed]
    + contour * sum e_c,k^2 + lag_anchor * sum e_l,k^2                [lag_anchor 2000, fixed]
    + goal * ((anchor_s - s_deadline) / lead)^2
    + goal * sum_{k>deadline} ((s_tail,k - s_k) / lead)^2             [pace tail, always on]
    + constraint * sum hinge(|e_c,k| - room_k)^2                      [walls, cartesian]
    + chance_weight * sum hinge(|e_c,k| + z*sigma_lat,k - room_k)^2   [chance_weight 1e4, fixed]

with sigma_lat,k from the plant's own noise (nc0 along, nc1 across) accumulated over the open-loop
window and projected on the wall normal. Fitted: jerk, contour, constraint, goal, free_velocity,
T0, z (7). Fixed: acc_max, lag_anchor, chance_weight, acc_weight, v_max, nc, budget, latency.

Keep (`mpcc_model.generate_mpcc`):
- Jerk parameterisation, free response, A_vel/A_pos/A_acc maps, SCALE_JERK.
- Kinematic progress on the fixed tangent schedule + fixed-point re-linearisation with the
  monotone best-iterate guard.
- Lead-normalised via-point at k_deadline; pace tail (`anchor_pace`).
- acc_max hinge; contour/lag_anchor split; corridor and cartesian hinges; warm start with shift.

Remove / make unconditional:
- `w_lag_anchor is None` legacy full-distance branch: `lag_anchor` becomes required.
- `desired_speed` argument and the `s_estimated`-based `cartesian_active` mask: use the schedule
  `s_sched` (already the plan's progress) to decide which nodes are on the path. Delete
  `planner_weights.desired_speed`, `.lag`, `.progress` everywhere (configs, fit specs, README).
- `chance_nc0` weight: read nc0/nc1 from `BumpParams.nc` (one source of truth for the noise the
  plant injects). `chance_t_ol` weight: take `t_ol` from the Fixation.
- Per-node `ref_path.curvature(s_b)` call inside the objective: precompute kappa on the schedule
  once per linearisation pass (it is fixed within a pass, like the tangents).
- `params.SteeringModelInput` fields no one reads in anchor mode: `tunnel` (TunnelInfo),
  `bump.Tp`, `planner_margin`, `clearance_profile`, `curvature_rate_profile`,
  `curvature_profile`. `model.py` then shrinks to state assembly + unpacking; fold it into
  `MotorModule.solve` and delete `model.py` (keep `FREE_SPACE_CLEARANCE_M` in gaze_module or a
  constants module).
- `baseline_model.py` and `generate_trajectory_with_start_and_end`: ruled out for pointing
  (its own docstring); pointing is the anchor model with the anchor on the path end. Delete both
  and the `#imports for baseline` block.
- `evaluate_tracking_errors`: diagnostic for the removed virtual-progress lag term; delete.

Change (the chance term): the uncommitted edit uses a pointwise proxy
  tighten_k = 0.5 * z * nc0 * |kappa_k| * v_k^2 * T_ol^2.
That is acceptable for the first probe, but it is the same functional form as the rejected bend
term, so if it fails do not conclude against the mechanism. The propagation form is:
  Sigma_0 = 0 ;  Sigma_k = Sigma_{k-1} + dt^2 v_{k-1}^2 (nc0^2 t_{k-1} t_{k-1}^T + nc1^2 n_{k-1} n_{k-1}^T)
  sigma_lat,k = sqrt(n_k^T Sigma_k n_k)
with t, n the schedule tangents/normals (fixed per pass, so sigma_lat is smooth in the jerks),
and Sigma reset at node 0 of every solve. It differs from the proxy in two ways that matter:
scatter accumulated on a straight approach converts to lateral error at a corner (corners cost
unless fixated), and the tightening grows over the plan rather than depending on local kappa
alone. Implement both behind `chance_form: "local" | "propagate"` and run the z-probe on each.

Add:
- dt-invariant stage costs (multiply per-node sums by dt/dt_ref). Refit B once; losses should hold.
- Free-space tightening: where room_k is the target radius (path end), the same hinge with
  R_eff = R - z*sigma_k; this is the pointing width term and replaces `goal_precision`.
- Exact-gradient / QP solve (item 2 of the review) behind `solver: "lbfgs" | "qp"`, with an
  agreement test on the S14 persona's probe trials. Not needed for the S15 decision; needed for
  the all-participant fit.

## 3. Simulator shell (`cursor_simulator.py`, `noise.py`)

Keep: pixel/metre mapping, reference-path generation and profiles, GazeModule/MotorModule loop,
carry_acceleration, dwell termination, `abort_on_breach_m`, diagnostics.

Remove: `Interval`/`Tp`/`Th`/`mouseGain`/`ddm_enabled` config keys (dead), `horizon_mode`
(only budget), `replan_mode` (only intermittent), `arrival_mode`, `anchor_lead_floor`,
`anchor_tail_pace`, `plan_turn_*`, `budget.lam/beta`, `horizon_min_steps`. Keep `_reject_pruned`
and extend it with these keys so S12-S14 configs fail loudly.

Add: `screen_width_m` as a task/config field (default 0.46); a `np.random.Generator` threaded
into `single_step_motor_and_device_noise` / `upper_limb_module.motor_noise` (replace the
global `np.random.seed`); expose `t_ol` in diagnostics.

## 4. Feedback loop (unchanged in structure, made explicit)

Fast loop (gaze -> cursor), once per fixation: anchor + T0 -> MPCC solve with Sigma_0 = 0 ->
open-loop execution under plant noise for t_ol.
Slow loop (cursor -> gaze), once per cycle: v_now enters the lead floor v*T_min; the realised
deviation ends a fixation early (0.3 of local width); the fixation onset re-seeds the plan from
the realised state (the observation). Prediction for D1: horizon shrink leads deceleration by
about one cycle (~0.35 s). If the human lag is one visuomotor delay instead, move the Sigma reset
to onset + delay.

## 5. Tests to add before fitting

- Golden metrics on the S14 persona, three trials, noiseless (CT, apex dip, lateral RMSE, 5 %).
- `test_budget_density_is_width_only` stays true again once lam/beta are gone; update its docstring.
- Chance term: on a straight corridor sigma_lat is nc1-only and the hinge is inactive at cruise;
  on a constant-curvature arc the closed-form cap v <= sqrt(2 room/(z nc0 kappa T_ol^2)) is
  reproduced within 10 % by the local form.
- Solver agreement (when the QP lands): L-BFGS-B vs QP within 1e-3 on the S14 probe trials.

## 6. Decision procedure

1. D1/D2 on the cleaned gaze events (data only).
2. Implement sections 1-3 removals + dt invariance; golden test passes on S14 with tau set to
   its S14 value (equivalence check before removing the rule).
3. Remove the rule; z-probe on B (z in {1,2,3}, both chance forms, noise off) for speed by width
   and type; noise on for HF lateral RMS, correction rate, acceleration distribution vs human.
4. If the 10 mm sharp sinusoid slows toward human and straights stay width-independent, fit S15
   (7 params) for B, then A, C. If not, S14's rule stays with D1/D2 as its justification and the
   chance term is reported as a tested negative.

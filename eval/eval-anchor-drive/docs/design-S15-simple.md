# S15: the simple design, and how to get there from S14 (09-01, rev 3)

One idea: a fixation cycle is a submovement. The gaze module decides WHERE the submovement goes
and WHEN the next one starts; the motor module plans it as an open-loop movement whose endpoint
scatter must land inside the corridor, and corrects with a delay when execution drifts. Speed is
never prescribed: width scaling comes from the gaze lead, curvature x width slowing from scatter
that must fit the walls, and time lost in narrow bends from delayed corrections under noise.

## Grounding

Gaze leads the effector by ~one look-ahead fixation: Land & Hayhoe 2001; Johansson et al. 2001;
two-point steering with a look-ahead point: Wilkie & Wann 2003, Salvucci & Gray 2004; preview
control: Donges 1978. Intermittent plan-execute-replan: Bye & Neilson 2008 (BUMP), Gawthrop et
al. 2011, Do et al. CHI 2021, Alvarez Martin et al. 2021 (refractory period, feedback delay).
Signal-dependent noise and margin: Harris & Wolpert 1998; optimal feedback control (planner
margin + delayed correction): Todorov & Jordan 2002. Submovement endpoint scatter and Fitts
chaining: Meyer et al. 1988; steering law as a chain of crossing tasks: Accot & Zhai 1997,
Pastel 2006 (corners). Chance-constrained MPC: Blackmore, Ono & Williams 2011.

## The design

Gaze module (per fixation), inputs: path, W(s), cursor s0 and v_now.
  lead    = D0 * W^gamma * W_ref^(1-gamma)          width-only budget, POPULATION gaze constants
  anchor  = min(path_end, max(s0 + lead, s0 + v_now*T_min, s0 + room(s0)))
  T_plan  = max(T0, lead / v_max)                   T0 = gaze crossing time, constant
  lifetime: arrival (within room of anchor, or passed) + lognormal latency; deviation trigger at
  30 % of local width; exhaustion backstop. Nothing about curvature.
  NEW: a deviation trigger does not replace the plan immediately — the old plan keeps executing
  for delta = 0.12 s (visuomotor delay, measured, not fitted) before the new solve takes over.

Motor module (MPCC, anchor mode), decision variables jerk only, N = round(T_plan/dt) + tail.
Five terms, each with one job:
  effort      jerk * sum |j_k|^2                                             fitted
  strategy    contour * sum e_lat,k^2          (adherence to the Phase-0 route)  fitted
  task        K_wall * sum hinge(|e_lat,k| - room_k)^2                       fixed high
              K_via  * ((anchor - s_deadline)/lead)^2 + pace tail            fixed high
              K_wall * hinge(z*sigma_wall(N) - room(anchor))^2   TERMINAL chance hinge, z fitted
  physiology  K_acc  * sum hinge(|a_k| - 4)^2                                fixed
  sigma_wall(N): scatter of the plan's endpoint from the plant's own noise (nc0 along, nc1 across,
  white per step, random walk over the plan) projected on the wall normal at the anchor; in free
  space room := R_target (replaces goal_precision). Per-node tightening is the fallback, not the
  default. Progress kinematic on the fixed tangent schedule, re-linearised (unchanged). Executed
  open-loop through the noise plant; the plan never sees the realised noise.

Removed from the objective: velocity damping (a hidden speed knob, fitted 0.31 in S14),
lag_anchor (a coupling the kinematic-progress formulation does not need), goal_precision, coast
safety, bend/plant lag, the free-space LQR, the speed profile, the time rule.

Fitted on cursor (4): jerk, contour, v_max, z. Fixed from gaze, pooled over all 13 participants:
D0, gamma, T0, T_min, latency median/CV, deviation 0.3, delta. Fixed physiology: acc_max 4,
nc [0.2, 0.02]. Fixed numerics: K_wall, K_via (verified inert over a decade).
Per-participant route: Phase-0 reference path (unchanged).

Fitting is NOISE-ON, seed-averaged (2-3 seeds). A design whose slowing comes from execution
uncertainty cannot win a noiseless fit by construction — with noise off the fastest deterministic
plan inside the walls is always cheapest, which is why every deterministic proxy looked inert.

## Two things to establish before building (data + offline, no solver work)

P1. Scope of the intermittent assumption. From the cleaned events, per width: early-termination
    fraction, corrections per cycle, cycle length. If at 10 mm bends most cycles end before
    crossing, that width is a closed-loop regime the one-plan-per-fixation model does not cover;
    state it as the model's limit and fit on the widths where the assumption holds. Same script
    as D1 (timing: horizon shrink vs deceleration lag) and D2 (deceleration onset vs vertex).
P2. Magnitude of the chance margin. Monte-Carlo motor_noise over the S14 plans on sharp-10,
    corner-20, sinus-50; tabulate z*sigma_wall/room at the anchor. Back-of-envelope at S14's
    10 mm cruise: sigma_wall ~ 0.6 mm (1.2 mm with the latency window) vs room 5 mm, i.e. z of
    4-6 to bind. If z <= 3 binds on the narrow bends, the terminal hinge is the mechanism; if not,
    the residual there is delayed-correction time under noise (delta + noise-on fitting), and the
    hinge stays only as the wall-breach guard. Do not fit z above 4.

## Remove from the code (cursor_simulator.py unless noted)

Config keys and their branches:
  budget.lam, budget.beta, budget.curvature_weighted  (intermittent.py too; density = width-only)
  plan_turn_time_s, plan_turn_width_exp                (the time rule)
  coast_safety, safety_steps, planner_weights.safety   (mpcc_model n_safety block)
  anchor_memory, corner_consume, corner_kappa
  motor_period_s and the whole motor_replan branch     (BUMP mode; pace tail superseded it)
  arrival_mode                                         (keep the distance-or-passed rule only)
  anchor_tail_pace, anchor_lead_floor                  (always on; drop the switches)
  goal_precision, nc0/nc1 in weights, target_radius in weights
  planner_weights.free_velocity, lag_anchor, lag, progress, desired_speed
  horizon_mode 'fixed', Th, TH_SCALE, pred_horizon     (budget is the only mode)
  speed_model section, GAM loading, _load_speed_model  (speed_model.py -> legacy/)
  replan_mode 'every_step'                             (intermittent is the only mode)
  ddm_enabled, carry_acceleration (always true), mouseGain, Interval vs Tp duplication
  budget_horizon.traverse_time and v_ref_prof placeholder (only the GAM path used them)
  curvature_rate_profile, compute_curvature_rate_profile, compute_curvature_spike_profile,
  compute_sharpness_profile, compute_local_curvature_integral (adapt.py: GAM features)
  curvature_profile (only the time rule and corner_consume read it)

mpcc_model.py: the speed_profile / speed_target smoothing, vs variables and SCALE_VS (n_blocks=2
always), free_space_mask / any_free / terminal LQR / _free_space_lqr_value, w_progress, w_lag,
w_lag_anchor, w_free_velocity / w_free_accel, the goal_precision block, the n_safety block,
bend/plant_lag remnants, evaluate_tracking_errors. Keep: jerk, contour, corridor + cartesian
hinges, acc hinge, via-point + tail, warm start, re-linearisation with the monotone safeguard.

model.py: the speed_model branch and the "speed_model required" error; the corridor 0.95
fallback. baseline_model.py + generate_trajectory_with_start_and_end: move to legacy/ (Fitts
baseline for the paper only). Tag the repo `S14-final` first so every removed variant stays
citable for the ablation appendix.

## Add

uncertainty.py: endpoint_scatter(v_plan, tangents, nc, dt, n_wall) -> sigma_wall (scalar at the
deadline node; optional per-node vector for the fallback); unit test against a Monte-Carlo of
upper_limb_module.motor_noise on a straight and on an arc.

mpcc_model.generate_mpcc: `chance_z` (default 0 = S14 walls): one hinge at the deadline node,
z*sigma_wall(N) <= room(anchor); in free space room = target radius. Multiply every stage sum by
dt/0.05 (dt-invariance) at the same time. Optional terminal rest cost K_rest*|v_N|^2 when the
anchor is the path end, behind a flag, for the damping-removal test.

intermittent.ReplanScheduler: `deviation_delay_steps` (delta/dt): a deviation trigger schedules
the replan delta later instead of firing it now; arrival/latency path unchanged.

cursor_simulator: GazeModule class (budget, floors, T_plan, trigger state — wraps
DifficultyBudgetHorizon + ReplanScheduler) and MotorModule class (builds SteeringModelInput,
solves, holds the plan); generate_trajectory_with_waypoints becomes a ~100-line loop over the two.
Declare anchor_s / deadline_steps / anchor_pace on SteeringModelInput. Pass a np.random.Generator
into the noise plant (drop the global np.random.seed). Make screen pitch a task/config field.

Solver throughput (prerequisite for noise-on fitting): analytic gradient or a QP on the
linearised anchor problem (review item 2). Until it lands, noise-on fitting uses a reduced trial
set (9 tunnels + 12 pointing rounds) and 2 seeds.

tests: golden-metrics test on the S14 persona with chance_z = 0, delta = 0 and the S14 weights
(three probe trials: completion time, apex dip, lateral RMSE within 5 %) so the refactor is
verified behaviour-preserving before any term is removed; then the same test pinned for S15.

fit_anchor.py: ANCHOR_SPEC = jerk, contour, plan_vmax, chance_z (bounds 0.5-4); everything else
fixed; loss noise-on seed-averaged; stability gate kept. Pooled gaze calibration script over the
13 cleaned participants writes the fixed block once.

## Order and checks

0. P1, P2, D1, D2 (data + offline; two days). They decide the scope statement and whether z is a
   mechanism or a guard. Nothing below depends on solver code.
1. Tag S14-final. Pooled gaze constants. Refactor + code removals with the S14 objective still
   reproducible behind flags; golden test must pass unchanged.
2. Term removals on B, S14 weights, noise-on 2 seeds, one at a time:
   a. free_velocity -> 0: pointing MT by radius within 0.1 s of S14, no overshoot beyond R
      (else enable K_rest on the final node; damping stays out).
   b. lag_anchor -> 0: max |e_long| over solved plans < 1 mm on corner20 / sharp10 and the
      re-linearisation converges in <= 8 passes (else a small fixed value, documented as numerics).
   c. K_wall, K_via fixed at 1e3 / 1e2: losses flat over one decade either way.
3. delta = 0.12 s on the deviation trigger, tau removed, z = 0: does narrow-bend completion time
   rise toward human with noise on (10 mm sharp CT ratio from ~0.5 toward 1)? Wall breaches per
   trial must not rise above the human abort rate.
4. Terminal chance hinge, z = 1/2/3: breaches drop to zero; 10 mm sharp speed and the 20/40 mm
   apex dips move toward human; straights stay width-independent; HF lateral RMS and correction
   rate within the human range (corner40 1.2-2.0 mm). Ablate acc_max at the best z.
   Per-node tightening only if mid-bend breaches remain.
5. Fit S15 (4 params, noise-on) for B, then A, C; then all 13 with the QP solver. Compare to S14
   on CT ratio by width and type, apex dips, steering-law slope, gaze rhythm, Fitts slope,
   easy-target MT, and the D1/D2 signatures reproduced by the model.
If step 4 fails at every z <= 3 and step 3 alone is insufficient, restore plan_turn_time_s from
the tag as the documented fallback, with D1/D2 as its empirical justification.

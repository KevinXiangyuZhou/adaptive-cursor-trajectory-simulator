# Review: hcs_package core design, and an improvement plan (09-01)

Scope: `hcs_package/src/hcs_package/{cursor_simulator,model,mpcc_model,intermittent,speed_model,noise}.py`
read against the S14 persona (`results/P170114_anchor_persona_S14.json`), the README history and
`docs/plan-horizon-led-mpc.md`. Priority order; each item names the file and the change.

## What the core design is today (S14, as executed by the code)

Gaze module (`intermittent.DifficultyBudgetHorizon` + `cursor_simulator` fixation block): the anchor
sits where the additive budget (W_ref/W)^gamma/W_ref + lam|kappa|(W_ref/W)^beta spends D0, floored at
v*T_min and at cursor + local room, capped at the path end. Time to that anchor is prescribed:
t_plan = max(T0, lead/v_max, t_acc) + tau*theta_lead*(W_ref/W)^beta_turn. The fixation lives
until distance-arrival + lognormal latency, 30 % deviation, or exhaustion.

Motor module (`mpcc_model.generate_mpcc`, anchor mode): jerk effort, velocity damping, |a| hinge at
4 m/s^2, lateral contour + stiff along-path lag_anchor (2000), corridor hinges, a lead-normalised
progress via-point at the deadline node, pace-holding tail. Progress is kinematic on a fixed tangent
schedule, re-linearised to a fixed point. Between fixations the plan runs open-loop under per-step
signal-dependent velocity noise (nc = [0.20 along, 0.02 across]).

The one-sentence diagnosis: the gaze module currently decides both WHERE (anchor) and HOW FAST
(t_plan); the motor module only decides the shape. Every residual you list — narrow-bend
time-to-anchor (kappa x W), the corner apex dip, A's stop-and-turn, the straight-tunnel pace — is a
speed residual, and speed is the one thing the MPC is not allowed to decide.

## 1. Make speed a motor-module decision: chance-constrained corridor from the existing noise model

This is the same recommendation as the horizon-led plan (drop the tau*theta surcharge, keep the
anchor + geometry channels) but with a concrete mechanism for the kappa x W residual that the three
rejected deterministic candidates (coast safety, room-normalised tracking, bend drift) could not
produce — and it introduces no new behavioural constant.

The plant already has the ingredient. `upper_limb_module.motor_noise` adds per-step velocity noise
0.20*v along the direction of motion and 0.02*v across it. Over k open-loop steps the along-track
position scatter is a random walk sigma_along ~ 0.2*v*dt*sqrt(k); the across-track scatter is ten
times smaller. On a straight the big component stays parallel to the walls, so a straight is nearly
width-independent (B's straights are). In a bend the tangent rotates by kappa*v*dt per step, so
scatter accumulated along the old tangent projects onto the new wall normal:
sigma_lat(k) ~ 0.2*kappa*v^2*dt^2*k^2/2 for small angles. Requiring z*sigma_lat <= room gives
v <= sqrt(2*room / (z*nc0*kappa*T_ol^2)) — a lateral-acceleration cap that scales with room, i.e.
exactly the kappa x W interaction, from constants the model already carries (nc0, the open-loop
window T_ol = deadline + latency). At a sharp corner the accumulated along-track scatter converts
to lateral error all at once — unless the plan is re-seeded at the apex, which is what a corner
fixation does. That is a mechanistic reason for corner fixations ("reset the uncertainty where it
would turn lateral") and for the "corners cheap, sustained bends dear" pattern you measured.

Implementation (`mpcc_model.generate_mpcc`, anchor mode only, config-gated `chance_z`, default 0):
propagate a 2x2 position covariance along the plan using the plan's own velocities and the
existing nc, in the same per-node loop that evaluates the corridor hinge; tighten the corridor
bounds at node k by z*sqrt(n^T Sigma_k n) with n the wall normal on the schedule. Because Sigma_k is
a function of the jerk variables, this is a smooth term for L-BFGS-B (and stays convex-ish in v^2).
Reset Sigma at node 0 (the realised state is observed at every solve). Then remove
`plan_turn_time_s` and let the via-point deadline be soft: lateness in bends emerges from the
tightened corridor + acc_max + jerk; straights keep the T0 crossing time.

Decision test before fitting (cheap, noiseless, B's S14 weights, `probe_anchor.py --quick`): with
z in {1, 2, 3} and tau = 0, does the 10 mm sharp sinusoid slow toward 0.026 m/s while the 50 mm
sinusoid stays near 0.25 and the straights stay width-independent? If yes, this replaces the
turning-time surcharge (S15 = plan doc + this term instead of the stopping hinge, or both).
If no, the D1/D2 experiments in the plan doc still decide leader vs shared trigger and the
calibrated tau stays with its empirical justification. Either way, run D1/D2 first as planned —
they cost nothing and settle the paper's framing.

Longer term, the same covariance can feed the gaze side (the postponed uncertainty-weighted
horizon): anchor where projected sigma_lat first reaches the room. Do not do this before the
motor-side test — the width-only lead is gaze-calibrated and should stay fixed as the control.

## 2. Replace the finite-difference L-BFGS-B solve with an exact QP on the linearised problem

`generate_mpcc` evaluates a Python loop over N nodes per objective call, and L-BFGS-B differentiates
it by finite differences over 2N variables, up to 500 iterations, up to 8 re-linearisation passes.
That is O(N^2) Python per iteration and is why one CMA-ES candidate costs 3–8 min and a fit needs a
10 h cluster job.

In anchor mode the problem is already almost a QP: progress is linear in the jerks on the fixed
schedule, positions are linear, and every term except ref_path(s_k) and the hinges is quadratic.
Linearise ref(s_k) ~ ref(s_sched_k) + tangent_k*(s_k - s_sched_k) (the schedule is re-linearised to
a fixed point anyway, so this converges to the same solution), and the hinges become
piecewise-quadratic — solvable exactly by OSQP or by a few active-set passes with an analytic
gradient. Expected: 20–100x per solve, deterministic convergence (no `success=False` plans, no
runaway 5e3 m/s iterates that needed the monotone safeguard), and a fit that finishes locally in
an hour instead of a cluster night. That changes what you can afford: all 13 participants,
seed-averaged noise-on fitting, cross-validation of every S-round instead of one B fit per idea.

Keep the current solver as the reference path behind a flag; assert the two agree on the S14
persona's three probe trials before switching the fits over.

## 3. Prune the variant graveyard and split the two modules into two classes

`CursorSimulator.__init__` carries ~25 config switches, and `generate_trajectory_with_waypoints`
is ~700 lines with the fixation state machine inline. Per the README, these are rejected or inert
in the current line: `bend`/`plant_lag_s`, `coast_safety`/`safety`, `via_tail`, `anchor_memory`,
`corner_consume`/`corner_kappa`, `budget.curvature_weighted`, `goal_precision`, `motor_period_s`
(BUMP mode, superseded by the pace tail), `arrival_mode: progress`, the GAM speed-profile branch,
the free-space LQR/`free_space_mask` branch. Three model families share one objective function.

Concretely: (a) tag the repo at S14 and delete those paths on main (the tag preserves them for the
paper's ablation appendix); (b) factor `GazeModule` (budget, floor, lead floor, fixation lifetime,
arrival/latency/deviation triggers — everything that produces (anchor_s, t_plan, T_ol)) and
`MotorModule` (build the SteeringModelInput, solve, expose the plan) so the code mirrors the
paper's two-module diagram; (c) make `SteeringModelInput` honest — `anchor_s`, `deadline_steps`,
`anchor_pace`, `safety_steps` are attached as ad-hoc attributes after construction; declare them.
This is a prerequisite for item 1 to be reviewable and for item 2 to be testable.

## 4. Make the objective dt-invariant before any further fitting

You already noted the 25 ms test was not clean because jerk, damping, hinge and tracking sums scale
with node count. The same bug is live at 50 ms: budget-mode horizons vary from 3 to 40 nodes across
fixations, so a plan through a corner (short lead, few nodes) and a plan on a straight (many nodes)
are weighted differently by the same jerk weight. Multiply every stage sum by dt/dt_ref (dt_ref =
0.05 so current fits keep their scale) and express the via-point and terminal terms per plan, not
per node. One-line change per term in `generate_mpcc`; refit B once to confirm the losses hold.

## 5. Doc/code drift and correctness nits (fix in one pass)

`intermittent.py` docstring and the `cursor_simulator` default-config comment say the lam|kappa|
toll was removed on 08-24 and the density is width-only; the code has `lam`/`beta` and the S14
persona runs lam 0.28, beta 2.34. `test_budget_density_is_width_only` pins the old claim. Update
the docstrings and the test to the additive form.

The fixation-exhaustion trigger (`cursor_simulator`, "trigger = 'exhausted'") relies on Python
parsing `A and B and C and D if cond else False` as `(A and B and C and D) if cond else False`;
it works, but parenthesise it.

`np.random.seed(seed)` in `__init__` seeds the global stream that `upper_limb_module.motor_noise`
draws from; under multiprocessing fits every worker inherits the same stream unless re-seeded, and
any numpy use elsewhere shifts the noise sequence. Pass a `np.random.Generator` into the noise
module the way `_replan_rng` already is.

`screen_width_m = 0.46` is hard-coded in `generate_trajectory_with_waypoints`: every task assumes a
46 cm wide screen, so pixel pitch (and therefore mm-scaled widths, the GAM features, the forearm
transfer) is wrong for any other display. Make it a task/config field with 0.46 as the default.

Tests cover the budget class and scheduler but nothing runs anchor-drive mode. Add one golden-metrics
test: S14 persona, three probe trials (corner40, sinus50 10 mm, straight30), noiseless, assert
completion time, apex dip and lateral RMSE within 5 % of stored values. Without it every S-round
change moves fitted results silently.

## 6. Pointing width term (lower priority, paper-visible)

Easy targets are ~0.3 s too fast and there is no tolerance term in pointing time (Fitts b 0.24 vs
human 0.17). Item 1 supplies one for free: at the goal the tightened "corridor" is the target disk,
so z*sigma(v) <= R caps approach speed by target radius — the classic speed-accuracy mechanism —
and the goal_precision well can go. Evaluate after item 1 rather than as separate work.

## Suggested order

D1/D2 (data only, this week) -> item 5 (half a day) -> item 4 (half a day + one B refit) ->
item 1 decision test on B (one day) -> item 3 pruning/refactor (two days, before any new fit
round) -> item 2 QP solver (three to five days, pays back on the first all-participant fit) ->
item 6.

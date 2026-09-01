# Plan: horizon-led MPC (synthesis of reviewer feedback, 09-01)

## What the feedback says, mapped to what we have

1. **Gaze module = leader; H_t = f(curvature ahead, width ahead, speed); outputs a look-ahead
   point.** Largely built: the additive budget rho = (W_ref/W)^gamma/W_ref + lam*|kappa|*(W_ref/W)^beta
   sets the lead from width and curvature ahead; the v*T_min floor gives the speed dependence
   (faster -> farther, the locomotion finding = our "slow loop" back-coupling); the anchor is the
   look-ahead point. Not built: an uncertainty weighting on the horizon.
2. **Motor module = MPC over horizon H_t; slowing emerges** from (a) a short horizon that cannot
   commit to speed it cannot see the end of, and (b) a terminal stopping/reachability constraint
   (able to stop or turn within the visible corridor), which caps feasible speed as H shrinks.
   This is the OPPOSITE resolution to our S12–S14 design: we slow via a calibrated time surcharge
   (T = max(T0, lead/v_max) + tau*theta*(W_ref/W)^beta) and we REMOVED the terminal coast-safety
   constraint (S12) because it was redundant *given* that time model. The feedback argues the
   emergent variant is the stronger claim for the paper. It is also exactly open-question #1
   (the "no-T ablation") plus a horizon-tied stopping constraint.
3. **Caution — H must not be the only channel:** the motor module must also receive the look-ahead
   point and the local geometry (corner vs narrowing produce different shapes). Already satisfied:
   the planner gets the anchor point, the reference path, and the wall corridor; H only sets the
   plan length. Keep it that way.

## Decision experiments (run FIRST — they discriminate, cheap, data-only)

D1. **Timing (leader vs shared trigger).** On the cleaned events: per participant, build the
    time series of horizon size (lead_corr at each fixation) and cursor speed; cross-correlate
    around corner passages. Gaze-leads is supported if the lead shrink precedes deceleration by
    ~100–250 ms (one visuomotor delay); simultaneous or reversed lag means a shared-trigger model.
D2. **Deceleration shape (emergent vs reactive).** From the human cursor traces: where does
    corner deceleration begin relative to the vertex (distance and time)? Anticipatory onset
    (well before the corner) is consistent with horizon/MPC-emergent slowing; onset at or after
    the vertex would indicate a reactive rule. Report per participant and width.

## Model variant to build if D1/D2 support it (S15, config-gated)

- Keep: additive-budget lead (clean constants), anchor + geometry channels, acc_max, pace tail,
  stability gate, routes, arrival/latency loop.
- Remove: the tau*theta time surcharge (T becomes max(T0, lead/v_max) only).
- Add: a terminal reachability constraint tied to the horizon — at the plan's last node the
  cursor must be able to stop (or turn to the visible corridor) within the remaining seen path:
  v_N^2 <= 2*a_max*(margin to horizon end), enforced as a stiff hinge like acc_max. Shrinking H
  then caps speed directly; corner slowing becomes an emergent prediction.
- Strengthen the slow loop if needed: h = max(budget lead, v*T_v) with T_v calibrated from the
  cleaned events (currently T_min = 0.1 fixed).

## Comparison and decision

Fit S15 for B (then A, C) and compare against S14 on: corner CT ratios and apex dips (20 vs 40 mm),
straight-tunnel speeds, smoothness metrics, joint losses, and D1/D2-style signatures reproduced by
the model itself. Choose the design that matches the human signatures; the paper's speed-adaptation
section then reads "emergent from horizon + reachability" (if S15 wins) or keeps the calibrated
turning-time with D1/D2 as its empirical justification (if S14 wins).

## Status hooks

S14: B and A fitted (evals in the chain), C paused at init — resume C only if S14 remains the
candidate after D1/D2. Uncertainty-weighted horizon: postponed; note as future work.

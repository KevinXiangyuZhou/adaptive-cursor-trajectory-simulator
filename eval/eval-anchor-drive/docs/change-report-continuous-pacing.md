# Change report: continuous pacing (time-density schedule)

**Status:** proposed, not implemented · **Owner:** simulation model · **Affects:** steering trajectories of all fitted personas

## One-paragraph summary
Fitted personas currently show a "stop–go–stop" speed pattern around corners and visible kinks/overshoots in the trajectory, which reviewers immediately read as un-human. The cause is not noise and not the planner's cost weights: it is how each plan's **time budget** is turned into a **pace**. Today the whole budget for a look-ahead window is spread evenly over that window, so a window that happens to contain a corner is driven slowly from start to end, and the next window — corner behind it — is driven fast from start to end. The proposed change keeps the exact same time budgets (nothing about the calibrated timing model changes) but spends them **where the geometry demands**: slow through the turn itself, fast on the straight parts. The speed profile becomes continuous across plans, the per-plan kinks disappear at the source, and no new parameters are introduced.

## What the user sees today
- Speed alternates from one look-ahead to the next (measured on participant B, corner 40 mm, deterministic run): 0.30 → 0.08 → 0.33 → 0.24 → 0.10 → 0.14 → 0.07 → 0.16 → 0.40 m/s, with an identical 48 mm look-ahead each time.
- Because the cursor's lateral behaviour is coupled to its speed, every pace jump also produces a kink or a swing in the path — the "dramatic jiggling/overshoot" in the Figure-3 panels.
- Humans on the same trial slow smoothly into the corner and speed up smoothly out of it; their speed changes follow the corners, not a clock.

## Root cause (plain language)
The gaze-timing model — calibrated from the eye data and not in question — says a plan gets `T = lead / v_max + τ · (turning inside the window) · (W_ref / W)`. That formula is correct as a **total**: a window with a 90° corner deserves ~0.3 s more than a straight window. The defect is in the next step: the planner converts `T` into a single average pace `lead / T` and asks the cursor to hold that pace across the entire window. So:
- window contains the corner → the *whole* window, including its straight parts, runs at ~0.08 m/s;
- next window is corner-free → the whole window runs at ~0.33 m/s.
The human spends the extra time *in the turn* and nowhere else.

## Proposed change
Replace the single average pace with a **time-density schedule**. Along the look-ahead, time per unit distance is
`dt/ds = 1 / v_max + τ · |κ(s)| · (W_ref / W(s))`
— exactly the integrand of the existing budget, so the total time for the window is unchanged to the millimetre. The plan's progress schedule `s(t)` (which the drive term, the sliding anchor of the motor replans, and the arrival logic already consume) is obtained by integrating this density instead of dividing lead by `T`. Consequences:
- Straight stretches inside a window are driven at `v_max`-pace; the turn is driven at its own slower pace.
- Two consecutive windows that overlap the same corner now agree on how fast to go through it, so the hand-over between plans is continuous in speed.
- The extra time is spent at the apex, which is also where the acceleration bound needs it — the arc through the corner becomes a smooth deceleration–turn–acceleration instead of a flat crawl followed by a jump.

## What does **not** change
- Calibrated timing constants (`T0`, `τ`, `β`, `v_max`), the gaze look-ahead model, the routes (Stage-1), all planner weights, the acceleration bound, the intermittent (plan–execute–replan) structure, the noise model.
- Total completion time predictions (the per-window budgets are identical); therefore the steering-law and completion-time results should be essentially unchanged.

## Expected impact
- Removes the fixation-frequency speed sawtooth (currently ±0.1 m/s on a 0.3 m/s base) and the associated path kinks — directly targets the smoothness complaint on Figure 3.
- Apex-speed statistics should move toward the human values (the model currently under-slows at the apex and over-slows on the legs; redistributing the same time fixes both directions at once).
- Likely improvement in speed-profile correlation in the joint loss, since the human profile is corner-shaped.

## Risks and side effects
- Implementation touches the plan-schedule construction in `cursor_simulator` (fixation plans and motor replans) — moderate surface, well-tested area; regression risk is in arrival/exhaustion bookkeeping, covered by the existing gaze-rhythm diagnostics (cycle length, arrival share).
- Personas fitted under the old pacing will need a refit (planner weights compensated for the sawtooth); budget ~1.5 h per participant on the local machine.
- Does not by itself fix the second smoothness problem (the collapsed lateral spring, contour ≈ 15); that one needs the smoothness statistics added to the fitting objective and is tracked separately.

## Validation plan
1. Deterministic trace on corner 40/20: per-plan pace must vary continuously (no alternation), total plan times unchanged.
2. Smoothness metrics vs human (high-frequency lateral RMS, correction rate, acceleration distribution) on the four Figure-3 conditions.
3. Full validation battery (eval-main completion times by width/type, gaze-lead rhythm) to confirm no regression.
4. Refit B, regenerate Figure-3 overlays, human-eye review.

## Effort
Implementation and deterministic checks: ~half a day. Refit + validation: ~3 h machine time. Decision needed: approve implementation.

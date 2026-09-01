# Anchor-drive planning (`speed_model: "none"`)

One model for steering and pointing with **no prescribed speed**. The GAM speed
prior, the free-space clearance gate, the LQR terminal cost and the lag weight
are all gone; what remains is one objective evaluated identically everywhere:

| term | form | note |
|---|---|---|
| effort | `jerk · Σ‖j_k‖²` | unchanged |
| tracking | `contour · Σ‖p_k − ref(s_k)‖²` | full distance to the progress point (lateral **and** along-path) |
| boundaries | corridor / cartesian hinges, weight `constraint` | unchanged, silent when far |
| damping | `free_velocity · Σ‖v_k‖²` | everywhere, not only in free space |
| **drive** | `goal · (anchor_s − s_{k_deadline})²` | via-point on **progress** at the deadline node |

`s_k` is kinematic progress — the plan's along-tangent velocity integrated along
the path — so there are no virtual progress variables. Tangents are taken on a
per-solve progress schedule and re-linearised to a fixed point
(`mpcc_model.ANCHOR_RELIN_PASSES`), which keeps the objective smooth on jagged
reference paths (a recursion through `tangent(s_k)` is chaotic when `v·dt·κ > 1`).

**Where speed comes from.** Every solve covers a fixed duration
`plan_deadline_s` (T_plan) and asks to be *at the gaze anchor* at that deadline.
The anchor is the difficulty-budget lookahead `h(W)` (Stage G, gaze-fitted), so
cruise speed is `h(W) / T_plan` — width adaptation is inherited from gaze, not
fitted to speed. Where the environment stops constraining, the budget density
vanishes, the anchor rests on the path end, and the same pursuit becomes
pointing (overshooting the end is tracking error, so braking into the target
emerges). Curvature adaptation is left to the effort term: in curves the plan
trades punctuality against jerk and arrives late, which is what the gaze data
show (onset lead/speed is width-invariant at ~0.19 s but rises from 0.14 s on
straights to 0.25 s in sharp sinusoids).

The realised cycle time is never scheduled: `T_cycle ≈ T_plan + Δκ + τ`, with the
existing arrival + lognormal-latency + deviation triggers deciding when a plan is
replaced.

## Config

```json
"speed_model": {"type": "none"},
"horizon_mode": "budget",          "replan_mode": "intermittent",
"plan_deadline_s": 0.19,
"planner_weights": {"jerk": ..., "contour": ..., "constraint": ..., "goal": ..., "free_velocity": ...}
```
`lag`, `progress`, `desired_speed`, `goal_precision` are ignored in this mode.

## Scripts

* `probe_anchor.py` — GAM persona vs anchor persona on the gaze cohort
  (A=P105835, B=P170114, C=P160254): fit-script tunnel/pointing losses (train /
  held-out), CT ratio, lateral RMSE, cruise speed, and the model's
  time-to-anchor by width and tunnel type (the zero-tuning check against the
  human table). `--quick` runs a straight/sharp/corner subset.
* `fit_anchor.py` — **one-stage** joint CMA-ES over the shared weights
  (`jerk, contour, constraint, goal, free_velocity`) + `plan_deadline_s`, on the
  training widths **and** training radii together (no tunnel/pointing stages).
  Writes `results/{pid}_anchor_config_s{seed}.json` (runnable persona) and
  `results/{pid}_anchor_fit_s{seed}.json` (fit record + held-out probe).

## Human reference (gaze cohort, fixation onsets)

| width | lead (m) | speed (m/s) | lead/speed (s) |
|---|---|---|---|
| 10 mm | 0.016 | 0.068 | 0.198 |
| 30 mm | 0.034 | 0.167 | 0.202 |
| 50 mm | 0.046 | 0.273 | 0.175 |

| straight | corner | gentle | mid | sharp |
|---|---|---|---|---|
| 0.136 s | 0.178 s | 0.224 s | 0.220 s | 0.254 s |

## Results so far (2026-08-28, quick subset, noiseless, hand weights)

* **Emerges:** width adaptation (model lead/v ≈ 0.2 s at every width, human 0.19 s;
  cruise 0.13→0.33 m/s across 10→50 mm for B), aggregate completion-time ratio ≈ 1.0
  (B: 0.99 train / 1.00 test), pointing from the same objective (MT ratio ≈ 1.0–1.3).
* **Does not emerge:** curvature braking. The model's time-to-anchor is the same for
  every tunnel type; humans go 0.136 s (straight) → 0.254 s (sharp). Over jerk
  5e-7…2e-5 and contour 3…58 the straight/sharp completion-time ratio spread stays
  ≈ 2.3–2.6 (`results/sweep_B*.log`); raising jerk slows all types uniformly;
  damping = 0 breaks pointing (nothing stops the cursor). Human onset lead is the
  same across types at a given width, so the missing channel is motor-side.
* Fitted GAM persona (B, same subset): tunnel loss 5.4 / 4.6 vs anchor-drive 9.6 / 7.6
  (best sweep row: jerk 6e-6, free_velocity 0.003, deadline 0.15–0.2 s).
* Joint CMA-ES: one candidate ≈ 33 sims ≈ 3–8 min → cluster job (`fit_anchor.py`),
  4 generations in 20 min locally.

## 2026-08-28 (later): centerline artifact, coast-safety result

* **Pipeline bug fixed** — `experiment/environment.py::_meters_to_pixels` truncated task
  waypoints to integer pixels (1 px = 1 mm). Narrow tunnels are sampled every ~2.5 mm, so
  ±0.5 mm quantisation made their centerlines zig-zag (|κ| 100–300 /m, ~200 sign changes);
  the cubic reference path interpolated the noise exactly. The sharp sinusoid is the **same
  curve at every width** (κ p90 ≈ 32 /m). All 8-26 personas were fitted on the jagged paths
  (B's GAM persona: quick-subset tunnel loss 5.4/4.6 → 7.7/6.8 on corrected paths).
* **Coast-safety (variant T)** implemented: `coast_safety` (default on in anchor mode),
  weight `planner_weights.safety`; the deadline state's ballistic continuation over the
  replan latency is pushed through the corridor hinge. Verified active (coasts sit on the
  bound) but on the true geometry it caps speed at ≈0.09 (10 mm) / ≈0.2 (50 mm) m/s and
  changes cruise by ≤12%; humans: 0.026 / 0.246. Deterministic geometry does not make a
  30 mm-radius bend slow; the human width×curvature interaction (sharp/straight speed
  0.33 at 10 mm → 0.73 at 50 mm, identical curve) is a margin-vs-execution-error signature.

## Bend-drift term (plant-lag-aware, tolerance-normalised) — 2026-08-28

`planner_weights.bend` (+ `plant_lag_s`, default 0.1 s; `bend_power`, default 2):
per node, drift = v·τ_p·tanh(κ·v·τ_p) (lateral drift of a lagged follower;
saturates at a sharp corner), cost = bend·(drift / room)^power, room = local
half-width. Zero on straights and in free space. Findings (B, quick subset):
* Sharp sinusoid by width: model 0.043/0.075/0.117/0.161/0.217 m/s vs human
  0.026/0.079/0.104/0.179/0.246 (bend = 1e-2) — the curvature × width pattern
  emerges; sharp CT ratio 1.10 (0.65 without the term).
* Corners over-braked (CT ratio ≈ 2.3, the 10 mm corner crawls) for any weight
  or exponent that fixes sinusoids. Humans: 0.073 m/s at 10 mm corners vs 0.078
  on straights vs 0.026 in the sharp sinusoid — corners are *cheap*, sustained
  bends are not. No single lag constant fits both (τ_p ≈ 0.03 s leaves corners
  alone but makes sinusoid drift negligible): corners are planned direction
  changes, not lag-limited steering.
* Pointing is untouched by the term; goal 50 / free_velocity 0.02 / contour 30
  brings anchor pointing to loss 7.1/11.8 (aug-26 GAM persona 6.0/6.8).
* Deadlines below 0.15 s (3-node horizon) destabilise solves — keep ≥ 0.175.
`compare_aug26.py` prints the full-set comparison against the aug-26 personas.

## Comparison with the aug-26 personas (full trial sets, corrected geometry, same build)

`compare_aug26.py` — aug-26 GAM personas (per-participant fitted, incl. GAM) vs
anchor-drive with ONE shared hand configuration (jerk 6e-6, free_velocity 0.02,
goal 50, contour 30, safety 5000, plant_lag 0.1, deadline = gaze t_cross):

| | tunnel loss tr/te (A / B / C) | pointing loss tr/te (A / B / C) |
|---|---|---|
| aug-26 GAM | 5.9/5.4 · 7.8/6.3 · 11.2/11.5 | 6.6/7.5 · 6.0/6.8 · 6.8/9.2 |
| anchor + bend | 10.6/9.1 · 12.2/11.5 · 25.1/24.9 | 11.5/11.6 · 11.9/12.8 · 10.2/12.5 |
| anchor no-bend | 11.2/7.8 · 10.7/8.5 · 20.1/20.4 | same |

The fitted GAM personas win on every loss. Two structural reasons on the anchor
side: (1) `free_velocity` serves two roles — pointing needs ≈0.02 to stop in the
target, tunnels need ≈0.003 to cruise (with 0.003 the tunnel losses were 9.5/7.6
for B, with 0.02 straights run at CT ratio 2.3–3.0); (2) the curvature channel:
the bend term fixes sinusoids but over-brakes corners. Width scaling and the
width-invariant time-to-anchor remain the anchor model's genuine wins; both are
things the GAM prescribes rather than explains.

## Local quick fit (B) — 2026-08-28

`fit_anchor.py --quick` (9 tunnels + 18 pointing rounds per candidate, 7 params incl.
`bend`): 10 generations in 32 min, joint loss 26.6 → 25.2, drifted to a weak-drive
region (goal 3.5, deadline 0.31). Full-set result vs aug-26 GAM persona (B): tunnel
11.7/10.5 vs 7.8/6.3, pointing 11.1/10.7 vs 6.0/6.8. Not a converged fit — use
`fit_anchor_all_participants.sh` (Great Lakes, ~200 generations in 10 h).
Also: `end_velocity` (velocity cost past the path end) was tried as a geometric
replacement for global damping at the target and destabilises solves (node-set
switch); left inert (weight 0).
* 25 ms `Interval` (to allow the 0.136 s straight-tunnel deadline): not a clean test —
  the objective is not dt-invariant (jerk, bend, safety and damping sums all scale with
  node count), and with the 50 ms weights the model ran 2× slower everywhere
  (lead/v 0.9–2.6 s). A dt-invariant objective (per-time rather than per-node weights)
  is a prerequisite for changing the control interval.

## 8-hour local fit (B) — launched 2026-08-29

`fit_anchor.py --pid P170114 --time-limit 28800 --popsize 12 --workers 12 --tag _8h`
on the full training set (15 tunnels + 18 pointing rounds per candidate; ~5 min per
generation → ~90 generations). Spec: jerk, contour, constraint, goal, free_velocity,
bend, plant_lag_s (0.02–0.5), plan_deadline_s (0.15–0.40, 50 ms steps). Outputs:
`results/P170114_anchor_config_8h_s42.json`, `results/P170114_anchor_fit_8h_s42.json`
(fit record + full-set held-out probe), `results/fit_8h_P170114.log`. Compare with
`python compare_aug26.py` after copying the probe summary, or read the record directly.

### 8-hour fit result (B) — 2026-08-29

299 generations, joint loss 25.3 → 17.8. Fitted: jerk 3.1e-4 (**at the upper bound**),
contour 346, constraint 552, goal 576, free_velocity 0.031, bend 2.1e-3, plant_lag 0.16,
deadline 0.25. Full-set held-out vs aug-26 GAM persona:

| | tunnel tr/te | CT ratio | pointing tr/te | MT ratio | Fitts b (human 0.170) |
|---|---|---|---|---|---|
| aug-26 GAM | 7.81 / 6.29 | 0.79 / 0.82 | 6.01 / 6.76 | 0.90 / 0.88 | 0.140 |
| anchor 8h | 8.28 / 7.19 | 1.18 / 1.29 | 10.43 / 10.09 | 1.02 / 0.80 | 0.533 |

By width the two err in opposite directions (GAM CT ratio 0.52 at 10 mm → 1.03 at
50 mm; anchor 0.93 → 1.30–1.42). By type the anchor model is punctual on straights
(lead/v 0.12 vs human 0.14) yet slow overall (CT ratio 1.98) — a slow, jerk-limited
start; the fit pushed jerk to its bound to tame pointing peak speeds (ratio 1.55),
because the deadline demands a 0.2 m target in 0.25 s. Added `plan_vmax` (max hand
speed, default 0.8 m/s): deadline = max(T_plan, lookahead / v_max) — inert in
tunnels, stretches the pointing deadline to a physiological demand.
* `plan_vmax` test (fitted 8h weights, B quick pointing): 0.8 inert (h/v ≈ deadline),
  0.5 worse at R=5 mm (1.8 s), 0.3 → peak-v ratio 0.92–1.24, loss 7.2/7.5 vs 6.7/7.7.
  Small-target approach (MT 1.15–1.8 s vs 0.78) is the remaining pointing gap, not peak
  speed. Continuation fit `_8h2` launched 2026-08-29: init at the 8h vector, jerk bound
  widened to 1e-2, `plan_vmax` (0.2–1.0) in the search.

### Continuation fit `_8h2` (B) — stopped at generation 182 (loss flat), 2026-08-29

Fitted: jerk 2.0e-6, contour 107, constraint 999 (upper bound), goal 110,
free_velocity 0.0136, bend 3.2e-4, plant_lag 0.092, **deadline 0.15 s (floor; human
straight-tunnel 0.136 s)**, plan_vmax 0.41 m/s. Joint loss 17.8 → 14.69.

| B, full sets | tunnel tr/te | CT ratio | lat | pointing tr/te | MT ratio | peak-v | Fitts b (hum 0.170) |
|---|---|---|---|---|---|---|---|
| aug-26 GAM | 7.81 / 6.29 | 0.79 / 0.82 | 4.0 mm | 6.01 / 6.76 | 0.90 / 0.88 | 1.02 | 0.140 |
| anchor 8h | 8.28 / 7.19 | 1.18 / 1.29 | 4.3 mm | 10.43 / 10.09 | 1.02 / 0.80 | 1.55 | 0.533 |
| **anchor 8h2** | **7.55 / 5.99** | **0.94 / 0.98** | 4.0 mm | 7.21 / 8.65 | 0.84 / 0.86 | 1.04 | 0.192 |

By width (CT ratio): GAM 0.52 / 0.72 / 0.90 / 0.92 / 1.03; anchor 0.76 / 1.02 / 1.02 / 0.95 / 1.07.
By type: anchor straight 1.44 (slow start-up), corner 1.00, gentle 0.83, sharp 0.79
(GAM 1.06 / 0.84 / 0.86 / 0.57). Pointing MT by radius matches the GAM persona within 0.05
at every R. Stochastic persona for eval-main: `results/P170114_anchor_persona_8h2.json`
(noise on, latency cv 0.7633); eval-main outputs in `results/eval-main-{anchor,gam}-B/`.

### eval-main law analysis (B, noise on, same build) — `results/eval-main-{anchor,gam}-B/`

| | Fitts (aligned) MT = a + b·ID | R² | TP | Steering law MT = a + b·ID | R² | per-trial CT ratio |
|---|---|---|---|---|---|---|
| human | 0.282 + 0.170·ID | 0.34 | 4.55 | −0.62 + 0.184·ID | 0.61 | — |
| aug-26 GAM | 0.255 + 0.140·ID | 0.91 | 4.71 | 1.40 + 0.049·ID | 0.69 | 0.82 |
| anchor 8h2 | −0.140 + 0.253·ID | 0.68 | 5.10 | 1.66 + 0.079·ID | 0.62 | 1.10 |

Both models under-slope the steering law (human 0.184 s per unit ID); the anchor persona
recovers more of it (0.079 vs 0.049) with completion times 10% above human vs 18% below.
Fitts: anchor slope 0.253 vs human 0.170 (GAM 0.140), lower R². ID4SCS regressions are
empty for the gaze cohort (too few segmented-tunnel trials for the 4-parameter fit).

## Simplification round (2026-08-29) — ablations on the fitted B persona (`ablate_anchor.py`)

Quick subset, noiseless losses; gaze stats from a noise-on gaze-lead run (human: cycle 0.38 s,
~80% of cycles end by arrival). Human straight tunnels: B cruises 0.56–0.70 m/s at every width
(CT 1.2–1.3 s) — ballistic to the end; the width-only budget capped the model at lead/0.15 s.

| variant | tunnel tr/te | corner | sharp | straight | cycle | arrival/early |
|---|---|---|---|---|---|---|
| full fit (8h2) | 10.48 / 7.19 | 1.29 | 0.87 | 1.62 | 0.20 s | .28 / .56 |
| − bend | 10.10 / 7.21 | 1.15 | 0.79 | 1.57 | 0.20 | .27 / .55 |
| − coast safety | 10.00 / 8.41 | 1.03 | 0.69 | 1.62 | 0.20 | .30 / .57 |
| S1 = −bend, distance arrival, deviation 0.3 | 9.94 / 7.47 | 1.18 | 0.81 | 1.69 | 0.35 | .61 / .15 |
| S1 + curvature-weighted budget D0 0.3 | 11.10 / 8.84 | 2.64 | 1.20 | 1.03 | 0.30 | .71 / .14 |
| **S1 + curvature-weighted budget D0 0.6** | **9.13 / 6.21** | 1.55 | 0.93 | 1.03 | 0.30 | .53 / .30 |
| S1 + curvature-weighted budget D0 1.2 | 8.89 / 6.82 | 1.36 | 0.78 | 1.03 | 0.30 | .37 / .47 |

Decisions: bend-drift term dropped (no effect); coast safety kept (held-out −1.2 without it);
`arrival_mode: distance` + `replan_deviation_frac 0.3` adopted on gaze grounds (rhythm matches
humans); gaze budget density changed to |κ|·(W_ref/W)^γ (`budget.curvature_weighted`): a
straight costs no budget → anchor at its end (10 mm straight CT 1.35 s vs human 1.30), a corner
is a lump the anchor stops at. Design "S2" = S1 + curvature-weighted budget, fitted with D0, γ
in the search (`fit_anchor.py --tag _S2 --patience 25`).

### S2 fit result (B) — early-stopped after 52 generations (2.9 h), 2026-08-29

Fitted: jerk 3.3e-7, contour 1329, constraint 13.4, goal 8684, free_velocity 0.274,
D0 0.58, γ 1.0 (bound), deadline 0.20 s, vmax 0.385. Joint loss 16.0 → 12.8.

| B, full sets | tunnel tr/te | CT ratio | lat | spd corr | pointing tr/te | MT ratio | Fitts b (hum 0.170) |
|---|---|---|---|---|---|---|---|
| aug-26 GAM | 7.81 / 6.29 | 0.79 / 0.82 | 4.0 | 0.22 | 6.01 / 6.76 | 0.90 / 0.88 | 0.140 |
| anchor 8h2 | 7.55 / 5.99 | 0.94 / 0.98 | 4.0 | 0.20 | 7.21 / 8.65 | 0.84 / 0.86 | 0.192 |
| **anchor S2** | **5.96 / 4.94** | **1.03 / 1.03** | 3.9 | **0.34** | 6.82 / 7.55 | 0.80 / 0.75 | 0.241 |

By type (CT ratio): straight 1.01 · gentle 0.97 · sharp 0.84 · corner 1.35 (GAM: 1.06 · 0.86 ·
0.57 · 0.84). By width: 0.86 / 0.99 / 1.06 / 1.08 / 1.21. Pointing MT by radius within 0.05 of
the GAM persona except R=10 mm (0.72 vs 1.00). Personas: `P170114_anchor_config_S2_s42.json`
(noiseless, as fitted), `P170114_anchor_persona_S2.json` (noise on); eval-main in
`results/eval-main-S2-B/`, gaze lead in `eval/eval-gaze-lead/model-gaze-lead-S2-B/`.

eval-main (noise on, B): S2 steering law model MT = 1.46 + 0.078·ID (human −0.62 + 0.184·ID),
per-trial CT ratio 1.00 (GAM 0.82, 8h2 1.10); Fitts aligned MT = −0.125 + 0.238·ID, R² 0.55,
TP 5.56 bps (human 0.170 slope, TP 4.55; GAM 0.140, TP 4.71). Gaze lead (S2): cycle 0.30 s
(human 0.38), arrival/early/exhausted 0.66/0.25/0.10 (was 0.28/0.56/0.17), onset lead by width
17/37/52/47/76 mm on non-straights (human 16/23/34/40/46) — wide-tunnel leads too long with
γ at its bound; lead/speed in bends 0.16–0.21 s (human 0.18–0.25). Plot:
`eval/eval-gaze-lead/model-gaze-lead-S2-B/`.

### Corner stall → anchor lead floor (S3) — 2026-08-30

Gaze-lead plateaus at corners: the curvature-weighted budget parks the anchor on the apex; once
the cursor is there the via-point drive is zero, and long latency draws outlast the padded plan
(`exhausted` replans from the same spot) → ~1 s stall per corner. Fix: `anchor_lead_floor`
(anchor ≥ cursor + arrival tolerance = local half-width; no new constant). Quick subset:

| | tunnel tr/te | corner | sharp | straight | gaze cycle | leads by width (mm) |
|---|---|---|---|---|---|---|
| S2 | 8.56 / 6.32 | 1.38 | 0.83 | 1.05 | 0.30 s | 20/40/51/48/76 |
| S2 + lead floor | 7.40 / 6.89 | **1.07** | 0.83 | 1.05 | 0.30 | 20/39/69/69/70 |

Wide-width leads are long (human 34–46 mm at 30–50 mm) with γ at its bound → S3 refit from the
S2 vector with γ ∈ (0.3, 1.5), `--patience 20` (`fit_S3_P170114.log`).

Pointing scatter (eval-main, noise on, B): S2 MT_kin 0.74±0.31 s vs human 0.90±0.33 (GAM 0.76±0.15);
residual sd around the Fitts line 0.21 s (human 0.28, GAM 0.07) — the anchor model's lower Fitts
R² (0.55) is human-like trial-to-trial variability, not misfit; the GAM persona is unnaturally tight.
Systematic part: easy targets too fast (ID 2–3: 0.43 vs 0.72 s) → negative Fitts intercept; the
deadline (0.2 s) + vmax bound short movements below the human ~0.5 s floor.

## Corner fixations ("wait at the apex") — 2026-08-29

**Human (gaze cohort, corner tunnels, 212 fixations):** fixations whose anchor sits on a corner
apex (16%, ≈ one per corner) last 0.445 s vs 0.325 s elsewhere (1.37×, Mann–Whitney p = 7e-5;
A/B/C 1.28–1.48×; 1.8× at 10 mm → 1.2× at 50 mm). Split: catch-up +0.10 s (t_cross 0.245 vs
0.145 s, speed dip to 0.16 of onset vs 0.34) and post-arrival dwell +0.10 s (0.29 vs 0.19 s);
corner leads shorter (17 vs 29 mm). Script: scratchpad `corner_human.py` (apex = polyline
turn > 20°, gaze within 12 mm).

**S2 model:** the curvature-weighted budget parks the anchor on the apex on 91% of cycles
(4 mm leads) — a burst of re-fixations of the same apex, corner cycles *shorter* (0.25 vs
0.35 s). Not the human pattern.

**Revisions tested (config flags, `cursor_simulator.py`):**
* `anchor_memory` — budget charged from the last anchor, not the cursor (eyes don't look
  back); alone insufficient (still 88% at apex: the lump spans several budgets).
* `corner_consume` — a fixated corner is consumed: next budget starts where its curvature
  ends. → one fixation per corner, corner cycle longer than others (stochastic 0.35 vs
  0.20 s), catch-up 0.20–0.275 s (human 0.245). But narrow corner tunnels become far too fast
  (10 mm CT ratio 0.36): a planned 90° turn is free for a deterministic planner.
* coast-safety at 100× the drive (`safety` 5e5) — binds (coasts on the bound) but only
  guards the tail; apex arrival speeds stay 0.2–0.4 m/s.
* `acc_max` (peak-acceleration hinge, `mpcc_model.py`; universal) — 4 m/s² gives the human
  corner shape (corner cycles 0.55 vs 0.50 s, catch-up +0.10 s, 10 mm CT ratio 0.88) but
  32–36% exhausted plans and a few stalls: the fixed deadline cannot be met through a turn.
  An acceleration-aware deadline floor (`1/2 a t² + v t = h`) does not bind at 4 m/s² — the
  lost time is in the turn, not the straight-line kinematics.
Candidate "S3" = S2 + memory + consume + safety 5e5 + acc_max 4; losses in `s3_full.log`.

**S3 vs S2 (unfitted S3 with S2's weights; `s3_probe.py`, `results/s3_probe.log`):**

| B | tunnel tr/te | CT ratio | lat | corner CTr | pointing tr/te | MT ratio |
|---|---|---|---|---|---|---|
| S2 (fitted) full | 5.96 / 4.94 | 1.03 / 1.03 | 4.0 mm | 1.35 | 6.82 / 7.55 | 0.80 / 0.75 |
| S3 full | 8.17 / 8.29 | 0.99 / 1.13 | 5.2–5.6 mm | 1.19 | 6.90 / 8.38 | 0.86 / 0.85 |

S3 fixes the corner *fixation pattern* (one longer fixation per apex, +0.1 s catch-up) and the
corner completion ratio (1.35 → 1.19) but, unfitted, loses on lateral error (far anchors past
consumed corners → deeper cuts) and needs a refit; pointing unchanged. Fit S3 before judging.

### S3 (S2 + anchor lead floor) — fit stopped at the 4 h cap (~45 generations), 2026-08-30

Fitted: jerk 2.0e-6, contour 1750, constraint 901, goal 3031, free_velocity 0.30, D0 0.51,
γ 1.37, deadline 0.25, vmax 0.40.

| B, full sets | tunnel tr/te | CT ratio | spd corr | pointing tr/te | Fitts b (R²) | steering b | gaze cycle |
|---|---|---|---|---|---|---|---|
| aug-26 GAM | 7.81 / 6.29 | 0.79 / 0.82 | 0.22 | 6.01 / 6.76 | 0.140 (0.91) | 0.049 | — |
| anchor S2 | 5.96 / 4.94 | 1.03 / 1.03 | 0.34 | 6.82 / 7.55 | 0.238 (0.55) | 0.078 | 0.30 s |
| **anchor S3** | **5.15 / 5.11** | 1.04 / 1.03 | **0.37** | 6.58 / 8.07 | **0.165 (0.34)** | **0.155** | **0.35 s** |
| human | | | | | 0.170 (0.34) | 0.184 | 0.38 s |

By type S3: straight 0.99 · gentle 1.03 · sharp 0.97 · corner 1.09; by width 0.96–1.11. Gaze
rhythm 0.66/0.25/0.09 arrival/early/exhausted. Remaining mismatch: onset leads 12/28/47/64/83 mm
vs human 16/23/34/40/46 — the cursor loss drives γ up (long leads at wide widths → faster
cruise); D0/γ should be calibrated on gaze leads (Stage G) and only planner weights fitted on
cursor. Plot: `eval/eval-gaze-lead/model-gaze-lead-S3-B/`.

### Gaze-calibrated budget → S4 (2026-08-30)

Curvature-weighted budget calibrated on human onset leads by width (noise-on gaze-lead runs,
non-straight tunnels; human 16/23/34/40/46 mm): γ 0.66, D0 0.40 → 17/23/39/41/38 mm
(log-RMSE 0.11; γ 1.0/D0 0.3 → 0.25; γ 0.66/D0 0.6 → 0.44). Design S4 = S3 with the budget
frozen at these values (Stage G from gaze) and only planner weights fitted on cursor
(`fit_anchor.py --fix-budget --patience 15 --tag _S4`).
Quick-subset check before fitting (S3 weights, calibrated budget): tunnel 7.62 / **4.78**
(S3 weights with fitted budget: 7.40 / 6.89), speed corr 0.54 / 0.63 (previous best 0.37),
types corner 1.16 · sharp 1.07 · straight 1.04; pointing 5.89 / 7.78 — the gaze-calibrated
lead improves held-out steering before any refit.

### S4 result (planner-only fit, early-stopped after ~25 generations)

Fitted: jerk 3.8e-7, contour 1797, constraint 499, goal 9325, free_velocity 0.31, deadline 0.20,
vmax 0.41; budget fixed γ 0.66 / D0 0.40.

| B, full sets | tunnel tr/te | CT ratio | spd corr | pointing tr/te | Fitts b (R²) | steering b | gaze cycle | leads (mm) |
|---|---|---|---|---|---|---|---|---|
| aug-26 GAM | 7.81 / 6.29 | 0.79 / 0.82 | 0.22 | 6.01 / 6.76 | 0.140 (0.91) | 0.049 | — | — |
| anchor S3 | 5.15 / 5.11 | 1.04 / 1.03 | 0.37 | 6.58 / 8.07 | 0.165 (0.34) | 0.155 | 0.35 s | 12/28/47/64/83 |
| **anchor S4** | 6.31 / **4.55** | 1.01 / 1.04 | 0.38 | 6.59 / 8.27 | 0.225 (0.48) | 0.064 | 0.30 s | **18/33/32/42/41** |
| human | | | | | 0.170 (0.34) | 0.184 | 0.38 s | 16/23/34/40/46 |

By type S4: straight 0.98 · gentle 1.02 · sharp 0.98 · corner 0.97 — all within 3%. Residual:
by width 10 mm 0.74 (too fast) … 50 mm 1.23 (slow) → shallow steering-law slope; humans' time-
to-anchor in *bends* is 0.47 s at 10 mm vs 0.19 s at 50 mm, i.e. the (κ, W) interaction.
The only remaining (κ, W) term is coast safety (weight 5000, never fitted) → sweep.
Coast-safety weight sweep on S4 (5e3 / 5e4 / 5e5): no change in the width pattern (10 mm
0.66–0.71, 50 mm 1.23–1.26) or losses → the term is inert on this design (candidate for removal).
Next test: room-normalised tracking `contour_norm·(e/half-width)²` (target radius in free space).
Room-normalised tracking on S4 (`contour_norm` 0.2 / 1 / 5): produces a width channel (10 mm CT
ratio 0.73 → 2.17 → 3.13) but slows narrow **straights** just as much (0.99 → 1.54 → 2.02) — W
without κ; B's straights are width-independent, so rejected. The residual needs κ × W. Re-testing
the bend-drift term now that the lead floor removes the corner-apex stall.
Bend-drift re-test on S4 (1e-3 / 3e-3 / 1e-2, plant_lag 0.1): width pattern unchanged (10 mm
0.75–0.86, 50 mm 1.23); 3e-3 destabilised a 10 mm trial. All three deterministic κ×W candidates
(coast safety, room-normalised tracking, bend drift) are now rejected on this design; the
narrow-bend time-to-anchor (human 0.47 s at 10 mm vs 0.19 s at 50 mm) remains the documented
residual attributable to execution uncertainty (kept out of the planner by decision).
Coast safety on/off on S4 (quick subset): 6.99 / 4.27 with vs 7.42 / 4.87 without; corners 0.97 vs
0.88 (faster without). Small but real → kept (one-line justification, no fitted constant).

## Current design (end of 2026-08-30 round): S4

* **Gaze module**: curvature-weighted difficulty budget |κ|·(W_ref/W)^γ, calibrated on human
  onset leads (γ 0.66, D0 0.40); T_min floor; anchor lead floor (anchor ≥ cursor + local room).
* **Planner (MPCC, one objective everywhere)**: jerk effort, full-distance tracking, corridor /
  cartesian hinges, velocity damping, arc-length via-point drive at the deadline node, coast-safety
  hinge over the replan latency. Kinematic progress on a fixed tangent schedule (re-linearised).
* **Timing**: fixed plan deadline (0.20 s fitted; floor lookahead / v_max, v_max 0.41 fitted).
* **Scheduler**: arrival = cursor within local room of the anchor point, then lognormal latency
  (gaze); early replan at 30% of local width; exhaustion backstop.
* **Removed this round**: bend-drift term + plant lag (inert), progress-based arrival (corner
  stalls), width-only budget (straights), cursor-fitted budget constants (long leads).
* **Fitted on cursor**: jerk, contour, constraint, goal, free_velocity, deadline, v_max (7).
* **Known residuals**: narrow-bend time-to-anchor (κ×W interaction; execution uncertainty),
  easy-target pointing too fast (~0.3 s vs human ~0.5–0.7 s floor).

## Review synthesis + fixes (2026-08-30, from the parallel session's independent review agent)

A peer session evolved this model in parallel (its S3 used anchor_memory / corner_consume /
acc_max — all config-gated, default OFF, so this line's results are uncontaminated) and had an
independent review agent audit it. Findings live for THIS line's S4, now fixed:
* **Deadline-before-lead-floor ordering** (floor extended the anchor after n_base was set):
  floor now applied before the deadline. Effect on S4 quick subset: test 4.27 → 4.55, corner
  0.97 → 1.04 (deadline now stretches with floored anchors — semantically correct; refit will
  re-balance).
* `_debug['solves']` unbounded growth in long fits → ring buffer (last 8).
* `np.resize` cyclic tiling on short schedules → pad with last value.
Deferred with evidence pending: put `safety` in the fit spec (currently hand-set 5000, shown
inert 5e3–5e5); reviewer's re-fixation suggestion (keep the anchor on a non-crossed cycle
instead of spending a fresh budget) — targets the crossed-fraction gap (S4 0.65 vs human 0.82);
next-round ablation. NOTE: two sessions edit these files concurrently — coordinate before
committing.

## Stall bug → via-schedule (S5) — 2026-08-30

User-priority bug: mid-tunnel stalls (speed → 0.01–0.02 m/s for 0.3–0.65 s; the flat near-zero
segments in the gaze-lead plots, e.g. trial 5). Mechanism, verified by trace: after arrival the
latency/exhaustion wait executes the plan's post-deadline tail, which has **no drive** — with
cheap jerk and damping 0.31 the optimizer brakes to ~0 right after the anchor. Humans dwell at
~90% cruise and overrun ~0.5·lead. Fix (no new constant class): the via-point defines a progress
**schedule** — be at the anchor at the deadline and continue at that pace through the tail,
targets clipped at the path end (goal settling preserved). `planner_weights.via_tail`.
Evidence (trial 5): min speed 0.012 → 0.22 m/s, zero sub-0.05 gaps, CT 2.75 → 1.60 s. At S4
weights everything becomes too fast (bends CT ratio 0.4–0.5) because the fitted times had
absorbed the stalls → refit S5 = S4 + via_tail, with `via_tail` and `safety` in the spec
(9 params, `--tag _S5 --patience 15`).

## BUMP-style two-timescale intermittency (S6) — 2026-08-30

Following Do/Chang/Lee CHI'21 (whose BUMP motor-control module our plant already derives from):
the scheduler had MERGED two timescales — fixations (2–3 Hz anchor updates) and motor
re-planning. BUMP keeps them separate: plans refresh every Tp (0.1 s) and only the first Tp of
any plan is executed, so an unplanned tail can never run (the stall is impossible by
construction); the ~0.19 s replan latency maps onto BUMP's SA→RP pipeline as a PERCEPTION delay
on anchor updates, not a motor freeze.

Implementation (`motor_period_s`, default 0 = old behaviour): fixation events unchanged
(curvature-weighted budget, arrival-within-room + lognormal latency, lead floor, fixation-level
exhaustion backstop); between fixations the MPCC re-solves every 0.1 s with the anchor slid
along the fixation's progress schedule (pace = lead/deadline, vmax-capped). Motor replans are
not logged as fixations (gaze plots/stats unchanged in meaning); arrival geometry carries over;
`via_tail` is superseded (removed from the spec). Manual tests (S4 weights): stalls 0/16 gaps
across trial 5 / sharp-10 / corner-40 (was 2/6 on trial 5 alone); speeds uniformly high at old
weights (pace saturates at fitted vmax) → S6 refit launched (8 params, --fix-budget,
--patience 15, vmax init lowered to 0.3).
BUMP-mode cost/bookkeeping: plan padding removed in BUMP mode (tails never execute; 4-node
solves), and the fixation lifetime backstop decoupled from solve length (deadline + 2·latency —
tying it to the unpadded plan length made fixations churn every 0.2 s and re-created stalls).
Sim cost in BUMP mode is still ~2–3× the single-plan mode (solves every 0.1 s); candidate for
later: cap re-linearisation passes on warm-started motor solves.
Slid motor-replan anchors get the same lead floor as fixations. Remaining sub-0.05 m/s windows
are all the FINAL fixation of a trial — the terminal deceleration into the end target (correct
behaviour); mid-tunnel stalls are zero on all four probe trials in BUMP mode. S6 fit relaunched
on the fixed code (tag _S6).

### S6 fit result (BUMP mode, 8 params, early-stopped after ~27 generations) — 2026-08-30

Fitted: jerk 5.6e-6, contour 2909, constraint 1000 (bound), goal 380, free_velocity 0.137,
safety 4.5e4, v_max 0.49, **deadline 0.40 (upper bound)**.

| B, full sets | tunnel tr/te | CT ratio | pointing tr/te | MT ratio | Fitts b (R²) | gaze cycle | arrival/early/exh |
|---|---|---|---|---|---|---|---|
| aug-26 GAM | 7.81 / 6.29 | 0.79 / 0.82 | 6.01 / 6.76 | 0.90 / 0.88 | 0.140 (0.91) | — | — |
| anchor S4 | 6.31 / 4.55 | 1.01 / 1.04 | 6.59 / 8.27 | 0.79 / 0.72 | 0.225 (0.48) | 0.30 s | .65/.27/.08 |
| anchor S6 | 8.07 / 5.94 | 0.96 / 0.96 | **5.52 / 6.85** | 0.98 / 0.92 | 0.232 (0.68) | 0.80 s | .29/.23/.47 |

Stalls gone and pointing best-ever (BUMP's 0.1 s re-planning helps the approach), but the gaze
rhythm broke: in BUMP mode the deadline sets the motor pace (lead/deadline), the fit used it as a
speed knob (→ 0.4 bound), and the fixation lifetime (deadline + 2·latency) went with it →
0.8 s cycles, 47% exhausted, leads 22–60 mm, lead/v 0.30–0.35. By width 0.57 → 1.43.
Fix (same principle as S4): the deadline is a gaze-measured crossing time → fix it at 0.2 s;
speed must come from v_max / damping / goal. Manual test then refit (S7).
Deadline-fixed test (S6 weights, deadline 0.2, v_max 0.3/0.4): gaze cycle back to 0.40 s but
cruise ≈ v_max at every width (10 mm CT ratio 0.31–0.37). Trace: on the 50 mm gentle sinusoid
the model is right (0.293 vs human ~0.30); on the 10 mm sharp sinusoid every fixation ends by
exhaustion — distance arrival (within the 5 mm room) is MISSED at speed (13 mm per step at
0.26 m/s), the fixation persists, the schedule keeps sliding, the cursor chases it at v_max,
and the next fixation starts downstream with a fresh 44–75 mm lead. Fix: arrival = reached OR
passed (distance within room, or progress beyond the anchor's arc position) — keeps the
corner-wedge immunity of distance arrival, closes the fast-pass hole.
**Bug (BUMP mode, found 08-31):** the plan's arrival geometry (`anchor_xy`/`anchor_s`/
`arrival_tol`) was rewritten on every 0.1 s motor replan with the *slid* anchor, which is ahead
of the cursor by construction — so after the first motor cycle arrival could never fire and
fixations ended by exhaustion (47% in S6). Fix: arrival is judged against the FIXATION's anchor
(stored at fixation time, restored on motor replans); arrival = reached OR passed. The S6 fit
and its validation predate this fix.
Re-test after the fix (S6 weights, deadline 0.2, v_max 0.3): gaze cycle 0.35 s (hum 0.38),
arrival/early/exhausted 0.72/0.22/0.06 (hum ~0.8/0.2), leads 21/36/39/44/45 mm (hum
16/23/34/40/46), lead/v in bends 0.13–0.23 s — rhythm fully restored. Tunnel losses at S6's
weights are poor (11.8/9.6; cruise ≈ 0.27 m/s everywhere, 10 mm CT ratio 0.40) because those
weights absorbed the runaway → S7 refit: BUMP mode, deadline 0.2 and budget FIXED, 7 weights
(`--fix-budget --fix-deadline --deadline 0.2 --vmax 0.25`, patience 15).

### S7 result (BUMP mode, arrival fix, deadline 0.2 + budget fixed; 7 weights, early-stopped ~gen 35) — 2026-08-31

Fitted: jerk 1.1e-7, contour 1878, constraint 258, goal 2.6, free_velocity 0.016, safety 2253,
v_max 0.69. Joint loss 18.1 → 13.12.

| B, full sets | tunnel tr/te | CT ratio | spd corr | pointing tr/te | MT ratio | Fitts b (R²) | gaze cycle | arrival/early/exh | leads (mm) |
|---|---|---|---|---|---|---|---|---|---|
| aug-26 GAM | 7.81 / 6.29 | 0.79 / 0.82 | 0.22 | 6.01 / 6.76 | 0.90 / 0.88 | 0.140 (0.91) | — | — | — |
| anchor S4 | 6.31 / 4.55 | 1.01 / 1.04 | 0.38 | 6.59 / 8.27 | 0.79 / 0.72 | 0.225 (0.48) | 0.30 s | .65/.27/.08 | 18/33/32/42/41 |
| anchor S6 | 8.07 / 5.94 | 0.96 / 0.96 | 0.36 | 5.52 / 6.85 | 0.98 / 0.92 | 0.232 (0.68) | 0.80 s | .29/.23/.47 | 22/48/48/60/42 |
| **anchor S7** | 7.34 / 6.06 | 0.86 / 0.90 | **0.43** | 5.78 / 7.34 | 0.97 / 0.89 | 0.239 (0.81) | **0.40 s** | **.70/.15/.15** | 23/31/44/44/43 |
| human | | | | | | 0.170 (0.34) | 0.38 s | ~.8/.2 | 16/23/34/40/46 |

By type S7: straight 0.98 · gentle 0.89 · sharp 0.75 · corner 0.86; by width 0.54 / 0.76 / 0.94 /
1.05 / 1.24. S7 is the first persona with stalls gone AND the gaze rhythm right; its steering loss
is above S4's because S4's narrow-tunnel times were partly stall time (right answer, wrong reason).
Residual unchanged: narrow-bend speed (κ×W). lead/v in bends 0.28–0.33 vs human 0.18–0.25.
Stall audit on the S7 persona (noise on, all 25 steering trials, mid-trial fixation windows,
"stall" = ≥0.15 s below 25% of the trial's cruise): 11 / 132 windows. Sinusoids of all
curvatures: 2 / 90 (one 0.15 s each). Corners: 7 / 36, mostly the 10 mm corner (0.15–0.6 s)
where humans also spend 12% of the time below 25% cruise (apex slowdown); one 0.6 s window
remains suspicious. The naive count (speed < 0.05 m/s anywhere) is confounded by the start-
from-rest fixation and by genuinely slow narrow tunnels (human 10 mm sharp: 0.026 m/s).

## Corner cutting / steering strategy (2026-08-31)

Why race-tracing vanished: S7 tracks the reference path to <0.1 mm (apex offsets 0.5 / 1.9 /
6.8 / 9.5 mm at corner20 / corner40 / sinus50 / gentle50) and the reference-path generator
cuts corners only 0.5–1.9 mm; the aug-26 persona cut 3.0 / 8.8 mm because its lateral weight
was tiny (contour 3.35) and the MPCC cut on its own. The merged full-distance tracking term in
anchor mode also carries the along-path progress coupling, so the fit drove it to ~2000 and
lateral freedom vanished with it.

Fix: separate weights again — `contour` = lateral adherence (strategy), `lag_anchor` =
along-path consistency (plumbing, stiff). B persona, lag_anchor = 1878:

| contour | corner20 cut (mean/p90 mm) | corner40 | sinus50 | corner40 CT |
|---|---|---|---|---|
| full-distance (1878) | 0.4 / 0.6 | 1.8 / 2.2 | 6.8 / 7.3 | 3.25 s |
| 300 | 1.0 / 2.5 | 1.9 / 2.5 | 6.8 / 7.2 | 2.88 s |
| 100 | 0.9 / 1.8 | 6.5 / 11.4 | 6.7 / 7.2 | 2.33 s |
| 30 | 2.4 / 4.8 | 8.4 / 16.4 | 6.5 / 7.3 | 2.10 s |
| human A / B / C | 2.8 / 2.8 / 3.0 | 3.3 / 4.3 / 2.8 | 6.4 / 8.1 / 13.2 | — |

Width dependence of cutting emerges from the walls; sinusoid cut depth is set by the
reference-path (Phase 0) parameters. Human strategy contrast (corner40): apex speed dip
A 0.13 (stop-and-go), B 0.65 (flowing), C 0.50; at corner20 everyone ~0.15.
Corner apex speed dip (model B persona, min speed at high-κ / median speed): full-distance
tracking 0.21 / 0.17 (corner20 / corner40); contour 100 → 0.27 / 0.33; contour 30 → 0.16 / 0.60.
Human B 0.17 / 0.65, A 0.15 / 0.13, C 0.16 / 0.50. One lateral weight moves the model from the
stop-and-go phenotype (A) to the flowing/cutting phenotype (B) and reproduces B's width-dependent
transition; the reference-path (Phase 0) parameters set sinusoid cut depth (A 6.4 / B 8.1 /
C 13.2 mm). `lag_anchor` is config-gated (None = legacy full-distance tracking).

## S8 round (2026-08-31): strategy weights

Phase-0 reference-path refit on corrected geometry (B, spatial-only, 4 generations): held-out
lateral RMSE 3.51 mm (old params) vs 3.56 mm (new) — the aug-26 Phase-0 parameters are
already adequate on the corrected centerlines; kept. S8 = S7 with `lag_anchor` = 2000 fixed
(along-path coupling) and `contour` free (lateral adherence = steering-strategy parameter);
same 7-parameter joint fit, deadline 0.2 / budget fixed. Evaluation of the paper's claims via
`strategy_stats.py` (cut depth + apex dip per condition, human vs personas).
A budget calibration (gaze leads, non-straight; human A 16/20/32/30/40 mm): γ 0.66, D0 0.3 →
11/25/28/25/39 (log-RMSE 0.24, cycle 0.45 s vs human 0.50, 76% arrival); D0 0.45 → 0.39;
D0 0.65 → 0.69. A persona: D0 0.35, γ 0.66, deadline 0.25 (A crossing time), A's gaze latency
(0.205 s, cv 0.84), A's aug-26 reference-path parameters; to be fitted with the S8 design after B.

## S9 (09-01) — corner-cutting / corner-speed phenotypes: width-only gaze lead + turning-time deadline

**Why S8 failed.** S8 (lag_anchor 2000 fixed, contour free) drove `contour` → 1573 and `safety` → 9.6e5:
the joint loss is nearly flat in `contour` (6.59 → 6.26 for 30 → 1573, noise on) and `safety` is
loss-neutral noise-off but blew up the noise-on persona (171 mm excursions at 1.65 m/s on corner 20 mm).
The model tracks the Phase-0 reference path exactly (model cut == ref-path cut: 0.5/1.9/6.9/9.1 mm at
corner20/40, sinus50, gentle50 vs human B 2.8/4.3/8.1/12.3) — nothing in the objective rewards a
shortcut (projected progress under-counts it; a spatial via-point |p−anchor|² was tried and dropped:
no extra cut, 9–25 % slower). Between-individual *route* strategy already lives in Phase-0 (w_cut A
0.30 vs B 0.79); the paper's contrast is mostly *speed at the apex* (A dip 0.13 vs B 0.65 at 40 mm).

**Gaze evidence (eval-gaze-cursor, `results/turn_time_calibration.json`).**
- Human fixations cross corners at every width (B corner-trial onset lead 11/31/37/50/55 mm at 10–50
  mm); the κ-weighted budget (turn toll 1.57·(W_ref/W)^γ ≫ D0 0.4) can never cross a 90° corner →
  truncated lead → apex over-braking (B corner40 dip 0.32 vs 0.65).
- Width-only lead fits: lead = D0·W^γ·W_ref^(1−γ): B γ 1.0/D0 1.21, A γ 0.66/D0 0.94, C 0.66/0.99.
- Time-to-anchor grows with turning angle in the lead and is tolerance-scaled: LAD fit
  T = T0 + τ·θ·(W_ref/W)^β, β = 1: B 0.13 + 0.23·θ·(W_ref/W) (corner crossing 0.46 s at 20 mm → 0.26 s
  at 40–50 mm); A 0.21 + 0.25·θ·(…) with ≈0.5 s corners at all widths (stop-and-turn); C 0.15 + 0.24·θ.
- B's straight-tunnel leads are ~40 mm at all widths with ttc ≈ 0.13 s (mean v 0.33–0.36 at all
  widths); A's straight speed falls with width (0.06 → 0.29 m/s).

**Design changes (all config-gated, defaults unchanged).**
- `budget.curvature_weighted=false` (width-only lead) + `plan_turn_time_s` τ, `plan_turn_width_exp` β:
  t_plan = max(T0, lead/v_max) + τ·θ_lead·(W_ref/W_local)^β; T0 = `plan_deadline_s` (gaze prior,
  fitted 0.08–0.30). Horizon floor is numerical only (3 nodes): if t_plan is shorter, the via-point is
  the schedule position at the horizon end (pace unchanged) — same rule the motor replans use.
- Via-point lateness normalised by the lead, goal·((anchor−s)/lead)²: stiffness no longer scales with
  lead² (fitted goal 2.6 was inert under short leads; goal 100 made walls soft → runaways).
- Numerical guards: monotone best-iterate safeguard in the anchor re-linearisation loop (2-node
  horizons + unbounded jerk ran to 5e3 m/s); `abort_on_breach_m` ends a trial that leaves the tunnel
  (the experiment restarts such trials) and the fit/probe score it as a failure; `safety` bound ≤ 1e4.
- Scripts: `contour_decomp.py` / `variant_decomp.py` (per-term loss + cut depth / apex dip per variant).

**Pre-fit check (B, S8c weights, goal 0.5–8, T0 0.13, γ 1.0, τ 0.23, β 1; noise off):**
corner20 CTr 0.87 (dip 0.47, human 0.17), corner40 CTr 0.82–1.16 (cut 1.5–2.8 mm, dip 0.59–0.84;
human 4.3 mm, 0.65), corner50 CTr 0.95–1.24 (cut 2.3–3.1, human 5.6), sinus50 CTr 1.10–1.16
(dip 0.74–0.78, human 0.97), straight30 CTr 1.5 (lead 36 mm / 0.13 s = 0.28 m/s vs human 0.35 mean).
Fit S9 launched for B (`fit_S9_P170114.log`): budget fixed, τ/β fixed from gaze, T0/v_max/6 weights free.

**S9 fit (B, 23 gens, loss 13.47):** jerk 1.2e-7, contour 228, constraint 15.8, goal 1.13, damping 0.268,
safety 399, T0 0.12, v_max 0.49. Validation: tunnel 6.29/5.79, CT ratio by width flat 0.87–0.98 (S8c
0.66–1.30), steering-law b 0.107 (human 0.184; S8c 0.065); strategy (noise on): corner40 cut 3.5/6.2 mm
dip 0.58 (human 4.3/6.3, 0.65), corner20 1.3/2.7 dip 0.41 (human 2.8/6.4, 0.17), sinus50 8.0 (8.1) —
width-dependent cutting emerges. Pointing regressed (MTr 0.74–0.85, Fitts R² 0.43).
**Rhythm bug in S9 (fixed before S9c):** the 3-node horizon floor made motor plans shorter than the
motor period, so the scheduler's plan exhaustion fired at every 0.2 s tick (94 % "exhausted", cycle
0.20 s). Floor is now max(3, motor_steps+2) for fixation and motor plans; afterwards all fixations end
by arrival, cycle 0.31–0.35 s (human 0.38), CTs at human values (corner40 2.65 vs 2.50 s, corner20
4.55 vs 5.33, sinus50 1.70 vs 1.68, straight30 1.70 vs 1.28). S9c = refit under the fixed rhythm (B,
then A with T0 0.21 / τ 0.25 / β 1 / D0 0.94 / γ 0.66).
Pointing note: under the fixed rhythm the S9 persona's pointing MT is 0.9–1.07 s (R 25→5 mm), i.e. a
weak width effect; the goal-precision well (`goal_precision` 1e-4…1e-2) adds ≤0.1 s at R 5 mm — a
Fitts-strength width term for pointing remains open.

**S9c (B, rhythm fixed; loss 12.89):** jerk 1.17e-7, contour 176, constraint 224, goal 1.12, damping
0.30, safety 225, T0 0.11, v_max 0.49. Validation: tunnel 6.29/5.22 (best held-out anchor loss without
stalls), CT ratio by width 0.99/1.00/1.01/0.94/1.05, by type corner 0.94 gentle 0.99 sharp 0.75
straight 1.46, steering-law b 0.118 (human 0.184; S8c 0.065), gaze cycle 0.35 s (human 0.38) with 90 %
arrival-triggered fixations, leads 9/16/27/35/46 mm (human 16/23/34/40/46). Pointing 6.60/8.18,
MTr 0.87/0.80, Fitts b 0.238 R² 0.59 (open: no tolerance term in pointing time).
Strategy (noise on, 3 runs): corner20 cut 1.0/2.0 mm dip 0.31; corner40 1.9/3.6 dip 0.56 (human
2.8/6.4, 0.17 and 4.3/6.3, 0.65); sinus50 6.3 (8.1); gentle50 8.8 (12.3). Width contrast present in cut
depth and apex speed, cut depth ≈ half the human; `contour` sweep on this persona pending.

**Corner-cut ceiling = Phase-0 route generator (S9c-B, noise off).** For every planner lateral weight
(contour 15–176) the loss is flat (6.23–6.28) and the cut depth does not move (corner40 1.7–1.9 mm);
for every reference-path setting the model's cut equals the reference path's own cut (corner40:
Phase-0 1.9 → model 1.5/2.3; w_cut 1.0 → 2.4 → 1.8/3.0; sinus50: 6.9 → 6.4, 8.8 → 8.1). The generator
cannot produce B's 4.3 mm corner cut with any of its parameters (it is a sinusoid-oriented cut window),
so the residual corner-cut amplitude is a route-generation limitation, not a planner one; the planner
reproduces the width contrast in apex speed (dip 0.31 → 0.56 at 20 → 40 mm; human 0.17 → 0.65).

**S9c (A; loss 16.26):** jerk 7.4e-6, contour 817, constraint 26, goal 1.58, damping 0.19, safety 198,
T0 0.16, v_max 0.64. Validation: tunnel 8.73/5.92 (aug-26 GAM 5.87/5.36), spdcorr 0.04, CT by width
0.57/0.87/1.06/1.04/1.12, gaze cycle 0.35 s (90 % arrival), leads 10/17/21/25/30 mm (human
16/23/34/40/46). Strategy: corner20 cut 0.8/1.4 dip 0.37, corner40 2.5/3.7 dip 0.43 (human 2.8/5.5,
0.15 and 3.3/6.4, 0.13). A's stop-and-turn is NOT reproduced: the model's A/B apex contrast at 40 mm is
0.43 vs 0.56 (human 0.13 vs 0.65). Raising the coast-safety weight to 1e4/1e5 does not change it (dip
0.39–0.44) — the hinge is not binding. Next candidate: A's short corner fixations are rest points
(pointing-like); the via-point is by design not a rest point.

**Correction (old-implementation / old-config comparison).** The earlier "corner-cut ceiling is the
generator" claim was wrong: `generate_optimal_reference_path` is byte-identical from the old commit
(83fe93b, the attached old reference_path.py) to HEAD — only `find_closest_theta` (global coarse
search + safeguarded Newton) and the vectorised `tangents()` changed, and `_smooth_offsets` is dead
code in both. The corner cut is limited by the aug-26 Phase-0 *parameter point*, specifically the wide
`cut_window_frac` 0.139: phi = ∫|κ|ds over the window drives exp(-w_suppress·phi), which fires at an
isolated 90° corner as hard as in dense curvature. Reference-path-only comparison on B's eval-main
corner tasks (cut mean/p90, mm): aug-26 B 1.9/2.1 at corner40 vs the OLD fitted config (w_cut 0.55,
window 0.023, suppress 1.11 — old dataset participant) 4.1/5.5 ≈ human B 4.3/6.3; B with the old
narrow window 6.3/7.6 (over), with w_suppress=0 6.7/9.5. A Phase-0 refit that does not let sinusoid
RMSE swallow the window (corner-weighted, or window fitted per type) can recover human-scale corner
cuts with the existing generator.

**Phase-0 corner-fair refit (S9d, B).** refit_phase0.py: per-trial spatial RMSE normalised by the
trial's human round-to-round RMSE (floor 1.5 mm) — the pipeline's human-variability normalisation
applied to Phase-0. Fitted (32 gens, 15 min): w_cut 0.523, w_suppress 0.539, w_width_exp 0.601,
cut_window_frac 0.199, global_clearance_ref 0.005 (task scaling saturated on). Held-out raw RMSE
unchanged (2.79 → 2.78 mm); ref-path corner cuts: corner20 0.5 → 1.4, corner40 1.9 → 3.3, corner50
2.5 → 3.8 mm (human 2.8/4.3/5.6), sinusoids unchanged. Executed persona (S9d = S9c weights + this
route, noise on): corner40 cut 2.8/4.2 mm (S9c 1.9/3.6; human 4.3/6.3), corner20 1.3/2.1 (human
2.8/6.4), dips unchanged; probe losses: tunnel 6.33/5.46 vs 6.29/5.22, pointing identical.
Reference-path visualisations (no model runs, max_steps=0): eval/eval-main/refpath-viz-S9-B and
refpath-viz-S9d-B (viz_ref_paths.py).

**Stage-1 identifiability + cut-profile fit (S9e/S9f).** Follow-up on the corner-fair refit: the
trajectory-RMSE family (raw or human-variability-normalised, per-round or against the mean path) is
strategy-blind — deep- and shallow-corner routes differ by <0.2 mm in held-out spatial RMSE, so the
optimiser cannot identify the corner cut from it. Resolution: Stage-1 loss = mean-path RMSE + cut-depth
profile match (|route cut − human cut| at high-curvature points, per curved trial; refit_phase0.py
--loss cutmatch). B (S9f route): corner20/40/50 1.1/4.4/5.7 mm, sinus 8.5 (human 2.8/4.3/5.6, 8.1) at
unchanged held-out RMSE (2.78 mm); executed noise-on: corner40 3.1/4.7 (human 4.3/6.3), corner20
1.1/2.2, sinus 7.9 — width contrast 1.1→3.1 (human 2.8→4.3); losses hold (tunnel 6.35/5.18, best test).
A: the cut term overshoots (executed 4.5 vs 3.3) because A's stiff tracker (contour 817) does not
attenuate; A uses the mean-path route (S9e): executed corner40 3.3 = human 3.3, sinus 6.8 vs 6.4.
Execution notes: B attenuates route cut ~30% at the short apex window; stiffer contour (500/1000)
REDUCES B's executed corner cut (progress-lag skips the apex window) — contour stays 176.
Figure-3-style overlays (A=S9e, B=S9f, 5 noise-on runs vs human rounds): results/fig3_overlays_S9.png
(fig3_overlays.py). Remaining gaps vs the paper figure: corner20 absolute cut (1.1 vs 2.8), gentle50
depth (8.7 vs 12.3), and A's apex dip (0.43 vs 0.13, stop-and-turn).

**S10/S10b (B): fitted peak-acceleration bound + noise-on stability gate.** Reflection: the paper
draft's Figure 3 arcs came from carried speed + soft tracking (old contour 16), with geometry emerging
from dynamics; our stiff tracker executed the route instead. S10 added `acc_max` to ANCHOR_SPEC (hinge
on |a| — cornering radius v²/a turns carried speed into arcs) and the fit immediately chose the soft
regime (contour 23, loss 12.75) — but noise-off fitting made it wall-fragile (strategy p90 16.7 mm,
strays in the figure). S10b adds a noise-ON stability gate to every fit evaluation (_noise_stability:
3 widest train trials, 1 seed, breach/incomplete = failure penalty). Fitted: contour 14.8 (old model:
16), constraint 2705 (containment moved from centering to walls), acc_max 3.7 m/s², v_max 0.53, loss
12.54 (best). Strategy (noise on): corner20 cut 2.4/5.0 dip 0.42 (human 2.8/6.4, 0.17), corner40
3.2/5.4 dip 0.48 (human 4.3/6.3, 0.65), sinus 7.6/9.5 (8.1/12.4), gentle 7.8/9.4 (12.3/15.6) — no
breaches. Validation: tunnel 6.42/5.40, CT by width 1.00–1.07, straight 1.59 (residual), pointing
6.35/7.81 MTr 0.83/0.80, Fitts R² 0.65. Figure: results/fig3_overlays_S10b.png — smooth inside arcs at
Cor W=40, tight at W=20, all runs in-corridor. Current personas: A=S9e, B=S10b. Open: straight-tunnel
pace (1.59×), corner apex dips (0.42/0.48 vs 0.17/0.65 — the 20/40 contrast direction is right but
compressed), gentle-50 depth, A's stop-and-turn, pointing width term.

**Reference-path generator redesign (09-01).** Diagnosis of the route "ears" (outward bumps at corner
entry/exit, also present with the old fitted config): (1) the cut side/magnitude came from the pointwise
curvature of the cubic centerline spline, which rings at a polyline vertex (10 sign flips within ±50 mm)
so the offset flipped to the wrong side; (2) the interpolating spline overshoots sharp offset
transitions (bump ∝ cut depth); (3) the base spline through 10 mm waypoints overshoots vertices by
0.8 mm. Deeper: a normal offset of a sharp vertex is another equally sharp vertex — it cannot make a
fillet, and a deep inward offset folds into a loop (routes leaving the tunnel once the ringing was
removed). New generator (reference_path.py): the smoothest path (minimum curvature energy) within the
inside slack band a = w_cut·f_room·room_inside, knots p = C + d·n + e·t with tangential freedom e (what
lets a vertex become an arc), turn direction from the smoothed waypoint-polyline heading, base spline
through the densified polyline, one-sided band (no outward swings), band → 0 at inflections/ends;
sparse bounded least squares (~60 ms/path). Parameters per participant: w_cut, w_width_exp, w_center
(lobe extent), global_clearance_ref (w_suppress/cut_window_frac dropped). No bumps/loops by
construction; all 25 routes in-corridor for A and B with the old parameters. Stage-1 refits (cutmatch)
rerun under the new generator; all planner personas need the new routes.

**S12 (B): corrected pipeline (w_center pass-through, BLAS pinned), qp3 route, single plan + pace tail,
acc_max 4 fixed, no coast-safety; 7 fitted params.** Fitted: jerk 1.1e-7, contour 4.8, constraint 103,
goal 1.89, damping 0.23, T0 0.15, v_max 0.48; joint loss 13.22 (15 gens). Strategy (noise on): corner20
3.0/4.6 dip 0.48 (human 2.8/6.4, 0.17); corner40 3.8/6.6 dip 0.66 (human 4.3/6.3, 0.65); sinus 8.2/9.6
(8.1/12.4); gentle 8.9/11.4 (12.3/15.6) — cut depths human-scale at both corner widths and sinusoids.
Validation: tunnel 7.19/6.39, CT 1.17/1.15 (straight 1.81, widths 10–30 mm 1.2–1.3×), steering b 0.172
(human 0.184), gaze cycle 0.35 s 85 % arrival, pointing 6.61/7.52. Smoothness: corner40 HF lateral RMS
1.8–2.0 mm (human 1.2–2.0), corner20 0.9–1.4 (human 0.6–0.7). Open: overall pace (+15 %), corner-20
apex dip, gentle-50 depth. A and C fits/evals in progress.
**S12 (A):** fitted jerk 4.1e-7, contour 26, constraint 660, goal 4.0, damping 0.31, T0 0.17, v_max 0.76;
loss 15.59 (S9c 16.26). Strategy (noise on): corner20 1.9/3.6 dip **0.15 (human 0.15)**; corner40 4.7/7.5
dip 0.43 (human 3.3/6.4, 0.13); sinus 7.7 (6.4); gentle 8.1 (8.1). Validation: tunnel 8.53/6.06,
CT by width 0.58/0.90/1.10/1.08/1.12, Fitts b 0.143 (human 0.138), steering b 0.158 (human 0.369),
gaze cycle 0.35 s 84 % arrival, leads 10–30 mm (human 16–46). Open for A: 10 mm tunnels too fast,
corner-40 apex dip, steering-law slope.
**S12 (C, P160254):** fitted jerk 5.5e-7, contour 40, constraint 1861, goal 39.6, damping 0.13, T0 0.12,
v_max 0.77; loss 22.94. Evaluation pending.
**S12 (C) evaluation:** strategy corner20 2.4/4.4 dip 0.32 (human 3.0/5.8, 0.16); corner40 5.1/6.9 dip 0.47
(human 2.8/5.0, 0.50); sinus 13.3/14.4 (13.2/16.6); gentle 12.3/13.6 (13.7/18.2). Validation: tunnel
13.84/12.85 (GAM-era scale for C), CT 0.91/1.10, by width 0.61/0.94/1.06/1.29/1.16, straight 1.54,
Fitts b 0.330 (human 0.138), steering b 0.127 (human 0.305), gaze cycle 0.25 s (human 0.38) 80 %
arrival, leads 9–30 mm (human 16–46). C's routes match its deep sinusoid cutting; corners over-cut at
40 mm and the leads are short — C's gaze constants (D0 0.99, γ 0.73) deserve a re-check.
Figure-3 overlays for S12: results/fig3_overlays_S12.png. All S12 outputs: eval-main-S12-{A,B,C},
model-gaze-lead-S12-{A,B,C}, refpath-viz-S12-{A,B,C}, jiggle_diag_S12_{A,B,C}.png, strategy_S12_*.log.

---
name: generate_model_report
description: Generate a detailed report of the CURRENT cursor-model implementation — the plan–execute cycle with all formulas, each term's meaning and design intuition, current fitted numbers, and a corner walk-through. Re-derive everything from the code on every invocation; never reuse a previous report's text.
---

# generate_model_report

Produce the "how the current planning and execution cycle works" report for the gaze-led
anchor-drive MPCC cursor model. The report must reflect the code **as it is now**: read the
sources below fresh each time and derive formulas, terms and numbers from them. Do not
paraphrase an earlier report from memory; if something in the code changed, the report changes.

Optional argument: a participant id / persona tag (e.g. `P170114 S10b`). Default: the newest
`eval/eval-anchor-drive/results/P170114_anchor_persona_*.json` (by modification time) and,
if present, the newest A persona (`P105835_...`) for the between-participant contrast.

## 1. Read these sources (in this order)

1. `hcs_package/src/hcs_package/reference_path.py` — `generate_optimal_reference_path`
   (route / race-tracing formula: cut fraction, width and curvature factors, suppression).
2. `hcs_package/src/hcs_package/intermittent.py` — `DifficultyBudgetHorizon` (lead / anchor
   selection: density, budget D0, floors, curvature weighting on/off) and `ReplanScheduler`
   (arrival, deviation, exhaustion triggers, latency).
3. `hcs_package/src/hcs_package/cursor_simulator.py` — the anchor-drive planning block
   (search for `plan_deadline_s`, `plan_turn_time_s`, `plan_vmax`, `acc_max`, `motor_period_s`,
   `anchor_lead_floor`, `abort_on_breach_m`, `fix = {`): deadline formula, pace/schedule,
   horizon floor, motor replans, fixation lifetime, arrival test, breach abort, noise/plant.
4. `hcs_package/src/hcs_package/mpcc_model.py` — the objective in anchor mode: every cost term
   (jerk, contour / lag split, via-point drive and its normalisation, damping, walls,
   coast-safety hinge, peak-acceleration hinge, any gated extras), the kinematic progress,
   the re-linearisation loop and its safeguards, decision-variable bounds.
5. `eval/eval-anchor-drive/fit_anchor.py` — `ANCHOR_SPEC` (which parameters are fitted, bounds),
   the joint loss composition (`_eval_joint`, `_tunnel_part`, `_pointing_part`, stability gate),
   and `eval/eval-anchor-drive/refit_phase0.py` (Stage-1 loss variants).
6. The persona JSON(s) chosen above — for the concrete numbers (planner weights, budget,
   deadline constants, acc_max, v_max, reference-path params, latency, noise).
7. `eval/eval-anchor-drive/README.md` — the latest recorded results and open issues (use only
   what is marked as current; flag anything the code contradicts).

Use `grep`/`sed -n` to locate and read the relevant regions; quote formulas from the code, not
from memory. If a term you expect is absent or gated off in the config, say so explicitly.

## 2. Report structure (all sections required)

**0. Before the trial: the route** — formula for `p_ref(s)`, each factor, what is fitted, the
participant's fitted values and what they imply (e.g. cut depth at corner 40 vs 20 if the
README has measured numbers).

**1. A fixation opens a plan** — (a) *where*: the lead/anchor formula with current budget mode
and constants, floors, caps; (b) *how long*: the deadline formula with every term explained and
the participant's constants; (c) the pace / schedule the fixation stores.

**2. The plan: one MPCC solve** — decision variables, dynamics, kinematic progress; then a table
with one row per cost term: formula · what it does · why it is there (design intuition) ·
current fitted value. Include gated/off terms as "off" rows if they exist in code. State the
horizon length rule and any numerical safeguards.

**3. Execute, correct, replan** — motor period, plant noise, motor replans (what the via-point
is on a motor replan), fixation-end triggers (arrival / deviation / exhaustion), latency
distribution, breach abort. Give the currently measured rhythm (cycle length, arrival share)
from the latest validation log if available.

**4. How a corner is handled** — a numeric walk-through for the participant at a wide and a
narrow corner: lead, θ_lead, T, pace, what limits cornering speed (acceleration hinge,
coast-safety, walls, route cut), what happens on exit; and where width dependence and
between-participant differences come from (route × speed × acceleration bound, gaze constants).

**5. Fitting** — which parameters come from gaze data, which from Stage 1, which from the joint
CMA-ES; the loss terms; the noise-on stability gate if present.

**6. Known issues / residuals** — from the README and from any discrepancy noticed while reading
the code (e.g. per-window pacing sawtooth, collapsed lateral spring); mark each as open or fixed.

## 3. Style

- Formulas in code-style or compact math (`T = max(T0, lead/v_max) + τ·θ·(W_ref/W)^β`), with
  every symbol defined once.
- Plain language for intuition; no marketing; state uncertainties and gated-off features.
- Concrete numbers in brackets next to each term ([B: 14.8]); name the persona/tag and the
  git commit (`git log -1 --format=%h`) at the top so reports are comparable over time.
- Length: as long as needed to be complete (typically 900–1500 words).

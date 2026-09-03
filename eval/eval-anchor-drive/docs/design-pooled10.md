# Pooled new-cohort model (2026-09-02): gaze-module redesign + pooled fit

## Verified conclusions → design

All from the corrected 10-participant dataset (per-block drift correction,
merged fixations, round pruning; eval-gaze-cursor analyses of 09-01/09-02):

| Finding (verified) | Design consequence |
| --- | --- |
| Constrained/unconstrained dichotomy is universal: in-tunnel lead ≈ 0.07–0.11 of remaining path; after the tunnel exit gaze anchors on the goal (c2u trials, every participant) | budget density is EXACTLY 0 where W > 0.15 m (explicit free-space mask in `DifficultyBudgetHorizon` — new; required because with γ=0 the width toll no longer vanishes at W→∞) |
| NO width grading of the lead for this cohort (condition-level partial ρ(lead, W) ≈ 0/negative for 6/8; the γ-free pooled fit chooses γ=0.14 with a 0.002 log-loss gain over γ=0) | γ = 0: corridor toll is a constant 1/W_ref per metre → budget lead = D0·W_ref per corridor stretch |
| Speed floor: leads grow with cursor speed (partial ρ(v) +0.2…+0.5) and never collapse at low W | h ≥ v·T_min kept (T_min pooled-fit) |
| Curvature slows the movement, not the look-ahead (leads cross corners; dwell grows with local turning, b_κ ≈ +0.6/rad, and with narrowness, b_w ≈ +0.5) | curvature enters via the turn-time deadline t_plan = max(T0_eff, lead/v_max) + τ·θ_lead·(W_ref/W)^β_t, NOT the budget |
| Width slows the movement too, even on straight targets (dwell ρ(dur, W \| straight) = −0.29 in 13/13; with γ=0 the first fitted persona ran width-flat pace: CT-by-width 0.4–0.7 at W=10 up to 1.5–1.8 at W=50) | **width-scaled base deadline** (new, `plan_width_time_exp`): T0_eff = T0·(W_ref/W_local)^β_w, β_w = 0.5 — with a constant lead, the width–speed coupling lives in the time channel; free space keeps bare T0 |
| Dwell/rhythm: fixation dwell = crossing + latency; pooled latency median 0.205 s, CV ≈ 1.0 | replan latency lognormal (median/CV pooled), cap 1.5 s |
| Width shapes behaviour through speed and dwell (fixation spacing ∝ v·cycle: ρ(spacing, W) +0.53 in 11/11) | emergent — no extra mechanism |

Width battery caveat: the new-cohort task uses widths {10,12,16,20,25,30,50} mm
(6/7 levels ≤ 30), which cannot resolve a width-graded lead even for A/C-like
participants (restricting A/C to W ≤ 30 collapses their ρ from ~0.5 to 0.1–0.2).
γ=0 is the right *pooled* structure for this cohort/task; it is not a claim that
γ=0 holds universally (B is a genuine width-scaler).

## Pooled gaze constants (fit_gaze_pooled.py; pool = p04 p06 p07 p09 p10; 1944 events)

    gamma   = 0
    D0      = 1.571      (corridor lead const = D0·W_ref ≈ 40.8 mm)
    T_min   = 0.138 s    (log-loss 0.660; γ-free baseline 0.659 → γ buys nothing)
    T0      = 0.231 s,  τ = 0.189 s/rad,  β_t = 1.5   (pooled LAD, n=1892; β_w=0)
    width-scaled deadline refit: T0 = 0.197 (→0.20), τ = 0.174, β_w = 0.5
    (crossing-time MAD near-flat over β_w 0.25–0.5 [0.1961 vs 0.1977]; 0.5
    chosen because the dwell regression b_w ≈ +0.5 and the observed CT-by-width
    correction [speed ratio ≈2.7 over the 5× width range → exp ≈0.6] agree there)
    latency = 0.205 s median, CV 1.02 (n=1405), cap 1.5 s

Pool selection: split-half reliability of condition-median leads ≥ 0.43 under
the canonical corrections (p04 .60, p06 .56, p07 .64, p09 .54, p10 .44);
p05/p08 unusable (gaze availability), p01/p02/p03 marginal (.31–.43).

## Pooled route (fit_route_pooled.py)

w_cut 0.30, w_width_exp 0.50, w_center 0.05, global_clearance_ref 0.015
(cutmatch loss 0.00355 over 125 pooled trials). The CMA run completed one
generation within its time cap and found no better candidate than this init —
treat these as a validated starting point, not a converged optimum; a longer
route fit is cheap follow-up work.

## Motor fit (fit_anchor_pooled.py)

ONE weight set for the cohort (pooled loss = mean over participants), not
per-individual. Fitted: jerk, contour, constraint, goal, free_velocity,
plan_vmax. Fixed: acc_max 4 m/s², lag_anchor 2000, gaze constants above,
pooled Stage-1 route (fit_route_pooled.py, cutmatch). Loss per participant:
quick-subset tunnel (straight/sharp/corner × train widths {10,30,50}) +
pointing (2 rounds/radius), human-variability scaled per participant;
noise-on stability gate on the widest corner/sharp trials of two participants.

## Results (2026-09-02, noise-on full-set evaluation per participant)

Three candidates evaluated (eval_pooled.py; results/pooled10_eval_{base,fitted,wt}.json):

| candidate | pooled fit loss | tunnel train (range) | tunnel test | CT-by-width spread |
| --- | --- | --- | --- | --- |
| base (S14-B weights + pooled gaze) | — | 19.6–28.8 | 7.6–12.9 | 0.4 → 1.8 (inverted) |
| CMA fit, flat deadline | 24.89 | 15.2–23.5 | 7.5–12.9 | 0.4 → 1.8 (inverted) |
| **CMA fit + width-scaled deadline (final)** | **18.46** | **8.6–14.9** | **7.1–13.4** | **0.5–0.95 → 1.0–1.3** |

Final persona: results/pooled10_anchor_config_wt_s42.json — jerk 1.63e-6,
contour 7.05, constraint 12.6, goal 3.41, free_velocity 0.16, plan_vmax 0.61
(all in normal persona ranges; the flat-deadline fit had drifted to a degenerate
corner, jerk 3e-4 / contour 400, which the width-time mechanism made unnecessary).

Per participant (final): 100 % completion everywhere; overall CT ratio 0.79–1.09;
gaze cycle 0.45 s with triggers arrival/deviation/exhaustion = 0.70/0.15/0.15;
model lead constant 40.8 mm at every width (the γ=0 design), human condition
medians 22–85 mm; pointing MT 0.98–1.19 s. The width-scaled deadline halved the
tunnel train losses and flattened the CT-by-width profile from a 2.5–4× spread
to ≤1.8× without hurting held-out widths (train ≈ test — no overfit).

Eval-main-style metrics (same run): steering-law slope b = 12.0 s/100·ID (model,
one pooled persona) vs human 14.4 (p04) / 14.4 (p06) / 21.2 (p10) / 26.2 (p09) /
29.0 (p07) — right direction, sits at the fast end of the pool (0.83× the two
closest humans, 0.4–0.6× the slowest). Corner strategy (cut mean mm / apex dip):
model 2.3–4.2 mm cuts at W ≤ 25 vs human 1.2–4.7 (right scale), under-cut at
W=50 (3.9 vs 5.6–10.6); dips rise with width in both (model 0.13 → 0.8,
human ≈ 0.2 → 0.4–0.7) — the width contrast direction is reproduced, model
brakes slightly harder narrow and lighter wide than the humans.

**Aggregate eval-main results** — results/eval-main-pooled10-ALL/ (merged across
the 5 pool participants by merge_eval_main.py; per-participant runs in
results/eval-main-pooled10-{p04,p06,p07,p09,p10}/; 750 steering + 450 Fitts rows):

    Steering law (per-condition means):  human MT = 0.211·ID − 1.07 (R² 0.71)
                                          model MT = 0.108·ID + 0.94 (R² 0.71)
    Fitts law (aligned MT_kin):           human MT = 0.216·ID − 0.06 (R² 0.97)
                                          model MT = 0.211·ID − 0.19 (R² 0.92)

The pooled persona reproduces Fitts' law almost exactly (slope 0.211 vs 0.216
s/bit) and the steering law's linear form at the same R², but at ~51 % of the
human slope — the model does not slow down as much as humans per unit of
steering difficulty (same residual direction as every per-individual anchor fit
to date; best previous was B S12 at 0.172/0.184 with per-individual weights).

Gaze-lead evaluation: eval/eval-gaze-lead/model-gaze-lead-pooled10/ — signed
lead-vs-time PDFs per pool participant (pooled persona, human overlay) +
model_lead_events.csv (1435 planning events). Model sawtooth cycles at ~40 mm
lead at human-like rate. NOTE: the human dots in these legacy PDFs are the RAW
exported lead (uncorrected — biased low for drift-corrected participants);
the calibrated human-lead comparison is pooled10_eval_wt.json /
eval-gaze-lead/human-gaze-lead-10p/.

lag_anchor = 2000 sanity sweep (sweep_lag_anchor.py, surviving pool p04/p07/p10,
values 50–32000): loss has a broad optimum containing 2000 (19.2 at 2000; 19.7–19.8
across 200–500), degrading softward (+8 % at 50, corner cut inflating 4.5 → 7.5 mm
as progress decouples from position) and stiffward (+13 % at 8000, +29 % at 32000,
apex dip stiffening 0.82 → 0.76 — the single-stiff-tracking pathology returning).
2000 stays fixed; fitting it would chase ≤3 % inside the plateau while re-opening
the S8 degenerate-drift failure mode.

Open residuals: straights still slow (CT 1.15–1.87, worst p07) — the familiar
straight-pace residual, now the dominant error; sharp sinusoids too fast
(0.46–0.79); p07/p09 remain fast at W=10 (0.5–0.6). The pooled route is a
validated init, not a converged optimum. Pointing was only lightly constrained
(2 rounds/radius in the fit) — a fuller pointing pass is future work.

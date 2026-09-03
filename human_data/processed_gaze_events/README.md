# Processed gaze fixation events

`{letter}_fixation_events_clean.csv` — one row per steering fixation: bias-corrected
gaze, forward-constrained projection (`lead_corr`), quality flags, and the
**near-coincident merge pass** (`gaze_cleaning.merge_events`): consecutive kept events
with a short gap (≤ 0.10 s), corrected-gaze separation ≤ 10 mm, and projected arc
separation ≤ 15 mm are one *functional* fixation — measured on all 13 participants
these are micro-saccade re-settlings on the same spot (0–2% of transitions), i.e. one
planning event, not a new anchor. The survivor (first fragment)
keeps the onset read (`t_onset`, `s_c`, `lead_onset`, `lead_corr`) and the union
dwell (`duration_s`, fragments counted in `n_merged`); absorbed fragments stay in
the table with `keep=False` and `merged_into=<survivor fixation_id>`.

**Per-block drift correction** (`gaze_cleaning.estimate_block_drift`): the p-cohort
sessions carry block-varying spatial calibration offsets (headset slip, up to ~60 mm)
that cancelled the true forward lead. For participants in
`gaze_cleaning.DRIFT_PARTICIPANTS` (p01–p03, p05–p09 — selected by split-half
reliability of condition-median leads; A/B/C, p04, p10 keep the pointing-dwell bias),
each steering block's offset is estimated as median(gaze − cursor advanced by the
block's gaze-lead lag) and replaces the global bias where they disagree by > 15 mm
(`drift_corrected`, `drift_x/y` columns). Caveat: the estimate uses the future cursor
path as the gaze-target proxy, so each corrected round's *constant* lead component is
re-derived through its cross-correlation lag rather than raw positions.

**Use the MERGED events (`keep == True`) for all model fitting** — the gaze budget
(D0, γ, λ, β), turning-time constants (T0, τ, β_t), and intermittency/latency
statistics. Onset-lead fits on unmerged events double-count long dwells and bias
corner leads downward (later fragments re-read the same gaze point with a smaller
lead). Regenerate with `eval/eval-gaze-cursor/regenerate_clean_events.py`
(`--no-merge` exists only for A/B comparisons — do not fit on its output).
Rhythm-based constants (cycle rate, replan latency median/CV, frac_crossed)
derived before the merge pass must be re-derived on merged events before being
compared against or fed into the model.

Maps: `eval/eval-gaze-cursor/results/fixmaps_merged/` (merged, canonical);
`fixmaps_clean/` is the pre-merge version kept for comparison.

"""Gaze data cleaning: calibration-bias correction, forward-constrained lead, noise filters.

1. Per-participant calibration bias: during the final approach of pointing trials
   (t_end - 0.6 s .. t_end - 0.1 s) the eyes are on the target the cursor is entering, so
   median(gaze - cursor) estimates the residual mapping offset. Subtract it everywhere.
2. Forward-constrained projection: a fixation's arc position is the nearest point on the
   path with s >= s_cursor - 30 mm (foveal gaze serves the upcoming path; unconstrained
   nearest-point snaps to other folds of a sinusoid or to passed segments).
   lead = s_gaze - s_cursor.
3. Flags (kept in the output, not silently dropped):
   blink_corrupted  - blink overlaps the fixation or its incoming saccade (existing rule)
   off_path         - corrected gaze farther than max(W/2 + 15 mm, 25 mm) from the path
   regressive       - lead < -20 mm (glance clearly behind the cursor: monitoring/distraction)
   keep = none of the above.
4. Merge pass (merge_events): tracker-split fixations — consecutive kept events in the
   same round with a sub-saccade gap and corrected-gaze separation — are one perceptual
   fixation. The survivor keeps the FIRST fragment's onset read (t_onset, s_c, lead_onset,
   lead_corr) and the union dwell; absorbed fragments get keep=False + merged_into.
5. Per-block drift correction (estimate_block_drift): the 10-participant batch shows
   block-varying spatial calibration offsets up to ~55 mm (headset slip) that cancel the
   true forward lead — gaze temporally leads the cursor (cross-correlation +0.2..0.4 s)
   while the mapped positions sit at/behind it. Per steering block: find the lag that
   best aligns gaze x to the future cursor x (offset-invariant), then the block offset =
   median(gaze - cursor advanced by that lag). Blocks with a valid estimate use it in
   place of the global pointing bias; others fall back to the global bias. Caveat: this
   uses the cursor's future path as the gaze-target proxy, so each round's CONSTANT lead
   component is re-derived through the lag estimate rather than raw positions;
   fixation-to-fixation and within-round lead variation is preserved.
"""
import numpy as np
import pandas as pd

BIAS_CAP_M = 0.025         # sanity cap on the estimated calibration offset
BACK_WIN_M = 0.03          # forward-constrained projection: allow 30 mm behind the cursor
OFF_PATH_M = 0.015         # extra room beyond the corridor half-width
OFF_PATH_MIN_M = 0.025
REGRESSIVE_M = -0.020

# Merge pass (Salvucci-Goldberg-style): consecutive events closer than this in
# time and corrected-gaze space are fragments of one fixation. ~30% of fixation
# transitions carry no labelled saccade (gaze_data.fixation_events), so the
# tracker demonstrably splits on jitter below its saccade threshold.
MERGE_GAP_S = 0.10         # max end -> next-onset gap
MERGE_DIST_M = 0.010       # max corrected-gaze centroid separation (~0.5 deg)
MERGE_ARC_M = 0.015        # max |Δ| of the projected arc positions (never bridge a fold)


POINTING_Y_OFFSET = {"top": 0.06, "middle": 0.0, "bottom": -0.06}


def estimate_bias(samples: pd.DataFrame) -> np.ndarray:
    """Gaze offset at moments the eyes are verifiably on a known point: the cursor is
    INSIDE the pointing target and nearly stopped (dwell before the click). Compares gaze
    to the TARGET CENTER (from the condition), so multi-repetition pointing blocks and
    return movements cannot contaminate the estimate. Capped at +/-BIAS_CAP_M."""
    import json as _json
    pt = samples[samples["tunnel_type"].astype(str).str.contains("pointing")
                 & samples["gaze_task_x"].notna() & samples["cursor_x"].notna()]
    rows = []
    for _, blk in pt.groupby("block_id"):
        cj = blk["condition_json"].dropna()
        if not len(cj):
            continue
        cond = _json.loads(cj.iloc[0])
        y_off = POINTING_Y_OFFSET.get(cond.get("targetPosition", "middle"), 0.0)
        tx, ty = float(cond["distance"]), 0.13 + y_off
        r = float(cond.get("targetRadius", 0.01))
        d = np.hypot(blk["cursor_x"] - tx, blk["cursor_y"] - ty)
        sp = pd.to_numeric(blk.get("speed"), errors="coerce")
        dwell = blk[(d < r) & (sp.fillna(1.0) < 0.03)]
        if len(dwell) >= 3:
            rows.append([np.median(dwell["gaze_task_x"]) - tx, np.median(dwell["gaze_task_y"]) - ty])
    if not rows:
        return np.array([0.0, 0.0])
    b = np.median(np.array(rows), axis=0)
    return np.clip(b, -BIAS_CAP_M, BIAS_CAP_M)


DRIFT_MIN_SAMPLES = 400    # valid gaze+cursor pairs a block needs for a drift estimate
DRIFT_MAX_LAG_S = 0.8      # search window for the gaze-ahead lag
DRIFT_MIN_R = 0.8          # min cross-correlation at the best lag
# Within a drift-corrected participant, a block's drift estimate is applied only when
# it disagrees with the global pointing bias by more than this; smaller disagreements
# keep the cleaner pointing-dwell bias.
DRIFT_GATE_M = 0.015

# WHO gets block-drift correction is decided per participant by split-half reliability
# of their condition-median leads (no geometric threshold separates the two groups):
#   drift better:   p01 .24->.33  p02 .27->.31  p03 .18->.43  p06 -.15->.56
#                   p07 .38->.64  p09 .18->.54  (+ p05/p08: leads only become
#                   physiological with it; too sparse to score)
#   no-drift better: A .53->.26   C .46->.16   p04 .60->.37   p10 .44->.28
#   B ties (.60/.66) and stays uncorrected so the canonical A/B/C leads that the
#   model's gaze constants were derived from are untouched.
DRIFT_PARTICIPANTS = {"p01", "p02", "p03", "p05", "p06", "p07", "p08", "p09"}


def estimate_block_drift(samples: pd.DataFrame, hz: float = 200.0) -> dict:
    """Per-steering-block calibration offset, robust to the behavioural lead.

    For each block: find the lag (0..DRIFT_MAX_LAG_S, gaze ahead) that maximises the
    x cross-correlation between gaze and cursor — a measure invariant to constant
    spatial offsets — then estimate the offset as median(gaze - cursor advanced by
    that lag). Returns {block_id: (off_x, off_y, lag_s, r)} for blocks passing the
    sample-count and correlation gates."""
    st = samples[samples["cursor_x"].notna() & samples["gaze_task_x"].notna()
                 & ~samples["tunnel_type"].astype(str).str.contains("pointing")]
    out = {}
    max_lag = int(round(DRIFT_MAX_LAG_S * hz))
    for bid, b in st.groupby("block_id"):
        gx = b["gaze_task_x"].to_numpy(); gy = b["gaze_task_y"].to_numpy()
        cx = b["cursor_x"].to_numpy();    cy = b["cursor_y"].to_numpy()
        m = np.isfinite(gx) & np.isfinite(cx) & np.isfinite(gy) & np.isfinite(cy)
        if m.sum() < DRIFT_MIN_SAMPLES:
            continue
        gx, gy, cx, cy = gx[m], gy[m], cx[m], cy[m]
        gxa = gx - gx.mean(); cxa = cx - cx.mean()
        best, bl = -2.0, 0
        for lag in range(0, max_lag + 1, 2):
            a = gxa[:len(gxa) - lag if lag else None]
            c = cxa[lag:]
            if len(a) < DRIFT_MIN_SAMPLES // 2:
                break
            r = float(np.corrcoef(a, c)[0, 1])
            if r > best:
                best, bl = r, lag
        if best < DRIFT_MIN_R:
            continue
        sl = slice(None, len(gx) - bl if bl else None)
        off_x = float(np.median(gx[sl] - cx[bl:]))
        off_y = float(np.median(gy[sl] - cy[bl:]))
        out[bid] = (off_x, off_y, bl / hz, best)
    return out


def constrained_project(geom, gx, gy, s_c):
    """Nearest path point with s >= s_c - BACK_WIN_M; returns (s_gaze, distance)."""
    m = geom.s >= (s_c - BACK_WIN_M)
    if not m.any():
        m = np.ones_like(geom.s, dtype=bool)
    pts = geom.path[m]
    d2 = (pts[:, 0] - gx) ** 2 + (pts[:, 1] - gy) ** 2
    i = int(np.argmin(d2))
    return float(geom.s[m][i]), float(np.sqrt(d2[i]))


def clean_events(ev: pd.DataFrame, geoms: dict, bias: np.ndarray,
                 drift: dict | None = None) -> pd.DataFrame:
    out = ev.copy()
    # Offset correction: a block's drift estimate replaces the global bias only when
    # the two disagree by more than DRIFT_GATE_M (the drift estimate subsumes the
    # bias — both are measured on the same raw gaze); small disagreements keep the
    # cleaner pointing-dwell bias.
    if drift:
        use = {b: v[:2] for b, v in drift.items()
               if np.hypot(v[0] - bias[0], v[1] - bias[1]) > DRIFT_GATE_M}
        off = np.array([use.get(b, (bias[0], bias[1])) for b in out["block_id"]])
        out["gaze_x_corr"] = out["gaze_task_x"] - off[:, 0]
        out["gaze_y_corr"] = out["gaze_task_y"] - off[:, 1]
        out["drift_corrected"] = [b in use for b in out["block_id"]]
        out["drift_x"] = off[:, 0]; out["drift_y"] = off[:, 1]
    else:
        out["gaze_x_corr"] = out["gaze_task_x"] - bias[0]
        out["gaze_y_corr"] = out["gaze_task_y"] - bias[1]
        out["drift_corrected"] = False
    lead_c = np.full(len(out), np.nan); dist_c = np.full(len(out), np.nan)
    for i, (_, r) in enumerate(out.iterrows()):
        g = geoms.get((r["participant"], r["trial_id"]))
        if g is None or not np.isfinite(r["gaze_x_corr"]) or not np.isfinite(r.get("s_c", np.nan)):
            continue
        s_g, d = constrained_project(g, r["gaze_x_corr"], r["gaze_y_corr"], r["s_c"])
        lead_c[i] = s_g - r["s_c"]; dist_c[i] = d
    out["lead_corr"] = lead_c
    out["gaze_path_dist"] = dist_c
    w = pd.to_numeric(out["width"], errors="coerce").fillna(0.05)
    out["off_path"] = out["gaze_path_dist"] > np.maximum(w / 2 + OFF_PATH_M, OFF_PATH_MIN_M)
    out["regressive"] = out["lead_corr"] < REGRESSIVE_M
    out["keep"] = (~out["off_path"]) & (~out["regressive"]) & (~out["blink_corrupted"]) & out["lead_corr"].notna()
    return out


def merge_events(ev: pd.DataFrame, gap_s: float = MERGE_GAP_S, dist_m: float = MERGE_DIST_M,
                 arc_m: float = MERGE_ARC_M, require_no_saccade: bool = False) -> pd.DataFrame:
    """Merge near-coincident fixations in a cleaned event table (one row per fixation).

    Consecutive KEPT events of the same round are one FUNCTIONAL fixation (one planning
    event: the anchor did not move) when the gap between them is short (<= gap_s), the
    corrected gaze moved less than dist_m, and their projected arc positions
    (s_c + lead_corr) differ by less than arc_m (the fold guard). Measured on the 13
    participants, such pairs are 0-2.3% of transitions and virtually ALL carry a
    labelled saccade between them — micro-saccades that re-settle on the same spot, not
    tracker dropouts — so require_no_saccade defaults to False (True would veto every
    merge); the tight spatial gates are the criterion. A dropped fragment (blink /
    off-path / regressive) breaks the chain: no bridging across corrupted data.

    The survivor is the FIRST fragment: its onset fields (t_onset, s_c, lead_onset,
    lead_corr, speed_onset) are the planning-event read and stay untouched; its
    duration_s becomes last-fragment end minus first onset (gaps included — dwell as a
    planning interval), and n_merged counts the fragments. Absorbed fragments get
    keep=False and merged_into=<survivor fixation_id> so nothing is silently deleted.
    """
    out = ev.copy()
    out["n_merged"] = 1
    out["merged_into"] = np.nan
    if not len(out):
        return out
    has_sacc = "saccade_followed" in out.columns
    chains = []
    for _, grp in out.groupby(["participant", "trial_id", "block_id"], sort=False):
        grp = grp.sort_values("t_onset")
        chain = []
        for idx in grp.index:
            r = out.loc[idx]
            if not bool(r["keep"]):
                if len(chain) > 1:
                    chains.append(chain)
                chain = []
                continue
            join = False
            if chain:
                p = out.loc[chain[-1]]
                gap = float(r["t_onset"]) - (float(p["t_onset"]) + float(p["duration_s"]))
                dist = float(np.hypot(r["gaze_x_corr"] - p["gaze_x_corr"],
                                      r["gaze_y_corr"] - p["gaze_y_corr"]))
                arc = abs((float(r["s_c"]) + float(r["lead_corr"]))
                          - (float(p["s_c"]) + float(p["lead_corr"])))
                sacc_between = bool(p["saccade_followed"]) if (require_no_saccade and has_sacc) else False
                join = (gap <= gap_s) and (dist <= dist_m) and (arc <= arc_m) and not sacc_between
            if join:
                chain.append(idx)
            else:
                if len(chain) > 1:
                    chains.append(chain)
                chain = [idx]
        if len(chain) > 1:
            chains.append(chain)
    for chain in chains:
        first, last = chain[0], chain[-1]
        t_end_last = float(out.loc[last, "t_onset"]) + float(out.loc[last, "duration_s"])
        out.loc[first, "duration_s"] = t_end_last - float(out.loc[first, "t_onset"])
        out.loc[first, "n_merged"] = len(chain)
        if has_sacc:
            # the merged fixation ends where its last fragment ends
            out.loc[first, "saccade_followed"] = bool(out.loc[last, "saccade_followed"])
        for idx in chain[1:]:
            out.loc[idx, "keep"] = False
            out.loc[idx, "merged_into"] = out.loc[first, "fixation_id"]
    return out

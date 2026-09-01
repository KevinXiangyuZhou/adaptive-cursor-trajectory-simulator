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
"""
import numpy as np
import pandas as pd

BIAS_CAP_M = 0.025         # sanity cap on the estimated calibration offset
BACK_WIN_M = 0.03          # forward-constrained projection: allow 30 mm behind the cursor
OFF_PATH_M = 0.015         # extra room beyond the corridor half-width
OFF_PATH_MIN_M = 0.025
REGRESSIVE_M = -0.020


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


def constrained_project(geom, gx, gy, s_c):
    """Nearest path point with s >= s_c - BACK_WIN_M; returns (s_gaze, distance)."""
    m = geom.s >= (s_c - BACK_WIN_M)
    if not m.any():
        m = np.ones_like(geom.s, dtype=bool)
    pts = geom.path[m]
    d2 = (pts[:, 0] - gx) ** 2 + (pts[:, 1] - gy) ** 2
    i = int(np.argmin(d2))
    return float(geom.s[m][i]), float(np.sqrt(d2[i]))


def clean_events(ev: pd.DataFrame, geoms: dict, bias: np.ndarray) -> pd.DataFrame:
    out = ev.copy()
    out["gaze_x_corr"] = out["gaze_task_x"] - bias[0]
    out["gaze_y_corr"] = out["gaze_task_y"] - bias[1]
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

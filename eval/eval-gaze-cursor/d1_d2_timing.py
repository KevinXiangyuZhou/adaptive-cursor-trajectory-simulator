"""D1/D2: the two discriminating experiments for gaze-led, anticipatory slowing.

D1 (leader vs shared trigger): for each corner passage, the time of the horizon minimum
(cleaned per-fixation lead, interpolated) vs the time of the cursor-speed minimum.
Positive lag (speed min after lead min) by ~0.1-0.25 s supports gaze-leads.
Also the cross-correlation of lead(t) and speed(t) over corner trials.

D2 (emergent vs reactive): where does corner deceleration BEGIN relative to the vertex?
Onset = last time speed falls below 80% of its pre-corner plateau before the vertex.
Anticipatory (onset well before the vertex) supports horizon/MPC-emergent slowing.

Usage: python d1_d2_timing.py [--letters A B C p04 p10]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import gaze_data as gd, lookahead_difficulty as ld

PROC = HERE.parents[1] / "human_data" / "processed_gaze_events"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--letters", nargs="*", default=["A", "B", "C", "p04", "p10"])
    a = ap.parse_args()
    out = {}
    for L in a.letters:
        s = gd.load_samples(L)
        cl = pd.read_csv(PROC / f"{L}_fixation_events_clean.csv"); cl = cl[cl["keep"] == True]  # noqa: E712
        ev = gd.fixation_events(s); ev = ev[~ev["tunnel_type"].astype(str).str.contains("pointing")]
        _, geoms = ld.attach_geometry(s, ev)
        lags, onsets_t, onsets_d, xcorr_lags = [], [], [], []
        corner_trials = cl[cl["tunnel_type"] == "corner"]["trial_id"].unique()
        for tid in corner_trials:
            g = geoms.get((L, tid))
            if g is None or len(g.turn_s) == 0: continue
            for bid, fx in cl[(cl["trial_id"] == tid)].groupby("block_id"):
                blk = s[(s["trial_id"] == tid) & (s["block_id"] == bid) & s["cursor_x"].notna()].sort_values("t")
                if len(blk) < 10: continue
                s_cur, _ = g.project(blk["cursor_x"].to_numpy(), blk["cursor_y"].to_numpy())
                s_cur = np.maximum.accumulate(s_cur); t = blk["t"].to_numpy(); v = pd.to_numeric(blk["speed"], errors="coerce").to_numpy()
                fx = fx.sort_values("t_onset")
                if len(fx) < 3: continue
                lead_t = np.interp(t, fx["t_onset"], fx["lead_corr"])
                # cross-correlation (detrended) over the whole round
                lv, vv = lead_t - np.nanmean(lead_t), v - np.nanmean(v)
                ok = np.isfinite(lv) & np.isfinite(vv)
                if ok.sum() > 20:
                    dt_med = np.median(np.diff(t))
                    max_shift = int(0.6 / dt_med)
                    cs = [np.corrcoef(lv[ok][sh:], vv[ok][:len(vv[ok]) - sh])[0, 1] if sh > 0 else
                          (np.corrcoef(lv[ok][:len(lv[ok]) + sh], vv[ok][-sh:])[0, 1] if sh < 0 else np.corrcoef(lv[ok], vv[ok])[0, 1])
                          for sh in range(-max_shift, max_shift + 1)]
                    best = int(np.nanargmax(cs)) - max_shift
                    xcorr_lags.append(best * dt_med)   # >0: lead leads speed
                for s_vertex in g.turn_s:
                    # window: from 120 mm before to 60 mm after the vertex
                    m = (s_cur > s_vertex - 0.12) & (s_cur < s_vertex + 0.06)
                    if m.sum() < 8: continue
                    tw, vw, lw, sw = t[m], v[m], lead_t[m], s_cur[m]
                    if not np.isfinite(vw).any() or not np.isfinite(lw).any(): continue
                    t_vmin = tw[np.nanargmin(vw)]; t_lmin = tw[np.nanargmin(lw)]
                    lags.append(t_vmin - t_lmin)
                    pre = vw[sw < s_vertex - 0.06]
                    if len(pre) >= 3 and np.isfinite(pre).any():
                        plateau = np.nanmedian(pre)
                        below = np.where((vw < 0.8 * plateau) & (sw < s_vertex))[0]
                        if len(below):
                            onsets_t.append(tw[below[0]] - tw[np.argmin(np.abs(sw - s_vertex))])
                            onsets_d.append(sw[below[0]] - s_vertex)
        out[L] = {"n_corner_passages": len(lags),
                  "D1_lag_speedmin_minus_leadmin_s": {"median": float(np.median(lags)) if lags else None,
                                                       "iqr": [float(np.percentile(lags, 25)), float(np.percentile(lags, 75))] if lags else None},
                  "D1_xcorr_peak_lag_s": {"median": float(np.median(xcorr_lags)) if xcorr_lags else None},
                  "D2_decel_onset_before_vertex_s": {"median": float(-np.median(onsets_t)) if onsets_t else None},
                  "D2_decel_onset_before_vertex_mm": {"median": float(-np.median(onsets_d) * 1000) if onsets_d else None}}
        print(L, json.dumps(out[L]), flush=True)
    json.dump(out, open(HERE / "results" / "d1_d2_timing.json", "w"), indent=2)
    print("saved results/d1_d2_timing.json"); print("DONE")


if __name__ == "__main__":
    main()

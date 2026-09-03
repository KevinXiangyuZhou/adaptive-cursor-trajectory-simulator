"""Does fixation duration depend on the LOCAL geometry of the fixated point —
curvature, width, or both?

Data: the merged canonical fixation dataset (human_data/processed_gaze_events,
keep=True). For each fixation the fixated point is s_f = s_c + lead_corr; its local
geometry is W_f = W(s_f) (usable width) and dphi = PHI(s_f+w) - PHI(s_f-w), the
turning angle inside a +/-7.5 mm window (rad; ~pi/2 at a 90-degree corner, ~0 on a
straight). Free-space targets (W_f > 0.15 m) are excluded.

Per participant:
  * Spearman rho(duration, dphi) and rho(duration, W_f), marginal.
  * rho(duration, dphi) WITHIN trial-width bins (n-weighted mean) — curvature effect
    with width held fixed; rho(duration, W_f) on near-straight targets only
    (dphi < 0.1 rad) — width effect with curvature held fixed.
  * corner ratio: median duration at turning targets (dphi > 0.5) / straight targets
    (dphi < 0.1).
  * OLS log(dur) ~ b_w * log(W_ref/W_f) + b_k * dphi, 1000-bootstrap 95% CIs.

Usage: python dwell_local_geometry.py [--letters A B C p01 ...]
Saves results/dwell_local_geometry.json.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import gaze_data as gd, lookahead_difficulty as ld

PROC = HERE.parents[1] / "human_data" / "processed_gaze_events"
W_REF = 0.026
WIN_M = 0.0075           # half-window for the local turning angle
DPHI_STRAIGHT = 0.10     # rad: below this the target counts as straight
DPHI_TURN = 0.50         # rad: above this the target sits in a turn
FREE_W = 0.15            # exclude free-space fixation targets


def build(L):
    s = gd.load_samples(L)
    cl = pd.read_csv(PROC / f"{L}_fixation_events_clean.csv")
    cl = cl[cl["keep"] == True]  # noqa: E712  (merged survivors only)
    ev = gd.fixation_events(s)
    ev = ev[~ev["tunnel_type"].astype(str).str.contains("pointing")]
    _, geoms = ld.attach_geometry(s, ev)
    rows = []
    for _, r in cl.iterrows():
        g = geoms.get((L, r["trial_id"]))
        if g is None or not np.isfinite(r["lead_corr"]):
            continue
        s_f = float(np.clip(r["s_c"] + r["lead_corr"], 0.0, g.s_end))
        W_f = float(np.interp(s_f, g.s, g.W))
        if not np.isfinite(W_f) or W_f > FREE_W:
            continue
        dphi = float(np.interp(min(s_f + WIN_M, g.s_end), g.s, g.PHI)
                     - np.interp(max(s_f - WIN_M, 0.0), g.s, g.PHI))
        if r["duration_s"] <= 0:
            continue
        rows.append((r["trial_id"], r["tunnel_type"], float(r["width"]),
                     float(r["duration_s"]), W_f, dphi))
    return pd.DataFrame(rows, columns=["trial_id", "tunnel_type", "width", "dur", "W_f", "dphi"])


def within_width_rho(d):
    """n-weighted mean Spearman(dur, dphi) inside trial-width bins (>=15 events)."""
    parts = []
    for wb, g in d.groupby(np.round(d["width"] * 1000)):
        if len(g) >= 15 and g["dphi"].std() > 1e-6:
            parts.append((len(g), spearmanr(g["dur"], g["dphi"]).statistic))
    if not parts:
        return np.nan, 0
    n = sum(p[0] for p in parts)
    return float(sum(p[0] * p[1] for p in parts) / n), len(parts)


def ols_boot(d, n_boot=1000, seed=0):
    """log(dur) ~ b0 + b_w log(W_ref/W_f) + b_k dphi; bootstrap 95% CIs."""
    X = np.c_[np.ones(len(d)), np.log(W_REF / d["W_f"]), d["dphi"]]
    y = np.log(d["dur"].to_numpy())
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    rng = np.random.default_rng(seed)
    bs = np.empty((n_boot, 3))
    for i in range(n_boot):
        idx = rng.integers(0, len(d), len(d))
        bs[i] = np.linalg.lstsq(X[idx], y[idx], rcond=None)[0]
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    return b, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--letters", nargs="*", default=["A", "B", "C"] + [f"p{i:02d}" for i in range(1, 11)])
    a = ap.parse_args()
    out = {}
    print(f"{'P':<5} {'n':>4} | rho(dur,dphi) within-W | rho(dur,W) straight | corner/straight dur "
          f"| b_w [95% CI]          | b_k [95% CI]")
    for L in a.letters:
        d = build(L)
        if len(d) < 40:
            print(f"{L:<5} {len(d):>4} | too few events"); continue
        rho_k_marg = spearmanr(d["dur"], d["dphi"]).statistic
        rho_k_w, nbins = within_width_rho(d)
        straight = d[d["dphi"] < DPHI_STRAIGHT]
        rho_w = (spearmanr(straight["dur"], straight["W_f"]).statistic
                 if len(straight) >= 30 and straight["W_f"].std() > 1e-6 else np.nan)
        turn = d[d["dphi"] > DPHI_TURN]
        ratio = (float(np.median(turn["dur"]) / np.median(straight["dur"]))
                 if len(turn) >= 10 and len(straight) >= 10 else np.nan)
        b, lo, hi = ols_boot(d)
        sig_w = "" if lo[1] <= 0 <= hi[1] else "*"
        sig_k = "" if lo[2] <= 0 <= hi[2] else "*"
        print(f"{L:<5} {len(d):>4} | {rho_k_marg:+.2f} marg / {rho_k_w:+.2f} ({nbins} bins) "
              f"| {rho_w:+.2f}" + (" " * 12) +
              f"| {ratio:4.2f}x            | {b[1]:+.3f} [{lo[1]:+.3f},{hi[1]:+.3f}]{sig_w:<2}"
              f"| {b[2]:+.3f} [{lo[2]:+.3f},{hi[2]:+.3f}]{sig_k}", flush=True)
        out[L] = {"n": int(len(d)), "rho_dphi_marginal": float(rho_k_marg),
                  "rho_dphi_within_width": float(rho_k_w), "n_width_bins": nbins,
                  "rho_W_straight_only": float(rho_w) if np.isfinite(rho_w) else None,
                  "corner_straight_dur_ratio": float(ratio) if np.isfinite(ratio) else None,
                  "ols_log_dur": {"b_width": float(b[1]), "b_width_ci": [float(lo[1]), float(hi[1])],
                                   "b_dphi": float(b[2]), "b_dphi_ci": [float(lo[2]), float(hi[2])]}}
    ks = [v for v in out.values()]
    if ks:
        med = lambda key, sub=None: float(np.nanmedian([(v[key][sub] if sub else v[key]) or np.nan for v in ks]))
        n_sig_k = sum(1 for v in ks if not (v["ols_log_dur"]["b_dphi_ci"][0] <= 0 <= v["ols_log_dur"]["b_dphi_ci"][1]))
        n_sig_w = sum(1 for v in ks if not (v["ols_log_dur"]["b_width_ci"][0] <= 0 <= v["ols_log_dur"]["b_width_ci"][1]))
        print(f"\npooled medians: rho(dur,dphi|W) {med('rho_dphi_within_width'):+.2f}, "
              f"rho(dur,W|straight) {med('rho_W_straight_only'):+.2f}, "
              f"corner/straight ratio {med('corner_straight_dur_ratio'):.2f}x, "
              f"b_w {med('ols_log_dur','b_width'):+.3f} (CI excl. 0: {n_sig_w}/{len(ks)}), "
              f"b_k {med('ols_log_dur','b_dphi'):+.3f} (CI excl. 0: {n_sig_k}/{len(ks)})")
    json.dump(out, open(HERE / "results" / "dwell_local_geometry.json", "w"), indent=2)
    print("saved results/dwell_local_geometry.json"); print("DONE")


if __name__ == "__main__":
    main()

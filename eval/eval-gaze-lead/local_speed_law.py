"""Fit the local cursor speed law on per-sample data (steering tasks, 10p batch).

For every cursor sample: v (exported speed), local tunnel width W(s_c), local
centerline curvature |kappa(s_c)| (gradient of the unwrapped tangent angle),
and ANTICIPATORY features over a forward window H_M: min width and max
curvature ahead. Candidate forms compared on log v (pooled + per participant
+ leave-one-trial-out grouped CV):

  M0  log v ~ log W_loc                        (steering law, local)
  M1  + log(kappa_loc + K_EPS)                 (two-thirds law says coef -1/3)
  M2  + interaction                            (separability test)
  M3  log W_min_ahead + log(kappa_max_ahead)   (anticipatory variant)
  M4  M1 + M3 features                         (local + lookahead)

Writes per-sample features to human-gaze-lead-10p/data/local_speed_samples.csv
(subsampled) and prints the comparison.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import maximum_filter1d, minimum_filter1d

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from human_gaze_lead import dense_grid, project_dense, PROC, SUBSAMPLE, gd, ld

LETTERS = ["p01", "p02", "p03", "p04", "p07", "p10"]
TYPES = {"gentle_sinusoidal": "gentle", "sharp_sinusoidal": "sharp",
         "None": "normal", "straight": "straight", "corner": "corner"}
BASE = SCRIPT_DIR / "human-gaze-lead-10p"
H_M = 0.05        # forward anticipation window (m)
K_EPS = 1.0       # rad/m, additive floor inside log for curvature
V_MIN = 0.002     # m/s, below = stationary, excluded from log fit
TRIM_S = 0.3      # s trimmed at round start/end (launch/stop transients)


def kappa_profile(geom):
    ang = np.unwrap(np.arctan2(*np.gradient(geom.path, axis=0).T[::-1]))
    ds = np.gradient(geom.s)
    return np.abs(np.gradient(ang) / (ds + 1e-12))


def main():
    rows = []
    step = 0.001
    win = int(H_M / step) + 1
    for L in LETTERS:
        s = gd.load_samples(L)
        cl = pd.read_csv(PROC / f"{L}_fixation_events_clean.csv")
        cl = cl[cl["keep"] == True]  # noqa: E712
        ev = gd.fixation_events(s)
        ev = ev[~ev["tunnel_type"].astype(str).str.contains("pointing")]
        _, geoms = ld.attach_geometry(s, ev)
        trials = cl.groupby("trial_id").first().reset_index()
        trials = trials[trials["tunnel_type"].astype(str).isin(TYPES)]
        for _, tr in trials.iterrows():
            tid = tr["trial_id"]; g = geoms.get((L, tid))
            if g is None:
                continue
            s_d, P_d = dense_grid(g, step)
            W_d = np.interp(s_d, g.s, g.W)
            K_d = np.interp(s_d, g.s, kappa_profile(g))
            # forward-window features on the dense grid: window [i, i+win-1]
            Wmin_d = minimum_filter1d(W_d, size=win, origin=-(win // 2), mode="nearest")
            Kmax_d = maximum_filter1d(K_d, size=win, origin=-(win // 2), mode="nearest")
            st = s[(s["trial_id"] == tid) & s["cursor_x"].notna() & s["speed"].notna()]
            for bid in sorted(st["block_id"].dropna().unique()):
                b = st[st["block_id"] == bid].iloc[::SUBSAMPLE]
                if len(b) < 10:
                    continue
                t = b["t"].to_numpy(); t = t - t[0]
                keep = (t > TRIM_S) & (t < t[-1] - TRIM_S)
                if keep.sum() < 10:
                    continue
                b = b[keep]
                s_c = project_dense(s_d, P_d, b["cursor_x"].to_numpy(),
                                    b["cursor_y"].to_numpy())
                idx = np.clip((s_c / step).astype(int), 0, len(s_d) - 1)
                rows.append(pd.DataFrame({
                    "participant": L, "type_label": TYPES[str(tr["tunnel_type"])],
                    "trial_id": tid, "block_id": bid,
                    "v": b["speed"].to_numpy(), "W_loc": W_d[idx], "k_loc": K_d[idx],
                    "W_ahead": Wmin_d[idx], "k_ahead": Kmax_d[idx]}))
        print(f"{L}: {sum(len(r) for r in rows[-60:])} samples so far", flush=True)

    d = pd.concat(rows, ignore_index=True)
    d = d[d["v"] >= V_MIN]
    d.iloc[::5].to_csv(BASE / "data" / "local_speed_samples.csv", index=False,
                       float_format="%.5f")
    print(f"\n{len(d)} samples in fit")

    y = np.log(d["v"]).to_numpy()
    lw = np.log(d["W_loc"]).to_numpy()
    lk = np.log(d["k_loc"] + K_EPS).to_numpy()
    lwa = np.log(d["W_ahead"]).to_numpy()
    lka = np.log(d["k_ahead"] + K_EPS).to_numpy()
    groups = d["participant"] + "/" + d["trial_id"].astype(str)

    def fit(X):
        X1 = np.column_stack([np.ones(len(y))] + X)
        b, *_ = np.linalg.lstsq(X1, y, rcond=None)
        r2 = 1 - ((y - X1 @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        return b, r2

    def cv_r2(X):
        X = np.column_stack([np.ones(len(y))] + X)
        ss_res = ss_tot = 0.0
        for gname in groups.unique():
            m = (groups == gname).to_numpy()
            b, *_ = np.linalg.lstsq(X[~m], y[~m], rcond=None)
            ss_res += ((y[m] - X[m] @ b) ** 2).sum()
            ss_tot += ((y[m] - y[~m].mean()) ** 2).sum()
        return 1 - ss_res / ss_tot

    models = {
        "M0 local W":            [lw],
        "M1 local W + kappa":    [lw, lk],
        "M2 + interaction":      [lw, lk, lw * lk],
        "M3 ahead W + kappa":    [lwa, lka],
        "M4 local + ahead":      [lw, lk, lwa, lka],
    }
    print(f"\nmodel comparison (log v; grouped CV = leave-one-trial-out):")
    for name, X in models.items():
        b, r2 = fit(X)
        print(f"  {name:22s} R2={r2:.3f}  CV={cv_r2(X):.3f}  coefs={np.round(b[1:], 3)}")

    print("\nper-participant M4 coefficients (logW, logk, logW_ahead, logk_ahead):")
    for L, gd_ in d.groupby("participant"):
        yy = np.log(gd_["v"]).to_numpy()
        X1 = np.column_stack([np.ones(len(yy)), np.log(gd_["W_loc"]),
                              np.log(gd_["k_loc"] + K_EPS), np.log(gd_["W_ahead"]),
                              np.log(gd_["k_ahead"] + K_EPS)])
        b, *_ = np.linalg.lstsq(X1, yy, rcond=None)
        r2 = 1 - ((yy - X1 @ b) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
        print(f"  {L}: {np.round(b[1:], 3)}  R2={r2:.3f}")

    # two-thirds-law check on curved samples only (kappa > 5 rad/m)
    cm = d["k_loc"] > 5
    b23, _, r23, p23, _ = stats.linregress(np.log(d.loc[cm, "k_loc"]),
                                           np.log(d.loc[cm, "v"]))
    print(f"\ntwo-thirds-law check (curved samples, kappa>5, n={cm.sum()}): "
          f"v ~ kappa^{b23:.3f} (literature -1/3; r={r23:.3f})")


if __name__ == "__main__":
    main()

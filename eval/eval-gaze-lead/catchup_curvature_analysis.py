"""Catch-up duration vs accumulated curvature over the caught-up path segment.

For every catch-up interval (fixation between two forward saccades, same
detector as saccade_width_analysis.py) we take the cursor arc positions at the
interval's start (s0, end of the reset saccade) and end (s1, start of the next
saccade) and integrate |curvature| of the reference centerline over [s0, s1]
using Geom.PHI (cumulative unsigned turn angle, radians). Questions: does more
curvature to consume mean a longer catch-up, and does it slow the cursor
(inverse speed = dur/dist) beyond what width already explains?

Reported: raw and per-participant Spearman rho(dPhi, dur); within-width-stratum
rho; partial rho of dur vs dPhi controlling log W and catch-up distance
(residual-on-residual); OLS of log dur on log W + dist + dPhi with delta-R2;
inverse-speed vs curvature-density rho.

Outputs (into human-gaze-lead-10p/):
  data/catchup_curvature.csv    one row per catch-up interval
  catchup_vs_curvature.png      binned medians by width stratum
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from human_gaze_lead import dense_grid, project_dense, PROC, SUBSAMPLE, gd, gc, ld
from saccade_width_analysis import V_ON, MAX_DT, A_MIN, A_MAX

LETTERS = ["p01", "p02", "p03", "p04", "p07", "p10"]
TYPES = {"gentle_sinusoidal": "gentle", "sharp_sinusoidal": "sharp",
         "None": "normal", "straight": "straight", "corner": "corner"}
BASE = SCRIPT_DIR / "human-gaze-lead-10p"
DUR_MIN, DUR_MAX = 0.05, 3.0
DIST_MIN = 0.003  # m, minimum caught-up path distance


def saccade_spans(t, lead):
    """Index spans [i, j+1] of forward saccades (as in saccade_width_analysis)."""
    dl = np.diff(lead); dt = np.diff(t)
    on = (dl / np.where(dt > 0, dt, np.inf) > V_ON) & (dt <= MAX_DT)
    spans = []
    i = 0
    while i < len(on):
        if on[i]:
            j = i
            while j + 1 < len(on) and on[j + 1]:
                j += 1
            if A_MIN <= lead[j + 1] - lead[i] <= A_MAX:
                spans.append((i, j + 1))
            i = j + 1
        i += 1
    return spans


def main():
    rows = []
    for L in LETTERS:
        s = gd.load_samples(L)
        bias = gc.estimate_bias(s)
        drift = gc.estimate_block_drift(s) if L in gc.DRIFT_PARTICIPANTS else {}
        off_by_block = {b: (v[0], v[1]) for b, v in drift.items()
                        if np.hypot(v[0] - bias[0], v[1] - bias[1]) > gc.DRIFT_GATE_M}
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
            s_d, P_d = dense_grid(g)
            st = s[(s["trial_id"] == tid) & s["cursor_x"].notna() & s["gaze_task_x"].notna()]
            for bid in sorted(st["block_id"].dropna().unique()):
                b = st[st["block_id"] == bid].iloc[::SUBSAMPLE]
                if len(b) < 10:
                    continue
                ox, oy = off_by_block.get(bid, (bias[0], bias[1]))
                s_c = project_dense(s_d, P_d, b["cursor_x"].to_numpy(), b["cursor_y"].to_numpy())
                s_g = project_dense(s_d, P_d, (b["gaze_task_x"] - ox).to_numpy(),
                                    (b["gaze_task_y"] - oy).to_numpy())
                t = b["t"].to_numpy(); t = t - t[0]
                lead = s_g - s_c
                spans = saccade_spans(t, lead)
                for (i0, j0), (i1, _) in zip(spans, spans[1:]):
                    dur = t[i1] - t[j0]
                    if not (DUR_MIN <= dur <= DUR_MAX):
                        continue
                    # drop intervals containing recording gaps
                    if np.max(np.diff(t[j0:i1 + 1])) > MAX_DT:
                        continue
                    s0, s1 = s_c[j0], s_c[i1]
                    dist = s1 - s0
                    if dist < DIST_MIN:
                        continue
                    dphi = np.interp(s1, g.s, g.PHI) - np.interp(s0, g.s, g.PHI)
                    rows.append(dict(
                        participant=L, type_label=TYPES[str(tr["tunnel_type"])],
                        trial_id=tid, width_mm=round(tr["width"] * 1000, 1),
                        dur_s=dur, dist_m=dist, dphi_rad=max(dphi, 0.0),
                        lead_post=lead[j0], inv_speed=dur / dist))
        print(f"{L}: {sum(r['participant'] == L for r in rows)} catch-up intervals", flush=True)

    d = pd.DataFrame(rows)
    d["kappa_density"] = d["dphi_rad"] / d["dist_m"]  # rad/m over the segment
    d.to_csv(BASE / "data" / "catchup_curvature.csv", index=False, float_format="%.6f")
    print(f"\n{len(d)} intervals total; dphi>0 on {(d['dphi_rad'] > 1e-6).mean():.0%} of them")

    # --- raw association
    print("\nSpearman rho(dphi, dur):")
    for L, g in d.groupby("participant"):
        rho, p = stats.spearmanr(g["dphi_rad"], g["dur_s"])
        print(f"  {L}: rho={rho:.3f} (p={p:.1e}) n={len(g)}")
    rho, p = stats.spearmanr(d["dphi_rad"], d["dur_s"])
    print(f"  pooled: rho={rho:.3f} (p={p:.1e})")

    # --- within width strata (curvature effect not explained by tolerance)
    print("\nwithin-width rho(dphi, dur):")
    for w, g in d.groupby("width_mm"):
        rho, p = stats.spearmanr(g["dphi_rad"], g["dur_s"])
        print(f"  W={w:g}: rho={rho:.3f} (p={p:.1e}) n={len(g)}")

    # --- partial rho: dur ~ dphi | (log W, dist)
    def resid(y, X):
        X1 = np.column_stack([np.ones(len(y)), X])
        beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
        return y - X1 @ beta
    ctrl = np.column_stack([np.log(d["width_mm"]), d["dist_m"]])
    r_dur = resid(np.log(d["dur_s"]).to_numpy(), ctrl)
    r_phi = resid(d["dphi_rad"].to_numpy(), ctrl)
    prho, pp = stats.spearmanr(r_phi, r_dur)
    print(f"\npartial rho(dphi, log dur | log W, dist) = {prho:.3f} (p={pp:.1e})")
    for L, g in d.groupby("participant"):
        c = np.column_stack([np.log(g["width_mm"]), g["dist_m"]])
        pr, ppp = stats.spearmanr(resid(g["dphi_rad"].to_numpy(), c),
                                  resid(np.log(g["dur_s"]).to_numpy(), c))
        print(f"  {L}: partial rho={pr:.3f} (p={ppp:.1e})")

    # --- OLS on log dur: delta-R2 of dphi
    y = np.log(d["dur_s"]).to_numpy()
    X0 = np.column_stack([np.log(d["width_mm"]), d["dist_m"]])
    X1 = np.column_stack([X0, d["dphi_rad"]])
    def r2(X):
        Xc = np.column_stack([np.ones(len(y)), X])
        beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
        return 1 - np.sum((y - Xc @ beta) ** 2) / np.sum((y - y.mean()) ** 2), beta
    r2_0, _ = r2(X0)
    r2_1, beta1 = r2(X1)
    print(f"\nOLS log dur ~ log W + dist (+ dphi): R2 {r2_0:.3f} -> {r2_1:.3f} "
          f"(delta {r2_1 - r2_0:.3f}); dphi coef {beta1[3]:.3f} log-units/rad "
          f"(x{np.exp(beta1[3]):.2f} duration per rad)")

    # --- speed channel: inverse speed vs curvature density
    rho_s, p_s = stats.spearmanr(d["kappa_density"], d["inv_speed"])
    print(f"inverse speed (dur/dist) vs curvature density (dphi/dist): "
          f"rho={rho_s:.3f} (p={p_s:.1e})")
    ctrl_w = np.log(d["width_mm"]).to_numpy()[:, None]
    pr_s, pp_s = stats.spearmanr(resid(d["kappa_density"].to_numpy(), ctrl_w),
                                 resid(np.log(d["inv_speed"]).to_numpy(), ctrl_w))
    print(f"  partial (controlling log W): rho={pr_s:.3f} (p={pp_s:.1e})")

    # --- figure: binned medians per width stratum
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    cmap = plt.get_cmap("viridis")
    widths = sorted(d["width_mm"].unique())
    for k, w in enumerate(widths):
        g = d[d["width_mm"] == w]
        bins = np.quantile(g["dphi_rad"], np.linspace(0, 1, 6))
        bins[0] -= 1e-9
        lab = pd.cut(g["dphi_rad"], np.unique(bins))
        m = g.groupby(lab).agg(x=("dphi_rad", "median"), y=("dur_s", "median"),
                               z=("inv_speed", "median"))
        axes[0].plot(m["x"], m["y"], "o-", color=cmap(k / 4), label=f"W={w:g} mm")
        axes[1].plot(m["x"], m["z"], "o-", color=cmap(k / 4), label=f"W={w:g} mm")
    axes[0].set_ylabel("median catch-up duration (s)")
    axes[1].set_ylabel("median inverse speed dur/dist (s/m)")
    for ax in axes:
        ax.set_xlabel("accumulated |curvature| over caught-up segment (rad)")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    axes[0].set_title("duration vs accumulated curvature")
    axes[1].set_title("slowing vs accumulated curvature")
    fig.suptitle("Catch-up vs curvature to consume — quintile bins within width strata", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(BASE / "catchup_vs_curvature.png", dpi=150)
    print(f"\nfigure -> {BASE / 'catchup_vs_curvature.png'}")


if __name__ == "__main__":
    main()

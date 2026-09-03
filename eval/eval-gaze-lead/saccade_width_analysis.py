"""Saccade distance vs tunnel width (tolerance), from the stored lead series.

Input: human-gaze-lead-10p/data/pXX_steering_lead.csv (the plotted per-round
signed-lead sawtooth). Forward saccades along the path are detected as
contiguous runs of lead velocity > V_ON (the sawtooth resets; fixation decay
is slow and negative), and the saccade distance is the lead change summed
over the run (cursor progress during the ~30 ms saccade is negligible).
Events crossing sample gaps (dt > MAX_DT) or outside [A_MIN, A_MAX] are
dropped (projection artifacts).

Outputs (into human-gaze-lead-10p/):
  data/saccade_events.csv        one row per detected forward saccade
  data/saccade_vs_width_summary.csv  per participant x width medians + fits
  saccade_vs_width.png           log-log summary figure

Prints per-participant and pooled power-law fits A = a * W^b and Spearman rho.
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR / "human-gaze-lead-10p"

V_ON = 0.5      # m/s lead velocity to mark saccade samples
MAX_DT = 0.05   # s, no event across recording gaps
A_MIN = 0.008   # m, minimum saccade distance (projection jitter below)
A_MAX = 0.40    # m, above = projection artifact (jump across path branches)


def detect(g):
    """Forward-saccade events in one round's (t, lead) series."""
    g = g.sort_values("t")
    t = g["t"].to_numpy(); lead = g["lead"].to_numpy()
    dl = np.diff(lead); dt = np.diff(t)
    on = (dl / np.where(dt > 0, dt, np.inf) > V_ON) & (dt <= MAX_DT)
    ev = []
    i = 0
    while i < len(on):
        if on[i]:
            j = i
            while j + 1 < len(on) and on[j + 1]:
                j += 1
            amp = lead[j + 1] - lead[i]
            if A_MIN <= amp <= A_MAX:
                ev.append((t[i], t[j + 1], amp, lead[i], lead[j + 1]))
            i = j + 1
        i += 1
    return ev


def main():
    files = sorted(glob.glob(str(BASE / "data" / "p*_steering_lead.csv")))
    if not files:
        sys.exit("no steering lead CSVs found")
    rows = []
    for f in files:
        d = pd.read_csv(f)
        # NOTE: not grouped on tunnel_type — pandas parses its "None" entries
        # (the normal sinusoid) as NaN, and groupby would drop those rows
        for (L, lab, tid, w, rnd), g in d.groupby(
                ["participant", "type_label", "trial_id", "width_mm", "round"]):
            for t0, t1, amp, l0, l1 in detect(g):
                rows.append(dict(participant=L, type_label=lab, trial_id=tid,
                                 width_mm=w, round=rnd, t=t0, t_end=t1, amp_m=amp,
                                 lead_pre=l0, lead_post=l1))
    ev = pd.DataFrame(rows)
    print(f"{len(ev)} forward saccades detected "
          f"({len(files)} participants, types: {sorted(ev['type_label'].unique())})")

    # per participant x width medians
    med = (ev.groupby(["participant", "width_mm"])["amp_m"]
             .agg(["median", "mean", "count"]).reset_index())

    # fits
    print("\nper-participant power law  amp = a * W^b  (on per-saccade data):")
    fits = []
    for L, g in ev.groupby("participant"):
        b, loga, r, p, se = stats.linregress(np.log(g["width_mm"]), np.log(g["amp_m"] * 1000))
        rho, prho = stats.spearmanr(g["width_mm"], g["amp_m"])
        fits.append(dict(participant=L, b=b, a_mm=np.exp(loga), r=r, rho=rho, n=len(g)))
        print(f"  {L}: b={b:.3f}  a={np.exp(loga):.2f} mm  pearson r(log)={r:.3f}  "
              f"spearman rho={rho:.3f} (p={prho:.1e})  n={len(g)}")
    fd = pd.DataFrame(fits)
    print(f"  mean exponent b = {fd['b'].mean():.3f} +- {fd['b'].std():.3f}")

    pooled_b, pooled_a, pr, pp, _ = stats.linregress(np.log(ev["width_mm"]),
                                                     np.log(ev["amp_m"] * 1000))
    prho, prho_p = stats.spearmanr(ev["width_mm"], ev["amp_m"])
    print(f"\npooled: b={pooled_b:.3f}, a={np.exp(pooled_a):.2f} mm, "
          f"spearman rho={prho:.3f} (p={prho_p:.1e}), n={len(ev)}")

    print("\nmedian saccade distance by width (pooled, mm):")
    byw = ev.groupby("width_mm")["amp_m"].agg(["median", "mean", "count"])
    for w, r in byw.iterrows():
        print(f"  W={w:g} mm: median={r['median']*1000:.1f}  mean={r['mean']*1000:.1f}  "
              f"ratio median/W={r['median']*1000/w:.2f}  n={int(r['count'])}")

    print("\nmedian saccade distance by width x type (mm):")
    piv = ev.pivot_table(index="type_label", columns="width_mm", values="amp_m",
                         aggfunc="median") * 1000
    print(piv.round(1).to_string())

    # ---- catch-up duration: fixation interval from a saccade's end to the
    # next saccade's start within the same round
    ev = ev.sort_values(["participant", "type_label", "trial_id", "round", "t"])
    grp = ev.groupby(["participant", "type_label", "trial_id", "round"])
    ev["catchup_s"] = grp["t"].shift(-1) - ev["t_end"]
    cu = ev[(ev["catchup_s"] >= 0.05) & (ev["catchup_s"] <= 3.0)].copy()
    ev.to_csv(BASE / "data" / "saccade_events.csv", index=False, float_format="%.6f")

    print(f"\ncatch-up durations: {len(cu)} intervals (0.05-3 s window)")
    print("per-participant power law  dur = a * W^b:")
    cfits = []
    for L, g in cu.groupby("participant"):
        b, loga, r, p, se = stats.linregress(np.log(g["width_mm"]), np.log(g["catchup_s"]))
        rho, prho = stats.spearmanr(g["width_mm"], g["catchup_s"])
        cfits.append(dict(participant=L, b=b, rho=rho))
        print(f"  {L}: b={b:.3f}  spearman rho={rho:.3f} (p={prho:.1e})  n={len(g)}")
    cfd = pd.DataFrame(cfits)
    print(f"  mean exponent b = {cfd['b'].mean():.3f} +- {cfd['b'].std():.3f}")
    cb, ca, _, _, _ = stats.linregress(np.log(cu["width_mm"]), np.log(cu["catchup_s"]))
    crho, crho_p = stats.spearmanr(cu["width_mm"], cu["catchup_s"])
    print(f"pooled: b={cb:.3f}, spearman rho={crho:.3f} (p={crho_p:.1e})")
    arho, arho_p = stats.spearmanr(cu["amp_m"], cu["catchup_s"])
    print(f"catch-up duration vs preceding saccade distance: rho={arho:.3f} (p={arho_p:.1e})")

    print("\nmedian catch-up duration by width (pooled, s):")
    for w, r in cu.groupby("width_mm")["catchup_s"].agg(["median", "mean", "count"]).iterrows():
        print(f"  W={w:g} mm: median={r['median']:.3f}  mean={r['mean']:.3f}  n={int(r['count'])}")
    print("\nmedian catch-up duration by width x type (s):")
    print(cu.pivot_table(index="type_label", columns="width_mm", values="catchup_s",
                         aggfunc="median").round(3).to_string())

    cmed = (cu.groupby(["participant", "width_mm"])["catchup_s"]
              .agg(catchup_median="median", catchup_mean="mean", catchup_count="count")
              .reset_index())
    med = med.merge(cmed, on=["participant", "width_mm"], how="left")
    med.to_csv(BASE / "data" / "saccade_vs_width_summary.csv", index=False,
               float_format="%.6f")

    # figure: per-participant medians, log-log, plus pooled fit
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    ax = axes[0]
    for L, g in med.groupby("participant"):
        ax.plot(g["width_mm"], g["median"] * 1000, "o-", label=L, alpha=0.8)
    wgrid = np.linspace(10, 50, 50)
    ax.plot(wgrid, np.exp(pooled_a) * wgrid ** pooled_b, "k--", lw=2,
            label=f"pooled fit $aW^b$, b={pooled_b:.2f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([10, 12.5, 16.5, 25, 50]); ax.set_xticklabels([10, 12.5, 16.5, 25, 50])
    ax.set_xlabel("tunnel width / tolerance W (mm)")
    ax.set_ylabel("median forward saccade distance (mm)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    ax.set_title("per-participant medians (log-log)")

    ax = axes[1]
    widths = sorted(ev["width_mm"].unique())
    ax.violinplot([ev.loc[ev["width_mm"] == w, "amp_m"] * 1000 for w in widths],
                  positions=range(len(widths)), showmedians=True)
    ax.set_xticks(range(len(widths))); ax.set_xticklabels([f"{w:g}" for w in widths])
    ax.set_xlabel("tunnel width / tolerance W (mm)")
    ax.set_ylabel("forward saccade distance (mm)")
    ax.set_title("pooled distributions"); ax.grid(alpha=0.3)

    ax = axes[2]
    for L, g in med.groupby("participant"):
        ax.plot(g["width_mm"], g["catchup_median"], "o-", label=L, alpha=0.8)
    ax.plot(wgrid, np.exp(ca) * wgrid ** cb, "k--", lw=2,
            label=f"pooled fit $aW^b$, b={cb:.2f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([10, 12.5, 16.5, 25, 50]); ax.set_xticklabels([10, 12.5, 16.5, 25, 50])
    ax.set_xlabel("tunnel width / tolerance W (mm)")
    ax.set_ylabel("median catch-up duration (s)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    ax.set_title("catch-up (inter-saccade fixation) duration")

    fig.suptitle("Saccade distance & catch-up duration vs tunnel width — steering tasks, "
                 "6 participants", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(BASE / "saccade_vs_width.png", dpi=150)
    print(f"\nfigure -> {BASE / 'saccade_vs_width.png'}")


if __name__ == "__main__":
    main()

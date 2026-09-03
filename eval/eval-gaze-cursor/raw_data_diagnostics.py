"""Raw gaze-data diagnostics: what is actually wrong with each session.

One PDF (results/raw_data_diagnostics.pdf): first a cohort summary page, then one
page per participant (A/B/C and p01-p10, INCLUDING the unusable ones) with:

  (1) a sample trial: tunnel centerline, cursor rounds (blue), RAW uncorrected
      gaze samples (color = round) — spatial offset and scatter are visible directly;
  (2) per-block calibration offset (drift estimate) over the session, x and y,
      with the global pointing bias for reference — headset slip shows as a
      wandering line, a good session as a flat one near the bias;
  (3) fixations per round and kept fraction — sparse-fixation sessions (p08)
      and heavy-drop sessions are visible;
  (4) raw exported signed lead (gaze_lead_signed) distribution — a drifting
      session sits left of zero;
  (5) event-drop breakdown (canonical cleaning).

Usage: python raw_data_diagnostics.py [--letters A B C p01 ...]
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import gaze_data as gd, lookahead_difficulty as ld, gaze_cleaning as gc

PROC = HERE.parents[1] / "human_data" / "processed_gaze_events"
ALL = ["A", "B", "C"] + [f"p{i:02d}" for i in range(1, 11)]


def collect(L):
    s = gd.load_samples(L)
    bias = gc.estimate_bias(s)
    drift = gc.estimate_block_drift(s)          # estimates for ALL blocks (no gate)
    cl = pd.read_csv(PROC / f"{L}_fixation_events_clean.csv")
    st = s[s["cursor_x"].notna() & ~s["tunnel_type"].astype(str).str.contains("pointing")]
    return s, st, bias, drift, cl


def page(pdf, L):
    s, st, bias, drift, cl = collect(L)
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1], hspace=0.35, wspace=0.3)

    # (1) sample trial overlay: prefer a sinusoidal type, else any steering trial
    ax = fig.add_subplot(gs[0, :2])
    ev = gd.fixation_events(s); ev = ev[~ev["tunnel_type"].astype(str).str.contains("pointing")]
    _, geoms = ld.attach_geometry(s, ev)
    pref = [t for t in st["tunnel_type"].dropna().unique() if "sinusoidal" in str(t)]
    ty = pref[0] if pref else str(st["tunnel_type"].dropna().iloc[0])
    tid = sorted(st[st["tunnel_type"] == ty]["trial_id"].unique())[0]
    g = geoms.get((L, tid))
    b = st[st["trial_id"] == tid]
    if g is not None:
        ax.plot(g.path[:, 0], g.path[:, 1], "-", color="0.75", lw=6, solid_capstyle="round", zorder=0)
    blocks = sorted(b["block_id"].unique())
    cols = plt.cm.autumn(np.linspace(0, 0.75, max(len(blocks), 2)))
    for ri, bid in enumerate(blocks):
        bb = b[b["block_id"] == bid].iloc[::2]
        ax.plot(bb["cursor_x"], bb["cursor_y"], "-", color="tab:blue", lw=0.8, alpha=0.6)
        ax.plot(bb["gaze_task_x"], bb["gaze_task_y"], ".", ms=2, color=cols[ri], alpha=0.5)
    ax.set_aspect("equal"); ax.set_title(f"{L}  t{tid} {ty}: RAW gaze (color=round) vs cursor (blue) — no corrections")
    ax.set_xticks([]); ax.set_yticks([])

    # (2) per-block drift over session
    ax = fig.add_subplot(gs[0, 2])
    if drift:
        t0s = st.groupby("block_id")["t"].min()
        tt = np.array([t0s.get(bid, np.nan) for bid in drift]) ; tt = (tt - np.nanmin(tt)) / 60.0
        ox = np.array([v[0] for v in drift.values()]) * 1000
        oy = np.array([v[1] for v in drift.values()]) * 1000
        o = np.argsort(tt)
        ax.plot(tt[o], ox[o], ".-", ms=3, lw=0.6, label="offset x")
        ax.plot(tt[o], oy[o], ".-", ms=3, lw=0.6, label="offset y")
        ax.axhline(bias[0]*1000, color="C0", ls="--", lw=0.8, alpha=0.6)
        ax.axhline(bias[1]*1000, color="C1", ls="--", lw=0.8, alpha=0.6)
        ax.axhline(0, color="0.5", lw=0.6)
    ax.set_title("per-round calibration offset (mm)\ndashed = global pointing bias")
    ax.set_xlabel("session time (min)"); ax.legend(fontsize=7); ax.grid(alpha=0.25)

    # (3) fixations per round + kept fraction
    ax = fig.add_subplot(gs[1, 0])
    per = cl.groupby(["trial_id", "block_id"]).agg(n=("keep", "size"), kept=("keep", "sum"))
    ax.hist(per["n"], bins=np.arange(0.5, 20.5), color="0.7", label="fixations/round")
    ax.axvline(per["n"].median(), color="k", lw=1)
    ax.set_title(f"fixations per round (median {per['n'].median():.0f}); "
                 f"rounds with >=3 kept: {(per['kept']>=3).mean():.0%}")
    ax.set_xlabel("fixations in round"); ax.grid(alpha=0.25)

    # (4) raw exported signed lead
    ax = fig.add_subplot(gs[1, 1])
    lead = pd.to_numeric(st["lead"], errors="coerce").dropna()
    ax.hist(np.clip(lead, -0.1, 0.15), bins=60, color="#8c6bb1", alpha=0.8)
    ax.axvline(0, color="k", lw=1)
    ax.axvline(np.median(lead), color="r", lw=1.2)
    ax.set_title(f"RAW exported signed lead (median {np.median(lead)*1000:+.0f} mm, red)")
    ax.set_xlabel("lead (task units)"); ax.grid(alpha=0.25)

    # (5) drop breakdown
    ax = fig.add_subplot(gs[1, 2])
    n = len(cl)
    parts = {"kept": int(cl["keep"].sum()),
             "blink": int(cl["blink_corrupted"].sum()),
             "off-path": int(cl["off_path"].sum()),
             "regressive": int(cl["regressive"].sum())}
    ax.bar(parts.keys(), [v / max(n, 1) for v in parts.values()],
           color=["#66a61e", "0.6", "#d95f02", "#7570b3"])
    ax.set_ylim(0, 1); ax.set_title(f"event fates ({n} fixations; flags overlap)")
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle(f"{L} — raw data diagnostics", fontsize=15)
    pdf.savefig(fig); plt.close(fig)
    per_ok = (per["kept"] >= 3).mean()
    dm = (np.median([np.hypot(v[0]-bias[0], v[1]-bias[1]) for v in drift.values()]) * 1000) if drift else np.nan
    return {"L": L, "fix_per_round": float(per["n"].median()), "rounds_ok": float(per_ok),
            "drift_disagree_mm": float(dm), "raw_lead_mm": float(np.median(lead)*1000),
            "kept_frac": parts["kept"]/max(n,1), "n_blocks_est": len(drift)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--letters", nargs="*", default=ALL)
    a = ap.parse_args()
    out = HERE / "results" / "raw_data_diagnostics.pdf"
    rows = []
    with PdfPages(out) as pdf:
        stats = []
        for L in a.letters:
            stats.append(page(pdf, L))
            print(f"{L} done", flush=True)
        # cohort summary as page 1? PdfPages appends; draw summary last instead.
        df = pd.DataFrame(stats).set_index("L")
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for ax, (col, title) in zip(axes, [
                ("drift_disagree_mm", "median block-offset vs bias (mm)\n= calibration drift"),
                ("fix_per_round", "median fixations per round"),
                ("kept_frac", "fraction of events kept"),
                ("raw_lead_mm", "RAW median signed lead (mm)")]):
            colors = ["#1b9e77" if i < 3 else "#d95f02" for i in range(len(df))]
            ax.bar(df.index, df[col], color=colors)
            ax.set_title(title, fontsize=9); ax.tick_params(axis="x", rotation=60, labelsize=7)
            ax.grid(alpha=0.25, axis="y")
            if col == "raw_lead_mm": ax.axhline(0, color="k", lw=0.8)
        fig.suptitle("cohort summary (green = original A/B/C, orange = new batch)", fontsize=12)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()

"""Fixation maps: where each participant's fixations land along every steering tunnel.

One PNG per participant: panels sorted by (tunnel type, width); each panel shows the
corridor (grey band, true local width), the cursor rounds (light blue), and the fixation
points in task coordinates (dots, size ~ dwell time, color = round). Panel title: type,
width, n fixations, median onset lead.

Usage: python fixation_maps.py [--letters p01 ... ] [--out-dir results/fixmaps]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import gaze_data as gd, lookahead_difficulty as ld

ORDER = ["straight", "corner", "mid_sinusoidal", "sinusoidal", "gentle_sinusoidal",
         "sharp_sinusoidal", "wide_to_narrow", "narrow_to_wide", "constrained_to_unconstrained"]


def band(ax, geom):
    p, W = geom.path, np.clip(geom.W, 0, 0.2)
    segs, widths = [], []
    for i in range(len(p) - 1):
        segs.append([p[i], p[i + 1]]); widths.append(W[i])
    # linewidth in points from data units after limits are known — approximate with a
    # second pass: draw once with lw=1, fix after autoscale
    lc = LineCollection(segs, colors="0.88", capstyle="round", joinstyle="round", zorder=0)
    ax.add_collection(lc)
    return lc, np.array(widths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--letters", nargs="*", default=[f"p{i:02d}" for i in range(1, 11)])
    ap.add_argument("--out-dir", default=str(HERE / "results" / "fixmaps"))
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for L in a.letters:
        s = gd.load_samples(L); ev = gd.fixation_events(s)
        ev = ev[(~ev["tunnel_type"].astype(str).str.contains("pointing")) & (~ev["blink_corrupted"])]
        ev, geoms = ld.attach_geometry(s, ev)
        trials = (ev.groupby("trial_id").first().reset_index()
                    .sort_values(by=["tunnel_type", "width"],
                                 key=lambda c: c.map(lambda v: ORDER.index(v) if v in ORDER else v if isinstance(v, float) else 99)))
        tids = trials["trial_id"].tolist()
        nc = 5; nr = int(np.ceil(len(tids) / nc))
        fig, axes = plt.subplots(nr, nc, figsize=(4.6 * nc, 2.1 * nr))
        for ax in np.ravel(axes): ax.axis("off")
        for ax, tid in zip(np.ravel(axes), tids):
            g = geoms.get((L, tid)); e = ev[ev["trial_id"] == tid]
            if g is None: continue
            ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
            lc, widths = band(ax, g)
            raw = s[(s["trial_id"] == tid) & s["cursor_x"].notna()]
            for _, blk in raw.groupby("block_id"):
                ax.plot(blk["cursor_x"], blk["cursor_y"], "-", color="tab:blue", lw=0.6, alpha=0.5)
            rounds = {b: i for i, b in enumerate(sorted(e["block_id"].unique()))}
            cols = plt.cm.autumn(np.linspace(0, 0.75, max(len(rounds), 2)))
            for _, r in e.iterrows():
                if np.isfinite(r.get("gaze_task_x", np.nan)):
                    ax.plot(r["gaze_task_x"], r["gaze_task_y"], "o", ms=2 + 8 * min(float(r["duration_s"]), 1.0),
                            color=cols[rounds[r["block_id"]]], alpha=0.75, mec="none")
            ax.set_aspect("equal"); ax.margins(0.05)
            fig.canvas.draw()
            ppu = float(np.hypot(*(ax.transData.transform((1, 0)) - ax.transData.transform((0, 0)))))
            lc.set_linewidths(np.clip(widths, 0, 0.2) * ppu * 72.0 / fig.dpi)
            ty = str(e["tunnel_type"].iloc[0]); w = e["width"].iloc[0]
            lead = np.nanmedian(e["lead_onset"]) * 1000
            ax.set_title(f"t{tid} {ty[:14]} W={w*1000:.0f} | {len(e)} fix, lead {lead:.0f}mm", fontsize=7)
        fig.suptitle(f"{L}: fixation landings (dot size = dwell; color = round; grey = corridor)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(out / f"fixmap_{L}.png", dpi=120); plt.close(fig)
        print(f"{L}: {len(tids)} trials -> {out}/fixmap_{L}.png", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()

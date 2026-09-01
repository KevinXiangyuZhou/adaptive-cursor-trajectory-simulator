"""Fixation maps from the PROCESSED gaze data: bias-corrected fixations with their
forward-constrained projection onto the path.

Per participant PNG: corridor (grey band, true local width), cursor rounds (blue),
kept fixations as corrected-gaze dots (size = dwell, color = round) connected to their
projected point ON the path (black dot at arc position s_cursor + lead_corr); dropped
fixations (blink / off-path / regressive) as faint grey x markers.

Usage: python fixation_maps_clean.py [--letters A B C p01 ...] [--out-dir results/fixmaps_clean]
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import gaze_data as gd, lookahead_difficulty as ld

PROC = HERE.parents[1] / "human_data" / "processed_gaze_events"
ORDER = ["straight", "corner", "mid_sinusoidal", "sinusoidal", "gentle_sinusoidal",
         "sharp_sinusoidal", "wide_to_narrow", "narrow_to_wide", "constrained_to_unconstrained"]


def path_at(geom, s):
    s = np.clip(s, geom.s[0], geom.s[-1])
    x = np.interp(s, geom.s, geom.path[:, 0]); y = np.interp(s, geom.s, geom.path[:, 1])
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--letters", nargs="*", default=["A", "B", "C"] + [f"p{i:02d}" for i in range(1, 11)])
    ap.add_argument("--out-dir", default=str(HERE / "results" / "fixmaps_clean"))
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for L in a.letters:
        s = gd.load_samples(L)
        cl = pd.read_csv(PROC / f"{L}_fixation_events_clean.csv")
        # geometry per trial (same builders as the cleaning)
        ev = gd.fixation_events(s); ev = ev[~ev["tunnel_type"].astype(str).str.contains("pointing")]
        _, geoms = ld.attach_geometry(s, ev)
        trials = (cl.groupby("trial_id").first().reset_index()
                    .sort_values(by=["tunnel_type", "width"],
                                 key=lambda c: c.map(lambda v: ORDER.index(v) if v in ORDER else v if isinstance(v, float) else 99)))
        tids = trials["trial_id"].tolist()
        nc = 5; nr = int(np.ceil(len(tids) / nc))
        fig, axes = plt.subplots(nr, nc, figsize=(4.6 * nc, 2.1 * nr))
        for ax in np.ravel(axes): ax.axis("off")
        for ax, tid in zip(np.ravel(axes), tids):
            g = geoms.get((L, tid)); e = cl[cl["trial_id"] == tid]
            if g is None: continue
            ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
            segs = [[g.path[i], g.path[i + 1]] for i in range(len(g.path) - 1)]
            lc = LineCollection(segs, colors="0.88", capstyle="round", joinstyle="round", zorder=0)
            ax.add_collection(lc)
            raw = s[(s["trial_id"] == tid) & s["cursor_x"].notna()]
            for _, blk in raw.groupby("block_id"):
                ax.plot(blk["cursor_x"], blk["cursor_y"], "-", color="tab:blue", lw=0.6, alpha=0.5)
            drop = e[~e["keep"]]
            ax.plot(drop["gaze_x_corr"], drop["gaze_y_corr"], "x", color="0.7", ms=3, mew=0.8, zorder=2)
            keep = e[e["keep"]]
            rounds = {b: i for i, b in enumerate(sorted(keep["block_id"].unique()))}
            cols = plt.cm.autumn(np.linspace(0, 0.75, max(len(rounds), 2)))
            for _, r in keep.iterrows():
                px, py = path_at(g, r["s_c"] + r["lead_corr"])
                ax.plot([r["gaze_x_corr"], px], [r["gaze_y_corr"], py], "-", color="0.6", lw=0.5, alpha=0.6, zorder=3)
                ax.plot(r["gaze_x_corr"], r["gaze_y_corr"], "o", ms=2 + 8 * min(float(r["duration_s"]), 1.0),
                        color=cols[rounds[r["block_id"]]], alpha=0.75, mec="none", zorder=4)
                ax.plot(px, py, "o", ms=2.5, color="k", zorder=5)
            ax.set_aspect("equal"); ax.margins(0.05)
            fig.canvas.draw()
            ppu = float(np.hypot(*(ax.transData.transform((1, 0)) - ax.transData.transform((0, 0)))))
            W = np.clip(g.W, 0, 0.2)
            lc.set_linewidths(W[:-1] * ppu * 72.0 / fig.dpi)
            ty = str(e["tunnel_type"].iloc[0]); w = e["width"].iloc[0]
            ml = np.nanmedian(keep["lead_corr"]) * 1000 if len(keep) else float("nan")
            ax.set_title(f"t{tid} {ty[:14]} W={w*1000:.0f} | {len(keep)} kept / {len(drop)} dropped, lead {ml:.0f}mm", fontsize=7)
        fig.suptitle(f"{L}: cleaned fixations (dots = corrected gaze, black = projection on path, grey x = dropped)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(out / f"fixmap_{L}.png", dpi=120); plt.close(fig)
        print(f"{L}: {len(tids)} trials -> {out}/fixmap_{L}.png", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()

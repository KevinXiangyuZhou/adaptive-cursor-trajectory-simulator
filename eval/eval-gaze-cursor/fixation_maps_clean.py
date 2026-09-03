"""Fixation maps from the PROCESSED gaze data: bias-corrected fixations with their
forward-constrained projection onto the path.

Encoding per trial: corridor (grey band, true local width), cursor rounds (blue),
kept fixations as corrected-gaze dots (size = dwell, color = round) connected to their
projected point ON the path (small black-edged dot at arc position s_cursor + lead_corr,
same round color); the cursor position at fixation onset (where the cursor was when the
saccade landed) as an open circle in the round color, joined to its gaze dot by a dotted
saccade-lead line; dropped fixations (blink / off-path / regressive) as faint x markers
in their round's color. Fragments absorbed by the merge pass (merged_into set) are NOT
drawn — their dwell is inside the survivor's dot, whose size reflects the merged duration.

Output per participant (results/fixmaps_merged/{L}/ — the merged dataset is the canonical
one for model fitting; the pre-merge maps in results/fixmaps_clean are left untouched):
  {L}/t{tid}_{type}_W{mm}mm.png — ONE TRIAL PER FILE, one round per stacked panel
                                  (rounds separated, not overplotted): the readable
                                  version for close inspection.
  {L}/overview.png              — the all-trials overview grid (small panels, all rounds
                                  together, for scanning).
Round k always draws in ROUND_COLORS[k-1], so panels and overview cross-reference.

Usage: python fixation_maps_clean.py [--letters A B C p01 ...] [--out-dir results/fixmaps_merged]
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import gaze_data as gd, lookahead_difficulty as ld

PROC = HERE.parents[1] / "human_data" / "processed_gaze_events"
ORDER = ["straight", "corner", "mid_sinusoidal", "sinusoidal", "gentle_sinusoidal",
         "sharp_sinusoidal", "wide_to_narrow", "narrow_to_wide", "constrained_to_unconstrained"]
LEGEND = ("filled dot = corrected gaze (size = merged dwell) | black-edged dot = projection on path | "
          "open circle = cursor at fixation onset (dotted line = saccade-time lead) | x = dropped | color = round")
# Qualitative round palette (Okabe-Ito / Tol muted mix), ordered so the common
# 3-4-round trials get maximally distinct hues; avoids the cursor-path blue and
# the corridor grey. Wraps past 10 rounds.
ROUND_COLORS = ["#D55E00", "#009E73", "#9467BD", "#E69F00", "#CC79A7",
                "#332288", "#8C564B", "#44AA99", "#999933", "#882255"]


def round_palette(n):
    return [ROUND_COLORS[i % len(ROUND_COLORS)] for i in range(n)]


def round_handles(n, ms=6):
    return [Line2D([], [], marker="o", ls="none", color=ROUND_COLORS[i % len(ROUND_COLORS)],
                   ms=ms, label=f"round {i + 1}") for i in range(n)]


def path_at(geom, s):
    s = np.clip(s, geom.s[0], geom.s[-1])
    x = np.interp(s, geom.s, geom.path[:, 0]); y = np.interp(s, geom.s, geom.path[:, 1])
    return x, y


def draw_trial(fig, ax, g, e, raw, scale=1.0, title_fs=9, block=None, round_idx=None):
    """Draw one trial's fixation map on ax. `scale` multiplies marker sizes and
    linewidths (1 for the overview grid, larger for the per-trial figures).
    With `block` set, only that round's cursor path and fixations are drawn
    (round_idx = its 0-based ordinal, for the fixed round color and the title)."""
    ax.set_xticks([]); ax.set_yticks([])
    segs = [[g.path[i], g.path[i + 1]] for i in range(len(g.path) - 1)]
    lc = LineCollection(segs, colors="0.88", capstyle="round", joinstyle="round", zorder=0)
    ax.add_collection(lc)
    # round colors from ALL blocks of the trial so dropped rows map too
    rounds = {b: i for i, b in enumerate(sorted(e["block_id"].unique()))}
    cols = round_palette(max(len(rounds), 1))
    if block is not None:
        e = e[e["block_id"] == block]
        raw = raw[raw["block_id"] == block]
        rounds = {block: round_idx}
        cols = {round_idx: ROUND_COLORS[round_idx % len(ROUND_COLORS)]}
    for _, blk in raw.groupby("block_id"):
        ax.plot(blk["cursor_x"], blk["cursor_y"], "-", color="tab:blue", lw=0.9 * scale, alpha=0.5)
    drop = e[~e["keep"]]
    if "merged_into" in e.columns:
        # absorbed merge fragments live inside their survivor's dot
        drop = drop[drop["merged_into"].isna()]
    keep = e[e["keep"]]
    for b, dg in drop.groupby("block_id"):
        ax.plot(dg["gaze_x_corr"], dg["gaze_y_corr"], "x", color=cols[rounds[b]],
                alpha=0.45, ms=4.2 * scale, mew=1.1 * scale, zorder=2)
    has_cursor = "cursor_x" in keep.columns
    for _, r in keep.iterrows():
        px, py = path_at(g, r["s_c"] + r["lead_corr"])
        c = cols[rounds[r["block_id"]]]
        ax.plot([r["gaze_x_corr"], px], [r["gaze_y_corr"], py], "-",
                color="0.6", lw=0.7 * scale, alpha=0.6, zorder=3)
        if has_cursor and np.isfinite(r["cursor_x"]):
            # cursor position at fixation onset (where the cursor was when the
            # saccade landed): open circle + dotted saccade-lead connector
            ax.plot([r["cursor_x"], r["gaze_x_corr"]], [r["cursor_y"], r["gaze_y_corr"]],
                    ":", color=c, lw=0.7 * scale, alpha=0.35, zorder=2.5)
            ax.plot(r["cursor_x"], r["cursor_y"], "o", ms=4.5 * scale, mfc="none",
                    mec=c, mew=1.1 * scale, alpha=0.9, zorder=4)
        ax.plot(r["gaze_x_corr"], r["gaze_y_corr"], "o",
                ms=(3 + 11 * min(float(r["duration_s"]), 1.0)) * scale,
                color=c, alpha=0.75, mec="none", zorder=4)
        ax.plot(px, py, "o", ms=3.5 * scale, color=c, mec="k", mew=0.6 * scale, zorder=5)
    ax.set_aspect("equal"); ax.margins(0.05)
    fig.canvas.draw()
    ppu = float(np.hypot(*(ax.transData.transform((1, 0)) - ax.transData.transform((0, 0)))))
    W = np.clip(g.W, 0, 0.2)
    lc.set_linewidths(W[:-1] * ppu * 72.0 / fig.dpi)
    ml = np.nanmedian(keep["lead_corr"]) * 1000 if len(keep) else float("nan")
    if block is not None:
        ax.set_title(f"round {round_idx + 1} | {len(keep)} kept / {len(drop)} dropped, "
                     f"median lead {ml:.0f}mm", fontsize=title_fs,
                     color=ROUND_COLORS[round_idx % len(ROUND_COLORS)])
    else:
        ty = str(e["tunnel_type"].iloc[0]); w = e["width"].iloc[0]
        tid = e["trial_id"].iloc[0]
        ax.set_title(f"t{tid} {ty[:20]} W={w*1000:.0f}mm | {len(keep)} kept / {len(drop)} dropped, "
                     f"median lead {ml:.0f}mm", fontsize=title_fs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--letters", nargs="*", default=["A", "B", "C"] + [f"p{i:02d}" for i in range(1, 11)])
    ap.add_argument("--out-dir", default=str(HERE / "results" / "fixmaps_merged"))
    ap.add_argument("--dpi", type=int, default=300, help="raster (PNG overview) resolution; the PDF is vector")
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

        pdir = out / L; pdir.mkdir(parents=True, exist_ok=True)

        # 1) one PNG per trial, ONE ROUND PER STACKED PANEL — the readable version
        for tid in tids:
            g = geoms.get((L, tid)); e = cl[cl["trial_id"] == tid]
            if g is None:
                continue
            blocks = sorted(e["block_id"].unique())
            raw = s[(s["trial_id"] == tid) & s["cursor_x"].notna()]
            fig, axes = plt.subplots(len(blocks), 1, figsize=(13, 3.6 * len(blocks)),
                                     squeeze=False, sharex=True, sharey=True)
            for k, (ax, b) in enumerate(zip(axes[:, 0], blocks)):
                draw_trial(fig, ax, g, e, raw, scale=1.6, title_fs=11, block=b, round_idx=k)
            ty = str(e["tunnel_type"].iloc[0]); w = float(e["width"].iloc[0])
            fig.suptitle(f"{L} t{tid} {ty} W={w*1000:.0f}mm — one round per panel", fontsize=13)
            fig.text(0.5, 0.008, LEGEND, ha="center", fontsize=8, color="0.35")
            fig.tight_layout(rect=[0, 0.02, 1, 0.97])
            fig.savefig(pdir / f"t{int(tid):02d}_{ty[:20]}_W{w*1000:.0f}mm.png", dpi=150)
            plt.close(fig)

        # 2) all-trials overview grid (rounds together, fixed round colors)
        nc = 5; nr = int(np.ceil(len(tids) / nc))
        fig, axes = plt.subplots(nr, nc, figsize=(7.0 * nc, 3.2 * nr))
        for ax in np.ravel(axes): ax.axis("off")
        n_rounds_max = int(cl.groupby("trial_id")["block_id"].nunique().max())
        for ax, tid in zip(np.ravel(axes), tids):
            g = geoms.get((L, tid)); e = cl[cl["trial_id"] == tid]
            if g is None: continue
            ax.axis("on")
            raw = s[(s["trial_id"] == tid) & s["cursor_x"].notna()]
            draw_trial(fig, ax, g, e, raw, scale=1.0, title_fs=9)
        fig.suptitle(f"{L}: cleaned + merged fixations ({LEGEND}) — canonical fitting dataset", fontsize=14)
        fig.legend(handles=round_handles(n_rounds_max), loc="upper right", ncol=min(n_rounds_max, 5),
                   fontsize=9, frameon=False)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(pdir / "overview.png", dpi=a.dpi)
        plt.close(fig)
        print(f"{L}: {len(tids)} trials -> {pdir}/ (per-trial round panels + overview.png)", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()

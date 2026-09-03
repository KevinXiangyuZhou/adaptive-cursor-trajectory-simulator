"""Human-only signed gaze-lead plots (no model runs).

One FOLDER per participant ({out-dir}/{L}/), one PNG per steering trial with ONE
ROUND PER STACKED PANEL (shared axes — rounds are separated, not overplotted):
the continuous signed gaze lead along the trial centerline vs time from round
start (dots), with the CANONICAL fixation-onset events (drift-corrected, merged;
human_data/processed_gaze_events) overlaid as black-edged markers. Round k
always draws in ROUND_COLORS[k-1] (shared with fixation_maps_clean.py, so the
lead traces cross-reference the fixation maps). {L}/summary_lead_by_width.png:
median onset lead by width.

The continuous lead is recomputed from the samples with the canonical drift
correction (per-block offsets for gaze_cleaning.DRIFT_PARTICIPANTS, block gate
DRIFT_GATE_M, else the global pointing bias) and then projected EXACTLY the way
the A/B/C `gaze_lead_signed` column was made (model_gaze_lead.ArcProjector):
dense ~1 mm resampling of the centerline, GLOBAL nearest point, no forward
constraint and no clamp — so decays render as smooth slopes and behind/off-path
episodes go negative instead of forming a floor stripe. The raw exported
`gaze_lead_signed` itself is NOT used: for the drifting sessions it is exactly
the corrupted signal.

Usage: python human_gaze_lead.py [--letters p01 ... p10] [--out-dir human-gaze-lead-10p]
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "eval" / "eval-gaze-cursor"))
import gaze_data as gd
import lookahead_difficulty as ld
import gaze_cleaning as gc
from fixation_maps_clean import ROUND_COLORS

PROC = PROJECT_ROOT / "human_data" / "processed_gaze_events"
SUBSAMPLE = 2          # plot every 2nd sample (100 Hz)
BACK_WIN_M = gc.BACK_WIN_M
ORDER = ["straight", "corner", "mid_sinusoidal", "sinusoidal", "gentle_sinusoidal",
         "sharp_sinusoidal", "wide_to_narrow", "narrow_to_wide", "constrained_to_unconstrained"]


def dense_grid(geom, step=0.001):
    """Centerline resampled to a ~1 mm grid (ArcProjector-style)."""
    s_d = np.arange(0.0, geom.s_end + step, step)
    P = np.column_stack([np.interp(s_d, geom.s, geom.path[:, 0]),
                         np.interp(s_d, geom.s, geom.path[:, 1])])
    return s_d, P


def project_dense(s_d, P, x, y, chunk=8000):
    """Arc position of the GLOBAL nearest dense-grid point for each (x, y) —
    the same projection the A/B/C gaze_lead_signed column used (no forward
    constraint, no clamp)."""
    pts = np.column_stack([x, y])
    out = np.empty(len(pts))
    for lo in range(0, len(pts), chunk):
        hi = min(lo + chunk, len(pts))
        d2 = ((pts[lo:hi, None, :] - P[None, :, :]) ** 2).sum(axis=2)
        out[lo:hi] = s_d[np.argmin(d2, axis=1)]
    return out


def nearest_vertex_s(geom, x, y, s_min=None, chunk=20000):
    """Legacy nearest-vertex projection with optional forward constraint —
    kept for diagnostics; the plots use project_dense (ABC method)."""
    pts = np.column_stack([x, y])
    out = np.empty(len(pts))
    P = geom.path
    for lo in range(0, len(pts), chunk):
        hi = min(lo + chunk, len(pts))
        d2 = ((pts[lo:hi, None, :] - P[None, :, :]) ** 2).sum(axis=2)
        if s_min is not None:
            mask = geom.s[None, :] < s_min[lo:hi, None]
            d2 = np.where(mask & ~mask.all(axis=1, keepdims=True), np.inf, d2)
        out[lo:hi] = geom.s[np.argmin(d2, axis=1)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--letters", nargs="*", default=[f"p{i:02d}" for i in range(1, 11)])
    ap.add_argument("--out-dir", default=str(SCRIPT_DIR / "human-gaze-lead-10p"))
    a = ap.parse_args()
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    for L in a.letters:
        s = gd.load_samples(L)
        bias = gc.estimate_bias(s)
        drift = gc.estimate_block_drift(s) if L in gc.DRIFT_PARTICIPANTS else {}
        # gated per-block offsets (same rule as the canonical dataset)
        off_by_block = {b: (v[0], v[1]) for b, v in drift.items()
                        if np.hypot(v[0] - bias[0], v[1] - bias[1]) > gc.DRIFT_GATE_M}
        cl = pd.read_csv(PROC / f"{L}_fixation_events_clean.csv")
        cl = cl[cl["keep"] == True]  # noqa: E712
        ev = gd.fixation_events(s)
        ev = ev[~ev["tunnel_type"].astype(str).str.contains("pointing")]
        _, geoms = ld.attach_geometry(s, ev)

        trials = (cl.groupby("trial_id").first().reset_index()
                    .sort_values(by=["tunnel_type", "width"],
                                 key=lambda c: c.map(lambda v: ORDER.index(v) if v in ORDER else 99)))
        pdir = out_dir / L; pdir.mkdir(parents=True, exist_ok=True)
        for k, (_, tr) in enumerate(trials.iterrows()):
            tid = tr["trial_id"]; g = geoms.get((L, tid))
            if g is None:
                continue
            s_d, P_d = dense_grid(g)
            st = s[(s["trial_id"] == tid) & s["cursor_x"].notna() & s["gaze_task_x"].notna()]
            blocks = [b for b in sorted(st["block_id"].unique())
                      if (st["block_id"] == b).sum() >= 10 * SUBSAMPLE]
            if not blocks:
                continue
            fig, axes = plt.subplots(len(blocks), 1, figsize=(11, 2.8 * len(blocks)),
                                     squeeze=False, sharex=True, sharey=True)
            n_corr = 0
            for ri, (ax, bid) in enumerate(zip(axes[:, 0], blocks)):
                b = st[st["block_id"] == bid].iloc[::SUBSAMPLE]
                c = ROUND_COLORS[ri % len(ROUND_COLORS)]
                ox, oy = off_by_block.get(bid, (bias[0], bias[1]))
                n_corr += bid in off_by_block
                s_c = project_dense(s_d, P_d, b["cursor_x"].to_numpy(), b["cursor_y"].to_numpy())
                s_g = project_dense(s_d, P_d, (b["gaze_task_x"] - ox).to_numpy(),
                                    (b["gaze_task_y"] - oy).to_numpy())
                t0 = b["t"].min()
                lead = s_g - s_c
                ax.plot(b["t"] - t0, lead, ".", ms=2.2, color=c, alpha=0.55)
                # canonical fixation onsets of this round
                e = cl[(cl["trial_id"] == tid) & (cl["block_id"] == bid)]
                ax.plot(e["t_onset"] - t0, e["lead_corr"], "o", ms=5, mfc=c,
                        mec="k", mew=0.7, alpha=0.9, zorder=5)
                ax.axhline(0.0, color="0.4", lw=0.8)
                ax.set_ylabel("signed lead")
                ax.set_title(f"round {ri + 1}" + (" (drift-corr)" if bid in off_by_block else "")
                             + f" | {len(e)} canonical onsets", fontsize=10, color=c)
                ax.grid(alpha=0.25)
            axes[-1, 0].set_xlabel("time since round start (s)")
            fig.suptitle(f"{L}  t{tid}  {tr['tunnel_type']}  W={tr['width']*1000:.0f}mm — one round per panel  "
                         f"({n_corr}/{len(blocks)} rounds drift-corrected; "
                         f"black-edged = canonical fixation onsets; ABC-style global projection)",
                         fontsize=10)
            fig.tight_layout(rect=[0, 0, 1, 0.965])
            ty = str(tr["tunnel_type"])
            fig.savefig(pdir / f"t{int(tid):02d}_{ty[:20]}_W{tr['width']*1000:.0f}mm.png", dpi=150)
            plt.close(fig)

        # summary: median onset lead by width
        fig, ax = plt.subplots(figsize=(7, 4.5))
        byw = cl.groupby(np.round(cl["width"] * 1000))["lead_corr"].median()
        ax.plot(byw.index, byw.values * 1000, "o-", color="#7b3294")
        ax.set_xlabel("tunnel width (mm)"); ax.set_ylabel("median onset lead (mm)")
        ax.set_title(f"{L}: median canonical onset lead by width "
                     f"({int(cl['drift_corrected'].sum())} of {len(cl)} kept events drift-corrected)")
        ax.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(pdir / "summary_lead_by_width.png", dpi=150)
        plt.close(fig)
        print(f"{L}: {len(trials)} trials -> {pdir}/ (per-trial round panels + summary)", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()

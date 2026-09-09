"""Evaluation figure: cursor speed moves the gaze horizon at equal width.

Two panels (human, model) from the cached cycle tables of
quantify_gaze_cycles.py (paper/quant/cycles_{human,model}.csv). Within each
width class, saccades are binned by the pre-saccade cursor speed (equal-count
bins pooled across widths); lines are the median saccade amplitude per width,
i.e. the speed effect at equal width that Sec. eval_cycle quantifies with the
partial correlations. The dashed lines are the median onset lead (negative =
overrun): amplitude grows with speed because the onset sinks, while the
landing lead stays width-set.

Usage:  python paper/fig_speed_horizon.py
Output: paper/figures/speed_horizon.{pdf,png}
"""
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

QUANT = Path(__file__).resolve().parent / "quant"
OUT_DIR = Path(__file__).resolve().parent / "figures"
V_PRE_MAX = 1.0          # m/s, as quantify_gaze_cycles.V_PRE_MAX
N_BINS = 5
WIDTHS = [10.0, 16.5, 25.0, 50.0]
CMAP = plt.get_cmap("viridis")


def panel(ax, d, title):
    d = d[(d["v_pre"] > 0) & (d["v_pre"] <= V_PRE_MAX)].copy()
    for k, w in enumerate(WIDTHS):
        dw = d[d["width_mm"] == w].copy()
        # speed quintiles WITHIN the width class: the x-axis spread is the
        # equal-width speed variation the partial correlations condition on
        dw["vbin"] = pd.qcut(dw["v_pre"], N_BINS, duplicates="drop")
        g = dw.groupby("vbin", observed=True)
        n = g.size()
        v = g["v_pre"].median() * 1000
        ok = n >= 20
        c = CMAP(0.15 + 0.7 * k / (len(WIDTHS) - 1))
        ax.plot(v[ok], g["amp"].median()[ok] * 1000, "o-", color=c,
                label=f"W = {w:g} mm", ms=4)
        ax.plot(v[ok], g["lead_pre"].median()[ok] * 1000, "o--", color=c,
                ms=3, alpha=0.6)
    ax.axhline(0.0, color="0.75", lw=0.8, zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("pre-saccade cursor speed (mm/s)")
    ax.set_title(title, fontsize=10)


def main():
    hum = pd.read_csv(QUANT / "cycles_human.csv")
    mod = pd.read_csv(QUANT / "cycles_model.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
    panel(axes[0], hum, "human")
    panel(axes[1], mod, "model")
    axes[0].set_ylabel("median saccade\namplitude (mm)")
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    axes[1].text(0.03, 0.06, "dashed: onset lead (overrun)",
                 transform=axes[1].transAxes, fontsize=7, color="0.35")
    fig.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"speed_horizon.{ext}", dpi=200)
    print("wrote", OUT_DIR / "speed_horizon.pdf")


if __name__ == "__main__":
    main()

"""Paper figures: steering-law and Fitts'-law regressions, Human vs. Model.

Reads the pooled-8 main eval (results-cluster-10p/eval-main-pooled8-local) and
regenerates the two law plots at publication quality (single-column CHI size,
PDF + PNG). Aggregation matches the eval pipeline exactly: one point per task
condition (trial id), flat pooled mean across all participants and rounds;
linear fit MT = a + b*ID per source. Fitts uses the *aligned* kinematic
movement time (final target entry - movement onset), which strips human
reaction/click latency and model dwell; steering uses the full trial time.

Usage:  python paper/plot_laws.py
Writes: paper/figures/steering_law.{pdf,png}, paper/figures/fitts_law.{pdf,png}
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO / "results-cluster-10p" / "eval-main-pooled8-local"
OUT_DIR = Path(__file__).resolve().parent / "figures"

# Okabe-Ito blue / vermillion — CVD-validated pair (ΔE 21.9 protan, 31.2 normal).
COLORS = {"Human": "#0072B2", "Model": "#D55E00"}
MARKERS = {"Human": "o", "Model": "^"}  # shape = secondary identity encoding

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,  # embed TrueType so text stays editable/searchable
    "ps.fonttype": 42,
})


def load_condition_means(csv_path, mt_col):
    """Collapse per-round rows to one (ID, mean MT) point per (source, tid).

    Flat pooled mean across participants and rounds — identical to the eval
    pipeline's steering_law_plot / fitts _collapse_by_tid aggregation.
    """
    per = defaultdict(lambda: ([], []))  # (src, tid) -> (IDs, MTs)
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r.get("timed_out") == "True":
                continue
            mt = r.get(mt_col)
            if not mt:
                continue
            src = "Model" if r["source"] == "Simulator" else "Human"
            ids, mts = per[(src, r["tid"])]
            ids.append(float(r["ID"]))
            mts.append(float(mt))
    out = {"Human": ([], []), "Model": ([], [])}
    for (src, tid), (ids, mts) in per.items():
        out[src][0].append(float(np.mean(ids)))
        out[src][1].append(float(np.mean(mts)))
    return {s: (np.array(x), np.array(y)) for s, (x, y) in out.items()}


def fit_line(x, y):
    b, a = np.polyfit(x, y, 1)
    r2 = np.corrcoef(x, y)[0, 1] ** 2
    return a, b, r2


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.88", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5)


def draw_law(ax, data, xlabel, unit):
    fits = {}
    for src in ("Human", "Model"):
        x, y = data[src]
        a, b, r2 = fit_line(x, y)
        fits[src] = (a, b, r2)
        xs = np.linspace(x.min(), x.max(), 2)
        ax.plot(xs, a + b * xs, color=COLORS[src], linewidth=1.2, zorder=2)
        ax.scatter(x, y, s=14, marker=MARKERS[src], color=COLORS[src],
                   linewidths=0, alpha=0.85, zorder=3, label=src)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Movement time (s)")
    ax.set_ylim(bottom=0)
    style_axes(ax)

    lines = [
        f"{src}: MT = {fits[src][0]:.2f} + {fits[src][1]:.3f} {unit}"
        f"  ($R^2$ = {fits[src][2]:.2f})"
        for src in ("Human", "Model")
    ]
    ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, fontsize=7,
            va="top", ha="left", linespacing=1.5)
    ax.legend(loc="lower right", frameon=False, handletextpad=0.2,
              borderaxespad=0.2)
    return fits


def save(fig, stem):
    OUT_DIR.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        p = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"saved {p.relative_to(REPO)}")
    plt.close(fig)


def main():
    # --- Steering law: MT vs ID = L/W, full trial time ---
    steer = load_condition_means(EVAL_DIR / "Steering" / "steering_results.csv", "MT_s")
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    fits = draw_law(ax, steer, "Index of difficulty $L/W$", "ID")
    save(fig, "steering_law")
    for s, (a, b, r2) in fits.items():
        print(f"  steering {s}: MT = {a:.3f} + {b:.4f} ID, R2 = {r2:.3f}, "
              f"n = {len(steer[s][0])} conditions")

    # --- Fitts' law: aligned kinematic MT vs ID (bits) ---
    fitts = load_condition_means(EVAL_DIR / "Fitts" / "fitts_results.csv", "MT_kin_s")
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    fits = draw_law(ax, fitts, "Index of difficulty (bits)", "ID")
    save(fig, "fitts_law")
    for s, (a, b, r2) in fits.items():
        print(f"  fitts {s}: MT = {a:.3f} + {b:.4f} ID, R2 = {r2:.3f}, "
              f"n = {len(fitts[s][0])} conditions")

    # Cross-check the Fitts refit against the eval pipeline's stored regression.
    ref = json.loads((EVAL_DIR / "Fitts" / "fitts_regression.json").read_text())
    for src, key in (("Human", "human"), ("Model", "model")):
        a, b, r2 = fits[src]
        ra, rb, rr2 = (ref["aligned"][key][k] for k in
                       ("a_intercept", "b_slope_s_per_bit", "r_squared"))
        ok = abs(a - ra) < 1e-3 and abs(b - rb) < 1e-3 and abs(r2 - rr2) < 1e-3
        print(f"  fitts {src} vs fitts_regression.json[aligned]: "
              f"{'MATCH' if ok else f'MISMATCH (json: a={ra}, b={rb}, R2={rr2})'}")


if __name__ == "__main__":
    main()

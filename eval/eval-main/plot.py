"""
Law-specific regression plots for the eval-new-data pipeline.

Trajectory / time-vs-speed / speed-vs-progress plots are NOT redefined here
— those are reused directly from eval/utils/plot_utils.py (plot_experiment_results,
plot_enhanced_speed_profiles, plot_speeds_vs_progress_enhanced) by run_eval.py.

This module holds the three per-law "MT vs difficulty" regression plots:
  - steering_law_plot      : migrated from figure5_steering_law (eval/*/plot.py,
                              17 byte-identical copies), re-pointed at a
                              data-driven trial_id set instead of a hardcoded one.
  - plot_fitts_regression   : ported from eval-fitts-continuous-well/run_eval.py.
  - plot_id4scs_regression  : net new, per CLAUDE.md's ID4SCS composite-ID plot.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def steering_law_plot(rows, output_path, title="Steering Law: MT vs Index of Difficulty (L/W)"):
    """
    MT vs ID (=L/W) scatter + per-source linear fit (steering law: MT = a + b*ID).

    Args:
        rows: list of dicts {"source": "Human"/"Simulator", "tid": int,
              "ID": float, "MT_s": float} — one row per (tid, round).
        output_path: path to save the .pdf to.
    """
    from collections import defaultdict

    by_source_tid = defaultdict(list)
    for r in rows:
        by_source_tid[(r["source"], r["tid"])].append(r["MT_s"])

    mean_by_source = defaultdict(dict)
    id_by_tid = {}
    for r in rows:
        id_by_tid[r["tid"]] = r["ID"]
    for (source, tid), mts in by_source_tid.items():
        mean_by_source[source][tid] = float(np.mean(mts))

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"Human": "#1f77b4", "Simulator": "#ff7f0e"}

    def _fit_line(ids, mts):
        ids_arr = np.array(ids)
        mts_arr = np.array(mts)
        if len(ids_arr) < 3:
            return None
        coeffs = np.polyfit(ids_arr, mts_arr, 1)
        r = np.corrcoef(ids_arr, mts_arr)[0, 1]
        return coeffs, r

    results = {}
    for source in ("Human", "Simulator"):
        tids = sorted(mean_by_source[source].keys())
        ids = [id_by_tid[t] for t in tids]
        mts = [mean_by_source[source][t] for t in tids]
        if not ids:
            continue
        color = colors[source]
        ax.scatter(ids, mts, color=color, s=60, alpha=0.8, linewidths=0,
                   zorder=3, label=source)
        fit = _fit_line(ids, mts)
        if fit:
            coeffs, r = fit
            x_range = np.linspace(min(ids), max(ids), 100)
            ax.plot(x_range, np.polyval(coeffs, x_range), color=color,
                    linewidth=2, alpha=0.7, zorder=4)
            results[source] = (coeffs, r)

    ax.set_xlabel("Index of Difficulty (L / W)", fontsize=12)
    ax.set_ylabel("Movement Time (s)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)

    ann_parts = [f"{source}: R²={r**2:.3f}" for source, (_, r) in results.items()]
    if ann_parts:
        ax.text(0.02, 0.98, "\n".join(ann_parts), transform=ax.transAxes,
                fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.legend(loc="upper left", fontsize=9, bbox_to_anchor=(0.02, 0.85))
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {output_path}")

    print("\n  === Steering Law Summary ===")
    for source, (coeffs, r) in results.items():
        print(f"  {source}: MT = {coeffs[0]:.4f} * ID + {coeffs[1]:.4f}  (R²={r**2:.3f})")


def plot_fitts_regression(model_rows, human_rows, model_reg, human_reg, output_path):
    """
    ID (bits) vs MT (s) scatter for every individual model and human run,
    with each side's fitted Fitts' line (MT = a + b*ID) overlaid.

    model_reg/human_reg: dicts with "a_intercept", "b_slope_s_per_bit",
    "r_squared" (may be {} if under-determined).
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    palette = {"Human": "tab:blue", "Simulator": "tab:orange"}

    def _scatter(rows, label):
        pts = [(r["ID"], r["MT_s"]) for r in rows if r["MT_s"] is not None]
        if not pts:
            return
        ids, mts = zip(*pts)
        ax.scatter(ids, mts, s=22, alpha=0.5, color=palette[label],
                   edgecolors="none", label=f"{label} (n={len(pts)})")

    _scatter(human_rows, "Human")
    _scatter(model_rows, "Simulator")

    def _line(reg, rows, label):
        if not reg:
            return
        ids = [r["ID"] for r in rows if r["MT_s"] is not None]
        if not ids:
            return
        a, b = reg["a_intercept"], reg["b_slope_s_per_bit"]
        x = np.linspace(min(ids), max(ids), 100)
        y = a + b * x
        r2 = reg.get("r_squared")
        sign = "+" if b >= 0 else "-"
        eq = f"MT={a:.2f}{sign}{abs(b):.2f}·ID"
        r2_str = f", R²={r2:.2f}" if r2 is not None else ""
        ax.plot(x, y, color=palette[label], linewidth=2,
                label=f"{label} fit: {eq}{r2_str}")

    _line(human_reg, human_rows, "Human")
    _line(model_reg, model_rows, "Simulator")

    ax.set_xlabel("ID (bits)", fontsize=11)
    ax.set_ylabel("MT (s)", fontsize=11)
    ax.set_title("Fitts' Law: MT vs. ID — Simulator vs. Human", fontsize=12)
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_id4scs_combined(side_rows, side_fits, output_path):
    """
    Aggregate ID4SCS plot across BOTH directions on a SHARED difficulty axis:
    composite ID = b*S1 + c*C + d*S2 computed with the HUMAN coefficients for
    both sides, so each condition has one x and the simulator's deviation
    from the human appears purely as vertical displacement. The human line is
    its own-law y = x + a; the simulator line is a least-squares fit of its
    MT against the human composite. Direction is encoded in the marker:
    o = wide-to-narrow, ^ = narrow-to-wide.

    Args:
        side_rows: {"human"/"model": [{"S1","C","S2","MT","direction"}, ...]}
                   with direction in {"wide_to_narrow", "narrow_to_wide"}.
        side_fits: {"human"/"model": dict from stats.fit_id4scs (a..d, r_squared)}.
        output_path: path to save the figure to.
    """
    colors = {"human": "tab:blue", "model": "tab:orange"}
    names = {"human": "Human", "model": "Simulator"}
    markers = {"wide_to_narrow": "o", "narrow_to_wide": "^"}

    ref = side_fits.get("human") or side_fits.get("model")
    if not ref:
        print(f"  Skipping {output_path}: no ID4SCS fit to define the composite axis")
        return

    def comp(r):
        return ref["b"] * r["S1"] + ref["c"] * r["C"] + ref["d"] * r["S2"]

    fig, ax = plt.subplots(figsize=(7, 6))
    ann = [f"axis: human fit  MT = {ref['a']:.2f} + {ref['b']:.3f}·S1 "
           f"{ref['c']:+.3f}·C {ref['d']:+.3f}·S2   (R²={ref['r_squared']:.2f})"]
    for side in ("human", "model"):
        rows = side_rows.get(side) or []
        if not rows:
            continue
        for direction, mk in markers.items():
            pts = [(comp(r), r["MT"]) for r in rows if r["direction"] == direction]
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, s=55, marker=mk, color=colors[side], alpha=0.85,
                       edgecolors="none", zorder=3,
                       label=f"{names[side]} — {direction.replace('_to_', ' → ')}")
        cs = np.array([comp(r) for r in rows])
        x = np.linspace(cs.min(), cs.max(), 100)
        if side == "human":
            ax.plot(x, x + ref["a"], color=colors[side], linewidth=2, alpha=0.8,
                    zorder=2)
        else:
            mts = np.array([r["MT"] for r in rows])
            if len(rows) >= 3 and np.ptp(cs) > 1e-9:
                slope, icpt = np.polyfit(cs, mts, 1)
                pred = icpt + slope * cs
                ss_tot = float(np.sum((mts - mts.mean()) ** 2))
                r_sq = 1.0 - float(np.sum((mts - pred) ** 2)) / ss_tot if ss_tot else 0.0
                ax.plot(x, icpt + slope * x, color=colors[side], linewidth=2,
                        alpha=0.8, zorder=2)
                ann.append(f"{names[side]} vs human composite: "
                           f"MT = {icpt:.2f} + {slope:.2f}·ID   (R²={r_sq:.2f})")

    ax.set_xlabel("Composite ID (b·S1 + c·C + d·S2, human coefficients)", fontsize=11)
    ax.set_ylabel("Observed MT (s)", fontsize=11)
    ax.set_title("ID4SCS: both directions, pooled per-condition means", fontsize=12,
                 fontweight="bold")
    if ann:
        ax.text(0.02, 0.98, "\n".join(ann), transform=ax.transAxes, fontsize=8.5,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_id4scs_regression(rows, reg, output_path, direction_label):
    """
    Composite-ID diagnostic plot for the ID4SCS model:
        Composite_ID_i = b*S1_i + c*C_i + d*S2_i   (using the FITTED b,c,d)
    plotted against observed MT, with a fixed-slope-1 reference line y = x + a.

    Args:
        rows: list of dicts {"S1", "C", "S2", "MT"}.
        reg: dict from stats.fit_id4scs (a, b, c, d, r_squared, n).
        direction_label: e.g. "Narrow-to-Wide" / "Wide-to-Narrow", for the title.
    """
    if not reg:
        print(f"  Skipping {output_path.name}: not enough rows to fit ID4SCS regression")
        return

    composite_id = [reg["b"] * r["S1"] + reg["c"] * r["C"] + reg["d"] * r["S2"] for r in rows]
    observed_mt = [r["MT"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(composite_id, observed_mt, s=30, alpha=0.7, color="tab:purple", zorder=3)

    x = np.linspace(min(composite_id), max(composite_id), 100)
    ax.plot(x, x + reg["a"], color="tab:red", linewidth=2,
            label=f"y = x + {reg['a']:.3f}  (R²={reg['r_squared']:.3f}, n={reg['n']})")

    ax.set_xlabel("Composite ID (b·S1 + c·C + d·S2)", fontsize=11)
    ax.set_ylabel("Observed MT (s)", fontsize=11)
    ax.set_title(f"ID4SCS: {direction_label}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")

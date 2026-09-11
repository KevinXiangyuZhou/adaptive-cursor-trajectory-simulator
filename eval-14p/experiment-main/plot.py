"""
Generate evaluation figures for the paper.

Figure 1: Human vs Model completion time variability — upper subplot shows human MT
          box plots across tasks (each box = distribution over participants), lower
          subplot shows model MT box plots (models fitted on different participants).
Figure 2: Scatter plots of Human MT vs Model MT (and speed), per participant × trial.
          Each dot = one participant's average on one trial. Colored by participant.
          Includes baseline comparison.

Reads per-participant results from results/participant_*/trial_*/results_summary.json.
Saves figures to eval/experiment-main/figures/.

Usage:
    python -m eval.experiment-main.plot
"""

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "hcs_package" / "src"))
RESULTS_DIR = SCRIPT_DIR / "results"
FIGURES_DIR = SCRIPT_DIR / "figures"

# Trial conditions (mirrored from run_fitting)
TRIAL_CONDITIONS = {
    1:  {"type": "sigmoidal", "width": 0.01, "curvature": 0.025, "label": "Sin W=10"},
    2:  {"type": "sigmoidal", "width": 0.02, "curvature": 0.025, "label": "Sin W=20"},
    3:  {"type": "sigmoidal", "width": 0.03, "curvature": 0.025, "label": "Sin W=30"},
    4:  {"type": "sigmoidal", "width": 0.04, "curvature": 0.025, "label": "Sin W=40"},
    5:  {"type": "sigmoidal", "width": 0.05, "curvature": 0.025, "label": "Sin W=50"},
    6:  {"type": "corner",    "width": 0.01, "num_corners": 2, "corner_offset": 0.1, "label": "Cor W=10"},
    7:  {"type": "corner",    "width": 0.02, "num_corners": 2, "corner_offset": 0.1, "label": "Cor W=20"},
    8:  {"type": "corner",    "width": 0.03, "num_corners": 2, "corner_offset": 0.1, "label": "Cor W=30"},
    9:  {"type": "corner",    "width": 0.04, "num_corners": 2, "corner_offset": 0.1, "label": "Cor W=40"},
    10: {"type": "corner",    "width": 0.05, "num_corners": 2, "corner_offset": 0.1, "label": "Cor W=50"},
    11: {"type": "straight",  "width": 0.01, "curvature": 0.0, "label": "Str W=10"},
    12: {"type": "straight",  "width": 0.02, "curvature": 0.0, "label": "Str W=20"},
    13: {"type": "straight",  "width": 0.03, "curvature": 0.0, "label": "Str W=30"},
    14: {"type": "straight",  "width": 0.04, "curvature": 0.0, "label": "Str W=40"},
    15: {"type": "straight",  "width": 0.05, "curvature": 0.0, "label": "Str W=50"},
    16: {"type": "gentle_sin","width": 0.01, "curvature": 0.015, "label": "Gen W=10"},
    17: {"type": "gentle_sin","width": 0.02, "curvature": 0.015, "label": "Gen W=20"},
    18: {"type": "gentle_sin","width": 0.03, "curvature": 0.015, "label": "Gen W=30"},
    19: {"type": "gentle_sin","width": 0.04, "curvature": 0.015, "label": "Gen W=40"},
    20: {"type": "gentle_sin","width": 0.05, "curvature": 0.015, "label": "Gen W=50"},
    21: {"type": "sharp_sin", "width": 0.01, "curvature": 0.05, "label": "Shp W=10"},
    22: {"type": "sharp_sin", "width": 0.02, "curvature": 0.05, "label": "Shp W=20"},
    23: {"type": "sharp_sin", "width": 0.03, "curvature": 0.05, "label": "Shp W=30"},
    24: {"type": "sharp_sin", "width": 0.04, "curvature": 0.05, "label": "Shp W=40"},
    25: {"type": "sharp_sin", "width": 0.05, "curvature": 0.05, "label": "Shp W=50"},
}

TRAIN_WIDTHS = {0.01, 0.03, 0.05}
TEST_WIDTHS  = {0.02, 0.04}
TRAIN_TIDS = {tid for tid, c in TRIAL_CONDITIONS.items() if c["width"] in TRAIN_WIDTHS}
TEST_TIDS  = {tid for tid, c in TRIAL_CONDITIONS.items() if c["width"] in TEST_WIDTHS}

# Trial ordering: group by type, then by width
TYPE_ORDER = ["sigmoidal", "corner", "straight", "gentle_sin", "sharp_sin"]
TYPE_LABELS = {
    "sigmoidal": "Sinusoidal", "corner": "Corner", "straight": "Straight",
    "gentle_sin": "Gentle Sin", "sharp_sin": "Sharp Sin",
}
TIDS_ORDERED = []
for t in TYPE_ORDER:
    TIDS_ORDERED.extend(
        sorted([tid for tid, c in TRIAL_CONDITIONS.items() if c["type"] == t],
               key=lambda tid: TRIAL_CONDITIONS[tid]["width"])
    )

TYPE_COLORS = {
    "sigmoidal": "#1f77b4", "corner": "#ff7f0e", "straight": "#2ca02c",
    "gentle_sin": "#d62728", "sharp_sin": "#9467bd",
}


def load_all_results(filter_baseline=True):
    """Load per-participant, per-trial results from JSON files.

    Args:
        filter_baseline: If True, filter out timed-out and diverged baseline runs.

    Returns dict: {pid: {tid: {"human_mt", "human_speed", "model_mt", "model_speed",
                                "baseline_mt", "baseline_speed", "n_human", "n_model"}}}
    """
    data = {}
    for pdir in sorted(RESULTS_DIR.glob("participant_*")):
        if not pdir.is_dir():
            continue
        pid = pdir.name.replace("participant_", "")
        data[pid] = {}
        for tid in range(1, 26):
            summary_path = pdir / f"trial_{tid}" / "results_summary.json"
            if not summary_path.exists():
                continue
            with open(summary_path) as f:
                s = json.load(f)
            h_rounds = s["human"].get("rounds", [])
            m_runs_all = s["model"].get("runs", [])
            h_cts = [r["completion_time"] for r in h_rounds]
            # Use n_valid_runs and avg from summary (already filtered)
            model_mt = s["model"]["avg_completion_time"]
            model_speed = s["model"]["avg_speed"]
            n_valid = s["model"].get("n_valid_runs", s["model"]["n_runs"])
            # For min/max, exclude outlier runs (same bounds as _process_trial)
            m_cts = []
            for ri, r in enumerate(m_runs_all):
                if r["completion_time"] >= 39.5:
                    print(f"  Filtered model: {pid} trial {tid} run {ri} "
                          f"— CT={r['completion_time']:.2f}s (hit max steps)")
                    continue
                m_cts.append(r["completion_time"])
            if len(m_cts) < len(m_runs_all):
                n_valid = len(m_cts)
            entry = {
                "human_mt": s["human"]["avg_completion_time"],
                "human_speed": s["human"]["avg_speed"],
                "human_mt_min": min(h_cts) if h_cts else s["human"]["avg_completion_time"],
                "human_mt_max": max(h_cts) if h_cts else s["human"]["avg_completion_time"],
                "model_mt": model_mt,
                "model_speed": model_speed,
                "model_mt_min": min(m_cts) if m_cts else model_mt,
                "model_mt_max": max(m_cts) if m_cts else model_mt,
                "n_human": s["human"]["n_rounds"],
                "n_model": n_valid,
            }
            if "baseline" in s and s["baseline"]:
                bl_runs = s["baseline"].get("runs", [])
                if filter_baseline:
                    MAX_BASELINE_CT = 39.5   # 800 steps * 0.05s = 40s
                    DIVERGE_LIMIT = 1.0      # meters
                    bl_runs_ok = []
                    for ri, r in enumerate(bl_runs):
                        if r["completion_time"] >= MAX_BASELINE_CT:
                            print(f"  Filtered baseline: {pid} trial {tid} run {ri} "
                                  f"— CT={r['completion_time']:.2f}s (hit max steps)")
                            continue
                        traj = r.get("trajectory", [])
                        if traj:
                            lp = traj[-1]
                            if abs(lp[0]) > DIVERGE_LIMIT or abs(lp[1]) > DIVERGE_LIMIT:
                                print(f"  Filtered baseline: {pid} trial {tid} run {ri} "
                                      f"— diverged (end=({lp[0]:.2f},{lp[1]:.2f}))")
                                continue
                        bl_runs_ok.append(r)
                else:
                    bl_runs_ok = bl_runs
                if bl_runs_ok:
                    bl_cts = [r["completion_time"] for r in bl_runs_ok]
                    bl_speeds = [r["avg_speed"] for r in bl_runs_ok]
                    entry["baseline_mt"] = float(np.mean(bl_cts))
                    entry["baseline_speed"] = float(np.mean(bl_speeds))
                    entry["baseline_mt_min"] = min(bl_cts)
                    entry["baseline_mt_max"] = max(bl_cts)
                    entry["n_baseline"] = len(bl_runs_ok)
                    n_filtered = len(bl_runs) - len(bl_runs_ok)
                    if n_filtered > 0:
                        entry["baseline_filtered"] = n_filtered
            data[pid][tid] = entry
    return data


def _make_boxplot(ax, box_data, labels, colors, ylabel, title):
    """Helper: draw a colored box plot on the given axes."""
    positions = np.arange(1, len(box_data) + 1)
    bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,
                    showfliers=True, flierprops=dict(markersize=3))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Type group separators
    prev_type = None
    for i, tid in enumerate(TIDS_ORDERED):
        t = TRIAL_CONDITIONS[tid]["type"]
        if prev_type is not None and t != prev_type:
            ax.axvline(i + 0.5, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
        prev_type = t
    return bp


def figure1_human_vs_model_variability(data):
    """Upper: human MT box plots across tasks. Lower: model MT box plots across tasks."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    human_mt_data = []
    model_mt_data = []
    labels = []
    colors = []

    for tid in TIDS_ORDERED:
        cond = TRIAL_CONDITIONS[tid]
        human_mts = [data[pid][tid]["human_mt"] for pid in data if tid in data[pid]]
        model_mts = [data[pid][tid]["model_mt"] for pid in data if tid in data[pid]]
        human_mt_data.append(human_mts)
        model_mt_data.append(model_mts)
        labels.append(cond["label"])
        colors.append(TYPE_COLORS[cond["type"]])

    _make_boxplot(axes[0], human_mt_data, labels, colors,
                  "Completion Time (s)", "Human Completion Time Across Participants")
    _make_boxplot(axes[1], model_mt_data, labels, colors,
                  "Completion Time (s)", "Model Completion Time (fitted per participant)")

    # X-axis labels on bottom subplot only
    positions = np.arange(1, len(TIDS_ORDERED) + 1)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    # Match y-axis range across both subplots for visual comparison
    y_max = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    axes[0].set_ylim(bottom=0, top=y_max)
    axes[1].set_ylim(bottom=0, top=y_max)

    # Legend
    legend_patches = [Patch(facecolor=TYPE_COLORS[t], alpha=0.6, label=TYPE_LABELS[t])
                      for t in TYPE_ORDER]
    axes[0].legend(handles=legend_patches, loc="upper right", fontsize=9)

    plt.tight_layout()
    out_path = FIGURES_DIR / "human_vs_model_variability.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Print summary stats
    print("\n=== Completion Time Variability Summary ===")
    print(f"{'Trial':<15} {'H min':>6} {'H max':>6} {'H IQR':>6} │ {'M min':>6} {'M max':>6} {'M IQR':>6}  N")
    for i, tid in enumerate(TIDS_ORDERED):
        h = np.array(human_mt_data[i])
        m = np.array(model_mt_data[i])
        if len(h) == 0:
            continue
        hq1, hq3 = np.percentile(h, [25, 75])
        mq1, mq3 = np.percentile(m, [25, 75])
        print(f"{labels[i]:<15} {h.min():6.2f} {h.max():6.2f} {hq3-hq1:6.2f} │ "
              f"{m.min():6.2f} {m.max():6.2f} {mq3-mq1:6.2f}  {len(h)}")

    # Variability ratio
    h_ratios, m_ratios = [], []
    for i, tid in enumerate(TIDS_ORDERED):
        h = [data[pid][tid]["human_mt"] for pid in data if tid in data[pid]]
        m = [data[pid][tid]["model_mt"] for pid in data if tid in data[pid]]
        if len(h) >= 2:
            h_ratios.append(max(h) / min(h))
            m_ratios.append(max(m) / min(m))
    print(f"\nMax/min MT ratio — Human: mean={np.mean(h_ratios):.2f}x, "
          f"Model: mean={np.mean(m_ratios):.2f}x")
    print(f"  Human range: {min(h_ratios):.2f}x – {max(h_ratios):.2f}x")
    print(f"  Model range: {min(m_ratios):.2f}x – {max(m_ratios):.2f}x")


def figure2_scatter(data):
    """Scatter: Human MT vs Model MT, colored by participant. Includes baseline."""
    pids = sorted(data.keys())
    n_pids = len(pids)
    cmap = matplotlib.colormaps.get_cmap("tab20").resampled(n_pids)
    pid_colors = {pid: cmap(i) for i, pid in enumerate(pids)}
    pid_short = {pid: f"P{i+1}" for i, pid in enumerate(pids)}

    has_baseline = any(
        "baseline_mt" in data[pid].get(tid, {})
        for pid in pids for tid in range(1, 26)
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Panel A: Completion Time ---
    ax = axes[0]
    human_mts, model_mts, baseline_mts = [], [], []
    human_mts_bl, human_speeds_bl = [], []
    baseline_speeds = []

    for pid in pids:
        for tid in TIDS_ORDERED:
            if tid not in data[pid]:
                continue
            e = data[pid][tid]
            hmt, mmt = e["human_mt"], e["model_mt"]
            human_mts.append(hmt)
            model_mts.append(mmt)
            marker = "o" if tid in TRAIN_TIDS else "^"
            ax.scatter(hmt, mmt, color=pid_colors[pid], marker=marker,
                       s=20, alpha=0.7, linewidths=0.3, edgecolors="k")
            if has_baseline and "baseline_mt" in e:
                human_mts_bl.append(hmt)
                baseline_mts.append(e["baseline_mt"])

    all_mt = human_mts + model_mts + baseline_mts
    lo, hi = min(all_mt) * 0.8, max(all_mt) * 1.1
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.5, label="y = x")

    if has_baseline and baseline_mts:
        ax.scatter(human_mts_bl, baseline_mts, color="gray", marker="x",
                   s=20, alpha=0.5, linewidths=0.8, label="Baseline")

    ax.set_xlabel("Human Completion Time (s)", fontsize=11)
    ax.set_ylabel("Model Completion Time (s)", fontsize=11)
    ax.set_title("Completion Time: Human vs Model", fontsize=12, fontweight="bold")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    # --- Panel B: Speed ---
    ax = axes[1]
    human_speeds, model_speeds = [], []

    for pid in pids:
        for tid in TIDS_ORDERED:
            if tid not in data[pid]:
                continue
            e = data[pid][tid]
            hs, ms = e["human_speed"] * 1000, e["model_speed"] * 1000  # mm/s
            human_speeds.append(hs)
            model_speeds.append(ms)
            marker = "o" if tid in TRAIN_TIDS else "^"
            ax.scatter(hs, ms, color=pid_colors[pid], marker=marker,
                       s=20, alpha=0.7, linewidths=0.3, edgecolors="k")
            if has_baseline and "baseline_speed" in e:
                human_speeds_bl.append(hs)
                baseline_speeds.append(e["baseline_speed"] * 1000)

    all_spd = human_speeds + model_speeds + baseline_speeds
    lo, hi = min(all_spd) * 0.8, max(all_spd) * 1.1
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.5, label="y = x")

    if has_baseline and baseline_speeds:
        ax.scatter(human_speeds_bl, baseline_speeds, color="gray", marker="x",
                   s=20, alpha=0.5, linewidths=0.8, label="Baseline")

    ax.set_xlabel("Human Speed (mm/s)", fontsize=11)
    ax.set_ylabel("Model Speed (mm/s)", fontsize=11)
    ax.set_title("Average Speed: Human vs Model", fontsize=12, fontweight="bold")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    # Combined legend
    handles = []
    for pid in pids:
        handles.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=pid_colors[pid], markersize=6,
                              label=pid_short[pid]))
    handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                          markersize=6, label="Train trial"))
    handles.append(Line2D([0], [0], marker="^", color="w", markerfacecolor="gray",
                          markersize=6, label="Test trial"))
    if has_baseline:
        handles.append(Line2D([0], [0], marker="x", color="gray",
                              markersize=6, linestyle="", label="Baseline"))
    handles.append(Line2D([0], [0], linestyle="--", color="k", alpha=0.5, label="y = x"))

    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
               fontsize=8, frameon=True)

    plt.tight_layout()
    out_path = FIGURES_DIR / "human_vs_model_scatter.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out_path}")

    # Print correlation stats
    human_mts_arr = np.array(human_mts)
    model_mts_arr = np.array(model_mts)
    human_spd_arr = np.array(human_speeds)
    model_spd_arr = np.array(model_speeds)

    r_mt = np.corrcoef(human_mts_arr, model_mts_arr)[0, 1]
    r_spd = np.corrcoef(human_spd_arr, model_spd_arr)[0, 1]
    mae_mt = np.mean(np.abs(human_mts_arr - model_mts_arr))
    mae_spd = np.mean(np.abs(human_spd_arr - model_spd_arr))

    print("\n=== Human vs Model Correlation ===")
    print(f"Completion Time: r={r_mt:.3f}, MAE={mae_mt:.2f}s  (N={len(human_mts)})")
    print(f"Average Speed:   r={r_spd:.3f}, MAE={mae_spd:.1f}mm/s  (N={len(human_speeds)})")

    # Train vs test breakdown
    for label, tid_set in [("Train", TRAIN_TIDS), ("Test", TEST_TIDS)]:
        mt_h, mt_m, spd_h, spd_m = [], [], [], []
        for pid in pids:
            for tid in tid_set:
                if tid in data[pid]:
                    e = data[pid][tid]
                    mt_h.append(e["human_mt"])
                    mt_m.append(e["model_mt"])
                    spd_h.append(e["human_speed"] * 1000)
                    spd_m.append(e["model_speed"] * 1000)
        if len(mt_h) > 2:
            r = np.corrcoef(mt_h, mt_m)[0, 1]
            mae = np.mean(np.abs(np.array(mt_h) - np.array(mt_m)))
            r_s = np.corrcoef(spd_h, spd_m)[0, 1]
            mae_s = np.mean(np.abs(np.array(spd_h) - np.array(spd_m)))
            print(f"  {label:5s} MT: r={r:.3f}, MAE={mae:.2f}s  |  "
                  f"Speed: r={r_s:.3f}, MAE={mae_s:.1f}mm/s  (N={len(mt_h)})")

    if has_baseline and baseline_mts:
        r_bl = np.corrcoef(human_mts_bl, baseline_mts)[0, 1]
        mae_bl = np.mean(np.abs(np.array(human_mts_bl) - np.array(baseline_mts)))
        print(f"\nBaseline MT: r={r_bl:.3f}, MAE={mae_bl:.2f}s  (N={len(baseline_mts)})")
        if baseline_speeds:
            r_bl_s = np.corrcoef(human_speeds_bl[:len(baseline_speeds)],
                                 baseline_speeds)[0, 1]
            mae_bl_s = np.mean(np.abs(
                np.array(human_speeds_bl[:len(baseline_speeds)]) -
                np.array(baseline_speeds)))
            print(f"Baseline Speed: r={r_bl_s:.3f}, MAE={mae_bl_s:.1f}mm/s  "
                  f"(N={len(baseline_speeds)})")


def _broken_y_transform(val, break_lo=20.0, break_hi=30.0, compress=0.2):
    """Map data value to display value with compressed region above break_lo.

    Below break_lo: identity (1:1 mapping).
    Above break_lo: compressed by `compress` factor.
    """
    if val <= break_lo:
        return val
    return break_lo + (val - break_lo) * compress


def figure3_paired_mt_pertask(data):
    """Per-task paired strip plot of raw MT showing individual differences.

    Two rows with broken y-axis: linear 0–20s, compressed 20–30s.
    Points above MT_CLIP are clipped.
    """
    MT_CLIP = 30.0
    BREAK_LO = 20.0
    COMPRESS = 0.2  # 20-30s range displayed at 20% of normal height

    pids = sorted(data.keys())
    n_pids = len(pids)
    cmap = matplotlib.colormaps.get_cmap("tab20").resampled(n_pids)
    pid_colors = {pid: cmap(i) for i, pid in enumerate(pids)}

    has_baseline = any(
        "baseline_mt" in data[pid].get(tid, {})
        for pid in pids for tid in range(1, 26)
    )

    # Two rows: Row 1 = Sin + Corner + Straight, Row 2 = Gentle Sin + Sharp Sin
    row1_types = ["sigmoidal", "corner", "straight"]
    row2_types = ["gentle_sin", "sharp_sin"]
    row1_tids = [tid for tid in TIDS_ORDERED
                 if TRIAL_CONDITIONS[tid]["type"] in row1_types]
    row2_tids = [tid for tid in TIDS_ORDERED
                 if TRIAL_CONDITIONS[tid]["type"] in row2_types]
    row_tids_list = [row1_tids, row2_tids]

    def ytransform(v):
        return _broken_y_transform(min(v, MT_CLIP), BREAK_LO, MT_CLIP, COMPRESS)

    # Display limits
    y_disp_max = ytransform(MT_CLIP) + 0.5

    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 1],
                           width_ratios=[1, 1, 1])
    # Row 1: single axes spanning all 3 columns
    ax_top = fig.add_subplot(gs[0, :])
    # Row 2: data axes spanning 2 columns, legend in 3rd
    ax_bot = fig.add_subplot(gs[1, 0:2])
    ax_legend = fig.add_subplot(gs[1, 2])
    ax_legend.axis("off")

    axes = [ax_top, ax_bot]

    spacing = 0.30
    all_h, all_m, all_b = [], [], []

    for row_idx, (ax, row_tids) in enumerate(zip(axes, row_tids_list)):
        labels = []
        row_type_set = row1_types if row_idx == 0 else row2_types

        for xi, tid in enumerate(row_tids):
            cond = TRIAL_CONDITIONS[tid]
            labels.append(cond["label"])
            x_pos = xi + 1

            if has_baseline:
                x_bl = x_pos - spacing
                x_h  = x_pos
                x_m  = x_pos + spacing
            else:
                x_bl = None
                x_h = x_pos - spacing
                x_m = x_pos + spacing

            for pid in pids:
                if tid not in data[pid]:
                    continue
                e = data[pid][tid]
                hmt_raw = min(e["human_mt"], MT_CLIP)
                mmt_raw = min(e["model_mt"], MT_CLIP)
                hmt = ytransform(hmt_raw)
                mmt = ytransform(mmt_raw)
                all_h.append(hmt_raw)
                all_m.append(mmt_raw)

                c = pid_colors[pid]
                ax.plot([x_h, x_m], [hmt, mmt],
                        color=c, linewidth=0.8, alpha=0.5, linestyle=":", zorder=2)
                ax.scatter(x_h, hmt, color=c, marker="o",
                           s=20, alpha=0.9, linewidths=0, zorder=3)
                ax.scatter(x_m, mmt, color=c, marker="^",
                           s=21, alpha=0.9, linewidths=0, zorder=3)

                if has_baseline and "baseline_mt" in e:
                    bmt_raw = min(e["baseline_mt"], MT_CLIP)
                    bmt = ytransform(bmt_raw)
                    all_b.append(bmt_raw)
                    ax.plot([x_bl, x_h], [bmt, hmt],
                            color=c, linewidth=0.8, alpha=0.5, linestyle=":", zorder=1)
                    ax.scatter(x_bl, bmt, color=c, marker="s",
                               s=17, alpha=0.9, linewidths=0, zorder=3)

        # Type group separators
        prev_type = None
        for i, tid in enumerate(row_tids):
            t = TRIAL_CONDITIONS[tid]["type"]
            if prev_type is not None and t != prev_type:
                ax.axvline(i + 0.5, color="gray", linewidth=0.5,
                           linestyle="--", alpha=0.5)
            prev_type = t

        # Shade background by task type
        for xi, tid in enumerate(row_tids):
            c = TYPE_COLORS[TRIAL_CONDITIONS[tid]["type"]]
            ax.axvspan(xi + 0.5, xi + 1.5, color=c, alpha=0.04, zorder=0)

        # X-axis
        positions = np.arange(1, len(row_tids) + 1)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        x_lo = 0.3 if not has_baseline else 0.3 - spacing
        ax.set_xlim(x_lo, len(row_tids) + 1.1)

        # Y-axis with broken scale
        ax.set_ylim(bottom=0, top=y_disp_max)
        # Tick positions: 0,5,10,15,20 (linear) + 25,30 (compressed)
        tick_vals = [0, 5, 10, 15, 20, 25, 30]
        tick_disp = [ytransform(v) for v in tick_vals]
        ax.set_yticks(tick_disp)
        ax.set_yticklabels([str(v) for v in tick_vals])
        ax.set_ylabel("Completion Time (s)", fontsize=11)

        # Draw break indicator
        break_disp = ytransform(BREAK_LO)
        ax.axhline(break_disp, color="gray", linewidth=0.5,
                   linestyle="-", alpha=0.3)

        # Per-task-group SD annotations
        group_end_idx = {}
        for xi, tid in enumerate(row_tids):
            ttype = TRIAL_CONDITIONS[tid]["type"]
            group_end_idx[ttype] = xi

        for ttype in row_type_set:
            tids_of_type = [tid for tid in row_tids
                            if TRIAL_CONDITIONS[tid]["type"] == ttype]
            g_h, g_m, g_b = [], [], []
            for tid in tids_of_type:
                for pid in pids:
                    if tid not in data[pid]:
                        continue
                    e = data[pid][tid]
                    g_h.append(e["human_mt"])
                    g_m.append(e["model_mt"])
                    if "baseline_mt" in e:
                        g_b.append(e["baseline_mt"])

            xi_end = group_end_idx[ttype]
            x_anchor = xi_end + 1 + spacing
            ann = f"Human SD={np.std(g_h):.2f}s\nModel SD={np.std(g_m):.2f}s"
            if g_b:
                ann += f"\nBaseline SD={np.std(g_b):.2f}s"
            ax.text(x_anchor, y_disp_max * 0.95, ann,
                    fontsize=10, verticalalignment="top",
                    horizontalalignment="right",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                              alpha=0.9, edgecolor="gray", linewidth=0.8))

    axes[0].set_title("Individual Comparison: Human vs Models",
                      fontsize=13, fontweight="bold")

    # Legend in the dedicated legend cell (bottom-right)
    handles = []
    if has_baseline:
        handles.append(Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
                              markersize=8, label="Baseline"))
    handles.extend([
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=8, label="Human"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="gray",
               markersize=8, label="Model"),
        Line2D([0], [0], color="gray", linewidth=1.0, linestyle=":", label="Paired"),
    ])
    for t in TYPE_ORDER:
        handles.append(Patch(facecolor=TYPE_COLORS[t], alpha=0.3,
                             label=TYPE_LABELS[t]))
    ax_legend.legend(handles=handles, loc="center", fontsize=15,
                     frameon=True, borderpad=1.5, ncol=2)

    plt.tight_layout()
    out_path = FIGURES_DIR / "paired_mt_individual_differences.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out_path}")

    # Console stats
    from scipy.stats import spearmanr
    all_h = np.array(all_h)
    all_m = np.array(all_m)
    all_b = np.array(all_b)
    r = np.corrcoef(all_h, all_m)[0, 1]
    mae = np.mean(np.abs(all_h - all_m))
    print("\n=== Paired MT Analysis (Individual Differences) ===")
    print(f"Overall: r = {r:.3f}, MAE = {mae:.2f}s (N={len(all_h)})")
    print(f"SD — Human: {np.std(all_h):.2f}s, Model: {np.std(all_m):.2f}s", end="")
    if len(all_b) > 0:
        print(f", Baseline: {np.std(all_b):.2f}s")
    else:
        print()

    rho_list = []
    for tid in TIDS_ORDERED:
        h_vals = [data[pid][tid]["human_mt"] for pid in pids if tid in data[pid]]
        m_vals = [data[pid][tid]["model_mt"] for pid in pids if tid in data[pid]]
        if len(h_vals) >= 5:
            rho, _ = spearmanr(h_vals, m_vals)
            rho_list.append(rho)
    if rho_list:
        print(f"Per-task rank preservation (Spearman rho):")
        print(f"  Mean={np.mean(rho_list):.3f}, "
              f"Median={np.median(rho_list):.3f}, "
              f"Min={np.min(rho_list):.3f}, Max={np.max(rho_list):.3f}")


def figure4_rank_preservation(data):
    """2-panel figure: per-task Spearman rho (top) + simplified strip plot (bottom).

    Panel A: Bar chart of per-task Spearman rho (model vs baseline), colored by
    tunnel type. Immediately shows rank preservation quality per task.
    Panel B: Cleaned-up strip plot — single color for human/model, connecting lines
    show per-participant pairing. No per-participant coloring.
    """
    from scipy.stats import spearmanr

    pids = sorted(data.keys())
    has_baseline = any(
        "baseline_mt" in data[pid].get(tid, {})
        for pid in pids for tid in range(1, 26)
    )

    # --- Compute per-task Spearman rho ---
    model_rhos = []
    baseline_rhos = []
    labels = []
    bar_colors = []

    for tid in TIDS_ORDERED:
        cond = TRIAL_CONDITIONS[tid]
        labels.append(cond["label"])
        bar_colors.append(TYPE_COLORS[cond["type"]])

        human_mts = {pid: data[pid][tid]["human_mt"]
                     for pid in pids if tid in data[pid]}
        if len(human_mts) < 5:
            model_rhos.append(np.nan)
            baseline_rhos.append(np.nan)
            continue

        pids_task = sorted(human_mts.keys())
        h_vals = [human_mts[p] for p in pids_task]
        m_vals = [data[p][tid]["model_mt"] for p in pids_task]
        rho_m, _ = spearmanr(h_vals, m_vals)
        model_rhos.append(rho_m)

        if has_baseline:
            b_vals = [data[p][tid].get("baseline_mt", np.nan) for p in pids_task]
            if (not any(np.isnan(v) for v in b_vals)
                    and np.std(b_vals) > 1e-9):
                rho_b, _ = spearmanr(h_vals, b_vals)
                baseline_rhos.append(rho_b)
            else:
                baseline_rhos.append(np.nan)
        else:
            baseline_rhos.append(np.nan)

    model_rhos = np.array(model_rhos)
    baseline_rhos = np.array(baseline_rhos)
    n_tasks = len(TIDS_ORDERED)
    positions = np.arange(1, n_tasks + 1)

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.5]})

    # ===================== Panel A: Spearman rho bars =====================
    ax = axes[0]
    bar_width = 0.35

    # Baseline bars (behind)
    if has_baseline and not np.all(np.isnan(baseline_rhos)):
        ax.bar(positions - bar_width / 2, baseline_rhos, bar_width,
               color="lightgray", edgecolor="gray", linewidth=0.5,
               label="Baseline", zorder=2)
        # Model bars (front)
        ax.bar(positions + bar_width / 2, model_rhos, bar_width,
               color=bar_colors, edgecolor="k", linewidth=0.5, alpha=0.8,
               label="Our model", zorder=3)
    else:
        ax.bar(positions, model_rhos, 0.6,
               color=bar_colors, edgecolor="k", linewidth=0.5, alpha=0.8,
               label="Our model", zorder=3)

    ax.axhline(0, color="k", linewidth=0.5)
    ax.axhline(0.5, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
    ax.set_ylabel("Spearman rho", fontsize=11)
    ax.set_title("Per-Task Rank Preservation Across Participants",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(-0.3, 1.05)
    ax.grid(axis="y", alpha=0.2)

    # Type group separators
    prev_type = None
    for i, tid in enumerate(TIDS_ORDERED):
        t = TRIAL_CONDITIONS[tid]["type"]
        if prev_type is not None and t != prev_type:
            axes[0].axvline(i + 0.5, color="gray", linewidth=0.5,
                            linestyle="--", alpha=0.5)
            axes[1].axvline(i + 0.5, color="gray", linewidth=0.5,
                            linestyle="--", alpha=0.5)
        prev_type = t

    # Annotate mean rho
    valid = ~np.isnan(model_rhos)
    mean_rho = model_rhos[valid].mean()
    ann = f"Mean rho = {mean_rho:.3f}"
    if has_baseline and not np.all(np.isnan(baseline_rhos)):
        bl_valid = ~np.isnan(baseline_rhos)
        mean_bl = baseline_rhos[bl_valid].mean()
        ann += f"  (baseline: {mean_bl:.3f})"
    ax.text(0.01, 0.95, ann, transform=ax.transAxes, fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Legend for panel A
    handles_a = []
    type_patches = [Patch(facecolor=TYPE_COLORS[t], alpha=0.8, edgecolor="k",
                          linewidth=0.5, label=TYPE_LABELS[t]) for t in TYPE_ORDER]
    handles_a.extend(type_patches)
    if has_baseline and not np.all(np.isnan(baseline_rhos)):
        handles_a.append(Patch(facecolor="lightgray", edgecolor="gray",
                               linewidth=0.5, label="Baseline"))
    ax.legend(handles=handles_a, loc="lower right", fontsize=7, ncol=3)

    # ===================== Panel B: Simplified strip plot =====================
    ax = axes[1]
    jitter = 0.15
    color_h = "#2c7bb6"  # blue for human
    color_m = "#d7191c"  # red for model

    for xi, tid in enumerate(TIDS_ORDERED):
        human_mts = {pid: data[pid][tid]["human_mt"]
                     for pid in pids if tid in data[pid]}
        if len(human_mts) < 3:
            continue
        vals = np.array(list(human_mts.values()))
        mu, sigma = vals.mean(), vals.std(ddof=1)
        if sigma < 1e-9:
            continue

        x_pos = xi + 1

        for pid, hmt in human_mts.items():
            hz = (hmt - mu) / sigma
            mz = (data[pid][tid]["model_mt"] - mu) / sigma

            # Connecting line (gray, subtle)
            ax.plot([x_pos - jitter, x_pos + jitter], [hz, mz],
                    color="gray", linewidth=0.4, alpha=0.4, zorder=2)
            # Human dot
            ax.scatter(x_pos - jitter, hz, color=color_h,
                       s=14, alpha=0.7, linewidths=0.2, edgecolors="k", zorder=3)
            # Model dot
            ax.scatter(x_pos + jitter, mz, color=color_m,
                       s=14, alpha=0.7, linewidths=0.2, edgecolors="k", zorder=3)

    # Shade background by task type
    for xi, tid in enumerate(TIDS_ORDERED):
        c = TYPE_COLORS[TRIAL_CONDITIONS[tid]["type"]]
        ax.axvspan(xi + 0.5, xi + 1.5, color=c, alpha=0.04, zorder=0)

    ax.axhline(0, color="gray", linewidth=0.8, alpha=0.4)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlim(0.3, n_tasks + 0.7)
    ax.set_ylabel("z-score (within task)", fontsize=11)
    ax.set_title("Per-Participant Deviations: Human vs Model",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.2)

    # Legend for panel B
    handles_b = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color_h,
               markersize=6, markeredgecolor="k", markeredgewidth=0.3,
               label="Human"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color_m,
               markersize=6, markeredgecolor="k", markeredgewidth=0.3,
               label="Model"),
        Line2D([0], [0], color="gray", linewidth=0.8, alpha=0.6,
               label="Paired participant"),
    ]
    ax.legend(handles=handles_b, loc="upper right", fontsize=8)

    plt.tight_layout()
    out_path = FIGURES_DIR / "rank_preservation.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out_path}")

    # Console summary
    print("\n=== Per-Task Rank Preservation (Spearman rho) ===")
    print(f"{'Task':<15} {'Model':>7} {'Baseline':>9}")
    for i, tid in enumerate(TIDS_ORDERED):
        bl_str = f"{baseline_rhos[i]:.3f}" if not np.isnan(baseline_rhos[i]) else "   –"
        m_str = f"{model_rhos[i]:.3f}" if not np.isnan(model_rhos[i]) else "   –"
        print(f"{labels[i]:<15} {m_str:>7} {bl_str:>9}")
    print(f"{'Mean':<15} {model_rhos[valid].mean():7.3f}", end="")
    if has_baseline and not np.all(np.isnan(baseline_rhos)):
        print(f" {baseline_rhos[~np.isnan(baseline_rhos)].mean():9.3f}")
    else:
        print()


def _centerline_length(trial_id):
    """Compute the centerline arc length for a trial's tunnel geometry."""
    from experiment.environment import create_environment
    cond = TRIAL_CONDITIONS[trial_id]
    ttype = cond["type"]
    width = cond["width"]

    if ttype in ("sigmoidal", "straight", "gentle_sin", "sharp_sin"):
        env_dict = {
            "env_type": "tunnel_steering_smooth",
            "screen_width": 460, "screen_height": 260,
            "tunnelWidth": width,
            "curvature": cond.get("curvature", 0.025),
            "max_steps": 800,
            "target_radius": width * 0.5,
        }
    elif ttype == "corner":
        env_dict = {
            "env_type": "tunnel_steering_corner",
            "screen_width": 460, "screen_height": 260,
            "tunnelWidth": width,
            "num_corners": cond["num_corners"],
            "corner_offset": cond["corner_offset"],
            "max_steps": 800,
            "target_radius": width * 0.5,
        }
    else:
        return 0.0

    environment = create_environment(env_dict)
    cl = environment["centerline"]
    length = 0.0
    for i in range(1, len(cl)):
        dx = cl[i][0] - cl[i - 1][0]
        dy = cl[i][1] - cl[i - 1][1]
        length += math.sqrt(dx * dx + dy * dy)
    return length


def figure5_steering_law(data):
    """Steering law plot: MT vs Index of Difficulty (ID = L/W).

    Three lines: human (mean across participants), our model, baseline.
    Points colored by tunnel type. Timeout rounds filtered for human.
    """
    pids = sorted(data.keys())
    has_baseline = any(
        "baseline_mt" in data[pid].get(tid, {})
        for pid in pids for tid in range(1, 26)
    )

    # Compute centerline length and ID for each trial
    cl_cache = {}
    trial_ids_by_id = {}  # {tid: ID}
    for tid in TIDS_ORDERED:
        cond = TRIAL_CONDITIONS[tid]
        width = cond["width"]
        if tid not in cl_cache:
            cl_cache[tid] = _centerline_length(tid)
        L = cl_cache[tid]
        ID = L / width
        trial_ids_by_id[tid] = ID

    # Collect mean MT per trial
    human_mt_per_tid = {}
    model_mt_per_tid = {}
    baseline_mt_per_tid = {}

    # Timeout threshold: filter human rounds > 20s
    MT_TIMEOUT = 20.0

    for tid in TIDS_ORDERED:
        h_mts = []
        m_mts = []
        b_mts = []
        for pid in pids:
            if tid not in data[pid]:
                continue
            e = data[pid][tid]
            if e["human_mt"] < MT_TIMEOUT:
                h_mts.append(e["human_mt"])
            m_mts.append(e["model_mt"])
            if has_baseline and "baseline_mt" in e:
                b_mts.append(e["baseline_mt"])

        if h_mts:
            human_mt_per_tid[tid] = np.mean(h_mts)
        if m_mts:
            model_mt_per_tid[tid] = np.mean(m_mts)
        if b_mts:
            baseline_mt_per_tid[tid] = np.mean(b_mts)

    fig, ax = plt.subplots(figsize=(8, 5))

    color_h = "#1f77b4"   # blue for human
    color_m = "#ff7f0e"   # orange for model
    color_bl = "#2ca02c"  # green for baseline

    # Collect all points
    all_human_ids, all_human_mts = [], []
    all_model_ids, all_model_mts = [], []
    all_bl_ids, all_bl_mts = [], []

    for tid in TIDS_ORDERED:
        ID = trial_ids_by_id[tid]
        if tid in human_mt_per_tid:
            all_human_ids.append(ID)
            all_human_mts.append(human_mt_per_tid[tid])
        if tid in model_mt_per_tid:
            all_model_ids.append(ID)
            all_model_mts.append(model_mt_per_tid[tid])
        if tid in baseline_mt_per_tid:
            all_bl_ids.append(ID)
            all_bl_mts.append(baseline_mt_per_tid[tid])

    # Scatter
    ax.scatter(all_human_ids, all_human_mts, color=color_h,
               s=60, alpha=0.8, linewidths=0, zorder=3, label="Human")
    ax.scatter(all_model_ids, all_model_mts, color=color_m,
               s=60, alpha=0.8, linewidths=0, zorder=3, label="Our Model")
    if all_bl_ids:
        ax.scatter(all_bl_ids, all_bl_mts, color=color_bl,
                   s=45, alpha=0.6, linewidths=0, zorder=2, label="Baseline")

    # Fit and draw regression lines (steering law: MT = a + b * ID)
    def _fit_line(ids, mts, color, label, linestyle="-"):
        ids_arr = np.array(ids)
        mts_arr = np.array(mts)
        if len(ids_arr) < 3:
            return
        coeffs = np.polyfit(ids_arr, mts_arr, 1)
        x_range = np.linspace(ids_arr.min(), ids_arr.max(), 100)
        ax.plot(x_range, np.polyval(coeffs, x_range), color=color,
                linestyle=linestyle, linewidth=2, alpha=0.7, zorder=4)
        r = np.corrcoef(ids_arr, mts_arr)[0, 1]
        return coeffs, r

    res_h = _fit_line(all_human_ids, all_human_mts, color_h, "Human")
    res_m = _fit_line(all_model_ids, all_model_mts, color_m, "Our Model")
    res_b = None
    if all_bl_ids:
        res_b = _fit_line(all_bl_ids, all_bl_mts, color_bl, "Baseline", linestyle="--")

    ax.set_xlabel("Index of Difficulty (L / W)", fontsize=12)
    ax.set_ylabel("Movement Time (s)", fontsize=12)
    ax.set_title("Steering Law: MT vs Index of Difficulty", fontsize=13, fontweight="bold")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)

    # Annotation: R² values
    ann_parts = []
    if res_h:
        ann_parts.append(f"Human: R²={res_h[1]**2:.3f}")
    if res_m:
        ann_parts.append(f"Model: R²={res_m[1]**2:.3f}")
    if res_b:
        ann_parts.append(f"Baseline: R²={res_b[1]**2:.3f}")
    if ann_parts:
        ax.text(0.02, 0.98, "\n".join(ann_parts),
                transform=ax.transAxes, fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.legend(loc="upper left", fontsize=9, bbox_to_anchor=(0.02, 0.85))

    # for spine in ax.spines.values():
    #     spine.set_visible(False)
    # ax.tick_params(left=True, bottom=True)

    plt.tight_layout()
    out_path = FIGURES_DIR / "steering_law.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=150, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"\nSaved: {out_path}")

    # Console summary
    print("\n=== Steering Law Summary ===")
    if res_h:
        print(f"Human:    MT = {res_h[0][0]:.4f} * ID + {res_h[0][1]:.4f}  (R²={res_h[1]**2:.3f})")
    if res_m:
        print(f"Model:    MT = {res_m[0][0]:.4f} * ID + {res_m[0][1]:.4f}  (R²={res_m[1]**2:.3f})")
    if res_b:
        print(f"Baseline: MT = {res_b[0][0]:.4f} * ID + {res_b[0][1]:.4f}  (R²={res_b[1]**2:.3f})")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate evaluation figures")
    parser.add_argument("--no-filter-baseline", action="store_true", default=False,
                        help="Disable post-processing filters on baseline runs "
                             "(timed-out / diverged)")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    filter_bl = not args.no_filter_baseline
    print(f"Loading per-participant results (baseline filtering: {'ON' if filter_bl else 'OFF'})...")
    data = load_all_results(filter_baseline=filter_bl)
    pids = sorted(data.keys())
    print(f"Found {len(pids)} participants, "
          f"{sum(len(v) for v in data.values())} participant×trial entries")

    if not pids:
        print("ERROR: No participant results found in", RESULTS_DIR)
        sys.exit(1)

    # figure1_human_vs_model_variability(data)
    # figure2_scatter(data)
    figure3_paired_mt_pertask(data)
    # figure4_rank_preservation(data)
    figure5_steering_law(data)
    print("\nDone.")


if __name__ == "__main__":
    main()

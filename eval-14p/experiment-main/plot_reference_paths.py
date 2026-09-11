"""
Plot fitted reference paths inside task tunnels — no simulator runs, no human data.

For every participant with a fitted config (eval/model_fitting/results/
{pid}_gam_config_s{seed}.json) and every trial condition in run_eval.TRIAL_CONDITIONS,
this script:
  1. rebuilds the tunnel (centerline + PathConstraint) exactly as run_eval does,
  2. loads the fitted config into CursorSimulator to obtain its reference_path params,
  3. builds the optimal (corner-cutting) reference path with the same call the
     simulator makes internally (generate_optimal_reference_path),
  4. plots tunnel walls + reference path to PDF.

Outputs (default: eval/experiment-main/figures/reference_paths/):
    {pid}/reference_path_t{tid}.pdf         one PDF per participant x trial
    {pid}_reference_paths.pdf               5x5 overview grid per participant
    all_participants_reference_paths.pdf    5x5 grid, all participants overlaid

Usage:
    python eval/experiment-main/plot_reference_paths.py
    python eval/experiment-main/plot_reference_paths.py --pids P204813 --trials 1 6 21
    python eval/experiment-main/plot_reference_paths.py --no-per-trial
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "hcs_package" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "eval"))
sys.path.insert(0, str(SCRIPT_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from experiment.utils import generateTunnelBoundaries
from hcs_package.cursor_simulator import CursorSimulator
from hcs_package.reference_path import ReferencePath, generate_optimal_reference_path
from hcs_package.constraints import PathConstraint, RectangleConstraint, PolygonConstraint
from hcs_package.constraint_utils import parse_constraints_from_json, convert_constraints_to_corridor_bounds

from run_eval import (
    TRIAL_CONDITIONS,
    FITTING_RESULTS_DIR,
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    _build_sigmoidal_config,
    _build_corner_config,
)

DEFAULT_OUT_DIR = SCRIPT_DIR / "figures" / "reference_paths"

# Overview grid: rows = tunnel family (in trial-id order), cols = width 10..50 mm
GRID_ROWS = [
    ("Sinusoidal (κ=0.025)", [1, 2, 3, 4, 5]),
    ("Corner",               [6, 7, 8, 9, 10]),
    ("Straight",             [11, 12, 13, 14, 15]),
    ("Gentle sin (κ=0.015)", [16, 17, 18, 19, 20]),
    ("Sharp sin (κ=0.05)",   [21, 22, 23, 24, 25]),
]
GRID_COLS = ["W = 10 mm", "W = 20 mm", "W = 30 mm", "W = 40 mm", "W = 50 mm"]

# Styling — tunnel is recessive gray (same as plot_utils), path carries one hue.
TUNNEL_FILL = "#e6e6e6"
TUNNEL_EDGE = "#8c8c8c"
CENTERLINE_COLOR = "#9a9a9a"
REF_COLOR = "#1f5fbf"
START_COLOR = "#2e8b57"
END_COLOR = "#c0392b"
TEXT_COLOR = "#333333"


# ---------------------------------------------------------------------------
# Tunnel + reference-path construction
# ---------------------------------------------------------------------------

def build_task(trial_id):
    """Return (task_config, centerline_m, tunnel_width) for a trial condition."""
    cond = TRIAL_CONDITIONS[trial_id]
    if cond["type"] == "sigmoidal":
        task_config, centerline = _build_sigmoidal_config(cond["width"], cond["curvature"])
    else:
        task_config, centerline = _build_corner_config(
            cond["width"], cond["num_corners"], cond["corner_offset"])
    centerline = np.asarray([[x, y] for x, y in centerline], dtype=float)
    return task_config, centerline, cond["width"]


def build_reference_path(sim, task_config):
    """Build the reference path exactly as CursorSimulator.generate_trajectory_with_waypoints
    does (use_optimal_path=True), without running the MPC.

    Mirrors hcs_package/src/hcs_package/cursor_simulator.py — keep in sync.
    Returns (reference_path, centerline_spline).
    """
    screen_width = float(task_config.get("screen_width", 460))
    screen_height = float(task_config.get("screen_height", 260))
    screen_width_m = 0.46
    screen_height_m = screen_height / screen_width * screen_width_m

    waypoints_norm = [
        (x / screen_width * screen_width_m, y / screen_height * screen_height_m)
        for x, y in task_config["waypoints"]
    ]

    constraint_config = None
    if task_config.get("constraints") is not None:
        constraint_config = parse_constraints_from_json(task_config["constraints"])

    cartesian_regions = []
    tunnel_width = None
    if constraint_config is not None:
        for region in constraint_config.regions:
            if isinstance(region.geometry, (RectangleConstraint, PolygonConstraint)):
                cartesian_regions.append(region)
            elif isinstance(region.geometry, PathConstraint) and tunnel_width is None:
                tunnel_width = float(region.geometry.width)
    if tunnel_width is None:
        distances = [
            np.linalg.norm(np.array(waypoints_norm[i + 1]) - np.array(waypoints_norm[i]))
            for i in range(len(waypoints_norm) - 1)
        ]
        avg_distance = np.mean(distances) if distances else 0.1
        tunnel_width = min(0.1, max(0.02, avg_distance * 0.3))

    centerline_spline = ReferencePath(waypoints_norm, s=0.0, k=3)
    corridor_bounds = None
    if constraint_config is not None:
        corridor_bounds = convert_constraints_to_corridor_bounds(
            constraint_config, centerline_spline, default_margin=sim.planner_margin)

    rp = sim.reference_path_params
    reference_path = generate_optimal_reference_path(
        tunnel_path=waypoints_norm,
        tunnel_width=tunnel_width,
        margin=sim.planner_margin,
        num_knots=None,
        w_cut=rp["w_cut"],
        w_suppress=rp["w_suppress"],
        w_width_exp=rp["w_width_exp"],
        cut_window_frac=rp["cut_window_frac"],
        global_clearance_ref=rp["global_clearance_ref"],
        cartesian_constraints=cartesian_regions if cartesian_regions else None,
        corridor_bounds=corridor_bounds,
        centerline_cache=centerline_spline,
    )
    return reference_path, centerline_spline


def sample_path(ref_path, n=600):
    s = np.linspace(0.0, ref_path.total_length, n)
    return np.asarray([ref_path(float(t)) for t in s], dtype=float)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_tunnel(ax, centerline, tunnel_width):
    left, right = generateTunnelBoundaries(centerline, tunnel_width)
    poly_x = np.concatenate([left[:, 0], right[::-1, 0]])
    poly_y = np.concatenate([left[:, 1], right[::-1, 1]])
    ax.fill(poly_x, poly_y, color=TUNNEL_FILL, zorder=1, linewidth=0)
    ax.plot(left[:, 0], left[:, 1], color=TUNNEL_EDGE, linewidth=0.8, zorder=2)
    ax.plot(right[:, 0], right[:, 1], color=TUNNEL_EDGE, linewidth=0.8, zorder=2)
    ax.plot(centerline[:, 0], centerline[:, 1], color=CENTERLINE_COLOR,
            linestyle=(0, (3, 3)), linewidth=0.7, zorder=3)


def draw_reference(ax, ref_pts, linewidth=1.8, alpha=1.0, markers=True):
    ax.plot(ref_pts[:, 0], ref_pts[:, 1], color=REF_COLOR, linewidth=linewidth,
            alpha=alpha, zorder=10, solid_capstyle="round")
    if markers:
        ax.scatter([ref_pts[0, 0]], [ref_pts[0, 1]], s=18, color=START_COLOR,
                   edgecolor="white", linewidth=0.6, zorder=11)
        ax.scatter([ref_pts[-1, 0]], [ref_pts[-1, 1]], s=18, color=END_COLOR,
                   edgecolor="white", linewidth=0.6, zorder=11)


def style_axes(ax, show_ticks=True):
    pad = 0.01
    ax.set_xlim(-pad, WINDOW_WIDTH + pad)
    ax.set_ylim(-pad, WINDOW_HEIGHT + pad)
    ax.invert_yaxis()  # screen coordinates, consistent with eval/utils/plot_utils.py
    ax.set_aspect("equal")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#bbbbbb")
    ax.tick_params(colors="#666666", labelsize=7, length=2)
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])


def legend_handles(n_participants=None):
    ref_label = "Reference path" if n_participants is None else f"Reference paths (n={n_participants})"
    return [
        Line2D([0], [0], color=REF_COLOR, linewidth=1.8, label=ref_label),
        Line2D([0], [0], color=CENTERLINE_COLOR, linestyle=(0, (3, 3)), linewidth=0.9, label="Tunnel centerline"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=START_COLOR, markersize=5, label="Start"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=END_COLOR, markersize=5, label="End"),
    ]


def rp_param_text(sim):
    rp = sim.reference_path_params
    return (f"w_cut={rp['w_cut']:.3f}  w_suppress={rp['w_suppress']:.3f}  "
            f"w_width_exp={rp['w_width_exp']:.2f}  cut_window_frac={rp['cut_window_frac']:.3f}  "
            f"clearance_ref={rp['global_clearance_ref']:.4f}")


def plot_single_trial(pid, sim, trial_id, tunnels, ref_pts, out_path):
    _, centerline, width = tunnels[trial_id]
    cond = TRIAL_CONDITIONS[trial_id]
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    draw_tunnel(ax, centerline, width)
    draw_reference(ax, ref_pts)
    style_axes(ax)
    ax.set_xlabel("x (m)", fontsize=8, color=TEXT_COLOR)
    ax.set_ylabel("y (m)", fontsize=8, color=TEXT_COLOR)
    ax.set_title(f"{pid} — trial {trial_id}: {cond['label']}", fontsize=9, color=TEXT_COLOR)
    ax.legend(handles=legend_handles(), loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=4, fontsize=7, frameon=False)
    fig.text(0.5, 0.005, rp_param_text(sim), ha="center", va="bottom", fontsize=6, color="#777777")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_overview_grid(title, subtitle, tunnels, ref_pts_by_trial, out_path,
                       n_participants=None, linewidth=1.6, alpha=1.0):
    """5x5 grid. ref_pts_by_trial: {trial_id: [ref_pts, ...]} (one or many paths per cell)."""
    fig, axes = plt.subplots(len(GRID_ROWS), len(GRID_COLS), figsize=(15, 9.2))
    for r, (row_label, tids) in enumerate(GRID_ROWS):
        for c, tid in enumerate(tids):
            ax = axes[r, c]
            if tid not in tunnels:
                ax.axis("off")
                continue
            _, centerline, width = tunnels[tid]
            draw_tunnel(ax, centerline, width)
            paths = ref_pts_by_trial.get(tid, [])
            for i, pts in enumerate(paths):
                draw_reference(ax, pts, linewidth=linewidth, alpha=alpha, markers=(i == 0))
            style_axes(ax, show_ticks=False)
            ax.text(0.01, 0.97, f"t{tid}", transform=ax.transAxes, fontsize=7,
                    color="#777777", ha="left", va="top")
            if r == 0:
                ax.set_title(GRID_COLS[c], fontsize=9, color=TEXT_COLOR)
            if c == 0:
                ax.set_ylabel(row_label, fontsize=9, color=TEXT_COLOR)
    fig.suptitle(title, fontsize=12, color=TEXT_COLOR, y=0.995)
    if subtitle:
        fig.text(0.5, 0.955, subtitle, ha="center", fontsize=7.5, color="#777777")
    fig.legend(handles=legend_handles(n_participants), loc="lower center", ncol=4,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def discover_pids(seed):
    return sorted(p.name.split("_gam_config_")[0]
                  for p in FITTING_RESULTS_DIR.glob(f"*_gam_config_s{seed}.json"))


def main():
    parser = argparse.ArgumentParser(description="Plot fitted reference paths in task tunnels (PDF).")
    parser.add_argument("--pids", nargs="+", default=None,
                        help="Participant IDs (default: all with fitted configs)")
    parser.add_argument("--trials", type=int, nargs="+", default=None,
                        help="Trial IDs (default: all in TRIAL_CONDITIONS)")
    parser.add_argument("--seed", type=int, default=42, help="Fitted-config seed suffix")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--no-per-trial", action="store_true",
                        help="Skip the one-PDF-per-trial outputs; only write overview grids")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pids = args.pids or discover_pids(args.seed)
    trial_ids = args.trials or sorted(TRIAL_CONDITIONS.keys())
    if not pids:
        sys.exit(f"No fitted configs found in {FITTING_RESULTS_DIR} for seed {args.seed}")

    print(f"Participants: {len(pids)}  Trials: {len(trial_ids)}  Output: {out_dir}")

    # Tunnels are participant-independent — build once.
    tunnels = {tid: build_task(tid) for tid in trial_ids}

    all_ref_pts = {tid: [] for tid in trial_ids}
    manifest = {}
    for pid in pids:
        cfg_path = FITTING_RESULTS_DIR / f"{pid}_gam_config_s{args.seed}.json"
        if not cfg_path.exists():
            print(f"  [skip] {pid}: {cfg_path.name} not found")
            continue
        sim = CursorSimulator(str(cfg_path))
        print(f"  {pid}: {rp_param_text(sim)}")

        pid_dir = out_dir / pid
        if not args.no_per_trial:
            pid_dir.mkdir(exist_ok=True)

        ref_by_trial = {}
        stats = {}
        for tid in trial_ids:
            task_config, centerline, width = tunnels[tid]
            ref_path, centerline_spline = build_reference_path(sim, task_config)
            pts = sample_path(ref_path)
            ref_by_trial[tid] = [pts]
            all_ref_pts[tid].append(pts)
            stats[tid] = {
                "ref_length_m": float(ref_path.total_length),
                "centerline_length_m": float(centerline_spline.total_length),
                "length_ratio": float(ref_path.total_length / centerline_spline.total_length),
            }
            if not args.no_per_trial:
                plot_single_trial(pid, sim, tid, tunnels, pts, pid_dir / f"reference_path_t{tid}.pdf")

        plot_overview_grid(
            title=f"Fitted reference paths — {pid}",
            subtitle=rp_param_text(sim),
            tunnels=tunnels,
            ref_pts_by_trial=ref_by_trial,
            out_path=out_dir / f"{pid}_reference_paths.pdf",
        )
        manifest[pid] = {"config": str(cfg_path.relative_to(PROJECT_ROOT)),
                         "reference_path_params": sim.reference_path_params,
                         "trials": stats}

    n = sum(1 for p in manifest)
    if n > 0:
        plot_overview_grid(
            title=f"Fitted reference paths — all participants (n={n})",
            subtitle=None,
            tunnels=tunnels,
            ref_pts_by_trial=all_ref_pts,
            out_path=out_dir / "all_participants_reference_paths.pdf",
            n_participants=n, linewidth=1.0, alpha=0.55,
        )

    with open(out_dir / "reference_path_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Done. Wrote outputs for {n} participants to {out_dir}")


if __name__ == "__main__":
    main()

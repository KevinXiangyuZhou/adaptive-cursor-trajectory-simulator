"""
Debug baseline simulator with trajectory visualization.

Runs a sampled baseline config on selected tunnel tasks and plots the
resulting trajectories overlaid on the tunnel, to diagnose issues like
the cursor going outside the tunnel or failing to reach the target.

Usage:
    # Run with first available config on a few sinusoidal tasks:
    python -m eval.experiment-main.debug_baseline

    # Specify participant and trials:
    python -m eval.experiment-main.debug_baseline --pid P204813 --trials 1 3 5 21 23

    # Run with noise enabled:
    python -m eval.experiment-main.debug_baseline --pid P204813 --noise --runs 3
"""

import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "hcs_package" / "src"))

from experiment.environment import create_environment, generate_task_config

BASELINE_PKG_DIR = PROJECT_ROOT / "eval" / "chi-26-ea_baseline_pacakage" / "src"
BASELINE_FITTING_DIR = PROJECT_ROOT / "eval" / "baseline_fitting" / "results"
OUTPUT_DIR = Path(__file__).resolve().parent / "debug_baseline_output"

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
    11: {"type": "sigmoidal", "width": 0.01, "curvature": 0.0,   "label": "Str W=10"},
    12: {"type": "sigmoidal", "width": 0.02, "curvature": 0.0,   "label": "Str W=20"},
    13: {"type": "sigmoidal", "width": 0.03, "curvature": 0.0,   "label": "Str W=30"},
    14: {"type": "sigmoidal", "width": 0.04, "curvature": 0.0,   "label": "Str W=40"},
    15: {"type": "sigmoidal", "width": 0.05, "curvature": 0.0,   "label": "Str W=50"},
    16: {"type": "sigmoidal", "width": 0.01, "curvature": 0.015, "label": "Gen W=10"},
    17: {"type": "sigmoidal", "width": 0.02, "curvature": 0.015, "label": "Gen W=20"},
    18: {"type": "sigmoidal", "width": 0.03, "curvature": 0.015, "label": "Gen W=30"},
    19: {"type": "sigmoidal", "width": 0.04, "curvature": 0.015, "label": "Gen W=40"},
    20: {"type": "sigmoidal", "width": 0.05, "curvature": 0.015, "label": "Gen W=50"},
    21: {"type": "sigmoidal", "width": 0.01, "curvature": 0.05,  "label": "Shp W=10"},
    22: {"type": "sigmoidal", "width": 0.02, "curvature": 0.05,  "label": "Shp W=20"},
    23: {"type": "sigmoidal", "width": 0.03, "curvature": 0.05,  "label": "Shp W=30"},
    24: {"type": "sigmoidal", "width": 0.04, "curvature": 0.05,  "label": "Shp W=40"},
    25: {"type": "sigmoidal", "width": 0.05, "curvature": 0.05,  "label": "Shp W=50"},
}


def build_task_config(trial_id):
    """Build task config and centerline for a trial."""
    cond = TRIAL_CONDITIONS[trial_id]
    if cond["type"] == "sigmoidal":
        env_dict = {
            "env_type": "tunnel_steering_smooth",
            "screen_width": 460, "screen_height": 260,
            "tunnelWidth": cond["width"],
            "curvature": cond["curvature"],
            "max_steps": 800,
            "target_radius": cond["width"] * 0.5,
        }
    else:
        env_dict = {
            "env_type": "tunnel_steering_corner",
            "screen_width": 460, "screen_height": 260,
            "tunnelWidth": cond["width"],
            "num_corners": cond["num_corners"],
            "corner_offset": cond["corner_offset"],
            "max_steps": 800,
            "target_radius": cond["width"] * 0.5,
        }
    environment = create_environment(env_dict)
    task_config = generate_task_config(environment, include_constraints=True)
    centerline = environment["centerline"]
    return task_config, centerline, cond


def load_baseline_sim(config_path, add_noise=False):
    """Load baseline simulator with module isolation."""
    # Read config and optionally enable noise
    with open(config_path) as f:
        cfg = json.load(f)
    cfg["add_noise"] = add_noise
    if not add_noise:
        cfg["nc"] = [0, 0]

    # Write patched config
    fd, tmp = tempfile.mkstemp(suffix=".json", prefix="bl_debug_")
    os.close(fd)
    with open(tmp, "w") as f:
        json.dump(cfg, f)

    # Module isolation
    saved = {k: sys.modules.pop(k)
             for k in list(sys.modules) if k == 'hcs_package' or k.startswith('hcs_package.')}
    sys.path.insert(0, str(BASELINE_PKG_DIR))
    try:
        from hcs_package.cursor_simulator import CursorSimulator as BaselineCls
        sim = BaselineCls(tmp)
    finally:
        for k in list(sys.modules):
            if k == 'hcs_package' or k.startswith('hcs_package.'):
                del sys.modules[k]
        sys.modules.update(saved)
        sys.path.remove(str(BASELINE_PKG_DIR))
        os.unlink(tmp)

    return sim, cfg


def run_one_trial(sim, task_config, max_steps=800, target_radius=0.01):
    """Run baseline sim on one task, return trajectory and metadata."""
    fd, task_file = tempfile.mkstemp(suffix=".json", prefix="bl_task_")
    os.close(fd)
    with open(task_file, "w") as f:
        json.dump(task_config, f)

    t0 = time.time()
    try:
        traj_raw = sim.generate_trajectory_with_waypoints(
            task_file=task_file,
            max_steps=max_steps,
            target_radius=target_radius,
            use_optimal_path=True,
        )
    finally:
        os.unlink(task_file)
    elapsed = time.time() - t0

    scale = 0.001
    traj = [[x * scale, y * scale] for x, y, _ in traj_raw]
    n = len(traj)
    interval = sim.interval
    ct = n * interval

    # Compute speeds
    speeds = []
    for i in range(1, n):
        d = math.sqrt((traj[i][0] - traj[i-1][0])**2 + (traj[i][1] - traj[i-1][1])**2)
        speeds.append(d / interval)
    if speeds:
        speeds.insert(0, speeds[0])

    return {
        "trajectory": traj,
        "speeds": speeds,
        "n_steps": n,
        "completion_time": ct,
        "wall_clock": elapsed,
        "hit_max_steps": n >= max_steps,
    }


def plot_trial_debug(trial_id, cond, centerline, results, tunnel_width, output_path):
    """Plot trajectory overlaid on tunnel for debugging."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    cl = np.array(centerline)
    half_w = tunnel_width / 2

    for ax_idx, ax in enumerate(axes):
        # Draw tunnel walls
        if len(cl) >= 2:
            # Compute normals
            diffs = np.diff(cl, axis=0)
            tangents = diffs / (np.linalg.norm(diffs, axis=1, keepdims=True) + 1e-12)
            normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
            # Extend normals to match centerline length
            normals = np.vstack([normals, normals[-1:]])

            wall_left = cl + normals * half_w
            wall_right = cl - normals * half_w

            ax.plot(wall_left[:, 0], wall_left[:, 1], 'k-', linewidth=1, alpha=0.5)
            ax.plot(wall_right[:, 0], wall_right[:, 1], 'k-', linewidth=1, alpha=0.5)
            ax.plot(cl[:, 0], cl[:, 1], 'k--', linewidth=0.5, alpha=0.3, label="Centerline")

        # Draw trajectories
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(results), 1)))
        for i, res in enumerate(results):
            traj = np.array(res["trajectory"])
            if len(traj) < 2:
                continue

            label = f"Run {i}: {res['completion_time']:.2f}s ({res['n_steps']} steps)"
            if res["hit_max_steps"]:
                label += " [TIMEOUT]"

            if ax_idx == 0:
                # Trajectory plot
                ax.plot(traj[:, 0], traj[:, 1], color=colors[i],
                        linewidth=0.8, alpha=0.8, label=label)
                ax.scatter(traj[0, 0], traj[0, 1], color=colors[i],
                           marker='o', s=30, zorder=5)
                ax.scatter(traj[-1, 0], traj[-1, 1], color=colors[i],
                           marker='x', s=30, zorder=5)
            else:
                # Lateral deviation from centerline
                lat_devs = []
                for pt in traj:
                    dists = np.linalg.norm(cl - pt, axis=1)
                    lat_devs.append(np.min(dists))
                progress = np.linspace(0, 1, len(lat_devs))
                ax.plot(progress, lat_devs, color=colors[i], linewidth=0.8,
                        alpha=0.8, label=label)

    # Target position
    if len(cl) >= 1:
        target = cl[-1]
        target_radius = cond["width"] * 0.5
        circle = plt.Circle(target, target_radius, fill=False,
                            edgecolor='red', linewidth=1.5, linestyle='--')
        axes[0].add_patch(circle)

    axes[0].set_aspect("equal")
    axes[0].set_title(f"Trial {trial_id}: {cond['label']} (W={cond['width']*1000:.0f}mm)",
                      fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=7, loc="upper left")
    axes[0].grid(alpha=0.2)
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")

    axes[1].axhline(half_w, color='red', linestyle='--', linewidth=0.8,
                     alpha=0.5, label=f"Wall (half_w={half_w*1000:.1f}mm)")
    axes[1].set_title("Lateral Deviation from Centerline", fontsize=12)
    axes[1].set_xlabel("Progress (normalized)")
    axes[1].set_ylabel("Distance from centerline (m)")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.2)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Debug baseline simulator")
    parser.add_argument("--pid", default=None,
                        help="Participant ID (default: first available)")
    parser.add_argument("--trials", type=int, nargs="+",
                        default=[1, 3, 5, 16, 18, 21, 23],
                        help="Trial IDs to test (default: sample of sinusoidal tasks)")
    parser.add_argument("--noise", action="store_true",
                        help="Enable motor noise (default: deterministic)")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of runs per trial (default: 1, use >1 with --noise)")
    parser.add_argument("--max-steps", type=int, default=800,
                        help="Max simulation steps (default: 800)")
    args = parser.parse_args()

    # Find config
    if args.pid:
        cfg_path = BASELINE_FITTING_DIR / f"{args.pid}_baseline_config_s42.json"
        if not cfg_path.exists():
            sys.exit(f"Config not found: {cfg_path}")
    else:
        configs = sorted(BASELINE_FITTING_DIR.glob("*_baseline_config_s42.json"))
        if not configs:
            sys.exit("No baseline configs found")
        cfg_path = configs[0]
        args.pid = cfg_path.name.split("_baseline_config")[0]

    print(f"Participant: {args.pid}")
    print(f"Config: {cfg_path}")
    print(f"Noise: {'ON' if args.noise else 'OFF'}")
    print(f"Runs per trial: {args.runs}")
    print(f"Trials: {args.trials}")

    # Load config and print key params
    with open(cfg_path) as f:
        raw_cfg = json.load(f)
    pw = raw_cfg.get("planner_weights", {})
    print(f"\nConfig params:")
    print(f"  Th={raw_cfg.get('Th'):.3f}  nc={raw_cfg.get('nc')}")
    print(f"  Planner weights:")
    for k, v in sorted(pw.items()):
        print(f"    {k:15s}: {v:.6g}")

    # Load simulator
    print("\nLoading baseline simulator...")
    sim, patched_cfg = load_baseline_sim(cfg_path, add_noise=args.noise)
    print(f"  Interval: {sim.interval}s")

    # Run trials
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput: {OUTPUT_DIR}")
    print(f"\n{'='*60}")

    summary = []
    for tid in args.trials:
        if tid not in TRIAL_CONDITIONS:
            print(f"Trial {tid}: unknown, skipping")
            continue

        cond = TRIAL_CONDITIONS[tid]
        print(f"\nTrial {tid}: {cond['label']} (W={cond['width']*1000:.0f}mm)")

        task_config, centerline, _ = build_task_config(tid)
        target_radius = task_config.get("target_radius", cond["width"] * 0.5)

        results = []
        for run_id in range(args.runs):
            res = run_one_trial(sim, task_config,
                                max_steps=args.max_steps,
                                target_radius=target_radius)
            status = "TIMEOUT" if res["hit_max_steps"] else "OK"
            print(f"  Run {run_id}: {res['n_steps']} steps, "
                  f"CT={res['completion_time']:.2f}s, "
                  f"wall={res['wall_clock']:.1f}s [{status}]")
            results.append(res)

        # Check if trajectory ends near target
        cl = np.array(centerline)
        target = cl[-1] if len(cl) > 0 else None
        for i, res in enumerate(results):
            traj = np.array(res["trajectory"])
            if len(traj) > 0 and target is not None:
                end_dist = np.linalg.norm(traj[-1] - target)
                res["end_dist_to_target"] = end_dist
                if end_dist > cond["width"]:
                    print(f"  Run {i}: END {end_dist*1000:.1f}mm from target "
                          f"(>{cond['width']*1000:.0f}mm tunnel width) — MISSED!")

        plot_trial_debug(
            tid, cond, centerline, results, cond["width"],
            OUTPUT_DIR / f"debug_t{tid}_{args.pid}.png"
        )

        summary.append({
            "trial": tid,
            "label": cond["label"],
            "n_runs": len(results),
            "timeouts": sum(1 for r in results if r["hit_max_steps"]),
            "avg_ct": np.mean([r["completion_time"] for r in results]),
            "avg_steps": np.mean([r["n_steps"] for r in results]),
            "avg_wall": np.mean([r["wall_clock"] for r in results]),
        })

    # Print summary table
    print(f"\n{'='*60}")
    print(f"{'Trial':<15} {'Runs':>4} {'T/O':>3} {'Avg CT':>8} {'Avg Steps':>10} {'Wall/run':>9}")
    for s in summary:
        print(f"{s['label']:<15} {s['n_runs']:4d} {s['timeouts']:3d} "
              f"{s['avg_ct']:8.2f}s {s['avg_steps']:10.0f} {s['avg_wall']:8.1f}s")


if __name__ == "__main__":
    main()

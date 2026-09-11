"""
Stop-and-Go Behavior Evaluation Experiment.

Evaluates the model's curvature-rate-based speed modulation by comparing
speed variability between corner and sigmoidal tunnels with the same width.

The curvature rate factor should produce:
- Corner tunnels: Strong slowdown at sharp turns (high SpeedCV, low MinSpeedRatio)
- Sigmoidal tunnels: Smooth speed changes (low SpeedCV, high MinSpeedRatio)

Usage:
    python run_eval.py [--config path/to/user_config.json] [--rounds N]
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.signal import find_peaks

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "hcs_package" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "eval" / "utils"))

from experiment.environment import create_environment, generate_task_config
from hcs_package.cursor_simulator import CursorSimulator
from trackpad_checker import check_participant
from oob import check_trajectory_oob

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HUMAN_DATA_DIR = PROJECT_ROOT / "eval" / "human_data" / "raw"
RESULTS_DIR = SCRIPT_DIR / "results"

# Same width conditions to isolate curvature effect
TRIAL_CONDITIONS = {
    1: {"type": "sigmoidal", "width": 0.02, "curvature": 0.025, "label": "sigmoidal W=20mm"},
    2: {"type": "sigmoidal", "width": 0.04, "curvature": 0.025, "label": "sigmoidal W=40mm"},
    3: {"type": "corner", "width": 0.02, "num_corners": 2, "corner_offset": 0.1, "label": "corner W=20mm"},
    4: {"type": "corner", "width": 0.04, "num_corners": 2, "corner_offset": 0.1, "label": "corner W=40mm"},
}

# Corner locations (approximate progress values where corners occur)
CORNER_PROGRESS_LOCATIONS = [0.25, 0.50, 0.75]  # For 2-corner tunnel

DEFAULT_CONFIG = PROJECT_ROOT / "experiment" / "user_configurations" / "customized.json"


# ===================================================================
# Utility Functions
# ===================================================================

def _path_length(trajectory):
    """Sum of Euclidean segment lengths."""
    total = 0.0
    for i in range(1, len(trajectory)):
        dx = trajectory[i][0] - trajectory[i - 1][0]
        dy = trajectory[i][1] - trajectory[i - 1][1]
        total += math.sqrt(dx * dx + dy * dy)
    return total


def _compute_speeds(trajectory, timestamps, window_size=5):
    """Central-difference speed with moving-average smoothing."""
    n = len(trajectory)
    if n < 2 or len(timestamps) != n:
        return []
    raw = []
    for i in range(n):
        if i == 0:
            p0, p1 = trajectory[0], trajectory[1]
            dt = (timestamps[1] - timestamps[0]) / 1000.0
        elif i == n - 1:
            p0, p1 = trajectory[-2], trajectory[-1]
            dt = (timestamps[-1] - timestamps[-2]) / 1000.0
        else:
            p0, p1 = trajectory[i - 1], trajectory[i + 1]
            dt = (timestamps[i + 1] - timestamps[i - 1]) / 1000.0
        dist = math.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)
        raw.append(dist / dt if dt > 0 else 0.0)

    half = window_size // 2
    smoothed = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        smoothed.append(sum(raw[lo:hi]) / (hi - lo))
    return smoothed


def compute_progress_along_path(trajectory, centerline):
    """Compute progress (0-1) for each trajectory point."""
    if not trajectory or not centerline:
        return [0.0] * len(trajectory)
    
    cl = np.array(centerline)
    # Compute cumulative arc length
    arc_lengths = [0.0]
    for i in range(1, len(cl)):
        arc_lengths.append(arc_lengths[-1] + np.linalg.norm(cl[i] - cl[i-1]))
    total_length = arc_lengths[-1]
    
    if total_length < 1e-6:
        return [0.0] * len(trajectory)
    
    progress = []
    for pt in trajectory:
        pt_np = np.array(pt)
        distances = np.linalg.norm(cl - pt_np, axis=1)
        closest_idx = np.argmin(distances)
        progress.append(arc_lengths[closest_idx] / total_length)
    
    return progress


# ===================================================================
# Stop-and-Go Specific Metrics
# ===================================================================

def compute_speed_cv(speeds):
    """Coefficient of variation of speed (std/mean)."""
    if not speeds or len(speeds) < 2:
        return 0.0
    speeds_arr = np.array(speeds)
    mean_spd = np.mean(speeds_arr)
    if mean_spd <= 0:
        return 0.0
    return float(np.std(speeds_arr) / mean_spd)


def compute_min_speed_ratio(speeds):
    """Ratio of minimum speed to mean speed."""
    if not speeds or len(speeds) < 2:
        return 1.0
    speeds_arr = np.array(speeds)
    mean_spd = np.mean(speeds_arr)
    min_spd = np.min(speeds_arr)
    if mean_spd <= 0:
        return 1.0
    return float(min_spd / mean_spd)


def compute_speed_dip_metrics(speeds, threshold_fraction=0.5):
    """
    Analyze speed dips (local minima below threshold).
    
    Returns:
        dict with dip_count, dip_depth_mean, dip_depth_max
    """
    if not speeds or len(speeds) < 5:
        return {"dip_count": 0, "dip_depth_mean": 0.0, "dip_depth_max": 0.0}
    
    speeds_arr = np.array(speeds)
    mean_spd = np.mean(speeds_arr)
    threshold = mean_spd * threshold_fraction
    
    # Find local minima by finding peaks in inverted signal
    inverted = -speeds_arr
    peaks, _ = find_peaks(inverted, distance=5)
    
    # Filter to only dips below threshold
    dip_indices = [p for p in peaks if speeds_arr[p] < threshold]
    
    if not dip_indices:
        return {"dip_count": 0, "dip_depth_mean": 0.0, "dip_depth_max": 0.0}
    
    dip_depths = [(mean_spd - speeds_arr[p]) / mean_spd for p in dip_indices]
    
    return {
        "dip_count": len(dip_indices),
        "dip_depth_mean": float(np.mean(dip_depths)),
        "dip_depth_max": float(np.max(dip_depths)),
    }


def compute_corner_slowdown_factor(speeds, progress_values, corner_locations, window=0.1):
    """
    Compute speed at corners vs speed at straights.
    
    Returns slowdown_factor: mean_speed_at_corners / mean_speed_at_straights
    Lower values indicate stronger stop-and-go.
    """
    if not speeds or not progress_values:
        return 1.0
    
    corner_speeds = []
    straight_speeds = []
    
    for prog, spd in zip(progress_values, speeds):
        at_corner = any(abs(prog - c) < window for c in corner_locations)
        if at_corner:
            corner_speeds.append(spd)
        else:
            straight_speeds.append(spd)
    
    if not corner_speeds or not straight_speeds:
        return 1.0
    
    return float(np.mean(corner_speeds) / np.mean(straight_speeds))


def compute_speed_smoothness(speeds, dt=0.05):
    """
    Compute smoothness of speed profile.
    Higher = smoother, Lower = more stop-and-go.
    """
    if len(speeds) < 3:
        return 1.0
    
    speed_arr = np.array(speeds)
    # Acceleration (first derivative)
    accel = np.diff(speed_arr) / dt
    # Jerk (second derivative)  
    jerk = np.diff(accel) / dt
    
    # RMS jerk (lower = smoother)
    rms_jerk = np.sqrt(np.mean(jerk ** 2))
    
    # Convert to smoothness score (higher = smoother)
    return float(1.0 / (1.0 + rms_jerk * 0.01))


# ===================================================================
# Load Human Data
# ===================================================================

def get_trackpad_participants(min_confidence=0.7):
    """Get participant IDs classified as trackpad users."""
    trackpad_pids = set()
    for fpath in sorted(HUMAN_DATA_DIR.glob("*.json")):
        result = check_participant(fpath)
        if (result.get("device_type") == "trackpad" and 
            result.get("confidence", 0) >= min_confidence):
            trackpad_pids.add(result.get("participant_id"))
    return trackpad_pids


def load_human_data(trackpad_confidence=None):
    """Load trials 1-4 (sigmoidal and corner) from raw participant JSONs."""
    trackpad_pids = None
    if trackpad_confidence is not None:
        trackpad_pids = get_trackpad_participants(trackpad_confidence)
    
    records = []
    for fpath in sorted(HUMAN_DATA_DIR.glob("*.json")):
        with open(fpath) as f:
            data = json.load(f)
        pid = data.get("participantId", fpath.stem)
        
        if trackpad_pids is not None and pid not in trackpad_pids:
            continue
        
        sessions = data.get("sessions", [])
        trial_list = sessions[0].get("trialData", []) if sessions else data.get("trialData", [])
        
        for trial in trial_list:
            tid = trial.get("trial_id")
            if tid not in TRIAL_CONDITIONS:
                continue
            
            cond = trial.get("condition", {})
            traj_raw = trial.get("trajectory", [])
            timestamps = trial.get("timestamps", [])
            ct = trial.get("completionTime", 0.0)
            
            if not traj_raw or not timestamps:
                continue
            
            traj = [(p["x"], p["y"]) for p in traj_raw]
            speeds = _compute_speeds(traj, timestamps)
            
            trial_cond = TRIAL_CONDITIONS[tid]
            records.append({
                "participant": pid,
                "trial_id": tid,
                "round": trial.get("round", 1),
                "tunnel_type": trial_cond["type"],
                "width": trial_cond["width"],
                "completion_time": ct,
                "avg_speed": _path_length(traj) / ct if ct > 0 else 0,
                "trajectory": traj,
                "speeds": speeds,
                "timestamps": timestamps,
            })
    
    return records


# ===================================================================
# Generate Centerlines
# ===================================================================

def generate_centerlines():
    """Generate tunnel centerlines for each condition."""
    from experiment.utils import generateTunnelPath, generateCornerPath
    
    centerlines = {}
    for tid, cond in TRIAL_CONDITIONS.items():
        if cond["type"] == "sigmoidal":
            path, _ = generateTunnelPath(
                width=cond["width"],
                curvature=cond["curvature"],
                start_x=0.0,
                end_x=0.46,
                y_base=0.13,
            )
        else:  # corner
            path, _ = generateCornerPath(
                width=cond["width"],
                start_x=0.0,
                end_x=0.46,
                y_base=0.13,
                num_corners=cond.get("num_corners", 2),
                corner_offset=cond.get("corner_offset", 0.1),
            )
        centerlines[tid] = path
    return centerlines


# ===================================================================
# Run Simulator
# ===================================================================

def _build_sigmoidal_config(width, curvature):
    """Create task config for a sigmoidal tunnel."""
    env_dict = {
        "env_type": "tunnel_steering_wave",
        "screen_width": 460,
        "screen_height": 260,
        "tunnelWidth": width,
        "curvature": curvature,
        "max_steps": 800,
        # Use tunnel width so the cursor reliably terminates at the tunnel end
        "target_radius": width * 0.5,
    }
    environment = create_environment(env_dict)
    task_config = generate_task_config(environment, include_constraints=True)
    centerline = environment["centerline"]
    return task_config, centerline


def _build_corner_config(width, num_corners, corner_offset):
    """Create task config for a corner tunnel."""
    env_dict = {
        "env_type": "tunnel_steering_corner",
        "screen_width": 460,
        "screen_height": 260,
        "tunnelWidth": width,
        "num_corners": num_corners,
        "corner_offset": corner_offset,
        "max_steps": 800,
        # Use tunnel width so the cursor reliably terminates at the tunnel end
        "target_radius": width * 0.5,
    }
    environment = create_environment(env_dict)
    task_config = generate_task_config(environment, include_constraints=True)
    centerline = environment["centerline"]
    return task_config, centerline


def run_simulator(user_config_path, rounds_per_condition=3):
    """Run simulator for all conditions.

    Returns:
        (records, oob_rounds) where oob_rounds is a list of dicts describing
        rounds that were excluded due to persistent out-of-bounds violations.
    """
    import tempfile

    sim = CursorSimulator(str(user_config_path))
    interval = sim.interval

    records = []
    oob_rounds = []
    for tid, cond in TRIAL_CONDITIONS.items():
        # Create environment and task config
        if cond["type"] == "sigmoidal":
            task_config, centerline = _build_sigmoidal_config(
                cond["width"], cond["curvature"]
            )
        else:
            task_config, centerline = _build_corner_config(
                cond["width"],
                cond.get("num_corners", 2),
                cond.get("corner_offset", 0.1)
            )

        half_width = cond["width"] / 2.0

        # Write task config to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
            json.dump(task_config, tf)
            task_file = tf.name

        try:
            for r in range(1, rounds_per_condition + 1):
                # Run simulation
                traj_raw = sim.generate_trajectory_with_waypoints(
                    task_file=task_file,
                    max_steps=task_config.get("max_steps", 800),
                    target_radius=task_config.get("target_radius", 0.01),
                    use_optimal_path=True,
                )

                # Convert from pixels to meters (1 pixel = 0.001 m)
                scale = 0.001
                traj = [[x * scale, y * scale] for x, y, _ in traj_raw]

                # Check for persistent out-of-bounds violations
                is_oob, oob_steps, max_consec = check_trajectory_oob(
                    traj, centerline, half_width, max_consecutive=5)
                if is_oob:
                    oob_rounds.append({
                        "trial_id": tid,
                        "round": r,
                        "label": cond["label"],
                        "oob_steps": oob_steps,
                        "max_consecutive": max_consec,
                    })
                    continue

                # Compute speeds
                speeds = []
                for i in range(1, len(traj)):
                    dx = traj[i][0] - traj[i-1][0]
                    dy = traj[i][1] - traj[i-1][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    speeds.append(dist / interval)

                ct = len(traj) * interval

                records.append({
                    "participant": "model",
                    "trial_id": tid,
                    "round": r,
                    "tunnel_type": cond["type"],
                    "width": cond["width"],
                    "completion_time": ct,
                    "avg_speed": _path_length(traj) / ct if ct > 0 else 0,
                    "trajectory": traj,
                    "speeds": speeds,
                })
        finally:
            os.unlink(task_file)

    return records, oob_rounds


# ===================================================================
# Compute Metrics
# ===================================================================

def compute_stop_and_go_metrics(records, centerlines):
    """Compute stop-and-go metrics for all records."""
    for r in records:
        speeds = r.get("speeds", [])
        traj = r.get("trajectory", [])
        tid = r.get("trial_id")
        centerline = centerlines.get(tid, [])
        
        # Basic metrics
        r["speed_cv"] = compute_speed_cv(speeds)
        r["min_speed_ratio"] = compute_min_speed_ratio(speeds)
        r["speed_smoothness"] = compute_speed_smoothness(speeds)
        
        # Dip metrics
        dip_metrics = compute_speed_dip_metrics(speeds)
        r["dip_count"] = dip_metrics["dip_count"]
        r["dip_depth_mean"] = dip_metrics["dip_depth_mean"]
        r["dip_depth_max"] = dip_metrics["dip_depth_max"]
        
        # Corner slowdown (only for corner tunnels)
        if r["tunnel_type"] == "corner" and centerline:
            progress = compute_progress_along_path(traj, centerline)
            r["corner_slowdown"] = compute_corner_slowdown_factor(
                speeds, progress, CORNER_PROGRESS_LOCATIONS
            )
            r["progress"] = progress
        else:
            r["corner_slowdown"] = 1.0
            r["progress"] = []
    
    return records


def summarize_by_condition(records):
    """Aggregate metrics by tunnel type and width, with mean and std."""
    groups = defaultdict(list)
    for r in records:
        key = (r["tunnel_type"], r["width"])
        groups[key].append(r)

    summary = []
    for (ttype, width), group in groups.items():
        speed_cvs = [r["speed_cv"] for r in group]
        min_ratios = [r["min_speed_ratio"] for r in group]
        dip_counts = [r["dip_count"] for r in group]
        dip_depths = [r["dip_depth_mean"] for r in group]
        avg_speeds = [r["avg_speed"] for r in group]

        summary.append({
            "tunnel_type": ttype,
            "width": width,
            "n_trials": len(group),
            # Raw per-trial values
            "speed_cv_values": speed_cvs,
            "dip_count_values": dip_counts,
            "min_ratio_values": min_ratios,
            # Point estimates
            "speed_cv_mean": np.mean(speed_cvs),
            "speed_cv_std": np.std(speed_cvs),
            "min_ratio_mean": np.mean(min_ratios),
            "min_ratio_std": np.std(min_ratios),
            "dip_count_mean": np.mean(dip_counts),
            "dip_count_std": np.std(dip_counts),
            "dip_depth_mean": np.mean(dip_depths),
            "smoothness_mean": np.mean([r["speed_smoothness"] for r in group]),
            "avg_speed_mean": np.mean(avg_speeds),
            "corner_slowdown_mean": np.mean([r["corner_slowdown"] for r in group]),
        })

    return summary



# ===================================================================
# Plotting Functions
# ===================================================================

def _ensure_results_dir():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)



def plot_speed_cv_comparison(human_summary, model_summary):
    """Bar chart comparing SpeedCV with std error bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ensure_results_dir()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, width, title in [
        (axes[0], 0.02, "Narrow Tunnel (W=20mm)"),
        (axes[1], 0.04, "Wide Tunnel (W=40mm)"),
    ]:
        ax.set_title(title, fontsize=12)

        types = ["sigmoidal", "corner"]
        x = np.arange(len(types))
        bar_w = 0.35

        for offset, summary, label, color in [
            (-bar_w / 2, human_summary, "Human", "#4287f5"),
            (bar_w / 2, model_summary, "Model", "#e8453c"),
        ]:
            means, stds = [], []
            for t in types:
                row = next((r for r in summary if r["tunnel_type"] == t and r["width"] == width), None)
                if row:
                    means.append(row["speed_cv_mean"])
                    stds.append(row["speed_cv_std"])
                else:
                    means.append(0)
                    stds.append(0)
            ax.bar(x + offset, means, bar_w, yerr=stds, label=label,
                   color=color, alpha=0.8, capsize=4, error_kw={"lw": 1.2})

        ax.set_xticks(x)
        ax.set_xticklabels(["Sigmoidal\n(smooth curves)", "Corner\n(90\u00b0 turns)"])
        ax.set_ylabel("Speed CV (std/mean)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Speed Variability: Higher = More Stop-and-Go\n(Error bars = \u00b11 std)", fontsize=14)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "stop_and_go_speed_cv.png", dpi=300)
    fig.savefig(RESULTS_DIR / "stop_and_go_speed_cv.pdf")
    plt.close(fig)
    print(f"  Saved {RESULTS_DIR / 'stop_and_go_speed_cv.png'} (.pdf)")


def plot_min_speed_ratio(human_summary, model_summary):
    """Bar chart comparing min speed ratio."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    _ensure_results_dir()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax, width, title in [
        (axes[0], 0.02, "Narrow Tunnel (W=20mm)"),
        (axes[1], 0.04, "Wide Tunnel (W=40mm)"),
    ]:
        ax.set_title(title, fontsize=12)
        
        types = ["sigmoidal", "corner"]
        x = np.arange(len(types))
        bar_w = 0.35
        
        for offset, summary, label, color in [
            (-bar_w/2, human_summary, "Human", "#4287f5"),
            (bar_w/2, model_summary, "Model", "#e8453c"),
        ]:
            means, stds = [], []
            for t in types:
                row = next((r for r in summary if r["tunnel_type"] == t and r["width"] == width), None)
                means.append(row["min_ratio_mean"] if row else 0)
                stds.append(row["min_ratio_std"] if row else 0)
            ax.bar(x + offset, means, bar_w, yerr=stds, label=label,
                   color=color, alpha=0.8, capsize=4)
        
        ax.set_xticks(x)
        ax.set_xticklabels(["Sigmoidal\n(smooth curves)", "Corner\n(90° turns)"])
        ax.set_ylabel("Min Speed / Mean Speed")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    
    fig.suptitle("Minimum Speed Ratio: Lower = Stronger Stops", fontsize=14)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "stop_and_go_min_ratio.png", dpi=200)
    plt.close(fig)
    print(f"  Saved {RESULTS_DIR / 'stop_and_go_min_ratio.png'}")


def plot_speed_profiles_by_progress(human_records, model_records, centerlines):
    """Plot speed vs progress for all conditions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
    _ensure_results_dir()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    n_bins = 20
    progress_bins = np.linspace(0, 1, n_bins + 1)
    progress_centers = (progress_bins[:-1] + progress_bins[1:]) / 2.0
    
    for ax, tid in zip(axes.flat, sorted(TRIAL_CONDITIONS.keys())):
        cond = TRIAL_CONDITIONS[tid]
        ttype = cond["type"]
        width = cond["width"]
        ax.set_title(f"{ttype.capitalize()} W={int(width*1000)}mm", fontsize=12)
        
        centerline = centerlines.get(tid, [])
        if not centerline:
            continue
        
        cl_np = np.array(centerline)
        all_data = []
        
        # Process human records
        for rec in human_records:
            if rec["trial_id"] != tid:
                continue
            traj = rec.get("trajectory", [])
            speeds = rec.get("speeds", [])
            if len(traj) < 2 or len(speeds) < 1:
                continue
            
            progress = compute_progress_along_path(traj, centerline)
            for i, speed in enumerate(speeds):
                if i + 1 < len(progress):
                    all_data.append({"Progress": progress[i+1], "Speed (m/s)": speed, "Type": "Human"})
        
        # Process model records
        for rec in model_records:
            if rec["trial_id"] != tid:
                continue
            traj = rec.get("trajectory", [])
            speeds = rec.get("speeds", [])
            if len(traj) < 2 or len(speeds) < 1:
                continue
            
            progress = compute_progress_along_path(traj, centerline)
            for i, speed in enumerate(speeds):
                if i + 1 < len(progress):
                    all_data.append({"Progress": progress[i+1], "Speed (m/s)": speed, "Type": "Model"})
        
        if all_data:
            df = pd.DataFrame(all_data)
            df["Progress_bin"] = pd.cut(df["Progress"], bins=progress_bins, 
                                        labels=progress_centers, include_lowest=True)
            df["Progress_bin"] = df["Progress_bin"].astype(float)
            
            palette = {"Human": "#4287f5", "Model": "#e8453c"}
            sns.lineplot(data=df, x="Progress_bin", y="Speed (m/s)", hue="Type",
                        ax=ax, errorbar="sd", palette=palette, linewidth=1.5)
            ax.legend(fontsize=9)
        
        # Add vertical lines at corner locations for corner tunnels
        if ttype == "corner":
            for loc in CORNER_PROGRESS_LOCATIONS:
                ax.axvline(x=loc, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.set_xlabel("Progress along path")
        ax.set_ylabel("Speed (m/s)")
        ax.grid(True, alpha=0.3)
    
    fig.suptitle("Speed vs Progress: Dashed lines = Corner Locations", fontsize=14)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "stop_and_go_speed_progress.png", dpi=200)
    plt.close(fig)
    print(f"  Saved {RESULTS_DIR / 'stop_and_go_speed_progress.png'}")


def plot_dip_analysis(human_summary, model_summary):
    """Plot speed dip count with std error bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ensure_results_dir()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    conditions = [("sigmoidal", 0.02), ("corner", 0.02), ("sigmoidal", 0.04), ("corner", 0.04)]
    labels = ["Sig\nW=20", "Cor\nW=20", "Sig\nW=40", "Cor\nW=40"]
    x = np.arange(len(conditions))
    bar_w = 0.35

    # Dip Count with std
    ax = axes[0]
    ax.set_title("Speed Dip Count", fontsize=12)

    for offset, summary, label, color in [
        (-bar_w / 2, human_summary, "Human", "#4287f5"),
        (bar_w / 2, model_summary, "Model", "#e8453c"),
    ]:
        means, stds = [], []
        for ttype, width in conditions:
            row = next((r for r in summary if r["tunnel_type"] == ttype and r["width"] == width), None)
            if row:
                means.append(row["dip_count_mean"])
                stds.append(row["dip_count_std"])
            else:
                means.append(0)
                stds.append(0)
        ax.bar(x + offset, means, bar_w, yerr=stds, label=label,
               color=color, alpha=0.8, capsize=4, error_kw={"lw": 1.2})

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Number of Speed Dips")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Dip Depth (unchanged — no per-trial CI data for depth)
    ax = axes[1]
    ax.set_title("Speed Dip Depth (mean)", fontsize=12)

    for offset, summary, label, color in [
        (-bar_w / 2, human_summary, "Human", "#4287f5"),
        (bar_w / 2, model_summary, "Model", "#e8453c"),
    ]:
        vals = []
        for ttype, width in conditions:
            row = next((r for r in summary if r["tunnel_type"] == ttype and r["width"] == width), None)
            vals.append(row["dip_depth_mean"] if row else 0)
        ax.bar(x + offset, vals, bar_w, label=label, color=color, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Dip Depth (fraction of mean)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Speed Dip Analysis (Error bars = \u00b11 std)", fontsize=14)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "stop_and_go_dip_analysis.png", dpi=300)
    fig.savefig(RESULTS_DIR / "stop_and_go_dip_analysis.pdf")
    plt.close(fig)
    print(f"  Saved {RESULTS_DIR / 'stop_and_go_dip_analysis.png'} (.pdf)")


def plot_effect_summary(human_summary, model_summary):
    """Summary plot showing stop-and-go effect size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    _ensure_results_dir()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Compute effect: corner - sigmoidal for same width
    effects = []
    for width in [0.02, 0.04]:
        for summary, source in [(human_summary, "Human"), (model_summary, "Model")]:
            sig_row = next((r for r in summary if r["tunnel_type"] == "sigmoidal" and r["width"] == width), None)
            cor_row = next((r for r in summary if r["tunnel_type"] == "corner" and r["width"] == width), None)
            
            if sig_row and cor_row:
                effects.append({
                    "width": f"W={int(width*1000)}mm",
                    "source": source,
                    "speed_cv_diff": cor_row["speed_cv_mean"] - sig_row["speed_cv_mean"],
                    "min_ratio_diff": sig_row["min_ratio_mean"] - cor_row["min_ratio_mean"],
                })
    
    # Plot
    x = np.arange(2)
    bar_w = 0.2
    
    for i, width_label in enumerate(["W=20mm", "W=40mm"]):
        human_eff = next((e for e in effects if e["width"] == width_label and e["source"] == "Human"), None)
        model_eff = next((e for e in effects if e["width"] == width_label and e["source"] == "Model"), None)
        
        if human_eff:
            ax.bar(i - bar_w/2, human_eff["speed_cv_diff"], bar_w, 
                   label="Human" if i == 0 else None, color="#4287f5", alpha=0.8)
        if model_eff:
            ax.bar(i + bar_w/2, model_eff["speed_cv_diff"], bar_w,
                   label="Model" if i == 0 else None, color="#e8453c", alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(["Narrow (W=20mm)", "Wide (W=40mm)"])
    ax.set_ylabel("SpeedCV Difference (Corner - Sigmoidal)")
    ax.set_title("Stop-and-Go Effect Size\n(Positive = Corner has more stop-and-go than Sigmoidal)", fontsize=12)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "stop_and_go_effect_summary.png", dpi=200)
    plt.close(fig)
    print(f"  Saved {RESULTS_DIR / 'stop_and_go_effect_summary.png'}")


def save_csv(human_records, model_records):
    """Save detailed results to CSV."""
    _ensure_results_dir()
    path = RESULTS_DIR / "stop_and_go_results.csv"
    
    fields = ["source", "participant", "trial_id", "round", "tunnel_type", "width",
              "completion_time", "avg_speed", "speed_cv", "min_speed_ratio",
              "dip_count", "dip_depth_mean", "dip_depth_max", "speed_smoothness",
              "corner_slowdown"]
    
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in human_records + model_records:
            writer.writerow({
                "source": "human" if r["participant"] != "model" else "model",
                "participant": r["participant"],
                "trial_id": r["trial_id"],
                "round": r["round"],
                "tunnel_type": r["tunnel_type"],
                "width": r["width"],
                "completion_time": f"{r['completion_time']:.3f}",
                "avg_speed": f"{r['avg_speed']:.4f}",
                "speed_cv": f"{r.get('speed_cv', 0):.3f}",
                "min_speed_ratio": f"{r.get('min_speed_ratio', 0):.3f}",
                "dip_count": r.get("dip_count", 0),
                "dip_depth_mean": f"{r.get('dip_depth_mean', 0):.3f}",
                "dip_depth_max": f"{r.get('dip_depth_max', 0):.3f}",
                "speed_smoothness": f"{r.get('speed_smoothness', 0):.3f}",
                "corner_slowdown": f"{r.get('corner_slowdown', 1.0):.3f}",
            })
    
    print(f"  Saved {path}")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Stop-and-Go Behavior Evaluation")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG),
                        help="Path to user configuration JSON")
    parser.add_argument("--rounds", type=int, default=3,
                        help="Number of simulator rounds per condition")
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)
    
    # Load config for display
    with open(config_path) as f:
        cfg = json.load(f)
    weights = cfg.get("planner_weights", {})
    
    print("=" * 70)
    print("Stop-and-Go Behavior Evaluation")
    print("=" * 70)
    print(f"  Config:           {config_path}")
    print(f"  desired_speed:    {weights.get('desired_speed', 0.12)}")
    print(f"  rate_percentile:  {weights.get('rate_percentile', 85.0)}")
    print(f"  rate_scale:       {weights.get('rate_scale', 1000.0)}")
    print(f"  rate_floor:       {weights.get('rate_floor', 0.3)}")
    print(f"  Rounds/cond:      {args.rounds}")
    print()
    
    # Step 1: Generate centerlines
    print("[1/6] Generating centerlines ...")
    centerlines = generate_centerlines()
    print(f"       Generated centerlines for {len(centerlines)} conditions")
    
    # Step 2: Load human data
    print("[2/6] Loading human data ...")
    human_records = load_human_data(trackpad_confidence=None)
    n_participants = len(set(r["participant"] for r in human_records))
    print(f"       Loaded {len(human_records)} trials from {n_participants} participants")
    
    # Step 3: Compute human metrics
    print("[3/6] Computing human stop-and-go metrics ...")
    human_records = compute_stop_and_go_metrics(human_records, centerlines)
    human_summary = summarize_by_condition(human_records)
    
    # Step 4: Run simulator
    print("[4/6] Running simulator ...")
    model_records, oob_rounds = run_simulator(config_path, rounds_per_condition=args.rounds)
    total_model_rounds = len(model_records) + len(oob_rounds)
    print(f"       Generated {len(model_records)} valid model trials"
          f" ({len(oob_rounds)} OOB out of {total_model_rounds})")
    if oob_rounds:
        for oob in oob_rounds:
            print(f"         OOB: {oob['label']} round {oob['round']}"
                  f" ({oob['max_consecutive']} consecutive steps)")
    
    # Step 5: Compute model metrics
    print("[5/6] Computing model stop-and-go metrics ...")
    model_records = compute_stop_and_go_metrics(model_records, centerlines)
    model_summary = summarize_by_condition(model_records)

    # Print summary table with mean +/- std
    print()
    print("  Stop-and-Go Metrics by Condition (mean \u00b1 std):")
    print("  " + "=" * 90)
    fmt_ms = lambda m, s: f"{m:.3f} \u00b1 {s:.3f}"

    for ttype in ["sigmoidal", "corner"]:
        for width in [0.02, 0.04]:
            h = next((r for r in human_summary if r["tunnel_type"] == ttype and r["width"] == width), None)
            m = next((r for r in model_summary if r["tunnel_type"] == ttype and r["width"] == width), None)
            if not h or not m:
                continue

            label = f"{ttype} W={int(width*1000)}mm"
            print(f"\n  {label} (Human n={h['n_trials']}, Model n={m['n_trials']})")
            print(f"    SpeedCV:  Human {fmt_ms(h['speed_cv_mean'], h['speed_cv_std'])}  "
                  f"Model {fmt_ms(m['speed_cv_mean'], m['speed_cv_std'])}")
            print(f"    DipCount: Human {fmt_ms(h['dip_count_mean'], h['dip_count_std'])}  "
                  f"Model {fmt_ms(m['dip_count_mean'], m['dip_count_std'])}")
            print(f"    MinRatio: Human {fmt_ms(h['min_ratio_mean'], h['min_ratio_std'])}  "
                  f"Model {fmt_ms(m['min_ratio_mean'], m['min_ratio_std'])}")

    # Compute effect sizes (Corner - Sigmoidal)
    print()
    print("  Stop-and-Go Effect (Corner - Sigmoidal SpeedCV difference):")
    for width in [0.02, 0.04]:
        h_sig = next((r for r in human_summary if r["tunnel_type"] == "sigmoidal" and r["width"] == width), None)
        h_cor = next((r for r in human_summary if r["tunnel_type"] == "corner" and r["width"] == width), None)
        m_sig = next((r for r in model_summary if r["tunnel_type"] == "sigmoidal" and r["width"] == width), None)
        m_cor = next((r for r in model_summary if r["tunnel_type"] == "corner" and r["width"] == width), None)

        if h_sig and h_cor and m_sig and m_cor:
            h_diff = h_cor["speed_cv_mean"] - h_sig["speed_cv_mean"]
            m_diff = m_cor["speed_cv_mean"] - m_sig["speed_cv_mean"]
            capture_pct = (m_diff / h_diff * 100) if h_diff > 0 else 0
            print(f"    W={int(width*1000)}mm: Human \u0394={h_diff:+.3f}, Model \u0394={m_diff:+.3f} "
                  f"({capture_pct:.0f}% captured)")

    # Step 6: Generate plots
    print()
    print("[6/6] Generating plots ...")
    plot_speed_cv_comparison(human_summary, model_summary)
    plot_min_speed_ratio(human_summary, model_summary)
    plot_speed_profiles_by_progress(human_records, model_records, centerlines)
    plot_dip_analysis(human_summary, model_summary)
    plot_effect_summary(human_summary, model_summary)
    save_csv(human_records, model_records)

    print()
    print("Done.")


if __name__ == "__main__":
    main()

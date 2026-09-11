"""
Corner-Cutting Strategy Evaluation Experiment.

Compares corner-cutting behavior between sigmoidal and corner tunnels with
the same width. Evaluates how well the model captures human strategy differences:

- Sigmoidal tunnels: Smooth curves encourage corner cutting (shorter paths)
- Corner tunnels: Sharp 90-degree turns encourage stop-and-go (follow centerline)

This aligns with Pastel (2006)'s observation of two distinct steering modes.

Usage:
    python run_eval.py [--config path/to/user_config.json] [--rounds N]
"""

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from collections import defaultdict

import numpy as np

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

TRIAL_CONDITIONS = {
    1: {"type": "sigmoidal", "width": 0.02, "curvature": 0.025, "label": "sigmoidal narrow"},
    2: {"type": "sigmoidal", "width": 0.04, "curvature": 0.025, "label": "sigmoidal wide"},
    3: {"type": "corner", "width": 0.02, "num_corners": 2, "corner_offset": 0.1, "label": "corner narrow"},
    4: {"type": "corner", "width": 0.04, "num_corners": 2, "corner_offset": 0.1, "label": "corner wide"},
}

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


# ===================================================================
# Corner-Cutting Metrics
# ===================================================================

def compute_path_efficiency(trajectory, centerline):
    """
    Compute path efficiency = centerline_length / actual_path_length.
    
    Values > 1.0 indicate corner cutting (trajectory shorter than centerline).
    Values ~ 1.0 indicate centerline following.
    Values < 1.0 indicate path overshooting.
    """
    traj_length = _path_length(trajectory)
    cl_length = _path_length(centerline)
    if traj_length <= 0:
        return 1.0
    return cl_length / traj_length


def compute_lateral_deviation(trajectory, centerline):
    """
    Compute RMS lateral deviation from centerline.
    
    For each trajectory point, find the closest point on centerline
    and compute the distance. Return the RMS of all distances.
    
    Higher values indicate more cutting (deviating from center).
    """
    if not trajectory or not centerline:
        return 0.0
    
    centerline_np = np.array(centerline)
    deviations = []
    
    for pt in trajectory:
        pt_np = np.array(pt)
        distances = np.linalg.norm(centerline_np - pt_np, axis=1)
        min_dist = np.min(distances)
        deviations.append(min_dist)
    
    return float(np.sqrt(np.mean(np.array(deviations) ** 2)))


def compute_speed_variability(speeds):
    """
    Compute coefficient of variation (CV) of speed.
    
    Higher CV indicates more stop-and-go behavior.
    Lower CV indicates more consistent speed (corner cutting).
    """
    if not speeds or len(speeds) < 2:
        return 0.0
    speeds_arr = np.array(speeds)
    mean_spd = np.mean(speeds_arr)
    if mean_spd <= 0:
        return 0.0
    return float(np.std(speeds_arr) / mean_spd)


def compute_min_speed_ratio(speeds):
    """
    Compute ratio of minimum speed to mean speed.
    
    Lower ratio indicates sharper slowdowns (stop-and-go).
    Higher ratio indicates more consistent speed (corner cutting).
    """
    if not speeds or len(speeds) < 2:
        return 1.0
    speeds_arr = np.array(speeds)
    mean_spd = np.mean(speeds_arr)
    min_spd = np.min(speeds_arr)
    if mean_spd <= 0:
        return 1.0
    return float(min_spd / mean_spd)


# ===================================================================
# Load Human Data
# ===================================================================

def get_trackpad_participants(min_confidence=0.7):
    """Get set of participant IDs that are classified as trackpad users.
    
    Args:
        min_confidence: Minimum confidence threshold for trackpad classification
        
    Returns:
        Set of participant IDs using trackpad with confidence >= threshold
    """
    trackpad_pids = set()
    for fpath in sorted(HUMAN_DATA_DIR.glob("*.json")):
        result = check_participant(fpath)
        if (result.get("device_type") == "trackpad" and 
            result.get("confidence", 0) >= min_confidence):
            trackpad_pids.add(result.get("participant_id"))
    return trackpad_pids


def load_human_data(trackpad_confidence=None):
    """Load trials 1-4 (sigmoidal and corner) from raw participant JSONs.
    
    Args:
        trackpad_confidence: If provided, filter to only include participants
            classified as trackpad users with confidence >= this threshold.
            If None, include all participants.
    
    Returns a list of dicts, each with keys:
        participant, trial_id, round, tunnel_type, width,
        completion_time, avg_speed, path_length, speeds, trajectory
    """
    # Get trackpad participants if filtering is enabled
    trackpad_pids = None
    if trackpad_confidence is not None:
        trackpad_pids = get_trackpad_participants(trackpad_confidence)
    
    records = []
    for fpath in sorted(HUMAN_DATA_DIR.glob("*.json")):
        with open(fpath) as f:
            data = json.load(f)
        pid = data.get("participantId", fpath.stem)
        
        # Skip non-trackpad participants if filtering is enabled
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
            
            if not traj_raw or not timestamps or ct <= 0:
                continue
            
            traj = [[p["x"], p["y"]] if isinstance(p, dict) else p for p in traj_raw]
            pl = _path_length(traj)
            speeds = _compute_speeds(traj, timestamps)
            
            trial_cond = TRIAL_CONDITIONS[tid]
            records.append({
                "participant": pid,
                "trial_id": tid,
                "round": trial.get("round", 1),
                "tunnel_type": trial_cond["type"],
                "width": trial_cond["width"],
                "completion_time": ct,
                "avg_speed": pl / ct if ct > 0 else 0.0,
                "path_length": pl,
                "speeds": speeds,
                "trajectory": traj,
            })
    
    return records


# ===================================================================
# Build Task Configs and Run Simulator
# ===================================================================

def _build_sigmoidal_config(width, curvature):
    """Create task config for a sigmoidal tunnel."""
    env_dict = {
        "env_type": "tunnel_steering_smooth",
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


def run_simulator_trials(config_path, n_rounds=3):
    """Run the simulator for each tunnel condition, n_rounds per condition.

    Returns:
        (records, oob_rounds) where oob_rounds lists rounds excluded due to
        persistent out-of-bounds violations.
    """
    sim = CursorSimulator(str(config_path))
    interval = sim.interval
    records = []
    oob_rounds = []

    for tid, cond in TRIAL_CONDITIONS.items():
        tunnel_type = cond["type"]
        width = cond["width"]
        half_width = width / 2.0

        if tunnel_type == "sigmoidal":
            task_config, centerline = _build_sigmoidal_config(width, cond["curvature"])
        else:  # corner
            task_config, centerline = _build_corner_config(
                width, cond["num_corners"], cond["corner_offset"]
            )

        cl_length = _path_length(centerline)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
            json.dump(task_config, tf)
            task_file = tf.name

        try:
            for rnd in range(1, n_rounds + 1):
                traj_raw = sim.generate_trajectory_with_waypoints(
                    task_file=task_file,
                    max_steps=task_config.get("max_steps", 800),
                    target_radius=task_config.get("target_radius", 0.01),
                    use_optimal_path=True,
                )

                # Model outputs in pixels; convert to meters (1 pixel = 0.001 m)
                scale = 0.001
                traj = [[x * scale, y * scale] for x, y, _ in traj_raw]

                # Check for persistent out-of-bounds violations
                is_oob, oob_steps, max_consec = check_trajectory_oob(
                    traj, centerline, half_width, max_consecutive=5)
                if is_oob:
                    oob_rounds.append({
                        "trial_id": tid,
                        "round": rnd,
                        "label": cond["label"],
                        "oob_steps": oob_steps,
                        "max_consecutive": max_consec,
                    })
                    continue

                pl = _path_length(traj)
                ct = len(traj_raw) * interval

                n_pts = len(traj)
                speeds = []
                for i in range(1, n_pts):
                    d = math.sqrt(
                        (traj[i][0] - traj[i - 1][0]) ** 2
                        + (traj[i][1] - traj[i - 1][1]) ** 2
                    )
                    speeds.append(d / interval)

                # Centerline from create_environment is already in meters
                cl_converted = [[x, y] for x, y in centerline]

                records.append({
                    "participant": "model",
                    "trial_id": tid,
                    "round": rnd,
                    "tunnel_type": tunnel_type,
                    "width": width,
                    "completion_time": ct,
                    "avg_speed": pl / ct if ct > 0 else 0.0,
                    "path_length": pl,
                    "speeds": speeds,
                    "trajectory": traj,
                    "centerline": cl_converted,
                    "centerline_length": _path_length(cl_converted),
                })
        finally:
            os.unlink(task_file)

    return records, oob_rounds


# ===================================================================
# Compute Strategy Metrics
# ===================================================================

def compute_strategy_metrics(records, centerlines=None):
    """
    Compute corner-cutting strategy metrics for each record.
    
    Args:
        records: List of trial records
        centerlines: Dict mapping trial_id -> centerline (for human data)
    
    Returns:
        Updated records with added metrics
    """
    for r in records:
        traj = r.get("trajectory", [])
        speeds = r.get("speeds", [])
        
        # Get centerline
        if "centerline" in r:
            centerline = r["centerline"]
        elif centerlines and r["trial_id"] in centerlines:
            centerline = centerlines[r["trial_id"]]
        else:
            centerline = None
        
        # Compute metrics
        if centerline:
            r["path_efficiency"] = compute_path_efficiency(traj, centerline)
            r["lateral_deviation"] = compute_lateral_deviation(traj, centerline)
        else:
            r["path_efficiency"] = 1.0
            r["lateral_deviation"] = 0.0
        
        r["speed_cv"] = compute_speed_variability(speeds)
        r["min_speed_ratio"] = compute_min_speed_ratio(speeds)
    
    return records


def compute_summary(records, source_label):
    """Aggregate records by tunnel type → mean/std of strategy metrics."""
    by_type = defaultdict(list)
    for r in records:
        by_type[r["tunnel_type"]].append(r)
    
    rows = []
    for ttype in ["sigmoidal", "corner"]:
        group = by_type.get(ttype, [])
        if not group:
            continue
        
        rows.append({
            "source": source_label,
            "tunnel_type": ttype,
            "n": len(group),
            "MT_mean": np.mean([r["completion_time"] for r in group]),
            "MT_std": np.std([r["completion_time"] for r in group]),
            "path_eff_mean": np.mean([r.get("path_efficiency", 1.0) for r in group]),
            "path_eff_std": np.std([r.get("path_efficiency", 1.0) for r in group]),
            "lat_dev_mean": np.mean([r.get("lateral_deviation", 0.0) for r in group]),
            "lat_dev_std": np.std([r.get("lateral_deviation", 0.0) for r in group]),
            "speed_cv_mean": np.mean([r.get("speed_cv", 0.0) for r in group]),
            "speed_cv_std": np.std([r.get("speed_cv", 0.0) for r in group]),
            "min_spd_ratio_mean": np.mean([r.get("min_speed_ratio", 1.0) for r in group]),
            "min_spd_ratio_std": np.std([r.get("min_speed_ratio", 1.0) for r in group]),
        })
    
    return rows


def compute_width_summary(records, source_label):
    """Aggregate records by tunnel type AND width → compare narrow vs wide."""
    by_type_width = defaultdict(list)
    for r in records:
        key = (r["tunnel_type"], r["width"])
        by_type_width[key].append(r)
    
    rows = []
    for ttype in ["sigmoidal", "corner"]:
        for width in [0.02, 0.04]:  # narrow, wide
            group = by_type_width.get((ttype, width), [])
            if not group:
                continue
            
            width_label = "narrow" if width < 0.03 else "wide"
            rows.append({
                "source": source_label,
                "tunnel_type": ttype,
                "width": width,
                "width_label": width_label,
                "n": len(group),
                "MT_mean": np.mean([r["completion_time"] for r in group]),
                "path_eff_mean": np.mean([r.get("path_efficiency", 1.0) for r in group]),
                "lat_dev_mean": np.mean([r.get("lateral_deviation", 0.0) for r in group]),
                "speed_cv_mean": np.mean([r.get("speed_cv", 0.0) for r in group]),
                "avg_speed_mean": np.mean([r.get("avg_speed", 0.0) for r in group]),
            })
    
    return rows


# ===================================================================
# Generate Centerlines for Human Data
# ===================================================================

def generate_centerlines():
    """Generate centerlines for each trial condition.
    
    Returns centerlines in meters to match human trajectory data format.
    """
    centerlines = {}
    
    for tid, cond in TRIAL_CONDITIONS.items():
        if cond["type"] == "sigmoidal":
            task_config, centerline = _build_sigmoidal_config(
                cond["width"], cond["curvature"]
            )
        else:
            task_config, centerline = _build_corner_config(
                cond["width"], cond["num_corners"], cond["corner_offset"]
            )
        
        # Centerline from create_environment is already in meters, same as human data
        centerlines[tid] = [[x, y] for x, y in centerline]
    
    return centerlines


# ===================================================================
# Plots
# ===================================================================

def _ensure_results_dir():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_path_efficiency(human_summary, model_summary):
    """Grouped bar chart of path efficiency by tunnel type."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    _ensure_results_dir()
    fig, ax = plt.subplots(figsize=(8, 5))
    
    types = ["sigmoidal", "corner"]
    x = np.arange(len(types))
    bar_w = 0.35
    
    for offset, summary, label, color in [
        (-bar_w/2, human_summary, "Human", "#4287f5"),
        (bar_w/2, model_summary, "Model", "#e8453c"),
    ]:
        means, stds = [], []
        for t in types:
            row = next((r for r in summary if r["tunnel_type"] == t), None)
            means.append(row["path_eff_mean"] if row else 1.0)
            stds.append(row["path_eff_std"] if row else 0.0)
        ax.bar(x + offset, means, bar_w, yerr=stds, label=label,
               color=color, alpha=0.8, capsize=4)
    
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Centerline following')
    ax.set_xticks(x)
    ax.set_xticklabels(["Sigmoidal\n(smooth curves)", "Corner\n(90° turns)"])
    ax.set_ylabel("Path Efficiency (centerline / actual)")
    ax.set_title("Path Efficiency by Tunnel Type\n(>1 = cutting, ~1 = following)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0.8, 1.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "strategy_path_efficiency.png", dpi=200)
    plt.close(fig)
    print(f"  Saved {RESULTS_DIR / 'strategy_path_efficiency.png'}")


def plot_lateral_deviation(human_summary, model_summary):
    """Grouped bar chart of lateral deviation by tunnel type."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    _ensure_results_dir()
    fig, ax = plt.subplots(figsize=(8, 5))
    
    types = ["sigmoidal", "corner"]
    x = np.arange(len(types))
    bar_w = 0.35
    
    for offset, summary, label, color in [
        (-bar_w/2, human_summary, "Human", "#4287f5"),
        (bar_w/2, model_summary, "Model", "#e8453c"),
    ]:
        means, stds = [], []
        for t in types:
            row = next((r for r in summary if r["tunnel_type"] == t), None)
            means.append(row["lat_dev_mean"] * 1000 if row else 0)  # Convert to mm
            stds.append(row["lat_dev_std"] * 1000 if row else 0)
        ax.bar(x + offset, means, bar_w, yerr=stds, label=label,
               color=color, alpha=0.8, capsize=4)
    
    ax.set_xticks(x)
    ax.set_xticklabels(["Sigmoidal\n(smooth curves)", "Corner\n(90° turns)"])
    ax.set_ylabel("RMS Lateral Deviation (mm)")
    ax.set_title("Lateral Deviation from Centerline by Tunnel Type\n(Higher = more cutting)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "strategy_lateral_deviation.png", dpi=200)
    plt.close(fig)
    print(f"  Saved {RESULTS_DIR / 'strategy_lateral_deviation.png'}")


def plot_speed_variability(human_summary, model_summary):
    """Grouped bar chart of speed CV by tunnel type."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    _ensure_results_dir()
    fig, ax = plt.subplots(figsize=(8, 5))
    
    types = ["sigmoidal", "corner"]
    x = np.arange(len(types))
    bar_w = 0.35
    
    for offset, summary, label, color in [
        (-bar_w/2, human_summary, "Human", "#4287f5"),
        (bar_w/2, model_summary, "Model", "#e8453c"),
    ]:
        means, stds = [], []
        for t in types:
            row = next((r for r in summary if r["tunnel_type"] == t), None)
            means.append(row["speed_cv_mean"] if row else 0)
            stds.append(row["speed_cv_std"] if row else 0)
        ax.bar(x + offset, means, bar_w, yerr=stds, label=label,
               color=color, alpha=0.8, capsize=4)
    
    ax.set_xticks(x)
    ax.set_xticklabels(["Sigmoidal\n(smooth curves)", "Corner\n(90° turns)"])
    ax.set_ylabel("Speed Coefficient of Variation")
    ax.set_title("Speed Variability by Tunnel Type\n(Higher = more stop-and-go)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "strategy_speed_variability.png", dpi=200)
    plt.close(fig)
    print(f"  Saved {RESULTS_DIR / 'strategy_speed_variability.png'}")


def plot_speed_profiles(human_records, model_records, interval=0.05):
    """Overlay mean speed profiles for sigmoidal vs corner tunnels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    _ensure_results_dir()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    
    for ax, ttype, title in [
        (axes[0], "sigmoidal", "Sigmoidal Tunnel (Smooth Curves)"),
        (axes[1], "corner", "Corner Tunnel (90° Turns)"),
    ]:
        ax.set_title(title)
        
        for records, label, color in [
            (human_records, "Human", "#4287f5"),
            (model_records, "Model", "#e8453c"),
        ]:
            group_speeds = [r["speeds"] for r in records 
                          if r["tunnel_type"] == ttype and r["speeds"]]
            if not group_speeds:
                continue
            
            max_len = max(len(s) for s in group_speeds)
            padded = np.full((len(group_speeds), max_len), np.nan)
            for i, s in enumerate(group_speeds):
                padded[i, :len(s)] = s
            
            mean_spd = np.nanmean(padded, axis=0)
            std_spd = np.nanstd(padded, axis=0)
            t = np.arange(len(mean_spd)) * interval
            
            ax.plot(t, mean_spd, color=color, label=label, linewidth=1.5)
            ax.fill_between(t, mean_spd - std_spd, mean_spd + std_spd,
                           color=color, alpha=0.15)
        
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Speed (m/s)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle("Speed Profiles: Human vs Model by Tunnel Type", fontsize=13)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "strategy_speed_profiles.png", dpi=200)
    plt.close(fig)
    print(f"  Saved {RESULTS_DIR / 'strategy_speed_profiles.png'}")


def plot_trajectory_samples(human_records, model_records, centerlines):
    """Plot sample trajectories overlaid on centerlines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    _ensure_results_dir()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for ax, tid in zip(axes.flat, sorted(TRIAL_CONDITIONS.keys())):
        cond = TRIAL_CONDITIONS[tid]
        ttype = cond["type"]
        width = cond["width"]
        ax.set_title(f"{ttype.capitalize()} W={int(width*1000)}mm")
        
        # Draw centerline (coordinates in meters, plot in mm)
        if tid in centerlines:
            cl = np.array(centerlines[tid])
            ax.plot(cl[:, 0] * 1000, cl[:, 1] * 1000, 'k-', linewidth=2, 
                   label='Centerline', alpha=0.6, color='gray')
        
        # Draw sample human trajectories
        human_trajs = [r["trajectory"] for r in human_records 
                      if r["trial_id"] == tid]  # Max 5 samples
        for i, traj in enumerate(human_trajs):
            traj_np = np.array(traj)
            ax.plot(traj_np[:, 0] * 1000, traj_np[:, 1] * 1000, 
                   color='blue', alpha=0.3, linewidth=1,
                   label='Human' if i == 0 else None)
        
        # Draw model trajectories
        model_trajs = [r["trajectory"] for r in model_records 
                      if r["trial_id"] == tid][:3]  # Max 3 samples
        for i, traj in enumerate(model_trajs):
            traj_np = np.array(traj)
            ax.plot(traj_np[:, 0] * 1000, traj_np[:, 1] * 1000, 
                   color='red', alpha=0.4, linewidth=1.5,
                   label='Model' if i == 0 else None)
        
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='datalim')
    
    fig.suptitle("Sample Trajectories by Tunnel Type", fontsize=14)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "strategy_trajectories.png", dpi=200)
    plt.close(fig)
    print(f"  Saved {RESULTS_DIR / 'strategy_trajectories.png'}")


def plot_speed_progress_grid(human_records, model_records, centerlines, interval=0.05, n_bins=20):
    """
    Plot 4-panel speed vs progress plot (like trajectory layout).
    
    Each panel shows speed profiles as a function of progress (0-100%) along the path,
    with human data in blue and model in red, using mean ± std bands.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    _ensure_results_dir()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Progress bins
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
        
        # Compute progress for a trajectory point
        def compute_progress(traj_point, cl):
            """Find closest point on centerline and return progress (0-1)."""
            dists = np.linalg.norm(cl - np.array(traj_point), axis=1)
            closest_idx = np.argmin(dists)
            # Compute arc length to closest point
            arc_lengths = np.zeros(len(cl))
            for i in range(1, len(cl)):
                arc_lengths[i] = arc_lengths[i-1] + np.linalg.norm(cl[i] - cl[i-1])
            total_len = arc_lengths[-1]
            if total_len < 1e-6:
                return 0.0
            return arc_lengths[closest_idx] / total_len
        
        # Collect all speed-progress data
        all_data = []
        
        # Process human records
        human_recs = [r for r in human_records if r["trial_id"] == tid]
        for rec in human_recs:
            traj = rec.get("trajectory", [])
            speeds = rec.get("speeds", [])
            if len(traj) < 2 or len(speeds) < 1:
                continue
            # Speed indices are offset by 1 from trajectory
            for i, speed in enumerate(speeds):
                if i + 1 < len(traj):
                    progress = compute_progress(traj[i + 1], cl_np)
                    all_data.append({"Progress": progress, "Speed (m/s)": speed, "Type": "Human"})
        
        # Process model records
        model_recs = [r for r in model_records if r["trial_id"] == tid]
        for rec in model_recs:
            traj = rec.get("trajectory", [])
            speeds = rec.get("speeds", [])
            if len(traj) < 2 or len(speeds) < 1:
                continue
            for i, speed in enumerate(speeds):
                if i + 1 < len(traj):
                    progress = compute_progress(traj[i + 1], cl_np)
                    all_data.append({"Progress": progress, "Speed (m/s)": speed, "Type": "Model"})
        
        if all_data:
            import pandas as pd
            df = pd.DataFrame(all_data)
            
            # Bin data by progress
            df["Progress_bin"] = pd.cut(df["Progress"], bins=progress_bins, labels=progress_centers, include_lowest=True)
            df["Progress_bin"] = df["Progress_bin"].astype(float)
            
            # Plot with seaborn
            palette = {"Human": "#4287f5", "Model": "#e8453c"}
            sns.lineplot(data=df, x="Progress_bin", y="Speed (m/s)", hue="Type",
                        ax=ax, errorbar="sd", palette=palette, linewidth=1.5)
            
            ax.legend(fontsize=9)
        
        # Format x-axis as percentage
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.set_xlabel("Progress along path")
        ax.set_ylabel("Speed (m/s)")
        ax.grid(True, alpha=0.3)
    
    fig.suptitle("Speed vs Progress: Human vs Model by Tunnel Type", fontsize=14)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "strategy_speed_progress.png", dpi=200)
    plt.close(fig)
    print(f"  Saved {RESULTS_DIR / 'strategy_speed_progress.png'}")


def save_csv(human_records, model_records):
    """Write per-trial results to CSV."""
    _ensure_results_dir()
    path = RESULTS_DIR / "strategy_results.csv"
    fields = ["source", "participant", "trial_id", "round", "tunnel_type", "width",
              "completion_time", "avg_speed", "path_length", 
              "path_efficiency", "lateral_deviation", "speed_cv", "min_speed_ratio"]
    
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
                "completion_time": round(r["completion_time"], 4),
                "avg_speed": round(r["avg_speed"], 6),
                "path_length": round(r["path_length"], 6),
                "path_efficiency": round(r.get("path_efficiency", 1.0), 4),
                "lateral_deviation": round(r.get("lateral_deviation", 0.0), 6),
                "speed_cv": round(r.get("speed_cv", 0.0), 4),
                "min_speed_ratio": round(r.get("min_speed_ratio", 1.0), 4),
            })
    print(f"  Saved {path}")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Corner-Cutting Strategy evaluation")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG),
                        help="Path to user configuration JSON")
    parser.add_argument("--rounds", type=int, default=2,
                        help="Number of simulation rounds per condition")
    parser.add_argument("--trackpad-confidence", type=float, default=0,
                        help="Minimum confidence threshold for trackpad classification. "
                             "Only participants classified as trackpad users with confidence >= "
                             "this value will be included. Set to 0 to disable filtering.")
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"Config not found: {config_path}")
    
    with open(config_path) as f:
        user_cfg = json.load(f)
    pw = user_cfg.get("planner_weights", {})
    
    # Determine trackpad filtering
    trackpad_conf = args.trackpad_confidence if args.trackpad_confidence > 0 else None
    
    print("=" * 70)
    print("Corner-Cutting Strategy Evaluation")
    print("=" * 70)
    print(f"  Config:         {config_path}")
    print(f"  desired_speed:  {pw.get('desired_speed', '?')}")
    print(f"  speed_alpha:    {pw.get('speed_alpha', 0.5)} (speed ~ clearance^alpha)")
    print(f"  speed_floor:    {pw.get('speed_floor', 0.3)} (min speed fraction)")
    print(f"  speed_ceil:     {pw.get('speed_ceil', 1.5)} (max speed fraction)")
    print(f"  tracking_alpha: {pw.get('tracking_alpha', 0.5)} (tracking ~ 1/clearance^alpha)")
    print(f"  tracking_max:   {pw.get('tracking_max', 3.0)} (max tracking boost)")
    print(f"  w_cut:          {pw.get('w_cut', 0.0)} (corner-cutting in ref path)")
    print(f"  w_suppress:     {pw.get('w_suppress', 5.0)}")
    print(f"  Rounds/cond:    {args.rounds}")
    if trackpad_conf is not None:
        print(f"  Trackpad filter: confidence >= {trackpad_conf}")
    else:
        print(f"  Trackpad filter: disabled (all participants)")
    print()
    
    # --- Step 1: Generate centerlines for reference ---
    print("[1/6] Generating reference centerlines ...")
    centerlines = generate_centerlines()
    print(f"       Generated centerlines for {len(centerlines)} conditions")
    
    # --- Step 2: Load human data ---
    print("[2/6] Loading human data ...")
    human_records = load_human_data(trackpad_confidence=trackpad_conf)
    n_participants = len(set(r["participant"] for r in human_records))
    print(f"       Loaded {len(human_records)} human trials from {n_participants} participants")
    
    # --- Step 3: Compute human metrics ---
    print("[3/6] Computing human strategy metrics ...")
    human_records = compute_strategy_metrics(human_records, centerlines)
    human_summary = compute_summary(human_records, "human")
    
    # --- Step 4: Run simulator ---
    print("[4/6] Running simulator ...")
    model_records, oob_rounds = run_simulator_trials(config_path, n_rounds=args.rounds)
    total_model_rounds = len(model_records) + len(oob_rounds)
    print(f"       Generated {len(model_records)} valid model trials"
          f" ({len(oob_rounds)} OOB out of {total_model_rounds})")
    if oob_rounds:
        for oob in oob_rounds:
            print(f"         OOB: {oob['label']} round {oob['round']}"
                  f" ({oob['max_consecutive']} consecutive steps)")
    
    # --- Step 5: Compute model metrics ---
    print("[5/6] Computing model strategy metrics ...")
    model_records = compute_strategy_metrics(model_records)
    model_summary = compute_summary(model_records, "model")
    human_width_summary = compute_width_summary(human_records, "human")
    model_width_summary = compute_width_summary(model_records, "model")
    
    # --- Print summary table by tunnel type ---
    print()
    print("  Summary by Tunnel Type (aggregated across widths):")
    print("  ┌─────────────────┬─────────────────────────────────────────────────────────────────┐")
    print("  │                 │        Human                             Model                  │")
    print("  │ Tunnel Type     │  PathEff  LatDev(mm)  SpeedCV     PathEff  LatDev(mm)  SpeedCV  │")
    print("  ├─────────────────┼─────────────────────────────────────────────────────────────────┤")
    
    for ttype in ["sigmoidal", "corner"]:
        h_row = next((r for r in human_summary if r["tunnel_type"] == ttype), None)
        m_row = next((r for r in model_summary if r["tunnel_type"] == ttype), None)
        
        h_pe = h_row["path_eff_mean"] if h_row else 0
        h_ld = h_row["lat_dev_mean"] * 1000 if h_row else 0  # meters to mm
        h_cv = h_row["speed_cv_mean"] if h_row else 0
        
        m_pe = m_row["path_eff_mean"] if m_row else 0
        m_ld = m_row["lat_dev_mean"] * 1000 if m_row else 0  # meters to mm
        m_cv = m_row["speed_cv_mean"] if m_row else 0
        
        print(f"  │ {ttype:15s} │   {h_pe:5.3f}     {h_ld:5.2f}      {h_cv:5.3f}        "
              f"{m_pe:5.3f}      {m_ld:5.2f}      {m_cv:5.3f}   │")
    
    print("  └─────────────────┴─────────────────────────────────────────────────────────────────┘")
    
    # --- Print width comparison table ---
    print()
    print("  Width Effect Analysis (narrow=20mm vs wide=40mm):")
    print("  ┌─────────────────────────┬──────────────────────────────────────────────────────────┐")
    print("  │                         │        Model Results                                     │")
    print("  │ Condition               │  LatDev(mm)   AvgSpd(m/s)   PathEff    SpeedCV   MT(s)  │")
    print("  ├─────────────────────────┼──────────────────────────────────────────────────────────┤")
    
    for ttype in ["sigmoidal", "corner"]:
        for width_label in ["narrow", "wide"]:
            m_row = next((r for r in model_width_summary 
                         if r["tunnel_type"] == ttype and r["width_label"] == width_label), None)
            if m_row:
                label = f"{ttype} {width_label}"
                ld = m_row["lat_dev_mean"] * 1000
                spd = m_row["avg_speed_mean"]
                pe = m_row["path_eff_mean"]
                cv = m_row["speed_cv_mean"]
                mt = m_row["MT_mean"]
                print(f"  │ {label:23s} │    {ld:5.2f}        {spd:5.3f}        {pe:5.3f}     {cv:5.3f}   {mt:5.2f}  │")
    
    print("  └─────────────────────────┴──────────────────────────────────────────────────────────┘")
    
    # --- Print expected behavior check ---
    print()
    print("  Expected Width Effect (narrow → wide):")
    for ttype in ["sigmoidal", "corner"]:
        narrow = next((r for r in model_width_summary 
                      if r["tunnel_type"] == ttype and r["width_label"] == "narrow"), None)
        wide = next((r for r in model_width_summary 
                    if r["tunnel_type"] == ttype and r["width_label"] == "wide"), None)
        if narrow and wide:
            ld_change = (wide["lat_dev_mean"] - narrow["lat_dev_mean"]) * 1000
            spd_change = wide["avg_speed_mean"] - narrow["avg_speed_mean"]
            ld_sign = "↑" if ld_change > 0 else "↓"
            spd_sign = "↑" if spd_change > 0 else "↓"
            print(f"    {ttype:10s}: LatDev {ld_sign} {abs(ld_change):+.2f}mm, AvgSpeed {spd_sign} {abs(spd_change):+.3f} m/s")
    
    print()
    print("  Key: ↑ = increase (expected for wide), ↓ = decrease")
    print("       Expected: Wide → higher LatDev (more cutting), higher Speed")
    print()
    
    # --- Step 6: Generate outputs ---
    print("[6/6] Generating plots and saving results ...")
    plot_path_efficiency(human_summary, model_summary)
    plot_lateral_deviation(human_summary, model_summary)
    plot_speed_variability(human_summary, model_summary)
    plot_speed_profiles(human_records, model_records)
    plot_trajectory_samples(human_records, model_records, centerlines)
    plot_speed_progress_grid(human_records, model_records, centerlines)
    save_csv(human_records, model_records)
    
    print()
    print("Done.")


if __name__ == "__main__":
    main()

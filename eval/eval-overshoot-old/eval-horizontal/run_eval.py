"""
Main Evaluation Experiment: Human vs Model Trajectory and Speed Profile Comparison.

This experiment compares human cursor trajectories and speed profiles with model
predictions across all task types. Creates per-participant directories with
per-trial plots.

Directory structure:
    results/
        participant_{pid}/
            trial_{tid}/
                trajectories_t{tid}.png
                speeds_t{tid}.png
                speeds_enhanced_t{tid}.png
                speeds_progress_t{tid}.png
                speeds_progress_enhanced_t{tid}.png
                trajectories_density_t{tid}.png
                results_summary.json
            trial_metadata.json

Usage:
    python run_eval.py [--config path/to/user_config.json] [--rounds N] [--trials 1 2 3 4]
    python run_eval.py --pid P084931  # Process single participant
"""

import argparse
import concurrent.futures
import csv
import json
import math
import os
import sys
import tempfile
import time
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
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

BASELINE_PKG_DIR = PROJECT_ROOT / "eval" / "chi-26-ea_baseline_pacakage" / "src"

from experiment.environment import create_environment, generate_task_config
from experiment.utils import generateTunnelPath, generateCornerPath, generateTunnelBoundaries
from hcs_package.cursor_simulator import CursorSimulator

from utils.plot_utils import (
    plot_experiment_results,
    plot_enhanced_speed_profiles,
    plot_speeds_vs_progress,
    plot_speeds_vs_progress_enhanced,
    plot_trajectory_density,
)
from utils.stats import (
    resample_by_progress,
    resample_speeds_by_progress,
    trajectory_rmse,
    speed_profile_rmse,
    speed_profile_correlation,
)
TRAIN_TIDS = set()
TEST_TIDS = set(range(11, 16))
STRAIGHT_TIDS = set(range(11, 16))
STEERING_TIDS = set()
MAX_ROUND_DURATION_S = 60

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HUMAN_DATA_DIR = PROJECT_ROOT / "human_data" / "raw"
RESULTS_DIR = SCRIPT_DIR / "results"

TRIAL_CONDITIONS = {
    # Straight (curvature=0)
    11: {"type": "sigmoidal", "width": 0.01, "curvature": 0.0, "label": "Straight W=10mm"},
    12: {"type": "sigmoidal", "width": 0.02, "curvature": 0.0, "label": "Straight W=20mm"},
    13: {"type": "sigmoidal", "width": 0.03, "curvature": 0.0, "label": "Straight W=30mm"},
    14: {"type": "sigmoidal", "width": 0.04, "curvature": 0.0, "label": "Straight W=40mm"},
    15: {"type": "sigmoidal", "width": 0.05, "curvature": 0.0, "label": "Straight W=50mm"},
}

DEFAULT_CONFIG = PROJECT_ROOT / "experiment" / "user_configurations" / "customized.json"
FITTING_RESULTS_DIR = PROJECT_ROOT / "eval" / "model_fitting" / "results"
BASELINE_FITTING_DIR = PROJECT_ROOT / "eval" / "baseline_fitting" / "results"

WINDOW_WIDTH = 0.46
WINDOW_HEIGHT = 0.26

SIM_CACHE_DIR = RESULTS_DIR / "sim_cache"


# ===================================================================
# Simulation result caching
# ===================================================================

def _sim_cache_path(config_path, label="model"):
    """Return the cache file path for a given config (round-count independent)."""
    config_name = Path(config_path).stem
    return SIM_CACHE_DIR / f"{label}_{config_name}.json"


def save_sim_cache(cache_dict, cache_path):
    """Save sim_cache dict {tid: [records]} to JSON."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {str(tid): records for tid, records in cache_dict.items()}
    with open(cache_path, "w") as f:
        json.dump(serializable, f)
    print(f"  Sim cache saved to {cache_path}")


def load_sim_cache(cache_path):
    """Load sim_cache from JSON. Returns {int_tid: [records]} or None."""
    if not cache_path.exists():
        return None
    try:
        with open(cache_path) as f:
            data = json.load(f)
        return {int(tid): records for tid, records in data.items()}
    except Exception as e:
        print(f"  Warning: failed to load sim cache ({e}), will re-run simulations")
        return None


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


def _normalize_trajectory(traj_raw):
    """Normalize trajectory to [[x, y], ...] format."""
    if not traj_raw:
        return []
    if isinstance(traj_raw[0], dict):
        return [[p["x"], p["y"]] for p in traj_raw]
    return traj_raw


def compute_progress_along_path(trajectory, tunnel_path):
    """Compute progress (0 to 1) along tunnel path for each point."""
    if not trajectory or not tunnel_path:
        return [0.0] * len(trajectory)
    
    path = np.asarray(tunnel_path, dtype=float)
    traj = np.asarray(trajectory, dtype=float)
    
    path_lengths = [0.0]
    cumulative_length = 0.0
    for i in range(len(path) - 1):
        seg_len = np.linalg.norm(path[i + 1] - path[i])
        cumulative_length += seg_len
        path_lengths.append(cumulative_length)
    
    total_path_length = path_lengths[-1]
    if total_path_length < 1e-6:
        return [0.0] * len(trajectory)
    
    progress_values = []
    for point in traj:
        best_dist2 = float('inf')
        best_i = 0
        best_t = 0.0
        
        for i in range(len(path) - 1):
            p0 = path[i]
            p1 = path[i + 1]
            seg = p1 - p0
            seg_len2 = float(np.dot(seg, seg))
            
            if seg_len2 <= 1e-18:
                cand = p0
                dist2 = float(np.dot(point - cand, point - cand))
                if dist2 < best_dist2:
                    best_dist2 = dist2
                    best_i = i
                    best_t = 0.0
            else:
                t = float(np.clip(np.dot(point - p0, seg) / seg_len2, 0.0, 1.0))
                cand = p0 + t * seg
                dist2 = float(np.dot(point - cand, point - cand))
                if dist2 < best_dist2:
                    best_dist2 = dist2
                    best_i = i
                    best_t = t
        
        arc_length = path_lengths[best_i]
        if best_i < len(path) - 1:
            seg_len = np.linalg.norm(path[best_i + 1] - path[best_i])
            arc_length += best_t * seg_len
        
        progress = arc_length / total_path_length
        progress_values.append(np.clip(progress, 0.0, 1.0))
    
    return progress_values


# ===================================================================
# Load Human Data by Participant
# ===================================================================

def load_trials_by_participant(trial_ids=None):
    """Load all trials from raw data files, organized by participant_id and trial_id.
    
    Returns:
        dict: {participant_id: {trial_id: {round: {trajectory, speeds, timestamps, condition, completion_time}}}}
    """
    if trial_ids is None:
        trial_ids = list(TRIAL_CONDITIONS.keys())
    trial_ids_set = set(trial_ids)
    
    all_data = {}
    
    for fpath in sorted(HUMAN_DATA_DIR.glob("*.json")):
        try:
            with open(fpath) as f:
                data = json.load(f)
            
            participant_id = data.get("participantId", fpath.stem)
            sessions = data.get("sessions", [])
            
            if not sessions:
                trial_data_array = data.get("trialData", [])
                sessions = [{"trialData": trial_data_array}] if trial_data_array else []
            
            if participant_id not in all_data:
                all_data[participant_id] = {}
            
            for session in sessions:
                trial_data_array = session.get("trialData", [])
                
                for trial_data in trial_data_array:
                    trial_id = trial_data.get("trial_id")
                    round_num = trial_data.get("round", 0)
                    
                    if trial_id is None or trial_id not in trial_ids_set:
                        continue
                    
                    if trial_id not in all_data[participant_id]:
                        all_data[participant_id][trial_id] = {}
                    
                    traj_raw = trial_data.get("trajectory", [])
                    trajectory = _normalize_trajectory(traj_raw)
                    timestamps = trial_data.get("timestamps", [])
                    
                    if len(trajectory) < 2 or len(timestamps) < 2:
                        continue

                    duration_s = (timestamps[-1] - timestamps[0]) / 1000.0
                    if duration_s > MAX_ROUND_DURATION_S:
                        print(f"  Filtered out {participant_id} trial {trial_id} round {round_num}: "
                              f"duration {duration_s:.1f}s > {MAX_ROUND_DURATION_S}s")
                        continue

                    speeds = _compute_speeds(trajectory, timestamps)
                    if len(speeds) != len(trajectory):
                        if len(speeds) < len(trajectory):
                            speeds.extend([speeds[-1] if speeds else 0.0] * (len(trajectory) - len(speeds)))
                        else:
                            speeds = speeds[:len(trajectory)]
                    
                    condition = trial_data.get("condition", {})
                    completion_time = trial_data.get("completionTime", 0.0)
                    
                    all_data[participant_id][trial_id][round_num] = {
                        "trajectory": trajectory,
                        "speeds": speeds,
                        "timestamps": timestamps,
                        "condition": condition,
                        "completion_time": completion_time,
                    }
        except Exception as e:
            print(f"Warning: Error loading {fpath.name}: {e}")
            continue
    
    return all_data


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


def generate_centerline(trial_id):
    """Generate centerline for a trial."""
    cond = TRIAL_CONDITIONS.get(trial_id, {})
    if cond.get("type") == "sigmoidal":
        _, centerline = _build_sigmoidal_config(cond["width"], cond["curvature"])
    elif cond.get("type") == "corner":
        _, centerline = _build_corner_config(cond["width"], cond["num_corners"], cond["corner_offset"])
    else:
        centerline = []
    return [[x, y] for x, y in centerline]


def run_simulator_for_trial(sim, trial_id, n_runs=1):
    """Run simulator for a specific trial condition."""
    cond = TRIAL_CONDITIONS.get(trial_id, {})
    if not cond:
        return []
    
    tunnel_type = cond["type"]
    width = cond["width"]
    
    if tunnel_type == "sigmoidal":
        task_config, centerline = _build_sigmoidal_config(width, cond["curvature"])
    else:
        task_config, centerline = _build_corner_config(width, cond["num_corners"], cond["corner_offset"])
    
    interval = sim.interval
    records = []
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        json.dump(task_config, tf)
        task_file = tf.name
    
    try:
        for run_id in range(n_runs):
            traj_raw = sim.generate_trajectory_with_waypoints(
                task_file=task_file,
                max_steps=task_config.get("max_steps", 800),
                target_radius=task_config.get("target_radius", 0.01),
                use_optimal_path=True,
            )
            
            scale = 0.001
            traj = [[x * scale, y * scale] for x, y, _ in traj_raw]
            
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
            if speeds:
                speeds.insert(0, speeds[0])
            
            records.append({
                "trajectory": traj,
                "speeds": speeds,
                "completion_time": ct,
                "path_length": pl,
                "avg_speed": pl / ct if ct > 0 else 0.0,
            })
    finally:
        os.unlink(task_file)
    
    return records


def _load_baseline_simulator():
    """Import CursorSimulator from the CHI-26-EA baseline package.

    Uses module isolation to avoid conflicts with the main hcs_package.
    """
    saved = {k: sys.modules.pop(k)
             for k in list(sys.modules) if k == 'hcs_package' or k.startswith('hcs_package.')}
    sys.path.insert(0, str(BASELINE_PKG_DIR))
    try:
        from hcs_package.cursor_simulator import CursorSimulator as Cls
    finally:
        for k in list(sys.modules):
            if k == 'hcs_package' or k.startswith('hcs_package.'):
                del sys.modules[k]
        sys.modules.update(saved)
        sys.path.remove(str(BASELINE_PKG_DIR))
    return Cls


def _run_baseline_in_worker(config_path, baseline_pkg_dir, trial_id, n_runs):
    """Run baseline simulation inside a pool worker with module isolation.

    Temporarily swaps out the main hcs_package for the baseline package for
    the full duration of sim creation + trajectory generation, ensuring the
    baseline's internal imports resolve correctly.
    """
    baseline_pkg_dir_str = str(baseline_pkg_dir)
    # Save and remove main hcs_package modules
    saved = {k: sys.modules.pop(k)
             for k in list(sys.modules) if k == 'hcs_package' or k.startswith('hcs_package.')}
    sys.path.insert(0, baseline_pkg_dir_str)
    try:
        from hcs_package.cursor_simulator import CursorSimulator as BaselineCls
        # Ensure noise is enabled for evaluation (fitted configs save add_noise=False
        # from the deterministic CMA-ES fitting phase)
        with open(str(config_path)) as _f:
            _cfg = json.load(_f)
        _cfg["add_noise"] = True
        import tempfile as _tf
        _fd, _tmp = _tf.mkstemp(suffix=".json", prefix="bl_eval_")
        os.close(_fd)
        try:
            with open(_tmp, "w") as _f:
                json.dump(_cfg, _f)
            sim = BaselineCls(_tmp)
            records = run_baseline_for_trial(sim, trial_id, n_runs=n_runs)
        finally:
            os.unlink(_tmp)
    finally:
        # Remove baseline modules and restore main hcs_package
        for k in list(sys.modules):
            if k == 'hcs_package' or k.startswith('hcs_package.'):
                del sys.modules[k]
        sys.modules.update(saved)
        sys.path.remove(baseline_pkg_dir_str)
    return records


def run_baseline_for_trial(baseline_sim, trial_id, n_runs=1,
                           wall_clock_timeout=120.0):
    """Run baseline simulator for a specific trial condition."""
    cond = TRIAL_CONDITIONS.get(trial_id, {})
    if not cond:
        return []

    tunnel_type = cond["type"]
    width = cond["width"]

    if tunnel_type == "sigmoidal":
        task_config, centerline = _build_sigmoidal_config(width, cond["curvature"])
    else:
        task_config, centerline = _build_corner_config(width, cond["num_corners"], cond["corner_offset"])

    interval = baseline_sim.interval
    records = []

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        json.dump(task_config, tf)
        task_file = tf.name

    # Bounding box for divergence detection: screen is 0.46 x 0.26 m,
    # allow some margin but flag anything way off-screen as diverged.
    DIVERGE_LIMIT = 1.0  # meters — well beyond the 0.46m screen

    try:
        for run_id in range(n_runs):
            t0 = time.time()
            bl_max_steps = task_config.get("max_steps", 800)
            traj_raw = baseline_sim.generate_trajectory_with_waypoints(
                task_file=task_file,
                max_steps=bl_max_steps,
                target_radius=task_config.get("target_radius", 0.01),
                use_optimal_path=True,
            )
            elapsed = time.time() - t0

            scale = 0.001
            traj = [[x * scale, y * scale] for x, y, _ in traj_raw]

            # Check for diverged trajectory (cursor exploded off-screen)
            if traj:
                last_pt = traj[-1]
                if abs(last_pt[0]) > DIVERGE_LIMIT or abs(last_pt[1]) > DIVERGE_LIMIT:
                    print(f"  WARNING: baseline trial {trial_id} run {run_id} diverged "
                          f"(end=({last_pt[0]:.2f},{last_pt[1]:.2f})), "
                          f"wall={elapsed:.1f}s — skipping", flush=True)
                    continue

            if elapsed > wall_clock_timeout:
                print(f"  WARNING: baseline trial {trial_id} run {run_id} "
                      f"took {elapsed:.1f}s (>{wall_clock_timeout}s), skipping remaining runs",
                      flush=True)
                break

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
            if speeds:
                speeds.insert(0, speeds[0])

            records.append({
                "trajectory": traj,
                "speeds": speeds,
                "completion_time": ct,
                "path_length": pl,
                "avg_speed": pl / ct if ct > 0 else 0.0,
            })
    finally:
        os.unlink(task_file)

    return records


N_PROGRESS_BINS = 100
# Speed metrics are evaluated over the central portion of the path,
# excluding startup acceleration and target-approach deceleration.
SPEED_TRIM_START = 10   # skip first 10% of progress
SPEED_TRIM_END   = 100   # looking at end of progress for evaluation of pointing model


def compute_trial_metrics(model_traj, model_speeds, human_traj, human_speeds,
                          human_timestamps, centerline):
    """Compute progress-aligned metrics between one model run and one human round.

    Returns dict with lateral_rmse, speed_rmse, speed_corr, time_diff.
    """
    _, _, lat_m = resample_by_progress(model_traj, centerline, N_PROGRESS_BINS)
    _, _, lat_h = resample_by_progress(human_traj, centerline, N_PROGRESS_BINS)
    _, spd_m = resample_speeds_by_progress(model_speeds, model_traj, centerline, N_PROGRESS_BINS)
    _, spd_h = resample_speeds_by_progress(human_speeds, human_traj, centerline, N_PROGRESS_BINS)

    lat_rmse = trajectory_rmse(lat_m, lat_h)

    # Trim speed profiles to central portion — excludes startup acceleration
    # and target-approach deceleration (task framing, not steering behavior)
    spd_m_trim = spd_m[SPEED_TRIM_START:SPEED_TRIM_END]
    spd_h_trim = spd_h[SPEED_TRIM_START:SPEED_TRIM_END]
    spd_rmse = speed_profile_rmse(spd_m_trim, spd_h_trim)
    spd_corr = speed_profile_correlation(spd_m_trim, spd_h_trim)

    human_time = (human_timestamps[-1] - human_timestamps[0]) / 1000.0 if len(human_timestamps) >= 2 else 1.0
    model_time = len(model_traj) * 0.05
    time_diff = abs(model_time - human_time) / max(human_time, 0.1)

    return {
        "lateral_rmse": float(lat_rmse),
        "speed_rmse": float(spd_rmse),
        "speed_corr": float(spd_corr),
        "time_diff": float(time_diff),
        "human_time": float(human_time),
        "model_time": float(model_time),
    }


def save_trial_results_summary(participant_id, trial_id, human_data, model_data,
                                centerline, output_path,
                                model_data_for_metrics=None,
                                baseline_data=None,
                                baseline_data_for_metrics=None):
    """Save results summary JSON for a trial, including progress-aligned metrics.

    Args:
        model_data: ALL model runs (unfiltered) — saved to results_summary.json.
        model_data_for_metrics: Only valid model runs (filtered) — used for
            metrics computation. If None, falls back to model_data.
        baseline_data: ALL baseline runs (unfiltered) — saved to results_summary.json.
        baseline_data_for_metrics: Only valid baseline runs (filtered) — used for
            metrics computation. If None, falls back to baseline_data.
    """
    h_trajs = human_data.get("trajectories", [])
    h_speeds = human_data.get("speeds", [])
    h_timestamps = human_data.get("timestamps", [])
    # All model runs (for saving)
    m_trajs_all = model_data.get("trajectories", [])
    m_speeds_all = model_data.get("speeds", [])
    # Filtered model runs (for metrics)
    m_metrics_src = model_data_for_metrics or model_data
    m_trajs = m_metrics_src.get("trajectories", [])
    m_speeds = m_metrics_src.get("speeds", [])
    # All baseline runs (for saving)
    b_trajs_all = baseline_data.get("trajectories", []) if baseline_data else []
    b_speeds_all = baseline_data.get("speeds", []) if baseline_data else []
    # Filtered baseline runs (for metrics)
    bl_metrics = baseline_data_for_metrics or baseline_data
    b_trajs = bl_metrics.get("trajectories", []) if bl_metrics else []
    b_speeds = bl_metrics.get("speeds", []) if bl_metrics else []

    # Compute metrics on mean profiles (averaged across all human rounds
    # and all model runs) — this matches what the speed-progress plots show.
    agg = {}
    human_rounds, model_rounds = [], []
    human_times = []
    all_spd_h, all_spd_m = [], []
    if m_trajs and centerline:
        # Resample all human rounds and model runs to progress bins
        all_spd_h, all_lat_h = [], []
        all_spd_m, all_lat_m = [], []
        human_times = []

        for i in range(len(h_trajs)):
            if len(h_trajs[i]) < 5:
                continue
            _, _, lat_h = resample_by_progress(h_trajs[i], centerline, N_PROGRESS_BINS)
            _, spd_h = resample_speeds_by_progress(h_speeds[i], h_trajs[i], centerline, N_PROGRESS_BINS)
            all_lat_h.append(lat_h)
            all_spd_h.append(spd_h)
            ht = (h_timestamps[i][-1] - h_timestamps[i][0]) / 1000.0 if len(h_timestamps[i]) >= 2 else 1.0
            human_times.append(ht)

        for j in range(len(m_trajs)):
            if len(m_trajs[j]) < 5:
                continue
            _, _, lat_m = resample_by_progress(m_trajs[j], centerline, N_PROGRESS_BINS)
            _, spd_m = resample_speeds_by_progress(m_speeds[j], m_trajs[j], centerline, N_PROGRESS_BINS)
            all_lat_m.append(lat_m)
            all_spd_m.append(spd_m)

        if all_lat_h and all_lat_m:
            # Mean profiles
            mean_lat_h = np.mean(all_lat_h, axis=0)
            mean_lat_m = np.mean(all_lat_m, axis=0)
            mean_spd_h = np.mean(all_spd_h, axis=0)
            mean_spd_m = np.mean(all_spd_m, axis=0)

            lat_rmse = trajectory_rmse(mean_lat_m, mean_lat_h)
            spd_m_trim = mean_spd_m[SPEED_TRIM_START:SPEED_TRIM_END]
            spd_h_trim = mean_spd_h[SPEED_TRIM_START:SPEED_TRIM_END]
            spd_rmse = speed_profile_rmse(spd_m_trim, spd_h_trim)
            spd_corr = speed_profile_correlation(spd_m_trim, spd_h_trim)

            mean_human_time = float(np.mean(human_times))
            mean_model_time = float(np.mean([len(mt) * 0.05 for mt in m_trajs]))
            time_diff = abs(mean_model_time - mean_human_time) / max(mean_human_time, 0.1)

            #Metrics to evaluate pointing model
            goal_spd_h = float(np.mean(mean_spd_h[90:100])) if len(mean_spd_h) >= 100 else None
            goal_spd_m = float(np.mean(mean_spd_m[90:100])) if len(mean_spd_m) >= 100 else None

            overshoot_distances = []
            if centerline and len(centerline) >= 2:
                goal = np.array(centerline[-1])
                path_dir = np.array(centerline[-1]) - np.array(centerline[-2])
                path_dir = path_dir / (np.linalg.norm(path_dir) + 1e-10)
                for traj in m_trajs:
                    if len(traj) < 2:
                        continue
                    last_pt = np.array(traj[-1])
                    overshoot_distances.append(float(np.dot(last_pt - goal, path_dir)))

            n_overshoot = sum(1 for d in overshoot_distances if d > 0)
            overshoots = [d for d in overshoot_distances if d > 0]
            undershoots = [d for d in overshoot_distances if d <= 0]

            agg = {
                "lateral_rmse_mean": float(lat_rmse),
                "speed_rmse_mean": float(spd_rmse),
                "speed_corr_mean": float(spd_corr),
                "time_diff_mean": float(time_diff),
                "goal_approach_speed_human": goal_spd_h,
                "goal_approach_speed_model": goal_spd_m,
                "goal_approach_speed_diff": float(abs(goal_spd_m - goal_spd_h) / max(goal_spd_h, 0.001)) if goal_spd_h is not None and goal_spd_m is not None else None,
                "overshoot_rate": float(n_overshoot / len(m_trajs)) if m_trajs else None,
                "mean_overshoot_dist_m": float(np.mean(overshoots)) if overshoots else None,
                "mean_undershoot_dist_m": float(np.mean(undershoots)) if undershoots else None,
                "mean_signed_dist_m": float(np.mean(overshoot_distances)) if overshoot_distances else None,
            }

            # Per-round stats (no cross-pairing — just individual round summaries)
            human_rounds = [
                {"round": i, "completion_time": float(ht),
                 "avg_speed": float(np.mean(all_spd_h[i]))}
                for i, ht in enumerate(human_times)
            ]
            model_rounds = [
                {"run": j, "completion_time": float(len(m_trajs[j]) * 0.05),
                 "avg_speed": float(np.mean(all_spd_m[j]))}
                for j in range(len(all_spd_m))
            ]

    # Baseline metrics (baseline vs human)
    baseline_agg = {}
    baseline_rounds = []
    all_spd_b = []
    if b_trajs and centerline and all_lat_h:
        all_lat_b, all_spd_b = [], []
        for j in range(len(b_trajs)):
            if len(b_trajs[j]) < 5:
                continue
            _, _, lat_b = resample_by_progress(b_trajs[j], centerline, N_PROGRESS_BINS)
            _, spd_b = resample_speeds_by_progress(b_speeds[j], b_trajs[j], centerline, N_PROGRESS_BINS)
            all_lat_b.append(lat_b)
            all_spd_b.append(spd_b)

        if all_lat_b and all_lat_h:
            mean_lat_b = np.mean(all_lat_b, axis=0)
            mean_spd_b = np.mean(all_spd_b, axis=0)
            mean_spd_h_ref = np.mean(all_spd_h, axis=0) if all_spd_h else mean_spd_b

            b_lat_rmse = trajectory_rmse(mean_lat_b, mean_lat_h)
            spd_b_trim = mean_spd_b[SPEED_TRIM_START:SPEED_TRIM_END]
            spd_h_trim2 = mean_spd_h_ref[SPEED_TRIM_START:SPEED_TRIM_END]
            b_spd_rmse = speed_profile_rmse(spd_b_trim, spd_h_trim2)
            b_spd_corr = speed_profile_correlation(spd_b_trim, spd_h_trim2)

            mean_human_time_b = float(np.mean(human_times)) if human_times else 1.0
            mean_baseline_time = float(np.mean([len(bt) * 0.05 for bt in b_trajs]))
            b_time_diff = abs(mean_baseline_time - mean_human_time_b) / max(mean_human_time_b, 0.1)

            baseline_agg = {
                "lateral_rmse_mean": float(b_lat_rmse),
                "speed_rmse_mean": float(b_spd_rmse),
                "speed_corr_mean": float(b_spd_corr),
                "time_diff_mean": float(b_time_diff),
            }
            baseline_rounds = [
                {"run": j, "completion_time": float(len(b_trajs[j]) * 0.05),
                 "avg_speed": float(np.mean(all_spd_b[j]))}
                for j in range(len(all_spd_b))
            ]

    summary = {
        "participant_id": participant_id,
        "trial_id": trial_id,
        "human": {
            "n_rounds": len(h_trajs),
            "avg_completion_time": float(np.mean(human_times)) if human_times else 0,
            "avg_speed": float(np.mean([np.mean(s) for s in all_spd_h])) if all_spd_h else 0,
            "rounds": human_rounds,
        },
        "model": {
            "n_runs": len(m_trajs_all),
            "n_valid_runs": len(m_trajs),
            "avg_completion_time": float(np.mean([len(mt) * 0.05 for mt in m_trajs])) if m_trajs else 0,
            "avg_speed": float(np.mean([np.mean(s) for s in all_spd_m])) if all_spd_m else 0,
            "runs": [
                {"run": j, "completion_time": float(len(m_trajs_all[j]) * 0.05),
                 "avg_speed": float(_path_length(m_trajs_all[j]) / max(len(m_trajs_all[j]) * 0.05, 0.001))}
                for j in range(len(m_trajs_all))
            ],
        },
        "metrics": agg,
    }
    if b_trajs_all:
        # Save ALL runs (including timed-out/diverged) as faithful record
        all_baseline_rounds = [
            {"run": j, "completion_time": float(len(b_trajs_all[j]) * 0.05),
             "avg_speed": float(_path_length(b_trajs_all[j]) / max(len(b_trajs_all[j]) * 0.05, 0.001))}
            for j in range(len(b_trajs_all))
        ]
        # avg_completion_time and avg_speed use filtered data only
        summary["baseline"] = {
            "n_runs": len(b_trajs_all),
            "n_valid_runs": len(b_trajs),
            "avg_completion_time": float(np.mean([len(bt) * 0.05 for bt in b_trajs])) if b_trajs else 0,
            "avg_speed": float(np.mean([np.mean(s) for s in all_spd_b])) if all_spd_b else 0,
            "runs": all_baseline_rounds,
        }
        summary["baseline_metrics"] = baseline_agg

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)


# ===================================================================
# Parallel trial processing
# ===================================================================

def _process_trial(config_path, speed_model_path, participant_id, trial_id,
                   trial_data_dict, n_rounds, existing_records,
                   trial_folder_str, centerline, include_baseline,
                   baseline_records_for_trial,
                   baseline_config_path=None, baseline_pkg_dir=None):
    """Process a single trial: simulate, plot, compute metrics.

    Designed to run in a process pool — creates its own simulator.
    When baseline_config_path is provided and existing baseline records are
    insufficient, runs baseline simulation inside the worker (parallelized).
    Returns dict with trial results including any new baseline records.
    """
    import matplotlib
    matplotlib.use("Agg")

    trial_folder = Path(trial_folder_str)
    trial_folder.mkdir(exist_ok=True)

    # -- human data --
    human_trajectories = []
    human_speeds_list = []
    human_timestamps_list = []
    human_records = []

    for round_num, round_data in trial_data_dict.items():
        traj = round_data["trajectory"]
        speeds = round_data["speeds"]
        timestamps = round_data["timestamps"]
        ct = round_data["completion_time"]
        if len(traj) >= 2:
            human_trajectories.append(traj)
            human_speeds_list.append(speeds)
            human_timestamps_list.append(timestamps)
            human_records.append({
                "completion_time": ct,
                "path_length": _path_length(traj),
                "avg_speed": _path_length(traj) / ct if ct > 0 else 0,
            })

    # -- model simulation (incremental) --
    existing = existing_records or []
    delta = n_rounds - len(existing)
    if delta > 0:
        sim = CursorSimulator(str(config_path))
        if speed_model_path:
            from hcs_package.speed_model import GAMSpeedModel
            sim.speed_model = GAMSpeedModel.load(str(speed_model_path))
        new_records = run_simulator_for_trial(sim, trial_id, n_runs=delta)
        all_model_records = existing + new_records
    else:
        all_model_records = existing[:n_rounds]

    all_model_records_unfiltered = all_model_records[:n_rounds]
    # Filter out model runs that diverged (trajectory goes far outside screen)
    _MAX_MODEL_CT = 39.5
    _MODEL_DIVERGE = 1.0
    model_records = []
    for ri, r in enumerate(all_model_records_unfiltered):
        traj = r.get("trajectory", [])
        if r["completion_time"] >= _MAX_MODEL_CT:
            print(f"  Filtered model: trial {trial_id} run {ri} "
                  f"— CT={r['completion_time']:.2f}s (hit max steps)", flush=True)
            continue
        if traj:
            xs = [p[0] for p in traj]
            ys = [p[1] for p in traj]
            if min(xs) < -0.1 or max(xs) > 0.56 or min(ys) < -0.1 or max(ys) > 0.36:
                print(f"  Filtered model: trial {trial_id} run {ri} "
                      f"— out of bounds x=[{min(xs):.3f},{max(xs):.3f}] "
                      f"y=[{min(ys):.3f},{max(ys):.3f}]", flush=True)
                continue
        model_records.append(r)
    model_trajectories = [r["trajectory"] for r in model_records]
    model_speeds_list = [r["speeds"] for r in model_records]

    # -- baseline (run in worker if needed, else use pre-computed) --
    baseline_trajectories = []
    baseline_speeds_list = []
    all_baseline_records_trial = list(baseline_records_for_trial) if baseline_records_for_trial else []
    if include_baseline and baseline_config_path and len(all_baseline_records_trial) < n_rounds:
        # Run baseline simulation inside the worker (parallelized with model)
        bl_delta = n_rounds - len(all_baseline_records_trial)
        new_bl_records = _run_baseline_in_worker(
            baseline_config_path, baseline_pkg_dir, trial_id, bl_delta)
        all_baseline_records_trial.extend(new_bl_records)
    if include_baseline and all_baseline_records_trial:
        # Filter out timed-out and diverged baseline runs before plotting/metrics
        _MAX_BL_CT = 39.5       # 800 steps * 0.05s
        _DIVERGE_LIM = 1.0      # meters
        bl_valid = []
        for ri, r in enumerate(all_baseline_records_trial[:n_rounds]):
            ct = r["completion_time"]
            if ct >= _MAX_BL_CT:
                print(f"  Filtered baseline: trial {trial_id} run {ri} "
                      f"— CT={ct:.2f}s (hit max steps)", flush=True)
                continue
            traj = r.get("trajectory", [])
            if traj:
                lp = traj[-1]
                if abs(lp[0]) > _DIVERGE_LIM or abs(lp[1]) > _DIVERGE_LIM:
                    print(f"  Filtered baseline: trial {trial_id} run {ri} "
                          f"— diverged (end=({lp[0]:.2f},{lp[1]:.2f}))", flush=True)
                    continue
            bl_valid.append(r)
        baseline_trajectories = [r["trajectory"] for r in bl_valid]
        baseline_speeds_list = [r["speeds"] for r in bl_valid]

    # -- plots --
    cond = TRIAL_CONDITIONS.get(trial_id, {})
    tunnel_width = cond.get("width", 0.02)
    trial_metadata = {trial_id: {participant_id: {0: {"condition": cond}}}}

    segment_data = {
        'human': {'trajectories': human_trajectories, 'speeds': human_speeds_list},
        'sim': {'trajectories': model_trajectories, 'speeds': model_speeds_list},
    }
    if baseline_trajectories:
        segment_data['baseline'] = {
            'trajectories': baseline_trajectories, 'speeds': baseline_speeds_list,
        }

    all_results = {trial_id: {0: segment_data}}
    tunnel_paths = {trial_id: centerline}
    trial_tunnel_widths = {trial_id: tunnel_width}

    plot_experiment_results(all_results, trial_folder, tunnel_paths, trial_tunnel_widths, trial_metadata)
    plot_enhanced_speed_profiles(all_results, trial_folder, time_step=0.05)
    if centerline:
        plot_speeds_vs_progress_enhanced(all_results, trial_folder, tunnel_paths, bin_size=0.1)

    # -- metrics & summary --
    # Save results_summary.json with ALL baseline runs (unfiltered) as a faithful record.
    # Use only filtered (valid) baseline runs for metrics computation.
    summary_path = trial_folder / "results_summary.json"
    bl_data_all = None
    if include_baseline and all_baseline_records_trial:
        bl_all_slice = all_baseline_records_trial[:n_rounds]
        bl_data_all = {
            "trajectories": [r["trajectory"] for r in bl_all_slice],
            "speeds": [r["speeds"] for r in bl_all_slice],
        }
    bl_data_filtered = None
    if baseline_trajectories:
        bl_data_filtered = {"trajectories": baseline_trajectories, "speeds": baseline_speeds_list}
    # Model: save all runs (unfiltered) but use filtered for metrics
    model_data_all = {
        "trajectories": [r["trajectory"] for r in all_model_records_unfiltered],
        "speeds": [r["speeds"] for r in all_model_records_unfiltered],
        "records": all_model_records_unfiltered,
    }
    model_data_filtered = {
        "trajectories": model_trajectories, "speeds": model_speeds_list,
        "records": model_records,
    }
    save_trial_results_summary(
        participant_id, trial_id,
        {"trajectories": human_trajectories, "speeds": human_speeds_list,
         "timestamps": human_timestamps_list, "records": human_records},
        model_data_all,
        centerline, summary_path,
        model_data_for_metrics=model_data_filtered,
        baseline_data=bl_data_all,
        baseline_data_for_metrics=bl_data_filtered,
    )

    # Read back metrics for printing
    metrics = {}
    try:
        with open(summary_path) as f:
            sm = json.load(f)
        metrics = sm.get("metrics", {})
    except Exception:
        pass

    # Model records for aggregate accumulation
    model_accum = []
    for r in model_records:
        model_accum.append({
            "trial_id": trial_id,
            "tunnel_type": cond.get("type", ""),
            "width": cond.get("width", 0),
            "completion_time": r["completion_time"],
            "avg_speed": r["avg_speed"],
            "path_length": r["path_length"],
        })

    return {
        "trial_id": trial_id,
        "all_sim_records": all_model_records,  # full cache (existing + new)
        "model_accum": model_accum,
        "metrics": metrics,
        "n_human": len(human_trajectories),
        "n_model": len(model_records),
        "n_baseline": len(baseline_trajectories),
        "all_baseline_records": all_baseline_records_trial,
    }


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Main evaluation: Human vs Model comparison")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG),
                        help="Path to user configuration JSON")
    parser.add_argument("--rounds", type=int, default=3,
                        help="Number of simulation rounds per condition")
    parser.add_argument("--trials", type=int, nargs='+', default=None,
                        help="List of trial IDs to process (default: all)")
    parser.add_argument("--pid", type=str, default=None,
                        help="Process only this participant ID")
    parser.add_argument("--speed-model", type=str, default=None,
                        help="Path to fitted GAM speed model (.pkl). "
                             "Overrides the config's speed_model section.")
    parser.add_argument("--include-baseline", action="store_true", default=False,
                        help="Include CHI-26-EA baseline results for comparison in plots")
    parser.add_argument("--per-participant", action="store_true", default=False,
                        help="Use per-participant fitted configs and speed models from "
                             "eval/model_fitting/results/{pid}_gam_config_s{seed}.json")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed suffix for per-participant model files (default: 42)")
    parser.add_argument("--aggregate-only", action="store_true", default=False,
                        help="Skip simulation, regenerate aggregate CSVs from existing "
                             "results_summary.json files across all participants")
    parser.add_argument("--no-filter-baseline", action="store_true", default=False,
                        help="Disable post-processing filters on baseline runs "
                             "(timed-out / diverged)")
    args = parser.parse_args()
    
    config_path = Path(args.config)
    trial_ids = args.trials if args.trials else list(TRIAL_CONDITIONS.keys())

    print("=" * 70)
    print("Main Evaluation: Human vs Model Trajectory & Speed Comparison")
    print("=" * 70)
    if args.per_participant:
        print(f"  Mode:           per-participant (seed={args.seed})")
    else:
        if not config_path.exists():
            sys.exit(f"Config not found: {config_path}")
        with open(config_path) as f:
            user_cfg = json.load(f)
        pw = user_cfg.get("planner_weights", {})
        print(f"  Config:         {config_path}")
        print(f"  desired_speed:  {pw.get('desired_speed', '?')}")
    print(f"  Trials:         {trial_ids}")
    print(f"  Rounds/cond:    {args.rounds}")
    if args.pid:
        print(f"  Participant:    {args.pid}")
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading human data by participant ...")
    all_data = load_trials_by_participant(trial_ids)

    if args.pid:
        if args.pid not in all_data:
            print(f"Error: Participant {args.pid} not found")
            print(f"Available: {list(all_data.keys())}")
            return
        if not args.aggregate_only:
            all_data = {args.pid: all_data[args.pid]}

    # In per-participant mode, filter to only participants with fitted models
    if args.per_participant:
        available_pids = {}
        for pid in list(all_data.keys()):
            cfg_file = FITTING_RESULTS_DIR / f"{pid}_gam_config_s{args.seed}.json"
            pkl_file = FITTING_RESULTS_DIR / f"{pid}_gam_s{args.seed}.pkl"
            if cfg_file.exists() and pkl_file.exists():
                available_pids[pid] = (cfg_file, pkl_file)
            else:
                print(f"  Warning: skipping {pid} — no fitted model (seed={args.seed})")
        all_data = {pid: all_data[pid] for pid in available_pids}
        if not all_data:
            sys.exit("No participants have fitted models. Run fit_speed_model.py first.")

        # Discover per-participant baseline configs
        available_baseline_cfgs = {}
        if args.include_baseline:
            for pid in list(all_data.keys()):
                bl_cfg = BASELINE_FITTING_DIR / f"{pid}_baseline_config_s{args.seed}.json"
                if bl_cfg.exists():
                    available_baseline_cfgs[pid] = bl_cfg
                else:
                    print(f"  Warning: no baseline config for {pid} (seed={args.seed})")

    print(f"       Loaded {len(all_data)} participants")

    if args.aggregate_only:
        print("\n  Aggregate-only mode — skipping simulation, regenerating CSVs ...")
        baseline_cache = {}
        all_model_records_accum = []
        n_model_filtered = 0
        # Load sim caches for out-of-bounds checking
        model_sim_caches = {}
        for participant_id in sorted(all_data.keys()):
            cache_path = SIM_CACHE_DIR / f"{participant_id}_s{args.seed}.json"
            if cache_path.exists():
                model_sim_caches[participant_id] = load_sim_cache(cache_path) or {}

        # Populate model records from existing results_summary.json files
        for participant_id in sorted(all_data.keys()):
            p_folder = RESULTS_DIR / f"participant_{participant_id}"
            sim_cache = model_sim_caches.get(participant_id, {})
            for tid in sorted(all_data[participant_id].keys()):
                summary_file = p_folder / f"trial_{tid}" / "results_summary.json"
                if not summary_file.exists():
                    continue
                with open(summary_file) as f:
                    sm = json.load(f)
                model_sec = sm.get("model", {})
                cond = TRIAL_CONDITIONS.get(tid, {})
                if model_sec.get("runs"):
                    label = f"model_{participant_id}" if args.per_participant else "model"
                    # Get cached trajectories for bounds checking
                    cached_runs = sim_cache.get(tid, sim_cache.get(str(tid), []))
                    for ri, run in enumerate(model_sec["runs"]):
                        ct = run["completion_time"]
                        # Check timeout
                        if ct >= 39.5:
                            print(f"  Filtered model: {participant_id} trial {tid} run {ri} "
                                  f"— CT={ct:.2f}s (hit max steps)")
                            n_model_filtered += 1
                            continue
                        # Check out-of-bounds from sim cache trajectory
                        if ri < len(cached_runs):
                            traj = cached_runs[ri].get("trajectory", [])
                            if traj:
                                xs = [p[0] for p in traj]
                                ys = [p[1] for p in traj]
                                if min(xs) < -0.1 or max(xs) > 0.56 or min(ys) < -0.1 or max(ys) > 0.36:
                                    print(f"  Filtered model: {participant_id} trial {tid} run {ri} "
                                          f"— out of bounds x=[{min(xs):.3f},{max(xs):.3f}] "
                                          f"y=[{min(ys):.3f},{max(ys):.3f}]")
                                    n_model_filtered += 1
                                    continue
                        all_model_records_accum.append({
                            "participant": label,
                            "trial_id": tid,
                            "tunnel_type": cond.get("type", ""),
                            "width": cond.get("width", 0),
                            "completion_time": ct,
                            "avg_speed": run["avg_speed"],
                            "path_length": run.get("path_length",
                                run["avg_speed"] * ct),
                        })
        if n_model_filtered:
            print(f"  Model filtering: {n_model_filtered} runs removed")
        # Jump directly to aggregate section (step 4/4)

    else:
        # Normal mode — proceed with simulation

        # Resolve config and speed model paths for each participant
        speed_model_path_str = None
        if not args.per_participant:
            if not config_path.exists():
                sys.exit(f"Config not found: {config_path}")
            if args.speed_model:
                smp = Path(args.speed_model)
                if not smp.exists():
                    sys.exit(f"Speed model not found: {smp}")
                speed_model_path_str = str(smp)
            print("[2/4] Checking simulator config ...")
            sim_test = CursorSimulator(str(config_path))
            if speed_model_path_str:
                from hcs_package.speed_model import GAMSpeedModel
                sim_test.speed_model = GAMSpeedModel.load(speed_model_path_str)
            if sim_test.speed_model is not None:
                print(f"       Speed model: {type(sim_test.speed_model).__name__}")
            else:
                print(f"       Speed model: None (no speed model loaded)")
            del sim_test
        else:
            print("[2/4] Per-participant mode — simulators created per trial worker")

        centerlines = {}
        for tid in trial_ids:
            centerlines[tid] = generate_centerline(tid)

        # Baseline setup
        baseline_cache = {}
        BaselineSimCls = None
        if args.include_baseline:
            BaselineSimCls = _load_baseline_simulator()
            if not args.per_participant:
                # Shared baseline (original behavior — single config for all participants)
                bl_cache_path = SIM_CACHE_DIR / "baseline_results.json"
                baseline_cache = load_sim_cache(bl_cache_path) or {}
                bl_needs_run = {}
                for tid in trial_ids:
                    existing = len(baseline_cache.get(tid, []))
                    if existing < args.rounds:
                        bl_needs_run[tid] = args.rounds - existing
                if bl_needs_run:
                    print("[2b/4] Running baseline simulations (CHI-26-EA) ...")
                    baseline_sim = BaselineSimCls("user_config")
                    for tid, delta in bl_needs_run.items():
                        new_records = run_baseline_for_trial(baseline_sim, tid, n_runs=delta)
                        baseline_cache[tid] = baseline_cache.get(tid, []) + new_records
                    save_sim_cache(baseline_cache, bl_cache_path)
                    print(f"       Baseline done ({len(bl_needs_run)} trials simulated)")
                else:
                    print(f"       Loaded cached baseline results ({len(baseline_cache)} trials)")
            else:
                print("[2b/4] Per-participant baseline — will run per participant")

        print("[3/4] Processing participants (parallel trials) ...")

        total_participants = len(all_data)
        all_model_records_accum = []
        n_workers = os.cpu_count() or 4

        for p_idx, (participant_id, participant_data) in enumerate(sorted(all_data.items())):
            print(f"\n{'='*60}")
            print(f"Participant {p_idx + 1}/{total_participants}: {participant_id}")
            print(f"{'='*60}")

            # Determine config + speed model for this participant
            if args.per_participant:
                cfg_file, pkl_file = available_pids[participant_id]
                p_config_path = str(cfg_file)
                p_speed_model_path = str(pkl_file)
                print(f"  Config: {cfg_file.name}")
                print(f"  Speed model: {pkl_file.name}")
                cache_path = SIM_CACHE_DIR / f"{participant_id}_s{args.seed}.json"
            else:
                p_config_path = str(config_path)
                p_speed_model_path = speed_model_path_str
                cache_path = _sim_cache_path(config_path, label="model")

            sim_cache = load_sim_cache(cache_path) or {}
            cached_count = sum(1 for tid in participant_data if tid in sim_cache
                              and len(sim_cache[tid]) >= args.rounds)
            partial_count = sum(1 for tid in participant_data if tid in sim_cache
                               and 0 < len(sim_cache[tid]) < args.rounds)
            need_sim = sum(1 for tid in participant_data if tid not in sim_cache
                           or len(sim_cache[tid]) < args.rounds)
            if sim_cache:
                parts = [f"{cached_count} fully cached"]
                if partial_count:
                    parts.append(f"{partial_count} partially cached")
                parts.append(f"{need_sim} need simulation")
                print(f"  Cache: {', '.join(parts)}")

            participant_folder = RESULTS_DIR / f"participant_{participant_id}"
            participant_folder.mkdir(exist_ok=True)

            participant_trial_ids = sorted(participant_data.keys())
            print(f"  Trials: {participant_trial_ids}")

            # Per-participant baseline: load cache and config path
            bl_cache_for_participant = {}
            bl_cfg_path_for_worker = None
            if args.include_baseline and args.per_participant:
                bl_cfg = available_baseline_cfgs.get(participant_id)
                if bl_cfg:
                    bl_cfg_path_for_worker = str(bl_cfg)
                    bl_cache_path_p = SIM_CACHE_DIR / f"{participant_id}_baseline_s{args.seed}.json"
                    bl_cache_for_participant = load_sim_cache(bl_cache_path_p) or {}
                    bl_cached = sum(1 for tid in participant_trial_ids
                                    if len(bl_cache_for_participant.get(tid, [])) >= args.rounds)
                    bl_need = len(participant_trial_ids) - bl_cached
                    if bl_need > 0:
                        print(f"  Baseline: {bl_cached} cached, {bl_need} need simulation (parallelized)")
                    else:
                        print(f"  Baseline: cached ({len(bl_cache_for_participant)} trials)")
                else:
                    print(f"  Baseline: no fitted config, skipping")
            elif args.include_baseline:
                bl_cache_for_participant = baseline_cache  # shared baseline

            # Build trial metadata file
            trial_metadata_file = {}
            for trial_id in participant_trial_ids:
                for round_num, round_data in participant_data[trial_id].items():
                    if trial_id not in trial_metadata_file:
                        trial_metadata_file[trial_id] = round_data.get("condition", {})

            # Submit all trials to process pool
            futures = {}
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                for trial_id in participant_trial_ids:
                    trial_folder = participant_folder / f"trial_{trial_id}"
                    existing_records = sim_cache.get(trial_id, [])
                    bl_records = bl_cache_for_participant.get(trial_id, []) if args.include_baseline else []

                    fut = executor.submit(
                        _process_trial,
                        p_config_path, p_speed_model_path,
                        participant_id, trial_id,
                        participant_data[trial_id],
                        args.rounds, existing_records,
                        str(trial_folder), centerlines.get(trial_id, []),
                        args.include_baseline, bl_records,
                        baseline_config_path=bl_cfg_path_for_worker,
                        baseline_pkg_dir=str(BASELINE_PKG_DIR),
                    )
                    futures[fut] = trial_id

                # Collect results as they complete (30 min total timeout)
                POOL_TIMEOUT = 1800
                try:
                    for fut in concurrent.futures.as_completed(futures, timeout=POOL_TIMEOUT):
                        tid = futures[fut]
                        try:
                            result = fut.result(timeout=60)
                            # Update sim cache with full records (existing + new)
                            sim_cache[tid] = result["all_sim_records"]
                            # Update baseline cache with records from worker
                            if result.get("all_baseline_records"):
                                bl_cache_for_participant[tid] = result["all_baseline_records"]
                            # Accumulate model records
                            label = f"model_{participant_id}" if args.per_participant else "model"
                            for rec in result["model_accum"]:
                                rec["participant"] = label
                                all_model_records_accum.append(rec)
                            # Print summary
                            m = result["metrics"]
                            m_str = ""
                            if m:
                                m_str = (f"  lat={m.get('lateral_rmse_mean', 0):.5f}"
                                         f"  spd_corr={m.get('speed_corr_mean', 0):.3f}"
                                         f"  spd_rmse={m.get('speed_rmse_mean', 0):.5f}"
                                         f"  time_diff={m.get('time_diff_mean', 0):.3f}")
                            print(f"  Trial {tid:2d}: H={result['n_human']} M={result['n_model']}"
                                  f" B={result['n_baseline']}{m_str}")
                        except Exception as e:
                            print(f"  Trial {tid}: ERROR — {e}")
                except concurrent.futures.TimeoutError:
                    timed_out = [tid for f, tid in futures.items() if not f.done()]
                    print(f"  WARNING: Pool timeout ({POOL_TIMEOUT}s) — "
                          f"cancelling stuck trials: {timed_out}")
                    for f in futures:
                        f.cancel()

            # Save updated sim cache
            save_sim_cache(sim_cache, cache_path)
            # Save updated baseline cache (per-participant)
            if args.include_baseline and args.per_participant and bl_cfg_path_for_worker and bl_cache_for_participant:
                save_sim_cache(bl_cache_for_participant, bl_cache_path_p)

            metadata_path = participant_folder / "trial_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(trial_metadata_file, f, indent=2)

    print("\n" + "=" * 70)
    print("[4/4] Generating aggregate summary ...")
    
    all_human_records = []
    all_model_records = list(all_model_records_accum)

    for participant_id, participant_data in all_data.items():
        for trial_id, trial_data_dict in participant_data.items():
            cond = TRIAL_CONDITIONS.get(trial_id, {})
            for round_num, round_data in trial_data_dict.items():
                traj = round_data["trajectory"]
                ct = round_data["completion_time"]
                if len(traj) >= 2 and ct > 0:
                    pl = _path_length(traj)
                    all_human_records.append({
                        "participant": participant_id,
                        "trial_id": trial_id,
                        "tunnel_type": cond.get("type", ""),
                        "width": cond.get("width", 0),
                        "completion_time": ct,
                        "avg_speed": pl / ct,
                        "path_length": pl,
                    })

    # Baseline: filter out timed-out runs (hit max_steps) and diverged runs
    filter_baseline = not args.no_filter_baseline
    MAX_BASELINE_CT = 39.5       # 800 steps * 0.05s = 40s
    DIVERGE_LIMIT_M = 1.0       # cursor exploded off-screen
    all_baseline_records = []
    n_bl_filtered_timeout = 0
    n_bl_filtered_diverge = 0
    print(f"  Baseline filtering: {'ON' if filter_baseline else 'OFF'}")

    def _filter_baseline_run(run, participant_id, tid, run_idx):
        """Return True if run should be kept, False if filtered out."""
        if not filter_baseline:
            return True
        nonlocal n_bl_filtered_timeout, n_bl_filtered_diverge
        ct = run["completion_time"]
        if ct >= MAX_BASELINE_CT:
            print(f"  Filtered baseline: {participant_id} trial {tid} run {run_idx} "
                  f"— CT={ct:.2f}s >= {MAX_BASELINE_CT}s (hit max steps)")
            n_bl_filtered_timeout += 1
            return False
        # Check for diverged trajectory if trajectory data is available
        traj = run.get("trajectory", [])
        if traj:
            last_pt = traj[-1]
            if abs(last_pt[0]) > DIVERGE_LIMIT_M or abs(last_pt[1]) > DIVERGE_LIMIT_M:
                print(f"  Filtered baseline: {participant_id} trial {tid} run {run_idx} "
                      f"— diverged (end=({last_pt[0]:.2f},{last_pt[1]:.2f}))")
                n_bl_filtered_diverge += 1
                return False
        return True

    if baseline_cache:
        # Shared baseline mode — records keyed by trial_id
        for trial_id, records in baseline_cache.items():
            cond = TRIAL_CONDITIONS.get(trial_id, {})
            for ri, r in enumerate(records):
                if not _filter_baseline_run(r, "shared", trial_id, ri):
                    continue
                all_baseline_records.append({
                    "participant": "baseline",
                    "trial_id": trial_id,
                    "tunnel_type": cond.get("type", ""),
                    "width": cond.get("width", 0),
                    "completion_time": r["completion_time"],
                    "avg_speed": r["avg_speed"],
                    "path_length": r["path_length"],
                })
    elif args.include_baseline and args.per_participant:
        # Per-participant baseline — read from results_summary.json files
        for participant_id in sorted(all_data.keys()):
            p_folder = RESULTS_DIR / f"participant_{participant_id}"
            for tid in sorted(all_data[participant_id].keys()):
                summary_file = p_folder / f"trial_{tid}" / "results_summary.json"
                if summary_file.exists():
                    with open(summary_file) as f:
                        sm = json.load(f)
                    bl = sm.get("baseline", {})
                    if bl and bl.get("runs"):
                        cond = TRIAL_CONDITIONS.get(tid, {})
                        for ri, run in enumerate(bl["runs"]):
                            if not _filter_baseline_run(run, participant_id, tid, ri):
                                continue
                            all_baseline_records.append({
                                "participant": f"baseline_{participant_id}",
                                "trial_id": tid,
                                "tunnel_type": cond.get("type", ""),
                                "width": cond.get("width", 0),
                                "completion_time": run["completion_time"],
                                "avg_speed": run["avg_speed"],
                                "path_length": run.get("path_length", run["avg_speed"] * run["completion_time"]),
                            })

    if n_bl_filtered_timeout or n_bl_filtered_diverge:
        print(f"\n  Baseline filtering: {n_bl_filtered_timeout} timed-out, "
              f"{n_bl_filtered_diverge} diverged "
              f"(kept {len(all_baseline_records)} runs)")

    csv_path = RESULTS_DIR / "main_results.csv"
    fields = ["participant", "trial_id", "tunnel_type", "width",
              "completion_time", "avg_speed", "path_length"]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in all_human_records + all_model_records + all_baseline_records:
            writer.writerow({k: round(v, 6) if isinstance(v, float) else v for k, v in r.items()})
    print(f"  Saved: {csv_path}")

    summary_by_trial = defaultdict(lambda: {"human": [], "model": [], "baseline": []})
    for r in all_human_records:
        summary_by_trial[r["trial_id"]]["human"].append(r)
    for r in all_model_records:
        summary_by_trial[r["trial_id"]]["model"].append(r)
    for r in all_baseline_records:
        summary_by_trial[r["trial_id"]]["baseline"].append(r)

    has_baseline = bool(all_baseline_records)

    print("\n  Summary by Trial:")
    if has_baseline:
        print("  ┌─────────────────────────────┬──────────────────────────────────────────────────────────────────────────┐")
        print("  │                             │        Human              Model             Baseline                    │")
        print("  │ Trial                       │  MT(s)   Speed(m/s)    MT(s)   Speed(m/s)  MT(s)   Speed(m/s)  n(H/M/B)│")
        print("  ├─────────────────────────────┼──────────────────────────────────────────────────────────────────────────┤")
    else:
        print("  ┌─────────────────────────────┬────────────────────────────────────────────────────┐")
        print("  │                             │        Human              Model                    │")
        print("  │ Trial                       │  MT(s)   Speed(m/s)    MT(s)   Speed(m/s)  n(H/M) │")
        print("  ├─────────────────────────────┼────────────────────────────────────────────────────┤")

    for tid in sorted(summary_by_trial.keys()):
        h_recs = summary_by_trial[tid]["human"]
        m_recs = summary_by_trial[tid]["model"]
        b_recs = summary_by_trial[tid]["baseline"]
        label = TRIAL_CONDITIONS.get(tid, {}).get("label", f"Trial {tid}")

        h_mt = np.mean([r["completion_time"] for r in h_recs]) if h_recs else 0
        h_spd = np.mean([r["avg_speed"] for r in h_recs]) if h_recs else 0
        m_mt = np.mean([r["completion_time"] for r in m_recs]) if m_recs else 0
        m_spd = np.mean([r["avg_speed"] for r in m_recs]) if m_recs else 0

        if has_baseline:
            b_mt = np.mean([r["completion_time"] for r in b_recs]) if b_recs else 0
            b_spd = np.mean([r["avg_speed"] for r in b_recs]) if b_recs else 0
            print(f"  │ {label:27s} │  {h_mt:5.2f}    {h_spd:.3f}       {m_mt:5.2f}    {m_spd:.3f}     {b_mt:5.2f}    {b_spd:.3f}   {len(h_recs):3d}/{len(m_recs)}/{len(b_recs):<3d}│")
        else:
            print(f"  │ {label:27s} │  {h_mt:5.2f}    {h_spd:.3f}       {m_mt:5.2f}    {m_spd:.3f}    {len(h_recs):3d}/{len(m_recs):<3d} │")

    if has_baseline:
        print("  └─────────────────────────────┴──────────────────────────────────────────────────────────────────────────┘")
    else:
        print("  └─────────────────────────────┴────────────────────────────────────────────────────┘")

    # Collect all per-participant, per-trial metrics from saved summaries
    aggregate_metrics = {}
    aggregate_baseline_metrics = {}
    for participant_id in sorted(all_data.keys()):
        p_folder = RESULTS_DIR / f"participant_{participant_id}"
        p_metrics = {}
        p_bl_metrics = {}
        for tid in sorted(all_data[participant_id].keys()):
            summary_file = p_folder / f"trial_{tid}" / "results_summary.json"
            if summary_file.exists():
                with open(summary_file) as f:
                    sm = json.load(f)
                p_metrics[str(tid)] = sm.get("metrics", {})
                bl_m = sm.get("baseline_metrics", {})
                bl_sec = sm.get("baseline", {})
                if bl_m:
                    # Skip baseline metrics if no valid runs remain after filtering
                    n_valid = bl_sec.get("n_valid_runs", bl_sec.get("n_runs", 0))
                    n_total = bl_sec.get("n_runs", 0)
                    # Cross-check: if all runs timed out, the metrics are invalid
                    bl_runs = bl_sec.get("runs", [])
                    n_ok = sum(1 for r in bl_runs if r.get("completion_time", 99) < 39.5)
                    if n_ok == 0 and n_total > 0:
                        print(f"  Skipped stale baseline metrics: {participant_id} trial {tid} "
                              f"(all {n_total} runs timed out)")
                    else:
                        p_bl_metrics[str(tid)] = bl_m
        if p_metrics:
            aggregate_metrics[participant_id] = p_metrics
        if p_bl_metrics:
            aggregate_baseline_metrics[participant_id] = p_bl_metrics

    # Compute grand average across participants and trials, split by train/test
    # and by task type (steering vs straight)
    metric_keys = ["lateral_rmse_mean", "speed_rmse_mean", "speed_corr_mean", "time_diff_mean",
               "goal_approach_speed_human", "goal_approach_speed_model",
               "goal_approach_speed_diff", "overshoot_rate",
               "mean_overshoot_dist_m", "mean_undershoot_dist_m", "mean_signed_dist_m"]

    def _accumulate_metrics(source):
        g = {k: [] for k in metric_keys}
        g_tr = {k: [] for k in metric_keys}
        g_te = {k: [] for k in metric_keys}
        g_st = {k: [] for k in metric_keys}
        g_str = {k: [] for k in metric_keys}
        pt = defaultdict(lambda: {k: [] for k in metric_keys})
        for pid, trials in source.items():
            for tid, m in trials.items():
                tid_int = int(tid)
                target = g_tr if tid_int in TRAIN_TIDS else g_te
                type_target = g_st if tid_int in STEERING_TIDS else g_str
                for k in metric_keys:
                    if k in m:
                        g[k].append(m[k])
                        target[k].append(m[k])
                        type_target[k].append(m[k])
                        pt[tid][k].append(m[k])
        avg = lambda d: {
            k: float(np.mean([x for x in v if x is not None])) 
            if any(x is not None for x in v) else None 
            for k, v in d.items()
        }
        ta = {}
        for tid in sorted(pt.keys()):
            ta[tid] = avg(pt[tid])
        return avg(g), avg(g_tr), avg(g_te), avg(g_st), avg(g_str), ta

    grand_avg, grand_train_avg, grand_test_avg, grand_steering_avg, grand_straight_avg, trial_avg = \
        _accumulate_metrics(aggregate_metrics)

    # Baseline metrics (if available)
    bl_grand_avg = bl_grand_train_avg = bl_grand_test_avg = None
    bl_grand_steering_avg = bl_grand_straight_avg = None
    bl_trial_avg = {}
    if aggregate_baseline_metrics:
        bl_grand_avg, bl_grand_train_avg, bl_grand_test_avg, \
            bl_grand_steering_avg, bl_grand_straight_avg, bl_trial_avg = \
            _accumulate_metrics(aggregate_baseline_metrics)

    # Print metrics table
    if has_baseline and bl_trial_avg:
        # Wide table with both Model and Baseline columns
        print("\n  Progress-Aligned Metrics (vs human, averaged across participants):")
        print("  ┌────┬─────────────────────────────┬─────────────────────────────────────────────┬─────────────────────────────────────────────┐")
        print("  │    │                             │              Model                          │             Baseline                        │")
        print("  │    │ Trial                       │ Lat RMSE  Spd RMSE  Spd Corr  Time Diff    │ Lat RMSE  Spd RMSE  Spd Corr  Time Diff    │")
        print("  ├────┼─────────────────────────────┼─────────────────────────────────────────────┼─────────────────────────────────────────────┤")
        for tid in sorted(trial_avg.keys(), key=int):
            tid_int = int(tid)
            label = TRIAL_CONDITIONS.get(tid_int, {}).get("label", f"Trial {tid}")
            split = "TR" if tid_int in TRAIN_TIDS else "TE"
            ta = trial_avg[tid]
            bt = bl_trial_avg.get(tid, {})
            m_str = f"{ta.get('lateral_rmse_mean', 0):9.5f} {ta.get('speed_rmse_mean', 0):9.5f} {ta.get('speed_corr_mean', 0):9.3f} {ta.get('time_diff_mean', 0):9.3f}   "
            if bt:
                b_str = f"{bt.get('lateral_rmse_mean', 0):9.5f} {bt.get('speed_rmse_mean', 0):9.5f} {bt.get('speed_corr_mean', 0):9.3f} {bt.get('time_diff_mean', 0):9.3f}   "
            else:
                b_str = "      -         -         -         -      "
            print(f"  │ {split} │ {label:27s} │ {m_str}│ {b_str}│")
        print("  ├────┼─────────────────────────────┼─────────────────────────────────────────────┼─────────────────────────────────────────────┤")

        def _print_summary_row(split, label, m_avg, b_avg):
            if m_avg and m_avg.get("lateral_rmse_mean") is not None:
                m_str = f"{m_avg['lateral_rmse_mean']:9.5f} {m_avg['speed_rmse_mean']:9.5f} {m_avg['speed_corr_mean']:9.3f} {m_avg['time_diff_mean']:9.3f}   "
            else:
                m_str = "      -         -         -         -      "
            if b_avg and b_avg.get("lateral_rmse_mean") is not None:
                b_str = f"{b_avg['lateral_rmse_mean']:9.5f} {b_avg['speed_rmse_mean']:9.5f} {b_avg['speed_corr_mean']:9.3f} {b_avg['time_diff_mean']:9.3f}   "
            else:
                b_str = "      -         -         -         -      "
            print(f"  │ {split} │ {label:27s} │ {m_str}│ {b_str}│")

        _print_summary_row("TR", "Train avg", grand_train_avg, bl_grand_train_avg)
        _print_summary_row("TE", "Test avg", grand_test_avg, bl_grand_test_avg)
        print("  ├────┼─────────────────────────────┼─────────────────────────────────────────────┼─────────────────────────────────────────────┤")
        _print_summary_row("  ", "Steering tasks (20)", grand_steering_avg, bl_grand_steering_avg)
        _print_summary_row("  ", "Straight tasks (5)", grand_straight_avg, bl_grand_straight_avg)
        _print_summary_row("  ", "Overall (25)", grand_avg, bl_grand_avg)
        print("  └────┴─────────────────────────────┴─────────────────────────────────────────────┴─────────────────────────────────────────────┘")
    else:
        print("\n  Progress-Aligned Metrics (model vs human, averaged across participants):")
        print("  ┌────┬─────────────────────────────┬───────────┬───────────┬───────────┬───────────┐")
        print("  │    │ Trial                       │ Lat RMSE  │ Spd RMSE  │ Spd Corr  │ Time Diff │")
        print("  ├────┼─────────────────────────────┼───────────┼───────────┼───────────┼───────────┤")
        for tid in sorted(trial_avg.keys(), key=int):
            tid_int = int(tid)
            label = TRIAL_CONDITIONS.get(tid_int, {}).get("label", f"Trial {tid}")
            split = "TR" if tid_int in TRAIN_TIDS else "TE"
            ta = trial_avg[tid]
            lr = ta.get("lateral_rmse_mean")
            sr = ta.get("speed_rmse_mean")
            sc = ta.get("speed_corr_mean")
            td = ta.get("time_diff_mean")
            print(f"  │ {split} │ {label:27s} │ {lr:9.5f} │ {sr:9.5f} │ {sc:9.3f} │ {td:9.3f} │")
        print("  ├────┼─────────────────────────────┼───────────┼───────────┼───────────┼───────────┤")
        if grand_train_avg.get("lateral_rmse_mean") is not None:
            print(f"  │ TR │ {'Train avg':27s} │ {grand_train_avg['lateral_rmse_mean']:9.5f} │ {grand_train_avg['speed_rmse_mean']:9.5f} │ {grand_train_avg['speed_corr_mean']:9.3f} │ {grand_train_avg['time_diff_mean']:9.3f} │")
        if grand_test_avg.get("lateral_rmse_mean") is not None:
            print(f"  │ TE │ {'Test avg':27s} │ {grand_test_avg['lateral_rmse_mean']:9.5f} │ {grand_test_avg['speed_rmse_mean']:9.5f} │ {grand_test_avg['speed_corr_mean']:9.3f} │ {grand_test_avg['time_diff_mean']:9.3f} │")
        print("  ├────┼─────────────────────────────┼───────────┼───────────┼───────────┼───────────┤")
        if grand_steering_avg.get("lateral_rmse_mean") is not None:
            print(f"  │    │ {'Steering tasks (20)':27s} │ {grand_steering_avg['lateral_rmse_mean']:9.5f} │ {grand_steering_avg['speed_rmse_mean']:9.5f} │ {grand_steering_avg['speed_corr_mean']:9.3f} │ {grand_steering_avg['time_diff_mean']:9.3f} │")
        if grand_straight_avg.get("lateral_rmse_mean") is not None:
            print(f"  │    │ {'Straight tasks (5)':27s} │ {grand_straight_avg['lateral_rmse_mean']:9.5f} │ {grand_straight_avg['speed_rmse_mean']:9.5f} │ {grand_straight_avg['speed_corr_mean']:9.3f} │ {grand_straight_avg['time_diff_mean']:9.3f} │")
        if grand_avg.get("lateral_rmse_mean") is not None:
            print(f"  │    │ {'Overall (25)':27s} │ {grand_avg['lateral_rmse_mean']:9.5f} │ {grand_avg['speed_rmse_mean']:9.5f} │ {grand_avg['speed_corr_mean']:9.3f} │ {grand_avg['time_diff_mean']:9.3f} │")
        print("  └────┴─────────────────────────────┴───────────┴───────────┴───────────┴───────────┘")

    # Save aggregate metrics JSON
    agg_out = {
        "n_participants": len(aggregate_metrics),
        "grand_average": grand_avg,
        "grand_average_train": grand_train_avg,
        "grand_average_test": grand_test_avg,
        "grand_average_steering": grand_steering_avg,
        "grand_average_straight": grand_straight_avg,
        "per_trial_average": trial_avg,
        "per_participant": aggregate_metrics,
    }
    if has_baseline and bl_grand_avg:
        agg_out["baseline_grand_average"] = bl_grand_avg
        agg_out["baseline_grand_average_train"] = bl_grand_train_avg
        agg_out["baseline_grand_average_test"] = bl_grand_test_avg
        agg_out["baseline_per_trial_average"] = bl_trial_avg
    agg_path = RESULTS_DIR / "aggregate_metrics.json"
    with open(agg_path, 'w') as f:
        json.dump(agg_out, f, indent=2)
    print(f"\n  Saved: {agg_path}")

    # --- CSV: Trial summary (one row per trial, averaged across participants) ---
    trial_summary_csv = RESULTS_DIR / "trial_summary.csv"
    ts_fields = ["trial_id", "label", "split", "tunnel_type", "width",
                "human_mt", "human_speed", "model_mt", "model_speed",
                "lat_rmse", "speed_rmse", "speed_corr", "time_diff",
                "goal_approach_speed_human", "goal_approach_speed_model",
                "goal_approach_speed_diff", "overshoot_rate",
                "mean_overshoot_dist_m", "mean_undershoot_dist_m", "mean_signed_dist_m",
                "n_human", "n_model"]
    if has_baseline:
        ts_fields += ["baseline_mt", "baseline_speed",
                      "bl_lat_rmse", "bl_speed_rmse", "bl_speed_corr", "bl_time_diff"]
    with open(trial_summary_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=ts_fields)
        writer.writeheader()
        for tid in sorted(summary_by_trial.keys()):
            cond = TRIAL_CONDITIONS.get(tid, {})
            h_recs = summary_by_trial[tid]["human"]
            m_recs = summary_by_trial[tid]["model"]
            tid_str = str(tid)
            ta = trial_avg.get(tid_str, {})
            row = {
                "trial_id": tid,
                "label": cond.get("label", f"Trial {tid}"),
                "split": "train" if tid in TRAIN_TIDS else "test",
                "tunnel_type": cond.get("type", ""),
                "width": cond.get("width", 0),
                "human_mt": round(float(np.mean([r["completion_time"] for r in h_recs])), 4) if h_recs else "",
                "human_speed": round(float(np.mean([r["avg_speed"] for r in h_recs])), 6) if h_recs else "",
                "model_mt": round(float(np.mean([r["completion_time"] for r in m_recs])), 4) if m_recs else "",
                "model_speed": round(float(np.mean([r["avg_speed"] for r in m_recs])), 6) if m_recs else "",
                "lat_rmse": round(ta["lateral_rmse_mean"], 6) if ta.get("lateral_rmse_mean") is not None else "",
                "speed_rmse": round(ta["speed_rmse_mean"], 6) if ta.get("speed_rmse_mean") is not None else "",
                "speed_corr": round(ta["speed_corr_mean"], 4) if ta.get("speed_corr_mean") is not None else "",
                "time_diff": round(ta["time_diff_mean"], 4) if ta.get("time_diff_mean") is not None else "",
                "goal_approach_speed_human": round(ta["goal_approach_speed_human"], 4) if ta.get("goal_approach_speed_human") is not None else "",
                "goal_approach_speed_model": round(ta["goal_approach_speed_model"], 4) if ta.get("goal_approach_speed_model") is not None else "",
                "goal_approach_speed_diff": round(ta["goal_approach_speed_diff"], 4) if ta.get("goal_approach_speed_diff") is not None else "",
                "overshoot_rate": round(ta["overshoot_rate"], 4) if ta.get("overshoot_rate") is not None else "",
                "mean_overshoot_dist_m": round(ta["mean_overshoot_dist_m"], 6) if ta.get("mean_overshoot_dist_m") is not None else "",
                "mean_undershoot_dist_m": round(ta["mean_undershoot_dist_m"], 6) if ta.get("mean_undershoot_dist_m") is not None else "",
                "mean_signed_dist_m": round(ta["mean_signed_dist_m"], 6) if ta.get("mean_signed_dist_m") is not None else "",
                "n_human": len(h_recs),
                "n_model": len(m_recs),
            }
            if has_baseline:
                b_recs = summary_by_trial[tid]["baseline"]
                bta = bl_trial_avg.get(tid_str, {})
                row["baseline_mt"] = round(float(np.mean([r["completion_time"] for r in b_recs])), 4) if b_recs else ""
                row["baseline_speed"] = round(float(np.mean([r["avg_speed"] for r in b_recs])), 6) if b_recs else ""
                row["bl_lat_rmse"] = round(bta["lateral_rmse_mean"], 6) if bta.get("lateral_rmse_mean") is not None else ""
                row["bl_speed_rmse"] = round(bta["speed_rmse_mean"], 6) if bta.get("speed_rmse_mean") is not None else ""
                row["bl_speed_corr"] = round(bta["speed_corr_mean"], 4) if bta.get("speed_corr_mean") is not None else ""
                row["bl_time_diff"] = round(bta["time_diff_mean"], 4) if bta.get("time_diff_mean") is not None else ""
            writer.writerow(row)
    print(f"  Saved: {trial_summary_csv}")

    # --- CSV: Progress-aligned metrics (one row per participant × trial) ---
    progress_csv = RESULTS_DIR / "progress_metrics.csv"
    pm_fields = ["participant", "trial_id", "label", "split", "tunnel_type", "width",
                "lat_rmse", "speed_rmse", "speed_corr", "time_diff",
                "goal_approach_speed_human", "goal_approach_speed_model",
                "goal_approach_speed_diff", "overshoot_rate",
                "mean_overshoot_dist_m", "mean_undershoot_dist_m", "mean_signed_dist_m"]
    if has_baseline:
        pm_fields += ["bl_lat_rmse", "bl_speed_rmse", "bl_speed_corr", "bl_time_diff"]
    with open(progress_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=pm_fields)
        writer.writeheader()
        for pid in sorted(aggregate_metrics.keys()):
            p_metrics = aggregate_metrics[pid]
            p_bl = aggregate_baseline_metrics.get(pid, {})
            for tid_str in sorted(p_metrics.keys(), key=int):
                tid_int = int(tid_str)
                cond = TRIAL_CONDITIONS.get(tid_int, {})
                m = p_metrics[tid_str]
                row = {
                    "participant": pid,
                    "trial_id": tid_int,
                    "label": cond.get("label", f"Trial {tid_str}"),
                    "split": "train" if tid_int in TRAIN_TIDS else "test",
                    "tunnel_type": cond.get("type", ""),
                    "width": cond.get("width", 0),
                    "lat_rmse": round(m["lateral_rmse_mean"], 6) if m.get("lateral_rmse_mean") is not None else "",
                    "speed_rmse": round(m["speed_rmse_mean"], 6) if m.get("speed_rmse_mean") is not None else "",
                    "speed_corr": round(m["speed_corr_mean"], 4) if m.get("speed_corr_mean") is not None else "",
                    "time_diff": round(m["time_diff_mean"], 4) if m.get("time_diff_mean") is not None else "",
                    "goal_approach_speed_human": round(m["goal_approach_speed_human"], 4) if m.get("goal_approach_speed_human") is not None else "",
                    "goal_approach_speed_model": round(m["goal_approach_speed_model"], 4) if m.get("goal_approach_speed_model") is not None else "",
                    "goal_approach_speed_diff": round(m["goal_approach_speed_diff"], 4) if m.get("goal_approach_speed_diff") is not None else "",
                    "overshoot_rate": round(m["overshoot_rate"], 4) if m.get("overshoot_rate") is not None else "",
                    "mean_overshoot_dist_m": round(m["mean_overshoot_dist_m"], 6) if m.get("mean_overshoot_dist_m") is not None else "",
                    "mean_undershoot_dist_m": round(m["mean_undershoot_dist_m"], 6) if m.get("mean_undershoot_dist_m") is not None else "",
                    "mean_signed_dist_m": round(m["mean_signed_dist_m"], 6) if m.get("mean_signed_dist_m") is not None else "",
                }
                if has_baseline:
                    bl_m = p_bl.get(tid_str, {})
                    row["bl_lat_rmse"] = round(bl_m["lateral_rmse_mean"], 6) if bl_m.get("lateral_rmse_mean") is not None else ""
                    row["bl_speed_rmse"] = round(bl_m["speed_rmse_mean"], 6) if bl_m.get("speed_rmse_mean") is not None else ""
                    row["bl_speed_corr"] = round(bl_m["speed_corr_mean"], 4) if bl_m.get("speed_corr_mean") is not None else ""
                    row["bl_time_diff"] = round(bl_m["time_diff_mean"], 4) if bl_m.get("time_diff_mean") is not None else ""
                writer.writerow(row)
    print(f"  Saved: {progress_csv}")

    print("\n" + "=" * 70)
    print("Evaluation complete!")
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

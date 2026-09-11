"""
Load human trajectories and regenerate model trajectories for video rendering.

Human data comes from eval/human_data/raw/*.json.
Model trajectories are regenerated using fitted configs from eval/model_fitting/results/.
Tunnel geometry is built from TRIAL_CONDITIONS.
"""

import json
import math
import sys
import tempfile
import os
from pathlib import Path

import numpy as np

# Project paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
HUMAN_DATA_DIR = PROJECT_ROOT / "eval" / "human_data" / "raw"
FITTING_RESULTS_DIR = PROJECT_ROOT / "eval" / "model_fitting" / "results"

# Add project paths for imports
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "hcs_package" / "src"))

from experiment.environment import create_environment, generate_task_config
from experiment.utils import generateTunnelBoundaries

# Trial conditions (copied from run_eval.py to avoid circular imports)
TRIAL_CONDITIONS = {
    1:  {"type": "sigmoidal", "width": 0.01, "curvature": 0.025, "label": "Sinusoidal W=10mm"},
    2:  {"type": "sigmoidal", "width": 0.02, "curvature": 0.025, "label": "Sinusoidal W=20mm"},
    3:  {"type": "sigmoidal", "width": 0.03, "curvature": 0.025, "label": "Sinusoidal W=30mm"},
    4:  {"type": "sigmoidal", "width": 0.04, "curvature": 0.025, "label": "Sinusoidal W=40mm"},
    5:  {"type": "sigmoidal", "width": 0.05, "curvature": 0.025, "label": "Sinusoidal W=50mm"},
    6:  {"type": "corner", "width": 0.01, "num_corners": 2, "corner_offset": 0.1, "label": "Corner W=10mm"},
    7:  {"type": "corner", "width": 0.02, "num_corners": 2, "corner_offset": 0.1, "label": "Corner W=20mm"},
    8:  {"type": "corner", "width": 0.03, "num_corners": 2, "corner_offset": 0.1, "label": "Corner W=30mm"},
    9:  {"type": "corner", "width": 0.04, "num_corners": 2, "corner_offset": 0.1, "label": "Corner W=40mm"},
    10: {"type": "corner", "width": 0.05, "num_corners": 2, "corner_offset": 0.1, "label": "Corner W=50mm"},
    11: {"type": "sigmoidal", "width": 0.01, "curvature": 0.0, "label": "Straight W=10mm"},
    12: {"type": "sigmoidal", "width": 0.02, "curvature": 0.0, "label": "Straight W=20mm"},
    13: {"type": "sigmoidal", "width": 0.03, "curvature": 0.0, "label": "Straight W=30mm"},
    14: {"type": "sigmoidal", "width": 0.04, "curvature": 0.0, "label": "Straight W=40mm"},
    15: {"type": "sigmoidal", "width": 0.05, "curvature": 0.0, "label": "Straight W=50mm"},
    16: {"type": "sigmoidal", "width": 0.01, "curvature": 0.015, "label": "Gentle Sin W=10mm"},
    17: {"type": "sigmoidal", "width": 0.02, "curvature": 0.015, "label": "Gentle Sin W=20mm"},
    18: {"type": "sigmoidal", "width": 0.03, "curvature": 0.015, "label": "Gentle Sin W=30mm"},
    19: {"type": "sigmoidal", "width": 0.04, "curvature": 0.015, "label": "Gentle Sin W=40mm"},
    20: {"type": "sigmoidal", "width": 0.05, "curvature": 0.015, "label": "Gentle Sin W=50mm"},
    21: {"type": "sigmoidal", "width": 0.01, "curvature": 0.05, "label": "Sharp Sin W=10mm"},
    22: {"type": "sigmoidal", "width": 0.02, "curvature": 0.05, "label": "Sharp Sin W=20mm"},
    23: {"type": "sigmoidal", "width": 0.03, "curvature": 0.05, "label": "Sharp Sin W=30mm"},
    24: {"type": "sigmoidal", "width": 0.04, "curvature": 0.05, "label": "Sharp Sin W=40mm"},
    25: {"type": "sigmoidal", "width": 0.05, "curvature": 0.05, "label": "Sharp Sin W=50mm"},
}


def _build_task_config(trial_id):
    """Build task config and centerline for a trial condition."""
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
    return task_config, centerline


def load_human_trajectory(pid, trial_id, round_num=0):
    """Load a single human trajectory from raw data.

    Returns:
        dict with keys: trajectory ([[x,y],...] in metres), timestamps ([ms,...])
    """
    fpath = HUMAN_DATA_DIR / f"participant_{pid}.json"
    with open(fpath) as f:
        data = json.load(f)

    sessions = data.get("sessions", [])
    if not sessions:
        trial_data_array = data.get("trialData", [])
        sessions = [{"trialData": trial_data_array}] if trial_data_array else []

    for session in sessions:
        for td in session.get("trialData", []):
            if td.get("trial_id") == trial_id and td.get("round", 0) == round_num:
                traj_raw = td.get("trajectory", [])
                if isinstance(traj_raw[0], dict):
                    trajectory = [[p["x"], p["y"]] for p in traj_raw]
                else:
                    trajectory = traj_raw
                timestamps = td.get("timestamps", [])
                return {"trajectory": trajectory, "timestamps": timestamps}

    raise ValueError(f"Trial {trial_id} round {round_num} not found for {pid}")


def load_all_human_rounds(pid, trial_id):
    """Load all rounds for a participant-trial pair."""
    fpath = HUMAN_DATA_DIR / f"participant_{pid}.json"
    with open(fpath) as f:
        data = json.load(f)

    sessions = data.get("sessions", [])
    if not sessions:
        trial_data_array = data.get("trialData", [])
        sessions = [{"trialData": trial_data_array}] if trial_data_array else []

    rounds = []
    for session in sessions:
        for td in session.get("trialData", []):
            if td.get("trial_id") == trial_id:
                traj_raw = td.get("trajectory", [])
                if isinstance(traj_raw[0], dict):
                    trajectory = [[p["x"], p["y"]] for p in traj_raw]
                else:
                    trajectory = traj_raw
                timestamps = td.get("timestamps", [])
                rounds.append({
                    "round": td.get("round", 0),
                    "trajectory": trajectory,
                    "timestamps": timestamps,
                })
    return sorted(rounds, key=lambda r: r["round"])


def regenerate_model_trajectories(pid, trial_id, n_runs=5, seed=42):
    """Regenerate multiple model trajectories using fitted config.

    Returns:
        list of dicts, each with keys: trajectory ([[x,y],...] in metres), interval (float)
    """
    from hcs_package import CursorSimulator

    config_path = FITTING_RESULTS_DIR / f"{pid}_gam_config_s{seed}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    sim = CursorSimulator(str(config_path))
    interval = sim.interval

    task_config, _ = _build_task_config(trial_id)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        json.dump(task_config, tf)
        task_file = tf.name

    runs = []
    try:
        for _ in range(n_runs):
            traj_raw = sim.generate_trajectory_with_waypoints(
                task_file=task_file,
                max_steps=task_config.get("max_steps", 800),
                target_radius=task_config.get("target_radius", 0.01),
                use_optimal_path=True,
            )
            trajectory = [[x * 0.001, y * 0.001] for x, y, _ in traj_raw]
            runs.append({"trajectory": trajectory, "interval": interval})
    finally:
        os.unlink(task_file)

    return runs


def regenerate_model_trajectory(pid, trial_id, seed=42):
    """Regenerate a single model trajectory. Convenience wrapper."""
    runs = regenerate_model_trajectories(pid, trial_id, n_runs=1, seed=seed)
    return runs[0]


def find_best_pair(pid, trial_id, n_model_runs=5, seed=42):
    """Find the (human_round, model_run) pair with best match.

    Selects the pair that minimizes a combined score of completion time
    difference and lateral RMSE (after resampling to equal-length arrays).

    Returns:
        (human_data, model_data, info) where info has match details
    """
    human_rounds = load_all_human_rounds(pid, trial_id)
    model_runs = regenerate_model_trajectories(pid, trial_id, n_runs=n_model_runs, seed=seed)

    best_score = float("inf")
    best_h = None
    best_m = None
    best_info = {}

    for h in human_rounds:
        h_traj = np.array(h["trajectory"], dtype=float)
        h_ts = np.array(h["timestamps"], dtype=float)
        h_dur = (h_ts[-1] - h_ts[0]) / 1000.0

        for mi, m in enumerate(model_runs):
            m_traj = np.array(m["trajectory"], dtype=float)
            m_dur = len(m_traj) * m["interval"]

            # Score 1: completion time difference (normalised)
            time_diff = abs(h_dur - m_dur) / max(h_dur, 0.1)

            # Score 2: lateral RMSE after resampling to 100 progress bins
            n_resample = 100
            h_progress = np.linspace(0, 1, len(h_traj))
            m_progress = np.linspace(0, 1, len(m_traj))
            target_progress = np.linspace(0, 1, n_resample)

            h_x = np.interp(target_progress, h_progress, h_traj[:, 0])
            h_y = np.interp(target_progress, h_progress, h_traj[:, 1])
            m_x = np.interp(target_progress, m_progress, m_traj[:, 0])
            m_y = np.interp(target_progress, m_progress, m_traj[:, 1])

            lat_rmse = np.sqrt(np.mean((h_x - m_x)**2 + (h_y - m_y)**2))

            # Combined score (weight time and spatial equally, normalised)
            score = time_diff + lat_rmse / 0.01  # 0.01m = 10mm as reference scale

            if score < best_score:
                best_score = score
                best_h = h
                best_m = m
                best_info = {
                    "human_round": h["round"],
                    "model_run": mi,
                    "human_duration": h_dur,
                    "model_duration": m_dur,
                    "time_diff": time_diff,
                    "lat_rmse": lat_rmse,
                    "score": score,
                }

    print(f"  Best pair: human round {best_info['human_round']}, "
          f"model run {best_info['model_run']} "
          f"(time_diff={best_info['time_diff']:.2f}, "
          f"lat_rmse={best_info['lat_rmse']*1000:.1f}mm, "
          f"score={best_info['score']:.3f})")

    return best_h, best_m, best_info


def build_tunnel_geometry(trial_id):
    """Build tunnel geometry for rendering.

    Returns:
        dict with: centerline, left_boundary, right_boundary (all np arrays in metres),
                   width (float), label (str)
    """
    cond = TRIAL_CONDITIONS[trial_id]
    _, centerline = _build_task_config(trial_id)

    centerline_arr = np.array(centerline, dtype=float)
    left_bnd, right_bnd = generateTunnelBoundaries(
        centerline, cond["width"]
    )

    return {
        "centerline": centerline_arr,
        "left_boundary": np.array(left_bnd, dtype=float),
        "right_boundary": np.array(right_bnd, dtype=float),
        "width": cond["width"],
        "label": cond["label"],
    }

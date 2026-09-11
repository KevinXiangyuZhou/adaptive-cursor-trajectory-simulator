"""
Fit MPCC planner weights to an individual participant's steering data.

Supports phased fitting to reduce cost and improve robustness:
  Phase 1 (base):   Fit task-independent dynamics (adaptation modules disabled)
  Phase 2 (width):  Fit steering-law params (width adaptation) on sigmoidal tasks
  Phase 3 (shape):  Fit stop-and-go params (shape adaptation) on wide tasks
  Phase 4 (refine): Global refinement of all params on all tasks

Usage:
    # Phased fitting (recommended):
    python -m eval.model_fitting.run_fitting --pid P111602 --time-limit 1800 --phase all

    # Single-phase fitting:
    python -m eval.model_fitting.run_fitting --pid P111602 --time-limit 600 --phase base

    # Legacy mode (all params, all tasks, single phase):
    python -m eval.model_fitting.run_fitting --pid P111602 --time-limit 600
"""

import argparse
import copy
import json
import math
import multiprocessing
import os
import sys
import tempfile
import time
import warnings
from collections import OrderedDict
from pathlib import Path

import cma
import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "hcs_package" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

from experiment.environment import create_environment, generate_task_config
from hcs_package.cursor_simulator import CursorSimulator
from utils.stats import (
    resample_by_progress,
    resample_speeds_by_progress,
    speed_profile_correlation,
    speed_profile_rmse,
    trajectory_rmse,
)

HUMAN_DATA_DIR = PROJECT_ROOT / "eval" / "human_data" / "raw"
BASE_CONFIG_PATH = SCRIPT_DIR / "initial_user_config.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"

# ---------------------------------------------------------------------------
# Trial conditions (tunnel types)
# ---------------------------------------------------------------------------
TRIAL_CONDITIONS = {
    # Sinusoidal (curvature=0.025)
    1:  {"type": "sigmoidal", "width": 0.01, "curvature": 0.025, "label": "Sinusoidal W=10mm"},
    2:  {"type": "sigmoidal", "width": 0.02, "curvature": 0.025, "label": "Sinusoidal W=20mm"},
    3:  {"type": "sigmoidal", "width": 0.03, "curvature": 0.025, "label": "Sinusoidal W=30mm"},
    4:  {"type": "sigmoidal", "width": 0.04, "curvature": 0.025, "label": "Sinusoidal W=40mm"},
    5:  {"type": "sigmoidal", "width": 0.05, "curvature": 0.025, "label": "Sinusoidal W=50mm"},
    # Corner (2 corners, offset=0.1)
    6:  {"type": "corner", "width": 0.01, "num_corners": 2, "corner_offset": 0.1, "label": "Corner W=10mm"},
    7:  {"type": "corner", "width": 0.02, "num_corners": 2, "corner_offset": 0.1, "label": "Corner W=20mm"},
    8:  {"type": "corner", "width": 0.03, "num_corners": 2, "corner_offset": 0.1, "label": "Corner W=30mm"},
    9:  {"type": "corner", "width": 0.04, "num_corners": 2, "corner_offset": 0.1, "label": "Corner W=40mm"},
    10: {"type": "corner", "width": 0.05, "num_corners": 2, "corner_offset": 0.1, "label": "Corner W=50mm"},
    # Straight (curvature=0)
    11: {"type": "sigmoidal", "width": 0.01, "curvature": 0.0, "label": "Straight W=10mm"},
    12: {"type": "sigmoidal", "width": 0.02, "curvature": 0.0, "label": "Straight W=20mm"},
    13: {"type": "sigmoidal", "width": 0.03, "curvature": 0.0, "label": "Straight W=30mm"},
    14: {"type": "sigmoidal", "width": 0.04, "curvature": 0.0, "label": "Straight W=40mm"},
    15: {"type": "sigmoidal", "width": 0.05, "curvature": 0.0, "label": "Straight W=50mm"},
    # Gentle sinusoidal (curvature=0.015)
    16: {"type": "sigmoidal", "width": 0.01, "curvature": 0.015, "label": "Gentle Sin W=10mm"},
    17: {"type": "sigmoidal", "width": 0.02, "curvature": 0.015, "label": "Gentle Sin W=20mm"},
    18: {"type": "sigmoidal", "width": 0.03, "curvature": 0.015, "label": "Gentle Sin W=30mm"},
    19: {"type": "sigmoidal", "width": 0.04, "curvature": 0.015, "label": "Gentle Sin W=40mm"},
    20: {"type": "sigmoidal", "width": 0.05, "curvature": 0.015, "label": "Gentle Sin W=50mm"},
    # Sharp sinusoidal (curvature=0.05)
    21: {"type": "sigmoidal", "width": 0.01, "curvature": 0.05, "label": "Sharp Sin W=10mm"},
    22: {"type": "sigmoidal", "width": 0.02, "curvature": 0.05, "label": "Sharp Sin W=20mm"},
    23: {"type": "sigmoidal", "width": 0.03, "curvature": 0.05, "label": "Sharp Sin W=30mm"},
    24: {"type": "sigmoidal", "width": 0.04, "curvature": 0.05, "label": "Sharp Sin W=40mm"},
    25: {"type": "sigmoidal", "width": 0.05, "curvature": 0.05, "label": "Sharp Sin W=50mm"},
}

N_PROGRESS_BINS = 100
MAX_ROUND_DURATION_S = 20.0  # Filter out rounds exceeding this duration (seconds)

# Speed metrics are evaluated over the central portion of the path,
# excluding startup acceleration and target-approach deceleration.
SPEED_TRIM_START = 10   # skip first 10% of progress
SPEED_TRIM_END   = 90   # skip last 10% of progress

# ---------------------------------------------------------------------------
# Width-based train / test split
# ---------------------------------------------------------------------------
TRAIN_WIDTHS = {0.01, 0.03, 0.05}   # 10, 30, 50 mm
TEST_WIDTHS  = {0.02, 0.04}          # 20, 40 mm

TRAIN_TIDS = {tid for tid, c in TRIAL_CONDITIONS.items() if c["width"] in TRAIN_WIDTHS}
TEST_TIDS  = {tid for tid, c in TRIAL_CONDITIONS.items() if c["width"] in TEST_WIDTHS}

# Task-type groupings (straight = curvature 0, all others are steering)
STRAIGHT_TIDS = {tid for tid, c in TRIAL_CONDITIONS.items() if c.get("curvature", 1) == 0.0}
STEERING_TIDS = {tid for tid in TRIAL_CONDITIONS if tid not in STRAIGHT_TIDS}

# CMA-ES subset: one per type at 30mm for efficiency (5 tasks instead of 15)
CMAES_TIDS = {3, 8, 13, 18, 23}

# ---------------------------------------------------------------------------
# Tunable parameter specification (full set)
# ---------------------------------------------------------------------------
PARAM_SPEC = [
    # Speed control (steering law)
    {"name": "desired_speed",   "log_scale": False, "bounds": (0.05, 0.50)},
    {"name": "speed_alpha",     "log_scale": False, "bounds": (0.2, 1.5)},
    {"name": "speed_reference", "log_scale": False, "bounds": (0.005, 0.04)},
    {"name": "speed_floor",     "log_scale": False, "bounds": (0.1, 0.5)},
    {"name": "speed_ceil",      "log_scale": False, "bounds": (1.5, 4.0)},
    # Trajectory tracking (base model)
    {"name": "contour",         "log_scale": True,  "bounds": (0.0, 2.0)},   # 1–100
    {"name": "lag",             "log_scale": True,  "bounds": (-2.0, 1.0)},   # 0.01–10
    {"name": "jerk",            "log_scale": True,  "bounds": (-8.0, -4.0)},  # 1e-8–1e-4
    {"name": "constraint",      "log_scale": True,  "bounds": (1.8, 3.0)},   # ~63–1000
    # Speed modulation (stop-and-go)
    {"name": "straight_boost",  "log_scale": False, "bounds": (1.0, 3.0)},
    {"name": "boost_sharpness", "log_scale": False, "bounds": (1.0, 6.0)},
    {"name": "rate_alpha",      "log_scale": False, "bounds": (0.5, 3.0)},
    {"name": "rate_scale",      "log_scale": True,  "bounds": (4.0, 8.0)},  # 1e4–1e8
    # Curvature-responsive speed modulation
    {"name": "kappa_ref",       "log_scale": True,  "bounds": (0.5, 2.0)},  # 3–100 (1/m)
    {"name": "kappa_alpha",     "log_scale": False, "bounds": (0.5, 3.0)},
    {"name": "kappa_floor",     "log_scale": False, "bounds": (0.3, 0.9)},
    # Reference path
    {"name": "w_cut",           "log_scale": False, "bounds": (0.7, 2.0),
     "config_key": "reference_path"},
    {"name": "Th",              "log_scale": False, "bounds": (0.2, 0.5),
     "discrete_step": 0.05, "config_key": "top_level"},
]

# Build lookup for quick access
_PARAM_SPEC_BY_NAME = {s["name"]: s for s in PARAM_SPEC}

# ---------------------------------------------------------------------------
# Phase definitions for phased fitting
# ---------------------------------------------------------------------------
PHASE_DEFS = OrderedDict([
    ("base", {
        "desc": "Base dynamics (adaptation disabled)",
        "params": ["desired_speed", "contour", "lag", "jerk", "constraint", "w_cut", "Th"],
        "tasks": [1, 2, 3, 4],
        # Disable all adaptation modules
        "overrides": {
            "speed_floor": 1.0,
            "speed_ceil": 1.0,
            "rate_percentile": 100.0,
            "straight_boost": 1.0,
            "kappa_floor": 1.0,
        },
        "time_frac": 0.25,
        "sigma0": 0.15,
        "popsize": 18,
    }),
    ("width", {
        "desc": "Width adaptation (steering law)",
        "params": ["speed_alpha", "speed_reference", "speed_floor", "speed_ceil"],
        "tasks": [1, 2],  # sigmoidal narrow + wide (vary width, same shape)
        # Keep stop-and-go and curvature factor disabled
        "overrides": {
            "rate_percentile": 100.0,
            "straight_boost": 1.0,
            "kappa_floor": 1.0,
        },
        "time_frac": 0.15,
        "sigma0": 0.20,
        "popsize": 14,
    }),
    ("curvature", {
        "desc": "Curvature adaptation (slow in curves)",
        "params": ["kappa_ref", "kappa_alpha", "kappa_floor"],
        "tasks": [1, 2],  # sigmoidal tunnels where curvature factor matters most
        # Keep stop-and-go disabled so it doesn't confound curvature fitting
        "overrides": {
            "rate_percentile": 100.0,
            "straight_boost": 1.0,
        },
        "time_frac": 0.10,
        "sigma0": 0.25,
        "popsize": 12,
    }),
    ("shape", {
        "desc": "Shape adaptation (stop-and-go)",
        "params": ["straight_boost", "boost_sharpness", "rate_alpha", "rate_scale"],
        "tasks": [2, 4],  # wide tunnels: sigmoidal + corner (vary shape, same width)
        "overrides": {},
        "time_frac": 0.15,
        "sigma0": 0.20,
        "popsize": 14,
    }),
    ("refine", {
        "desc": "Global refinement (all params, all tasks)",
        "params": None,  # all params
        "tasks": [1, 2, 3, 4],
        "overrides": {},
        "time_frac": 0.35,
        "sigma0": 0.10,  # smaller sigma — starting from good initialization
        "popsize": 24,
    }),
])

# ---------------------------------------------------------------------------
# Loss weights and constants
# ---------------------------------------------------------------------------
# Weights express relative importance (after normalization by human variability).
LOSS_WEIGHTS = {
    "lateral_rmse": 1.0,
    "speed_rmse": 1.0,
    "speed_corr": 1.0,
    "time_diff": 1.0,
}

# Fallback scales if human variability can't be computed
DEFAULT_METRIC_SCALES = {
    "lateral_rmse": 0.003,
    "speed_rmse": 0.05,
    "speed_corr": 0.3,
    "time_diff": 0.15,
}

# Will be set by compute_human_variability_scales() at fitting time
METRIC_SCALES = dict(DEFAULT_METRIC_SCALES)


def compute_human_variability_scales(train_data, centerlines):
    """Compute per-metric scales from human inter-round variability.

    For each task with ≥2 rounds, compute metrics between all pairs of human
    rounds (treating one as "model" and the other as "human").  The median
    value per metric across all pairs becomes the scale, so that a loss of 1.0
    per metric corresponds to human-level variability.

    Updates the global METRIC_SCALES dict in-place and returns it.
    """
    from itertools import combinations

    pair_metrics = {"lateral_rmse": [], "speed_rmse": [], "speed_corr": [], "time_diff": []}

    for tid, rounds in train_data.items():
        if len(rounds) < 2 or tid not in centerlines:
            continue
        centerline = centerlines[tid]

        for ra, rb in combinations(rounds, 2):
            # Compute progress-aligned metrics between two human rounds
            _, _, lat_a = resample_by_progress(ra["trajectory"], centerline, N_PROGRESS_BINS)
            _, _, lat_b = resample_by_progress(rb["trajectory"], centerline, N_PROGRESS_BINS)
            _, spd_a = resample_speeds_by_progress(ra["speeds"], ra["trajectory"], centerline, N_PROGRESS_BINS)
            _, spd_b = resample_speeds_by_progress(rb["speeds"], rb["trajectory"], centerline, N_PROGRESS_BINS)

            pair_metrics["lateral_rmse"].append(trajectory_rmse(lat_a, lat_b))
            spd_a_t = spd_a[SPEED_TRIM_START:SPEED_TRIM_END]
            spd_b_t = spd_b[SPEED_TRIM_START:SPEED_TRIM_END]
            pair_metrics["speed_rmse"].append(speed_profile_rmse(spd_a_t, spd_b_t))
            corr = speed_profile_correlation(spd_a_t, spd_b_t)
            pair_metrics["speed_corr"].append(1.0 - corr)

            ta = (ra["timestamps"][-1] - ra["timestamps"][0]) / 1000.0
            tb = (rb["timestamps"][-1] - rb["timestamps"][0]) / 1000.0
            pair_metrics["time_diff"].append(abs(ta - tb) / max(tb, 0.1))

    if pair_metrics["lateral_rmse"]:
        for key in pair_metrics:
            med = float(np.median(pair_metrics[key]))
            METRIC_SCALES[key] = max(med, 1e-6)  # avoid division by zero
        print(f"  Human variability scales: {METRIC_SCALES}")
    else:
        print("  Warning: not enough round pairs, using default scales")

    return METRIC_SCALES


CMA_SIGMA0 = 0.15
CMA_POPSIZE = 24
MAX_SIM_STEPS = 120
INCOMPLETE_PENALTY = 100.0
WALL_MARGIN_WEIGHT = 20.0


# ---------------------------------------------------------------------------
# Parameter encoding / decoding (supports full or subset)
# ---------------------------------------------------------------------------

def _get_active_specs(param_names=None):
    """Get PARAM_SPEC entries for given param names, or all if None."""
    if param_names is None:
        return PARAM_SPEC
    return [_PARAM_SPEC_BY_NAME[n] for n in param_names if n in _PARAM_SPEC_BY_NAME]


def encode(param_dict, active_specs=None):
    """Map parameter values → normalised [0,1] vector."""
    specs = active_specs or PARAM_SPEC
    vec = np.zeros(len(specs))
    for i, spec in enumerate(specs):
        val = param_dict[spec["name"]]
        lo, hi = spec["bounds"]
        if spec["log_scale"]:
            val = math.log10(val)
        vec[i] = (val - lo) / (hi - lo)
    return np.clip(vec, 0.0, 1.0)


def decode(vec, active_specs=None):
    """Map normalised [0,1] vector → parameter dict."""
    specs = active_specs or PARAM_SPEC
    out = {}
    for i, spec in enumerate(specs):
        v = float(np.clip(vec[i], 0.0, 1.0))
        lo, hi = spec["bounds"]
        val = lo + v * (hi - lo)
        if spec["log_scale"]:
            val = 10.0 ** val
        if "discrete_step" in spec:
            step = spec["discrete_step"]
            val = round(round(val / step) * step, 6)
            val = max(lo, min(hi, val))
        out[spec["name"]] = val
    return out


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _normalize_traj(raw):
    if not raw:
        return []
    if isinstance(raw[0], dict):
        return [[p["x"], p["y"]] for p in raw]
    return raw


def _compute_speeds(traj, timestamps, window=5):
    n = len(traj)
    if n < 2 or len(timestamps) != n:
        return []
    raw = []
    for i in range(n):
        if i == 0:
            p0, p1 = traj[0], traj[1]
            dt = (timestamps[1] - timestamps[0]) / 1000.0
        elif i == n - 1:
            p0, p1 = traj[-2], traj[-1]
            dt = (timestamps[-1] - timestamps[-2]) / 1000.0
        else:
            p0, p1 = traj[i - 1], traj[i + 1]
            dt = (timestamps[i + 1] - timestamps[i - 1]) / 1000.0
        dist = math.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)
        raw.append(dist / dt if dt > 0 else 0.0)
    half = window // 2
    smoothed = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        smoothed.append(sum(raw[lo:hi]) / (hi - lo))
    return smoothed


def load_participant_data(pid):
    """Load all trial rounds for a participant."""
    participant_file = None
    for fpath in sorted(HUMAN_DATA_DIR.glob("*.json")):
        try:
            with open(fpath) as f:
                data = json.load(f)
            if data.get("participantId", "") == pid:
                participant_file = fpath
                break
        except Exception:
            continue

    if participant_file is None:
        raise FileNotFoundError(f"No data file found for participant {pid}")

    with open(participant_file) as f:
        data = json.load(f)

    sessions = data.get("sessions", [])
    if not sessions:
        td = data.get("trialData", [])
        sessions = [{"trialData": td}] if td else []

    result = {}
    for sess in sessions:
        for td in sess.get("trialData", []):
            tid = td.get("trial_id")
            if tid not in TRIAL_CONDITIONS:
                continue
            traj = _normalize_traj(td.get("trajectory", []))
            ts = td.get("timestamps", [])
            if len(traj) < 5 or len(ts) < 5:
                continue
            duration_s = (ts[-1] - ts[0]) / 1000.0
            if duration_s > MAX_ROUND_DURATION_S:
                print(f"  Filtered out trial {tid} round {td.get('round', '?')}: "
                      f"duration {duration_s:.1f}s > {MAX_ROUND_DURATION_S}s")
                continue
            speeds = _compute_speeds(traj, ts)
            if tid not in result:
                result[tid] = []
            result[tid].append({
                "trajectory": traj,
                "speeds": speeds,
                "timestamps": ts,
                "round": td.get("round", 1),
            })
    return result


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def split_train_test(participant_trials, rng=None):
    """Width-based train/test split: train on {10,30,50}mm, test on {20,40}mm.

    All rounds of each task go entirely to train or test (no within-task splitting).
    rng is accepted for API compatibility but not used.
    """
    train, test = {}, {}
    for tid, rounds in participant_trials.items():
        if not rounds:
            continue
        if tid in TRAIN_TIDS:
            train[tid] = rounds
        elif tid in TEST_TIDS:
            test[tid] = rounds
    return train, test


# ---------------------------------------------------------------------------
# Environment / centerline setup
# ---------------------------------------------------------------------------

def build_task(cond):
    """Build task config and extract centerline for a trial condition."""
    t_radius = cond["width"] * 0.5
    if cond["type"] == "sigmoidal":
        env_dict = {
            "env_type": "tunnel_steering_smooth",
            "screen_width": 460, "screen_height": 260,
            "tunnelWidth": cond["width"], "curvature": cond["curvature"],
            "max_steps": MAX_SIM_STEPS, "target_radius": t_radius,
        }
    else:
        env_dict = {
            "env_type": "tunnel_steering_corner",
            "screen_width": 460, "screen_height": 260,
            "tunnelWidth": cond["width"],
            "num_corners": cond["num_corners"],
            "corner_offset": cond["corner_offset"],
            "max_steps": MAX_SIM_STEPS, "target_radius": t_radius,
        }
    env = create_environment(env_dict)
    task_config = generate_task_config(env, include_constraints=True)
    centerline = [[x, y] for x, y in env["centerline"]]
    return task_config, centerline


def build_all_tasks():
    """Build task configs and centerlines for all 4 conditions."""
    tasks, centerlines = {}, {}
    for tid, cond in TRIAL_CONDITIONS.items():
        tc, cl = build_task(cond)
        tasks[tid] = tc
        centerlines[tid] = cl
    return tasks, centerlines


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_single_sim(sim, task_config):
    """Run one simulation, return trajectory, speeds (m/s), and interval (s)."""
    interval = sim.interval
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        json.dump(task_config, tf)
        task_file = tf.name
    try:
        traj_raw = sim.generate_trajectory_with_waypoints(
            task_file=task_file,
            max_steps=task_config.get("max_steps", 100),
            target_radius=task_config.get("target_radius", 0.01),
            use_optimal_path=True,
        )
    finally:
        os.unlink(task_file)

    scale = 0.001
    traj = [[x * scale, y * scale] for x, y, _ in traj_raw]
    n = len(traj)
    # Central-difference + 5-point moving average (matching _compute_speeds for human data)
    raw_speeds = []
    for i in range(n):
        if i == 0:
            p0, p1 = traj[0], traj[1] if n > 1 else traj[0]
            dt_local = interval
        elif i == n - 1:
            p0, p1 = traj[-2], traj[-1]
            dt_local = interval
        else:
            p0, p1 = traj[i - 1], traj[i + 1]
            dt_local = 2.0 * interval
        d = math.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)
        raw_speeds.append(d / dt_local if dt_local > 0 else 0.0)
    window = 5
    half = window // 2
    speeds = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        speeds.append(sum(raw_speeds[lo:hi]) / (hi - lo))
    return traj, speeds, interval


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_trial_metrics(model_traj, model_speeds, human_trial, centerline,
                          sim_interval=0.05, tunnel_half_width=None):
    """Compute progress-aligned metrics between model and one human trial."""
    h_traj = human_trial["trajectory"]
    h_speeds = human_trial["speeds"]
    h_ts = human_trial["timestamps"]

    _, _, lat_m = resample_by_progress(model_traj, centerline, N_PROGRESS_BINS)
    _, _, lat_h = resample_by_progress(h_traj, centerline, N_PROGRESS_BINS)
    _, spd_m = resample_speeds_by_progress(model_speeds, model_traj, centerline, N_PROGRESS_BINS)
    _, spd_h = resample_speeds_by_progress(h_speeds, h_traj, centerline, N_PROGRESS_BINS)

    lat_rmse = trajectory_rmse(lat_m, lat_h)

    # Trim speed profiles to central portion — excludes startup acceleration
    # and target-approach deceleration (task framing, not steering behavior)
    spd_m_trim = spd_m[SPEED_TRIM_START:SPEED_TRIM_END]
    spd_h_trim = spd_h[SPEED_TRIM_START:SPEED_TRIM_END]
    spd_rmse = speed_profile_rmse(spd_m_trim, spd_h_trim)
    spd_corr = speed_profile_correlation(spd_m_trim, spd_h_trim)

    # Symmetric relative time difference
    human_time = (h_ts[-1] - h_ts[0]) / 1000.0 if len(h_ts) >= 2 else 1.0
    model_time = len(model_traj) * sim_interval
    time_diff = abs(model_time - human_time) / max(human_time, 0.1)

    # Wall margin penalty: penalize trajectory points near tunnel walls
    wall_margin = 0.0
    if tunnel_half_width is not None and tunnel_half_width > 0:
        lat_m_arr = np.array(lat_m)
        wall_proximity = np.abs(lat_m_arr) / tunnel_half_width
        violations = np.maximum(wall_proximity - 0.6, 0.0)
        wall_margin = float(np.mean(violations ** 2))

    return {
        "lateral_rmse": lat_rmse,
        "speed_rmse": spd_rmse,
        "speed_corr": spd_corr,
        "time_diff": time_diff,
        "wall_margin": wall_margin,
    }


def metrics_to_loss(metrics):
    """Weighted combination of metrics → scalar loss.

    Each metric is normalized by its scale (human inter-round variability)
    so that a loss of 1.0 per metric ≈ human-level variability.
    """
    loss = 0.0
    for key in ("lateral_rmse", "speed_rmse", "time_diff"):
        scale = METRIC_SCALES.get(key, 1.0)
        loss += LOSS_WEIGHTS[key] * metrics[key] / scale
    # speed_corr: higher is better, so use (1 - corr) as cost
    corr_scale = METRIC_SCALES.get("speed_corr", 1.0)
    loss += LOSS_WEIGHTS["speed_corr"] * (1.0 - metrics["speed_corr"]) / corr_scale
    if "wall_margin" in metrics:
        loss += WALL_MARGIN_WEIGHT * metrics["wall_margin"]
    return loss


# ---------------------------------------------------------------------------
# Objective function (top-level for pickling with multiprocessing)
# ---------------------------------------------------------------------------

_TOP_LEVEL_PARAMS = {s["name"] for s in PARAM_SPEC if s.get("config_key") == "top_level"}
_TOP_LEVEL_PARAMS.add("Th")

_REF_PATH_PARAMS = {s["name"] for s in PARAM_SPEC if s.get("config_key") == "reference_path"}


def _apply_params(cfg, param_dict):
    """Apply parameters to a config dict (in-place)."""
    for key, val in param_dict.items():
        if key in _TOP_LEVEL_PARAMS:
            cfg[key] = val
        elif key in _REF_PATH_PARAMS:
            cfg.setdefault("reference_path", {})[key] = val
        else:
            cfg["planner_weights"][key] = val


def _eval_single(args):
    """Evaluate one parameter vector. Designed for multiprocessing.Pool.map().

    args: (param_vec, base_config, train_data, task_configs, centerlines,
           active_spec_names, active_tasks, overrides)
    """
    (param_vec, base_config, train_data, task_configs, centerlines,
     active_spec_names, active_tasks, overrides) = args

    # Decode only active params
    active_specs = _get_active_specs(active_spec_names)
    active_params = decode(param_vec, active_specs)

    cfg = copy.deepcopy(base_config)
    _apply_params(cfg, active_params)

    # Apply phase overrides (e.g., disable adaptation modules)
    if overrides:
        _apply_params(cfg, overrides)

    # Fit noiseless
    cfg["nc"] = [0, 0]

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".json", prefix="fit_worker_")
    os.close(tmp_fd)
    try:
        with open(tmp_path, "w") as f:
            json.dump(cfg, f)

        try:
            sim = CursorSimulator(str(tmp_path))
        except Exception:
            return 1e6

        total_loss = 0.0
        n_tasks = 0
        for tid in sorted(active_tasks):
            if tid not in train_data:
                continue
            rounds = train_data[tid]
            n_tasks += 1

            try:
                model_traj, model_speeds, sim_interval = run_single_sim(sim, task_configs[tid])
            except Exception:
                total_loss += 1e6
                continue

            if len(model_traj) < 5:
                total_loss += 1e6
                continue

            # Check completion
            cl = np.array(centerlines[tid])
            last_pt = np.array(model_traj[-1])
            cl_diffs = np.diff(cl, axis=0)
            cl_lens = np.linalg.norm(cl_diffs, axis=1)
            cl_cum = np.concatenate([[0.0], np.cumsum(cl_lens)])
            cl_total = cl_cum[-1]
            best_arc = 0.0
            for i in range(len(cl) - 1):
                seg = cl[i + 1] - cl[i]
                seg_len2 = np.dot(seg, seg)
                t = 0.0 if seg_len2 < 1e-18 else float(np.clip(np.dot(last_pt - cl[i], seg) / seg_len2, 0.0, 1.0))
                arc = cl_cum[i] + t * cl_lens[i]
                if arc > best_arc:
                    best_arc = arc
            completion = best_arc / cl_total if cl_total > 0 else 0.0

            if len(model_traj) >= MAX_SIM_STEPS and completion < 0.95:
                total_loss += INCOMPLETE_PENALTY * (1.0 - completion)
                continue

            half_w = TRIAL_CONDITIONS[tid]["width"] * 0.5
            round_losses = []
            for human_trial in rounds:
                metrics = compute_trial_metrics(
                    model_traj, model_speeds, human_trial, centerlines[tid],
                    sim_interval=sim_interval,
                    tunnel_half_width=half_w,
                )
                round_losses.append(metrics_to_loss(metrics))
            total_loss += sum(round_losses) / len(round_losses)

        # Normalize by number of tasks so loss is comparable across phases
        if n_tasks > 0:
            total_loss = total_loss * 4 / n_tasks

        return total_loss
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# CMA-ES fitting (single phase)
# ---------------------------------------------------------------------------

def run_fitting(base_config, train_data, task_configs, centerlines,
                time_limit, seed, n_workers=None,
                active_params=None, active_tasks=None, overrides=None,
                sigma0=None, popsize=None):
    """Run CMA-ES optimisation for one phase.

    active_params: list of param names to optimize (None = all)
    active_tasks: list of trial IDs to evaluate on (None = all)
    overrides: dict of param→value to force during this phase
    """
    active_specs = _get_active_specs(active_params)
    active_spec_names = [s["name"] for s in active_specs]
    if active_tasks is None:
        active_tasks = sorted(TRIAL_CONDITIONS.keys())
    if overrides is None:
        overrides = {}
    if sigma0 is None:
        sigma0 = CMA_SIGMA0
    if popsize is None:
        popsize = CMA_POPSIZE
    if n_workers is None:
        n_workers = min(multiprocessing.cpu_count(), popsize, 8)

    # Initial point from base config
    initial_params = {}
    for spec in active_specs:
        name = spec["name"]
        if name in _TOP_LEVEL_PARAMS:
            initial_params[name] = base_config.get(name,
                base_config["planner_weights"].get(name))
        elif name in _REF_PATH_PARAMS:
            initial_params[name] = base_config.get("reference_path", {}).get(name,
                base_config["planner_weights"].get(name))
        else:
            initial_params[name] = base_config["planner_weights"][name]
    x0 = encode(initial_params, active_specs)

    # Build shared args for workers
    shared_args = (base_config, train_data, task_configs, centerlines,
                   active_spec_names, active_tasks, overrides)

    def obj_fn(x):
        return _eval_single((x, *shared_args))

    # Evaluate initial point
    initial_loss = obj_fn(x0)
    print(f"  Initial loss: {initial_loss:.6f}")
    print(f"  Using {n_workers} parallel workers, {len(active_specs)} params, "
          f"tasks={active_tasks}")

    n_params = len(active_specs)
    es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, {
        "bounds": [[0.0] * n_params, [1.0] * n_params],
        "popsize": popsize,
        "seed": seed,
        "verb_disp": 0,
        "verb_log": 0,
        "verb_filenameprefix": "",
        "verbose": -9,
    })

    best_x = x0.copy()
    best_loss = initial_loss
    loss_history = [{
        "generation": 0,
        "best_loss": initial_loss,
        "mean_loss": initial_loss,
        "elapsed_sec": 0.0,
    }]

    start = time.time()
    generation = 0

    with multiprocessing.Pool(processes=n_workers) as pool:
        while not es.stop():
            elapsed = time.time() - start
            if elapsed > time_limit:
                print(f"  Time limit reached ({elapsed:.0f}s)")
                break

            solutions = es.ask()
            eval_args = [(x, *shared_args) for x in solutions]
            fitness = pool.map(_eval_single, eval_args)
            es.tell(solutions, fitness)

            generation += 1
            gen_best_idx = int(np.argmin(fitness))
            if fitness[gen_best_idx] < best_loss:
                best_loss = fitness[gen_best_idx]
                best_x = np.array(solutions[gen_best_idx]).copy()

            elapsed = time.time() - start
            loss_history.append({
                "generation": generation,
                "best_loss": float(best_loss),
                "mean_loss": float(np.mean(fitness)),
                "elapsed_sec": round(elapsed, 1),
            })

            print(
                f"  Gen {generation:3d}: best={best_loss:.6f}  "
                f"mean={np.mean(fitness):.6f}  "
                f"elapsed={elapsed:.0f}s"
            )

    elapsed_total = time.time() - start

    fitted_params = decode(best_x, active_specs)
    return fitted_params, best_loss, loss_history, initial_params, elapsed_total, generation


# ---------------------------------------------------------------------------
# Phased fitting orchestrator
# ---------------------------------------------------------------------------

def run_phased_fitting(base_config, train_data, task_configs, centerlines,
                       total_time_limit, seed, n_workers=None, phases=None):
    """Run multi-phase fitting: base → width → shape → refine.

    Each phase optimizes a subset of params on a subset of tasks,
    then passes the best params forward to the next phase.
    """
    if phases is None:
        phases = list(PHASE_DEFS.keys())

    cfg = copy.deepcopy(base_config)
    all_history = {}
    all_params = {}

    # Collect initial params (all)
    initial_params = {}
    for spec in PARAM_SPEC:
        name = spec["name"]
        if name in _TOP_LEVEL_PARAMS:
            initial_params[name] = cfg.get(name, cfg["planner_weights"].get(name))
        elif name in _REF_PATH_PARAMS:
            initial_params[name] = cfg.get("reference_path", {}).get(name,
                cfg["planner_weights"].get(name))
        else:
            initial_params[name] = cfg["planner_weights"][name]

    total_start = time.time()

    for phase_name in phases:
        phase_def = PHASE_DEFS[phase_name]
        elapsed_so_far = time.time() - total_start
        remaining = total_time_limit - elapsed_so_far

        if remaining < 30:
            print(f"\n--- Skipping phase '{phase_name}': insufficient time ({remaining:.0f}s) ---")
            continue

        # Compute time budget for this phase
        # Remaining phases get proportional share of remaining time
        remaining_phases = [p for p in phases if p not in all_history]
        total_frac = sum(PHASE_DEFS[p]["time_frac"] for p in remaining_phases)
        phase_time = remaining * (phase_def["time_frac"] / total_frac) if total_frac > 0 else remaining

        print(f"\n{'='*60}")
        print(f"Phase: {phase_name} — {phase_def['desc']}")
        print(f"  Time budget: {phase_time:.0f}s  "
              f"Params: {phase_def['params'] or 'all'}")
        print(f"  Tasks: {phase_def['tasks']}")
        if phase_def["overrides"]:
            print(f"  Overrides: {phase_def['overrides']}")
        print(f"{'='*60}")

        fitted, loss, history, init_p, elapsed, n_gens = run_fitting(
            cfg, train_data, task_configs, centerlines,
            time_limit=phase_time,
            seed=seed,
            n_workers=n_workers,
            active_params=phase_def["params"],
            active_tasks=phase_def["tasks"],
            overrides=phase_def["overrides"],
            sigma0=phase_def["sigma0"],
            popsize=phase_def["popsize"],
        )

        print(f"\n  Phase '{phase_name}' complete: {n_gens} generations, "
              f"best loss: {loss:.6f}")
        for name, val in fitted.items():
            print(f"    {name:20s}: {init_p[name]:.6g} → {val:.6g}")

        # Update config with fitted params for next phase
        _apply_params(cfg, fitted)
        all_history[phase_name] = history
        all_params[phase_name] = fitted

    # Collect final params (all, from the updated config)
    final_params = {}
    for spec in PARAM_SPEC:
        name = spec["name"]
        if name in _TOP_LEVEL_PARAMS:
            final_params[name] = cfg.get(name, cfg["planner_weights"].get(name))
        elif name in _REF_PATH_PARAMS:
            final_params[name] = cfg.get("reference_path", {}).get(name,
                cfg["planner_weights"].get(name))
        else:
            final_params[name] = cfg["planner_weights"][name]

    total_elapsed = time.time() - total_start
    return final_params, initial_params, all_history, all_params, cfg, total_elapsed


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_on_data(fitted_params, base_config, data, task_configs, centerlines):
    """Evaluate fitted params against a set of trials (train or test)."""
    cfg = copy.deepcopy(base_config)
    _apply_params(cfg, fitted_params)
    cfg["nc"] = [0, 0]

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".json", prefix="eval_cfg_")
    os.close(tmp_fd)
    try:
        with open(tmp_path, "w") as f:
            json.dump(cfg, f)
        sim = CursorSimulator(str(tmp_path))

        results = {}
        for tid in sorted(data.keys()):
            rounds = data[tid]
            if not rounds:
                continue

            try:
                model_traj, model_speeds, sim_interval = run_single_sim(sim, task_configs[tid])
            except Exception:
                results[str(tid)] = {"error": "simulation failed"}
                continue

            if len(model_traj) < 5:
                results[str(tid)] = {"error": "degenerate trajectory"}
                continue

            all_metrics = []
            for human_trial in rounds:
                m = compute_trial_metrics(
                    model_traj, model_speeds, human_trial, centerlines[tid],
                    sim_interval=sim_interval,
                )
                all_metrics.append(m)

            results[str(tid)] = {
                "n_rounds": len(all_metrics),
                "lateral_rmse_mean": float(np.mean([m["lateral_rmse"] for m in all_metrics])),
                "speed_rmse_mean": float(np.mean([m["speed_rmse"] for m in all_metrics])),
                "speed_corr_mean": float(np.mean([m["speed_corr"] for m in all_metrics])),
                "time_diff_mean": float(np.mean([m["time_diff"] for m in all_metrics])),
                "loss_mean": float(np.mean([metrics_to_loss(m) for m in all_metrics])),
            }
    finally:
        os.unlink(tmp_path)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fit MPCC parameters to a participant's steering data."
    )
    parser.add_argument("--pid", required=True, help="Participant ID (e.g. P111602)")
    parser.add_argument("--time-limit", type=int, default=600,
                        help="Total time limit in seconds (default: 600)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: auto)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: eval/model_fitting/results/)")
    parser.add_argument("--phase", type=str, default=None,
                        help="Fitting mode: 'all' for phased (base→width→shape→refine), "
                             "or single phase name (base/width/shape/refine), "
                             "or omit for legacy single-phase all-params")
    parser.add_argument("--mode", type=str, default="parametric",
                        choices=["parametric", "gam"],
                        help="Speed model type: 'parametric' (default) or 'gam' "
                             "(two-stage GAM fitting)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(BASE_CONFIG_PATH) as f:
        base_config = json.load(f)
    base_config.pop("_tuning_notes", None)

    print(f"=== Model Fitting: {args.pid} ===")
    print(f"  Time limit: {args.time_limit}s, seed: {args.seed}, "
          f"workers: {args.workers or 'auto'}, phase: {args.phase or 'legacy'}")

    # Load participant data
    print("\nLoading participant data...")
    participant_trials = load_participant_data(args.pid)
    for tid in sorted(participant_trials.keys()):
        n = len(participant_trials[tid])
        print(f"  Trial {tid} ({TRIAL_CONDITIONS[tid]['label']}): {n} rounds")

    if not participant_trials:
        print("ERROR: No trial data found.")
        sys.exit(1)

    rng = np.random.default_rng(args.seed)
    train_data, test_data = split_train_test(participant_trials, rng)
    print(f"\nTrain/test split (width-based):")
    print(f"  Train tasks ({len(train_data)}): {sorted(train_data.keys())}")
    print(f"  Test tasks  ({len(test_data)}): {sorted(test_data.keys())}")

    print("\nBuilding task environments...")
    task_configs, centerlines = build_all_tasks()

    # ---- GAM mode: two-stage fitting ----
    if args.mode == "gam":
        from eval.model_fitting.fit_speed_model import run_two_stage_fitting
        result = run_two_stage_fitting(
            pid=args.pid,
            base_config_path=str(BASE_CONFIG_PATH),
            time_limit=args.time_limit,
            seed=args.seed,
        )
        return

    # ---- Phased fitting ----
    if args.phase:
        if args.phase == "all":
            phases = list(PHASE_DEFS.keys())
        elif args.phase in PHASE_DEFS:
            phases = [args.phase]
        else:
            print(f"ERROR: Unknown phase '{args.phase}'. "
                  f"Options: all, {', '.join(PHASE_DEFS.keys())}")
            sys.exit(1)

        final_params, initial_params, all_history, all_params, full_cfg, elapsed = \
            run_phased_fitting(
                base_config, train_data, task_configs, centerlines,
                args.time_limit, args.seed, n_workers=args.workers,
                phases=phases,
            )

        print(f"\n{'='*60}")
        print(f"Phased fitting complete in {elapsed:.1f}s")
        print(f"\n  Final fitted parameters:")
        for spec in PARAM_SPEC:
            name = spec["name"]
            init_val = initial_params[name]
            fit_val = final_params[name]
            changed = " *" if abs(fit_val - init_val) / max(abs(init_val), 1e-10) > 0.01 else ""
            print(f"    {name:20s}: {init_val:.6g} → {fit_val:.6g}{changed}")

        # Flatten loss history
        flat_history = []
        for phase_name, hist in all_history.items():
            for entry in hist:
                flat_history.append({**entry, "phase": phase_name})

        # Evaluate
        print("\nEvaluating on training data...")
        train_metrics = evaluate_on_data(
            final_params, base_config, train_data, task_configs, centerlines)
        for tid, m in sorted(train_metrics.items()):
            if "error" in m:
                print(f"  Trial {tid}: {m['error']}")
            else:
                print(f"  Trial {tid}: lat_rmse={m['lateral_rmse_mean']:.5f}  "
                      f"spd_corr={m['speed_corr_mean']:.3f}  loss={m['loss_mean']:.5f}")

        print("\nEvaluating on test data...")
        test_metrics = evaluate_on_data(
            final_params, base_config, test_data, task_configs, centerlines)
        for tid, m in sorted(test_metrics.items()):
            if "error" in m:
                print(f"  Trial {tid}: {m['error']}")
            else:
                print(f"  Trial {tid}: lat_rmse={m['lateral_rmse_mean']:.5f}  "
                      f"spd_corr={m['speed_corr_mean']:.3f}  loss={m['loss_mean']:.5f}  "
                      f"({m['n_rounds']} rounds)")

        print("\nEvaluating initial (unfitted) params on test data...")
        initial_test_metrics = evaluate_on_data(
            initial_params, base_config, test_data, task_configs, centerlines)
        for tid, m in sorted(initial_test_metrics.items()):
            if "error" in m:
                print(f"  Trial {tid}: {m['error']}")
            else:
                print(f"  Trial {tid}: lat_rmse={m['lateral_rmse_mean']:.5f}  "
                      f"spd_corr={m['speed_corr_mean']:.3f}  loss={m['loss_mean']:.5f}")

        # Compute best total train loss
        train_losses = [m["loss_mean"] for m in train_metrics.values()
                        if isinstance(m, dict) and "loss_mean" in m]
        best_train_loss = sum(train_losses) if train_losses else float("inf")

        out = {
            "participant_id": args.pid,
            "seed": args.seed,
            "time_limit_sec": args.time_limit,
            "elapsed_sec": round(elapsed, 1),
            "mode": "phased",
            "phases_run": list(all_history.keys()),
            "fitted_parameters": {k: round(v, 8) for k, v in final_params.items()},
            "initial_parameters": {k: round(v, 8) for k, v in initial_params.items()},
            "train_loss": round(best_train_loss, 6),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "initial_test_metrics": initial_test_metrics,
            "loss_history": flat_history,
            "phase_params": {k: {pk: round(pv, 8) for pk, pv in v.items()}
                            for k, v in all_params.items()},
            "test_rounds": {str(k): v for k, v in test_rounds.items()},
            "full_config": full_cfg,
        }

        out_path = output_dir / f"{args.pid}_fit_s{args.seed}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to: {out_path}")

    # ---- Legacy single-phase fitting ----
    else:
        print(f"\nStarting CMA-ES optimisation (time limit={args.time_limit}s)...")
        fitted_params, best_loss, loss_history, initial_params, elapsed, n_gens = run_fitting(
            base_config, train_data, task_configs, centerlines,
            args.time_limit, args.seed, n_workers=args.workers,
        )

        print(f"\nFitting complete: {n_gens} generations in {elapsed:.1f}s")
        print(f"  Best training loss: {best_loss:.6f}")
        print(f"\n  Fitted parameters:")
        for spec in PARAM_SPEC:
            name = spec["name"]
            init_val = initial_params[name]
            fit_val = fitted_params[name]
            print(f"    {name:20s}: {init_val:.6g} → {fit_val:.6g}")

        print("\nEvaluating on training data...")
        train_metrics = evaluate_on_data(
            fitted_params, base_config, train_data, task_configs, centerlines)
        for tid, m in sorted(train_metrics.items()):
            if "error" in m:
                print(f"  Trial {tid}: {m['error']}")
            else:
                print(f"  Trial {tid}: lat_rmse={m['lateral_rmse_mean']:.5f}  "
                      f"spd_corr={m['speed_corr_mean']:.3f}  loss={m['loss_mean']:.5f}")

        print("\nEvaluating on test data...")
        test_metrics = evaluate_on_data(
            fitted_params, base_config, test_data, task_configs, centerlines)
        for tid, m in sorted(test_metrics.items()):
            if "error" in m:
                print(f"  Trial {tid}: {m['error']}")
            else:
                print(f"  Trial {tid}: lat_rmse={m['lateral_rmse_mean']:.5f}  "
                      f"spd_corr={m['speed_corr_mean']:.3f}  loss={m['loss_mean']:.5f}  "
                      f"({m['n_rounds']} rounds)")

        print("\nEvaluating initial (unfitted) params on test data...")
        initial_test_metrics = evaluate_on_data(
            initial_params, base_config, test_data, task_configs, centerlines)
        for tid, m in sorted(initial_test_metrics.items()):
            if "error" in m:
                print(f"  Trial {tid}: {m['error']}")
            else:
                print(f"  Trial {tid}: lat_rmse={m['lateral_rmse_mean']:.5f}  "
                      f"spd_corr={m['speed_corr_mean']:.3f}  loss={m['loss_mean']:.5f}")

        full_config = copy.deepcopy(base_config)
        _apply_params(full_config, fitted_params)

        out = {
            "participant_id": args.pid,
            "seed": args.seed,
            "time_limit_sec": args.time_limit,
            "elapsed_sec": round(elapsed, 1),
            "n_generations": n_gens,
            "mode": "legacy",
            "fitted_parameters": {k: round(v, 8) for k, v in fitted_params.items()},
            "initial_parameters": {k: round(v, 8) for k, v in initial_params.items()},
            "train_loss": round(best_loss, 6),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "initial_test_metrics": initial_test_metrics,
            "loss_history": loss_history,
            "test_rounds": {str(k): v for k, v in test_rounds.items()},
            "full_config": full_config,
        }

        out_path = output_dir / f"{args.pid}_fit_s{args.seed}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()

"""
Shared statistics helpers for the eval-new-data pipeline.

Re-exports the general-purpose trajectory/speed-profile stats from
eval/utils/stats.py, and adds the law-specific helpers (Fitts ID,
centerline geometry, ID4SCS regression) that don't belong in that
shared, law-agnostic module.
"""

import math
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

from utils.stats import (  # noqa: E402
    resample_by_progress,
    resample_speeds_by_progress,
    trajectory_rmse,
    speed_profile_rmse,
    speed_profile_correlation,
)

__all__ = [
    "resample_by_progress",
    "resample_speeds_by_progress",
    "trajectory_rmse",
    "speed_profile_rmse",
    "speed_profile_correlation",
    "centerline_arc_length",
    "segment_arc_lengths",
    "fitts_id",
    "fit_id4scs",
    "compute_comparison_metrics",
]

# Mirrors eval-fitts-continuous-well/run_eval.py exactly: skip first 10% of
# progress (startup acceleration), evaluate to 100% (full path incl. goal
# approach) when computing speed_rmse/speed_corr.
SPEED_TRIM_START = 10
SPEED_TRIM_END = 100
N_PROGRESS_BINS = 100
GOAL_APPROACH_BINS = (90, 100)


def centerline_arc_length(centerline):
    """Sum of Euclidean segment lengths along a centerline/path."""
    total = 0.0
    for i in range(1, len(centerline)):
        dx = centerline[i][0] - centerline[i - 1][0]
        dy = centerline[i][1] - centerline[i - 1][1]
        total += math.sqrt(dx * dx + dy * dy)
    return total


def segment_arc_lengths(centerline):
    """
    Split a two-segment tunnel's centerline at the same midpoint index used
    by _create_tunnel_env (experiment/environment.py) to change corridor
    width, and return (A1, A2): the arc length of each half.
    """
    n = len(centerline)
    if n < 2:
        return 0.0, 0.0
    mid = n // 2
    seg1 = centerline[: mid + 1]
    seg2 = centerline[mid:]
    return centerline_arc_length(seg1), centerline_arc_length(seg2)


def fitts_id(D, R):
    """Shannon ID = log2(D/(2R) + 1). R is target radius; 2R is target width."""
    return math.log2(D / (2 * R) + 1) if R > 0 else 0.0


def fit_id4scs(rows):
    """
    Multiple linear regression for the ID4SCS model:
        MT = a + b*S1 + c*C + d*S2

    Args:
        rows: list of dicts with keys 'S1', 'C', 'S2', 'MT'.

    Returns:
        dict with a, b, c, d, r_squared, n. Empty dict if too few rows.
    """
    if len(rows) < 5:
        return {}

    X = np.column_stack([
        np.ones(len(rows)),
        [r["S1"] for r in rows],
        [r["C"] for r in rows],
        [r["S2"] for r in rows],
    ])
    y = np.array([r["MT"] for r in rows])

    coeffs, _residuals, _rank, _sv = np.linalg.lstsq(X, y, rcond=None)
    a, b, c, d = coeffs
    pred = X @ coeffs
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "a": round(float(a), 6),
        "b": round(float(b), 6),
        "c": round(float(c), 6),
        "d": round(float(d), 6),
        "r_squared": round(r_sq, 4),
        "n": len(rows),
    }


def compute_comparison_metrics(model_trajs, model_speeds, model_cts,
                                human_trajs, human_speeds, human_cts,
                                centerline):
    """
    Experiment-main-style human-vs-model comparison metrics, ported from
    eval-fitts-continuous-well/run_eval.py's _build_condition_summary:
    lateral RMSE, speed-profile RMSE/correlation (trimmed to bins 10-100,
    i.e. skip the first 10% of progress), goal-approach speed (mean speed
    over the last 10% of progress) for both sides plus their relative
    difference, and relative completion-time difference.

    Requires >= 5 points on both sides of `centerline` (a straight 2-point
    line is fine for point-to-point Fitts tasks) to resample against;
    returns {} if there isn't enough data to compute a meaningful profile.
    """
    if not centerline or len(centerline) < 2:
        return {}

    all_spd_m, all_lat_m = [], []
    for traj, speeds in zip(model_trajs, model_speeds):
        if len(traj) < 5 or len(traj) != len(speeds):
            continue
        _, _, lat_m = resample_by_progress(traj, centerline, N_PROGRESS_BINS)
        _, spd_m = resample_speeds_by_progress(speeds, traj, centerline, N_PROGRESS_BINS)
        all_lat_m.append(lat_m)
        all_spd_m.append(spd_m)
    if not all_spd_m:
        return {}

    mean_spd_m = np.mean(all_spd_m, axis=0)
    mean_lat_m = np.mean(all_lat_m, axis=0)
    lo, hi = GOAL_APPROACH_BINS
    goal_spd_m = float(np.mean(mean_spd_m[lo:hi])) if len(mean_spd_m) >= hi else None

    result = {"goal_approach_speed_model": round(goal_spd_m, 4) if goal_spd_m is not None else None}

    all_spd_h, all_lat_h = [], []
    for traj, speeds in zip(human_trajs, human_speeds):
        if len(traj) < 5 or len(traj) != len(speeds):
            continue
        _, _, lat_h = resample_by_progress(traj, centerline, N_PROGRESS_BINS)
        _, spd_h = resample_speeds_by_progress(speeds, traj, centerline, N_PROGRESS_BINS)
        all_lat_h.append(lat_h)
        all_spd_h.append(spd_h)

    if all_lat_h and all_lat_m:
        mean_lat_h = np.mean(all_lat_h, axis=0)
        mean_spd_h = np.mean(all_spd_h, axis=0)

        lat_rmse = float(trajectory_rmse(mean_lat_m, mean_lat_h))

        spd_m_trim = mean_spd_m[SPEED_TRIM_START:SPEED_TRIM_END]
        spd_h_trim = mean_spd_h[SPEED_TRIM_START:SPEED_TRIM_END]
        spd_rmse = float(speed_profile_rmse(spd_m_trim, spd_h_trim))
        spd_corr = float(speed_profile_correlation(spd_m_trim, spd_h_trim))

        goal_spd_h = float(np.mean(mean_spd_h[lo:hi])) if len(mean_spd_h) >= hi else None
        goal_spd_diff = None
        if goal_spd_h is not None and goal_spd_m is not None and goal_spd_h > 0:
            goal_spd_diff = abs(goal_spd_m - goal_spd_h) / goal_spd_h

        mean_human_time = float(np.mean(human_cts)) if human_cts else None
        mean_model_time = float(np.mean(model_cts)) if model_cts else None
        time_diff = None
        if mean_human_time and mean_model_time and mean_human_time > 0:
            time_diff = abs(mean_model_time - mean_human_time) / mean_human_time

        result.update({
            "lateral_rmse": round(lat_rmse, 6),
            "speed_rmse": round(spd_rmse, 6),
            "speed_corr": round(spd_corr, 4),
            "goal_approach_speed_human": round(goal_spd_h, 4) if goal_spd_h is not None else None,
            "goal_approach_speed_diff": round(goal_spd_diff, 4) if goal_spd_diff is not None else None,
            "time_diff": round(time_diff, 4) if time_diff is not None else None,
            "human_time_mean_s": round(mean_human_time, 4) if mean_human_time is not None else None,
            "model_time_mean_s": round(mean_model_time, 4) if mean_model_time is not None else None,
        })

    return result

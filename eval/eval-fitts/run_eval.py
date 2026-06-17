"""
Fitts' Law Evaluation Experiment.

Systematically varies target radius R across multiple widths W to evaluate
whether the model's goal_precision term produces Fitts'-like behavior:
    MT = a + b * ID,  ID = log2(D/(2R) + 1)

W (tunnel width) governs steering difficulty; R (target radius) governs
endpoint precision. ID varies through R — this isolates the goal_precision
term. Human data exists only at R = W/2 (hard-stop termination); all other
R values are model-only. Metrics against human are still computed at R = W/2
and at all other R values (where they reflect model-vs-human comparison with
the caveat that termination mechanics differ). The instructor noted that
speed_rmse and speed_corr are the most important metrics — these are
preserved in full.

SPEED_TRIM_START / SPEED_TRIM_END follow experiment-main exactly:
    - skip first 10% of progress (startup acceleration)
    - evaluate up to 100% (full path, including goal approach)
    - SPEED_TRIM_END=100 also drives the x-axis start of progress plots
      via plot_speeds_vs_progress_enhanced's xlim(0.1, 1.0)

Per-participant mode (default for evaluation):
    python run_eval.py --per-participant --rounds 3 --trials 11 12 13 14 15

Directory structure:
    results/
        participant_{pid}/
            trial_{tid}_R{r_mult:.2f}/
                trajectories_t{tid}.png
                speeds_enhanced_t{tid}.png
                speeds_progress_enhanced_t{tid}.png
                results_summary.json
            speed_profiles_W{width_mm}mm.png   <- one subplot per R value
                                                   (same width), shared y-axis
                                                   so deceleration depth is
                                                   comparable across R/W
        fitts_results.csv           <- one row per run
        fitts_condition_summary.csv <- one row per (participant, tid, r_mult)
        progress_metrics.csv        <- one row per (participant, tid, r_mult), all metrics
        fitts_regression.json       <- regression table for instructor
"""

import argparse
import concurrent.futures
import csv
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "hcs_package" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

from experiment.environment import create_environment, generate_task_config
from hcs_package.cursor_simulator import CursorSimulator

EXPERIMENT_MAIN_DIR = PROJECT_ROOT / "eval" / "experiment-main"
sys.path.insert(0, str(EXPERIMENT_MAIN_DIR))
from utils.plot_utils import (
    plot_experiment_results,
    plot_enhanced_speed_profiles,
    plot_speeds_vs_progress_enhanced,
    compute_progress_along_path,
)
from utils.stats import (
    resample_by_progress,
    resample_speeds_by_progress,
    trajectory_rmse,
    speed_profile_rmse,
    speed_profile_correlation,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HUMAN_DATA_DIR      = PROJECT_ROOT / "human_data" / "raw"
RESULTS_DIR         = SCRIPT_DIR / "results"
SIM_CACHE_DIR       = RESULTS_DIR / "sim_cache"
DEFAULT_CONFIG      = PROJECT_ROOT / "experiment" / "user_configurations" / "customized.json"
FITTING_RESULTS_DIR = PROJECT_ROOT / "eval" / "model_fitting" / "results"

WINDOW_WIDTH         = 0.46
WINDOW_HEIGHT        = 0.26
MAX_ROUND_DURATION_S = 60

# Speed metric trim — mirrors experiment-main exactly.
# SPEED_TRIM_START skips startup acceleration (first 10% of progress).
# SPEED_TRIM_END=100 evaluates the full path including goal approach,
# which is critical for detecting Fitts'-like deceleration.
# These also drive the x-axis of plot_speeds_vs_progress_enhanced
# (xlim 0.1 -> 1.0 corresponds to bins 10-100).
SPEED_TRIM_START = 10
SPEED_TRIM_END   = 100
N_PROGRESS_BINS  = 100

# R multiplier that matches human data (R = W/2)
HUMAN_R_MULT = 0.5

# ---------------------------------------------------------------------------
# Condition definitions
# ---------------------------------------------------------------------------
BASE_CONDITIONS = {
    11: {"type": "sigmoidal", "width": 0.01, "curvature": 0.0, "label": "Straight W=10mm"},
    12: {"type": "sigmoidal", "width": 0.02, "curvature": 0.0, "label": "Straight W=20mm"},
    13: {"type": "sigmoidal", "width": 0.03, "curvature": 0.0, "label": "Straight W=30mm"},
    14: {"type": "sigmoidal", "width": 0.04, "curvature": 0.0, "label": "Straight W=40mm"},
    15: {"type": "sigmoidal", "width": 0.05, "curvature": 0.0, "label": "Straight W=50mm"},
}

DEFAULT_R_MULTIPLIERS = [0.1, 0.25, 0.5, 1.0, 2.0]


def build_fitts_conditions(trial_ids, r_mults):
    """Return dict keyed by (tid, r_mult) with full condition info."""
    conditions = {}
    for tid in trial_ids:
        base = BASE_CONDITIONS[tid]
        for r_mult in r_mults:
            conditions[(tid, r_mult)] = {
                **base,
                "target_radius": base["width"] * r_mult,
                "r_mult":        r_mult,
                "label":         f"W={int(base['width']*1000)}mm R={r_mult}W",
                "has_human":     (r_mult == HUMAN_R_MULT),
            }
    return conditions


def condition_folder_name(tid, r_mult):
    return f"trial_{tid}_R{r_mult:.2f}"


def condition_cache_key(tid, r_mult):
    return f"{tid}_R{r_mult:.2f}"


# ---------------------------------------------------------------------------
# Sim cache helpers
# ---------------------------------------------------------------------------

def save_sim_cache(cache_dict, cache_path):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache_dict, f)
    print(f"  Sim cache saved to {cache_path}")


def load_sim_cache(cache_path):
    if not cache_path.exists():
        return None
    try:
        with open(cache_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  Warning: failed to load sim cache ({e}), will re-run")
        return None


# ---------------------------------------------------------------------------
# Environment builders
# ---------------------------------------------------------------------------

def _build_sigmoidal_config(width, curvature, target_radius=None):
    """
    Build task config for a sigmoidal tunnel.
    target_radius overrides the default W/2 to drive the Fitts' sweep.
    """
    if target_radius is None:
        target_radius = width * 0.5

    env_dict = {
        "env_type":    "tunnel_steering_smooth",
        "screen_width":  460,
        "screen_height": 260,
        "tunnelWidth":   width,
        "curvature":     curvature,
        "max_steps":     800,
        "target_radius": target_radius,
    }
    environment = create_environment(env_dict)
    task_config = generate_task_config(environment, include_constraints=True)
    centerline  = environment["centerline"]
    return task_config, centerline


def generate_centerline(tid):
    cond = BASE_CONDITIONS[tid]
    _, centerline = _build_sigmoidal_config(cond["width"], cond["curvature"])
    return [[x, y] for x, y in centerline]


def centerline_arc_length(centerline):
    total = 0.0
    for i in range(1, len(centerline)):
        dx = centerline[i][0] - centerline[i - 1][0]
        dy = centerline[i][1] - centerline[i - 1][1]
        total += math.sqrt(dx * dx + dy * dy)
    return total


def fitts_id(D, R):
    """Shannon ID = log2(D/(2R) + 1). R is target radius; 2R is target width."""
    return math.log2(D / (2 * R) + 1) if R > 0 else 0.0


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _path_length(trajectory):
    total = 0.0
    for i in range(1, len(trajectory)):
        dx = trajectory[i][0] - trajectory[i - 1][0]
        dy = trajectory[i][1] - trajectory[i - 1][1]
        total += math.sqrt(dx * dx + dy * dy)
    return total


def _compute_speeds(trajectory, timestamps, window_size=5):
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
    if not traj_raw:
        return []
    if isinstance(traj_raw[0], dict):
        return [[p["x"], p["y"]] for p in traj_raw]
    return traj_raw


def _approach_speed(speeds, trim_pct=0.90):
    """Mean speed over the last (1-trim_pct) fraction of the trajectory."""
    if not speeds:
        return 0.0
    start = max(0, int(len(speeds) * trim_pct))
    tail  = speeds[start:]
    return float(np.mean(tail)) if tail else 0.0


# ---------------------------------------------------------------------------
# Human data loading
# ---------------------------------------------------------------------------

def load_trials_by_participant(trial_ids):
    """
    Load human data from raw JSON files, organised by participant and trial.
    Mirrors experiment-main's load_trials_by_participant exactly.

    Returns:
        {pid: {tid: {round: {trajectory, speeds, timestamps, completion_time}}}}
    """
    trial_ids_set = set(trial_ids)
    all_data = {}

    for fpath in sorted(HUMAN_DATA_DIR.glob("*.json")):
        try:
            with open(fpath) as f:
                data = json.load(f)

            pid      = data.get("participantId", fpath.stem)
            sessions = data.get("sessions", [])
            if not sessions:
                td_arr   = data.get("trialData", [])
                sessions = [{"trialData": td_arr}] if td_arr else []

            if pid not in all_data:
                all_data[pid] = {}

            for session in sessions:
                for trial_data in session.get("trialData", []):
                    tid       = trial_data.get("trial_id")
                    round_num = trial_data.get("round", 0)
                    if tid not in trial_ids_set:
                        continue

                    if tid not in all_data[pid]:
                        all_data[pid][tid] = {}

                    traj_raw   = _normalize_trajectory(trial_data.get("trajectory", []))
                    timestamps = trial_data.get("timestamps", [])
                    if len(traj_raw) < 2 or len(timestamps) < 2:
                        continue

                    duration_s = (timestamps[-1] - timestamps[0]) / 1000.0
                    if duration_s > MAX_ROUND_DURATION_S:
                        print(f"  Filtered {pid} trial {tid} round {round_num}: "
                              f"{duration_s:.1f}s > {MAX_ROUND_DURATION_S}s")
                        continue

                    speeds = _compute_speeds(traj_raw, timestamps)
                    if len(speeds) != len(traj_raw):
                        if len(speeds) < len(traj_raw):
                            speeds.extend([speeds[-1] if speeds else 0.0]
                                          * (len(traj_raw) - len(speeds)))
                        else:
                            speeds = speeds[:len(traj_raw)]

                    all_data[pid][tid][round_num] = {
                        "trajectory":      traj_raw,
                        "speeds":          speeds,
                        "timestamps":      timestamps,
                        "completion_time": trial_data.get("completionTime", duration_s),
                    }
        except Exception as e:
            print(f"  Warning: error loading {fpath.name}: {e}")

    return all_data


# ---------------------------------------------------------------------------
# Simulator runner
# ---------------------------------------------------------------------------

def run_simulator_for_condition(sim, tid, target_radius, n_runs=1):
    """
    Run simulator for one (tid, target_radius) condition.
    target_radius is stored in each record for CSV verification.
    """
    cond = BASE_CONDITIONS.get(tid, {})
    if not cond:
        return []

    task_config, _ = _build_sigmoidal_config(
        cond["width"], cond["curvature"], target_radius=target_radius
    )

    interval = sim.interval
    records  = []

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        json.dump(task_config, tf)
        task_file = tf.name

    try:
        for _ in range(n_runs):
            traj_raw = sim.generate_trajectory_with_waypoints(
                task_file=task_file,
                max_steps=task_config.get("max_steps", 800),
                target_radius=task_config.get("target_radius", target_radius),
                use_optimal_path=True,
            )

            scale  = 0.001
            traj   = [[x * scale, y * scale] for x, y, _ in traj_raw]
            n_pts  = len(traj)
            speeds = []
            for i in range(1, n_pts):
                d = math.sqrt(
                    (traj[i][0] - traj[i - 1][0]) ** 2 +
                    (traj[i][1] - traj[i - 1][1]) ** 2
                )
                speeds.append(d / interval)
            if speeds:
                speeds.insert(0, speeds[0])

            pl  = _path_length(traj)
            ct  = len(traj_raw) * interval
            app = _approach_speed(speeds)

            records.append({
                "trajectory":      traj,
                "speeds":          speeds,
                "completion_time": ct,
                "path_length":     pl,
                "avg_speed":       pl / ct if ct > 0 else 0.0,
                "approach_speed":  app,
                "target_radius":   target_radius,   # verification column
            })
    finally:
        os.unlink(task_file)

    return records


# ---------------------------------------------------------------------------
# Per-condition metrics (mirrors save_trial_results_summary from experiment-main)
# ---------------------------------------------------------------------------

def _build_condition_summary(
    participant_id, tid, r_mult, cond,
    valid_records, all_records,
    centerline,
    human_traj_list=None,
    human_speeds_list=None,
    human_timestamps_list=None,
):
    """
    Compute and return results_summary dict for one (pid, tid, r_mult) condition.

    human_* lists are only non-None when r_mult == HUMAN_R_MULT, i.e. the
    condition that matches the human data. For all other R values, the
    experiment-main metrics (speed_rmse, speed_corr, lateral_rmse, time_diff)
    are still computed against the human R=W/2 data as a reference — your
    instructor wants to see how these metrics change as R varies, even though
    the termination mechanics differ at non-W/2 radii.
    """
    W  = cond["width"]
    R  = cond["target_radius"]
    D  = centerline_arc_length(centerline) if centerline else 0.0
    ID = fitts_id(D, R)

    m_trajs  = [r["trajectory"]      for r in valid_records]
    m_speeds = [r["speeds"]          for r in valid_records]
    m_cts    = [r["completion_time"] for r in valid_records]
    m_app    = [r["approach_speed"]  for r in valid_records]
    m_tps    = [ID / ct for ct in m_cts if ct > 0]

    h_trajs  = human_traj_list      or []
    h_speeds = human_speeds_list    or []
    h_ts     = human_timestamps_list or []

    agg = {}
    if m_trajs and centerline:
        all_spd_m, all_lat_m = [], []
        for j in range(len(m_trajs)):
            if len(m_trajs[j]) < 5:
                continue
            _, _, lat_m = resample_by_progress(m_trajs[j], centerline, N_PROGRESS_BINS)
            _, spd_m    = resample_speeds_by_progress(
                m_speeds[j], m_trajs[j], centerline, N_PROGRESS_BINS
            )
            all_lat_m.append(lat_m)
            all_spd_m.append(spd_m)

        if all_spd_m:
            mean_spd_m = np.mean(all_spd_m, axis=0)
            mean_lat_m = np.mean(all_lat_m, axis=0)

            # Goal approach speed from progress bins 90-100 (experiment-main metric)
            goal_spd_m = (float(np.mean(mean_spd_m[90:100]))
                          if len(mean_spd_m) >= 100 else None)

            # ----------------------------------------------------------------
            # Human-comparison metrics
            # These are computed whenever human data is available (passed in).
            # At R=W/2 this is the fair comparison. At other R values it still
            # shows how the model speed profile compares to human R=W/2 data —
            # useful for understanding how much R changes the profile shape.
            # ----------------------------------------------------------------
            lat_rmse   = None
            spd_rmse   = None
            spd_corr   = None
            time_diff  = None
            goal_spd_h = None
            goal_spd_diff = None
            human_time = None

            if h_trajs and centerline:
                all_spd_h, all_lat_h = [], []
                human_times = []

                for i in range(len(h_trajs)):
                    if len(h_trajs[i]) < 5:
                        continue
                    _, _, lat_h = resample_by_progress(
                        h_trajs[i], centerline, N_PROGRESS_BINS
                    )
                    _, spd_h = resample_speeds_by_progress(
                        h_speeds[i], h_trajs[i], centerline, N_PROGRESS_BINS
                    )
                    all_lat_h.append(lat_h)
                    all_spd_h.append(spd_h)
                    if len(h_ts[i]) >= 2:
                        human_times.append(
                            (h_ts[i][-1] - h_ts[i][0]) / 1000.0
                        )

                if all_lat_h and all_lat_m:
                    mean_lat_h = np.mean(all_lat_h, axis=0)
                    mean_spd_h = np.mean(all_spd_h, axis=0)

                    lat_rmse = float(trajectory_rmse(mean_lat_m, mean_lat_h))

                    # Apply SPEED_TRIM_START / SPEED_TRIM_END exactly as
                    # experiment-main does in save_trial_results_summary
                    spd_m_trim = mean_spd_m[SPEED_TRIM_START:SPEED_TRIM_END]
                    spd_h_trim = mean_spd_h[SPEED_TRIM_START:SPEED_TRIM_END]
                    spd_rmse   = float(speed_profile_rmse(spd_m_trim, spd_h_trim))
                    spd_corr   = float(speed_profile_correlation(spd_m_trim, spd_h_trim))

                    goal_spd_h = (float(np.mean(mean_spd_h[90:100]))
                                  if len(mean_spd_h) >= 100 else None)
                    if goal_spd_h is not None and goal_spd_m is not None and goal_spd_h > 0:
                        goal_spd_diff = float(
                            abs(goal_spd_m - goal_spd_h) / goal_spd_h
                        )

                    mean_human_time  = float(np.mean(human_times)) if human_times else None
                    mean_model_time  = float(np.mean(m_cts)) if m_cts else None
                    if mean_human_time and mean_model_time and mean_human_time > 0:
                        time_diff = float(
                            abs(mean_model_time - mean_human_time) / mean_human_time
                        )
                    human_time = mean_human_time

            agg = {
                # ---- Fitts-specific ----
                "ID":                         round(ID, 4),
                "D_m":                        round(D, 4),
                "target_radius_m":            round(R, 6),
                "R_over_W":                   round(r_mult, 4),
                "has_human_data":             bool(h_trajs),
                "throughput_mean_bps":        round(float(np.mean(m_tps)), 4) if m_tps else None,
                "throughput_std_bps":         round(float(np.std(m_tps)),  4) if m_tps else None,
                "MT_mean_s":                  round(float(np.mean(m_cts)), 4) if m_cts else None,
                "MT_std_s":                   round(float(np.std(m_cts)),  4) if m_cts else None,
                "approach_speed_mean_ms":     round(float(np.mean(m_app)), 6) if m_app else None,
                "approach_speed_std_ms":      round(float(np.std(m_app)),  6) if m_app else None,
                # ---- Experiment-main metrics (preserved in full) ----
                "lateral_rmse_mean":          round(lat_rmse,  6) if lat_rmse  is not None else None,
                "speed_rmse_mean":            round(spd_rmse,  6) if spd_rmse  is not None else None,
                "speed_corr_mean":            round(spd_corr,  4) if spd_corr  is not None else None,
                "time_diff_mean":             round(time_diff, 4) if time_diff is not None else None,
                "goal_approach_speed_human":  round(goal_spd_h,    4) if goal_spd_h    is not None else None,
                "goal_approach_speed_model":  round(goal_spd_m,    4) if goal_spd_m    is not None else None,
                "goal_approach_speed_diff":   round(goal_spd_diff, 4) if goal_spd_diff is not None else None,
                # ---- Human timing reference ----
                "human_time_mean_s":          round(human_time, 4) if human_time is not None else None,
            }

    return {
        "participant_id": participant_id,
        "trial_id":       tid,
        "r_mult":         r_mult,
        "condition":      {k: v for k, v in cond.items()},
        "human": {
            "n_rounds":            len(h_trajs) if (human_traj_list or []) else 0,
            "avg_completion_time": agg.get("human_time_mean_s"),
        },
        "model": {
            "n_runs":              len(all_records),
            "n_valid_runs":        len(valid_records),
            "avg_completion_time": float(np.mean(m_cts)) if m_cts else 0.0,
            "avg_speed":           float(np.mean([r["avg_speed"] for r in valid_records]))
                                   if valid_records else 0.0,
            "runs": [
                {
                    "run":             j,
                    "completion_time": r["completion_time"],
                    "avg_speed":       r["avg_speed"],
                    "approach_speed":  r["approach_speed"],
                    "target_radius":   r["target_radius"],  # verification
                }
                for j, r in enumerate(all_records)
            ],
        },
        "metrics": agg,
    }


# ---------------------------------------------------------------------------
# Per-condition worker (runs in process pool)
# ---------------------------------------------------------------------------

def _process_condition(
    config_path, speed_model_path,
    participant_id, tid, r_mult, cond,
    trial_data_dict,                   # {round: {trajectory, speeds, timestamps, ...}}
    n_rounds, existing_records,
    trial_folder_str, centerline,
):
    """
    Simulate + plot + save summary for one (participant, tid, r_mult).
    Designed to run inside a ProcessPoolExecutor.
    """
    import matplotlib
    matplotlib.use("Agg")

    trial_folder = Path(trial_folder_str)
    trial_folder.mkdir(parents=True, exist_ok=True)

    # ---- human trajectories for this tid ----
    human_trajectories   = []
    human_speeds_list    = []
    human_timestamps_list = []
    for round_num, round_data in (trial_data_dict or {}).items():
        traj = round_data["trajectory"]
        if len(traj) >= 2:
            human_trajectories.append(traj)
            human_speeds_list.append(round_data["speeds"])
            human_timestamps_list.append(round_data["timestamps"])

    # ---- simulation (incremental) ----
    existing = existing_records or []
    delta    = n_rounds - len(existing)
    if delta > 0:
        sim = CursorSimulator(str(config_path))
        if speed_model_path:
            from hcs_package.speed_model import GAMSpeedModel
            sim.speed_model = GAMSpeedModel.load(str(speed_model_path))
        new_recs    = run_simulator_for_condition(sim, tid, cond["target_radius"], n_runs=delta)
        all_records = existing + new_recs
    else:
        all_records = existing[:n_rounds]

    all_records_unfiltered = all_records[:n_rounds]

    # ---- filter diverged / timed-out runs ----
    _MAX_CT = 39.5
    valid_records = []
    for ri, r in enumerate(all_records_unfiltered):
        if r["completion_time"] >= _MAX_CT:
            print(f"  Filtered: {participant_id} tid={tid} R={r_mult} run={ri} "
                  f"CT={r['completion_time']:.2f}s", flush=True)
            continue
        traj = r.get("trajectory", [])
        if traj:
            xs = [p[0] for p in traj]
            ys = [p[1] for p in traj]
            if min(xs) < -0.1 or max(xs) > 0.56 or min(ys) < -0.1 or max(ys) > 0.36:
                print(f"  Filtered: {participant_id} tid={tid} R={r_mult} run={ri} "
                      f"out of bounds", flush=True)
                continue
        valid_records.append(r)

    model_trajectories = [r["trajectory"] for r in valid_records]
    model_speeds_list  = [r["speeds"]     for r in valid_records]

    # ---- plots ----
    segment_data = {
        "sim": {"trajectories": model_trajectories, "speeds": model_speeds_list},
    }
    # CHANGED: human data is now overlaid at every R value (not just R=W/2),
    # purely for visual comparison in the trajectory and speeds-vs-progress
    # plots. This is independent of the metrics computation in
    # _build_condition_summary, which already compares against human R=W/2
    # data at all R values — this only affects what gets drawn.
    if human_trajectories:
        segment_data["human"] = {
            "trajectories": human_trajectories,
            "speeds":       human_speeds_list,
        }

    all_results    = {tid: {0: segment_data}}
    tunnel_paths   = {tid: centerline}
    tunnel_widths  = {tid: cond["width"]}   # tunnel width for rendering, not R
    trial_metadata = {tid: {0: {0: {"condition": cond}}}}

    plot_experiment_results(
        all_results, trial_folder, tunnel_paths, tunnel_widths, trial_metadata
    )
    plot_enhanced_speed_profiles(all_results, trial_folder, time_step=0.05)
    if centerline:
        plot_speeds_vs_progress_enhanced(
            all_results, trial_folder, tunnel_paths, bin_size=0.1
        )

    # ---- summary JSON ----
    # Pass human data for all R values so metrics are always computed where
    # human data exists. At non-W/2 R values, the metrics reflect the
    # model-vs-human comparison with differing termination mechanics — this
    # is noted in the summary and is still scientifically useful.
    summary = _build_condition_summary(
        participant_id, tid, r_mult, cond,
        valid_records, all_records_unfiltered,
        centerline,
        human_traj_list=human_trajectories,
        human_speeds_list=human_speeds_list,
        human_timestamps_list=human_timestamps_list,
    )
    summary_path = trial_folder / "results_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    metrics = summary.get("metrics", {})
    return {
        "participant_id":  participant_id,
        "tid":             tid,
        "r_mult":          r_mult,
        "all_sim_records": all_records,
        "valid_records":   valid_records,
        "n_model":         len(valid_records),
        "n_human":         len(human_trajectories),
        "metrics":         metrics,
        # Returned so main() can build the per-participant, per-width
        # "speed_profiles_W{width_mm}mm.png" summary plot after all R
        # conditions for this (participant, tid) have been processed.
        # Kept minimal (no full all_records) to limit IPC payload size.
        "segment_data":    segment_data,
    }


# ---------------------------------------------------------------------------
# Per-width speed-profile summary plot (one PNG per participant per width,
# laying every R/W condition's speeds_progress_enhanced-style subplot side
# by side on a shared y-axis so deceleration depth is comparable across R).
# ---------------------------------------------------------------------------

def _binned_speed_rows(trajectories, speeds_list, tunnel_path, label,
                        bin_size=0.1):
    """
    Replicates the per-run progress-binning from plot_speeds_vs_progress_enhanced
    (mean speed per progress bin, per run), returned as long-form rows
    ready to feed into a seaborn dataframe: {"Progress", "Speed (m/s)", "Type"}.
    """
    rows = []
    if not tunnel_path:
        return rows

    progress_bins    = np.arange(0.0, 1.0 + bin_size, bin_size)
    progress_centers = (progress_bins[:-1] + progress_bins[1:]) / 2.0
    progress_centers[-1] = 1.0

    for traj, speeds in zip(trajectories, speeds_list):
        if len(traj) != len(speeds) or len(traj) == 0:
            continue

        progress_values = compute_progress_along_path(traj, tunnel_path)

        binned_speeds = [[] for _ in range(len(progress_centers))]
        for progress, speed in zip(progress_values, speeds):
            bin_idx = int(np.clip(progress / bin_size, 0, len(progress_centers) - 1))
            binned_speeds[bin_idx].append(speed)

        for bin_idx, progress_center in enumerate(progress_centers):
            if binned_speeds[bin_idx]:
                rows.append({
                    "Progress":     progress_center,
                    "Speed (m/s)":  float(np.mean(binned_speeds[bin_idx])),
                    "Type":         label,
                })
    return rows


def plot_speed_profiles_by_width(width_mm, r_segment_data, fitts_conditions_for_width,
                                  tunnel_path, output_path, bin_size=0.1):
    """
    One figure per tunnel width: one subplot per R value tested for that
    width, all sharing the same y-axis scale so the deceleration depth near
    the goal is visually comparable across R/W conditions.

    Args:
        width_mm: tunnel width in mm, used only for the figure title.
        r_segment_data: dict {r_mult: segment_data} where segment_data is
            the same {"sim": {...}, "human": {...}} structure built in
            _process_condition for that (tid, r_mult).
        fitts_conditions_for_width: dict {r_mult: cond} for axis titles/labels.
        tunnel_path: centerline for this width (same across all R values).
        output_path: full path to save the PNG to.
        bin_size: progress bin size (default 0.1, matches the other
            speeds_progress_enhanced plots).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    r_mults = sorted(r_segment_data.keys())
    if not r_mults:
        return

    n_r = len(r_mults)
    fig, axes = plt.subplots(1, n_r, figsize=(5 * n_r, 4), squeeze=False)
    axes = axes[0]

    palette = {"Human": "tab:blue", "Simulator": "tab:orange"}

    # First pass: build each subplot's dataframe and track the global y-max
    # so every subplot can share the same y-axis scale.
    per_axis_df = {}
    global_y_max = 0.0
    for r_mult in r_mults:
        segment_data = r_segment_data[r_mult]
        rows = []
        if "human" in segment_data:
            rows.extend(_binned_speed_rows(
                segment_data["human"]["trajectories"],
                segment_data["human"]["speeds"],
                tunnel_path, "Human", bin_size=bin_size,
            ))
        if "sim" in segment_data:
            rows.extend(_binned_speed_rows(
                segment_data["sim"]["trajectories"],
                segment_data["sim"]["speeds"],
                tunnel_path, "Simulator", bin_size=bin_size,
            ))
        df = pd.DataFrame(rows) if rows else None
        per_axis_df[r_mult] = df
        if df is not None and not df.empty:
            global_y_max = max(global_y_max, float(df["Speed (m/s)"].max()))

    y_top = global_y_max * 1.05 if global_y_max > 0 else 1.0

    for ax, r_mult in zip(axes, r_mults):
        df = per_axis_df[r_mult]
        if df is not None and not df.empty:
            sns.lineplot(data=df, x="Progress", y="Speed (m/s)", hue="Type",
                         ax=ax, errorbar="sd", alpha=0.8, legend=False,
                         palette=palette)

        cond  = fitts_conditions_for_width.get(r_mult, {})
        label = cond.get("label", f"R={r_mult}W")
        ax.set_title(label, fontsize=11)
        ax.set_xticks([0.1, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["10%", "25%", "50%", "75%", "100%"])
        ax.set_xlim(0.1, 1.0)
        ax.set_ylim(0, y_top)
        ax.set_xlabel("Progress", fontsize=10)
        ax.set_ylabel("Speed (m/s)", fontsize=10)
        ax.grid(False)

    fig.suptitle(f"Speed vs. Progress by R — W={width_mm:.0f}mm", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Regression helpers
# ---------------------------------------------------------------------------

def _compute_fitts_regression(rows):
    sub = [r for r in rows if r["MT_s"] is not None]
    if len(sub) < 3:
        return {}

    ids = np.array([r["ID"]  for r in sub])
    mts = np.array([r["MT_s"] for r in sub])
    tps = np.array([r["TP"]   for r in sub if r["TP"] is not None])

    b, a   = np.polyfit(ids, mts, 1)
    pred   = a + b * ids
    ss_res = np.sum((mts - pred) ** 2)
    ss_tot = np.sum((mts - np.mean(mts)) ** 2)
    r_sq   = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    width_rhos = {}
    by_width   = defaultdict(list)
    for r in sub:
        by_width[r["width_mm"]].append(r)
    for w, wr in by_width.items():
        if len(wr) >= 3:
            rho, _ = spearmanr([x["R_over_W"] for x in wr],
                               [x["MT_s"]     for x in wr])
            width_rhos[str(int(w))] = round(float(rho), 4)

    return {
        "a_intercept":                   round(float(a),   6),
        "b_slope_s_per_bit":             round(float(b),   6),
        "r_squared":                     round(float(r_sq), 4),
        "throughput_mean_bps":           round(float(np.mean(tps)), 4) if len(tps) else None,
        "throughput_std_bps":            round(float(np.std(tps)),  4) if len(tps) else None,
        "n_runs":                        len(sub),
        "monotonicity_rho_per_width_mm": width_rhos,
    }


def _per_width_regression(rows):
    """Per-width MT = a + b*ID regression.
    """
    sub      = [r for r in rows if r["MT_s"] is not None]
    by_width = defaultdict(list)
    for r in sub:
        by_width[r["width_mm"]].append(r)

    out = {}
    for w, wr in sorted(by_width.items()):
        if len(wr) < 3:
            continue
        ids = np.array([r["ID"]  for r in wr])
        mts = np.array([r["MT_s"] for r in wr])
        tps = np.array([r["TP"]   for r in wr if r["TP"] is not None])
        b, a   = np.polyfit(ids, mts, 1)
        pred   = a + b * ids
        ss_res = np.sum((mts - pred) ** 2)
        ss_tot = np.sum((mts - np.mean(mts)) ** 2)
        r_sq   = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        out[str(int(w))] = {
            "a_intercept":         round(float(a),   6),
            "b_slope_s_per_bit":   round(float(b),   6),
            "r_squared":           round(float(r_sq), 4),
            "throughput_mean_bps": round(float(np.mean(tps)), 4) if len(tps) else None,
            "n_runs":              len(wr),
        }
    return out


# ---------------------------------------------------------------------------
# Aggregate CSV + JSON outputs
# ---------------------------------------------------------------------------

def write_aggregate_outputs(all_rows, aggregate_metrics, fitts_conditions,
                            centerline_lengths):
    """
    Write all aggregate outputs:
      - fitts_results.csv          (per-run flat file)
      - fitts_condition_summary.csv (per participant × condition, all metrics)
      - progress_metrics.csv        (per participant × condition, metrics only)
      - fitts_regression.json
    """

    # ------------------------------------------------------------------
    # 1. fitts_results.csv  (per-run, mirrors main_results.csv structure)
    # ------------------------------------------------------------------
    csv_path = RESULTS_DIR / "fitts_results.csv"
    fields = [
        "source", "participant", "tid", "width_mm", "R_mm", "R_over_W",
        "target_radius_m",
        "D_m", "ID", "MT_s",
        "approach_speed_ms",
        "TP",
        "avg_speed_ms",
        "path_length_m",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in all_rows:
            writer.writerow({
                k: round(v, 6) if isinstance(v, float) else (v if v is not None else "")
                for k, v in r.items()
            })
    print(f"  Saved: {csv_path}")

    # ------------------------------------------------------------------
    # 2. fitts_condition_summary.csv
    #    One row per (participant, tid, r_mult) with all experiment-main
    #    metrics plus Fitts-specific metrics.
    #    Mirrors trial_summary.csv + progress_metrics.csv combined.
    # ------------------------------------------------------------------
    cond_summary_path = RESULTS_DIR / "fitts_condition_summary.csv"
    cond_fields = [
        "participant", "tid", "label", "width_mm", "R_mm", "R_over_W",
        "target_radius_m", "D_m", "ID", "has_human_data",
        # Fitts metrics
        "MT_mean_s", "MT_std_s",
        "throughput_mean_bps", "throughput_std_bps",
        "approach_speed_mean_ms", "approach_speed_std_ms",
        # Experiment-main metrics (preserved in full per instructor)
        "lateral_rmse_mean",
        "speed_rmse_mean",
        "speed_corr_mean",
        "time_diff_mean",
        "goal_approach_speed_human",
        "goal_approach_speed_model",
        "goal_approach_speed_diff",
        # Counts
        "n_valid_runs", "n_human_rounds",
    ]
    with open(cond_summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cond_fields)
        writer.writeheader()
        for (pid, tid, r_mult), m in sorted(aggregate_metrics.items()):
            cond = fitts_conditions.get((tid, r_mult), {})
            W    = cond.get("width", 0)
            R    = cond.get("target_radius", 0)
            D    = centerline_lengths.get(tid, 0.0)
            ID   = fitts_id(D, R)
            row  = {
                "participant":            pid,
                "tid":                    tid,
                "label":                  cond.get("label", ""),
                "width_mm":               round(W * 1000, 1),
                "R_mm":                   round(R * 1000, 3),
                "R_over_W":               round(r_mult, 4),
                "target_radius_m":        round(R, 6),
                "D_m":                    round(D, 6),
                "ID":                     round(ID, 4),
                "has_human_data":         m.get("has_human_data", False),
                "MT_mean_s":              m.get("MT_mean_s"),
                "MT_std_s":               m.get("MT_std_s"),
                "throughput_mean_bps":    m.get("throughput_mean_bps"),
                "throughput_std_bps":     m.get("throughput_std_bps"),
                "approach_speed_mean_ms": m.get("approach_speed_mean_ms"),
                "approach_speed_std_ms":  m.get("approach_speed_std_ms"),
                "lateral_rmse_mean":      m.get("lateral_rmse_mean"),
                "speed_rmse_mean":        m.get("speed_rmse_mean"),
                "speed_corr_mean":        m.get("speed_corr_mean"),
                "time_diff_mean":         m.get("time_diff_mean"),
                "goal_approach_speed_human":  m.get("goal_approach_speed_human"),
                "goal_approach_speed_model":  m.get("goal_approach_speed_model"),
                "goal_approach_speed_diff":   m.get("goal_approach_speed_diff"),
                "n_valid_runs":           m.get("n_valid_runs"),
                "n_human_rounds":         m.get("n_human_rounds"),
            }
            writer.writerow({
                k: round(v, 6) if isinstance(v, float) else (v if v is not None else "")
                for k, v in row.items()
            })
    print(f"  Saved: {cond_summary_path}")

    # ------------------------------------------------------------------
    # 3. progress_metrics.csv  (metrics only, keyed by participant × condition)
    #    Mirrors experiment-main's progress_metrics.csv exactly.
    # ------------------------------------------------------------------
    pm_path = RESULTS_DIR / "progress_metrics.csv"
    pm_fields = [
        "participant", "tid", "label", "R_over_W", "width_mm", "ID",
        "has_human_data",
        "lat_rmse", "speed_rmse", "speed_corr", "time_diff",
        "goal_approach_speed_human", "goal_approach_speed_model",
        "goal_approach_speed_diff",
        "throughput_mean_bps", "approach_speed_mean_ms",
    ]
    with open(pm_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pm_fields)
        writer.writeheader()
        for (pid, tid, r_mult), m in sorted(aggregate_metrics.items()):
            cond = fitts_conditions.get((tid, r_mult), {})
            W    = cond.get("width", 0)
            R    = cond.get("target_radius", 0)
            D    = centerline_lengths.get(tid, 0.0)
            ID   = fitts_id(D, R)
            writer.writerow({
                "participant":               pid,
                "tid":                       tid,
                "label":                     cond.get("label", ""),
                "R_over_W":                  round(r_mult, 4),
                "width_mm":                  round(W * 1000, 1),
                "ID":                        round(ID, 4),
                "has_human_data":            m.get("has_human_data", False),
                "lat_rmse":                  m.get("lateral_rmse_mean") or "",
                "speed_rmse":                m.get("speed_rmse_mean")   or "",
                "speed_corr":                m.get("speed_corr_mean")   or "",
                "time_diff":                 m.get("time_diff_mean")    or "",
                "goal_approach_speed_human": m.get("goal_approach_speed_human") or "",
                "goal_approach_speed_model": m.get("goal_approach_speed_model") or "",
                "goal_approach_speed_diff":  m.get("goal_approach_speed_diff")  or "",
                "throughput_mean_bps":       m.get("throughput_mean_bps")    or "",
                "approach_speed_mean_ms":    m.get("approach_speed_mean_ms") or "",
            })
    print(f"  Saved: {pm_path}")

    # ------------------------------------------------------------------
    # 4. fitts_regression.json
    # ------------------------------------------------------------------
    model_rows = [r for r in all_rows if r["source"] != "human"]
    human_rows = [r for r in all_rows if r["source"] == "human"]

    overall_reg   = _compute_fitts_regression(model_rows)
    per_width_reg = _per_width_regression(model_rows)

    m_tps = [r["TP"] for r in model_rows if r["TP"] is not None]
    tp_cv = (float(np.std(m_tps)) / float(np.mean(m_tps))
             if m_tps and np.mean(m_tps) > 0 else None)

    human_ref_out = []
    seen = set()
    for r in human_rows:
        key = (r["tid"], r["width_mm"])
        if key in seen:
            continue
        seen.add(key)
        mt_vals = [x["MT_s"] for x in human_rows
                   if x["tid"] == r["tid"] and x["MT_s"] is not None]
        human_ref_out.append({
            "tid":       r["tid"],
            "width_mm":  r["width_mm"],
            "R_mult":    HUMAN_R_MULT,
            "R_mm":      r["R_mm"],
            "ID":        r["ID"],
            "MT_s_mean": round(float(np.mean(mt_vals)), 4) if mt_vals else None,
            "n_rounds":  len(mt_vals),
        })

    regression_out = {
        "notes": {
            "ID_formula":         "log2(D/(2R) + 1)  [Shannon; 2R = target width]",
            "MT_model":           "MT = a + b * ID",
            "SPEED_TRIM_START":   SPEED_TRIM_START,
            "SPEED_TRIM_END":     SPEED_TRIM_END,
            "termination_note":   (
                "Model uses dwell-based termination (1s inside R); "
                "human data used hard-stop. MTs are not directly comparable "
                "but speed_rmse and speed_corr still reflect profile shape."
            ),
            "human_reference":    f"R = W/2 per width (r_mult={HUMAN_R_MULT})",
            "non_human_R_note":   (
                "speed_rmse/speed_corr at R != W/2 compare model profile "
                "against human R=W/2 profiles — useful for seeing how "
                "deceleration shape changes with R."
            ),
            "success_criteria": {
                "r_squared":           "> 0.7 overall",
                "b_slope_s_per_bit":   "positive; human steering ~0.1-0.3 s/bit",
                "throughput_cv":       "< 0.30 for Fitts constancy",
                "monotonicity_rho":    "< -0.5 per width (larger R -> faster)",
                "approach_speed_drop": "goal speed at R=0.1W ~3-5x lower than R=2.0W",
                "speed_corr":          "> 0.7 at R=W/2 for good profile match",
            },
            "tuning_guide": {
                "no_MT_change":         "w_precision too small or target_radius not reaching MPCC",
                "slope_too_shallow":    "increase w_precision (effect ~linear)",
                "timeout_at_small_R":   "w_precision too large",
                "MT_up_but_speed_flat": "check dwell termination inflating MT",
            },
        },
        "overall":         overall_reg,
        "throughput_cv":   round(tp_cv, 4) if tp_cv is not None else None,
        "per_width_mm":    per_width_reg,
        "human_reference": sorted(human_ref_out, key=lambda x: x["width_mm"]),
    }

    reg_path = RESULTS_DIR / "fitts_regression.json"
    with open(reg_path, "w") as f:
        json.dump(regression_out, f, indent=2)
    print(f"  Saved: {reg_path}")

    return regression_out


# ---------------------------------------------------------------------------
# Console summary table
# ---------------------------------------------------------------------------

def print_summary_table(aggregate_metrics, fitts_conditions, centerline_lengths):
    print("\n  Summary by condition (averaged across participants):")
    print("  ┌──────────────────────────────┬───────┬───────┬──────────┬──────────┬──────────┬──────────┐")
    print("  │ Condition                    │  ID   │  MT   │ App.Spd  │    TP    │ Spd.Corr │  n runs  │")
    print("  ├──────────────────────────────┼───────┼───────┼──────────┼──────────┼──────────┼──────────┤")

    by_cond = defaultdict(list)
    for (pid, tid, r_mult), m in aggregate_metrics.items():
        by_cond[(tid, r_mult)].append(m)

    for (tid, r_mult), cond in sorted(fitts_conditions.items()):
        metrics_list = by_cond[(tid, r_mult)]
        label        = cond["label"]
        if not metrics_list:
            print(f"  │ {label:28s} │     – │     – │        – │        – │        – │        – │")
            continue

        W  = cond["width"]
        R  = cond["target_radius"]
        D  = centerline_lengths.get(tid, 0.0)
        ID = fitts_id(D, R)

        mts  = [m["MT_mean_s"]            for m in metrics_list if m.get("MT_mean_s")]
        apps = [m["approach_speed_mean_ms"] for m in metrics_list if m.get("approach_speed_mean_ms")]
        tps  = [m["throughput_mean_bps"]   for m in metrics_list if m.get("throughput_mean_bps")]
        cors = [m["speed_corr_mean"]       for m in metrics_list if m.get("speed_corr_mean")]

        mt_str   = f"{np.mean(mts):5.2f}"   if mts  else "    –"
        app_str  = f"{np.mean(apps):8.4f}"  if apps else "       –"
        tp_str   = f"{np.mean(tps):8.4f}"   if tps  else "       –"
        cor_str  = f"{np.mean(cors):8.3f}"  if cors else "       –"
        n_str    = str(sum(m.get("n_valid_runs", 0) for m in metrics_list))

        print(f"  │ {label:28s} │ {ID:5.2f} │ {mt_str} │ {app_str} │ {tp_str} │ {cor_str} │ {n_str:>8} │")

    print("  └──────────────────────────────┴───────┴───────┴──────────┴──────────┴──────────┴──────────┘")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fitts' Law Evaluation: sweep target radius across tunnel widths"
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--trials", type=int, nargs="+", default=None)
    parser.add_argument("--r-mults", type=float, nargs="+", default=DEFAULT_R_MULTIPLIERS)
    parser.add_argument("--speed-model", type=str, default=None)
    parser.add_argument("--aggregate-only", action="store_true", default=False)
    parser.add_argument("--per-participant", action="store_true", default=False,
                        help="Use per-participant fitted configs from "
                             "eval/model_fitting/results/{pid}_gam_config_s{seed}.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pid", type=str, default=None,
                        help="Process only this participant ID")
    args = parser.parse_args()

    config_path = Path(args.config)
    trial_ids   = args.trials if args.trials else list(BASE_CONDITIONS.keys())
    r_mults     = args.r_mults

    fitts_conditions = build_fitts_conditions(trial_ids, r_mults)

    print("=" * 70)
    print("Fitts' Law Evaluation")
    print("=" * 70)
    if args.per_participant:
        print(f"  Mode:         per-participant (seed={args.seed})")
    else:
        print(f"  Config:       {config_path}")
    print(f"  Trials:       {trial_ids}")
    print(f"  R multipliers:{r_mults}")
    print(f"  Rounds/cond:  {args.rounds}")
    print(f"  Conditions:   {len(fitts_conditions)}")
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SIM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ---- centerlines (geometry fixed across R values) ----
    print("[1/4] Generating centerlines ...")
    centerlines        = {}
    centerline_lengths = {}
    for tid in trial_ids:
        cl = generate_centerline(tid)
        centerlines[tid]        = cl
        centerline_lengths[tid] = centerline_arc_length(cl)
        W      = BASE_CONDITIONS[tid]["width"]
        D      = centerline_lengths[tid]
        ID_ref = fitts_id(D, W * HUMAN_R_MULT)   # ID at R=W/2 as reference
        print(f"  Trial {tid}: D={D:.4f}m  W={W*1000:.0f}mm  "
              f"ID at R=W/2: {ID_ref:.3f} bits")

    # ---- human data ----
    print("\n[2/4] Loading human data ...")
    all_human_data = load_trials_by_participant(trial_ids)

    if args.pid:
        if args.pid not in all_human_data:
            print(f"  Participant {args.pid} not found. Available: {list(all_human_data.keys())}")
            return
        all_human_data = {args.pid: all_human_data[args.pid]}

    # Per-participant mode: filter to participants with fitted models
    available_pids = {}
    if args.per_participant:
        for pid in list(all_human_data.keys()):
            cfg_file = FITTING_RESULTS_DIR / f"{pid}_gam_config_s{args.seed}.json"
            pkl_file = FITTING_RESULTS_DIR / f"{pid}_gam_s{args.seed}.pkl"
            if cfg_file.exists() and pkl_file.exists():
                available_pids[pid] = (cfg_file, pkl_file)
            else:
                print(f"  Warning: skipping {pid} — no fitted model (seed={args.seed})")
        all_human_data = {pid: all_human_data[pid] for pid in available_pids}
        if not all_human_data:
            print("  No participants have fitted models. Run fit_speed_model.py first.")
            return
        print(f"  {len(all_human_data)} participants with fitted models")
    else:
        if not config_path.exists():
            print(f"  Config not found: {config_path}")
            return
        print(f"  {len(all_human_data)} participants loaded (shared config)")

    all_rows          = []   # for fitts_results.csv
    aggregate_metrics = {}   # {(pid, tid, r_mult): metrics_dict}

    if not args.aggregate_only:
        print(f"\n[3/4] Running simulations ...")
        n_workers = os.cpu_count() or 4

        for pid, participant_data in sorted(all_human_data.items()):
            print(f"\n  Participant: {pid}")

            # Resolve config and speed model
            if args.per_participant:
                cfg_file, pkl_file = available_pids[pid]
                p_config_path      = str(cfg_file)
                p_speed_model_path = str(pkl_file)
            else:
                p_config_path      = str(config_path)
                p_speed_model_path = args.speed_model

            # Per-participant cache (keyed by condition string to avoid R collisions)
            cache_path = SIM_CACHE_DIR / f"{pid}_fitts_s{args.seed}.json"
            sim_cache  = load_sim_cache(cache_path) or {}

            p_folder = RESULTS_DIR / f"participant_{pid}"
            p_folder.mkdir(exist_ok=True)

            futures = {}
            condition_segment_data = {}   # {(tid, r_mult): segment_data}
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                for (tid, r_mult), cond in sorted(fitts_conditions.items()):
                    cache_key    = condition_cache_key(tid, r_mult)
                    existing     = sim_cache.get(cache_key, [])
                    trial_folder = p_folder / condition_folder_name(tid, r_mult)

                    # Human data for this tid (only exists at R=W/2 in raw data,
                    # but we pass it for all R values so metrics are always computed)
                    trial_data_dict = participant_data.get(tid, {})

                    fut = executor.submit(
                        _process_condition,
                        p_config_path, p_speed_model_path,
                        pid, tid, r_mult, cond,
                        trial_data_dict,
                        args.rounds, existing,
                        str(trial_folder), centerlines[tid],
                    )
                    futures[fut] = (tid, r_mult)

                try:
                    for fut in concurrent.futures.as_completed(futures, timeout=3600):
                        tid, r_mult = futures[fut]
                        try:
                            result    = fut.result(timeout=120)
                            cache_key = condition_cache_key(tid, r_mult)
                            sim_cache[cache_key] = result["all_sim_records"]

                            m = result["metrics"]
                            # Store n counts for summary CSV
                            m["n_valid_runs"]  = result["n_model"]
                            m["n_human_rounds"] = result["n_human"]
                            aggregate_metrics[(pid, tid, r_mult)] = m
                            condition_segment_data[(tid, r_mult)] = result["segment_data"]

                            cor_str = (f"spd_corr={m.get('speed_corr_mean','?'):.3f}"
                                       if m.get("speed_corr_mean") is not None else "spd_corr=–")
                            print(f"    tid={tid} R={r_mult:.2f}W "
                                  f"n={result['n_model']} "
                                  f"MT={m.get('MT_mean_s','?')} "
                                  f"{cor_str} "
                                  f"TP={m.get('throughput_mean_bps','?')}")
                        except Exception as e:
                            print(f"    ERROR tid={tid} R={r_mult}: {e}")
                except concurrent.futures.TimeoutError:
                    print("  WARNING: pool timeout")
                    for f in futures:
                        f.cancel()

            save_sim_cache(sim_cache, cache_path)

            # ---- per-width speed-profile summary plots (one per tid/width) ----
            # Lays every R/W condition's speed-vs-progress profile side by
            # side, all on the same y-axis scale, so deceleration depth near
            # the goal is visually comparable across R values for this width.
            for tid in trial_ids:
                width_mm = BASE_CONDITIONS[tid]["width"] * 1000
                r_segment_data = {
                    r_mult: condition_segment_data[(tid, r_mult)]
                    for r_mult in r_mults
                    if (tid, r_mult) in condition_segment_data
                }
                if not r_segment_data:
                    continue
                fitts_conditions_for_width = {
                    r_mult: fitts_conditions[(tid, r_mult)]
                    for r_mult in r_mults
                    if (tid, r_mult) in fitts_conditions
                }
                out_path = p_folder / f"speed_profiles_W{width_mm:.0f}mm.png"
                try:
                    plot_speed_profiles_by_width(
                        width_mm, r_segment_data, fitts_conditions_for_width,
                        centerlines[tid], out_path, bin_size=0.1,
                    )
                except Exception as e:
                    print(f"    ERROR plotting speed_profiles_W{width_mm:.0f}mm.png: {e}")


    # ---- collect rows from saved summaries (supports --aggregate-only) ----
    print("\n[4/4] Generating aggregate outputs ...")

    for pid in sorted(all_human_data.keys()):
        p_folder = RESULTS_DIR / f"participant_{pid}"
        for (tid, r_mult), cond in sorted(fitts_conditions.items()):
            W = cond["width"]
            R = cond["target_radius"]
            D = centerline_lengths.get(tid, 0.0)
            ID = fitts_id(D, R)

            summary_path = p_folder / condition_folder_name(tid, r_mult) / "results_summary.json"
            if not summary_path.exists():
                continue

            with open(summary_path) as f:
                sm = json.load(f)

            # Load metrics into aggregate_metrics if not already there
            # (covers --aggregate-only mode)
            key = (pid, tid, r_mult)
            if key not in aggregate_metrics:
                m = sm.get("metrics", {})
                m["n_valid_runs"]   = sm["model"].get("n_valid_runs", sm["model"]["n_runs"])
                m["n_human_rounds"] = sm["human"].get("n_rounds", 0)
                aggregate_metrics[key] = m

            # Per-run rows for fitts_results.csv
            for run in sm["model"]["runs"]:
                MT  = run["completion_time"]
                app = run.get("approach_speed")
                TP  = ID / MT if MT and MT > 0 else None
                label = f"model_{pid}" if args.per_participant else "model"
                all_rows.append({
                    "source":          label,
                    "participant":     pid,
                    "tid":             tid,
                    "width_mm":        W * 1000,
                    "R_mm":            R * 1000,
                    "R_over_W":        r_mult,
                    "target_radius_m": run.get("target_radius", R),
                    "D_m":             round(D, 6),
                    "ID":              round(ID, 4),
                    "MT_s":            MT,
                    "approach_speed_ms": app,
                    "TP":              TP,
                    "avg_speed_ms":    run.get("avg_speed"),
                    "path_length_m":   run.get("path_length"),
                })

        # Human reference rows (R = W/2 only, one row per human round)
        for tid in trial_ids:
            W  = BASE_CONDITIONS[tid]["width"]
            R  = W * HUMAN_R_MULT
            D  = centerline_lengths.get(tid, 0.0)
            ID = fitts_id(D, R)
            tid_data = all_human_data.get(pid, {}).get(tid, {})
            for round_num, rd in tid_data.items():
                traj = rd["trajectory"]
                ts   = rd["timestamps"]
                if len(ts) < 2:
                    continue
                MT = (ts[-1] - ts[0]) / 1000.0
                TP = ID / MT if MT > 0 else None
                pl = _path_length(traj)
                all_rows.append({
                    "source":          "human",
                    "participant":     pid,
                    "tid":             tid,
                    "width_mm":        W * 1000,
                    "R_mm":            R * 1000,
                    "R_over_W":        HUMAN_R_MULT,
                    "target_radius_m": R,
                    "D_m":             round(D, 6),
                    "ID":              round(ID, 4),
                    "MT_s":            MT,
                    "approach_speed_ms": None,
                    "TP":              TP,
                    "avg_speed_ms":    pl / MT if MT > 0 else None,
                    "path_length_m":   pl,
                })

    reg = write_aggregate_outputs(
        all_rows, aggregate_metrics, fitts_conditions, centerline_lengths
    )
    print_summary_table(aggregate_metrics, fitts_conditions, centerline_lengths)

    # Quick pass/fail
    overall = reg.get("overall", {})
    print("\n  === Quick verification ===")
    r_sq = overall.get("r_squared")
    b    = overall.get("b_slope_s_per_bit")
    tp_m = overall.get("throughput_mean_bps")
    tp_cv = reg.get("throughput_cv")
    if r_sq is not None:
        print(f"  R²      : {r_sq:.3f}  {'✓' if r_sq > 0.7 else '✗'} (target > 0.7)")
    if b is not None:
        print(f"  b slope : {b:.4f} s/bit  {'✓ positive' if b > 0 else '✗ non-positive'}")
    if tp_m is not None:
        print(f"  Mean TP : {tp_m:.3f} bits/s")
    if tp_cv is not None:
        print(f"  TP CV   : {tp_cv:.3f}  {'✓' if tp_cv < 0.30 else '✗'} (target < 0.30)")

    mono = overall.get("monotonicity_rho_per_width_mm", {})
    if mono:
        print("  Monotonicity rho (R_over_W vs MT, should be negative):")
        for w, rho in sorted(mono.items()):
            print(f"    W={w}mm: rho={rho:.3f}  {'✓' if rho < -0.5 else '✗'}")

    print("\n" + "=" * 70)
    print("Fitts evaluation complete.")
    print(f"Results: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
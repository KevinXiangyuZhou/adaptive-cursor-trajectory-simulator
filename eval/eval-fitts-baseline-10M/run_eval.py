"""
Fitts' Law Evaluation Experiment — Simulator Only (Baseline vs. MPCC).

⚠️ RULED OUT for pointing tasks (2026-08): the "baseline" (--model-type
baseline, hcs_package.baseline_model.generate_baseline_mpc) side of this
comparison shows jittery start/end behavior and a severe distance-driven
completion-time blowup vs. human data. Do not use the baseline model as the
default pointing-task model in new code — MPCC-in-a-bypass-tunnel replaced
it (see eval/eval-new-data/run_eval.py's module docstring for the full
writeup). This script itself is a pre-existing historical artifact, kept
as-is for reference.

Systematically sweeps an ABSOLUTE target radius R (in meters — NOT a
multiplier of tunnel width) inside a single, identical 10-meter-wide bypass
tunnel, to compare the constraint-free BUMP-style Pointing Baseline model
directly against the MPCC steering model under a matched independent
variable (target_radius). Only the model type (--model-type) and the
target radius change between conditions:

    MT = a + b * ID,  ID = log2(D/(2R) + 1)

There is no human data anywhere in this script: no human trajectories are
loaded, no human traces are plotted or overlaid, and no human-comparison
metrics (speed_rmse, speed_corr, lateral_rmse, time_diff) are computed.
Those columns are still present in the CSV outputs (structure is preserved
exactly, per spec) but are always blank/None now that there is nothing to
compare against.

Fitts' regression (MT = a + b*ID, R² extraction) and throughput are computed
and plotted solely for the simulator — run this script once with
--model-type mpcc and once with --model-type baseline (each writes into its
own participant_*/cache/plot namespace) to compare the two model types.

SPEED_TRIM_START / SPEED_TRIM_END follow experiment-main exactly:
    - skip first 10% of progress (startup acceleration)
    - evaluate up to 100% (full path, including goal approach)
    - SPEED_TRIM_END=100 also drives the x-axis start of progress plots
      via plot_speeds_vs_progress_enhanced's xlim(0.1, 1.0)

Usage:
    python run_eval.py --model-type mpcc
    python run_eval.py --model-type baseline
    python run_eval.py --model-type baseline --radii 0.0025 0.005 0.0075 0.010

Directory structure:
    results/
        participant_{base_pid}_{model_type}/
            trial_{tid}_R_abs_{radius:.4f}/
                trajectories_t{tid}.png
                speeds_enhanced_t{tid}.png
                speeds_progress_enhanced_t{tid}.png
                results_summary.json
            speed_profiles_W{width_mm}mm_{model_type}.png
        fitts_results.csv           <- one row per run
        fitts_condition_summary.csv <- one row per (participant, tid, radius)
        progress_metrics.csv        <- one row per (participant, tid, radius)
        fitts_regression.json       <- regression table
        fitts_regression_plot_{model_type}.png
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
)
# NOTE: trajectory_rmse / speed_profile_rmse / speed_profile_correlation are
# intentionally NOT imported anymore — those were exclusively used to
# compare model output against human data, which this script no longer
# loads or references at all (see module docstring).

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_DIR         = SCRIPT_DIR / "results"
SIM_CACHE_DIR       = RESULTS_DIR / "sim_cache"
DEFAULT_CONFIG      = PROJECT_ROOT / "experiment" / "user_configurations" / "customized.json"
FITTING_RESULTS_DIR = PROJECT_ROOT / "eval" / "model_fitting" / "results"

WINDOW_WIDTH         = 0.46
WINDOW_HEIGHT        = 0.26

# Speed metric trim — mirrors experiment-main exactly.
# SPEED_TRIM_START skips startup acceleration (first 10% of progress).
# SPEED_TRIM_END=100 evaluates the full path including goal approach,
# which is critical for detecting Fitts'-like deceleration.
# These also drive the x-axis of plot_speeds_vs_progress_enhanced
# (xlim 0.1 -> 1.0 corresponds to bins 10-100).
SPEED_TRIM_START = 10
SPEED_TRIM_END   = 100
N_PROGRESS_BINS  = 100

# Both the Baseline and MPCC models run inside this identical, effectively
# unconstrained corridor (10m >> the ~0.46m screen and >> any swept target
# radius), so tunnel-boundary effects never engage for either model — the
# ONLY thing that varies between conditions is target_radius / model_type.
BYPASS_TUNNEL_WIDTH_M = 10.0

# ---------------------------------------------------------------------------
# Condition definitions
# ---------------------------------------------------------------------------
# Only a single trial ID (11) is kept — intentionally, per spec — so that
# directory names, cache-file names, and any external database keyed on
# these trial IDs stay valid. Its "width" is declared directly as the
# bypass-tunnel width (meters) so BASE_CONDITIONS and _build_sigmoidal_config
# never disagree about what tunnel geometry is actually being simulated.
BASE_CONDITIONS = {
    11: {"type": "sigmoidal", "width": BYPASS_TUNNEL_WIDTH_M, "curvature": 0.0,
         "label": "Bypass Tunnel W=10m (constraint-free)"},
}

# Absolute target radii to sweep, in meters (2.5mm / 5.0mm / 7.5mm / 10.0mm).
# This REPLACES the old width-relative multiplier sweep
# (DEFAULT_R_MULTIPLIERS = [0.25, 0.375, 0.5, 0.75, 1.0]) — R is now the
# independent variable directly, not a fraction of tunnel width.
DEFAULT_ABSOLUTE_RADII = [0.0025, 0.005, 0.0075, 0.010, 0.015, 0.020, 0.030, 0.050]


def build_fitts_conditions(trial_ids, radii):
    """Return dict keyed by (tid, radius) with full condition info.

    `radii` are ABSOLUTE target radii in meters. `R_over_W` is still
    computed and stored (CSV schema compatibility — see write_aggregate_outputs)
    but it is now a DERIVED quantity (radius / width), not the sweep variable.

    Type/key-safety note: `radii` values are cast to `float` so that
    dictionary keys built from them (and later float-formatted into folder
    names / cache keys via condition_folder_name / condition_cache_key)
    behave consistently whether they arrived from CLI strings (argparse
    already applies type=float) or were passed in-process as Python floats.
    """
    conditions = {}
    for tid in trial_ids:
        base  = BASE_CONDITIONS[tid]
        width = base["width"]
        for radius in radii:
            radius = float(radius)
            conditions[(tid, radius)] = {
                **base,
                "target_radius": radius,
                "radius":        radius,
                "r_over_w":      (radius / width) if width else 0.0,
                "label":         f"R={radius * 1000:.2f}mm (W={width:.0f}m bypass)",
            }
    return conditions


def condition_folder_name(tid, radius):
    return f"trial_{tid}_R_abs_{radius:.4f}"


def condition_cache_key(tid, radius):
    return f"{tid}_R_abs_{radius:.4f}"


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

    `width` is intentionally IGNORED and hardcoded to BYPASS_TUNNEL_WIDTH_M
    (10.0 m) unconditionally, per spec — this guarantees the MPCC model and
    the constraint-free Baseline model are always compared inside the exact
    same wide-open corridor, even if a caller ever passes a stale/different
    width in by mistake. target_radius overrides the default width/2 to
    drive the Fitts' sweep.

    Note: target_radius=None is only a defensive fallback (mirroring the
    original function's signature) — every call in this script always
    passes an explicit absolute target_radius from DEFAULT_ABSOLUTE_RADII,
    so `width * 0.5` (which would be 5.0m — nonsensical as a target radius)
    is never actually reached in practice.
    """
    width = BYPASS_TUNNEL_WIDTH_M  # hardcoded unconditionally

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


def _approach_speed(speeds, trim_pct=0.90):
    """Mean speed over the last (1-trim_pct) fraction of the trajectory."""
    if not speeds:
        return 0.0
    start = max(0, int(len(speeds) * trim_pct))
    tail  = speeds[start:]
    return float(np.mean(tail)) if tail else 0.0


# ---------------------------------------------------------------------------
# Simulator configuration discovery (replaces human-data-driven participant
# discovery entirely — see module docstring: no human data is used anywhere)
# ---------------------------------------------------------------------------

def discover_participants(args):
    """
    Resolve the set of simulator configurations to run.

    Returns:
        dict {base_pid: (config_path, speed_model_path_or_None)}

    In --per-participant mode, "participants" are discovered directly from
    the fitted GAM config files under FITTING_RESULTS_DIR (these are fitted
    motor-noise models, not raw human trajectories — using them does not
    reintroduce human trajectory/plot data anywhere in this script).

    In shared-config mode (default), there is exactly one nominal entry,
    "sim", pointing at --config / --speed-model. This preserves the
    existing participant-keyed directory/cache architecture (per spec: "to
    avoid changing the surrounding pipeline architecture") without requiring
    any human data file to exist.

    Safe key extraction note: participant IDs recovered from filenames are
    plain strings (Path.stem based), so downstream f"{base_pid}_{model_type}"
    concatenation and dict lookups keyed by that string are type-consistent
    throughout (no int/str key mismatches).
    """
    participants = {}

    if args.per_participant:
        suffix = f"_gam_config_s{args.seed}"
        for cfg_file in sorted(FITTING_RESULTS_DIR.glob(f"*{suffix}.json")):
            name = cfg_file.stem
            pid  = name[:-len(suffix)] if name.endswith(suffix) else name
            pkl_file = FITTING_RESULTS_DIR / f"{pid}_gam_s{args.seed}.pkl"
            if pkl_file.exists():
                participants[pid] = (cfg_file, pkl_file)
            else:
                print(f"  Warning: skipping {pid} — no matching .pkl for "
                      f"fitted GAM (seed={args.seed})")
        if args.pid:
            participants = {k: v for k, v in participants.items() if k == args.pid}
        if not participants:
            print("  No participants have fitted models. Run fit_speed_model.py first.")
    else:
        if args.pid:
            print("  Note: --pid only applies with --per-participant; ignoring.")
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        speed_model_path = Path(args.speed_model) if args.speed_model else None
        participants["sim"] = (config_path, speed_model_path)

    return participants


# ---------------------------------------------------------------------------
# Simulator runner
# ---------------------------------------------------------------------------

def run_simulator_for_condition(sim, tid, target_radius, n_runs=1, model_type="mpcc"):
    """
    Run simulator for one (tid, target_radius) condition, dispatching to
    either the MPCC steering model or the constraint-free pointing Baseline
    model according to `model_type`.

    API Unification note: generate_trajectory_with_waypoints (MPCC) and
    generate_trajectory_with_start_and_end (Baseline) share the exact same
    unit conventions now — waypoints in screen "pixels" (this codebase's
    pixel units are millimeters: screen_width=460 <-> 0.46m, see
    WINDOW_WIDTH), target_radius natively in meters. No coordinate or
    target_radius scaling is needed before calling either method.

    Deviation from the literal call template worth flagging: the Baseline
    branch below explicitly passes screen_width/screen_height from
    task_config rather than relying on generate_trajectory_with_start_and_end's
    own defaults (1920x1080). This codebase's task_config uses
    screen_width=460/screen_height=260 (millimeter-scale "pixels"), NOT
    1920x1080 — using the method's defaults here would silently misconvert
    every coordinate by roughly 4x. The MPCC branch doesn't need this because
    generate_trajectory_with_waypoints reads screen_width/height from the
    task_file itself, which already carries the correct 460/260 values.
    """
    cond = BASE_CONDITIONS.get(tid, {})
    if not cond:
        return []

    task_config, _ = _build_sigmoidal_config(
        cond["width"], cond["curvature"], target_radius=target_radius
    )

    interval = sim.interval
    records  = []

    # Only the MPCC path needs a task_file on disk; the Baseline path
    # consumes task_config["waypoints"] directly, in-memory.
    task_file = None
    if model_type != "baseline":
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
            json.dump(task_config, tf)
            task_file = tf.name

    try:
        for _ in range(n_runs):
            if model_type == "baseline":
                traj_raw = sim.generate_trajectory_with_start_and_end(
                    waypoints=task_config["waypoints"],
                    screen_width=task_config.get("screen_width", 460.0),
                    screen_height=task_config.get("screen_height", 260.0),
                    max_steps=task_config.get("max_steps", 800),
                    target_radius=target_radius,
                )
            else:
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
        if task_file is not None:
            os.unlink(task_file)

    return records


# ---------------------------------------------------------------------------
# Per-condition metrics (simulator-only — no human comparison anywhere)
# ---------------------------------------------------------------------------

def _build_condition_summary(
    participant_id, tid, radius, cond,
    valid_records, all_records,
    centerline, model_type,
):
    """
    Compute and return results_summary dict for one (pid, tid, radius)
    condition.

    Simulator-only: no human trajectories/speeds/timestamps are loaded,
    compared against, or referenced here in any way. `radius` is the
    absolute target radius (meters) for this condition. `model_type`
    ("mpcc" or "baseline") is stamped into the summary so results from the
    two model types can always be told apart downstream (Section 3).
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
            # mean_lat_m is computed for parity with the lateral-tracking
            # signal used elsewhere, even though nothing consumes it now
            # that human lateral_rmse is gone.
            mean_lat_m = np.mean(all_lat_m, axis=0)  # noqa: F841

            # Goal approach speed from progress bins 90-100 — a purely
            # model-side metric (experiment-main's definition), unaffected
            # by the removal of human data.
            goal_spd_m = (float(np.mean(mean_spd_m[90:100]))
                          if len(mean_spd_m) >= 100 else None)

            agg = {
                # ---- Fitts-specific ----
                "ID":                        round(ID, 4),
                "D_m":                       round(D, 4),
                "target_radius_m":           round(R, 6),
                "R_over_W":                  round(R / W, 6) if W else None,
                "has_human_data":            False,  # always False: no human data exists in this script
                "throughput_mean_bps":       round(float(np.mean(m_tps)), 4) if m_tps else None,
                "throughput_std_bps":        round(float(np.std(m_tps)),  4) if m_tps else None,
                "MT_mean_s":                 round(float(np.mean(m_cts)), 4) if m_cts else None,
                "MT_std_s":                  round(float(np.std(m_cts)),  4) if m_cts else None,
                "approach_speed_mean_ms":    round(float(np.mean(m_app)), 6) if m_app else None,
                "approach_speed_std_ms":     round(float(np.std(m_app)),  6) if m_app else None,
                # ---- Human-comparison fields: kept in the schema for CSV
                #      compatibility (Section 4 mandate) but always None —
                #      there is nothing to compare against anymore.
                "lateral_rmse_mean":         None,
                "speed_rmse_mean":           None,
                "speed_corr_mean":           None,
                "time_diff_mean":            None,
                "goal_approach_speed_human": None,
                "goal_approach_speed_model": round(goal_spd_m, 4) if goal_spd_m is not None else None,
                "goal_approach_speed_diff":  None,
                "human_time_mean_s":         None,
                "model_type":                model_type,
            }

    return {
        "participant_id": participant_id,
        "trial_id":       tid,
        "target_radius":  radius,
        "model_type":     model_type,
        "condition":      {k: v for k, v in cond.items()},
        # Stub kept for schema stability (some external tooling may still
        # read summary["human"]); always zeroed since no human data is used.
        "human": {"n_rounds": 0, "avg_completion_time": None},
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
                    "path_length":     r["path_length"],
                }
                for j, r in enumerate(valid_records)
            ],
        },
        "metrics": agg,
    }


# ---------------------------------------------------------------------------
# Per-condition worker (runs in process pool)
# ---------------------------------------------------------------------------

def _process_condition(
    config_path, speed_model_path,
    participant_id, tid, radius, cond,
    n_rounds, existing_records,
    trial_folder_str, centerline,
    model_type,
):
    """
    Simulate + plot + save summary for one (participant, tid, radius).
    Designed to run inside a ProcessPoolExecutor.

    `model_type` ("mpcc" or "baseline") is threaded through to
    run_simulator_for_condition (which model to call) and into the saved
    summary (so results are always traceable to the model that produced
    them). No human trajectory, speed, or timestamp data is loaded, plotted,
    or referenced anywhere in this function, under any circumstance.
    """
    import matplotlib
    matplotlib.use("Agg")

    trial_folder = Path(trial_folder_str)
    trial_folder.mkdir(parents=True, exist_ok=True)

    # ---- simulation (incremental) ----
    existing = existing_records or []
    delta    = n_rounds - len(existing)
    if delta > 0:
        sim = CursorSimulator(str(config_path))
        if speed_model_path:
            from hcs_package.speed_model import GAMSpeedModel
            sim.speed_model = GAMSpeedModel.load(str(speed_model_path))
        new_recs    = run_simulator_for_condition(
            sim, tid, cond["target_radius"], n_runs=delta, model_type=model_type
        )
        all_records = existing + new_recs
    else:
        all_records = existing[:n_rounds]

    all_records_unfiltered = all_records[:n_rounds]

    # ---- filter diverged / timed-out runs (bounds checks fully preserved) ----
    _MAX_CT = 39.5
    valid_records = []
    for ri, r in enumerate(all_records_unfiltered):
        if r["completion_time"] >= _MAX_CT:
            print(f"  Filtered: {participant_id} tid={tid} R={radius:.4f} run={ri} "
                  f"CT={r['completion_time']:.2f}s", flush=True)
            continue
        traj = r.get("trajectory", [])
        if traj:
            xs = [p[0] for p in traj]
            ys = [p[1] for p in traj]
            if min(xs) < -0.1 or max(xs) > 0.56 or min(ys) < -0.1 or max(ys) > 0.36:
                print(f"  Filtered: {participant_id} tid={tid} R={radius:.4f} run={ri} "
                      f"out of bounds", flush=True)
                continue
        valid_records.append(r)

    model_trajectories = [r["trajectory"] for r in valid_records]
    model_speeds_list  = [r["speeds"]     for r in valid_records]

    # ---- plots: simulator-only, no human overlay under any circumstance ----
    segment_data = {
        "sim": {"trajectories": model_trajectories, "speeds": model_speeds_list},
    }

    all_results    = {tid: {0: segment_data}}
    tunnel_paths   = {tid: centerline}
    tunnel_widths  = {tid: cond["width"]}
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
    summary = _build_condition_summary(
        participant_id, tid, radius, cond,
        valid_records, all_records_unfiltered,
        centerline, model_type,
    )
    summary_path = trial_folder / "results_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    metrics = summary.get("metrics", {})
    return {
        "participant_id":  participant_id,
        "tid":             tid,
        "radius":          radius,
        "all_sim_records": all_records,
        "valid_records":   valid_records,
        "n_model":         len(valid_records),
        "metrics":         metrics,
        # Returned so main() can build the per-participant, per-width
        # "speed_profiles_W{width_mm}mm_{model_type}.png" summary plot after
        # all radius conditions for this (participant, tid) have been
        # processed. Kept minimal (no full all_records) to limit IPC size.
        "segment_data":    segment_data,
    }


# ---------------------------------------------------------------------------
# Per-width speed-profile summary plot (simulator-only)
# ---------------------------------------------------------------------------

def _binned_speed_rows(trajectories, speeds_list, tunnel_path, label,
                        bin_size=0.1):
    """
    Replicates the per-run progress-binning from plot_speeds_vs_progress_enhanced
    (mean speed per progress bin, per run), returned as long-form rows
    ready to feed into a seaborn dataframe: {"Progress", "Speed (m/s)", "Type"}.

    Generic over `trajectories`/`speeds_list`/`label` — this function never
    referenced human data directly even in the original version, so no
    human-specific code needed to be dropped here; it is now only ever
    called with the "Simulator" label.
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
                                  tunnel_path, output_path, model_type, bin_size=0.1):
    """
    One figure per tunnel width: one subplot per absolute target radius
    tested for that width, all sharing the same y-axis scale so the
    deceleration depth near the goal is visually comparable across R
    conditions. Simulator-only — no human series is plotted under any
    circumstance.

    Args:
        width_mm: tunnel width in mm, used only for the figure title.
        r_segment_data: dict {radius: segment_data} where segment_data is
            the {"sim": {...}} structure built in _process_condition for
            that (tid, radius).
        fitts_conditions_for_width: dict {radius: cond} for axis titles/labels.
        tunnel_path: centerline for this width (same across all radii).
        output_path: full path to save the PNG to.
        model_type: "mpcc" or "baseline" — used in the figure title only.
        bin_size: progress bin size (default 0.1).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    radii = sorted(r_segment_data.keys())
    if not radii:
        return

    n_r = len(radii)
    fig, axes = plt.subplots(1, n_r, figsize=(5 * n_r, 4), squeeze=False)
    axes = axes[0]

    palette = {"Simulator": "tab:orange"}

    # First pass: build each subplot's dataframe and track the global y-max
    # so every subplot can share the same y-axis scale.
    per_axis_df = {}
    global_y_max = 0.0
    for radius in radii:
        segment_data = r_segment_data[radius]
        rows = []
        if "sim" in segment_data:
            rows.extend(_binned_speed_rows(
                segment_data["sim"]["trajectories"],
                segment_data["sim"]["speeds"],
                tunnel_path, "Simulator", bin_size=bin_size,
            ))
        df = pd.DataFrame(rows) if rows else None
        per_axis_df[radius] = df
        if df is not None and not df.empty:
            global_y_max = max(global_y_max, float(df["Speed (m/s)"].max()))

    y_top = global_y_max * 1.05 if global_y_max > 0 else 1.0

    for ax, radius in zip(axes, radii):
        df = per_axis_df[radius]
        if df is not None and not df.empty:
            sns.lineplot(data=df, x="Progress", y="Speed (m/s)", hue="Type",
                         ax=ax, errorbar="sd", alpha=0.8, legend=False,
                         palette=palette)

        cond  = fitts_conditions_for_width.get(radius, {})
        label = cond.get("label", f"R={radius * 1000:.2f}mm")
        ax.set_title(label, fontsize=11)
        ax.set_xticks([0.1, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["10%", "25%", "50%", "75%", "100%"])
        ax.set_xlim(0.1, 1.0)
        ax.set_ylim(0, y_top)
        ax.set_xlabel("Progress", fontsize=10)
        ax.set_ylabel("Speed (m/s)", fontsize=10)
        ax.grid(False)

    fig.suptitle(f"Speed vs. Progress by R — W={width_mm:.0f}mm — model={model_type}",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Regression helpers (math UNCHANGED from the original — per spec, Section 4)
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
    """Per-width MT = a + b*ID regression."""
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


def plot_fitts_regression(model_rows, model_reg, model_type, output_path):
    """
    ID (bits) vs MT (s) scatter for every individual simulator run, with the
    fitted Fitts' line (MT = a + b*ID) overlaid.

    Simulator-only: there is no human series to plot anymore. To compare
    Baseline vs. MPCC, run this script once per --model-type (each writes
    its own fitts_regression_plot_{model_type}.png) and compare the two
    output files/regression tables side by side.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    color        = "tab:orange"
    series_label = f"Simulator ({model_type})"

    pts = [(r["ID"], r["MT_s"]) for r in model_rows if r["MT_s"] is not None]
    if pts:
        ids, mts = zip(*pts)
        ax.scatter(ids, mts, s=22, alpha=0.5, color=color,
                   edgecolors="none", label=f"{series_label} (n={len(pts)})")

    if model_reg:
        ids = [r["ID"] for r in model_rows if r["MT_s"] is not None]
        if ids:
            a, b = model_reg["a_intercept"], model_reg["b_slope_s_per_bit"]
            x = np.linspace(min(ids), max(ids), 100)
            y = a + b * x
            r2 = model_reg.get("r_squared")
            sign = "+" if b >= 0 else "-"
            eq = f"MT={a:.2f}{sign}{abs(b):.2f}\u00b7ID"
            r2_str = f", R\u00b2={r2:.2f}" if r2 is not None else ""
            ax.plot(x, y, color=color, linewidth=2,
                    label=f"{series_label} fit: {eq}{r2_str}")

    ax.set_xlabel("ID (bits)", fontsize=11)
    ax.set_ylabel("MT (s)", fontsize=11)
    ax.set_title(f"Fitts' Law: MT vs. ID \u2014 {series_label}", fontsize=12)
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Aggregate CSV + JSON outputs
# ---------------------------------------------------------------------------

def write_aggregate_outputs(all_rows, aggregate_metrics, fitts_conditions,
                            centerline_lengths, model_type):
    """
    Write all aggregate outputs:
      - fitts_results.csv          (per-run flat file)
      - fitts_condition_summary.csv (per participant × condition, all metrics)
      - progress_metrics.csv        (per participant × condition, metrics only)
      - fitts_regression.json

    CSV headers/columns are UNCHANGED from the original human-comparison
    version (Section 4 mandate) — human-only columns (speed_rmse_mean,
    lateral_rmse_mean, has_human_data, n_human_rounds, etc.) are still
    present, but their values are always blank/None/False now, since no
    human data is loaded anywhere in this script.
    """
    # ------------------------------------------------------------------
    # 1. fitts_results.csv — fields UNCHANGED
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
    # 2. fitts_condition_summary.csv — fields UNCHANGED
    # ------------------------------------------------------------------
    cond_summary_path = RESULTS_DIR / "fitts_condition_summary.csv"
    cond_fields = [
        "participant", "tid", "label", "width_mm", "R_mm", "R_over_W",
        "target_radius_m", "D_m", "ID", "has_human_data",
        "MT_mean_s", "MT_std_s",
        "throughput_mean_bps", "throughput_std_bps",
        "approach_speed_mean_ms", "approach_speed_std_ms",
        "lateral_rmse_mean",
        "speed_rmse_mean",
        "speed_corr_mean",
        "time_diff_mean",
        "goal_approach_speed_human",
        "goal_approach_speed_model",
        "goal_approach_speed_diff",
        "n_valid_runs", "n_human_rounds",
    ]
    with open(cond_summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cond_fields)
        writer.writeheader()
        for (pid, tid, radius), m in sorted(aggregate_metrics.items(),
                                             key=lambda kv: (str(kv[0][0]), kv[0][1], kv[0][2])):
            # Safe key extraction: fitts_conditions is keyed by (tid, radius)
            # with radius as a plain float (see build_fitts_conditions), the
            # same type used to build aggregate_metrics's keys below — so
            # this lookup never silently misses due to an int/float/str
            # key-type mismatch.
            cond = fitts_conditions.get((tid, radius), {})
            W    = cond.get("width", 0)
            R    = cond.get("target_radius", radius)
            D    = centerline_lengths.get(tid, 0.0)
            ID   = fitts_id(D, R)
            row  = {
                "participant":            pid,
                "tid":                    tid,
                "label":                  cond.get("label", ""),
                "width_mm":               round(W * 1000, 1),
                "R_mm":                   round(R * 1000, 3),
                "R_over_W":               round(R / W, 6) if W else None,
                "target_radius_m":        round(R, 6),
                "D_m":                    round(D, 6),
                "ID":                     round(ID, 4),
                "has_human_data":         False,
                "MT_mean_s":              m.get("MT_mean_s"),
                "MT_std_s":               m.get("MT_std_s"),
                "throughput_mean_bps":    m.get("throughput_mean_bps"),
                "throughput_std_bps":     m.get("throughput_std_bps"),
                "approach_speed_mean_ms": m.get("approach_speed_mean_ms"),
                "approach_speed_std_ms":  m.get("approach_speed_std_ms"),
                "lateral_rmse_mean":      None,
                "speed_rmse_mean":        None,
                "speed_corr_mean":        None,
                "time_diff_mean":         None,
                "goal_approach_speed_human":  None,
                "goal_approach_speed_model":  m.get("goal_approach_speed_model"),
                "goal_approach_speed_diff":   None,
                "n_valid_runs":           m.get("n_valid_runs"),
                "n_human_rounds":         0,
            }
            writer.writerow({
                k: round(v, 6) if isinstance(v, float) else (v if v is not None else "")
                for k, v in row.items()
            })
    print(f"  Saved: {cond_summary_path}")

    # ------------------------------------------------------------------
    # 3. progress_metrics.csv — fields UNCHANGED
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
        for (pid, tid, radius), m in sorted(aggregate_metrics.items(),
                                             key=lambda kv: (str(kv[0][0]), kv[0][1], kv[0][2])):
            cond = fitts_conditions.get((tid, radius), {})
            W    = cond.get("width", 0)
            R    = cond.get("target_radius", radius)
            D    = centerline_lengths.get(tid, 0.0)
            ID   = fitts_id(D, R)
            writer.writerow({
                "participant":               pid,
                "tid":                       tid,
                "label":                     cond.get("label", ""),
                "R_over_W":                  round(R / W, 6) if W else "",
                "width_mm":                  round(W * 1000, 1),
                "ID":                        round(ID, 4),
                "has_human_data":            False,
                "lat_rmse":                  "",
                "speed_rmse":                "",
                "speed_corr":                "",
                "time_diff":                 "",
                "goal_approach_speed_human": "",
                "goal_approach_speed_model": m.get("goal_approach_speed_model") or "",
                "goal_approach_speed_diff":  "",
                "throughput_mean_bps":       m.get("throughput_mean_bps")    or "",
                "approach_speed_mean_ms":    m.get("approach_speed_mean_ms") or "",
            })
    print(f"  Saved: {pm_path}")

    # ------------------------------------------------------------------
    # 4. fitts_regression.json — simulator-only, this model_type's run
    # ------------------------------------------------------------------
    # Defensive filter (all_rows never contains "human" rows anymore, since
    # nothing appends them — see main()) kept so this function degrades
    # safely even if a caller feeds it legacy-shaped rows.
    model_rows = [r for r in all_rows if r["source"] != "human"]

    overall_reg   = _compute_fitts_regression(model_rows)
    per_width_reg = _per_width_regression(model_rows)

    plot_fitts_regression(
        model_rows, overall_reg, model_type,
        RESULTS_DIR / f"fitts_regression_plot_{model_type}.png",
    )

    m_tps = [r["TP"] for r in model_rows if r["TP"] is not None]
    tp_cv = (float(np.std(m_tps)) / float(np.mean(m_tps))
             if m_tps and np.mean(m_tps) > 0 else None)

    regression_out = {
        "notes": {
            "ID_formula":       "log2(D/(2R) + 1)  [Shannon; 2R = target width]",
            "MT_model":         "MT = a + b * ID",
            "SPEED_TRIM_START": SPEED_TRIM_START,
            "SPEED_TRIM_END":   SPEED_TRIM_END,
            "model_type":       model_type,
            "sweep_note": (
                "Absolute target-radius sweep in meters (see "
                "DEFAULT_ABSOLUTE_RADII), not a width-relative multiplier. "
                "Both MPCC and the Baseline model run inside an identical "
                f"{BYPASS_TUNNEL_WIDTH_M:.0f}m-wide bypass tunnel so only "
                "target_radius / model_type varies between conditions."
            ),
            "success_criteria": {
                "r_squared":         "> 0.7 overall",
                "b_slope_s_per_bit": "positive",
                "throughput_cv":     "< 0.30 for Fitts constancy",
                "monotonicity_rho":  "< -0.5 (larger R -> faster)",
            },
        },
        "overall":       overall_reg,
        "throughput_cv": round(tp_cv, 4) if tp_cv is not None else None,
        "per_width_mm":  per_width_reg,
    }

    reg_path = RESULTS_DIR / "fitts_regression.json"
    with open(reg_path, "w") as f:
        json.dump(regression_out, f, indent=2)
    print(f"  Saved: {reg_path}")

    return regression_out


# ---------------------------------------------------------------------------
# Console summary table (math/format UNCHANGED, per spec)
# ---------------------------------------------------------------------------

def print_summary_table(aggregate_metrics, fitts_conditions, centerline_lengths):
    print("\n  Summary by condition (averaged across configurations):")
    print("  ┌──────────────────────────────┬───────┬───────┬──────────┬──────────┬──────────┬──────────┐")
    print("  │ Condition                    │  ID   │  MT   │ App.Spd  │    TP    │ Spd.Corr │  n runs  │")
    print("  ├──────────────────────────────┼───────┼───────┼──────────┼──────────┼──────────┼──────────┤")

    by_cond = defaultdict(list)
    for (pid, tid, radius), m in aggregate_metrics.items():
        by_cond[(tid, radius)].append(m)

    for (tid, radius), cond in sorted(fitts_conditions.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        metrics_list = by_cond[(tid, radius)]
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
        # Note: speed_corr_mean is always None now (no human data to
        # correlate against), so `cors` will always be empty and this
        # column will always print "–". This is expected, not a bug — the
        # column/format is preserved exactly per spec, its content is just
        # vacuous now that human comparison is gone.
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
        description="Fitts' Law Evaluation (simulator-only): sweep an "
                     "absolute target radius inside an identical bypass "
                     "tunnel, comparing the Baseline pointing model against "
                     "the MPCC steering model."
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--trials", type=int, nargs="+", default=None)
    parser.add_argument("--radii", type=float, nargs="+", default=DEFAULT_ABSOLUTE_RADII,
                        help="Absolute target radii to sweep, in meters "
                             "(default: %(default)s)")
    parser.add_argument("--model-type", type=str, choices=["mpcc", "baseline"],
                        default="mpcc",
                        help="Which trajectory generator to evaluate.")
    parser.add_argument("--speed-model", type=str, default=None)
    parser.add_argument("--aggregate-only", action="store_true", default=False)
    parser.add_argument("--per-participant", action="store_true", default=False,
                        help="Use per-participant fitted GAM configs from "
                             "eval/model_fitting/results/{pid}_gam_config_s{seed}.json "
                             "instead of a single shared --config.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pid", type=str, default=None,
                        help="Process only this fitted-model ID "
                             "(only meaningful with --per-participant)")
    args = parser.parse_args()

    # Safe key handling: trial_ids defaults to BASE_CONDITIONS' own keys
    # (currently just [11]), and every trial_id is validated against
    # BASE_CONDITIONS up front so a typo'd --trials value fails fast with a
    # clear message instead of a KeyError deep inside a worker process.
    trial_ids = args.trials if args.trials else list(BASE_CONDITIONS.keys())
    unknown_trials = [tid for tid in trial_ids if tid not in BASE_CONDITIONS]
    if unknown_trials:
        print(f"  Unknown trial id(s) {unknown_trials}; only {list(BASE_CONDITIONS.keys())} "
              f"are defined in this simulator-only evaluation.")
        return

    radii = [float(r) for r in args.radii]

    fitts_conditions = build_fitts_conditions(trial_ids, radii)

    print("=" * 70)
    print("Fitts' Law Evaluation (Simulator Only: Baseline vs. MPCC)")
    print("=" * 70)
    print(f"  Model type:   {args.model_type}")
    if args.per_participant:
        print(f"  Mode:         per-participant fitted GAM configs (seed={args.seed})")
    else:
        print(f"  Config:       {args.config}")
    print(f"  Trials:       {trial_ids}")
    print(f"  Abs. radii:   {radii}")
    print(f"  Rounds/cond:  {args.rounds}")
    print(f"  Conditions:   {len(fitts_conditions)}")
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SIM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ---- centerlines (geometry fixed across R values; width is hardcoded
    #      to the bypass tunnel inside _build_sigmoidal_config) ----
    print("[1/3] Generating centerlines ...")
    centerlines        = {}
    centerline_lengths = {}
    for tid in trial_ids:
        cl = generate_centerline(tid)
        centerlines[tid]        = cl
        centerline_lengths[tid] = centerline_arc_length(cl)
        W = BASE_CONDITIONS[tid]["width"]
        D = centerline_lengths[tid]
        if radii:
            ID_ref = fitts_id(D, radii[0])
            print(f"  Trial {tid}: D={D:.4f}m  W={W:.1f}m (bypass)  "
                  f"ID at R={radii[0] * 1000:.2f}mm: {ID_ref:.3f} bits")
        else:
            print(f"  Trial {tid}: D={D:.4f}m  W={W:.1f}m (bypass)  (no radii to sweep)")

    # ---- simulator configuration discovery (no human data anywhere) ----
    print("\n[2/3] Resolving simulator configuration(s) ...")
    try:
        participants = discover_participants(args)
    except FileNotFoundError as e:
        print(f"  {e}")
        return
    if not participants:
        print("  No usable simulator configuration found.")
        return
    print(f"  {len(participants)} configuration(s) to run: {list(participants.keys())}")

    all_rows          = []   # for fitts_results.csv
    aggregate_metrics = {}   # {(pid, tid, radius): metrics_dict}

    if not args.aggregate_only:
        print(f"\n[3/3] Running simulations (model_type={args.model_type}) ...")
        n_workers = os.cpu_count() or 4

        for base_pid, (p_config_path, p_speed_model_path) in sorted(participants.items()):
            # Namespace every output by model_type so a "baseline" run and
            # an "mpcc" run never collide on disk/cache/CSV rows, and are
            # always distinguishable downstream (Section 3).
            pid = f"{base_pid}_{args.model_type}"
            print(f"\n  Run: {pid}")

            cache_path = SIM_CACHE_DIR / f"{pid}_fitts_s{args.seed}.json"
            sim_cache  = load_sim_cache(cache_path) or {}

            p_folder = RESULTS_DIR / f"participant_{pid}"
            p_folder.mkdir(exist_ok=True)

            futures = {}
            condition_segment_data = {}   # {(tid, radius): segment_data}
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                for (tid, radius), cond in sorted(fitts_conditions.items(),
                                                   key=lambda kv: (kv[0][0], kv[0][1])):
                    cache_key    = condition_cache_key(tid, radius)
                    existing     = sim_cache.get(cache_key, [])
                    trial_folder = p_folder / condition_folder_name(tid, radius)

                    fut = executor.submit(
                        _process_condition,
                        p_config_path, p_speed_model_path,
                        pid, tid, radius, cond,
                        args.rounds, existing,
                        str(trial_folder), centerlines[tid],
                        args.model_type,
                    )
                    futures[fut] = (tid, radius)

                try:
                    for fut in concurrent.futures.as_completed(futures, timeout=3600):
                        tid, radius = futures[fut]
                        try:
                            result    = fut.result(timeout=120)
                            cache_key = condition_cache_key(tid, radius)
                            sim_cache[cache_key] = result["all_sim_records"]

                            m = result["metrics"]
                            # Store n counts for summary CSV
                            m["n_valid_runs"]   = result["n_model"]
                            m["n_human_rounds"] = 0
                            aggregate_metrics[(pid, tid, radius)]   = m
                            condition_segment_data[(tid, radius)]   = result["segment_data"]

                            print(f"    tid={tid} R={radius:.4f}m "
                                  f"n={result['n_model']} "
                                  f"MT={m.get('MT_mean_s', '?')} "
                                  f"TP={m.get('throughput_mean_bps', '?')}")
                        except Exception as e:
                            print(f"    ERROR tid={tid} R={radius}: {e}")
                except concurrent.futures.TimeoutError:
                    print("  WARNING: pool timeout")
                    for f in futures:
                        f.cancel()

            save_sim_cache(sim_cache, cache_path)

            # ---- per-width speed-profile summary plots ----
            for tid in trial_ids:
                width_mm = BASE_CONDITIONS[tid]["width"] * 1000
                r_segment_data = {
                    radius: condition_segment_data[(tid, radius)]
                    for radius in radii
                    if (tid, radius) in condition_segment_data
                }
                if not r_segment_data:
                    continue
                fitts_conditions_for_width = {
                    radius: fitts_conditions[(tid, radius)]
                    for radius in radii
                    if (tid, radius) in fitts_conditions
                }
                out_path = p_folder / f"speed_profiles_W{width_mm:.0f}mm_{args.model_type}.png"
                try:
                    plot_speed_profiles_by_width(
                        width_mm, r_segment_data, fitts_conditions_for_width,
                        centerlines[tid], out_path, args.model_type, bin_size=0.1,
                    )
                except Exception as e:
                    print(f"    ERROR plotting speed_profiles_W{width_mm:.0f}mm: {e}")

    # ---- collect rows from saved summaries (supports --aggregate-only) ----
    print("\nGenerating aggregate outputs ...")

    for base_pid in sorted(participants.keys()):
        pid      = f"{base_pid}_{args.model_type}"
        p_folder = RESULTS_DIR / f"participant_{pid}"
        for (tid, radius), cond in sorted(fitts_conditions.items(),
                                           key=lambda kv: (kv[0][0], kv[0][1])):
            W  = cond["width"]
            R  = cond["target_radius"]
            D  = centerline_lengths.get(tid, 0.0)
            ID = fitts_id(D, R)

            summary_path = p_folder / condition_folder_name(tid, radius) / "results_summary.json"
            if not summary_path.exists():
                continue

            with open(summary_path) as f:
                sm = json.load(f)

            # Load metrics into aggregate_metrics if not already there
            # (covers --aggregate-only mode). Key type consistency: `radius`
            # here is the same float used everywhere else this dict is
            # keyed/read (build_fitts_conditions, the live-run branch above),
            # so this lookup is safe.
            key = (pid, tid, radius)
            if key not in aggregate_metrics:
                m = sm.get("metrics", {})
                m["n_valid_runs"]   = sm["model"].get("n_valid_runs", sm["model"]["n_runs"])
                m["n_human_rounds"] = 0
                aggregate_metrics[key] = m

            # Per-run rows for fitts_results.csv
            for run in sm["model"]["runs"]:
                MT  = run["completion_time"]
                app = run.get("approach_speed")
                TP  = ID / MT if MT and MT > 0 else None
                all_rows.append({
                    "source":            f"model_{args.model_type}",
                    "participant":       pid,
                    "tid":               tid,
                    "width_mm":          W * 1000,
                    "R_mm":              R * 1000,
                    "R_over_W":          round(R / W, 6) if W else None,
                    "target_radius_m":   run.get("target_radius", R),
                    "D_m":               round(D, 6),
                    "ID":                round(ID, 4),
                    "MT_s":              MT,
                    "approach_speed_ms": app,
                    "TP":                TP,
                    "avg_speed_ms":      run.get("avg_speed"),
                    "path_length_m":     run.get("path_length"),
                })

    reg = write_aggregate_outputs(
        all_rows, aggregate_metrics, fitts_conditions, centerline_lengths, args.model_type
    )
    print_summary_table(aggregate_metrics, fitts_conditions, centerline_lengths)

    # Quick pass/fail
    overall = reg.get("overall", {})
    print("\n  === Quick verification ===")
    r_sq  = overall.get("r_squared")
    b     = overall.get("b_slope_s_per_bit")
    tp_m  = overall.get("throughput_mean_bps")
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
        print("  Monotonicity rho (target_radius vs MT, should be negative):")
        for w, rho in sorted(mono.items()):
            print(f"    W={w}mm: rho={rho:.3f}  {'✓' if rho < -0.5 else '✗'}")

    print("\n" + "=" * 70)
    print(f"Fitts evaluation complete ({args.model_type}).")
    print(f"Results: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""
ID4SCS / Fitts' Law / Steering Law evaluation for the new single-participant
dataset (human_data/raw_august_2026/), per CLAUDE.md.

Splits trials into three mutually-exclusive buckets by inspecting each
trial's `condition` dict (field-presence based, not hardcoded trial_id
ranges — see bucket_condition()):
    - Steering   : constant-width tunnels (sinusoidal, corner, straight,
                   gentle/sharp sinusoidal). ID = L / W.
    - ID4SCS     : segmented tunnels (wide-to-narrow, narrow-to-wide).
                   MT = a + b*(A1-5W1)/W1 + c*log2(5W1/W2+1) + d*(A2/W2),
                   fit independently per direction.
    - Fitts      : genuine unconstrained pointing tasks (no tunnel).
                   ID = log2(D/(2R)+1), Shannon.
`constrained_to_unconstrained` trials are excluded entirely, per spec.

Uses the shared office_worker.json simulator persona (not a per-participant
fitted config) and CursorSimulator.generate_trajectory_with_waypoints for
all three buckets, including Fitts — the MPCC model, run inside a very
wide ("bypass") tunnel for Fitts tasks so its corridor/contour-tracking
machinery stays active without the corridor actually being binding (see
build_fitts_bypass_config). This replaced an earlier version that used the
constraint-free baseline model (generate_trajectory_with_start_and_end)
for Fitts, which showed jittery start/end behavior and a much larger
distance-driven completion-time gap vs. human data; MPCC-in-a-bypass-tunnel
was verified empirically (with this specific config) to converge without
timeouts across the full real target-radius/distance range.

⚠️ RULED OUT for pointing tasks (2026-08): CursorSimulator.
generate_trajectory_with_start_and_end / hcs_package.baseline_model
(the constraint-free BUMP-style model referenced above) — do not
reintroduce it as the default Fitts-bucket model here without addressing
the jitter/distance-blowup issues described above. Its pygame visual
debugger (run_pointing_experiment/render_pointing_frame/
_create_pointing_env, plus their unconstrained_pointing.json config) was
removed; the method itself still exists on CursorSimulator and can be
called directly if you need to compare against it again.

Directory structure:
    results/
        Fitts/participant_{pid}/trial_{tid}_{targetPosition}/...
        Fitts/{fitts_results.csv, fitts_regression.json, fitts_regression_plot.png}
        Steering/participant_{pid}/trial_{tid}/...
        Steering/{steering_results.csv, steering_law.pdf}
        ID4SCS/{wide_to_narrow,narrow_to_wide}/participant_{pid}/trial_{tid}/...
        ID4SCS/{wide_to_narrow,narrow_to_wide}/id4scs_*_{results.csv,regression.json,regression_plot.png}

Usage:
    python run_eval.py --config <path to persona json>  [--pid P005628] [--aggregate-only]
"""

import argparse
import json
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "hcs_package" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "eval"))
sys.path.insert(0, str(SCRIPT_DIR))

from experiment.environment import create_environment, generate_task_config  # noqa: E402
from hcs_package.cursor_simulator import CursorSimulator  # noqa: E402

from utils.plot_utils import (  # noqa: E402
    plot_experiment_results,
    plot_enhanced_speed_profiles,
    plot_speeds_vs_progress_enhanced,
)

import stats as law_stats  # noqa: E402
import plot as law_plot  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HUMAN_DATA_DIR = PROJECT_ROOT / "human_data" / "raw_august_2026"
RESULTS_DIR = SCRIPT_DIR / "results"
SIM_CACHE_DIR = RESULTS_DIR / "sim_cache"
DEFAULT_CONFIG = (
    PROJECT_ROOT / "hcs_package" / "src" / "hcs_package" / "user_configurations" / "office_worker.json"
)

FITTS_DIR = RESULTS_DIR / "Fitts"
STEERING_DIR = RESULTS_DIR / "Steering"
ID4SCS_DIR = RESULTS_DIR / "ID4SCS"
ID4SCS_W2N_DIR = ID4SCS_DIR / "wide_to_narrow"
ID4SCS_N2W_DIR = ID4SCS_DIR / "narrow_to_wide"

MAX_ROUND_DURATION_S = 60
EXCLUDED_TUNNEL_TYPE = "constrained_to_unconstrained"

BUCKET_LAW_DIR = {
    "steering": STEERING_DIR,
    "id4scs_w2n": ID4SCS_W2N_DIR,
    "id4scs_n2w": ID4SCS_N2W_DIR,
    "fitts": FITTS_DIR,
}


# ---------------------------------------------------------------------------
# Bucketing (field-presence driven — see plan: trial_ids 1-5 have no
# "tunnelType" key at all, so a naive tunnelType string match would drop them)
# ---------------------------------------------------------------------------

def bucket_condition(cond):
    """Returns one of: 'fitts', 'steering', 'id4scs_w2n', 'id4scs_n2w', 'excluded'."""
    ttype = cond.get("tunnelType")
    if ttype == EXCLUDED_TUNNEL_TYPE:
        return "excluded"
    has_seg = "segment1Width" in cond and "segment2Width" in cond
    has_dist = "distance" in cond
    if has_seg and has_dist:
        return "excluded"  # belt-and-suspenders: constrained_to_unconstrained also has both
    if has_seg:
        return "id4scs_n2w" if cond["segment1Width"] < cond["segment2Width"] else "id4scs_w2n"
    if has_dist:
        return "fitts"
    if "tunnelWidth" in cond:
        return "steering"
    return "excluded"


def scan_conditions(data_dir):
    """
    Lightweight pre-pass over the raw JSON (condition/trial_id only, no
    trajectory parsing) to build the trial_id -> condition/bucket maps
    before the full trajectory loader runs.
    """
    tid_to_condition = {}
    tid_to_bucket = {}
    for fpath in sorted(data_dir.glob("*.json")):
        with open(fpath) as f:
            data = json.load(f)
        sessions = data.get("sessions", [])
        if not sessions:
            td_arr = data.get("trialData", [])
            sessions = [{"trialData": td_arr}] if td_arr else []
        for session in sessions:
            for trial_data in session.get("trialData", []):
                tid = trial_data.get("trial_id")
                if tid is None or tid in tid_to_condition:
                    continue
                cond = trial_data.get("condition", {})
                tid_to_condition[tid] = cond
                tid_to_bucket[tid] = bucket_condition(cond)
    return tid_to_condition, tid_to_bucket


# ---------------------------------------------------------------------------
# Trajectory / speed helpers (ported from eval-fitts-continuous-well/run_eval.py)
# ---------------------------------------------------------------------------

def _normalize_trajectory(traj_raw):
    if not traj_raw:
        return []
    if isinstance(traj_raw[0], dict):
        return [[p["x"], p["y"]] for p in traj_raw]
    return traj_raw


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


def _path_length(trajectory):
    total = 0.0
    for i in range(1, len(trajectory)):
        dx = trajectory[i][0] - trajectory[i - 1][0]
        dy = trajectory[i][1] - trajectory[i - 1][1]
        total += math.sqrt(dx * dx + dy * dy)
    return total


def _approach_speed(speeds, trim_pct=0.90):
    if not speeds:
        return 0.0
    start = max(0, int(len(speeds) * trim_pct))
    tail = speeds[start:]
    return float(np.mean(tail)) if tail else 0.0


# ---------------------------------------------------------------------------
# Human data loading
# ---------------------------------------------------------------------------

def load_trials_by_participant(trial_ids):
    """
    Returns {pid: {tid: {round: {trajectory, speeds, timestamps,
                                  completion_time, condition}}}}
    """
    trial_ids_set = set(trial_ids)
    all_data = {}

    for fpath in sorted(HUMAN_DATA_DIR.glob("*.json")):
        try:
            with open(fpath) as f:
                data = json.load(f)

            pid = data.get("participantId", fpath.stem)
            sessions = data.get("sessions", [])
            if not sessions:
                td_arr = data.get("trialData", [])
                sessions = [{"trialData": td_arr}] if td_arr else []

            if pid not in all_data:
                all_data[pid] = {}

            for session in sessions:
                for trial_data in session.get("trialData", []):
                    tid = trial_data.get("trial_id")
                    round_num = trial_data.get("round", 0)
                    if tid not in trial_ids_set:
                        continue

                    if tid not in all_data[pid]:
                        all_data[pid][tid] = {}

                    traj_raw = _normalize_trajectory(trial_data.get("trajectory", []))
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
                        "trajectory": traj_raw,
                        "speeds": speeds,
                        "timestamps": timestamps,
                        "completion_time": trial_data.get("completionTime", duration_s),
                        "condition": trial_data.get("condition", {}),
                    }
        except Exception as e:
            print(f"  Warning: error loading {fpath.name}: {e}")

    return all_data


# ---------------------------------------------------------------------------
# Environment / task-config builders
# ---------------------------------------------------------------------------

def _build_sigmoidal_config(width, curvature, target_radius=None):
    if target_radius is None:
        target_radius = width * 0.5
    env_dict = {
        "env_type": "tunnel_steering_smooth",
        "screen_width": 460,
        "screen_height": 260,
        "tunnelWidth": width,
        "curvature": curvature,
        "max_steps": 800,
        "target_radius": target_radius,
    }
    environment = create_environment(env_dict)
    task_config = generate_task_config(environment, include_constraints=True)
    return task_config, environment["centerline"]


def _build_corner_config(width, num_corners, corner_offset):
    env_dict = {
        "env_type": "tunnel_steering_corner",
        "screen_width": 460,
        "screen_height": 260,
        "tunnelWidth": width,
        "num_corners": num_corners,
        "corner_offset": corner_offset,
        "max_steps": 800,
        "target_radius": width * 0.5,
    }
    environment = create_environment(env_dict)
    task_config = generate_task_config(environment, include_constraints=True)
    return task_config, environment["centerline"]


def _build_wide_to_narrow_config(segment1_width, segment2_width, curvature=0.0):
    env_dict = {
        "env_type": "tunnel_steering_smooth",
        "screen_width": 460,
        "screen_height": 260,
        "tunnelWidth": segment1_width,
        "segment1Width": segment1_width,
        "segment2Width": segment2_width,
        "curvature": curvature,
        "max_steps": 800,
        # Terminate reliably at the tunnel end: use segment2 (the exit segment).
        "target_radius": segment2_width * 0.5,
    }
    environment = create_environment(env_dict)
    task_config = generate_task_config(environment, include_constraints=True)
    return task_config, environment["centerline"]


def build_steering_task_config(cond):
    """
    Dispatch a Steering-bucket condition dict to the right env builder.
    Condition keys are camelCase, matching the raw data file
    (numCorners, cornerOffset, radiusRatio) — not the snake_case builder
    *parameter* names.
    """
    if cond.get("tunnelType") == "corner":
        return _build_corner_config(cond["tunnelWidth"], cond["numCorners"], cond["cornerOffset"])
    width = cond["tunnelWidth"]
    radius_ratio = cond.get("radiusRatio")
    target_radius = width * radius_ratio if radius_ratio is not None else None
    return _build_sigmoidal_config(width, cond.get("curvature", 0.0), target_radius=target_radius)


def make_width_profile(centerline, w1, w2):
    """Mirrors _create_tunnel_env's segment split (experiment/environment.py):
    width changes from w1 to w2 at the centerline midpoint index."""
    n = len(centerline)
    mid = n // 2
    return [w1] * mid + [w2] * (n - mid)


# Very wide "bypass" tunnel width for approximating unconstrained pointing
# with the MPCC model: >> the ~0.46m screen and >> any realistic target
# radius, so the corridor constraint is never actually binding. Verified
# empirically (see plan notes) that office_worker.json + this bypass width
# converges (no timeouts) across the full real P005628 distance/radius
# range — unlike a per-participant fitted config tested separately, which
# timed out at small radii; the two configs behave differently here.
BYPASS_TUNNEL_WIDTH_M = 10.0


def build_fitts_bypass_config(human_round, target_radius, max_steps=800):
    """MPCC-based unconstrained-pointing task: a straight, very wide
    ('bypass') tunnel between a human round's actual recorded start/end
    points, so generate_trajectory_with_waypoints has a (non-binding)
    corridor to track against. Returns (task_config, centerline, tunnel_width)."""
    traj = human_round["trajectory"]
    start_m, end_m = traj[0], traj[-1]
    env_dict = {
        "env_type": "unconstrained_pointing_mpcc",
        "screen_width": 460, "screen_height": 260,
        "start_x": start_m[0], "start_y": start_m[1],
        "distance": end_m[0] - start_m[0],
        "target_y_offset": end_m[1] - start_m[1],
        "target_radius": target_radius,
        "max_steps": max_steps,
        "tunnel_width": BYPASS_TUNNEL_WIDTH_M,
    }
    environment = create_environment(env_dict)
    task_config = generate_task_config(environment, include_constraints=True)
    return task_config, environment["centerline"], environment["tunnel_width"]


# ---------------------------------------------------------------------------
# Simulator runners
# ---------------------------------------------------------------------------

def _build_record(traj_raw, interval, target_radius, max_steps):
    scale = 0.001
    traj = [[x * scale, y * scale] for x, y, _ in traj_raw]
    n_pts = len(traj)
    speeds = []
    for i in range(1, n_pts):
        d = math.sqrt((traj[i][0] - traj[i - 1][0]) ** 2 + (traj[i][1] - traj[i - 1][1]) ** 2)
        speeds.append(d / interval)
    if speeds:
        speeds.insert(0, speeds[0])
    pl = _path_length(traj)
    ct = n_pts * interval
    app = _approach_speed(speeds)
    # The simulator ran out of its step budget without settling in the
    # target (never reached the dwell-based termination condition), rather
    # than genuinely completing the movement — i.e. it "timed out."
    timed_out = n_pts >= max_steps
    return {
        "trajectory": traj,
        "speeds": speeds,
        "completion_time": ct,
        "path_length": pl,
        "avg_speed": pl / ct if ct > 0 else 0.0,
        "approach_speed": app,
        "target_radius": target_radius,
        "timed_out": timed_out,
    }


def run_tunnel_simulator(sim, task_config, target_radius, n_runs):
    """Steering / ID4SCS: generate_trajectory_with_waypoints, respects the
    tunnel corridor."""
    interval = sim.interval
    max_steps = task_config.get("max_steps", 800)
    records = []
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        json.dump(task_config, tf)
        task_file = tf.name
    try:
        for _ in range(n_runs):
            traj_raw = sim.generate_trajectory_with_waypoints(
                task_file=task_file,
                max_steps=max_steps,
                target_radius=target_radius,
                use_optimal_path=True,
            )
            records.append(_build_record(traj_raw, interval, target_radius, max_steps))
    finally:
        os.unlink(task_file)
    return records


# ---------------------------------------------------------------------------
# Sim cache helpers
# ---------------------------------------------------------------------------

def load_sim_cache(cache_path):
    if not cache_path.exists():
        return {}
    try:
        with open(cache_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  Warning: failed to load sim cache ({e}), will re-run")
        return {}


def save_sim_cache(cache_dict, cache_path):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache_dict, f)


# ---------------------------------------------------------------------------
# Per-condition processing (plots + results_summary.json)
# ---------------------------------------------------------------------------

def process_condition(law_dir, pid, tid, folder_name, human_rounds, model_records,
                       tunnel_path, tunnel_width, condition):
    trial_folder = law_dir / f"participant_{pid}" / folder_name
    trial_folder.mkdir(parents=True, exist_ok=True)

    human_trajectories = [r["trajectory"] for r in human_rounds]
    human_speeds_list = [r["speeds"] for r in human_rounds]
    model_trajectories = [r["trajectory"] for r in model_records]
    model_speeds_list = [r["speeds"] for r in model_records]

    segment_data = {"sim": {"trajectories": model_trajectories, "speeds": model_speeds_list}}
    if human_trajectories:
        segment_data["human"] = {"trajectories": human_trajectories, "speeds": human_speeds_list}

    all_results = {tid: {0: segment_data}}
    tunnel_paths = {tid: tunnel_path}
    tunnel_widths = {tid: tunnel_width}

    plot_experiment_results(all_results, trial_folder, tunnel_paths, tunnel_widths)
    plot_enhanced_speed_profiles(all_results, trial_folder, time_step=0.05)
    if tunnel_path and len(tunnel_path) >= 2:
        plot_speeds_vs_progress_enhanced(all_results, trial_folder, tunnel_paths, bin_size=0.1)

    # Experiment-main-style human-vs-model comparison metrics (lateral RMSE,
    # speed RMSE/correlation, goal-approach speed, time diff) — additive on
    # top of the raw per-round completion times/path lengths below.
    metrics = law_stats.compute_comparison_metrics(
        model_trajectories, model_speeds_list, [r["completion_time"] for r in model_records],
        human_trajectories, human_speeds_list, [r["completion_time"] for r in human_rounds],
        tunnel_path,
    )

    summary = {
        "trial_id": tid,
        "condition": condition,
        "human": {
            "n_rounds": len(human_rounds),
            "completion_times": [r["completion_time"] for r in human_rounds],
        },
        "model": {
            "n_valid_runs": len(model_records),
            "completion_times": [r["completion_time"] for r in model_records],
            "path_lengths": [r["path_length"] for r in model_records],
        },
        "metrics": metrics,
    }
    with open(trial_folder / "results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ---------------------------------------------------------------------------
# Per-condition job preparation (parent process) + worker (child process)
#
# Each condition (tid) runs in its own ProcessPoolExecutor worker, matching
# the pattern already used by eval-fitts-continuous-well/run_eval.py: a
# hung/slow MPCC solve in one worker doesn't block the others, and results
# are collected with a per-future timeout so a stuck condition is detected
# and reported instead of silently hanging the whole script. CursorSimulator
# is NOT passed across the process boundary (it isn't designed to be
# pickled) — each worker reconstructs its own instance from config_path.
# ---------------------------------------------------------------------------

PER_CONDITION_TIMEOUT_S = 180
POOL_TIMEOUT_S = 3600


def build_condition_job(pid, tid, bucket, cond, human_rounds, cached_records,
                         config_path_str, tid_geometry, simulate):
    """Prepare a picklable job dict for one (pid, tid) condition."""
    law_dir = BUCKET_LAW_DIR[bucket]
    n_needed = (len(human_rounds) - len(cached_records)) if simulate else 0
    n_needed = max(n_needed, 0)

    job = {
        "pid": pid, "tid": tid, "bucket": bucket, "cond": cond,
        "human_rounds": human_rounds, "cached_records": cached_records,
        "n_needed": n_needed, "config_path": config_path_str,
        "law_dir": str(law_dir),
    }

    if bucket == "steering":
        task_config, centerline = tid_geometry[tid]
        job.update({
            "task_config": task_config, "centerline": centerline,
            "tunnel_width": cond["tunnelWidth"], "folder_name": f"trial_{tid}",
        })
    elif bucket in ("id4scs_w2n", "id4scs_n2w"):
        task_config, centerline = tid_geometry[tid]
        W1, W2 = cond["segment1Width"], cond["segment2Width"]
        A1, A2 = law_stats.segment_arc_lengths(centerline)
        job.update({
            "task_config": task_config, "centerline": centerline,
            "tunnel_width": make_width_profile(centerline, W1, W2),
            "folder_name": f"trial_{tid}",
            "W1": W1, "W2": W2, "A1": A1, "A2": A2,
        })
    elif bucket == "fitts":
        target_position = cond.get("targetPosition", "unknown")
        starts = [r["trajectory"][0] for r in human_rounds]
        ends = [r["trajectory"][-1] for r in human_rounds]
        start_mean = [float(np.mean([s[0] for s in starts])), float(np.mean([s[1] for s in starts]))]
        end_mean = [float(np.mean([e[0] for e in ends])), float(np.mean([e[1] for e in ends]))]
        job.update({
            "canonical_path": [start_mean, end_mean],
            "folder_name": f"trial_{tid}_{target_position}",
        })

    return job


def _run_condition_job(job):
    """
    Worker entry point (runs in a ProcessPoolExecutor child process). Takes
    only plain dicts/lists/strings (picklable), reconstructs its own
    CursorSimulator if new simulation is needed, and returns a plain dict.
    """
    import matplotlib
    matplotlib.use("Agg")

    bucket = job["bucket"]
    tid = job["tid"]
    pid = job["pid"]
    cond = job["cond"]
    human_rounds = job["human_rounds"]
    cached = job["cached_records"]
    n_needed = job["n_needed"]

    new_records = []
    if bucket in ("steering", "id4scs_w2n", "id4scs_n2w"):
        task_config, centerline = job["task_config"], job["centerline"]
        target_radius = task_config["target_radius"]
        if n_needed > 0:
            sim = CursorSimulator(job["config_path"])
            new_records = run_tunnel_simulator(sim, task_config, target_radius, n_needed)
        tunnel_path = centerline
        tunnel_width = job["tunnel_width"]
    else:  # fitts
        target_radius = cond["targetRadius"]
        if n_needed > 0:
            sim = CursorSimulator(job["config_path"])
            for i in range(n_needed):
                round_idx = len(cached) + i
                task_config, _centerline, _width = build_fitts_bypass_config(
                    human_rounds[round_idx], target_radius
                )
                new_records.extend(run_tunnel_simulator(sim, task_config, target_radius, n_runs=1))
        tunnel_path = job["canonical_path"]
        tunnel_width = BYPASS_TUNNEL_WIDTH_M

    model_records = (cached + new_records)[: len(human_rounds)]
    if not model_records:
        return {"tid": tid, "bucket": bucket, "new_records": new_records, "rows": [],
                "condition_summary_row": None, "skipped": True}

    law_dir = Path(job["law_dir"])
    summary = process_condition(law_dir, pid, tid, job["folder_name"], human_rounds,
                                 model_records, tunnel_path, tunnel_width, cond)
    metrics = summary.get("metrics", {})

    n_timed_out = sum(1 for r in model_records if r.get("timed_out"))

    rows = []
    cond_summary_row = {"tid": tid, "n_human": len(human_rounds), "n_model": len(model_records),
                         "n_timed_out": n_timed_out}
    cond_summary_row.update(metrics)

    if bucket == "steering":
        width = cond["tunnelWidth"]
        L = law_stats.centerline_arc_length(tunnel_path)
        ID = L / width if width > 0 else 0.0
        for r in human_rounds:
            rows.append({"source": "Human", "tid": tid, "ID": ID, "MT_s": r["completion_time"]})
        for r in model_records:
            rows.append({"source": "Simulator", "tid": tid, "ID": ID, "MT_s": r["completion_time"],
                         "timed_out": r.get("timed_out", False)})
        cond_summary_row.update({
            "width_mm": round(width * 1000, 2), "ID_L_over_W": round(ID, 4),
            "target_radius_mm": round(target_radius * 1000, 3),
            "radius_ratio": cond.get("radiusRatio"),  # only set for "straight" trials (tids 11-25)
        })

    elif bucket in ("id4scs_w2n", "id4scs_n2w"):
        W1, W2, A1, A2 = job["W1"], job["W2"], job["A1"], job["A2"]

        def _mk_row(source, mt, timed_out=None):
            s1 = (A1 - 5 * W1) / W1 if W1 > 0 else 0.0
            c = math.log2(5 * W1 / W2 + 1) if W2 > 0 else 0.0
            s2 = A2 / W2 if W2 > 0 else 0.0
            row = {"source": source, "tid": tid, "S1": s1, "C": c, "S2": s2, "MT": mt}
            if timed_out is not None:
                row["timed_out"] = timed_out
            return row

        for r in human_rounds:
            rows.append(_mk_row("Human", r["completion_time"]))
        for r in model_records:
            rows.append(_mk_row("Simulator", r["completion_time"], r.get("timed_out", False)))
        cond_summary_row.update({
            "W1_mm": round(W1 * 1000, 2), "W2_mm": round(W2 * 1000, 2),
            "A1_m": round(A1, 4), "A2_m": round(A2, 4),
        })

    else:  # fitts
        target_position = cond.get("targetPosition", "unknown")

        def _fitts_row(source, mt, traj, timed_out=None):
            D = math.hypot(traj[-1][0] - traj[0][0], traj[-1][1] - traj[0][1])
            ID = law_stats.fitts_id(D, target_radius)
            TP = ID / mt if mt and mt > 0 else None
            row = {"source": source, "tid": tid, "ID": ID, "MT_s": mt, "TP": TP,
                   "D_m": D, "target_position": target_position}
            if timed_out is not None:
                row["timed_out"] = timed_out
            return row

        for r in human_rounds:
            rows.append(_fitts_row("Human", r["completion_time"], r["trajectory"]))
        for human_r, model_r in zip(human_rounds, model_records):
            rows.append(_fitts_row("Simulator", model_r["completion_time"], human_r["trajectory"],
                                    model_r.get("timed_out", False)))

        model_tps = [r["TP"] for r in rows if r["source"] == "Simulator" and r["TP"] is not None]
        cond_summary_row.update({
            "target_position": target_position, "target_radius_mm": round(target_radius * 1000, 2),
            "throughput_mean_bps": round(float(np.mean(model_tps)), 4) if model_tps else None,
        })

    return {
        "tid": tid, "bucket": bucket, "new_records": new_records,
        "rows": rows, "condition_summary_row": cond_summary_row, "skipped": False,
    }


def process_participant(pid, participant_data, config_path_str, tid_to_condition, tid_to_bucket,
                         tid_geometry, sim_cache, simulate=True):
    """
    Dispatches every included tid this participant has data for to a
    ProcessPoolExecutor worker (see _run_condition_job), collects results
    with a per-condition timeout, updates sim_cache with any newly
    simulated records, and returns:
        rows_by_bucket:      {"steering": [...], ...}  (for regression CSVs/plots)
        condition_summaries: {"steering": [...], ...}  (one row per tid, incl.
                              the experiment-main RMSE/correlation metrics)
    """
    import concurrent.futures

    jobs = {}
    for tid in sorted(participant_data.keys()):
        bucket = tid_to_bucket.get(tid)
        if bucket in (None, "excluded"):
            continue
        cond = tid_to_condition[tid]
        human_rounds_dict = participant_data[tid]
        round_nums = sorted(human_rounds_dict.keys())
        human_rounds = [human_rounds_dict[r] for r in round_nums]
        if not human_rounds:
            continue

        cache_key = f"{pid}_{tid}"
        cached = sim_cache.get(cache_key, [])
        n_needed = max(len(human_rounds) - len(cached), 0) if simulate else 0
        if n_needed > 0:
            print(f"    tid={tid} ({bucket}): cache has {len(cached)}/{len(human_rounds)} "
                  f"runs, simulating {n_needed} more", flush=True)
        else:
            print(f"    tid={tid} ({bucket}): cache hit, reusing {min(len(cached), len(human_rounds))} "
                  f"cached run(s)" + ("" if simulate else " [--aggregate-only]"), flush=True)

        jobs[tid] = build_condition_job(
            pid, tid, bucket, cond, human_rounds, cached, config_path_str, tid_geometry, simulate
        )

    rows_by_bucket = defaultdict(list)
    condition_summaries = defaultdict(list)

    if not jobs:
        return rows_by_bucket, condition_summaries

    n_workers = os.cpu_count() or 4
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_run_condition_job, job): tid for tid, job in jobs.items()}
        try:
            for fut in concurrent.futures.as_completed(futures, timeout=POOL_TIMEOUT_S):
                tid = futures[fut]
                try:
                    result = fut.result(timeout=PER_CONDITION_TIMEOUT_S)
                except concurrent.futures.TimeoutError:
                    print(f"    ERROR tid={tid}: worker exceeded {PER_CONDITION_TIMEOUT_S}s timeout, skipped",
                          flush=True)
                    continue
                except Exception as e:
                    print(f"    ERROR tid={tid}: {e}", flush=True)
                    continue

                bucket = result["bucket"]
                if result["new_records"]:
                    cache_key = f"{pid}_{tid}"
                    sim_cache[cache_key] = jobs[tid]["cached_records"] + result["new_records"]
                if result["skipped"]:
                    print(f"    tid={tid}: no model records available, skipped", flush=True)
                    continue

                rows_by_bucket[bucket].extend(result["rows"])
                condition_summaries[bucket].append(result["condition_summary_row"])
                print(f"    tid={tid} ({bucket}) done: {len(result['rows'])} rows", flush=True)
        except concurrent.futures.TimeoutError:
            print(f"    WARNING: pool exceeded overall {POOL_TIMEOUT_S}s timeout, "
                  f"remaining conditions cancelled", flush=True)
            for f in futures:
                f.cancel()

    return rows_by_bucket, condition_summaries


# ---------------------------------------------------------------------------
# Aggregate outputs
# ---------------------------------------------------------------------------

def _write_dict_rows_csv(rows, out_path):
    """Writes a list of (possibly heterogeneous-keyed) dicts to CSV, taking
    the union of keys across all rows as the header."""
    import csv

    if not rows:
        return
    fieldnames = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (r.get(k) if r.get(k) is not None else "") for k in fieldnames})
    print(f"  Saved: {out_path}")


def _exclude_timed_out(rows):
    """Drop simulator rows that hit the step-budget timeout (diverged,
    never actually reached/settled at the target) before fitting a
    regression or plotting — matches the older reference pipelines'
    _MAX_CT divergence filter. Human rows (no 'timed_out' key) pass
    through unaffected. The raw per-run CSVs still include everything,
    including timed-out rows, for full auditability."""
    return [r for r in rows if not r.get("timed_out")]


def write_steering_outputs(rows, condition_summaries):
    STEERING_DIR.mkdir(parents=True, exist_ok=True)
    _write_dict_rows_csv(rows, STEERING_DIR / "steering_results.csv")

    # Per-condition experiment-main-style comparison metrics (lateral/speed
    # RMSE, speed correlation, goal-approach speed, time diff) — additive
    # on top of the per-run results CSV above.
    _write_dict_rows_csv(sorted(condition_summaries, key=lambda r: r["tid"]),
                          STEERING_DIR / "steering_condition_summary.csv")

    law_plot.steering_law_plot(_exclude_timed_out(rows), STEERING_DIR / "steering_law.pdf")


def write_id4scs_outputs(rows, condition_summaries, direction_label, out_dir, filename_prefix):
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_dict_rows_csv(rows, out_dir / f"{filename_prefix}_results.csv")
    _write_dict_rows_csv(sorted(condition_summaries, key=lambda r: r["tid"]),
                          out_dir / f"{filename_prefix}_condition_summary.csv")

    clean_rows = _exclude_timed_out(rows)
    for source, tag in (("Human", "human"), ("Simulator", "model")):
        sub_rows = [r for r in clean_rows if r["source"] == source]
        reg = law_stats.fit_id4scs(sub_rows)
        reg_path = out_dir / f"{filename_prefix}_{tag}_regression.json"
        with open(reg_path, "w") as f:
            json.dump(reg, f, indent=2)
        print(f"  Saved: {reg_path}")
        if reg:
            plot_path = out_dir / f"{filename_prefix}_{tag}_regression_plot.png"
            law_plot.plot_id4scs_regression(sub_rows, reg, plot_path, f"{direction_label} ({source})")
        else:
            print(f"  Skipped {tag} regression plot for {direction_label}: "
                  f"only {len(sub_rows)} rows (need >= 5)")


def write_fitts_outputs(rows, condition_summaries):
    from collections import defaultdict as dd

    FITTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_dict_rows_csv(rows, FITTS_DIR / "fitts_results.csv")
    _write_dict_rows_csv(sorted(condition_summaries, key=lambda r: r["tid"]),
                          FITTS_DIR / "fitts_condition_summary.csv")

    def _fit(rows_sub):
        sub = [r for r in rows_sub if r["MT_s"] is not None and not r.get("timed_out")]
        if len(sub) < 3:
            return {}
        ids = np.array([r["ID"] for r in sub])
        mts = np.array([r["MT_s"] for r in sub])
        tps = np.array([r["TP"] for r in sub if r.get("TP") is not None])
        b, a = np.polyfit(ids, mts, 1)
        pred = a + b * ids
        ss_res = np.sum((mts - pred) ** 2)
        ss_tot = np.sum((mts - np.mean(mts)) ** 2)
        r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return {
            "a_intercept": round(float(a), 6),
            "b_slope_s_per_bit": round(float(b), 6),
            "r_squared": round(float(r_sq), 4),
            "throughput_mean_bps": round(float(np.mean(tps)), 4) if len(tps) else None,
            "throughput_std_bps": round(float(np.std(tps)), 4) if len(tps) else None,
            "n_runs": len(sub),
        }

    clean_rows = _exclude_timed_out(rows)
    model_rows = [r for r in clean_rows if r["source"] == "Simulator"]
    human_rows = [r for r in clean_rows if r["source"] == "Human"]
    model_reg = _fit(model_rows)
    human_reg = _fit(human_rows)

    law_plot.plot_fitts_regression(
        model_rows, human_rows, model_reg, human_reg, FITTS_DIR / "fitts_regression_plot.png"
    )

    n_excluded = sum(1 for r in rows if r["source"] == "Simulator" and r.get("timed_out"))
    regression_out = {
        "notes": {
            "ID_formula": "log2(D/(2R) + 1)  [Shannon; 2R = target width]",
            "D_formula": "Euclidean sqrt(x^2+y^2) between each round's recorded "
                         "trajectory start/end points (not the nominal 'distance' field)",
            "MT_model": "MT = a + b * ID",
            "timed_out_runs_excluded": n_excluded,
        },
        "model": model_reg,
        "human": human_reg,
    }
    reg_path = FITTS_DIR / "fitts_regression.json"
    with open(reg_path, "w") as f:
        json.dump(regression_out, f, indent=2)
    print(f"  Saved: {reg_path}")
    if n_excluded:
        print(f"  ({n_excluded} timed-out simulator run(s) excluded from the regression/plot; "
              f"still counted in fitts_condition_summary.csv's n_timed_out and present in fitts_results.csv)")

    # --- "too-short pointing distances" diagnostic (CLAUDE.md sec. 2) ---
    by_cond = dd(list)
    for r in human_rows:
        by_cond[(r["tid"], r["target_position"])].append(r["ID"])
    if by_cond:
        mean_ids = {k: float(np.mean(v)) for k, v in by_cond.items()}
        id_min, id_max = min(mean_ids.values()), max(mean_ids.values())
        print(f"\n  Fitts distance diagnostic: ID range across {len(mean_ids)} "
              f"conditions spans {id_min:.2f} to {id_max:.2f} bits.")
        if id_max - id_min < 2.0:
            print("  -> Narrow ID range: pointing distances collected so far may be "
                  "too short/undifferentiated to fit a reliable Fitts' Law regression.")
        else:
            print("  -> ID range looks reasonably wide for a Fitts' Law regression.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ID4SCS / Fitts' Law / Steering Law evaluation for the new single-participant dataset"
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--pid", type=str, default=None, help="Process only this participant ID")
    parser.add_argument("--aggregate-only", action="store_true", default=False,
                        help="Skip simulation, regenerate CSVs/plots from cached results only "
                             "(conditions with no prior cached run are skipped)")
    parser.add_argument("--fresh-sim", action="store_true", default=False,
                        help="Ignore any cached simulator runs and resimulate every condition "
                             "from scratch. The simulator applies stochastic per-step motor/"
                             "device noise, so this produces a different random draw than "
                             "whatever is currently cached, not just a repeat of it.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return

    for d in (FITTS_DIR, STEERING_DIR, ID4SCS_W2N_DIR, ID4SCS_N2W_DIR, SIM_CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Eval New Data: ID4SCS / Fitts' Law / Steering Law")
    print("=" * 70)
    print(f"  Config: {config_path}")

    print("\n[1/4] Scanning trial conditions ...")
    tid_to_condition, tid_to_bucket = scan_conditions(HUMAN_DATA_DIR)
    bucket_counts = Counter(tid_to_bucket.values())
    for b in ("steering", "id4scs_w2n", "id4scs_n2w", "fitts", "excluded"):
        print(f"  {b:12s}: {bucket_counts.get(b, 0):3d} trial_ids")
    n_included = sum(v for k, v in bucket_counts.items() if k != "excluded")
    print(f"  Total: {n_included} included, {bucket_counts.get('excluded', 0)} excluded, "
          f"{len(tid_to_bucket)} scanned")

    included_tids = sorted(t for t, b in tid_to_bucket.items() if b != "excluded")

    print("\n[2/4] Loading human data ...")
    all_human_data = load_trials_by_participant(included_tids)
    if args.pid:
        all_human_data = {args.pid: all_human_data.get(args.pid, {})}
    if not all_human_data or all(not v for v in all_human_data.values()):
        print("  No human data found.")
        return
    print(f"  Participants: {sorted(all_human_data.keys())}")

    # Fixed geometry (task_config, centerline) per tid — same across rounds/participants.
    tid_geometry = {}
    for tid, bucket in tid_to_bucket.items():
        cond = tid_to_condition[tid]
        if bucket == "steering":
            tid_geometry[tid] = build_steering_task_config(cond)
        elif bucket in ("id4scs_w2n", "id4scs_n2w"):
            tid_geometry[tid] = _build_wide_to_narrow_config(
                cond["segment1Width"], cond["segment2Width"], cond.get("curvature", 0.0)
            )

    all_rows = defaultdict(list)             # bucket -> rows (across all participants)
    all_condition_summaries = defaultdict(list)  # bucket -> one row per (pid, tid)
    simulate = not args.aggregate_only

    if args.fresh_sim and args.aggregate_only:
        print("  --fresh-sim and --aggregate-only are contradictory; ignoring --aggregate-only.")
        simulate = True

    print("\n[3/4] Running simulations ..." if simulate
          else "\n[3/4] Aggregating from cached simulator runs (--aggregate-only) ...")
    for pid, participant_data in sorted(all_human_data.items()):
        print(f"\n  Participant: {pid}")
        cache_path = SIM_CACHE_DIR / f"{pid}_sim_cache.json"
        sim_cache = {} if args.fresh_sim else load_sim_cache(cache_path)
        if args.fresh_sim:
            print("    --fresh-sim: ignoring any existing cache, resimulating every condition "
                  "(stochastic noise means these will differ from the previous cached run)")

        rows_by_bucket, condition_summaries = process_participant(
            pid, participant_data, str(config_path), tid_to_condition, tid_to_bucket, tid_geometry,
            sim_cache, simulate=simulate,
        )
        if simulate:
            save_sim_cache(sim_cache, cache_path)

        for bucket, rows in rows_by_bucket.items():
            all_rows[bucket].extend(rows)
        for bucket, summaries in condition_summaries.items():
            all_condition_summaries[bucket].extend(summaries)
        print(f"    steering={len(rows_by_bucket.get('steering', []))} rows, "
              f"id4scs_w2n={len(rows_by_bucket.get('id4scs_w2n', []))} rows, "
              f"id4scs_n2w={len(rows_by_bucket.get('id4scs_n2w', []))} rows, "
              f"fitts={len(rows_by_bucket.get('fitts', []))} rows")

    print("\n[4/4] Generating aggregate outputs ...")
    if all_rows.get("steering"):
        write_steering_outputs(all_rows["steering"], all_condition_summaries["steering"])
    if all_rows.get("id4scs_w2n"):
        write_id4scs_outputs(all_rows["id4scs_w2n"], all_condition_summaries["id4scs_w2n"],
                              "Wide-to-Narrow", ID4SCS_W2N_DIR, "id4scs_wide_to_narrow")
    if all_rows.get("id4scs_n2w"):
        write_id4scs_outputs(all_rows["id4scs_n2w"], all_condition_summaries["id4scs_n2w"],
                              "Narrow-to-Wide", ID4SCS_N2W_DIR, "id4scs_narrow_to_wide")
    if all_rows.get("fitts"):
        write_fitts_outputs(all_rows["fitts"], all_condition_summaries["fitts"])

    print("\n" + "=" * 70)
    print("Eval complete.")
    print(f"Results: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

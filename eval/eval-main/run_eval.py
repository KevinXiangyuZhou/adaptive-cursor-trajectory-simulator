"""
Main evaluation: ID4SCS / Fitts' Law / Steering Law on the Prolific dataset
(human_data/aug-26-prolific/ by default; override with --data-dir).

Derived from eval/eval-new-data/run_eval.py. Adds a `--human-only` mode that
never instantiates CursorSimulator (no simulation, no sim cache): it plots the
human trajectories / speed profiles per trial, writes a per-participant
overview grid of every trial (results/overview/participant_{pid}_overview.png)
and a per-round QC table (results/overview/human_rounds_qc.csv) so the raw
data can be checked for cleanliness by eye before any model comparison.
In --human-only mode the constrained_to_unconstrained trials (excluded from
the law fits) are ALSO plotted, under results/ConstrainedToUnconstrained/.

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
`constrained_to_unconstrained` (c2u) trials are simulated and reported but
take part in no law regression and no fitting stage (transfer test only).

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
    python run_eval.py --human-only                       # visualize human data only
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

from experiment.environment import create_environment, generate_task_config, POINTING_Y_OFFSET  # noqa: E402
from experiment.utils import generateTunnelBoundaries  # noqa: E402
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
HUMAN_DATA_DIR = PROJECT_ROOT / "human_data" / "aug-26-prolific"  # overridable via --data-dir
# Override with HCS_EVAL_RESULTS_DIR (Great Lakes: .../projects/chi-27/results/eval-main)
RESULTS_DIR = Path(os.environ.get("HCS_EVAL_RESULTS_DIR", SCRIPT_DIR / "results"))
SIM_CACHE_DIR = RESULTS_DIR / "sim_cache"
DEFAULT_CONFIG = (
    PROJECT_ROOT / "hcs_package" / "src" / "hcs_package" / "user_configurations" / "office_worker.json"
)

FITTS_DIR = RESULTS_DIR / "Fitts"
STEERING_DIR = RESULTS_DIR / "Steering"
ID4SCS_DIR = RESULTS_DIR / "ID4SCS"
ID4SCS_W2N_DIR = ID4SCS_DIR / "wide_to_narrow"
ID4SCS_N2W_DIR = ID4SCS_DIR / "narrow_to_wide"
C2U_DIR = RESULTS_DIR / "ConstrainedToUnconstrained"   # only populated in --human-only mode
OVERVIEW_DIR = RESULTS_DIR / "overview"

MAX_ROUND_DURATION_S = 60
EXCLUDED_TUNNEL_TYPE = "constrained_to_unconstrained"

BUCKET_LAW_DIR = {
    "steering": STEERING_DIR,
    "id4scs_w2n": ID4SCS_W2N_DIR,
    "id4scs_n2w": ID4SCS_N2W_DIR,
    "fitts": FITTS_DIR,
    "c2u": C2U_DIR,
}


def _set_results_dir(base):
    """Redirect every output/cache path to `base` (for --results-dir, so
    model variants can be evaluated side by side without clobbering)."""
    global RESULTS_DIR, SIM_CACHE_DIR, FITTS_DIR, STEERING_DIR, ID4SCS_DIR
    global ID4SCS_W2N_DIR, ID4SCS_N2W_DIR, C2U_DIR, OVERVIEW_DIR, BUCKET_LAW_DIR
    RESULTS_DIR = Path(base)
    SIM_CACHE_DIR = RESULTS_DIR / "sim_cache"
    FITTS_DIR = RESULTS_DIR / "Fitts"
    STEERING_DIR = RESULTS_DIR / "Steering"
    ID4SCS_DIR = RESULTS_DIR / "ID4SCS"
    ID4SCS_W2N_DIR = ID4SCS_DIR / "wide_to_narrow"
    ID4SCS_N2W_DIR = ID4SCS_DIR / "narrow_to_wide"
    C2U_DIR = RESULTS_DIR / "ConstrainedToUnconstrained"
    OVERVIEW_DIR = RESULTS_DIR / "overview"
    BUCKET_LAW_DIR = {
        "steering": STEERING_DIR,
        "id4scs_w2n": ID4SCS_W2N_DIR,
        "id4scs_n2w": ID4SCS_N2W_DIR,
        "fitts": FITTS_DIR,
        "c2u": C2U_DIR,
    }


def _resolve_pid_config(config_dir, pid, seed):
    """Per-participant config lookup in --config-dir: fitted-config naming
    first, then plain {pid}.json, then a shared default.json."""
    for name in (f"{pid}_gam_config_s{seed}.json", f"{pid}.json", "default.json"):
        cand = Path(config_dir) / name
        if cand.exists():
            return cand
    return None


# ---------------------------------------------------------------------------
# Bucketing (field-presence driven — see plan: trial_ids 1-5 have no
# "tunnelType" key at all, so a naive tunnelType string match would drop them)
# ---------------------------------------------------------------------------

def bucket_condition(cond):
    """Returns one of: 'fitts', 'steering', 'id4scs_w2n', 'id4scs_n2w', 'c2u',
    'excluded'. c2u (constrained_to_unconstrained) is simulated and reported
    but takes part in NO law regression and NO fitting stage — it is the
    zero-free-parameter transfer test for the tunnel->free-space transition."""
    ttype = cond.get("tunnelType")
    if ttype == EXCLUDED_TUNNEL_TYPE:
        return "c2u"
    has_seg = "segment1Width" in cond and "segment2Width" in cond
    has_dist = "distance" in cond
    if has_seg and has_dist:
        return "c2u"  # belt-and-suspenders: constrained_to_unconstrained also has both
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

def load_trials_by_participant(trial_ids, data_dir=None):
    """
    Returns {pid: {tid: {round: {trajectory, speeds, timestamps,
                                  completion_time, condition}}}}
    """
    data_dir = Path(data_dir) if data_dir is not None else HUMAN_DATA_DIR
    trial_ids_set = set(trial_ids)
    all_data = {}

    for fpath in sorted(data_dir.glob("*.json")):
        try:
            with open(fpath) as f:
                data = json.load(f)

            pid = data.get("participantId", fpath.stem)
            sessions = data.get("sessions", [])
            if not sessions:
                td_arr = data.get("trialData", [])
                sessions = [{"trialData": td_arr}] if td_arr else []
            if not sessions:
                continue  # e.g. participants_index.json — not a participant file

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


def pointing_target_center(human_round):
    """Nominal target centre of a pointing round: start + (distance, y-offset
    of the round's own targetPosition) — the same rule the web experiment
    uses (experiment.environment.POINTING_Y_OFFSET). targetPosition varies
    across participants for the same trial_id, so this must be per round.
    Falls back to the recorded end point when the condition lacks the fields."""
    traj = human_round["trajectory"]
    cond = human_round.get("condition", {}) or {}
    start = traj[0]
    if "distance" in cond:
        off = POINTING_Y_OFFSET.get(cond.get("targetPosition", "middle"), 0.0)
        return [float(start[0] + cond["distance"]), float(start[1] + off)]
    return [float(traj[-1][0]), float(traj[-1][1])]


# ---------------------------------------------------------------------------
# Human/model time alignment for pointing
#   onset      : last sample before the cursor has moved > ONSET_DISP_M from
#                its start (strips the per-round reaction latency)
#   final entry: first sample of the last uninterrupted stay inside the target
#   MT_kin     : final entry - onset  (kinematic movement time, no click/dwell)
#   click_lat  : end - final entry    (human: click latency; model: dwell)
# ---------------------------------------------------------------------------
ONSET_DISP_M = 0.002


def movement_onset_time(traj, times):
    p0 = np.asarray(traj[0], dtype=float)
    disp = np.linalg.norm(np.asarray(traj, dtype=float) - p0, axis=1)
    idx = np.argmax(disp > ONSET_DISP_M) if (disp > ONSET_DISP_M).any() else 0
    return float(times[max(idx - 1, 0)])


def final_entry_time(traj, times, center, radius):
    d = np.linalg.norm(np.asarray(traj, dtype=float) - np.asarray(center, dtype=float), axis=1)
    r_eff = max(float(radius), float(d[-1]) * 1.001)   # click/end registered just outside -> tolerate
    outside = np.where(d > r_eff)[0]
    if len(outside) == 0:
        return float(times[0])
    idx = outside[-1] + 1
    return float(times[min(idx, len(times) - 1)])


def align_round(traj, times, center, radius):
    """Returns dict(onset_s, final_entry_s, mt_kin_s, end_lat_s, total_s)."""
    t0 = float(times[0])
    onset = movement_onset_time(traj, times) - t0
    fe = final_entry_time(traj, times, center, radius) - t0
    total = float(times[-1]) - t0
    return {"onset_s": onset, "final_entry_s": fe, "mt_kin_s": max(fe - onset, 0.0),
            "end_lat_s": max(total - fe, 0.0), "total_s": total}


def build_c2u_task_config(cond, start, max_steps=800):
    """Constrained->unconstrained hybrid: a straight tunnel of width
    segment1Width from `start` to the recorded transition x, then
    unconstrained pointing to the round's nominal target (the same
    start + (distance, POINTING_Y_OFFSET[targetPosition]) rule as the
    pointing bucket). The sim constraint gives the free leg the bypass
    width, so clearance saturates there and the planner's free-space
    objective takes over node-by-node as its horizon crosses the exit.
    Returns (task_config, centerline, display_width_profile, trans, target);
    the display profile draws the free leg as a target-diameter corridor
    instead of the 10 m bypass so plots stay readable."""
    W1 = float(cond["segment1Width"])
    R = float(cond["targetRadius"])
    tp = cond.get("transition_point", {}) or {}
    trans = [float(tp.get("taskX", start[0] + float(cond.get("distance", 0.3)) * 0.6)),
             float(start[1])]
    off = POINTING_Y_OFFSET.get(cond.get("targetPosition", "middle"), 0.0)
    target = [float(start[0] + float(cond["distance"])), float(start[1] + off)]

    def _seg(a, b, step=0.004):
        n = max(2, int(np.ceil(math.hypot(b[0] - a[0], b[1] - a[1]) / step)))
        return [[a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
                for t in np.linspace(0.0, 1.0, n, endpoint=False)]

    seg1 = _seg(list(start), trans)
    seg2 = _seg(trans, target)
    centerline = seg1 + seg2 + [list(target)]
    width_sim = [W1] * len(seg1) + [BYPASS_TUNNEL_WIDTH_M] * (len(seg2) + 1)
    width_disp = [W1] * len(seg1) + [max(2.0 * R, 0.01)] * (len(seg2) + 1)
    environment = {
        "env_type": "tunnel_steering_smooth",
        "screen_width": 460, "screen_height": 260,
        "window_width_m": 0.46, "window_height_m": 0.26,
        "centerline": centerline, "tunnel_width": W1, "width_profile": width_sim,
        "target_radius": R, "max_steps": max_steps,
    }
    task_config = generate_task_config(environment, include_constraints=True)
    return task_config, centerline, width_disp, trans, target


def build_fitts_bypass_config(human_round, target_radius, max_steps=800):
    """MPCC-based unconstrained-pointing task: a straight, very wide
    ('bypass') tunnel from a human round's recorded start point to the
    round's nominal target centre, so generate_trajectory_with_waypoints has
    a (non-binding) corridor to track against. Returns (task_config,
    centerline, tunnel_width)."""
    traj = human_round["trajectory"]
    start_m, end_m = traj[0], pointing_target_center(human_round)
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
                         config_path_str, tid_geometry, simulate, human_only=False,
                         min_runs=0):
    """Prepare a picklable job dict for one (pid, tid) condition.

    min_runs > len(human_rounds) simulates extra model runs per condition
    (cycling over the human rounds' geometry) so aggregate statistics and
    regressions average over more noise realisations than the human round
    count — the per-condition simulator stream is deterministic under the
    config's random_seed, so single-draw comparisons conflate realisation
    noise with model differences.
    """
    law_dir = BUCKET_LAW_DIR[bucket]
    n_target = max(len(human_rounds), int(min_runs))
    n_needed = (n_target - len(cached_records)) if (simulate and not human_only) else 0
    n_needed = max(n_needed, 0)

    job = {
        "pid": pid, "tid": tid, "bucket": bucket, "cond": cond, "n_model_runs": n_target,
        "human_rounds": human_rounds, "cached_records": cached_records,
        "n_needed": n_needed, "config_path": config_path_str,
        "law_dir": str(law_dir), "human_only": human_only,
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
        starts = [r["trajectory"][0] for r in human_rounds]
        centers = [pointing_target_center(r) for r in human_rounds]
        start_mean = [float(np.mean([s[0] for s in starts])), float(np.mean([s[1] for s in starts]))]
        end_mean = [float(np.mean([e[0] for e in centers])), float(np.mean([e[1] for e in centers]))]
        job.update({
            "canonical_path": [start_mean, end_mean],
            "folder_name": f"trial_{tid}",   # targetPosition varies per round/participant
        })
    elif bucket == "c2u":
        # Geometry is per participant (targetPosition / transition point vary
        # by pid for the same tid), so build it from this pid's own rounds.
        rcond = (human_rounds[0].get("condition") or cond)
        starts = [r["trajectory"][0] for r in human_rounds]
        start_mean = [float(np.mean([s[0] for s in starts])), float(np.mean([s[1] for s in starts]))]
        task_config, centerline, width_disp, trans, target = build_c2u_task_config(rcond, start_mean)
        job["cond"] = rcond
        job.update({
            "task_config": task_config, "centerline": centerline,
            "tunnel_width": width_disp, "folder_name": f"trial_{tid}",
            "trans_x": trans[0], "target": target,
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
    elif bucket == "fitts":
        target_radius = cond["targetRadius"]
        if n_needed > 0:
            sim = CursorSimulator(job["config_path"])
            for i in range(n_needed):
                round_idx = (len(cached) + i) % len(human_rounds)
                task_config, _centerline, _width = build_fitts_bypass_config(
                    human_rounds[round_idx], target_radius
                )
                new_records.extend(run_tunnel_simulator(sim, task_config, target_radius, n_runs=1))
        tunnel_path = job["canonical_path"]
        tunnel_width = BYPASS_TUNNEL_WIDTH_M
    elif bucket == "c2u":
        target_radius = float(cond.get("targetRadius", 0.01))
        if n_needed > 0:
            sim = CursorSimulator(job["config_path"])
            new_records = run_tunnel_simulator(sim, job["task_config"], target_radius, n_needed)
        tunnel_path = job["centerline"]
        tunnel_width = job["tunnel_width"]
    else:
        raise ValueError(f"unknown bucket {bucket!r}")

    model_records = (cached + new_records)[: job.get("n_model_runs", len(human_rounds))]
    if not model_records and not job.get("human_only"):
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
            rows.append({"source": "Human", "pid": pid, "tid": tid, "ID": ID, "MT_s": r["completion_time"]})
        for r in model_records:
            rows.append({"source": "Simulator", "pid": pid, "tid": tid, "ID": ID, "MT_s": r["completion_time"],
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

    elif bucket == "fitts":
        def _fitts_row(source, mt, human_r, al, timed_out=None):
            center = pointing_target_center(human_r)
            s0 = human_r["trajectory"][0]
            D = math.hypot(center[0] - s0[0], center[1] - s0[1])
            ID = law_stats.fitts_id(D, target_radius)
            mtk = al["mt_kin_s"]
            row = {"source": source, "pid": pid, "tid": tid, "ID": ID, "MT_s": mt, "MT_kin_s": mtk,
                   "TP": ID / mt if mt and mt > 0 else None,
                   "TP_kin": ID / mtk if mtk and mtk > 0 else None,
                   "onset_s": al["onset_s"], "final_entry_s": al["final_entry_s"],
                   "end_lat_s": al["end_lat_s"], "D_m": D,
                   "target_position": (human_r.get("condition") or {}).get("targetPosition", "")}
            if timed_out is not None:
                row["timed_out"] = timed_out
            return row

        h_al, m_al = [], []
        for r in human_rounds:
            ts = [(t - r["timestamps"][0]) / 1000.0 for t in r["timestamps"]]
            al = align_round(r["trajectory"], ts, pointing_target_center(r), target_radius)
            h_al.append(al)
            rows.append(_fitts_row("Human", r["completion_time"], r, al))
        interval = 0.05
        import itertools
        for human_r, model_r in zip(itertools.cycle(human_rounds), model_records):
            mtraj = model_r["trajectory"]
            ts = [i * interval for i in range(len(mtraj))]
            al = align_round(mtraj, ts, pointing_target_center(human_r), target_radius)
            m_al.append(al)
            rows.append(_fitts_row("Simulator", model_r["completion_time"], human_r, al,
                                    model_r.get("timed_out", False)))

        model_tps = [r["TP"] for r in rows if r["source"] == "Simulator" and r["TP"] is not None]
        cond_summary_row.update({
            "target_position": "/".join(sorted({(r.get("condition") or {}).get("targetPosition", "?") for r in human_rounds})),
            "target_radius_mm": round(target_radius * 1000, 2),
            "throughput_mean_bps": round(float(np.mean(model_tps)), 4) if model_tps else None,
            "human_onset_mean_s": round(float(np.mean([a["onset_s"] for a in h_al])), 3),
            "human_click_lat_mean_s": round(float(np.mean([a["end_lat_s"] for a in h_al])), 3),
            "human_mt_kin_mean_s": round(float(np.mean([a["mt_kin_s"] for a in h_al])), 3),
            "model_mt_kin_mean_s": round(float(np.mean([a["mt_kin_s"] for a in m_al])), 3) if m_al else None,
            "model_dwell_mean_s": round(float(np.mean([a["end_lat_s"] for a in m_al])), 3) if m_al else None,
        })

    elif bucket == "c2u":
        target = job["target"]
        trans_x = job["trans_x"]

        def _c2u_row(source, mt, traj, times, speeds, timed_out=None):
            al = align_round(traj, times, target, target_radius)
            k = next((i for i, p in enumerate(traj) if p[0] >= trans_x), None)
            v_exit = float(speeds[k]) if (speeds and k is not None and k < len(speeds)) else None
            v_peak_free = float(max(speeds[k:])) if (speeds and k is not None and k < len(speeds)) else None
            row = {"source": source, "pid": pid, "tid": tid,
                   "W1_mm": round(float(cond.get("segment1Width", 0)) * 1000, 1),
                   "R_mm": round(target_radius * 1000, 1),
                   "target_position": cond.get("targetPosition", ""),
                   "MT_s": mt, "MT_kin_s": al["mt_kin_s"],
                   "v_exit": v_exit, "v_peak_free": v_peak_free}
            if timed_out is not None:
                row["timed_out"] = timed_out
            return row

        h_rows, m_rows = [], []
        for r in human_rounds:
            ts = [(t - r["timestamps"][0]) / 1000.0 for t in r["timestamps"]]
            h_rows.append(_c2u_row("Human", r["completion_time"], r["trajectory"], ts, r["speeds"]))
        interval = 0.05
        for model_r in model_records:
            mtraj = model_r["trajectory"]
            ts = [i * interval for i in range(len(mtraj))]
            m_rows.append(_c2u_row("Simulator", model_r["completion_time"], mtraj, ts,
                                    model_r.get("speeds"), model_r.get("timed_out", False)))
        rows.extend(h_rows); rows.extend(m_rows)

        def _mean(rs, k):
            v = [r[k] for r in rs if r.get(k) is not None]
            return float(np.mean(v)) if v else None
        cond_summary_row.update({
            "target_position": cond.get("targetPosition", "unknown"),
            "segment1Width_mm": round(float(cond.get("segment1Width", 0)) * 1000, 2),
            "target_radius_mm": round(target_radius * 1000, 2),
            "human_mt_kin_mean_s": round(_mean(h_rows, "MT_kin_s"), 3) if h_rows else None,
            "model_mt_kin_mean_s": round(_mean(m_rows, "MT_kin_s"), 3) if m_rows else None,
            "human_v_exit": _mean(h_rows, "v_exit"), "model_v_exit": _mean(m_rows, "v_exit"),
            "human_v_peak_free": _mean(h_rows, "v_peak_free"),
            "model_v_peak_free": _mean(m_rows, "v_peak_free"),
        })

    return {
        "tid": tid, "bucket": bucket, "new_records": new_records,
        "rows": rows, "condition_summary_row": cond_summary_row, "skipped": False,
    }


def process_participant(pid, participant_data, config_path_str, tid_to_condition, tid_to_bucket,
                         tid_geometry, sim_cache, simulate=True, human_only=False,
                         min_runs=0):
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
        n_target = max(len(human_rounds), int(min_runs)) if not human_only else len(human_rounds)
        n_needed = max(n_target - len(cached), 0) if (simulate and not human_only) else 0
        if human_only:
            print(f"    tid={tid} ({bucket}): {len(human_rounds)} human round(s) [--human-only]", flush=True)
        elif n_needed > 0:
            print(f"    tid={tid} ({bucket}): cache has {len(cached)}/{len(human_rounds)} "
                  f"runs, simulating {n_needed} more", flush=True)
        else:
            print(f"    tid={tid} ({bucket}): cache hit, reusing {min(len(cached), len(human_rounds))} "
                  f"cached run(s)" + ("" if simulate else " [--aggregate-only]"), flush=True)

        jobs[tid] = build_condition_job(
            pid, tid, bucket, cond, human_rounds, cached, config_path_str, tid_geometry, simulate,
            human_only=human_only, min_runs=min_runs,
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

    # Averaged-across-participants points, same intent as Steering/Fitts, but
    # collapsed only to one point per (participant, tid) — i.e. averaging away
    # each participant's within-condition repetitions — not all the way down
    # to one point per tid. With only 3 distinct trial_ids for ID4SCS, a full
    # per-tid collapse leaves 3 points, below fit_id4scs's minimum of 5 to fit
    # a,b,c,d at all. condition_summaries already has exactly this granularity
    # (one row per participant x tid); S1/C/S2 recomputed here with the same
    # formula _mk_row uses above (W1/W2 in metres).
    def _agg_points(mt_key):
        pts = []
        for r in condition_summaries:
            if r.get("n_timed_out", 0) not in (0, "0"):
                continue
            W1, W2 = r["W1_mm"] / 1000.0, r["W2_mm"] / 1000.0
            A1, A2 = r["A1_m"], r["A2_m"]
            s1 = (A1 - 5 * W1) / W1 if W1 > 0 else 0.0
            c = math.log2(5 * W1 / W2 + 1) if W2 > 0 else 0.0
            s2 = A2 / W2 if W2 > 0 else 0.0
            pts.append({"S1": s1, "C": c, "S2": s2, "MT": r[mt_key]})
        return pts

    for source, tag, mt_key in (("Human", "human", "human_time_mean_s"), ("Simulator", "model", "model_time_mean_s")):
        sub_rows = _agg_points(mt_key)
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
                  f"only {len(sub_rows)} rows (need >= 5) after averaging within participants")


def write_fitts_outputs(rows, condition_summaries):
    from collections import defaultdict as dd

    FITTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_dict_rows_csv(rows, FITTS_DIR / "fitts_results.csv")
    _write_dict_rows_csv(sorted(condition_summaries, key=lambda r: r["tid"]),
                          FITTS_DIR / "fitts_condition_summary.csv")

    def _fit(rows_sub, mt_key="MT_s", tp_key="TP"):
        sub = [r for r in rows_sub if r.get(mt_key) is not None and not r.get("timed_out")]
        if len(sub) < 3:
            return {}
        ids = np.array([r["ID"] for r in sub])
        mts = np.array([r[mt_key] for r in sub])
        tps = np.array([r[tp_key] for r in sub if r.get(tp_key) is not None])
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

    def _collapse_by_tid(rows_sub, value_keys):
        """One point per trial condition, averaged across ALL participants —
        matches the Steering Law plot's methodology (and the original paper's
        Fig. 3: "each point represents one task condition averaged across
        participants"). Flat pooled mean per key (not a per-participant-first
        hierarchical mean) — identical to steering_law_plot's mechanism."""
        groups = dd(lambda: dd(list))
        for r in rows_sub:
            for k in value_keys:
                if r.get(k) is not None:
                    groups[r["tid"]][k].append(r[k])
        return [
            {"tid": tid, **{k: float(np.mean(v)) for k, v in vals.items() if v}}
            for tid, vals in groups.items()
        ]

    clean_rows = _exclude_timed_out(rows)
    model_rows = [r for r in clean_rows if r["source"] == "Simulator"]
    human_rows = [r for r in clean_rows if r["source"] == "Human"]

    # Collapse to one point per trial_id, averaged across participants, before
    # fitting/plotting — raw per-round rows (fitts_results.csv) are unaffected.
    model_rows_agg = _collapse_by_tid(model_rows, ["ID", "MT_s", "TP"])
    human_rows_agg = _collapse_by_tid(human_rows, ["ID", "MT_s", "TP"])
    model_rows_kin_agg = _collapse_by_tid(model_rows, ["ID", "MT_kin_s", "TP_kin"])
    human_rows_kin_agg = _collapse_by_tid(human_rows, ["ID", "MT_kin_s", "TP_kin"])

    model_reg = _fit(model_rows_agg)
    human_reg = _fit(human_rows_agg)
    # Aligned (kinematic) MT: onset -> final target entry, for both sides.
    model_reg_kin = _fit(model_rows_kin_agg, "MT_kin_s", "TP_kin")
    human_reg_kin = _fit(human_rows_kin_agg, "MT_kin_s", "TP_kin")

    law_plot.plot_fitts_regression(
        model_rows_agg, human_rows_agg, model_reg, human_reg, FITTS_DIR / "fitts_regression_plot_raw.png"
    )
    def _as_kin(rs):
        return [dict(r, MT_s=r.get("MT_kin_s"), TP=r.get("TP_kin")) for r in rs]
    law_plot.plot_fitts_regression(
        _as_kin(model_rows_kin_agg), _as_kin(human_rows_kin_agg), model_reg_kin, human_reg_kin,
        FITTS_DIR / "fitts_regression_plot.png"
    )
    h_on = [r["onset_s"] for r in human_rows if r.get("onset_s") is not None]
    h_cl = [r["end_lat_s"] for r in human_rows if r.get("end_lat_s") is not None]
    m_dw = [r["end_lat_s"] for r in model_rows if r.get("end_lat_s") is not None]

    n_excluded = sum(1 for r in rows if r["source"] == "Simulator" and r.get("timed_out"))
    regression_out = {
        "notes": {
            "ID_formula": "log2(D/(2R) + 1)  [Shannon; 2R = target width]",
            "D_formula": "Euclidean distance from each round's recorded start point to the "
                         "round's nominal target centre (start + (distance, y-offset[targetPosition]))",
            "MT_model": "MT = a + b * ID",
            "aligned": "MT_kin = final target entry - movement onset (>2 mm displacement), "
                       "both sides; strips human reaction latency and click latency, and the "
                       "model's dwell. 'raw' uses human click time vs model time incl. dwell.",
            "aggregation": "Regression/plot fit on one point per trial_id, averaged across all "
                           "participants (flat pooled mean, not a per-participant-first hierarchical "
                           "mean) — matches the Steering Law plot's methodology and the original "
                           "paper's Fig. 3 ('each point represents one task condition averaged "
                           "across participants'). fitts_results.csv retains every raw per-round row.",
            "timed_out_runs_excluded": n_excluded,
        },
        "aligned": {"model": model_reg_kin, "human": human_reg_kin},
        "raw": {"model": model_reg, "human": human_reg},
        "latencies": {
            "human_onset_s": {"mean": round(float(np.mean(h_on)), 3), "median": round(float(np.median(h_on)), 3),
                              "sd": round(float(np.std(h_on)), 3)} if h_on else None,
            "human_click_lat_s": {"mean": round(float(np.mean(h_cl)), 3), "median": round(float(np.median(h_cl)), 3),
                                  "sd": round(float(np.std(h_cl)), 3)} if h_cl else None,
            "model_dwell_s": {"mean": round(float(np.mean(m_dw)), 3)} if m_dw else None,
        },
        # backwards-compatible top-level keys = raw
        "model": model_reg,
        "human": human_reg,
    }
    reg_path = FITTS_DIR / "fitts_regression.json"
    with open(reg_path, "w") as f:
        json.dump(regression_out, f, indent=2)
    print(f"  Saved: {reg_path}")
    if human_reg_kin and model_reg_kin:
        print(f"  Fitts (aligned MT_kin): human MT={human_reg_kin['a_intercept']:.3f}+{human_reg_kin['b_slope_s_per_bit']:.3f}*ID "
              f"(R2={human_reg_kin['r_squared']:.2f}) | model MT={model_reg_kin['a_intercept']:.3f}+{model_reg_kin['b_slope_s_per_bit']:.3f}*ID "
              f"(R2={model_reg_kin['r_squared']:.2f})")
    if h_on:
        print(f"  Human onset latency: mean {np.mean(h_on):.2f}s (sd {np.std(h_on):.2f}); "
              f"click latency: mean {np.mean(h_cl):.2f}s (sd {np.std(h_cl):.2f}); model dwell: {np.mean(m_dw) if m_dw else float('nan'):.2f}s")
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
# Human-only QC outputs: per-participant overview grid + per-round QC table
# ---------------------------------------------------------------------------

_BUCKET_SHORT = {"steering": "steer", "id4scs_w2n": "W2N", "id4scs_n2w": "N2W",
                 "fitts": "fitts", "c2u": "C2U", "excluded": "excl"}


def _tid_display_geometry(tid, bucket, cond, human_rounds, tid_geometry):
    """(centerline, width_or_profile, target_center, target_radius) for drawing."""
    if bucket == "steering":
        _cfg, centerline = tid_geometry[tid]
        return centerline, cond["tunnelWidth"], None, cond.get("targetRadius")
    if bucket in ("id4scs_w2n", "id4scs_n2w"):
        _cfg, centerline = tid_geometry[tid]
        return centerline, make_width_profile(centerline, cond["segment1Width"], cond["segment2Width"]), None, None
    starts = np.array([r["trajectory"][0] for r in human_rounds], dtype=float)
    ends = np.array([r["trajectory"][-1] for r in human_rounds], dtype=float)
    start_mean, end_mean = starts.mean(axis=0).tolist(), ends.mean(axis=0).tolist()
    if bucket == "fitts":
        return [start_mean, end_mean], None, end_mean, cond.get("targetRadius")
    # excluded / constrained_to_unconstrained
    tp = cond.get("transition_point", {}) or {}
    trans_x = float(tp.get("taskX", start_mean[0] + cond.get("distance", 0.0) * 0.6))
    return [start_mean, [trans_x, start_mean[1]]], cond.get("segment1Width", 0.02), end_mean, cond.get("targetRadius")


def _dist_to_polyline(points, centerline):
    """Per-point (distance to nearest centerline segment, index of that segment)."""
    P = np.asarray(points, dtype=float)
    C = np.asarray(centerline, dtype=float)
    A, B = C[:-1], C[1:]
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(axis=1), 1e-12)
    d = np.empty((len(P), len(A)))
    for j in range(len(A)):
        t = np.clip(((P - A[j]) @ AB[j]) / L2[j], 0.0, 1.0)
        proj = A[j] + t[:, None] * AB[j]
        d[:, j] = np.linalg.norm(P - proj, axis=1)
    idx = d.argmin(axis=1)
    return d[np.arange(len(P)), idx], idx


def _round_qc_row(pid, tid, bucket, cond, round_num, r, centerline, width):
    traj = np.asarray(r["trajectory"], dtype=float)
    ts = np.asarray(r["timestamps"], dtype=float)
    dts = np.diff(ts) if len(ts) > 1 else np.array([0.0])
    duration_s = float((ts[-1] - ts[0]) / 1000.0) if len(ts) > 1 else 0.0
    path_len = _path_length(traj.tolist())
    straight = float(np.linalg.norm(traj[-1] - traj[0]))
    n_dup = int(np.sum(np.all(np.diff(traj, axis=0) == 0, axis=1))) if len(traj) > 1 else 0
    speeds = np.asarray(r["speeds"], dtype=float)
    row = {
        "pid": pid, "tid": tid, "bucket": _BUCKET_SHORT.get(bucket, bucket), "round": round_num,
        "target_position": cond.get("targetPosition", ""),
        "n_samples": len(traj), "duration_s": round(duration_s, 3),
        "completion_time_s": round(float(r["completion_time"]), 3),
        "path_length_m": round(path_len, 4), "straight_dist_m": round(straight, 4),
        "path_ratio": round(path_len / straight, 3) if straight > 1e-6 else None,
        "median_dt_ms": round(float(np.median(dts)), 1), "max_dt_ms": round(float(dts.max()), 1),
        "n_repeated_points": n_dup,
        "mean_speed_mps": round(float(speeds.mean()), 4) if len(speeds) else None,
        "peak_speed_mps": round(float(speeds.max()), 4) if len(speeds) else None,
        "start_x": round(float(traj[0, 0]), 4), "start_y": round(float(traj[0, 1]), 4),
        "end_x": round(float(traj[-1, 0]), 4), "end_y": round(float(traj[-1, 1]), 4),
        "outside_tunnel_frac": None,
    }
    if bucket in ("steering", "id4scs_w2n", "id4scs_n2w", "c2u") and centerline is not None and len(centerline) >= 2:
        pts = traj
        if bucket == "c2u":  # only the constrained segment (x <= transition) is bounded
            pts = traj[traj[:, 0] <= centerline[-1][0] + 1e-9]
        if len(pts):
            d, seg_idx = _dist_to_polyline(pts, centerline)
            if np.isscalar(width) or width is None:
                half_w = np.full(len(pts), (width or 0.0) / 2.0)
            else:
                wprof = np.asarray(width, dtype=float)
                half_w = wprof[np.minimum(seg_idx, len(wprof) - 1)] / 2.0
            row["outside_tunnel_frac"] = round(float(np.mean(d > half_w)), 3)
    flags = []
    if row["n_samples"] < 10:
        flags.append("few_samples")
    if row["max_dt_ms"] > 250:
        flags.append("sampling_gap")
    if row["duration_s"] > 20:
        flags.append("long_duration")
    if abs(row["duration_s"] - row["completion_time_s"]) > 0.5:
        flags.append("ct_mismatch")
    if row["path_ratio"] is not None and row["path_ratio"] > 3:
        flags.append("wandering")
    if row["outside_tunnel_frac"] is not None and row["outside_tunnel_frac"] > 0.2:
        flags.append("outside_tunnel")
    if len(traj) > 1 and n_dup > 0.5 * (len(traj) - 1):
        flags.append("mostly_static")
    row["flags"] = "|".join(flags)
    return row


def write_participant_overview(pid, participant_data, tid_to_condition, tid_to_bucket, tid_geometry,
                               out_dir, ncols=6):
    """One panel per trial_id (all buckets, incl. constrained_to_unconstrained)
    with the tunnel/target geometry and every human round overlaid. Returns QC rows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    tids = sorted(participant_data.keys())
    qc_rows = []
    if not tids:
        return qc_rows
    nrows = math.ceil(len(tids) / ncols)
    # Fixed task-window view so every panel is directly comparable (task space is
    # ~0.46 m x 0.26 m; a few trajectories poke slightly outside, hence the margins).
    XLIM, YLIM = (-0.02, 0.50), (-0.02, 0.28)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.55 * nrows), squeeze=False)
    round_colors = plt.cm.tab10.colors

    for k, tid in enumerate(tids):
        ax = axes[k // ncols][k % ncols]
        bucket = tid_to_bucket.get(tid, "excluded")
        cond = tid_to_condition[tid]
        rounds_dict = participant_data[tid]
        round_nums = sorted(rounds_dict.keys())
        human_rounds = [rounds_dict[rn] for rn in round_nums]
        centerline, width, target_c, target_r = _tid_display_geometry(tid, bucket, cond, human_rounds, tid_geometry)

        if centerline is not None and width is not None and len(centerline) >= 2:
            lb, rb = generateTunnelBoundaries(centerline, width)
            ax.fill(np.concatenate([lb[:, 0], rb[::-1, 0]]), np.concatenate([lb[:, 1], rb[::-1, 1]]),
                    color="lightgray", zorder=1)
        elif centerline is not None:
            cl = np.asarray(centerline)
            ax.plot(cl[:, 0], cl[:, 1], color="gray", linestyle=":", linewidth=0.8, zorder=1)
        if target_c is not None and target_r:
            ax.add_patch(Circle(target_c, target_r, facecolor="salmon", edgecolor="red", alpha=0.5, zorder=2))

        cts = []
        for i, (rn, r) in enumerate(zip(round_nums, human_rounds)):
            t = np.asarray(r["trajectory"], dtype=float)
            col = round_colors[i % len(round_colors)]
            ax.plot(t[:, 0], t[:, 1], color=col, linewidth=0.8, alpha=0.9, zorder=10, label=f"r{rn}")
            ax.scatter(t[0, 0], t[0, 1], color="green", s=12, zorder=11)
            ax.scatter(t[-1, 0], t[-1, 1], color=col, edgecolor="black", s=14, zorder=11)
            cts.append(r["completion_time"])
            qc_rows.append(_round_qc_row(pid, tid, bucket, cond, rn, r, centerline, width))

        desc = (cond.get("description", "")
                .replace("constrained-to-unconstrained, ", "")
                .replace("unconstrained pointing, ", "")
                .replace("distance ", "D=").replace("radius ", "R=").replace("width ", "W="))
        ct_str = "/".join(f"{c:.1f}" for c in cts)
        ax.set_title(f"t{tid} [{_BUCKET_SHORT.get(bucket, bucket)}] {desc}   CT={ct_str}s", fontsize=6.5)
        ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=5)
        ax.grid(alpha=0.2)

    for k in range(len(tids), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle(f"Participant {pid} — human trajectories, all trials (green=start, dot=end)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"participant_{pid}_overview.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return qc_rows

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
    parser.add_argument("--human-only", action="store_true", default=False,
                        help="Never call the simulator. Plot human trajectories/speeds per trial "
                             "(incl. constrained_to_unconstrained trials), write a per-participant "
                             "overview grid and a per-round QC CSV under results/overview/.")
    parser.add_argument("--data-dir", type=str, default=str(HUMAN_DATA_DIR),
                        help="Directory of per-participant human data JSON files")
    parser.add_argument("--buckets", type=str, nargs="+", default=None,
                        choices=["steering", "id4scs_w2n", "id4scs_n2w", "fitts", "c2u"],
                        help="Restrict processing to these task buckets (default: all)")
    parser.add_argument("--per-participant", action="store_true", default=False,
                        help="Use each participant's fitted persona from "
                             "eval/model_fitting/results/{pid}_gam_config_s{seed}.json "
                             "(participants without a fitted config are skipped)")
    parser.add_argument("--seed", type=int, default=42, help="seed suffix of fitted configs (--per-participant)")
    parser.add_argument("--config-dir", type=str, default=None,
                        help="Directory of per-participant simulator configs. Resolution per "
                             "pid: {pid}_gam_config_s{seed}.json, {pid}.json, default.json "
                             "(participants without a match are skipped). Overrides --config "
                             "and --per-participant.")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Write all outputs (and the sim cache) under this directory "
                             "instead of results/ — relative paths resolve against the "
                             "script dir, so model variants can be compared side by side. "
                             "Takes precedence over HCS_EVAL_RESULTS_DIR.")
    parser.add_argument("--min-runs", type=int, default=0,
                        help="Simulate at least this many model runs per condition (cycling "
                             "the human rounds' geometry) so statistics average over more "
                             "noise realisations — per-condition streams are deterministic "
                             "under the config's random_seed, so single draws conflate "
                             "realisation noise with model differences. Default: one run "
                             "per human round.")
    parser.add_argument("--fresh-sim", action="store_true", default=False,
                        help="Ignore any cached simulator runs and resimulate every condition "
                             "from scratch. The simulator applies stochastic per-step motor/"
                             "device noise, so this produces a different random draw than "
                             "whatever is currently cached, not just a repeat of it.")
    args = parser.parse_args()

    config_path = Path(args.config)
    human_only = args.human_only
    if args.results_dir:
        rd = Path(args.results_dir)
        _set_results_dir(rd if rd.is_absolute() else SCRIPT_DIR / rd)
    config_dir = Path(args.config_dir) if args.config_dir else None
    if config_dir is not None and not config_dir.is_dir():
        print(f"Config dir not found: {config_dir}")
        return
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"Data dir not found: {data_dir}")
        return
    if (not human_only and config_dir is None and not args.per_participant
            and not config_path.exists()):
        print(f"Config not found: {config_path}")
        return

    dirs = [FITTS_DIR, STEERING_DIR, ID4SCS_W2N_DIR, ID4SCS_N2W_DIR, C2U_DIR]
    dirs += [OVERVIEW_DIR] if human_only else [SIM_CACHE_DIR]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Eval Main: ID4SCS / Fitts' Law / Steering Law" + ("  [HUMAN-ONLY]" if human_only else ""))
    print("=" * 70)
    print(f"  Data dir: {data_dir}")
    print(f"  Results:  {RESULTS_DIR}")
    if config_dir is not None:
        print(f"  Configs:  per-participant from {config_dir}")
    if human_only:
        print("  Mode:     --human-only (simulator is never called)")
    else:
        print(f"  Config:   {config_path}")

    print("\n[1/4] Scanning trial conditions ...")
    tid_to_condition, tid_to_bucket = scan_conditions(data_dir)
    bucket_counts = Counter(tid_to_bucket.values())
    for b in ("steering", "id4scs_w2n", "id4scs_n2w", "fitts", "c2u", "excluded"):
        print(f"  {b:12s}: {bucket_counts.get(b, 0):3d} trial_ids")
    n_included = sum(v for k, v in bucket_counts.items() if k != "excluded")
    # (c2u counts as included: simulated + reported, but outside every law fit)
    print(f"  Total: {n_included} included, {bucket_counts.get('excluded', 0)} excluded, "
          f"{len(tid_to_bucket)} scanned")

    included_tids = sorted(t for t, b in tid_to_bucket.items()
                           if b != "excluded"
                           and (args.buckets is None or b in args.buckets))

    print("\n[2/4] Loading human data ...")
    all_human_data = load_trials_by_participant(included_tids, data_dir)
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
    simulate = not args.aggregate_only and not human_only
    all_qc_rows = []

    if args.fresh_sim and args.aggregate_only:
        print("  --fresh-sim and --aggregate-only are contradictory; ignoring --aggregate-only.")
        simulate = True

    if human_only:
        print("\n[3/4] Plotting human trajectories (--human-only) ...")
    elif simulate:
        print("\n[3/4] Running simulations ...")
    else:
        print("\n[3/4] Aggregating from cached simulator runs (--aggregate-only) ...")
    fitting_dir = Path(os.environ.get("HCS_FIT_RESULTS_DIR", PROJECT_ROOT / "eval" / "model_fitting" / "results"))
    for pid, participant_data in sorted(all_human_data.items()):
        print(f"\n  Participant: {pid}")
        pid_config_path = config_path
        if config_dir is not None and not human_only:
            pid_config_path = _resolve_pid_config(config_dir, pid, args.seed)
            if pid_config_path is None:
                print(f"    no config for {pid} in {config_dir} — skipped", flush=True)
                continue
            print(f"    persona: {pid_config_path.name}")
        elif args.per_participant and not human_only:
            pid_config_path = fitting_dir / f"{pid}_gam_config_s{args.seed}.json"
            if not pid_config_path.exists():
                print(f"    no fitted config {pid_config_path.name} — skipped", flush=True)
                continue
            print(f"    persona: {pid_config_path.name}")
        cache_path = SIM_CACHE_DIR / f"{pid}_sim_cache.json"
        sim_cache = {} if (args.fresh_sim or human_only) else load_sim_cache(cache_path)
        if args.fresh_sim:
            print("    --fresh-sim: ignoring any existing cache, resimulating every condition "
                  "(stochastic noise means these will differ from the previous cached run)")

        rows_by_bucket, condition_summaries = process_participant(
            pid, participant_data, str(pid_config_path), tid_to_condition, tid_to_bucket, tid_geometry,
            sim_cache, simulate=simulate, human_only=human_only, min_runs=args.min_runs,
        )
        if simulate:
            save_sim_cache(sim_cache, cache_path)
        if human_only:
            all_qc_rows.extend(write_participant_overview(
                pid, participant_data, tid_to_condition, tid_to_bucket, tid_geometry, OVERVIEW_DIR))

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
    if all_rows.get("c2u"):
        _write_dict_rows_csv(all_rows["c2u"], C2U_DIR / "c2u_results.csv")
        _write_dict_rows_csv(sorted(all_condition_summaries["c2u"], key=lambda r: r["tid"]),
                              C2U_DIR / "c2u_condition_summary.csv")
    if human_only:
        qc_path = OVERVIEW_DIR / "human_rounds_qc.csv"
        _write_dict_rows_csv(sorted(all_qc_rows, key=lambda r: (r["pid"], r["tid"], r["round"])), qc_path)
        print(f"  Saved: {qc_path}")
        flagged = [r for r in all_qc_rows if r["flags"]]
        print(f"\n  QC: {len(all_qc_rows)} human rounds, {len(flagged)} flagged")
        flag_counts = Counter(f for r in flagged for f in r["flags"].split("|"))
        for fl, n in flag_counts.most_common():
            print(f"    {fl:16s}: {n}")
        by_pid = Counter(r["pid"] for r in flagged)
        for pid, n in sorted(by_pid.items()):
            print(f"    {pid}: {n} flagged rounds")

    print("\n" + "=" * 70)
    print("Eval complete.")
    print(f"Results: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

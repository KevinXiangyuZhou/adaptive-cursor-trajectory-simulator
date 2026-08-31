"""
Per-participant model fitting — revised 2026-08 for the aug-26-prolific
dataset (steering + ID4SCS + unconstrained pointing) and the current model
(fixed plant, free-space LQR objective, dwell termination).

Pipeline (all simulation noiseless — add_noise=False; the persona's nc is kept so
the goal-precision well, which is scaled by nc^2, stays active during fitting):

  Stage G  (gaze participants only, runs OUTSIDE this script, before it):
           intermittent-control module parameters fitted from the gaze data —
           difficulty-budget lookahead (D0, gamma, T_min;
           eval/eval-gaze-cursor/refit_floor.py) and replan trigger timing
           (tau, cv; intermittency_analysis.py) — baked into
           base_configs_gaze/{pid}.json and held FIXED here. Because the
           budget lookahead determines the prediction horizon, Th is
           excluded from the Stage 2 search under horizon_mode=budget.

  Phase 0  Reference-path params (spatial-only, no simulation).
           Loss: lateral RMSE of human trajectories w.r.t. the generated
           reference path, on the steering training tasks.
  Stage 1  GAM speed model fitted directly from human tunnel data (all
           steering widths incl. straight, plus ID4SCS), using samples in the
           central 10–90 % of progress (cruise) with the features computed on
           the fitted reference path exactly as the simulator does.
  Stage 2  MPCC tunnel weights via CMA-ES (contour, lag, jerk, constraint,
           progress, Th) on the steering training widths {10, 30, 50} mm.
           Loss = human-variability-normalised lateral RMSE + speed-profile
           RMSE + (1 - speed corr) + relative time diff + wall margin.
  Stage 3  Free-space (pointing) LQ weights via CMA-ES (goal, free_velocity;
           jerk fixed from Stage 2; goal_precision stays at the base config's
           value, default 0) on the pointing training radii {5, 15, 25} mm. Loss = normalised |relative MT_kin diff| +
           speed-profile RMSE + (1 - corr) + endpoint-depth diff + lateral
           RMSE, with MT_kin = movement onset -> final target entry on both
           sides (human reaction and click latency are NOT fitted; the model
           dwell_s stands in for click latency).

Held-out evaluation: steering widths {20, 40} mm, ID4SCS (both directions),
pointing radii {10, 20} mm. Constrained->unconstrained trials are not used.

Outputs (eval/model_fitting/results/):
    {pid}_gam_s{seed}.pkl           fitted GAM speed model
    {pid}_gam_config_s{seed}.json   persona config with all fitted params
                                    (loadable by CursorSimulator / eval-main
                                    --per-participant)
    {pid}_gam_fit_s{seed}.json      full record: params, losses, train/test tids

Usage:
    python -m eval.model_fitting.fit_speed_model --pid P6a0aa037f7816b7befeb15e6 --time-limit 43200
    python -m eval.model_fitting.fit_speed_model --pid ... --stages tunnel      # skip pointing
    python -m eval.model_fitting.fit_speed_model --pid ... --stages pointing    # reuse tunnel fit
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
from itertools import combinations
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("MPLBACKEND", "Agg")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "hcs_package" / "src", PROJECT_ROOT / "eval",
          PROJECT_ROOT / "eval" / "eval-main", SCRIPT_DIR):
    sys.path.insert(0, str(p))

import run_eval as em  # eval/eval-main/run_eval.py: data loading, task builders, alignment  # noqa: E402
from hcs_package.cursor_simulator import CursorSimulator  # noqa: E402
from hcs_package.speed_model import GAMSpeedModel  # noqa: E402
from hcs_package.reference_path import ReferencePath, generate_optimal_reference_path  # noqa: E402
from hcs_package.constraints import PathConstraint, RectangleConstraint, PolygonConstraint  # noqa: E402
from hcs_package.constraint_utils import parse_constraints_from_json, convert_constraints_to_corridor_bounds  # noqa: E402
from utils.stats import (  # noqa: E402
    resample_by_progress, resample_speeds_by_progress,
    trajectory_rmse, speed_profile_rmse, speed_profile_correlation,
)
from extract_speed_data import extract_all_speed_data_v2  # noqa: E402  (same dir)

# Overridable per cohort: HCS_HUMAN_DATA_DIR env or --data-dir (e.g. the
# gaze cohort lives in human_data/gaze_cursor_data).
HUMAN_DATA_DIR = Path(os.environ.get("HCS_HUMAN_DATA_DIR", em.HUMAN_DATA_DIR))
DEFAULT_BASE_CONFIG = PROJECT_ROOT / "hcs_package" / "src" / "hcs_package" / "user_configurations" / "office_worker.json"
POPULATION_GAM = DEFAULT_BASE_CONFIG.parent / "population_gam.pkl"
# Override with HCS_FIT_RESULTS_DIR (Great Lakes: .../projects/chi-27/results/model_fitting)
RESULTS_DIR = Path(os.environ.get("HCS_FIT_RESULTS_DIR", SCRIPT_DIR / "results"))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_SIM_STEPS = 600            # 30 s at 50 ms; narrow tunnels can take 15-25 s
N_PROGRESS_BINS = 100
SPEED_TRIM_START, SPEED_TRIM_END = 10, 90   # tunnel speed metrics: central 10-90 % of progress
GAM_PROGRESS_WINDOW = (0.10, 0.90)          # cruise samples for the GAM
INCOMPLETE_PENALTY = 100.0
WALL_MARGIN_WEIGHT = 20.0

TRAIN_WIDTHS = {0.01, 0.03, 0.05}
TEST_WIDTHS = {0.02, 0.04}
POINT_TRAIN_R = {0.005, 0.015, 0.025}
POINT_TEST_R = {0.010, 0.020}

REF_PATH_PARAM_SPEC = [
    {"name": "w_cut",                "bounds": (0.0, 1.0)},
    {"name": "w_suppress",           "bounds": (0.0, 2.0)},
    {"name": "w_width_exp",          "bounds": (0.0, 3.0)},
    {"name": "cut_window_frac",      "bounds": (0.02, 0.20)},
    {"name": "global_clearance_ref", "bounds": (0.005, 0.05)},
]
REF_PATH_KEYS = {s["name"] for s in REF_PATH_PARAM_SPEC}

# Stage 2: tunnel MPCC weights (log10 bounds unless noted)
TUNNEL_PARAM_SPEC = [
    {"name": "contour",    "log_scale": True,  "bounds": (0.5, 2.5)},     # 3 - 316
    {"name": "lag",        "log_scale": True,  "bounds": (-2.0, 1.0)},    # 0.01 - 10
    {"name": "jerk",       "log_scale": True,  "bounds": (-7.0, -4.5)},   # 1e-7 - 3e-5
    {"name": "constraint", "log_scale": True,  "bounds": (1.0, 2.5)},     # 10 - 316
    {"name": "progress",   "log_scale": True,  "bounds": (-8.0, -3.0)},   # 1e-8 - 1e-3
    {"name": "Th",         "log_scale": False, "bounds": (0.2, 0.6), "discrete_step": 0.05,
     "config_key": "top_level"},
]
# Stage 3: free-space LQ weights (jerk fixed from Stage 2 — only ratios matter
# for the LQR). goal_precision is NOT fitted: the well is off by default
# (2026-08-21 decision; seed-averaged ablation showed ~10% slope effect at
# 1e-4 and nothing in steering) — re-add here only if the endpoint-depth
# objective demands it.
POINTING_PARAM_SPEC = [
    {"name": "goal",           "log_scale": True, "bounds": (-1.0, 1.0)},   # 0.1 - 10
    {"name": "free_velocity",  "log_scale": True, "bounds": (-2.5, 0.0)},   # 0.003 - 1
]

# log_time replaces time_diff in the loss (kept in metrics for reporting):
# |log(model_CT / human_CT)| is symmetric in slow/fast misses, unlike the
# relative diff, and its weight is doubled — the 8-24 fits showed a 2-3x CT
# miss on narrow curved tunnels can survive when the equal-weight shape terms
# are satisfied.
TUNNEL_LOSS_WEIGHTS = {"lateral_rmse": 1.0, "speed_rmse": 1.0, "speed_corr": 1.0, "log_time": 2.0}
POINT_LOSS_WEIGHTS = {"mt_rel": 1.0, "speed_rmse": 1.0, "speed_corr": 1.0, "end_depth": 1.0, "lateral_rmse": 1.0}
DEFAULT_TUNNEL_SCALES = {"lateral_rmse": 0.003, "speed_rmse": 0.05, "speed_corr": 0.3, "time_diff": 0.15,
                         "log_time": 0.15}
DEFAULT_POINT_SCALES = {"mt_rel": 0.2, "speed_rmse": 0.1, "speed_corr": 0.2, "end_depth": 0.2, "lateral_rmse": 0.004}


# ---------------------------------------------------------------------------
# Generic [0,1] encoding for a param spec
# ---------------------------------------------------------------------------

def encode(params, spec):
    vec = np.zeros(len(spec))
    for i, s in enumerate(spec):
        val = params[s["name"]]
        lo, hi = s["bounds"]
        if s.get("log_scale"):
            val = math.log10(max(val, 1e-300))
        vec[i] = (val - lo) / (hi - lo)
    return np.clip(vec, 0.0, 1.0)


def decode(vec, spec):
    out = {}
    for i, s in enumerate(spec):
        v = float(np.clip(vec[i], 0.0, 1.0))
        lo, hi = s["bounds"]
        val = lo + v * (hi - lo)
        if s.get("log_scale"):
            val = 10.0 ** val
        if "discrete_step" in s:
            st = s["discrete_step"]
            val = max(lo, min(hi, round(round(val / st) * st, 6)))
        out[s["name"]] = val
    return out


def apply_params(cfg, params):
    """Apply fitted params to a persona config (in place)."""
    for k, v in params.items():
        if k in ("Th", "plan_deadline_s", "plan_vmax"):
            cfg[k] = v          # top-level simulator keys, not planner weights
        elif k in ("D0", "gamma", "T_min"):
            cfg.setdefault("budget", {})[k] = v   # gaze-budget constants
        elif k in REF_PATH_KEYS:
            cfg.setdefault("reference_path", {})[k] = v
        else:
            cfg.setdefault("planner_weights", {})[k] = v


# ---------------------------------------------------------------------------
# Data + tasks
# ---------------------------------------------------------------------------

def load_participant(pid):
    """Returns (rounds_by_tid, tid_to_condition, tid_to_bucket).
    rounds_by_tid: {tid: [round dict with trajectory, speeds, timestamps, completion_time, condition]}"""
    tid_to_condition, tid_to_bucket = em.scan_conditions(HUMAN_DATA_DIR)
    tids = [t for t, b in tid_to_bucket.items() if b in ("steering", "id4scs_w2n", "id4scs_n2w", "fitts")]
    all_data = em.load_trials_by_participant(tids, HUMAN_DATA_DIR)
    if pid not in all_data:
        raise FileNotFoundError(f"No data for participant {pid} in {HUMAN_DATA_DIR}")
    rounds_by_tid = {tid: [d[k] for k in sorted(d)] for tid, d in all_data[pid].items() if d}
    return rounds_by_tid, tid_to_condition, tid_to_bucket


def build_tunnel_tasks(tid_to_condition, tid_to_bucket):
    """{tid: (task_config, centerline, half_width)} for steering + ID4SCS."""
    out = {}
    for tid, b in tid_to_bucket.items():
        cond = tid_to_condition[tid]
        if b == "steering":
            tc, cl = em.build_steering_task_config(cond)
            hw = cond["tunnelWidth"] * 0.5
        elif b in ("id4scs_w2n", "id4scs_n2w"):
            tc, cl = em._build_wide_to_narrow_config(cond["segment1Width"], cond["segment2Width"], cond.get("curvature", 0.0))
            hw = min(cond["segment1Width"], cond["segment2Width"]) * 0.5
        else:
            continue
        tc = dict(tc); tc["max_steps"] = MAX_SIM_STEPS
        out[tid] = (tc, [list(map(float, p)) for p in cl], hw)
    return out


def split_tunnel(rounds_by_tid, tid_to_condition, tid_to_bucket):
    train, test = {}, {}
    for tid, rounds in rounds_by_tid.items():
        b = tid_to_bucket.get(tid)
        if b == "steering":
            w = round(tid_to_condition[tid]["tunnelWidth"], 3)
            (train if w in TRAIN_WIDTHS else test)[tid] = rounds
        elif b in ("id4scs_w2n", "id4scs_n2w"):
            test[tid] = rounds
    return train, test


def split_pointing(rounds_by_tid, tid_to_condition, tid_to_bucket):
    train, test = {}, {}
    for tid, rounds in rounds_by_tid.items():
        if tid_to_bucket.get(tid) != "fitts":
            continue
        R = round(tid_to_condition[tid]["targetRadius"], 4)
        (train if R in POINT_TRAIN_R else test)[tid] = rounds
    return train, test


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def _smooth_speeds(traj, interval):
    n = len(traj)
    raw = []
    for i in range(n):
        if n < 2:
            raw.append(0.0); continue
        if i == 0:
            p0, p1, dt = traj[0], traj[1], interval
        elif i == n - 1:
            p0, p1, dt = traj[-2], traj[-1], interval
        else:
            p0, p1, dt = traj[i - 1], traj[i + 1], 2.0 * interval
        raw.append(math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / dt)
    half = 2
    return [sum(raw[max(0, i - half):min(n, i + half + 1)]) / (min(n, i + half + 1) - max(0, i - half)) for i in range(n)]


def run_single_sim(sim, task_config, target_radius=None):
    """One noiseless-or-not simulation. Returns (traj_m, speeds, interval)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        json.dump(task_config, tf)
        path = tf.name
    try:
        traj_raw = sim.generate_trajectory_with_waypoints(
            task_file=path,
            max_steps=task_config.get("max_steps", MAX_SIM_STEPS),
            target_radius=target_radius if target_radius is not None else task_config.get("target_radius", 0.01),
            use_optimal_path=True,
        )
    finally:
        os.unlink(path)
    traj = [[x * 0.001, y * 0.001] for x, y, _ in traj_raw]
    return traj, _smooth_speeds(traj, sim.interval), sim.interval


def _make_sim(cfg, gam_path=None):
    fd, path = tempfile.mkstemp(suffix=".json", prefix="fit_cfg_")
    os.close(fd)
    with open(path, "w") as f:
        json.dump(cfg, f)
    try:
        sim = CursorSimulator(path)
    finally:
        os.unlink(path)
    if gam_path is not None:
        sim.speed_model = GAMSpeedModel.load(gam_path)
    return sim


def _completion(traj, centerline):
    cl = np.asarray(centerline, float); last = np.asarray(traj[-1], float)
    lens = np.linalg.norm(np.diff(cl, axis=0), axis=1); cum = np.concatenate([[0.0], np.cumsum(lens)])
    if cum[-1] <= 0:
        return 0.0
    best = 0.0
    for i in range(len(cl) - 1):
        seg = cl[i + 1] - cl[i]; l2 = float(seg @ seg)
        t = 0.0 if l2 < 1e-18 else float(np.clip((last - cl[i]) @ seg / l2, 0.0, 1.0))
        best = max(best, cum[i] + t * lens[i])
    return best / cum[-1]


# ---------------------------------------------------------------------------
# Phase 0: reference-path params (spatial only)
# ---------------------------------------------------------------------------

def _waypoints_m(task_config):
    sw, sh = task_config["screen_width"], task_config["screen_height"]
    swm = 0.46; shm = sh / sw * swm
    return [(x / sw * swm, y / sh * shm) for x, y in task_config["waypoints"]]


def _precompute_task_geometry(task_configs, margin=0.0):
    geometry = {}
    for tid, tc in task_configs.items():
        wp = _waypoints_m(tc)
        cc = None; cart = []; width = None
        if tc.get("constraints") is not None:
            cc = parse_constraints_from_json(tc["constraints"])
            for region in cc.regions:
                if isinstance(region.geometry, PathConstraint):
                    w = region.geometry.width
                    width = float(min(w) if isinstance(w, (list, tuple)) else w)
                elif isinstance(region.geometry, (RectangleConstraint, PolygonConstraint)):
                    cart.append(region)
        if width is None:
            width = 0.02
        spline = ReferencePath(wp, s=0.0, k=3)
        cb = convert_constraints_to_corridor_bounds(cc, spline, default_margin=margin) if cc is not None else None
        geometry[tid] = {"waypoints_norm": wp, "tunnel_width": width, "centerline_spline": spline,
                         "corridor_bounds": cb, "cartesian_regions": cart, "margin": margin}
    return geometry


def _build_ref_path_from_geometry(geom, rp):
    ref = generate_optimal_reference_path(
        tunnel_path=geom["waypoints_norm"], tunnel_width=geom["tunnel_width"], margin=geom["margin"],
        w_cut=rp["w_cut"], w_suppress=rp["w_suppress"], w_width_exp=rp["w_width_exp"],
        cut_window_frac=rp["cut_window_frac"], global_clearance_ref=rp["global_clearance_ref"],
        cartesian_constraints=geom["cartesian_regions"] or None, corridor_bounds=geom["corridor_bounds"],
        centerline_cache=geom["centerline_spline"])
    thetas = np.linspace(0, ref.total_length, 200)
    return np.array([ref(t) for t in thetas]), ref


def _eval_ref_path_spatial(vec, train_data, geometry):
    rp = decode(vec, REF_PATH_PARAM_SPEC)
    total, n = 0.0, 0
    for tid in sorted(train_data):
        rounds = train_data[tid]
        if not rounds or tid not in geometry:
            continue
        n += 1
        try:
            poly, _ = _build_ref_path_from_geometry(geometry[tid], rp)
        except Exception:
            total += 1e6; continue
        rm = []
        for h in rounds:
            _, _, lat = resample_by_progress(h["trajectory"], poly, 50)
            rm.append(float(np.sqrt(np.mean(np.asarray(lat) ** 2))))
        total += float(np.mean(rm))
    return total / max(n, 1)


def fit_reference_path_params(train_data, task_configs, base_config, time_limit=120.0, seed=42, sigma0=0.3, popsize=8):
    import cma
    print("\n--- Phase 0: reference-path params (spatial-only) ---", flush=True)
    geometry = _precompute_task_geometry(task_configs)
    rp = base_config.get("reference_path", {}); pw = base_config.get("planner_weights", {})
    sim_defaults = {"w_cut": 0.01, "w_suppress": 0.0, "w_width_exp": 1.0, "cut_window_frac": 0.05, "global_clearance_ref": 0.025}
    init = {s["name"]: rp.get(s["name"], pw.get(s["name"], sim_defaults[s["name"]])) for s in REF_PATH_PARAM_SPEC}
    x0 = encode(init, REF_PATH_PARAM_SPEC)
    obj = lambda x: _eval_ref_path_spatial(x, train_data, geometry)
    best_loss = obj(x0); best_x = x0.copy()
    print(f"  initial loss {best_loss:.6f}  params {init}")
    es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, {"bounds": [[0.0] * len(x0), [1.0] * len(x0)], "popsize": popsize,
                                                        "seed": seed, "maxiter": 300, "verb_disp": 0, "verb_log": 0,
                                                        "verb_filenameprefix": "", "verbose": -9})
    t0 = time.time(); gen = 0; hist = [{"generation": 0, "best_loss": best_loss, "elapsed_sec": 0.0}]
    while not es.stop() and time.time() - t0 < time_limit:
        sols = es.ask(); fit = [obj(x) for x in sols]; es.tell(sols, fit); gen += 1
        i = int(np.argmin(fit))
        if fit[i] < best_loss:
            best_loss, best_x = fit[i], np.array(sols[i]).copy()
        hist.append({"generation": gen, "best_loss": float(best_loss), "mean_loss": float(np.mean(fit)), "elapsed_sec": round(time.time() - t0, 1)})
        if gen % 20 == 0 or gen <= 3:
            print(f"  gen {gen:3d} best {best_loss:.6f} mean {np.mean(fit):.6f} ({time.time() - t0:.0f}s)", flush=True)
    fitted = decode(best_x, REF_PATH_PARAM_SPEC)
    print(f"  Phase 0 done: {gen} generations, {time.time() - t0:.0f}s; fitted {fitted}")
    return fitted, float(best_loss), hist


# ---------------------------------------------------------------------------
# Stage 1: GAM speed model
# ---------------------------------------------------------------------------

def fit_gam_speed_model(tunnel_data, task_configs, ref_params):
    print("\n--- Stage 1: GAM speed model from human tunnel data ---", flush=True)
    t0 = time.time()
    geometry = _precompute_task_geometry(task_configs)
    data = extract_all_speed_data_v2(tunnel_data, geometry, ref_params, _build_ref_path_from_geometry,
                                     progress_window=GAM_PROGRESS_WINDOW)
    if data is None:
        raise ValueError("no speed data extracted")
    valid = data["speed"] > 0.005
    cl, ka, dk, sp = data["clearance"][valid], data["kappa"][valid], data["dkappa_ds"][valid], data["speed"][valid]
    rid = data["round_id"][valid]
    # Equal weight per ROUND, not per sample: observations arrive per
    # timestep, so a slow round of a trial contributes proportionally more
    # samples than a fast round of the same trial (a 15 s round has ~6x the
    # samples of a 2.5 s one) and drags the regression toward the slow tail.
    # This produced narrow-width speed targets ~2-3x below the participant's
    # typical rounds in the 8-24 fits.
    _, inv, counts = np.unique(rid, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inv]
    w *= len(w) / w.sum()   # mean weight 1 for conditioning
    print(f"  {int(valid.sum())} observations from {len(np.unique(data['trial_id']))} tasks "
          f"({len(counts)} rounds, equal-round weighting); "
          f"clearance [{cl.min():.4f}, {cl.max():.4f}] m, kappa [{ka.min():.2f}, {ka.max():.2f}], "
          f"median speed {np.median(sp):.3f} m/s (round-weighted {np.average(sp, weights=w):.3f})")
    gam = GAMSpeedModel(base_speed=float(np.median(sp)), floor=0.01, ceil=0.6)
    gam.fit(cl, ka, dk, sp, sample_weight=w)
    pred = gam.compute_speed_profile(np.arange(len(sp)), cl, ka, dk)
    print(f"  fitted in {time.time() - t0:.1f}s; train corr {np.corrcoef(sp, pred)[0, 1]:.3f}, RMSE {np.sqrt(np.mean((sp - pred) ** 2)):.4f} m/s")
    return gam


# ---------------------------------------------------------------------------
# Stage 2: tunnel MPCC weights
# ---------------------------------------------------------------------------
TUNNEL_SCALES = dict(DEFAULT_TUNNEL_SCALES)
POINT_SCALES = dict(DEFAULT_POINT_SCALES)


def tunnel_metrics(model_traj, model_speeds, human, centerline, sim_interval, half_width):
    _, _, lat_m = resample_by_progress(model_traj, centerline, N_PROGRESS_BINS)
    _, _, lat_h = resample_by_progress(human["trajectory"], centerline, N_PROGRESS_BINS)
    _, spd_m = resample_speeds_by_progress(model_speeds, model_traj, centerline, N_PROGRESS_BINS)
    _, spd_h = resample_speeds_by_progress(human["speeds"], human["trajectory"], centerline, N_PROGRESS_BINS)
    sm, sh = spd_m[SPEED_TRIM_START:SPEED_TRIM_END], spd_h[SPEED_TRIM_START:SPEED_TRIM_END]
    ts = human["timestamps"]
    human_time = (ts[-1] - ts[0]) / 1000.0 if len(ts) >= 2 else 1.0
    model_time = len(model_traj) * sim_interval
    wall = 0.0
    if half_width:
        prox = np.abs(np.asarray(lat_m)) / half_width
        wall = float(np.mean(np.maximum(prox - 0.6, 0.0) ** 2))
    return {"lateral_rmse": trajectory_rmse(lat_m, lat_h), "speed_rmse": speed_profile_rmse(sm, sh),
            "speed_corr": speed_profile_correlation(sm, sh),
            "time_diff": abs(model_time - human_time) / max(human_time, 0.1),
            "log_time": abs(math.log(max(model_time, 0.05) / max(human_time, 0.05))),
            "wall_margin": wall}


def tunnel_loss(m):
    loss = 0.0
    for k in ("lateral_rmse", "speed_rmse", "log_time"):
        loss += TUNNEL_LOSS_WEIGHTS[k] * m[k] / TUNNEL_SCALES[k]
    loss += TUNNEL_LOSS_WEIGHTS["speed_corr"] * (1.0 - m["speed_corr"]) / TUNNEL_SCALES["speed_corr"]
    return loss + WALL_MARGIN_WEIGHT * m.get("wall_margin", 0.0)


def compute_tunnel_scales(train_data, tasks):
    pm = {"lateral_rmse": [], "speed_rmse": [], "speed_corr": [], "time_diff": [], "log_time": []}
    for tid, rounds in train_data.items():
        if len(rounds) < 2 or tid not in tasks:
            continue
        cl = tasks[tid][1]
        for a, b in combinations(rounds, 2):
            _, _, la = resample_by_progress(a["trajectory"], cl, N_PROGRESS_BINS)
            _, _, lb = resample_by_progress(b["trajectory"], cl, N_PROGRESS_BINS)
            _, sa = resample_speeds_by_progress(a["speeds"], a["trajectory"], cl, N_PROGRESS_BINS)
            _, sb = resample_speeds_by_progress(b["speeds"], b["trajectory"], cl, N_PROGRESS_BINS)
            pm["lateral_rmse"].append(trajectory_rmse(la, lb))
            sa, sb = sa[SPEED_TRIM_START:SPEED_TRIM_END], sb[SPEED_TRIM_START:SPEED_TRIM_END]
            pm["speed_rmse"].append(speed_profile_rmse(sa, sb)); pm["speed_corr"].append(1.0 - speed_profile_correlation(sa, sb))
            ta = (a["timestamps"][-1] - a["timestamps"][0]) / 1000.0; tb = (b["timestamps"][-1] - b["timestamps"][0]) / 1000.0
            pm["time_diff"].append(abs(ta - tb) / max(tb, 0.1))
            pm["log_time"].append(abs(math.log(max(ta, 0.05) / max(tb, 0.05))))
    if pm["lateral_rmse"]:
        for k in pm:
            TUNNEL_SCALES[k] = max(float(np.median(pm[k])), 1e-6)
    print(f"  tunnel human-variability scales: { {k: round(v, 4) for k, v in TUNNEL_SCALES.items()} }")
    return TUNNEL_SCALES


def _eval_tunnel(args):
    """One CMA-ES candidate on the tunnel training set. Top-level for Pool.
    The parameter spec travels in the args (not via the module global) so the
    effective spec — e.g. Th excluded under horizon_mode=budget — reaches
    spawn-based Pool workers unchanged."""
    vec, spec, base_config, gam_path, train_data, tasks, scales = args
    TUNNEL_SCALES.update(scales)
    cfg = copy.deepcopy(base_config); apply_params(cfg, decode(vec, spec))
    cfg["add_noise"] = False          # noiseless sim (nc kept in the persona)
    cfg["replan_latency_cv"] = 0.0    # deterministic replan latency during fitting;
                                      # the SAVED config keeps the base cv for generation
    try:
        sim = _make_sim(cfg, gam_path)
    except Exception:
        return 1e6
    total, n = 0.0, 0
    for tid in sorted(train_data):
        rounds = train_data[tid]; tc, cl, hw = tasks[tid]; n += 1
        try:
            traj, spd, dt = run_single_sim(sim, tc)
        except Exception:
            total += 1e6; continue
        if len(traj) < 5:
            total += 1e6; continue
        comp = _completion(traj, cl)
        if len(traj) >= MAX_SIM_STEPS and comp < 0.95:
            total += INCOMPLETE_PENALTY * (1.0 - comp); continue
        total += float(np.mean([tunnel_loss(tunnel_metrics(traj, spd, h, cl, dt, hw)) for h in rounds]))
    return total / max(n, 1)


# ---------------------------------------------------------------------------
# Stage 3: free-space (pointing) LQ weights
# ---------------------------------------------------------------------------

def _pointing_profile(traj, speeds, times, center, R):
    """Aligned kinematic descriptors of one pointing round (human or model)."""
    al = em.align_round(traj, times, center, R)
    t = np.asarray(times, float)
    i0 = int(np.searchsorted(t, al["onset_s"] + t[0])); i1 = int(np.searchsorted(t, al["final_entry_s"] + t[0]))
    i1 = max(i1, i0 + 2); i0 = min(i0, len(traj) - 3)
    seg_traj = traj[i0:i1 + 1]; seg_spd = speeds[i0:i1 + 1]
    return {"mt_kin": al["mt_kin_s"], "traj": seg_traj, "speeds": seg_spd,
            "end_depth": float(np.linalg.norm(np.asarray(traj[-1]) - np.asarray(center)) / R)}


def pointing_metrics(model_prof, human_prof, canonical):
    _, _, lat_m = resample_by_progress(model_prof["traj"], canonical, N_PROGRESS_BINS)
    _, _, lat_h = resample_by_progress(human_prof["traj"], canonical, N_PROGRESS_BINS)
    _, spd_m = resample_speeds_by_progress(model_prof["speeds"], model_prof["traj"], canonical, N_PROGRESS_BINS)
    _, spd_h = resample_speeds_by_progress(human_prof["speeds"], human_prof["traj"], canonical, N_PROGRESS_BINS)
    return {"mt_rel": abs(model_prof["mt_kin"] - human_prof["mt_kin"]) / max(human_prof["mt_kin"], 0.1),
            "speed_rmse": speed_profile_rmse(spd_m, spd_h), "speed_corr": speed_profile_correlation(spd_m, spd_h),
            "end_depth": abs(model_prof["end_depth"] - human_prof["end_depth"]),
            "lateral_rmse": trajectory_rmse(lat_m, lat_h)}


def pointing_loss(m):
    loss = 0.0
    for k in ("mt_rel", "speed_rmse", "end_depth", "lateral_rmse"):
        loss += POINT_LOSS_WEIGHTS[k] * m[k] / POINT_SCALES[k]
    return loss + POINT_LOSS_WEIGHTS["speed_corr"] * (1.0 - m["speed_corr"]) / POINT_SCALES["speed_corr"]


def _human_pointing_profiles(rounds):
    out = []
    for h in rounds:
        R = h["condition"]["targetRadius"]; ctr = em.pointing_target_center(h)
        times = [(t - h["timestamps"][0]) / 1000.0 for t in h["timestamps"]]
        prof = _pointing_profile(h["trajectory"], h["speeds"], times, ctr, R)
        prof["center"] = ctr; prof["R"] = R; prof["start"] = list(map(float, h["trajectory"][0]))
        prof["canonical"] = [prof["start"], list(map(float, ctr))]; prof["round"] = h
        out.append(prof)
    return out


def compute_pointing_scales(train_data):
    pm = {k: [] for k in POINT_LOSS_WEIGHTS}
    for tid, rounds in train_data.items():
        profs = _human_pointing_profiles(rounds)
        for a, b in combinations(profs, 2):
            m = pointing_metrics(a, b, b["canonical"])
            for k in pm:
                pm[k].append(1.0 - m[k] if k == "speed_corr" else m[k])
    if pm["mt_rel"]:
        for k in pm:
            POINT_SCALES[k] = max(float(np.median(pm[k])), 1e-6)
    print(f"  pointing human-variability scales: { {k: round(v, 4) for k, v in POINT_SCALES.items()} }")
    return POINT_SCALES


def _eval_pointing(args):
    """One CMA-ES candidate on the pointing training set (one noiseless sim per human round)."""
    vec, base_config, gam_path, train_data, scales = args
    POINT_SCALES.update(scales)
    cfg = copy.deepcopy(base_config); apply_params(cfg, decode(vec, POINTING_PARAM_SPEC))
    cfg["add_noise"] = False          # noiseless sim (nc kept in the persona)
    cfg["replan_latency_cv"] = 0.0    # deterministic replan latency during fitting
    try:
        sim = _make_sim(cfg, gam_path)
    except Exception:
        return 1e6
    total, n = 0.0, 0
    for tid in sorted(train_data):
        for hp in _human_pointing_profiles(train_data[tid]):
            n += 1
            try:
                tc, _, _ = em.build_fitts_bypass_config(hp["round"], hp["R"], max_steps=MAX_SIM_STEPS)
                traj, spd, dt = run_single_sim(sim, tc, target_radius=hp["R"])
            except Exception:
                total += 1e6; continue
            if len(traj) < 5:
                total += 1e6; continue
            if len(traj) >= MAX_SIM_STEPS:
                total += INCOMPLETE_PENALTY; continue
            mp = _pointing_profile(traj, spd, [i * dt for i in range(len(traj))], hp["center"], hp["R"])
            total += pointing_loss(pointing_metrics(mp, hp, hp["canonical"]))
    return total / max(n, 1)


# ---------------------------------------------------------------------------
# CMA-ES driver (shared by Stages 2 and 3)
# ---------------------------------------------------------------------------

def run_cmaes(label, spec, initial_params, eval_fn, shared_args, time_limit, seed, popsize, n_workers, sigma0=0.15, patience=None, min_rel_improve=0.01):
    import cma
    print(f"\n--- {label}: CMA-ES over {[s['name'] for s in spec]} ---", flush=True)
    x0 = encode(initial_params, spec)
    n_workers = n_workers or min(multiprocessing.cpu_count(), popsize)
    initial_loss = eval_fn((x0, *shared_args))
    print(f"  initial loss {initial_loss:.4f}; {n_workers} workers, popsize {popsize}, budget {time_limit:.0f}s", flush=True)
    es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, {"bounds": [[0.0] * len(x0), [1.0] * len(x0)], "popsize": popsize,
                                                        "seed": seed, "verb_disp": 0, "verb_log": 0, "verb_filenameprefix": "", "verbose": -9})
    best_x, best_loss = x0.copy(), initial_loss
    hist = [{"generation": 0, "best_loss": float(best_loss), "elapsed_sec": 0.0}]
    t0 = time.time(); gen = 0
    last_improve_gen, ref_best = 0, best_loss
    with multiprocessing.Pool(processes=n_workers) as pool:
        while not es.stop() and time.time() - t0 < time_limit:
            if patience and gen - last_improve_gen >= patience:
                print(f"  early stop: no >{min_rel_improve*100:.0f}% improvement in {patience} generations", flush=True); break
            sols = es.ask()
            fit = pool.map(eval_fn, [(x, *shared_args) for x in sols])
            es.tell(sols, fit); gen += 1
            i = int(np.argmin(fit))
            if fit[i] < best_loss:
                best_loss, best_x = fit[i], np.array(sols[i]).copy()
            if best_loss < ref_best * (1.0 - min_rel_improve):
                ref_best, last_improve_gen = best_loss, gen
            hist.append({"generation": gen, "best_loss": float(best_loss), "mean_loss": float(np.mean(fit)), "elapsed_sec": round(time.time() - t0, 1)})
            print(f"  gen {gen:3d} best {best_loss:.4f} mean {np.mean(fit):.4f} ({time.time() - t0:.0f}s)", flush=True)
    fitted = decode(best_x, spec)
    print(f"  {label} done: {gen} generations, {time.time() - t0:.0f}s")
    for k, v in fitted.items():
        print(f"    {k:16s}: {initial_params[k]:.6g} -> {v:.6g}")
    return fitted, float(best_loss), hist


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_fitting(pid, base_config_path, time_limit, seed, popsize, n_workers, stages="all"):
    print(f"Fitting {pid} | base config {base_config_path} | budget {time_limit}s | seed {seed} | stages {stages}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(base_config_path) as f:
        base = json.load(f)
    base.pop("_description", None); base.pop("_tuning_notes", None)
    sm = base.get("speed_model") or {}
    if sm.get("path") and not Path(sm["path"]).is_absolute():   # temp configs are written elsewhere
        sm["path"] = str((Path(base_config_path).resolve().parent / sm["path"]).resolve())

    rounds_by_tid, tid_to_condition, tid_to_bucket = load_participant(pid)
    tunnel_tasks = build_tunnel_tasks(tid_to_condition, tid_to_bucket)
    tun_train, tun_test = split_tunnel(rounds_by_tid, tid_to_condition, tid_to_bucket)
    pt_train, pt_test = split_pointing(rounds_by_tid, tid_to_condition, tid_to_bucket)
    print(f"tunnel train {sorted(tun_train)}\ntunnel test  {sorted(tun_test)}\npointing train {sorted(pt_train)}\npointing test  {sorted(pt_test)}")

    gam_path = str(RESULTS_DIR / f"{pid}_gam_s{seed}.pkl")
    cfg_path = RESULTS_DIR / f"{pid}_gam_config_s{seed}.json"
    fit_path = RESULTS_DIR / f"{pid}_gam_fit_s{seed}.json"
    result = {"participant_id": pid, "seed": seed, "base_config": str(base_config_path),
              "tunnel_train_tids": sorted(tun_train), "tunnel_test_tids": sorted(tun_test),
              "pointing_train_tids": sorted(pt_train), "pointing_test_tids": sorted(pt_test)}
    if stages == "pointing" and cfg_path.exists():
        # Reuse the saved tunnel fit. The config (fitted tunnel weights) and
        # the GAM pkl are saved as each stage completes; the fit RECORD is
        # only written at the very end, so it may be missing when a previous
        # run was killed late (e.g. the 8-25 cluster jobs, whose wall equalled
        # the CMA budget) — the record is optional here, the config is not.
        if fit_path.exists():
            with open(fit_path) as f:
                prev = json.load(f)
            result.update({k: v for k, v in prev.items() if k not in result})
        with open(cfg_path) as f:
            base = json.load(f)
        base["speed_model"] = {"type": "gam", "path": gam_path}
        print("  reusing tunnel fit from", cfg_path)
    elif stages == "pointing":
        raise FileNotFoundError(
            f"--stages pointing needs the saved tunnel config {cfg_path}")
    T0 = time.time()

    # Under horizon_mode=budget the prediction horizon is DETERMINED by the
    # gaze-fitted difficulty-budget lookahead (D0, gamma, T_min in the
    # base config) — Th is unused by the simulator, so it is excluded from
    # the Stage 2 search instead of being fitted as an inert dimension.
    tunnel_spec = TUNNEL_PARAM_SPEC
    if base.get("horizon_mode") == "budget":
        tunnel_spec = [sp for sp in TUNNEL_PARAM_SPEC if sp["name"] != "Th"]
        print("  horizon_mode=budget: Th excluded from Stage 2 — the prediction horizon "
              "is set by the gaze-fitted budget lookahead (fixed during this fit).")
    if stages in ("all", "tunnel"):
        # Phase 0
        steer_train_tasks = {t: tunnel_tasks[t][0] for t in tun_train}
        ref_params, ref_loss, ref_hist = fit_reference_path_params(
            tun_train, steer_train_tasks, base, time_limit=min(300.0, 0.05 * time_limit), seed=seed)
        base.setdefault("reference_path", {}).update(ref_params)
        result.update({"fitted_ref_path_params": ref_params, "ref_path_loss": ref_loss})
        # Stage 1
        all_tunnel = {**tun_train, **tun_test}
        gam = fit_gam_speed_model(all_tunnel, {t: tunnel_tasks[t][0] for t in all_tunnel}, ref_params)
        gam.save(gam_path); print(f"  GAM saved to {gam_path}")
        base["speed_model"] = {"type": "gam", "path": gam_path}
        # Stage 2
        compute_tunnel_scales(tun_train, tunnel_tasks)
        init = {s["name"]: (base["Th"] if s["name"] == "Th" else base["planner_weights"][s["name"]]) for s in tunnel_spec}
        remaining = time_limit - (time.time() - T0)
        budget = remaining * (0.80 if stages == "all" else 1.0)
        fitted_t, best_t, hist_t = run_cmaes("Stage 2 (tunnel)", tunnel_spec, init, _eval_tunnel,
                                              (tunnel_spec, base, gam_path, tun_train, tunnel_tasks, dict(TUNNEL_SCALES)),
                                              budget, seed, popsize, n_workers)
        apply_params(base, fitted_t)
        bx = encode(fitted_t, tunnel_spec)
        train_loss = _eval_tunnel((bx, tunnel_spec, base, gam_path, tun_train, tunnel_tasks, dict(TUNNEL_SCALES)))
        test_loss = _eval_tunnel((bx, tunnel_spec, base, gam_path, tun_test, tunnel_tasks, dict(TUNNEL_SCALES))) if tun_test else None
        print(f"  tunnel: train loss {train_loss:.4f}  test loss {test_loss}")
        result.update({"fitted_tunnel_params": fitted_t, "tunnel_best_loss": best_t, "tunnel_train_loss": train_loss,
                       "tunnel_test_loss": test_loss, "tunnel_scales": dict(TUNNEL_SCALES), "tunnel_history": hist_t})
        _save_config(base, cfg_path, pid, seed)

    if stages in ("all", "pointing") and pt_train:
        compute_pointing_scales(pt_train)
        init = {s["name"]: base["planner_weights"][s["name"]] for s in POINTING_PARAM_SPEC}
        remaining = max(time_limit - (time.time() - T0), 60.0)
        fitted_p, best_p, hist_p = run_cmaes("Stage 3 (pointing)", POINTING_PARAM_SPEC, init, _eval_pointing,
                                              (base, gam_path, pt_train, dict(POINT_SCALES)), remaining, seed, popsize, n_workers)
        apply_params(base, fitted_p)
        bx = encode(fitted_p, POINTING_PARAM_SPEC)
        p_train = _eval_pointing((bx, base, gam_path, pt_train, dict(POINT_SCALES)))
        p_test = _eval_pointing((bx, base, gam_path, pt_test, dict(POINT_SCALES))) if pt_test else None
        print(f"  pointing: train loss {p_train:.4f}  test loss {p_test}")
        result.update({"fitted_pointing_params": fitted_p, "pointing_best_loss": best_p, "pointing_train_loss": p_train,
                       "pointing_test_loss": p_test, "pointing_scales": dict(POINT_SCALES), "pointing_history": hist_p})
        _save_config(base, cfg_path, pid, seed)

    result["total_elapsed_sec"] = round(time.time() - T0, 1)
    result["final_config"] = _clean_config(base, pid, seed)
    with open(fit_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved {cfg_path}\nSaved {fit_path}")
    return result


def _clean_config(cfg, pid, seed):
    out = copy.deepcopy(cfg)
    out["speed_model"] = {"type": "gam", "path": f"{pid}_gam_s{seed}.pkl"}   # relative to results dir
    out["_description"] = f"Fitted persona for {pid} (seed {seed}); see {pid}_gam_fit_s{seed}.json"
    return out


def _save_config(cfg, path, pid, seed):
    with open(path, "w") as f:
        json.dump(_clean_config(cfg, pid, seed), f, indent=2, default=str)


def main():
    ap = argparse.ArgumentParser(description="Per-participant fitting: ref path + GAM + tunnel MPCC + pointing LQ weights")
    ap.add_argument("--pid", required=True)
    ap.add_argument("--config", default=str(DEFAULT_BASE_CONFIG), help="base persona JSON (default: office_worker.json)")
    ap.add_argument("--time-limit", type=int, default=1800, help="total budget in seconds")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--popsize", type=int, default=12)
    ap.add_argument("--n-workers", type=int, default=None)
    ap.add_argument("--stages", choices=["all", "tunnel", "pointing"], default="all")
    ap.add_argument("--data-dir", default=None,
                    help="Human data directory (default: env HCS_HUMAN_DATA_DIR or the "
                         "aug-26-prolific dataset). Gaze cohort: human_data/gaze_cursor_data")
    a = ap.parse_args()
    if a.data_dir:
        global HUMAN_DATA_DIR
        HUMAN_DATA_DIR = Path(a.data_dir)
    run_fitting(a.pid, a.config, a.time_limit, a.seed, a.popsize, a.n_workers, a.stages)


if __name__ == "__main__":
    main()

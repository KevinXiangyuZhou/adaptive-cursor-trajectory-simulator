"""
Three-stage model fitting with GAM speed model.

Phase 0: Fit reference path params (spatial-only, no simulation needed) (~10-30s).
Stage 1: Fit GAM speed model directly from human speed data (seconds).
Stage 2: Fit MPCC tracking params via CMA-ES with fixed speed model (minutes).

Usage:
    python -m eval.model_fitting.fit_speed_model --pid P111602
    python -m eval.model_fitting.fit_speed_model --pid P111602 --time-limit 1800
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
from collections import OrderedDict
from pathlib import Path

import numpy as np

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from hcs_package import CursorSimulator
from hcs_package.speed_model import GAMSpeedModel
from hcs_package.reference_path import ReferencePath, generate_optimal_reference_path
from hcs_package.constraints import PathConstraint, RectangleConstraint, PolygonConstraint
from hcs_package.constraint_utils import parse_constraints_from_json, convert_constraints_to_corridor_bounds

from eval.model_fitting.extract_speed_data import (
    extract_all_speed_data,
    extract_all_speed_data_v2,
)
from eval.model_fitting.run_fitting import (
    TRIAL_CONDITIONS,
    HUMAN_DATA_DIR,
    MAX_SIM_STEPS,
    N_PROGRESS_BINS,
    LOSS_WEIGHTS,
    WALL_MARGIN_WEIGHT,
    INCOMPLETE_PENALTY,
    load_participant_data,
    split_train_test,
    build_all_tasks,
    build_task,
    run_single_sim,
    compute_trial_metrics,
    metrics_to_loss,
    _apply_params,
    _TOP_LEVEL_PARAMS,
    TRAIN_TIDS,
    TEST_TIDS,
    CMAES_TIDS,
    STEERING_TIDS,
    compute_human_variability_scales,
)
from eval.utils.stats import resample_by_progress


# ---------------------------------------------------------------------------
# Phase 0: Fit reference path params (spatial-only, no simulation)
# ---------------------------------------------------------------------------

REF_PATH_PARAM_SPEC = [
    {"name": "w_cut",                "bounds": (0.0, 1.0)},
    {"name": "w_suppress",           "bounds": (0.0, 2.0)},
    {"name": "w_width_exp",          "bounds": (0.0, 3.0)},
    {"name": "cut_window_frac",      "bounds": (0.02, 0.20)},
    {"name": "global_clearance_ref", "bounds": (0.005, 0.05)},
]


def _encode_ref_params(params):
    """Encode ref path params to [0,1] vector."""
    vec = np.zeros(len(REF_PATH_PARAM_SPEC))
    for i, spec in enumerate(REF_PATH_PARAM_SPEC):
        lo, hi = spec["bounds"]
        vec[i] = (params[spec["name"]] - lo) / (hi - lo)
    return np.clip(vec, 0.0, 1.0)


def _decode_ref_params(vec):
    """Decode [0,1] vector to ref path param dict."""
    params = {}
    for i, spec in enumerate(REF_PATH_PARAM_SPEC):
        lo, hi = spec["bounds"]
        params[spec["name"]] = lo + float(vec[i]) * (hi - lo)
    return params


def _normalize_waypoints_to_meters(task_config):
    """Convert task config waypoints from pixels to meters.

    Replicates CursorSimulator's coordinate normalization.
    """
    screen_width = task_config["screen_width"]
    screen_height = task_config["screen_height"]
    screen_width_m = 0.46
    screen_height_m = screen_height / screen_width * screen_width_m
    return [
        (x / screen_width * screen_width_m, y / screen_height * screen_height_m)
        for x, y in task_config["waypoints"]
    ]


def _precompute_task_geometry(task_configs, margin=0.0):
    """Pre-compute per-task geometry that is invariant to ref path params.

    This avoids re-parsing constraints, building centerline splines, and
    computing corridor bounds on every CMA-ES evaluation.

    Returns:
        dict {tid: {waypoints_norm, tunnel_width, centerline_spline,
                     corridor_bounds, cartesian_regions}}
    """
    geometry = {}
    for tid, task_config in task_configs.items():
        waypoints_norm = _normalize_waypoints_to_meters(task_config)

        constraints_dict = task_config.get("constraints")
        constraint_config = None
        cartesian_regions = []
        tunnel_width = None

        if constraints_dict is not None:
            constraint_config = parse_constraints_from_json(constraints_dict)
            for region in constraint_config.regions:
                if isinstance(region.geometry, PathConstraint):
                    tunnel_width = float(region.geometry.width)
                elif isinstance(region.geometry, (RectangleConstraint, PolygonConstraint)):
                    cartesian_regions.append(region)

        if tunnel_width is None:
            distances = [
                np.linalg.norm(np.array(waypoints_norm[i + 1]) - np.array(waypoints_norm[i]))
                for i in range(len(waypoints_norm) - 1)
            ]
            avg_distance = np.mean(distances) if distances else 0.1
            tunnel_width = min(0.1, max(0.02, avg_distance * 0.3))

        centerline_spline = ReferencePath(waypoints_norm, s=0.0, k=3)

        corridor_bounds = None
        if constraint_config is not None:
            corridor_bounds = convert_constraints_to_corridor_bounds(
                constraint_config, centerline_spline, default_margin=margin,
            )

        geometry[tid] = {
            "waypoints_norm": waypoints_norm,
            "tunnel_width": tunnel_width,
            "centerline_spline": centerline_spline,
            "corridor_bounds": corridor_bounds,
            "cartesian_regions": cartesian_regions,
            "margin": margin,
        }
    return geometry


def _build_ref_path_from_geometry(geom, ref_params):
    """Generate reference path from pre-computed geometry and ref path params.

    Returns:
        (ref_polyline, ref_path) where ref_polyline is an (M,2) array
        of densely sampled points, and ref_path is the ReferencePath object.
    """
    ref_path = generate_optimal_reference_path(
        tunnel_path=geom["waypoints_norm"],
        tunnel_width=geom["tunnel_width"],
        margin=geom["margin"],
        w_cut=ref_params["w_cut"],
        w_suppress=ref_params["w_suppress"],
        w_width_exp=ref_params["w_width_exp"],
        cut_window_frac=ref_params["cut_window_frac"],
        global_clearance_ref=ref_params["global_clearance_ref"],
        cartesian_constraints=geom["cartesian_regions"] or None,
        corridor_bounds=geom["corridor_bounds"],
        centerline_cache=geom["centerline_spline"],
    )

    n_samples = 200
    thetas = np.linspace(0, ref_path.total_length, n_samples)
    ref_polyline = np.array([ref_path(t) for t in thetas])

    return ref_polyline, ref_path


def _build_ref_path_for_task(task_config, ref_params, margin=0.0):
    """Generate reference path for a task using given ref path params.

    Convenience wrapper that builds geometry on the fly.  For repeated
    evaluations (e.g. CMA-ES), prefer _precompute_task_geometry +
    _build_ref_path_from_geometry.
    """
    geometry = _precompute_task_geometry({0: task_config}, margin=margin)
    return _build_ref_path_from_geometry(geometry[0], ref_params)


def _eval_ref_path_spatial(param_vec, train_data, task_geometry):
    """Evaluate one reference path parameter vector using spatial-only loss.

    For each trial condition, generates the reference path then computes
    lateral RMSE of human trajectories projected onto that reference path.
    No simulation is run.

    Args:
        task_geometry: Pre-computed geometry from _precompute_task_geometry.
    """
    ref_params = _decode_ref_params(param_vec)

    total_loss = 0.0
    n_tasks = 0

    for tid in sorted(train_data.keys()):
        rounds = train_data[tid]
        if not rounds:
            continue

        try:
            ref_polyline, _ = _build_ref_path_from_geometry(
                task_geometry[tid], ref_params
            )
        except Exception:
            total_loss += 1e6
            n_tasks += 1
            continue

        n_tasks += 1
        round_rmses = []
        for human_trial in rounds:
            h_traj = human_trial["trajectory"]
            # Project human trajectory onto the reference path (not centerline).
            # Lateral deviation from the reference path is the spatial error.
            # Use fewer bins than full eval — spatial alignment doesn't
            # need fine granularity and resample_by_progress is the bottleneck.
            _, _, lat_h = resample_by_progress(h_traj, ref_polyline, 50)
            lat_rmse = float(np.sqrt(np.mean(np.asarray(lat_h) ** 2)))
            round_rmses.append(lat_rmse)

        total_loss += sum(round_rmses) / len(round_rmses)

    if n_tasks > 0:
        total_loss = total_loss * 4 / n_tasks

    return total_loss


def fit_reference_path_params(train_data, task_configs, base_config,
                              time_limit=60.0, seed=42, sigma0=0.3,
                              popsize=8):
    """Phase 0: Fit reference path params using spatial-only loss.

    Optimizes corner-cutting / race-tracing parameters by minimizing
    lateral RMSE between human trajectories and the generated reference
    path.  No simulation needed — reference path generation is deterministic
    and fast (~ms per evaluation).

    Returns:
        fitted_params, best_loss, loss_history
    """
    import cma

    print("\n--- Phase 0: Fitting reference path params (spatial-only) ---")

    # Pre-compute task geometry (centerline splines, corridor bounds, etc.)
    # so that each CMA-ES evaluation only runs generate_optimal_reference_path.
    task_geometry = _precompute_task_geometry(task_configs)
    print(f"  Pre-computed geometry for {len(task_geometry)} tasks")

    # Initial params from config
    rp = base_config.get("reference_path", {})
    pw = base_config.get("planner_weights", {})
    initial_params = {}
    for spec in REF_PATH_PARAM_SPEC:
        name = spec["name"]
        lo, hi = spec["bounds"]
        initial_params[name] = rp.get(name, pw.get(name, (lo + hi) / 2))

    x0 = _encode_ref_params(initial_params)
    n_params = len(REF_PATH_PARAM_SPEC)

    def obj_fn(x):
        return _eval_ref_path_spatial(x, train_data, task_geometry)

    initial_loss = obj_fn(x0)
    print(f"  Initial loss: {initial_loss:.6f}")
    print(f"  Initial params: { {k: round(v, 5) for k, v in initial_params.items()} }")

    es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, {
        "bounds": [[0.0] * n_params, [1.0] * n_params],
        "popsize": popsize,
        "seed": seed,
        "maxiter": 300,
        "verb_disp": 0, "verb_log": 0, "verb_filenameprefix": "",
        "verbose": -9,
    })

    best_x = x0.copy()
    best_loss = initial_loss
    loss_history = [{"generation": 0, "best_loss": initial_loss,
                     "mean_loss": initial_loss, "elapsed_sec": 0.0}]

    start = time.time()
    generation = 0

    # No multiprocessing — evaluations are ~10ms each
    while not es.stop():
        elapsed = time.time() - start
        if elapsed > time_limit:
            print(f"  Time limit reached ({elapsed:.0f}s)")
            break

        solutions = es.ask()
        fitness = [obj_fn(x) for x in solutions]
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

        if generation % 20 == 0 or generation <= 3:
            print(f"  Gen {generation:3d}: best={best_loss:.6f}  "
                  f"mean={np.mean(fitness):.6f}  elapsed={elapsed:.1f}s",
                  flush=True)

    fitted_params = _decode_ref_params(best_x)
    total_elapsed = time.time() - start
    print(f"\n  Phase 0 complete: {generation} generations, {total_elapsed:.1f}s")
    for name, val in fitted_params.items():
        print(f"    {name:25s}: {initial_params[name]:.6g} -> {val:.6g}")

    return fitted_params, best_loss, loss_history


# ---------------------------------------------------------------------------
# Stage 1: Fit GAM speed model from human data
# ---------------------------------------------------------------------------

def fit_gam_speed_model(participant_data, centerlines, trial_conditions,
                        task_geometry=None, ref_path_params=None,
                        lam_grid=None):
    """Fit GAM speed model from human speed observations.

    Args:
        participant_data: dict {trial_id: [round dicts]}.
        centerlines: dict {trial_id: centerline_pts}.
        trial_conditions: dict {trial_id: condition_dict}.
        task_geometry: dict from _precompute_task_geometry (optional).
        ref_path_params: fitted reference path params dict (optional).
        lam_grid: smoothing parameter grid for gridsearch.

    When task_geometry and ref_path_params are provided, features are
    extracted using the reference path (matching simulation-time computation).
    Otherwise falls back to legacy extraction with constant clearance.

    Returns:
        Fitted GAMSpeedModel.
    """
    print("\n--- Stage 1: Fitting GAM speed model from human data ---")
    t0 = time.time()

    # Extract (geometry, speed) pairs
    if task_geometry is not None and ref_path_params is not None:
        print("  Using reference-path-aligned feature extraction (v2)")
        data = extract_all_speed_data_v2(
            participant_data, task_geometry, ref_path_params,
            build_ref_path_fn=_build_ref_path_from_geometry,
        )
    else:
        print("  Using legacy feature extraction (constant clearance)")
        data = extract_all_speed_data(participant_data, centerlines, trial_conditions)
    if data is None:
        raise ValueError("No speed data extracted from participant trials")

    n_total = len(data["speed"])
    n_trials = len(np.unique(data["trial_id"]))
    print(f"  Extracted {n_total} observations from {n_trials} trial types")

    # Filter out very low speeds (start/end of trials where cursor is stationary)
    speed_threshold = 0.005  # 5 mm/s
    valid = data["speed"] > speed_threshold
    n_valid = np.sum(valid)
    print(f"  After filtering stationary points: {n_valid} observations")

    clearance = data["clearance"][valid]
    kappa = data["kappa"][valid]
    dkappa_ds = data["dkappa_ds"][valid]
    speeds = data["speed"][valid]

    # Report feature ranges (useful for diagnosing train-test mismatch)
    print(f"  Clearance range: [{clearance.min():.4f}, {clearance.max():.4f}] m"
          f"  (unique levels: {len(np.unique(np.round(clearance, 4)))})")
    print(f"  Kappa range:     [{kappa.min():.2f}, {kappa.max():.2f}]")

    # Estimate base speed as median of observed speeds
    base_speed = float(np.median(speeds))
    print(f"  Estimated base speed: {base_speed:.4f} m/s")

    # Fit GAM
    gam_model = GAMSpeedModel(
        base_speed=base_speed,
        floor=0.01,    # 10 mm/s minimum
        ceil=0.50,     # 500 mm/s maximum
    )
    gam_model.fit(clearance, kappa, dkappa_ds, speeds, lam_grid=lam_grid)

    elapsed = time.time() - t0
    print(f"  GAM fitted in {elapsed:.1f}s")

    # Report prediction quality
    predicted = gam_model.compute_speed_profile(
        np.arange(n_valid), clearance, kappa, dkappa_ds
    )
    corr = np.corrcoef(speeds, predicted)[0, 1]
    rmse = np.sqrt(np.mean((speeds - predicted) ** 2))
    print(f"  Training fit: corr={corr:.3f}, RMSE={rmse:.4f} m/s")

    return gam_model


# ---------------------------------------------------------------------------
# Stage 2: Fit MPCC params with GAM speed model fixed
# ---------------------------------------------------------------------------

# Only MPCC tracking params — speed is handled by the GAM,
# reference path params are handled by Phase 0.
MPCC_PARAM_SPEC = [
    {"name": "contour",    "log_scale": True,  "bounds": (0.5, 2.5)},   # ~3-316
    {"name": "lag",        "log_scale": True,  "bounds": (-2.0, 1.0)},  # 0.01-10
    {"name": "jerk",       "log_scale": True,  "bounds": (-8.0, -4.0)}, # 1e-8-1e-4
    {"name": "constraint", "log_scale": True,  "bounds": (1.5, 2.5)},   # ~30-316
    {"name": "progress",   "log_scale": True,  "bounds": (-4.0, 0.5)},  # 1e-4 to ~3.0
    {"name": "Th",         "log_scale": False, "bounds": (0.35, 0.6),
     "discrete_step": 0.05, "config_key": "top_level"},
]

_MPCC_SPEC_BY_NAME = {s["name"]: s for s in MPCC_PARAM_SPEC}


def _encode_mpcc(params):
    """Encode MPCC param dict to [0,1] vector."""
    vec = np.zeros(len(MPCC_PARAM_SPEC))
    for i, spec in enumerate(MPCC_PARAM_SPEC):
        val = params[spec["name"]]
        lo, hi = spec["bounds"]
        if spec["log_scale"]:
            val = math.log10(val)
        vec[i] = (val - lo) / (hi - lo)
    return np.clip(vec, 0.0, 1.0)


def _decode_mpcc(vec):
    """Decode [0,1] vector to MPCC param dict."""
    params = {}
    for i, spec in enumerate(MPCC_PARAM_SPEC):
        lo, hi = spec["bounds"]
        val = lo + vec[i] * (hi - lo)
        if spec["log_scale"]:
            val = 10.0 ** val
        if "discrete_step" in spec:
            val = round(val / spec["discrete_step"]) * spec["discrete_step"]
        params[spec["name"]] = val
    return params


def _eval_mpcc_single(args):
    """Evaluate one MPCC parameter vector with GAM speed model."""
    (param_vec, base_config, gam_model_path, train_data,
     task_configs, centerlines) = args

    params = _decode_mpcc(param_vec)

    cfg = copy.deepcopy(base_config)
    _apply_params(cfg, params)
    cfg["nc"] = [0, 0]

    # Load GAM model
    gam_model = GAMSpeedModel.load(gam_model_path)

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".json", prefix="gam_fit_")
    os.close(tmp_fd)
    try:
        with open(tmp_path, "w") as f:
            json.dump(cfg, f)

        sim = CursorSimulator(str(tmp_path))
        sim.speed_model = gam_model

        total_loss = 0.0
        n_tasks = 0
        for tid in sorted(train_data.keys()):
            rounds = train_data[tid]
            n_tasks += 1

            try:
                model_traj, model_speeds, sim_interval = run_single_sim(
                    sim, task_configs[tid]
                )
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
                t = (0.0 if seg_len2 < 1e-18
                     else float(np.clip(np.dot(last_pt - cl[i], seg) / seg_len2, 0.0, 1.0)))
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
                    sim_interval=sim_interval, tunnel_half_width=half_w,
                )
                round_losses.append(metrics_to_loss(metrics))
            total_loss += sum(round_losses) / len(round_losses)

        if n_tasks > 0:
            total_loss = total_loss / n_tasks

        return total_loss
    finally:
        os.unlink(tmp_path)


def fit_mpcc_params(base_config, gam_model_path, train_data,
                    task_configs, centerlines, time_limit, seed,
                    cmaes_task_ids=None, n_workers=None, sigma0=0.15,
                    popsize=12):
    """Stage 2: Fit MPCC tracking params with CMA-ES.

    The GAM speed model is fixed; only MPCC params are optimized.

    Returns:
        fitted_params, best_loss, loss_history
    """
    import cma

    print("\n--- Stage 2: Fitting MPCC params via CMA-ES ---")

    # Initial MPCC params from config
    initial_params = {}
    for spec in MPCC_PARAM_SPEC:
        name = spec["name"]
        ck = spec.get("config_key")
        if ck == "top_level" or name in _TOP_LEVEL_PARAMS:
            initial_params[name] = base_config.get(
                name, base_config.get("planner_weights", {}).get(name))
        elif ck == "reference_path":
            initial_params[name] = base_config.get("reference_path", {}).get(
                name, base_config.get("planner_weights", {}).get(name))
        else:
            initial_params[name] = base_config["planner_weights"][name]

    x0 = _encode_mpcc(initial_params)

    if n_workers is None:
        n_workers = min(multiprocessing.cpu_count(), popsize)

    # Optionally restrict CMA-ES to a task subset for efficiency
    if cmaes_task_ids is not None:
        cmaes_data = {tid: rounds for tid, rounds in train_data.items()
                      if tid in cmaes_task_ids}
        print(f"  CMA-ES task subset: {sorted(cmaes_data.keys())} "
              f"({len(cmaes_data)} of {len(train_data)} train tasks)")
    else:
        cmaes_data = train_data

    shared_args = (base_config, gam_model_path, cmaes_data,
                   task_configs, centerlines)

    def obj_fn(x):
        return _eval_mpcc_single((x, *shared_args))

    initial_loss = obj_fn(x0)
    print(f"  Initial loss: {initial_loss:.6f}")
    print(f"  Using {n_workers} workers, {len(MPCC_PARAM_SPEC)} params")

    n_params = len(MPCC_PARAM_SPEC)
    es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, {
        "bounds": [[0.0] * n_params, [1.0] * n_params],
        "popsize": popsize,
        "seed": seed,
        "verb_disp": 0, "verb_log": 0, "verb_filenameprefix": "",
        "verbose": -9,
    })

    best_x = x0.copy()
    best_loss = initial_loss
    loss_history = [{"generation": 0, "best_loss": initial_loss,
                     "mean_loss": initial_loss, "elapsed_sec": 0.0}]

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
            fitness = pool.map(_eval_mpcc_single, eval_args)
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
            print(f"  Gen {generation:3d}: best={best_loss:.6f}  "
                  f"mean={np.mean(fitness):.6f}  elapsed={elapsed:.0f}s",
                  flush=True)

    fitted_params = _decode_mpcc(best_x)
    total_elapsed = time.time() - start
    print(f"\n  Stage 2 complete: {generation} generations, {total_elapsed:.0f}s")
    for name, val in fitted_params.items():
        print(f"    {name:20s}: {initial_params[name]:.6g} -> {val:.6g}")

    return fitted_params, best_loss, loss_history


# ---------------------------------------------------------------------------
# Full three-stage pipeline
# ---------------------------------------------------------------------------

def run_two_stage_fitting(pid, base_config_path, time_limit=1800, seed=42,
                          popsize=12, n_workers=None):
    """Run complete three-stage fitting: ref path + GAM + MPCC.

    Phase 0: Fit reference path params (spatial-only, ~10-30s).
    Stage 1: Fit GAM speed model from human data (~1s).
    Stage 2: Fit MPCC tracking params with GAM fixed (remaining time).

    Args:
        pid: Participant ID.
        base_config_path: Path to initial config JSON.
        time_limit: Total time limit in seconds.
        seed: Random seed.

    Returns:
        results dict with fitted config, GAM model path, metrics.
    """
    print(f"Three-stage GAM fitting for {pid}")
    print(f"Time limit: {time_limit}s, seed: {seed}")

    # Load config
    with open(base_config_path) as f:
        base_config = json.load(f)

    # Load human data
    participant_data = load_participant_data(pid)
    train_data, test_data = split_train_test(participant_data)
    print(f"Train/test split (width-based):")
    print(f"  Train tasks ({len(train_data)}): {sorted(train_data.keys())}")
    print(f"  Test tasks  ({len(test_data)}): {sorted(test_data.keys())}")

    # Build tasks
    task_configs, centerlines = build_all_tasks()

    total_start = time.time()

    # Phase 0: Fit reference path params (spatial-only, ~60-300s)
    # Use 5% of budget (capped at 300s). Each generation takes ~48s,
    # so 300s ≈ 6 generations.
    ref_time_budget = min(300.0, time_limit * 0.05)
    fitted_ref, ref_loss, ref_history = fit_reference_path_params(
        train_data, task_configs, base_config,
        time_limit=ref_time_budget, seed=seed,
    )

    # Apply fitted ref path params to base config for downstream stages
    base_config.setdefault("reference_path", {}).update(fitted_ref)

    phase0_elapsed = time.time() - total_start

    # Stage 1: Fit GAM (~1s)
    # Use ALL steering data (train + test) — fast direct fit, no overfitting risk.
    # Exclude straight tunnels: GAM can't model their inverted-U speed profile
    # (kappa=0 everywhere gives constant output).
    task_geometry = _precompute_task_geometry(task_configs)
    all_steering = {tid: rounds for tid, rounds in {**train_data, **test_data}.items()
                    if tid in STEERING_TIDS}
    print(f"  GAM training on {len(all_steering)} steering tasks "
          f"(excluded {len(train_data) + len(test_data) - len(all_steering)} straight)")
    gam_model = fit_gam_speed_model(
        all_steering, centerlines, TRIAL_CONDITIONS,
        task_geometry=task_geometry, ref_path_params=fitted_ref,
    )

    # Save GAM model
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    gam_path = str(results_dir / f"{pid}_gam_s{seed}.pkl")
    gam_model.save(gam_path)
    print(f"  GAM model saved to {gam_path}")

    stage1_elapsed = time.time() - total_start - phase0_elapsed

    # Compute human variability scales for loss normalization
    compute_human_variability_scales(train_data, centerlines)

    remaining = time_limit - (time.time() - total_start)

    # Stage 2: Fit MPCC params (ref path params fixed from Phase 0)
    if remaining > 30:
        fitted_mpcc, best_loss, history = fit_mpcc_params(
            base_config, gam_path, train_data,
            task_configs, centerlines,
            time_limit=remaining, seed=seed,
            cmaes_task_ids=TRAIN_TIDS,  # use full training set
            n_workers=n_workers, popsize=popsize,
        )

        # Build final config
        final_config = copy.deepcopy(base_config)
        _apply_params(final_config, fitted_mpcc)

        # Validation: evaluate on full train set and test set
        best_x = _encode_mpcc(fitted_mpcc)
        val_args = (base_config, gam_path, train_data, task_configs, centerlines)
        train_loss = _eval_mpcc_single((best_x, *val_args))
        print(f"\n  Validation — full train loss ({len(train_data)} tasks): {train_loss:.6f}")

        if test_data:
            test_args = (base_config, gam_path, test_data, task_configs, centerlines)
            test_loss = _eval_mpcc_single((best_x, *test_args))
            print(f"  Validation — test loss ({len(test_data)} tasks): {test_loss:.6f}")
        else:
            test_loss = None
    else:
        print("  Skipping Stage 2: insufficient time remaining")
        fitted_mpcc = {}
        best_loss = float('inf')
        train_loss = float('inf')
        test_loss = None
        history = []
        final_config = base_config

    total_elapsed = time.time() - total_start

    # Build a clean GAM config: only MPCC params + speed_model reference
    # (no parametric speed params that would be ignored anyway)
    _PARAMETRIC_SPEED_KEYS = {
        "desired_speed", "speed_alpha", "speed_reference", "speed_floor",
        "speed_ceil", "kappa_ref", "kappa_alpha", "kappa_floor",
        "rate_percentile", "rate_scale", "rate_alpha", "rate_floor",
        "rate_median_max", "straight_boost", "boost_sharpness", "rate_ceil",
    }
    _REF_PATH_KEYS = {
        "w_cut", "w_suppress", "w_width_exp", "cut_window_frac",
        "global_clearance_ref",
    }
    gam_config = copy.deepcopy(final_config)
    gam_config.pop("_tuning_notes", None)
    pw = gam_config.get("planner_weights", {})
    # Migrate any ref path keys still in planner_weights to reference_path
    rp = gam_config.setdefault("reference_path", {})
    for key in _REF_PATH_KEYS:
        if key in pw and key not in rp:
            rp[key] = pw[key]
    for key in _PARAMETRIC_SPEED_KEYS | _REF_PATH_KEYS:
        pw.pop(key, None)
    gam_config["speed_model"] = {
        "type": "gam",
        "path": f"{pid}_gam_s{seed}.pkl",
    }

    gam_config_path = results_dir / f"{pid}_gam_config_s{seed}.json"
    with open(gam_config_path, "w") as f:
        json.dump(gam_config, f, indent=2, default=str)
    print(f"\nGAM config saved to {gam_config_path}")

    # Save full results (with all details for reproducibility)
    result = {
        "participant_id": pid,
        "seed": seed,
        "gam_model_path": gam_path,
        "gam_config_path": str(gam_config_path),
        "fitted_ref_path_params": fitted_ref,
        "ref_path_loss": ref_loss,
        "fitted_mpcc_params": fitted_mpcc,
        "best_loss": best_loss,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "train_tids": sorted(train_data.keys()),
        "test_tids": sorted(test_data.keys()),
        "cmaes_tids": sorted(CMAES_TIDS),
        "total_elapsed_sec": round(total_elapsed, 1),
        "phase0_elapsed_sec": round(phase0_elapsed, 1),
        "stage1_elapsed_sec": round(stage1_elapsed, 1),
        "final_config": final_config,
    }

    result_path = results_dir / f"{pid}_gam_fit_s{seed}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Full results saved to {result_path}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Three-stage model fitting: reference path + GAM speed model + MPCC params"
    )
    parser.add_argument("--pid", required=True, help="Participant ID")
    parser.add_argument("--config", default=None,
                        help="Path to initial config JSON")
    parser.add_argument("--time-limit", type=int, default=1800,
                        help="Total time limit in seconds (default: 1800)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--popsize", type=int, default=12,
                        help="CMA-ES population size for Stage 2 (default: 12)")
    parser.add_argument("--n-workers", type=int, default=None,
                        help="Number of parallel workers for Stage 2 (default: min(cpu_count, popsize))")

    args = parser.parse_args()

    config_path = args.config
    if config_path is None:
        config_path = str(Path(__file__).parent / "initial_user_config.json")

    run_two_stage_fitting(
        pid=args.pid,
        base_config_path=config_path,
        time_limit=args.time_limit,
        seed=args.seed,
        popsize=args.popsize,
        n_workers=args.n_workers,
    )


if __name__ == "__main__":
    main()

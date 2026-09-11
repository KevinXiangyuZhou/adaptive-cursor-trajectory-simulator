"""
Fit CHI-26-EA baseline model to individual participants via CMA-ES.

The baseline model has 7 tunable parameters:
  - 6 planner weights: jerk, progress, wall, contour, lag, desired_speed
  - 1 top-level: Th (prediction horizon)

Usage:
    # Fit single participant:
    python -m eval.baseline_fitting.fit_baseline --pid P111602 --time-limit 3600

    # Fit all participants:
    python -m eval.baseline_fitting.fit_baseline --all --time-limit 3600
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
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "hcs_package" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

BASELINE_PKG_DIR = PROJECT_ROOT / "eval" / "chi-26-ea_baseline_pacakage" / "src"
RESULTS_DIR = SCRIPT_DIR / "results"

from eval.model_fitting.run_fitting import (
    TRIAL_CONDITIONS,
    TRAIN_TIDS,
    TEST_TIDS,
    STEERING_TIDS,
    MAX_SIM_STEPS,
    N_PROGRESS_BINS,
    LOSS_WEIGHTS,
    WALL_MARGIN_WEIGHT,
    INCOMPLETE_PENALTY,
    load_participant_data,
    split_train_test,
    build_all_tasks,
    run_single_sim,
    compute_trial_metrics,
    metrics_to_loss,
    compute_human_variability_scales,
)

# ---------------------------------------------------------------------------
# Default baseline config (matches CHI-26-EA user_config.json)
# ---------------------------------------------------------------------------
DEFAULT_BASELINE_CONFIG = {
    "Interval": 0.05,
    "Tp": 0.05,
    "Th": 0.30,
    "nc": [0.20, 0.020],
    "forearm": 0.35,
    "mouseGain": 1,
    "planner_weights": {
        "jerk": 1.227e-06,
        "progress": 0.106e-06,
        "wall": 200.37,
        "contour": 20.87,
        "lag": 0.0581,
        "desired_speed": 0.208,
    },
    "planner_margin": 0.001,
    "add_noise": False,
    "random_seed": 1000,
}

# ---------------------------------------------------------------------------
# Parameter specification (7 params)
# ---------------------------------------------------------------------------
BASELINE_PARAM_SPEC = [
    {"name": "jerk",          "log_scale": True,  "bounds": (-10.0, -4.0)},   # 1e-10 to 1e-4
    {"name": "progress",      "log_scale": True,  "bounds": (-8.0, -4.0)},    # 1e-8 to 1e-4
    {"name": "wall",          "log_scale": True,  "bounds": (1.0, 3.5)},      # 10 to ~3162
    {"name": "contour",       "log_scale": True,  "bounds": (0.5, 3.5)},      # ~3 to ~3162
    {"name": "lag",           "log_scale": True,  "bounds": (-3.0, 1.0)},     # 0.001 to 10
    {"name": "desired_speed", "log_scale": True,  "bounds": (-1.5, 0.1)},     # 0.01 to ~3.16
    {"name": "Th",            "log_scale": False, "bounds": (0.20, 0.60),
     "discrete_step": 0.05, "top_level": True},
]

_SPEC_BY_NAME = {s["name"]: s for s in BASELINE_PARAM_SPEC}


def _encode(params):
    """Encode param dict to [0,1] vector."""
    vec = np.zeros(len(BASELINE_PARAM_SPEC))
    for i, spec in enumerate(BASELINE_PARAM_SPEC):
        val = params[spec["name"]]
        lo, hi = spec["bounds"]
        if spec["log_scale"]:
            val = math.log10(val)
        vec[i] = (val - lo) / (hi - lo)
    return np.clip(vec, 0.0, 1.0)


def _decode(vec):
    """Decode [0,1] vector to param dict."""
    params = {}
    for i, spec in enumerate(BASELINE_PARAM_SPEC):
        lo, hi = spec["bounds"]
        val = lo + float(vec[i]) * (hi - lo)
        if spec["log_scale"]:
            val = 10.0 ** val
        if "discrete_step" in spec:
            val = round(val / spec["discrete_step"]) * spec["discrete_step"]
        params[spec["name"]] = val
    return params


def _apply_params(cfg, params):
    """Apply fitted params to a baseline config dict (in-place)."""
    for name, val in params.items():
        if _SPEC_BY_NAME[name].get("top_level"):
            cfg[name] = val
        else:
            cfg["planner_weights"][name] = val


def _load_baseline_simulator_class():
    """Import CursorSimulator from CHI-26-EA baseline with module isolation."""
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


def _run_baseline_sim(sim, task_config):
    """Run one baseline simulation, return trajectory, speeds, interval.

    Same as run_single_sim but works with the baseline CursorSimulator.
    """
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
    # Central-difference + 5-point moving average
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
# CMA-ES objective (top-level for pickling)
# ---------------------------------------------------------------------------

def _eval_baseline_single(args):
    """Evaluate one baseline parameter vector. For multiprocessing.Pool.map()."""
    (param_vec, base_config, train_data, task_configs, centerlines,
     baseline_cls_module_path) = args

    params = _decode(param_vec)

    cfg = copy.deepcopy(base_config)
    _apply_params(cfg, params)
    cfg["nc"] = [0, 0]  # disable noise for deterministic evaluation

    # Write config to temp file and create baseline simulator
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".json", prefix="bl_fit_")
    os.close(tmp_fd)
    try:
        with open(tmp_path, "w") as f:
            json.dump(cfg, f)

        # Load baseline class via module isolation in each worker
        saved = {k: sys.modules.pop(k)
                 for k in list(sys.modules) if k == 'hcs_package' or k.startswith('hcs_package.')}
        sys.path.insert(0, str(baseline_cls_module_path))
        try:
            from hcs_package.cursor_simulator import CursorSimulator as BaselineCls
        finally:
            for k in list(sys.modules):
                if k == 'hcs_package' or k.startswith('hcs_package.'):
                    del sys.modules[k]
            sys.modules.update(saved)
            sys.path.remove(str(baseline_cls_module_path))

        sim = BaselineCls(tmp_path)

        total_loss = 0.0
        n_tasks = 0
        for tid in sorted(train_data.keys()):
            rounds = train_data[tid]
            n_tasks += 1

            try:
                model_traj, model_speeds, sim_interval = _run_baseline_sim(
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


# ---------------------------------------------------------------------------
# Fitting pipeline
# ---------------------------------------------------------------------------

def fit_baseline(pid, time_limit=3600, seed=42, popsize=12, n_workers=None):
    """Fit baseline model to a single participant via CMA-ES.

    Returns:
        results dict with fitted config, metrics, timing.
    """
    import cma

    print(f"\n{'='*60}")
    print(f"Fitting baseline model for {pid}")
    print(f"Time limit: {time_limit}s, seed: {seed}, popsize: {popsize}")
    print(f"{'='*60}")

    # Load human data
    participant_data = load_participant_data(pid)
    train_data, test_data = split_train_test(participant_data)
    print(f"Train tasks ({len(train_data)}): {sorted(train_data.keys())}")
    print(f"Test tasks  ({len(test_data)}): {sorted(test_data.keys())}")

    total_rounds = sum(len(r) for r in train_data.values())
    print(f"Total training rounds: {total_rounds}")

    # Build tasks
    task_configs, centerlines = build_all_tasks()

    # Compute human variability scales
    compute_human_variability_scales(train_data, centerlines)

    # Initial params from default config
    base_config = copy.deepcopy(DEFAULT_BASELINE_CONFIG)
    initial_params = {}
    for spec in BASELINE_PARAM_SPEC:
        name = spec["name"]
        if spec.get("top_level"):
            initial_params[name] = base_config[name]
        else:
            initial_params[name] = base_config["planner_weights"][name]

    x0 = _encode(initial_params)
    n_params = len(BASELINE_PARAM_SPEC)

    if n_workers is None:
        n_workers = min(multiprocessing.cpu_count(), popsize)

    # Use all training tasks for CMA-ES evaluation
    cmaes_data = train_data
    print(f"CMA-ES training tasks: {sorted(cmaes_data.keys())} ({len(cmaes_data)} tasks)")

    baseline_pkg = str(BASELINE_PKG_DIR)
    shared_args = (base_config, cmaes_data, task_configs, centerlines, baseline_pkg)

    def obj_fn(x):
        return _eval_baseline_single((x, *shared_args))

    print("\nEvaluating initial parameters...")
    initial_loss = obj_fn(x0)
    print(f"Initial loss: {initial_loss:.6f}")
    print(f"Using {n_workers} workers, {n_params} params")

    es = cma.CMAEvolutionStrategy(x0.tolist(), 0.20, {
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
            fitness = pool.map(_eval_baseline_single, eval_args)
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

    fitted_params = _decode(best_x)
    total_elapsed = time.time() - start
    print(f"\nFitting complete: {generation} generations, {total_elapsed:.0f}s")
    for name, val in fitted_params.items():
        print(f"  {name:20s}: {initial_params[name]:.6g} -> {val:.6g}")

    # Build final config
    final_config = copy.deepcopy(base_config)
    _apply_params(final_config, fitted_params)

    # Validation on full train set
    val_args = (base_config, train_data, task_configs, centerlines, baseline_pkg)
    train_loss = _eval_baseline_single((_encode(fitted_params), *val_args))
    print(f"\nValidation — full train loss ({len(train_data)} tasks): {train_loss:.6f}")

    test_loss = None
    if test_data:
        test_args = (base_config, test_data, task_configs, centerlines, baseline_pkg)
        test_loss = _eval_baseline_single((_encode(fitted_params), *test_args))
        print(f"Validation — test loss ({len(test_data)} tasks): {test_loss:.6f}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    config_path = RESULTS_DIR / f"{pid}_baseline_config_s{seed}.json"
    with open(config_path, "w") as f:
        json.dump(final_config, f, indent=2)
    print(f"\nConfig saved to {config_path}")

    result = {
        "participant_id": pid,
        "seed": seed,
        "config_path": str(config_path),
        "initial_params": {k: float(v) for k, v in initial_params.items()},
        "fitted_params": {k: float(v) for k, v in fitted_params.items()},
        "initial_loss": float(initial_loss),
        "best_loss": float(best_loss),
        "train_loss": float(train_loss),
        "test_loss": float(test_loss) if test_loss is not None else None,
        "train_tids": sorted(train_data.keys()),
        "test_tids": sorted(test_data.keys()),
        "cmaes_tids": sorted(cmaes_data.keys()),
        "generations": generation,
        "total_elapsed_sec": round(total_elapsed, 1),
        "loss_history": loss_history,
    }

    result_path = RESULTS_DIR / f"{pid}_baseline_fit_s{seed}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {result_path}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fit CHI-26-EA baseline model to participants via CMA-ES"
    )
    parser.add_argument("--pid", default=None, help="Participant ID")
    parser.add_argument("--all", action="store_true",
                        help="Fit all participants found in human_data/raw")
    parser.add_argument("--time-limit", type=int, default=3600,
                        help="Time limit per participant in seconds (default: 3600)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--popsize", type=int, default=12,
                        help="CMA-ES population size (default: 12)")
    parser.add_argument("--n-workers", type=int, default=None,
                        help="Number of parallel workers (default: min(cpu_count, popsize))")

    args = parser.parse_args()

    if args.all:
        # Discover all participant IDs
        human_dir = PROJECT_ROOT / "eval" / "human_data" / "raw"
        pids = set()
        for fpath in sorted(human_dir.glob("*.json")):
            try:
                with open(fpath) as f:
                    data = json.load(f)
                pid = data.get("participantId", "")
                if pid:
                    pids.add(pid)
            except Exception:
                continue
        pids = sorted(pids)
        print(f"Found {len(pids)} participants: {pids}")
        for pid in pids:
            try:
                fit_baseline(pid, time_limit=args.time_limit, seed=args.seed,
                             popsize=args.popsize, n_workers=args.n_workers)
            except Exception as e:
                print(f"ERROR fitting {pid}: {e}")
                continue
    elif args.pid:
        fit_baseline(args.pid, time_limit=args.time_limit, seed=args.seed,
                     popsize=args.popsize, n_workers=args.n_workers)
    else:
        parser.error("Specify --pid or --all")


if __name__ == "__main__":
    main()

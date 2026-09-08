"""CHI-26-EA baseline simulator: isolated import + fit/eval adapters.

The baseline package (eval/chi-26-ea_baseline_pacakage) is ALSO named
``hcs_package``, so it cannot be imported next to the current simulator by
name. ``load_baseline_simulator_class`` swaps ``sys.modules`` for the
duration of one import and restores the current package afterwards; the
returned class keeps references to its own (EA) modules, so it stays usable
after the swap. The class is cached: do the import once per process
(parent, before a fork) and never re-import in a worker that has already
imported the current package.

Adapters mirror ``fit_speed_model._make_sim`` / ``run_single_sim`` so the
fitting objective and the eval runner can drive either simulator through
the same two calls.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINE_PKG_DIR = ROOT / "eval" / "chi-26-ea_baseline_pacakage" / "src"
BASE_CONFIG_DIR = ROOT / "eval" / "model_fitting" / "base_configs_gaze"

# CHI-26-EA defaults (the package's built-in office_worker persona), used as
# the starting point of every baseline fit. Fitted: the six planner weights
# + curvature_scale (planner_weights) and Th (top level).
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
        "curvature_scale": 10.0,
    },
    "planner_margin": 0.001,
    "add_noise": False,
    "random_seed": 1000,
}

_CLS = None


def load_baseline_simulator_class():
    """Import the EA ``CursorSimulator`` with module isolation (cached)."""
    global _CLS
    if _CLS is not None:
        return _CLS
    saved = {k: sys.modules.pop(k) for k in list(sys.modules)
             if k == "hcs_package" or k.startswith("hcs_package.")}
    sys.path.insert(0, str(BASELINE_PKG_DIR))
    try:
        from hcs_package.cursor_simulator import CursorSimulator as Cls
    finally:
        for k in list(sys.modules):
            if k == "hcs_package" or k.startswith("hcs_package."):
                del sys.modules[k]
        sys.modules.update(saved)
        sys.path.remove(str(BASELINE_PKG_DIR))
    _CLS = Cls
    return Cls


def baseline_base_config(pid=None):
    """EA defaults with the participant's noise/plant constants (nc, forearm,
    Interval) copied from the current-model base persona when one exists, so
    both models share the same plant."""
    import copy
    cfg = copy.deepcopy(DEFAULT_BASELINE_CONFIG)
    if pid is not None:
        p = BASE_CONFIG_DIR / f"{pid}.json"
        if p.exists():
            persona = json.load(open(p))
            for k in ("nc", "forearm", "Interval", "Tp"):
                if k in persona:
                    cfg[k] = persona[k]
    return cfg


def make_baseline_sim(cfg):
    Cls = load_baseline_simulator_class()
    fd, path = tempfile.mkstemp(suffix=".json", prefix="bl_cfg_")
    os.close(fd)
    with open(path, "w") as f:
        json.dump(cfg, f)
    try:
        return Cls(path)
    finally:
        os.unlink(path)


def _smooth_speeds(traj, interval):
    for p in (ROOT / "eval" / "model_fitting",):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from fit_speed_model import _smooth_speeds as _s
    return _s(traj, interval)


def run_baseline_sim(sim, task_config, target_radius=None):
    """One baseline simulation; returns (traj_m, speeds, interval) exactly as
    ``fit_speed_model.run_single_sim`` does for the current model."""
    task_config = dict(task_config)
    # ID4SCS tasks carry a per-sample width list; the EA package reads a
    # scalar for its reference-path slack (the corridor itself comes from
    # the constraints block, which it parses fine). Use the narrowest width
    # so the race-tracing path never plans outside the corridor.
    if isinstance(task_config.get("tunnel_width"), (list, tuple)):
        task_config["tunnel_width"] = float(min(task_config["tunnel_width"]))
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        json.dump(task_config, tf)
        path = tf.name
    try:
        traj_raw = sim.generate_trajectory_with_waypoints(
            task_file=path,
            max_steps=task_config.get("max_steps", 600),
            target_radius=(target_radius if target_radius is not None
                           else task_config.get("target_radius", 0.01)),
            use_optimal_path=True,
        )
    finally:
        os.unlink(path)
    traj = [[x * 0.001, y * 0.001] for x, y, _ in traj_raw]
    return traj, _smooth_speeds(traj, sim.interval), sim.interval

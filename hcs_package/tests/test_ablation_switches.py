"""Ablation switches (paper evaluation): `horizon_mode: fixed_lead` and
`catchup_mode: constant`.

The first test is the bit-identity guard: a persona that does not mention
either key must produce exactly the trajectory it produced before the
switches existed, and spelling the defaults out must change nothing.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "hcs_package" / "src" / "hcs_package" / "user_configurations"


def _make_sim(overrides, seed=11):
    from hcs_package.cursor_simulator import CursorSimulator
    cfg = json.load(open(CONFIG_DIR / "office_worker.json"))
    cfg["random_seed"] = seed
    cfg.update(overrides)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
    return CursorSimulator(f.name)


@pytest.fixture(scope="module")
def sinusoidal_task():
    import sys
    sys.path.insert(0, str(ROOT))
    from experiment.environment import create_environment, generate_task_config
    env = create_environment(
        {"env_type": "tunnel_steering_smooth", "tunnelWidth": 0.03,
         "curvature": 0.025})
    task = generate_task_config(env, include_constraints=True)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(task, f)
    return f.name


def _run(sim, task, max_steps=300):
    traj = sim.generate_trajectory_with_waypoints(
        task_file=task, max_steps=max_steps, target_radius=0.015)
    return np.asarray([[x, y] for x, y, _ in traj], dtype=float), sim.last_diagnostics


def test_defaults_are_bit_identical(sinusoidal_task):
    # keys absent vs. keys spelled out at their defaults: identical output
    base = {"add_noise": False, "replan_latency_cv": 0.0}
    t_absent, d_absent = _run(_make_sim(base), sinusoidal_task)
    t_explicit, d_explicit = _run(_make_sim({**base, "horizon_mode": "budget",
                                             "catchup_mode": "pace_law",
                                             "fixed_lead_m": 0.03}), sinusoidal_task)
    assert t_absent.shape == t_explicit.shape
    assert np.array_equal(t_absent, t_explicit)
    assert d_absent["n_solves"] == d_explicit["n_solves"]
    assert d_absent["horizon_mode"] == "budget"
    assert d_absent["catchup_mode"] == "pace_law"


def test_old_fixed_horizon_flag_still_refused():
    with pytest.raises(ValueError, match="horizon_mode='fixed'"):
        _make_sim({"horizon_mode": "fixed"})


def test_unknown_switch_values_refused():
    with pytest.raises(ValueError):
        _make_sim({"horizon_mode": "banana"})
    with pytest.raises(ValueError):
        _make_sim({"catchup_mode": "banana"})
    with pytest.raises(ValueError):
        _make_sim({"horizon_mode": "fixed_lead", "fixed_lead_m": 0.0})


def test_fixed_lead_places_constant_lookahead(sinusoidal_task):
    lead = 0.02
    sim = _make_sim({"add_noise": False, "replan_latency_cv": 0.0,
                     "horizon_mode": "fixed_lead", "fixed_lead_m": lead,
                     "budget": {"D0": 1.0, "T_min": 0.0, "gamma": 0.66, "W_ref": 0.026}})
    _, d = _run(sim, sinusoidal_task)
    ev = d["replan_events"]
    assert len(ev) >= 3
    leads = np.array([e["anchor"] - e["theta"] for e in ev[:-1]])   # last may be path-end capped
    assert np.allclose(leads, lead, atol=1e-9)


def test_fixed_lead_differs_from_budget(sinusoidal_task):
    base = {"add_noise": False, "replan_latency_cv": 0.0}
    t_budget, _ = _run(_make_sim(base), sinusoidal_task)
    t_fixed, _ = _run(_make_sim({**base, "horizon_mode": "fixed_lead",
                                 "fixed_lead_m": 0.01}), sinusoidal_task)
    assert not (t_budget.shape == t_fixed.shape and np.array_equal(t_budget, t_fixed))


def test_constant_catchup_ignores_pace_law(sinusoidal_task):
    base = {"add_noise": False, "replan_latency_cv": 0.0}
    t_pace, d_pace = _run(_make_sim(base), sinusoidal_task)
    t_const, d_const = _run(_make_sim({**base, "catchup_mode": "constant"}), sinusoidal_task)
    assert d_const["catchup_mode"] == "constant"
    assert d_const["traversal_speed_model"] == "gam_traversal"   # loaded, just unused
    assert not (t_pace.shape == t_const.shape and np.array_equal(t_pace, t_const))

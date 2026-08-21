"""Tests for the difficulty-budget horizon and intermittent replan scheduler.

The backward-compatibility guarantee (horizon_mode="fixed" +
replan_mode="every_step" reproduces the pre-module simulator bit-for-bit) was
verified against git HEAD via trajectory fingerprints; the tests here cover
the module math, the trigger state machine, and the simulator-level behavior
of the new modes.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from hcs_package.intermittent import (
    DifficultyBudgetHorizon,
    ReplanEvent,
    ReplanScheduler,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "hcs_package" / "src" / "hcs_package" / "user_configurations"


# ---------------------------------------------------- DifficultyBudgetHorizon

def _uniform_horizon(width, v=0.2, length=2.0, n=2001, D0=1.4, lam=1.0,
                     kappa=0.0):
    s = np.linspace(0.0, length, n)
    return DifficultyBudgetHorizon(
        s_profile=s,
        width_profile=np.full(n, width),
        kappa_profile=np.full(n, kappa),
        v_ref_profile=np.full(n, v),
        D0=D0,
        lam=lam,
    )


def test_budget_anchor_constant_width():
    # density = 1/W, so h = D0 * W exactly on a straight uniform corridor
    bh = _uniform_horizon(width=0.03, D0=1.4)
    h = bh.anchor(0.5) - 0.5
    assert h == pytest.approx(1.4 * 0.03, rel=1e-3)


def test_budget_anchor_narrower_is_shorter():
    h_wide = _uniform_horizon(width=0.05).anchor(0.0)
    h_narrow = _uniform_horizon(width=0.01).anchor(0.0)
    assert h_narrow < h_wide
    assert h_narrow == pytest.approx(1.4 * 0.01, rel=1e-3)


def test_budget_curvature_consumes_budget():
    # adding curvature raises the density, shortening the lookahead
    h_straight = _uniform_horizon(width=0.03, kappa=0.0).anchor(0.0)
    h_curved = _uniform_horizon(width=0.03, kappa=20.0).anchor(0.0)
    assert h_curved < h_straight
    # analytic: h = D0 / (1/W + lam*kappa)
    assert h_curved == pytest.approx(1.4 / (1 / 0.03 + 20.0), rel=1e-3)


def test_budget_anchor_caps_at_path_end():
    bh = _uniform_horizon(width=0.05, length=0.04)  # whole path cheaper than D0
    assert bh.anchor(0.0) == pytest.approx(0.04)
    assert bh.anchor(0.03) == pytest.approx(0.04)


def test_budget_traverse_time_constant_speed():
    bh = _uniform_horizon(width=0.03, v=0.25)
    assert bh.traverse_time(0.2, 0.7) == pytest.approx(0.5 / 0.25, rel=1e-6)


def test_budget_lam_zero_is_width_only():
    bh = _uniform_horizon(width=0.02, kappa=50.0, lam=0.0)
    assert bh.anchor(0.0) == pytest.approx(1.4 * 0.02, rel=1e-3)


def test_budget_floor_binds_at_speed():
    # narrow corridor: budget lead D0*W = 0.014; at v=0.2 the visuomotor
    # floor v*T_min = 0.03 wins
    s = np.linspace(0.0, 2.0, 2001)
    bh = DifficultyBudgetHorizon(
        s_profile=s, width_profile=np.full(2001, 0.01),
        kappa_profile=np.zeros(2001), v_ref_profile=np.full(2001, 0.2),
        D0=1.4, lam=1.0, T_min=0.15)
    assert bh.anchor(0.5, v_now=0.0) - 0.5 == pytest.approx(0.014, rel=1e-3)
    assert bh.anchor(0.5, v_now=0.2) - 0.5 == pytest.approx(0.03, rel=1e-3)
    # wide corridor: budget lead 0.07 exceeds the floor -> unchanged
    bh_wide = DifficultyBudgetHorizon(
        s_profile=s, width_profile=np.full(2001, 0.05),
        kappa_profile=np.zeros(2001), v_ref_profile=np.full(2001, 0.2),
        D0=1.4, lam=1.0, T_min=0.15)
    assert bh_wide.anchor(0.5, v_now=0.2) - 0.5 == pytest.approx(0.07, rel=1e-3)


def test_budget_floor_caps_at_path_end():
    bh = _uniform_horizon(width=0.05, length=0.04)
    bh.T_min = 10.0
    assert bh.anchor(0.03, v_now=1.0) == pytest.approx(0.04)


# --------------------------------------------------------- ReplanScheduler

def _event(step=0, t=0.0, theta=0.0, anchor=1.0, n_steps=10, trigger="init"):
    return ReplanEvent(step=step, t=t, theta=theta, anchor=anchor,
                       n_steps=n_steps, trigger=trigger)


def test_scheduler_every_step_always_replans():
    sch = ReplanScheduler(mode="every_step")
    assert sch.needs_replan(0.0) == "init"
    sch.on_replan(_event(), anchor=np.inf, plan_len=6)
    sch.on_step_executed()
    assert sch.needs_replan(0.0) == "every_step"


def test_scheduler_intermittent_arrival_then_latency():
    sch = ReplanScheduler(mode="intermittent", latency_steps=3)
    assert sch.needs_replan(-np.inf) == "init"
    sch.on_replan(_event(anchor=1.0, n_steps=20), anchor=1.0, plan_len=20)

    # before arrival: keep executing
    for _ in range(4):
        assert sch.needs_replan(0.5) is None
        sch.on_step_executed()

    # arrival starts the countdown (this step + 2 more execute), replan on the
    # 3rd step after arrival
    assert sch.needs_replan(1.01) is None
    sch.on_step_executed()
    assert sch.needs_replan(1.05) is None
    sch.on_step_executed()
    assert sch.needs_replan(1.10) is None
    sch.on_step_executed()
    assert sch.needs_replan(1.15) == "arrival+latency"


def test_scheduler_intermittent_zero_latency_fires_on_arrival():
    sch = ReplanScheduler(mode="intermittent", latency_steps=0)
    sch.on_replan(_event(), anchor=1.0, plan_len=20)
    assert sch.needs_replan(0.9) is None
    assert sch.needs_replan(1.0) == "arrival+latency"


def test_scheduler_intermittent_exhaustion_backstop():
    sch = ReplanScheduler(mode="intermittent", latency_steps=4)
    sch.on_replan(_event(), anchor=np.inf, plan_len=3)
    for _ in range(3):
        assert sch.needs_replan(0.0) is None
        sch.on_step_executed()
    assert sch.needs_replan(0.0) == "exhausted"


def test_scheduler_latency_cv_zero_is_deterministic():
    sch = ReplanScheduler(mode="intermittent", latency_steps=4, latency_cv=0.0)
    assert all(sch._sample_latency_steps() == 4 for _ in range(20))


def test_scheduler_latency_cv_samples_lognormal_around_median():
    np.random.seed(3)
    sch = ReplanScheduler(mode="intermittent", latency_steps=4, latency_cv=0.9)
    draws = np.array([sch._sample_latency_steps() for _ in range(4000)])
    assert draws.std() > 0
    assert abs(np.median(draws) - 4) <= 1   # median preserved
    assert draws.min() >= 0


def test_scheduler_deviation_triggers_early_replan():
    sch = ReplanScheduler(mode="intermittent", latency_steps=4,
                          deviation_frac=0.3)
    sch.on_replan(_event(anchor=1.0, n_steps=20), anchor=1.0, plan_len=20)
    sch.on_step_executed()
    # below threshold, before arrival: keep executing
    assert sch.needs_replan(0.5, deviation_ratio=0.2) is None
    # above threshold: replan even though the anchor was never reached
    assert sch.needs_replan(0.5, deviation_ratio=0.4) == "deviation"


def test_scheduler_deviation_disabled_by_default():
    sch = ReplanScheduler(mode="intermittent", latency_steps=4)
    sch.on_replan(_event(anchor=1.0, n_steps=20), anchor=1.0, plan_len=20)
    sch.on_step_executed()
    assert sch.needs_replan(0.5, deviation_ratio=5.0) is None


def test_scheduler_wants_theta_only_while_hunting_arrival():
    sch = ReplanScheduler(mode="intermittent", latency_steps=2)
    assert not sch.wants_theta  # no plan yet
    sch.on_replan(_event(), anchor=1.0, plan_len=10)
    assert sch.wants_theta
    sch.needs_replan(1.5)  # arrival -> latency countdown
    assert not sch.wants_theta  # no projection needed during countdown
    sch_every = ReplanScheduler(mode="every_step")
    sch_every.on_replan(_event(), anchor=1.0, plan_len=10)
    assert not sch_every.wants_theta


# ----------------------------------------------------- warm start across N

def test_warm_start_survives_horizon_change():
    """The MPCC warm start pads/truncates the shifted previous solution, so
    budget mode's per-solve horizon changes keep the warm start alive."""
    from hcs_package.reference_path import ReferencePath
    from hcs_package.mpcc_model import generate_mpcc, reset_warm_start

    ref = ReferencePath([(0.0, 0.0), (0.2, 0.0)], s=0.0, k=1)
    state = [0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0]
    weights = {"jerk": 1.2e-6, "progress": 1e-7, "contour": 20, "lag": 0.05}
    limits = {"acc_max": 100.0}

    reset_warm_start()
    for n in (10, 6, 14):  # shrink then grow the horizon
        controls, info = generate_mpcc(
            ref, state, n, 0.05, weights, limits,
            speed_profile=np.full(n, 0.1), desired_speed=0.1,
            corridor_bounds=(0.02, 0.02), warm_shift=2)
        assert controls.shape == (n, 3)
        assert np.all(np.isfinite(controls))


# ------------------------------------------------------- simulator behavior

def _make_sim(overrides, seed=11):
    from hcs_package.cursor_simulator import CursorSimulator
    cfg = json.load(open(CONFIG_DIR / "office_worker.json"))
    cfg["speed_model"]["path"] = str(CONFIG_DIR / "population_gam.pkl")
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


def test_sim_every_step_solves_each_step(sinusoidal_task):
    sim = _make_sim({})
    sim.generate_trajectory_with_waypoints(
        task_file=sinusoidal_task, max_steps=120, target_radius=0.015)
    d = sim.last_diagnostics
    assert d["n_solves"] == d["n_steps_executed"]
    assert all(e["trigger"] in ("init", "every_step") for e in d["replan_events"])


def test_sim_intermittent_replans_sparsely(sinusoidal_task):
    sim = _make_sim({"replan_mode": "intermittent"})
    sim.generate_trajectory_with_waypoints(
        task_file=sinusoidal_task, max_steps=300, target_radius=0.015)
    d = sim.last_diagnostics
    assert d["n_solves"] <= d["n_steps_executed"] / 3
    cycles = np.diff([e["t"] for e in d["replan_events"]])
    assert np.median(cycles) >= 0.2  # plan-execute cycles, not per-step
    # cursor keeps moving between solves
    assert d["n_steps_executed"] > 50


def test_sim_intermittent_noiseless_arrives_at_anchor(sinusoidal_task):
    # deterministic execution follows the plan exactly, so every non-final
    # cycle should end at/past its anchor (arrival-triggered, not exhausted)
    sim = _make_sim({"replan_mode": "intermittent", "add_noise": False})
    sim.generate_trajectory_with_waypoints(
        task_file=sinusoidal_task, max_steps=300, target_radius=0.015)
    ev = sim.last_diagnostics["replan_events"]
    assert len(ev) >= 3
    non_init = [e for e in ev[1:]]
    arrival = [e for e in non_init if e["trigger"] == "arrival+latency"]
    assert len(arrival) >= 0.6 * len(non_init)
    for prev, nxt in zip(ev[:-1], ev[1:]):
        if nxt["trigger"] == "arrival+latency":
            assert nxt["theta"] >= prev["anchor"] - 1e-9


def test_sim_budget_horizon_adapts_to_width():
    import sys
    sys.path.insert(0, str(ROOT))
    from experiment.environment import create_environment, generate_task_config

    anchors = {}
    for width in (0.01, 0.05):
        env = create_environment(
            {"env_type": "tunnel_steering_smooth", "tunnelWidth": width,
             "curvature": 0.025})
        task = generate_task_config(env, include_constraints=True)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(task, f)
        sim = _make_sim({"horizon_mode": "budget",
                         "replan_mode": "intermittent"})
        sim.generate_trajectory_with_waypoints(
            task_file=f.name, max_steps=300, target_radius=width / 2)
        ev = sim.last_diagnostics["replan_events"]
        # lookahead distance per planning event (skip end-capped events)
        looks = [e["anchor"] - e["theta"] for e in ev]
        anchors[width] = float(np.median(looks[: max(1, len(looks) - 2)]))
    assert anchors[0.01] < anchors[0.05]


def test_sim_budget_every_step_matches_baseline_shape(sinusoidal_task):
    # budget + every_step: solves each step, but with adaptive horizons
    sim = _make_sim({"horizon_mode": "budget"})
    sim.generate_trajectory_with_waypoints(
        task_file=sinusoidal_task, max_steps=120, target_radius=0.015)
    d = sim.last_diagnostics
    assert d["n_solves"] == d["n_steps_executed"]
    horizons = {e["n_steps"] for e in d["replan_events"]}
    assert len(horizons) > 1  # horizon varies along the path


def test_sim_budget_solve_horizon_floored_in_time(sinusoidal_task):
    # With T_min > 0 the solve horizon never collapses below ceil(T_min/dt),
    # even when the anchor caps at the path end (the old stall/timeout mode).
    sim = _make_sim({"horizon_mode": "budget",
                     "budget": {"D0": 1.66, "lam": 0.5, "T_min": 0.2}})
    sim.generate_trajectory_with_waypoints(
        task_file=sinusoidal_task, max_steps=300, target_radius=0.015)
    d = sim.last_diagnostics
    floor = int(np.ceil(0.2 / d["interval"]))
    assert all(e["n_steps"] >= floor for e in d["replan_events"])


def test_sim_deviation_trigger_fires_under_noise(sinusoidal_task):
    # A tight threshold plus motor noise must produce early (pre-arrival)
    # replans; the same run without the threshold has none.
    sim = _make_sim({"replan_mode": "intermittent",
                     "replan_deviation_frac": 0.05})
    sim.generate_trajectory_with_waypoints(
        task_file=sinusoidal_task, max_steps=300, target_radius=0.015)
    triggers = [e["trigger"] for e in sim.last_diagnostics["replan_events"]]
    assert "deviation" in triggers

    sim_off = _make_sim({"replan_mode": "intermittent",
                         "replan_deviation_frac": 0.0})
    sim_off.generate_trajectory_with_waypoints(
        task_file=sinusoidal_task, max_steps=300, target_radius=0.015)
    triggers_off = [e["trigger"] for e in sim_off.last_diagnostics["replan_events"]]
    assert "deviation" not in triggers_off


def test_sim_stochastic_latency_varies_cycles(sinusoidal_task):
    # add_noise=False isolates the latency draw as the only stochastic part:
    # cycle lengths must vary with cv > 0 and be constant-ish without.
    sim = _make_sim({"replan_mode": "intermittent", "add_noise": False,
                     "replan_latency_cv": 0.9}, seed=7)
    sim.generate_trajectory_with_waypoints(
        task_file=sinusoidal_task, max_steps=300, target_radius=0.015)
    ev = sim.last_diagnostics["replan_events"]
    arrivals = [e for e in ev if e["trigger"] == "arrival+latency"]
    assert len(arrivals) >= 2
    cycles = np.diff([e["step"] for e in ev])
    assert len(set(cycles.tolist())) > 1

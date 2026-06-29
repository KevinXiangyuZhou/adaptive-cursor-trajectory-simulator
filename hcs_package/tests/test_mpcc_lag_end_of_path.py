"""Tests for MPCC lag-term behaviour near the end of the reference path.

The lag error ``e_lag`` penalises the cursor for falling behind the virtual
progress ``s_traj`` on the reference path.  At the path terminus,
``ref_path(s_traj)`` is clamped to the endpoint, so ``e_lag`` no longer moves
with ``vs`` — only with the planned cursor position ``(px, py)``.  When the
goal-precision term (#5) is active it penalises speed whenever
``s_traj >= path_length``, which can freeze ``e_lag`` across the horizon even
while the cursor remains outside ``target_radius``.

These tests check that ``e_lag`` is still *updating* (changing along the
horizon) when lag catch-up is working, and flag the stagnation pattern that
appears when precision dominates at small target radii.
"""

import numpy as np
import pytest

from hcs_package.reference_path import ReferencePath
from hcs_package.mpcc_model import (
    generate_mpcc,
    reset_warm_start,
    evaluate_tracking_errors,
)


@pytest.fixture
def straight_path():
    """Horizontal 0.4 m straight reference path."""
    return ReferencePath([(0.0, 0.0), (0.4, 0.0)], s=0.0, k=1)


@pytest.fixture
def mpcc_setup(straight_path):
    """Shared MPCC parameters for end-of-path scenarios."""
    dt = 0.05
    num_steps = 6
    speed_profile = np.full(num_steps, 0.13)
    limits = {'acc_max': 100.0}
    base_weights = {
        'jerk': 1.2e-6,
        'progress': 0.1e-6,
        'contour': 20.0,
        'lag': 0.05,
        'nc0': 0.2,
        'nc1': 0.02,
    }
    return {
        'ref_path': straight_path,
        'dt': dt,
        'num_steps': num_steps,
        'speed_profile': speed_profile,
        'limits': limits,
        'base_weights': base_weights,
        'desired_speed': 0.15,
    }


def _run_mpcc(setup, cursor_x, cursor_y=0.0, vx=0.01, vy=0.0,
              goal_precision=0.0, target_radius=0.01):
    ref_path = setup['ref_path']
    reset_warm_start()
    s0 = ref_path.find_closest_theta([cursor_x, cursor_y])
    state_0 = [cursor_x, cursor_y, vx, vy, 0.0, 0.0, s0]
    weights = dict(setup['base_weights'])
    weights['goal_precision'] = goal_precision
    weights['target_radius'] = target_radius

    controls, opt_info = generate_mpcc(
        ref_path=ref_path,
        state_0=state_0,
        num_steps=setup['num_steps'],
        dt=setup['dt'],
        weights=weights,
        limits=setup['limits'],
        speed_profile=setup['speed_profile'],
        desired_speed=setup['desired_speed'],
    )
    errors = evaluate_tracking_errors(
        ref_path, state_0, controls, setup['num_steps'], setup['dt'],
    )
    return controls, errors, opt_info, state_0


def test_e_lag_changes_with_cursor_position_when_past_path_end(straight_path):
    """e_lag must still respond to cursor position once s_traj is clamped."""
    s_end = straight_path.total_length
    tangent = straight_path.tangent(s_end)
    ref_k = straight_path(s_end)
    cos_phi, sin_phi = tangent[0], tangent[1]
    R = np.array([
        [sin_phi, -cos_phi],
        [-cos_phi, -sin_phi],
    ])

    e_lags = []
    for cursor_x in [0.38, 0.39, 0.395]:
        e_lags.append((R @ (np.array([cursor_x, 0.0]) - ref_k))[1])

    assert all(e > 0.0 for e in e_lags), "all positions are behind the endpoint"
    assert e_lags == sorted(e_lags, reverse=True), (
        "e_lag should decrease as the cursor approaches the endpoint"
    )


def test_reference_point_clamped_when_s_traj_exceeds_path_length(straight_path):
    """Once s_traj passes total_length the reference point stops advancing."""
    s_end = straight_path.total_length
    ref_at_end = straight_path(s_end)
    ref_past = straight_path(s_end + 0.05)
    np.testing.assert_allclose(ref_at_end, ref_past, atol=1e-12)


def test_e_lag_updates_along_horizon_without_goal_precision(mpcc_setup):
    """Without term #5, lag drives catch-up: e_lag should change across the horizon."""
    # Cursor is ~5 mm short of the 0.4 m goal; virtual progress already at end.
    _, errors, opt_info, _ = _run_mpcc(
        mpcc_setup,
        cursor_x=0.395,
        goal_precision=0.0,
        target_radius=0.0025,
    )

    assert opt_info['success']
    assert errors['at_path_end'].all(), "scenario should have s_traj at path end"
    assert errors['e_lag'][0] > 0.0, "cursor should start behind the endpoint"

    lag_delta = errors['e_lag'][-1] - errors['e_lag'][0]
    assert lag_delta < -0.001, (
        "e_lag should decrease along the horizon as the planner catches up; "
        f"got delta={lag_delta:.6f}"
    )
    assert errors['px'][-1] > 0.398, "planned endpoint should move toward the goal"


@pytest.mark.parametrize("target_radius", [0.01, 0.0025])
def test_e_lag_stagnates_with_goal_precision_at_path_end(
    mpcc_setup, target_radius,
):
    """With term #5 active, e_lag can freeze even while the cursor is still short."""
    _, errors_no_prec, _, _ = _run_mpcc(
        mpcc_setup,
        cursor_x=0.395,
        goal_precision=0.0,
        target_radius=target_radius,
    )
    _, errors_with_prec, _, _ = _run_mpcc(
        mpcc_setup,
        cursor_x=0.395,
        goal_precision=0.01,
        target_radius=target_radius,
    )

    assert errors_with_prec['at_path_end'].all()
    assert errors_with_prec['e_lag'][0] > 0.0, "cursor still behind endpoint"

    delta_no_prec = abs(errors_no_prec['e_lag'][-1] - errors_no_prec['e_lag'][0])
    delta_with_prec = abs(errors_with_prec['e_lag'][-1] - errors_with_prec['e_lag'][0])

    # Lag is "updating" without precision but nearly flat with it.
    assert delta_no_prec > 10.0 * delta_with_prec, (
        f"e_lag horizon delta without precision ({delta_no_prec:.6f}) should "
        f"exceed delta with precision ({delta_with_prec:.6f}) by 10x"
    )
    assert delta_with_prec < 0.001, (
        f"e_lag barely changes with goal_precision active (delta={delta_with_prec:.6f})"
    )


def test_precision_term_leaves_cursor_short_of_goal(mpcc_setup):
    """Small target_radius + goal_precision can stop before reaching the button."""
    _, errors_no_prec, _, _ = _run_mpcc(
        mpcc_setup, cursor_x=0.395, goal_precision=0.0, target_radius=0.0025,
    )
    _, errors_with_prec, _, _ = _run_mpcc(
        mpcc_setup, cursor_x=0.395, goal_precision=0.01, target_radius=0.0025,
    )

    goal_x = mpcc_setup['ref_path'].total_length
    shortfall_no_prec = goal_x - errors_no_prec['px'][-1]
    shortfall_with_prec = goal_x - errors_with_prec['px'][-1]

    assert shortfall_with_prec > shortfall_no_prec, (
        "goal_precision should leave the planned endpoint farther from the goal"
    )
    assert shortfall_with_prec > 0.0025, (
        f"planned endpoint misses a 2.5 mm target (shortfall={shortfall_with_prec*1000:.2f} mm)"
    )


def test_lag_cost_still_nonzero_when_e_lag_stagnates(mpcc_setup):
    """Even when e_lag stops updating, the lag penalty itself is still present."""
    _, errors, _, _ = _run_mpcc(
        mpcc_setup,
        cursor_x=0.395,
        goal_precision=0.01,
        target_radius=0.0025,
    )

    w_lag = mpcc_setup['base_weights']['lag']
    lag_cost = w_lag * np.sum(errors['e_lag'] ** 2)
    assert lag_cost > 0.0, "lag penalty should be nonzero while cursor is behind"
    assert abs(errors['e_lag'][-1] - errors['e_lag'][0]) < 0.001, (
        "diagnostic: e_lag is stagnant even though lag cost is nonzero"
    )

"""Steering model — orchestrates SpeedModel and MPCC solver."""

import numpy as np
from .params import SteeringModelInput
from .reference_path import ReferencePath
from .speed_model import SpeedModel
from .mpcc_model import (
    generate_mpcc,
    reset_warm_start,
    _build_A_vel_from_jerk,
    _build_A_pos_from_jerk,
)


# Clearance above which a horizon node is treated as unconstrained space.
# Study tunnels are <= 5 cm wide (clearance <= 2.5 cm) and the GAM speed
# model saturates at its ceiling well below 0.1 m of clearance.
FREE_SPACE_CLEARANCE_M = 0.1


def model(model_input: SteeringModelInput):
    """Steering model using structured parameters with reference path MPC."""

    state_cog = model_input.state_cog
    pred_horizon = model_input.bump.pred_horizon
    interval = model_input.env.interval
    tunnel_info = (
        model_input.tunnel.tunnel_path,
        model_input.tunnel.tunnel_width,
        model_input.tunnel.top_wall,
        model_input.tunnel.bottom_wall,
    )

    cursor_pos_x, cursor_pos_y, cursor_vel_x, cursor_vel_y = state_cog
    tunnel_path, tunnel_width, top_wall, bottom_wall = tunnel_info

    if model_input.reference_path is not None:
        ref_path = model_input.reference_path
    else:
        ref_path = ReferencePath(tunnel_path, s=0.0, k=3)

    current_pos = np.array([cursor_pos_x, cursor_pos_y])
    theta_0 = ref_path.find_closest_theta(current_pos)

    current_acc = getattr(model_input, 'current_acc', (0.0, 0.0))
    if current_acc is None:
        current_acc = (0.0, 0.0)
    ax0, ay0 = current_acc

    state_0 = [cursor_pos_x, cursor_pos_y, cursor_vel_x, cursor_vel_y, ax0, ay0, theta_0]

    limits = {'acc_max': 100.0}

    corridor_bounds = getattr(model_input, 'corridor_bounds', None)
    if corridor_bounds is None:
        cartesian_constraints = getattr(model_input, 'cartesian_constraints', None)
        if cartesian_constraints is None and tunnel_width is not None:
            half_width = float(tunnel_width) / 2.0
            bound_value = half_width * 0.95
            corridor_bounds = (bound_value, bound_value)
    else:
        cartesian_constraints = getattr(model_input, 'cartesian_constraints', None)

    desired_speed = 0.12
    if model_input.planner_weights and 'desired_speed' in model_input.planner_weights:
        desired_speed = float(model_input.planner_weights['desired_speed'])

    speed_model = getattr(model_input, 'speed_model', None)
    anchor_s = getattr(model_input, 'anchor_s', None)
    if speed_model is None and anchor_s is None:
        raise ValueError("speed_model is required in SteeringModelInput "
                         "(or anchor_s for anchor-drive planning)")

    s0_init = theta_0
    # Project the look-ahead at the cursor's actual speed (not a fixed nominal)
    # so braking anticipation emerges from the horizon. desired_speed is just a
    # start-up floor (the cursor begins at rest).
    speed_now = float(np.hypot(cursor_vel_x, cursor_vel_y))
    v_proj = max(speed_now, desired_speed)
    s_estimated = s0_init + v_proj * interval * np.arange(1, pred_horizon + 1)

    clearance_profile = getattr(model_input, 'clearance_profile', None)
    curvature_rate_profile = getattr(model_input, 'curvature_rate_profile', None)
    curvature_profile = getattr(model_input, 'curvature_profile', None)

    clearance_at_s = np.full(pred_horizon, 0.01)
    kappa_at_s = np.zeros(pred_horizon)
    dkappa_at_s = np.zeros(pred_horizon)

    if clearance_profile is not None:
        s_prof, c_prof = clearance_profile
        clearance_at_s = np.interp(s_estimated, s_prof, c_prof)
    if curvature_profile is not None:
        s_prof, k_prof = curvature_profile
        kappa_at_s = np.interp(s_estimated, s_prof, k_prof)
    if curvature_rate_profile is not None:
        s_prof, r_prof = curvature_rate_profile
        dkappa_at_s = np.interp(s_estimated, s_prof, r_prof)

    anchor = None
    if anchor_s is not None:
        # Anchor-drive planning: no prescribed speed and no free-space
        # gate. ONE drive term everywhere — a via-point cost at the deadline
        # node aimed at the gaze anchor. In a tunnel the anchor recedes at
        # every replan (cruise emerges as lookahead / deadline); where the
        # environment stops constraining, the budget density vanishes, the
        # anchor rests on the goal and the same pursuit becomes pointing.
        speed_profile = None
        free_space_mask = None
        deadline_steps = int(getattr(model_input, 'deadline_steps', pred_horizon) or pred_horizon)
        k_deadline = int(np.clip(deadline_steps, 1, pred_horizon)) - 1
        # Initial progress schedule for the linearisation: the current
        # speed (gentle floor; re-linearised to self-consistency inside).
        s_sched0 = s0_init + max(speed_now, 0.05) * interval * np.arange(1, pred_horizon + 1)
        anchor = (float(np.clip(anchor_s, 0.0, ref_path.total_length)), k_deadline, s_sched0,
                  int(getattr(model_input, 'safety_steps', 0) or 0),
                  float(getattr(model_input, 'anchor_pace', 0.0) or 0.0))
    else:
        speed_profile = speed_model.compute_speed_profile(
            s_estimated, clearance_at_s, kappa_at_s, dkappa_at_s,
        )

        # Horizon nodes in unconstrained space (clearance far beyond any tunnel
        # half-width; the tunnel-adaptive speed model is saturated there) switch
        # the MPCC from corridor-following to goal-directed pointing.
        free_space_mask = clearance_at_s > FREE_SPACE_CLEARANCE_M

    weights_with_nc = dict(model_input.planner_weights) if model_input.planner_weights else {}
    weights_with_nc['nc0'] = model_input.bump.nc[0]
    weights_with_nc['nc1'] = model_input.bump.nc[1]
    weights_with_nc['target_radius'] = model_input.target_radius

    controls, opt_info = generate_mpcc(
        ref_path=ref_path,
        state_0=state_0,
        num_steps=pred_horizon,
        dt=interval,
        weights=weights_with_nc,
        limits=limits,
        speed_profile=speed_profile,
        desired_speed=desired_speed,
        corridor_bounds=corridor_bounds,
        cartesian_constraints=cartesian_constraints,
        free_space_mask=free_space_mask,
        anchor=anchor,
        curvature_profile=curvature_profile,
        warm_shift=int(getattr(model_input, 'warm_shift', 1) or 1),
    )

    jx = controls[:, 0]
    jy = controls[:, 1]

    A_vel = _build_A_vel_from_jerk(pred_horizon, interval)
    A_pos = _build_A_pos_from_jerk(pred_horizon, interval)

    t_vec = np.arange(1, pred_horizon + 1) * interval

    vx_free = cursor_vel_x + ax0 * t_vec
    vy_free = cursor_vel_y + ay0 * t_vec
    px_free = cursor_pos_x + cursor_vel_x * t_vec + 0.5 * ax0 * (t_vec**2)
    py_free = cursor_pos_y + cursor_vel_y * t_vec + 0.5 * ay0 * (t_vec**2)

    pos_x = px_free + A_pos @ jx
    pos_y = py_free + A_pos @ jy
    vel_x = vx_free + A_vel @ jx
    vel_y = vy_free + A_vel @ jy

    c_vel_x = np.insert(vel_x, 0, cursor_vel_x)
    c_vel_y = np.insert(vel_y, 0, cursor_vel_y)

    all_pos_x = np.insert(pos_x, 0, cursor_pos_x)
    all_pos_y = np.insert(pos_y, 0, cursor_pos_y)
    c_pos_dx = np.diff(all_pos_x)
    c_pos_dy = np.diff(all_pos_y)

    cursor_info = c_pos_dx, c_pos_dy, c_vel_x, c_vel_y

    ref_target = ref_path(theta_0)
    plan_debug = {
        "ideal_segment": (pos_x.tolist(), pos_y.tolist()),
        "anchor": None if anchor is None else (float(anchor[0]), int(anchor[1])),
        "target_waypoint": (float(ref_target[0]), float(ref_target[1])),
        "theta": float(theta_0),
        "opt_info": opt_info,
    }
    return cursor_info, plan_debug

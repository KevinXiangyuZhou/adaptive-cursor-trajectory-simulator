"""Steering model — adapts SteeringModelInput to the anchor-drive MPCC solver."""

import numpy as np
from .params import SteeringModelInput
from .reference_path import ReferencePath
from .mpcc_model import (
    generate_mpcc,
    reset_warm_start,
    _build_A_vel_from_jerk,
    _build_A_pos_from_jerk,
)


# Clearance above which a point is treated as unconstrained space.
# Study tunnels are <= 5 cm wide (clearance <= 2.5 cm); anything this wide is
# free space (pointing), not a corridor.
FREE_SPACE_CLEARANCE_M = 0.1


def model(model_input: SteeringModelInput):
    """Anchor-drive steering model: one MPCC solve toward the gaze anchor.

    No prescribed speed and no free-space gate. ONE drive term everywhere — a
    via-point cost at the deadline node aimed at the gaze anchor. In a tunnel
    the anchor recedes at every replan (cruise emerges as lookahead /
    deadline); where the environment stops constraining, the budget density
    vanishes, the anchor rests on the goal and the same pursuit becomes
    pointing.
    """

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

    current_acc = model_input.current_acc
    if current_acc is None:
        current_acc = (0.0, 0.0)
    ax0, ay0 = current_acc

    state_0 = [cursor_pos_x, cursor_pos_y, cursor_vel_x, cursor_vel_y, ax0, ay0, theta_0]

    corridor_bounds = model_input.corridor_bounds
    cartesian_constraints = model_input.cartesian_constraints
    if corridor_bounds is None and cartesian_constraints is None and tunnel_width is not None:
        half_width = float(tunnel_width) / 2.0
        bound_value = half_width * 0.95
        corridor_bounds = (bound_value, bound_value)

    desired_speed = 0.12
    if model_input.planner_weights and 'desired_speed' in model_input.planner_weights:
        desired_speed = float(model_input.planner_weights['desired_speed'])

    anchor_s = model_input.anchor_s
    if anchor_s is None:
        raise ValueError("anchor_s is required in SteeringModelInput "
                         "(anchor-drive planning is the only model; variants "
                         "live at git tag s14-variant-graveyard)")

    s0_init = theta_0
    speed_now = float(np.hypot(cursor_vel_x, cursor_vel_y))

    deadline_steps = int(model_input.deadline_steps or pred_horizon)
    k_deadline = int(np.clip(deadline_steps, 1, pred_horizon)) - 1
    # Initial progress schedule for the linearisation: the current
    # speed (gentle floor; re-linearised to self-consistency inside).
    s_sched0 = s0_init + max(speed_now, 0.05) * interval * np.arange(1, pred_horizon + 1)

    controls, opt_info = generate_mpcc(
        ref_path=ref_path,
        state_0=state_0,
        num_steps=pred_horizon,
        dt=interval,
        weights=dict(model_input.planner_weights) if model_input.planner_weights else {},
        anchor_s=float(np.clip(anchor_s, 0.0, ref_path.total_length)),
        k_deadline=k_deadline,
        s_schedule=s_sched0,
        anchor_pace=float(model_input.anchor_pace or 0.0),
        corridor_bounds=corridor_bounds,
        cartesian_constraints=cartesian_constraints,
        desired_speed=desired_speed,
        warm_shift=int(model_input.warm_shift or 1),
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
        "anchor": (float(np.clip(anchor_s, 0.0, ref_path.total_length)), k_deadline),
        "target_waypoint": (float(ref_target[0]), float(ref_target[1])),
        "theta": float(theta_0),
        "opt_info": opt_info,
    }
    return cursor_info, plan_debug

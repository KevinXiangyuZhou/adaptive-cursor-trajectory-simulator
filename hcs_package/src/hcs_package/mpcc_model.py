"""Anchor-drive MPCC solver for cursor steering.

Solves for jerk controls that carry the cursor to the gaze anchor (a progress
via-point on the reference path) by the plan deadline, subject to corridor and
Cartesian constraints. Speed is never prescribed: cruise emerges as
lookahead / deadline.

Pruned variants (GAM speed-profile tracking, the free-space LQR /
free_space_mask branch, the goal_precision well, coast safety) are preserved
at git tag ``s14-variant-graveyard``.
"""

import os
import numpy as np
from scipy.optimize import minimize

from .constraints import ConstraintType, RectangleConstraint, PolygonConstraint
from .constraint_utils import _point_in_polygon, _distance_to_polygon_boundary


# Anchor-drive progress linearisation: max fixed-point passes and the
# schedule change (m) below which the linearisation is accepted.
ANCHOR_RELIN_PASSES = 8
ANCHOR_RELIN_TOL = 1e-3

_warm_start_cache = {
    'x_prev': None,
    'num_steps': None,
}


def reset_warm_start():
    """Reset the warm-start cache (call at the start of each new trajectory)."""
    _warm_start_cache['x_prev'] = None
    _warm_start_cache['num_steps'] = None


def evaluate_tracking_errors(ref_path, state_0, controls, num_steps, dt):
    """Compute per-horizon contour/lag errors used in the MPCC objective.

    Args:
        ref_path: ReferencePath instance.
        state_0: Initial state [px, py, vx, vy, ax, ay, s].
        controls: (N, 3) array of [jx, jy, vs] from generate_mpcc.
        num_steps: Horizon length N.
        dt: Time step.

    Returns:
        dict with arrays ``px``, ``py``, ``s_traj``, ``e_contour``, ``e_lag``,
        and boolean mask ``at_path_end`` (True where s_traj >= path length).
    """
    px0, py0, vx0, vy0, ax0, ay0, s0 = state_0
    jx = controls[:, 0]
    jy = controls[:, 1]
    vs = controls[:, 2]

    A_pos_mat = _build_A_pos_from_jerk(num_steps, dt)
    t_vec = np.arange(1, num_steps + 1) * dt
    t2_vec = 0.5 * t_vec ** 2

    px = px0 + vx0 * t_vec + ax0 * t2_vec + A_pos_mat @ jx
    py = py0 + vy0 * t_vec + ay0 * t2_vec + A_pos_mat @ jy

    S_mat = np.tril(np.ones((num_steps, num_steps))) * dt
    s_traj = s0 + S_mat @ vs
    s_end = ref_path.total_length

    e_contour = np.zeros(num_steps)
    e_lag = np.zeros(num_steps)
    for k in range(num_steps):
        ref_k = ref_path(float(s_traj[k]))
        tangent = ref_path.tangent(s_traj[k])
        cos_phi, sin_phi = tangent[0], tangent[1]
        pos_error = np.array([px[k], py[k]], dtype=float) - ref_k
        R = np.array([
            [sin_phi, -cos_phi],
            [-cos_phi, -sin_phi],
        ], dtype=float)
        e_k = R @ pos_error
        e_contour[k] = e_k[0]
        e_lag[k] = e_k[1]

    return {
        'px': px,
        'py': py,
        's_traj': s_traj,
        'e_contour': e_contour,
        'e_lag': e_lag,
        'at_path_end': s_traj >= s_end - 1e-9,
    }


def _build_A_acc(num_steps, dt):
    """Matrix mapping jerk to acceleration via integration."""
    A_acc = np.tril(np.ones((num_steps, num_steps))) * dt
    return A_acc


def _build_A_vel_from_jerk(num_steps, dt):
    """Matrix mapping jerk to velocity."""
    A_v = np.zeros((num_steps, num_steps))
    for k in range(num_steps):
        for i in range(k + 1):
            A_v[k, i] = (k - i + 0.5) * (dt ** 2)
    return A_v


def _build_A_pos_from_jerk(num_steps, dt):
    """Matrix mapping jerk to position."""
    A_p = np.zeros((num_steps, num_steps))
    for i in range(num_steps):
        dp, dv, da = 0.0, 0.0, 0.0
        for k in range(num_steps):
            j_val = 1.0 if k == i else 0.0
            term_p = dp + dv * dt + 0.5 * da * (dt**2) + (1.0/6.0) * j_val * (dt**3)
            term_v = dv + da * dt + 0.5 * j_val * (dt**2)
            term_a = da + j_val * dt
            dp, dv, da = term_p, term_v, term_a
            A_p[k, i] = dp
    return A_p


def generate_mpcc(
    ref_path,
    state_0,
    num_steps,
    dt,
    weights,
    anchor_s,
    k_deadline,
    s_schedule,
    anchor_pace=0.0,
    corridor_bounds=None,
    cartesian_constraints=None,
    desired_speed=1.0,
    warm_shift=1,
):
    """
    Generate an anchor-drive MPCC plan.

    The objective is the SAME everywhere: jerk effort, contour tracking,
    boundary penalties, and one drive — a progress via-point

        goal * ((anchor_s - s_{k_deadline}) / lead)^2

    at the deadline node only, in the path's own arc-length coordinate (the
    drive acts along the path, never across a corner). Progress is KINEMATIC —
    the plan's along-tangent velocity integrated along the path, with the
    tangents taken on a fixed per-solve progress schedule (s_schedule, the
    lookahead at the current speed) so progress is LINEAR in the plan's
    velocities: a recursion through tangent(s_k) is chaotic on wiggly paths
    (amplification ~ v*dt*kappa per node) and wrecks finite-difference
    gradients. No virtual progress variables, no lag term; the drive acts on
    the jerk decision variables directly (a via-point routed through a stiff
    lag coupling stalls a first-order solver; a Cartesian via-point pulls
    across corners). The anchor is passed through, not settled at: nodes after
    the deadline carry no attractor and coast (the old plan keeps executing
    through the anchor during the post-arrival latency, as the gaze data
    show); with anchor_pace > 0 the schedule instead continues at the
    fixation's pace beyond the deadline (pace-holding tail). The reference
    path is a curve WITH ENDS: where progress saturates at the path end the
    tracking error is the full distance to the end point, so overshooting the
    goal costs contour error and braking into the target emerges. Speed is
    never prescribed: cruise emerges as lookahead / deadline, and pointing is
    the same pursuit with the anchor resting on the path end.

    Args:
        ref_path: ReferencePath object.
        state_0: Initial state [px, py, vx, vy, ax, ay, s].
        num_steps: Prediction horizon (N).
        dt: Time step.
        weights: Dictionary of MPCC weights (jerk, goal, contour, lag_anchor,
            constraint, acc_max, ...).
        anchor_s: Arc length of the gaze anchor (the via-point target).
        k_deadline: 0-based horizon node index at which the plan must be at
            the anchor.
        s_schedule: (N,) initial progress schedule for the tangent
            linearisation (re-linearised to a fixed point internally).
        anchor_pace: Progress pace (m/s) of the pace-holding tail beyond the
            deadline node; 0 disables the tail.
        corridor_bounds: Tuple (bound_left, bound_right) for path-relative
            corridor constraints.
        cartesian_constraints: List[ConstraintRegion] for world-space
            constraints.
        desired_speed: Nominal speed used only to estimate which horizon nodes
            are still on the path for the Cartesian-constraint mask.
        warm_shift: Steps executed since the previous solve (warm-start shift).

    Returns:
        controls: (N, 3) array of [jx, jy, vs] (vs = kinematic progress rate).
        opt_info: optimization result info dict.
    """
    if weights is None:
        weights = {}

    w_jerk = weights.get('jerk', 1.5e-6)
    w_constraint = weights.get('constraint', 1e3)
    w_corridor = w_constraint  # one fitted weight for all boundary penalties
    # Peak-acceleration bound: the hand cannot produce arbitrary acceleration;
    # a stiff hinge on |a_k| above acc_max (m/s^2). One physiological
    # constant, universal (pointing, corners, bends): cornering speed is
    # capped at sqrt(acc_max * cut radius), so the corner phenotype's width
    # dependence comes from the corner-cut radius.
    acc_max_w = float(weights.get('acc_max', 0.0) or 0.0)
    w_acc = float(weights.get('acc_weight', 1e4))
    # Chance-constraint corridor tightening (config-gated, chance_z > 0): the wall
    # hinge activates early by the lateral scatter that along-track motor noise
    # (nc0, the 10x channel) acquires where the tangent turns:
    #   tighten_k = 0.5 * z * nc0 * |kappa(s_k)| * (v_k * T_ol)^2
    # T_ol = the actual open-loop interval (replan latency). Zero on straights;
    # binds as speed * curvature outruns the room left by execution scatter.
    chance_z = float(weights.get('chance_z', 0.0) or 0.0)
    chance_nc0 = float(weights.get('chance_nc0', 0.2))
    chance_tol = float(weights.get('chance_t_ol', 0.2))
    # Enforced as a penalty-method CONSTRAINT: its own stiff hinge (like acc_weight),
    # not the fitted wall weight — a behavioral weight cannot enforce a constraint.
    w_chance = float(weights.get('chance_weight', 1e4))
    w_contour = weights.get('contour', 1.0)
    # Lateral adherence ('contour') and along-path consistency ('lag_anchor')
    # are different things — the first is a steering STRATEGY (how much the
    # planner cuts inside the walls), the second is the coupling that lets the
    # progress via-point drive the cursor and must be stiff. A single
    # full-distance weight forces both stiff and removes corner-cutting.
    # None = legacy full-distance tracking with 'contour'.
    w_lag_anchor = weights.get('lag_anchor', None)
    w_goal = weights.get('goal', 1.0)

    anchor_s_target = float(np.clip(float(anchor_s), 0.0, ref_path.total_length))
    k_deadline = int(np.clip(int(k_deadline), 0, num_steps - 1))
    anchor_lead_norm = max(anchor_s_target - float(state_0[6]), 0.005)
    anchor_pace = float(anchor_pace or 0.0)
    s_sched = np.clip(np.asarray(s_schedule, dtype=float), 0.0, ref_path.total_length)
    if s_sched.shape[0] != num_steps:
        # Pad with the last value (np.resize tiles cyclically — review-flagged latent trap).
        if s_sched.shape[0] < num_steps:
            s_sched = np.concatenate([s_sched, np.full(num_steps - s_sched.shape[0], s_sched[-1] if len(s_sched) else 0.0)])
        else:
            s_sched = s_sched[:num_steps]

    px0, py0, vx0, vy0, ax0, ay0, s0 = state_0

    def _tangents_on(sched):
        # Tangent at the previous node of the schedule (node 0: at s0).
        return ref_path.tangents(np.concatenate([[state_0[6]], sched[:-1]]))
    # Mutable holder so the objective closure sees re-linearised tangents
    # (fixed-point iteration below).
    tan_sched_box = [_tangents_on(s_sched)]

    SCALE_JERK = 1000.0

    A_acc_mat = _build_A_acc(num_steps, dt)
    A_vel_mat = _build_A_vel_from_jerk(num_steps, dt)
    A_pos_mat = _build_A_pos_from_jerk(num_steps, dt)

    t_vec = np.arange(1, num_steps + 1) * dt
    t2_vec = 0.5 * t_vec ** 2

    # Free (zero-jerk) response
    px_free = px0 + vx0 * t_vec + ax0 * t2_vec
    py_free = py0 + vy0 * t_vec + ay0 * t2_vec
    vx_free = vx0 + ax0 * t_vec
    vy_free = vy0 + ay0 * t_vec
    ax_free = np.full(num_steps, ax0)
    ay_free = np.full(num_steps, ay0)

    # x = [jx_0..N-1, jy_0..N-1]; no virtual progress variables (progress is
    # kinematic).
    n_blocks = 2
    n_vars = n_blocks * num_steps
    idx_jx = slice(0, num_steps)
    idx_jy = slice(num_steps, 2 * num_steps)
    s_end_total = float(ref_path.total_length)

    if cartesian_constraints is not None:
        s_estimated = s0 + desired_speed * dt * np.arange(1, num_steps + 1)
        cartesian_active = s_estimated < ref_path.total_length
    else:
        cartesian_active = None

    def unpack_x(x):
        jx = x[idx_jx] * SCALE_JERK
        jy = x[idx_jy] * SCALE_JERK
        return jx, jy

    def kinematic_progress(vx, vy):
        """Arc-length progress from the plan's along-tangent velocity
        (trapezoid over each step) with tangents on the fixed schedule —
        linear in the velocities, hence smooth in the jerk variables."""
        tan_sched = tan_sched_box[0]
        vpx = np.concatenate([[vx0], vx[:-1]])
        vpy = np.concatenate([[vy0], vy[:-1]])
        v_t = 0.5 * ((vpx + vx) * tan_sched[:, 0] + (vpy + vy) * tan_sched[:, 1])
        # Unclipped: progress may run past the path end, so overshooting the
        # goal anchor is lateness in reverse and the via-point drive brakes
        # into the target (path lookups clip internally; a clipped progress
        # left overshoot to the weak tracking term -> hunting at the goal).
        return s0 + np.cumsum(v_t) * dt

    def objective(x):
        jx, jy = unpack_x(x)

        # 1. Jerk smoothness
        j_cost = np.sum(jx**2 + jy**2) * w_jerk

        # 2. Kinematic progress; no speed target and no velocity damping
        # (the free_velocity |v|^2 term was removed with the finalized cycle
        # design: the GAM traversal deadline sets the pace, and a quadratic
        # drag on every node only fought the via-point drive).
        vx = vx_free + A_vel_mat @ jx
        vy = vy_free + A_vel_mat @ jy
        s_traj = kinematic_progress(vx, vy)

        prog_cost = 0.0
        if acc_max_w > 0.0:
            a_mag = np.hypot(ax_free + A_acc_mat @ jx, ay_free + A_acc_mat @ jy)
            prog_cost += w_acc * float(np.sum(np.maximum(a_mag - acc_max_w, 0.0) ** 2))

        # 3. Tracking error + via-point drive
        px = px_free + A_pos_mat @ jx
        py = py_free + A_pos_mat @ jy

        ref_pts = ref_path(s_traj).T
        if ref_pts.shape != (num_steps, 2):
            ref_pts = np.zeros((num_steps, 2))
            for k in range(num_steps):
                ref_pts[k] = ref_path(float(s_traj[k]))

        rx, ry = ref_pts[:, 0], ref_pts[:, 1]
        tangents_all = ref_path.tangents(s_traj)
        tracking_cost = 0.0

        for k in range(num_steps):
            pos_k = np.array([px[k], py[k]], dtype=float)
            ref_k = np.array([rx[k], ry[k]], dtype=float)
            pos_error = pos_k - ref_k

            cos_phi = tangents_all[k, 0]
            sin_phi = tangents_all[k, 1]

            R = np.array([
                [sin_phi, -cos_phi],
                [-cos_phi, -sin_phi]
            ], dtype=float)

            e_k = R @ pos_error

            # Tracking error = full distance to the progress point (one
            # term, one weight). Along-path mismatch is what appears when
            # the plan coasts past a corner (progress freezes while the
            # cursor runs on) or past the path end; lateral mismatch is
            # the contour error. No special cases.
            if w_lag_anchor is None:
                tracking_cost += w_contour * float(pos_error @ pos_error)
            else:
                tracking_cost += w_contour * e_k[0]**2 + float(w_lag_anchor) * e_k[1]**2
            if k == k_deadline:
                # Progress via-point: be at the anchor at the deadline.
                # Lateness is normalised by the planned lead (dimensionless)
                # so the drive's stiffness does not scale with lead^2 — a
                # 60 mm and a 300 mm plan are held to the same fraction.
                tracking_cost += w_goal * ((anchor_s_target - float(s_traj[k])) / anchor_lead_norm)**2
            elif k > k_deadline and anchor_pace > 0.0:
                s_tail = min(anchor_s_target + anchor_pace * dt * (k - k_deadline), ref_path.total_length)
                tracking_cost += w_goal * ((s_tail - float(s_traj[k])) / anchor_lead_norm)**2

            # 4a. Path-relative corridor penalty
            if corridor_bounds is not None:
                b_left_in, b_right_in = corridor_bounds
                s_b = float(min(max(s_traj[k], 0.0), s_end_total))
                w_left = b_left_in(s_b) if callable(b_left_in) else float(b_left_in)
                w_right = b_right_in(s_b) if callable(b_right_in) else float(b_right_in)
                if chance_z > 0.0:
                    kap_k = abs(float(ref_path.curvature(s_b)))
                    v_sq = float(vx[k] * vx[k] + vy[k] * vy[k])
                    tighten = 0.5 * chance_z * chance_nc0 * kap_k * v_sq * (chance_tol ** 2)
                    # The z-sigma scatter ellipse must FIT in the room: hinge on
                    # (|e_c| + tighten - room). At e_c=0 this binds as soon as the
                    # scatter alone exceeds the room, capping speed at
                    # v <= sqrt(2*room/(z*nc0*kappa*T_ol^2)).
                    tracking_cost += w_chance * (max(0.0, e_k[0] + tighten - w_left) ** 2
                                                 + max(0.0, -e_k[0] + tighten - w_right) ** 2)
                violation_left = max(0.0, e_k[0] - w_left)
                violation_right = max(0.0, -e_k[0] - w_right)
                tracking_cost += w_corridor * (violation_left**2 + violation_right**2)

            # 4b. Cartesian constraint penalty
            if cartesian_constraints and cartesian_active[k]:
                pk_x, pk_y = px[k], py[k]
                for region in cartesian_constraints:
                    geom = region.geometry
                    if isinstance(geom, RectangleConstraint):
                        margin = 0.0
                        x_min, x_max = geom.x, geom.x + geom.width
                        y_min, y_max = geom.y, geom.y + geom.height

                        if region.constraint_type == ConstraintType.KEEP_IN:
                            dist_left = pk_x - x_min
                            dist_right = x_max - pk_x
                            dist_bottom = pk_y - y_min
                            dist_top = y_max - pk_y

                            viol_left = max(0.0, margin - dist_left)
                            viol_right = max(0.0, margin - dist_right)
                            viol_bottom = max(0.0, margin - dist_bottom)
                            viol_top = max(0.0, margin - dist_top)

                            tracking_cost += w_constraint * (viol_left**2 + viol_right**2 + viol_bottom**2 + viol_top**2)

                        else:  # KEEP_OUT
                            if x_min <= pk_x <= x_max and y_min <= pk_y <= y_max:
                                dist = min(pk_x - x_min, x_max - pk_x,
                                           pk_y - y_min, y_max - pk_y)
                                tracking_cost += w_constraint * (dist + margin)**2
                            else:
                                ddx = max(x_min - pk_x, 0.0, pk_x - x_max)
                                ddy = max(y_min - pk_y, 0.0, pk_y - y_max)
                                dist_outside = np.sqrt(ddx**2 + ddy**2)
                                if dist_outside < margin:
                                    tracking_cost += w_constraint * (margin - dist_outside)**2

                    elif isinstance(geom, PolygonConstraint):
                        point = np.array([pk_x, pk_y])
                        vertices = np.array(geom.vertices)
                        inside = _point_in_polygon(point, vertices)
                        dist_to_boundary = _distance_to_polygon_boundary(point, vertices)

                        if region.constraint_type == ConstraintType.KEEP_IN:
                            if not inside:
                                tracking_cost += w_constraint * dist_to_boundary**2
                        else:  # KEEP_OUT
                            if inside:
                                tracking_cost += w_constraint * dist_to_boundary**2

        return j_cost + prog_cost + tracking_cost

    bounds = []
    bounds.extend([(None, None)] * n_vars)

    x0_cold = np.zeros(n_vars)

    # Warm-start: shift the previous solution forward by the number of steps
    # executed since that solve (1 under per-step replanning; larger under
    # intermittent replanning). The shifted tail is truncated or padded
    # (zero jerk) to the new horizon length, so the warm start survives the
    # per-solve horizon changes of budget mode.
    x0_warm = None
    x_prev = _warm_start_cache.get('x_prev')
    prev_n = _warm_start_cache.get('num_steps')
    k_shift = int(warm_shift)
    if (x_prev is not None and prev_n is not None and 1 <= k_shift < prev_n
            and len(x_prev) == n_blocks * prev_n):
        prev_jx = x_prev[:prev_n]
        prev_jy = x_prev[prev_n:2*prev_n]

        def _fit(tail, fill):
            if len(tail) >= num_steps:
                return tail[:num_steps]
            return np.append(tail, np.full(num_steps - len(tail), fill))

        x0_warm = np.zeros(n_vars)
        x0_warm[idx_jx] = _fit(prev_jx[k_shift:], 0.0)
        x0_warm[idx_jy] = _fit(prev_jy[k_shift:], 0.0)

    x0_guess = x0_cold
    if x0_warm is not None:
        cost_cold = objective(x0_cold)
        cost_warm = objective(x0_warm)
        if cost_warm < cost_cold:
            x0_guess = x0_warm

    result = minimize(
        objective,
        x0_guess,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-6, 'gtol': 1e-5, 'maxfun': 5000}
    )
    # Fixed-point iteration on the progress linearisation: re-take the
    # tangents on the solved plan's own progress and re-solve
    # (warm-started) until the schedule is self-consistent.
    # Monotone safeguard: the fixed-point iteration is not a descent
    # method (each pass changes the objective's linearisation), so keep the
    # best (cost, plan, schedule) seen and return that if a later pass
    # diverges — a numerical guard against runaway plans, no behaviour.
    best = (float(result.fun), result.x.copy(), s_sched.copy(), tan_sched_box[0].copy())
    for _ in range(ANCHOR_RELIN_PASSES - 1):
        jx_p, jy_p = unpack_x(result.x)
        s_new = kinematic_progress(vx_free + A_vel_mat @ jx_p, vy_free + A_vel_mat @ jy_p)
        if float(np.max(np.abs(s_new - s_sched))) < ANCHOR_RELIN_TOL:
            break
        s_sched = s_new
        tan_sched_box[0] = _tangents_on(s_sched)
        result = minimize(
            objective, result.x, method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-6, 'gtol': 1e-5, 'maxfun': 5000})
        if np.isfinite(result.fun) and float(result.fun) < best[0]:
            best = (float(result.fun), result.x.copy(), s_sched.copy(), tan_sched_box[0].copy())
    if not (np.isfinite(result.fun) and float(result.fun) <= best[0]):
        s_sched = best[2]; tan_sched_box[0] = best[3]
        result.x = best[1]; result.fun = best[0]

    _warm_start_cache['x_prev'] = result.x.copy()
    _warm_start_cache['num_steps'] = num_steps

    jx_opt, jy_opt = unpack_x(result.x)
    # Report the kinematic progress rate in the vs column for callers.
    vx_o = vx_free + A_vel_mat @ jx_opt
    vy_o = vy_free + A_vel_mat @ jy_opt
    vs_opt = np.diff(np.concatenate([[s0], kinematic_progress(vx_o, vy_o)])) / dt
    controls = np.column_stack((jx_opt, jy_opt, vs_opt))

    opt_info = {
        'success': result.success,
        'cost': result.fun,
        'message': result.message,
        'nit': result.nit
    }

    return controls, opt_info

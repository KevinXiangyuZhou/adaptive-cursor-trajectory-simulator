"""Model Predictive Contouring Control (MPCC) solver for cursor steering.

Receives a pre-computed speed profile (from a SpeedModel) and solves for
optimal jerk controls.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d

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

    Useful for diagnosing whether the lag term is still driving catch-up when
    virtual progress ``s_traj`` has reached the end of the reference path.

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


_lqr_cache = {}


def _free_space_lqr_value(dt, q, r, rho, w_jerk):
    """Infinite-horizon discrete LQR value matrix P (3x3) for one axis of the
    jerk-driven triple integrator x=[e, v, a], u=j, with stage cost
    q e^2 + r v^2 + rho a^2 + w_jerk j^2. Cached per weight tuple."""
    key = (round(dt, 6), q, r, rho, w_jerk)
    P = _lqr_cache.get(key)
    if P is None:
        from scipy.linalg import solve_discrete_are
        A = np.array([[1.0, dt, 0.5 * dt * dt], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
        B = np.array([[dt ** 3 / 6.0], [0.5 * dt * dt], [dt]])
        Q = np.diag([max(q, 0.0), max(r, 0.0), max(rho, 0.0)])
        R = np.array([[max(w_jerk, 1e-12)]])
        P = solve_discrete_are(A, B, Q, R)
        _lqr_cache[key] = P
    return P


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
    limits,
    speed_profile,
    desired_speed=1.0,
    corridor_bounds=None,
    cartesian_constraints=None,
    free_space_mask=None,
    anchor=None,
    curvature_profile=None,
    warm_shift=1,
):
    """
    Generate MPCC (Model Predictive Contouring Control) plan.

    Args:
        ref_path: ReferencePath object.
        state_0: Initial state [px, py, vx, vy, ax, ay, s].
        num_steps: Prediction horizon (N).
        dt: Time step.
        weights: Dictionary of MPCC weights (jerk, progress, tracking, wall, etc.).
        limits: Dictionary of limits (acc_max).
        speed_profile: 1-D array of desired speeds at the horizon positions (N,).
            Pre-computed by a SpeedModel externally.
        desired_speed: Base desired speed (used for scaling only).
        corridor_bounds: Tuple (bound_left, bound_right) for path-relative
            corridor constraints.
        cartesian_constraints: List[ConstraintRegion] for world-space constraints.
        anchor: Optional (anchor_s, k_deadline, s_schedule) for anchor-drive planning
            (speed_profile None, free_space_mask ignored). The objective is
            then the SAME everywhere: jerk effort, contour tracking,
            boundary penalties, velocity damping free_velocity*|v|^2 at
            every node, and one drive — a progress via-point
                goal * (anchor_s - s_{k_deadline})^2
            at the deadline node only, in the path's own arc-length
            coordinate (the drive acts along the path, never across a
            corner). Progress is KINEMATIC — the plan's along-tangent
            velocity integrated along the path, with the tangents taken on
            a fixed per-solve progress schedule (s_schedule, the lookahead
            at the current speed) so progress is LINEAR in the plan's
            velocities: a recursion through tangent(s_k) is chaotic on
            wiggly paths (amplification ~ v*dt*kappa per node) and wrecks
            finite-difference gradients. No virtual progress variables, no
            lag term; the drive acts on the jerk decision variables
            directly (a via-point routed through a stiff lag coupling
            stalls a first-order solver; a Cartesian via-point pulls across
            corners). The anchor is passed through,
            not settled at: nodes after the deadline carry no attractor
            and coast (the old plan keeps executing through the anchor
            during the post-arrival latency, as the gaze data show). The
            reference path is a curve WITH ENDS: where progress saturates
            at the path end the tracking error is the full distance to the
            end point, so overshooting the goal costs contour error and
            braking into the target emerges. Speed is never prescribed:
            cruise emerges as lookahead / deadline, and pointing is the
            same pursuit with the anchor resting on the path end.
        free_space_mask: Optional bool array (N,). True at horizon nodes that
            lie in unconstrained space (clearance far larger than any tunnel).
            At those nodes the corridor-following machinery is replaced by
            goal-directed pointing with a linear-quadratic objective:
              * stage cost  goal*|p-goal|^2 + free_velocity*|v|^2
                            + free_accel*|a|^2 + jerk*|j|^2
                (the tunnel speed-profile term and the lag term are off);
              * terminal cost x_N' P x_N (per axis, x=[p-goal, v, a]) where P
                is the infinite-horizon LQR value function of that same stage
                cost for the jerk-driven triple integrator. With it the
                receding-horizon planner reproduces the infinite-horizon LQR
                regardless of horizon length, so free-space behaviour is set
                by the closed-loop poles of (goal, free_velocity, free_accel,
                jerk) alone: straight paths, peak speed proportional to
                distance, MT growing with log(D/R), a single-peaked bell whose
                shape does not depend on D. Nothing prescribes a cruise speed
                and no per-horizon heuristics are needed.

    Returns:
        controls: (N, 3) array of [jx, jy, vs].
        opt_info: optimization result info dict.
    """
    if weights is None:
        weights = {}

    w_jerk = weights.get('jerk', 1.5e-6)
    w_progress = weights.get('progress', 1.0e-5)
    w_constraint = weights.get('constraint', 1e3)
    w_corridor = w_constraint  # one fitted weight for all boundary penalties
    # Coast-safety hinge weight (anchor-drive terminal safety); a constraint
    # must be enforced as one, so this defaults to the boundary weight but
    # is expected to sit far above it.
    w_safety = weights.get('safety', w_constraint)
    # Peak-acceleration bound (anchor-drive): the hand cannot produce
    # arbitrary acceleration; a stiff hinge on |a_k| above acc_max (m/s^2).
    # One physiological constant, universal (pointing, corners, bends):
    # cornering speed is capped at sqrt(acc_max * cut radius), so the
    # corner phenotype's width dependence comes from the corner-cut radius.
    acc_max_w = float(weights.get('acc_max', 0.0) or 0.0)
    w_acc = float(weights.get('acc_weight', 1e4))
    w_contour = weights.get('contour', 1.0)
    w_lag = weights.get('lag', 0.1)
    # Free-space (pointing) LQ weights: q on |p-goal|^2, r on |v|^2, rho on
    # |a|^2 (jerk weight shared with the tunnel objective). Provisional
    # defaults; to be replaced by formal fitting to the pointing data.
    w_goal = weights.get('goal', 1.0)
    w_free_velocity = weights.get('free_velocity', 0.08)
    w_free_accel = weights.get('free_accel', 0.0)
    anchor_mode = anchor is not None
    if anchor_mode:
        anchor_s_target = float(np.clip(float(anchor[0]), 0.0, ref_path.total_length))
        k_deadline = int(np.clip(int(anchor[1]), 0, num_steps - 1))
        free_space_mask = None
        s_sched = np.clip(np.asarray(anchor[2], dtype=float), 0.0, ref_path.total_length)
        # Coast-safety: number of latency steps over which the deadline
        # state's ballistic continuation must stay inside the corridor.
        n_safety = int(anchor[3]) if len(anchor) > 3 and anchor[3] else 0
        t_safety = dt * np.arange(1, n_safety + 1)
        if s_sched.shape[0] != num_steps:
            # Pad with the last value (np.resize tiles cyclically — review-flagged latent trap).
            if s_sched.shape[0] < num_steps:
                s_sched = np.concatenate([s_sched, np.full(num_steps - s_sched.shape[0], s_sched[-1] if len(s_sched) else 0.0)])
            else:
                s_sched = s_sched[:num_steps]

        def _tangents_on(sched):
            # Tangent at the previous node of the schedule (node 0: at s0).
            return ref_path.tangents(np.concatenate([[state_0[6]], sched[:-1]]))
        # Mutable holder so the objective closure sees re-linearised tangents
        # (fixed-point iteration below).
        tan_sched_box = [_tangents_on(s_sched)]
    if free_space_mask is None:
        free_mask = np.zeros(num_steps, dtype=bool)
    else:
        free_mask = np.asarray(free_space_mask, dtype=bool)
        if free_mask.shape[0] != num_steps:
            free_mask = np.resize(free_mask, num_steps)
    any_free = bool(free_mask.any())
    terminal_lqr = any_free and bool(free_mask[-1])
    if any_free:
        _p_goal = ref_path(ref_path.total_length)
        goal_xy = np.array([float(_p_goal[0]), float(_p_goal[1])])
    if terminal_lqr:
        P_lqr = _free_space_lqr_value(dt, w_goal, w_free_velocity, w_free_accel, w_jerk)

    # Light Gaussian smoothing prevents chasing step-like changes over the short horizon
    if speed_profile is None:
        speed_target = None
    else:
        speed_target = np.asarray(speed_profile, dtype=float).copy()
        if len(speed_target) > 3:
            speed_target = gaussian_filter1d(speed_target, sigma=1.5)

    SCALE_JERK = 1000.0
    SCALE_VS = max(0.1, desired_speed)


    px0, py0, vx0, vy0, ax0, ay0, s0 = state_0

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

    # x = [jx_0..N-1, jy_0..N-1, vs_0..N-1]; anchor mode has no virtual
    # progress variables (progress is kinematic), so x = [jx, jy] only.
    n_blocks = 2 if anchor_mode else 3
    n_vars = n_blocks * num_steps
    idx_jx = slice(0, num_steps)
    idx_jy = slice(num_steps, 2 * num_steps)
    idx_vs = slice(2 * num_steps, 3 * num_steps)
    s_end_total = float(ref_path.total_length)

    S_mat = np.tril(np.ones((num_steps, num_steps))) * dt

    if cartesian_constraints is not None:
        s_estimated = s0 + desired_speed * dt * np.arange(1, num_steps + 1)
        cartesian_active = s_estimated < ref_path.total_length
    else:
        cartesian_active = None

    def unpack_x(x):
        jx = x[idx_jx] * SCALE_JERK
        jy = x[idx_jy] * SCALE_JERK
        vs = None if anchor_mode else x[idx_vs] * SCALE_VS
        return jx, jy, vs

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
        jx, jy, vs = unpack_x(x)

        # 1. Jerk smoothness (+ free-space acceleration/effort cost)
        j_cost = np.sum(jx**2 + jy**2) * w_jerk
        if any_free:
            ax_h = ax_free + A_acc_mat @ jx
            ay_h = ay_free + A_acc_mat @ jy
            if w_free_accel > 0.0:
                j_cost += w_free_accel * float(np.sum(np.where(free_mask, ax_h**2 + ay_h**2, 0.0)))

        # 2. Progress / speed tracking
        vx = vx_free + A_vel_mat @ jx
        vy = vy_free + A_vel_mat @ jy
        s_traj = kinematic_progress(vx, vy) if anchor_mode else s0 + S_mat @ vs
        physical_speed = np.sqrt(vx**2 + vy**2)

        if anchor_mode:
            # Velocity damping everywhere; no speed target.
            prog_cost = w_free_velocity * float(np.sum(vx**2 + vy**2))
            if acc_max_w > 0.0:
                a_mag = np.hypot(ax_free + A_acc_mat @ jx, ay_free + A_acc_mat @ jy)
                prog_cost += w_acc * float(np.sum(np.maximum(a_mag - acc_max_w, 0.0) ** 2))
        else:
            speed_error = physical_speed - speed_target
            if any_free:
                v_sq = vx**2 + vy**2
                prog_cost = float(np.sum(np.where(
                    free_mask, w_free_velocity * v_sq, w_progress * speed_error**2)))
            else:
                prog_cost = np.sum(speed_error**2) * w_progress

        # 3. Contour + lag tracking error
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
            e_contour = e_k[0]
            e_lag = e_k[1]

            if anchor_mode and n_safety > 0 and k == k_deadline and corridor_bounds is not None:
                # Terminal safety across the intermittency: the plan hands
                # over at the deadline and no new plan can arrive for the
                # latency tau. The handover state must be one whose
                # ballistic continuation p + v t (t <= tau) stays inside the
                # walls — a recursive-feasibility hedge. Zero on straights
                # and in free space; on a bend it caps speed at
                # sqrt(2 m / (kappa tau^2)) with no new constant.
                vt_k = vx[k] * cos_phi + vy[k] * sin_phi
                s_e = np.clip(float(s_traj[k]) + vt_k * t_safety, 0.0, s_end_total)
                ref_e = ref_path(s_e)
                ref_e = np.asarray(ref_e, dtype=float).reshape(2, -1).T if np.ndim(ref_e) > 1 else np.atleast_2d(ref_e)
                tan_e = ref_path.tangents(s_e)
                b_l, b_r = corridor_bounds
                for i in range(n_safety):
                    ex = px[k] + vx[k] * t_safety[i] - ref_e[i, 0]
                    ey = py[k] + vy[k] * t_safety[i] - ref_e[i, 1]
                    e_c = tan_e[i, 1] * ex - tan_e[i, 0] * ey
                    wl = b_l(s_e[i]) if callable(b_l) else float(b_l)
                    wr = b_r(s_e[i]) if callable(b_r) else float(b_r)
                    tracking_cost += w_safety * (max(0.0, e_c - wl) ** 2 + max(0.0, -e_c - wr) ** 2)
            if anchor_mode:
                # Tracking error = full distance to the progress point (one
                # term, one weight). Along-path mismatch is what appears when
                # the plan coasts past a corner (progress freezes while the
                # cursor runs on) or past the path end; lateral mismatch is
                # the contour error. No special cases.
                tracking_cost += w_contour * float(pos_error @ pos_error)
                if k == k_deadline:
                    # Progress via-point: be at the anchor at the deadline.
                    tracking_cost += w_goal * (anchor_s_target - float(s_traj[k]))**2
            elif any_free and free_mask[k]:
                # Goal-directed pointing: lateral error stays w.r.t. the path
                # (keeps the movement on the straight line); the drive is the
                # squared distance to the goal.
                g_err = pos_k - goal_xy
                tracking_cost += (w_contour * e_contour**2) + (w_goal * float(g_err @ g_err))
            else:
                tracking_cost += (w_contour * e_contour**2) + (w_lag * e_lag**2)

            # 4a. Path-relative corridor penalty
            if corridor_bounds is not None:
                b_left_in, b_right_in = corridor_bounds
                s_b = float(min(max(s_traj[k], 0.0), s_end_total))
                w_left = b_left_in(s_b) if callable(b_left_in) else float(b_left_in)
                w_right = b_right_in(s_b) if callable(b_right_in) else float(b_right_in)
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

        # 5. Goal precision — continuous potential-well formulation.
        # Signal-dependent motor noise variance (nc * v)^2 is penalized at every
        # horizon node, weighted by proximity to the target.  Nodes far from the
        # target contribute little; nodes close to it are strongly penalised,
        # naturally encouraging slow, accurate arrivals without any arc-length gate.
        w_precision = weights.get('goal_precision', 0.0)
        target_radius = weights.get('target_radius', 0.01)
        if w_precision > 0.0:
            # both nc0 and nc1 are signal-dependent noise coefficients
            nc0 = weights.get('nc0', 0.2)
            nc1 = weights.get('nc1', 0.02)
            # predicted velocity-scatter variance at each node: (nc * v)^2
            scatter_var = (nc0**2 + nc1**2) * (vx**2 + vy**2)
            # physical (x, y) of the path endpoint — no arc-length gating needed
            p_end = ref_path(ref_path.total_length)
            x_target, y_target = float(p_end[0]), float(p_end[1])
            # squared Euclidean distance from every predicted position to the target
            dist_sq = (px - x_target)**2 + (py - y_target)**2
            # potential-well: penalty is large near the target, fades with distance
            r2 = target_radius**2
            per_node_penalty = scatter_var / (dist_sq + r2)
            goal_cost = w_precision * np.sum(per_node_penalty)
        else:
            goal_cost = 0.0

        # 6. Free-space terminal cost = LQR value function of the stage cost
        #    (per axis, state [p-goal, v, a]) — makes the short-horizon plan
        #    equal to the infinite-horizon optimum.
        term_cost = 0.0
        if terminal_lqr:
            xN = np.array([px[-1] - goal_xy[0], vx[-1], ax_h[-1]])
            yN = np.array([py[-1] - goal_xy[1], vy[-1], ay_h[-1]])
            term_cost = float(xN @ P_lqr @ xN + yN @ P_lqr @ yN)

        return j_cost + prog_cost + tracking_cost + goal_cost + term_cost

    bounds = []
    bounds.extend([(None, None)] * n_vars)

    x0_cold = np.zeros(n_vars)
    if not anchor_mode:
        x0_cold[idx_vs] = speed_target / SCALE_VS

    # Warm-start: shift the previous solution forward by the number of steps
    # executed since that solve (1 under per-step replanning; larger under
    # intermittent replanning). The shifted tail is truncated or padded
    # (zero jerk, held progress speed) to the new horizon length, so the
    # warm start survives the per-solve horizon changes of budget mode.
    x0_warm = None
    x_prev = _warm_start_cache.get('x_prev')
    prev_n = _warm_start_cache.get('num_steps')
    k_shift = int(warm_shift)
    if (x_prev is not None and prev_n is not None and 1 <= k_shift < prev_n
            and len(x_prev) == n_blocks * prev_n):
        prev_jx = x_prev[:prev_n]
        prev_jy = x_prev[prev_n:2*prev_n]
        prev_vs = None if anchor_mode else x_prev[2*prev_n:3*prev_n]

        def _fit(tail, fill):
            if len(tail) >= num_steps:
                return tail[:num_steps]
            return np.append(tail, np.full(num_steps - len(tail), fill))

        x0_warm = np.zeros(n_vars)
        x0_warm[idx_jx] = _fit(prev_jx[k_shift:], 0.0)
        x0_warm[idx_jy] = _fit(prev_jy[k_shift:], 0.0)
        if not anchor_mode:
            x0_warm[idx_vs] = _fit(prev_vs[k_shift:], prev_vs[-1])

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
    if anchor_mode:
        # Fixed-point iteration on the progress linearisation: re-take the
        # tangents on the solved plan's own progress and re-solve
        # (warm-started) until the schedule is self-consistent.
        for _ in range(ANCHOR_RELIN_PASSES - 1):
            jx_p, jy_p, _ = unpack_x(result.x)
            s_new = kinematic_progress(vx_free + A_vel_mat @ jx_p, vy_free + A_vel_mat @ jy_p)
            if float(np.max(np.abs(s_new - s_sched))) < ANCHOR_RELIN_TOL:
                break
            s_sched = s_new
            tan_sched_box[0] = _tangents_on(s_sched)
            result = minimize(
                objective, result.x, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-6, 'gtol': 1e-5, 'maxfun': 5000})

    _warm_start_cache['x_prev'] = result.x.copy()
    _warm_start_cache['num_steps'] = num_steps

    jx_opt, jy_opt, vs_opt = unpack_x(result.x)
    if vs_opt is None:
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

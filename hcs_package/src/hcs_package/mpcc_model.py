"""Model Predictive Contouring Control (MPCC) solver for cursor steering.

Receives a pre-computed speed profile (from a SpeedModel) and solves for
optimal jerk controls.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d

from .constraints import ConstraintType, RectangleConstraint, PolygonConstraint
from .constraint_utils import _point_in_polygon, _distance_to_polygon_boundary


_warm_start_cache = {
    'x_prev': None,
    'num_steps': None,
}


def reset_warm_start():
    """Reset the warm-start cache (call at the start of each new trajectory)."""
    _warm_start_cache['x_prev'] = None
    _warm_start_cache['num_steps'] = None


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
    w_contour = weights.get('contour', 1.0)
    w_lag = weights.get('lag', 0.1)

    # Light Gaussian smoothing prevents chasing step-like changes over the short horizon
    speed_target = np.asarray(speed_profile, dtype=float).copy()
    if len(speed_target) > 3:
        speed_target = gaussian_filter1d(speed_target, sigma=1.5)

    SCALE_JERK = 1000.0
    SCALE_VS = max(0.1, desired_speed)

    acc_max = limits.get('acc_max', 100.0)

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

    # x = [jx_0..N-1, jy_0..N-1, vs_0..N-1]
    n_vars = 3 * num_steps
    idx_jx = slice(0, num_steps)
    idx_jy = slice(num_steps, 2 * num_steps)
    idx_vs = slice(2 * num_steps, 3 * num_steps)

    S_mat = np.tril(np.ones((num_steps, num_steps))) * dt

    if cartesian_constraints is not None:
        s_estimated = s0 + desired_speed * dt * np.arange(1, num_steps + 1)
        cartesian_active = s_estimated < ref_path.total_length
    else:
        cartesian_active = None

    def unpack_x(x):
        jx = x[idx_jx] * SCALE_JERK
        jy = x[idx_jy] * SCALE_JERK
        vs = x[idx_vs] * SCALE_VS
        return jx, jy, vs

    def objective(x):
        jx, jy, vs = unpack_x(x)

        # 1. Jerk smoothness
        j_cost = np.sum(jx**2 + jy**2) * w_jerk

        # 2. Progress / speed tracking
        s_traj = s0 + S_mat @ vs
        vx = vx_free + A_vel_mat @ jx
        vy = vy_free + A_vel_mat @ jy
        physical_speed = np.sqrt(vx**2 + vy**2)

        speed_error = physical_speed - speed_target
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
        tracking_cost = 0.0

        for k in range(num_steps):
            pos_k = np.array([px[k], py[k]], dtype=float)
            ref_k = np.array([rx[k], ry[k]], dtype=float)
            pos_error = pos_k - ref_k

            tangent = ref_path.tangent(s_traj[k])
            cos_phi = tangent[0]
            sin_phi = tangent[1]

            R = np.array([
                [sin_phi, -cos_phi],
                [-cos_phi, -sin_phi]
            ], dtype=float)

            e_k = R @ pos_error
            e_contour = e_k[0]
            e_lag = e_k[1]

            tracking_cost += (w_contour * e_contour**2) + (w_lag * e_lag**2)

            # 4a. Path-relative corridor penalty
            if corridor_bounds is not None:
                b_left_in, b_right_in = corridor_bounds
                w_left = b_left_in(s_traj[k]) if callable(b_left_in) else float(b_left_in)
                w_right = b_right_in(s_traj[k]) if callable(b_right_in) else float(b_right_in)
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

    # vs >= 50% of speed target
    vs_min_per_step = 0.5 * speed_target / SCALE_VS
    bounds = []
    bounds.extend([(None, None)] * num_steps)
    bounds.extend([(None, None)] * num_steps)
    bounds.extend([(float(vs_min_per_step[i]), None) for i in range(num_steps)])

    x0_cold = np.zeros(n_vars)
    x0_cold[idx_vs] = speed_target / SCALE_VS

    # Warm-start: shift previous solution forward by one step
    x0_warm = None
    x_prev = _warm_start_cache.get('x_prev')
    prev_n = _warm_start_cache.get('num_steps')
    if x_prev is not None and prev_n == num_steps:
        x0_warm = np.zeros(n_vars)
        prev_jx = x_prev[:num_steps]
        prev_jy = x_prev[num_steps:2*num_steps]
        prev_vs = x_prev[2*num_steps:3*num_steps]
        x0_warm[idx_jx] = np.append(prev_jx[1:], 0.0)
        x0_warm[idx_jy] = np.append(prev_jy[1:], 0.0)
        x0_warm[idx_vs] = np.append(prev_vs[1:], prev_vs[-1])

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

    _warm_start_cache['x_prev'] = result.x.copy()
    _warm_start_cache['num_steps'] = num_steps

    jx_opt, jy_opt, vs_opt = unpack_x(result.x)
    controls = np.column_stack((jx_opt, jy_opt, vs_opt))

    opt_info = {
        'success': result.success,
        'cost': result.fun,
        'message': result.message,
        'nit': result.nit
    }

    return controls, opt_info

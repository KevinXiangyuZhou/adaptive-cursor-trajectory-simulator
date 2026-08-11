"""Pointing baseline solver — simplified BUMP (Basic Unit of Motor Production)
optimal trajectory generator (OTG) for point-to-point Fitts' Law tasks.

This is a stripped-down sibling of ``mpcc_model.generate_mpcc``: it keeps the
same jerk-to-kinematics mapping and the same contour/lag tracking error
formulation (evaluated against a straight-line reference path), but drops
everything that only matters for constrained tunnel steering — the
goal-precision potential well, path-relative corridor penalties, and
cartesian (rectangle/polygon) keep-in/keep-out constraints. In their place it
adds a direct acceleration-magnitude penalty, matching the classic
minimum-jerk/minimum-acceleration compromise used in BUMP-style OTG models.

⚠️ RULED OUT for pointing tasks (2026-08): shows jittery start/end behavior
(the desired-speed reference fed into this solver never re-anchors to real
elapsed time or progress) and a severe distance-driven completion-time
blowup vs. human data. Replaced by MPCC-in-a-bypass-tunnel (see
``experiment.environment._create_pointing_bypass_env`` /
``eval/eval-new-data/run_eval.py::build_fitts_bypass_config``). Its pygame
visual debugger was removed; do not use this as the default pointing-task
model in new code. See ``eval/eval-new-data/run_eval.py``'s module
docstring for the full writeup.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d

from .mpcc_model import (
    _warm_start_cache,
    _build_A_acc,
    _build_A_vel_from_jerk,
    _build_A_pos_from_jerk,
)


def generate_baseline_mpc(
    ref_path,
    state_0,
    num_steps,
    dt,
    weights,
    limits,
    speed_profile,
    desired_speed=1.0,
):
    """
    Generate a simplified pointing-baseline MPC plan.

    Minimizes a weighted sum of jerk, acceleration, speed-profile tracking
    error, and contour/lag tracking error against a straight-line reference
    path. Unlike ``generate_mpcc``, this solver has no boundary constraints,
    no cartesian keep-in/keep-out regions, and no goal-precision potential
    well — it is meant for open-field point-to-point pointing tasks only.

    Args:
        ref_path: ReferencePath object (a straight line for pointing tasks).
        state_0: Initial state [px, py, vx, vy, ax, ay, s].
        num_steps: Prediction horizon (N).
        dt: Time step.
        weights: Dictionary of weights — ``jerk``, ``acceleration``,
            ``progress``, ``contour``, ``lag``.
        limits: Dictionary of limits (``acc_max``). Kept for interface
            parity with ``generate_mpcc``; not otherwise used since
            acceleration is penalized softly via the ``acceleration`` weight
            rather than hard-bounded.
        speed_profile: 1-D array of desired speeds at the horizon nodes
            (N,) — typically a pre-computed minimum-jerk bell-shaped
            velocity profile.
        desired_speed: Base desired speed (used for scaling only).

    Returns:
        controls: (N, 3) array of [jx, jy, vs].
        opt_info: optimization result info dict.
    """
    if weights is None:
        weights = {}

    w_jerk = weights.get('jerk', 1.5e-6)
    w_acc = weights.get('acceleration', 1e-4)
    w_progress = weights.get('progress', 1.0e-5)
    w_contour = weights.get('contour', 1.0)
    w_lag = weights.get('lag', 0.1)

    # Light Gaussian smoothing prevents chasing step-like changes over the short horizon
    speed_target = np.asarray(speed_profile, dtype=float).copy()
    if len(speed_target) > 3:
        speed_target = gaussian_filter1d(speed_target, sigma=1.5)

    SCALE_JERK = 1000.0
    SCALE_VS = max(0.1, desired_speed)

    acc_max = limits.get('acc_max', 100.0)  # kept for parity with generate_mpcc; unused (soft penalty only)

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

    def unpack_x(x):
        jx = x[idx_jx] * SCALE_JERK
        jy = x[idx_jy] * SCALE_JERK
        vs = x[idx_vs] * SCALE_VS
        return jx, jy, vs

    def objective(x):
        jx, jy, vs = unpack_x(x)

        # 1. Jerk smoothness
        j_cost = np.sum(jx**2 + jy**2) * w_jerk

        # 2. Acceleration magnitude (brand-new term vs. generate_mpcc)
        ax = ax_free + A_acc_mat @ jx
        ay = ay_free + A_acc_mat @ jy
        acc_cost = np.sum(ax**2 + ay**2) * w_acc

        # 3. Progress / speed tracking
        s_traj = s0 + S_mat @ vs
        vx = vx_free + A_vel_mat @ jx
        vy = vy_free + A_vel_mat @ jy
        physical_speed = np.sqrt(vx**2 + vy**2)

        speed_error = physical_speed - speed_target
        prog_cost = np.sum(speed_error**2) * w_progress

        # 4. Contour + lag tracking error (straight-line reference path)
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
                [-cos_phi, -sin_phi],
            ], dtype=float)

            e_k = R @ pos_error
            e_contour = e_k[0]
            e_lag = e_k[1]

            tracking_cost += (w_contour * e_contour**2) + (w_lag * e_lag**2)

        # No goal-precision potential well, no corridor bounds, no cartesian
        # constraints — this is intentionally a constraint-free baseline.

        return j_cost + acc_cost + prog_cost + tracking_cost

    bounds = [(None, None)] * n_vars

    x0_cold = np.zeros(n_vars)
    x0_cold[idx_vs] = speed_target / SCALE_VS

    # Warm-start: shares the same cache as generate_mpcc (reset_warm_start()
    # clears it), shifting the previous solution forward by one step.
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
        'nit': result.nit,
    }

    return controls, opt_info

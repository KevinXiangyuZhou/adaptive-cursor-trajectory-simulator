"""Reference path construction and race-tracing optimization."""

import numpy as np
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d
from .adapt import compute_local_curvature_integral, compute_qp_bounds, compute_clearance_profile


class ReferencePath:
    """Smooth reference path constructed from tunnel centerline using cubic splines."""
    def __init__(self, waypoints, s=0.0, k=3):
        """
        Args:
            waypoints: List of (x, y) points defining the centerline
            s: Smoothing factor (0 = interpolation, >0 = smoothing)
            k: Spline degree (3 = cubic)
        """
        waypoints = self._validate_and_prepare_waypoints(waypoints)

        self.waypoints = waypoints
        max_k = min(k, waypoints.shape[0] - 1)
        if max_k < 1:
            max_k = 1

        try:
            self.tck, self.u = splprep([waypoints[:, 0], waypoints[:, 1]], s=s, k=max_k)
        except ValueError as e:
            # Retry with linear spline if higher-degree fit fails
            if max_k > 1:
                try:
                    self.tck, self.u = splprep([waypoints[:, 0], waypoints[:, 1]], s=s, k=1)
                except ValueError:
                    raise ValueError(f"splprep failed with waypoints shape {waypoints.shape}: {e}")
            else:
                raise ValueError(f"splprep failed with waypoints shape {waypoints.shape}: {e}")

        u_dense = np.linspace(0, 1, 1000)
        xy_dense = np.array(splev(u_dense, self.tck))
        diffs = np.diff(xy_dense, axis=1)
        ds = np.sqrt(np.sum(diffs**2, axis=0))
        self.arclengths = np.concatenate([[0], np.cumsum(ds)])
        self.u_dense = u_dense
        self.total_length = self.arclengths[-1]
    
    def __call__(self, theta):
        """Evaluate position at arclength theta."""
        u = self._theta_to_u(theta)
        xy = splev(u, self.tck)
        return np.array(xy, dtype=float)
    
    def _validate_and_prepare_waypoints(self, waypoints):
        """Clean waypoints (remove NaN/dup) and ensure splprep can fit them."""
        waypoints = np.asarray(waypoints, dtype=float)

        valid_mask = np.isfinite(waypoints).all(axis=1)
        if not valid_mask.all():
            waypoints = waypoints[valid_mask]
            if len(waypoints) == 0:
                raise ValueError("All waypoints are invalid (NaN or inf)")

        if len(waypoints) > 1:
            diffs = np.diff(waypoints, axis=0)
            dists = np.linalg.norm(diffs, axis=1)
            keep_mask = np.concatenate([[True], dists > 1e-10])
            waypoints = waypoints[keep_mask]

        if waypoints.shape[0] < 2:
            raise ValueError(f"Need at least 2 distinct waypoints, got {waypoints.shape[0]}")

        # Pad to >= 4 points so cubic splprep can fit. Resample each segment
        # with EVENLY spaced points: the previous scheme inserted a point
        # 1e-6 away from an existing one, and two near-coincident interior
        # points make the chord-length parametrisation degenerate — the
        # cubic then bulges (a 2-point straight path came out ~39% longer
        # with cm-scale lateral swings).
        if waypoints.shape[0] < 4:
            n_seg = waypoints.shape[0] - 1
            per_seg = int(np.ceil(4 / n_seg))  # points per segment (excl. segment end)
            pieces = [np.linspace(waypoints[i], waypoints[i + 1], per_seg + 1)[:-1]
                      for i in range(n_seg)]
            waypoints = np.vstack(pieces + [waypoints[-1:]])

        # Break collinearity to avoid degenerate splprep fits
        if waypoints.shape[0] >= 3:
            vec1 = waypoints[1] - waypoints[0]
            vec2 = waypoints[-1] - waypoints[0]
            cross = vec1[0] * vec2[1] - vec1[1] * vec2[0]
            if abs(cross) < 1e-10:
                mid_idx = len(waypoints) // 2
                dir_vec = waypoints[-1] - waypoints[0]
                perp = np.array([-dir_vec[1], dir_vec[0]])
                perp = perp / (np.linalg.norm(perp) + 1e-10)
                waypoints[mid_idx] = waypoints[mid_idx] + perp * 1e-6

        return waypoints
    
    def tangent(self, theta):
        """Return unit tangent vector at arclength theta."""
        u = self._theta_to_u(theta)
        dxy = splev(u, self.tck, der=1)
        t = np.array([dxy[0], dxy[1]], dtype=float)
        norm = np.linalg.norm(t)
        if norm < 1e-9:
            return np.array([1.0, 0.0])
        return t / norm
    
    def tangents(self, thetas):
        """Unit tangents at an array of arclengths — (N, 2), vectorised
        counterpart of ``tangent`` (one splev call)."""
        u = self._theta_to_u(np.asarray(thetas, dtype=float))
        dx, dy = splev(u, self.tck, der=1)
        t = np.column_stack([np.asarray(dx, dtype=float), np.asarray(dy, dtype=float)])
        n = np.linalg.norm(t, axis=1)
        bad = n < 1e-9
        t[bad] = (1.0, 0.0); n[bad] = 1.0
        return t / n[:, None]

    def normal(self, theta):
        """Return right-pointing unit normal vector at arclength theta."""
        t = self.tangent(theta)
        return np.array([t[1], -t[0]], dtype=float)
    
    def curvature(self, theta):
        """Compute curvature κ at arclength theta."""
        u = self._theta_to_u(theta)
        k = self.tck[2] if hasattr(self.tck, '__len__') else 3
        if k < 2:
            return 0.0  # linear spline has zero curvature
        dxy = splev(u, self.tck, der=1)
        ddxy = splev(u, self.tck, der=2)
        dx, dy = dxy[0], dxy[1]
        ddx, ddy = ddxy[0], ddxy[1]
        num = dx * ddy - dy * ddx
        den = (dx**2 + dy**2)**1.5
        if den < 1e-12:
            return 0.0
        return num / den
    
    def find_closest_theta(self, pos, initial_guess=None, min_theta=None):
        """Find arclength θ of closest point on path to given position.

        Coarse sample search followed by Newton refinement on
        f(u) = ||c(u) - pos||². ``min_theta`` enforces forward progress.
        """
        pos = np.asarray(pos, dtype=float)
        px, py = float(pos[0]), float(pos[1])

        min_u = 0.0
        if min_theta is not None:
            min_theta_val = float(min_theta)
            min_u = float(np.interp(min_theta_val, self.arclengths, self.u_dense))

        # ALWAYS global coarse search, then Newton refinement. A warm-started
        # local search is an absorbing trap: when the guess lags the cursor
        # by a fold of the path (e.g. after an intermittent-control pause the
        # arc estimate is only refreshed at the next replan), Newton's f''
        # can turn negative there, the step clips to u=0, and every later
        # call re-enters at 0 — the simulator's theta froze at the path
        # start while the cursor completed the trial (the "diving" gaze-lead
        # pages). Task centerlines do not approach themselves, so the global
        # nearest point is the right projection, and the dense evaluation is
        # the same cost the guess-free branch always paid. ``initial_guess``
        # is retained for API compatibility but no longer trusted.
        xy = np.array(splev(self.u_dense, self.tck))
        dx = xy[0] - px
        dy = xy[1] - py
        dist2 = dx * dx + dy * dy
        if min_u > 0.0:
            dist2 = np.where(self.u_dense < min_u, np.inf, dist2)
        u0 = float(self.u_dense[int(np.argmin(dist2))])

        # Newton step needs second derivative — undefined for linear splines
        spline_k = self.tck[2] if hasattr(self.tck, '__len__') else 3
        if spline_k < 2:
            theta = float(np.interp(u0, self.u_dense, self.arclengths))
            return theta

        max_iter = 5
        tol = 1e-6
        # Safeguarded Newton: the coarse-grid minimum is only refined by
        # steps that (a) are descent steps (f'' > 0 — on a jagged path the
        # squared distance is locally concave and an unsafeguarded step
        # runs away, clips to u=0 and reports theta=0 while the cursor is
        # mid-path), (b) stay within a few grid cells of the grid minimum,
        # and (c) actually reduce the distance.
        du_max = 3.0 * float(self.u_dense[1] - self.u_dense[0])

        def _f(u):
            c = splev(u, self.tck, der=0)
            return (c[0] - px) ** 2 + (c[1] - py) ** 2

        f0 = _f(u0)
        for _ in range(max_iter):
            c = splev(u0, self.tck, der=0)
            c1 = splev(u0, self.tck, der=1)
            c2 = splev(u0, self.tck, der=2)

            rx = c[0] - px
            ry = c[1] - py
            c1x, c1y = c1[0], c1[1]
            c2x, c2y = c2[0], c2[1]

            f_prime = 2.0 * (rx * c1x + ry * c1y)
            f_second = 2.0 * ((c1x * c1x + c1y * c1y) + (rx * c2x + ry * c2y))

            if f_second <= 1e-12:
                break

            du = float(np.clip(-f_prime / f_second, -du_max, du_max))
            if abs(du) < tol:
                break

            u_new = float(np.clip(u0 + du, min_u, 1.0))
            f_new = _f(u_new)
            if f_new >= f0 or abs(u_new - u0) < tol:
                break
            u0, f0 = u_new, f_new

        theta = float(np.interp(u0, self.u_dense, self.arclengths))
        return theta
    
    def _theta_to_u(self, theta):
        """Convert arclength θ to spline parameter u ∈ [0, 1]."""
        theta = np.clip(theta, 0.0, self.total_length)
        u = np.interp(theta, self.arclengths, self.u_dense)
        if np.ndim(theta) == 0:
            return float(u)
        return u


def _has_loop(path_points: np.ndarray, tol: float = 1e-6) -> bool:
    """True if any non-adjacent segment pair intersects at interior points."""
    if len(path_points) < 4:
        return False

    for i in range(len(path_points) - 1):
        p0 = path_points[i]
        p1 = path_points[i + 1]

        for j in range(i + 2, len(path_points) - 1):
            p2 = path_points[j]
            p3 = path_points[j + 1]

            denom = (p1[0] - p0[0]) * (p3[1] - p2[1]) - (p1[1] - p0[1]) * (p3[0] - p2[0])

            if abs(denom) > tol:
                t = ((p2[0] - p0[0]) * (p3[1] - p2[1]) - (p2[1] - p0[1]) * (p3[0] - p2[0])) / denom
                u = ((p2[0] - p0[0]) * (p1[1] - p0[1]) - (p2[1] - p0[1]) * (p1[0] - p0[0])) / denom

                if tol < t < 1.0 - tol and tol < u < 1.0 - tol:
                    return True

    return False


def _remove_loops_from_path(path_points: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Greedy loop removal: skip points whose new segment crosses an earlier one."""
    if len(path_points) < 4:
        return path_points

    if not _has_loop(path_points, tol):
        return path_points

    cleaned_indices = [0]
    i = 1

    while i < len(path_points):
        test_indices = cleaned_indices + [i]
        test_path = path_points[test_indices]

        has_intersection = False
        if len(test_indices) >= 4:
            last_seg_start = test_indices[-2]
            last_seg_end = test_indices[-1]
            p2 = path_points[last_seg_start]
            p3 = path_points[last_seg_end]

            for j in range(len(test_indices) - 3):
                p0 = path_points[test_indices[j]]
                p1 = path_points[test_indices[j + 1]]

                denom = (p1[0] - p0[0]) * (p3[1] - p2[1]) - (p1[1] - p0[1]) * (p3[0] - p2[0])

                if abs(denom) > tol:
                    t = ((p2[0] - p0[0]) * (p3[1] - p2[1]) - (p2[1] - p0[1]) * (p3[0] - p2[0])) / denom
                    u = ((p2[0] - p0[0]) * (p1[1] - p0[1]) - (p2[1] - p0[1]) * (p1[0] - p0[0])) / denom

                    if tol < t < 1.0 - tol and tol < u < 1.0 - tol:
                        has_intersection = True
                        break

        if not has_intersection:
            cleaned_indices.append(i)

        i += 1

    if cleaned_indices[-1] != len(path_points) - 1:
        cleaned_indices.append(len(path_points) - 1)

    cleaned_path = path_points[cleaned_indices]

    # Aggressive fallback: keep every Nth point if loops persist
    if _has_loop(cleaned_path, tol):
        step = max(2, len(path_points) // 20)
        cleaned_path = path_points[::step]
        if len(cleaned_path) > 0 and not np.array_equal(cleaned_path[-1], path_points[-1]):
            cleaned_path = np.vstack([cleaned_path, path_points[-1:]])

    return cleaned_path


def _smooth_offsets(
    d: np.ndarray,
    s: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    sigma_frac: float = 0.015,
) -> np.ndarray:
    """Gaussian-smooth lateral offsets and clip to [lb, ub].

    Kernel must stay narrow relative to each turn's arc length — wider
    kernels span turns of opposite sign and cancel the offsets to ~0.
    sigma_frac=0.015 (≈1.5% of L) blends only the inflection transition.
    """
    L = s[-1] - s[0]
    sigma = max(sigma_frac * L, 1e-9)
    n = len(d)
    d_smooth = np.empty(n)
    for i in range(n):
        w = np.exp(-0.5 * ((s - s[i]) / sigma) ** 2)
        w /= w.sum()
        d_smooth[i] = w @ d
    return np.clip(d_smooth, lb, ub)


def generate_optimal_reference_path(
    tunnel_path,
    tunnel_width,
    margin=0.001,
    num_knots=None,
    w_cut=0.01,          # base corner-cutting aggressiveness (0 = stay on centerline)
    w_suppress=0.0,      # suppression sensitivity: exp(-w_suppress * phi_k)
                         # 0 = always cut; larger = suppress in dense/sharp regions
    w_width_exp=1.0,     # width sensitivity exponent (0 = ignore width, 1 = linear)
    cut_window_frac=0.05,  # window for phi_k as fraction of path length (scale-invariant)
    global_clearance_ref=0.025,  # reference clearance for absolute task scaling (2.5cm default)
    cartesian_constraints=None,  # List[ConstraintRegion] for constraint-aware bounds
    corridor_bounds=None,  # Optional (left_bound, right_bound) for path-relative constraints
    centerline_cache=None,  # Optional pre-built ReferencePath to avoid redundant spline creation
):
    """Generate a race-tracing reference path inside the tunnel.

    Path model (right-normal Frenet frame):
        p(s) = C(s) + d(s) * n_R(s),   n_R(s) = [t_y, -t_x]

    Signed curvature: κ > 0 → left turn (cut toward lb), κ < 0 → right turn
    (cut toward ub).

    Corner-cutting: width ENABLES cutting, curvature MOTIVATES it.
        cut_fraction = w_cut * width_factor * kappa_factor * exp(-w_suppress * phi)

    width_factor scales with local clearance ^ w_width_exp; kappa_factor
    scales with |κ| / max(|κ|); phi is ∫|κ| ds over a local window.
    Wide+curved sections cut maximally; narrow or straight sections don't.

    Args:
        tunnel_path:           List[(x,y)] centerline samples.
        tunnel_width:          Fallback corridor width (used when no constraints).
        margin:                Safety margin to walls.
        num_knots:             Arc-length discretisation (auto if None).
        w_cut:                 Max cutting fraction in [0,1] (1 = cut to wall).
        w_suppress:            Suppression vs. dense/sharp turn regions.
        w_width_exp:           Width sensitivity exponent (0 = ignore width).
        cut_window_frac:       phi window as fraction of total path length.
        cartesian_constraints: Optional List[ConstraintRegion] for keep-in bounds.
        corridor_bounds:       Optional (left_bound, right_bound) for path-relative
                               constraints.

    Returns:
        ReferencePath through optimised waypoints (cubic, C2-continuous).
    """
    waypoints = np.asarray(tunnel_path, dtype=float)
    centerline = centerline_cache if centerline_cache is not None else ReferencePath(waypoints, s=0.0, k=3)
    L = centerline.total_length

    if num_knots is None:
        num_knots = max(40, min(300, 2 * len(waypoints)))
    s_knots = np.linspace(0.0, L, num_knots)

    C       = np.stack([centerline(theta) for theta in s_knots], axis=0)         # (N,2)
    T       = np.stack([centerline.tangent(theta) for theta in s_knots], axis=0) # (N,2)
    N_right = np.stack([[t[1], -t[0]] for t in T], axis=0)                       # (N,2)
    kappa   = np.array([centerline.curvature(theta) for theta in s_knots])       # (N,)
    ds      = np.diff(s_knots, prepend=s_knots[0])
    ds[0]   = ds[1] if len(ds) > 1 else 0.0

    half_w = np.full(num_knots, 0.5 * float(tunnel_width) - float(margin))
    half_w = np.maximum(half_w, 1e-6)

    N = num_knots

    lb, ub = compute_qp_bounds(
        centerline, s_knots, C, N_right, half_w,
        cartesian_constraints=cartesian_constraints,
        margin=float(margin),
    )

    # Pin endpoints on the centerline
    lb[0] = ub[0] = 0.0
    lb[-1] = ub[-1] = 0.0

    clearance = compute_clearance_profile(
        centerline, s_knots,
        corridor_bounds=corridor_bounds,
        cartesian_constraints=cartesian_constraints,
    )

    # No constraints → fall back to lateral bounds for clearance
    if np.isinf(clearance).all() or (clearance == clearance[0]).all():
        clearance = np.minimum(np.abs(lb), np.abs(ub))
        clearance = np.maximum(clearance, 1e-6)

    # Width factor combines absolute (task-level) and relative (within-task)
    # components: wider tasks AND wider spots within a task enable more cutting.
    task_clearance_ref = np.percentile(clearance, 90)
    if task_clearance_ref < 1e-6:
        task_clearance_ref = np.max(clearance)

    task_scale = np.tanh(task_clearance_ref / global_clearance_ref) if task_clearance_ref > 1e-6 else 0.0

    local_norm = np.clip(clearance / task_clearance_ref, 0.0, 2.0) if task_clearance_ref > 1e-6 else np.ones(N)
    local_factor = local_norm ** w_width_exp

    width_factor = task_scale * local_factor

    cut_window = max(cut_window_frac * L, 1e-6)
    phi = compute_local_curvature_integral(centerline, s_knots, window=cut_window)

    # Smooth κ — cubic-spline corners produce extreme spikes that need spreading
    # over ~sigma_knots knots so cutting transitions are bell-shaped, not jumpy.
    sigma_knots = max(5.0, 0.03 * N)
    kappa_smoothed = gaussian_filter1d(kappa.astype(float), sigma=sigma_knots)

    kappa_sm_abs = np.abs(kappa_smoothed)
    kappa_sm_max = float(np.max(kappa_sm_abs))
    # Normalising by max|κ| turns the numerical-noise curvature of a
    # (near-)straight centreline (~1e-3..1e-2 1/m from the spline fit) into an
    # O(1) cutting profile; with a wide corridor that becomes cm-scale wobble
    # on a path that should be straight. Only cut when there is a physically
    # meaningful bend (radius of curvature < KAPPA_CUT_MIN^-1 = 10 m).
    KAPPA_CUT_MIN = 0.1
    kappa_factor = (kappa_sm_abs / kappa_sm_max) if kappa_sm_max > KAPPA_CUT_MIN else np.zeros_like(kappa)

    cut_fraction = np.clip(w_cut, 0.0, 1.0) * width_factor * kappa_factor * np.exp(-w_suppress * phi)

    # Cut toward the inside of each turn (lb for left turns, ub for right).
    d_desired = np.where(
        kappa_smoothed >= 0,
        cut_fraction * lb,
        cut_fraction * ub,
    )
    d_opt = np.clip(d_desired, lb, ub)

    # Re-pin endpoints — kappa_smoothed is non-zero there from gaussian boundary effects
    d_opt[0] = 0.0
    d_opt[-1] = 0.0

    P = C + d_opt[:, None] * N_right
    P = np.asarray(P)

    P = _remove_loops_from_path(P)

    return ReferencePath(P, s=0.0, k=3)

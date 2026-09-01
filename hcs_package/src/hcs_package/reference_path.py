"""Reference path construction and race-tracing optimization."""

import numpy as np
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d
from .adapt import compute_qp_bounds


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


def densify_polyline(points, max_spacing=0.002):
    """Resample a polyline so consecutive points are at most ``max_spacing`` apart
    (linear interpolation along each segment; vertices are kept). A cubic spline
    through a coarse polyline overshoots each sharp vertex by ~spacing^2 — 0.8 mm
    for the 10 mm task waypoints — which shows up as a bump outside every corner;
    through the densified polyline the overshoot is ~0.03 mm."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return pts
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / max(max_spacing, 1e-9))))
        for i in range(1, n + 1):
            out.append(a + (b - a) * (i / n))
    return np.asarray(out)


def generate_optimal_reference_path(
    tunnel_path,
    tunnel_width,
    margin=0.001,
    num_knots=None,
    w_cut=0.5,                    # fraction of the inside room the person is willing to use, [0, 1]
    w_width_exp=1.0,              # room sensitivity: f_room = tanh(room / ref) ** w_width_exp
    w_center=1.0,                 # reluctance to leave the centerline where it buys no smoothness
    global_clearance_ref=0.025,   # room (m) at which f_room reaches tanh(1)
    cartesian_constraints=None,   # List[ConstraintRegion] for constraint-aware bounds
    corridor_bounds=None,         # Optional (left_bound, right_bound), path-relative
    centerline_cache=None,        # Optional pre-built centerline ReferencePath
    **legacy,                     # older keys (w_suppress, cut_window_frac, cut_scale_frac) ignored
):
    """Race-tracing reference path = the smoothest path inside the lateral slack the
    participant is willing to use.

        p_k = C_k + d_k n_k + e_k t_k                       (knots on the centerline spline)
        minimise  sum_k |Δ²p_k / ds²|²  (curvature energy)
                + w_center * sum_k (d_k / room_k²)²  +  sum_k (e_k / room_k²)²
        subject to d_k in the INSIDE slack band of the local turn:
                   [-a_k, 0] for a left turn, [0, a_k] for a right turn,
                   a_k = w_cut * f_room(k) * room_inside(k),  f_room = tanh(room/ref) ** w_width_exp,
                   a_k -> 0 through inflections and at both ends; |e_k| <= a_k.

    * C, n, t: cubic spline through the densified waypoint polyline (no vertex overshoot).
    * Turn direction from the Gaussian-smoothed heading of the waypoint polyline (a clean
      step at a corner, smooth along a sinusoid, zero on straights).
    * The tangential freedom e_k is what lets a sharp vertex become a fillet: a pure
      normal offset of a corner is another corner of the same angle, so it cannot lower
      the curvature energy — the classic inward-offset folds into a loop instead.
    * Curvature motivates, room enables: slack is spent only where it lowers curvature
      (corner fillets, inside shifts on bends); a straight is already optimal and stays put.
      The one-sided band forbids outward swings; a minimum-curvature path has no bumps
      and no loops by construction. w_center sets how early the path leaves the
      centerline ahead of a turn (the lobe extent).

    Per-participant parameters: w_cut, w_width_exp, w_center, global_clearance_ref.
    Bounded least squares (scipy lsq_linear, trf with exact inner solves).
    """
    import scipy.sparse as sps
    from scipy.optimize import lsq_linear

    waypoints = np.asarray(tunnel_path, dtype=float)
    dense = densify_polyline(waypoints)
    centerline = centerline_cache if centerline_cache is not None else ReferencePath(dense, s=0.0, k=3)
    L = centerline.total_length
    if num_knots is None:
        num_knots = int(np.clip(round(L / 0.006), 40, 250))   # ~6 mm knots
    N = num_knots
    s_knots = np.linspace(0.0, L, N)
    ds = s_knots[1] - s_knots[0] if N > 1 else 1.0

    C = np.stack([centerline(theta) for theta in s_knots], axis=0)
    T = centerline.tangents(s_knots)
    N_right = np.column_stack([T[:, 1], -T[:, 0]])

    half_w = np.maximum(np.full(N, 0.5 * float(tunnel_width) - float(margin)), 1e-6)
    lb, ub = compute_qp_bounds(
        centerline, s_knots, C, N_right, half_w,
        cartesian_constraints=cartesian_constraints, margin=float(margin),
    )
    lb = np.minimum(lb, 0.0)
    ub = np.maximum(ub, 0.0)

    # --- which side is inside: signed curvature of the smoothed polyline heading ----------
    seg = np.diff(waypoints, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    keep = seg_len > 1e-12
    seg, seg_len = seg[keep], seg_len[keep]
    if len(seg) == 0:
        return ReferencePath(dense, s=0.0, k=3)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    s_poly = s_knots * (cum[-1] / max(L, 1e-12))
    idx = np.clip(np.searchsorted(cum, s_poly, side="right") - 1, 0, len(seg) - 1)
    psi = np.unwrap(np.arctan2(seg[idx, 1], seg[idx, 0]))
    sigma_k = max(0.005 / max(ds, 1e-9), 1.0)            # 5 mm: only to de-noise the sign
    kappa_s = np.gradient(gaussian_filter1d(psi, sigma=sigma_k, mode="nearest"), s_knots)
    k_max = float(np.max(np.abs(kappa_s)))
    if k_max < 1e-6:
        return ReferencePath(dense, s=0.0, k=3)          # straight tunnel: centerline

    # --- inside slack band ------------------------------------------------------------
    room_in = np.where(kappa_s >= 0.0, -lb, ub)          # kappa>0: left turn, inside = left (d<0)
    f_room = np.tanh(room_in / max(global_clearance_ref, 1e-6)) ** float(w_width_exp)
    a = np.clip(w_cut, 0.0, 1.0) * f_room * room_in
    a = a * np.clip((np.abs(kappa_s) / k_max) / 0.05, 0.0, 1.0)   # fade through inflections
    lo = np.where(kappa_s >= 0.0, -a, 0.0)
    hi = np.where(kappa_s >= 0.0, 0.0, a)
    import os as _os
    if _os.environ.get('HCS_DEBUG_ROUTE'):
        print(f"[route] L={L*1000:.0f}mm N={N} tunnel_width={tunnel_width*1000:.1f}mm margin={margin*1000:.2f}mm "
              f"lb[min,med]={np.min(lb)*1000:.1f},{np.median(lb)*1000:.1f} ub[med,max]={np.median(ub)*1000:.1f},{np.max(ub)*1000:.1f} "
              f"room_in[med,max]={np.median(room_in)*1000:.1f},{np.max(room_in)*1000:.1f} f_room[med]={np.median(f_room):.2f} "
              f"a[max]={np.max(a)*1000:.1f}mm k_max={k_max:.1f} cart={'y' if cartesian_constraints else 'n'} corr={'y' if corridor_bounds is not None else 'n'} "
              f"nwp={len(waypoints)}", flush=True)
    lo[[0, -1]] = 0.0; hi[[0, -1]] = 0.0
    hi = np.maximum(hi, lo + 1e-9)
    e_lim = np.maximum(np.maximum(-lo, hi), 1e-9)
    e_lo, e_hi = -e_lim, e_lim.copy()
    e_lo[[0, -1]] = -1e-9; e_hi[[0, -1]] = 1e-9

    # --- smoothest path in the band: sparse bounded least squares in (d, e) --------------
    D2 = sps.diags([np.ones(N - 2), -2.0 * np.ones(N - 2), np.ones(N - 2)], [0, 1, 2], shape=(N - 2, N))
    sc = 1.0 / (ds * ds)
    room_ref = np.maximum(np.maximum(-lb, ub), 1e-6)
    A = sps.vstack([
        sps.hstack([D2 @ sps.diags(N_right[:, 0]), D2 @ sps.diags(T[:, 0])]) * sc,
        sps.hstack([D2 @ sps.diags(N_right[:, 1]), D2 @ sps.diags(T[:, 1])]) * sc,
        sps.hstack([sps.diags(np.sqrt(max(float(w_center), 0.0)) / room_ref ** 2), sps.csr_matrix((N, N))]),
        sps.hstack([sps.csr_matrix((N, N)), sps.diags(1.0 / room_ref ** 2)]),
    ]).tocsr()
    bvec = np.concatenate([-(D2 @ C[:, 0]) * sc, -(D2 @ C[:, 1]) * sc, np.zeros(2 * N)])
    # Dense exact inner solves: the sparse lsmr variant stalls far from the optimum on
    # long, smooth bands (gentle sinusoids); 'exact' converges in ~15-100 iterations.
    res = lsq_linear(A.toarray(), bvec, bounds=(np.concatenate([lo, e_lo]), np.concatenate([hi, e_hi])),
                     method="trf", lsq_solver="exact", tol=1e-8, max_iter=500)
    d = np.clip(res.x[:N], lo, hi)
    e = np.clip(res.x[N:], e_lo, e_hi)
    P = C + d[:, None] * N_right + e[:, None] * T
    return ReferencePath(np.asarray(P), s=0.0, k=3)

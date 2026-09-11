"""
Reference path construction and optimization.

Provides ReferencePath class for smooth path representation using cubic splines,
and functions for generating optimal reference paths within tunnels.
"""

import numpy as np
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize


class ReferencePath:
    """Smooth reference path constructed from tunnel centerline using cubic splines."""
    def __init__(self, waypoints, s=0.0, k=3):
        """
        Args:
            waypoints: List of (x, y) points defining the centerline
            s: Smoothing factor (0 = interpolation, >0 = smoothing)
            k: Spline degree (3 = cubic)
        """
        # Validate and prepare waypoints
        waypoints = self._validate_and_prepare_waypoints(waypoints)
        
        # Parameterize by cumulative chord length
        self.waypoints = waypoints
        max_k = min(k, waypoints.shape[0] - 1)
        if max_k < 1:
            max_k = 1
        
        try:
            self.tck, self.u = splprep([waypoints[:, 0], waypoints[:, 1]], s=s, k=max_k)
        except ValueError as e:
            # If splprep fails, try with linear spline (k=1)
            if max_k > 1:
                try:
                    self.tck, self.u = splprep([waypoints[:, 0], waypoints[:, 1]], s=s, k=1)
                except ValueError:
                    raise ValueError(f"splprep failed with waypoints shape {waypoints.shape}: {e}")
            else:
                raise ValueError(f"splprep failed with waypoints shape {waypoints.shape}: {e}")
        
        # Compute approximate arclength mapping
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
        """
        Validate and prepare waypoints for spline fitting.
        
        Args:
            waypoints: Array of (x, y) points
        
        Returns:
            Cleaned and validated waypoints array
        """
        waypoints = np.asarray(waypoints, dtype=float)
        
        # Remove any NaN or inf values
        valid_mask = np.isfinite(waypoints).all(axis=1)
        if not valid_mask.all():
            waypoints = waypoints[valid_mask]
            if len(waypoints) == 0:
                raise ValueError("All waypoints are invalid (NaN or inf)")
        
        # Remove duplicate consecutive points
        if len(waypoints) > 1:
            diffs = np.diff(waypoints, axis=0)
            dists = np.linalg.norm(diffs, axis=1)
            # Keep first point and points that are not duplicates
            keep_mask = np.concatenate([[True], dists > 1e-10])
            waypoints = waypoints[keep_mask]
        
        # Ensure we have at least 2 points
        if waypoints.shape[0] < 2:
            raise ValueError(f"Need at least 2 distinct waypoints, got {waypoints.shape[0]}")
        
        # If we have exactly 2 points, add a middle point to make it 3
        if waypoints.shape[0] == 2:
            mid_point = (waypoints[0] + waypoints[1]) / 2.0
            waypoints = np.vstack([waypoints[0:1], mid_point.reshape(1, -1), waypoints[1:2]])
        
        # If we have 3 points, add one more to make it 4 for cubic spline
        if waypoints.shape[0] == 3:
            # Add a point slightly offset from the middle to avoid collinearity
            mid_point = waypoints[1]
            # Create a small perpendicular offset
            dir_vec = waypoints[2] - waypoints[0]
            perp = np.array([-dir_vec[1], dir_vec[0]])
            perp = perp / (np.linalg.norm(perp) + 1e-10)
            offset_point = mid_point + perp * 1e-6
            waypoints = np.vstack([waypoints[0:1], waypoints[1:2], offset_point.reshape(1, -1), waypoints[2:3]])
        
        # Check for collinearity - if all points are collinear, add a small perpendicular offset
        if waypoints.shape[0] >= 3:
            # Check if points are collinear
            vec1 = waypoints[1] - waypoints[0]
            vec2 = waypoints[-1] - waypoints[0]
            cross = vec1[0] * vec2[1] - vec1[1] * vec2[0]
            if abs(cross) < 1e-10:
                # Points are collinear, add a small perpendicular offset to middle point
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
    
    def normal(self, theta):
        """Return right-pointing unit normal vector at arclength theta."""
        t = self.tangent(theta)
        return np.array([t[1], -t[0]], dtype=float)
    
    def curvature(self, theta):
        """Compute curvature κ at arclength theta."""
        u = self._theta_to_u(theta)
        dxy = splev(u, self.tck, der=1)
        ddxy = splev(u, self.tck, der=2)
        dx, dy = dxy[0], dxy[1]
        ddx, ddy = ddxy[0], ddxy[1]
        num = dx * ddy - dy * ddx  # 
        den = (dx**2 + dy**2)**1.5
        if den < 1e-12:
            return 0.0
        return num / den
    
    def find_closest_theta(self, pos, initial_guess=None, min_theta=None):
        """Find arclength θ of closest point on path to given position.
        
        Uses a two-stage approach:
        1. Coarse search through sampled points to find initial guess
        2. Fine optimization using scipy.optimize.minimize_scalar
        
        Args:
            pos: Target position (x, y)
            initial_guess: Optional initial guess for theta (for faster convergence)
            min_theta: Optional minimum theta value (to enforce forward progress)
        
        Returns:
            Arclength theta of closest point on path
        """
        pos = np.asarray(pos, dtype=float)
        px, py = float(pos[0]), float(pos[1])

        # --- 1) Get initial guess in spline parameter u ∈ [0, 1] ---

        if initial_guess is not None:
            # initial_guess is in arclength θ, clip and map to u
            theta0 = float(np.clip(initial_guess, 0.0, self.total_length))
            u0 = float(np.interp(theta0, self.arclengths, self.u_dense))
        else:
            # Coarse search over pre-sampled path (vectorized, cheap)
            xy = np.array(splev(self.u_dense, self.tck))
            dx = xy[0] - px
            dy = xy[1] - py
            dist2 = dx * dx + dy * dy
            idx = int(np.argmin(dist2))
            u0 = float(self.u_dense[idx])
        
        # If min_theta is specified, ensure we only search forward
        min_u = 0.0
        if min_theta is not None:
            min_theta_val = float(min_theta)
            min_u = float(np.interp(min_theta_val, self.arclengths, self.u_dense))
            # Clip u0 to be at least min_u
            u0 = max(u0, min_u)

        # --- 2) Refine with a few Newton steps on f(u) = ||c(u) - pos||^2 ---

        max_iter = 5
        tol = 1e-6

        for _ in range(max_iter):
            # Position c(u), first and second derivatives
            c = splev(u0, self.tck, der=0)
            c1 = splev(u0, self.tck, der=1)
            c2 = splev(u0, self.tck, der=2)

            rx = c[0] - px
            ry = c[1] - py
            c1x, c1y = c1[0], c1[1]
            c2x, c2y = c2[0], c2[1]

            # f(u) = (c(u) - pos)·(c(u) - pos)
            # f'(u) = 2 (c - pos)·c'
            # f''(u) = 2 (|c'|^2 + (c - pos)·c'')
            f_prime = 2.0 * (rx * c1x + ry * c1y)
            f_second = 2.0 * ((c1x * c1x + c1y * c1y) + (rx * c2x + ry * c2y))

            if abs(f_second) < 1e-12:
                break  # avoid division by zero / ill-conditioned

            du = -f_prime / f_second
            if abs(du) < tol:
                break

            u_new = u0 + du
            # Keep u in valid range (enforce min_u if min_theta was specified)
            u_new = float(np.clip(u_new, min_u, 1.0))

            if abs(u_new - u0) < tol:
                u0 = u_new
                break

            u0 = u_new

        # --- 3) Map refined u back to arclength θ ---

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
    """
    Check if a path has self-intersections (loops).
    
    Args:
        path_points: Array of shape (N, 2) representing path points
        tol: Tolerance for intersection detection
    
    Returns:
        True if path has loops, False otherwise
    """
    if len(path_points) < 4:
        return False
    
    for i in range(len(path_points) - 1):
        p0 = path_points[i]
        p1 = path_points[i + 1]
        
        # Check if this segment intersects with any future segment (not adjacent)
        for j in range(i + 2, len(path_points) - 1):
            p2 = path_points[j]
            p3 = path_points[j + 1]
            
            # Line segment intersection check
            denom = (p1[0] - p0[0]) * (p3[1] - p2[1]) - (p1[1] - p0[1]) * (p3[0] - p2[0])
            
            if abs(denom) > tol:
                t = ((p2[0] - p0[0]) * (p3[1] - p2[1]) - (p2[1] - p0[1]) * (p3[0] - p2[0])) / denom
                u = ((p2[0] - p0[0]) * (p1[1] - p0[1]) - (p2[1] - p0[1]) * (p1[0] - p0[0])) / denom
                
                # If segments intersect at interior points, this is a loop
                if tol < t < 1.0 - tol and tol < u < 1.0 - tol:
                    return True
    
    return False


def _remove_loops_from_path(path_points: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """
    Remove loops from a path by detecting self-intersections and removing looped segments.
    
    Uses a greedy approach: when a loop is detected, skip the segment that causes the loop.
    
    Args:
        path_points: Array of shape (N, 2) representing path points
        tol: Tolerance for intersection detection
    
    Returns:
        Cleaned path points with loops removed
    """
    if len(path_points) < 4:
        return path_points
    
    # First check if there are any loops
    if not _has_loop(path_points, tol):
        return path_points
    
    # Use a more aggressive approach: remove points that cause intersections
    cleaned_indices = [0]  # Always keep first point
    i = 1
    
    while i < len(path_points):
        # Try adding this point
        test_indices = cleaned_indices + [i]
        test_path = path_points[test_indices]
        
        # Check if adding this point creates a loop
        has_intersection = False
        if len(test_indices) >= 4:
            # Check if the last segment intersects with any previous segment
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
    
    # Always ensure we end with the final point
    if cleaned_indices[-1] != len(path_points) - 1:
        cleaned_indices.append(len(path_points) - 1)
    
    cleaned_path = path_points[cleaned_indices]
    
    # Verify no loops remain
    if _has_loop(cleaned_path, tol):
        # If still has loops, use a more aggressive approach: keep only every Nth point
        # This is a fallback
        step = max(2, len(path_points) // 20)
        cleaned_path = path_points[::step]
        if len(cleaned_path) > 0 and not np.array_equal(cleaned_path[-1], path_points[-1]):
            cleaned_path = np.vstack([cleaned_path, path_points[-1:]])
    
    return cleaned_path


def generate_optimal_reference_path(
    tunnel_path,
    tunnel_width,
    margin=0.001,
    num_knots=None,
    alpha=5.0,         # weight on Δd^2 (lateral slope)
    beta=5.0,         # weight on Δ²d^2 (lateral bending)
    lambda_length=0.01, # weight on length linear term
    gamma_center=1e-2  # tiny center bias (keeps solution off walls unless useful)
):
    """
    Generate a smooth, length-efficient reference path inside the tunnel using a convex QP in d(s).
    
    Path model (right-normal Frenet frame):
        p(s) = C(s) + d(s) * n_R(s)
    With right-pointing normal n_R(s) = [t_y, -t_x].
    
    Length linearization (with right normal):
        L(d) ≈ ∑ Δs_k * (1 + κ_k * d_k)   → linear term  +κ_k Δs_k · d_k
        so f_k = +lambda_length * κ_k * Δs_k

    Args:
        tunnel_path: List[(x,y)] centerline samples (entry→exit).
        tunnel_width: scalar width (use your per-sample width if available).
        margin: safety margin to walls.
        num_knots: discretization along arc length (auto if None).
        alpha, beta, lambda_length, gamma_center: objective weights.

    Returns:
        ReferencePath for the optimized path (spline through solved waypoints).
    """
    waypoints = np.asarray(tunnel_path, dtype=float)
    centerline = ReferencePath(waypoints, s=0.0, k=3)
    L = centerline.total_length

    # Discretize along arclength
    if num_knots is None:
        # ~2 points per original sample, but clamp
        num_knots = max(40, min(300, 2 * len(waypoints)))
    s_knots = np.linspace(0.0, L, num_knots)

    # Evaluate C(s), t(s), n_R(s), κ(s)
    C = np.stack([centerline(theta) for theta in s_knots], axis=0)            # (N,2)
    T = np.stack([centerline.tangent(theta) for theta in s_knots], axis=0)    # (N,2)
    N_right = np.stack([[t[1], -t[0]] for t in T], axis=0)                    # (N,2)
    kappa = np.array([centerline.curvature(theta) for theta in s_knots])      # (N,)
    ds = np.diff(s_knots, prepend=s_knots[0])
    ds[0] = ds[1] if len(ds) > 1 else 0.0

    # Corridor bounds (constant width here; replace with per-sample if you have it)
    half_w = np.full(num_knots, 0.5 * float(tunnel_width) - float(margin))
    half_w = np.maximum(half_w, 1e-6)  # avoid negative/zero

    N = num_knots

    # Build finite-difference matrices
    D1 = np.zeros((N-1, N))
    for k in range(N-1):
        D1[k, k]   = -1.0
        D1[k, k+1] =  1.0

    D2 = np.zeros((N-2, N))
    for k in range(N-2):
        D2[k, k]   =  1.0
        D2[k, k+1] = -2.0
        D2[k, k+2] =  1.0

    # Quadratic term (PSD)
    H = alpha * (D1.T @ D1) + beta * (D2.T @ D2) + gamma_center * np.eye(N)

    # Linear term from length (RIGHT normal → +κ d)
    f = lambda_length * (kappa * ds)

    # Bounds and pinned endpoints (keep entry/exit on centerline; change if needed)
    lb = (-half_w).copy()
    ub = (+half_w).copy()
    lb[0] = ub[0] = 0.0
    lb[-1] = ub[-1] = 0.0
    bounds = list(zip(lb, ub))

    # Solve the box-constrained quadratic with L-BFGS-B
    def objective(d):
        return 0.5 * d @ H @ d + f @ d
    
    def gradient(d):
        # analytic gradient: H d + f
        return H @ d + f

    x0 = np.zeros(N)
    res = minimize(
        objective, x0=np.zeros(N),
        method='L-BFGS-B',
        jac=gradient,                     # <<< provide gradient
        bounds=bounds,
        options={
            'maxiter': 5000,              # raise caps a bit
            'ftol': 1e-10,                # strict function tolerance
            'gtol': 1e-8,                 # gradient norm tolerance
            'maxls': 50                   # more line-search steps if needed
        }
    )

    if not res.success:
        # fallback to centerline offsets = 0
        d_opt = np.zeros(N)
    else:
        d_opt = res.x

    # Reconstruct the 2D path
    P = C + d_opt[:, None] * N_right
    P = np.asarray(P)

    # Check for loops in the optimized path and remove them
    P = _remove_loops_from_path(P)

    # Return a smoothed spline through the optimized points
    return ReferencePath(P, s=0.0, k=3)

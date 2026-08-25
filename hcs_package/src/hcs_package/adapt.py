"""Pure precomputation helpers for MPCC and reference-path generation."""

import numpy as np
from typing import Optional, Tuple, List

from .constraints import ConstraintType, RectangleConstraint, PolygonConstraint
from .constraint_utils import _point_in_polygon, _distance_to_polygon_boundary

_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))


def compute_local_curvature_integral(
    ref_path,
    s_samples: np.ndarray,
    window: float = 0.1,
) -> np.ndarray:
    """Total turning angle in a window: phi_k = ∫|kappa(s)| ds.

    Captures both local sharpness (tight 90° corner ≈ π/2) and turn density
    (three such corners ≈ 3π/2) in one signal — no peak detection needed.
    """
    s_samples = np.asarray(s_samples, dtype=float)
    kappa = np.array([abs(ref_path.curvature(float(s))) for s in s_samples])
    half_w = window / 2.0
    n = len(s_samples)
    phi = np.empty(n)
    for i in range(n):
        s_lo = s_samples[i] - half_w
        s_hi = s_samples[i] + half_w
        mask = (s_samples >= s_lo) & (s_samples <= s_hi)
        s_win = s_samples[mask]
        k_win = kappa[mask]
        if len(s_win) >= 2:
            phi[i] = float(_trapezoid(k_win, s_win))
        elif len(s_win) == 1:
            phi[i] = float(k_win[0] * min(window, s_samples[-1] - s_samples[0] + 1e-9))
        else:
            phi[i] = 0.0
    return phi


def compute_sharpness_profile(
    ref_path,
    s_samples: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute curvature and normalized curvature-rate (sharpness) along the path.

    Args:
        ref_path:  ReferencePath object.
        s_samples: 1-D array of arc-length positions at which to evaluate.

    Returns:
        sigma   : Normalized |∂κ/∂s| ∈ [0, 1].  Near 1 at sharp corners,
                  near 0 on straight or gently curved sections.
        kappa   : |κ(s)| at each sample.
        dkappa  : |∂κ/∂s| (un-normalized) at each sample.
    """
    s_samples = np.asarray(s_samples, dtype=float)
    kappa = np.array([abs(ref_path.curvature(float(s))) for s in s_samples])
    dkappa = np.abs(np.gradient(kappa, s_samples))
    sigma = dkappa / (np.max(dkappa) + 1e-9)
    return sigma, kappa, dkappa


def compute_curvature_rate_profile(
    ref_path,
    s_samples: np.ndarray,
    smooth_sigma: float = 4.0,
    trim_fraction: float = 0.05,
) -> np.ndarray:
    """Corner-difficulty signal κ(s) × |∂κ/∂s| — fires only where curvature
    is both high AND changing fast (i.e., at corners).
    """
    from scipy.ndimage import gaussian_filter1d

    s_samples = np.asarray(s_samples, dtype=float)
    n = len(s_samples)
    if n < 2:
        return np.zeros(n)

    kappa = np.array([abs(ref_path.curvature(float(s))) for s in s_samples])
    dkappa_ds = np.abs(np.gradient(kappa, s_samples))

    if smooth_sigma > 0:
        dkappa_ds = gaussian_filter1d(dkappa_ds, sigma=smooth_sigma)
        kappa_smooth = gaussian_filter1d(kappa, sigma=smooth_sigma)
    else:
        kappa_smooth = kappa

    signal = kappa_smooth * dkappa_ds

    if trim_fraction > 0 and n > 10:
        trim_samples = max(1, int(n * trim_fraction))
        signal[:trim_samples] = signal[trim_samples]
        signal[-trim_samples:] = signal[-trim_samples - 1]

    return signal


def compute_curvature_spike_profile(
    ref_path,
    s_samples: np.ndarray,
    window_samples: int = 30,
    smooth_sigma: float = 2.0,
) -> np.ndarray:
    """Curvature-rate spike metric: max(0, rate - local_median).

    Distinguishes CORNER tunnels (low baseline + sharp spikes) from
    SIGMOIDAL tunnels (continuously elevated rate, no spikes).
    """
    from scipy.ndimage import gaussian_filter1d, median_filter

    s_samples = np.asarray(s_samples, dtype=float)
    n = len(s_samples)
    if n < 2:
        return np.zeros(n)

    kappa = np.array([abs(ref_path.curvature(float(s))) for s in s_samples])
    rate = np.abs(np.gradient(kappa, s_samples))

    if smooth_sigma > 0:
        rate = gaussian_filter1d(rate, sigma=smooth_sigma)

    local_baseline = median_filter(rate, size=window_samples, mode='reflect')

    spike = np.maximum(0.0, rate - local_baseline)

    return spike


def _ray_to_polygon_boundary(origin: np.ndarray, direction: np.ndarray,
                              vertices: np.ndarray) -> float:
    """Distance from origin along direction to nearest polygon edge, or inf."""
    min_t = float('inf')
    n = len(vertices)
    dx, dy = float(direction[0]), float(direction[1])
    ox, oy = float(origin[0]), float(origin[1])

    for i in range(n):
        j = (i + 1) % n
        ex, ey = float(vertices[i][0]), float(vertices[i][1])
        fx, fy = float(vertices[j][0]), float(vertices[j][1])

        sx, sy = fx - ex, fy - ey
        denom = dx * sy - dy * sx
        if abs(denom) < 1e-12:
            continue

        t = ((ex - ox) * sy - (ey - oy) * sx) / denom  # ray parameter
        u = ((ex - ox) * dy - (ey - oy) * dx) / denom  # edge parameter

        if t > 1e-9 and 0.0 <= u <= 1.0:
            min_t = min(min_t, t)

    return min_t


def compute_clearance_profile(
    ref_path,
    s_samples: np.ndarray,
    corridor_bounds=None,
    cartesian_constraints: Optional[List] = None,
    unconstrained: str = "max",
) -> np.ndarray:
    """Local usable width — twice the distance from path to the nearest
    active constraint boundary (for a corridor, w_left + w_right).

    Geometry-agnostic generalisation of tunnel width, on a single scale so
    the difficulty budget's D0 and the speed model transfer between
    corridor-defined and cartesian-constraint tasks:
    * PathConstraint  (corridor_bounds):    w_left(s) + w_right(s)
    * PolygonConstraint KEEP_IN:            2 x distance to polygon boundary
    * RectangleConstraint KEEP_IN:          2 x distance to nearest rect edge

    Unconstrained samples receive the max finite clearance so they don't
    artificially cap speed (``unconstrained="max"``, legacy default), or
    stay at ``np.inf`` with ``unconstrained="inf"`` — the honest task-width
    semantics: where no constraint is active there is no width, and
    consumers that need a finite number must cap it themselves.
    """
    s_samples = np.asarray(s_samples, dtype=float)
    n = len(s_samples)
    clearance = np.full(n, np.inf)

    for i, s_i in enumerate(s_samples):
        p = ref_path(float(s_i))

        if corridor_bounds is not None:
            b_left_in, b_right_in = corridor_bounds
            wl = b_left_in(s_i) if callable(b_left_in) else float(b_left_in)
            wr = b_right_in(s_i) if callable(b_right_in) else float(b_right_in)
            clearance[i] = min(clearance[i], wl + wr)

        if cartesian_constraints:
            for region in cartesian_constraints:
                geom = region.geometry

                if isinstance(geom, PolygonConstraint):
                    if region.constraint_type == ConstraintType.KEEP_IN:
                        verts = np.array(geom.vertices, dtype=float)
                        if _point_in_polygon(p, verts):
                            d = _distance_to_polygon_boundary(p, verts)
                            clearance[i] = min(clearance[i], 2.0 * d)

                elif isinstance(geom, RectangleConstraint):
                    if region.constraint_type == ConstraintType.KEEP_IN:
                        d_left   = p[0] - geom.x
                        d_right  = geom.x + geom.width  - p[0]
                        d_bottom = p[1] - geom.y
                        d_top    = geom.y + geom.height - p[1]
                        d = min(d_left, d_right, d_bottom, d_top)
                        if d > 0:
                            clearance[i] = min(clearance[i], 2.0 * d)

    if unconstrained == "inf":
        return clearance

    finite_mask = np.isfinite(clearance)
    if finite_mask.any():
        clearance[~finite_mask] = float(np.max(clearance[finite_mask]))
    else:
        clearance[:] = 1.0  # fully unconstrained path

    return clearance


def compute_qp_bounds(
    centerline,
    s_knots: np.ndarray,
    C: np.ndarray,
    N_right: np.ndarray,
    tunnel_half_w: np.ndarray,
    cartesian_constraints: Optional[List] = None,
    margin: float = 0.001,
    n_probe: int = 40,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-knot lateral lower/upper bounds for the reference-path QP.

    Path model: p(s_k) = C(s_k) + d_k * n_R(s_k). For each knot, find the
    maximum |d| in each direction that stays inside every KEEP_IN constraint.
    Falls back to symmetric ``tunnel_half_w`` when no constraints are given.

    Returns (lb, ub) with lb <= 0 and ub >= 0 (lateral-offset convention).
    """
    N = len(s_knots)
    lb = -tunnel_half_w.copy()
    ub =  tunnel_half_w.copy()

    if not cartesian_constraints:
        return lb, ub

    keep_in_geoms = []
    for region in cartesian_constraints:
        if region.constraint_type != ConstraintType.KEEP_IN:
            continue
        geom = region.geometry
        if isinstance(geom, PolygonConstraint):
            keep_in_geoms.append(('polygon', np.array(geom.vertices, dtype=float)))
        elif isinstance(geom, RectangleConstraint):
            keep_in_geoms.append(('rect', geom))

    if not keep_in_geoms:
        return lb, ub

    def _inside_all(pt):
        for gtype, gdata in keep_in_geoms:
            if gtype == 'polygon':
                if not _point_in_polygon(pt, gdata):
                    return False
            else:
                g = gdata
                if not (g.x <= pt[0] <= g.x + g.width and
                        g.y <= pt[1] <= g.y + g.height):
                    return False
        return True

    max_d = float(np.max(tunnel_half_w)) * 2.0
    probe_offsets = np.linspace(0.0, max_d, n_probe)

    for k in range(N):
        c_k = C[k]
        n_k = N_right[k]

        # Largest contiguous feasible d starting from 0, in each direction.
        inside_pos = np.array([_inside_all(c_k + d * n_k) for d in probe_offsets])
        if inside_pos[0]:
            exit_idx = np.argmin(inside_pos)
            if inside_pos[exit_idx]:
                best_pos = probe_offsets[-1]
            else:
                best_pos = probe_offsets[max(0, exit_idx - 1)]
        else:
            best_pos = 0.0
        ub[k] = min(ub[k], max(best_pos - margin, 0.0))

        inside_neg = np.array([_inside_all(c_k - d * n_k) for d in probe_offsets])
        if inside_neg[0]:
            exit_idx = np.argmin(inside_neg)
            if inside_neg[exit_idx]:
                best_neg = probe_offsets[-1]
            else:
                best_neg = probe_offsets[max(0, exit_idx - 1)]
        else:
            best_neg = 0.0
        lb[k] = max(lb[k], -max(best_neg - margin, 0.0))

    return lb, ub

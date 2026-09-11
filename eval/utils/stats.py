"""Statistical utilities for evaluation."""

import numpy as np


def _centerline_arcs(centerline):
    cl = np.asarray(centerline, dtype=float)
    seg = np.diff(cl, axis=0)                       # (M, 2)
    seg_len = np.linalg.norm(seg, axis=1)           # (M,)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    return cl, seg, seg_len, cum


def project_progress(trajectory, centerline):
    """Progress (arc length / total, in [0, 1]) and signed lateral offset of
    every trajectory point at its closest point on the polyline centerline.

    Vectorised (2026-09-11) replacement of the point x segment Python loop:
    all N x M clipped segment projections at once, first minimum-distance
    segment per point (np.argmin), so ties resolve as the loop's strict '<'
    did. Same arithmetic per element, results equal to ~1e-15; the old loop
    was ~4 s per condition and the whole of an eval aggregate's run time.

    Returns (progress (N,), lateral (N,), cl_total). lateral is positive to
    the left of the path direction; 0 on degenerate segments.
    """
    traj = np.asarray(trajectory, dtype=float).reshape(-1, 2)
    cl, seg, seg_len, cum = _centerline_arcs(centerline)
    cl_total = cum[-1]
    n, m = len(traj), len(seg)
    if cl_total < 1e-9 or n == 0 or m == 0:
        return np.zeros(n), np.zeros(n), cl_total
    seg_len2 = np.einsum("ij,ij->i", seg, seg)                    # (M,)
    rel = traj[:, None, :] - cl[None, :-1, :]                     # (N, M, 2) = pt - cl[i]
    dots = rel[..., 0] * seg[None, :, 0] + rel[..., 1] * seg[None, :, 1]
    ok = seg_len2 >= 1e-18
    t = np.zeros((n, m))
    t[:, ok] = np.clip(dots[:, ok] / seg_len2[ok], 0.0, 1.0)
    proj = cl[None, :-1, :] + t[..., None] * seg[None, :, :]      # (N, M, 2)
    off = traj[:, None, :] - proj                                 # (N, M, 2) = pt - proj
    d2 = off[..., 0] ** 2 + off[..., 1] ** 2
    best = np.argmin(d2, axis=1)                                  # first minimum, as the loop's '<'
    rows = np.arange(n)
    t_b = t[rows, best]
    arc = cum[best] + t_b * seg_len[best]
    # signed lateral: perp = (-seg_y, seg_x) / |seg|
    perp = np.stack([-seg[:, 1], seg[:, 0]], axis=1)
    perp_len = np.linalg.norm(perp, axis=1)
    lat = np.zeros(n)
    good = perp_len[best] > 1e-12
    if good.any():
        pb = perp[best[good]] / perp_len[best[good]][:, None]
        ob = off[rows[good], best[good]]
        lat[good] = ob[:, 0] * pb[:, 0] + ob[:, 1] * pb[:, 1]
    return arc / cl_total, lat, cl_total


def resample_by_progress(trajectory, centerline, n_bins=100):
    """Resample a trajectory at uniform progress points along the centerline.

    Returns:
        progress: (n_bins,) array of progress values [0, 1]
        positions: (n_bins, 2) array of interpolated trajectory positions
        lateral: (n_bins,) array of signed lateral deviation from centerline
    """
    traj = np.asarray(trajectory, dtype=float)
    traj_progress, traj_lateral, cl_total = project_progress(traj, centerline)
    if cl_total < 1e-9:
        return np.linspace(0, 1, n_bins), np.tile(traj[0], (n_bins, 1)), np.zeros(n_bins)

    # Resample at uniform progress bins
    progress_bins = np.linspace(0, 1, n_bins)
    lat_resampled = np.interp(progress_bins, traj_progress, traj_lateral)

    # Also resample x, y positions
    pos_x = np.interp(progress_bins, traj_progress, traj[:, 0])
    pos_y = np.interp(progress_bins, traj_progress, traj[:, 1])
    positions = np.column_stack([pos_x, pos_y])

    return progress_bins, positions, lat_resampled


def resample_speeds_by_progress(speeds, trajectory, centerline, n_bins=100):
    """Resample speed profile at uniform progress along centerline."""
    traj = np.asarray(trajectory, dtype=float)
    spd = np.asarray(speeds, dtype=float)
    traj_progress, _lat, cl_total = project_progress(traj, centerline)
    if cl_total < 1e-9 or len(spd) < 2:
        return np.linspace(0, 1, n_bins), np.zeros(n_bins)

    # Truncate speeds to trajectory length
    spd = spd[:len(traj)]
    if len(spd) < len(traj):
        spd = np.pad(spd, (0, len(traj) - len(spd)), mode='edge')

    progress_bins = np.linspace(0, 1, n_bins)
    spd_resampled = np.interp(progress_bins, traj_progress, spd)
    return progress_bins, spd_resampled


def trajectory_rmse(lat1, lat2):
    """RMSE of lateral deviation between two resampled trajectories."""
    return float(np.sqrt(np.mean((np.asarray(lat1) - np.asarray(lat2)) ** 2)))


def speed_profile_correlation(spd1, spd2):
    """Pearson correlation between two resampled speed profiles."""
    s1, s2 = np.asarray(spd1), np.asarray(spd2)
    if s1.std() < 1e-12 or s2.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(s1, s2)[0, 1])


def speed_profile_rmse(spd1, spd2):
    """RMSE between two resampled speed profiles."""
    return float(np.sqrt(np.mean((np.asarray(spd1) - np.asarray(spd2)) ** 2)))

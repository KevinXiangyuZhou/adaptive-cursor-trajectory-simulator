"""Statistical utilities for evaluation."""

import numpy as np


def resample_by_progress(trajectory, centerline, n_bins=100):
    """Resample a trajectory at uniform progress points along the centerline.

    Returns:
        progress: (n_bins,) array of progress values [0, 1]
        positions: (n_bins, 2) array of interpolated trajectory positions
        lateral: (n_bins,) array of signed lateral deviation from centerline
    """
    traj = np.asarray(trajectory, dtype=float)
    cl = np.asarray(centerline, dtype=float)

    # Compute centerline arc-length
    cl_diffs = np.diff(cl, axis=0)
    cl_seg_lens = np.linalg.norm(cl_diffs, axis=1)
    cl_cum = np.concatenate([[0.0], np.cumsum(cl_seg_lens)])
    cl_total = cl_cum[-1]
    if cl_total < 1e-9:
        return np.linspace(0, 1, n_bins), np.tile(traj[0], (n_bins, 1)), np.zeros(n_bins)

    # For each trajectory point, find closest point on centerline → progress
    traj_progress = []
    traj_lateral = []
    for pt in traj:
        best_dist2 = np.inf
        best_arc = 0.0
        best_lateral = 0.0
        for i in range(len(cl) - 1):
            seg = cl[i + 1] - cl[i]
            seg_len2 = np.dot(seg, seg)
            if seg_len2 < 1e-18:
                t = 0.0
            else:
                t = np.clip(np.dot(pt - cl[i], seg) / seg_len2, 0.0, 1.0)
            proj = cl[i] + t * seg
            d2 = np.sum((pt - proj) ** 2)
            if d2 < best_dist2:
                best_dist2 = d2
                best_arc = cl_cum[i] + t * cl_seg_lens[i]
                # Signed lateral: positive = left of path direction
                perp = np.array([-seg[1], seg[0]])
                perp_len = np.linalg.norm(perp)
                if perp_len > 1e-12:
                    best_lateral = np.dot(pt - proj, perp / perp_len)
                else:
                    best_lateral = 0.0
        traj_progress.append(best_arc / cl_total)
        traj_lateral.append(best_lateral)

    traj_progress = np.array(traj_progress)
    traj_lateral = np.array(traj_lateral)

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
    cl = np.asarray(centerline, dtype=float)
    spd = np.asarray(speeds, dtype=float)

    # Minimal: use same progress computation
    cl_diffs = np.diff(cl, axis=0)
    cl_seg_lens = np.linalg.norm(cl_diffs, axis=1)
    cl_cum = np.concatenate([[0.0], np.cumsum(cl_seg_lens)])
    cl_total = cl_cum[-1]
    if cl_total < 1e-9 or len(spd) < 2:
        return np.linspace(0, 1, n_bins), np.zeros(n_bins)

    traj_progress = []
    for pt in traj:
        best_dist2 = np.inf
        best_arc = 0.0
        for i in range(len(cl) - 1):
            seg = cl[i + 1] - cl[i]
            seg_len2 = np.dot(seg, seg)
            t = 0.0 if seg_len2 < 1e-18 else np.clip(np.dot(pt - cl[i], seg) / seg_len2, 0.0, 1.0)
            proj = cl[i] + t * seg
            d2 = np.sum((pt - proj) ** 2)
            if d2 < best_dist2:
                best_dist2 = d2
                best_arc = cl_cum[i] + t * cl_seg_lens[i]
        traj_progress.append(best_arc / cl_total)

    traj_progress = np.array(traj_progress)
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

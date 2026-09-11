"""Out-of-bounds detection for model trajectories.

Detects when the cursor persistently exits tunnel boundaries due to noise.
Brief exits (a few steps) are tolerated; only persistent violations trigger
an OOB flag, causing the round to be excluded from aggregated metrics.
"""

import numpy as np


def check_trajectory_oob(trajectory, centerline, half_width, max_consecutive=5):
    """Check if a trajectory has persistent out-of-bounds violations.

    For each trajectory point, finds the nearest centerline point and checks
    whether the lateral distance exceeds the tunnel half-width.  Brief exits
    (fewer than *max_consecutive* consecutive steps) are tolerated.

    Args:
        trajectory: List of (x, y) points in meters.
        centerline: List of (x, y) centerline points in meters.
        half_width: Tunnel half-width in meters (scalar).
        max_consecutive: Number of consecutive OOB steps before flagging.

    Returns:
        Tuple of (is_oob, total_oob_steps, max_consecutive_oob).
    """
    cl = np.asarray(centerline, dtype=float)
    consecutive = 0
    max_consec = 0
    total_oob = 0

    for pt in trajectory:
        dists = np.linalg.norm(cl - np.asarray(pt, dtype=float), axis=1)
        min_dist = float(np.min(dists))

        if min_dist > half_width:
            consecutive += 1
            total_oob += 1
            max_consec = max(max_consec, consecutive)
        else:
            consecutive = 0

    return (max_consec >= max_consecutive), total_oob, max_consec

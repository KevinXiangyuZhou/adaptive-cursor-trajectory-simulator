"""
Extract (geometry, speed) training pairs from human trial data.

For each timestep in each human trial, computes:
  - clearance: local distance to nearest boundary
  - kappa: |curvature| of the path at the cursor's progress
  - dkappa_ds: |d_kappa/ds| of the centerline at the cursor's progress
  - speed: human speed (central-difference + 5pt moving average)

These features are used to fit the GAM speed model.

Two variants:
  - extract_speed_features / extract_all_speed_data:
      Legacy — uses constant tunnel_half_width for clearance and
      centerline curvature. Fast but creates train-test mismatch
      with simulation.

  - extract_speed_features_v2 / extract_all_speed_data_v2:
      Uses fitted reference path to compute features identically to
      simulation time (compute_clearance_profile on ref path,
      curvature from ref path, curvature rate from centerline remapped
      to ref path progress).
"""

import numpy as np
from hcs_package.reference_path import ReferencePath
from hcs_package.adapt import compute_clearance_profile, compute_curvature_rate_profile


def _compute_progress(trajectory, centerline_pts):
    """Map each trajectory point to arc-length progress along the centerline.

    Uses nearest-segment projection with a monotonicity constraint so that
    progress never decreases.  A forward-search window avoids O(N*M)
    worst-case when both trajectory and polyline are long.

    Returns:
        progress: array of arc-length values (N,).
        centerline_lengths: total arc-length of the centerline.
    """
    cl = np.array(centerline_pts, dtype=float)
    cl_diffs = np.diff(cl, axis=0)
    cl_lens = np.linalg.norm(cl_diffs, axis=1)
    cl_cum = np.concatenate([[0.0], np.cumsum(cl_lens)])
    cl_total = cl_cum[-1]

    traj = np.array(trajectory, dtype=float)
    n_traj = len(traj)
    n_seg = len(cl) - 1
    progress = np.empty(n_traj)

    # Search window: start from the segment of the previous match
    search_start = 0
    prev_arc = 0.0

    for i in range(n_traj):
        pt = traj[i]
        best_dist = float('inf')
        best_arc = prev_arc  # default: don't go backwards

        # Search forward from the previous best segment, with a lookback
        # buffer to handle minor trajectory jitter.
        lookback = min(5, search_start)
        seg_lo = search_start - lookback
        seg_hi = min(n_seg, search_start + n_seg)  # search all remaining

        for j in range(seg_lo, seg_hi):
            jj = j % n_seg if j >= n_seg else j
            seg = cl[jj + 1] - cl[jj]
            seg_len2 = np.dot(seg, seg)
            if seg_len2 < 1e-18:
                t = 0.0
            else:
                t = float(np.clip(np.dot(pt - cl[jj], seg) / seg_len2, 0.0, 1.0))
            closest = cl[jj] + t * seg
            dist = float(np.linalg.norm(pt - closest))
            arc = cl_cum[jj] + t * cl_lens[jj]

            # Only consider this segment if it doesn't move us backwards
            if arc >= prev_arc and dist < best_dist:
                best_dist = dist
                best_arc = arc
                search_start = jj

        progress[i] = best_arc
        prev_arc = best_arc

    return progress, cl_total


def extract_speed_features(human_trial, centerline_pts, tunnel_half_width):
    """Extract per-timestep geometric features and speed from one human trial.

    Args:
        human_trial: dict with keys 'trajectory', 'speeds', 'timestamps'.
        centerline_pts: list of [x, y] centerline points.
        tunnel_half_width: half-width of the tunnel (meters).

    Returns:
        dict with arrays: {s, clearance, kappa, dkappa_ds, speed}.
        All arrays have length N (number of timesteps).
    """
    traj = human_trial["trajectory"]
    speeds = human_trial["speeds"]

    if len(traj) < 5 or len(speeds) < 5:
        return None

    # Build centerline spline
    cl_path = ReferencePath(centerline_pts, s=0.0, k=3)
    cl_total = cl_path.total_length

    # Map trajectory to progress along centerline
    progress, _ = _compute_progress(traj, centerline_pts)
    # Normalize to arc-length on the spline
    s_values = progress / max(progress[-1], 1e-9) * cl_total
    s_values = np.clip(s_values, 0, cl_total)

    # Compute curvature and curvature rate on the centerline
    n_profile = 500
    s_profile = np.linspace(0, cl_total, n_profile)
    kappa_profile = np.array([abs(cl_path.curvature(float(s))) for s in s_profile])
    dkappa_profile = np.abs(np.gradient(kappa_profile, s_profile))

    # Interpolate features at each trajectory point's progress
    kappa_at_s = np.interp(s_values, s_profile, kappa_profile)
    dkappa_at_s = np.interp(s_values, s_profile, dkappa_profile)

    # Clearance: for current tunnel types, width is constant along the path.
    # Use tunnel half-width as clearance (this generalizes to variable-width
    # tunnels when the clearance profile varies along s).
    clearance_at_s = np.full(len(traj), tunnel_half_width)

    # Human speeds from _compute_speeds() are in the same units as the
    # trajectory coordinates.  For this project, trajectory coordinates are
    # already in meters and timestamps in milliseconds, so speeds are in m/s.
    speed_arr = np.array(speeds, dtype=float)

    return {
        "s": s_values,
        "clearance": clearance_at_s,
        "kappa": kappa_at_s,
        "dkappa_ds": dkappa_at_s,
        "speed": speed_arr,
    }


def extract_all_speed_data(participant_data, centerlines, trial_conditions):
    """Extract features from all trials of one participant.

    Args:
        participant_data: dict {trial_id: [list of round dicts]}.
        centerlines: dict {trial_id: centerline_pts}.
        trial_conditions: dict {trial_id: condition_dict} with 'width' key.

    Returns:
        Combined dict {clearance, kappa, dkappa_ds, speed, trial_id} with
        concatenated arrays across all trials and rounds.
    """
    all_clearance = []
    all_kappa = []
    all_dkappa = []
    all_speed = []
    all_tid = []

    for tid, rounds in participant_data.items():
        if tid not in centerlines or tid not in trial_conditions:
            continue
        half_w = trial_conditions[tid]["width"] * 0.5
        cl = centerlines[tid]

        for rd in rounds:
            features = extract_speed_features(rd, cl, half_w)
            if features is None:
                continue
            n = len(features["speed"])
            all_clearance.append(features["clearance"])
            all_kappa.append(features["kappa"])
            all_dkappa.append(features["dkappa_ds"])
            all_speed.append(features["speed"])
            all_tid.append(np.full(n, tid))

    if not all_clearance:
        return None

    return {
        "clearance": np.concatenate(all_clearance),
        "kappa": np.concatenate(all_kappa),
        "dkappa_ds": np.concatenate(all_dkappa),
        "speed": np.concatenate(all_speed),
        "trial_id": np.concatenate(all_tid),
    }


# ---------------------------------------------------------------------------
# V2: Reference-path-aligned feature extraction
# ---------------------------------------------------------------------------

def extract_speed_features_v2(human_trial, ref_path, ref_polyline,
                               centerline_spline, corridor_bounds,
                               cartesian_regions):
    """Extract per-timestep features aligned with simulation-time computation.

    Features are computed on the fitted reference path (not the centerline),
    matching how the simulator pre-computes profiles in cursor_simulator.py.

    Args:
        human_trial:        dict with 'trajectory', 'speeds'.
        ref_path:           ReferencePath object (fitted, from Phase 0).
        ref_polyline:       (M, 2) densely sampled points on ref_path.
        centerline_spline:  ReferencePath object for the centerline.
        corridor_bounds:    Tuple (left_bound, right_bound) or None.
        cartesian_regions:  List of ConstraintRegion objects or [].

    Returns:
        dict with arrays: {s, clearance, kappa, dkappa_ds, speed}.
    """
    traj = human_trial["trajectory"]
    speeds = human_trial["speeds"]

    if len(traj) < 5 or len(speeds) < 5:
        return None

    ref_total = ref_path.total_length

    # Map human trajectory → progress along reference path polyline
    progress, _ = _compute_progress(traj, ref_polyline)
    # Scale polyline progress to ref_path arc-length
    polyline_total = _polyline_length(ref_polyline)
    s_values = progress / max(polyline_total, 1e-9) * ref_total
    s_values = np.clip(s_values, 0, ref_total)

    # Pre-compute profiles on a dense grid (matching cursor_simulator.py)
    n_profile = 500
    s_profile = np.linspace(0, ref_total, n_profile)

    # Clearance: distance from reference path to corridor walls
    # (matches cursor_simulator.py lines 365-369)
    c_profile = compute_clearance_profile(
        ref_path, s_profile,
        corridor_bounds=corridor_bounds,
        cartesian_constraints=cartesian_regions if cartesian_regions else None,
    )
    clearance_at_s = np.interp(s_values, s_profile, c_profile)

    # Kappa: curvature on the reference path
    # (matches cursor_simulator.py line 393)
    kappa_profile = np.array([abs(ref_path.curvature(float(s)))
                              for s in s_profile])
    kappa_at_s = np.interp(s_values, s_profile, kappa_profile)

    # Dkappa_ds: curvature rate from CENTERLINE, remapped to ref path progress
    # (matches cursor_simulator.py lines 380-388)
    cl_total = centerline_spline.total_length
    n_cl = 500
    s_cl = np.linspace(0, cl_total, n_cl)
    rate_cl = compute_curvature_rate_profile(centerline_spline, s_cl)
    # Remap via normalized progress
    progress_cl = s_cl / cl_total
    progress_opt = s_profile / ref_total
    rate_profile = np.interp(progress_opt, progress_cl, rate_cl)
    dkappa_at_s = np.interp(s_values, s_profile, rate_profile)

    speed_arr = np.array(speeds, dtype=float)

    return {
        "s": s_values,
        "clearance": clearance_at_s,
        "kappa": kappa_at_s,
        "dkappa_ds": dkappa_at_s,
        "speed": speed_arr,
    }


def _polyline_length(pts):
    """Total arc length of a polyline given as (M, 2) array."""
    pts = np.asarray(pts, dtype=float)
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def extract_all_speed_data_v2(participant_data, task_geometry, ref_path_params,
                               build_ref_path_fn):
    """Extract features from all trials using reference-path-aligned computation.

    Args:
        participant_data:  dict {trial_id: [round dicts]}.
        task_geometry:     dict {trial_id: geometry_dict} from _precompute_task_geometry.
        ref_path_params:   dict with fitted reference path parameters.
        build_ref_path_fn: callable(geom, ref_params) -> (ref_polyline, ref_path).

    Returns:
        Combined dict {clearance, kappa, dkappa_ds, speed, trial_id}.
    """
    all_clearance = []
    all_kappa = []
    all_dkappa = []
    all_speed = []
    all_tid = []

    for tid, rounds in participant_data.items():
        if tid not in task_geometry:
            continue
        geom = task_geometry[tid]

        ref_polyline, ref_path = build_ref_path_fn(geom, ref_path_params)

        for rd in rounds:
            features = extract_speed_features_v2(
                rd, ref_path, ref_polyline,
                centerline_spline=geom["centerline_spline"],
                corridor_bounds=geom["corridor_bounds"],
                cartesian_regions=geom["cartesian_regions"],
            )
            if features is None:
                continue
            n = len(features["speed"])
            all_clearance.append(features["clearance"])
            all_kappa.append(features["kappa"])
            all_dkappa.append(features["dkappa_ds"])
            all_speed.append(features["speed"])
            all_tid.append(np.full(n, tid))

    if not all_clearance:
        return None

    return {
        "clearance": np.concatenate(all_clearance),
        "kappa": np.concatenate(all_kappa),
        "dkappa_ds": np.concatenate(all_dkappa),
        "speed": np.concatenate(all_speed),
        "trial_id": np.concatenate(all_tid),
    }

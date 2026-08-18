"""
Extract (geometry, speed) training pairs from human tunnel trials for the
GAM speed model.

For each timestep of each human round we compute — on the FITTED reference
path, exactly as the simulator does at run time (cursor_simulator.py) —
  - clearance : local distance from the reference path to the corridor walls
  - kappa     : |curvature| of the reference path at the cursor's progress
  - dkappa_ds : |curvature rate| of the centreline, remapped to ref-path progress
  - speed     : human speed (central difference + 5-pt moving average)
  - progress  : normalised arc-length progress in [0, 1]

Ported from the neuromorphic-simulation repo (v2 extraction only; the legacy
constant-clearance variant is dropped).
"""

import numpy as np

from hcs_package.adapt import compute_clearance_profile, compute_curvature_rate_profile


def _compute_progress(trajectory, centerline_pts):
    """Map each trajectory point to arc-length progress along a polyline.

    Nearest-segment projection with a monotonicity constraint (progress never
    decreases) and a forward search window.

    Returns:
        progress (N,), total polyline length.
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

    search_start = 0
    prev_arc = 0.0
    for i in range(n_traj):
        pt = traj[i]
        best_dist = float("inf")
        best_arc = prev_arc
        lookback = min(5, search_start)
        for jj in range(search_start - lookback, n_seg):
            seg = cl[jj + 1] - cl[jj]
            seg_len2 = float(np.dot(seg, seg))
            t = 0.0 if seg_len2 < 1e-18 else float(np.clip(np.dot(pt - cl[jj], seg) / seg_len2, 0.0, 1.0))
            closest = cl[jj] + t * seg
            dist = float(np.linalg.norm(pt - closest))
            arc = cl_cum[jj] + t * cl_lens[jj]
            if arc >= prev_arc and dist < best_dist:
                best_dist = dist
                best_arc = arc
                search_start = jj
        progress[i] = best_arc
        prev_arc = best_arc
    return progress, cl_total


def _polyline_length(pts):
    pts = np.asarray(pts, dtype=float)
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def extract_speed_features_v2(human_trial, ref_path, ref_polyline,
                              centerline_spline, corridor_bounds,
                              cartesian_regions):
    """Per-timestep features aligned with simulation-time computation.

    Returns dict {s, progress, clearance, kappa, dkappa_ds, speed} or None.
    """
    traj = human_trial["trajectory"]
    speeds = human_trial["speeds"]
    if len(traj) < 5 or len(speeds) < 5:
        return None

    ref_total = ref_path.total_length
    progress, _ = _compute_progress(traj, ref_polyline)
    polyline_total = _polyline_length(ref_polyline)
    s_values = np.clip(progress / max(polyline_total, 1e-9) * ref_total, 0, ref_total)

    n_profile = 500
    s_profile = np.linspace(0, ref_total, n_profile)

    c_profile = compute_clearance_profile(
        ref_path, s_profile,
        corridor_bounds=corridor_bounds,
        cartesian_constraints=cartesian_regions if cartesian_regions else None,
    )
    clearance_at_s = np.interp(s_values, s_profile, c_profile)

    kappa_profile = np.array([abs(ref_path.curvature(float(s))) for s in s_profile])
    kappa_at_s = np.interp(s_values, s_profile, kappa_profile)

    cl_total = centerline_spline.total_length
    s_cl = np.linspace(0, cl_total, 500)
    rate_cl = compute_curvature_rate_profile(centerline_spline, s_cl)
    rate_profile = np.interp(s_profile / ref_total, s_cl / cl_total, rate_cl)
    dkappa_at_s = np.interp(s_values, s_profile, rate_profile)

    return {
        "s": s_values,
        "progress": s_values / max(ref_total, 1e-9),
        "clearance": clearance_at_s,
        "kappa": kappa_at_s,
        "dkappa_ds": dkappa_at_s,
        "speed": np.array(speeds, dtype=float),
    }


def extract_all_speed_data_v2(participant_data, task_geometry, ref_path_params,
                              build_ref_path_fn, progress_window=None):
    """Extract features from all rounds of all tunnel trials.

    Args:
        participant_data:  {trial_id: [round dicts]}.
        task_geometry:     {trial_id: geometry dict} (see fit_speed_model._precompute_task_geometry).
        ref_path_params:   fitted reference-path params.
        build_ref_path_fn: callable(geom, ref_params) -> (ref_polyline, ref_path).
        progress_window:   optional (lo, hi) in [0,1]; keep only samples whose
                           progress lies inside (drops start-up / final-approach
                           samples so the GAM learns cruise speed, which is what
                           the planner asks it for).

    Returns dict {clearance, kappa, dkappa_ds, speed, trial_id} or None.
    """
    cols = {"clearance": [], "kappa": [], "dkappa_ds": [], "speed": [], "trial_id": []}
    for tid, rounds in participant_data.items():
        if tid not in task_geometry:
            continue
        geom = task_geometry[tid]
        ref_polyline, ref_path = build_ref_path_fn(geom, ref_path_params)
        for rd in rounds:
            f = extract_speed_features_v2(
                rd, ref_path, ref_polyline,
                centerline_spline=geom["centerline_spline"],
                corridor_bounds=geom["corridor_bounds"],
                cartesian_regions=geom["cartesian_regions"],
            )
            if f is None:
                continue
            keep = np.ones(len(f["speed"]), dtype=bool)
            if progress_window is not None:
                lo, hi = progress_window
                keep &= (f["progress"] >= lo) & (f["progress"] <= hi)
            if not keep.any():
                continue
            for k in ("clearance", "kappa", "dkappa_ds", "speed"):
                cols[k].append(f[k][keep])
            cols["trial_id"].append(np.full(int(keep.sum()), tid))
    if not cols["speed"]:
        return None
    return {k: np.concatenate(v) for k, v in cols.items()}

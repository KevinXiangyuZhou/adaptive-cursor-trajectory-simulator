"""
Shared statistics helpers for the eval-new-data pipeline, plus an
aggregate-statistics CLI for eval-main results directories.

Helper side (imported by run_eval.py): re-exports the general-purpose
trajectory/speed-profile stats from eval/utils/stats.py, and adds the
law-specific helpers (Fitts ID, centerline geometry, ID4SCS regression)
that don't belong in that shared, law-agnostic module.

CLI side: aggregates a results directory produced by run_eval.py
(Steering / Fitts / ID4SCS / ConstrainedToUnconstrained). Because each
per-participant run overwrites the shared top-level CSVs, the CLI reads
the per-trial results_summary.json files under participant_*/trial_*
(complete for every participant) and uses the top-level CSVs only for
participant-independent tid -> geometry maps (steering ID, ID4SCS
S1/C/S2 terms).

Usage:
    python stats.py <results_root>            # e.g. results/eval-main-8-25-26
    python stats.py <results_root> --by-participant
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

from utils.stats import (  # noqa: E402
    resample_by_progress,
    resample_speeds_by_progress,
    trajectory_rmse,
    speed_profile_rmse,
    speed_profile_correlation,
)

__all__ = [
    "resample_by_progress",
    "resample_speeds_by_progress",
    "trajectory_rmse",
    "speed_profile_rmse",
    "speed_profile_correlation",
    "centerline_arc_length",
    "segment_arc_lengths",
    "fitts_id",
    "fit_id4scs",
    "compute_comparison_metrics",
]

# Mirrors eval-fitts-continuous-well/run_eval.py exactly: skip first 10% of
# progress (startup acceleration), evaluate to 100% (full path incl. goal
# approach) when computing speed_rmse/speed_corr.
SPEED_TRIM_START = 10
SPEED_TRIM_END = 100
N_PROGRESS_BINS = 100
GOAL_APPROACH_BINS = (90, 100)


def centerline_arc_length(centerline):
    """Sum of Euclidean segment lengths along a centerline/path."""
    total = 0.0
    for i in range(1, len(centerline)):
        dx = centerline[i][0] - centerline[i - 1][0]
        dy = centerline[i][1] - centerline[i - 1][1]
        total += math.sqrt(dx * dx + dy * dy)
    return total


def segment_arc_lengths(centerline):
    """
    Split a two-segment tunnel's centerline at the same midpoint index used
    by _create_tunnel_env (experiment/environment.py) to change corridor
    width, and return (A1, A2): the arc length of each half.
    """
    n = len(centerline)
    if n < 2:
        return 0.0, 0.0
    mid = n // 2
    seg1 = centerline[: mid + 1]
    seg2 = centerline[mid:]
    return centerline_arc_length(seg1), centerline_arc_length(seg2)


def fitts_id(D, R):
    """Shannon ID = log2(D/(2R) + 1). R is target radius; 2R is target width."""
    return math.log2(D / (2 * R) + 1) if R > 0 else 0.0


def fit_id4scs(rows):
    """
    Multiple linear regression for the ID4SCS model:
        MT = a + b*S1 + c*C + d*S2

    Args:
        rows: list of dicts with keys 'S1', 'C', 'S2', 'MT'.

    Returns:
        dict with a, b, c, d, r_squared, n. Empty dict if too few rows.
    """
    if len(rows) < 5:
        return {}

    X = np.column_stack([
        np.ones(len(rows)),
        [r["S1"] for r in rows],
        [r["C"] for r in rows],
        [r["S2"] for r in rows],
    ])
    y = np.array([r["MT"] for r in rows])

    coeffs, _residuals, _rank, _sv = np.linalg.lstsq(X, y, rcond=None)
    a, b, c, d = coeffs
    pred = X @ coeffs
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "a": round(float(a), 6),
        "b": round(float(b), 6),
        "c": round(float(c), 6),
        "d": round(float(d), 6),
        "r_squared": round(r_sq, 4),
        "n": len(rows),
    }


def compute_comparison_metrics(model_trajs, model_speeds, model_cts,
                                human_trajs, human_speeds, human_cts,
                                centerline):
    """
    Experiment-main-style human-vs-model comparison metrics, ported from
    eval-fitts-continuous-well/run_eval.py's _build_condition_summary:
    lateral RMSE, speed-profile RMSE/correlation (trimmed to bins 10-100,
    i.e. skip the first 10% of progress), goal-approach speed (mean speed
    over the last 10% of progress) for both sides plus their relative
    difference, and relative completion-time difference.

    Requires >= 5 points on both sides of `centerline` (a straight 2-point
    line is fine for point-to-point Fitts tasks) to resample against;
    returns {} if there isn't enough data to compute a meaningful profile.
    """
    if not centerline or len(centerline) < 2:
        return {}

    all_spd_m, all_lat_m = [], []
    for traj, speeds in zip(model_trajs, model_speeds):
        if len(traj) < 5 or len(traj) != len(speeds):
            continue
        _, _, lat_m = resample_by_progress(traj, centerline, N_PROGRESS_BINS)
        _, spd_m = resample_speeds_by_progress(speeds, traj, centerline, N_PROGRESS_BINS)
        all_lat_m.append(lat_m)
        all_spd_m.append(spd_m)
    if not all_spd_m:
        return {}

    mean_spd_m = np.mean(all_spd_m, axis=0)
    mean_lat_m = np.mean(all_lat_m, axis=0)
    lo, hi = GOAL_APPROACH_BINS
    goal_spd_m = float(np.mean(mean_spd_m[lo:hi])) if len(mean_spd_m) >= hi else None

    result = {"goal_approach_speed_model": round(goal_spd_m, 4) if goal_spd_m is not None else None}

    all_spd_h, all_lat_h = [], []
    for traj, speeds in zip(human_trajs, human_speeds):
        if len(traj) < 5 or len(traj) != len(speeds):
            continue
        _, _, lat_h = resample_by_progress(traj, centerline, N_PROGRESS_BINS)
        _, spd_h = resample_speeds_by_progress(speeds, traj, centerline, N_PROGRESS_BINS)
        all_lat_h.append(lat_h)
        all_spd_h.append(spd_h)

    if all_lat_h and all_lat_m:
        mean_lat_h = np.mean(all_lat_h, axis=0)
        mean_spd_h = np.mean(all_spd_h, axis=0)

        lat_rmse = float(trajectory_rmse(mean_lat_m, mean_lat_h))

        spd_m_trim = mean_spd_m[SPEED_TRIM_START:SPEED_TRIM_END]
        spd_h_trim = mean_spd_h[SPEED_TRIM_START:SPEED_TRIM_END]
        spd_rmse = float(speed_profile_rmse(spd_m_trim, spd_h_trim))
        spd_corr = float(speed_profile_correlation(spd_m_trim, spd_h_trim))

        goal_spd_h = float(np.mean(mean_spd_h[lo:hi])) if len(mean_spd_h) >= hi else None
        goal_spd_diff = None
        if goal_spd_h is not None and goal_spd_m is not None and goal_spd_h > 0:
            goal_spd_diff = abs(goal_spd_m - goal_spd_h) / goal_spd_h

        mean_human_time = float(np.mean(human_cts)) if human_cts else None
        mean_model_time = float(np.mean(model_cts)) if model_cts else None
        time_diff = None
        if mean_human_time and mean_model_time and mean_human_time > 0:
            time_diff = abs(mean_model_time - mean_human_time) / mean_human_time

        result.update({
            "lateral_rmse": round(lat_rmse, 6),
            "speed_rmse": round(spd_rmse, 6),
            "speed_corr": round(spd_corr, 4),
            "goal_approach_speed_human": round(goal_spd_h, 4) if goal_spd_h is not None else None,
            "goal_approach_speed_diff": round(goal_spd_diff, 4) if goal_spd_diff is not None else None,
            "time_diff": round(time_diff, 4) if time_diff is not None else None,
            "human_time_mean_s": round(mean_human_time, 4) if mean_human_time is not None else None,
            "model_time_mean_s": round(mean_model_time, 4) if mean_model_time is not None else None,
        })

    return result


# ---------------------------------------------------------------------------
# Aggregate-statistics CLI over an eval-main results directory
# ---------------------------------------------------------------------------

METRIC_KEYS = ["lateral_rmse", "speed_rmse", "speed_corr", "time_diff"]

# (display name, subdirectory relative to the results root)
TASK_DIRS = [
    ("Steering", Path("Steering")),
    ("Fitts", Path("Fitts")),
    ("ID4SCS narrow_to_wide", Path("ID4SCS") / "narrow_to_wide"),
    ("ID4SCS wide_to_narrow", Path("ID4SCS") / "wide_to_narrow"),
    ("C2U", Path("ConstrainedToUnconstrained")),
]


def _read_csv_rows(path):
    if not Path(path).exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def load_trial_rows(results_root):
    """
    Walk participant_*/trial_*/results_summary.json for every task and
    return one row per (task, participant, trial) with the per-trial
    comparison metrics and both sides' completion times.
    """
    rows = []
    for task, sub in TASK_DIRS:
        base = results_root / sub
        if not base.is_dir():
            continue
        for pdir in sorted(base.glob("participant_*")):
            pid = pdir.name.replace("participant_", "")
            for tdir in sorted(pdir.glob("trial_*"),
                               key=lambda p: int(p.name.split("_")[-1])):
                sp = tdir / "results_summary.json"
                if not sp.exists():
                    continue
                with open(sp) as f:
                    data = json.load(f)
                met = data.get("metrics") or {}
                h_ct = met.get("human_time_mean_s")
                m_ct = met.get("model_time_mean_s")
                rows.append({
                    "task": task,
                    "pid": pid,
                    "tid": int(data.get("trial_id", tdir.name.split("_")[-1])),
                    "condition": data.get("condition", {}),
                    "metrics": met,
                    "human_cts": (data.get("human") or {}).get("completion_times", []),
                    "model_cts": (data.get("model") or {}).get("completion_times", []),
                    "human_ct": h_ct,
                    "model_ct": m_ct,
                    "ct_ratio": (m_ct / h_ct) if (h_ct and m_ct and h_ct > 0) else None,
                })
    return rows


def _agg(values):
    """Return (mean, std, n) for a list of numeric values, skipping None."""
    valid = [v for v in values if v is not None]
    if not valid:
        return None, None, 0
    std = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
    return float(np.mean(valid)), std, len(valid)


def _group_stats(rows):
    """Aggregate a list of trial rows into {metric: (mean, std, n), ...}."""
    stats = {k: _agg([r["metrics"].get(k) for r in rows]) for k in METRIC_KEYS}
    ratios = [r["ct_ratio"] for r in rows if r["ct_ratio"] is not None]
    stats["ct_ratio_med"] = float(np.median(ratios)) if ratios else None
    stats["ct_ratio_iqr"] = ((float(np.percentile(ratios, 25)),
                              float(np.percentile(ratios, 75)))
                             if ratios else None)
    stats["human_ct"], _, _ = _agg([r["human_ct"] for r in rows])
    stats["model_ct"], _, _ = _agg([r["model_ct"] for r in rows])
    stats["n"] = len(rows)
    return stats


def _print_group_table(title, groups):
    """groups: list of (label, stats-dict from _group_stats)."""
    print(f"\n  {title}")
    header = (f"  {'group':<26s}{'latRMSE':>10s}{'spdRMSE':>10s}{'spdCorr':>9s}"
              f"{'timeDiff':>10s}{'hCT(s)':>9s}{'mCT(s)':>9s}"
              f"{'CTratio med [IQR]':>20s}{'n':>5s}")
    print("  " + "-" * (len(header) - 2))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, st in groups:
        cells = [f"  {label:<26s}"]
        for k, w in zip(METRIC_KEYS, (10, 10, 9, 10)):
            m, _s, _n = st[k]
            cells.append(f"{m:>{w}.4f}" if m is not None else f"{'-':>{w}s}")
        for k in ("human_ct", "model_ct"):
            v = st[k]
            cells.append(f"{v:>9.3f}" if v is not None else f"{'-':>9s}")
        if st["ct_ratio_med"] is not None:
            lo, hi = st["ct_ratio_iqr"]
            cells.append(f"{st['ct_ratio_med']:>7.2f} [{lo:.2f},{hi:.2f}]".rjust(20))
        else:
            cells.append(f"{'-':>20s}")
        cells.append(f"{st['n']:>5d}")
        print("".join(cells))
    print("  " + "-" * (len(header) - 2))


def _wilcoxon_ct(rows, label):
    """Paired Wilcoxon on per-(participant, trial) human vs model CT."""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return
    pairs = [(r["human_ct"], r["model_ct"]) for r in rows
             if r["human_ct"] is not None and r["model_ct"] is not None]
    if len(pairs) < 6:
        return
    h = np.array([p[0] for p in pairs])
    m = np.array([p[1] for p in pairs])
    try:
        stat, p = wilcoxon(m, h)
    except ValueError:
        return
    direction = "model slower" if np.median(m - h) > 0 else "model faster"
    sig = ("***" if p < 0.001 else "**" if p < 0.01 else
           "*" if p < 0.05 else "n.s.")
    print(f"  Wilcoxon CT (model vs human, {label}): W={stat:.1f}, "
          f"p={p:.2e} {sig}  ({direction}, n={len(pairs)} pairs)")


def _pooled_condition_means(rows, side):
    """
    One MT per trial_id: flat pooled mean across all participants' rounds —
    matches run_eval.py's regression/plot methodology.
    side: 'human_cts' or 'model_cts'.
    """
    by_tid = defaultdict(list)
    for r in rows:
        by_tid[r["tid"]].extend(r[side])
    return {t: float(np.mean(v)) for t, v in by_tid.items() if v}


def _linear_law_fit(id_by_tid, mt_by_tid):
    """Fit MT = a + b*ID over conditions present in both maps."""
    tids = sorted(set(id_by_tid) & set(mt_by_tid))
    if len(tids) < 3:
        return None
    xs = np.array([id_by_tid[t] for t in tids])
    ys = np.array([mt_by_tid[t] for t in tids])
    b, a = np.polyfit(xs, ys, 1)
    pred = a + b * xs
    ss_res = float(np.sum((ys - pred) ** 2))
    ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"a": float(a), "b": float(b), "r_squared": r_sq, "n": len(tids)}


def _steering_id_map(results_root):
    """tid -> ID (L/W). Geometry-derived, identical across participants, so
    reading it from the (last-run-overwritten) top-level CSV is safe."""
    id_map = {}
    for r in _read_csv_rows(results_root / "Steering" / "steering_results.csv"):
        try:
            id_map[int(r["tid"])] = float(r["ID"])
        except (KeyError, ValueError):
            continue
    return id_map


def _id4scs_geometry(results_root):
    """(direction, tid) -> (S1, C, S2), from the per-direction results CSVs."""
    geo = {}
    for sub in ("narrow_to_wide", "wide_to_narrow"):
        p = results_root / "ID4SCS" / sub / f"id4scs_{sub}_results.csv"
        for r in _read_csv_rows(p):
            try:
                geo[(sub, int(r["tid"]))] = (float(r["S1"]), float(r["C"]),
                                             float(r["S2"]))
            except (KeyError, ValueError):
                continue
    return geo


def _fitts_id_map(rows):
    """tid -> nominal Shannon ID from each trial's condition geometry."""
    id_map = {}
    for r in rows:
        cond = r["condition"]
        D, R = cond.get("distance"), cond.get("targetRadius")
        if D and R:
            id_map[r["tid"]] = fitts_id(float(D), float(R))
    return id_map


def _law_str(fit):
    return (f"MT = {fit['b']:.3f}*ID + {fit['a']:.3f}   "
            f"(R²={fit['r_squared']:.3f}, n={fit['n']} conditions)")


def save_aggregate_csv(all_groups, out_path):
    fields = ["task", "grouping", "group", "metric", "mean", "std", "n"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for task, grouping, groups in all_groups:
            for label, st in groups:
                for k in METRIC_KEYS:
                    mean, std, n = st[k]
                    w.writerow({
                        "task": task, "grouping": grouping, "group": label,
                        "metric": k,
                        "mean": round(mean, 6) if mean is not None else "",
                        "std": round(std, 6) if std is not None else "",
                        "n": n,
                    })
                for k in ("ct_ratio_med", "human_ct", "model_ct"):
                    v = st.get(k)
                    w.writerow({
                        "task": task, "grouping": grouping, "group": label,
                        "metric": k,
                        "mean": round(v, 6) if v is not None else "",
                        "std": "", "n": st["n"],
                    })


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate statistics for an eval-main results directory")
    parser.add_argument("results_root", nargs="?",
                        default=str(PROJECT_ROOT / "results" / "eval-main-8-25-26"),
                        help="results dir produced by run_eval.py")
    parser.add_argument("--by-participant", action="store_true",
                        help="also print per-participant tables for each task")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    if not results_root.is_dir():
        sys.exit(f"Not found: {results_root}")

    rows = load_trial_rows(results_root)
    if not rows:
        sys.exit(f"No participant_*/trial_*/results_summary.json under {results_root}")

    pids = sorted({r["pid"] for r in rows})
    print("=" * 70)
    print(f"Aggregate statistics — {results_root}")
    print(f"Participants: {', '.join(pids)}  ({len(rows)} trial entries)")
    print("=" * 70)

    csv_pids = sorted({r.get("pid") for r in
                       _read_csv_rows(results_root / "Steering" / "steering_results.csv")
                       if r.get("pid")})
    if csv_pids and set(csv_pids) != set(pids):
        print(f"  NOTE: top-level CSVs / fitts_regression.json contain only "
              f"{', '.join(csv_pids)} (each per-participant run overwrote them);"
              f"\n  all tables below are aggregated from the per-trial JSONs instead.")

    all_groups = []

    # Overall cross-task summary
    task_groups = []
    for task, _sub in TASK_DIRS:
        trows = [r for r in rows if r["task"] == task]
        if trows:
            task_groups.append((task, _group_stats(trows)))
    _print_group_table("Overall (per task, all participants pooled)", task_groups)
    all_groups.append(("all", "task", task_groups))

    # --- Steering ---------------------------------------------------------
    steer = [r for r in rows if r["task"] == "Steering"]
    if steer:
        print(f"\n[Steering]  {len({r['tid'] for r in steer})} conditions × "
              f"{len({r['pid'] for r in steer})} participants")
        by_type = defaultdict(list)
        for r in steer:
            cond = r["condition"]
            # baseline sinusoidal trials omit tunnelType; recover it from
            # the "sinusoidal, width 0.01"-style description
            ttype = cond.get("tunnelType") or \
                cond.get("description", "?").split(",")[0].strip()
            by_type[ttype].append(r)
        groups = [(t, _group_stats(v)) for t, v in sorted(by_type.items())]
        _print_group_table("By tunnel type", groups)
        all_groups.append(("Steering", "tunnel_type", groups))

        by_width = defaultdict(list)
        for r in steer:
            w = r["condition"].get("tunnelWidth")
            if w is not None:
                by_width[float(w)].append(r)
        groups = [(f"W={w*1000:.0f}mm", _group_stats(v))
                  for w, v in sorted(by_width.items())]
        _print_group_table("By tunnel width", groups)
        all_groups.append(("Steering", "width", groups))

        id_map = _steering_id_map(results_root)
        if id_map:
            print("\n  Steering law (pooled per-condition means):")
            for side, key in (("human", "human_cts"), ("model", "model_cts")):
                fit = _linear_law_fit(id_map, _pooled_condition_means(steer, key))
                if fit:
                    print(f"    {side:<6s} {_law_str(fit)}")
        _wilcoxon_ct(steer, "Steering")

    # --- Fitts ------------------------------------------------------------
    fitts = [r for r in rows if r["task"] == "Fitts"]
    if fitts:
        print(f"\n[Fitts]  {len({r['tid'] for r in fitts})} conditions × "
              f"{len({r['pid'] for r in fitts})} participants")
        by_r = defaultdict(list)
        for r in fitts:
            rad = r["condition"].get("targetRadius")
            if rad is not None:
                by_r[float(rad)].append(r)
        groups = [(f"R={rad*1000:.0f}mm", _group_stats(v))
                  for rad, v in sorted(by_r.items())]
        _print_group_table("By target radius", groups)
        all_groups.append(("Fitts", "target_radius", groups))

        id_map = _fitts_id_map(fitts)
        print("\n  Fitts' law, raw completion times (pooled per-condition means):")
        for side, key in (("human", "human_cts"), ("model", "model_cts")):
            fit = _linear_law_fit(id_map, _pooled_condition_means(fitts, key))
            if fit:
                tps = [id_map[r["tid"]] / ct for r in fitts
                       if r["tid"] in id_map for ct in r[key] if ct > 0]
                print(f"    {side:<6s} {_law_str(fit)}   "
                      f"TP={np.mean(tps):.2f}±{np.std(tps):.2f} bps")
        reg_path = results_root / "Fitts" / "fitts_regression.json"
        if reg_path.exists():
            reg = json.load(open(reg_path))
            aligned = reg.get("aligned", {})
            if aligned:
                who = f" ({', '.join(csv_pids)} only)" if csv_pids else ""
                print(f"  Saved aligned (MT_kin) fit from fitts_regression.json{who}:")
                for side in ("human", "model"):
                    m = aligned.get(side) or {}
                    if m:
                        print(f"    {side:<6s} MT = {m['b_slope_s_per_bit']:.3f}*ID "
                              f"+ {m['a_intercept']:.3f}   (R²={m['r_squared']:.3f}, "
                              f"TP={m['throughput_mean_bps']:.2f} bps)")
        _wilcoxon_ct(fitts, "Fitts")

    # --- ID4SCS -----------------------------------------------------------
    id4_rows = {"narrow_to_wide": [r for r in rows if r["task"] == "ID4SCS narrow_to_wide"],
                "wide_to_narrow": [r for r in rows if r["task"] == "ID4SCS wide_to_narrow"]}
    if any(id4_rows.values()):
        print("\n[ID4SCS]")
        groups = [(sub, _group_stats(v)) for sub, v in id4_rows.items() if v]
        _print_group_table("By direction", groups)
        all_groups.append(("ID4SCS", "direction", groups))

        geo = _id4scs_geometry(results_root)
        if geo:
            print("\n  ID4SCS regression MT = a + b*S1 + c*C + d*S2 "
                  "(both directions combined, pooled per-condition means):")
            side_rows, side_fits = {}, {}
            for side, key in (("human", "human_cts"), ("model", "model_cts")):
                reg_rows = []
                for sub, v in id4_rows.items():
                    mts = _pooled_condition_means(v, key)
                    for tid, mt in mts.items():
                        if (sub, tid) in geo:
                            s1, c, s2 = geo[(sub, tid)]
                            reg_rows.append({"S1": s1, "C": c, "S2": s2, "MT": mt,
                                             "direction": sub})
                fit = fit_id4scs(reg_rows)
                side_rows[side], side_fits[side] = reg_rows, fit
                if fit:
                    print(f"    {side:<6s} a={fit['a']:.3f} b={fit['b']:.3f} "
                          f"c={fit['c']:.3f} d={fit['d']:.3f}   "
                          f"(R²={fit['r_squared']:.3f}, n={fit['n']} conditions)")
                else:
                    print(f"    {side:<6s} too few conditions to fit (n={len(reg_rows)})")
            if any(side_fits.values()):
                sys.path.insert(0, str(SCRIPT_DIR))
                from plot import plot_id4scs_combined
                plot_id4scs_combined(side_rows, side_fits,
                                     results_root / "ID4SCS" / "id4scs_combined_plot.png")
        _wilcoxon_ct(id4_rows["narrow_to_wide"] + id4_rows["wide_to_narrow"], "ID4SCS")

    # --- Constrained-to-Unconstrained ------------------------------------
    c2u = [r for r in rows if r["task"] == "C2U"]
    if c2u:
        print(f"\n[Constrained-to-Unconstrained]  {len({r['tid'] for r in c2u})} "
              f"conditions × {len({r['pid'] for r in c2u})} participants")
        by_pos = defaultdict(list)
        for r in c2u:
            by_pos[r["condition"].get("targetPosition", "?")].append(r)
        groups = [(p, _group_stats(v)) for p, v in sorted(by_pos.items())]
        _print_group_table("By target position", groups)
        all_groups.append(("C2U", "target_position", groups))

        by_w = defaultdict(list)
        for r in c2u:
            w = r["condition"].get("segment1Width")
            if w is not None:
                by_w[float(w)].append(r)
        groups = [(f"W1={w*1000:.0f}mm", _group_stats(v))
                  for w, v in sorted(by_w.items())]
        _print_group_table("By tunnel-segment width", groups)
        all_groups.append(("C2U", "segment1_width", groups))
        _wilcoxon_ct(c2u, "C2U")

    # --- Per participant --------------------------------------------------
    if args.by_participant:
        for task, _sub in TASK_DIRS:
            trows = [r for r in rows if r["task"] == task]
            if not trows:
                continue
            by_pid = defaultdict(list)
            for r in trows:
                by_pid[r["pid"]].append(r)
            groups = [(pid, _group_stats(v)) for pid, v in sorted(by_pid.items())]
            _print_group_table(f"{task} — by participant", groups)
            all_groups.append((task, "participant", groups))

    out_path = results_root / "aggregate_stats.csv"
    save_aggregate_csv(all_groups, out_path)
    print(f"\nSaved: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

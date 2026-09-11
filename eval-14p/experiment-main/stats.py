"""
Compute aggregate statistics from evaluation result CSVs.

Reads:
  results/progress_metrics.csv   — per-participant × per-trial progress-aligned metrics
  results/trial_summary.csv      — per-trial summary (averaged across participants)
  results/main_results.csv       — per-round raw data (human, model, baseline)

Prints summary tables and saves aggregate statistics to results/aggregate_stats.csv.

Usage:
    python -m eval.experiment-main.stats
    python -m eval.experiment-main.stats --by-type       # group by tunnel type
    python -m eval.experiment-main.stats --by-width      # group by width
    python -m eval.experiment-main.stats --by-participant # per-participant summary
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"

METRIC_KEYS = ["lat_rmse", "speed_rmse", "speed_corr", "time_diff"]
BASELINE_METRIC_KEYS = ["bl_lat_rmse", "bl_speed_rmse", "bl_speed_corr", "bl_time_diff"]

# Trial type groupings
TUNNEL_TYPES = {
    "sinusoidal": {"label": "Sinusoidal", "tids": {1, 2, 3, 4, 5}},
    "corner":     {"label": "Corner",     "tids": {6, 7, 8, 9, 10}},
    "straight":   {"label": "Straight",   "tids": {11, 12, 13, 14, 15}},
    "gentle_sin": {"label": "Gentle Sin", "tids": {16, 17, 18, 19, 20}},
    "sharp_sin":  {"label": "Sharp Sin",  "tids": {21, 22, 23, 24, 25}},
}

TRAIN_TIDS = {1, 3, 5, 6, 8, 10, 11, 13, 15, 16, 18, 20, 21, 23, 25}
TEST_TIDS  = {2, 4, 7, 9, 12, 14, 17, 19, 22, 24}


def load_progress_metrics():
    """Load progress_metrics.csv -> list of dicts."""
    csv_path = RESULTS_DIR / "progress_metrics.csv"
    if not csv_path.exists():
        sys.exit(f"Not found: {csv_path}\nRun evaluation first.")
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["trial_id"] = int(row["trial_id"])
            row["width"] = float(row["width"])
            for k in METRIC_KEYS + BASELINE_METRIC_KEYS:
                if k in row and row[k] != "":
                    row[k] = float(row[k])
                elif k in row:
                    row[k] = None
            rows.append(row)
    return rows


def load_trial_summary():
    """Load trial_summary.csv -> list of dicts."""
    csv_path = RESULTS_DIR / "trial_summary.csv"
    if not csv_path.exists():
        sys.exit(f"Not found: {csv_path}\nRun evaluation first.")
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["trial_id"] = int(row["trial_id"])
            row["width"] = float(row["width"])
            for k in ["human_mt", "human_speed", "model_mt", "model_speed",
                       "lat_rmse", "speed_rmse", "speed_corr", "time_diff",
                       "n_human", "n_model",
                       "baseline_mt", "baseline_speed",
                       "bl_lat_rmse", "bl_speed_rmse", "bl_speed_corr", "bl_time_diff"]:
                if k in row and row[k] != "":
                    row[k] = float(row[k])
                elif k in row:
                    row[k] = None
            rows.append(row)
    return rows


def _agg(values):
    """Return (mean, std, n) for a list of numeric values, skipping None."""
    valid = [v for v in values if v is not None]
    if not valid:
        return None, None, 0
    return float(np.mean(valid)), float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0, len(valid)


def _print_metric_table(title, groups, keys=METRIC_KEYS, show_baseline=False):
    """Print a formatted table of metric statistics."""
    print(f"\n  {title}")
    header_parts = ["Group"]
    for k in keys:
        header_parts.append(f"{k:>12s}")
    header = "  ".join(header_parts)
    print(f"  {'─' * len(header)}")
    print(f"  {header}")
    print(f"  {'─' * len(header)}")

    for group_label, metrics_dict in groups:
        parts = [f"{group_label:<28s}"]
        for k in keys:
            mean, std, n = metrics_dict.get(k, (None, None, 0))
            if mean is not None:
                parts.append(f"{mean:>8.4f}±{std:.4f}" if std else f"{mean:>12.4f}")
            else:
                parts.append(f"{'—':>12s}")
        print(f"  {'  '.join(parts)}  (n={metrics_dict.get('n', (0,0,0))[2]})")

    print(f"  {'─' * len(header)}")


def aggregate_by_split(rows):
    """Aggregate progress metrics by train/test split."""
    buckets = {"train": defaultdict(list), "test": defaultdict(list), "all": defaultdict(list)}
    for row in rows:
        tid = row["trial_id"]
        split = "train" if tid in TRAIN_TIDS else "test"
        for k in METRIC_KEYS:
            if row.get(k) is not None:
                buckets[split][k].append(row[k])
                buckets["all"][k].append(row[k])

    results = []
    for label, bucket in [("Train (15 tasks)", buckets["train"]),
                           ("Test (10 tasks)", buckets["test"]),
                           ("Overall (25 tasks)", buckets["all"])]:
        stats = {k: _agg(bucket[k]) for k in METRIC_KEYS}
        stats["n"] = _agg(bucket[METRIC_KEYS[0]])  # n from any key
        results.append((label, stats))
    return results


def aggregate_by_type(rows):
    """Aggregate progress metrics by tunnel type."""
    buckets = {ttype: defaultdict(list) for ttype in TUNNEL_TYPES}
    for row in rows:
        tid = row["trial_id"]
        for ttype, info in TUNNEL_TYPES.items():
            if tid in info["tids"]:
                for k in METRIC_KEYS:
                    if row.get(k) is not None:
                        buckets[ttype][k].append(row[k])
                break

    results = []
    for ttype in ["sinusoidal", "gentle_sin", "sharp_sin", "corner", "straight"]:
        info = TUNNEL_TYPES[ttype]
        bucket = buckets[ttype]
        stats = {k: _agg(bucket[k]) for k in METRIC_KEYS}
        stats["n"] = _agg(bucket[METRIC_KEYS[0]])
        results.append((info["label"], stats))
    return results


def aggregate_by_type_and_split(rows):
    """Aggregate progress metrics by tunnel type, separated by train/test."""
    buckets = {}
    for ttype in TUNNEL_TYPES:
        buckets[ttype] = {
            "train": defaultdict(list),
            "test": defaultdict(list),
            "all": defaultdict(list),
        }
    for row in rows:
        tid = row["trial_id"]
        split = "train" if tid in TRAIN_TIDS else "test"
        for ttype, info in TUNNEL_TYPES.items():
            if tid in info["tids"]:
                for k in METRIC_KEYS:
                    if row.get(k) is not None:
                        buckets[ttype][split][k].append(row[k])
                        buckets[ttype]["all"][k].append(row[k])
                break

    results = []
    for ttype in ["sinusoidal", "gentle_sin", "sharp_sin", "corner", "straight"]:
        info = TUNNEL_TYPES[ttype]
        for split_label, split_key in [("Train", "train"), ("Test", "test"), ("All", "all")]:
            bucket = buckets[ttype][split_key]
            if not bucket[METRIC_KEYS[0]]:
                continue
            stats = {k: _agg(bucket[k]) for k in METRIC_KEYS}
            stats["n"] = _agg(bucket[METRIC_KEYS[0]])
            results.append((f"{info['label']} ({split_label})", stats))
    return results


def aggregate_by_width(rows):
    """Aggregate progress metrics by tunnel width."""
    buckets = defaultdict(lambda: defaultdict(list))
    for row in rows:
        w = row["width"]
        for k in METRIC_KEYS:
            if row.get(k) is not None:
                buckets[w][k].append(row[k])

    results = []
    for w in sorted(buckets.keys()):
        bucket = buckets[w]
        stats = {k: _agg(bucket[k]) for k in METRIC_KEYS}
        stats["n"] = _agg(bucket[METRIC_KEYS[0]])
        results.append((f"W={w*1000:.0f}mm", stats))
    return results


def aggregate_by_participant(rows):
    """Aggregate progress metrics per participant."""
    buckets = defaultdict(lambda: defaultdict(list))
    for row in rows:
        pid = row["participant"]
        for k in METRIC_KEYS:
            if row.get(k) is not None:
                buckets[pid][k].append(row[k])

    results = []
    for pid in sorted(buckets.keys()):
        bucket = buckets[pid]
        stats = {k: _agg(bucket[k]) for k in METRIC_KEYS}
        stats["n"] = _agg(bucket[METRIC_KEYS[0]])
        results.append((pid, stats))
    return results


def print_trial_table(trial_rows):
    """Print per-trial summary table with Human/Model MT, Speed, and counts."""
    has_bl = any(r.get("baseline_mt") is not None for r in trial_rows)

    print("\n  Summary by Trial (averaged across participants):")
    if has_bl:
        print("  ┌────┬─────────────────────────────┬────────────────────────────────────────────────────┬────────────────────────────────────────────────────┐")
        print("  │    │                             │        Human              Model                    │        Human              Baseline                │")
        print("  │    │ Trial                       │  MT(s)   Speed(m/s)    MT(s)   Speed(m/s)  n(H/M) │  MT(s)   Speed(m/s)    MT(s)   Speed(m/s)  n(H/B) │")
        print("  ├────┼─────────────────────────────┼────────────────────────────────────────────────────┼────────────────────────────────────────────────────┤")
    else:
        print("  ┌────┬─────────────────────────────┬────────────────────────────────────────────────────┐")
        print("  │    │                             │        Human              Model                    │")
        print("  │    │ Trial                       │  MT(s)   Speed(m/s)    MT(s)   Speed(m/s)  n(H/M) │")
        print("  ├────┼─────────────────────────────┼────────────────────────────────────────────────────┤")

    for row in sorted(trial_rows, key=lambda r: r["trial_id"]):
        tid = row["trial_id"]
        label = row["label"]
        split = "TR" if tid in TRAIN_TIDS else "TE"

        h_mt = row.get("human_mt")
        h_spd = row.get("human_speed")
        m_mt = row.get("model_mt")
        m_spd = row.get("model_speed")
        n_h = int(row["n_human"]) if row.get("n_human") is not None else 0
        n_m = int(row["n_model"]) if row.get("n_model") is not None else 0

        m_str = (f"{h_mt:6.2f}    {h_spd:.3f}     {m_mt:6.2f}    {m_spd:.3f}   {n_h:3d}/{n_m:<3d}"
                 if h_mt is not None else "     -        -          -        -       -   ")

        if has_bl:
            bl_mt = row.get("baseline_mt")
            bl_spd = row.get("baseline_speed")
            b_str = (f"{h_mt:6.2f}    {h_spd:.3f}     {bl_mt:6.2f}    {bl_spd:.3f}   {n_h:3d}/{n_m:<3d}"
                     if bl_mt is not None else "     -        -          -        -       -   ")
            print(f"  │ {split} │ {label:27s} │ {m_str} │ {b_str} │")
        else:
            print(f"  │ {split} │ {label:27s} │ {m_str} │")

    # Summary rows
    if has_bl:
        print("  ├────┼─────────────────────────────┼────────────────────────────────────────────────────┼────────────────────────────────────────────────────┤")
    else:
        print("  ├────┼─────────────────────────────┼────────────────────────────────────────────────────┤")

    for split_label, split_name, tid_set in [("TR", "Train avg", TRAIN_TIDS),
                                              ("TE", "Test avg", TEST_TIDS),
                                              ("  ", "Overall", None)]:
        subset = [r for r in trial_rows if (tid_set is None or r["trial_id"] in tid_set)]
        h_mts = [r["human_mt"] for r in subset if r.get("human_mt") is not None]
        h_spds = [r["human_speed"] for r in subset if r.get("human_speed") is not None]
        m_mts = [r["model_mt"] for r in subset if r.get("model_mt") is not None]
        m_spds = [r["model_speed"] for r in subset if r.get("model_speed") is not None]
        total_h = sum(int(r["n_human"]) for r in subset if r.get("n_human") is not None)
        total_m = sum(int(r["n_model"]) for r in subset if r.get("n_model") is not None)

        if h_mts:
            m_str = (f"{np.mean(h_mts):6.2f}    {np.mean(h_spds):.3f}     "
                     f"{np.mean(m_mts):6.2f}    {np.mean(m_spds):.3f}   "
                     f"{total_h:3d}/{total_m:<3d}")
            if has_bl:
                bl_mts = [r["baseline_mt"] for r in subset if r.get("baseline_mt") is not None]
                bl_spds = [r["baseline_speed"] for r in subset if r.get("baseline_speed") is not None]
                if bl_mts:
                    b_str = (f"{np.mean(h_mts):6.2f}    {np.mean(h_spds):.3f}     "
                             f"{np.mean(bl_mts):6.2f}    {np.mean(bl_spds):.3f}   "
                             f"{total_h:3d}/{total_m:<3d}")
                else:
                    b_str = "     -        -          -        -       -   "
                print(f"  │ {split_label} │ {split_name:27s} │ {m_str} │ {b_str} │")
            else:
                print(f"  │ {split_label} │ {split_name:27s} │ {m_str} │")

    if has_bl:
        print("  └────┴─────────────────────────────┴────────────────────────────────────────────────────┴────────────────────────────────────────────────────┘")
    else:
        print("  └────┴─────────────────────────────┴────────────────────────────────────────────────────┘")


def completion_time_statistical_test():
    """Paired statistical test: is our model's MT error significantly smaller
    than the baseline's MT error?

    For each (participant, trial) pair, compute:
      model_error  = |model_mt - human_mt|
      baseline_error = |baseline_mt - human_mt|
    Then run a paired Wilcoxon signed-rank test on model_error vs baseline_error.
    """
    from scipy.stats import wilcoxon
    import json

    results_dir = RESULTS_DIR
    model_errors = []
    baseline_errors = []
    pairs_detail = []

    # Iterate over participant directories
    for pdir in sorted(results_dir.iterdir()):
        if not pdir.is_dir() or not pdir.name.startswith("participant_"):
            continue
        pid = pdir.name.replace("participant_", "")

        for tdir in sorted(pdir.iterdir()):
            if not tdir.is_dir() or not tdir.name.startswith("trial_"):
                continue
            tid = int(tdir.name.replace("trial_", ""))

            summary_path = tdir / "results_summary.json"
            if not summary_path.exists():
                continue

            with open(summary_path) as f:
                data = json.load(f)

            human_mt = data.get("human", {}).get("avg_completion_time")
            model_mt = data.get("model", {}).get("avg_completion_time")
            baseline_data = data.get("baseline", {})
            baseline_mt = baseline_data.get("avg_completion_time")

            if human_mt is None or model_mt is None or baseline_mt is None:
                continue

            m_err = abs(model_mt - human_mt)
            b_err = abs(baseline_mt - human_mt)
            model_errors.append(m_err)
            baseline_errors.append(b_err)
            pairs_detail.append((pid, tid, human_mt, model_mt, baseline_mt, m_err, b_err))

    if len(model_errors) < 5:
        print("\n  Insufficient data for statistical test.")
        return

    # Aggregate to participant level to avoid pseudoreplication
    pid_model_errors = defaultdict(list)
    pid_baseline_errors = defaultdict(list)
    for pid, tid, h_mt, m_mt, b_mt, m_err, b_err in pairs_detail:
        pid_model_errors[pid].append(m_err)
        pid_baseline_errors[pid].append(b_err)

    pids_sorted = sorted(pid_model_errors.keys())
    participant_model_err = np.array([np.mean(pid_model_errors[p]) for p in pids_sorted])
    participant_baseline_err = np.array([np.mean(pid_baseline_errors[p]) for p in pids_sorted])
    n_participants = len(pids_sorted)

    if n_participants < 5:
        print("\n  Insufficient participants for statistical test.")
        return

    # Paired Wilcoxon signed-rank test at participant level
    stat, p_value = wilcoxon(participant_model_err, participant_baseline_err, alternative="less")

    print(f"\n  Completion Time Statistical Test (paired Wilcoxon signed-rank)")
    print(f"  {'─' * 65}")
    print(f"  Unit of analysis: participant-level mean |ΔMT|")
    print(f"  H0: |model_MT - human_MT| >= |baseline_MT - human_MT|")
    print(f"  H1: |model_MT - human_MT| <  |baseline_MT - human_MT|")
    print(f"  {'─' * 65}")
    print(f"  N participants:       {n_participants}")
    print(f"  N trial pairs total:  {len(model_errors)}")
    print(f"  Model |ΔMT| mean:     {np.mean(participant_model_err):.3f}s  (median: {np.median(participant_model_err):.3f}s)")
    print(f"  Baseline |ΔMT| mean:  {np.mean(participant_baseline_err):.3f}s  (median: {np.median(participant_baseline_err):.3f}s)")
    print(f"  Wilcoxon statistic:   {stat:.1f}")
    print(f"  p-value (one-sided):  {p_value:.2e}")
    if p_value < 0.001:
        print(f"  Result: *** p < 0.001 — model MT error is significantly smaller")
    elif p_value < 0.01:
        print(f"  Result: **  p < 0.01")
    elif p_value < 0.05:
        print(f"  Result: *   p < 0.05")
    else:
        print(f"  Result: not significant at α=0.05")
    print(f"  {'─' * 65}")


def metrics_statistical_tests():
    """Paired Wilcoxon tests for all four metrics (model vs baseline),
    aggregated at participant level to avoid pseudoreplication."""
    from scipy.stats import wilcoxon
    import json

    results_dir = RESULTS_DIR
    # Collect per (participant, trial) metrics
    pairs = []  # (pid, tid, model_metrics, baseline_metrics)

    for pdir in sorted(results_dir.iterdir()):
        if not pdir.is_dir() or not pdir.name.startswith("participant_"):
            continue
        pid = pdir.name.replace("participant_", "")

        for tdir in sorted(pdir.iterdir()):
            if not tdir.is_dir() or not tdir.name.startswith("trial_"):
                continue
            tid = int(tdir.name.replace("trial_", ""))

            summary_path = tdir / "results_summary.json"
            if not summary_path.exists():
                continue

            with open(summary_path) as f:
                data = json.load(f)

            m_metrics = data.get("metrics", {})
            b_metrics = data.get("baseline_metrics", {})

            if not m_metrics or not b_metrics:
                continue

            pairs.append((pid, tid, m_metrics, b_metrics))

    if not pairs:
        print("\n  No paired metric data found for statistical tests.")
        return

    # Aggregate to participant level
    pid_data = defaultdict(lambda: defaultdict(lambda: {"model": [], "baseline": []}))
    for pid, tid, m_met, b_met in pairs:
        for key in ["lateral_rmse_mean", "speed_rmse_mean", "speed_corr_mean", "time_diff_mean"]:
            mv = m_met.get(key)
            bv = b_met.get(key)
            if mv is not None and bv is not None:
                pid_data[pid][key]["model"].append(mv)
                pid_data[pid][key]["baseline"].append(bv)

    metric_labels = {
        "lateral_rmse_mean": ("Lateral RMSE", "less", "lower is better"),
        "speed_rmse_mean": ("Speed RMSE", "less", "lower is better"),
        "speed_corr_mean": ("Speed Correlation", "greater", "higher is better"),
        "time_diff_mean": ("Time Difference", "less", "lower is better"),
    }

    print(f"\n  Metric-Level Statistical Tests (paired Wilcoxon, participant-level)")
    print(f"  {'─' * 80}")
    print(f"  {'Metric':<22s}  {'Model mean':>12s}  {'Base mean':>12s}  {'W':>8s}  {'p-value':>12s}  {'Sig':>5s}")
    print(f"  {'─' * 80}")

    for key, (label, alternative, direction) in metric_labels.items():
        pids_sorted = sorted(pid_data.keys())
        model_avgs = []
        baseline_avgs = []
        for p in pids_sorted:
            m_vals = pid_data[p][key]["model"]
            b_vals = pid_data[p][key]["baseline"]
            if m_vals and b_vals:
                model_avgs.append(np.mean(m_vals))
                baseline_avgs.append(np.mean(b_vals))

        if len(model_avgs) < 5:
            print(f"  {label:<22s}  insufficient data")
            continue

        model_avgs = np.array(model_avgs)
        baseline_avgs = np.array(baseline_avgs)

        stat, p_value = wilcoxon(model_avgs, baseline_avgs, alternative=alternative)

        sig = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else "n.s."))
        print(f"  {label:<22s}  {np.mean(model_avgs):12.4f}  {np.mean(baseline_avgs):12.4f}  {stat:8.1f}  {p_value:12.2e}  {sig:>5s}")

    print(f"  {'─' * 80}")
    print(f"  N participants: {len([p for p in pid_data if pid_data[p]])}")


def save_aggregate_csv(all_groups, output_path):
    """Save all aggregate stats to a single CSV."""
    fields = ["grouping", "group", "metric", "mean", "std", "n"]
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for grouping_name, groups in all_groups:
            for group_label, stats in groups:
                for k in METRIC_KEYS:
                    mean, std, n = stats.get(k, (None, None, 0))
                    writer.writerow({
                        "grouping": grouping_name,
                        "group": group_label,
                        "metric": k,
                        "mean": round(mean, 6) if mean is not None else "",
                        "std": round(std, 6) if std is not None else "",
                        "n": n,
                    })


def main():
    parser = argparse.ArgumentParser(description="Compute aggregate statistics from evaluation CSVs")
    parser.add_argument("--by-type", action="store_true", help="Group by tunnel type")
    parser.add_argument("--by-width", action="store_true", help="Group by tunnel width")
    parser.add_argument("--by-participant", action="store_true", help="Per-participant summary")
    args = parser.parse_args()

    # If no flags, show all
    show_all = not (args.by_type or args.by_width or args.by_participant)

    print("=" * 70)
    print("Aggregate Statistics from Evaluation Results")
    print("=" * 70)

    rows = load_progress_metrics()
    trial_rows = load_trial_summary()
    n_participants = len(set(r["participant"] for r in rows))
    n_trials = len(set(r["trial_id"] for r in rows))
    print(f"  Loaded {len(rows)} entries: {n_participants} participants × {n_trials} trials")

    has_baseline = any(r.get("bl_lat_rmse") is not None for r in rows)

    # Per-trial table (always shown)
    print_trial_table(trial_rows)

    all_groups = []

    # By split (always shown)
    split_groups = aggregate_by_split(rows)
    _print_metric_table("By Train/Test Split", split_groups)
    all_groups.append(("split", split_groups))

    if show_all or args.by_type:
        type_groups = aggregate_by_type(rows)
        _print_metric_table("By Tunnel Type", type_groups)
        all_groups.append(("type", type_groups))

        type_split_groups = aggregate_by_type_and_split(rows)
        _print_metric_table("By Tunnel Type × Train/Test Split", type_split_groups)
        all_groups.append(("type_split", type_split_groups))

    if show_all or args.by_width:
        width_groups = aggregate_by_width(rows)
        _print_metric_table("By Width", width_groups)
        all_groups.append(("width", width_groups))

    if show_all or args.by_participant:
        pid_groups = aggregate_by_participant(rows)
        _print_metric_table("By Participant", pid_groups)
        all_groups.append(("participant", pid_groups))

    # Baseline comparison (if available)
    if has_baseline:
        bl_buckets = {"train": defaultdict(list), "test": defaultdict(list), "all": defaultdict(list)}
        for row in rows:
            tid = row["trial_id"]
            split = "train" if tid in TRAIN_TIDS else "test"
            for k, bk in zip(METRIC_KEYS, BASELINE_METRIC_KEYS):
                if row.get(bk) is not None:
                    bl_buckets[split][k].append(row[bk])
                    bl_buckets["all"][k].append(row[bk])

        bl_stats_all = {k: _agg(bl_buckets["all"][k]) for k in METRIC_KEYS}
        bl_stats_all["n"] = _agg(bl_buckets["all"][METRIC_KEYS[0]])
        bl_stats_train = {k: _agg(bl_buckets["train"][k]) for k in METRIC_KEYS}
        bl_stats_train["n"] = _agg(bl_buckets["train"][METRIC_KEYS[0]])
        bl_stats_test = {k: _agg(bl_buckets["test"][k]) for k in METRIC_KEYS}
        bl_stats_test["n"] = _agg(bl_buckets["test"][METRIC_KEYS[0]])

        # Model stats by split
        model_all = [g for label, g in split_groups if "Overall" in label][0]
        model_train = [g for label, g in split_groups if "Train" in label][0]
        model_test = [g for label, g in split_groups if "Test" in label][0]

        print("\n  Model vs Baseline (by split):")
        print(f"  {'─' * 76}")
        print(f"  {'':28s}  {'lat_rmse':>12s}  {'speed_rmse':>12s}  {'speed_corr':>12s}  {'time_diff':>12s}")
        print(f"  {'─' * 76}")
        for lbl, st in [("Model (Train)", model_train),
                         ("Baseline (Train)", bl_stats_train),
                         ("Model (Test)", model_test),
                         ("Baseline (Test)", bl_stats_test),
                         ("Model (Overall)", model_all),
                         ("Baseline (Overall)", bl_stats_all)]:
            parts = [f"{lbl:<28s}"]
            for k in METRIC_KEYS:
                m, s, n = st.get(k, (None, None, 0))
                parts.append(f"{m:>8.4f}±{s:.4f}" if m is not None else f"{'—':>12s}")
            print(f"  {'  '.join(parts)}")
        print(f"  {'─' * 76}")

    # Statistical tests
    if has_baseline:
        completion_time_statistical_test()
        metrics_statistical_tests()

    # Save to CSV
    out_path = RESULTS_DIR / "aggregate_stats.csv"
    save_aggregate_csv(all_groups, out_path)
    print(f"\n  Saved: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

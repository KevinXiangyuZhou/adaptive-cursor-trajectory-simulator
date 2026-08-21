"""Side-by-side comparison of eval-main results directories (model variants).

Usage:
    python compare_results.py resultsdirA resultsdirB ... [--labels a b ...]
                              [--out comparison.csv]

For each results dir (as produced by run_eval.py, optionally via
--results-dir) it summarises:
  Steering : model/human CT ratio (median, IQR), lateral RMSE, speed corr,
             steering-law slope/intercept, timeouts
  Fitts    : aligned MT_kin ratio (median, IQR), aligned Fitts fit a+b*ID
             (human fit printed once), throughput, timeouts
  ID4SCS   : model/human MT ratio (median across both directions)
Rows are per variant; the human reference row is derived from the first
directory that has the corresponding human data (identical across variants
run on the same cohort).
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _read_csv(path):
    if not Path(path).exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def summarize(results_dir):
    d = Path(results_dir)
    out = {"dir": str(d)}

    rows = _read_csv(d / "Steering" / "steering_condition_summary.csv")
    if rows:
        r = [float(x["model_time_mean_s"]) / float(x["human_time_mean_s"])
             for x in rows if float(x.get("human_time_mean_s", 0) or 0) > 0]
        out["steer_ct_ratio_med"] = round(float(np.median(r)), 3)
        out["steer_ct_ratio_iqr"] = (round(float(np.percentile(r, 25)), 2),
                                      round(float(np.percentile(r, 75)), 2))
        out["steer_lat_rmse_m"] = round(float(np.mean(
            [float(x["lateral_rmse"]) for x in rows if x.get("lateral_rmse")])), 4)
        out["steer_speed_corr"] = round(float(np.mean(
            [float(x["speed_corr"]) for x in rows if x.get("speed_corr")])), 3)
        out["steer_timeouts"] = sum(int(x.get("n_timed_out", 0) or 0) for x in rows)
    srows = _read_csv(d / "Steering" / "steering_results.csv")
    if srows:
        for src, key in (("Simulator", "steer_law_model"), ("Human", "steer_law_human")):
            sub = [x for x in srows if x["source"] == src and not
                   (x.get("timed_out", "").lower() in ("true", "1"))]
            by_tid = defaultdict(list)
            ids = {}
            for x in sub:
                by_tid[x["tid"]].append(float(x["MT_s"]))
                ids[x["tid"]] = float(x["ID"])
            if len(by_tid) >= 3:
                xs = np.array([ids[t] for t in by_tid])
                ys = np.array([np.mean(v) for v in by_tid.values()])
                b, a = np.polyfit(xs, ys, 1)
                out[key] = f"{b:.3f}*ID+{a:.2f}"

    frows = _read_csv(d / "Fitts" / "fitts_results.csv")
    if frows:
        by = defaultdict(list)
        for x in frows:
            if x.get("MT_kin_s"):
                by[(x["source"], x["tid"])].append(float(x["MT_kin_s"]))
        tids = sorted({t for (_, t) in by})
        r = [np.mean(by[("Simulator", t)]) / np.mean(by[("Human", t)])
             for t in tids if by.get(("Human", t)) and by.get(("Simulator", t))]
        if r:
            out["fitts_mtkin_ratio_med"] = round(float(np.median(r)), 3)
            out["fitts_mtkin_ratio_iqr"] = (round(float(np.percentile(r, 25)), 2),
                                             round(float(np.percentile(r, 75)), 2))
        out["fitts_timeouts"] = sum(
            1 for x in frows if x["source"] == "Simulator"
            and x.get("timed_out", "").lower() in ("true", "1"))
    reg_path = d / "Fitts" / "fitts_regression.json"
    if reg_path.exists():
        reg = json.load(open(reg_path))
        for side in ("model", "human"):
            m = reg.get("aligned", {}).get(side) or {}
            if m:
                out[f"fitts_kin_{side}"] = (f"{m['b_slope_s_per_bit']:.3f}*ID"
                                            f"{m['a_intercept']:+.2f}")
                out[f"fitts_tp_{side}"] = m.get("throughput_mean_bps")

    id_r = []
    for sub in ("wide_to_narrow", "narrow_to_wide"):
        prefix = "id4scs_" + sub
        rows = _read_csv(d / "ID4SCS" / sub / f"{prefix}_condition_summary.csv")
        for x in rows:
            h, m = x.get("human_time_mean_s"), x.get("model_time_mean_s")
            if h and m and float(h) > 0:
                id_r.append(float(m) / float(h))
    if id_r:
        out["id4scs_ct_ratio_med"] = round(float(np.median(id_r)), 3)

    return out


COLUMNS = [
    ("steer_ct_ratio_med", "steerCT"),
    ("steer_lat_rmse_m", "latRMSE"),
    ("steer_speed_corr", "spdCorr"),
    ("steer_law_model", "steer law (model)"),
    ("steer_timeouts", "sTO"),
    ("fitts_mtkin_ratio_med", "fittsMTkin"),
    ("fitts_kin_model", "fitts law (model)"),
    ("fitts_tp_model", "TP"),
    ("fitts_timeouts", "fTO"),
    ("id4scs_ct_ratio_med", "id4CT"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dirs", nargs="+")
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--out", default=None, help="optional CSV output path")
    args = ap.parse_args()

    labels = args.labels or [Path(p).name for p in args.results_dirs]
    assert len(labels) == len(args.results_dirs), "--labels count mismatch"

    summaries = [summarize(p) for p in args.results_dirs]

    human = {}
    for s in summaries:
        for k in ("steer_law_human", "fitts_kin_human", "fitts_tp_human"):
            if k in s and k not in human:
                human[k] = s[k]
    if human:
        print("Human reference: "
              + "  ".join(f"{k.replace('_human','')}={v}" for k, v in human.items()))
        print()

    widths = [max(len(lbl) for lbl in labels + ["variant"])] + \
             [max(len(h), 10) for _, h in COLUMNS]
    header = ["variant"] + [h for _, h in COLUMNS]
    print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
    for lbl, s in zip(labels, summaries):
        cells = [lbl]
        for key, _ in COLUMNS:
            v = s.get(key, "-")
            cells.append(str(v))
        print("  ".join(c.ljust(w) for c, w in zip(cells, widths)))

    if args.out:
        keys = ["label"] + [k for k, _ in COLUMNS] + list(human.keys())
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for lbl, s in zip(labels, summaries):
                row = {"label": lbl}
                row.update({k: s.get(k) for k, _ in COLUMNS})
                row.update(human)
                w.writerow(row)
        print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()

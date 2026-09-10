"""Steering agreement table (paper): metrics by tunnel type and train/test split.

Reads the pooled-8 main eval's steering_condition_summary.csv (one row per
participant x condition) and prints LaTeX rows of mean +/- SD for lateral RMSE
(mm), speed RMSE (m/s), speed correlation, and time ratio (model/human
completion time). Train conditions are the fitted widths {10, 50} mm
(fit_speed_model.TRAIN_WIDTHS); test conditions are {12.5, 25} mm.
NOTE: runs fitted before 2026-09-10 used train widths {10, 50} — pass
--train-widths 10,50 when tabulating one of those.

Usage: python paper/table_steering_metrics.py [--csv PATH]
       (default: the pooled-8 main eval; pass a run's
        eval/Steering/steering_condition_summary.csv for per-participant fits)
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
CSV = (REPO / "results-cluster-10p" / "eval-main-pooled8-local" / "Steering"
       / "steering_condition_summary.csv")

TRAIN_WIDTHS_MM = {10.0, 16.5, 50.0}

# tid -> tunnel type (from the trial conditions in the raw participant JSON)
TYPE_BY_TID = {}
for lo, name in ((1, "Sinusoidal"), (6, "Corner"), (11, "Straight"),
                 (26, "Gentle sinusoidal"), (31, "Sharp sinusoidal")):
    for t in range(lo, lo + 5):
        TYPE_BY_TID[t] = name

ROW_ORDER = ["Straight", "Corner", "Sinusoidal", "Gentle sinusoidal",
             "Sharp sinusoidal"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(CSV),
                    help="steering_condition_summary.csv to tabulate")
    ap.add_argument("--train-widths", default=None,
                    help="comma-separated train widths in mm (default: the "
                         "current fit split; use 10,50 for pre-2026-09-10 runs)")
    a = ap.parse_args()
    train_widths = (TRAIN_WIDTHS_MM if a.train_widths is None
                    else {float(w) for w in a.train_widths.split(",")})
    groups = defaultdict(lambda: defaultdict(list))  # (type, split) -> metric -> vals
    with open(a.csv) as f:
        for r in csv.DictReader(f):
            ttype = TYPE_BY_TID[int(r["tid"])]
            split = "Train" if float(r["width_mm"]) in train_widths else "Test"
            vals = {
                "lat_mm": float(r["lateral_rmse"]) * 1000.0,
                "spd": float(r["speed_rmse"]),
                "corr": float(r["speed_corr"]),
                "ratio": float(r["model_time_mean_s"]) / float(r["human_time_mean_s"]),
            }
            for key in (ttype, "All"):
                for m, v in vals.items():
                    groups[(key, split)][m].append(v)

    def cell(g, m, prec):
        v = np.array(g[m])
        return f"${np.mean(v):.{prec}f} \\pm {np.std(v):.{prec}f}$"

    for ttype in ROW_ORDER + ["All"]:
        if ttype == "All":
            print("        \\midrule")
        for i, split in enumerate(("Train", "Test")):
            g = groups[(ttype, split)]
            label = ttype if i == 0 else ""
            n = len(g["lat_mm"])
            print(f"        {label:<18} & {split:<5} & {cell(g, 'lat_mm', 1)} & "
                  f"{cell(g, 'spd', 3)} & {cell(g, 'corr', 2)} & "
                  f"{cell(g, 'ratio', 2)} \\\\  % n={n}")


if __name__ == "__main__":
    main()

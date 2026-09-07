"""Steering agreement table (paper): metrics by tunnel type and train/test split.

Reads the pooled-8 main eval's steering_condition_summary.csv (one row per
participant x condition) and prints LaTeX rows of mean +/- SD for lateral RMSE
(mm), speed RMSE (m/s), speed correlation, and time ratio (model/human
completion time). Train conditions are the fitted widths {10, 50} mm
(fit_speed_model.TRAIN_WIDTHS); test conditions are {12.5, 16.5, 25} mm.

Usage: python paper/table_steering_metrics.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
CSV = (REPO / "results-cluster-10p" / "eval-main-pooled8-local" / "Steering"
       / "steering_condition_summary.csv")

TRAIN_WIDTHS_MM = {10.0, 50.0}

# tid -> tunnel type (from the trial conditions in the raw participant JSON)
TYPE_BY_TID = {}
for lo, name in ((1, "Sinusoidal"), (6, "Corner"), (11, "Straight"),
                 (26, "Gentle sinusoidal"), (31, "Sharp sinusoidal")):
    for t in range(lo, lo + 5):
        TYPE_BY_TID[t] = name

ROW_ORDER = ["Straight", "Corner", "Sinusoidal", "Gentle sinusoidal",
             "Sharp sinusoidal"]


def main():
    groups = defaultdict(lambda: defaultdict(list))  # (type, split) -> metric -> vals
    with open(CSV) as f:
        for r in csv.DictReader(f):
            ttype = TYPE_BY_TID[int(r["tid"])]
            split = "Train" if float(r["width_mm"]) in TRAIN_WIDTHS_MM else "Test"
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

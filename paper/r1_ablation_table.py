"""Reviewer-1 ablation table: full model vs baseline vs the two crossed
ablations, with the paper's own aggregation.

    lookahead \\ catch-up | pace law            | fixed target speed
    ----------------------+---------------------+---------------------
    adaptive gaze budget  | full model (Ours)   | V2 fixed_speed
    fixed horizon         | V1 fixed_horizon    | baseline (CHI-26-EA)

For every run: steering law MT = a + b ID (ID = L/W) and the
curvature-aware ID_k law (lambda fit on the human condition means, as in
plot_laws.py), Fitts law on the aligned movement time, the steering
agreement metrics of table_steering_metrics.py (all tunnel types pooled,
Train / Test widths), and the model/human completion-time ratio per width
(where a mechanism that cannot scale speed with width shows up first).
Aggregation is plot_laws.py's: one point per condition, flat pooled mean
over participants and rounds; timed-out runs excluded.

Usage:
  python paper/r1_ablation_table.py \\
      --run Ours=results-cluster-10p/runs/mpcc-full-perpid-s42-20260909-1257-22c349c \\
      --run Baseline=results-cluster-10p/runs/baseline-full-perpid-s42-20260908-1800-c96338d \\
      --run V1_fixed_horizon=results-cluster-10p/runs/<mpcc-fixed_horizon-perpid-...> \\
      --run V2_fixed_speed=results-cluster-10p/runs/<mpcc-fixed_speed-perpid-...> \\
      [--train-widths 10,50] [--out paper/quant_r1/r1_ablation_table.csv]

--train-widths: the fitted steering widths in mm. The paper's Study-1 runs
(fitted at 22c349c, before the 2026-09-10 split change) trained on 10,50;
runs fitted at HEAD train on 10,16.5,50. Pass a run directory (with eval/)
or an eval directory itself. The Human rows are taken from the first run.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

import plot_laws as pl   # same folder; aggregation + geometry helpers

REPO = Path(__file__).resolve().parent.parent
TYPE_BY_TID = {}
for lo, name in ((1, "sinusoidal"), (6, "corner"), (11, "straight"),
                 (26, "gentle_sinusoidal"), (31, "sharp_sinusoidal")):
    for t in range(lo, lo + 5):
        TYPE_BY_TID[t] = name


def eval_dir(p):
    p = Path(p)
    if not p.is_absolute():
        p = REPO / p
    return p / "eval" if (p / "eval").is_dir() else p


def mt_by_tid(csv_path, mt_col, want_human):
    """{tid: mean MT} for the Simulator (or Human) rows of one results CSV."""
    per = defaultdict(list)
    ids = {}
    for src, tid, id_, mt in pl._iter_rows(csv_path, mt_col, "Sim"):
        if (src == "Human") == want_human:
            per[tid].append(mt)
            ids[tid] = id_
    return {t: float(np.mean(v)) for t, v in per.items()}, ids


def law(mt, ids):
    tids = sorted(mt, key=int)
    x = np.array([ids[t] for t in tids]); y = np.array([mt[t] for t in tids])
    a, b, r2 = pl.fit_line(x, y)
    return a, b, r2


def budget_law(mt, geoms, lam):
    tids = sorted(mt, key=int)
    w = np.array([geoms[t][0] for t in tids]); L = np.array([geoms[t][1] for t in tids])
    phi = np.array([geoms[t][2] for t in tids]); y = np.array([mt[t] for t in tids])
    return pl.fit_line((L + lam * phi) / w, y)


def steering_metrics(summary_csv, train_widths):
    g = defaultdict(lambda: defaultdict(list))
    ratio_w = defaultdict(list)
    with open(summary_csv) as f:
        for r in csv.DictReader(f):
            split = "Train" if float(r["width_mm"]) in train_widths else "Test"
            ratio = float(r["model_time_mean_s"]) / float(r["human_time_mean_s"])
            vals = {"lat_mm": float(r["lateral_rmse"]) * 1000.0, "spd": float(r["speed_rmse"]),
                    "corr": float(r["speed_corr"]), "ratio": ratio}
            for m, v in vals.items():
                g[split][m].append(v)
            ratio_w[float(r["width_mm"])].append(ratio)
    return g, {w: float(np.mean(v)) for w, v in sorted(ratio_w.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=DIR")
    ap.add_argument("--train-widths", default="10,50")
    ap.add_argument("--out", default=str(REPO / "paper" / "quant_r1" / "r1_ablation_table.csv"))
    a = ap.parse_args()
    train_w = {float(w) for w in a.train_widths.split(",")}
    runs = [(s.split("=", 1)[0], eval_dir(s.split("=", 1)[1])) for s in a.run]
    geoms = pl.steering_geometry()

    first = runs[0][1]
    h_steer, ids = mt_by_tid(first / "Steering" / "steering_results.csv", "MT_s", True)
    fcsv = first / "Fitts" / "fitts_results.csv"
    h_fitts, fids = mt_by_tid(fcsv, "MT_kin_s", True) if fcsv.exists() else ({}, {})
    if any(t not in geoms for t in h_steer):
        raise SystemExit("steering conditions not in the 8-participant battery: this table is "
                         "for Study 1 runs; use stats_study2.py --s2-model for eval14p runs")
    _, lam = pl.budget_id_data({"Human": h_steer}, geoms)

    rows = []
    def add(label, steer, fitts, summary):
        row = {"source": label}
        row["steer_a"], row["steer_b"], row["steer_r2"] = law(steer, ids)
        row["steerK_a"], row["steerK_b"], row["steerK_r2"] = budget_law(steer, geoms, lam)
        if fitts:
            row["fitts_a"], row["fitts_b"], row["fitts_r2"] = law(fitts, fids)
        if summary is not None:
            g, rw = steering_metrics(summary, train_w)
            for split in ("Train", "Test"):
                for m in ("lat_mm", "spd", "corr", "ratio"):
                    v = np.array(g[split][m]) if g[split][m] else np.array([np.nan])
                    row[f"{split}_{m}_mean"] = float(np.mean(v)); row[f"{split}_{m}_sd"] = float(np.std(v))
            for w, r in rw.items():
                row[f"ratio_W{w:g}"] = r
        rows.append(row)

    add("Human", h_steer, h_fitts, None)
    for label, ed in runs:
        steer, _ = mt_by_tid(ed / "Steering" / "steering_results.csv", "MT_s", False)
        fp = ed / "Fitts" / "fitts_results.csv"
        fitts = mt_by_tid(fp, "MT_kin_s", False)[0] if fp.exists() else {}
        add(label, steer, fitts, ed / "Steering" / "steering_condition_summary.csv")

    print(f"curvature-aware ID: lambda = {lam:.3f} m/rad (fit on human); train widths {sorted(train_w)} mm\n")
    print(f"{'source':<20} {'steer b':>8} {'R2':>5} {'ID_k b':>8} {'R2':>5} {'Fitts b':>8} {'R2':>5}")
    for r in rows:
        print(f"{r['source']:<20} {r['steer_b']:8.4f} {r['steer_r2']:5.2f} {r['steerK_b']:8.4f} "
              f"{r['steerK_r2']:5.2f} {r.get('fitts_b', np.nan):8.4f} {r.get('fitts_r2', np.nan):5.2f}")
    print(f"\n{'source':<20} {'split':<5} {'lat RMSE mm':>13} {'speed RMSE':>13} {'speed corr':>12} {'time ratio':>12}")
    for r in rows[1:]:
        for s in ("Train", "Test"):
            print(f"{r['source']:<20} {s:<5} "
                  + " ".join(f"{r[f'{s}_{m}_mean']:{p}} ±{r[f'{s}_{m}_sd']:{q}}"
                             for m, p, q in (("lat_mm", "6.2f", "5.2f"), ("spd", "6.3f", "5.3f"),
                                             ("corr", "5.2f", "5.2f"), ("ratio", "5.2f", "5.2f"))))
    wcols = sorted({k for r in rows for k in r if k.startswith("ratio_W")}, key=lambda k: float(k[7:]))
    print(f"\nmodel/human time ratio by width: " + "  ".join(k[7:] + " mm" for k in wcols))
    for r in rows[1:]:
        print(f"  {r['source']:<20} " + "  ".join(f"{r.get(k, np.nan):5.2f}" for k in wcols))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

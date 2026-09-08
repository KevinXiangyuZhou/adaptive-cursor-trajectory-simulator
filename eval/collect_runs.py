"""Collect every isolated run under <root>/runs/*/ into one SUMMARY.csv.

One row per run: identity (run_id, model, variant, kind, seed, commit,
status), fit summary (best loss, held-out train/test loss), Steering-Law and
Fitts'-law regressions (slope, R^2, throughput) recomputed from the run's own
eval CSV/JSON the way the eval plots do (one point per condition averaged
across participants), and trajectory-level agreement on training vs held-out
widths (lateral RMSE, speed RMSE, speed corr, time ratio).

    python eval/collect_runs.py [--root <RESULTS_ROOT>] [--out SUMMARY.csv] [--quiet]

Status: 'done' when the aggregate wrote DONE; 'fit_done' when fitted personas
exist but the eval has not finished; 'pending' otherwise; 'failed' when any
SLURM .err log ends with a Python traceback and no later success marker.
Paper tables and figures must cite RUN_IDs from this file (paper/RUNS_USED.md),
never "the latest directory".
"""
import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

DEFAULT_ROOT = os.environ.get("RESULTS_ROOT", "/home/xiangyz/ondemand/data/sys/myjobs/projects/chi-27/results")
TRAIN_WIDTHS_MM = {10.0, 30.0, 50.0}   # fit_speed_model.TRAIN_WIDTHS


def _linfit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return dict(slope=None, intercept=None, r2=None, n=int(ok.sum()))
    b, a = np.polyfit(x[ok], y[ok], 1)
    yhat = a + b * x[ok]
    ss = float(np.sum((y[ok] - yhat) ** 2)); st = float(np.sum((y[ok] - y[ok].mean()) ** 2))
    return dict(slope=float(b), intercept=float(a), r2=(1 - ss / st if st > 0 else None), n=int(ok.sum()))


def _read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _steering(run_dir):
    p = run_dir / "eval" / "Steering" / "steering_condition_summary.csv"
    if not p.exists():
        return {}
    rows = _read_csv(p)
    by_tid = {}
    for r in rows:
        try:
            by_tid.setdefault(r["tid"], []).append(r)
        except KeyError:
            continue
    out = {}
    # law: one point per condition, averaged across participants
    ids, th, tm = [], [], []
    for tid, rs in by_tid.items():
        try:
            ids.append(float(rs[0]["ID_L_over_W"]))
            th.append(np.mean([float(r["human_time_mean_s"]) for r in rs]))
            tm.append(np.mean([float(r["model_time_mean_s"]) for r in rs]))
        except (KeyError, ValueError):
            continue
    fh, fm = _linfit(ids, th), _linfit(ids, tm)
    out.update({"steer_slope_human": fh["slope"], "steer_r2_human": fh["r2"],
                "steer_slope_model": fm["slope"], "steer_r2_model": fm["r2"], "steer_n_cond": fm["n"]})
    # trajectory agreement by train/test width
    for split in ("train", "test"):
        sel = [r for r in rows if r.get("width_mm") and
               ((float(r["width_mm"]) in TRAIN_WIDTHS_MM) == (split == "train"))]
        def m(key):
            v = [float(r[key]) for r in sel if r.get(key) not in (None, "", "nan")]
            return float(np.mean(v)) if v else None
        ratios = [float(r["model_time_mean_s"]) / float(r["human_time_mean_s"]) for r in sel
                  if r.get("human_time_mean_s") and float(r["human_time_mean_s"]) > 0]
        out.update({f"lat_rmse_mm_{split}": (m("lateral_rmse") * 1000 if m("lateral_rmse") is not None else None),
                    f"speed_rmse_{split}": m("speed_rmse"), f"speed_corr_{split}": m("speed_corr"),
                    f"time_ratio_{split}": (float(np.mean(ratios)) if ratios else None),
                    f"timeouts_{split}": sum(int(float(r.get("n_timed_out", 0) or 0)) for r in sel),
                    f"n_rows_{split}": len(sel)})
    return out


def _fitts(run_dir):
    p = run_dir / "eval" / "Fitts" / "fitts_regression.json"
    if not p.exists():
        return {}
    j = json.load(open(p)).get("aligned", {})
    out = {}
    for side in ("human", "model"):
        s = j.get(side, {})
        out.update({f"fitts_slope_{side}": s.get("b_slope_s_per_bit"), f"fitts_intercept_{side}": s.get("a_intercept"),
                    f"fitts_r2_{side}": s.get("r_squared"), f"fitts_tp_{side}": s.get("throughput_mean_bps")})
    return out


def _fit_summary(run_dir, kind):
    stage = run_dir / "fit" / "stages" / ("pooled8" if kind == "pooled8" else "base")
    recs = sorted(stage.glob("*_fit_s*.json")) if stage.exists() else []
    if not recs:
        return {"n_fitted": 0}
    best, ho = [], {"tunnel": {"train": [], "test": []}, "pointing": {"train": [], "test": []}}
    for r in recs:
        j = json.load(open(r))
        if j.get("best_loss") is not None:
            best.append(float(j["best_loss"]))
        h = j.get("heldout") or {}
        if kind == "pooled8":      # {pid: {tunnel: {train, test}, pointing: {...}}}
            for pid, hh in h.items():
                for k in ho:
                    for s in ho[k]:
                        v = (hh.get(k) or {}).get(s)
                        if v is not None: ho[k][s].append(float(v))
        else:
            for k in ho:
                for s in ho[k]:
                    v = (h.get(k) or {}).get(s)
                    if v is not None: ho[k][s].append(float(v))
    def mean(v): return float(np.mean(v)) if v else None
    return {"n_fitted": len(recs), "fit_best_loss": mean(best),
            "heldout_tunnel_train": mean(ho["tunnel"]["train"]), "heldout_tunnel_test": mean(ho["tunnel"]["test"]),
            "heldout_point_train": mean(ho["pointing"]["train"]), "heldout_point_test": mean(ho["pointing"]["test"])}


def _status(run_dir, kind, superseded=""):
    if superseded:
        # a run marked superseded (cancelled, preflight, refit ...) is
        # reported as such regardless of what its tree contains
        return f"superseded:{superseded}"
    if (run_dir / "DONE").exists():
        return "done"
    for err in sorted((run_dir / "logs").glob("*.err")):
        try:
            tail = err.read_text()[-4000:]
        except OSError:
            continue
        if "Traceback (most recent call last)" in tail and "DONE" not in tail:
            return "failed"
    stage = run_dir / "fit" / "stages" / ("pooled8" if kind == "pooled8" else "base")
    if stage.exists() and list(stage.glob("*_config_s*.json")):
        return "fit_done"
    return "pending"


def collect(root):
    rows = []
    for info_p in sorted(glob.glob(str(Path(root) / "runs" / "*" / "RUN_INFO.json"))):
        run_dir = Path(info_p).parent
        info = json.load(open(info_p))
        kind = info.get("kind", "perpid")
        row = {"run_id": info.get("run_id", run_dir.name), "model": info.get("model"), "variant": info.get("variant"),
               "kind": kind, "seed": info.get("seed"), "commit": (info.get("commit") or "")[:7],
               "dirty": info.get("dirty"), "submitted": info.get("submitted"),
               "status": _status(run_dir, kind, info.get("superseded_by", "")),
               "superseded_by": info.get("superseded_by", ""),
               "path": str(run_dir)}
        row.update(_fit_summary(run_dir, kind))
        row.update(_steering(run_dir))
        row.update(_fitts(run_dir))
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out", default=None, help="default <root>/runs/SUMMARY.csv")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    rows = collect(a.root)
    if not rows:
        print(f"no runs under {a.root}/runs"); return
    cols = []
    for r in rows:
        for k in r:
            if k not in cols: cols.append(k)
    out = Path(a.out) if a.out else Path(a.root) / "runs" / "SUMMARY.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})
    if not a.quiet:
        def fmt(v, nd=3):
            return "" if v in (None, "") else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))
        print(f"{'run_id':58s} {'status':8s} {'steer b(m/h)':14s} {'R2 m/h':11s} {'fitts b(m/h)':14s} {'TR tr/te':11s} {'lat te':6s}")
        for r in rows:
            print(f"{r['run_id'][:58]:58s} {r['status']:8s} "
                  f"{fmt(r.get('steer_slope_model')):>6}/{fmt(r.get('steer_slope_human')):<6} "
                  f"{fmt(r.get('steer_r2_model'),2):>5}/{fmt(r.get('steer_r2_human'),2):<5} "
                  f"{fmt(r.get('fitts_slope_model')):>6}/{fmt(r.get('fitts_slope_human')):<6} "
                  f"{fmt(r.get('time_ratio_train'),2):>5}/{fmt(r.get('time_ratio_test'),2):<5} "
                  f"{fmt(r.get('lat_rmse_mm_test'),1):>6}")
        print(f"\nwrote {out} ({len(rows)} runs)")


if __name__ == "__main__":
    main()

"""Generalisation summary of one eval-14p run: how well a persona fitted on
the 8-participant cohort reproduces the 14 NEW participants' steering
trials, from the harness outputs (eval/eval-main/run_eval.py) alone.

    python eval-14p/summarize_14p.py <results dir> [--data-dir eval-14p/human_data/raw]
        [--reference <8p Steering/steering_condition_summary.csv>]
        [--reference-data-dir human_data/task_aligned_all] [--out PREFIX]

<results dir> holds Steering/steering_condition_summary.csv (one row per
participant x condition; no pid column) and Steering/participant_<pid>/
trial_<tid>/results_summary.json (the same metrics, per participant).
Tunnel type and width come from the trial conditions in --data-dir.

Tables (printed, and written to PREFIX.txt / PREFIX.csv):
  by tunnel type x width, by width, by tunnel type, overall — each with the
  Steering-Law fit (MT = a + b * L/W, one point per condition averaged over
  participants) and the per-condition agreement means the fitting reports:
  MT ratio model/human, lateral RMSE, speed RMSE, speed correlation,
  relative time difference, time-outs. Widths are split into those the
  fitting cohort also had (10 and 50 mm) and the new ones (20/30/40 mm).
  by participant: the same means per new participant.
With --reference the same cohort-level tables are computed from the source
run's own summary on the 8 fitting participants (in-sample), so the two
can be read side by side.
"""
import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = ["mt_ratio", "lat_rmse_mm", "speed_rmse", "speed_corr", "rel_time_diff", "timeouts"]
# 8p cohort tid layout (human_data/task_aligned_all), used only when that
# data dir is not available to scan
_REF_TID_TYPES = {**{t: "sinusoidal" for t in range(1, 6)}, **{t: "corner" for t in range(6, 11)},
                  **{t: "straight" for t in range(11, 16)}, **{t: "gentle_sinusoidal" for t in range(26, 31)},
                  **{t: "sharp_sinusoidal" for t in range(31, 36)}}
TYPE_ORDER = ["straight", "gentle_sinusoidal", "sinusoidal", "sharp_sinusoidal", "corner"]


def tunnel_type(cond):
    t = cond.get("tunnelType")
    if t:
        return t
    return "straight" if float(cond.get("curvature", 0) or 0) == 0 else "sinusoidal"


def scan_types(data_dir):
    """tid -> (tunnel type, width mm) for the steering conditions of a data dir."""
    out = {}
    for fpath in sorted(Path(data_dir).glob("*.json")):
        try:
            data = json.load(open(fpath))
        except Exception:
            continue
        sessions = data.get("sessions") or ([{"trialData": data.get("trialData", [])}])
        for s in sessions:
            for t in s.get("trialData", []):
                tid, cond = t.get("trial_id"), t.get("condition") or {}
                if tid is None or tid in out or "tunnelWidth" not in cond or "segment1Width" in cond:
                    continue
                out[tid] = (tunnel_type(cond), round(float(cond["tunnelWidth"]) * 1000, 1))
        if out:
            break   # every participant saw the same conditions
    return out


def _f(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def row_metrics(r):
    """Per-(participant, condition) agreement row -> metric dict."""
    h, m = _f(r.get("human_time_mean_s")), _f(r.get("model_time_mean_s"))
    return {
        "mt_ratio": (m / h) if (h and m is not None and h > 0) else None,
        "lat_rmse_mm": (_f(r.get("lateral_rmse")) or 0) * 1000 if _f(r.get("lateral_rmse")) is not None else None,
        "speed_rmse": _f(r.get("speed_rmse")),
        "speed_corr": _f(r.get("speed_corr")),
        "rel_time_diff": _f(r.get("time_diff")),
        "timeouts": _f(r.get("n_timed_out")) or 0,
        "human_mt": h, "model_mt": m, "ID": _f(r.get("ID_L_over_W")),
    }


def load_condition_rows(csv_path, tid_types):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            tid = int(r["tid"])
            ttype, width = tid_types.get(tid, ("?", _f(r.get("width_mm"))))
            rows.append({"tid": tid, "type": ttype, "width": width, **row_metrics(r)})
    return rows


def load_participant_rows(results_dir, tid_types):
    """pid -> rows, from Steering/participant_<pid>/trial_<tid>/results_summary.json."""
    per = defaultdict(list)
    for p in sorted(Path(results_dir, "Steering").glob("participant_*")):
        pid = p.name[len("participant_"):]
        for tdir in sorted(p.glob("trial_*")):
            js = tdir / "results_summary.json"
            if not js.exists():
                continue
            d = json.load(open(js))
            tid = int(d.get("trial_id", tdir.name.split("_")[1]))
            met = dict(d.get("metrics", {}))
            hct, mct = d.get("human", {}).get("completion_times", []), d.get("model", {}).get("completion_times", [])
            met.setdefault("human_time_mean_s", np.mean(hct) if hct else None)
            met.setdefault("model_time_mean_s", np.mean(mct) if mct else None)
            met.setdefault("n_timed_out", d.get("model", {}).get("n_timed_out", 0))
            ttype, width = tid_types.get(tid, ("?", None))
            per[pid].append({"tid": tid, "type": ttype, "width": width, **row_metrics(met)})
    return per


def steering_law(rows):
    """MT = a + b * ID, one point per condition (tid) averaged across participants."""
    by = defaultdict(list)
    for r in rows:
        if r["ID"] is not None:
            by[r["tid"]].append(r)
    ids, th, tm = [], [], []
    for tid, rs in by.items():
        h = [r["human_mt"] for r in rs if r["human_mt"] is not None]
        m = [r["model_mt"] for r in rs if r["model_mt"] is not None]
        if h and m:
            ids.append(rs[0]["ID"]); th.append(np.mean(h)); tm.append(np.mean(m))
    def fit(y):
        if len(ids) < 3:
            return (None, None, None)
        b, a = np.polyfit(ids, y, 1)
        yhat = a + b * np.asarray(ids)
        ss, st = np.sum((np.asarray(y) - yhat) ** 2), np.sum((np.asarray(y) - np.mean(y)) ** 2)
        return (float(a), float(b), float(1 - ss / st) if st > 0 else None)
    return fit(th), fit(tm), len(ids)


def agg(rows):
    out = {"n_rows": len(rows), "n_cond": len({r["tid"] for r in rows})}
    for k in METRICS:
        v = [r[k] for r in rows if r[k] is not None]
        out[k] = (float(np.sum(v)) if k == "timeouts" else float(np.mean(v))) if v else None
    for k in ("human_mt", "model_mt"):
        v = [r[k] for r in rows if r[k] is not None]
        out[k] = float(np.mean(v)) if v else None
    return out


def fmt(v, nd=3):
    return "   -  " if v is None else (f"{v:6.{nd}f}" if isinstance(v, float) else f"{v:6d}")


HEADER = f"{'group':32s} {'n':>4s} {'cond':>4s} {'MT_h':>6s} {'MT_m':>6s} {'ratio':>6s} {'latRMSEmm':>9s} {'spdRMSE':>7s} {'spdCorr':>7s} {'relTdiff':>8s} {'timeouts':>8s}"


def line(name, a):
    return (f"{name:32s} {a['n_rows']:4d} {a['n_cond']:4d} {fmt(a['human_mt'])} {fmt(a['model_mt'])} {fmt(a['mt_ratio'])} "
            f"{fmt(a['lat_rmse_mm'], 2):>9s} {fmt(a['speed_rmse']):>7s} {fmt(a['speed_corr']):>7s} {fmt(a['rel_time_diff']):>8s} "
            f"{int(a['timeouts'] or 0):8d}")


def cohort_tables(rows, label, shared_widths):
    """Text lines + long-format records for one cohort's condition rows."""
    lines, recs = [], []
    def add(table, group, sel):
        if not sel:
            return
        a = agg(sel); lines.append(line(group, a))
        recs.append({"cohort": label, "table": table, "group": group, **{k: a[k] for k in ("n_rows", "n_cond", "human_mt", "model_mt", *METRICS)}})
    widths = sorted({r["width"] for r in rows if r["width"] is not None})
    types = [t for t in TYPE_ORDER if any(r["type"] == t for r in rows)] + sorted({r["type"] for r in rows} - set(TYPE_ORDER))
    (ah, bh, r2h), (am, bm, r2m), ncond = steering_law(rows)
    lines.append(f"== {label}: {len(rows)} participant x condition rows, {ncond} conditions ==")
    if bh is not None:
        lines.append(f"  Steering law  human: MT = {ah:.3f} + {bh:.4f} * L/W (R2 {r2h:.3f})   model: MT = {am:.3f} + {bm:.4f} * L/W (R2 {r2m:.3f})")
    lines.append(HEADER)
    add("overall", "all conditions", rows)
    add("overall", "widths shared with fit cohort", [r for r in rows if r["width"] in shared_widths])
    add("overall", "widths new to the model", [r for r in rows if r["width"] is not None and r["width"] not in shared_widths])
    lines.append("-- by width --")
    for w in widths:
        add("by_width", f"W={w:g}mm" + (" (shared)" if w in shared_widths else " (new)"), [r for r in rows if r["width"] == w])
    lines.append("-- by tunnel type --")
    for t in types:
        add("by_type", t, [r for r in rows if r["type"] == t])
    lines.append("-- by tunnel type x width --")
    for t in types:
        for w in widths:
            add("by_type_width", f"{t} W={w:g}mm", [r for r in rows if r["type"] == t and r["width"] == w])
    return lines, recs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("results_dir")
    ap.add_argument("--data-dir", default="eval-14p/human_data/raw")
    ap.add_argument("--reference", default=None, help="fitting cohort's Steering/steering_condition_summary.csv")
    ap.add_argument("--reference-data-dir", default="human_data/task_aligned_all")
    ap.add_argument("--out", default=None, help="output prefix (default <results_dir>/SUMMARY_14p)")
    args = ap.parse_args()

    rd = Path(args.results_dir)
    csv_path = rd / "Steering" / "steering_condition_summary.csv"
    if not csv_path.exists():
        raise SystemExit(f"missing {csv_path} (run the aggregate first)")
    tid_types = scan_types(args.data_dir)
    rows = load_condition_rows(csv_path, tid_types)
    per_pid = load_participant_rows(rd, tid_types)

    ref_rows = []
    if args.reference:
        ref_types = scan_types(args.reference_data_dir) if os.path.isdir(args.reference_data_dir) else {}
        if not ref_types:
            ref_types = {t: (ty, None) for t, ty in _REF_TID_TYPES.items()}
        ref_rows = load_condition_rows(args.reference, ref_types)
    # widths the fitting cohort had (in-sample for the reference; 10 and 50 mm
    # are the only ones the 14p protocol shares when no reference is given)
    ref_widths = {r["width"] for r in ref_rows if r["width"] is not None}
    shared = ref_widths if ref_widths else {10.0, 50.0}

    lines, recs = cohort_tables(rows, f"NEW participants ({len(per_pid)} pids, {rd})", shared)
    lines.append("")
    lines.append("-- by participant (new cohort) --")
    lines.append(HEADER)
    for pid, prs in sorted(per_pid.items()):
        a = agg(prs); lines.append(line(pid, a))
        recs.append({"cohort": "new", "table": "by_participant", "group": pid, **{k: a[k] for k in ("n_rows", "n_cond", "human_mt", "model_mt", *METRICS)}})
    if ref_rows:
        lines.append("")
        rl, rr = cohort_tables(ref_rows, f"REFERENCE fitting cohort, in-sample ({args.reference})", shared)
        lines += rl; recs += rr
    text = "\n".join(lines)
    print(text)

    out = Path(args.out) if args.out else rd / "SUMMARY_14p"
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(out) + ".txt").write_text(text + "\n")
    with open(str(out) + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cohort", "table", "group", "n_rows", "n_cond", "human_mt", "model_mt", *METRICS])
        w.writeheader(); w.writerows(recs)
    print(f"\nSaved: {out}.txt, {out}.csv")


if __name__ == "__main__":
    main()

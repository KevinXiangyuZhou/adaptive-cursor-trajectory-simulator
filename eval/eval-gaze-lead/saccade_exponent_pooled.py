"""Saccade-distance-vs-width exponent, human and pooled model, ONE estimator.

Motivation: the per-saccade log-log fit (saccade_width_analysis.py) is biased
flat by the detector's A_MIN floor censoring small saccades at narrow widths
(12-28% of W=10 events sit within 4 mm of the floor), giving b_human ~ 0.55,
while per-condition medians give ~0.72 for the same participants. Any human
vs model exponent comparison must therefore use one estimator on both sides.

This script uses per-condition MEDIANS throughout:
  human: data/saccade_events.csv (committed detector output, 6 participants,
         5 tunnel types x 5 widths) -> per-participant, per-width median
         amplitude (pooled over types) -> log-log OLS over the 5 widths ->
         b per participant, mean +/- SD.
  model: the POOLED-8 persona re-simulated on the full 25-trial steering
         battery (p04 geometry, noise on), SEEDS seeds; forward saccades =
         contiguous positive lead-velocity runs, same A_MIN/A_MAX amplitude
         gates -> per-seed, per-width medians -> b per seed, mean +/- SD.

Also prints the straight/sinusoid equal-width median amplitude factor for
both sides (the emergent shape effect), same estimator.

Usage: python eval/eval-gaze-lead/saccade_exponent_pooled.py [--seeds 5]
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "eval" / "model_fitting"))
import model_gaze_lead as mg      # noqa: E402
import fit_speed_model as fsm     # noqa: E402

EVENTS = SCRIPT_DIR / "human-gaze-lead-10p" / "data" / "saccade_events.csv"
POOLED_CONFIG = (PROJECT_ROOT / "results-cluster-10p" / "anchor_fitting_pooled8"
                 / "stages" / "pooled8" / "pooled8_anchor_config_s42.json")
GEOMETRY_PID = "p04"   # battery geometry is nominally identical across pids

A_MIN, A_MAX = 0.008, 0.40   # same amplitude gates as saccade_width_analysis
CU_MIN, CU_MAX = 0.05, 3.0   # same catch-up duration window

TYPE_BY_TID = {}
for lo, name in ((1, "normal"), (6, "corner"), (11, "straight"),
                 (26, "gentle"), (31, "sharp")):
    for t in range(lo, lo + 5):
        TYPE_BY_TID[t] = name
WIDTH_BY_TID = {t: w for lo in (1, 6, 11, 26, 31)
                for t, w in zip(range(lo, lo + 5), (10.0, 12.5, 16.5, 25.0, 50.0))}


def loglog_b(widths, medians):
    b, loga, r, p, se = stats.linregress(np.log(widths), np.log(medians))
    return b, r * r


def human_side():
    ev = pd.read_csv(EVENTS)
    print(f"human: {len(ev)} events, {ev.participant.nunique()} participants, "
          f"types {sorted(ev.type_label.unique())}")
    bs = []
    for L, g in ev.groupby("participant"):
        med = g.groupby("width_mm")["amp_m"].median()
        b, r2 = loglog_b(med.index.to_numpy(), med.to_numpy())
        bs.append(b)
        print(f"  {L}: median-based b={b:.2f} (log-log r2={r2:.3f}, n={len(g)})")
    print(f"  HUMAN median-based b = {np.mean(bs):.2f} +/- {np.std(bs):.2f} SD "
          f"(n={len(bs)} participants)")

    # catch-up duration vs width, same estimator (catchup_s is stored on
    # each event by saccade_width_analysis; same 0.05-3 s window)
    cu = ev[(ev["catchup_s"] >= CU_MIN) & (ev["catchup_s"] <= CU_MAX)]
    cbs = []
    for L, g in cu.groupby("participant"):
        med = g.groupby("width_mm")["catchup_s"].median()
        b, r2 = loglog_b(med.index.to_numpy(), med.to_numpy())
        cbs.append((b, r2))
        print(f"  {L}: catch-up median-based b={b:.2f} (r2={r2:.3f}, n={len(g)})")
    cb = [b for b, _ in cbs]
    print(f"  HUMAN catch-up b = {np.mean(cb):.2f} +/- {np.std(cb):.2f} SD, "
          f"r2 range {min(r for _, r in cbs):.2f}-{max(r for _, r in cbs):.2f}")

    # shape factor: straight / normal median amplitude at equal width
    ratios = []
    for L, g in ev.groupby("participant"):
        m = g.groupby(["type_label", "width_mm"])["amp_m"].median()
        r = [m.get(("straight", w), np.nan) / m.get(("normal", w), np.nan)
             for w in (10.0, 12.5, 16.5, 25.0, 50.0)]
        ratios.append(np.nanmedian(r))
    print(f"  HUMAN straight/sinusoid factor = {np.median(ratios):.2f} "
          f"(median over participants; per-p {np.round(ratios, 2)})")
    return np.mean(bs), np.std(bs)


def detect_model(t, lead):
    """Forward saccades in a model lead trace: contiguous positive-velocity
    runs; amplitude = summed lead increase, gated to [A_MIN, A_MAX].
    Catch-up = time from the end of one accepted jump to the start of the
    next (the human detector's definition), gated to [CU_MIN, CU_MAX]."""
    amps, catchups = [], []
    cur, jump_end_t = 0.0, None
    d = np.diff(lead)
    for i, dl in enumerate(d):
        if dl > 0:
            if cur == 0.0 and jump_end_t is not None:
                cu = t[i] - jump_end_t
                if CU_MIN <= cu <= CU_MAX:
                    catchups.append(cu)
                jump_end_t = None
            cur += dl
        else:
            if cur > 0.0:
                if A_MIN <= cur <= A_MAX:
                    amps.append(cur)
                    jump_end_t = t[i]
                cur = 0.0
    if A_MIN <= cur <= A_MAX:
        amps.append(cur)
    return amps, catchups


def model_side(n_seeds):
    cfg = json.load(open(POOLED_CONFIG))
    cfg.pop("_description", None)
    cfg["add_noise"] = True
    if not float(cfg.get("replan_latency_cv", 0.0) or 0.0):
        cfg["replan_latency_cv"] = 0.89
    sim = fsm._make_sim(cfg)
    rounds_by_tid, t2c, t2b = fsm.load_participant(GEOMETRY_PID)
    print(f"model: pooled-8 persona (gamma={cfg['budget']['gamma']}), "
          f"{n_seeds} seeds x 25 trials")

    tasks = {}
    for tid in TYPE_BY_TID:
        built = mg.build_task(tid, t2b.get(tid), t2c.get(tid),
                              rounds_by_tid.get(tid, []))
        if built is not None:
            tasks[tid] = built

    bs, ratios, cbs = [], [], []
    for s in range(n_seeds):
        rows, cu_rows = [], []
        for tid, (task_cfg, cl) in tasks.items():
            np.random.seed(1000 * (s + 1) + tid)
            res = mg.model_lead_trace(sim, task_cfg, cl)
            if res is None:
                print(f"  seed {s}: t{tid} FAILED")
                continue
            amps, catchups = detect_model(np.asarray(res[0]), np.asarray(res[1]))
            rows += [(TYPE_BY_TID[tid], WIDTH_BY_TID[tid], a) for a in amps]
            cu_rows += [(WIDTH_BY_TID[tid], c) for c in catchups]
        df = pd.DataFrame(rows, columns=["type_label", "width_mm", "amp_m"])
        med = df.groupby("width_mm")["amp_m"].median()
        b, r2 = loglog_b(med.index.to_numpy(), med.to_numpy())
        bs.append(b)
        m = df.groupby(["type_label", "width_mm"])["amp_m"].median()
        r = [m.get(("straight", w), np.nan) / m.get(("normal", w), np.nan)
             for w in (10.0, 12.5, 16.5, 25.0, 50.0)]
        ratios.append(np.nanmedian(r))
        cdf = pd.DataFrame(cu_rows, columns=["width_mm", "catchup_s"])
        cmed = cdf.groupby("width_mm")["catchup_s"].median()
        cb, cr2 = loglog_b(cmed.index.to_numpy(), cmed.to_numpy())
        cbs.append((cb, cr2))
        print(f"  seed {s}: b={b:.2f} (r2={r2:.3f}, {len(df)} saccades), "
              f"straight/sin factor {ratios[-1]:.2f}, "
              f"catch-up b={cb:.2f} (r2={cr2:.3f}, n={len(cdf)})")
    print(f"  MODEL median-based b = {np.mean(bs):.2f} +/- {np.std(bs):.2f} SD "
          f"(n={len(bs)} seeds)")
    print(f"  MODEL straight/sinusoid factor = {np.median(ratios):.2f}")
    cb = [b for b, _ in cbs]
    print(f"  MODEL catch-up b = {np.mean(cb):.2f} +/- {np.std(cb):.2f} SD, "
          f"r2 range {min(r for _, r in cbs):.2f}-{max(r for _, r in cbs):.2f}")
    return np.mean(bs), np.std(bs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    hb, hsd = human_side()
    mb, msd = model_side(a.seeds)
    print(f"\nSAME-ESTIMATOR COMPARISON: human b = {hb:.2f} +/- {hsd:.2f}, "
          f"model b = {mb:.2f} +/- {msd:.2f}")


if __name__ == "__main__":
    main()

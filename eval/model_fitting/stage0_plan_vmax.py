"""Stage 0: measure plan_vmax from pointing data instead of fitting it.

plan_vmax bounds the planned PACE of a free-space reach (the plan's travel
time is never shorter than distance / plan_vmax), so the matching human
measurement is per-round pace D / MT_kin — straight-line distance from the
round's start to the target centre over the aligned kinematic movement time
(movement onset -> final target entry). Raw speed peaks are the wrong
estimator: the plant's bell profile peaks ~1.9x above its mean, while
plan_vmax constrains the mean.

The pinned value is the POOLED p90 across all participants' pointing rounds:
the cap should be the fast-reach envelope (it binds on far/easy targets),
not the typical pace. 2026-09-03 measurement on the 10p batch (6 kept
participants, 267 rounds): median 0.43, p75 0.53, p90 0.66 m/s -> 0.66.

Usage:
  python -m eval.model_fitting.stage0_plan_vmax            # print stats
  python -m eval.model_fitting.stage0_plan_vmax --write    # also set
      plan_vmax in eval/model_fitting/base_configs_gaze/{pid}.json
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for p in (PROJECT_ROOT / "eval" / "eval-main", PROJECT_ROOT / "hcs_package" / "src",
          PROJECT_ROOT / "eval", SCRIPT_DIR):
    sys.path.insert(0, str(p))
import run_eval as em            # noqa: E402
import fit_speed_model as fsm    # noqa: E402

LETTERS = ["p01", "p02", "p03", "p04", "p07", "p10"]
QUANTILE = 90       # fast-reach envelope
MT_MIN_S = 0.1      # drop degenerate rounds
BASE_CONFIG_DIR = SCRIPT_DIR / "base_configs_gaze"


def round_pace(r, center, R):
    traj = r["trajectory"]
    times = [(t - r["timestamps"][0]) / 1000.0 for t in r["timestamps"]]
    onset = em.movement_onset_time(traj, times)
    d2c = np.hypot(np.asarray(traj)[:, 0] - center[0],
                   np.asarray(traj)[:, 1] - center[1])
    inside = np.flatnonzero(d2c < R)
    if not len(inside):
        return None
    out = np.flatnonzero(d2c >= R)
    fe_idx = out[-1] + 1 if len(out) and out[-1] + 1 < len(times) else inside[0]
    mt = times[fe_idx] - onset
    if mt < MT_MIN_S:
        return None
    D = float(np.hypot(traj[0][0] - center[0], traj[0][1] - center[1]))
    return D / mt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="set the pooled value as plan_vmax in the base configs")
    a = ap.parse_args()

    pooled = []
    for L in LETTERS:
        rounds, t2c, t2b = fsm.load_participant(L)
        vs = []
        for tid, rs in rounds.items():
            if t2b.get(tid) != "fitts":
                continue
            R = t2c[tid]["targetRadius"]
            for r in rs:
                center = em.pointing_target_center(r)
                v = round_pace(r, center, R)
                if v is not None:
                    vs.append(v)
        pooled.extend(vs)
        print(f"{L}: n={len(vs)}  median={np.median(vs):.3f}  "
              f"p{QUANTILE}={np.percentile(vs, QUANTILE):.3f}")
    val = round(float(np.percentile(pooled, QUANTILE)), 2)
    print(f"\nPOOLED (n={len(pooled)}): median={np.median(pooled):.3f}  "
          f"p{QUANTILE}={np.percentile(pooled, QUANTILE):.3f}  -> plan_vmax = {val}")

    if a.write:
        for L in LETTERS:
            path = BASE_CONFIG_DIR / f"{L}.json"
            cfg = json.load(open(path))
            cfg["plan_vmax"] = val
            json.dump(cfg, open(path, "w"), indent=2)
            print(f"  wrote plan_vmax={val} -> {path.name}")


if __name__ == "__main__":
    main()

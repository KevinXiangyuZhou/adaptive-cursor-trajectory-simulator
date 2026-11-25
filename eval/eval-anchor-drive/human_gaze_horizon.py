"""Per-participant mean human gaze lead -> the fixed planning horizon of the
R1 "fixed_horizon" ablation (fit_anchor.ABLATIONS).

The model's planning horizon is the anchor's arc-length lead over the cursor
at a planning event, so its human counterpart is the gaze lead at FIXATION
ONSET (not the time-averaged continuous lead, which sits near zero because
the cursor catches up to and passes each anchor). Events are the canonical
table the gaze constants were calibrated on (fit_gaze_pooled.build_pool:
merged keep=True fixations, round-quality pruned, lead > 3 mm, end-capped
excluded), restricted to the constant-width steering battery the evaluation
uses (bucket "steering": sinusoidal, corner, straight, gentle and sharp
sinusoidal x 10 / 12.5 / 16.5 / 25 / 50 mm).

Writes human_gaze_horizon_10p.json next to this file (committed, so cluster
snapshots need no raw gaze data):
    {"participants": {"p01": {"mean_lead_m", "median_lead_m", "n_events"}, ...},
     "pooled": {"mean_of_participant_means_m", "event_mean_m", "n_events"}, ...}

Usage: python eval/eval-anchor-drive/human_gaze_horizon.py [--letters p01 ...]
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "eval" / "eval-gaze-cursor"))
sys.path.insert(0, str(ROOT / "eval" / "eval-main"))
sys.path.insert(0, str(ROOT / "hcs_package" / "src"))
sys.path.insert(0, str(ROOT))

LETTERS = ["p01", "p02", "p03", "p04", "p06", "p07", "p08", "p10"]
OUT = HERE / "human_gaze_horizon_10p.json"


def steering_tids():
    import run_eval as em
    t2c, t2b = em.scan_conditions(ROOT / "human_data" / "task_aligned_all")
    return sorted(int(t) for t, b in t2b.items() if b == "steering")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--letters", nargs="+", default=LETTERS)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    import fit_gaze_pooled as fgp
    tids = steering_tids()
    ev, _ = fgp.build_pool(a.letters)
    ev = ev[ev["tid"].astype(int).isin(tids)]
    per = {}
    for L in a.letters:
        g = ev.loc[ev["L"] == L, "lead"].to_numpy(float)
        per[L] = {"mean_lead_m": round(float(np.mean(g)), 6),
                  "median_lead_m": round(float(np.median(g)), 6),
                  "n_events": int(len(g))}
    means = [per[L]["mean_lead_m"] for L in a.letters]
    out = {
        "description": "mean gaze lead at fixation onset (arc length, m) over the "
                       "constant-width steering battery; fixed planning horizon of "
                       "the fixed_horizon ablation",
        "events": "fit_gaze_pooled.build_pool (merged keep=True, round-quality pruned, "
                  "lead > 3 mm, end-capped excluded), steering-bucket trial ids only",
        "steering_tids": tids,
        "participants": per,
        "pooled": {"mean_of_participant_means_m": round(float(np.mean(means)), 6),
                   "event_mean_m": round(float(ev["lead"].mean()), 6),
                   "n_events": int(len(ev))},
        "generated": datetime.date.today().isoformat(),
    }
    json.dump(out, open(a.out, "w"), indent=2)
    for L in a.letters:
        print(f"  {L}: mean {per[L]['mean_lead_m'] * 1000:.1f} mm  median "
              f"{per[L]['median_lead_m'] * 1000:.1f} mm  (n={per[L]['n_events']})")
    print(f"  pooled mean of means {out['pooled']['mean_of_participant_means_m'] * 1000:.1f} mm -> {a.out}")


if __name__ == "__main__":
    main()

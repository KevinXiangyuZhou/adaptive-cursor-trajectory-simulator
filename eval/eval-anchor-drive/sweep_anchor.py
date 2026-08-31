"""Design sweep for the anchor-drive persona: does curvature braking emerge
from the effort weight, and where should the deadline sit?

Grid over jerk x plan_deadline_s (other weights fixed), quick subset
(straight / sharp sinusoidal / corner at every width) + pointing subset.
Reports, per combo: tunnel loss (train/test), CT ratio by tunnel type,
model time-to-anchor (lead/v) by type vs the human values, pointing loss.

Usage: python sweep_anchor.py --pid P170114 [--jerk 6.5e-6 2e-5 6e-5 2e-4] [--deadline 0.14 0.19 0.235]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_anchor as pa

HUMAN_TH = {"straight": 0.136, "corner": 0.178, "sharp_sinusoidal": 0.254}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", default="P170114")
    ap.add_argument("--jerk", nargs="+", type=float, default=[6.5e-6, 2e-5, 6e-5, 2e-4])
    ap.add_argument("--deadline", nargs="+", type=float, default=[0.14, 0.19, 0.235])
    ap.add_argument("--weights", default=None, help="JSON base weights (default: the persona's own contour/constraint + goal 50)")
    ap.add_argument("--fv", nargs="+", type=float, default=[0.01], help="free_velocity (damping) grid")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--no-pointing", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    base_w = json.loads(a.weights) if a.weights else {}
    rows = []
    buckets = ("tunnel",) if a.no_pointing else ("tunnel", "pointing")
    for T in a.deadline:
      for FV in a.fv:
        for J in a.jerk:
            ov = {"planner_weights": {**base_w, "jerk": J, "free_velocity": FV}, "plan_deadline_s": T}
            r = pa.run_probe(a.pid, "anchor", ov, quick=True, n_workers=a.workers, buckets=buckets, verbose=False)
            t = r["tunnel"]; bt = t["by_type"]; p = r.get("pointing") or {}
            row = {"deadline": T, "jerk": J, "fv": FV, "loss_tr": t["train"]["loss"], "loss_te": t["test"]["loss"],
                   "ctr_tr": t["train"]["ct_ratio"], "timeouts": t["train"]["timeouts"] + t["test"]["timeouts"],
                   "types": {ty: (d["ct_ratio"], d["lead_over_v"]) for ty, d in bt.items()},
                   "widths": {w: d["v_cruise"] for w, d in t["by_width"].items()},
                   "pt_loss": (p.get("train") or {}).get("loss", np.nan), "mt_ratio": (p.get("train") or {}).get("mt_ratio", np.nan),
                   "peak_v": (p.get("train") or {}).get("peak_v_ratio", np.nan)}
            rows.append(row)
            ty = row["types"]
            print(f"T={T:.3f} fv={FV:g} jerk={J:.1e} | tunnel loss {row['loss_tr']:6.2f}/{row['loss_te']:6.2f} TO {row['timeouts']} | "
                  + " ".join(f"{k[:5]} CTr={v[0]:.2f} Th={v[1]:.2f}" for k, v in ty.items())
                  + f" | v10={row['widths'].get('0.01', np.nan):.3f} v50={row['widths'].get('0.05', np.nan):.3f}"
                  + f" | pt loss {row['pt_loss']:.2f} MTr {row['mt_ratio']:.2f} pkv {row['peak_v']:.2f}", flush=True)
    print("\nhuman lead/v: straight 0.136, corner 0.178, sharp 0.254")
    if a.out:
        json.dump(rows, open(a.out, "w"), default=float, indent=1)


if __name__ == "__main__":
    main()

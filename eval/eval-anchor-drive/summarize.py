"""Compact GAM-vs-anchor comparison from fit records and GAM probe outputs.

Usage: python summarize.py [--gam-json results/gam_probe.json] [--seed 42]
The GAM probe JSON is produced by
  python probe_anchor.py --pids P105835 P170114 P160254 --personas gam --out results/gam_probe.json
"""
import argparse, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
LETTER = {"P105835": "A", "P170114": "B", "P160254": "C"}
HUMAN_TH = {"straight": 0.136, "corner": 0.178, "gentle_sinusoidal": 0.224, "mid_sinusoidal": 0.220, "sharp_sinusoidal": 0.254}


def row(name, t, p):
    tr, te = t.get("train", {}), t.get("test", {})
    ptr, pte = (p or {}).get("train", {}), (p or {}).get("test", {})
    return (f"{name:14s} tunnel loss {tr.get('loss', np.nan):6.2f}/{te.get('loss', np.nan):6.2f}  CTr {tr.get('ct_ratio', np.nan):4.2f}/{te.get('ct_ratio', np.nan):4.2f}"
            f"  lat {tr.get('lat_rmse', np.nan)*1000:4.1f}mm  TO {tr.get('timeouts', 0)}+{te.get('timeouts', 0)}"
            f" | pointing loss {ptr.get('loss', np.nan):6.2f}/{pte.get('loss', np.nan):6.2f}  MTr {ptr.get('mt_ratio', np.nan):4.2f}/{pte.get('mt_ratio', np.nan):4.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gam-json", default=str(HERE / "results" / "probes" / "gam_probe.json"))
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    gam = {}
    if Path(a.gam_json).exists():
        for r in json.load(open(a.gam_json)):
            gam[r["pid"]] = r
    print("train/test values; CTr = geometric-mean model/human completion-time ratio; MTr = kinematic movement-time ratio\n")
    for pid, L in LETTER.items():
        fp = HERE / "results" / "stages" / "base" / f"{pid}_anchor_fit_s{a.seed}.json"
        if not fp.exists():
            print(f"{pid} ({L}): no anchor fit yet"); continue
        rec = json.load(open(fp))
        print(f"=== {pid} ({L})  deadline {rec['deadline']:.3f}s  fitted {json.dumps({k: round(float(v), 6) for k, v in rec['fitted'].items()})}")
        if pid in gam:
            print("  " + row("GAM persona", gam[pid]["tunnel"], gam[pid].get("pointing")))
        print("  " + row("anchor-drive", rec["tunnel"], rec.get("pointing")))
        bw = rec["tunnel"]["by_width"]; bt = rec["tunnel"]["by_type"]
        print("  width:  " + "  ".join(f"{float(w)*1000:.0f}mm v={d['v_cruise']:.3f} lead/v={d['lead_over_v']:.2f} CTr={d['ct_ratio']:.2f}" for w, d in bw.items()))
        print("  type:   " + "  ".join(f"{ty[:8]} lead/v={d['lead_over_v']:.2f} (hum {HUMAN_TH.get(ty, float('nan')):.2f}) CTr={d['ct_ratio']:.2f}" for ty, d in bt.items()))
        if rec.get("pointing") and "fitts_model" in rec["pointing"]:
            fm, fh = rec["pointing"]["fitts_model"], rec["pointing"]["fitts_human"]
            print(f"  Fitts:  model MT = {fm[1]:.3f} + {fm[0]:.3f}·ID | human MT = {fh[1]:.3f} + {fh[0]:.3f}·ID")
        print()


if __name__ == "__main__":
    main()

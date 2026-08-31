"""Side-by-side: aug-26 GAM personas vs anchor-drive (with / without bend) on
the same simulator build and corrected task geometry, full trial sets.

Usage: python compare_aug26.py   (reads results/gam_full.json,
       results/anchor_full_bend.json, results/anchor_full_nobend.json)
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent / "results"
LETTER = {"P105835": "A", "P170114": "B", "P160254": "C"}
HUMAN_TH = {"straight": 0.136, "corner": 0.178, "gentle_sinusoidal": 0.224, "mid_sinusoidal": 0.220, "sharp_sinusoidal": 0.254}
FILES = [("aug-26 GAM", "gam_full.json"), ("anchor+bend", "anchor_full_bend.json"), ("anchor no-bend", "anchor_full_nobend.json")]


def load(fn):
    p = HERE / fn
    return {r["pid"]: r for r in json.load(open(p))} if p.exists() else {}


def main():
    data = [(name, load(fn)) for name, fn in FILES]
    for pid, L in LETTER.items():
        print(f"\n=== {pid} ({L})")
        print(f"{'model':15s} {'tunnel loss tr/te':>18s} {'CTr tr/te':>10s} {'lat mm':>7s} {'TO':>3s} | {'point loss tr/te':>17s} {'MTr tr/te':>10s} | Fitts b (human)")
        for name, d in data:
            if pid not in d: print(f"{name:15s} (missing)"); continue
            r = d[pid]; t = r["tunnel"]; p = r.get("pointing") or {}
            tr, te = t["train"], t["test"]; ptr, pte = p.get("train", {}), p.get("test", {})
            fb = p.get("fitts_model", [np.nan])[0]; fh = p.get("fitts_human", [np.nan])[0]
            print(f"{name:15s} {tr['loss']:8.2f}/{te['loss']:<8.2f} {tr['ct_ratio']:4.2f}/{te['ct_ratio']:<4.2f} {tr['lat_rmse']*1000:6.1f} {tr['timeouts']+te['timeouts']:3d} | "
                  f"{ptr.get('loss', np.nan):8.2f}/{pte.get('loss', np.nan):<8.2f} {ptr.get('mt_ratio', np.nan):4.2f}/{pte.get('mt_ratio', np.nan):<4.2f} | {fb:.3f} ({fh:.3f})")
        print("  by type — CT ratio | model lead/v (human):")
        types = ["straight", "gentle_sinusoidal", "mid_sinusoidal", "sharp_sinusoidal", "corner"]
        hdr = "  " + " " * 15 + "".join(f"{ty[:8]:>16s}" for ty in types); print(hdr)
        for name, d in data:
            if pid not in d: continue
            bt = d[pid]["tunnel"]["by_type"]
            print(f"  {name:15s}" + "".join(f"{bt[ty]['ct_ratio']:6.2f} | {bt[ty]['lead_over_v']:.2f}({HUMAN_TH[ty]:.2f})" if ty in bt else " " * 16 for ty in types))
        print("  by width — cruise speed (m/s):")
        for name, d in data:
            if pid not in d: continue
            bw = d[pid]["tunnel"]["by_width"]
            print(f"  {name:15s}" + "".join(f"  {float(w)*1000:2.0f}mm {v['v_cruise']:.3f}" for w, v in bw.items()))


if __name__ == "__main__":
    main()

"""Per-term tunnel loss + strategy statistics (cut depth / apex dip) for a list of planner
variants, all other persona settings fixed.  Companion of contour_decomp.py.

Usage: python variant_decomp.py --pid P170114 --persona results/P170114_anchor_persona_S8c.json \
          --noise off --variants '{"contour":30}' '{"anchor_spatial":1,"contour":30}' ...
"""
import argparse, json, sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import probe_anchor as pa, fit_speed_model as fsm
from contour_decomp import job, TERMS

CONDS = [("corner", 0.02), ("corner", 0.04), ("sinusoidal", 0.05), ("gentle_sinusoidal", 0.05)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pid", default="P170114"); ap.add_argument("--persona", required=True)
    ap.add_argument("--variants", nargs="+", required=True); ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--noise", default="off"); ap.add_argument("--fit-record", default=None)
    a = ap.parse_args()
    rounds, t2c, t2b = fsm.load_participant(a.pid); tasks = fsm.build_tunnel_tasks(t2c, t2b)
    base = json.load(open(a.persona)); base.pop("_description", None)
    if a.noise == "off":
        base["add_noise"] = False; base["replan_latency_cv"] = 0.0
    scales = {'lateral_rmse': 0.0055, 'speed_rmse': 0.0759, 'speed_corr': 0.8405, 'time_diff': 0.1332, 'log_time': 0.1415}
    if a.fit_record:
        scales = json.load(open(a.fit_record)).get("tunnel_scales", scales)
    tids = [t for t in tasks if t in rounds and t2b[t] == "steering"]
    hum = {}
    for ty, w in CONDS:
        tid = next((t for t in tids if abs(t2c[t]["tunnelWidth"] - w) < 1e-6 and (t2c[t].get("tunnelType") or "sinusoidal") == ty), None)
        if tid is None: continue
        import strategy_stats as ss
        from hcs_package.reference_path import ReferencePath
        tc = tasks[tid][0]; sp = ReferencePath(fsm._waypoints_m(tc), s=0.0, k=3)
        ks = np.array([abs(sp.curvature(float(x))) for x in np.linspace(0, sp.total_length, 1500)])
        hum[(ty, w)] = np.nanmean([ss.stats(h["trajectory"], h["speeds"], sp, np.percentile(ks, 85)) for h in rounds[tid]], axis=0)
    print(f"{a.pid}: {len(tids)} steering tids, noise {a.noise}, {a.seeds} seed(s); loss terms scaled by human variability (log_time x2)")
    print("human            | " + " | ".join(f"{ty[:6]}{w*1000:.0f} cut {hum[(ty,w)][0]:4.1f} dip {hum[(ty,w)][2]:.2f}" for ty, w in CONDS if (ty, w) in hum))
    print(f"{'variant':40s} | {'total':>6} {'lat':>5} {'spdR':>5} {'corr':>5} {'time':>5} | CTr  | " + " | ".join(f"{ty[:6]}{w*1000:.0f} cut/dip" for ty, w in CONDS) + " | to")
    for v in a.variants:
        ov = json.loads(v); cfg = json.loads(json.dumps(base)); cfg["budget"].update(ov.pop("budget", {}))
        for kk in [k for k in ov if k.startswith("plan_")]: cfg[kk] = ov.pop(kk)
        cfg["planner_weights"].update(ov)
        args = [(cfg, tid, tasks[tid][0], tasks[tid][1], tasks[tid][2], rounds[tid], scales, 1000 + s, t2c[tid]) for tid in tids for s in range(a.seeds)]
        with Pool(a.workers) as pool:
            rows = pool.map(job, args)
        parts = {k: np.mean([r["parts"][k] for r in rows]) for k in TERMS}; tot = sum(parts.values())
        ctr = np.exp(np.mean([np.log(r["ct_ratio"]) for r in rows]))
        cols = []
        for ty, w in CONDS:
            sel = [r for r in rows if r["ttype"] == ty and abs(r["width"] - w) < 1e-6]
            cols.append(f"{np.nanmean([r['cut'] for r in sel]):4.1f}mm {np.nanmean([r['dip'] for r in sel]):.2f}" if sel else "   n/a   ")
        print(f"{v:40s} | {tot:6.2f} {parts['lateral_rmse']:5.2f} {parts['speed_rmse']:5.2f} {parts['speed_corr']:5.2f} {parts['log_time']:5.2f} | {ctr:.2f} | " + " | ".join(cols) + f" | {sum(r['timed_out'] for r in rows)}", flush=True)


if __name__ == "__main__":
    main()

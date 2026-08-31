"""Why does the joint fit push `contour` (lateral tracking weight) stiff?  Decompose the tunnel
loss per term as a function of contour, with the fitted S8 persona otherwise unchanged, and
report the corner cut depth / apex dip alongside.

Usage: python contour_decomp.py --pid P170114 --persona results/P170114_anchor_persona_S8.json --contours 30 100 300 1000 1573
"""
import argparse, json, sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import probe_anchor as pa, fit_speed_model as fsm, strategy_stats as ss
from hcs_package.reference_path import ReferencePath

TERMS = ("lateral_rmse", "speed_rmse", "speed_corr", "log_time")


def job(args):
    cfg, tid, tc, cl, hw, rounds, scales, seed, cond = args
    fsm.TUNNEL_SCALES.update(scales)
    c = json.loads(json.dumps(cfg)); c["random_seed"] = seed
    traj, spd, dt, diag, ref = pa._sim_with_diag(c, tc)
    comp = fsm._completion(traj, cl); timed_out = len(traj) >= fsm.MAX_SIM_STEPS and comp < 0.95
    mets = [fsm.tunnel_metrics(traj, spd, h, cl, dt, hw) for h in rounds]
    W, S = fsm.TUNNEL_LOSS_WEIGHTS, fsm.TUNNEL_SCALES
    parts = {k: float(np.mean([(W[k] * (1 - m[k]) / S[k]) if k == "speed_corr" else W[k] * m[k] / S[k] for m in mets])) for k in TERMS}
    ht = float(np.mean([(h["timestamps"][-1] - h["timestamps"][0]) / 1000.0 for h in rounds]))
    cl_sp = ReferencePath(fsm._waypoints_m(tc), s=0.0, k=3)
    ks = np.array([abs(cl_sp.curvature(float(x))) for x in np.linspace(0, cl_sp.total_length, 1500)])
    cut, cut90, dip = ss.stats(traj, spd, cl_sp, np.percentile(ks, 85))
    return {"tid": tid, "timed_out": timed_out, "parts": parts, "ct_ratio": len(traj) * dt / ht,
            "width": cond["tunnelWidth"], "ttype": cond.get("tunnelType") or "sinusoidal", "cut": cut, "dip": dip}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pid", default="P170114"); ap.add_argument("--persona", required=True)
    ap.add_argument("--contours", nargs="*", type=float, default=[30, 100, 300, 1000, 1573]); ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--fit-record", default=None)
    a = ap.parse_args()
    rounds, t2c, t2b = fsm.load_participant(a.pid); tasks = fsm.build_tunnel_tasks(t2c, t2b)
    base = json.load(open(a.persona)); base.pop("_description", None)
    scales = {'lateral_rmse': 0.0055, 'speed_rmse': 0.0759, 'speed_corr': 0.8405, 'time_diff': 0.1332, 'log_time': 0.1415}
    if a.fit_record:
        rec = json.load(open(a.fit_record)); scales = rec.get("tunnel_scales", scales)
    tids = [t for t in tasks if t in rounds and t2b[t] == "steering"]
    print(f"{a.pid}: {len(tids)} tunnel tids, {a.seeds} seeds; loss terms are scaled (units of human variability), log_time weight 2")
    print(f"{'contour':>8} | {'total':>6} {'lat':>5} {'spdR':>5} {'corr':>5} {'time':>5} | CTr  | corner20 cut/dip | corner40 cut/dip | to")
    for cw in a.contours:
        cfg = json.loads(json.dumps(base)); cfg["planner_weights"]["contour"] = float(cw)
        args = [(cfg, tid, tasks[tid][0], tasks[tid][1], tasks[tid][2], rounds[tid], scales, 1000 + s, t2c[tid]) for tid in tids for s in range(a.seeds)]
        with Pool(a.workers) as pool:
            rows = pool.map(job, args)
        parts = {k: np.mean([r["parts"][k] for r in rows]) for k in TERMS}; tot = sum(parts.values())
        ctr = np.exp(np.mean([np.log(r["ct_ratio"]) for r in rows]))
        def cd(ty, w):
            sel = [r for r in rows if r["ttype"] == ty and abs(r["width"] - w) < 1e-6]
            return (np.nanmean([r["cut"] for r in sel]), np.nanmean([r["dip"] for r in sel])) if sel else (np.nan, np.nan)
        c20, c40 = cd("corner", 0.02), cd("corner", 0.04)
        print(f"{cw:8.0f} | {tot:6.2f} {parts['lateral_rmse']:5.2f} {parts['speed_rmse']:5.2f} {parts['speed_corr']:5.2f} {parts['log_time']:5.2f} | {ctr:.2f} | {c20[0]:5.1f}mm {c20[1]:.2f} | {c40[0]:5.1f}mm {c40[1]:.2f} | {sum(r['timed_out'] for r in rows)}", flush=True)


if __name__ == "__main__":
    main()

"""Steering-strategy statistics for the paper's qualitative claims: corner-cutting depth and
apex speed dip, human vs personas, by task condition.

  cut depth  = |lateral offset from the centerline| at high-curvature points (top 15% |kappa|),
               mean / p90 in mm — larger = more race-tracing
  apex dip   = min speed at high-curvature points / median trial speed — small = stop-and-go

Usage: python strategy_stats.py --pid P170114 --personas results/P170114_anchor_persona_S8.json \
           [--gam] [--runs 2] [--conds corner:0.02 corner:0.04 sinusoidal:0.05 gentle_sinusoidal:0.05]
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import probe_anchor as pa, fit_speed_model as fsm
from hcs_package.reference_path import ReferencePath


def offsets(traj, cl_sp):
    out = []
    for p in traj:
        th = cl_sp.find_closest_theta(np.array(p)); c = cl_sp(th); t = cl_sp.tangent(th); n = np.array([t[1], -t[0]])
        out.append(((np.array(p) - c) @ n, abs(cl_sp.curvature(th))))
    return np.array(out)


def stats(traj, speeds, cl_sp, kap_thr):
    o = offsets(traj, cl_sp); hi = o[:, 1] > kap_thr
    if hi.sum() < 3:
        return np.nan, np.nan, np.nan
    sp = np.asarray(speeds, float)
    dip = float(np.min(sp[hi]) / max(np.median(sp), 1e-6)) if len(sp) == len(o) else np.nan
    return float(np.mean(np.abs(o[hi, 0]))) * 1000, float(np.percentile(np.abs(o[hi, 0]), 90)) * 1000, dip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", default="P170114"); ap.add_argument("--personas", nargs="*", default=[])
    ap.add_argument("--gam", action="store_true", help="also run the aug-26 GAM persona")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--conds", nargs="*", default=["corner:0.02", "corner:0.04", "sinusoidal:0.05", "gentle_sinusoidal:0.05"])
    a = ap.parse_args()
    rounds, t2c, t2b = fsm.load_participant(a.pid); tasks = fsm.build_tunnel_tasks(t2c, t2b)
    models = []
    for pth in a.personas:
        c = json.load(open(pth)); c.pop("_description", None); models.append((Path(pth).stem.replace(f"{a.pid}_anchor_persona_", ""), c))
    if a.gam:
        g = json.load(open(ROOT / f"model_fitting-8-26-26-1/{a.pid}_gam_config_s42.json")); g["speed_model"]["path"] = os.path.abspath(ROOT / "model_fitting-8-26-26-1" / g["speed_model"]["path"]); models.append(("aug-26 GAM", g))
    print(f"{a.pid}: cut depth mean/p90 (mm) and apex dip, by condition")
    for cond in a.conds:
        ty, w = cond.split(":"); w = float(w)
        tid = next((t for t in tasks if t2b[t] == "steering" and abs(t2c[t]["tunnelWidth"] - w) < 1e-6 and (t2c[t].get("tunnelType") or "sinusoidal") == ty), None)
        if tid is None or tid not in rounds:
            print(f"  {cond}: n/a"); continue
        tc, cl, hw = tasks[tid]; cl_sp = ReferencePath(fsm._waypoints_m(tc), s=0.0, k=3)
        ks = np.array([abs(cl_sp.curvature(float(x))) for x in np.linspace(0, cl_sp.total_length, 1500)]); kap_thr = np.percentile(ks, 85)
        hm = np.nanmean([stats(h["trajectory"], h["speeds"], cl_sp, kap_thr) for h in rounds[tid]], axis=0)
        line = f"  {ty[:6]}{w*1000:3.0f}mm | human cut {hm[0]:4.1f}/{hm[1]:4.1f} dip {hm[2]:.2f}"
        for name, cfg in models:
            vals = []
            for i in range(a.runs):
                c = json.loads(json.dumps(cfg)); c["random_seed"] = 1000 + i
                traj, spd, dt, diag, ref = pa._sim_with_diag(c, tc); vals.append(stats(traj, spd, cl_sp, kap_thr))
            m = np.nanmean(vals, axis=0); line += f" | {name} cut {m[0]:4.1f}/{m[1]:4.1f} dip {m[2]:.2f}"
        print(line, flush=True)


if __name__ == "__main__":
    main()

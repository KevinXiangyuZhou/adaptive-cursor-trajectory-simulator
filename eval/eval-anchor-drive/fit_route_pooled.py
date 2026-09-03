"""Pooled Stage-1 (reference-path) fit for the new-cohort model: ONE route
parameter set (w_cut, w_width_exp, w_center, global_clearance_ref) fitted on the
POOLED steering trials of the good-quality new participants, cutmatch loss
(mean-path RMSE + |route cut - human cut| at high-curvature points), each trial
normalised implicitly by pooling means.

Usage: python fit_route_pooled.py [--pool p04 p06 p07 p09 p10] [--time-limit 1200]
Saves results/pooled10_route.json.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import probe_anchor as pa  # noqa: F401  (sets sys.path)
import fit_speed_model as fsm
import strategy_stats as ss
from hcs_package.reference_path import ReferencePath

PID = {"p01": "P103405", "p02": "P113109", "p03": "P123702", "p04": "P130409",
       "p05": "P134305", "p06": "P163215", "p07": "P170149", "p08": "P174303",
       "p09": "P190710", "p10": "P132427"}


def load_one(pid):
    rounds, t2c, t2b = fsm.load_participant(PID[pid])
    tasks = fsm.build_tunnel_tasks(t2c, t2b)
    steer = {t: r for t, r in rounds.items() if t2b.get(t) == "steering" and r}
    geometry = fsm._precompute_task_geometry({t: tasks[t][0] for t in tasks if t in steer})
    cutgeo, human_cut = {}, {}
    for tid in steer:
        sp = ReferencePath(fsm._waypoints_m(tasks[tid][0]), s=0.0, k=3)
        ks = np.array([abs(sp.curvature(float(x))) for x in np.linspace(0, sp.total_length, 1200)])
        if np.ptp(ks) < 1e-9:
            continue
        thr = float(np.percentile(ks, 85))
        if thr < 0.5:
            continue
        vals = [ss.stats(h["trajectory"], h["speeds"], sp, thr)[0] for h in steer[tid]]
        hc = float(np.nanmean(vals)) / 1000.0
        if np.isfinite(hc):
            human_cut[tid] = hc
            cutgeo[tid] = (sp, thr)
    return steer, geometry, cutgeo, human_cut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", nargs="*", default=["p04", "p06", "p07", "p09", "p10"])
    ap.add_argument("--time-limit", type=float, default=1200.0)
    ap.add_argument("--popsize", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    import cma
    data = {p: load_one(p) for p in a.pool}
    n_tr = sum(len(d[0]) for d in data.values())
    print(f"pool {a.pool}: {n_tr} steering trials "
          f"({ {p: len(d[0]) for p, d in data.items()} })", flush=True)

    def eval_vec(vec):
        rp = fsm.decode(vec, fsm.REF_PATH_PARAM_SPEC)
        total, n = 0.0, 0
        for p, (steer, geometry, cutgeo, human_cut) in data.items():
            for tid in sorted(steer):
                if tid not in geometry:
                    continue
                n += 1
                try:
                    poly, _ = fsm._build_ref_path_from_geometry(geometry[tid], rp)
                except Exception:
                    total += 1e6; continue
                lats = [np.asarray(fsm.resample_by_progress(h["trajectory"], poly, 50)[2])
                        for h in steer[tid]]
                mean_resid = np.mean(np.stack(lats), axis=0)
                term = float(np.sqrt(np.mean(mean_resid ** 2)))
                if tid in cutgeo:
                    sp, thr = cutgeo[tid]
                    o = ss.offsets(poly[::3], sp); hi = o[:, 1] > thr
                    route_cut = float(np.mean(np.abs(o[hi, 0]))) if hi.any() else 0.0
                    term += abs(route_cut - human_cut[tid])
                total += term
        return total / max(n, 1)

    init = {"w_cut": 0.3, "w_width_exp": 0.5, "w_center": 0.05, "global_clearance_ref": 0.015}
    x0 = fsm.encode(init, fsm.REF_PATH_PARAM_SPEC)
    best_loss, best_x = eval_vec(x0), np.array(x0, float)
    print(f"init loss {best_loss:.5f}", flush=True)
    es = cma.CMAEvolutionStrategy(np.asarray(x0, float).tolist(), 0.3,
                                  {"bounds": [[0.0] * len(x0), [1.0] * len(x0)], "popsize": a.popsize,
                                   "seed": a.seed, "maxiter": 300, "verb_disp": 0, "verb_log": 0,
                                   "verb_filenameprefix": "", "verbose": -9})
    t0, gen = time.time(), 0
    while not es.stop() and time.time() - t0 < a.time_limit:
        sols = es.ask(); fit = [eval_vec(x) for x in sols]; es.tell(sols, fit); gen += 1
        i = int(np.argmin(fit))
        if fit[i] < best_loss:
            best_loss, best_x = float(fit[i]), np.array(sols[i]).copy()
        if gen % 3 == 0:
            print(f"  gen {gen:3d} best {best_loss:.5f} ({time.time()-t0:.0f}s)", flush=True)
    fitted = {k: float(v) for k, v in fsm.decode(best_x, fsm.REF_PATH_PARAM_SPEC).items()}
    print(f"fitted route (pooled): {json.dumps({k: round(v, 4) for k, v in fitted.items()})}  loss {best_loss:.5f}")
    json.dump({"pool": a.pool, "fitted": fitted, "loss": best_loss, "gens": gen},
              open(HERE / "results" / "pooled10" / "pooled10_route.json", "w"), indent=2)
    print("saved results/pooled10/pooled10_route.json"); print("DONE")


if __name__ == "__main__":
    main()

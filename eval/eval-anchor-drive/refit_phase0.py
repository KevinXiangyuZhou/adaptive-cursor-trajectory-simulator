"""Phase-0 (reference-path) refit with human-variability-normalised spatial loss.

The original Phase-0 objective is the raw mean per-trial RMSE of the human lateral
offsets around the generated reference path; sinusoid trials carry the largest
magnitudes, so the fit trades the corner cut away (wide cut_window_frac -> the
exp(-w_suppress*phi) suppression fires at isolated 90-degree corners).  Here each
trial's RMSE is divided by that trial's human round-to-round spatial RMSE (the same
normalisation the tunnel/pointing losses use): trials where the participant is
consistent — corners, straights — count more; trials where rounds differ count less.

Usage: python refit_phase0.py --pid P170114 [--time-limit 900] [--seed 42]
Saves results/{pid}_refpath_cornerfair.json and prints the before/after cut table.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "eval-main"))
import probe_anchor as pa, fit_speed_model as fsm, strategy_stats as ss
from hcs_package.reference_path import ReferencePath

SCALE_FLOOR_M = 0.0015   # measurement floor: below ~1.5 mm rounds are indistinguishable


def trial_scale(rounds):
    """Human round-to-round spatial RMSE (m), floored."""
    vals = []
    for i, hi in enumerate(rounds):
        for j, hj in enumerate(rounds):
            if i == j:
                continue
            poly = np.asarray(hj["trajectory"], dtype=float)
            _, _, lat = fsm.resample_by_progress(hi["trajectory"], poly, 50)
            vals.append(float(np.sqrt(np.mean(np.asarray(lat) ** 2))))
    return max(float(np.mean(vals)) if vals else SCALE_FLOOR_M, SCALE_FLOOR_M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", default="P170114"); ap.add_argument("--time-limit", type=float, default=900.0)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--popsize", type=int, default=8)
    ap.add_argument("--loss", choices=["norm", "meanpath", "cutmatch"], default="norm",
                    help="norm: per-round RMSE / human scale; meanpath: RMSE of the mean-over-rounds "
                         "residual profile (matches the participant's mean path — the round-to-round "
                         "variance no longer pulls the route to the centerline)")
    ap.add_argument("--tag", default="cornerfair")
    ap.add_argument("--init-params", default=None, help="JSON dict of reference_path params to seed CMA-ES from (default: the S9c config's)")
    ap.add_argument("--no-scale", action="store_true", help="equal per-trial weights (raw mm) instead of dividing by the human round-to-round scale")
    a = ap.parse_args()
    import cma
    rounds, t2c, t2b = fsm.load_participant(a.pid); tasks = fsm.build_tunnel_tasks(t2c, t2b)
    tun_train, tun_test = fsm.split_tunnel(rounds, t2c, t2b)
    geometry = fsm._precompute_task_geometry({tid: tasks[tid][0] for tid in tasks})
    scales = {tid: (1.0 if a.no_scale else trial_scale(rounds[tid])) for tid in {**tun_train, **tun_test} if tid in rounds}
    # Per-trial human cut depth (mean |offset| at high-curvature centerline points) and
    # the geometry needed to score a candidate route the same way — used by --loss cutmatch.
    cutgeo, human_cut = {}, {}
    if a.loss == "cutmatch":
        for tid in {**tun_train, **tun_test}:
            if tid not in rounds or t2b.get(tid) != "steering":
                continue
            sp = ReferencePath(fsm._waypoints_m(tasks[tid][0]), s=0.0, k=3)
            ks = np.array([abs(sp.curvature(float(x))) for x in np.linspace(0, sp.total_length, 1200)])
            if np.ptp(ks) < 1e-9:
                continue   # straight: no apex, no cut term
            thr = float(np.percentile(ks, 85))
            if thr < 0.5:
                continue   # effectively straight: no apex
            vals = [ss.stats(h["trajectory"], h["speeds"], sp, thr)[0] for h in rounds[tid]]
            hc = float(np.nanmean(vals)) / 1000.0
            if not np.isfinite(hc):
                continue
            human_cut[tid] = hc
            cutgeo[tid] = (sp, thr)
    print(f"{a.pid}: {len(tun_train)} train / {len(tun_test)} test tids; per-trial human scales (mm): "
          f"{ {t: round(s*1000,1) for t, s in sorted(scales.items())} }", flush=True)

    def eval_norm(vec, data):
        rp = fsm.decode(vec, fsm.REF_PATH_PARAM_SPEC)
        total, n = 0.0, 0
        for tid in sorted(data):
            if not data[tid] or tid not in geometry:
                continue
            n += 1
            try:
                poly, _ = fsm._build_ref_path_from_geometry(geometry[tid], rp)
            except Exception:
                total += 1e6; continue
            if a.loss == "cutmatch":
                lats = [np.asarray(fsm.resample_by_progress(h["trajectory"], poly, 50)[2]) for h in data[tid]]
                mean_resid = np.mean(np.stack(lats), axis=0)
                term = float(np.sqrt(np.mean(mean_resid ** 2)))
                if tid in cutgeo:
                    sp, thr = cutgeo[tid]
                    o = ss.offsets(poly[::3], sp); hi = o[:, 1] > thr
                    route_cut = float(np.mean(np.abs(o[hi, 0]))) if hi.any() else 0.0
                    term += abs(route_cut - human_cut[tid])
                total += term
            elif a.loss == "meanpath":
                lats = [np.asarray(fsm.resample_by_progress(h["trajectory"], poly, 50)[2]) for h in data[tid]]
                mean_resid = np.mean(np.stack(lats), axis=0)
                total += float(np.sqrt(np.mean(mean_resid ** 2))) / scales[tid]
            else:
                rm = []
                for h in data[tid]:
                    _, _, lat = fsm.resample_by_progress(h["trajectory"], poly, 50)
                    rm.append(float(np.sqrt(np.mean(np.asarray(lat) ** 2))))
                total += float(np.mean(rm)) / scales[tid]
        return total / max(n, 1)

    def eval_raw(vec, data):
        return fsm._eval_ref_path_spatial(vec, data, geometry)

    base_path = HERE / "results" / "stages" / "S9c" / f"{a.pid}_anchor_config_S9c_s42.json"
    if base_path.exists():
        base = json.load(open(base_path))
    else:   # participant without an anchor config yet: start from the aug-26 GAM config's route params
        base = json.load(open(HERE.parents[1] / "model_fitting-8-26-26-1" / f"{a.pid}_gam_config_s42.json"))
    init_src = dict(json.loads(a.init_params) if a.init_params else base["reference_path"])
    init_src.setdefault("w_center", 1.0)
    init = {s["name"]: init_src[s["name"]] for s in fsm.REF_PATH_PARAM_SPEC}
    x0 = fsm.encode(init, fsm.REF_PATH_PARAM_SPEC)
    best_loss = eval_norm(x0, tun_train); best_x = np.array(x0, dtype=float)
    print(f"  initial normalised loss {best_loss:.4f}  params {json.dumps({k: round(v,4) for k,v in init.items()})}", flush=True)
    es = cma.CMAEvolutionStrategy(np.asarray(x0, dtype=float).tolist(), 0.3,
                                  {"bounds": [[0.0] * len(x0), [1.0] * len(x0)], "popsize": a.popsize,
                                   "seed": a.seed, "maxiter": 400, "verb_disp": 0, "verb_log": 0,
                                   "verb_filenameprefix": "", "verbose": -9})
    t0 = time.time(); gen = 0
    while not es.stop() and time.time() - t0 < a.time_limit:
        sols = es.ask(); fit = [eval_norm(x, tun_train) for x in sols]; es.tell(sols, fit); gen += 1
        i = int(np.argmin(fit))
        if fit[i] < best_loss:
            best_loss, best_x = float(fit[i]), np.array(sols[i]).copy()
        if gen % 5 == 0 or gen <= 3:
            print(f"  gen {gen:3d} best {best_loss:.4f} mean {np.mean(fit):.4f} ({time.time()-t0:.0f}s)", flush=True)
    fitted = fsm.decode(best_x, fsm.REF_PATH_PARAM_SPEC)
    print(f"  done: {gen} gens; fitted {json.dumps({k: round(float(v),4) for k,v in fitted.items()})}", flush=True)

    for name, vec in (("aug-26", x0), ("cornerfair", best_x)):
        print(f"  {name}: norm train {eval_norm(vec, tun_train):.4f} test {eval_norm(vec, tun_test):.4f} | "
              f"raw train {eval_raw(vec, tun_train)*1000:.2f} mm test {eval_raw(vec, tun_test)*1000:.2f} mm", flush=True)

    # cut table on the strategy conditions
    print("  ref-path cut mean/p90 (mm) at high-curvature points:", flush=True)
    for ty, w in (("corner", 0.02), ("corner", 0.04), ("corner", 0.05), ("sinusoidal", 0.05), ("gentle_sinusoidal", 0.05)):
        tid = next((t for t in tasks if t2b[t] == "steering" and abs(t2c[t]["tunnelWidth"] - w) < 1e-6
                    and (t2c[t].get("tunnelType") or "sinusoidal") == ty), None)
        if tid is None or tid not in rounds:
            continue
        sp = ReferencePath(fsm._waypoints_m(tasks[tid][0]), s=0.0, k=3)
        ks = np.array([abs(sp.curvature(float(x))) for x in np.linspace(0, sp.total_length, 1500)]); thr = np.percentile(ks, 85)
        line = f"    {ty[:6]}{w*1000:.0f}:"
        for name, vec in (("aug-26", x0), ("cornerfair", best_x)):
            poly, _ = fsm._build_ref_path_from_geometry(geometry[tid], fsm.decode(vec, fsm.REF_PATH_PARAM_SPEC))
            o = ss.offsets(poly, sp); hi = o[:, 1] > thr
            line += f"  {name} {np.mean(np.abs(o[hi,0]))*1000:4.1f}/{np.percentile(np.abs(o[hi,0]),90)*1000:4.1f}"
        hm = np.nanmean([ss.stats(h["trajectory"], h["speeds"], sp, thr) for h in rounds[tid]], axis=0)
        print(line + f"  | human {hm[0]:4.1f}/{hm[1]:4.1f}", flush=True)

    out = HERE / "results" / "refpath" / f"{a.pid}_refpath_{a.tag}.json"
    json.dump({"pid": a.pid, "fitted": {k: float(v) for k, v in fitted.items()},
               "norm_loss_train": best_loss, "seed": a.seed,
               "loss_kind": a.loss,
               "_description": "Phase-0 refit; loss=" + a.loss},
              open(out, "w"), indent=2)
    print(f"  saved {out}", flush=True)


if __name__ == "__main__":
    main()

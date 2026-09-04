"""ONE pooled anchor-drive model fitted jointly on all eight participants.

Output is a single persona: the pooled Stage-0 GAM (speed_model
gam_traversal, the shipped artifact trained on all 8 participants' samples)
plus ONE set of fitted parameters. CMA-ES searches the same five parameters
as the per-participant fit (jerk, contour, constraint, goal, D0; gamma and
plan_vmax stay pinned, T0 calibrated post-fit) against the POOLED loss

    L(theta) = mean over participants of
               [ tunnel train loss + w_pt * pointing train loss
                 + noise-on stability penalty ]

Parallelism: the work unit is (candidate x participant) — popsize 12 x 8
participants = 96 jobs per generation — so a wide node stays saturated even
though one candidate's full evaluation spans eight datasets. Each
participant's training data is loaded once in the parent and reaches the
workers by fork (copy-on-write), so jobs carry only (vec, pid).

Post-fit: pooled T0 calibration (mean pointing loss over participants per
grid point), then a held-out probe per participant with the SAME pooled
persona (per-pid train/test summaries in the fit record).

Outputs (results/stages/pooled8/):
    pooled8_anchor_config_s{seed}.json   the one persona (noise restored)
    pooled8_anchor_fit_s{seed}.json      params, history, T0 scan, per-pid probes

Usage:
  python fit_anchor_pooled8.py --time-limit 18000 --workers 36
  python fit_anchor_pooled8.py --quick --time-limit 120 --workers 8 \
      --letters p01 p02 --skip-probe          # local smoke
"""
import argparse
import copy
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_anchor as pa                 # sets sys.path / HCS_HUMAN_DATA_DIR
import fit_anchor as fa                   # per-participant loss parts + spec + T0 grid
import fit_speed_model as fsm

LETTERS = ["p01", "p02", "p03", "p04", "p06", "p07", "p08", "p10"]
RESULTS = HERE / "results"

# Per-participant training data, loaded once in the parent and inherited by
# forked workers (copy-on-write): pid -> dict(tun_train, tasks, scales,
# pt_train, pscales, stab).
_DATA = {}


def _load_all(letters, quick):
    for L in letters:
        rounds_by_tid, t2c, t2b = fsm.load_participant(L)
        tasks = fsm.build_tunnel_tasks(t2c, t2b)
        tun_train, tun_test = fsm.split_tunnel(rounds_by_tid, t2c, t2b)
        tun_train = {t: r for t, r in tun_train.items() if t2b[t] == "steering"}
        pt_train, pt_test = fsm.split_pointing(rounds_by_tid, t2c, t2b)
        if quick:
            keep = {}
            for t in sorted(tun_train):
                key = (t2c[t]["tunnelWidth"], t2c[t].get("tunnelType"))
                if key not in keep.values() and t2c[t].get("tunnelType") in (
                        "straight", "sharp_sinusoidal", "corner"):
                    keep[t] = key
            tun_train = {t: tun_train[t] for t in keep}
            pt_train = {t: r[:2] for t, r in pt_train.items()}
        fsm.compute_tunnel_scales(tun_train, tasks)
        fsm.compute_pointing_scales(pt_train)
        stab = []
        for ty, w in (("corner", 0.05), ("sinusoidal", 0.05)):
            tid = next((t for t in tun_train if abs(t2c[t]["tunnelWidth"] - w) < 1e-6
                        and (t2c[t].get("tunnelType") or "sinusoidal") == ty), None)
            if tid is None:
                continue
            tc, cl, hw = tasks[tid]
            ct_h = float(np.mean([(h["timestamps"][-1] - h["timestamps"][0]) / 1000.0
                                  for h in tun_train[tid]]))
            tc = dict(tc)
            tc["max_steps"] = int(min(fsm.MAX_SIM_STEPS, max(60, 3.0 * ct_h / 0.05)))
            stab.append((tc, cl))
        _DATA[L] = dict(tun_train=tun_train, tasks=tasks,
                        scales=dict(fsm.TUNNEL_SCALES), pt_train=pt_train,
                        pscales=dict(fsm.POINT_SCALES), stab=stab)
        print(f"  {L}: tunnel train {len(tun_train)} tids, pointing train "
              f"{len(pt_train)} tids, stability {len(stab)}", flush=True)


def _eval_unit(args):
    """One (candidate, participant) unit of the pooled loss."""
    vec, spec, base, pid, w_pt = args
    d = _DATA[pid]
    fsm.TUNNEL_SCALES.update(d["scales"])
    fsm.POINT_SCALES.update(d["pscales"])
    cfg = copy.deepcopy(base)
    fsm.apply_params(cfg, fsm.decode(np.asarray(vec), spec))
    cfg["add_noise"] = False
    cfg["replan_latency_cv"] = 0.0
    lt = fa._tunnel_part(cfg, d["tun_train"], d["tasks"])
    lp = fa._pointing_part(cfg, d["pt_train"]) if d["pt_train"] else 0.0
    ls = fa._noise_stability(cfg, d["stab"]) if d["stab"] else 0.0
    return lt + w_pt * lp + ls


def _eval_t0_unit(args):
    """One (T0 grid point, participant) unit of the pooled calibration."""
    t0, base, pid = args
    d = _DATA[pid]
    fsm.POINT_SCALES.update(d["pscales"])
    cfg = copy.deepcopy(base)
    cfg["plan_deadline_s"] = float(t0)
    cfg["add_noise"] = False
    cfg["replan_latency_cv"] = 0.0
    return float(fa._pointing_part(cfg, d["pt_train"]))


def pooled_cmaes(spec, init, base, letters, w_pt, time_limit, seed, popsize,
                 workers, sigma0=0.2):
    """CMA-ES over the pooled loss; map over (candidate x participant)."""
    import cma
    x0 = fsm.encode(init, spec)
    es = cma.CMAEvolutionStrategy(
        x0, sigma0, {"bounds": [0, 1], "seed": seed, "popsize": popsize,
                     "verbose": -9})
    ctx = mp.get_context("fork")   # workers inherit _DATA copy-on-write
    best, best_vec, hist = np.inf, x0, []
    t_start = time.time()
    gen = 0
    with ctx.Pool(processes=workers) as pool:
        while time.time() - t_start < time_limit:
            sols = es.ask()
            jobs = [(list(x), spec, base, pid, w_pt)
                    for x in sols for pid in letters]
            unit = pool.map(_eval_unit, jobs)
            # reduce: mean over participants per candidate
            fit = [float(np.mean(unit[i * len(letters):(i + 1) * len(letters)]))
                   for i in range(len(sols))]
            es.tell(sols, fit)
            gen += 1
            i = int(np.argmin(fit))
            if fit[i] < best:
                best, best_vec = fit[i], sols[i]
            hist.append({"gen": gen, "best": float(best),
                         "gen_best": float(fit[i]),
                         "elapsed": round(time.time() - t_start, 1)})
            print(f"  gen {gen}: best {best:.4f} (this gen {fit[i]:.4f}), "
                  f"{time.time() - t_start:.0f}s", flush=True)
    return fsm.decode(np.asarray(best_vec), spec), best, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=int, default=18000,
                    help="CMA budget (s); T0 calibration + probes run after it")
    ap.add_argument("--popsize", type=int, default=12)
    ap.add_argument("--workers", type=int, default=36)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--w-point", type=float, default=1.0)
    ap.add_argument("--letters", nargs="+", default=LETTERS)
    ap.add_argument("--quick", action="store_true",
                    help="straight/sharp/corner subset + 2 pointing rounds per radius")
    ap.add_argument("--skip-probe", action="store_true",
                    help="skip the per-participant held-out probes")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    # One pooled base persona: the (identical) per-pid Stage-G base config.
    base = pa.load_persona(a.letters[0], "anchor")
    base.pop("_description", None)

    print(f"pooled fit over {a.letters} | budget {a.time_limit}s | "
          f"popsize {a.popsize} x {len(a.letters)} pids = "
          f"{a.popsize * len(a.letters)} units/gen on {a.workers} workers", flush=True)
    _load_all(a.letters, a.quick)

    spec = list(fa.ANCHOR_SPEC)
    def _init_val(name):
        if name in ("plan_deadline_s", "plan_vmax"):
            return base[name]
        if name in ("D0", "gamma", "T_min"):
            return base["budget"][name]
        return base["planner_weights"][name]
    init = {s["name"]: _init_val(s["name"]) for s in spec}

    t_all = time.time()
    fitted, best, hist = pooled_cmaes(spec, init, base, a.letters, a.w_point,
                                      a.time_limit, a.seed, a.popsize, a.workers)
    fsm.apply_params(base, fitted)
    print(f"\npooled fitted: {json.dumps({k: float(v) for k, v in fitted.items()})} "
          f"| best pooled loss {best:.4f}", flush=True)

    # Pooled T0 calibration: mean pointing loss over participants per grid point.
    t0_scan = None
    pt_letters = [L for L in a.letters if _DATA[L]["pt_train"]]
    if pt_letters:
        ctx = mp.get_context("fork")
        jobs = [(t0, base, pid) for t0 in fa.T0_GRID for pid in pt_letters]
        with ctx.Pool(processes=min(a.workers, len(jobs))) as pool:
            unit = pool.map(_eval_t0_unit, jobs)
        losses = [float(np.mean(unit[i * len(pt_letters):(i + 1) * len(pt_letters)]))
                  for i in range(len(fa.T0_GRID))]
        i = int(np.argmin(losses))
        t0_scan = {"grid": fa.T0_GRID, "losses": losses, "best": fa.T0_GRID[i]}
        print(f"pooled T0 calibration: {base['plan_deadline_s']}s -> {fa.T0_GRID[i]}s "
              f"(edge={'YES' if i in (0, len(fa.T0_GRID) - 1) else 'no'})", flush=True)
        base["plan_deadline_s"] = fa.T0_GRID[i]

    stage_dir = RESULTS / "stages" / ((a.tag.strip("_") or "pooled8"))
    stage_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = stage_dir / f"pooled8_anchor_config{a.tag}_s{a.seed}.json"
    save_cfg = copy.deepcopy(base)
    save_cfg["add_noise"] = True                     # fit ran noiseless
    save_cfg["replan_latency_cv"] = 0.89
    save_cfg["_description"] = (
        f"Pooled anchor-drive persona fitted jointly on {a.letters} "
        f"(seed {a.seed}); pooled Stage-0 GAM + one parameter set. "
        f"See pooled8_anchor_fit{a.tag}_s{a.seed}.json")
    with open(cfg_path, "w") as f:
        json.dump(save_cfg, f, indent=2)
    print(f"saved {cfg_path}", flush=True)

    # Held-out probes: the SAME pooled persona against each participant.
    probes = {}
    if not a.skip_probe:
        probe_ov = {k: v for k, v in base.items()
                    if k not in ("speed_model", "reference_path", "_description")}
        for L in a.letters:
            res = pa.run_probe(L, "anchor", override=probe_ov, quick=False,
                               n_workers=a.workers)
            probes[L] = {"tunnel": res["tunnel"], "pointing": res.get("pointing")}
            print(f"  probe {L} done", flush=True)

    rec = {"letters": a.letters, "fitted": fitted, "best_loss": best,
           "history": hist, "t0_scan": t0_scan,
           "deadline": base["plan_deadline_s"], "probes": probes,
           "elapsed": time.time() - t_all}
    with open(stage_dir / f"pooled8_anchor_fit{a.tag}_s{a.seed}.json", "w") as f:
        json.dump(rec, f, indent=2, default=float)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

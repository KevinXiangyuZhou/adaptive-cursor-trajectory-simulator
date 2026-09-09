"""ONE pooled persona fitted jointly on all eight participants — for the
current model, its ablations (--ablation), or the CHI-26-EA baseline
(--model baseline). Same protocol as the per-participant fit (fit_anchor.py).

Output is a single persona: for mpcc the pooled Stage-0 GAM (speed_model
gam_traversal, the shipped artifact trained on all 8 participants' samples)
plus ONE set of fitted parameters. CMA-ES searches the same parameters as
the per-participant fit (mpcc: jerk, contour, constraint, goal, D0; gamma and
plan_vmax stay pinned, T0 calibrated post-fit; baseline: the eight EA
parameters) against the POOLED loss.
gamma defaults to the base-config value (0.66, legacy A/B/C onset-lead
exponent); --gamma overrides it before the fit. A 0.55 pin was considered
and rejected 2026-09-07: the cohort's per-saccade exponent 0.55 is biased
flat by detection-floor censoring, and the same-estimator comparison
(eval-gaze-lead/saccade_exponent_pooled.py, per-condition medians on both
sides) gives cohort b = 0.72 +/- 0.08 vs the gamma=0.66 model's emergent
0.81 +/- 0.13 — so 0.66 stays.

    L(theta) = mean over participants of
               [ tunnel train loss + w_pt * pointing train loss
                 + noise-on stability penalty ]

Parallelism: the work unit is one TRIAL of one candidate on one participant
— (candidate x participant x {tunnel condition | pointing round | stability
trial}) — ~2400 units per generation of 12 candidates, so a generation ends
after the longest single trial, not the slowest candidate. Each
participant's data are loaded once in the parent and reach the workers by
fork (copy-on-write); jobs carry only (vector, pid, kind, key).

Post-fit: pooled T0 calibration (mean pointing loss over participants per
grid point; mpcc without a jointly-fitted T0 only), then a held-out
train/test loss per participant with the SAME pooled persona, plus the
detailed anchor probe for mpcc.

Outputs ($HCS_FIT_RESULTS_DIR/stages/<tag>/, default tag pooled8 or
pooled8-<ablation>):
    pooled8_{anchor|baseline}_config_s{seed}.json   the one persona (noise restored)
    pooled8_{anchor|baseline}_fit_s{seed}.json      params, history, T0 scan, per-pid held-out

Usage:
  python fit_anchor_pooled8.py --time-limit 18000 --workers 36
  python fit_anchor_pooled8.py --model baseline --time-limit 18000 --workers 36
  python fit_anchor_pooled8.py --ablation no_pace --quick --time-limit 120 --workers 8 \
      --letters p01 p02 --skip-probe          # local smoke
"""
import argparse
import copy
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_anchor as pa                 # sets sys.path / HCS_HUMAN_DATA_DIR
import fit_anchor as fa                   # trial units + specs + setup + T0 grid
import fit_speed_model as fsm

LETTERS = ["p01", "p02", "p03", "p04", "p06", "p07", "p08", "p10"]
RESULTS = Path(os.environ.get("HCS_FIT_RESULTS_DIR", HERE / "results"))

# Per-participant training data (fit_anchor.load_training dicts) and worker
# contexts, built once in the parent and inherited by forked workers.
_DATA = {}
_CTXS = {}


def _load_all(letters, quick):
    for L in letters:
        _DATA[L] = fa.load_training(L, quick)
        d = _DATA[L]
        print(f"  {L}: tunnel train {len(d['tun_train'])} tids, pointing train "
              f"{len(d['pt_train'])} tids, stability {len(d['stab'])}", flush=True)


def _eval_pid_unit(job):
    """One (candidate, participant, trial) unit of the pooled loss."""
    vec, pid, kind, key = job
    ctx = _CTXS[pid]
    fa._CTX = ctx
    fsm.TUNNEL_SCALES.update(ctx["scales"]); fsm.POINT_SCALES.update(ctx["pscales"])
    return fa._eval_unit((vec, kind, key))


def _eval_t0_unit(args):
    """One (T0 grid point, participant) unit of the pooled calibration."""
    t0, base, pid = args
    d = _DATA[pid]
    fsm.POINT_SCALES.update(d["pscales"])
    cfg = copy.deepcopy(base)
    cfg["plan_deadline_s"] = float(t0)
    cfg["add_noise"] = False
    cfg["replan_latency_cv"] = 0.0
    return float(fa._pointing_part(cfg, d["pt_train"], "mpcc"))


def pooled_cmaes(spec, init, base, letters, w_pt, time_limit, seed, popsize,
                 workers, model, sigma0=0.2):
    """CMA-ES over the pooled loss; map over (candidate x participant x trial)."""
    import cma
    global _CTXS
    _CTXS = {L: fa.make_ctx(base, spec, _DATA[L], model) for L in letters}
    x0 = fsm.encode(init, spec)
    es = cma.CMAEvolutionStrategy(
        x0, sigma0, {"bounds": [0, 1], "seed": seed, "popsize": popsize,
                     "verbose": -9})
    ctx = mp.get_context("fork")   # workers inherit _DATA / _CTXS copy-on-write
    n_units = sum(len(fa.unit_jobs(None, _DATA[L])) for L in letters)
    print(f"  {n_units} trial units per candidate x popsize {popsize} = {n_units * popsize} units/gen "
          f"on {workers} workers", flush=True)

    def _jobs(x):
        return [(list(x), L, k, key) for L in letters for (_, k, key) in fa.unit_jobs(None, _DATA[L])]

    def _reduce(jobs, vals):
        per_pid = []
        for L in letters:
            sel = [(j, v) for j, v in zip(jobs, vals) if j[1] == L]
            f, _, _, _ = fa.reduce_units([(None, j[2], j[3]) for j, _ in sel], [v for _, v in sel], w_pt)
            per_pid.append(f)
        return float(np.mean(per_pid))

    best, best_vec, hist = np.inf, x0, []
    t_start = time.time()
    gen = 0
    with ctx.Pool(processes=workers) as pool:
        j0 = _jobs(x0)
        best = _reduce(j0, pool.map(_eval_pid_unit, j0, chunksize=1))
        print(f"  initial pooled loss {best:.4f} ({time.time() - t_start:.0f}s)", flush=True)
        hist.append({"gen": 0, "best": float(best), "elapsed": round(time.time() - t_start, 1)})
        while time.time() - t_start < time_limit:
            tg = time.time()
            sols = es.ask()
            per_cand = [_jobs(x) for x in sols]
            flat = [j for jobs in per_cand for j in jobs]
            vals = pool.map(_eval_pid_unit, flat, chunksize=1)
            fit, pos = [], 0
            for jobs in per_cand:
                fit.append(_reduce(jobs, vals[pos:pos + len(jobs)])); pos += len(jobs)
            es.tell(sols, fit)
            gen += 1
            i = int(np.argmin(fit))
            if fit[i] < best:
                best, best_vec = fit[i], sols[i]
            hist.append({"gen": gen, "best": float(best), "gen_best": float(fit[i]),
                         "gen_sec": round(time.time() - tg, 1),
                         "elapsed": round(time.time() - t_start, 1)})
            print(f"  gen {gen}: best {best:.4f} (this gen {fit[i]:.4f}), "
                  f"{time.time() - tg:.0f}s gen, {time.time() - t_start:.0f}s", flush=True)
    return fsm.decode(np.asarray(best_vec), spec), best, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=fa.MODELS, default="mpcc")
    ap.add_argument("--ablation", choices=list(fa.ABLATIONS), default="none",
                    help="(mpcc only) remove one gaze mechanism; see fit_anchor.ABLATIONS")
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
    ap.add_argument("--gamma", type=float, default=None,
                    help="pin the budget width exponent to this value instead "
                         "of the base config's (0.66); use 0.55 for the "
                         "cohort-measured saccade-distance exponent")
    ap.add_argument("--override", default=None,
                    help="JSON applied to the base persona (planner_weights/budget merged)")
    ap.add_argument("--tag", default="", help="stage folder (default pooled8 or pooled8-<ablation>)")
    a = ap.parse_args()

    # One pooled base persona: the (identical) per-pid Stage-G base config
    # (mpcc) / the EA defaults with the cohort plant constants (baseline).
    # pooled=True: the pooled traversal GAM, never a.letters[0]'s own.
    su = fa.build_setup(a.model, a.letters[0], a.ablation,
                        override=(json.loads(a.override) if a.override else None),
                        pooled=True)
    base, spec, init, model = su["base"], su["spec"], su["init"], su["model"]
    base.pop("_description", None)
    if a.gamma is not None:
        if model != "mpcc":
            raise SystemExit("--gamma applies to --model mpcc only")
        print(f"gamma override: {base['budget']['gamma']} -> {a.gamma}", flush=True)
        base["budget"]["gamma"] = float(a.gamma)
    tag = a.tag.strip("_") or ("pooled8" if su["ablation"] == "none" else f"pooled8-{su['ablation']}")

    print(f"pooled fit [{model} / {su['ablation']}] over {a.letters} | budget {a.time_limit}s | "
          f"popsize {a.popsize} on {a.workers} workers | search {[s['name'] for s in spec]}", flush=True)
    _load_all(a.letters, a.quick)

    t_all = time.time()
    fitted, best, hist = pooled_cmaes(spec, init, base, a.letters, a.w_point,
                                      a.time_limit, a.seed, a.popsize, a.workers, model)
    fsm.apply_params(base, fitted)
    print(f"\npooled fitted: {json.dumps({k: float(v) for k, v in fitted.items()})} "
          f"| best pooled loss {best:.4f}", flush=True)

    # Pooled T0 calibration: mean pointing loss over participants per grid point.
    t0_scan = None
    pt_letters = [L for L in a.letters if _DATA[L]["pt_train"]]
    if pt_letters and not su["skip_t0"]:
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

    stage_dir = RESULTS / "stages" / tag
    stage_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = stage_dir / f"pooled8_{su['tag_model']}_config_s{a.seed}.json"
    save_cfg = fa.restore_stochasticity(copy.deepcopy(base), model, a.letters[0])
    save_cfg["_description"] = (
        f"Pooled {model}/{su['ablation']} persona fitted jointly on {a.letters} "
        f"(seed {a.seed}); one parameter set. See pooled8_{su['tag_model']}_fit_s{a.seed}.json")
    save_cfg["_fit"] = {"model": model, "ablation": su["ablation"], "pooled": a.letters,
                        "seed": a.seed, "tag": tag, "search": [s["name"] for s in spec]}
    with open(cfg_path, "w") as f:
        json.dump(save_cfg, f, indent=2)
    print(f"saved {cfg_path}", flush=True)

    # Held-out: the SAME pooled persona against each participant.
    heldout, probes = {}, {}
    if not a.skip_probe:
        for L in a.letters:
            heldout[L] = fa.heldout_losses(base, _DATA[L], model, workers=a.workers, w_pt=a.w_point)
        print(f"held-out: {json.dumps(heldout)}", flush=True)
        if model == "mpcc":
            probe_ov = {k: v for k, v in base.items()
                        if k not in ("speed_model", "reference_path", "_description", "_fit")}
            for L in a.letters:
                res = pa.run_probe(L, "anchor", override=probe_ov, quick=False,
                                   n_workers=a.workers)
                probes[L] = {"tunnel": res["tunnel"], "pointing": res.get("pointing")}
                print(f"  probe {L} done", flush=True)

    rec = {"letters": a.letters, "model": model, "ablation": su["ablation"], "tag": tag,
           "seed": a.seed, "search": [s["name"] for s in spec],
           "fitted": fitted, "best_loss": best,
           "gamma_pinned": (base.get("budget") or {}).get("gamma"),
           "history": hist, "t0_scan": t0_scan,
           "caps": {"tunnel_mult": fa.TUNNEL_CAP_MULT, "tunnel_min_steps": fa.TUNNEL_CAP_MIN_STEPS,
                    "point_steps": fa.POINT_CAP_STEPS},
           "deadline": base.get("plan_deadline_s"), "heldout": heldout, "probes": probes,
           "elapsed": time.time() - t_all}
    with open(stage_dir / f"pooled8_{su['tag_model']}_fit_s{a.seed}.json", "w") as f:
        json.dump(rec, f, indent=2, default=float)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

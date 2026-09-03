"""Joint CMA-ES fit of the anchor-drive persona (one stage, one set of weights
for tunnels AND pointing). The plan deadline is the gaze-measured time-to-
anchor and is held fixed; the gaze budget (D0, gamma, T_min) and replan
latency come from the Stage G base config. CMA-ES fits the five motor/budget
parameters: jerk, contour, constraint, goal, D0. Timing parameters are all
set outside the search: gamma at the gaze-derived constant (0.66), plan_vmax
at the Stage-0 pooled pace measurement (0.66 m/s, stage0_plan_vmax.py), and
plan_deadline_s (the terminal free-space plan-time floor) by a post-fit 1-D
calibration scan on the pointing loss with the fitted persona frozen — the
free-space analog of the Stage-0 GAM. free_velocity was dropped 2026-09-03
with the MPCC damping term.

Loss = mean tunnel loss on the training widths (fit_speed_model.tunnel_loss,
human-variability scaled) + mean pointing loss on the training radii
(fit_speed_model.pointing_loss). Held-out: test widths / radii.

Usage: python fit_anchor.py --pid P105835 --time-limit 900 --popsize 12
"""
import argparse, copy, json, math, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_anchor as pa                 # sets sys.path / HCS_HUMAN_DATA_DIR
import fit_speed_model as fsm
import run_eval as em

ANCHOR_SPEC = [
    {"name": "jerk", "bounds": (-7.0, -2.0), "log_scale": True},
    {"name": "contour", "bounds": (0.0, 3.5), "log_scale": True},
    {"name": "constraint", "bounds": (1.0, 4.0), "log_scale": True},
    {"name": "goal", "bounds": (0.0, 4.0), "log_scale": True},
    # free_velocity removed (2026-09-03): |v|^2 damping dropped from the MPCC.
    # coast-safety hinge removed from the fit (S12): with walls and the pace-holding
    # plan tail it was redundant (B: tunnel loss 8.77/5.59 -> 6.74/5.54 without it).
    # peak hand acceleration is NOT fitted: fixed at 4 m/s^2 in the base config
    # (minimum-jerk peak for a ~15 cm / 0.5 s reach; above every participant's
    # observed p99 cursor acceleration: A 1.6, B 2.8, C 3.4 m/s^2; B's fitted
    # value was 3.7). Set via --override planner_weights.acc_max.
    # gaze budget quota, re-estimated on cursor data. gamma is NOT fitted
    # (2026-09-03 decision): it stays at the gaze-derived constant 0.66 from
    # the base config — the width exponent is independently measured (onset
    # lead b=0.66, saccade amplitude b~0.55, local speed ~W^1) and letting
    # CMA-ES move it traded it off against D0 on cursor loss alone.
    {"name": "D0", "bounds": (0.2, 1.5)},
    # plan_vmax is NOT fitted (2026-09-03 decision): it is a Stage-0
    # measurement — the pooled p90 of per-round pointing pace D/MT_kin
    # across all six 10p participants (eval/model_fitting/
    # stage0_plan_vmax.py -> 0.66 m/s, baked into the base configs). Like
    # gamma, it is data-derived and held constant; --vmax still overrides.
    # plan_deadline_s is NOT in the CMA search either (2026-09-03): it is
    # inert on the whole steering half of the loss (corridor plan times come
    # from the GAM), so the joint fit runs with the base prior (0.19 s) and
    # the value is CALIBRATED afterwards by a 1-D scan on the pointing loss
    # with the fitted persona frozen (T0_GRID below) — the free-space analog
    # of the Stage-0 GAM: each timing regime is fitted at its own stage
    # against the data regime it governs.
]
# Post-fit T0 calibration grid (s): terminal free-space plan-time floor.
T0_GRID = [round(0.08 + 0.01 * i, 2) for i in range(23)]   # 0.08 .. 0.30
RESULTS = HERE / "results"


def _tunnel_part(cfg, train_data, tasks):
    sim = fsm._make_sim(cfg)
    total, n = 0.0, 0
    for tid in sorted(train_data):
        rounds = train_data[tid]; tc, cl, hw = tasks[tid]; n += 1
        # Per-trial step cap: 3x the human completion time (floor 3 s). A
        # candidate that crawls is a failure either way; this stops it from
        # burning the 30 s cap on every trial (fit gen 1 took 25 min).
        ct_h = float(np.mean([(h["timestamps"][-1] - h["timestamps"][0]) / 1000.0 for h in rounds]))
        tc = dict(tc); tc["max_steps"] = int(min(fsm.MAX_SIM_STEPS, max(60, 3.0 * ct_h / 0.05)))
        try:
            traj, spd, dt = fsm.run_single_sim(sim, tc)
        except Exception:
            total += 1e6; continue
        if len(traj) < 5:   # aborted within 5 steps (breach at start-up): a failed trial, not a crash
            total += fsm.INCOMPLETE_PENALTY; continue
        comp = fsm._completion(traj, cl)
        if comp < 0.95:   # timed out or aborted (wall breach): trial failure
            total += fsm.INCOMPLETE_PENALTY * (1.0 - comp); continue
        total += float(np.mean([fsm.tunnel_loss(fsm.tunnel_metrics(traj, spd, h, cl, dt, hw)) for h in rounds]))
    return total / max(n, 1)


def _pointing_part(cfg, train_data):
    sim = fsm._make_sim(cfg)
    total, n = 0.0, 0
    for tid in sorted(train_data):
        for hp in fsm._human_pointing_profiles(train_data[tid]):
            n += 1
            try:
                # Per-trial cap: 3 s (human pointing MT <= ~1 s); a timed-out
                # trial is a failure either way.
                pt_cap = min(fsm.MAX_SIM_STEPS, 60)
                tc, _, _ = em.build_fitts_bypass_config(hp["round"], hp["R"], max_steps=pt_cap)
                traj, spd, dt = fsm.run_single_sim(sim, tc, target_radius=hp["R"])
            except Exception:
                total += 1e6; continue
            if len(traj) < 5:
                total += fsm.INCOMPLETE_PENALTY; continue
            if len(traj) >= pt_cap:
                total += fsm.INCOMPLETE_PENALTY; continue
            mp = fsm._pointing_profile(traj, spd, [i * dt for i in range(len(traj))], hp["center"], hp["R"])
            total += fsm.pointing_loss(fsm.pointing_metrics(mp, hp, hp["canonical"]))
    return total / max(n, 1)


def _noise_stability(cfg, stab):
    """Noise-on wall-breach check: a persona must survive its own motor noise.
    Runs each stability trial once with noise on (latency cv 0); an aborted or
    incomplete run scores the failure penalty. Keeps noise-off-only optima
    (soft lateral weights that breach walls under noise) out of the fit."""
    cfg_n = copy.deepcopy(cfg); cfg_n["add_noise"] = True
    cfg_n["replan_latency_cv"] = 0.0; cfg_n["random_seed"] = 777
    sim = fsm._make_sim(cfg_n)
    pen = 0.0
    for tc, cl in stab:
        try:
            traj, spd, dt = fsm.run_single_sim(sim, tc)
        except Exception:
            pen += fsm.INCOMPLETE_PENALTY; continue
        comp = fsm._completion(traj, cl) if len(traj) >= 5 else 0.0
        if comp < 0.95:
            pen += fsm.INCOMPLETE_PENALTY * (1.0 - min(comp, 1.0))
    return pen


def _eval_joint(args):
    vec, spec, base, tun_train, tasks, scales, pt_train, pscales, w_pt, stab = args
    fsm.TUNNEL_SCALES.update(scales); fsm.POINT_SCALES.update(pscales)
    cfg = copy.deepcopy(base); fsm.apply_params(cfg, fsm.decode(vec, spec))
    cfg["add_noise"] = False; cfg["replan_latency_cv"] = 0.0
    t0 = time.time()
    lt = _tunnel_part(cfg, tun_train, tasks)
    lp = _pointing_part(cfg, pt_train) if pt_train else 0.0
    ls = _noise_stability(cfg, stab) if stab else 0.0
    el = time.time() - t0
    if el > 240:
        print(f"    [slow candidate {el:.0f}s] tunnel {lt:.2f} pointing {lp:.2f} params {json.dumps({k: round(float(v), 4) for k, v in fsm.decode(vec, spec).items()})}", file=sys.stderr, flush=True)
    return lt + w_pt * lp + ls


def _eval_t0(args):
    """One T0 candidate of the post-fit calibration: pointing loss of the
    frozen fitted persona with plan_deadline_s = t0 (noiseless, like the
    CMA objective)."""
    t0, base, pt_train, pscales = args
    fsm.POINT_SCALES.update(pscales)
    cfg = copy.deepcopy(base)
    cfg["plan_deadline_s"] = float(t0)
    cfg["add_noise"] = False
    cfg["replan_latency_cv"] = 0.0
    return float(_pointing_part(cfg, pt_train))


def calibrate_t0(base, pt_train, pscales, workers):
    """Stage T0: 1-D scan of the terminal free-space plan-time floor on the
    pointing loss, everything else frozen. Simulation-based on purpose: T0's
    behavioural imprint is filtered through the replan cycle and the plant,
    so a raw data regression would bake in model error — the scan absorbs it.
    Returns (best_t0, {grid, losses})."""
    from multiprocessing import Pool
    jobs = [(t0, base, pt_train, pscales) for t0 in T0_GRID]
    with Pool(processes=max(1, min(workers, len(jobs)))) as pool:
        losses = pool.map(_eval_t0, jobs)
    i = int(np.argmin(losses))
    return T0_GRID[i], {"grid": T0_GRID, "losses": [float(x) for x in losses],
                        "best": T0_GRID[i]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", required=True)
    ap.add_argument("--time-limit", type=int, default=900)
    ap.add_argument("--popsize", type=int, default=12)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--w-point", type=float, default=1.0, help="weight of the pointing loss in the joint loss")
    ap.add_argument("--init", default=None, help="JSON dict of initial planner weights")
    ap.add_argument("--deadline", type=float, default=None, help="override plan_deadline_s (default: gaze t_cross)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--override", default=None, help="JSON applied to the base persona (top-level keys; planner_weights/budget merged)")
    ap.add_argument("--patience", type=int, default=None, help="stop when the best loss has not improved >1%% for this many generations")
    ap.add_argument("--fix-budget", action="store_true", help="keep D0 fixed at the base (gaze-calibrated) value; fit planner weights only (gamma is always fixed at the base constant)")
    ap.add_argument("--fix-deadline", action="store_true", help="skip the post-fit T0 calibration scan (keep the base persona's plan_deadline_s)")
    ap.add_argument("--quick", action="store_true", help="fit on the straight/sharp/corner subset + 2 pointing rounds per radius")
    a = ap.parse_args()
    RESULTS.mkdir(exist_ok=True, parents=True)

    base = pa.load_persona(a.pid, "anchor")
    if a.deadline is not None:
        base["plan_deadline_s"] = a.deadline
    if a.vmax is not None:
        base["plan_vmax"] = a.vmax
    if a.init:
        base["planner_weights"].update(json.loads(a.init))
    if a.override:
        for k, v in json.loads(a.override).items():
            if k in ("planner_weights", "budget"): base.setdefault(k, {}).update(v)
            else: base[k] = v
    rounds_by_tid, t2c, t2b = fsm.load_participant(a.pid)
    tasks = fsm.build_tunnel_tasks(t2c, t2b)
    tun_train, tun_test = fsm.split_tunnel(rounds_by_tid, t2c, t2b)
    tun_train = {t: r for t, r in tun_train.items() if t2b[t] == "steering"}
    pt_train, pt_test = fsm.split_pointing(rounds_by_tid, t2c, t2b)
    if a.quick:
        keep = {}
        for t in sorted(tun_train):
            key = (t2c[t]["tunnelWidth"], t2c[t].get("tunnelType"))
            if key not in keep.values() and t2c[t].get("tunnelType") in ("straight", "sharp_sinusoidal", "corner"):
                keep[t] = key
        tun_train = {t: tun_train[t] for t in keep}
        pt_train = {t: r[:2] for t, r in pt_train.items()}
    print(f"{a.pid}: tunnel train {len(tun_train)} tids, test {len(tun_test)}; pointing train {len(pt_train)} tids "
          f"({sum(len(v) for v in pt_train.values())} rounds); deadline {base['plan_deadline_s']}s", flush=True)
    fsm.compute_tunnel_scales(tun_train, tasks); fsm.compute_pointing_scales(pt_train)
    scales, pscales = dict(fsm.TUNNEL_SCALES), dict(fsm.POINT_SCALES)
    base.setdefault("plan_vmax", 0.8)
    def _init_val(name):
        if name in ("plan_deadline_s", "plan_vmax"): return base[name]
        if name in ("D0", "gamma", "T_min"): return base["budget"][name]
        return base["planner_weights"][name]
    spec = [sp for sp in ANCHOR_SPEC if not (a.fix_budget and sp["name"] == "D0")]
    init = {s["name"]: _init_val(s["name"]) for s in spec}
    # Noise-on stability trials: the widest corner/sinusoid TRAIN conditions,
    # capped at 3x the human completion time like the fit trials.
    stab = []
    for ty, w in (("corner", 0.05), ("corner", 0.03), ("sinusoidal", 0.05)):
        tid = next((t for t in tun_train if abs(t2c[t]["tunnelWidth"] - w) < 1e-6
                    and (t2c[t].get("tunnelType") or "sinusoidal") == ty), None)
        if tid is None:
            continue
        tc, cl, hw = tasks[tid]
        ct_h = float(np.mean([(h["timestamps"][-1] - h["timestamps"][0]) / 1000.0 for h in tun_train[tid]]))
        tc = dict(tc); tc["max_steps"] = int(min(fsm.MAX_SIM_STEPS, max(60, 3.0 * ct_h / 0.05)))
        stab.append((tc, cl))
    print(f"  noise-on stability trials: {len(stab)}")
    shared = (spec, base, tun_train, tasks, scales, pt_train, pscales, a.w_point, stab)
    t0 = time.time()
    fitted, best, hist = fsm.run_cmaes(f"anchor joint fit {a.pid}", spec, init, _eval_joint, shared,
                                       a.time_limit, a.seed, a.popsize, a.workers, sigma0=0.2, patience=a.patience)
    fsm.apply_params(base, fitted)
    # Stage T0: calibrate the terminal free-space plan-time floor (skipped
    # with --fix-deadline, or when there is no pointing training data). If
    # the calibrated value lands at a grid edge, treat it as a diagnostic of
    # the endgame model, not just a parameter.
    t0_scan = None
    if pt_train and not a.fix_deadline:
        best_t0, t0_scan = calibrate_t0(base, pt_train, pscales, a.workers)
        print(f"T0 calibration: {base['plan_deadline_s']}s -> {best_t0}s "
              f"(pointing loss {min(t0_scan['losses']):.3f}; "
              f"edge={'YES' if best_t0 in (T0_GRID[0], T0_GRID[-1]) else 'no'})", flush=True)
        base["plan_deadline_s"] = best_t0
    stage_dir = RESULTS / "stages" / (a.tag.strip("_") or "base"); stage_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = stage_dir / f"{a.pid}_anchor_config{a.tag}_s{a.seed}.json"
    # The fit ran noiseless/deterministic (load_persona forces add_noise off
    # and latency cv 0 for the CMA objective) — the SAVED persona must get
    # its stochasticity back or every downstream eval runs a single
    # deterministic trajectory. Restore from the raw base persona.
    save_cfg = copy.deepcopy(base)
    try:
        raw = json.load(open(pa.BASE_CONFIG_DIR / f"{a.pid}.json"))
    except FileNotFoundError:
        raw = {}
    save_cfg["add_noise"] = bool(raw.get("add_noise", True))
    save_cfg["replan_latency_cv"] = float(raw.get("replan_latency_cv", 0.89))
    with open(cfg_path, "w") as f:
        json.dump(save_cfg, f, indent=2)
    print(f"\nfitted: {json.dumps({k: float(v) for k, v in fitted.items()})}\nbest joint loss {best:.4f}; saved {cfg_path}")
    # held-out evaluation via the probe (train/test, by width/type, t_cross)
    probe_ov = {k: v for k, v in base.items() if k not in ("speed_model", "reference_path", "_description")}
    res = pa.run_probe(a.pid, "anchor", override=probe_ov,
                       quick=False, n_workers=a.workers)
    rec = {"pid": a.pid, "fitted": fitted, "best_loss": best, "history": hist, "deadline": base["plan_deadline_s"],
           "t0_scan": t0_scan,
           "tunnel": res["tunnel"], "pointing": res.get("pointing"), "elapsed": time.time() - t0,
           "scales": scales, "pscales": pscales}
    with open(stage_dir / f"{a.pid}_anchor_fit{a.tag}_s{a.seed}.json", "w") as f:
        json.dump(rec, f, indent=2, default=float)


if __name__ == "__main__":
    main()

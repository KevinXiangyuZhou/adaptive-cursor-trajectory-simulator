"""Joint CMA-ES fit of one persona (one stage, one set of weights for tunnels
AND pointing) — for the current model, its ablations, and the CHI-26-EA
baseline, under ONE protocol.

    --model mpcc      (default) the anchor-drive simulator. CMA-ES fits the
                      five motor/budget parameters jerk, contour, constraint,
                      goal, D0. gamma stays at the gaze-derived constant
                      (0.66), plan_vmax at the Stage-0 pooled pace (0.66 m/s),
                      and plan_deadline_s (free-space plan-time floor) is
                      calibrated post-fit by a 1-D scan on the pointing loss.
    --ablation NAME   (mpcc only) removes one gaze mechanism; see ABLATIONS.
    --model baseline  the CHI-26-EA MPCC (eval/chi-26-ea_baseline_pacakage):
                      fixed horizon Th, per-step replanning, tracked reference
                      velocity v_des/(1+c|kappa|). CMA-ES fits eight
                      parameters: jerk, progress, wall, contour, lag,
                      desired_speed, Th, curvature_scale. No T0 scan (there is
                      no free-space deadline).

Loss = mean tunnel loss on the training widths (fit_speed_model.tunnel_loss,
human-variability scaled) + w_pt * mean pointing loss on the training radii
(fit_speed_model.pointing_loss) + a noise-on stability penalty. Held-out:
test widths / radii. Fits run noiseless; the saved persona has noise (and
latency variability) restored.

Results go to $HCS_FIT_RESULTS_DIR (default: ./results next to this file):
    stages/<tag>/{pid}_{anchor|baseline}_config_s{seed}.json
    stages/<tag>/{pid}_{anchor|baseline}_fit_s{seed}.json
<tag> defaults to the ablation name ('base' for the full model / baseline).

Usage:
  python fit_anchor.py --pid p01 --time-limit 900 --popsize 12
  python fit_anchor.py --pid p01 --ablation no_pace --quick --time-limit 120
  python fit_anchor.py --pid p01 --model baseline --quick --time-limit 120
"""
import argparse, copy, json, math, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_anchor as pa                 # sets sys.path / HCS_HUMAN_DATA_DIR
import fit_speed_model as fsm
import run_eval as em
sys.path.insert(0, str(HERE.parents[1] / "eval"))
from utils import baseline_loader as bl   # noqa: E402

MODELS = ("mpcc", "baseline")

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
# Ablation-only search entries.
FIXED_LEAD_SPEC = {"name": "fixed_lead_m", "bounds": (0.01, 0.15)}
T0_SPEC = {"name": "plan_deadline_s", "bounds": (0.08, 0.40)}

# Ablations of the current model (paper §Evaluation). Each removes one of
# the three gaze mechanisms; parameters that the removed mechanism owned
# leave the search and the ablation's own knob(s) enter it.
ABLATIONS = {
    "none": {"override": {}, "drop": [], "add": [], "skip_t0": False},
    # C1: no gaze module at all — constant lead, constant catch-up time,
    # per-step replanning (the in-framework analogue of the EA baseline).
    "no_gaze": {"override": {"horizon_mode": "fixed_lead", "fixed_lead_m": 0.03,
                             "catchup_mode": "constant", "replan_mode": "every_step"},
                "drop": ["D0"], "add": [FIXED_LEAD_SPEC, T0_SPEC], "skip_t0": True},
    # C2: no adaptive lookahead — constant lead; pace-law catch-up and
    # arrival-triggered replanning kept.
    "no_lookahead": {"override": {"horizon_mode": "fixed_lead", "fixed_lead_m": 0.03},
                     "drop": ["D0"], "add": [FIXED_LEAD_SPEC], "skip_t0": False},
    # C3: no pace law — budget lookahead kept; catch-up time is the constant
    # free-space rule everywhere (T0 fitted jointly, no post-fit scan).
    "no_pace": {"override": {"catchup_mode": "constant"},
                "drop": [], "add": [T0_SPEC], "skip_t0": True},
    # C4: no intermittency — budget + pace law kept; replan every step.
    "no_intermittent": {"override": {"replan_mode": "every_step"},
                        "drop": [], "add": [], "skip_t0": False},
}

BASELINE_SPEC = [
    {"name": "jerk", "log_scale": True, "bounds": (-10.0, -4.0)},        # 1e-10 .. 1e-4
    {"name": "progress", "log_scale": True, "bounds": (-8.0, -4.0)},     # 1e-8 .. 1e-4
    {"name": "wall", "log_scale": True, "bounds": (1.0, 3.5)},           # 10 .. ~3162
    {"name": "contour", "log_scale": True, "bounds": (0.5, 3.5)},        # ~3 .. ~3162
    {"name": "lag", "log_scale": True, "bounds": (-3.0, 1.0)},           # 0.001 .. 10
    {"name": "desired_speed", "log_scale": True, "bounds": (-1.5, 0.1)}, # 0.03 .. 1.26 m/s
    {"name": "Th", "log_scale": False, "bounds": (0.20, 0.60), "discrete_step": 0.05},
    {"name": "curvature_scale", "log_scale": True, "bounds": (0.0, 2.0)},  # 1 .. 100
]

# Post-fit T0 calibration grid (s): terminal free-space plan-time floor.
T0_GRID = [round(0.08 + 0.01 * i, 2) for i in range(23)]   # 0.08 .. 0.30
RESULTS = Path(os.environ.get("HCS_FIT_RESULTS_DIR", HERE / "results"))

# Simulator adapters per model: (make_sim(cfg) -> sim, run_sim(sim, tc, target_radius=None)).
SIM_FNS = {
    "mpcc": (fsm._make_sim, fsm.run_single_sim),
    "baseline": (bl.make_baseline_sim, bl.run_baseline_sim),
}


def _sim_fns(model):
    return SIM_FNS[model]


# ---------------------------------------------------------------- loss parts

def _tunnel_part(cfg, train_data, tasks, model="mpcc"):
    make_sim, run_sim = _sim_fns(model)
    sim = make_sim(cfg)
    total, n = 0.0, 0
    for tid in sorted(train_data):
        rounds = train_data[tid]; tc, cl, hw = tasks[tid]; n += 1
        # Per-trial step cap: 3x the human completion time (floor 3 s). A
        # candidate that crawls is a failure either way; this stops it from
        # burning the 30 s cap on every trial (fit gen 1 took 25 min).
        ct_h = float(np.mean([(h["timestamps"][-1] - h["timestamps"][0]) / 1000.0 for h in rounds]))
        tc = dict(tc); tc["max_steps"] = int(min(fsm.MAX_SIM_STEPS, max(60, 3.0 * ct_h / 0.05)))
        try:
            traj, spd, dt = run_sim(sim, tc)
        except Exception:
            total += 1e6; continue
        if len(traj) < 5:   # aborted within 5 steps (breach at start-up): a failed trial, not a crash
            total += fsm.INCOMPLETE_PENALTY; continue
        comp = fsm._completion(traj, cl)
        if comp < 0.95:   # timed out or aborted (wall breach): trial failure
            total += fsm.INCOMPLETE_PENALTY * (1.0 - comp); continue
        total += float(np.mean([fsm.tunnel_loss(fsm.tunnel_metrics(traj, spd, h, cl, dt, hw)) for h in rounds]))
    return total / max(n, 1)


def _pointing_part(cfg, train_data, model="mpcc"):
    make_sim, run_sim = _sim_fns(model)
    sim = make_sim(cfg)
    total, n = 0.0, 0
    for tid in sorted(train_data):
        for hp in fsm._human_pointing_profiles(train_data[tid]):
            n += 1
            try:
                # Per-trial cap: 5 s (human pointing MT <= ~1.3 s); a timed-out
                # trial is a failure either way. (Raised from 3 s on 2026-09-08
                # so a slow-but-moving baseline candidate is scored on its
                # profile rather than flat-penalised; the same cap applies to
                # every model.)
                pt_cap = min(fsm.MAX_SIM_STEPS, 100)
                tc, _, _ = em.build_fitts_bypass_config(hp["round"], hp["R"], max_steps=pt_cap)
                traj, spd, dt = run_sim(sim, tc, target_radius=hp["R"])
            except Exception:
                total += 1e6; continue
            if len(traj) < 5:
                total += fsm.INCOMPLETE_PENALTY; continue
            if len(traj) >= pt_cap:
                total += fsm.INCOMPLETE_PENALTY; continue
            mp = fsm._pointing_profile(traj, spd, [i * dt for i in range(len(traj))], hp["center"], hp["R"])
            total += fsm.pointing_loss(fsm.pointing_metrics(mp, hp, hp["canonical"]))
    return total / max(n, 1)


def _noise_stability(cfg, stab, model="mpcc"):
    """Noise-on wall-breach check: a persona must survive its own motor noise.
    Runs each stability trial once with noise on (latency cv 0); an aborted or
    incomplete run scores the failure penalty. Keeps noise-off-only optima
    (soft lateral weights that breach walls under noise) out of the fit."""
    make_sim, run_sim = _sim_fns(model)
    cfg_n = copy.deepcopy(cfg); cfg_n["add_noise"] = True
    cfg_n["replan_latency_cv"] = 0.0; cfg_n["random_seed"] = 777
    sim = make_sim(cfg_n)
    pen = 0.0
    for tc, cl in stab:
        try:
            traj, spd, dt = run_sim(sim, tc)
        except Exception:
            pen += fsm.INCOMPLETE_PENALTY; continue
        comp = fsm._completion(traj, cl) if len(traj) >= 5 else 0.0
        if comp < 0.95:
            pen += fsm.INCOMPLETE_PENALTY * (1.0 - min(comp, 1.0))
    return pen


def _eval_joint(args):
    vec, spec, base, tun_train, tasks, scales, pt_train, pscales, w_pt, stab, model = args
    fsm.TUNNEL_SCALES.update(scales); fsm.POINT_SCALES.update(pscales)
    cfg = copy.deepcopy(base); fsm.apply_params(cfg, fsm.decode(vec, spec))
    cfg["add_noise"] = False; cfg["replan_latency_cv"] = 0.0
    t0 = time.time()
    lt = _tunnel_part(cfg, tun_train, tasks, model)
    lp = _pointing_part(cfg, pt_train, model) if pt_train else 0.0
    ls = _noise_stability(cfg, stab, model) if stab else 0.0
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
    return float(_pointing_part(cfg, pt_train, "mpcc"))


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


# ---------------------------------------------------------- held-out losses

def _eval_split(args):
    kind, cfg, data, tasks, scales, pscales, model = args
    fsm.TUNNEL_SCALES.update(scales); fsm.POINT_SCALES.update(pscales)
    cfg = copy.deepcopy(cfg); cfg["add_noise"] = False; cfg["replan_latency_cv"] = 0.0
    if not data:
        return None
    if kind == "tunnel":
        return float(_tunnel_part(cfg, data, tasks, model))
    return float(_pointing_part(cfg, data, model))


def heldout_losses(cfg, tun_train, tun_test, pt_train, pt_test, tasks, model, workers=4):
    """Train/test loss of a frozen persona, noiseless, with the same parts as
    the CMA objective. Works for every model (the eval pipeline produces the
    detailed metrics; this is the fit record's quick summary)."""
    from multiprocessing import Pool
    scales, pscales = dict(fsm.TUNNEL_SCALES), dict(fsm.POINT_SCALES)
    jobs = [("tunnel", cfg, tun_train, tasks, scales, pscales, model),
            ("tunnel", cfg, tun_test, tasks, scales, pscales, model),
            ("pointing", cfg, pt_train, None, scales, pscales, model),
            ("pointing", cfg, pt_test, None, scales, pscales, model)]
    with Pool(processes=max(1, min(workers, 4))) as pool:
        r = pool.map(_eval_split, jobs)
    return {"tunnel": {"train": r[0], "test": r[1]},
            "pointing": {"train": r[2], "test": r[3]}}


# --------------------------------------------------------------- model setup

def _init_val(base, name):
    if name in base:
        return base[name]
    if name in (base.get("budget") or {}):
        return base["budget"][name]
    return base["planner_weights"][name]


def build_setup(model, pid, ablation="none", *, deadline=None, vmax=None,
                init=None, override=None, fix_budget=False):
    """Base persona + CMA spec for (model, ablation). Returns a dict with
    base, spec, init, skip_t0, model, ablation, tag_model."""
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {model!r}")
    if model == "baseline":
        if ablation not in (None, "none"):
            raise ValueError("--ablation applies to --model mpcc only")
        base = bl.baseline_base_config(pid)
        base["add_noise"] = False
        spec = list(BASELINE_SPEC)
        skip_t0 = True
    else:
        if ablation not in ABLATIONS:
            raise ValueError(f"unknown ablation {ablation!r}; choose from {list(ABLATIONS)}")
        ab = ABLATIONS[ablation]
        base = pa.load_persona(pid, "anchor")
        for k, v in ab["override"].items():
            base[k] = v
        spec = [sp for sp in ANCHOR_SPEC if sp["name"] not in ab["drop"]]
        spec = [sp for sp in spec if not (fix_budget and sp["name"] == "D0")]
        spec += [sp for sp in ab["add"] if sp["name"] not in {s["name"] for s in spec}]
        skip_t0 = bool(ab["skip_t0"])
        base.setdefault("plan_vmax", 0.8)
    if deadline is not None:
        base["plan_deadline_s"] = deadline
    if vmax is not None:
        base["plan_vmax"] = vmax
    if init:
        base["planner_weights"].update(init)
    if override:
        for k, v in override.items():
            if k in ("planner_weights", "budget"): base.setdefault(k, {}).update(v)
            else: base[k] = v
    return {"model": model, "ablation": (ablation or "none"), "base": base, "spec": spec,
            "init": {s["name"]: _init_val(base, s["name"]) for s in spec},
            "skip_t0": skip_t0, "tag_model": ("anchor" if model == "mpcc" else "baseline")}


def restore_stochasticity(save_cfg, model, pid):
    """The fit ran noiseless/deterministic; the SAVED persona must get its
    stochasticity back or every downstream eval runs a single deterministic
    trajectory. Restore from the raw base persona (mpcc) / EA default (baseline)."""
    if model == "baseline":
        save_cfg["add_noise"] = True
        return save_cfg
    try:
        raw = json.load(open(pa.BASE_CONFIG_DIR / f"{pid}.json"))
    except FileNotFoundError:
        raw = {}
    save_cfg["add_noise"] = bool(raw.get("add_noise", True))
    save_cfg["replan_latency_cv"] = float(raw.get("replan_latency_cv", 0.89))
    return save_cfg


def load_training(pid, quick=False):
    rounds_by_tid, t2c, t2b = fsm.load_participant(pid)
    tasks = fsm.build_tunnel_tasks(t2c, t2b)
    tun_train, tun_test = fsm.split_tunnel(rounds_by_tid, t2c, t2b)
    tun_train = {t: r for t, r in tun_train.items() if t2b[t] == "steering"}
    pt_train, pt_test = fsm.split_pointing(rounds_by_tid, t2c, t2b)
    if quick:
        keep = {}
        for t in sorted(tun_train):
            key = (t2c[t]["tunnelWidth"], t2c[t].get("tunnelType"))
            if key not in keep.values() and t2c[t].get("tunnelType") in ("straight", "sharp_sinusoidal", "corner"):
                keep[t] = key
        tun_train = {t: tun_train[t] for t in keep}
        pt_train = {t: r[:2] for t, r in pt_train.items()}
        pt_test = {t: r[:2] for t, r in pt_test.items()}
    fsm.compute_tunnel_scales(tun_train, tasks); fsm.compute_pointing_scales(pt_train)
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
    return dict(tun_train=tun_train, tun_test=tun_test, tasks=tasks,
                pt_train=pt_train, pt_test=pt_test, stab=stab,
                scales=dict(fsm.TUNNEL_SCALES), pscales=dict(fsm.POINT_SCALES),
                t2c=t2c, t2b=t2b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", required=True)
    ap.add_argument("--model", choices=MODELS, default="mpcc")
    ap.add_argument("--ablation", choices=list(ABLATIONS), default="none",
                    help="(mpcc only) remove one gaze mechanism; see ABLATIONS")
    ap.add_argument("--time-limit", type=int, default=900)
    ap.add_argument("--popsize", type=int, default=12)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--w-point", type=float, default=1.0, help="weight of the pointing loss in the joint loss")
    ap.add_argument("--init", default=None, help="JSON dict of initial planner weights")
    ap.add_argument("--deadline", type=float, default=None, help="override plan_deadline_s (default: gaze t_cross)")
    ap.add_argument("--tag", default="", help="stage folder name (default: ablation name, 'base' for none)")
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--override", default=None, help="JSON applied to the base persona (top-level keys; planner_weights/budget merged)")
    ap.add_argument("--patience", type=int, default=None, help="stop when the best loss has not improved >1%% for this many generations")
    ap.add_argument("--fix-budget", action="store_true", help="keep D0 fixed at the base (gaze-calibrated) value; fit planner weights only (gamma is always fixed at the base constant)")
    ap.add_argument("--fix-deadline", action="store_true", help="skip the post-fit T0 calibration scan (keep the base persona's plan_deadline_s)")
    ap.add_argument("--quick", action="store_true", help="fit on the straight/sharp/corner subset + 2 pointing rounds per radius")
    ap.add_argument("--skip-probe", action="store_true", help="skip the held-out probe after the fit")
    a = ap.parse_args()
    RESULTS.mkdir(exist_ok=True, parents=True)

    su = build_setup(a.model, a.pid, a.ablation, deadline=a.deadline, vmax=a.vmax,
                     init=(json.loads(a.init) if a.init else None),
                     override=(json.loads(a.override) if a.override else None),
                     fix_budget=a.fix_budget)
    base, spec, init, model = su["base"], su["spec"], su["init"], su["model"]
    tag = (a.tag.strip("_") or (su["ablation"] if su["ablation"] != "none" else "base"))
    d = load_training(a.pid, a.quick)
    print(f"{a.pid} [{model} / {su['ablation']}] -> stages/{tag}: tunnel train {len(d['tun_train'])} tids, "
          f"test {len(d['tun_test'])}; pointing train {len(d['pt_train'])} tids "
          f"({sum(len(v) for v in d['pt_train'].values())} rounds); "
          f"search {[s['name'] for s in spec]}; deadline {base.get('plan_deadline_s')}", flush=True)
    print(f"  noise-on stability trials: {len(d['stab'])}")
    shared = (spec, base, d["tun_train"], d["tasks"], d["scales"], d["pt_train"], d["pscales"],
              a.w_point, d["stab"], model)
    t0 = time.time()
    fitted, best, hist = fsm.run_cmaes(f"{model}/{su['ablation']} joint fit {a.pid}", spec, init, _eval_joint,
                                       shared, a.time_limit, a.seed, a.popsize, a.workers,
                                       sigma0=0.2, patience=a.patience)
    fsm.apply_params(base, fitted)
    # Stage T0: calibrate the terminal free-space plan-time floor (skipped
    # with --fix-deadline, for the baseline, for ablations that fit T0
    # jointly, or when there is no pointing training data). If the
    # calibrated value lands at a grid edge, treat it as a diagnostic of the
    # endgame model, not just a parameter.
    t0_scan = None
    if d["pt_train"] and not a.fix_deadline and not su["skip_t0"]:
        best_t0, t0_scan = calibrate_t0(base, d["pt_train"], d["pscales"], a.workers)
        print(f"T0 calibration: {base['plan_deadline_s']}s -> {best_t0}s "
              f"(pointing loss {min(t0_scan['losses']):.3f}; "
              f"edge={'YES' if best_t0 in (T0_GRID[0], T0_GRID[-1]) else 'no'})", flush=True)
        base["plan_deadline_s"] = best_t0
    stage_dir = RESULTS / "stages" / tag; stage_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = stage_dir / f"{a.pid}_{su['tag_model']}_config_s{a.seed}.json"
    save_cfg = restore_stochasticity(copy.deepcopy(base), model, a.pid)
    save_cfg["_fit"] = {"model": model, "ablation": su["ablation"], "pid": a.pid, "seed": a.seed,
                        "tag": tag, "search": [s["name"] for s in spec]}
    with open(cfg_path, "w") as f:
        json.dump(save_cfg, f, indent=2)
    print(f"\nfitted: {json.dumps({k: float(v) for k, v in fitted.items()})}\nbest joint loss {best:.4f}; saved {cfg_path}")
    rec = {"pid": a.pid, "model": model, "ablation": su["ablation"], "tag": tag, "seed": a.seed,
           "search": [s["name"] for s in spec], "fitted": fitted, "best_loss": best, "history": hist,
           "deadline": base.get("plan_deadline_s"), "t0_scan": t0_scan,
           "scales": d["scales"], "pscales": d["pscales"]}
    if not a.skip_probe:
        # held-out evaluation: train/test loss of the frozen persona (all
        # models) plus the detailed anchor probe (mpcc: by width/type, t_cross)
        rec["heldout"] = heldout_losses(base, d["tun_train"], d["tun_test"], d["pt_train"], d["pt_test"],
                                        d["tasks"], model, workers=a.workers)
        print(f"held-out: {json.dumps(rec['heldout'])}", flush=True)
        if model == "mpcc":
            probe_ov = {k: v for k, v in base.items() if k not in ("speed_model", "reference_path", "_description")}
            res = pa.run_probe(a.pid, "anchor", override=probe_ov, quick=False, n_workers=a.workers)
            rec["tunnel"] = res["tunnel"]; rec["pointing"] = res.get("pointing")
    rec["elapsed"] = time.time() - t0
    with open(stage_dir / f"{a.pid}_{su['tag_model']}_fit_s{a.seed}.json", "w") as f:
        json.dump(rec, f, indent=2, default=float)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

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

Parallelism (2026-09-08): the work unit is one TRIAL of one candidate —
(candidate x tunnel condition | pointing round | stability trial) — not one
candidate, so a generation of 12 candidates is ~300 independent simulations
that spread over every core of a node and the generation ends after the
longest single trial, not the slowest candidate. Training data reach the
workers once (pool initializer); jobs carry only (vector, kind, key).
Per-trial step caps: 2x the human completion time (floor 3 s) for tunnels,
5 s for pointing — the same for every model.

Results go to $HCS_FIT_RESULTS_DIR (default: ./results next to this file):
    stages/<tag>/{pid}_{anchor|baseline}_config_s{seed}.json
    stages/<tag>/{pid}_{anchor|baseline}_fit_s{seed}.json
<tag> defaults to the ablation name ('base' for the full model / baseline).

Usage:
  python fit_anchor.py --pid p01 --time-limit 900 --popsize 12 --workers 36
  python fit_anchor.py --pid p01 --ablation no_pace --quick --time-limit 120
  python fit_anchor.py --pid p01 --model baseline --quick --time-limit 120
"""
import argparse, copy, json, math, multiprocessing, os, sys, time
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
# Per-trial step caps (identical for every model). A candidate that crawls is
# a failure either way; the cap stops it from burning the 30 s hard cap.
TUNNEL_CAP_MULT = 2.0       # x human completion time (2026-09-08: 3 -> 2)
TUNNEL_CAP_MIN_STEPS = 60   # 3 s floor
POINT_CAP_STEPS = 100       # 5 s (human pointing MT <= ~1.3 s)

# Simulator adapters per model: (make_sim(cfg) -> sim, run_sim(sim, tc, target_radius=None)).
SIM_FNS = {
    "mpcc": (fsm._make_sim, fsm.run_single_sim),
    "baseline": (bl.make_baseline_sim, bl.run_baseline_sim),
}


def _sim_fns(model):
    return SIM_FNS[model]


def _cap_tunnel_task(tc, rounds):
    ct_h = float(np.mean([(h["timestamps"][-1] - h["timestamps"][0]) / 1000.0 for h in rounds]))
    tc = dict(tc)
    tc["max_steps"] = int(min(fsm.MAX_SIM_STEPS, max(TUNNEL_CAP_MIN_STEPS, TUNNEL_CAP_MULT * ct_h / 0.05)))
    return tc


# ------------------------------------------------------------- trial units

def _tunnel_trial(cfg, rounds, tc, cl, hw, model, sim=None):
    """Loss of one tunnel condition (mean over the human rounds)."""
    make_sim, run_sim = _sim_fns(model)
    sim = sim or make_sim(cfg)
    tc = _cap_tunnel_task(tc, rounds)
    try:
        traj, spd, dt = run_sim(sim, tc)
    except Exception:
        return 1e6
    if len(traj) < 5:   # aborted within 5 steps (breach at start-up): a failed trial, not a crash
        return fsm.INCOMPLETE_PENALTY
    comp = fsm._completion(traj, cl)
    if comp < 0.95:     # timed out or aborted (wall breach): trial failure
        return fsm.INCOMPLETE_PENALTY * (1.0 - comp)
    return float(np.mean([fsm.tunnel_loss(fsm.tunnel_metrics(traj, spd, h, cl, dt, hw)) for h in rounds]))


def _pointing_trial(cfg, hp, model, sim=None):
    """Loss of one pointing round."""
    make_sim, run_sim = _sim_fns(model)
    sim = sim or make_sim(cfg)
    try:
        pt_cap = min(fsm.MAX_SIM_STEPS, POINT_CAP_STEPS)
        tc, _, _ = em.build_fitts_bypass_config(hp["round"], hp["R"], max_steps=pt_cap)
        traj, spd, dt = run_sim(sim, tc, target_radius=hp["R"])
    except Exception:
        return 1e6
    if len(traj) < 5 or len(traj) >= pt_cap:
        return fsm.INCOMPLETE_PENALTY
    mp = fsm._pointing_profile(traj, spd, [i * dt for i in range(len(traj))], hp["center"], hp["R"])
    return float(fsm.pointing_loss(fsm.pointing_metrics(mp, hp, hp["canonical"])))


def _stability_trial(cfg, tc, cl, model, sim=None):
    """Noise-on wall-breach check for one trial: a persona must survive its
    own motor noise (latency cv 0, fixed seed); an aborted or incomplete run
    scores the failure penalty. Keeps noise-off-only optima (soft lateral
    weights that breach walls under noise) out of the fit."""
    make_sim, run_sim = _sim_fns(model)
    if sim is None:
        cfg_n = copy.deepcopy(cfg); cfg_n["add_noise"] = True
        cfg_n["replan_latency_cv"] = 0.0; cfg_n["random_seed"] = 777
        sim = make_sim(cfg_n)
    try:
        traj, spd, dt = run_sim(sim, tc)
    except Exception:
        return fsm.INCOMPLETE_PENALTY
    comp = fsm._completion(traj, cl) if len(traj) >= 5 else 0.0
    return fsm.INCOMPLETE_PENALTY * (1.0 - min(comp, 1.0)) if comp < 0.95 else 0.0


# ---------------------------------------------------- sequential loss parts
# (post-fit use: T0 scan, small probes; the CMA loop uses the unit pool)

def _tunnel_part(cfg, train_data, tasks, model="mpcc"):
    sim = _sim_fns(model)[0](cfg)
    vals = [_tunnel_trial(cfg, train_data[t], *tasks[t], model, sim=sim) for t in sorted(train_data)]
    return float(np.mean(vals)) if vals else 0.0


def _pointing_part(cfg, train_data, model="mpcc"):
    sim = _sim_fns(model)[0](cfg)
    vals = [_pointing_trial(cfg, hp, model, sim=sim)
            for t in sorted(train_data) for hp in fsm._human_pointing_profiles(train_data[t])]
    return float(np.mean(vals)) if vals else 0.0


def _noise_stability(cfg, stab, model="mpcc"):
    cfg_n = copy.deepcopy(cfg); cfg_n["add_noise"] = True
    cfg_n["replan_latency_cv"] = 0.0; cfg_n["random_seed"] = 777
    sim = _sim_fns(model)[0](cfg_n)
    return float(sum(_stability_trial(cfg, tc, cl, model, sim=sim) for tc, cl in stab))


# ------------------------------------------------------- unit pool machinery

_CTX = {}


def _init_worker(ctx):
    """Pool initializer: training data reach each worker once."""
    global _CTX
    _CTX = ctx
    fsm.TUNNEL_SCALES.update(ctx["scales"]); fsm.POINT_SCALES.update(ctx["pscales"])


def _cfg_for(vec):
    if vec is None:
        return _CTX["fixed_cfg"]
    cfg = copy.deepcopy(_CTX["base"]); fsm.apply_params(cfg, fsm.decode(np.asarray(vec), _CTX["spec"]))
    cfg["add_noise"] = False; cfg["replan_latency_cv"] = 0.0
    return cfg


def _eval_unit(job):
    """One (candidate, trial) unit. job = (vec | None, kind, key)."""
    vec, kind, key = job
    cfg = _cfg_for(vec)
    model = _CTX["model"]
    if kind == "tunnel":
        rounds, tc, cl, hw = _CTX["tun"][key]
        return _tunnel_trial(cfg, rounds, tc, cl, hw, model)
    if kind == "pointing":
        return _pointing_trial(cfg, _CTX["pt"][key], model)
    tc, cl = _CTX["stab"][key]
    return _stability_trial(cfg, tc, cl, model)


def make_ctx(base, spec, data, model, fixed_cfg=None):
    """Worker context: every tunnel condition (train + test), every pointing
    round (train + test) and the stability trials, keyed for the job lists."""
    tun = {}
    for split in ("tun_train", "tun_test"):
        for t, rounds in data[split].items():
            tc, cl, hw = data["tasks"][t]
            tun[t] = (rounds, tc, cl, hw)
    pt = {}
    for split in ("pt_train", "pt_test"):
        for t, rounds in data[split].items():
            for i, hp in enumerate(fsm._human_pointing_profiles(rounds)):
                pt[(t, i)] = hp
    return {"base": base, "spec": spec, "model": model, "fixed_cfg": fixed_cfg,
            "tun": tun, "pt": pt, "stab": list(data["stab"]),
            "scales": data["scales"], "pscales": data["pscales"]}


def unit_jobs(vec, data, split="train"):
    """Job list of one candidate on one split (+ stability on train)."""
    jobs = [(vec, "tunnel", t) for t in sorted(data[f"tun_{split}"])]
    jobs += [(vec, "pointing", (t, i)) for t in sorted(data[f"pt_{split}"])
             for i in range(len(fsm._human_pointing_profiles(data[f"pt_{split}"][t])))]
    if split == "train":
        jobs += [(vec, "stab", i) for i in range(len(data["stab"]))]
    return jobs


def reduce_units(jobs, vals, w_pt):
    """(tunnel mean over conditions) + w_pt * (pointing mean over rounds) + (stability sum)."""
    tun = [v for (_, k, _), v in zip(jobs, vals) if k == "tunnel"]
    pt = [v for (_, k, _), v in zip(jobs, vals) if k == "pointing"]
    st = [v for (_, k, _), v in zip(jobs, vals) if k == "stab"]
    lt = float(np.mean(tun)) if tun else 0.0
    lp = float(np.mean(pt)) if pt else 0.0
    return lt + w_pt * lp + float(sum(st)), lt, lp, float(sum(st))


def run_cmaes_units(label, spec, init, base, data, model, w_pt, time_limit, seed,
                    popsize, workers, sigma0=0.2, patience=None, min_rel_improve=0.01):
    """CMA-ES with (candidate x trial) work units over one process pool."""
    import cma
    x0 = fsm.encode(init, spec)
    ctx = make_ctx(base, spec, data, model)
    n_units = len(unit_jobs(None, data))
    print(f"\n--- {label}: CMA-ES over {[s['name'] for s in spec]} ---", flush=True)
    print(f"  {n_units} trial units per candidate x popsize {popsize} = {n_units * popsize} units/gen "
          f"on {workers} workers; budget {time_limit:.0f}s", flush=True)
    es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, {"bounds": [[0.0] * len(x0), [1.0] * len(x0)],
                                                        "popsize": popsize, "seed": seed, "verb_disp": 0,
                                                        "verb_log": 0, "verb_filenameprefix": "", "verbose": -9})
    t0 = time.time(); gen = 0
    with multiprocessing.Pool(processes=workers, initializer=_init_worker, initargs=(ctx,)) as pool:
        j0 = unit_jobs(list(x0), data)
        initial_loss, lt, lp, ls = reduce_units(j0, pool.map(_eval_unit, j0), w_pt)
        print(f"  initial loss {initial_loss:.4f} (tunnel {lt:.3f} pointing {lp:.3f} stability {ls:.1f}; {time.time() - t0:.0f}s)", flush=True)
        best_x, best_loss = x0.copy(), initial_loss
        hist = [{"generation": 0, "best_loss": float(best_loss), "elapsed_sec": round(time.time() - t0, 1)}]
        last_improve_gen, ref_best = 0, best_loss
        while not es.stop() and time.time() - t0 < time_limit:
            if patience and gen - last_improve_gen >= patience:
                print(f"  early stop: no >{min_rel_improve*100:.0f}% improvement in {patience} generations", flush=True); break
            tg = time.time()
            sols = es.ask()
            per_cand = [unit_jobs(list(x), data) for x in sols]
            flat = [j for jobs in per_cand for j in jobs]
            vals = pool.map(_eval_unit, flat, chunksize=1)
            fit, pos = [], 0
            for jobs in per_cand:
                f, _, _, _ = reduce_units(jobs, vals[pos:pos + len(jobs)], w_pt); pos += len(jobs)
                fit.append(f)
            es.tell(sols, fit); gen += 1
            i = int(np.argmin(fit))
            if fit[i] < best_loss:
                best_loss, best_x = fit[i], np.array(sols[i]).copy()
            if best_loss < ref_best * (1.0 - min_rel_improve):
                ref_best, last_improve_gen = best_loss, gen
            hist.append({"generation": gen, "best_loss": float(best_loss), "mean_loss": float(np.mean(fit)),
                         "gen_sec": round(time.time() - tg, 1), "elapsed_sec": round(time.time() - t0, 1)})
            print(f"  gen {gen:3d} best {best_loss:.4f} mean {np.mean(fit):.4f} ({time.time() - tg:.0f}s gen, {time.time() - t0:.0f}s)", flush=True)
    fitted = fsm.decode(best_x, spec)
    print(f"  {label} done: {gen} generations, {time.time() - t0:.0f}s")
    for k, v in fitted.items():
        print(f"    {k:16s}: {init[k]:.6g} -> {v:.6g}")
    return fitted, float(best_loss), hist


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

def heldout_losses(cfg, data, model, workers=4, w_pt=1.0):
    """Train/test loss of a frozen persona, noiseless, with the same units as
    the CMA objective (stability excluded), parallel over trials. Works for
    every model (the eval pipeline produces the detailed metrics; this is the
    fit record's quick summary)."""
    cfg = copy.deepcopy(cfg); cfg["add_noise"] = False; cfg["replan_latency_cv"] = 0.0
    ctx = make_ctx(cfg, [], data, model, fixed_cfg=cfg)
    out = {"tunnel": {}, "pointing": {}}
    with multiprocessing.Pool(processes=max(1, workers), initializer=_init_worker, initargs=(ctx,)) as pool:
        for split in ("train", "test"):
            jobs = [j for j in unit_jobs(None, data, split) if j[1] != "stab"]
            vals = pool.map(_eval_unit, jobs, chunksize=1) if jobs else []
            _, lt, lp, _ = reduce_units(jobs, vals, w_pt)
            out["tunnel"][split] = lt if any(j[1] == "tunnel" for j in jobs) else None
            out["pointing"][split] = lp if any(j[1] == "pointing" for j in jobs) else None
    return out


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
    # capped like the fit trials.
    stab = []
    for ty, w in (("corner", 0.05), ("corner", 0.03), ("sinusoidal", 0.05)):
        tid = next((t for t in tun_train if abs(t2c[t]["tunnelWidth"] - w) < 1e-6
                    and (t2c[t].get("tunnelType") or "sinusoidal") == ty), None)
        if tid is None:
            continue
        tc, cl, hw = tasks[tid]
        stab.append((_cap_tunnel_task(tc, tun_train[tid]), cl))
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
          f"({sum(len(v) for v in d['pt_train'].values())} rounds); stability {len(d['stab'])}; "
          f"search {[s['name'] for s in spec]}; deadline {base.get('plan_deadline_s')}", flush=True)
    t0 = time.time()
    fitted, best, hist = run_cmaes_units(f"{model}/{su['ablation']} joint fit {a.pid}", spec, init, base, d, model,
                                         a.w_point, a.time_limit, a.seed, a.popsize, a.workers,
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
           "caps": {"tunnel_mult": TUNNEL_CAP_MULT, "tunnel_min_steps": TUNNEL_CAP_MIN_STEPS, "point_steps": POINT_CAP_STEPS},
           "scales": d["scales"], "pscales": d["pscales"]}
    if not a.skip_probe:
        # held-out evaluation: train/test loss of the frozen persona (all
        # models) plus the detailed anchor probe (mpcc: by width/type, t_cross)
        rec["heldout"] = heldout_losses(base, d, model, workers=a.workers, w_pt=a.w_point)
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

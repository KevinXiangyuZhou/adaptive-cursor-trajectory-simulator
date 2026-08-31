"""Joint CMA-ES fit of the anchor-drive persona (one stage, one set of weights
for tunnels AND pointing). The plan deadline is the gaze-measured time-to-
anchor and is held fixed; the gaze budget (D0, gamma, T_min) and replan
latency come from the Stage G base config. Fitted: jerk, contour, lag,
constraint, goal, free_velocity, plan_deadline_s (lag is inert in anchor mode).

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
    {"name": "constraint", "bounds": (1.0, 3.0), "log_scale": True},
    {"name": "goal", "bounds": (0.0, 4.0), "log_scale": True},
    {"name": "free_velocity", "bounds": (-4.0, -0.5), "log_scale": True},
    # coast-safety hinge weight (review: was hand-set)
    {"name": "safety", "bounds": (2.0, 6.0), "log_scale": True},
    # curvature-weighted gaze budget constants (re-estimated on cursor data)
    {"name": "D0", "bounds": (0.2, 1.5)},
    {"name": "gamma", "bounds": (0.3, 1.5)},
    # intended time-to-anchor (s), quantised to the 50 ms step; a 3-node
    # horizon (<0.15 s) destabilises solves, so the floor is 0.15
    {"name": "plan_deadline_s", "bounds": (0.15, 0.40), "discrete_step": 0.05},
    # max hand speed (m/s): deadline >= lookahead / v_max (pointing only in practice)
    {"name": "plan_vmax", "bounds": (0.2, 1.0)},
]
RESULTS = HERE / "results"


def _tunnel_part(cfg, train_data, tasks):
    sim = fsm._make_sim(cfg)
    total, n = 0.0, 0
    for tid in sorted(train_data):
        rounds = train_data[tid]; tc, cl, hw = tasks[tid]; n += 1
        try:
            traj, spd, dt = fsm.run_single_sim(sim, tc)
        except Exception:
            total += 1e6; continue
        if len(traj) < 5:
            total += 1e6; continue
        comp = fsm._completion(traj, cl)
        if len(traj) >= fsm.MAX_SIM_STEPS and comp < 0.95:
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
                tc, _, _ = em.build_fitts_bypass_config(hp["round"], hp["R"], max_steps=fsm.MAX_SIM_STEPS)
                traj, spd, dt = fsm.run_single_sim(sim, tc, target_radius=hp["R"])
            except Exception:
                total += 1e6; continue
            if len(traj) < 5:
                total += 1e6; continue
            if len(traj) >= fsm.MAX_SIM_STEPS:
                total += fsm.INCOMPLETE_PENALTY; continue
            mp = fsm._pointing_profile(traj, spd, [i * dt for i in range(len(traj))], hp["center"], hp["R"])
            total += fsm.pointing_loss(fsm.pointing_metrics(mp, hp, hp["canonical"]))
    return total / max(n, 1)


def _eval_joint(args):
    vec, spec, base, tun_train, tasks, scales, pt_train, pscales, w_pt = args
    fsm.TUNNEL_SCALES.update(scales); fsm.POINT_SCALES.update(pscales)
    cfg = copy.deepcopy(base); fsm.apply_params(cfg, fsm.decode(vec, spec))
    cfg["add_noise"] = False; cfg["replan_latency_cv"] = 0.0
    lt = _tunnel_part(cfg, tun_train, tasks)
    lp = _pointing_part(cfg, pt_train) if pt_train else 0.0
    return lt + w_pt * lp


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
    ap.add_argument("--fix-budget", action="store_true", help="keep D0/gamma fixed at the base (gaze-calibrated) values; fit planner weights only")
    ap.add_argument("--fix-deadline", action="store_true", help="drop plan_deadline_s from the search (gaze-measured crossing time; in BUMP mode it sets the motor pace)")
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
    base["planner_weights"].setdefault("safety", 5000.0)
    base.setdefault("plan_vmax", 0.8)
    def _init_val(name):
        if name in ("plan_deadline_s", "plan_vmax"): return base[name]
        if name in ("D0", "gamma", "T_min"): return base["budget"][name]
        return base["planner_weights"][name]
    spec = [sp for sp in ANCHOR_SPEC if not (a.fix_budget and sp["name"] in ("D0", "gamma"))
            and not (a.fix_deadline and sp["name"] == "plan_deadline_s")]
    init = {s["name"]: _init_val(s["name"]) for s in spec}
    shared = (spec, base, tun_train, tasks, scales, pt_train, pscales, a.w_point)
    t0 = time.time()
    fitted, best, hist = fsm.run_cmaes(f"anchor joint fit {a.pid}", spec, init, _eval_joint, shared,
                                       a.time_limit, a.seed, a.popsize, a.workers, sigma0=0.2, patience=a.patience)
    fsm.apply_params(base, fitted)
    cfg_path = RESULTS / f"{a.pid}_anchor_config{a.tag}_s{a.seed}.json"
    with open(cfg_path, "w") as f:
        json.dump(base, f, indent=2)
    print(f"\nfitted: {json.dumps({k: float(v) for k, v in fitted.items()})}\nbest joint loss {best:.4f}; saved {cfg_path}")
    # held-out evaluation via the probe (train/test, by width/type, t_cross)
    probe_ov = {k: v for k, v in base.items() if k not in ("speed_model", "reference_path", "_description")}
    res = pa.run_probe(a.pid, "anchor", override=probe_ov,
                       quick=False, n_workers=a.workers)
    rec = {"pid": a.pid, "fitted": fitted, "best_loss": best, "history": hist, "deadline": base["plan_deadline_s"],
           "tunnel": res["tunnel"], "pointing": res.get("pointing"), "elapsed": time.time() - t0,
           "scales": scales, "pscales": pscales}
    with open(RESULTS / f"{a.pid}_anchor_fit{a.tag}_s{a.seed}.json", "w") as f:
        json.dump(rec, f, indent=2, default=float)


if __name__ == "__main__":
    main()

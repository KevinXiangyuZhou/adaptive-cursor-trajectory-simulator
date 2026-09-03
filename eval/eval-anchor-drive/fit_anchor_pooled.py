"""POOLED joint CMA-ES fit of the anchor-drive model on the good-quality new
participants — ONE weight set for the cohort, not one per individual.

Gaze module is the 2026-09-02 redesign, constants FIXED from the pooled gaze
calibration (eval-gaze-cursor/results/pooled10_gaze_constants.json):
  gamma = 0 (width-ungraded corridor toll, zero density in free space),
  D0, T_min, turn-time (T0, tau, beta_t), replan latency median/CV.
Route (Stage 1) params are the pooled fit from results/pooled10_route.json.
Fitted here (6 motor params): jerk, contour, constraint, goal, free_velocity,
plan_vmax.  Loss = mean over pool participants of (quick-subset tunnel loss +
w_pt * pointing loss), + noise-on stability penalty (widest corner/sinusoid
train trials of two participants). Per-participant human-variability scales.

Usage: python fit_anchor_pooled.py [--time-limit 12600] [--popsize 10]
"""
import argparse, copy, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_anchor as pa                 # noqa: F401  (sets sys.path)
import fit_speed_model as fsm
import run_eval as em
from fit_anchor import _tunnel_part, _pointing_part, _noise_stability

RESULTS = HERE / "results" / "pooled10"
PID = {"p01": "P103405", "p02": "P113109", "p03": "P123702", "p04": "P130409",
       "p05": "P134305", "p06": "P163215", "p07": "P170149", "p08": "P174303",
       "p09": "P190710", "p10": "P132427"}

SPEC = [
    {"name": "jerk", "bounds": (-7.0, -2.0), "log_scale": True},
    {"name": "contour", "bounds": (0.0, 3.5), "log_scale": True},
    {"name": "constraint", "bounds": (1.0, 4.0), "log_scale": True},
    {"name": "goal", "bounds": (0.0, 4.0), "log_scale": True},
    {"name": "free_velocity", "bounds": (-4.0, -0.5), "log_scale": True},
    {"name": "plan_vmax", "bounds": (0.25, 1.0)},
]


def build_base():
    gz = json.load(open(HERE.parent / "eval-gaze-cursor" / "results" / "pooled10_gaze_constants.json"))
    rt_path = RESULTS / "pooled10_route.json"
    if rt_path.exists():
        rt = json.load(open(rt_path))
    else:
        print("WARNING: pooled10_route.json missing — using provisional route params", flush=True)
        rt = {"fitted": {"w_cut": 0.3, "w_width_exp": 0.5, "w_center": 0.05,
                          "global_clearance_ref": 0.015}}
    base = json.load(open(HERE / "results" / "stages" / "S14" / "P170114_anchor_persona_S14.json"))
    base.pop("_description", None)
    base["speed_model"] = {"type": "none"}
    base["budget"] = {"D0": gz["lead"]["D0"], "T_min": gz["lead"]["T_min"],
                       "gamma": 0.0, "W_ref": gz["lead"]["W_ref"]}
    # Width-scaled deadline (2026-09-02): T0*(W_ref/W)^beta_w + tau*theta*(W_ref/W)^beta_t.
    # beta_w = 0.5 chosen where the crossing-time MAD is near-flat (0.25-0.5)
    # because the dwell regression (b_w ~ +0.5) and the CT-by-width correction
    # both point there; T0/tau from the beta_w-conditioned pooled LAD refit.
    if "turn_time_widthscaled" in gz:
        base["plan_deadline_s"] = 0.20      # T0 0.197 quantised to the 50 ms grid
        base["plan_width_time_exp"] = 0.5
        base["plan_turn_time_s"] = round(gz["turn_time_widthscaled"]["tau"], 3)
        base["plan_turn_width_exp"] = gz["turn_time_widthscaled"]["beta_t"]
    else:
        base["plan_deadline_s"] = round(gz["turn_time"]["T0"], 2)
        base["plan_turn_time_s"] = round(gz["turn_time"]["tau"], 3)
        base["plan_turn_width_exp"] = gz["turn_time"]["beta_t"]
    base["replan_latency_s"] = round(gz["latency"]["median_s"], 3)
    base["replan_latency_cv"] = round(gz["latency"]["cv"], 2)
    base["replan_latency_max_s"] = 1.5
    base["reference_path"] = rt["fitted"]
    base["add_noise"] = False
    base["_description"] = "pooled new-cohort anchor persona (gaze redesign 2026-09-02)"
    return base


def prep_participant(pid_short, quick=True):
    rounds, t2c, t2b = fsm.load_participant(PID[pid_short])
    tasks = fsm.build_tunnel_tasks(t2c, t2b)
    tun_train, tun_test = fsm.split_tunnel(rounds, t2c, t2b)
    tun_train = {t: r for t, r in tun_train.items() if t2b[t] == "steering"}
    pt_train, pt_test = fsm.split_pointing(rounds, t2c, t2b)
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
    scales = dict(fsm.TUNNEL_SCALES); pscales = dict(fsm.POINT_SCALES)
    # per-trial step cap: 3x human completion (floor 3 s)
    capped = {}
    for tid in tun_train:
        tc, cl, hw = tasks[tid]
        ct_h = float(np.mean([(h["timestamps"][-1] - h["timestamps"][0]) / 1000.0 for h in tun_train[tid]]))
        tc = dict(tc); tc["max_steps"] = int(min(fsm.MAX_SIM_STEPS, max(60, 2.5 * ct_h / 0.05)))
        capped[tid] = (tc, cl, hw)
    return {"tun_train": tun_train, "tasks": {**tasks, **capped}, "pt_train": pt_train,
            "scales": scales, "pscales": pscales, "t2c": t2c, "t2b": t2b}


def make_stab(pdata, pids):
    stab = []
    for p in pids:
        d = pdata[p]
        cands = sorted(d["tun_train"],
                       key=lambda t: -float(d["t2c"][t]["tunnelWidth"]))
        for t in cands:
            if d["t2c"][t].get("tunnelType") in ("corner", "sharp_sinusoidal"):
                tc, cl, hw = d["tasks"][t]
                stab.append((dict(tc), cl))
                break
    return stab


def _eval_pooled(args):
    vec, spec, base, pdata, w_pt, stab = args
    cfg0 = copy.deepcopy(base)
    fsm.apply_params(cfg0, fsm.decode(vec, spec))
    cfg0["add_noise"] = False; cfg0["replan_latency_cv"] = 0.0
    total, n = 0.0, 0
    for p, d in pdata.items():
        fsm.TUNNEL_SCALES.clear(); fsm.TUNNEL_SCALES.update(d["scales"])
        fsm.POINT_SCALES.clear(); fsm.POINT_SCALES.update(d["pscales"])
        lt = _tunnel_part(cfg0, d["tun_train"], d["tasks"])
        lp = _pointing_part(cfg0, d["pt_train"]) if d["pt_train"] else 0.0
        total += lt + w_pt * lp; n += 1
    ls = _noise_stability(cfg0, stab) if stab else 0.0
    return total / max(n, 1) + ls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", nargs="*", default=["p04", "p06", "p07", "p09", "p10"])
    ap.add_argument("--time-limit", type=int, default=12600)
    ap.add_argument("--popsize", type=int, default=10)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--w-point", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--sigma0", type=float, default=0.25)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    base = build_base()
    pdata = {p: prep_participant(p) for p in a.pool}
    for p, d in pdata.items():
        print(f"{p}: tunnel quick {len(d['tun_train'])} tids, pointing {len(d['pt_train'])} tids "
              f"({sum(len(v) for v in d['pt_train'].values())} rounds)", flush=True)
    stab = make_stab(pdata, a.pool[:2])
    print(f"stability trials: {len(stab)}; deadline {base['plan_deadline_s']}s; "
          f"budget D0 {base['budget']['D0']:.3f} gamma 0; route {base['reference_path']}", flush=True)
    init = {s["name"]: (base["planner_weights"].get(s["name"]) if s["name"] != "plan_vmax"
                         else base.get("plan_vmax", 0.6)) for s in SPEC}
    shared = (SPEC, base, pdata, a.w_point, stab)
    t0 = time.time()
    fitted, best, hist = fsm.run_cmaes(f"pooled anchor fit {'+'.join(a.pool)}", SPEC, init,
                                       _eval_pooled, shared, a.time_limit, a.seed,
                                       a.popsize, a.workers, sigma0=a.sigma0, patience=a.patience)
    fsm.apply_params(base, fitted)
    cfg_path = RESULTS / f"pooled10_anchor_config{a.tag}_s{a.seed}.json"
    json.dump(base, open(cfg_path, "w"), indent=2)
    rec = {"pool": a.pool, "fitted": fitted, "best_loss": best, "history": hist,
           "elapsed": time.time() - t0}
    json.dump(rec, open(RESULTS / f"pooled10_anchor_fit{a.tag}_s{a.seed}.json", "w"), indent=2, default=float)
    print(f"\nfitted: {json.dumps({k: float(v) for k, v in fitted.items()})}\n"
          f"best pooled loss {best:.4f}; saved {cfg_path}")


if __name__ == "__main__":
    main()

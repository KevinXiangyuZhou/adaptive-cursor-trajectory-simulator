"""Sanity-check the fixed lag_anchor = 2000 on the pooled persona.

Design claim: lag_anchor is a consistency constraint (kinematic progress <->
Cartesian position); the loss should be FLAT for any sufficiently stiff value
and only break when it is soft enough to decouple progress from motion. If the
sweep instead shows a systematic trend, the weight deserves fitting.

For each value: the pooled fit loss (quick-subset tunnel + pointing per
participant, noise-off, same measure the CMA fit minimised) + a noise-on corner
probe (widest corner trial of p04: completion, cut mean mm, apex dip).

Usage: python sweep_lag_anchor.py [--values 50 200 500 2000 8000 32000]
Saves results/pooled10/lag_anchor_sweep.json.
"""
import argparse, copy, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_anchor as pa   # noqa: F401
import fit_speed_model as fsm
import strategy_stats as ss
from hcs_package.reference_path import ReferencePath
import fit_anchor_pooled as fap

RESULTS = HERE / "results" / "pooled10"
# p05/p06/p08/p09 data removed from the study (2026-09-02); sweep on the
# surviving members of the original fit pool. NOTE the persona itself was
# fitted with p06/p09 included — the sweep still answers the lag_anchor
# stiffness question, but absolute losses are not comparable to the fit's.
POOL = ["p04", "p07", "p10"]

_shared = {}


def _init():
    base = json.load(open(RESULTS / "pooled10_anchor_config_wt_s42.json"))
    base.pop("_description", None)
    pdata = {p: fap.prep_participant(p) for p in POOL}
    stab = fap.make_stab(pdata, POOL[:2])
    # noise-on corner probe geometry: widest corner train trial of p04
    d = pdata["p04"]
    tid = max((t for t in d["tun_train"] if d["t2c"][t].get("tunnelType") == "corner"),
              key=lambda t: d["t2c"][t]["tunnelWidth"])
    tc, cl, hw = d["tasks"][tid]
    sp = ReferencePath(fsm._waypoints_m(tc), s=0.0, k=3)
    ks = np.array([abs(sp.curvature(float(x))) for x in np.linspace(0, sp.total_length, 1200)])
    thr = float(np.percentile(ks, 85))
    _shared.update(base=base, pdata=pdata, stab=stab,
                   probe=(dict(tc), cl, sp, thr, d["t2c"][tid]["tunnelWidth"]))


def eval_value(v):
    base = copy.deepcopy(_shared["base"])
    base["planner_weights"]["lag_anchor"] = float(v)
    # pooled quick loss, exactly the fit's measure (noise-off + stability gate)
    vec = fsm.encode({s["name"]: (base["planner_weights"].get(s["name"])
                                    if s["name"] != "plan_vmax" else base["plan_vmax"])
                      for s in fap.SPEC}, fap.SPEC)
    loss = fap._eval_pooled((vec, fap.SPEC, base, _shared["pdata"], 1.0, _shared["stab"]))
    # noise-on corner probe
    tc, cl, sp, thr, W = _shared["probe"]
    cfg = copy.deepcopy(base); cfg["add_noise"] = True; cfg["random_seed"] = 1000
    sim = fsm._make_sim(cfg)
    try:
        traj, spd, dt = fsm.run_single_sim(sim, tc)
        comp = fsm._completion(traj, cl)
        cut, cut90, dip = ss.stats(traj, spd, sp, thr)
    except Exception:
        comp, cut, dip = 0.0, np.nan, np.nan
    return {"lag_anchor": float(v), "pooled_loss": float(loss),
            "corner_W_mm": int(W * 1000), "completion": float(comp),
            "cut_mm": float(cut), "dip": float(dip)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--values", nargs="*", type=float,
                    default=[50, 200, 500, 2000, 8000, 32000])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    # macOS spawns workers (no fork inheritance): each worker loads its own
    # data copy via the initializer.
    with Pool(min(a.workers, len(a.values)), initializer=_init) as pool:
        rows = pool.map(eval_value, a.values)
    print(f"{'lag_anchor':>10} {'pooled loss':>12} {'corner comp':>11} {'cut mm':>7} {'dip':>6}")
    for r in sorted(rows, key=lambda r: r["lag_anchor"]):
        print(f"{r['lag_anchor']:>10.0f} {r['pooled_loss']:>12.3f} {r['completion']:>11.2f} "
              f"{r['cut_mm']:>7.2f} {r['dip']:>6.2f}")
    json.dump(rows, open(RESULTS / "lag_anchor_sweep.json", "w"), indent=2)
    print("saved results/pooled10/lag_anchor_sweep.json"); print("DONE")


if __name__ == "__main__":
    main()

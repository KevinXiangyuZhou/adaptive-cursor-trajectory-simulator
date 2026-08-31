"""Anchor-drive probe: GAM persona vs. anchor-drive persona on the gaze cohort.

For each participant (A=P105835, B=P170114, C=P160254) and persona:
  * steering trials: fit-script tunnel loss (human-variability scaled) on the
    train / test widths, CT ratio, lateral RMSE, and the model's
    time-to-anchor (fixation onset -> cursor crosses the anchor) by width and
    tunnel type — the zero-tuning check against the human table
    (width-invariant ~0.19 s, rising ~1.8x from straight to sharp sinusoidal).
  * pointing trials: fit-script pointing loss and MT_kin ratio by radius.

Usage:
  python probe_anchor.py --pids P105835 [--personas gam anchor] [--quick]
        [--override '{"planner_weights": {...}, "plan_deadline_s": 0.19}']
"""
import argparse, copy, json, math, os, sys, tempfile, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HCS_HUMAN_DATA_DIR", str(ROOT / "human_data" / "gaze_cursor_data"))
os.environ.setdefault("MPLBACKEND", "Agg")
for p in (ROOT, ROOT / "hcs_package" / "src", ROOT / "eval", ROOT / "eval" / "eval-main",
          ROOT / "eval" / "model_fitting"):
    sys.path.insert(0, str(p))
import fit_speed_model as fsm            # noqa: E402
import run_eval as em                    # noqa: E402
from hcs_package.cursor_simulator import CursorSimulator  # noqa: E402

FITTED_DIR = ROOT / "model_fitting-8-26-26-1"
LETTER = {"P105835": "A", "P170114": "B", "P160254": "C"}
# Gaze-measured intended time-to-anchor (fixation onset -> anchor crossing).
T_CROSS = {"P105835": 0.235, "P170114": 0.175, "P160254": 0.175}
ANCHOR_DEFAULT = {"goal": 50.0, "free_velocity": 0.01}


def load_persona(pid, kind, override=None):
    with open(FITTED_DIR / f"{pid}_gam_config_s42.json") as f:
        cfg = json.load(f)
    cfg.pop("_description", None)
    cfg["speed_model"]["path"] = str(FITTED_DIR / cfg["speed_model"]["path"])
    cfg["add_noise"] = False
    cfg["replan_latency_cv"] = 0.0
    if kind == "anchor":
        cfg["speed_model"] = {"type": "none"}
        cfg["plan_deadline_s"] = T_CROSS[pid]
        cfg["planner_weights"].update(ANCHOR_DEFAULT)
    if override:
        for k, v in override.items():
            if k == "planner_weights":
                cfg["planner_weights"].update(v)
            else:
                cfg[k] = v
    return cfg


def _sim_with_diag(cfg, task_config, target_radius=None):
    sim = fsm._make_sim(cfg)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        json.dump(task_config, tf); path = tf.name
    try:
        traj_raw, ref = sim.generate_trajectory_with_waypoints(
            task_file=path, max_steps=task_config.get("max_steps", fsm.MAX_SIM_STEPS),
            target_radius=target_radius if target_radius is not None else task_config.get("target_radius", 0.01),
            use_optimal_path=True, return_reference_path=True)
    finally:
        os.unlink(path)
    traj = [[x * 0.001, y * 0.001] for x, y, _ in traj_raw]
    return traj, fsm._smooth_speeds(traj, sim.interval), sim.interval, sim.last_diagnostics, ref


def crossing_stats(traj, speeds, diag, ref):
    """Per replan event: lead, speed at solve, time to cross the anchor, cycle."""
    dt = diag["interval"]; ev = diag["replan_events"]
    theta = np.array([ref.find_closest_theta(np.array(p)) for p in traj])
    out = []
    for i, e in enumerate(ev):
        lead = e["anchor"] - e["theta"]
        nxt = ev[i + 1]["step"] if i + 1 < len(ev) else len(traj)
        idx = np.where(theta[e["step"]:] >= e["anchor"] - 1e-9)[0]
        t_cross = float(idx[0]) * dt if len(idx) else np.nan
        v = speeds[e["step"]] if e["step"] < len(speeds) else np.nan
        out.append({"lead": lead, "v": v, "t_cross": t_cross, "cycle": (nxt - e["step"]) * dt,
                    "trigger": e["trigger"], "th_emp": lead / v if v > 1e-3 else np.nan})
    return out


def _tunnel_job(args):
    cfg, tid, tc, cl, hw, rounds, cond = args
    fsm.TUNNEL_SCALES.update(args_scales)
    try:
        traj, spd, dt, diag, ref = _sim_with_diag(cfg, tc)
    except Exception as ex:
        return {"tid": tid, "error": repr(ex)}
    comp = fsm._completion(traj, cl)
    timed_out = comp < 0.95   # timed out or aborted (wall breach)
    mets = [fsm.tunnel_metrics(traj, spd, h, cl, dt, hw) for h in rounds]
    loss = fsm.INCOMPLETE_PENALTY * (1 - comp) if timed_out else float(np.mean([fsm.tunnel_loss(m) for m in mets]))
    ht = float(np.mean([(h["timestamps"][-1] - h["timestamps"][0]) / 1000.0 for h in rounds]))
    cs = crossing_stats(traj, spd, diag, ref)
    # cruise stats over the middle 10-90 % of the trajectory
    n = len(spd); mid = spd[int(0.1 * n):int(0.9 * n)] if n > 10 else spd
    return {"tid": tid, "width": cond["tunnelWidth"], "ttype": cond.get("tunnelType", "?"),
            "ct_model": len(traj) * dt, "ct_human": ht, "timed_out": timed_out, "completion": comp,
            "loss": loss, "lateral_rmse": float(np.mean([m["lateral_rmse"] for m in mets])),
            "speed_corr": float(np.mean([m["speed_corr"] for m in mets])),
            "v_cruise": float(np.median(mid)), "n_solves": diag["n_solves"], "cross": cs}


def _pointing_job(args):
    cfg, tid, hp = args
    fsm.POINT_SCALES.update(args_pscales)
    try:
        tc, _, _ = em.build_fitts_bypass_config(hp["round"], hp["R"], max_steps=fsm.MAX_SIM_STEPS)
        traj, spd, dt, diag, ref = _sim_with_diag(cfg, tc, target_radius=hp["R"])
    except Exception as ex:
        return {"tid": tid, "error": repr(ex)}
    timed_out = len(traj) >= fsm.MAX_SIM_STEPS
    mp = fsm._pointing_profile(traj, spd, [i * dt for i in range(len(traj))], hp["center"], hp["R"])
    m = fsm.pointing_metrics(mp, hp, hp["canonical"])
    loss = fsm.INCOMPLETE_PENALTY if timed_out else fsm.pointing_loss(m)
    D = float(np.hypot(hp["center"][0] - hp["start"][0], hp["center"][1] - hp["start"][1]))
    return {"tid": tid, "R": hp["R"], "D": D, "ID": math.log2(D / (2 * hp["R"]) + 1),
            "mt_model": mp["mt_kin"], "mt_human": hp["mt_kin"], "timed_out": timed_out, "loss": loss,
            "peak_v_model": float(np.max(spd)), "peak_v_human": float(np.max(hp["speeds"])) if len(hp["speeds"]) else np.nan,
            "end_depth_model": mp["end_depth"], "end_depth_human": hp["end_depth"], "n_solves": diag["n_solves"]}


args_scales = {}; args_pscales = {}


def _init_worker(scales, pscales):
    global args_scales, args_pscales
    args_scales = scales; args_pscales = pscales


def summarize_tunnel(rows, train_tids, test_tids):
    rows = [r for r in rows if "error" not in r]
    def agg(sel):
        sel = list(sel)
        if not sel: return {}
        return {"n": len(sel), "loss": float(np.mean([r["loss"] for r in sel])),
                "ct_ratio": float(np.exp(np.mean([np.log(r["ct_model"] / r["ct_human"]) for r in sel]))),
                "lat_rmse": float(np.mean([r["lateral_rmse"] for r in sel])),
                "spd_corr": float(np.mean([r["speed_corr"] for r in sel])),
                "timeouts": int(sum(r["timed_out"] for r in sel))}
    out = {"train": agg(r for r in rows if r["tid"] in train_tids),
           "test": agg(r for r in rows if r["tid"] in test_tids), "by_width": {}, "by_type": {}}
    for key, field in (("by_width", "width"), ("by_type", "ttype")):
        for val in sorted({r[field] for r in rows}, key=str):
            sel = [r for r in rows if r[field] == val]
            cs = [c for r in sel for c in r["cross"] if c["trigger"] != "init" and np.isfinite(c["t_cross"])]
            th = [c["th_emp"] for r in sel for c in r["cross"] if np.isfinite(c["th_emp"])]
            out[key][str(val)] = {**agg(sel),
                                  "v_cruise": float(np.median([r["v_cruise"] for r in sel])),
                                  "t_cross": float(np.median([c["t_cross"] for c in cs])) if cs else np.nan,
                                  "lead_over_v": float(np.median(th)) if th else np.nan,
                                  "cycle": float(np.median([c["cycle"] for c in cs])) if cs else np.nan}
    return out


def summarize_pointing(rows, train_tids, test_tids):
    rows = [r for r in rows if "error" not in r]
    def agg(sel):
        sel = list(sel)
        if not sel: return {}
        return {"n": len(sel), "loss": float(np.mean([r["loss"] for r in sel])),
                "mt_ratio": float(np.exp(np.mean([np.log(max(r["mt_model"], 0.05) / max(r["mt_human"], 0.05)) for r in sel]))),
                "peak_v_ratio": float(np.nanmedian([r["peak_v_model"] / r["peak_v_human"] for r in sel])),
                "timeouts": int(sum(r["timed_out"] for r in sel))}
    out = {"train": agg(r for r in rows if r["tid"] in train_tids),
           "test": agg(r for r in rows if r["tid"] in test_tids), "by_R": {}}
    for R in sorted({r["R"] for r in rows}):
        sel = [r for r in rows if r["R"] == R]
        out["by_R"][str(R)] = {**agg(sel), "mt_model": float(np.mean([r["mt_model"] for r in sel])),
                               "mt_human": float(np.mean([r["mt_human"] for r in sel]))}
    # Fitts slopes (kinematic MT vs ID)
    if len(rows) >= 3:
        ids = np.array([r["ID"] for r in rows]); out["fitts_model"] = list(map(float, np.polyfit(ids, [r["mt_model"] for r in rows], 1)))
        out["fitts_human"] = list(map(float, np.polyfit(ids, [r["mt_human"] for r in rows], 1)))
    return out


def run_probe(pid, kind, override=None, quick=False, n_workers=12, buckets=("tunnel", "pointing"), verbose=True):
    import multiprocessing as mp
    cfg = load_persona(pid, kind, override)
    rounds_by_tid, t2c, t2b = fsm.load_participant(pid)
    tasks = fsm.build_tunnel_tasks(t2c, t2b)
    tun_train, tun_test = fsm.split_tunnel(rounds_by_tid, t2c, t2b)
    pt_train, pt_test = fsm.split_pointing(rounds_by_tid, t2c, t2b)
    steer = {t: r for t, r in {**tun_train, **tun_test}.items() if t2b[t] == "steering"}
    if quick:
        keep = {}
        for t, r in sorted(steer.items()):
            key = (t2c[t]["tunnelWidth"], t2c[t].get("tunnelType"))
            if key not in keep.values() and (t2c[t].get("tunnelType") in ("straight", "sharp_sinusoidal", "corner")):
                keep[t] = key
        steer = {t: steer[t] for t in keep}
    res = {"pid": pid, "kind": kind, "cfg": cfg}
    t0 = time.time()
    with mp.Pool(n_workers, initializer=_init_worker, initargs=(dict(fsm.TUNNEL_SCALES), dict(fsm.POINT_SCALES))) as pool:
        if "tunnel" in buckets:
            fsm.compute_tunnel_scales(tun_train, tasks)
            pool._initargs = (dict(fsm.TUNNEL_SCALES), dict(fsm.POINT_SCALES))
            jobs = [(cfg, t, tasks[t][0], tasks[t][1], tasks[t][2], steer[t], t2c[t]) for t in sorted(steer)]
            # scales must reach workers: re-create pool after computing them
            pool.close(); pool.join()
            with mp.Pool(n_workers, initializer=_init_worker, initargs=(dict(fsm.TUNNEL_SCALES), dict(fsm.POINT_SCALES))) as pool2:
                rows = pool2.map(_tunnel_job, jobs)
            res["tunnel_rows"] = rows
            res["tunnel"] = summarize_tunnel(rows, set(tun_train), set(tun_test))
        if "pointing" in buckets and pt_train:
            fsm.compute_pointing_scales(pt_train)
            allpt = {**pt_train, **pt_test}
            if quick:
                allpt = {t: allpt[t][:2] for t in sorted(allpt)[:4]}
            jobs = [(cfg, t, hp) for t in sorted(allpt) for hp in fsm._human_pointing_profiles(allpt[t])]
            with mp.Pool(n_workers, initializer=_init_worker, initargs=(dict(fsm.TUNNEL_SCALES), dict(fsm.POINT_SCALES))) as pool3:
                prow = pool3.map(_pointing_job, jobs)
            res["pointing_rows"] = prow
            res["pointing"] = summarize_pointing(prow, set(pt_train), set(pt_test))
    res["elapsed"] = time.time() - t0
    if verbose:
        print_summary(res)
    return res


def print_summary(res):
    print(f"\n=== {res['pid']} ({LETTER.get(res['pid'], '?')}) persona={res['kind']}  [{res['elapsed']:.0f}s]")
    t = res.get("tunnel")
    if t:
        for split in ("train", "test"):
            a = t[split]
            if a: print(f"  tunnel {split:5s}: loss {a['loss']:.3f} | CT ratio {a['ct_ratio']:.2f} | lat RMSE {a['lat_rmse']*1000:.1f}mm | spd corr {a['spd_corr']:.2f} | timeouts {a['timeouts']}/{a['n']}")
        print("  width   loss  CTr  v_cruise  t_cross lead/v cycle")
        for w, a in t["by_width"].items():
            print(f"  {float(w)*1000:4.0f}mm {a['loss']:6.2f} {a['ct_ratio']:5.2f}  {a['v_cruise']:.3f}    {a['t_cross']:.3f}  {a['lead_over_v']:.3f}  {a['cycle']:.3f}")
        print("  type                 loss  CTr  v_cruise  t_cross lead/v")
        for ty, a in t["by_type"].items():
            print(f"  {ty:20s} {a['loss']:6.2f} {a['ct_ratio']:5.2f}  {a['v_cruise']:.3f}    {a['t_cross']:.3f}  {a['lead_over_v']:.3f}")
    p = res.get("pointing")
    if p:
        for split in ("train", "test"):
            a = p[split]
            if a: print(f"  pointing {split:5s}: loss {a['loss']:.3f} | MT ratio {a['mt_ratio']:.2f} | peak-v ratio {a['peak_v_ratio']:.2f} | timeouts {a['timeouts']}/{a['n']}")
        for R, a in p["by_R"].items():
            print(f"    R={float(R)*1000:4.1f}mm  MT model {a['mt_model']:.2f}s human {a['mt_human']:.2f}s  loss {a['loss']:.2f}")
        if "fitts_model" in p:
            print(f"    Fitts MT=a+b*ID  model b={p['fitts_model'][0]:.3f} a={p['fitts_model'][1]:.3f} | human b={p['fitts_human'][0]:.3f} a={p['fitts_human'][1]:.3f}")
    errs = [r for r in res.get("tunnel_rows", []) + res.get("pointing_rows", []) if "error" in r]
    if errs: print("  ERRORS:", errs[:3])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pids", nargs="+", default=["P105835"])
    ap.add_argument("--personas", nargs="+", default=["gam", "anchor"])
    ap.add_argument("--override", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--buckets", nargs="+", default=["tunnel", "pointing"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    ov = json.loads(a.override) if a.override else None
    allres = []
    for pid in a.pids:
        for kind in a.personas:
            allres.append(run_probe(pid, kind, ov, a.quick, a.workers, tuple(a.buckets)))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(allres, f, default=float)

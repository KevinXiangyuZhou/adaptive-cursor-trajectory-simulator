"""Module-comparison eval: {fixed|budget} horizon x {every_step|intermittent}
replanning against the gaze-cursor cohort (A/B/C).

Four variants of the same persona (office_worker + population GAM):

  baseline      fixed Th, replan every step (current model)
  budget        difficulty-budget horizon, replan every step
  intermittent  fixed Th, plan-execute-replan cycles
  full          budget horizon + plan-execute-replan cycles

Per-participant module parameters come from the gaze fits, not hand-tuning:
D0 = median difficulty budget per fixation (lookahead_summary.json
budget_constancy D_median, lam=1) and tau = median post-arrival dwell
(intermittency_summary.json post_dwell_median_s).

For every steering trial of each participant the script simulates N_SEEDS
runs per variant, then reports
  * kinematic fit vs the participant's own rounds: completion time,
    lateral RMSE, speed-profile RMSE/correlation (eval-main's
    compute_comparison_metrics), timeout fraction;
  * planning-process fit (intermittent variants): replan-cycle durations,
    anchor-overshoot fraction, and lookahead-vs-width, compared against the
    same statistics measured from gaze (intermittency_summary.json /
    horizon_summary.json);
  * compute cost: MPCC solves per simulated second.

Run:  python3 eval/eval-intermittent/run_eval.py [--n-seeds 3] [--workers 7]
      [--tids 1,2,...]   (default: all steering tids)
Results under eval/eval-intermittent/results/.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
# This repo's hcs_package must shadow any installed copy from another checkout.
sys.path.insert(0, str(PROJECT_ROOT / "eval" / "eval-main"))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "hcs_package" / "src"))

import numpy as np

import run_eval as em                    # eval-main driver (loaders, builders)
from stats import compute_comparison_metrics

RESULTS_DIR = SCRIPT_DIR / "results"
GAZE_DATA_DIR = PROJECT_ROOT / "human_data" / "gaze_cursor_data"
GAZE_RESULTS_DIR = PROJECT_ROOT / "eval" / "eval-gaze-cursor" / "results"
CONFIG_DIR = PROJECT_ROOT / "hcs_package" / "src" / "hcs_package" / "user_configurations"

# participant letter -> prolific id (human_data/gaze_cursor_data filenames)
PARTICIPANTS = {"A": "P105835", "B": "P170114", "C": "P160254"}

# Early-replan interrupt threshold (fraction of local usable width),
# calibrated so the model's share of deviation-triggered cycles matches the
# human non-crossed fraction ~0.2 (intermittency_summary.json frac_crossed
# ~0.8). Sweep on tids {1,3,5,8,33} x A/B/C x 3 seeds, full variant:
# 0.08 -> 0.50, 0.12 -> 0.29, 0.15 -> 0.18, 0.2 -> 0.10 deviation share.
DEVIATION_FRAC = 0.15


def load_gaze_fit():
    """Per-participant module parameters from the gaze fits:
    (T_min, lam, D0) = refit_floor.py sim_params (cross-validated,
    magnitude-calibrated); tau/cv = post-arrival dwell median and CV
    (intermittency_analysis.py). Falls back to the pre-floor values if the
    refit output is missing."""
    fit = {
        "A": {"D0": 1.395, "lam": 1.0, "T_min": 0.0, "tau": 0.205, "cv": 0.0},
        "B": {"D0": 1.549, "lam": 1.0, "T_min": 0.0, "tau": 0.190, "cv": 0.0},
        "C": {"D0": 1.243, "lam": 1.0, "T_min": 0.0, "tau": 0.170, "cv": 0.0},
    }
    floor_path = GAZE_RESULTS_DIR / "lookahead_floor_summary.json"
    if floor_path.exists():
        floor = json.load(open(floor_path))
        for letter in fit:
            sp = floor.get(letter, {}).get("sim_params")
            if sp:
                fit[letter].update(
                    D0=sp["D0"], lam=sp["lam"], T_min=sp["T_min"])
    inter_path = GAZE_RESULTS_DIR / "intermittency_summary.json"
    if inter_path.exists():
        trig = json.load(open(inter_path))["trigger"]
        for letter in fit:
            if letter in trig:
                fit[letter].update(
                    tau=trig[letter]["post_dwell_median_s"],
                    cv=trig[letter]["cv_post_dwell"])
    return fit

VARIANTS = {
    "baseline": {},
    "budget": {"horizon_mode": "budget"},
    "intermittent": {"replan_mode": "intermittent"},
    "full": {"horizon_mode": "budget", "replan_mode": "intermittent"},
}


def build_configs(n_seeds, gaze_fit, deviation_frac=DEVIATION_FRAC):
    """Write one persona JSON per (variant, participant); seeds are applied
    per run via np.random.seed in the worker."""
    base = json.load(open(CONFIG_DIR / "office_worker.json"))
    base["speed_model"]["path"] = str(CONFIG_DIR / "population_gam.pkl")
    cfg_dir = RESULTS_DIR / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for vname, overrides in VARIANTS.items():
        for letter, fit in gaze_fit.items():
            cfg = json.loads(json.dumps(base))
            cfg.update(overrides)
            cfg["budget"] = {"D0": fit["D0"], "lam": fit["lam"],
                             "T_min": fit["T_min"]}
            cfg["replan_latency_s"] = fit["tau"]
            cfg["replan_latency_cv"] = fit["cv"]
            cfg["replan_deviation_frac"] = deviation_frac
            p = cfg_dir / f"{vname}_{letter}.json"
            with open(p, "w") as f:
                json.dump(cfg, f, indent=1)
            paths[(vname, letter)] = str(p)
    return paths


def _replan_stats(diag):
    ev = diag["replan_events"]
    if len(ev) < 2:
        return {}
    t = np.array([e["t"] for e in ev])
    theta = np.array([e["theta"] for e in ev])
    anchor = np.array([e["anchor"] for e in ev])
    cycles = np.diff(t)
    lead = anchor - theta                       # lookahead distance at solve
    lead_end = anchor[:-1] - theta[1:]          # residual at next solve
    with np.errstate(invalid="ignore", divide="ignore"):
        frac_rem = lead_end / np.where(lead[:-1] > 1e-9, lead[:-1], np.nan)
    return {
        "cycles": cycles.tolist(),
        "leads": lead[:-1].tolist(),
        "frac_remaining": frac_rem[np.isfinite(frac_rem)].tolist(),
        "overshoot_frac": float(np.mean(lead_end < 0)),
        "triggers": [e["trigger"] for e in ev],
    }


def run_job(job):
    """One (participant, tid): all variants x seeds. Runs in a worker."""
    letter, tid, cond, config_paths, n_seeds, max_steps = (
        job["letter"], job["tid"], job["cond"], job["config_paths"],
        job["n_seeds"], job["max_steps"])

    from hcs_package.cursor_simulator import CursorSimulator

    task_config, centerline = em.build_steering_task_config(cond)
    task_config["max_steps"] = max_steps
    target_radius = task_config.get("target_radius")
    if target_radius is None:
        rr = cond.get("radiusRatio")
        target_radius = cond["tunnelWidth"] * (rr if rr is not None else 0.5)

    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(task_config, tf)
        task_file = tf.name

    out = {"letter": letter, "tid": tid, "cond": cond,
           "centerline": centerline, "variants": {}}
    try:
        for vname in VARIANTS:
            sim = CursorSimulator(config_paths[(vname, letter)])
            records, diags, walls = [], [], []
            for seed in range(n_seeds):
                np.random.seed(1000 + 97 * seed + tid)
                t0 = time.time()
                traj_raw = sim.generate_trajectory_with_waypoints(
                    task_file=task_file, max_steps=max_steps,
                    target_radius=target_radius, use_optimal_path=True)
                walls.append(time.time() - t0)
                records.append(em._build_record(
                    traj_raw, sim.interval, target_radius, max_steps))
                diags.append({
                    "n_solves": sim.last_diagnostics["n_solves"],
                    "n_steps": sim.last_diagnostics["n_steps_executed"],
                    "replan": _replan_stats(sim.last_diagnostics),
                })
            out["variants"][vname] = {
                "records": records, "diags": diags,
                "wall_per_run": float(np.mean(walls)),
            }
    finally:
        os.unlink(task_file)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--max-steps", type=int, default=800)
    ap.add_argument("--tids", type=str, default=None,
                    help="comma-separated steering tids (default: all)")
    ap.add_argument("--deviation-frac", type=float, default=DEVIATION_FRAC,
                    help="early-replan threshold (fraction of local width)")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    gaze_fit = load_gaze_fit()
    print("gaze-fit params:", json.dumps(gaze_fit), flush=True)
    config_paths = build_configs(args.n_seeds, gaze_fit,
                                 deviation_frac=args.deviation_frac)

    # human data + steering conditions (identical tids across A/B/C)
    jobs = []
    human = {}
    for letter, pid in PARTICIPANTS.items():
        tid_to_condition, tid_to_bucket = em.scan_conditions(GAZE_DATA_DIR)
        steer_tids = sorted(t for t, b in tid_to_bucket.items() if b == "steering")
        if args.tids:
            keep = {int(x) for x in args.tids.split(",")}
            steer_tids = [t for t in steer_tids if t in keep]
        trials = em.load_trials_by_participant(steer_tids, GAZE_DATA_DIR)
        human[letter] = trials.get(pid, {})
        for tid in steer_tids:
            if tid not in human[letter]:
                continue
            jobs.append({"letter": letter, "tid": tid,
                         "cond": tid_to_condition[tid],
                         "config_paths": config_paths,
                         "n_seeds": args.n_seeds,
                         "max_steps": args.max_steps})

    print(f"{len(jobs)} (participant, tid) jobs x {len(VARIANTS)} variants "
          f"x {args.n_seeds} seeds", flush=True)

    rows, gaze_rows = [], []
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_job, j): (j["letter"], j["tid"]) for j in jobs}
        for i, fut in enumerate(as_completed(futs)):
            letter, tid = futs[fut]
            try:
                res = fut.result()
            except Exception as exc:  # keep going; report at the end
                print(f"  job {letter}/t{tid} FAILED: {exc}", flush=True)
                continue
            rounds = human[letter][tid]
            h_trajs = [r["trajectory"] for r in rounds.values()]
            h_speeds = [r["speeds"] for r in rounds.values()]
            h_cts = [r["completion_time"] for r in rounds.values()]
            for vname, v in res["variants"].items():
                recs = v["records"]
                metrics = compute_comparison_metrics(
                    [r["trajectory"] for r in recs],
                    [r["speeds"] for r in recs],
                    [r["completion_time"] for r in recs],
                    h_trajs, h_speeds, h_cts, res["centerline"])
                solves = sum(d["n_solves"] for d in v["diags"])
                steps = sum(d["n_steps"] for d in v["diags"])
                row = {
                    "variant": vname, "participant": letter, "tid": tid,
                    "width": res["cond"].get("tunnelWidth"),
                    "tunnel_type": res["cond"].get("tunnelType", "sinusoidal"),
                    "n_timed_out": sum(r["timed_out"] for r in recs),
                    "n_runs": len(recs),
                    "model_mt": float(np.mean([r["completion_time"] for r in recs])),
                    "human_mt": float(np.mean(h_cts)),
                    "solves_per_s": solves / (steps * 0.05) if steps else None,
                    "wall_per_run": v["wall_per_run"],
                }
                row.update({k: metrics.get(k) for k in (
                    "lateral_rmse", "speed_rmse", "speed_corr", "time_diff")})
                rows.append(row)
                for d in v["diags"]:
                    rp = d["replan"]
                    if rp:
                        gaze_rows.append({
                            "variant": vname, "participant": letter, "tid": tid,
                            "width": row["width"], "cycles": rp["cycles"],
                            "leads": rp["leads"],
                            "frac_remaining": rp["frac_remaining"],
                            "overshoot_frac": rp["overshoot_frac"],
                            "triggers": rp["triggers"],
                        })
            done = i + 1
            print(f"  [{done}/{len(jobs)}] {letter}/t{tid} "
                  f"({time.time()-t_start:.0f}s)", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "comparison.csv", index=False)
    with open(RESULTS_DIR / "replan_stats.json", "w") as f:
        json.dump(gaze_rows, f)
    print(f"wrote {RESULTS_DIR}/comparison.csv ({len(df)} rows) "
          f"and replan_stats.json ({len(gaze_rows)} runs)")


if __name__ == "__main__":
    main()

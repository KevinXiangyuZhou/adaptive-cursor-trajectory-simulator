"""Full-model replan events for the non-steering tasks of the gaze cohort.

Simulates the FULL variant (budget horizon + intermittent replanning,
per-participant gaze-fitted D0/tau) on the remaining task families A/B/C
performed, collecting the planning events needed for the horizon-vs-gaze
comparison:

  tids 36-38 wide_to_narrow, 39-41 narrow_to_wide  (eval-main builder)
  tids 42-56 unconstrained_pointing                (fitts bypass builder)
  tids 57-83 constrained_to_unconstrained          (tunnel segment to the
        transition x, then a 10 m bypass to the target — the construction
        from eval/eval-main/capability/c2u_probe.py)

Writes results/replan_stats_ext.json (same row format as replan_stats.json).

Run: python3 eval/eval-intermittent/run_ext_tasks.py [--n-seeds 2] [--workers 7]
"""

import argparse
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "eval" / "eval-main"))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "hcs_package" / "src"))

import numpy as np

import run_eval as em
from experiment.environment import POINTING_Y_OFFSET

RESULTS_DIR = SCRIPT_DIR / "results"
GAZE_DATA_DIR = PROJECT_ROOT / "human_data" / "gaze_cursor_data"
PARTICIPANTS = {"A": "P105835", "B": "P170114", "C": "P160254"}
EXT_TIDS = list(range(36, 84))


def build_c2u_config(human_round, cond):
    """Tunnel of segment1Width from the recorded start to transition taskX,
    then a 10 m-wide bypass leg to the nominal target (c2u_probe recipe)."""
    traj = human_round["trajectory"]
    s = traj[0]
    W = float(cond["segment1Width"])
    R = float(cond["targetRadius"])
    tx = float(cond["transition_point"]["taskX"])
    ctr = [s[0] + float(cond["distance"]),
           s[1] + POINTING_Y_OFFSET.get(cond.get("targetPosition", "middle"), 0.0)]
    seg1 = [[s[0] + (tx - s[0]) * k / 8.0, s[1]] for k in range(9)]
    seg2 = [[tx + (ctr[0] - tx) * k / 8.0,
             s[1] + (ctr[1] - s[1]) * k / 8.0] for k in range(1, 9)]
    pts = seg1 + seg2
    widths = [W] * len(seg1) + [10.0] * len(seg2)
    wp_px = [[p[0] / 0.46 * 460, p[1] / 0.26 * 260] for p in pts]
    task = {
        "waypoints": wp_px, "screen_width": 460, "screen_height": 260,
        "target_radius": R, "max_steps": 800,
        "constraints": {
            "coordinate_system": "normalized", "default_margin": 0.0,
            "regions": [{"constraint_type": "keep_in",
                         "geometry": {"type": "path", "path": pts,
                                      "width": widths},
                         "enabled": True}],
        },
    }
    return task, R


def _replan_stats(diag):
    ev = diag["replan_events"]
    if len(ev) < 2:
        return {}
    t = np.array([e["t"] for e in ev])
    theta = np.array([e["theta"] for e in ev])
    anchor = np.array([e["anchor"] for e in ev])
    lead = anchor - theta
    lead_end = anchor[:-1] - theta[1:]
    return {
        "cycles": np.diff(t).tolist(),
        "leads": lead.tolist(),
        "overshoot_frac": float(np.mean(lead_end < 0)),
        "triggers": [e["trigger"] for e in ev],
    }


def run_job(job):
    from hcs_package.cursor_simulator import CursorSimulator

    letter, tid, cond, rnd, n_seeds = (
        job["letter"], job["tid"], job["cond"], job["round"], job["n_seeds"])
    ttype = cond.get("tunnelType")
    if ttype in ("wide_to_narrow", "narrow_to_wide"):
        task, _ = em._build_wide_to_narrow_config(
            cond["segment1Width"], cond["segment2Width"],
            cond.get("curvature", 0.0))
        target_radius = task.get("target_radius",
                                 cond["segment2Width"] * 0.5)
    elif ttype == "unconstrained_pointing":
        task, _, _ = em.build_fitts_bypass_config(rnd, cond["targetRadius"])
        target_radius = float(cond["targetRadius"])
    else:  # constrained_to_unconstrained
        task, target_radius = build_c2u_config(rnd, cond)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(task, tf)
        task_file = tf.name

    out = []
    try:
        sim = CursorSimulator(str(RESULTS_DIR / "configs" / f"full_{letter}.json"))
        for seed in range(n_seeds):
            np.random.seed(2000 + 97 * seed + tid)
            sim.generate_trajectory_with_waypoints(
                task_file=task_file, max_steps=800,
                target_radius=target_radius, use_optimal_path=True)
            rp = _replan_stats(sim.last_diagnostics)
            if rp:
                rp.update({"variant": "full", "participant": letter,
                           "tid": tid, "tunnel_type": ttype,
                           "timed_out": sim.last_diagnostics["n_steps_executed"] >= 800})
                out.append(rp)
    finally:
        os.unlink(task_file)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=2)
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args()

    jobs = []
    for letter, pid in PARTICIPANTS.items():
        trials = em.load_trials_by_participant(EXT_TIDS, GAZE_DATA_DIR)
        for tid, rounds in trials.get(pid, {}).items():
            rnd = rounds[min(rounds)]
            jobs.append({"letter": letter, "tid": tid,
                         "cond": rnd["condition"], "round": rnd,
                         "n_seeds": args.n_seeds})
    print(f"{len(jobs)} jobs x {args.n_seeds} seeds", flush=True)

    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_job, j): (j["letter"], j["tid"]) for j in jobs}
        for i, fut in enumerate(as_completed(futs)):
            letter, tid = futs[fut]
            try:
                rows.extend(fut.result())
            except Exception as exc:
                print(f"  {letter}/t{tid} FAILED: {exc}", flush=True)
            if (i + 1) % 20 == 0:
                print(f"  [{i + 1}/{len(jobs)}] ({time.time() - t0:.0f}s)",
                      flush=True)

    with open(RESULTS_DIR / "replan_stats_ext.json", "w") as f:
        json.dump(rows, f)
    n_to = sum(r["timed_out"] for r in rows)
    print(f"wrote replan_stats_ext.json: {len(rows)} runs, {n_to} timed out")


if __name__ == "__main__":
    main()

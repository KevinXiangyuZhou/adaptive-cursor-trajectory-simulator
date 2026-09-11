"""
Cross-Validated Trajectory Similarity Evaluation.

For each participant P_i and each condition, compute:
  - d(model, P_i)   : distance from model trajectory to P_i
  - d(P_j,   P_i)   : distance from every other human P_j to P_i

If d(model, P_i) falls within the distribution of d(P_j, P_i),
the model is "within human variability."

Metrics (all computed on progress-aligned profiles):
  - Lateral deviation RMSE (trajectory shape similarity)
  - Speed profile RMSE   (speed adaptation similarity)
  - Speed profile correlation (speed pattern similarity)

Usage:
    python -m eval.experiment-similarity.run_eval [--rounds N] [--config path]
"""

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "hcs_package" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

from experiment.environment import create_environment, generate_task_config
from hcs_package.cursor_simulator import CursorSimulator
from utils.stats import (
    resample_by_progress,
    resample_speeds_by_progress,
    trajectory_rmse,
    speed_profile_correlation,
    speed_profile_rmse,
)

# ---------------------------------------------------------------------------
HUMAN_DATA_DIR = PROJECT_ROOT / "eval" / "human_data" / "raw"
RESULTS_DIR = SCRIPT_DIR / "results"

TRIAL_CONDITIONS = {
    1: {"type": "sigmoidal", "width": 0.02, "curvature": 0.025,
        "label": "sigmoidal W=20mm"},
    2: {"type": "sigmoidal", "width": 0.04, "curvature": 0.025,
        "label": "sigmoidal W=40mm"},
    3: {"type": "corner",    "width": 0.02, "num_corners": 2,
        "corner_offset": 0.1, "label": "corner W=20mm"},
    4: {"type": "corner",    "width": 0.04, "num_corners": 2,
        "corner_offset": 0.1, "label": "corner W=40mm"},
}

DEFAULT_CONFIG = (PROJECT_ROOT / "experiment" / "user_configurations"
                  / "customized.json")

N_PROGRESS_BINS = 100  # uniform progress bins for resampling


# ===================================================================
# Data Loading
# ===================================================================

def _normalize_traj(raw):
    if not raw:
        return []
    if isinstance(raw[0], dict):
        return [[p["x"], p["y"]] for p in raw]
    return raw


def _compute_speeds(traj, timestamps, window=5):
    n = len(traj)
    if n < 2 or len(timestamps) != n:
        return []
    raw = []
    for i in range(n):
        if i == 0:
            p0, p1 = traj[0], traj[1]
            dt = (timestamps[1] - timestamps[0]) / 1000.0
        elif i == n - 1:
            p0, p1 = traj[-2], traj[-1]
            dt = (timestamps[-1] - timestamps[-2]) / 1000.0
        else:
            p0, p1 = traj[i - 1], traj[i + 1]
            dt = (timestamps[i + 1] - timestamps[i - 1]) / 1000.0
        dist = math.sqrt((p1[0] - p0[0])**2 + (p1[1] - p0[1])**2)
        raw.append(dist / dt if dt > 0 else 0.0)
    half = window // 2
    smoothed = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        smoothed.append(sum(raw[lo:hi]) / (hi - lo))
    return smoothed


def load_human_trials():
    """Load all human trials for conditions 1-4.

    Returns:
        {participant_id: {trial_id: [{trajectory, speeds, timestamps}, ...]}}
    """
    data = {}
    for fpath in sorted(HUMAN_DATA_DIR.glob("*.json")):
        try:
            with open(fpath) as f:
                raw = json.load(f)
        except Exception:
            continue
        pid = raw.get("participantId", fpath.stem)
        sessions = raw.get("sessions", [])
        if not sessions:
            td = raw.get("trialData", [])
            sessions = [{"trialData": td}] if td else []

        if pid not in data:
            data[pid] = {}

        for sess in sessions:
            for td in sess.get("trialData", []):
                tid = td.get("trial_id")
                if tid not in TRIAL_CONDITIONS:
                    continue
                traj = _normalize_traj(td.get("trajectory", []))
                ts = td.get("timestamps", [])
                if len(traj) < 5 or len(ts) < 5:
                    continue
                speeds = _compute_speeds(traj, ts)
                if tid not in data[pid]:
                    data[pid][tid] = []
                data[pid][tid].append({
                    "trajectory": traj,
                    "speeds": speeds,
                    "timestamps": ts,
                    "round": td.get("round", 0),
                })
    return data


# ===================================================================
# Environment / Centerline
# ===================================================================

def build_task(cond):
    # Use tunnel width as target radius so the cursor reliably terminates
    # at the tunnel end instead of overshooting and wandering.
    t_radius = cond["width"] * 0.5
    if cond["type"] == "sigmoidal":
        env_dict = {
            "env_type": "tunnel_steering_smooth",
            "screen_width": 460, "screen_height": 260,
            "tunnelWidth": cond["width"], "curvature": cond["curvature"],
            "max_steps": 800, "target_radius": t_radius,
        }
    else:
        env_dict = {
            "env_type": "tunnel_steering_corner",
            "screen_width": 460, "screen_height": 260,
            "tunnelWidth": cond["width"],
            "num_corners": cond["num_corners"],
            "corner_offset": cond["corner_offset"],
            "max_steps": 800, "target_radius": t_radius,
        }
    env = create_environment(env_dict)
    task_config = generate_task_config(env, include_constraints=True)
    centerline = [[x, y] for x, y in env["centerline"]]
    return task_config, centerline


# ===================================================================
# Model simulation
# ===================================================================

def run_model(sim, task_config, n_rounds):
    """Run model on a task, return list of trajectory dicts."""
    interval = sim.interval
    records = []
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        json.dump(task_config, tf)
        task_file = tf.name
    try:
        for _ in range(n_rounds):
            traj_raw = sim.generate_trajectory_with_waypoints(
                task_file=task_file,
                max_steps=task_config.get("max_steps", 800),
                target_radius=task_config.get("target_radius", 0.01),
                use_optimal_path=True,
            )
            scale = 0.001
            traj = [[x * scale, y * scale] for x, y, _ in traj_raw]
            n = len(traj)
            speeds = []
            for i in range(1, n):
                d = math.sqrt((traj[i][0] - traj[i-1][0])**2
                              + (traj[i][1] - traj[i-1][1])**2)
                speeds.append(d / interval)
            if speeds:
                speeds.insert(0, speeds[0])
            records.append({"trajectory": traj, "speeds": speeds})
    finally:
        os.unlink(task_file)
    return records


# ===================================================================
# Pairwise distance computation
# ===================================================================

def compute_distances(traj_a, speeds_a, traj_b, speeds_b, centerline):
    """Compute trajectory and speed distances between two trials."""
    _, _, lat_a = resample_by_progress(traj_a, centerline, N_PROGRESS_BINS)
    _, _, lat_b = resample_by_progress(traj_b, centerline, N_PROGRESS_BINS)

    _, spd_a = resample_speeds_by_progress(speeds_a, traj_a, centerline,
                                           N_PROGRESS_BINS)
    _, spd_b = resample_speeds_by_progress(speeds_b, traj_b, centerline,
                                           N_PROGRESS_BINS)

    return {
        "lateral_rmse": trajectory_rmse(lat_a, lat_b),
        "speed_rmse": speed_profile_rmse(spd_a, spd_b),
        "speed_corr": speed_profile_correlation(spd_a, spd_b),
    }


# ===================================================================
# Cross-validated evaluation
# ===================================================================

def cross_validate(human_data, model_records, centerline, condition_label):
    """Leave-one-out cross-validation across participants.

    For each reference participant P_i, compute:
      - d(model, P_i) averaged over model rounds × P_i rounds
      - d(P_j,   P_i) averaged over all rounds of P_j × rounds of P_i
    """
    pids = sorted(human_data.keys())
    if len(pids) < 2:
        return None

    rows = []  # one row per (reference_pid, source)

    for ref_pid in pids:
        ref_trials = human_data[ref_pid]

        # --- Model-to-reference distances ---
        model_dists = {"lateral_rmse": [], "speed_rmse": [], "speed_corr": []}
        for ref_trial in ref_trials:
            for model_trial in model_records:
                d = compute_distances(
                    model_trial["trajectory"], model_trial["speeds"],
                    ref_trial["trajectory"], ref_trial["speeds"],
                    centerline,
                )
                for k in model_dists:
                    model_dists[k].append(d[k])

        rows.append({
            "condition": condition_label,
            "reference_pid": ref_pid,
            "source": "model",
            "lateral_rmse_mean": np.mean(model_dists["lateral_rmse"]),
            "speed_rmse_mean": np.mean(model_dists["speed_rmse"]),
            "speed_corr_mean": np.mean(model_dists["speed_corr"]),
            "n_pairs": len(model_dists["lateral_rmse"]),
        })

        # --- Human-to-reference distances (leave-one-out) ---
        for other_pid in pids:
            if other_pid == ref_pid:
                continue
            other_trials = human_data[other_pid]
            h_dists = {"lateral_rmse": [], "speed_rmse": [], "speed_corr": []}
            for ref_trial in ref_trials:
                for oth_trial in other_trials:
                    d = compute_distances(
                        oth_trial["trajectory"], oth_trial["speeds"],
                        ref_trial["trajectory"], ref_trial["speeds"],
                        centerline,
                    )
                    for k in h_dists:
                        h_dists[k].append(d[k])

            rows.append({
                "condition": condition_label,
                "reference_pid": ref_pid,
                "source": f"human_{other_pid[:8]}",
                "lateral_rmse_mean": np.mean(h_dists["lateral_rmse"]),
                "speed_rmse_mean": np.mean(h_dists["speed_rmse"]),
                "speed_corr_mean": np.mean(h_dists["speed_corr"]),
                "n_pairs": len(h_dists["lateral_rmse"]),
            })

    return rows


# ===================================================================
# Summary and plotting
# ===================================================================

def summarize_results(all_rows):
    """Aggregate cross-validation results per condition.

    For each condition, compare distribution of model-to-ref vs human-to-ref.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = sorted(set(r["condition"] for r in all_rows))
    metrics = ["lateral_rmse_mean", "speed_rmse_mean", "speed_corr_mean"]
    metric_labels = {
        "lateral_rmse_mean": "Lateral RMSE (m)",
        "speed_rmse_mean": "Speed RMSE (m/s)",
        "speed_corr_mean": "Speed Correlation",
    }

    summary_lines = []
    summary_lines.append("=" * 72)
    summary_lines.append("Cross-Validated Trajectory Similarity")
    summary_lines.append("=" * 72)

    stats_results = {}

    for cond in conditions:
        cond_rows = [r for r in all_rows if r["condition"] == cond]
        model_rows = [r for r in cond_rows if r["source"] == "model"]
        human_rows = [r for r in cond_rows if r["source"].startswith("human_")]

        summary_lines.append(f"\n  {cond}:")
        summary_lines.append(f"    N references = {len(set(r['reference_pid'] for r in cond_rows))}")
        summary_lines.append(f"    Model pairs  = {len(model_rows)}")
        summary_lines.append(f"    Human pairs  = {len(human_rows)}")

        cond_stats = {}
        for metric in metrics:
            m_vals = [r[metric] for r in model_rows]
            h_vals = [r[metric] for r in human_rows]

            m_mean = np.mean(m_vals)
            m_std = np.std(m_vals)
            h_mean = np.mean(h_vals)
            h_std = np.std(h_vals)

            label = metric_labels[metric]
            summary_lines.append(
                f"    {label:25s}  "
                f"Model: {m_mean:.4f} \u00b1 {m_std:.4f}  "
                f"Human: {h_mean:.4f} \u00b1 {h_std:.4f}"
            )

            cond_stats[metric] = {
                "model_mean": m_mean, "model_std": m_std,
                "human_mean": h_mean, "human_std": h_std,
                "model_vals": m_vals,
                "human_vals": h_vals,
            }
        stats_results[cond] = cond_stats

    # --- Print summary ---
    summary_lines.append("")

    for line in summary_lines:
        print(line)

    # --- Generate figure ---
    _plot_similarity(stats_results, conditions, metric_labels)

    return stats_results


def _plot_similarity(stats_results, conditions, metric_labels):
    """Bar chart: model vs human distances per condition with std error bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = list(metric_labels.keys())
    n_cond = len(conditions)
    n_met = len(metrics)

    fig, axes = plt.subplots(1, n_met, figsize=(4.5 * n_met, 4.5), dpi=300)
    if n_met == 1:
        axes = [axes]

    bar_w = 0.35
    x = np.arange(n_cond)
    colors = {"model": "#e8453c", "human": "#4287f5"}

    for ax, metric in zip(axes, metrics):
        m_means, m_stds, h_means, h_stds = [], [], [], []
        for cond in conditions:
            s = stats_results[cond][metric]
            m_means.append(s["model_mean"])
            m_stds.append(s["model_std"])
            h_means.append(s["human_mean"])
            h_stds.append(s["human_std"])

        ax.bar(x - bar_w / 2, m_means, bar_w, yerr=m_stds,
               label="Model\u2192Human", color=colors["model"],
               capsize=3, alpha=0.85)
        ax.bar(x + bar_w / 2, h_means, bar_w, yerr=h_stds,
               label="Human\u2192Human", color=colors["human"],
               capsize=3, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([c.replace(" W=", "\nW=") for c in conditions],
                           fontsize=8)
        ax.set_ylabel(metric_labels[metric], fontsize=9)
        ax.legend(fontsize=8)
        ax.set_title(metric_labels[metric], fontsize=10)

    plt.tight_layout()
    out_path = RESULTS_DIR / "similarity_comparison.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")

    # Also save PDF for publication
    out_pdf = RESULTS_DIR / "similarity_comparison.pdf"
    fig2, axes2 = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4.5), dpi=300)
    if len(metrics) == 1:
        axes2 = [axes2]
    # Re-draw in PDF
    for ax, metric in zip(axes2, metrics):
        m_means, m_stds, h_means, h_stds = [], [], [], []
        for cond in conditions:
            s = stats_results[cond][metric]
            m_means.append(s["model_mean"])
            m_stds.append(s["model_std"])
            h_means.append(s["human_mean"])
            h_stds.append(s["human_std"])
        ax.bar(x - bar_w / 2, m_means, bar_w, yerr=m_stds,
               label="Model\u2192Human", color=colors["model"], capsize=3, alpha=0.85)
        ax.bar(x + bar_w / 2, h_means, bar_w, yerr=h_stds,
               label="Human\u2192Human", color=colors["human"], capsize=3, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace(" W=", "\nW=") for c in conditions], fontsize=8)
        ax.set_ylabel(metric_labels[metric], fontsize=9)
        ax.legend(fontsize=8)
        ax.set_title(metric_labels[metric], fontsize=10)
    plt.tight_layout()
    fig2.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved {out_pdf}")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cross-validated trajectory similarity evaluation")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--rounds", type=int, default=5,
                        help="Model rounds per condition")
    args = parser.parse_args()

    print("=" * 72)
    print("Cross-Validated Trajectory Similarity Evaluation")
    print("=" * 72)
    print(f"  Config:  {args.config}")
    print(f"  Rounds:  {args.rounds}")
    print()

    # 1. Load human data
    print("[1/4] Loading human data ...")
    human_data = load_human_trials()
    n_participants = len(human_data)
    n_trials = sum(len(trials) for pid_data in human_data.values()
                   for trials in pid_data.values())
    print(f"       {n_participants} participants, {n_trials} trials")

    # 2. Generate centerlines
    print("[2/4] Generating centerlines ...")
    centerlines = {}
    task_configs = {}
    for tid, cond in TRIAL_CONDITIONS.items():
        tc, cl = build_task(cond)
        task_configs[tid] = tc
        centerlines[tid] = cl

    # 3. Run model
    print("[3/4] Running model simulations ...")
    sim = CursorSimulator(args.config)
    model_records = {}
    for tid, cond in TRIAL_CONDITIONS.items():
        model_records[tid] = run_model(sim, task_configs[tid], args.rounds)
        print(f"       {cond['label']}: {len(model_records[tid])} trajectories")

    # 4. Cross-validated comparison
    print("[4/4] Computing cross-validated distances ...")
    all_rows = []
    for tid, cond in TRIAL_CONDITIONS.items():
        # Gather per-participant trial lists for this condition
        cond_human = {}
        for pid, pid_data in human_data.items():
            if tid in pid_data and pid_data[tid]:
                cond_human[pid] = pid_data[tid]

        if len(cond_human) < 2:
            print(f"       Skipping {cond['label']}: <2 participants")
            continue

        rows = cross_validate(cond_human, model_records[tid],
                              centerlines[tid], cond["label"])
        if rows:
            all_rows.extend(rows)

    if not all_rows:
        print("  No results to report.")
        return

    # 5. Summarize and plot
    stats_results = summarize_results(all_rows)

    # 6. Save CSV
    csv_path = RESULTS_DIR / "similarity_results.csv"
    fieldnames = ["condition", "reference_pid", "source",
                  "lateral_rmse_mean", "speed_rmse_mean",
                  "speed_corr_mean", "n_pairs"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"  Saved {csv_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()

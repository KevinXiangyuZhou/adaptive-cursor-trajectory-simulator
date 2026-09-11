"""
Steering Law evaluation experiment.

Compares simulated completion times and average speeds against human data
across different tunnel widths.  The Steering Law predicts MT ~ L/W, so
doubling the width should roughly halve movement time.

Usage:
    python run_eval.py [--config path/to/user_config.json] [--rounds N]
"""

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "hcs_package" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "eval" / "utils"))

from experiment.utils import generateTunnelPath
from experiment.environment import (
    create_environment,
    generate_task_config,
)
from hcs_package.cursor_simulator import CursorSimulator
from trackpad_checker import check_participant
from oob import check_trajectory_oob

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HUMAN_DATA_DIR = PROJECT_ROOT / "eval" / "human_data" / "raw"
RESULTS_DIR = SCRIPT_DIR / "results"

TRIAL_CONDITIONS = {
    1: {"width": 0.02, "curvature": 0.025, "label": "narrow (W=0.02)"},
    2: {"width": 0.04, "curvature": 0.025, "label": "wide (W=0.04)"},
}

DEFAULT_CONFIG = PROJECT_ROOT / "experiment" / "user_configurations" / "customized.json"


# ===================================================================
# Step 1 — Load human data
# ===================================================================

def _path_length(trajectory):
    """Sum of Euclidean segment lengths."""
    total = 0.0
    for i in range(1, len(trajectory)):
        dx = trajectory[i][0] - trajectory[i - 1][0]
        dy = trajectory[i][1] - trajectory[i - 1][1]
        total += math.sqrt(dx * dx + dy * dy)
    return total


def _compute_speeds(trajectory, timestamps, window_size=5):
    """Central-difference speed with moving-average smoothing."""
    n = len(trajectory)
    if n < 2 or len(timestamps) != n:
        return []
    raw = []
    for i in range(n):
        if i == 0:
            p0, p1 = trajectory[0], trajectory[1]
            dt = (timestamps[1] - timestamps[0]) / 1000.0
        elif i == n - 1:
            p0, p1 = trajectory[-2], trajectory[-1]
            dt = (timestamps[-1] - timestamps[-2]) / 1000.0
        else:
            p0, p1 = trajectory[i - 1], trajectory[i + 1]
            dt = (timestamps[i + 1] - timestamps[i - 1]) / 1000.0
        dist = math.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)
        raw.append(dist / dt if dt > 0 else 0.0)

    half = window_size // 2
    smoothed = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        smoothed.append(sum(raw[lo:hi]) / (hi - lo))
    return smoothed


def get_trackpad_participants(min_confidence=0.7):
    """Get set of participant IDs that are classified as trackpad users.
    
    Args:
        min_confidence: Minimum confidence threshold for trackpad classification
        
    Returns:
        Set of participant IDs using trackpad with confidence >= threshold
    """
    trackpad_pids = set()
    for fpath in sorted(HUMAN_DATA_DIR.glob("*.json")):
        result = check_participant(fpath)
        if (result.get("device_type") == "trackpad" and 
            result.get("confidence", 0) >= min_confidence):
            trackpad_pids.add(result.get("participant_id"))
    return trackpad_pids


def _get_centerline_lengths():
    """Pre-compute centerline lengths for each trial condition.
    
    Returns:
        dict mapping trial_id -> centerline_length (in meters)
    """
    centerline_lengths = {}
    for tid, cond in TRIAL_CONDITIONS.items():
        env_dict = {
            "env_type": "tunnel_steering_smooth",
            "screen_width": 460,
            "screen_height": 260,
            "tunnelWidth": cond["width"],
            "curvature": cond["curvature"],
            "max_steps": 600,
            # Use tunnel width so the cursor reliably terminates at the tunnel end
            "target_radius": cond["width"] * 0.5,
        }
        environment = create_environment(env_dict)
        centerline = environment["centerline"]
        centerline_lengths[tid] = _path_length(centerline)
    return centerline_lengths


def load_human_data(trackpad_confidence=None):
    """Load all wave-tunnel trials (trial_id 1 & 2) from raw participant JSONs.

    Args:
        trackpad_confidence: If provided, filter to only include participants
            classified as trackpad users with confidence >= this threshold.
            If None, include all participants.

    Returns a list of dicts, each with keys:
        participant, trial_id, round, width, curvature,
        completion_time, avg_speed, path_length, centerline_length, speeds
    """
    # Get trackpad participants if filtering is enabled
    trackpad_pids = None
    if trackpad_confidence is not None:
        trackpad_pids = get_trackpad_participants(trackpad_confidence)
    
    # Pre-compute centerline lengths for ID calculation
    centerline_lengths = _get_centerline_lengths()
    
    records = []
    for fpath in sorted(HUMAN_DATA_DIR.glob("*.json")):
        with open(fpath) as f:
            data = json.load(f)
        pid = data.get("participantId", fpath.stem)
        
        # Skip non-trackpad participants if filtering is enabled
        if trackpad_pids is not None and pid not in trackpad_pids:
            continue
        
        sessions = data.get("sessions", [])
        trial_list = sessions[0].get("trialData", []) if sessions else data.get("trialData", [])
        for trial in trial_list:
            tid = trial.get("trial_id")
            if tid not in TRIAL_CONDITIONS:
                continue
            cond = trial.get("condition", {})
            traj_raw = trial.get("trajectory", [])
            timestamps = trial.get("timestamps", [])
            ct = trial.get("completionTime", 0.0)
            if not traj_raw or not timestamps or ct <= 0:
                continue

            traj = [[p["x"], p["y"]] if isinstance(p, dict) else p for p in traj_raw]
            pl = _path_length(traj)
            speeds = _compute_speeds(traj, timestamps)
            cl_length = centerline_lengths[tid]
            records.append({
                "participant": pid,
                "trial_id": tid,
                "round": trial.get("round", 1),
                "width": cond.get("tunnelWidth", TRIAL_CONDITIONS[tid]["width"]),
                "curvature": TRIAL_CONDITIONS[tid]["curvature"],
                "completion_time": ct,
                "avg_speed": pl / ct if ct > 0 else 0.0,
                "path_length": pl,
                "centerline_length": cl_length,
                "speeds": speeds,
            })
    return records


# ===================================================================
# Step 2 & 3 — Generate environments and run simulator
# ===================================================================

def _build_task_config(width, curvature):
    """Create an hcs_package-compatible task config for a wave tunnel."""
    env_dict = {
        "env_type": "tunnel_steering_smooth",
        "screen_width": 460,
        "screen_height": 260,
        "tunnelWidth": width,
        "curvature": curvature,
        "max_steps": 600,
        # Use tunnel width so the cursor reliably terminates at the tunnel end
        "target_radius": width * 0.5,
    }
    environment = create_environment(env_dict)
    task_config = generate_task_config(environment, include_constraints=True)
    centerline = environment["centerline"]
    return task_config, centerline


def run_simulator_trials(config_path, n_rounds=3):
    """Run the simulator for each width condition, *n_rounds* per condition.

    Returns:
        (records, oob_rounds) where oob_rounds lists rounds excluded due to
        persistent out-of-bounds violations.
    """
    sim = CursorSimulator(str(config_path))
    interval = sim.interval
    records = []
    oob_rounds = []

    for tid, cond in TRIAL_CONDITIONS.items():
        task_config, centerline = _build_task_config(cond["width"], cond["curvature"])
        cl_length = _path_length(centerline)
        half_width = cond["width"] / 2.0

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
            json.dump(task_config, tf)
            task_file = tf.name

        try:
            for rnd in range(1, n_rounds + 1):
                traj_raw = sim.generate_trajectory_with_waypoints(
                    task_file=task_file,
                    max_steps=task_config.get("max_steps", 600),
                    target_radius=task_config.get("target_radius", 0.01),
                    use_optimal_path=True,
                )
                # traj_raw: List[(x_px, y_px, delay)]
                sw = task_config["screen_width"]
                sh = task_config["screen_height"]
                wm = 0.46
                hm = wm * sh / sw
                traj = [[x / sw * wm, y / sh * hm] for x, y, _ in traj_raw]

                # Check for persistent out-of-bounds violations
                is_oob, oob_steps, max_consec = check_trajectory_oob(
                    traj, centerline, half_width, max_consecutive=5)
                if is_oob:
                    oob_rounds.append({
                        "trial_id": tid,
                        "round": rnd,
                        "label": cond["label"],
                        "oob_steps": oob_steps,
                        "max_consecutive": max_consec,
                    })
                    continue

                pl = _path_length(traj)
                ct = len(traj_raw) * interval
                n_pts = len(traj)
                speeds = []
                for i in range(1, n_pts):
                    d = math.sqrt(
                        (traj[i][0] - traj[i - 1][0]) ** 2
                        + (traj[i][1] - traj[i - 1][1]) ** 2
                    )
                    speeds.append(d / interval)

                records.append({
                    "participant": "model",
                    "trial_id": tid,
                    "round": rnd,
                    "width": cond["width"],
                    "curvature": cond["curvature"],
                    "completion_time": ct,
                    "avg_speed": pl / ct if ct > 0 else 0.0,
                    "path_length": pl,
                    "speeds": speeds,
                    "centerline_length": cl_length,
                })
        finally:
            os.unlink(task_file)

    return records, oob_rounds


# ===================================================================
# Step 4 — Compute metrics & steering-law fit
# ===================================================================

def compute_summary(records, source_label):
    """Aggregate records by width → mean/std of MT and avg_speed."""
    from collections import defaultdict
    by_width = defaultdict(list)
    for r in records:
        by_width[r["width"]].append(r)

    rows = []
    for w in sorted(by_width):
        group = by_width[w]
        mts = [r["completion_time"] for r in group]
        spds = [r["avg_speed"] for r in group]
        pls = [r["path_length"] for r in group]

        rows.append({
            "source": source_label,
            "width": w,
            "n": len(group),
            "MT_values": mts,
            "speed_values": spds,
            "MT_mean": np.mean(mts),
            "MT_std": np.std(mts),
            "speed_mean": np.mean(spds),
            "speed_std": np.std(spds),
            "PL_mean": np.mean(pls),
        })
    return rows


def steering_law_fit(records):
    """Fit MT = a + b * ID where ID = L / W.  Returns (a, b, r^2).
    
    The model is: MT = a + b * ID, where a is intercept and b is slope.
    """
    ids, mts = [], []
    for r in records:
        L = r.get("centerline_length") or r["path_length"]
        ids.append(L / r["width"])
        mts.append(r["completion_time"])
    ids, mts = np.array(ids), np.array(mts)
    if len(ids) < 2:
        return 0.0, 0.0, 0.0
    
    # Use polyfit for clarity: returns [slope, intercept] for degree 1
    coeffs = np.polyfit(ids, mts, 1)
    slope, intercept = coeffs[0], coeffs[1]
    
    # Compute R²
    predicted = intercept + slope * ids
    ss_res = np.sum((mts - predicted) ** 2)
    ss_tot = np.sum((mts - np.mean(mts)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    return float(intercept), float(slope), float(r2)


# ===================================================================
# Step 5 — Plots
# ===================================================================

def _ensure_results_dir():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_mt_vs_id(human_records, model_records):
    """Scatter + regression of MT vs ID."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ensure_results_dir()
    fig, ax = plt.subplots(figsize=(7, 5))

    for records, label, color, marker in [
        (human_records, "Human", "#4287f5", "o"),
        (model_records, "Model", "#e8453c", "s"),
    ]:
        ids, mts = [], []
        for r in records:
            L = r.get("centerline_length") or r["path_length"]
            ids.append(L / r["width"])
            mts.append(r["completion_time"])
        ids, mts = np.array(ids), np.array(mts)
        ax.scatter(ids, mts, c=color, alpha=0.35, s=18, marker=marker)

        a, b, r2 = steering_law_fit(records)
        id_range = np.linspace(ids.min() * 0.9, ids.max() * 1.1, 50)
        ax.plot(id_range, a + b * id_range, color=color, linewidth=2,
                label=f"{label}  MT = {a:.2f} + {b:.4f}\u00b7ID  (R\u00b2={r2:.3f})")

    ax.set_xlabel("Index of Difficulty  (L / W)")
    ax.set_ylabel("Movement Time (s)")
    ax.set_title("Steering Law: MT vs ID")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "steering_law_MT.png", dpi=300)
    fig.savefig(RESULTS_DIR / "steering_law_MT.pdf")
    plt.close(fig)
    print(f"  Saved {RESULTS_DIR / 'steering_law_MT.png'} (.pdf)")


def plot_speed_by_width(human_summary, model_summary):
    """Grouped bar chart of average speed per width with std error bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ensure_results_dir()
    fig, ax = plt.subplots(figsize=(6, 4.5))

    widths_set = sorted(set(r["width"] for r in human_summary + model_summary))
    x = np.arange(len(widths_set))
    bar_w = 0.32

    for offset, summary, label, color in [
        (-bar_w / 2, human_summary, "Human", "#4287f5"),
        (bar_w / 2, model_summary, "Model", "#e8453c"),
    ]:
        means, stds = [], []
        for w in widths_set:
            row = next((r for r in summary if r["width"] == w), None)
            if row:
                means.append(row["speed_mean"])
                stds.append(row["speed_std"])
            else:
                means.append(0)
                stds.append(0)
        ax.bar(x + offset, means, bar_w, yerr=stds, label=label,
               color=color, alpha=0.8, capsize=4, error_kw={"lw": 1.2})

    ax.set_xticks(x)
    ax.set_xticklabels([f"W = {w}" for w in widths_set])
    ax.set_ylabel("Average Speed (m/s)")
    ax.set_title("Average Steering Speed by Tunnel Width\n(Error bars = \u00b11 std)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "steering_law_speed.png", dpi=300)
    fig.savefig(RESULTS_DIR / "steering_law_speed.pdf")
    plt.close(fig)
    print(f"  Saved {RESULTS_DIR / 'steering_law_speed.png'} (.pdf)")


def plot_speed_profiles(human_records, model_records, interval=0.05):
    """Overlay mean human vs model speed profiles for each width."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ensure_results_dir()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for ax, tid in zip(axes, sorted(TRIAL_CONDITIONS)):
        cond = TRIAL_CONDITIONS[tid]
        w = cond["width"]
        ax.set_title(cond["label"])

        for records, label, color in [
            (human_records, "Human", "#4287f5"),
            (model_records, "Model", "#e8453c"),
        ]:
            group_speeds = [r["speeds"] for r in records if r["trial_id"] == tid and r["speeds"]]
            if not group_speeds:
                continue
            max_len = max(len(s) for s in group_speeds)
            padded = np.full((len(group_speeds), max_len), np.nan)
            for i, s in enumerate(group_speeds):
                padded[i, : len(s)] = s
            mean_spd = np.nanmean(padded, axis=0)
            std_spd = np.nanstd(padded, axis=0)
            t = np.arange(len(mean_spd)) * interval
            ax.plot(t, mean_spd, color=color, label=label, linewidth=1.5)
            ax.fill_between(t, mean_spd - std_spd, mean_spd + std_spd,
                            color=color, alpha=0.15)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Speed (m/s)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Speed Profiles: Human vs Model", fontsize=13)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "steering_law_speed_profiles.png", dpi=200)
    plt.close(fig)
    print(f"  Saved {RESULTS_DIR / 'steering_law_speed_profiles.png'}")


def save_csv(human_records, model_records):
    """Write per-trial results to CSV."""
    _ensure_results_dir()
    path = RESULTS_DIR / "steering_law_results.csv"
    fields = ["source", "participant", "trial_id", "round", "width",
              "completion_time", "avg_speed", "path_length"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in human_records + model_records:
            writer.writerow({
                "source": "human" if r["participant"] != "model" else "model",
                "participant": r["participant"],
                "trial_id": r["trial_id"],
                "round": r["round"],
                "width": r["width"],
                "completion_time": round(r["completion_time"], 4),
                "avg_speed": round(r["avg_speed"], 6),
                "path_length": round(r["path_length"], 6),
            })
    print(f"  Saved {path}")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Steering Law evaluation")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG),
                        help="Path to user configuration JSON")
    parser.add_argument("--rounds", type=int, default=3,
                        help="Number of simulation rounds per condition")
    parser.add_argument("--trackpad-confidence", type=float, default=0,
                        help="Minimum confidence threshold for trackpad classification. "
                             "Only participants classified as trackpad users with confidence >= "
                             "this value will be included. Set to 0 to disable filtering.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"Config not found: {config_path}")

    with open(config_path) as f:
        user_cfg = json.load(f)
    steering_exp = user_cfg.get("planner_weights", {}).get("steering_exponent", 0.0)
    desired_speed = user_cfg.get("planner_weights", {}).get("desired_speed", "?")

    # Determine trackpad filtering
    trackpad_conf = args.trackpad_confidence if args.trackpad_confidence > 0 else None

    print("=" * 60)
    print("Steering Law Evaluation")
    print("=" * 60)
    print(f"  Config:             {config_path}")
    print(f"  steering_exponent:  {steering_exp}")
    print(f"  desired_speed:      {desired_speed}")
    print(f"  Rounds per cond:    {args.rounds}")
    if trackpad_conf is not None:
        print(f"  Trackpad filter:    confidence >= {trackpad_conf}")
    else:
        print(f"  Trackpad filter:    disabled (all participants)")
    print()

    # --- Step 1: Human data ---
    print("[1/5] Loading human data \u2026")
    human_records = load_human_data(trackpad_confidence=trackpad_conf)
    n_participants = len(set(r["participant"] for r in human_records))
    print(f"       Loaded {len(human_records)} human trials from {n_participants} participants")

    # --- Step 2 & 3: Simulator ---
    print("[2/5] Running simulator …")
    model_records, oob_rounds = run_simulator_trials(config_path, n_rounds=args.rounds)
    total_model_rounds = len(model_records) + len(oob_rounds)
    print(f"       Generated {len(model_records)} valid model trials"
          f" ({len(oob_rounds)} OOB out of {total_model_rounds})")
    if oob_rounds:
        for oob in oob_rounds:
            print(f"         OOB: {oob['label']} round {oob['round']}"
                  f" ({oob['max_consecutive']} consecutive steps)")

    # --- Step 3: Metrics ---
    print("[3/5] Computing metrics …")
    human_summary = compute_summary(human_records, "human")
    model_summary = compute_summary(model_records, "model")

    h_a, h_b, h_r2 = steering_law_fit(human_records)
    m_a, m_b, m_r2 = steering_law_fit(model_records)

    # MT ratio (narrow / wide)
    h_mt = {r["width"]: r["MT_mean"] for r in human_summary}
    m_mt = {r["width"]: r["MT_mean"] for r in model_summary}
    h_ratio = h_mt.get(0.02, 0) / h_mt.get(0.04, 1)
    m_ratio = m_mt.get(0.02, 0) / m_mt.get(0.04, 1)

    fmt_ms = lambda m, s: f"{m:.3f} \u00b1 {s:.3f}"

    print()
    print("  Steering Law Metrics (mean \u00b1 std):")
    print("  " + "=" * 70)
    for w in sorted(h_mt):
        h_row = next((r for r in human_summary if r["width"] == w), None)
        m_row = next((r for r in model_summary if r["width"] == w), None)
        if h_row and m_row:
            print(f"\n  W={int(w*1000)}mm (Human n={h_row['n']}, Model n={m_row['n']})")
            print(f"    MT:    Human {fmt_ms(h_row['MT_mean'], h_row['MT_std'])}s  "
                  f"Model {fmt_ms(m_row['MT_mean'], m_row['MT_std'])}s")
            print(f"    Speed: Human {fmt_ms(h_row['speed_mean'], h_row['speed_std'])}  "
                  f"Model {fmt_ms(m_row['speed_mean'], m_row['speed_std'])}")

    print(f"\n  Steering Law Fit:")
    print(f"    Human: MT = {h_a:.2f} + {h_b:.4f}*ID  (R\u00b2={h_r2:.3f})")
    print(f"    Model: MT = {m_a:.2f} + {m_b:.4f}*ID  (R\u00b2={m_r2:.3f})")

    print(f"\n  MT Ratio (W=20mm / W=40mm):")
    print(f"    Human: {h_ratio:.2f}    Model: {m_ratio:.2f}")
    print()

    # --- Step 4: Output ---
    print("[4/5] Generating plots ...")
    plot_mt_vs_id(human_records, model_records)
    plot_speed_by_width(human_summary, model_summary)
    plot_speed_profiles(human_records, model_records)

    print("[5/5] Saving CSV ...")
    save_csv(human_records, model_records)

    print()
    print("Done.")


if __name__ == "__main__":
    main()

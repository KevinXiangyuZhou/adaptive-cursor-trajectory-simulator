"""Per-trial gaze analysis for the corner tunnels (trials 6-10).

Question: within individual corner trials, is gaze pointing-like (saccade to a
corner turn point, dwell until the cursor arrives, hop to the next) or
steering-like (continuous lead ~ g(width), landings spread along the path)?

Method: gaze is mapped from screen pixels into task coordinates with a
per-trial affine fit (cursor_screenvid_px ~ cursor_traj task coords, which the
CSV provides as aligned pairs), the corner centerline is rebuilt with
experiment.utils.generateCornerPath from the trial's condition_json, and each
fixation onset is scored by its distance to the nearest 90-degree turn point.
Chance level = fraction of centerline arc length within the same radius of a
turn point.

Outputs: results/corner_gaze_{A,B,C}.png + printed per-trial stats.
Run: python3 corner_gaze_analysis.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiment.utils import generateCornerPath  # noqa: E402

import gaze_data as gd  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
NEAR_R = 0.02  # task units; "landed at a corner" radius


def corner_centerline(cond: dict) -> np.ndarray:
    path, _ = generateCornerPath(
        width=cond.get("tunnelWidth", 0.03),
        num_corners=cond.get("numCorners", 2),
        corner_offset=cond.get("cornerOffset", 0.05),
    )
    return np.asarray(path, dtype=float)


def turn_points(path: np.ndarray) -> np.ndarray:
    """Vertices where the polyline direction changes by more than 45 degrees."""
    d = np.diff(path, axis=0)
    d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
    dots = np.sum(d[:-1] * d[1:], axis=1)
    idx = np.where(dots < np.cos(np.pi / 4))[0] + 1
    # merge consecutive indices (a rounded corner spans a few vertices)
    pts, last = [], -10
    for i in idx:
        if i - last > 3:
            pts.append(path[i])
        last = i
    return np.asarray(pts)


def chance_near_fraction(path: np.ndarray, corners: np.ndarray, r: float) -> float:
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    samples = np.linspace(0, s[-1], 400)
    xy = np.column_stack([
        np.interp(samples, s, path[:, 0]), np.interp(samples, s, path[:, 1]),
    ])
    dmin = np.min(np.linalg.norm(xy[:, None, :] - corners[None, :, :], axis=2), axis=1)
    return float((dmin < r).mean())


def gaze_to_task(trial: pd.DataFrame) -> pd.DataFrame:
    """Per-trial affine screen->task map, fitted on aligned cursor pairs."""
    d = trial.dropna(subset=["cursor_px_x", "cursor_px_y", "cursor_x", "cursor_y"])
    out = trial.copy()
    for px, task, dst in (("cursor_px_x", "cursor_x", "gaze_task_x"),
                          ("cursor_px_y", "cursor_y", "gaze_task_y")):
        a, b = np.polyfit(d[px], d[task], 1)
        src = "gaze_x_px" if dst == "gaze_task_x" else "gaze_y_px"
        out[dst] = a * out[src] + b
        r = np.corrcoef(d[px], d[task])[0, 1]
        if abs(r) < 0.99:
            print(f"  [warn] weak affine fit ({px}: r={r:.3f})")
    return out


def analyze_participant(letter: str, axes):
    samples = gd.load_samples(letter)
    corner = samples[samples["tunnel_type"] == "corner"].copy()
    stats = []
    for row, tid in enumerate(sorted(corner["trial_id"].dropna().unique())):
        trial = corner[corner["trial_id"] == tid]
        cond = json.loads(trial["condition_json"].dropna().iloc[0])
        path = corner_centerline(cond)
        corners = turn_points(path)
        trial = gaze_to_task(trial)

        fx = (trial.dropna(subset=["fixation_id", "lead"])
              .groupby(["block_id", "fixation_id"], sort=True)
              .agg(gx=("gaze_task_x", "first"), gy=("gaze_task_y", "first"),
                   lead_onset=("lead", "first"), lead_end=("lead", "last"),
                   dur=("t", lambda s: s.max() - s.min()),
                   speed_onset=("speed", "first"))
              .reset_index())
        if len(fx) == 0:
            continue
        dmin = np.min(np.linalg.norm(
            fx[["gx", "gy"]].to_numpy()[:, None, :] - corners[None, :, :], axis=2), axis=1)
        near = dmin < NEAR_R
        chance = chance_near_fraction(path, corners, NEAR_R)
        stats.append({
            "trial": int(tid), "width": cond.get("tunnelWidth"),
            "n_corners": len(corners), "n_fix": len(fx),
            "frac_at_corner": float(near.mean()), "chance": chance,
            "corner_excess": float(near.mean()) - chance,
            "lead_onset_p50": float(fx["lead_onset"].median()),
            "lead_end_p50": float(fx["lead_end"].median()),
            "dwell_p50_s": float(fx["dur"].median()),
        })

        ax = axes[row]
        w = cond.get("tunnelWidth", 0.03)
        ax.plot(path[:, 0], path[:, 1], "-", color="0.75", lw=1)
        ax.plot(trial["cursor_x"], trial["cursor_y"], "-", color="tab:blue",
                lw=0.8, alpha=0.6)
        ax.scatter(fx["gx"], fx["gy"], s=8 + 60 * fx["dur"].clip(0, 1.5),
                   c=np.where(near, "tab:red", "tab:orange"), alpha=0.75, zorder=3)
        ax.scatter(corners[:, 0], corners[:, 1], marker="x", s=50, c="k", zorder=4)
        ax.set_title(f"t{int(tid)} W={w}  fix@corner {near.mean():.0%} "
                     f"(chance {chance:.0%})", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlim(-0.02, 0.5)
    return stats


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    all_stats = {}
    for letter in gd.PARTICIPANTS:
        print(f"=== participant {letter} ===")
        fig, axes = plt.subplots(5, 1, figsize=(9, 13))
        stats = analyze_participant(letter, axes)
        all_stats[letter] = stats
        for s in stats:
            print("  " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                   for k, v in s.items()))
        fig.suptitle(
            f"Participant {letter} — corner trials: fixation landings "
            f"(red = within {NEAR_R} of a corner, x = corner turn points, "
            f"dot size = dwell time)", fontsize=10)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / f"corner_gaze_{letter}.png", dpi=150)
        plt.close(fig)
    with open(RESULTS_DIR / "corner_gaze_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"wrote plots + corner_gaze_stats.json to {RESULTS_DIR}")


if __name__ == "__main__":
    main()

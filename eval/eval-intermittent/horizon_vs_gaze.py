"""How well does the model's planning horizon match the measured gaze lead?

Compares the full model's lookahead (anchor - cursor progress at each planning
event, from replan_stats.json) against the human gaze-cursor lead at fixation
onset (gaze CSVs), on the same steering conditions and in the same units
(task meters, 0.46 m screen).

Panels of results/horizon_vs_gaze.png:
  A  lead-vs-time sawtooth: one human round vs one model run, same condition
  B  lead distributions by tunnel width (human vs model, pooled A/B/C)
  C  condition-level medians (participant x width x tunnel type):
     human vs model scatter + Spearman rho
  D  tunnel-type effect at fixed width: does the curvature term in the
     budget reproduce the human type ordering?

Run: python3 eval/eval-intermittent/horizon_vs_gaze.py
"""

import json
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "hcs_package" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "eval" / "eval-gaze-cursor"))

import gaze_data as gd

RESULTS_DIR = SCRIPT_DIR / "results"

MIN_SPEED = 0.01
MIN_LEAD = 1e-4
# model tid -> human tunnelType label; comparison.csv 'sinusoidal' (tids 1-5,
# curvature 0.025) is labeled mid_sinusoidal in the gaze CSVs
TYPE_MAP = {"sinusoidal": "mid_sinusoidal"}
STEER_TYPES = ["straight", "gentle_sinusoidal", "mid_sinusoidal",
               "sharp_sinusoidal", "corner"]
LETTER = {"A": "A", "B": "B", "C": "C"}


def spearman(x, y):
    return float(np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1])


# ------------------------------------------------------------ data loading

def human_events():
    _, events = gd.load_all()
    ev = events[
        events["tunnel_type"].isin(STEER_TYPES)
        & (events["speed_onset"] > MIN_SPEED)
        & (events["lead_onset"] > MIN_LEAD)
    ].copy()
    ev["width"] = ev["width"].round(3)
    return ev


def model_events():
    rows = json.load(open(RESULTS_DIR / "replan_stats.json"))
    comp = pd.read_csv(RESULTS_DIR / "comparison.csv")
    tid_type = dict(zip(comp["tid"], comp["tunnel_type"]))
    recs = []
    for r in rows:
        if r["variant"] != "full":
            continue
        ttype = TYPE_MAP.get(tid_type[r["tid"]], tid_type[r["tid"]])
        if ttype not in STEER_TYPES:
            continue
        # drop the last two events of each run: their anchors are capped at
        # the path end, deflating the lead (humans get the same end cap but
        # the fixation filter keeps their events all along the path)
        for lead in r["leads"][:-2]:
            if lead > MIN_LEAD:
                recs.append({"participant": r["participant"],
                             "tunnel_type": ttype,
                             "width": round(float(r["width"]), 3),
                             "lead": float(lead)})
    return pd.DataFrame(recs)


# ------------------------------------------------- panel A: sawtooth traces

def model_sawtooth(width=0.03, curvature=0.025, seed=5):
    """One full-model run on the mid-sinusoidal tunnel; returns (t, lead)
    with lead(t) = active anchor - cursor arc-length progress."""
    from experiment.environment import create_environment, generate_task_config
    from hcs_package.cursor_simulator import CursorSimulator
    from hcs_package.reference_path import ReferencePath

    env = create_environment({"env_type": "tunnel_steering_smooth",
                              "tunnelWidth": width, "curvature": curvature})
    task = generate_task_config(env, include_constraints=True)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(task, f)
        task_file = f.name

    cfg_path = RESULTS_DIR / "configs" / "full_C.json"
    sim = CursorSimulator(str(cfg_path))
    np.random.seed(seed)
    traj, ref_path = sim.generate_trajectory_with_waypoints(
        task_file=task_file, max_steps=600, target_radius=width / 2,
        return_timestamps=True, return_reference_path=True)
    d = sim.last_diagnostics

    # cursor progress theta(t): trajectory is in px on a 460x260 frame
    scale = 0.46 / task.get("screen_width", 460)
    pts = np.array([(x * scale, y * scale) for x, y, _ in traj])
    t = np.array([ts for _, _, ts in traj])
    theta = np.empty(len(pts))
    guess = 0.0
    for i, p in enumerate(pts):
        guess = float(ref_path.find_closest_theta(p, initial_guess=guess))
        theta[i] = guess

    ev = d["replan_events"]
    anchors = np.full(len(t), np.nan)
    for k, e in enumerate(ev):
        t_next = ev[k + 1]["t"] if k + 1 < len(ev) else np.inf
        anchors[(t >= e["t"]) & (t < t_next)] = e["anchor"]
    lead = anchors - theta
    events_xy = [(e["t"], e["anchor"] - e["theta"]) for e in ev]
    return t, lead, events_xy


def human_sawtooth(letter="C", trial_id=3):
    samples = gd.load_samples(letter)
    s = samples[(samples["trial_id"] == trial_id)].dropna(subset=["lead"])
    if not len(s):
        return None
    blocks = s.groupby("block_id").size()
    blk = blocks.idxmax()
    g = s[s["block_id"] == blk]
    t = g["t"].to_numpy()
    return t - t[0], g["lead"].to_numpy()


# --------------------------------------------------------------- the figure

def main():
    hev = human_events()
    mev = model_events()

    # condition-level medians (participant x width x tunnel_type)
    hm = (hev.groupby(["participant", "width", "tunnel_type"])["lead_onset"]
          .agg(["median", "size"]).reset_index()
          .rename(columns={"median": "human_lead", "size": "n_h"}))
    mm = (mev.groupby(["participant", "width", "tunnel_type"])["lead"]
          .median().reset_index().rename(columns={"lead": "model_lead"}))
    cells = hm.merge(mm, on=["participant", "width", "tunnel_type"])
    cells = cells[cells["n_h"] >= 8]
    rho = spearman(cells["human_lead"], cells["model_lead"])
    ratio = float((cells["model_lead"] / cells["human_lead"]).median())

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))

    # A: sawtooth overlay
    ax = axes[0, 0]
    hs = human_sawtooth("C", 3)
    if hs is not None:
        ax.plot(hs[0], hs[1], color="tab:blue", lw=1.0,
                label="human C, trial 3 (gaze lead)")
    t, lead, evxy = model_sawtooth()
    ax.plot(t, lead, color="tab:red", lw=1.0,
            label="full model, same condition (anchor lead)")
    ax.scatter([e[0] for e in evxy], [e[1] for e in evxy], color="tab:red",
               s=16, zorder=3)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("time in trial (s)")
    ax.set_ylabel("lead ahead of cursor (m)")
    ax.set_title("A  Lookahead sawtooth: model vs gaze (w=0.03 sinusoidal)")
    ax.legend(fontsize=8)

    # B: distributions by width (pooled participants, all steering types)
    ax = axes[0, 1]
    widths = sorted(w for w in hev["width"].unique()
                    if w in set(mev["width"].unique()))
    pos = np.arange(len(widths))
    hdata = [hev[hev["width"] == w]["lead_onset"].to_numpy() for w in widths]
    mdata = [mev[mev["width"] == w]["lead"].to_numpy() for w in widths]
    b1 = ax.boxplot(hdata, positions=pos - 0.18, widths=0.3, showfliers=False,
                    patch_artist=True)
    b2 = ax.boxplot(mdata, positions=pos + 0.18, widths=0.3, showfliers=False,
                    patch_artist=True)
    for p in b1["boxes"]:
        p.set_facecolor("tab:blue"); p.set_alpha(0.5)
    for p in b2["boxes"]:
        p.set_facecolor("tab:red"); p.set_alpha(0.5)
    ax.set_xticks(pos, [f"{w:.2f}" for w in widths])
    ax.set_xlabel("tunnel width (m)")
    ax.set_ylabel("lead (m)")
    ax.set_title("B  Lead distributions by width (blue human, red model)")

    # C: condition-median scatter
    ax = axes[1, 0]
    markers = {"A": "o", "B": "s", "C": "^"}
    wcolors = {0.01: "#440154", 0.02: "#3b528b", 0.03: "#21918c",
               0.04: "#5ec962", 0.05: "#fde725"}
    for _, r in cells.iterrows():
        ax.scatter(r["human_lead"], r["model_lead"],
                   marker=markers[r["participant"]],
                   color=wcolors.get(r["width"], "gray"), s=42,
                   edgecolor="k", linewidth=0.4)
    lim = max(cells["human_lead"].max(), cells["model_lead"].max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("human median gaze lead (m)")
    ax.set_ylabel("model median lookahead (m)")
    ax.set_title(f"C  Condition medians (rho={rho:.2f}, "
                 f"model/human ratio={ratio:.2f})")
    for p, mk in markers.items():
        ax.scatter([], [], marker=mk, color="gray", label=f"participant {p}")
    for w, c in wcolors.items():
        ax.scatter([], [], marker="o", color=c, label=f"w={w:.2f}")
    ax.legend(fontsize=7, ncol=2)

    # D: tunnel-type ordering at fixed width
    ax = axes[1, 1]
    w0 = 0.03
    order = ["straight", "gentle_sinusoidal", "mid_sinusoidal",
             "sharp_sinusoidal", "corner"]
    hb = [hev[(hev["width"] == w0) & (hev["tunnel_type"] == tt)]["lead_onset"]
          .median() for tt in order]
    mb = [mev[(mev["width"] == w0) & (mev["tunnel_type"] == tt)]["lead"]
          .median() for tt in order]
    xpos = np.arange(len(order))
    ax.bar(xpos - 0.18, hb, width=0.36, color="tab:blue", alpha=0.7,
           label="human")
    ax.bar(xpos + 0.18, mb, width=0.36, color="tab:red", alpha=0.7,
           label="model")
    ax.set_xticks(xpos, [o.replace("_sinusoidal", "\nsinus.") for o in order],
                  fontsize=8)
    ax.set_ylabel("median lead (m)")
    ax.set_title(f"D  Curvature effect at w={w0:.2f}: type ordering")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "horizon_vs_gaze.png", dpi=150)
    plt.close(fig)

    summary = {
        "n_cells": int(len(cells)),
        "rho_condition_medians": rho,
        "model_over_human_ratio_median": ratio,
        "by_width_median": {
            f"{w:.2f}": {"human": float(np.median(h)), "model": float(np.median(m))}
            for w, h, m in zip(widths, hdata, mdata)
        },
        "by_type_at_w03": {tt: {"human": float(h) if h == h else None,
                                "model": float(m) if m == m else None}
                           for tt, h, m in zip(order, hb, mb)},
    }
    with open(RESULTS_DIR / "horizon_vs_gaze_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

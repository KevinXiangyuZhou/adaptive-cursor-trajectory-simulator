"""Evaluation figure: model reproduces the saccade-fixate-catch-up cycle.

Composes one grid from hand-picked columns of the cluster
results-cluster-10p/gaze-lead-10p/{p}/lead_by_curvature/{type}.png figures:
each selection (participant, tunnel type, human round) becomes one column,
rows are the five tunnel widths.  Human lead comes from the committed
human-gaze-lead-10p/data CSVs; the model trace is re-simulated here with the
participant's fitted persona (noise on, seeded), exactly as gaze_lead_grids.py
did on the cluster, and drawn as per-timestep dots like the human series.

Usage:  python paper/fig_gaze_lead_cycles.py  [--seed 42]
Output: paper/figures/gaze_lead_cycles.{pdf,png}
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "eval" / "eval-gaze-lead"))
sys.path.insert(0, str(PROJECT_ROOT / "eval" / "model_fitting"))
import model_gaze_lead as mg                      # noqa: E402
import fit_speed_model as fsm                     # noqa: E402

DATA_DIR = PROJECT_ROOT / "eval" / "eval-gaze-lead" / "human-gaze-lead-10p" / "data"
CONFIG_DIR = PROJECT_ROOT / "results-cluster-10p" / "anchor_fitting_10p" / "stages" / "base"
OUT_DIR = Path(__file__).resolve().parent / "figures"

# (participant, type_label, human round 1-3) -> one column, in this order
SELECTIONS = [
    ("p04", "normal", 2),
    ("p06", "normal", 2),
    ("p10", "normal", 1),
    ("p04", "straight", 3),
    ("p06", "straight", 2),
    ("p10", "straight", 1),
]
TASK_SHORT = {"normal": "sinusoid", "straight": "straight"}
HUMAN_COLOR, MODEL_COLOR = mg.HUMAN_COLOR, mg.MODEL_COLOR


def make_sim(config_path):
    """Fitted persona with noise on, as eval_10p.sh staged it."""
    cfg = json.load(open(config_path))
    cfg.pop("_description", None)
    cfg["add_noise"] = True
    if not float(cfg.get("replan_latency_cv", 0.0) or 0.0):
        cfg["replan_latency_cv"] = 0.89
    return fsm._make_sim(cfg)


def load_participant_series(letter, types, seed):
    """(human, model, widths): human[(type, width, round)] and
    model[(type, width)] time/lead series for this participant."""
    hum = pd.read_csv(DATA_DIR / f"{letter}_steering_lead.csv")
    hum = hum[hum["type_label"].isin(types)]
    rounds_by_tid, t2c, t2b = fsm.load_participant(letter)
    sim = make_sim(CONFIG_DIR / f"{letter}_anchor_config_s42.json")

    human, model = {}, {}
    trials = hum.groupby("trial_id").first().reset_index()
    for _, tr in trials.iterrows():
        tid, ty, w = int(tr["trial_id"]), tr["type_label"], float(tr["width_mm"])
        for rnd, g in hum[hum["trial_id"] == tid].groupby("round"):
            human[(ty, w, int(rnd))] = (g["t"].to_numpy(), g["lead"].to_numpy())
        built = mg.build_task(tid, t2b.get(tid), t2c.get(tid), rounds_by_tid.get(tid, []))
        if built is None:
            continue
        np.random.seed(seed + 100 * int(letter[1:]) + tid)
        res = mg.model_lead_trace(sim, built[0], built[1])
        if res is not None:
            model[(ty, w)] = (res[0], res[1])
        print(f"  [{letter}] t{tid} {ty} W{w:g}: model {'ok' if res else 'FAILED'}",
              flush=True)
    widths = sorted(trials["width_mm"].unique())
    return human, model, widths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    types_by_letter = {}
    for letter, ty, _ in SELECTIONS:
        types_by_letter.setdefault(letter, set()).add(ty)
    series = {letter: load_participant_series(letter, tys, a.seed)
              for letter, tys in types_by_letter.items()}
    widths = series[SELECTIONS[0][0]][2]

    # global y limits over everything drawn, per-column x limits
    all_lead, col_tmax = [], []
    for letter, ty, rnd in SELECTIONS:
        human, model, _ = series[letter]
        tmax = 0.0
        for w in widths:
            for s in (human.get((ty, w, rnd)), model.get((ty, w))):
                if s is not None:
                    all_lead.append(s[1])
                    tmax = max(tmax, float(s[0].max()))
        col_tmax.append(tmax)
    lead = np.concatenate(all_lead)
    lo, hi = np.percentile(lead, [0.5, 99.5])
    pad = 0.06 * (hi - lo + 1e-9)
    ylim = (min(lo, 0.0) - pad, max(hi, 0.0) + pad)

    n_rows, n_cols = len(widths), len(SELECTIONS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.3 * n_cols, 1.55 * n_rows),
                             squeeze=False, sharey=True)
    for c, (letter, ty, rnd) in enumerate(SELECTIONS):
        human, model, _ = series[letter]
        for r, w in enumerate(widths):
            ax = axes[r, c]
            h, m = human.get((ty, w, rnd)), model.get((ty, w))
            if h is not None:
                ax.plot(h[0], h[1], ".", ms=1.4, color=HUMAN_COLOR, alpha=0.5)
            if m is not None:
                ax.plot(m[0], m[1], ".", ms=1.4, color=MODEL_COLOR, alpha=0.8)
            ax.axhline(0.0, color="0.4", lw=0.6)
            ax.set_xlim(-0.02 * col_tmax[c], 1.02 * col_tmax[c])
            ax.set_ylim(*ylim)
            ax.tick_params(labelsize=7)
            if r == 0:
                ax.set_title(f"{letter} — {TASK_SHORT[ty]}", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"W={w:g} mm\nsigned lead (m)", fontsize=8)
            if r == n_rows - 1:
                ax.set_xlabel("time (s)", fontsize=8)
            else:
                ax.tick_params(labelbottom=False)
    fig.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"gaze_lead_cycles.{ext}", dpi=250)
    print(f"wrote {OUT_DIR / 'gaze_lead_cycles.pdf'}")


if __name__ == "__main__":
    main()

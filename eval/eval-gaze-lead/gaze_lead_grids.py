"""Model-vs-human gaze-lead PNGs in the human-gaze-lead-10p layout.

For one participant and a fitted persona, this simulates every constant-width
steering trial (gentle/sharp/normal sinusoid, straight, corner; N_RUNS noisy
runs each) and renders, into --out-dir:

  individual/t{tid}_{type}_W{w}mm.png   one round per stacked panel (human
                                        lead dots, as in human_gaze_lead.py)
                                        with model run k overlaid on panel k
  lead_by_width/W{w}mm.png              rows = gentle/sharp/normal, cols =
                                        rounds 1-3 (human dots + model line
                                        per cell, one shared x/y scale)
  lead_by_curvature/{type}.png          rows = widths, cols = rounds, same
                                        overlay, for all five types

Human series come from the stored plotting data
(human-gaze-lead-10p/data/{letter}_steering_lead.csv — committed, so no gaze
recomputation on the cluster). Model lead = anchor - cursor on the trial
centerline via model_gaze_lead.model_lead_trace (the model's sawtooth).

Usage:
  python gaze_lead_grids.py --letters p01 --config <fitted persona> \
      [--out-dir DIR] [--noise on|off] [--runs 3]
"""
import argparse
import copy
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

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "eval" / "model_fitting"))
import model_gaze_lead as mg                      # noqa: E402
import fit_speed_model as fsm                     # noqa: E402

DATA_DIR = SCRIPT_DIR / "human-gaze-lead-10p" / "data"
TYPES_BY_WIDTH = ["gentle", "sharp", "normal"]    # rows of lead_by_width
ALL_TYPES = ["gentle", "sharp", "normal", "straight", "corner"]
HUMAN_COLOR, MODEL_COLOR = mg.HUMAN_COLOR, mg.MODEL_COLOR


def make_sim(config_path, noise_on):
    cfg = json.load(open(config_path))
    cfg.pop("_description", None)
    if noise_on:
        cfg["add_noise"] = True
        if not float(cfg.get("replan_latency_cv", 0.0) or 0.0):
            cfg["replan_latency_cv"] = 0.89
    else:
        cfg["add_noise"] = False
        cfg["replan_latency_cv"] = 0.0
    return fsm._make_sim(cfg)


def model_runs(sim, tid, cond, rounds, n_runs):
    """[(t, lead)] for n_runs noisy simulations of one steering trial."""
    built = mg.build_task(tid, "steering", cond, rounds)
    if built is None:
        return []
    tc, cl = built
    out = []
    for _ in range(n_runs):
        res = mg.model_lead_trace(sim, tc, cl)
        if res is not None:
            out.append((res[0], res[1]))
    return out


def shared_limits(series):
    lead = np.concatenate([l for _, l in series])
    lo, hi = np.percentile(lead, [0.5, 99.5])
    pad = 0.06 * (hi - lo + 1e-9)
    tmax = max(t.max() for t, _ in series)
    return (-0.02 * tmax, 1.02 * tmax), (min(lo, 0.0) - pad, max(hi, 0.0) + pad)


def draw_cell(ax, human, model):
    if human is not None:
        ax.plot(human[0], human[1], ".", ms=1.4, color=HUMAN_COLOR, alpha=0.5)
    if model is not None:
        ax.plot(model[0], model[1], "-", lw=1.2, color=MODEL_COLOR, alpha=0.9)
    ax.axhline(0.0, color="0.4", lw=0.7)
    ax.grid(alpha=0.25)


def render_rounds_grid(out_png, rows, human_cells, model_cells, suptitle):
    """rows: [(key, label)]; cells keyed (key, round_idx)."""
    series = ([v for v in human_cells.values() if v is not None]
              + [v for v in model_cells.values() if v is not None])
    if not series:
        return
    xlim, ylim = shared_limits(series)
    n_cols = 3
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(4.2 * n_cols, 2.4 * len(rows)),
                             squeeze=False, sharex=True, sharey=True)
    for r, (rk, rlabel) in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r, c]
            draw_cell(ax, human_cells.get((rk, c)), model_cells.get((rk, c)))
            if r == 0:
                ax.set_title(f"round {c + 1} / model run {c + 1}", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{rlabel}\nsigned lead (m)")
            if r == len(rows) - 1:
                ax.set_xlabel("time since round start (s)")
    axes[0, 0].set_xlim(*xlim); axes[0, 0].set_ylim(*ylim)
    fig.suptitle(suptitle + "\n(dots = human lead; line = fitted-model sawtooth, "
                 "one noisy run per column)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def render_individual(out_png, human_rounds_, model_runs_, title):
    n = max(len(human_rounds_), 1)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.4 * n), squeeze=False, sharex=True)
    series = human_rounds_ + model_runs_
    xlim, ylim = shared_limits(series) if series else ((0, 1), (-0.05, 0.05))
    for k, ax in enumerate(axes[:, 0]):
        h = human_rounds_[k] if k < len(human_rounds_) else None
        m = model_runs_[k % len(model_runs_)] if model_runs_ else None
        draw_cell(ax, h, m)
        ax.set_ylabel(f"round {k + 1}\nlead (m)")
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    axes[-1, 0].set_xlabel("time since round start (s)")
    fig.suptitle(title + "  (dots = human; line = model run)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def run_letter(letter, config_path, out_dir, noise_on, n_runs):
    hum = pd.read_csv(DATA_DIR / f"{letter}_steering_lead.csv")
    rounds_by_tid, t2c, t2b = fsm.load_participant(letter)
    sim = make_sim(config_path, noise_on)

    trials = (hum.groupby("trial_id").first().reset_index()
                 [["trial_id", "type_label", "width_mm"]])
    # human series per (tid, round-index) and model runs per tid
    hser = {}
    for (tid_r, rnd), g in hum.groupby(["trial_id", "round"]):
        hser[(int(tid_r), int(rnd) - 1)] = (g["t"].to_numpy(), g["lead"].to_numpy())
    mruns = {}
    for _, tr in trials.iterrows():
        tid = int(tr["trial_id"])
        cond = t2c.get(tid)
        if cond is None or t2b.get(tid) != "steering":
            continue
        mruns[tid] = model_runs(sim, tid, cond, rounds_by_tid.get(tid, []), n_runs)
        print(f"  [{letter}] t{tid} {tr['type_label']} W{tr['width_mm']:g}: "
              f"{len(mruns[tid])} model runs", flush=True)

    out = Path(out_dir)
    (out / "individual").mkdir(parents=True, exist_ok=True)
    (out / "lead_by_width").mkdir(exist_ok=True)
    (out / "lead_by_curvature").mkdir(exist_ok=True)

    # ---- individual task PNGs
    for _, tr in trials.iterrows():
        tid = int(tr["trial_id"])
        if tid not in mruns:
            continue
        hr = [hser[(tid, k)] for k in range(3) if (tid, k) in hser]
        render_individual(
            out / "individual" / f"t{tid:02d}_{tr['type_label']}_W{tr['width_mm']:g}mm.png",
            hr, mruns[tid],
            f"{letter}  t{tid}  {tr['type_label']}  W={tr['width_mm']:g} mm")

    # ---- lead_by_width: one figure per width, rows = gentle/sharp/normal
    tid_of = {(r["type_label"], r["width_mm"]): int(r["trial_id"])
              for _, r in trials.iterrows()}
    widths = sorted({w for (ty, w) in tid_of if ty in TYPES_BY_WIDTH
                     if all((t2, w) in tid_of for t2 in TYPES_BY_WIDTH)})
    for w in widths:
        hc, mc, rows = {}, {}, []
        for ty in TYPES_BY_WIDTH:
            tid = tid_of.get((ty, w))
            if tid is None:
                continue
            rows.append((ty, f"{ty}  (t{tid})"))
            for c in range(3):
                hc[(ty, c)] = hser.get((tid, c))
                runs = mruns.get(tid, [])
                mc[(ty, c)] = runs[c % len(runs)] if runs else None
        render_rounds_grid(out / "lead_by_width" / f"W{w:g}mm.png", rows, hc, mc,
                           f"{letter} — W={w:g} mm: sinusoid tasks x rounds")

    # ---- lead_by_curvature: one figure per type, rows = widths
    for ty in ALL_TYPES:
        ws = sorted({w for (t2, w) in tid_of if t2 == ty})
        if not ws:
            continue
        hc, mc, rows = {}, {}, []
        for w in ws:
            tid = tid_of[(ty, w)]
            rows.append((w, f"W={w:g} mm  (t{tid})"))
            for c in range(3):
                hc[(w, c)] = hser.get((tid, c))
                runs = mruns.get(tid, [])
                mc[(w, c)] = runs[c % len(runs)] if runs else None
        render_rounds_grid(out / "lead_by_curvature" / f"{ty}.png", rows, hc, mc,
                           f"{letter} — {ty}: widths x rounds")
    print(f"  [{letter}] grids -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--letters", nargs="+", required=True)
    ap.add_argument("--config", required=True, help="fitted persona JSON")
    ap.add_argument("--out-dir", default=None,
                    help="output folder (default: model-gaze-lead-grids/{letter})")
    ap.add_argument("--noise", choices=["on", "off"], default="on")
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()
    for letter in a.letters:
        out = Path(a.out_dir) if a.out_dir else SCRIPT_DIR / "model-gaze-lead-grids" / letter
        run_letter(letter, a.config, out, a.noise == "on", a.runs)


if __name__ == "__main__":
    main()

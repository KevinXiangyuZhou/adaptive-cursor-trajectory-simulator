"""Alternative gaze-lead figure: p04 human vs the model, side by side.

Four columns — p04 sinusoid, p04 straight (human gaze lead, committed CSVs),
then model sinusoid, model straight (re-simulated, noise on, seeded) — rows
are the five tunnel widths. Unlike fig_gaze_lead_cycles.py, human and model
are drawn in separate columns, never overlaid.

--persona picks the model column's persona:
    pooled  (default) the pooled single model fitted to all eight
            participants (results-cluster-10p/anchor_fitting_pooled8)
    perpid  the participant's own fitted persona (mpcc-full-perpid-s42 run),
            with the participant's own traversal GAM (gam_traversal_{pid}.pkl)
            — that fit predates the per-participant-GAM pipeline fix, so the
            GAM path is patched in here exactly as stage_persona now does

Usage:  python paper/fig_gaze_lead_p4_pooled.py  [--seed 42]
        [--participant p04] [--persona pooled|perpid] [--config PATH]
        [--round-normal N] [--round-straight N]
Output: paper/figures/gaze_lead_{participant}_{persona}.{pdf,png}
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
POOLED_CONFIG = (PROJECT_ROOT / "results-cluster-10p" / "anchor_fitting_pooled8"
                 / "stages" / "pooled8" / "pooled8_anchor_config_s42.json")
PERPID_RUN = (PROJECT_ROOT / "results-cluster-10p" / "runs"
              / "mpcc-full-perpid-s42-20260908-1800-c96338d")
PACKAGE_MODELS = PROJECT_ROOT / "hcs_package" / "src" / "hcs_package" / "models"
OUT_DIR = Path(__file__).resolve().parent / "figures"

# default human rounds per participant: straight/normal are the ones picked in
# fig_gaze_lead_cycles; gentle uses round 2 (good sample coverage at all widths)
DEFAULT_ROUNDS = {"p04": {"normal": 2, "straight": 3, "gentle": 2},
                  "p06": {"normal": 2, "straight": 2, "gentle": 2},
                  "p10": {"normal": 1, "straight": 1, "gentle": 2}}
LETTER = "p04"
HUMAN_ROUND = DEFAULT_ROUNDS["p04"]
TYPES = ["normal", "straight"]
TASK_SHORT = {"normal": "sinusoid", "straight": "straight",
              "gentle": "gentle sinusoid"}
HUMAN_COLOR, MODEL_COLOR = mg.HUMAN_COLOR, mg.MODEL_COLOR


def make_sim(persona, config_path=None):
    """Simulator for the model columns: the pooled-8 persona, or the
    participant's own fitted persona with its own traversal GAM."""
    if config_path is None:
        config_path = (POOLED_CONFIG if persona == "pooled" else
                       PERPID_RUN / "fit" / "stages" / "base"
                       / f"{LETTER}_anchor_config_s{42}.json")
    cfg = json.load(open(config_path))
    cfg.pop("_description", None)
    cfg["add_noise"] = True
    if not float(cfg.get("replan_latency_cv", 0.0) or 0.0):
        cfg["replan_latency_cv"] = 0.89
    if persona == "perpid":
        sm = cfg.get("speed_model")
        if (isinstance(sm, dict) and sm.get("type") == "gam_traversal"
                and not sm.get("path")):
            name = f"gam_traversal_{LETTER}.pkl"
            if not (PACKAGE_MODELS / name).exists():
                raise FileNotFoundError(f"per-participant GAM {name} not found")
            cfg["speed_model"] = {"type": "gam_traversal", "path": name}
    print(f"  model persona: {config_path}\n  speed_model: {cfg.get('speed_model')}",
          flush=True)
    return fsm._make_sim(cfg)


def load_series(seed, persona, config_path=None):
    """human[(type, width)] (selected round) and model[(type, width)] traces."""
    hum = pd.read_csv(DATA_DIR / f"{LETTER}_steering_lead.csv")
    hum = hum[hum["type_label"].isin(TYPES)]
    rounds_by_tid, t2c, t2b = fsm.load_participant(LETTER)
    sim = make_sim(persona, config_path)

    human, model = {}, {}
    trials = hum.groupby("trial_id").first().reset_index()
    for _, tr in trials.iterrows():
        tid, ty, w = int(tr["trial_id"]), tr["type_label"], float(tr["width_mm"])
        g = hum[(hum["trial_id"] == tid) & (hum["round"] == HUMAN_ROUND[ty])]
        if len(g):
            human[(ty, w)] = (g["t"].to_numpy(), g["lead"].to_numpy())
        built = mg.build_task(tid, t2b.get(tid), t2c.get(tid), rounds_by_tid.get(tid, []))
        if built is None:
            continue
        np.random.seed(seed + 100 * int(LETTER[1:]) + tid)
        res = mg.model_lead_trace(sim, built[0], built[1])
        if res is not None:
            model[(ty, w)] = (res[0], res[1])
        print(f"  t{tid} {ty} W{w:g}: model {'ok' if res else 'FAILED'}", flush=True)
    widths = sorted(trials["width_mm"].unique())
    return human, model, widths


def main():
    global LETTER, HUMAN_ROUND
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--participant", default="p04")
    ap.add_argument("--round-normal", type=int, default=None)
    ap.add_argument("--round-task2", type=int, default=None)
    ap.add_argument("--task2", default="straight", choices=["straight", "gentle"],
                    help="second task shown right of the separator")
    ap.add_argument("--persona", default="pooled", choices=["pooled", "perpid"],
                    help="model columns: pooled-8 persona or the participant's "
                         "own fitted persona (with its own traversal GAM)")
    ap.add_argument("--config", default=None,
                    help="explicit persona JSON for the model columns "
                         "(overrides the --persona default path)")
    a = ap.parse_args()
    LETTER = a.participant
    TYPES[1] = a.task2
    HUMAN_ROUND = dict(DEFAULT_ROUNDS.get(
        LETTER, {"normal": 1, "straight": 1, "gentle": 1}))
    if a.round_normal is not None:
        HUMAN_ROUND["normal"] = a.round_normal
    if a.round_task2 is not None:
        HUMAN_ROUND[a.task2] = a.round_task2

    human, model, widths = load_series(a.seed, a.persona, a.config)

    # column spec: (series dict, type, color, title) — human/model pairs per
    # task, with a vertical separator between the two tasks
    ty2 = TYPES[1]
    cols = [
        (human, "normal", HUMAN_COLOR, f"{LETTER} — {TASK_SHORT['normal']}"),
        (model, "normal", MODEL_COLOR, f"model — {TASK_SHORT['normal']}"),
        (human, ty2, HUMAN_COLOR, f"{LETTER} — {TASK_SHORT[ty2]}"),
        (model, ty2, MODEL_COLOR, f"model — {TASK_SHORT[ty2]}"),
    ]

    # global y limits; x limits shared per task (human and model columns of
    # the same task get identical x ticks)
    all_lead, task_tmax = [], {}
    for series, ty, _, _ in cols:
        for w in widths:
            s = series.get((ty, w))
            if s is not None:
                all_lead.append(s[1])
                task_tmax[ty] = max(task_tmax.get(ty, 0.0), float(s[0].max()))
    col_tmax = [task_tmax[ty] for _, ty, _, _ in cols]
    lead = np.concatenate(all_lead)
    lo, hi = np.percentile(lead, [0.5, 99.5])
    pad = 0.06 * (hi - lo + 1e-9)
    ylim = (min(lo, 0.0) - pad, max(hi, 0.0) + pad)

    n_rows, n_cols = len(widths), len(cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.3 * n_cols, 1.55 * n_rows),
                             squeeze=False, sharey=True)
    for c, (series, ty, color, title) in enumerate(cols):
        for r, w in enumerate(widths):
            ax = axes[r, c]
            s = series.get((ty, w))
            if s is not None:
                ax.plot(s[0], s[1], ".", ms=1.4, color=color,
                        alpha=0.5 if color == HUMAN_COLOR else 0.8)
            ax.axhline(0.0, color="0.4", lw=0.6)
            ax.set_xlim(-0.02 * col_tmax[c], 1.02 * col_tmax[c])
            ax.set_ylim(*ylim)
            ax.tick_params(labelsize=7)
            if r == 0:
                ax.set_title(title, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"W={w:g} mm\nsigned lead (m)", fontsize=8)
            if r == n_rows - 1:
                ax.set_xlabel("time (s)", fontsize=8)
            else:
                ax.tick_params(labelbottom=False)
    fig.tight_layout()
    # vertical separator between the two tasks (between columns 2 and 3)
    left = axes[0, 1].get_position().x1
    right = axes[0, 2].get_position().x0
    x_mid = 0.5 * (left + right)
    fig.add_artist(matplotlib.lines.Line2D(
        [x_mid, x_mid], [0.02, 0.98], transform=fig.transFigure,
        color="0.5", lw=0.8))
    OUT_DIR.mkdir(exist_ok=True)
    stem = f"gaze_lead_{LETTER}_{a.persona}"
    if ty2 != "straight":
        stem += f"_{ty2}"
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", dpi=250)
    print(f"wrote {OUT_DIR / (stem + '.pdf')}")


if __name__ == "__main__":
    main()

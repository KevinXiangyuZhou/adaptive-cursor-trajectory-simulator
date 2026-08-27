"""Per-trial fixation-sequence comparison: human gaze vs model planning.

For each supported trial (steering, ID4SCS, unconstrained pointing) two
figures are produced:

  1. ROUNDS figure — one spatial panel per round, vertically aligned and
     drawn in the eval-main trajectory-plot style (lightgray corridor via
     generateTunnelBoundaries, salmon target disk, fixed task window,
     inverted y, no ticks/grid): every recorded HUMAN round and an equal
     number of stochastic MODEL runs. Lines are CURSOR trajectories only
     (human: the experiment-JSON round trajectory, the same source the
     eval-main results plots use), with the eye-fixation sequence layered
     on top as unconnected circles. Human fixations sit at the median gaze
     position of each fixation (task units, via the cursor-pair px->task
     mapping), circle AREA proportional to fixation duration, numbered in
     scanpath order; blink-corrupted fixations (blink overlapping the
     fixation or its incoming saccade) are excluded. Model "fixations" are
     the intermittent controller's replan anchors: the anchor is placed at
     a replan (fixation onset) and dwells until the next replan (fixation
     duration).
  2. SEQUENCE figure — arc position along the trial centerline vs time for
     every round of both sides: fixation circles (again sized by duration)
     riding ahead of each round's cursor progress trace s(t).

The model is run exactly as eval-main's run_eval.py runs it: a fresh
CursorSimulator per condition built straight from the fitted persona
config (noise on by default — add_noise defaults to True — with the
per-condition deterministic noise stream that implies), max_steps 800,
one run per human round. With noise on, the executed cursor can drift
outside the corridor between replans (open-loop execution), exactly as in
the eval-main results.

Outputs, per participant letter (A/B/C):
    results-fixation-sequences/{letter}_fixation_sequences.pdf
        two pages per trial (rounds page, sequence page)
    results-fixation-sequences/{letter}_trial{tid}_rounds.png     (--png)
    results-fixation-sequences/{letter}_trial{tid}_sequence.png   (--png)

Usage:
    python plot_fixation_sequences.py [--letters A B C] [--noise {on,off}]
                                      [--png TID [TID ...]]
"""

import argparse
import json
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "hcs_package" / "src",
          PROJECT_ROOT / "eval" / "eval-main",
          PROJECT_ROOT / "eval" / "eval-gaze-cursor",
          PROJECT_ROOT / "eval" / "eval-gaze-lead"):
    sys.path.insert(0, str(p))

import run_eval as em  # noqa: E402
import gaze_data  # noqa: E402
import model_gaze_lead as mgl  # noqa: E402  (task builders, ArcProjector)
from hcs_package.cursor_simulator import CursorSimulator  # noqa: E402

GAZE_DATA_DIR = PROJECT_ROOT / "human_data" / "gaze_cursor_data"
OUT_DIR = SCRIPT_DIR / "results-fixation-sequences"
DT = mgl.DT
SCALE = mgl.SCALE

# eval-main plot_experiment_results conventions: human blue / simulator
# orange, lightgray tunnel fill, fixed task window with inverted y (screen
# coordinates), no ticks, no grid, dpi 300.
HUMAN_COLORS = ["#1f77b4", "#6baed6"]      # round 1 dark, round 2 light
MODEL_COLORS = ["#ff7f0e", "#fdae6b"]
WINDOW_W, WINDOW_H = 0.46, 0.26
# circle area (pt^2) per second of fixation duration, shared by all panels
AREA_PER_S = 700.0
LEGEND_DURS = (0.2, 0.5, 1.0)
DPI = 300


# ---------------------------------------------------------------------------
# Model side: replan anchors as predicted fixations
# ---------------------------------------------------------------------------

def model_fixations(sim, task_config, centerline):
    """Run one simulation; return (traj_pts, fixations) with traj_pts the
    cursor trajectory in task units and fixations a list of dicts
    (t, x, y, duration_s, trigger) — one per replan anchor."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(task_config, tf)
        task_file = tf.name
    traj_raw, ref_path = sim.generate_trajectory_with_waypoints(
        task_file=task_file, target_radius=task_config["target_radius"],
        max_steps=task_config.get("max_steps", mgl.MAX_STEPS),
        return_reference_path=True)
    diag = sim.last_diagnostics
    pts = np.array([[x * SCALE, y * SCALE] for x, y, _ in traj_raw])
    n = len(pts)
    if n < 2:
        return None

    events = sorted(diag["replan_events"], key=lambda e: e["step"])
    fixations = []
    for k, e in enumerate(events):
        lo = int(e["step"])
        if lo >= n:
            continue
        hi = int(events[k + 1]["step"]) if k + 1 < len(events) else n
        s_a = min(float(e["anchor"]), float(ref_path.total_length))
        xy = np.asarray(ref_path(s_a), dtype=float).reshape(2)
        fixations.append({
            "t": lo * DT, "x": float(xy[0]), "y": float(xy[1]),
            "duration_s": (min(hi, n) - lo) * DT, "trigger": e["trigger"],
        })
    return pts, fixations


# ---------------------------------------------------------------------------
# Human side
# ---------------------------------------------------------------------------

def _blocks_to_rounds(d_all, n_rounds):
    """Group the gaze CSV's block_ids for one trial into rounds. load_samples
    segments blocks on >0.5 s gaps in the gaze clock, which also splits on
    within-round gaze dropouts — so a trial can have more blocks than
    recorded rounds. Merge time-ordered blocks back into n_rounds groups by
    splitting at the (n_rounds - 1) largest inter-block gaps."""
    spans = (d_all.groupby("block_id")["t"].agg(["min", "max"])
             .sort_values("min"))
    blocks = list(spans.index)
    if len(blocks) <= n_rounds or n_rounds < 1:
        return [[b] for b in blocks]
    gaps = spans["min"].to_numpy()[1:] - spans["max"].to_numpy()[:-1]
    cut_after = set(np.argsort(gaps)[-(n_rounds - 1):]) if n_rounds > 1 else set()
    groups, cur = [], [blocks[0]]
    for i, b in enumerate(blocks[1:]):
        if i in cut_after:
            groups.append(cur)
            cur = []
        cur.append(b)
    groups.append(cur)
    return groups


def human_rounds(samples, events, tid, json_rounds):
    """All recorded rounds of this trial, in order: list of dicts
    (traj, traj_t, fixations). Cursor trajectories/timestamps come from the
    experiment JSON rounds (the exact logged cursor, same source as the
    eval-main results plots); fixations come from the gaze CSV blocks
    grouped per round, times re-zeroed to each round's first in-trial gaze
    sample. Blink-corrupted fixations are dropped."""
    d_all = samples[(samples["trial_id"] == tid).fillna(False)]
    groups = _blocks_to_rounds(d_all, len(json_rounds))
    out = []
    for i, r in enumerate(json_rounds):
        traj = np.asarray(r["trajectory"], dtype=float)
        ts = np.asarray(r["timestamps"], dtype=float)
        traj_t = (ts - ts[0]) / 1000.0 if len(ts) else np.arange(len(traj)) * 0.0
        fixations = []
        if i < len(groups):
            blocks = groups[i]
            t0 = float(d_all[d_all["block_id"].isin(blocks)]["t"].min())
            ev = events[events["block_id"].isin(blocks)
                        & (~events["blink_corrupted"])]
            ev = ev.dropna(subset=["gaze_task_x", "gaze_task_y"]).sort_values("t_onset")
            fixations = [{"t": float(rr["t_onset"]) - t0,
                          "x": float(rr["gaze_task_x"]),
                          "y": float(rr["gaze_task_y"]),
                          "duration_s": float(rr["duration_s"])}
                         for _, rr in ev.iterrows()]
        out.append({"traj": traj, "traj_t": traj_t, "fixations": fixations})
    return out


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_geometry(ax, centerline, width, target_c, target_r):
    """Trial geometry exactly as eval-main's trajectory plots draw it:
    generateTunnelBoundaries corridor filled lightgray with gray dashed
    boundary lines, salmon target disk, dotted centerline when there is no
    corridor (unconstrained pointing)."""
    cl = np.asarray(centerline, dtype=float)
    if width is not None and len(cl) >= 2:
        lb, rb = em.generateTunnelBoundaries(centerline, width)
        ax.fill(np.concatenate([lb[:, 0], rb[::-1, 0]]),
                np.concatenate([lb[:, 1], rb[::-1, 1]]),
                color="lightgray", zorder=2)
        for b in (lb, rb):
            ax.plot(b[:, 0], b[:, 1], color="gray", linestyle="--",
                    linewidth=0.8, zorder=1)
    else:
        ax.plot(cl[:, 0], cl[:, 1], color="gray", linestyle=":",
                linewidth=0.8, zorder=1)
    if target_c is not None and target_r:
        ax.add_patch(plt.Circle(target_c, target_r, facecolor="salmon",
                                edgecolor="red", alpha=0.5, zorder=3))


def style_task_panel(ax):
    """Fixed task-window view in screen orientation (y down), no ticks or
    grid — matching plot_experiment_results."""
    ax.set_xlim(0.0, WINDOW_W)
    ax.set_ylim(0.0, WINDOW_H)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def draw_round_panel(ax, traj, fixations, color, label):
    """One spatial panel: cursor trajectory (green start dot) with the
    fixation sequence layered on top as unconnected duration-sized circles
    (lines on these plots are cursor trajectories only; the scanpath order
    is given by the numbers)."""
    if traj is not None and len(traj):
        ax.plot(traj[:, 0], traj[:, 1], "-", lw=0.8, color=color, alpha=0.85,
                zorder=10)
        ax.scatter(traj[0, 0], traj[0, 1], color="green", s=14, zorder=11)
    if fixations:
        xs = [f["x"] for f in fixations]
        ys = [f["y"] for f in fixations]
        sizes = [max(f["duration_s"], 0.04) * AREA_PER_S for f in fixations]
        ax.scatter(xs, ys, s=sizes, facecolor=color, alpha=0.25,
                   edgecolor=color, linewidth=1.0, zorder=13)
        for i, f in enumerate(fixations):
            ax.annotate(str(i + 1), (f["x"], f["y"]), fontsize=5.5,
                        ha="center", va="center", color="0.15", zorder=14)
    ax.set_title(label, fontsize=9, loc="left")


def duration_legend(ax, color="0.4"):
    handles = [Line2D([], [], marker="o", linestyle="none", color=color,
                      alpha=0.4, markersize=np.sqrt(d * AREA_PER_S),
                      label=f"{d:.1f} s")
               for d in LEGEND_DURS]
    ax.legend(handles=handles, title="fixation duration", fontsize=7,
              title_fontsize=7, loc="upper right", framealpha=0.85)


def plot_trial_rounds(tid, bucket, cond, h_rounds, m_rounds, geometry,
                      n_trials, k):
    """Figure 1: every human round and model run as vertically aligned
    task-space panels."""
    panels = ([(f"HUMAN round {i + 1} — {len(r['fixations'])} fixations",
                HUMAN_COLORS[i % len(HUMAN_COLORS)], r["traj"], r["fixations"])
               for i, r in enumerate(h_rounds)]
              + [(f"MODEL run {i + 1} — {len(f)} fixations (replan anchors)",
                  MODEL_COLORS[i % len(MODEL_COLORS)], traj, f)
                 for i, (traj, f) in enumerate(m_rounds)])
    n = max(len(panels), 1)
    fig, axes = plt.subplots(n, 1, figsize=(7.2, 1.2 + 4.15 * n),
                             squeeze=False)
    geo_cl, width, target_c, target_r = geometry
    for ax, (label, color, traj, fixations) in zip(axes[:, 0], panels):
        draw_geometry(ax, geo_cl, width, target_c, target_r)
        draw_round_panel(ax, traj, fixations, color, label)
        style_task_panel(ax)
    duration_legend(axes[0, 0])
    fig.suptitle(f"Trial {k + 1}/{n_trials} — {mgl.trial_label(cond, bucket)} "
                 f"(id {tid})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    return fig


def plot_trial_sequence(tid, bucket, cond, h_rounds, m_rounds, centerline,
                        n_trials, k):
    """Figure 2: arc position along the centerline vs time — the fixation
    sequences of every round over their cursor progress traces."""
    cl_proj = mgl.ArcProjector(centerline)
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    sides = ([(f"human r{i + 1}", HUMAN_COLORS[i % len(HUMAN_COLORS)],
               r["traj"], r["traj_t"], r["fixations"])
              for i, r in enumerate(h_rounds)]
             + [(f"model r{i + 1}", MODEL_COLORS[i % len(MODEL_COLORS)],
                 traj, np.arange(len(traj)) * DT, f)
                for i, (traj, f) in enumerate(m_rounds)])
    for name, color, traj, traj_t, fixations in sides:
        if traj is not None and len(traj):
            s_curs = np.array([cl_proj.theta(p) for p in traj])
            ax.plot(traj_t, s_curs, "-", lw=1.0, color=color, alpha=0.6,
                    label=f"{name} cursor s(t)")
        if fixations:
            ts = [f["t"] for f in fixations]
            ss = [cl_proj.theta((f["x"], f["y"])) for f in fixations]
            sz = [max(f["duration_s"], 0.04) * AREA_PER_S for f in fixations]
            ax.scatter(ts, ss, s=sz, facecolor=color, alpha=0.3,
                       edgecolor=color, linewidth=1.0, zorder=3,
                       label=f"{name} fixations")
    ax.set_xlabel("time since trial start (s)")
    ax.set_ylabel("arc position along centerline (task units)")
    ax.grid(False)
    ax.legend(fontsize=7, loc="lower right", ncols=2)
    ax.set_title(f"Trial {k + 1}/{n_trials} — {mgl.trial_label(cond, bucket)} "
                 f"(id {tid}) — fixation sequences vs cursor progress",
                 fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------

def sim_config_file(letter, noise_on):
    """Config path for CursorSimulator, matching eval-main's setup: the
    fitted persona config used verbatim (noise on — its default). For
    --noise off, a temp copy with add_noise/latency-jitter disabled."""
    pid = mgl.PARTICIPANTS[letter]
    cfg_path = mgl.FIT_DIR / f"{pid}_gam_config_s42.json"
    if noise_on:
        return str(cfg_path)
    cfg = json.load(open(cfg_path))
    cfg["add_noise"] = False
    cfg["replan_latency_cv"] = 0.0
    sm = cfg.get("speed_model", {})
    if sm.get("path") and not Path(sm["path"]).is_absolute():
        sm["path"] = str(mgl.FIT_DIR / sm["path"])
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(cfg, tf)
        return tf.name


def run_participant(letter, noise_on=True, png_tids=()):
    pid = mgl.PARTICIPANTS[letter]
    cfg_file = sim_config_file(letter, noise_on)

    tid_to_condition, tid_to_bucket = em.scan_conditions(GAZE_DATA_DIR)
    tids = sorted(t for t, b in tid_to_bucket.items()
                  if b in ("steering", "id4scs_w2n", "id4scs_n2w", "fitts"))
    rounds_all = em.load_trials_by_participant(tids, GAZE_DATA_DIR).get(pid, {})
    samples = gaze_data.load_samples(letter)
    events = gaze_data.fixation_events(samples)

    OUT_DIR.mkdir(exist_ok=True)
    pdf_path = OUT_DIR / f"{letter}_fixation_sequences.pdf"
    n_pages = 0
    with PdfPages(pdf_path) as pdf:
        for k, tid in enumerate(tids):
            bucket = tid_to_bucket[tid]
            cond = tid_to_condition[tid]
            rounds = [rounds_all[tid][r] for r in sorted(rounds_all.get(tid, {}))]
            task = mgl.build_task(tid, bucket, cond, rounds)
            if task is None:
                continue
            task_config, centerline = task
            task_config["max_steps"] = 800   # run_eval default, not mgl's 600
            h_rounds = human_rounds(samples, events, tid, rounds)
            # one stochastic model run per human round (>=1) on a FRESH
            # simulator per condition — run_eval's workers do exactly this,
            # so each condition sees the same deterministic noise stream
            sim = CursorSimulator(cfg_file)
            m_rounds = []
            for _ in range(max(len(h_rounds), 1)):
                out = model_fixations(sim, task_config, centerline)
                if out is not None:
                    m_rounds.append(out)
            if not h_rounds and not m_rounds:
                continue
            # display geometry via eval-main's canonical helper (corridor /
            # width profile / target disk, same as the participant overview)
            geometry = em._tid_display_geometry(tid, bucket, cond, rounds,
                                                {tid: (task_config, centerline)})
            fig_r = plot_trial_rounds(tid, bucket, cond, h_rounds, m_rounds,
                                      geometry, len(tids), k)
            fig_s = plot_trial_sequence(tid, bucket, cond, h_rounds, m_rounds,
                                        centerline, len(tids), k)
            pdf.savefig(fig_r)
            pdf.savefig(fig_s)
            if tid in png_tids:
                for fig, tag in ((fig_r, "rounds"), (fig_s, "sequence")):
                    png = OUT_DIR / f"{letter}_trial{tid}_{tag}.png"
                    fig.savefig(png, dpi=DPI, bbox_inches="tight")
                    print(f"  [{letter}] saved {png.name}")
            plt.close(fig_r)
            plt.close(fig_s)
            n_pages += 2
            nh = sum(len(r["fixations"]) for r in h_rounds)
            nm = sum(len(f) for _, f in m_rounds)
            print(f"  [{letter}] tid {tid} ({bucket}): {len(h_rounds)} human "
                  f"rounds / {len(m_rounds)} model runs, {nh}/{nm} fixations",
                  flush=True)
    print(f"Saved: {pdf_path} ({n_pages} pages)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--letters", nargs="+", default=list(mgl.PARTICIPANTS),
                    choices=list(mgl.PARTICIPANTS))
    ap.add_argument("--noise", choices=["on", "off"], default="on")
    ap.add_argument("--png", nargs="*", type=int, default=[],
                    help="trial ids to additionally save as standalone PNGs")
    a = ap.parse_args()
    for letter in a.letters:
        run_participant(letter, noise_on=(a.noise == "on"), png_tids=set(a.png))


if __name__ == "__main__":
    main()

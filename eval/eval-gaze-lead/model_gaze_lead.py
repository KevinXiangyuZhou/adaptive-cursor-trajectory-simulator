"""Model gaze-lead plots, in the format of human-gaze-lead/ PDFs.

The model's "gaze" is the planning anchor of the intermittent controller: at
each replan event the difficulty-budget horizon places the anchor a lookahead
distance ahead of the cursor (the model analog of a fixation onset), and the
anchor stays put while the plan executes open-loop (the analog of the fixation
dwell). Signed gaze lead is therefore

    lead(t) = s_centerline(anchor of the active plan) - s_centerline(cursor)

in task units (arc length along the trial's tunnel centerline — the same
coordinate as the human `gaze_lead_signed` column), which produces the same
sawtooth: saccade jump at each replan, roughly linear decay while the cursor
closes the gap, negative overshoot during the post-arrival latency.

For each participant (A/B/C) this script runs the FITTED persona
(model_fitting_8-24-26/{pid}_gam_config_s42.json: budget horizon +
intermittent replanning, goal_precision 0) once per supported trial (steering,
ID4SCS, unconstrained pointing; constrained->unconstrained trials have no
simulator task builder and are skipped), extracts the model lead trace, and
writes one PDF per participant:

    model-gaze-lead/{letter}_task_aligned_signed_gazelead_vs_time.pdf
        one page per trial: model lead trace (line) overlaid on the human
        round-1 gaze-lead samples (dots), time from trial start;
        final page: onset-lead-by-width and cycle-rate summary.
    model-gaze-lead/model_lead_events.csv
        one row per model replan event (the model's "fixation onsets").

Usage:  python model_gaze_lead.py [--letters A B C] [--noise {on,off}]
"""

import argparse
import json
import sys
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "hcs_package" / "src",
          PROJECT_ROOT / "eval" / "eval-main",
          PROJECT_ROOT / "eval" / "eval-gaze-cursor"):
    sys.path.insert(0, str(p))

import run_eval as em  # noqa: E402  (eval-main: task builders, data loading)
import gaze_data  # noqa: E402  (eval-gaze-cursor: human gaze CSVs)
from hcs_package.cursor_simulator import CursorSimulator  # noqa: E402
from hcs_package.reference_path import ReferencePath  # noqa: E402


class ArcProjector:
    """Global nearest-point projection onto a polyline, in arc length.

    ReferencePath.find_closest_theta is NOT usable for this: for linear
    (k=1) splines it returns the initial guess unchanged (Newton refinement
    needs a second derivative), and for k=3 the warm-started search is
    local. Here every query does a dense global argmin — the centerlines
    (sinusoids, corners, straight tunnels) do not approach themselves, so
    the global nearest point is the right projection, matching how the
    human gaze samples were projected onto the task centerline.
    """

    def __init__(self, centerline, n_dense=4000):
        cl = np.asarray(centerline, dtype=float)
        seg = np.linalg.norm(np.diff(cl, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        self.total_length = float(arc[-1])
        s_dense = np.linspace(0.0, self.total_length, n_dense)
        self.pts = np.column_stack([
            np.interp(s_dense, arc, cl[:, 0]),
            np.interp(s_dense, arc, cl[:, 1]),
        ])
        self.s_dense = s_dense

    def theta(self, p):
        d2 = np.sum((self.pts - np.asarray(p, dtype=float)) ** 2, axis=1)
        return float(self.s_dense[int(np.argmin(d2))])

GAZE_DATA_DIR = PROJECT_ROOT / "human_data" / "gaze_cursor_data"
FIT_DIR = PROJECT_ROOT / "model_fitting_8-24-26"
OUT_DIR = SCRIPT_DIR / "model-gaze-lead"
PARTICIPANTS = {"A": "P105835", "B": "P170114", "C": "P160254"}
DT = 0.05
SCALE = 0.001          # task screen units (px @460x260) -> meters/task units
MAX_STEPS = 600

HUMAN_COLOR = "#8c6bb1"
MODEL_COLOR = "#e6550d"


def build_task(tid, bucket, cond, rounds):
    """(task_config, centerline) for one trial, or None if unsupported."""
    if bucket == "steering":
        tc, cl = em.build_steering_task_config(cond)
    elif bucket in ("id4scs_w2n", "id4scs_n2w"):
        tc, cl = em._build_wide_to_narrow_config(
            cond["segment1Width"], cond["segment2Width"], cond.get("curvature", 0.0))
    elif bucket == "fitts":
        if not rounds:
            return None
        tc, cl, _w = em.build_fitts_bypass_config(rounds[0], cond["targetRadius"])
    else:
        return None
    tc = dict(tc)
    tc["max_steps"] = MAX_STEPS
    return tc, [list(map(float, p)) for p in cl]


def model_lead_trace(sim, task_config, centerline):
    """Run one simulation; return (t, lead, events) with lead in task units
    along the trial CENTERLINE (human gaze-lead coordinate), events a list of
    dicts (one per replan) with onset lead/speed."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(task_config, tf)
        task_file = tf.name
    traj_raw, ref_path = sim.generate_trajectory_with_waypoints(
        task_file=task_file, target_radius=task_config["target_radius"],
        max_steps=task_config.get("max_steps", MAX_STEPS),
        return_reference_path=True)
    diag = sim.last_diagnostics
    pts = np.array([[x * SCALE, y * SCALE] for x, y, _ in traj_raw])
    n = len(pts)
    if n < 2:
        return None

    cl_path = ArcProjector(centerline)

    # cursor arc position on the centerline (global projection per step)
    theta_cl = np.array([cl_path.theta(p) for p in pts])

    # anchor (model ref-path arc) -> xy -> centerline arc, per replan event
    events = sorted(diag["replan_events"], key=lambda e: e["step"])
    anchor_cl = np.full(n, np.nan)
    ev_rows = []
    for k, e in enumerate(events):
        s_a = min(float(e["anchor"]), float(ref_path.total_length))
        xy = np.asarray(ref_path(s_a), dtype=float).reshape(2)
        a_cl = cl_path.theta(xy)
        lo = int(e["step"])
        hi = int(events[k + 1]["step"]) if k + 1 < len(events) else n
        anchor_cl[lo:min(hi, n)] = a_cl
        if lo < n:
            spd = (np.linalg.norm(pts[min(lo + 1, n - 1)] - pts[lo]) / DT
                   if n > 1 else 0.0)
            ev_rows.append({
                "step": lo, "t": lo * DT, "trigger": e["trigger"],
                "lead_onset": a_cl - theta_cl[lo], "speed_onset": spd,
            })
    lead = anchor_cl - theta_cl
    t = np.arange(n) * DT
    return t, lead, ev_rows


def human_round1(samples, tid):
    """(t_rel, lead) of the first recorded round of this trial, or None."""
    d = samples[(samples["trial_id"] == tid) & samples["lead"].notna()]
    if d.empty:
        return None
    first_block = d["block_id"].iloc[0]
    d = d[d["block_id"] == first_block]
    if d.empty:
        return None
    t = d["t"].to_numpy()
    return t - t.min(), d["lead"].to_numpy()


def trial_label(cond, bucket):
    tt = cond.get("tunnelType", "sinusoidal" if "curvature" in cond else "?")
    if bucket == "steering":
        return f"{tt or 'sinusoidal'}, width {cond.get('tunnelWidth', '?')}"
    if bucket.startswith("id4scs"):
        return f"{tt}, {cond.get('segment1Width')}->{cond.get('segment2Width')}"
    return (f"pointing, R={cond.get('targetRadius', 0) * 1000:.0f}mm, "
            f"D={cond.get('distance', 0):.2f}")


def steering_width(cond):
    w = cond.get("tunnelWidth")
    return round(float(w), 3) if w is not None else None


def run_participant(letter, noise_on=True):
    pid = PARTICIPANTS[letter]
    cfg_path = FIT_DIR / f"{pid}_gam_config_s42.json"
    if noise_on:
        sim = CursorSimulator(str(cfg_path))
    else:
        cfg = json.load(open(cfg_path))
        cfg["add_noise"] = False
        cfg["replan_latency_cv"] = 0.0
        sm = cfg.get("speed_model", {})
        if sm.get("path") and not Path(sm["path"]).is_absolute():
            sm["path"] = str(FIT_DIR / sm["path"])
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            json.dump(cfg, tf)
            sim = CursorSimulator(tf.name)

    tid_to_condition, tid_to_bucket = em.scan_conditions(GAZE_DATA_DIR)
    tids = sorted(t for t, b in tid_to_bucket.items()
                  if b in ("steering", "id4scs_w2n", "id4scs_n2w", "fitts"))
    rounds_all = em.load_trials_by_participant(tids, GAZE_DATA_DIR).get(pid, {})
    samples = gaze_data.load_samples(letter)
    hum_events = gaze_data.fixation_events(samples)

    OUT_DIR.mkdir(exist_ok=True)
    pdf_path = OUT_DIR / f"{letter}_task_aligned_signed_gazelead_vs_time.pdf"
    all_ev = []
    onset_by_width = defaultdict(list)
    total_solves, total_time = 0, 0.0

    with PdfPages(pdf_path) as pdf:
        for k, tid in enumerate(tids):
            bucket = tid_to_bucket[tid]
            cond = tid_to_condition[tid]
            rounds = [rounds_all[tid][r] for r in sorted(rounds_all.get(tid, {}))]
            task = build_task(tid, bucket, cond, rounds)
            if task is None:
                continue
            out = model_lead_trace(sim, *task)
            if out is None:
                continue
            t_m, lead_m, ev = out
            for row in ev:
                row.update({"participant": letter, "tid": tid, "bucket": bucket})
            all_ev.extend(ev)
            w = steering_width(cond)
            if bucket == "steering" and w is not None:
                onset_by_width[w].extend(
                    [r["lead_onset"] for r in ev if r["trigger"] != "init"])
            total_solves += len(ev)
            total_time += t_m[-1] if len(t_m) else 0.0

            fig, ax = plt.subplots(figsize=(8.5, 4.2))
            hr = human_round1(samples, tid)
            if hr is not None:
                ax.plot(hr[0], hr[1], ".", ms=3, color=HUMAN_COLOR, alpha=0.7,
                        label="human (round 1 gaze)")
            ax.plot(t_m, lead_m, "-", lw=1.4, color=MODEL_COLOR,
                    label="model (anchor - cursor)")
            for row in ev:
                if row["trigger"] != "init":
                    ax.axvline(row["t"], color=MODEL_COLOR, lw=0.4, alpha=0.25)
            ax.axhline(0.0, color="0.4", lw=0.8)
            ax.set_xlabel("time since trial start (s)")
            ax.set_ylabel("signed gaze lead along centerline (task units)\n"
                          "(+ gaze ahead / - cursor ahead)")
            ax.set_title(f"Trial {k + 1}/{len(tids)}  round 1  —  "
                         f"{trial_label(cond, bucket)}  (id {tid})", fontsize=10)
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(alpha=0.25)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            print(f"  [{letter}] tid {tid} ({bucket}): "
                  f"{len(ev)} solves over {t_m[-1]:.1f}s", flush=True)

        # ---- summary page: onset lead by width + cycle rate ----
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        hsteer = hum_events[(hum_events["constrained"].astype(str) == "constrained")
                            & hum_events["width"].notna()]
        widths = sorted(onset_by_width)
        hm = [float(hsteer[np.isclose(hsteer["width"], w, atol=2e-3)]
                    ["lead_onset"].median()) for w in widths]
        mm = [float(np.median(onset_by_width[w])) for w in widths]
        axes[0].plot(widths, hm, "o-", color=HUMAN_COLOR, label="human")
        axes[0].plot(widths, mm, "s-", color=MODEL_COLOR, label="model")
        axes[0].set_xlabel("tunnel width (task units)")
        axes[0].set_ylabel("median onset lead (task units)")
        axes[0].set_title("fixation/replan onset lead by width")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # human rate = fixations per second of in-trial constrained time
        con = samples[samples["constrained"].astype(str) == "constrained"]
        h_time = float((con.groupby("block_id")["t"].max()
                        - con.groupby("block_id")["t"].min()).sum())
        hrate = len(hsteer) / max(h_time, 1e-9) if len(hsteer) else np.nan
        mrate = total_solves / max(total_time, 1e-9)
        axes[1].bar(["human\nfixations", "model\nreplans"], [hrate, mrate],
                    color=[HUMAN_COLOR, MODEL_COLOR], width=0.5)
        axes[1].set_ylabel("events / s")
        axes[1].set_title("planning-event rate")
        axes[1].grid(alpha=0.3, axis="y")
        fig.suptitle(f"participant {letter} ({pid}) — summary "
                     f"(steering trials; noise {'on' if noise_on else 'off'})")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Saved: {pdf_path}")
    return all_ev


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--letters", nargs="+", default=list(PARTICIPANTS),
                    choices=list(PARTICIPANTS))
    ap.add_argument("--noise", choices=["on", "off"], default="on",
                    help="motor noise + stochastic replan latency (default on: "
                         "matches how humans generated their traces)")
    a = ap.parse_args()

    all_rows = []
    for letter in a.letters:
        all_rows.extend(run_participant(letter, noise_on=(a.noise == "on")))

    import csv
    OUT_DIR.mkdir(exist_ok=True)
    out_csv = OUT_DIR / "model_lead_events.csv"
    with open(out_csv, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(all_rows)
    print(f"Saved: {out_csv} ({len(all_rows)} model planning events)")


if __name__ == "__main__":
    main()

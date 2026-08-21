"""Intermittent-planning analysis from gaze-cursor data.

Hypothesis: humans do not replan continuously (the MPCC model re-solves every
10 ms step); they plan intermittently. The gaze-cursor lead signal should then
be a sawtooth: a saccade jumps the gaze anchor ahead (lead spikes to the
planning-horizon distance), gaze dwells while the cursor closes the gap
open-loop, and a new saccade fires when some trigger condition is met.

This script (1) documents the sawtooth, and (2) runs a trigger-law
competition: what predicts *when* the next planning event (fixation end /
next saccade) happens?

  T1 fixed period      duration ~ const                    (clock-driven)
  T2 arrival/threshold duration ~ (lead_onset - delta)/v   (state-driven:
                       replan when the cursor has closed to within delta of
                       the gaze anchor; th_emp = lead_onset/v is the 0-delta
                       arrival-time predictor)
  T3 components        duration ~ lead_onset alone / 1/v alone (which part
                       of the arrival time carries the correlation)

If T2 wins, the model change is: execute the MPCC plan open-loop and re-solve
only when the cursor has consumed the planned segment down to a residual
delta (equivalently a fraction of the planned horizon), not every step.

Outputs under results/: intermittency_example_trace.png,
intermittency_sawtooth_profile.png, intermittency_trigger.png,
intermittency_summary.json.

Run: python3 intermittency_analysis.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gaze_data as gd

RESULTS_DIR = Path(__file__).resolve().parent / "results"
STEER_TYPES = [
    "corner", "gentle_sinusoidal", "mid_sinusoidal", "sharp_sinusoidal",
    "straight", "wide_to_narrow", "narrow_to_wide",
]
MIN_SPEED = 0.01   # task units/s; cursor must be moving at fixation onset
MIN_LEAD = 1e-4    # forward (positive-lead) fixations only
MAX_DURATION = 1.5  # s; drop end-of-trial dwells that are not steering cycles
MIN_SAMPLES = 3    # >= 3 gaze samples (15 ms) so lead_end is meaningful
N_PROFILE_BINS = 20


def spearman(x, y):
    return float(np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1])


def robust_cv(x):
    """IQR / |median| — scale-free dispersion; low = 'held constant'."""
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    iqr = np.percentile(x, 75) - np.percentile(x, 25)
    return float(iqr / max(abs(med), 1e-9))


def steering_events(events: pd.DataFrame) -> pd.DataFrame:
    ev = events[
        events["tunnel_type"].isin(STEER_TYPES)
        & (events["speed_onset"] > MIN_SPEED)
        & (events["lead_onset"] > MIN_LEAD)
        & (events["duration_s"] > 0)
        & (events["duration_s"] <= MAX_DURATION)
        & (events["n_samples"] >= MIN_SAMPLES)
        & events["lead_end"].notna()
    ].copy()
    ev["frac_remaining"] = ev["lead_end"] / ev["lead_onset"]
    return ev


def add_cycle_intervals(events: pd.DataFrame) -> pd.DataFrame:
    """Saccade-to-saccade (fixation-onset-to-onset) interval within a block,
    computed over ALL fixation events so gaps from filtered-out fixations do
    not inflate the cycle. block_id is only unique within a participant, so
    always key on (participant, block_id)."""
    events = events.sort_values(["participant", "block_id", "t_onset"]).copy()
    events["cycle_s"] = (
        events.groupby(["participant", "block_id"])["t_onset"].diff().shift(-1)
    )
    return events


# ------------------------------------------------------------- sawtooth

def sawtooth_profile(samples: pd.DataFrame, ev: pd.DataFrame) -> dict:
    """Mean time-normalized lead trajectory within a fixation. If execution
    is a ballistic catch-up, lead decays monotonically from onset toward ~0."""
    keys = set(zip(ev["participant"], ev["block_id"], ev["fixation_id"]))
    d = samples.dropna(subset=["fixation_id", "lead"]).copy()
    d["fixation_id"] = pd.to_numeric(d["fixation_id"], errors="coerce")

    grid = np.linspace(0.0, 1.0, N_PROFILE_BINS + 1)
    profiles = []
    for (part, blk, fid), g in d.groupby(
            ["participant", "block_id", "fixation_id"], sort=False):
        if (part, blk, fid) not in keys or len(g) < 5:
            continue
        t = g["t"].to_numpy()
        span = t[-1] - t[0]
        if span <= 0:
            continue
        tau = (t - t[0]) / span
        lead0 = g["lead"].to_numpy()[0]
        if lead0 <= MIN_LEAD:
            continue
        profiles.append(np.interp(grid, tau, g["lead"].to_numpy()) / lead0)
    prof = np.array(profiles)
    return {
        "grid": grid,
        "median": np.median(prof, axis=0),
        "q25": np.percentile(prof, 25, axis=0),
        "q75": np.percentile(prof, 75, axis=0),
        "n": len(prof),
    }


def plot_example_trace(samples: pd.DataFrame, ev: pd.DataFrame, outdir: Path):
    """Lead vs time for the block with the most analyzed fixations: the raw
    sawtooth, with fixation onsets marked."""
    part, blk = ev.groupby(["participant", "block_id"]).size().idxmax()
    g = samples[
        (samples["participant"] == part) & (samples["block_id"] == blk)
    ].dropna(subset=["lead"])
    e = ev[(ev["participant"] == part) & (ev["block_id"] == blk)]

    fig, ax = plt.subplots(figsize=(10, 3.6))
    t0 = g["t"].min()
    ax.plot(g["t"] - t0, g["lead"], lw=0.8, color="tab:gray", label="gaze-cursor lead")
    ax.scatter(e["t_onset"] - t0, e["lead_onset"], color="tab:red", s=18,
               zorder=3, label="fixation onset (saccade lands)")
    ax.scatter(e["t_onset"] - t0 + e["duration_s"], e["lead_end"],
               color="tab:blue", s=18, zorder=3, label="fixation end")
    ax.axhline(0, color="k", lw=0.6)
    row = e.iloc[0]
    ax.set_xlabel("time in trial (s)")
    ax.set_ylabel("signed gaze lead (task units)")
    ax.set_title(
        f"Sawtooth lead: participant {row['participant']}, "
        f"{row['tunnel_type']}, trial {row['trial_id']}"
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "intermittency_example_trace.png", dpi=150)
    plt.close(fig)


def plot_profiles(profiles: dict, outdir: Path):
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    colors = {"pooled": "k", "A": "tab:blue", "B": "tab:orange", "C": "tab:green"}
    for tag, prof in profiles.items():
        ax.plot(prof["grid"], prof["median"], color=colors[tag],
                lw=2 if tag == "pooled" else 1.2,
                label=f"{tag} (n={prof['n']})")
        if tag == "pooled":
            ax.fill_between(prof["grid"], prof["q25"], prof["q75"],
                            color="k", alpha=0.12)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("normalized time within fixation")
    ax.set_ylabel("lead / lead_onset")
    ax.set_title("Within-fixation catch-up (median, IQR pooled)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "intermittency_sawtooth_profile.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------- arrival decomposition

def crossing_decomposition(samples: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    """Split each fixation at the moment the cursor reaches the gaze anchor
    (first lead<=0 sample): duration = t_cross + post_dwell. If the trigger is
    'arrival event + fixed sensorimotor latency', post_dwell should be roughly
    constant while t_cross carries the state-dependence."""
    d = samples.dropna(subset=["fixation_id", "lead"]).copy()
    d["fixation_id"] = pd.to_numeric(d["fixation_id"], errors="coerce")
    keys = set(zip(ev["participant"], ev["block_id"], ev["fixation_id"]))

    rows = []
    for (part, blk, fid), g in d.groupby(
            ["participant", "block_id", "fixation_id"], sort=False):
        if (part, blk, fid) not in keys:
            continue
        t = g["t"].to_numpy()
        lead = g["lead"].to_numpy()
        below = np.nonzero(lead <= 0)[0]
        t_cross = float(t[below[0]] - t[0]) if len(below) else np.nan
        rows.append({"participant": part, "block_id": blk,
                     "fixation_id": fid, "t_cross": t_cross})
    return ev.merge(pd.DataFrame(rows),
                    on=["participant", "block_id", "fixation_id"], how="left")


# ------------------------------------------------------- trigger competition

def fit_duration_model(ev: pd.DataFrame, xcol: str) -> dict:
    """OLS duration ~ a + b*x, plus Spearman rho."""
    x = ev[xcol].to_numpy()
    y = ev["duration_s"].to_numpy()
    b, a = np.polyfit(x, y, 1)
    pred = a + b * x
    r2 = 1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    return {"slope": float(b), "intercept_s": float(a), "r2": float(r2),
            "rho": spearman(x, y)}


def trigger_stats(ev: pd.DataFrame) -> dict:
    out = {}
    for tag, g in [("pooled", ev)] + list(ev.groupby("participant")):
        g = g.copy()
        g["inv_speed"] = 1.0 / g["speed_onset"]
        cyc = g["cycle_s"].dropna()
        crossed = g.dropna(subset=["t_cross"]).copy()
        crossed["post_dwell"] = crossed["duration_s"] - crossed["t_cross"]
        out[tag] = {
            # arrival decomposition (events where the cursor reached the anchor)
            "frac_crossed": float(g["t_cross"].notna().mean()),
            "t_cross_median_s": float(crossed["t_cross"].median()),
            "post_dwell_median_s": float(crossed["post_dwell"].median()),
            "cv_t_cross": robust_cv(crossed["t_cross"].clip(lower=1e-3)),
            "cv_post_dwell": robust_cv(crossed["post_dwell"]),
            "cv_duration_crossed": robust_cv(crossed["duration_s"]),
            "rho_tcross_themp": spearman(crossed["t_cross"], crossed["th_emp"]),
            "rho_duration_tcross": spearman(crossed["t_cross"], crossed["duration_s"]),
            "rho_postdwell_tcross": spearman(crossed["t_cross"], crossed["post_dwell"]),
            "rho_postdwell_width": spearman(
                crossed["width"].fillna(crossed["width"].median()),
                crossed["post_dwell"]),
        } | {
            "n": int(len(g)),
            "duration_median_s": float(g["duration_s"].median()),
            "cycle_median_s": float(cyc.median()) if len(cyc) else None,
            "replan_rate_hz": float(1.0 / cyc.median()) if len(cyc) else None,
            "lead_onset_median": float(g["lead_onset"].median()),
            "lead_end_median": float(g["lead_end"].median()),
            "frac_remaining_median": float(g["frac_remaining"].median()),
            "frac_events_full_catchup": float((g["lead_end"] < 0.25 * g["lead_onset"]).mean()),
            "frac_events_overshoot_anchor": float((g["lead_end"] < 0).mean()),
            # T1: a clock would hold duration constant
            "cv_duration": robust_cv(g["duration_s"]),
            # T2: a threshold would hold the residual lead constant
            "cv_lead_end": robust_cv(g["lead_end"]),
            "cv_frac_remaining": robust_cv(g["frac_remaining"]),
            "cv_lead_onset": robust_cv(g["lead_onset"]),
            # duration prediction: arrival time vs its components
            "duration_vs_th_emp": fit_duration_model(g, "th_emp"),
            "duration_vs_lead_onset": {"rho": spearman(g["lead_onset"], g["duration_s"])},
            "duration_vs_inv_speed": {"rho": spearman(g["inv_speed"], g["duration_s"])},
            "duration_vs_width": {"rho": spearman(g["width"].fillna(g["width"].median()), g["duration_s"])},
        }
    return out


def plot_trigger(ev: pd.DataFrame, stats: dict, outdir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))

    ax = axes[0]
    m = stats["pooled"]["duration_vs_th_emp"]
    lim = float(np.percentile(ev["th_emp"], 98))
    ax.scatter(ev["th_emp"], ev["duration_s"], s=6, alpha=0.25)
    xs = np.linspace(0, lim, 50)
    ax.plot(xs, m["intercept_s"] + m["slope"] * xs, "r-", lw=2,
            label=f"OLS: {m['intercept_s']:.3f} + {m['slope']:.2f}·t_arr\n"
                  f"rho={m['rho']:.2f}")
    ax.plot(xs, xs, "k--", lw=1, label="duration = arrival time")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, float(np.percentile(ev["duration_s"], 98)))
    ax.set_xlabel("implied arrival time  lead_onset / v  (s)")
    ax.set_ylabel("fixation duration (s)")
    ax.set_title("T2: dwell tracks cursor arrival at anchor")
    ax.legend(fontsize=8)

    ax = axes[1]
    bins = np.linspace(-1.0, 1.5, 60)
    ax.hist(np.clip(ev["frac_remaining"], -1, 1.5), bins=bins, alpha=0.8,
            color="tab:blue")
    med = float(ev["frac_remaining"].median())
    ax.axvline(med, color="r", lw=2, label=f"median = {med:.2f}")
    ax.axvline(1.0, color="k", ls="--", lw=1, label="no catch-up")
    ax.axvline(0.0, color="k", lw=0.6)
    ax.set_xlabel("lead_end / lead_onset (fraction of plan left at replan)")
    ax.set_ylabel("fixation events")
    ax.set_title("Residual plan at next saccade")
    ax.legend(fontsize=8)

    ax = axes[2]
    labels = ["duration\n(T1 clock)", "lead_end\n(T2 abs threshold)",
              "frac_remaining\n(T2 rel threshold)", "lead_onset\n(reference)"]
    keys = ["cv_duration", "cv_lead_end", "cv_frac_remaining", "cv_lead_onset"]
    parts = ["A", "B", "C"]
    xpos = np.arange(len(keys))
    for i, p in enumerate(parts):
        ax.bar(xpos + (i - 1) * 0.25, [stats[p][k] for k in keys], width=0.25,
               label=p)
    ax.set_xticks(xpos, labels, fontsize=8)
    ax.set_ylabel("robust CV (IQR/|median|)")
    ax.set_title("Which quantity is held constant at replan?")
    ax.legend(title="participant", fontsize=8)

    fig.tight_layout()
    fig.savefig(outdir / "intermittency_trigger.png", dpi=150)
    plt.close(fig)


# -------------------------------------------------------------------- main

def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    samples, events = gd.load_all()
    events = add_cycle_intervals(events)
    ev = steering_events(events)
    ev = crossing_decomposition(samples, ev)

    stats = trigger_stats(ev)

    profiles = {"pooled": sawtooth_profile(samples, ev)}
    for p, g in ev.groupby("participant"):
        profiles[p] = sawtooth_profile(samples[samples["participant"] == p], g)

    plot_example_trace(samples, ev, RESULTS_DIR)
    plot_profiles(profiles, RESULTS_DIR)
    plot_trigger(ev, stats, RESULTS_DIR)

    summary = {
        "n_events_analyzed": int(len(ev)),
        "filters": {
            "steer_types": STEER_TYPES, "min_speed": MIN_SPEED,
            "min_lead": MIN_LEAD, "max_duration_s": MAX_DURATION,
            "min_samples": MIN_SAMPLES,
        },
        "trigger": stats,
        "sawtooth_profile_pooled": {
            "grid": profiles["pooled"]["grid"].tolist(),
            "median": profiles["pooled"]["median"].tolist(),
            "q25": profiles["pooled"]["q25"].tolist(),
            "q75": profiles["pooled"]["q75"].tolist(),
            "n": profiles["pooled"]["n"],
        },
    }
    with open(RESULTS_DIR / "intermittency_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(stats, indent=2))
    print(f"\nwrote results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()

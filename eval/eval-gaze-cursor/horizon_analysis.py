"""Adaptive planning-horizon analysis from gaze-cursor data.

Hypothesis: the arc-length gap between gaze anchor and cursor at fixation
onset is the behavioural planning-horizon distance. The MPCC model currently
uses a fixed preview time (lead = v * Th, Th ~= 0.3 s); this script tests that
against the data and fits the predictive horizon model

    lead = a * width^b          (fit as OLS in log-log space)

Model-selection notes (kept so the choice is reproducible):
- curvature: flat partial effect with wide CIs, and barely varies within
  trials -> dropped.
- speed: small high-speed effect, largely collinear with width (wider ->
  faster via the speed model) -> dropped; preview *time* still falls with
  speed because the horizon is a distance (Th_adaptive = lead / v).
- GAM vs power law: a monotone log-space GAM on log width matched the power
  law to three decimals (pooled log-R2 0.1276 vs 0.1274) -> the spline adds
  nothing over 5 discrete width levels; use the 2-parameter form.
- proportional (lead = k*w, the linear-noise-growth prediction): worse
  (pooled log-R2 0.092); b is significantly < 1 (pooled 0.66 [0.55, 0.75]).

Outputs under results/: evidence plots, per-participant + pooled power-law
fits and plots, and horizon_summary.json with all statistics.

Run: python3 horizon_analysis.py
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
MIN_SPEED = 0.01  # task units/s; cursor must be moving at fixation onset
MIN_LEAD = 1e-4   # log-space target: fit on forward (positive-lead) fixations
FIXED_TH = 0.3    # the current model default being tested
N_BOOT = 300


def spearman(x, y):
    return float(np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1])


def steering_events(events: pd.DataFrame) -> pd.DataFrame:
    ev = events[
        events["tunnel_type"].isin(STEER_TYPES)
        & (events["speed_onset"] > MIN_SPEED)
        & events["lead_onset"].notna()
    ].copy()
    ev["abs_curv"] = ev["curvature"].abs().fillna(0.0)
    ev["width"] = ev["width"].fillna(ev["width"].median())
    return ev


# ---------------------------------------------------------------- evidence

def evidence_stats(ev: pd.DataFrame) -> dict:
    out = {}
    for p, g in ev.groupby("participant"):
        within_w = {
            f"{w:.2f}": spearman(gg["speed_onset"], gg["lead_onset"])
            for w, gg in g.groupby(g["width"].round(2))
            if len(gg) >= 30
        }
        g2 = g.copy()
        g2["st"] = pd.qcut(g2["speed_onset"], 3, labels=False)
        within_v = {
            f"tercile_{int(t)}": spearman(gg["width"], gg["lead_onset"])
            for t, gg in g2.groupby("st")
        }
        out[p] = {
            "n_events": int(len(g)),
            "lead_onset_median": float(g["lead_onset"].median()),
            "th_emp_median": float(g["th_emp"].median()),
            "rho_lead_speed": spearman(g["speed_onset"], g["lead_onset"]),
            "rho_lead_width": spearman(g["width"], g["lead_onset"]),
            "rho_lead_abscurv": spearman(g["abs_curv"], g["lead_onset"]),
            "rho_themp_speed": spearman(g["speed_onset"], g["th_emp"]),
            "rho_lead_speed_within_width": within_w,
            "rho_lead_width_within_speed_tercile": within_v,
            "fixed_th_lead_median": float((FIXED_TH * g["speed_onset"]).median()),
        }
    return out


# --------------------------------------------------------- power-law model

def fit_horizon_power(ev: pd.DataFrame, seed: int = 0) -> tuple[dict, pd.DataFrame]:
    """lead = a * width^b, OLS in log-log space on positive-lead fixations."""
    d = ev[ev["lead_onset"] > MIN_LEAD].copy()
    lw = np.log(d["width"].clip(lower=1e-3)).to_numpy()
    ll = np.log(d["lead_onset"]).to_numpy()
    b, log_a = np.polyfit(lw, ll, 1)

    rng = np.random.default_rng(seed)
    n = len(d)
    boots = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        boots.append(np.polyfit(lw[idx], ll[idx], 1)[0])
    b_lo, b_hi = np.percentile(boots, [2.5, 97.5])

    pred = log_a + b * lw
    log_r2 = 1.0 - np.sum((ll - pred) ** 2) / np.sum((ll - ll.mean()) ** 2)
    params = {
        "a": float(np.exp(log_a)),
        "b": float(b),
        "b_ci95": [float(b_lo), float(b_hi)],
        "n_fit": n,
        "log_r2": float(log_r2),
        "rho_pred": spearman(np.exp(pred), d["lead_onset"]),
        "rho_fixed_th": spearman(FIXED_TH * d["speed_onset"], d["lead_onset"]),
    }
    return params, d


def predict_lead(params: dict, width) -> np.ndarray:
    w = np.clip(np.asarray(width, dtype=float), 1e-3, None)
    return params["a"] * w ** params["b"]


# ------------------------------------------------------------------- plots

def plot_evidence(ev: pd.DataFrame, outdir: Path):
    colors = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    for p, g in ev.groupby("participant"):
        m = g.groupby(g["width"].round(2))["lead_onset"].median()
        q1 = g.groupby(g["width"].round(2))["lead_onset"].quantile(0.25)
        q3 = g.groupby(g["width"].round(2))["lead_onset"].quantile(0.75)
        ax.plot(m.index, m.values, "o-", color=colors[p], label=p)
        ax.fill_between(m.index, q1.values, q3.values, color=colors[p], alpha=0.15)
    ax.set_xlabel("tunnel width (task units)")
    ax.set_ylabel("gaze lead at fixation onset (task units)")
    ax.set_title("Horizon distance grows with corridor width")
    ax.legend(title="participant")

    ax = axes[1]
    for p, g in ev.groupby("participant"):
        g = g.copy()
        g["sq"] = pd.qcut(g["speed_onset"], 5, labels=False, duplicates="drop")
        agg = g.groupby("sq").agg(v=("speed_onset", "median"), th=("th_emp", "median"))
        ax.plot(agg["v"], agg["th"], "o-", color=colors[p], label=p)
    ax.axhline(FIXED_TH, color="k", ls="--", lw=1, label=f"fixed Th={FIXED_TH}s")
    ax.set_xlabel("cursor speed at fixation onset (task units/s)")
    ax.set_ylabel("implied preview time lead/v (s)")
    ax.set_title("Preview time is not constant: falls with speed")
    ax.legend()

    ax = axes[2]
    order = ev.groupby("tunnel_type")["lead_onset"].median().sort_values()
    ax.barh(range(len(order)), order.values, color="tab:gray")
    ax.set_yticks(range(len(order)), order.index)
    ax.set_xlabel("median gaze lead at fixation onset (task units)")
    ax.set_title("Lead by tunnel type")

    fig.tight_layout()
    fig.savefig(outdir / "horizon_evidence.png", dpi=150)
    plt.close(fig)


def plot_fit(params: dict, d: pd.DataFrame, tag: str, outdir: Path):
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    widths = np.linspace(d["width"].min(), d["width"].max(), 100)
    ax.scatter(d["width"], d["lead_onset"], s=8, alpha=0.25, label="fixation events")
    ax.plot(widths, predict_lead(params, widths), "r-", lw=2,
            label=f"lead = {params['a']:.2f}·w^{params['b']:.2f}")
    ax.set_xlabel("tunnel width (task units)")
    ax.set_ylabel("gaze lead at fixation onset (task units)")
    ax.set_ylim(0, float(d["lead_onset"].quantile(0.98)))
    ax.set_title(f"Horizon power law ({tag})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"horizon_fit_{tag}.png", dpi=150)
    plt.close(fig)


def plot_pred_vs_obs(params: dict, d: pd.DataFrame, tag: str, outdir: Path):
    pred = predict_lead(params, d["width"])
    fixed = FIXED_TH * d["speed_onset"]
    lim = float(np.percentile(np.concatenate([d["lead_onset"], pred, fixed]), 99))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4), sharex=True, sharey=True)
    for ax, yhat, name in [
        (axes[0], pred, "horizon power law"),
        (axes[1], fixed, f"fixed Th={FIXED_TH}s (current model)"),
    ]:
        ax.scatter(d["lead_onset"], yhat, s=6, alpha=0.3)
        ax.plot([0, lim], [0, lim], "k--", lw=1)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("observed lead at fixation onset")
        ax.set_title(f"{name}  (rho={spearman(yhat, d['lead_onset']):.2f})")
    axes[0].set_ylabel("predicted lead")
    fig.tight_layout()
    fig.savefig(outdir / f"horizon_pred_vs_obs_{tag}.png", dpi=150)
    plt.close(fig)


# -------------------------------------------------------------------- main

def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    _, events = gd.load_all()
    ev = steering_events(events)

    summary = {
        "n_fixation_events_total": int(len(events)),
        "n_steering_moving_events": int(len(ev)),
        "fixed_th_tested_s": FIXED_TH,
        "model": "lead = a * width^b (log-log OLS)",
        "evidence": evidence_stats(ev),
        "fits": {},
    }

    plot_evidence(ev, RESULTS_DIR)

    for tag, sub in [("pooled", ev)] + [(p, g) for p, g in ev.groupby("participant")]:
        params, d = fit_horizon_power(sub)
        summary["fits"][tag] = params
        plot_fit(params, d, tag, RESULTS_DIR)
        plot_pred_vs_obs(params, d, tag, RESULTS_DIR)

    with open(RESULTS_DIR / "horizon_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary["fits"], indent=2))
    print(f"\nwrote results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()

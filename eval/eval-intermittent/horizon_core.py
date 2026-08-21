"""Core horizon comparison: predicted lookahead vs real gaze lead, all tasks.

Two panels (results/horizon_core.png):

  LEFT  — the full battery: all 25 steering conditions (5 tunnel families x
          5 widths), events pooled across A/B/C. Median +- IQR of the human
          gaze lead at fixation onset (blue) vs the full model's planning
          lookahead (red), condition by condition.
  RIGHT — the structural difference: lead vs width on log-log axes with
          fitted power-law exponents. Humans are sublinear (lead ~ w^0.6:
          a lookahead floor in narrow tunnels, restraint in wide ones);
          the linear difficulty budget gives ~w^1.

Run: python3 eval/eval-intermittent/horizon_core.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "eval" / "eval-gaze-cursor"))

import gaze_data as gd

RESULTS_DIR = SCRIPT_DIR / "results"

MIN_SPEED = 0.01
MIN_LEAD = 1e-4
FAMILIES = ["straight", "gentle_sinusoidal", "mid_sinusoidal",
            "sharp_sinusoidal", "corner"]
FAMILY_LABEL = {"straight": "straight", "gentle_sinusoidal": "gentle sinus.",
                "mid_sinusoidal": "mid sinus.", "sharp_sinusoidal": "sharp sinus.",
                "corner": "corner"}
TYPE_MAP = {"sinusoidal": "mid_sinusoidal"}   # model label -> human label
WIDTHS = [0.01, 0.02, 0.03, 0.04, 0.05]


def spearman(x, y):
    return float(np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1])


def load_events():
    _, events = gd.load_all()
    hev = events[
        events["tunnel_type"].isin(FAMILIES)
        & (events["speed_onset"] > MIN_SPEED)
        & (events["lead_onset"] > MIN_LEAD)
    ].copy()
    hev["width"] = hev["width"].round(3)
    hev = hev.rename(columns={"lead_onset": "lead"})

    rows = json.load(open(RESULTS_DIR / "replan_stats.json"))
    comp = pd.read_csv(RESULTS_DIR / "comparison.csv")
    tid_type = dict(zip(comp["tid"], comp["tunnel_type"]))
    recs = []
    for r in rows:
        if r["variant"] != "full":
            continue
        ttype = TYPE_MAP.get(tid_type[r["tid"]], tid_type[r["tid"]])
        if ttype not in FAMILIES:
            continue
        # drop the final 2 events per run (anchors capped at the path end)
        for lead in r["leads"][:-2]:
            if lead > MIN_LEAD:
                recs.append({"tunnel_type": ttype,
                             "width": round(float(r["width"]), 3),
                             "lead": float(lead)})
    return hev[["tunnel_type", "width", "lead"]], pd.DataFrame(recs)


def condition_stats(df):
    out = {}
    for (tt, w), g in df.groupby(["tunnel_type", "width"]):
        out[(tt, w)] = (float(g["lead"].median()),
                        float(g["lead"].quantile(0.25)),
                        float(g["lead"].quantile(0.75)),
                        int(len(g)))
    return out


def power_fit(medians_by_width):
    ws = np.array(sorted(medians_by_width))
    ms = np.array([medians_by_width[w] for w in ws])
    b, log_a = np.polyfit(np.log(ws), np.log(ms), 1)
    return float(np.exp(log_a)), float(b), ws, ms


def main():
    hev, mev = load_events()
    hstats, mstats = condition_stats(hev), condition_stats(mev)

    conds = [(tt, w) for tt in FAMILIES for w in WIDTHS
             if (tt, w) in hstats and (tt, w) in mstats]
    h_med = np.array([hstats[c][0] for c in conds])
    m_med = np.array([mstats[c][0] for c in conds])
    rho = spearman(h_med, m_med)
    ratio = float(np.median(m_med / h_med))

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.9, 1.0]})

    # ---- LEFT: all 25 conditions
    x = np.arange(len(conds))
    for src, stats, color, label in [
        ("human", hstats, "tab:blue", "human gaze lead (median, IQR)"),
        ("model", mstats, "tab:red", "model lookahead (median, IQR)"),
    ]:
        med = [stats[c][0] for c in conds]
        q25 = [stats[c][1] for c in conds]
        q75 = [stats[c][2] for c in conds]
        ax.plot(x, med, "o-", color=color, lw=1.6, ms=4, label=label)
        ax.fill_between(x, q25, q75, color=color, alpha=0.15)

    # family separators + labels (above the axes so the legend can't cover them)
    for i in range(1, len(FAMILIES)):
        ax.axvline(i * len(WIDTHS) - 0.5, color="k", lw=0.6, alpha=0.4)
    for i, tt in enumerate(FAMILIES):
        ax.text(i * len(WIDTHS) + 2.0, 1.01, FAMILY_LABEL[tt],
                transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=9)
    ax.set_xticks(x, [f"{int(w * 1000)}" for _, w in conds], fontsize=7)
    ax.set_xlabel("tunnel width (mm), grouped by tunnel family")
    ax.set_ylabel("lookahead distance (m)")
    ax.set_title(f"Planning horizon vs gaze lead — all 25 steering tasks, "
                 f"A/B/C pooled (rho={rho:.2f}, model/human={ratio:.2f})",
                 pad=22)
    ax.legend(fontsize=9, loc="upper left")

    # ---- RIGHT: scaling law (pooled across families per width)
    h_by_w = {w: np.median([hstats[(tt, w)][0] for tt in FAMILIES
                            if (tt, w) in hstats]) for w in WIDTHS}
    m_by_w = {w: np.median([mstats[(tt, w)][0] for tt in FAMILIES
                            if (tt, w) in mstats]) for w in WIDTHS}
    ha, hb, ws, hms = power_fit(h_by_w)
    ma, mb, _, mms = power_fit(m_by_w)
    wgrid = np.linspace(0.009, 0.055, 60)
    ax2.plot(ws, hms, "o", color="tab:blue", ms=7)
    ax2.plot(wgrid, ha * wgrid ** hb, "-", color="tab:blue",
             label=f"human: lead ∝ w^{hb:.2f}")
    ax2.plot(ws, mms, "o", color="tab:red", ms=7)
    ax2.plot(wgrid, ma * wgrid ** mb, "-", color="tab:red",
             label=f"model: lead ∝ w^{mb:.2f}")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xticks(WIDTHS, [f"{int(w * 1000)}" for w in WIDTHS])
    ax2.set_xlabel("tunnel width (mm, log)")
    ax2.set_ylabel("median lookahead (m, log)")
    ax2.set_title("The structural difference:\nhumans are sublinear in width")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "horizon_core.png", dpi=150)
    plt.close(fig)

    summary = {
        "n_conditions": len(conds),
        "n_human_events": int(len(hev)),
        "n_model_events": int(len(mev)),
        "rho_condition_medians_pooled": rho,
        "model_over_human_ratio_median": ratio,
        "human_width_exponent": hb,
        "model_width_exponent": mb,
        "per_family_ratio": {
            tt: float(np.median([mstats[(tt, w)][0] / hstats[(tt, w)][0]
                                 for w in WIDTHS if (tt, w) in mstats and (tt, w) in hstats]))
            for tt in FAMILIES
        },
    }
    with open(RESULTS_DIR / "horizon_core_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

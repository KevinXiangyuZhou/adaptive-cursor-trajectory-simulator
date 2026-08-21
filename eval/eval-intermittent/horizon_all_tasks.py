"""Horizon vs gaze lead across ALL tasks A/B/C performed.

Builds one figure (results/horizon_all_tasks.png):
  LEFT  — per-cell median +- IQR of human gaze lead at fixation onset vs the
          full model's planning lookahead, across the whole battery:
          25 steering cells (5 families x 5 widths), 6 two-segment cells,
          3 pointing cells (by distance, radii pooled), 9 C2U cells
          (segment width x distance, radii pooled). Log y (pointing leads
          are an order of magnitude larger than tunnel leads).
  RIGHT — the same 43 cells as a human-vs-model scatter (log-log) with the
          rank correlation.

Needs results/replan_stats.json (steering sweep) and
results/replan_stats_ext.json (run_ext_tasks.py).

Run: python3 eval/eval-intermittent/horizon_all_tasks.py
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
sys.path.insert(0, str(PROJECT_ROOT / "eval" / "eval-main"))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "eval" / "eval-gaze-cursor"))

import gaze_data as gd
import run_eval as em

RESULTS_DIR = SCRIPT_DIR / "results"
GAZE_DATA_DIR = PROJECT_ROOT / "human_data" / "gaze_cursor_data"

MIN_SPEED = 0.01
MIN_LEAD = 1e-4
STEER_FAMILY_TIDS = {
    "straight": [11, 12, 13, 14, 15],
    "gentle sinus.": [26, 27, 28, 29, 30],
    "mid sinus.": [1, 2, 3, 4, 5],
    "sharp sinus.": [31, 32, 33, 34, 35],
    "corner": [6, 7, 8, 9, 10],
}
GROUP_COLOR = {"steering": "0.25", "two-segment": "tab:purple",
               "pointing": "tab:green", "C2U": "tab:orange"}


def spearman(x, y):
    return float(np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1])


def build_cells(tid_cond):
    """Ordered list of cells: (label, group, [tids])."""
    cells = []
    for fam, tids in STEER_FAMILY_TIDS.items():
        for tid in tids:
            w = tid_cond[tid]["tunnelWidth"]
            cells.append((f"{int(round(w * 1000))}", f"steering:{fam}", [tid]))
    for tid in (36, 37, 38, 39, 40, 41):
        c = tid_cond[tid]
        lab = f"{int(c['segment1Width'] * 1000)}→{int(c['segment2Width'] * 1000)}"
        cells.append((lab, "two-segment", [tid]))
    by_d = {}
    for tid in range(42, 57):
        by_d.setdefault(round(tid_cond[tid]["distance"], 3), []).append(tid)
    for d in sorted(by_d):
        cells.append((f"D{int(round(d * 1000))}", "pointing", by_d[d]))
    by_wd = {}
    for tid in range(57, 84):
        c = tid_cond[tid]
        key = (round(c["segment1Width"], 3), round(c["distance"], 3))
        by_wd.setdefault(key, []).append(tid)
    for (w, d) in sorted(by_wd):
        cells.append((f"{int(w * 1000)}|{int(round(d * 1000))}", "C2U", by_wd[(w, d)]))
    return cells


def human_leads_by_tid():
    _, events = gd.load_all()
    ev = events[(events["speed_onset"] > MIN_SPEED)
                & (events["lead_onset"] > MIN_LEAD)]
    out = {}
    for tid, g in ev.groupby("trial_id"):
        out[int(tid)] = g["lead_onset"].to_numpy()
    return out


def model_leads_by_tid():
    out = {}
    rows = json.load(open(RESULTS_DIR / "replan_stats.json"))
    for r in rows:
        if r["variant"] != "full":
            continue
        # steering: drop the final 2 events (anchor capped at the path end)
        leads = [x for x in r["leads"][:-2] if x > MIN_LEAD]
        out.setdefault(int(r["tid"]), []).extend(leads)
    rows = json.load(open(RESULTS_DIR / "replan_stats_ext.json"))
    for r in rows:
        if r["tunnel_type"] in ("wide_to_narrow", "narrow_to_wide"):
            leads = [x for x in r["leads"][:-2] if x > MIN_LEAD]
        else:
            # pointing / C2U: the end cap IS the prediction (anchor on the
            # target); keep all positive-lead planning events
            leads = [x for x in r["leads"] if x > MIN_LEAD]
        out.setdefault(int(r["tid"]), []).extend(leads)
    return out


def main():
    tid_cond, _ = em.scan_conditions(GAZE_DATA_DIR)
    cells = build_cells(tid_cond)
    hleads = human_leads_by_tid()
    mleads = model_leads_by_tid()

    rows = []
    for label, group, tids in cells:
        h = np.concatenate([hleads.get(t, np.array([])) for t in tids])
        m = np.concatenate([np.asarray(mleads.get(t, []), dtype=float)
                            for t in tids])
        if len(h) < 5 or len(m) < 5:
            continue
        rows.append({
            "label": label, "group": group,
            "h_med": np.median(h), "h_q25": np.percentile(h, 25),
            "h_q75": np.percentile(h, 75), "n_h": len(h),
            "m_med": np.median(m), "m_q25": np.percentile(m, 25),
            "m_q75": np.percentile(m, 75), "n_m": len(m),
        })
    df = pd.DataFrame(rows)
    rho_all = spearman(df["h_med"], df["m_med"])
    ratio = float((df["m_med"] / df["h_med"]).median())

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(16.5, 5.4), gridspec_kw={"width_ratios": [2.3, 1.0]})

    x = np.arange(len(df))
    ax.plot(x, df["h_med"], "o-", color="tab:blue", lw=1.5, ms=4,
            label="human gaze lead (median, IQR)")
    ax.fill_between(x, df["h_q25"], df["h_q75"], color="tab:blue", alpha=0.15)
    ax.plot(x, df["m_med"], "o-", color="tab:red", lw=1.5, ms=4,
            label="model lookahead (median, IQR)")
    ax.fill_between(x, df["m_q25"], df["m_q75"], color="tab:red", alpha=0.15)
    ax.set_yscale("log")

    groups = df["group"].tolist()
    bounds = [i for i in range(1, len(groups)) if groups[i] != groups[i - 1]]
    for b in bounds:
        ax.axvline(b - 0.5, color="k", lw=0.6, alpha=0.4)
    start = 0
    for i, b in enumerate(bounds + [len(groups)]):
        g = groups[start]
        lab = g.replace("steering:", "")
        ax.text((start + b - 1) / 2.0, 1.01, lab,
                transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=8)
        start = b
    ax.set_xticks(x, df["label"], fontsize=6.5, rotation=45)
    ax.set_xlabel("task cell (steering: width mm | two-segment: w1→w2 | "
                  "pointing: distance | C2U: tunnel width | distance)")
    ax.set_ylabel("lookahead distance (m, log)")
    ax.set_title(f"Planning horizon vs gaze lead — every task family A/B/C "
                 f"performed (rho={rho_all:.2f}, model/human={ratio:.2f})",
                 pad=22)
    ax.legend(fontsize=9, loc="upper left")

    for _, r in df.iterrows():
        grp = r["group"].split(":")[0] if r["group"].startswith("steering") \
            else r["group"]
        ax2.scatter(r["h_med"], r["m_med"], s=44,
                    color=GROUP_COLOR["steering" if grp == "steering" else grp],
                    edgecolor="k", linewidth=0.4, zorder=3)
    lims = [df[["h_med", "m_med"]].min().min() * 0.7,
            df[["h_med", "m_med"]].max().max() * 1.4]
    ax2.plot(lims, lims, "k--", lw=1)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlim(lims); ax2.set_ylim(lims)
    for g, c in GROUP_COLOR.items():
        ax2.scatter([], [], color=c, label=g)
    ax2.set_xlabel("human median gaze lead (m, log)")
    ax2.set_ylabel("model median lookahead (m, log)")
    ax2.set_title(f"All {len(df)} task cells (rho={rho_all:.2f})")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "horizon_all_tasks.png", dpi=150)
    plt.close(fig)

    per_group = {}
    for g, gdf in df.groupby(df["group"].str.replace("steering:.*", "steering",
                                                     regex=True)):
        per_group[g] = {
            "n_cells": int(len(gdf)),
            "rho": spearman(gdf["h_med"], gdf["m_med"]) if len(gdf) > 2 else None,
            "ratio_median": float((gdf["m_med"] / gdf["h_med"]).median()),
        }
    summary = {"n_cells": int(len(df)), "rho_all": rho_all,
               "ratio_all": ratio, "per_group": per_group,
               "cells": df.to_dict("records")}
    with open(RESULTS_DIR / "horizon_all_tasks_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps({k: v for k, v in summary.items() if k != "cells"},
                     indent=1, default=float))


if __name__ == "__main__":
    main()

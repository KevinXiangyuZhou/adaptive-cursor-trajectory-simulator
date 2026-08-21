"""Aggregate the module-comparison sweep and score it against human data.

Reads results/comparison.csv + results/replan_stats.json (from run_eval.py)
and the human gaze references (eval-gaze-cursor results JSONs). Produces:

  results/summary.json           per-variant aggregate metrics
  results/kinematics.png         model-vs-human MT + metric bars
  results/process.png            replan-cycle / lookahead comparison vs gaze

Run: python3 eval/eval-intermittent/analyze.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
GAZE_RESULTS = SCRIPT_DIR.parents[0] / "eval-gaze-cursor" / "results"

VARIANT_ORDER = ["baseline", "budget", "intermittent", "full"]
COLORS = {"baseline": "tab:gray", "budget": "tab:green",
          "intermittent": "tab:orange", "full": "tab:red"}


def load_human_refs():
    inter = json.load(open(GAZE_RESULTS / "intermittency_summary.json"))["trigger"]
    horiz = json.load(open(GAZE_RESULTS / "horizon_summary.json"))["fits"]
    return inter, horiz


def kinematic_summary(df):
    out = {}
    for v, g in df.groupby("variant"):
        out[v] = {
            "n_conditions": int(len(g)),
            "timeout_rate": float(g["n_timed_out"].sum() / g["n_runs"].sum()),
            "mt_ratio_median": float((g["model_mt"] / g["human_mt"]).median()),
            "abs_mt_err_median": float(
                ((g["model_mt"] - g["human_mt"]).abs() / g["human_mt"]).median()),
            "lateral_rmse_median_mm": float(g["lateral_rmse"].median() * 1000),
            "speed_rmse_median": float(g["speed_rmse"].median()),
            "speed_corr_median": float(g["speed_corr"].median()),
            "solves_per_sim_s": float(g["solves_per_s"].median()),
            "wall_per_run_s": float(g["wall_per_run"].median()),
        }
    return out


def process_summary(replans, inter_ref):
    """Replan-cycle statistics per (variant, participant) vs gaze."""
    out = {}
    rows = pd.DataFrame(replans)
    for (v, p), g in rows.groupby(["variant", "participant"]):
        cycles = np.concatenate([np.asarray(c) for c in g["cycles"]]) if len(g) else []
        frac = np.concatenate([np.asarray(c) for c in g["frac_remaining"]]) if len(g) else []
        trig = sum((list(t) for t in g["triggers"]), [])
        n_arr = sum(1 for t in trig if t == "arrival+latency")
        n_dev = sum(1 for t in trig if t == "deviation")
        n_evt = sum(1 for t in trig if t != "init")
        href = inter_ref[p]
        out.setdefault(v, {})[p] = {
            "cycle_median_s": float(np.median(cycles)) if len(cycles) else None,
            "human_cycle_median_s": href["cycle_median_s"],
            "frac_remaining_median": float(np.median(frac)) if len(frac) else None,
            "human_frac_remaining_median": href["frac_remaining_median"],
            "overshoot_frac": float(np.mean([
                np.mean(o) for o in g["overshoot_frac"]])),
            "human_overshoot_frac": href["frac_events_overshoot_anchor"],
            "arrival_trigger_frac": n_arr / n_evt if n_evt else None,
            # model analog of 1 - frac_crossed (human ~0.2): cycles that end
            # before anchor arrival because the plan drifted
            "deviation_trigger_frac": n_dev / n_evt if n_evt else None,
            "human_noncrossed_frac": 1.0 - href["frac_crossed"],
            "n_replan_events": int(n_evt),
        }
    return out


def lead_by_width(replans):
    rows = pd.DataFrame(replans)
    out = {}
    for (v, p, w), g in rows.groupby(["variant", "participant", "width"]):
        leads = np.concatenate([np.asarray(x) for x in g["leads"]])
        leads = leads[leads > 1e-4]
        if len(leads):
            out.setdefault(v, {}).setdefault(p, {})[float(w)] = float(np.median(leads))
    return out


def plot_kinematics(df, summary, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axes[0]
    for v in VARIANT_ORDER:
        g = df[df["variant"] == v]
        ax.scatter(g["human_mt"], g["model_mt"], s=14, alpha=0.55,
                   color=COLORS[v], label=v)
    lim = max(df["human_mt"].max(), df["model_mt"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("human MT (s)"); ax.set_ylabel("model MT (s)")
    ax.set_title("Completion time per steering condition")
    ax.legend(fontsize=8)

    ax = axes[1]
    metrics = [("lateral_rmse_median_mm", "lateral RMSE (mm)"),
               ("speed_rmse_median", "speed RMSE (m/s)"),
               ("abs_mt_err_median", "|MT err| (rel)"),
               ("speed_corr_median", "speed corr")]
    x = np.arange(len(metrics))
    for i, v in enumerate(VARIANT_ORDER):
        vals = [summary[v][k] for k, _ in metrics]
        # scale lateral mm down for shared axis readability
        vals[0] = vals[0] / 10.0
        ax.bar(x + (i - 1.5) * 0.19, vals, width=0.19, color=COLORS[v], label=v)
    ax.set_xticks(x, ["lat RMSE\n(cm)", "speed\nRMSE", "|MT err|\n(rel)",
                      "speed\ncorr"], fontsize=8)
    ax.set_title("Kinematic fit vs participant rounds (medians)")
    ax.legend(fontsize=8)

    ax = axes[2]
    for i, v in enumerate(VARIANT_ORDER):
        ax.bar(i, summary[v]["solves_per_sim_s"], color=COLORS[v])
        ax.text(i, summary[v]["solves_per_sim_s"] + 0.3,
                f'{summary[v]["solves_per_sim_s"]:.1f}', ha="center", fontsize=9)
    ax.set_xticks(range(len(VARIANT_ORDER)), VARIANT_ORDER, fontsize=8)
    ax.set_ylabel("MPCC solves per simulated second")
    ax.set_title("Planning cost")

    fig.tight_layout()
    fig.savefig(outdir / "kinematics.png", dpi=150)
    plt.close(fig)


def plot_process(replans, proc, leads_w, inter_ref, horiz_ref, outdir):
    rows = pd.DataFrame(replans)
    inter_variants = [v for v in ("intermittent", "full") if v in set(rows["variant"])]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axes[0]
    data, labels, cols = [], [], []
    for v in inter_variants:
        g = rows[rows["variant"] == v]
        data.append(np.concatenate([np.asarray(c) for c in g["cycles"]]))
        labels.append(v); cols.append(COLORS[v])
    bp = ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True)
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(0.5)
    for p, href in inter_ref.items():
        if p in ("pooled",):
            continue
        ax.axhline(href["cycle_median_s"], color="k", ls=":", lw=1)
    ax.axhline(inter_ref["pooled"]["cycle_median_s"], color="k", lw=1.5,
               label="human medians (A/B/C, pooled)")
    ax.set_ylabel("replan cycle (s)")
    ax.set_title("Plan-execute-replan cycle durations")
    ax.legend(fontsize=8)

    ax = axes[1]
    for v in inter_variants:
        g = rows[rows["variant"] == v]
        frac = np.concatenate([np.asarray(c) for c in g["frac_remaining"]])
        ax.hist(np.clip(frac, -1.5, 1.5), bins=40, alpha=0.5, density=True,
                color=COLORS[v], label=f"{v} (med {np.median(frac):.2f})")
    ax.axvline(inter_ref["pooled"]["frac_remaining_median"], color="k", lw=1.5,
               label=f"human med {inter_ref['pooled']['frac_remaining_median']:.2f}")
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("residual lead at next replan / lead at onset")
    ax.set_title("Anchor overshoot (execution during latency)")
    ax.legend(fontsize=8)

    ax = axes[2]
    widths = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    for v in inter_variants + (["budget"] if "budget" in set(rows["variant"]) else []):
        med_by_w = {}
        for p, d in leads_w.get(v, {}).items():
            for w, m in d.items():
                med_by_w.setdefault(w, []).append(m)
        if not med_by_w:
            continue
        ws = sorted(med_by_w)
        ax.plot(ws, [np.median(med_by_w[w]) for w in ws], "o-",
                color=COLORS[v], label=f"{v} model lead")
    for tag, fit in horiz_ref.items():
        if tag == "pooled":
            ax.plot(widths, fit["a"] * widths ** fit["b"], "k-", lw=2,
                    label="human lead fit (pooled)")
        else:
            ax.plot(widths, fit["a"] * widths ** fit["b"], "k:", lw=0.8)
    ax.set_xlabel("tunnel width (m, task units)")
    ax.set_ylabel("lookahead / gaze lead at onset (m)")
    ax.set_title("Planning horizon vs width")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(outdir / "process.png", dpi=150)
    plt.close(fig)


def main():
    df = pd.read_csv(RESULTS_DIR / "comparison.csv")
    replans = json.load(open(RESULTS_DIR / "replan_stats.json"))
    inter_ref, horiz_ref = load_human_refs()

    summary = {
        "kinematics": kinematic_summary(df),
        "process": process_summary(replans, inter_ref),
        "lead_by_width": lead_by_width(replans),
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    plot_kinematics(df, summary["kinematics"], RESULTS_DIR)
    plot_process(replans, summary["process"], summary["lead_by_width"],
                 inter_ref, horiz_ref, RESULTS_DIR)

    print(json.dumps(summary["kinematics"], indent=1))
    for v, per_p in summary["process"].items():
        for p, s in per_p.items():
            print(f"{v:13s} {p}: cycle {s['cycle_median_s']:.3f}s "
                  f"(human {s['human_cycle_median_s']:.3f}) "
                  f"overshoot {s['overshoot_frac']:.2f} "
                  f"(human {s['human_overshoot_frac']:.2f}) "
                  f"frac_rem {s['frac_remaining_median']:.2f} "
                  f"(human {s['human_frac_remaining_median']:.2f}) "
                  f"arrival% {100 * (s['arrival_trigger_frac'] or 0):.0f}")


if __name__ == "__main__":
    main()

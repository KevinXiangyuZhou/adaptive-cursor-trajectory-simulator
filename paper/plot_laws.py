"""Paper figures: steering-law and Fitts'-law regressions, Human vs. Model.

Reads the pooled-8 main eval (results-cluster-10p/eval-main-pooled8-local) and
regenerates the two law plots at publication quality (single-column CHI size,
PDF + PNG). Aggregation matches the eval pipeline exactly: one point per task
condition (trial id), flat pooled mean across all participants and rounds;
linear fit MT = a + b*ID per source. Fitts uses the *aligned* kinematic
movement time (final target entry - movement onset), which strips human
reaction/click latency and model dwell; steering uses the full trial time.

The curvature-aware steering variant replaces ID = L/W with
ID_k = int (1 + lam*|kappa|)/W ds = (L + lam*PHI)/W, where PHI is the
centerline's total absolute turning (rad), lam (m/rad) converts turning into
extra effective length, and the single lam is fit on the human condition
means (max linear-fit R^2), then applied unchanged to the model. The
curvature-width interaction follows Chen & Fels, "Curves Ahead" (CHI 2025).

Usage:  python paper/plot_laws.py [--eval pooled8|perpid]
Writes: paper/figures/{steering_law, steering_law_budget, fitts_law}{,_perpid}.{pdf,png}

--eval perpid swaps both sources for the per-participant runs (each
participant simulated with their own fitted persona) and suffixes the output
stems with _perpid; the aggregation is unchanged (flat pooled mean per
condition), only the personas behind the Simulator rows differ.
"""

import argparse
import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "results-cluster-10p" / "runs"
# (model eval dir, baseline eval dir, output-stem suffix) per --eval choice.
# Baseline runs: CHI-26-EA persona (Simulator rows = baseline model; its
# Human rows are dropped — the model run's serve both sources).
EVAL_SOURCES = {
    "pooled8": (REPO / "results-cluster-10p" / "eval-main-pooled8-local",
                RUNS / "baseline-full-pooled8-s42-20260908-1543-1adbed9" / "eval",
                ""),
    "perpid": (RUNS / "mpcc-full-perpid-s42-20260909-1257-22c349c" / "eval",
               RUNS / "baseline-full-perpid-s42-20260908-1800-c96338d" / "eval",
               "_perpid"),
}
PANEL_TITLES = {"perpid": "Per-participant fits", "pooled8": "Pooled fit"}
OUT_DIR = Path(__file__).resolve().parent / "figures"

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "hcs_package" / "src"))
from experiment.environment import create_environment  # noqa: E402

# Okabe-Ito blue / vermillion — CVD-validated pair (ΔE 21.9 protan, 31.2 normal);
# neutral gray for the baseline so the two focal series keep the contrast.
COLORS = {"Human": "#0072B2", "Ours": "#D55E00", "Baseline": "#8C8C8C"}
MARKERS = {"Human": "o", "Ours": "^", "Baseline": "s"}  # shape = secondary identity encoding
SOURCES = ("Human", "Ours", "Baseline")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,  # embed TrueType so text stays editable/searchable
    "ps.fonttype": 42,
})


def _iter_rows(csv_path, mt_col, sim_label):
    """Yield (source, tid, ID, MT) rows; Simulator rows get sim_label."""
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r.get("timed_out") == "True":
                continue
            mt = r.get(mt_col)
            if not mt:
                continue
            src = sim_label if r["source"] == "Simulator" else "Human"
            yield src, r["tid"], float(r["ID"]), float(mt)


def load_condition_means(csv_path, mt_col, baseline_csv=None):
    """Collapse per-round rows to one (ID, mean MT) point per (source, tid).

    Flat pooled mean across participants and rounds — identical to the eval
    pipeline's steering_law_plot / fitts _collapse_by_tid aggregation.
    With baseline_csv, its Simulator rows are added as a "Baseline" source
    (its Human rows are dropped — verified identical to csv_path's).
    """
    per = defaultdict(lambda: ([], []))  # (src, tid) -> (IDs, MTs)
    for src, tid, id_, mt in _iter_rows(csv_path, mt_col, "Ours"):
        per[(src, tid)][0].append(id_)
        per[(src, tid)][1].append(mt)
    if baseline_csv is not None:
        for src, tid, id_, mt in _iter_rows(baseline_csv, mt_col, "Baseline"):
            if src == "Baseline":
                per[(src, tid)][0].append(id_)
                per[(src, tid)][1].append(mt)
    out = defaultdict(lambda: ([], []))
    for (src, tid), (ids, mts) in per.items():
        out[src][0].append(float(np.mean(ids)))
        out[src][1].append(float(np.mean(mts)))
    return {s: (np.array(x), np.array(y)) for s, (x, y) in out.items()}


def load_mt_means_by_tid(csv_path, mt_col, baseline_csv=None):
    """Same aggregation as load_condition_means, but keyed by tid so the
    per-condition MT means can be joined with tunnel geometry."""
    per = defaultdict(list)
    for src, tid, _, mt in _iter_rows(csv_path, mt_col, "Ours"):
        per[(src, tid)].append(mt)
    if baseline_csv is not None:
        for src, tid, _, mt in _iter_rows(baseline_csv, mt_col, "Baseline"):
            if src == "Baseline":
                per[(src, tid)].append(mt)
    srcs = sorted({s for s, _ in per}, key=SOURCES.index)
    return {src: {tid: float(np.mean(v)) for (s, tid), v in per.items() if s == src}
            for src in srcs}


def steering_geometry():
    """Per-tid tunnel geometry {tid: (W, L, PHI)} with PHI = int |kappa| ds
    (total absolute turning of the centerline, rad). Conditions are read from
    a raw experiment file; centerlines are regenerated by the same
    create_environment call the eval pipeline uses."""
    raw = sorted(glob.glob(str(REPO / "human_data" / "task_aligned_all" / "*.json")))[0]
    data = json.loads(Path(raw).read_text())
    sessions = data.get("sessions") or [{"trialData": data.get("trialData", [])}]
    geoms = {}
    for s in sessions:
        for td in s.get("trialData", []):
            tid, c = td.get("trial_id"), td.get("condition", {})
            if tid is None or str(tid) in geoms:
                continue
            if "tunnelWidth" not in c or "distance" in c or "segment1Width" in c:
                continue  # not a Steering-bucket condition
            if c.get("tunnelType") == "corner":
                env = {"env_type": "tunnel_steering_corner",
                       "num_corners": c["numCorners"], "corner_offset": c["cornerOffset"]}
            else:
                env = {"env_type": "tunnel_steering_smooth",
                       "curvature": c.get("curvature", 0.0)}
            env.update({"screen_width": 460, "screen_height": 260,
                        "tunnelWidth": c["tunnelWidth"], "max_steps": 800,
                        "target_radius": c["tunnelWidth"] * 0.5})
            cl = np.asarray(create_environment(env)["centerline"], float)
            seg = np.diff(cl, axis=0)
            ds = np.hypot(seg[:, 0], seg[:, 1])
            seg = seg[ds > 1e-9]
            th = np.arctan2(seg[:, 1], seg[:, 0])
            phi = float(np.abs(np.angle(np.exp(1j * np.diff(th)))).sum())
            geoms[str(tid)] = (c["tunnelWidth"], float(ds.sum()), phi)
    return geoms


def budget_id_data(mt_by_tid, geoms):
    """Curvature-aware ID_k = int (1 + lam*|kappa|)/W ds = (L + lam*PHI)/W
    per condition: total turning PHI converts to extra effective length at
    lam metres per radian, and the whole path is width-normalized — the
    curvature-width interaction form suggested by the L*K term of Chen &
    Fels, "Curves Ahead" (CHI 2025, doi 10.1145/3706598.3713102). The single
    lam maximizes the linear-fit R^2 on the *human* condition means and is
    applied unchanged to the model. (The additive variant L/W + lam*PHI
    scores human R^2 0.84 vs 0.91 for this form.)"""
    tids = sorted(mt_by_tid["Human"], key=int)
    w = np.array([geoms[t][0] for t in tids])
    length = np.array([geoms[t][1] for t in tids])
    phi = np.array([geoms[t][2] for t in tids])
    y_h = np.array([mt_by_tid["Human"][t] for t in tids])
    lams = np.linspace(0.0, 1.0, 10001)  # m/rad
    lam = float(lams[np.argmax([np.corrcoef((length + l * phi) / w, y_h)[0, 1] ** 2
                                for l in lams])])
    data = {src: ((length + lam * phi) / w,
                  np.array([mt_by_tid[src][t] for t in tids]))
            for src in mt_by_tid}
    return data, lam


def fit_line(x, y):
    b, a = np.polyfit(x, y, 1)
    r2 = np.corrcoef(x, y)[0, 1] ** 2
    return a, b, r2


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.88", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5)


def draw_law(ax, data, xlabel, unit, ylabel=True, legend=True):
    fits = {}
    line_lo = 0.0
    srcs = [s for s in SOURCES if s in data]
    for src in srcs:
        x, y = data[src]
        a, b, r2 = fit_line(x, y)
        fits[src] = (a, b, r2)
        xs = np.linspace(x.min(), x.max(), 2)
        ax.plot(xs, a + b * xs, color=COLORS[src], linewidth=1.2, zorder=2)
        ax.scatter(x, y, s=14, marker=MARKERS[src], color=COLORS[src],
                   linewidths=0, alpha=0.85, zorder=3, label=src)
        line_lo = min(line_lo, float((a + b * xs).min()))
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel("Movement time (s)")
    # Keep the full regression lines visible: only clamp the bottom at 0 when
    # no line dips below it; otherwise pad below the lowest line endpoint.
    ax.set_ylim(bottom=line_lo if line_lo == 0.0 else line_lo - 0.3)
    if line_lo < 0.0:
        ax.axhline(0.0, color="0.75", linewidth=0.5, zorder=1)
    style_axes(ax)

    lines = [
        f"{src}: MT = {fits[src][0]:.2f} + {fits[src][1]:.3f} {unit}"
        f"  ($R^2$ = {fits[src][2]:.2f})"
        for src in srcs
    ]
    ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, fontsize=6,
            va="top", ha="left", linespacing=1.5)
    if legend:
        ax.legend(loc="lower right", frameon=False, handletextpad=0.2,
                  borderaxespad=0.2)
    return fits


def save(fig, stem):
    OUT_DIR.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        p = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"saved {p.relative_to(REPO)}")
    plt.close(fig)


LAW_SPECS = {   # law -> (data key, xlabel, unit)
    "steering_law": ("steer", "Index of difficulty $L/W$", "ID"),
    "steering_law_budget": ("budget",
                            r"Curvature-aware ID  $\int (1 + \lambda|\kappa|)/W\,ds$",
                            "ID"),
    "fitts_law": ("fitts", r"Index of difficulty $\log_2(D/W+1)$", "ID"),
}


def load_law_data(eval_key, geoms):
    """All three laws' condition-mean data for one eval source."""
    eval_dir, baseline_dir, _ = EVAL_SOURCES[eval_key]
    steer = load_condition_means(eval_dir / "Steering" / "steering_results.csv", "MT_s",
                                 baseline_dir / "Steering" / "steering_results.csv")
    mt_by_tid = load_mt_means_by_tid(eval_dir / "Steering" / "steering_results.csv", "MT_s",
                                     baseline_dir / "Steering" / "steering_results.csv")
    budget, lam = budget_id_data(mt_by_tid, geoms)
    fitts = load_condition_means(eval_dir / "Fitts" / "fitts_results.csv", "MT_kin_s",
                                 baseline_dir / "Fitts" / "fitts_results.csv")
    return {"steer": steer, "budget": budget, "lam": lam, "fitts": fitts}


def report_fits(label, fits, unit="ID"):
    for s, (a_, b, r2) in fits.items():
        print(f"  {label} {s}: MT = {a_:.3f} + {b:.4f} {unit}, R2 = {r2:.3f}")


def check_fitts_regression(eval_key, fits):
    """Cross-check the Fitts refit against the eval pipeline's stored regression."""
    ref = json.loads((EVAL_SOURCES[eval_key][0] / "Fitts"
                      / "fitts_regression.json").read_text())
    for src, key in (("Human", "human"), ("Ours", "model")):
        a_, b, r2 = fits[src]
        ra, rb, rr2 = (ref["aligned"][key][k] for k in
                       ("a_intercept", "b_slope_s_per_bit", "r_squared"))
        ok = abs(a_ - ra) < 1e-3 and abs(b - rb) < 1e-3 and abs(r2 - rr2) < 1e-3
        print(f"  fitts [{eval_key}] {src} vs fitts_regression.json[aligned]: "
              f"{'MATCH' if ok else f'MISMATCH (json: a={ra}, b={rb}, R2={rr2})'}")


def lam_note(ax, lam):
    ax.text(0.03, 0.70, rf"$\lambda$ = {lam:.2f} m/rad (fit on human)",
            transform=ax.transAxes, fontsize=6, va="top", ha="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="pooled8",
                    choices=sorted(EVAL_SOURCES) + ["both"],
                    help="which eval feeds the Simulator rows: the pooled-8 "
                         "personas (default), the per-participant fits, or "
                         "'both' — one figure per law with two panels "
                         "(per-participant | pooled), stems suffixed _compare")
    a = ap.parse_args()

    if a.eval == "both":
        geoms = steering_geometry()
        keys = ("perpid", "pooled8")
        data = {k: load_law_data(k, geoms) for k in keys}
        for k in keys:
            print(f"[{k}] model: {EVAL_SOURCES[k][0].relative_to(REPO)}")
        for law, (dkey, xlabel, unit) in LAW_SPECS.items():
            fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5))
            for i, k in enumerate(keys):
                fits = draw_law(axes[i], data[k][dkey], xlabel, unit,
                                ylabel=(i == 0), legend=(i == 1))
                if dkey == "budget":
                    lam_note(axes[i], data[k]["lam"])
                axes[i].set_title(PANEL_TITLES[k])
                report_fits(f"{law} [{k}]", fits, unit)
                if dkey == "fitts":
                    check_fitts_regression(k, fits)
            # shared y scale so the two panels compare directly
            lo = min(ax.get_ylim()[0] for ax in axes)
            hi = max(ax.get_ylim()[1] for ax in axes)
            for ax in axes:
                ax.set_ylim(lo, hi)
            fig.tight_layout()
            save(fig, f"{law}_compare")
        return

    EVAL_DIR, BASELINE_DIR, sfx = EVAL_SOURCES[a.eval]
    print(f"eval: {a.eval}\n  model:    {EVAL_DIR}\n  baseline: {BASELINE_DIR}")
    data = load_law_data(a.eval, steering_geometry())
    for law, (dkey, xlabel, unit) in LAW_SPECS.items():
        fig, ax = plt.subplots(figsize=(3.4, 2.5))
        fits = draw_law(ax, data[dkey], xlabel, unit)
        if dkey == "budget":
            lam_note(ax, data["lam"])
            print(f"  budget lam = {data['lam']:.4f} m/rad (fit on human condition means)")
        save(fig, f"{law}{sfx}")
        report_fits(law, fits, unit)
        if dkey == "fitts":
            check_fitts_regression(a.eval, fits)


if __name__ == "__main__":
    main()

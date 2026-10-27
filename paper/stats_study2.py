"""Section 5.4 (generalization to unseen users): every number, Table 3 and
Figure 5 for the Study 2 evaluation of the POOLED train-all models
(fitted jointly on the 8 Study 1 participants, 12 h CMA-ES budget) on the
14 Study 2 participants they never saw.

Sources (results-cluster-10p mirror of the cluster runs):
  Study 2 model     runs-14p/eval14p-mpcc-pooled8all-c70a5dd-s42-20260911-1709-ca2b4a7
  Study 2 baseline  runs-14p/eval14p-baseline-pooled8all-c70a5dd-s42-20260911-1709-ca2b4a7
  Study 1 reference runs/mpcc-full-perpid-s42-20260909-1257-22c349c (+ its
                    baseline), the per-participant runs behind Figure 3 /
                    Tables 1-2, train widths {10, 50} mm, test {12.5, 16.5, 25}

Timing: one point per Study 2 condition (mean over the 14 participants and
their rounds, as plot_laws.py does for Study 1), curvature-aware
ID_k = (L + lam*PHI)/W with lam refit on the Study 2 human condition means
(as Figure 3a fits it on Study 1's; both values reported), linear fits MT = a + b*ID_k for the human,
model and baseline; slope agreement and mean absolute error over conditions.
Spread: SD of per-participant mean completion time across the 14
participants per condition, against the pooled model's run-to-run SD (5
runs per condition, 3 for straight).
Trajectories (Table 3): the four Section 5.3 metrics per participant x
condition pair from steering_condition_summary.csv (lateral RMSE, speed RMSE,
speed correlation, relative time difference), mean +/- SD by corridor type,
for the Study 1 test split (per-participant model) and Study 2 (pooled
model, pooled baseline); one-sided paired Wilcoxon over participants
(n = 14) model vs baseline.
Figure 5: paper/figures/steering_law_budget_study2.{pdf,png}, the Study 2
counterpart of Figure 3a (same drawing code).

Usage: python paper/stats_study2.py
Writes: paper/quant_study2/{stats.txt, table3_rows.tex, conditions.csv}
"""
import csv
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "paper"))
import plot_laws as pl  # noqa: E402  (figure style, ID_k, fits; sets Agg backend)
from experiment.environment import create_environment  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

RUNS = REPO / "results-cluster-10p"
S2_MODEL = RUNS / "runs-14p" / "eval14p-mpcc-pooled8all-c70a5dd-s42-20260911-1709-ca2b4a7" / "eval"
S2_BASE = RUNS / "runs-14p" / "eval14p-baseline-pooled8all-c70a5dd-s42-20260911-1709-ca2b4a7" / "eval"
S1_MODEL = RUNS / "runs" / "mpcc-full-perpid-s42-20260909-1257-22c349c" / "eval"
S1_BASE = RUNS / "runs" / "baseline-full-perpid-s42-20260908-1800-c96338d" / "eval"
S1_TRAIN_WIDTHS_MM = {10.0, 50.0}          # the split of that run (pre 2026-09-10)
S2_RAW = REPO / "eval-14p" / "human_data" / "raw"
S1_RAW = REPO / "human_data" / "task_aligned_all"
S1_HUMAN_SLOPE_PAPER = 0.059               # Figure 3a, verified by plot_laws --eval perpid
OUT = REPO / "paper" / "quant_study2"
TYPE_ORDER = ["Straight", "Corner", "Sinusoidal", "Gentle sinusoidal", "Sharp sinusoidal"]
TYPE_LABEL = {"straight": "Straight", "corner": "Corner", "sinusoidal": "Sinusoidal",
              "gentle_sinusoidal": "Gentle sinusoidal", "sharp_sinusoidal": "Sharp sinusoidal"}
S1_TYPE_BY_TID = {}
for lo, name in ((1, "Sinusoidal"), (6, "Corner"), (11, "Straight"),
                 (26, "Gentle sinusoidal"), (31, "Sharp sinusoidal")):
    for t in range(lo, lo + 5):
        S1_TYPE_BY_TID[t] = name

_log = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    _log.append(s)


# ---------------------------------------------------------------------------
# geometry: tid -> (W, L, PHI, type) from a raw data dir, centerlines regenerated
# by the eval pipeline's create_environment (same as plot_laws.steering_geometry)
# ---------------------------------------------------------------------------
def geometry(raw_dir):
    raw = sorted(glob.glob(str(raw_dir / "*.json")))[0]
    data = json.loads(Path(raw).read_text())
    sessions = data.get("sessions") or [{"trialData": data.get("trialData", [])}]
    geoms = {}
    for s in sessions:
        for td in s.get("trialData", []):
            tid, c = td.get("trial_id"), td.get("condition", {})
            if tid is None or str(tid) in geoms or "tunnelWidth" not in c \
                    or "distance" in c or "segment1Width" in c:
                continue
            if c.get("tunnelType") == "corner":
                env = {"env_type": "tunnel_steering_corner",
                       "num_corners": c["numCorners"], "corner_offset": c["cornerOffset"]}
            else:
                env = {"env_type": "tunnel_steering_smooth", "curvature": c.get("curvature", 0.0)}
            env.update({"screen_width": 460, "screen_height": 260, "tunnelWidth": c["tunnelWidth"],
                        "max_steps": 800, "target_radius": c["tunnelWidth"] * 0.5})
            cl = np.asarray(create_environment(env)["centerline"], float)
            seg = np.diff(cl, axis=0)
            ds = np.hypot(seg[:, 0], seg[:, 1])
            seg = seg[ds > 1e-9]
            th = np.arctan2(seg[:, 1], seg[:, 0])
            phi = float(np.abs(np.angle(np.exp(1j * np.diff(th)))).sum())
            ttype = c.get("tunnelType") or ("straight" if float(c.get("curvature", 0) or 0) == 0 else "sinusoidal")
            geoms[str(tid)] = (c["tunnelWidth"], float(ds.sum()), phi, TYPE_LABEL[ttype])
    return geoms


def id_kappa(geoms, tids, lam):
    return np.array([(geoms[t][1] + lam * geoms[t][2]) / geoms[t][0] for t in tids])


# ---------------------------------------------------------------------------
# per-round rows -> per (source, pid, tid) lists of MT
# ---------------------------------------------------------------------------
def runs_by_pid_tid(csv_path, sim_label):
    per = defaultdict(list)
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r.get("timed_out") == "True":
                continue
            src = sim_label if r["source"] == "Simulator" else "Human"
            per[(src, r["pid"], r["tid"])].append(float(r["MT_s"]))
    return per


def spread_stats(per, src):
    """(mean over conditions of the SD across participants of per-participant
    mean MT, mean run-to-run SD within participant x condition, n pids)."""
    by_tid = defaultdict(dict)
    within = []
    for (s, pid, tid), v in per.items():
        if s != src:
            continue
        by_tid[tid][pid] = float(np.mean(v))
        if len(v) > 1:
            within.append(float(np.std(v, ddof=1)))
    across = [float(np.std(list(d.values()), ddof=1)) for d in by_tid.values() if len(d) > 1]
    n_pid = max(len(d) for d in by_tid.values())
    return float(np.mean(across)), float(np.mean(within)), n_pid, by_tid


# ---------------------------------------------------------------------------
# condition-summary metrics (one row per participant x condition)
# ---------------------------------------------------------------------------
METRICS = (("lat_mm", "lateral_rmse", 1000.0, 1), ("spd", "speed_rmse", 1.0, 3),
           ("corr", "speed_corr", 1.0, 2), ("tdiff_pct", "time_diff", 100.0, 1))


def load_summary(csv_path, type_of_tid, keep=lambda r: True, pid_of=None):
    """rows: dicts with type, width, pid (if the summary lacks a pid column
    the per-pid Wilcoxon uses the participant folders' order — the eval
    writes one row per participant x tid in participant order)."""
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if not keep(r):
                continue
            d = {"tid": int(r["tid"]), "type": type_of_tid(int(r["tid"])), "width": float(r["width_mm"]),
                 "ratio": float(r["model_time_mean_s"]) / float(r["human_time_mean_s"]),
                 "abs_err": abs(float(r["model_time_mean_s"]) - float(r["human_time_mean_s"])),
                 "n_timed_out": int(r["n_timed_out"])}
            for k, col, scale, _ in METRICS:
                d[k] = float(r[col]) * scale
            rows.append(d)
    return rows


def cell(vals, prec):
    v = np.asarray(vals, float)
    return f"${np.mean(v):.{prec}f} \\pm {np.std(v):.{prec}f}$"


def group(rows, ttype):
    return [r for r in rows if ttype == "All" or r["type"] == ttype]


def rows_by_pid(rows, csv_path, truth=None, pids=None):
    """The condition summary has no participant column: identify each row's
    participant by its human mean completion time within the condition
    (unique to < 1 ms across participants; asserted), matched against the
    same run's per-round file steering_results.csv, which carries pids."""
    per = runs_by_pid_tid(Path(csv_path).parent / "steering_results.csv", "x")
    truth = {(p, t): float(np.mean(v)) for (s_, p, t), v in per.items() if s_ == "Human"}
    pids = sorted({p for p, _ in truth})
    out = defaultdict(list)
    raw = list(csv.DictReader(open(csv_path)))
    assert len(raw) == len(rows)
    for r, rr in zip(rows, raw):
        tid = str(r["tid"]); h = float(rr["human_time_mean_s"])
        hits = [p for p in pids if (p, tid) in truth and abs(truth[(p, tid)] - h) < 5e-4]
        assert len(hits) == 1, f"tid {tid}: {len(hits)} participants match human mean {h}"
        out[hits[0]].append(r)
    n = Counter(len(v) for v in out.values())
    if len(n) > 1:   # a run may lack conditions for a participant (worker time-out in its eval)
        print(f"  note: {Path(csv_path).parts[-4]}: rows per participant {dict(n)}")
    return out


def per_pid_means(rows_by_pid, key):
    return np.array([np.mean([r[key] for r in rows]) for _, rows in sorted(rows_by_pid.items())])


def main():
    OUT.mkdir(exist_ok=True)
    say("=== sources ===")
    for k, v in (("Study 2 model", S2_MODEL), ("Study 2 baseline", S2_BASE),
                 ("Study 1 model", S1_MODEL), ("Study 1 baseline", S1_BASE)):
        say(f"  {k:17s} {v.relative_to(REPO)}")

    # ---- Study 1 reference: lam, fits (must match Figure 3a) -----------------
    g1 = geometry(S1_RAW)
    mt1 = pl.load_mt_means_by_tid(S1_MODEL / "Steering" / "steering_results.csv", "MT_s",
                                  S1_BASE / "Steering" / "steering_results.csv")
    g1_pl = {t: v[:3] for t, v in g1.items()}
    d1, lam = pl.budget_id_data(mt1, g1_pl)
    say(f"\n=== Study 1 (per-participant fits) with lam = {lam:.4f} m/rad fit on Study 1 human ===")
    s1_fits = {s: pl.fit_line(*d1[s]) for s in d1}
    for s, (a, b, r2) in s1_fits.items():
        say(f"  {s:8s} MT = {a:.3f} + {b:.4f} ID_k  (R2 {r2:.3f})")
    assert abs(s1_fits["Human"][1] - S1_HUMAN_SLOPE_PAPER) < 5e-4, "Study 1 human slope no longer 0.059"

    # ---- Study 2 timing --------------------------------------------------------
    g2 = geometry(S2_RAW)
    mt2 = pl.load_mt_means_by_tid(S2_MODEL / "Steering" / "steering_results.csv", "MT_s",
                                  S2_BASE / "Steering" / "steering_results.csv")
    tids = sorted(mt2["Human"], key=int)
    # lam refit on the Study 2 human condition means, as Figure 3a fits it on
    # Study 1's (max linear-fit R^2); the Study 1 value is kept for reference
    lam1 = lam
    g2_pl = {t: v[:3] for t, v in g2.items()}
    d2, lam = pl.budget_id_data(mt2, g2_pl)
    x = d2["Human"][0]
    say(f"\n=== Study 2 timing: {len(tids)} conditions, lam = {lam:.4f} m/rad fit on Study 2 human (Study 1: {lam1:.4f}) ===")
    say(f"  ID_k range {x.min():.1f} .. {x.max():.1f} (Study 1: {d1['Human'][0].min():.1f} .. {d1['Human'][0].max():.1f})")
    for s_ in ("Human", "Ours", "Baseline"):
        a_, b_, r2_ = pl.fit_line(id_kappa(g2, tids, lam1), np.array([mt2[s_][t] for t in tids]))
        say(f"  (with Study 1 lam {lam1:.2f} fixed) {s_:8s} MT = {a_:.3f} + {b_:.4f} ID_k  (R2 {r2_:.3f})")
    s2_fits = {s: pl.fit_line(*d2[s]) for s in d2}
    for s, (a, b, r2) in s2_fits.items():
        say(f"  {s:8s} MT = {a:.3f} + {b:.4f} ID_k  (R2 {r2:.3f})")
    bh1, bh2 = s1_fits["Human"][1], s2_fits["Human"][1]
    say(f"  Study 2 vs Study 1 human slope: {bh2:.4f} vs {bh1:.4f} -> {100 * abs(bh2 - bh1) / bh1:.1f}% apart")
    say(f"  Study 2 vs Study 1 human intercept: {s2_fits['Human'][0]:.3f} vs {s1_fits['Human'][0]:.3f}")
    yh = d2["Human"][1]
    for s in ("Ours", "Baseline"):
        a, b, r2 = s2_fits[s]
        ym = d2[s][1]
        mae = float(np.mean(np.abs(ym - yh)))
        say(f"  {s:8s} slope within {100 * abs(b - bh2) / bh2:.1f}% of Study 2 human; intercept diff {a - s2_fits['Human'][0]:+.2f} s; "
            f"MAE over conditions {mae:.2f} s = {100 * mae / yh.mean():.1f}% of mean human MT ({yh.mean():.2f} s); "
            f"mean per-condition rel. error {100 * np.mean(np.abs(ym - yh) / yh):.1f}%; mean MT ratio {np.mean(ym / yh):.2f}")
        s1a, s1b, _ = s1_fits[s]
        say(f"           Study 1 counterpart: slope {100 * abs(s1b - bh1) / bh1:.1f}% off human, intercept diff {s1a - s1_fits['Human'][0]:+.2f} s")
    # per width and per type, human vs pooled model condition means
    say("  per width (human MT | model MT | ratio | baseline MT):")
    for w in sorted({g2[t][0] for t in tids}):
        sel = [i for i, t in enumerate(tids) if g2[t][0] == w]
        say(f"    W={w * 1000:4.0f} mm  {yh[sel].mean():5.2f} | {d2['Ours'][1][sel].mean():5.2f} | "
            f"{np.mean(d2['Ours'][1][sel] / yh[sel]):.2f} | {d2['Baseline'][1][sel].mean():5.2f}")
    say("  per type (human MT | model MT | ratio | baseline MT):")
    for ty in TYPE_ORDER:
        sel = [i for i, t in enumerate(tids) if g2[t][3] == ty]
        say(f"    {ty:18s} {yh[sel].mean():5.2f} | {d2['Ours'][1][sel].mean():5.2f} | "
            f"{np.mean(d2['Ours'][1][sel] / yh[sel]):.2f} | {d2['Baseline'][1][sel].mean():5.2f}")
    with open(OUT / "conditions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tid", "type", "width_mm", "L_m", "PHI_rad", "ID_k", "human_MT", "model_MT", "baseline_MT"])
        for i, t in enumerate(tids):
            w.writerow([t, g2[t][3], g2[t][0] * 1000, f"{g2[t][1]:.4f}", f"{g2[t][2]:.4f}", f"{x[i]:.3f}",
                        f"{yh[i]:.4f}", f"{d2['Ours'][1][i]:.4f}", f"{d2['Baseline'][1][i]:.4f}"])

    # ---- Figure 5 --------------------------------------------------------------
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    fits = pl.draw_law(ax, d2, r"Curvature-aware ID  $\int (1 + \lambda|\kappa|)/W\,ds$", "ID")
    # headroom: the baseline's slowest conditions (~18 s) would otherwise sit
    # under the equation box at the top left
    ax.set_ylim(top=1.5 * max(y.max() for _, y in d2.values()))
    ax.text(0.03, 0.70, rf"$\lambda$ = {lam:.2f} m/rad (fit on human)",
            transform=ax.transAxes, fontsize=6, va="top", ha="left")
    pl.save(fig, "steering_law_budget_study2")
    for s in fits:
        assert np.allclose(fits[s], s2_fits[s]), "figure fits differ from reported fits"

    # ---- spread ----------------------------------------------------------------
    say("\n=== Study 2 spread of completion time ===")
    per_m = runs_by_pid_tid(S2_MODEL / "Steering" / "steering_results.csv", "Ours")
    per_b = runs_by_pid_tid(S2_BASE / "Steering" / "steering_results.csv", "Baseline")
    h_across, h_within, n_h, h_by_tid = spread_stats(per_m, "Human")
    m_across, m_within, n_m, _ = spread_stats(per_m, "Ours")
    b_across, b_within, n_b, _ = spread_stats(per_b, "Baseline")
    say(f"  human ({n_h} participants): SD across participants of per-participant mean MT, mean over conditions = {h_across:.2f} s; "
        f"within-participant round-to-round SD = {h_within:.2f} s")
    # the pooled persona is simulated with the same seeded noise sequence for
    # every participant (identical draws per condition, across-participant SD
    # of the model is exactly 0), so the model's spread is its run-to-run SD
    # over the 5 draws (3 for straight) per condition
    say(f"  pooled model: run-to-run SD within a condition = {m_within:.2f} s (identical draws for every participant: "
        f"across-participant SD {m_across:.2f} s) -> human across-participant SD / model run-to-run SD = {h_across / m_within:.1f}x")
    say(f"  pooled baseline: run-to-run SD = {b_within:.2f} s")
    # per-type human spread for the text
    for ty in TYPE_ORDER:
        sd = [np.std(list(h_by_tid[t].values()), ddof=1) for t in h_by_tid if g2[t][3] == ty]
        sdm = [np.std(v, ddof=1) for (s, p, t), v in per_m.items() if s == "Ours" and g2[t][3] == ty and len(v) > 1]
        say(f"    {ty:18s} human SD across participants {np.mean(sd):.2f} s | model run-to-run SD {np.mean(sdm):.2f} s")
    # Study 1 reference spread (8 participants, per-participant models): same statistic
    per1 = runs_by_pid_tid(S1_MODEL / "Steering" / "steering_results.csv", "Ours")
    h1a, h1w, n1, _ = spread_stats(per1, "Human")
    m1a, m1w, _, _ = spread_stats(per1, "Ours")
    say(f"  Study 1 reference ({n1} participants): human SD across participants {h1a:.2f} s, per-participant model SD across participants {m1a:.2f} s, "
        f"model run-to-run SD {m1w:.2f} s")

    # ---- Table 3 ---------------------------------------------------------------
    say("\n=== Table 3: trajectory metrics (mean +/- SD over participant x condition rows) ===")
    # Study 1 reference: the per-participant models over ALL their conditions
    # (Table 1 "Model (Overall)"); used only as the reference row of the All block
    s1_rows = load_summary(S1_MODEL / "Steering" / "steering_condition_summary.csv", S1_TYPE_BY_TID.get)
    s2_type = {int(t): g2[t][3] for t in g2}
    s2m_rows = load_summary(S2_MODEL / "Steering" / "steering_condition_summary.csv", s2_type.get)
    s2b_rows = load_summary(S2_BASE / "Steering" / "steering_condition_summary.csv", s2_type.get)
    say(f"  rows: Study 1 all {len(s1_rows)}, Study 2 model {len(s2m_rows)}, Study 2 baseline {len(s2b_rows)}; "
        f"timed-out/diverged model runs in Study 2: model {sum(r['n_timed_out'] for r in s2m_rows)}, baseline {sum(r['n_timed_out'] for r in s2b_rows)}")
    tex = []   # single-column layout: one row per corridor type x source, 4 metric columns
    SRC_LABEL = (("Pooled model", s2m_rows), ("Pooled baseline", s2b_rows))
    hdr = f"{'type':18s} | {'S2 model lat':>12s} {'spd':>7s} {'corr':>6s} {'tdiff%':>7s} | {'S2 base lat':>12s} {'spd':>7s} {'corr':>6s} {'tdiff%':>7s}"
    say(hdr)
    for ty in TYPE_ORDER + ["All"]:
        line = f"  {ty:18s}"
        if ty == "All":
            tex.append("        \\midrule")
        for i, (label, rows) in enumerate(SRC_LABEL):
            gr = group(rows, ty)
            cols = [cell([r[k] for r in gr], p) for k, _, _, p in METRICS]
            line += " | " + " ".join(f"{np.mean([r[k] for r in gr]):7.{p}f}" for k, _, _, p in METRICS)
            tex.append(f"        {(ty if i == 0 else ''):<18} & {label:<26} & " + " & ".join(cols) + " \\\\")
        say(line)
    gr = s1_rows
    cols = [cell([r[k] for r in gr], p) for k, _, _, p in METRICS]
    tex.append(f"        {'':<18} & {'Per-participant, Study 1':<26} & " + " & ".join(cols) + " \\\\")
    say("  Study 1 reference (per-participant models, all conditions, Table 1 Model Overall): " +
        " ".join(f"{np.mean([r[k] for r in gr]):7.{p}f}" for k, _, _, p in METRICS) + f"  (n={len(gr)})")
    (OUT / "table3_rows.tex").write_text("\n".join(tex) + "\n")
    say("  time ratio (model/human) by type, Study 2 model vs baseline:")
    for ty in TYPE_ORDER + ["All"]:
        say(f"    {ty:18s} model {np.mean([r['ratio'] for r in group(s2m_rows, ty)]):.2f}  baseline {np.mean([r['ratio'] for r in group(s2b_rows, ty)]):.2f}")
    say("  Study 2 pooled model by width (shared with fit cohort: 10, 50 mm; new: 20, 30, 40):")
    for wmm in (10.0, 20.0, 30.0, 40.0, 50.0):
        gr = [r for r in s2m_rows if r["width"] == wmm]
        say(f"    W={wmm:3.0f} lat {np.mean([r['lat_mm'] for r in gr]):.1f} mm  spd {np.mean([r['spd'] for r in gr]):.3f}  corr {np.mean([r['corr'] for r in gr]):.2f}  "
            f"tdiff {np.mean([r['tdiff_pct'] for r in gr]):.1f}%  ratio {np.mean([r['ratio'] for r in gr]):.2f}")

    # ---- Wilcoxon, model vs baseline, per participant (n = 14) -----------------
    say("\n=== paired one-sided Wilcoxon over Study 2 participants, pooled model vs pooled baseline ===")
    # the condition summary has no pid column; rows are written participant by
    # participant in sorted-pid order, 25 conditions each — recover pids from
    # the per-round CSV (same order, same count) and check the count matches
    pids = sorted({p for (s, p, t) in per_m if s == "Human"})
    n_cond = len(tids)
    assert len(s2m_rows) == len(pids) * n_cond == len(s2b_rows), "unexpected row count"
    m_by = rows_by_pid(s2m_rows, S2_MODEL / "Steering" / "steering_condition_summary.csv")
    b_by = rows_by_pid(s2b_rows, S2_BASE / "Steering" / "steering_condition_summary.csv")
    for k, label, alt in (("lat_mm", "lateral RMSE", "less"), ("spd", "speed RMSE", "less"),
                          ("corr", "speed corr", "greater"), ("tdiff_pct", "time diff", "less"),
                          ("abs_err", "|dT| (s)", "less")):
        m, b = per_pid_means(m_by, k), per_pid_means(b_by, k)
        st = wilcoxon(m, b, alternative=alt)
        better = int(np.sum(m < b) if alt == "less" else np.sum(m > b))
        say(f"  {label:13s} model {m.mean():7.3f}  baseline {b.mean():7.3f}  W={st.statistic:.0f} p={st.pvalue:.4f}  model better in {better}/{len(pids)} participants")

    # ---- Study 1 (Section 5.3): the same test for the per-participant runs ---
    say("\n=== Study 1 (per-participant models vs per-participant baseline), one-sided paired Wilcoxon over participants ===")
    s1b_rows = load_summary(S1_BASE / "Steering" / "steering_condition_summary.csv", S1_TYPE_BY_TID.get)
    m1_by = rows_by_pid(s1_rows, S1_MODEL / "Steering" / "steering_condition_summary.csv")
    b1_by = rows_by_pid(s1b_rows, S1_BASE / "Steering" / "steering_condition_summary.csv")
    assert sorted(m1_by) == sorted(b1_by), "participant sets differ between the Study 1 runs"
    # compare on the (participant, condition) pairs present in BOTH runs
    common = {pid: {r["tid"] for r in m1_by[pid]} & {r["tid"] for r in b1_by[pid]} for pid in m1_by}
    say(f"  common participant x condition pairs: {sum(len(v) for v in common.values())} "
        f"(model rows {len(s1_rows)}, baseline rows {len(s1b_rows)})")
    for split, keep in (("all", lambda r: True), ("train", lambda r: r["width"] in S1_TRAIN_WIDTHS_MM),
                        ("test", lambda r: r["width"] not in S1_TRAIN_WIDTHS_MM)):
        say(f"  [{split}]")
        for k, label, alt in (("lat_mm", "lateral RMSE", "less"), ("spd", "speed RMSE", "less"),
                              ("corr", "speed corr", "greater"), ("tdiff_pct", "time diff", "less"),
                              ("abs_err", "|dT| (s)", "less")):
            m = np.array([np.mean([r[k] for r in rows if keep(r) and r["tid"] in common[pid]]) for pid, rows in sorted(m1_by.items())])
            b = np.array([np.mean([r[k] for r in rows if keep(r) and r["tid"] in common[pid]]) for pid, rows in sorted(b1_by.items())])
            st = wilcoxon(m, b, alternative=alt)
            better = int(np.sum(m < b) if alt == "less" else np.sum(m > b))
            say(f"    {label:13s} model {m.mean():7.3f}  baseline {b.mean():7.3f}  W={st.statistic:.0f} p={st.pvalue:.4f}  model better in {better}/{len(m)} participants")

    (OUT / "stats.txt").write_text("\n".join(_log) + "\n")
    print(f"\nwrote {OUT.relative_to(REPO)}/stats.txt, table3_rows.tex, conditions.csv")


if __name__ == "__main__":
    main()

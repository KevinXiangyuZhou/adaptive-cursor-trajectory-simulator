"""Quantify the saccade-fixate-catch-up cycle, human vs model (Sec. eval_cycle).

Fills the TODOs of the evaluation draft with numbers from the eight fitted
participants (participants_10p.txt):

  1. saccade distance vs width power law  amp = a * W^b   (human, model)
     + straight vs sinusoidal saccade distance at equal width
  2. catch-up duration vs width power law                  (human, model)
     + partial Spearman rho of log duration vs accumulated |curvature| of the
       caught-up segment, controlling log W and segment distance
  3. overrun: % of cycles where the cursor crosses the anchor before the next
     saccade, and median overrun depth as a fraction of the post-saccade lead
  4. cursor speed -> saccade lead at equal width: partial Spearman rho of the
     pre-saccade cursor speed vs the post-saccade lead / amplitude / onset
     lead (all | log W), and an OLS mediation of the straight-vs-sinusoid
     amplitude gap by speed (does the straight dummy collapse once speed
     enters?). The pre-saccade speed is the endpoint slope -d(lead)/dt over
     the PRE_WIN window before onset: during a fixation the gaze anchor is
     stationary, so the lead decays at the cursor's arc speed — the same
     estimator applies to human traces and model traces.

Human side: the committed lead series (human-gaze-lead-10p/data/
p*_steering_lead.csv) with the SAME forward-saccade detector as
saccade_width_analysis.py; the curvature intervals are regenerated from the
raw sessions for all eight participants via catchup_curvature_analysis.py
(its LETTERS list is overridden here).

Model side: each participant's fitted persona (results-cluster-10p/
anchor_fitting_10p/stages/base, noise on, seeded) is re-simulated on every
steering trial; the model lead is the anchor-minus-cursor arc distance on the
trial centerline (as in gaze_lead_grids.py) and a saccade is a single-step
forward jump of the lead within the same [A_MIN, A_MAX] amplitude gates
(the anchor moves only at replans, so jumps are one dt=0.05 s step wide;
the human run-based velocity detector reduces to the same event).

Stages (cached as CSVs under paper/quant/):
  python paper/quantify_gaze_cycles.py --stage human       # fast
  python paper/quantify_gaze_cycles.py --stage human-curv  # slow (projection)
  python paper/quantify_gaze_cycles.py --stage model       # slow (600 sims)
  python paper/quantify_gaze_cycles.py --stage stats       # prints TODO block
  python paper/quantify_gaze_cycles.py --stage all
"""
import argparse
import glob
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "eval" / "eval-gaze-lead"))
sys.path.insert(0, str(PROJECT_ROOT / "eval" / "model_fitting"))
import model_gaze_lead as mg                       # noqa: E402
import fit_speed_model as fsm                      # noqa: E402
from saccade_width_analysis import V_ON, MAX_DT, A_MIN, A_MAX  # noqa: E402

LETTERS = ["p01", "p02", "p03", "p04", "p06", "p07", "p08", "p10"]
DATA_DIR = PROJECT_ROOT / "eval" / "eval-gaze-lead" / "human-gaze-lead-10p" / "data"
CONFIG_DIR = PROJECT_ROOT / "results-cluster-10p" / "anchor_fitting_10p" / "stages" / "base"
OUT = Path(__file__).resolve().parent / "quant"
SINUSOIDS = {"gentle", "sharp", "normal"}
DUR_MIN, DUR_MAX = 0.05, 3.0     # catch-up window, as catchup_curvature_analysis
DIST_MIN = 0.003                  # m, minimum caught-up distance for curvature
PRE_WIN, PRE_MIN = 0.20, 0.08     # s, pre-saccade speed window (and its floor)
V_PRE_MAX = 1.0                   # m/s, above = window contaminated by a
                                  # backward gaze saccade, not cursor motion


# ---------------------------------------------------------------- detection
def spans_from_series(t, lead, run_based):
    """Forward-saccade index spans [(i, j)] with lead[j]-lead[i] the amplitude.

    run_based=True: the committed human detector (contiguous lead velocity
    > V_ON runs). False: model traces, where the anchor jumps in exactly one
    step — each positive single-step jump is its own span. Both apply the
    same [A_MIN, A_MAX] amplitude gates.
    """
    dl = np.diff(lead)
    dt = np.diff(t)
    spans = []
    if run_based:
        on = (dl / np.where(dt > 0, dt, np.inf) > V_ON) & (dt <= MAX_DT)
        i = 0
        while i < len(on):
            if on[i]:
                j = i
                while j + 1 < len(on) and on[j + 1]:
                    j += 1
                if A_MIN <= lead[j + 1] - lead[i] <= A_MAX:
                    spans.append((i, j + 1))
                i = j + 1
            i += 1
    else:
        for i in np.where((dl >= A_MIN) & (dl <= A_MAX))[0]:
            spans.append((int(i), int(i) + 1))
    return spans


def cycle_rows(t, lead, spans, theta=None, phi_of_s=None):
    """One row per saccade; catch-up/overrun fields describe the interval to
    the NEXT saccade (NaN for the round's last saccade or across a gap)."""
    rows = []
    for k, (i, j) in enumerate(spans):
        r = dict(t=t[i], amp=lead[j] - lead[i], lead_pre=lead[i], lead_post=lead[j],
                 v_pre=np.nan, catchup_s=np.nan, min_lead=np.nan, dist=np.nan,
                 dphi=np.nan)
        # Pre-saccade cursor arc speed: -d(lead)/dt over the last PRE_WIN
        # seconds of the preceding fixation (anchor stationary there, so the
        # lead decays at the cursor's speed; endpoint slope is robust to
        # gaze jitter). Confined to after the previous saccade's end and to
        # gap-free samples.
        j_prev = spans[k - 1][1] if k > 0 else 0
        k0 = i
        while (k0 > j_prev and t[i] - t[k0 - 1] <= PRE_WIN
               and t[k0] - t[k0 - 1] <= MAX_DT + 1e-9):
            k0 -= 1
        if t[i] - t[k0] >= PRE_MIN:
            r["v_pre"] = float((lead[k0] - lead[i]) / (t[i] - t[k0]))
        if k + 1 < len(spans):
            i1 = spans[k + 1][0]
            # i1 == j: back-to-back saccades, no catch-up interval (the
            # zero-length cycle would be dropped by DUR_MIN anyway).
            if i1 > j and np.max(np.diff(t[j:i1 + 1])) <= MAX_DT + 1e-9:
                r["catchup_s"] = t[i1] - t[j]
                r["min_lead"] = float(np.min(lead[j:i1 + 1]))
                if theta is not None:
                    r["dist"] = float(theta[i1] - theta[j])
                    if phi_of_s is not None:
                        r["dphi"] = max(float(phi_of_s(theta[i1]) - phi_of_s(theta[j])), 0.0)
        rows.append(r)
    return rows


# ---------------------------------------------------------------- human
def stage_human():
    rows = []
    for f in sorted(glob.glob(str(DATA_DIR / "p*_steering_lead.csv"))):
        d = pd.read_csv(f)
        d = d[d["participant"].isin(LETTERS)]
        for (L, lab, tid, w, rnd), g in d.groupby(
                ["participant", "type_label", "trial_id", "width_mm", "round"]):
            g = g.sort_values("t")
            t, lead = g["t"].to_numpy(), g["lead"].to_numpy()
            for r in cycle_rows(t, lead, spans_from_series(t, lead, run_based=True)):
                rows.append(dict(participant=L, type_label=lab, trial_id=int(tid),
                                 width_mm=float(w), round=int(rnd), **r))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "cycles_human.csv", index=False, float_format="%.6f")
    print(f"human: {len(d)} saccades, {d['catchup_s'].notna().sum()} cycles "
          f"({d['participant'].nunique()} participants)")


def stage_human_curv():
    """Regenerate catchup_curvature.csv from raw sessions for all 8."""
    import catchup_curvature_analysis as cca
    cca.LETTERS = LETTERS
    cca.main()
    d = pd.read_csv(DATA_DIR / "catchup_curvature.csv")
    d.to_csv(OUT / "curv_human.csv", index=False)
    print(f"human curvature intervals: {len(d)} "
          f"({d['participant'].nunique()} participants)")


# ---------------------------------------------------------------- model
def make_sim(config_path):
    cfg = json.load(open(config_path))
    cfg.pop("_description", None)
    cfg["add_noise"] = True
    if not float(cfg.get("replan_latency_cv", 0.0) or 0.0):
        cfg["replan_latency_cv"] = 0.89
    return fsm._make_sim(cfg)


def model_trace_with_theta(sim, task_config, centerline):
    """(t, lead, theta_cl): mg.model_lead_trace plus the cursor arc series."""
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(task_config, tf)
        task_file = tf.name
    traj_raw, ref_path = sim.generate_trajectory_with_waypoints(
        task_file=task_file, target_radius=task_config["target_radius"],
        max_steps=task_config.get("max_steps", mg.MAX_STEPS),
        return_reference_path=True)
    diag = sim.last_diagnostics
    pts = np.array([[x * mg.SCALE, y * mg.SCALE] for x, y, _ in traj_raw])
    n = len(pts)
    if n < 2:
        return None
    cl_path = mg.ArcProjector(centerline)
    theta_cl = np.array([cl_path.theta(p) for p in pts])
    events = sorted(diag["replan_events"], key=lambda e: e["step"])
    anchor_cl = np.full(n, np.nan)
    for k, e in enumerate(events):
        s_a = min(float(e["anchor"]), float(ref_path.total_length))
        xy = np.asarray(ref_path(s_a), dtype=float).reshape(2)
        lo = int(e["step"])
        hi = int(events[k + 1]["step"]) if k + 1 < len(events) else n
        anchor_cl[lo:min(hi, n)] = cl_path.theta(xy)
    t = np.arange(n) * mg.DT
    return t, anchor_cl - theta_cl, theta_cl


def centerline_phi(centerline):
    """s -> cumulative unsigned turn angle (rad) of the polyline."""
    cl = np.asarray(centerline, float)
    seg = np.diff(cl, axis=0)
    keep = np.linalg.norm(seg, axis=1) > 1e-12
    cl = np.vstack([cl[:1], cl[1:][keep]])
    seg = np.diff(cl, axis=0)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(seg, axis=1))])
    ang = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
    phi = np.concatenate([[0.0, 0.0], np.cumsum(np.abs(np.diff(ang)))])
    return lambda q: float(np.interp(q, s, phi))


def stage_model(seed, n_runs):
    rows = []
    for L in LETTERS:
        hum = pd.read_csv(DATA_DIR / f"{L}_steering_lead.csv")
        trials = (hum.groupby("trial_id").first().reset_index()
                     [["trial_id", "type_label", "width_mm"]])
        rounds_by_tid, t2c, t2b = fsm.load_participant(L)
        sim = make_sim(CONFIG_DIR / f"{L}_anchor_config_s42.json")
        for _, tr in trials.iterrows():
            tid, lab, w = int(tr["trial_id"]), tr["type_label"], float(tr["width_mm"])
            if t2b.get(tid) != "steering" or t2c.get(tid) is None:
                continue
            built = mg.build_task(tid, "steering", t2c[tid], rounds_by_tid.get(tid, []))
            if built is None:
                continue
            tc, cl = built
            phi = centerline_phi(cl)
            for run in range(n_runs):
                np.random.seed((seed * 1000003 + int(L[1:]) * 10007
                                + tid * 101 + run) % (2 ** 32))
                res = model_trace_with_theta(sim, tc, cl)
                if res is None:
                    continue
                t, lead, theta = res
                spans = spans_from_series(t, lead, run_based=False)
                for r in cycle_rows(t, lead, spans, theta=theta, phi_of_s=phi):
                    rows.append(dict(participant=L, type_label=lab, trial_id=tid,
                                     width_mm=w, round=run + 1, **r))
            print(f"  [{L}] t{tid} {lab} W{w:g}: "
                  f"{sum(r['participant'] == L and r['trial_id'] == tid for r in rows)}"
                  " saccades", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "cycles_model.csv", index=False, float_format="%.6f")
    print(f"model: {len(d)} saccades, {d['catchup_s'].notna().sum()} cycles")


# ---------------------------------------------------------------- stats
def powerlaw(d, ycol, label):
    """Per-participant exponents of y = a * W^b (log-log OLS) + pooled."""
    bs = []
    for L, g in d.groupby("participant"):
        b, _, r, p, se = stats.linregress(np.log(g["width_mm"]), np.log(g[ycol]))
        bs.append(b)
    pb, _, pr, pp, _ = stats.linregress(np.log(d["width_mm"]), np.log(d[ycol]))
    bs = np.array(bs)
    print(f"  {label}: b = {bs.mean():.3f} +- {bs.std():.3f} (per-participant, "
          f"n={len(bs)}); pooled b = {pb:.3f} (r={pr:.3f}, p={pp:.1e}, "
          f"n={len(d)})")
    return bs.mean(), bs.std(), pb


def _resid(y, X):
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return y - X1 @ beta


def partial_rho(d, label):
    """Spearman rho(dphi, log dur | log W, dist), pooled + per-participant."""
    ctrl = np.column_stack([np.log(d["width_mm"]), d["dist_m"]])
    rho, p = stats.spearmanr(_resid(d["dphi_rad"].to_numpy(), ctrl),
                             _resid(np.log(d["dur_s"]).to_numpy(), ctrl))
    per = []
    for L, g in d.groupby("participant"):
        c = np.column_stack([np.log(g["width_mm"]), g["dist_m"]])
        r, _ = stats.spearmanr(_resid(g["dphi_rad"].to_numpy(), c),
                               _resid(np.log(g["dur_s"]).to_numpy(), c))
        per.append(r)
    per = np.array(per)
    print(f"  {label}: pooled partial rho = {rho:.3f} (p={p:.1e}, n={len(d)}); "
          f"per-participant {per.mean():.3f} +- {per.std():.3f}")
    return rho, p, per.mean(), per.std()


def overrun_stats(d, label):
    c = d[d["catchup_s"].between(DUR_MIN, DUR_MAX) & (d["lead_post"] > 0)]
    crossed = c["min_lead"] < 0
    depth = (-c.loc[crossed, "min_lead"] / c.loc[crossed, "lead_post"])
    per = c.assign(x=crossed).groupby("participant")["x"].mean()
    print(f"  {label}: crossed in {crossed.mean() * 100:.1f}% of {len(c)} cycles "
          f"(per-participant {per.mean() * 100:.1f} +- {per.std() * 100:.1f}%); "
          f"median depth = {depth.median():.2f} of the post-saccade lead")
    return crossed.mean(), depth.median()


def speed_lead_partial(d, ycol, label, logy=True):
    """Spearman rho(log v_pre, y | log W): does speed move ``ycol`` at
    equal width? Pooled + per-participant."""
    d = d[(d["v_pre"] > 0) & (d["v_pre"] <= V_PRE_MAX)].copy()
    y = d[ycol].to_numpy()
    if logy:
        d = d[d[ycol] > 0]
        y = np.log(d[ycol].to_numpy())
    x = np.log(d["v_pre"].to_numpy())
    ctrl = np.log(d["width_mm"].to_numpy())[:, None]
    rho, p = stats.spearmanr(_resid(x, ctrl), _resid(y, ctrl))
    per = []
    for L, g in d.groupby("participant"):
        c = np.log(g["width_mm"].to_numpy())[:, None]
        gy = np.log(g[ycol].to_numpy()) if logy else g[ycol].to_numpy()
        r, _ = stats.spearmanr(_resid(np.log(g["v_pre"].to_numpy()), c),
                               _resid(gy, c))
        per.append(r)
    per = np.array(per)
    print(f"  {label}: rho(v_pre, {ycol} | W) = {rho:.3f} (p={p:.1e}, "
          f"n={len(d)}); per-participant {per.mean():.3f} +- {per.std():.3f}")
    return rho, p


def _ols(y, X):
    """OLS betas, t-stats, p-values (X without intercept column)."""
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    r = y - X1 @ beta
    dof = len(y) - X1.shape[1]
    s2 = float(r @ r) / dof
    cov = s2 * np.linalg.inv(X1.T @ X1)
    t = beta / np.sqrt(np.diag(cov))
    p = 2 * stats.t.sf(np.abs(t), dof)
    return beta, t, p


def speed_mediation(d, label):
    """Does pre-saccade speed absorb the straight-vs-sinusoid amplitude gap?

    OLS log amp ~ log W + 1[straight], then + log v_pre. The straight
    dummy's coefficient is the log gap at equal width; the fraction of it
    that vanishes when speed enters is the share the speed channel carries.
    """
    d = d[d["type_label"].isin(SINUSOIDS | {"straight"})]
    d = d[(d["v_pre"] > 0) & (d["v_pre"] <= V_PRE_MAX) & (d["amp"] > 0)].copy()
    straight = (d["type_label"] == "straight").to_numpy(float)
    y = np.log(d["amp"].to_numpy())
    logw = np.log(d["width_mm"].to_numpy())
    logv = np.log(d["v_pre"].to_numpy())
    b1, _, p1 = _ols(y, np.column_stack([logw, straight]))
    b2, _, p2 = _ols(y, np.column_stack([logw, straight, logv]))
    absorbed = 100.0 * (1.0 - b2[2] / b1[2]) if b1[2] != 0 else np.nan
    # speed gap itself at equal width (premise: straight is traversed faster)
    bv, _, pv = _ols(logv, np.column_stack([logw, straight]))
    print(f"  {label}: straight beta = {b1[2]:.3f} (x{np.exp(b1[2]):.2f}, "
          f"p={p1[2]:.1e}) -> {b2[2]:.3f} (x{np.exp(b2[2]):.2f}, "
          f"p={p2[2]:.1e}) with log v_pre in; {absorbed:.0f}% absorbed; "
          f"beta_v = {b2[3]:.3f} (p={p2[3]:.1e}); speed gap at equal width "
          f"x{np.exp(bv[2]):.2f} (p={pv[2]:.1e}); n={len(d)}")
    return b1[2], b2[2], absorbed


def straight_vs_sinusoid(d, label):
    d = d[d["type_label"].isin(SINUSOIDS | {"straight"})].copy()
    d["cls"] = np.where(d["type_label"] == "straight", "straight", "sinusoid")
    piv = d.pivot_table(index="width_mm", columns="cls", values="amp",
                        aggfunc="median") * 1000
    piv["ratio"] = piv["straight"] / piv["sinusoid"]
    print(f"  {label}: median saccade distance (mm) by width\n"
          + piv.round(2).to_string())
    return piv


def stage_stats():
    hum = pd.read_csv(OUT / "cycles_human.csv")
    mod = pd.read_csv(OUT / "cycles_model.csv")
    curv_h = pd.read_csv(OUT / "curv_human.csv")
    curv_h = curv_h[curv_h["participant"].isin(LETTERS)]
    # model curvature intervals with the human analysis' filters
    curv_m = mod[(mod["catchup_s"].between(DUR_MIN, DUR_MAX))
                 & (mod["dist"] >= DIST_MIN)].rename(
        columns={"catchup_s": "dur_s", "dist": "dist_m", "dphi": "dphi_rad"})

    print("\n== 1. saccade distance vs width (power law amp = a * W^b) ==")
    bh = powerlaw(hum, "amp", "human")
    bm = powerlaw(mod, "amp", "model")
    print("\n   straight vs sinusoidal saccade distance:")
    straight_vs_sinusoid(hum, "human")
    straight_vs_sinusoid(mod, "model")

    print("\n== 2. catch-up duration vs width (power law dur = a * W^b) ==")
    ch = hum[hum["catchup_s"].between(DUR_MIN, DUR_MAX)]
    cm = mod[mod["catchup_s"].between(DUR_MIN, DUR_MAX)]
    powerlaw(ch, "catchup_s", "human")
    powerlaw(cm, "catchup_s", "model")
    print("\n   duration vs accumulated curvature (partial, | log W, dist):")
    partial_rho(curv_h, "human")
    partial_rho(curv_m, "model")

    print("\n== 3. overrun below zero before the next saccade ==")
    overrun_stats(hum, "human")
    overrun_stats(mod, "model")

    print("\n== 4. cursor speed -> saccade lead at equal width ==")
    print("   post-saccade lead vs pre-saccade speed (| log W):")
    speed_lead_partial(hum, "lead_post", "human")
    speed_lead_partial(mod, "lead_post", "model")
    print("   saccade amplitude vs pre-saccade speed (| log W):")
    speed_lead_partial(hum, "amp", "human")
    speed_lead_partial(mod, "amp", "model")
    print("   onset lead (overrun channel; expect negative) vs speed (| log W):")
    speed_lead_partial(hum, "lead_pre", "human", logy=False)
    speed_lead_partial(mod, "lead_pre", "model", logy=False)
    print("   straight-vs-sinusoid amplitude gap, mediated by speed:")
    speed_mediation(hum, "human")
    speed_mediation(mod, "model")

    print("\n(cycle counts: human", len(ch), "model", len(cm),
          "| curvature intervals: human", len(curv_h), "model", len(curv_m), ")")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "human", "human-curv", "model", "stats"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--config-dir", default=None,
                    help="directory of fitted personas {letter}_anchor_config_s42.json "
                         "(default: the legacy anchor_fitting_10p stage; use a run's fit/stages/base)")
    ap.add_argument("--quant-dir", default=None,
                    help="output/cache directory (default paper/quant); use a separate one per persona set")
    a = ap.parse_args()
    global CONFIG_DIR, OUT
    if a.config_dir:
        CONFIG_DIR = Path(a.config_dir).resolve()
    if a.quant_dir:
        OUT = Path(a.quant_dir).resolve()
    OUT.mkdir(exist_ok=True, parents=True)
    if a.stage in ("all", "human"):
        stage_human()
    if a.stage in ("all", "human-curv"):
        stage_human_curv()
    if a.stage in ("all", "model"):
        stage_model(a.seed, a.runs)
    if a.stage in ("all", "stats"):
        stage_stats()


if __name__ == "__main__":
    main()

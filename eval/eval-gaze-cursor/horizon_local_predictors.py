"""Does the gaze horizon (onset lead: arc distance from the cursor at fixation onset
to the fixated point) depend on width, curvature, cursor speed — or which combination?

Data: merged canonical fixation events (keep=True), steering only, lead_corr > 3 mm,
end-of-path fixations excluded (fixated point within 5 mm of the tunnel end — the lead
is capped there, not chosen). Predictors, all at/ahead of the CURSOR at onset:
    W_c        usable width at s_c (free space W_c > 0.15 excluded)
    dphi_ahead turning angle over the next 30 mm of path (rad) — upcoming curvature
    v          cursor speed at onset (floored at 0.02 m/s)

Per participant:
  * marginal Spearman rho(lead, x) for each predictor;
  * partial Spearman (rank OLS residuals) for each predictor given the other two;
  * OLS log(lead) ~ b_w log(W_c/W_ref) + b_k dphi_ahead + b_v log(v), bootstrap CIs;
  * 5-fold CV R^2 of log(lead) for all 7 predictor subsets — which combination is
    actually needed.

Usage: python horizon_local_predictors.py [--letters A B C p01 ...]
Saves results/horizon_local_predictors.json.
"""
import argparse, itertools, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import gaze_data as gd, lookahead_difficulty as ld

PROC = HERE.parents[1] / "human_data" / "processed_gaze_events"
W_REF = 0.026
AHEAD_M = 0.030          # forward window for upcoming turning angle
LEAD_MIN = 0.003
END_PAD = 0.005
FREE_W = 0.15
V_FLOOR = 0.02


GAZE_VALID_MIN = 0.5     # round pruning: min fraction of samples with mapped gaze
KEPT_MIN = 3             # round pruning: min kept fixations in the round
SPAN_MIN = 0.6           # round pruning: min tunnel fraction the round's fixations span


def round_quality(s, cl_all, geoms, L):
    """Per-round (trial_id, block_id) quality gate. A round is pruned when the gaze
    mapping is mostly missing (tracking loss / failed block calibration), when almost
    no fixation survived cleaning (uninformative), or when the round's fixations span
    only a fraction of the tunnel (incomplete round: aborted trial or truncated data).
    Returns (kept set, stats dict)."""
    ok, why = set(), {"gaze": 0, "few_kept": 0, "span": 0}
    st = s[s["cursor_x"].notna()]
    gaze_frac = st.groupby(["trial_id", "block_id"])["gaze_task_x"].apply(lambda x: x.notna().mean())
    for (tid, bid), grp in cl_all.groupby(["trial_id", "block_id"]):
        g = geoms.get((L, tid))
        if g is None:
            continue
        gf = float(gaze_frac.get((tid, bid), 0.0))
        n_kept = int(grp["keep"].sum())
        span = float((grp["s_c"].max() - grp["s_c"].min()) / max(g.s_end, 1e-9))
        if gf < GAZE_VALID_MIN:
            why["gaze"] += 1
        elif n_kept < KEPT_MIN:
            why["few_kept"] += 1
        elif span < SPAN_MIN:
            why["span"] += 1
        else:
            ok.add((tid, bid))
    return ok, why


def _global_lead(cl, geoms, L, step=0.001):
    """Recompute each event's onset lead with the ABC-style projection: dense
    global nearest point, NO forward constraint (the corrected gaze position
    gaze_x/y_corr is projected as-is). Returns the new lead array (NaN where
    no geometry)."""
    out = np.full(len(cl), np.nan)
    grids = {}
    for i, (_, r) in enumerate(cl.iterrows()):
        g = geoms.get((L, r["trial_id"]))
        if g is None:
            continue
        if r["trial_id"] not in grids:
            s_d = np.arange(0.0, g.s_end + step, step)
            P = np.column_stack([np.interp(s_d, g.s, g.path[:, 0]),
                                 np.interp(s_d, g.s, g.path[:, 1])])
            grids[r["trial_id"]] = (s_d, P)
        s_d, P = grids[r["trial_id"]]
        d2 = (P[:, 0] - r["gaze_x_corr"]) ** 2 + (P[:, 1] - r["gaze_y_corr"]) ** 2
        out[i] = s_d[int(np.argmin(d2))] - r["s_c"]
    return out


def build_with_trial(L, prune=True, abc_proj=False):
    s = gd.load_samples(L)
    cl_all = pd.read_csv(PROC / f"{L}_fixation_events_clean.csv")
    cl = cl_all[cl_all["keep"] == True]  # noqa: E712
    ev = gd.fixation_events(s)
    ev = ev[~ev["tunnel_type"].astype(str).str.contains("pointing")]
    ev, geoms = ld.attach_geometry(s, ev)
    prune_stats = None
    if prune:
        ok, why = round_quality(s, cl_all, geoms, L)
        n_rounds = cl_all.groupby(["trial_id", "block_id"]).ngroups
        cl = cl[[tuple(x) in ok for x in cl[["trial_id", "block_id"]].to_numpy()]]
        prune_stats = {"rounds_total": int(n_rounds), "rounds_kept": len(ok), "pruned_by": why}
    if abc_proj:
        cl = cl.copy()
        cl["lead_corr"] = _global_lead(cl, geoms, L)
    cl = cl.merge(ev[["trial_id", "block_id", "fixation_id", "speed_onset"]],
                  on=["trial_id", "block_id", "fixation_id"], how="left")
    rows, n_end = [], 0
    for _, r in cl.iterrows():
        g = geoms.get((L, r["trial_id"]))
        if g is None or not np.isfinite(r["lead_corr"]) or r["lead_corr"] <= LEAD_MIN:
            continue
        s_c = float(r["s_c"]); s_f = s_c + float(r["lead_corr"])
        if s_f >= g.s_end - END_PAD:      # lead capped by the tunnel end, not chosen
            n_end += 1; continue
        W_c = float(np.interp(s_c, g.s, g.W))
        if not np.isfinite(W_c) or W_c > FREE_W:
            continue
        dphi = float(np.interp(min(s_c + AHEAD_M, g.s_end), g.s, g.PHI)
                     - np.interp(s_c, g.s, g.PHI))
        v = max(float(r["speed_onset"]) if np.isfinite(r["speed_onset"]) else V_FLOOR, V_FLOOR)
        rows.append((int(r["trial_id"]), float(r["lead_corr"]), W_c, dphi, v))
    d = pd.DataFrame(rows, columns=["trial_id", "lead", "W_c", "dphi", "v"])
    return d, n_end, prune_stats


def partial_spearman(y, X, j):
    """Partial rank correlation of y with X[:,j] given the other columns."""
    ry = rankdata(y); rX = np.column_stack([rankdata(X[:, k]) for k in range(X.shape[1])])
    others = np.c_[np.ones(len(y)), np.delete(rX, j, axis=1)]
    res_y = ry - others @ np.linalg.lstsq(others, ry, rcond=None)[0]
    res_x = rX[:, j] - others @ np.linalg.lstsq(others, rX[:, j], rcond=None)[0]
    denom = np.std(res_y) * np.std(res_x)
    return float(np.mean(res_y * res_x) / denom) if denom > 1e-12 else np.nan


def cv_r2(X, y, cols, k=5, seed=0):
    """5-fold CV R^2 of OLS y ~ X[:, cols] (with intercept)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y)); folds = np.array_split(idx, k)
    ss_res, ss_tot = 0.0, 0.0
    for f in folds:
        tr = np.setdiff1d(idx, f)
        A_tr = np.c_[np.ones(len(tr)), X[np.ix_(tr, cols)]]
        A_te = np.c_[np.ones(len(f)), X[np.ix_(f, cols)]]
        b = np.linalg.lstsq(A_tr, y[tr], rcond=None)[0]
        ss_res += float(np.sum((y[f] - A_te @ b) ** 2))
        ss_tot += float(np.sum((y[f] - np.mean(y[tr])) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--letters", nargs="*", default=["A", "B", "C"] + [f"p{i:02d}" for i in range(1, 11)])
    ap.add_argument("--no-prune", action="store_true", help="skip the round-quality pruning")
    ap.add_argument("--abc-proj", action="store_true",
                    help="recompute onset leads with the unconstrained global projection "
                         "(ABC gaze_lead_signed method) instead of the forward-constrained one")
    a = ap.parse_args()
    names = ["w", "k", "v"]
    combos = [c for r in (1, 2, 3) for c in itertools.combinations(range(3), r)]
    out = {}
    print(f"{'P':<5} {'n':>4} | partial rho: W    kappa  v     | best CV-R2 model (top three)")
    for L in a.letters:
        d, n_end, ps = build_with_trial(L, prune=not a.no_prune, abc_proj=a.abc_proj)
        if ps:
            print(f"{L:<5} rounds kept {ps['rounds_kept']}/{ps['rounds_total']} "
                  f"(pruned: gaze-loss {ps['pruned_by']['gaze']}, <{KEPT_MIN} kept {ps['pruned_by']['few_kept']}, "
                  f"span<{SPAN_MIN} {ps['pruned_by']['span']})", flush=True)
        if len(d) < 40:
            print(f"{L:<5} {len(d):>4} | too few events"); continue
        # predictors for regression / CV (log-lead target)
        X = np.c_[np.log(W_REF / d["W_c"]), d["dphi"], np.log(d["v"])]
        y = np.log(d["lead"].to_numpy())
        Xr = np.c_[d["W_c"], d["dphi"], d["v"]]          # raw for rank partials
        marg = [spearmanr(d["lead"], Xr[:, j]).statistic for j in range(3)]
        part = [partial_spearman(d["lead"].to_numpy(), Xr, j) for j in range(3)]
        # bootstrap CIs on the full OLS
        rng = np.random.default_rng(1)
        bs = np.empty((1000, 4))
        A = np.c_[np.ones(len(d)), X]
        for i in range(1000):
            ii = rng.integers(0, len(d), len(d))
            bs[i] = np.linalg.lstsq(A[ii], y[ii], rcond=None)[0]
        b = np.linalg.lstsq(A, y, rcond=None)[0]
        lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        r2 = {"".join(names[j] for j in c): cv_r2(X, y, list(c)) for c in combos}
        top = sorted(r2.items(), key=lambda kv: -kv[1])[:3]
        print(f"{L:<5} {len(d):>4} | {part[0]:+.2f}  {part[1]:+.2f}  {part[2]:+.2f} | "
              + "  ".join(f"{k}:{v:.2f}" for k, v in top)
              + f"   (end-capped excl. {n_end})", flush=True)
        # --- condition level: per-trial medians (how the budget constants were fit;
        # event noise averaged out, ~25 steering conditions per participant) ---
        cond = d.groupby("trial_id").agg(lead=("lead", "median"), W_c=("W_c", "median"),
                                         dphi=("dphi", "median"), v=("v", "median"),
                                         n=("lead", "size"))
        cond = cond[cond["n"] >= 5]
        Xc = np.c_[cond["W_c"], cond["dphi"], cond["v"]]
        cmarg = [spearmanr(cond["lead"], Xc[:, j]).statistic for j in range(3)]
        cpart = [partial_spearman(cond["lead"].to_numpy(), Xc, j) for j in range(3)]
        print(f"      cond-level ({len(cond)} trials): marginal rho W {cmarg[0]:+.2f} k {cmarg[1]:+.2f} "
              f"v {cmarg[2]:+.2f} | partial W {cpart[0]:+.2f} k {cpart[1]:+.2f} v {cpart[2]:+.2f}", flush=True)
        out[L] = {"n": int(len(d)), "n_end_capped_excluded": int(n_end),
                  "spearman_marginal": {"W": marg[0], "dphi": marg[1], "v": marg[2]},
                  "spearman_partial": {"W": part[0], "dphi": part[1], "v": part[2]},
                  "ols_log_lead": {nm: {"b": float(b[j + 1]), "ci": [float(lo[j + 1]), float(hi[j + 1])]}
                                    for j, nm in enumerate(["log(Wref/W)", "dphi_ahead", "log(v)"])},
                  "cv_r2": {k: float(v) for k, v in r2.items()},
                  "pruning": ps,
                  "condition_level": {"n_trials": int(len(cond)),
                                       "marginal": {"W": cmarg[0], "dphi": cmarg[1], "v": cmarg[2]},
                                       "partial": {"W": cpart[0], "dphi": cpart[1], "v": cpart[2]}}}
    if out:
        vs = list(out.values())
        medp = {k: float(np.nanmedian([v["spearman_partial"][k] for v in vs])) for k in ("W", "dphi", "v")}
        medr2 = {k: float(np.nanmedian([v["cv_r2"][k] for v in vs])) for k in vs[0]["cv_r2"]}
        best = sorted(medr2.items(), key=lambda kv: -kv[1])
        medc = {lvl: {k: float(np.nanmedian([v["condition_level"][lvl][k] for v in vs]))
                      for k in ("W", "dphi", "v")} for lvl in ("marginal", "partial")}
        print(f"\npooled median partial rho (event level): W {medp['W']:+.2f}, kappa {medp['dphi']:+.2f}, v {medp['v']:+.2f}")
        print("pooled median CV-R2 by model (event level): " + "  ".join(f"{k}:{v:.2f}" for k, v in best))
        print(f"pooled median condition-level rho: marginal W {medc['marginal']['W']:+.2f} k {medc['marginal']['dphi']:+.2f} "
              f"v {medc['marginal']['v']:+.2f} | partial W {medc['partial']['W']:+.2f} k {medc['partial']['dphi']:+.2f} "
              f"v {medc['partial']['v']:+.2f}")
    fname = "horizon_local_predictors.json" if a.no_prune else "horizon_local_predictors_pruned.json"
    if a.abc_proj:
        fname = fname.replace(".json", "_abcproj.json")
    json.dump(out, open(HERE / "results" / fname, "w"), indent=2)
    print(f"saved results/{fname}"); print("DONE")


if __name__ == "__main__":
    main()

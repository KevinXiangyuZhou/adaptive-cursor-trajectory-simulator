"""Refit of the horizon-prediction module: visuomotor floor + width exponent.

Motivation (eval-intermittent horizon_vs_gaze): the bare 1/W difficulty
budget is linear in width while human leads are sublinear (lead ~ w^0.66):
it underestimates lead at the narrowest widths (human ~0.015 at W=0.01 vs
~0.006 predicted) and overestimates at the widest. Two candidate fixes are
fitted jointly and model-compared under identical CV:

  * visuomotor-delay floor:  h >= v * T_min
  * width exponent gamma:    density_W = (W_ref/W)^gamma / W_ref
    (gamma=1 reduces to 1/W; W_ref = geometric mean of the studied widths
    keeps the density in 1/length so D0 and lam stay dimensionless —
    rescaling W_ref is absorbed by D0, it is not a behavioural knob)

Full model:

    h_pred = clip( max( v * T_min,  h_budget(gamma, lam, D0) ),  0, remaining )

with h_budget the smallest h s.t.
int_s^{s+h} [(W_ref/W)^gamma / W_ref + lam|kappa|] ds' = D0 (end-capped),
v the cursor speed at fixation onset.

Fitting protocol (improvements over lookahead_difficulty.budget_fit):
  * grouped 5-fold cross-validation (folds split by trial; D0 quantiles and
    the power-law fit recomputed on each train split);
  * selection by median absolute log error (MALE) of predicted vs observed
    lead — the simulator consumes lead MAGNITUDES; Spearman rho reported
    alongside for comparability with lookahead_summary.json.

Model comparison under the same CV: full (gamma, T_min free), gamma-only
(T_min=0), floor-only-budget (gamma=1, T_min free), bare budget (gamma=1,
T_min=0), pure floor (h=v*T_min), width power law lead=a*w^b.

Acceptance extras: per-width median lead calibration and the steering->c2u
transfer rho (no refit) with the selected pooled parameters.

Outputs results/lookahead_floor_summary.json, including per-participant
sim_params {T_min, lam, D0, gamma, W_ref} for the simulator/eval configs.
Run: python3 refit_floor.py   (from eval/eval-gaze-cursor)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gaze_data as gd  # noqa: E402
import horizon_analysis as ha  # noqa: E402
import lookahead_difficulty as ld  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"

T_MIN_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]
# Curvature term REMOVED from the gaze model (2026-08-24): lead shrinks with
# width only. Corner-dwell analysis showed lam>0 shrinks the corner anchor
# lead enough to cancel the emergent corner dwell (humans: ~1.4-1.5x longer
# fixations at corners; model reproduces it iff lam=0) while buying ~1% CV.
# The lam plumbing below is kept so old summaries stay comparable, but only
# lam=0 is searched and sim_params no longer emits a lam key.
LAM_GRID = [0.0]
GAMMA_GRID = [0.5, 0.6, 0.66, 0.7, 0.8, 0.9, 1.0]
D0_QUANTS = [0.35, 0.5, 0.65, 0.8, 0.9]
N_FOLDS = 5
# Geometric mean of the studied width levels {0.01..0.05}; dimensional
# reference only — any rescaling is absorbed by the fitted D0.
W_REF = 0.026
# Dense D0 grid for the h(D0) precompute; candidate D0 values (train-fold
# quantiles) are then evaluated by interpolation along this axis.
D0_DENSE = np.geomspace(0.05, 30.0, 120)


def _male(h_pred, h_obs):
    """Median absolute log error — robust, symmetric magnitude criterion."""
    ok = (h_pred > 1e-6) & (h_obs > 1e-6)
    if ok.sum() < 3:
        return np.inf
    return float(np.median(np.abs(np.log(h_pred[ok] / h_obs[ok]))))


def _gamma_cum(geom, gamma, lam):
    """Cumulative difficulty IC(s) on the geom grid for (gamma, lam)."""
    invw = ((W_REF / np.clip(geom.W, 1e-6, None)) ** gamma) / W_REF
    seg = np.diff(geom.s)
    IW = np.concatenate([[0.0], np.cumsum(0.5 * (invw[1:] + invw[:-1]) * seg)])
    return IW + lam * geom.PHI


class BudgetTable:
    """Per-(gamma, lam) precompute of h(D0) on the dense D0 grid and the
    actual-window difficulty D_act for every event. Any candidate D0 value
    is then an O(events) interpolation — the trial loop runs once per
    (gamma, lam), not once per (gamma, lam, D0, fold)."""

    def __init__(self, ev, geoms, gamma, lam):
        n = len(ev)
        self.h_grid = np.empty((n, len(D0_DENSE)))
        self.D_act = np.empty(n)
        for (p, tid), g in ev.groupby(["participant", "trial_id"]):
            geom = geoms[(p, tid)]
            IC = _gamma_cum(geom, gamma, lam)
            loc = ev.index.get_indexer(g.index)
            s_c = g["s_c"].to_numpy()
            ic0 = np.interp(s_c, geom.s, IC)
            # h(D0) for all events x dense-D0 at once (end-capped: interp
            # saturates at s[-1] because IC targets beyond IC[-1] clip there)
            targets = ic0[:, None] + D0_DENSE[None, :]
            s2 = np.interp(targets.ravel(), IC, geom.s).reshape(targets.shape)
            self.h_grid[loc] = s2 - s_c[:, None]
            # difficulty inside the observed lookahead window
            h_obs = np.clip(g["lead_onset"].to_numpy(), 0.0, None)
            self.D_act[loc] = np.interp(s_c + h_obs, geom.s, IC) - ic0

    def leads(self, D0):
        """Budget leads for a scalar D0 (interpolated along the dense axis)."""
        j = np.searchsorted(D0_DENSE, D0)
        j = int(np.clip(j, 1, len(D0_DENSE) - 1))
        d0, d1 = D0_DENSE[j - 1], D0_DENSE[j]
        w = (D0 - d0) / (d1 - d0)
        return (1 - w) * self.h_grid[:, j - 1] + w * self.h_grid[:, j]


def _floored(h_budget, v, T_min, remaining):
    return np.clip(np.maximum(v * T_min, h_budget), 0.0, remaining)


def _fold_assign(ev, n_folds, seed=0):
    """Grouped folds: all events of a trial share a fold."""
    trials = ev["trial_id"].astype(str) + "_" + ev["participant"].astype(str)
    uniq = np.unique(trials)
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(uniq, rng.integers(0, n_folds, len(uniq))))
    return trials.map(fold_of).to_numpy()


def fit_group(ev, geoms, label=""):
    """CV model comparison + final fit for one event group (participant or
    pooled). Returns a summary dict."""
    ev = ev.reset_index(drop=True)
    v = ev["speed_onset"].to_numpy(float)
    h_obs = ev["lead_onset"].to_numpy(float)
    remaining = (ev["s_end"] - ev["s_c"]).to_numpy(float)
    folds = _fold_assign(ev, N_FOLDS)

    tables = {(g, l): BudgetTable(ev, geoms, g, l)
              for g in GAMMA_GRID for l in LAM_GRID}

    # ---- CV over the (gamma, lam, q, T_min) grid
    combo_pred = {}   # (gamma, lam, q, T_min) -> out-of-fold predictions
    pl_pred = np.full(len(ev), np.nan)
    for f in range(N_FOLDS):
        tr = folds != f
        te = folds == f
        if te.sum() == 0 or tr.sum() < 20:
            continue
        for (gam, lam), tab in tables.items():
            for q in D0_QUANTS:
                D0 = float(np.quantile(tab.D_act[tr], q))
                hb = tab.leads(D0)
                for T_min in T_MIN_GRID:
                    hp = _floored(hb[te], v[te], T_min, remaining[te])
                    combo_pred.setdefault((gam, lam, q, T_min),
                                          np.full(len(ev), np.nan))[te] = hp
        pl_params, _ = ha.fit_horizon_power(ev.loc[tr])
        pl_pred[te] = ha.predict_lead(pl_params, ev.loc[te, "width"])

    def score(pred):
        ok = np.isfinite(pred)
        return {"male_cv": _male(pred[ok], h_obs[ok]),
                "rho_cv": float(ha.spearman(pred[ok], h_obs[ok]))}

    scores = {c: score(p) for c, p in combo_pred.items()}

    def best(filt):
        c = min((c for c in scores if filt(c)),
                key=lambda c: scores[c]["male_cv"])
        return {"gamma": c[0], "lam": c[1], "D0_quantile": c[2],
                "T_min": c[3], **scores[c]}, c

    full_s, full_c = best(lambda c: True)
    gamma_only_s, _ = best(lambda c: c[3] == 0.0)
    floor_budget_s, _ = best(lambda c: c[0] == 1.0)
    bare_s, _ = best(lambda c: c[0] == 1.0 and c[3] == 0.0)
    floor_only = {}
    for T_min in T_MIN_GRID:
        if T_min == 0.0:
            continue
        floor_only[T_min] = score(np.clip(v * T_min, 0.0, remaining))
    best_floor = min(floor_only, key=lambda t: floor_only[t]["male_cv"])

    # ---- final fit on all events with the selected combo
    gam, lam, q, T_min = full_c
    tab = tables[(gam, lam)]
    D0 = float(np.quantile(tab.D_act, q))
    hb = tab.leads(D0)
    h_final = _floored(hb, v, T_min, remaining)
    frac_floor = float(np.mean((v * T_min > hb) & (h_final > 0)))

    by_width = {}
    for w, g in ev.groupby(ev["width"].round(3)):
        loc = g.index.to_numpy()
        by_width[f"{w:g}"] = {
            "human": float(np.median(h_obs[loc])),
            "model": float(np.median(h_final[loc])),
            "n": int(len(loc)),
        }

    return {
        "n": int(len(ev)),
        "cv": {
            "full": full_s,
            "gamma_only": gamma_only_s,
            "floored_budget_gamma1": floor_budget_s,
            "bare_budget": bare_s,
            "floor_only": {"T_min": best_floor, **floor_only[best_floor]},
            "power_law": score(pl_pred),
        },
        "sim_params": {"T_min": T_min, "D0": D0,
                       "gamma": gam, "W_ref": W_REF},
        "insample_rho": float(ha.spearman(h_final, h_obs)),
        "insample_male": _male(h_final, h_obs),
        "frac_floor_active": frac_floor,
        "lead_by_width": by_width,
    }


def c2u_transfer(samples, events, sim_params):
    """Steering-fitted params applied to constrained-to-unconstrained trials
    with NO refit — the budget model's transfer test (bare-budget reference:
    rho 0.53 pooled, lookahead_summary.json unconstrained section)."""
    ev = events[
        (events["tunnel_type"] == "constrained_to_unconstrained")
        & (events["speed_onset"] > ha.MIN_SPEED)
        & events["lead_onset"].notna()
    ].copy()
    ev, geoms = ld.attach_geometry(samples, ev)
    ev = ev[ev["s_c"].notna() & (ev["lead_onset"] > ha.MIN_LEAD)]
    ev = ev.reset_index(drop=True)
    if not len(ev):
        return {}
    tab = BudgetTable(ev, geoms, sim_params["gamma"], sim_params.get("lam", 0.0))
    hb = tab.leads(sim_params["D0"])
    remaining = (ev["s_end"] - ev["s_c"]).to_numpy(float)
    hp = _floored(hb, ev["speed_onset"].to_numpy(float),
                  sim_params["T_min"], remaining)
    h_obs = ev["lead_onset"].to_numpy(float)
    out = {"pooled": {"rho": float(ha.spearman(hp, h_obs)),
                      "male": _male(hp, h_obs), "n": int(len(ev))}}
    for p, g in ev.groupby("participant"):
        loc = g.index.to_numpy()
        out[p] = {"rho": float(ha.spearman(hp[loc], h_obs[loc])),
                  "male": _male(hp[loc], h_obs[loc]), "n": int(len(g))}
    return out


def ttc_validation(ev_pos: pd.DataFrame, fitted: dict) -> dict:
    """Fit-free check of T_min against time_to_catch = lead/v at fixation
    onset (re-exported CSVs). The floor h >= v*T_min is exactly
    ttc_onset >= T_min at the planning event, so the fitted T_min should sit
    near the lower quantiles of the onset distribution (instantaneous-speed
    noise blurs the hard edge; use quantiles, not the minimum)."""
    if "ttc_onset" not in ev_pos.columns:
        return {}
    out = {}
    groups = [("pooled", ev_pos)] + list(ev_pos.groupby("participant"))
    for name, g in groups:
        ttc = pd.to_numeric(g["ttc_onset"], errors="coerce").dropna()
        ttc = ttc[(ttc > 0) & (ttc < 3.0)]
        if len(ttc) < 10:
            continue
        t_min = fitted.get(name, {}).get("sim_params", {}).get("T_min")
        out[name] = {
            "n": int(len(ttc)),
            "q10": float(np.percentile(ttc, 10)),
            "q25": float(np.percentile(ttc, 25)),
            "q50": float(np.percentile(ttc, 50)),
            "fitted_T_min": t_min,
            "frac_above_T_min": (float((ttc >= t_min).mean())
                                 if t_min is not None else None),
        }
    return out


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    samples, events = gd.load_all()
    ev = ha.steering_events(events)
    ev, geoms = ld.attach_geometry(samples, ev)
    ev = ev[ev["s_c"].notna()]
    ev_pos = ev[ev["lead_onset"] > ha.MIN_LEAD].copy()
    print(f"[info] {len(ev_pos)} positive-lead steering events")

    out = {"n_events": int(len(ev_pos)),
           "model": ("h = clip(max(v*T_min, budget(gamma, lam, D0)), 0, "
                     "remaining); density_W = (W_ref/W)^gamma / W_ref"),
           "W_ref": W_REF,
           "selection": "grouped 5-fold CV, median |log(pred/obs)|"}
    out["pooled"] = fit_group(ev_pos, geoms, "pooled")
    for p, g in ev_pos.groupby("participant"):
        out[p] = fit_group(g, geoms, p)
        print(f"[{p}] {json.dumps(out[p]['cv'], default=float)}")
    print(f"[pooled] {json.dumps(out['pooled']['cv'], default=float)}")

    out["c2u_transfer"] = c2u_transfer(samples, events,
                                       out["pooled"]["sim_params"])
    print(f"[c2u] {json.dumps(out['c2u_transfer'], default=float)}")

    out["ttc_validation"] = ttc_validation(ev_pos, out)
    print(f"[ttc] {json.dumps(out['ttc_validation'], default=float)}")

    with open(RESULTS_DIR / "lookahead_floor_summary.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"wrote {RESULTS_DIR / 'lookahead_floor_summary.json'}")


if __name__ == "__main__":
    main()

"""Refit of the horizon-prediction module with a visuomotor-delay floor.

Motivation (eval-intermittent horizon_vs_gaze): the bare difficulty budget
underestimates gaze lead at the narrowest widths (human median lead ~0.015
at W=0.01 vs ~0.006 predicted) — the 1/W density explodes there and the
budget lead shrinks proportionally, but human gaze stays at least one
visuomotor delay ahead of the cursor at its current speed. Floored model:

    h_pred = clip( max( v * T_min,  h_budget(lam, D0) ),  0, remaining )

with h_budget the smallest h s.t. int_s^{s+h} (1/W + lam|kappa|) ds' = D0
(capped at the path end), v the cursor speed at fixation onset.

Fitting improvements over lookahead_difficulty.budget_fit:
  * grouped 5-fold cross-validation (folds split by trial, D0 quantiles and
    the power-law fit recomputed on each train split) instead of in-sample
    selection;
  * selection by median absolute log error (MALE) of predicted vs observed
    lead — the simulator consumes lead MAGNITUDES, so the fit should be
    calibrated, not just rank-correlated. Spearman rho is reported alongside
    for comparability with lookahead_summary.json.

Baselines under the same CV: bare budget (T_min=0), floor-only (h=v*T_min),
width power law lead=a*w^b.

Outputs results/lookahead_floor_summary.json, including per-participant
sim_params {T_min, lam, D0} for the simulator/eval configs.
Run: python3 refit_floor.py   (from eval/eval-gaze-cursor)
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gaze_data as gd  # noqa: E402
import horizon_analysis as ha  # noqa: E402
import lookahead_difficulty as ld  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"

T_MIN_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]
LAM_GRID = [0.0, 0.25, 0.5, 1.0, 2.0]
D0_QUANTS = [0.35, 0.5, 0.65, 0.8, 0.9]
N_FOLDS = 5


def _male(h_pred, h_obs):
    """Median absolute log error — robust, symmetric magnitude criterion."""
    ok = (h_pred > 1e-6) & (h_obs > 1e-6)
    if ok.sum() < 3:
        return np.inf
    return float(np.median(np.abs(np.log(h_pred[ok] / h_obs[ok]))))


def _budget_leads(ev, geoms, lam, D0):
    """Vector of budget leads h_budget(lam, D0) for every event (end-capped)."""
    h = np.empty(len(ev))
    for (p, tid), g in ev.groupby(["participant", "trial_id"]):
        geom = geoms[(p, tid)]
        hp, _ = geom.solve_h(g["s_c"].to_numpy(), lam, D0)
        h[ev.index.get_indexer(g.index)] = hp
    return h


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

    # Cache budget leads per (lam, D0) — D0 values are train-quantiles, so
    # cache keyed on the rounded value to reuse across folds when equal.
    cache = {}

    def budget_leads(lam, D0):
        key = (lam, round(D0, 6))
        if key not in cache:
            cache[key] = _budget_leads(ev, geoms, lam, D0)
        return cache[key]

    # ---- CV over the (lam, q, T_min) grid
    combo_pred = {}   # (lam, q, T_min) -> out-of-fold predictions
    pl_pred = np.full(len(ev), np.nan)
    for f in range(N_FOLDS):
        tr = folds != f
        te = folds == f
        if te.sum() == 0 or tr.sum() < 20:
            continue
        for lam in LAM_GRID:
            D_act = (ev.loc[tr, "D_W_act"] + lam * ev.loc[tr, "D_K_act"]).to_numpy()
            for q in D0_QUANTS:
                D0 = float(np.quantile(D_act, q))
                hb = budget_leads(lam, D0)
                for T_min in T_MIN_GRID:
                    hp = _floored(hb[te], v[te], T_min, remaining[te])
                    combo_pred.setdefault((lam, q, T_min),
                                          np.full(len(ev), np.nan))[te] = hp
        pl_params, _ = ha.fit_horizon_power(ev.loc[tr])
        pl_pred[te] = ha.predict_lead(pl_params, ev.loc[te, "width"])

    def score(pred):
        ok = np.isfinite(pred)
        return {"male_cv": _male(pred[ok], h_obs[ok]),
                "rho_cv": float(ha.spearman(pred[ok], h_obs[ok]))}

    scores = {c: score(p) for c, p in combo_pred.items()}
    best_full = min(scores, key=lambda c: scores[c]["male_cv"])
    best_bare = min((c for c in scores if c[2] == 0.0),
                    key=lambda c: scores[c]["male_cv"])
    floor_only = {}
    for T_min in T_MIN_GRID:
        if T_min == 0.0:
            continue
        hp = np.clip(v * T_min, 0.0, remaining)
        floor_only[T_min] = score(hp)  # no fitted piece -> CV unnecessary
    best_floor = min(floor_only, key=lambda t: floor_only[t]["male_cv"])

    # ---- final fit on all events with the selected (lam, q, T_min)
    lam, q, T_min = best_full
    D_act = (ev["D_W_act"] + lam * ev["D_K_act"]).to_numpy()
    D0 = float(np.quantile(D_act, q))
    h_final = _floored(budget_leads(lam, D0), v, T_min, remaining)
    frac_floor = float(np.mean((v * T_min > budget_leads(lam, D0)) & (h_final > 0)))

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
            "floored_budget": {"lam": lam, "D0_quantile": q, "T_min": T_min,
                               **scores[best_full]},
            "bare_budget": {"lam": best_bare[0], "D0_quantile": best_bare[1],
                            **scores[best_bare]},
            "floor_only": {"T_min": best_floor, **floor_only[best_floor]},
            "power_law": score(pl_pred),
        },
        "sim_params": {"T_min": T_min, "lam": lam, "D0": D0},
        "insample_rho": float(ha.spearman(h_final, h_obs)),
        "insample_male": _male(h_final, h_obs),
        "frac_floor_active": frac_floor,
        "lead_by_width": by_width,
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    samples, events = gd.load_all()
    ev = ha.steering_events(events)
    ev, geoms = ld.attach_geometry(samples, ev)
    ev = ev[ev["s_c"].notna()]
    ev_pos = ev[ev["lead_onset"] > ha.MIN_LEAD].copy()
    print(f"[info] {len(ev_pos)} positive-lead steering events")

    out = {"n_events": int(len(ev_pos)),
           "model": "h = clip(max(v*T_min, budget(lam, D0)), 0, remaining)",
           "selection": "grouped 5-fold CV, median |log(pred/obs)|"}
    out["pooled"] = fit_group(ev_pos, geoms, "pooled")
    for p, g in ev_pos.groupby("participant"):
        out[p] = fit_group(g, geoms, p)
        print(f"[{p}] {json.dumps(out[p]['cv'], default=float)}")
    print(f"[pooled] {json.dumps(out['pooled']['cv'], default=float)}")

    with open(RESULTS_DIR / "lookahead_floor_summary.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"wrote {RESULTS_DIR / 'lookahead_floor_summary.json'}")


if __name__ == "__main__":
    main()

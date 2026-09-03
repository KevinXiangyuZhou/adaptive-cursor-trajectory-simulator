"""Pooled gaze-module calibration on the good-quality new participants.

Redesign (verified 2026-09-02 on the corrected 10-participant batch):
  * constrained/unconstrained dichotomy is universal -> density 0 in free space;
  * NO width grading of the corridor toll for this cohort -> gamma = 0:
        rho(s) = 1/W_ref   in a corridor (W <= 0.15 m),   0 in free space
    so the budget lead is a constant distance D0*W_ref per corridor stretch;
  * visuomotor speed floor  h >= v * T_min  (positive speed dependence verified);
  * curvature slows the MOVEMENT, not the lookahead: turn-time deadline
        t_plan = max(T0, lead/v_max) + tau * theta_lead * (W_ref/W)^beta_t.

Fits, POOLED over PARTICIPANTS (one constant set for the cohort model):
  1. lead model: D0, T_min  (gamma=0), vs. the gamma-free baseline for reference;
  2. turn-time: T0, tau, beta_t (LAD, pooled crossing events);
  3. replan latency: median / CV of (fixation duration - crossing time).

Events: canonical merged keep=True, round-pruned (horizon_local_predictors.
round_quality), lead > 3 mm, end-capped excluded.

Usage: python fit_gaze_pooled.py [--pool p04 p06 p07 p09 p10]
Saves results/pooled10_gaze_constants.json.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import gaze_data as gd, lookahead_difficulty as ld
import horizon_local_predictors as hp
from turn_time_clean import lad

PROC = HERE.parents[1] / "human_data" / "processed_gaze_events"
W_REF = 0.026
FREE_W = 0.15
V_FLOOR = 0.02


def build_pool(letters):
    """Pooled event table + per-trial geometry caches."""
    events, geos = [], {}
    for L in letters:
        s = gd.load_samples(L)
        cl_all = pd.read_csv(PROC / f"{L}_fixation_events_clean.csv")
        ev = gd.fixation_events(s)
        ev = ev[~ev["tunnel_type"].astype(str).str.contains("pointing")]
        ev, geoms = ld.attach_geometry(s, ev)
        ok, _ = hp.round_quality(s, cl_all, geoms, L)
        cl = cl_all[cl_all["keep"] & np.array(
            [tuple(x) in ok for x in cl_all[["trial_id", "block_id"]].to_numpy()])].copy()
        cl = cl.merge(ev[["trial_id", "block_id", "fixation_id", "speed_onset"]],
                      on=["trial_id", "block_id", "fixation_id"], how="left")
        for _, r in cl.iterrows():
            g = geoms.get((L, r["trial_id"]))
            if g is None or not np.isfinite(r["lead_corr"]) or r["lead_corr"] <= 0.003:
                continue
            if r["s_c"] + r["lead_corr"] >= g.s_end - 0.005:
                continue
            geos[(L, r["trial_id"])] = g
            events.append((L, r["trial_id"], float(r["s_c"]), float(r["lead_corr"]),
                           max(float(r["speed_onset"]) if np.isfinite(r["speed_onset"]) else V_FLOOR, V_FLOOR)))
        # keep samples/geoms for the turn-time pass
        geos[("_samples", L)] = (s, cl, geoms)
    return pd.DataFrame(events, columns=["L", "tid", "s_c", "lead", "v"]), geos


def cum_density(g, gamma):
    W = np.clip(np.asarray(g.W, float), 1e-3, None)
    rho = (W_REF / W) ** gamma / W_REF
    rho = np.where(np.asarray(g.W, float) > FREE_W, 0.0, rho)
    ds = np.diff(g.s)
    return np.concatenate([[0.0], np.cumsum(0.5 * (rho[1:] + rho[:-1]) * ds)])


def predict(ev, geos, D0, T_min, gamma):
    cds = {}
    out = np.empty(len(ev))
    for i, r in enumerate(ev.itertuples()):
        key = (r.L, r.tid)
        if key not in cds:
            cds[key] = cum_density(geos[key], gamma)
        C = cds[key]; g = geos[key]
        target = float(np.interp(r.s_c, g.s, C)) + D0
        s_b = g.s_end if target >= C[-1] else float(np.interp(target, C, g.s))
        h = max(s_b - r.s_c, r.v * T_min)
        out[i] = np.clip(min(h, g.s_end - r.s_c), 1e-4, None)
    return out


def loss(ev, geos, D0, T_min, gamma):
    p = predict(ev, geos, D0, T_min, gamma)
    return float(np.mean(np.abs(np.log(p / ev["lead"].to_numpy()))))


def fit_lead(ev, geos, gamma_free):
    best = None
    if gamma_free:
        starts = [(1.2, 0.1, 1.0), (0.8, 0.2, 0.5), (1.5, 0.05, 0.2)]
        f = lambda x: loss(ev, geos, max(x[0], 0.05), np.clip(x[1], 0.0, 0.8), np.clip(x[2], 0.0, 1.5))
    else:
        starts = [(1.0, 0.15), (1.6, 0.05), (0.6, 0.3)]
        f = lambda x: loss(ev, geos, max(x[0], 0.05), np.clip(x[1], 0.0, 0.8), 0.0)
    for x0 in starts:
        r = minimize(f, x0, method="Nelder-Mead",
                     options={"maxiter": 200, "xatol": 1e-3, "fatol": 1e-4})
        if best is None or r.fun < best.fun:
            best = r
    return best


def crossings(letters, geos):
    rows = []
    for L in letters:
        s, cl, geoms = geos[("_samples", L)]
        for (tid, bid), grp in cl.groupby(["trial_id", "block_id"]):
            g = geoms.get((L, tid))
            if g is None:
                continue
            blk = s[(s["trial_id"] == tid) & (s["block_id"] == bid) & s["cursor_x"].notna()].sort_values("t")
            if len(blk) < 5:
                continue
            s_cur, _ = g.project(blk["cursor_x"].to_numpy(), blk["cursor_y"].to_numpy())
            s_mono = np.maximum.accumulate(s_cur)
            t_arr = blk["t"].to_numpy()
            for _, r in grp.iterrows():
                if not np.isfinite(r["lead_corr"]) or r["lead_corr"] <= 0.003:
                    continue
                s_t = r["s_c"] + r["lead_corr"]
                if s_t > g.s_end - 1e-6:
                    continue
                j = np.searchsorted(s_mono, s_t)
                if j >= len(t_arr):
                    continue
                T = t_arr[j] - r["t_onset"]
                if not (0 < T < 1.5):
                    continue
                th = float(np.interp(s_t, g.s, g.PHI) - np.interp(r["s_c"], g.s, g.PHI))
                W = float(np.clip(g.width_at(r["s_c"]), 1e-3, 1.0))
                rows.append((T, th, W, float(r["duration_s"])))
    return np.array(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", nargs="*", default=["p04", "p06", "p07", "p09", "p10"])
    a = ap.parse_args()
    ev, geos = build_pool(a.pool)
    print(f"pool {a.pool}: {len(ev)} lead events "
          f"({ {L: int((ev['L']==L).sum()) for L in a.pool} })", flush=True)

    r0 = fit_lead(ev, geos, gamma_free=False)
    D0, T_min = float(max(r0.x[0], 0.05)), float(np.clip(r0.x[1], 0.0, 0.8))
    print(f"redesign (gamma=0):  D0={D0:.3f} (lead const {D0*W_REF*1000:.1f}mm) "
          f"T_min={T_min:.3f}s  log-loss {r0.fun:.4f}")
    r1 = fit_lead(ev, geos, gamma_free=True)
    print(f"baseline (gamma free): D0={r1.x[0]:.3f} T_min={np.clip(r1.x[1],0,0.8):.3f} "
          f"gamma={np.clip(r1.x[2],0,1.5):.3f}  log-loss {r1.fun:.4f}", flush=True)
    per = {L: loss(ev[ev["L"] == L], geos, D0, T_min, 0.0) for L in a.pool}
    print("per-participant redesign loss:", {k: round(v, 3) for k, v in per.items()})

    d = crossings(a.pool, geos)
    T, th, W, dur = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
    best = None
    for beta in (0.0, 0.5, 1.0, 1.5):
        coef, mad = lad(th * (W_REF / W) ** beta, T)
        if best is None or mad < best[2]:
            best = (beta, coef, mad)
    beta_t, coef, mad = best
    print(f"turn-time (pooled, n={len(d)}): T = {coef[0]:.3f} + {coef[1]:.3f}*theta*(Wref/W)^{beta_t:g}  MAD {mad:.3f}")
    lat = np.clip(dur - T, 0.0, None)
    lat = lat[lat > 0]
    lat_med = float(np.median(lat)); lat_cv = float(np.std(lat) / max(np.mean(lat), 1e-9))
    print(f"replan latency (dwell - crossing): median {lat_med:.3f}s CV {lat_cv:.2f} (n={len(lat)})")

    out = {"pool": a.pool, "n_events": int(len(ev)),
           "lead": {"gamma": 0.0, "D0": D0, "T_min": T_min, "W_ref": W_REF,
                     "log_loss": float(r0.fun), "per_participant_loss": per,
                     "baseline_gamma_free": {"D0": float(r1.x[0]), "T_min": float(np.clip(r1.x[1],0,0.8)),
                                              "gamma": float(np.clip(r1.x[2],0,1.5)), "log_loss": float(r1.fun)}},
           "turn_time": {"T0": float(coef[0]), "tau": float(coef[1]), "beta_t": float(beta_t),
                          "mad": float(mad), "n": int(len(d))},
           "latency": {"median_s": lat_med, "cv": lat_cv, "n": int(len(lat))}}
    json.dump(out, open(HERE / "results" / "pooled10_gaze_constants.json", "w"), indent=2)
    print("saved results/pooled10_gaze_constants.json"); print("DONE")


if __name__ == "__main__":
    main()

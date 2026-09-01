"""Calibrate the additive gaze look-ahead density per participant and benchmark it:

    rho(s) = (W_ref/W(s))^gamma / W_ref  +  lam * |kappa(s)| * (W_ref/W(s))^beta

    lead h at cursor position s1 solves  int_{s1}^{s1+h} rho = D0.

Special cases: width-only (lam=0, the current model) and curvature-weighted
(first term ~0). Fitted on steering fixation-onset events (lead > 3 mm, blink-
filtered), objective = mean |log(h_pred/h_obs)|; events whose window is capped by
the tunnel end are kept (prediction capped too). Reports per-type x width median
predicted vs observed leads — the diagnostic is straight vs curved at W=10.

Usage: python fit_budget_additive.py [--letters A B C] [--time-limit 120]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import gaze_data as gd, lookahead_difficulty as ld

W_REF = 0.026


def build_events(L):
    s = gd.load_samples(L); ev = gd.fixation_events(s)
    ev = ev[(~ev["tunnel_type"].astype(str).str.contains("pointing")) & (ev["lead_onset"] > 0.003) & (~ev["blink_corrupted"])]
    ev, geoms = ld.attach_geometry(s, ev)
    ev = ev.dropna(subset=["s_c", "lead_onset"])
    return ev, geoms


def predict(ev, geoms, D0, gamma, lam, beta):
    out = np.full(len(ev), np.nan)
    for i, (_, r) in enumerate(ev.iterrows()):
        g = geoms.get((r["participant"], r["trial_id"]))
        if g is None: continue
        W = np.clip(g.W, 1e-4, 1.0)
        rho = (W_REF / W) ** gamma / W_REF
        kap = np.gradient(g.PHI, g.s)          # PHI is cumulative ∫|κ|ds
        rho = rho + lam * kap * (W_REF / W) ** beta
        C = np.concatenate([[0.0], np.cumsum(0.5 * (rho[1:] + rho[:-1]) * np.diff(g.s))])
        c1 = np.interp(r["s_c"], g.s, C); target = c1 + D0
        if target >= C[-1]:
            out[i] = g.s_end - r["s_c"]
        else:
            out[i] = np.interp(target, C, g.s) - r["s_c"]
    return np.clip(out, 1e-4, None)


def loss(ev, geoms, params):
    D0, gamma, lam, beta = params
    h = predict(ev, geoms, D0, gamma, lam, beta)
    obs = ev["lead_onset"].to_numpy()
    ok = np.isfinite(h)
    return float(np.mean(np.abs(np.log(h[ok] / obs[ok]))))


def fit(ev, geoms, x0, bounds, fixed=None, time_limit=120, seed=42):
    import cma, time
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    def obj(z):
        p = lo + (hi - lo) * np.clip(z, 0, 1)
        if fixed:
            for k, v in fixed.items(): p[k] = v
        return loss(ev, geoms, p)
    z0 = (np.array(x0) - lo) / (hi - lo)
    es = cma.CMAEvolutionStrategy(z0.tolist(), 0.25, {"bounds": [[0]*4, [1]*4], "popsize": 10, "seed": seed, "verbose": -9, "verb_log": 0, "verb_filenameprefix": ""})
    t0 = time.time(); best = (obj(z0), np.array(z0))
    while not es.stop() and time.time() - t0 < time_limit:
        sols = es.ask(); fits = [obj(np.array(z)) for z in sols]; es.tell(sols, fits)
        i = int(np.argmin(fits))
        if fits[i] < best[0]: best = (fits[i], np.array(sols[i]))
    p = lo + (hi - lo) * np.clip(best[1], 0, 1)
    if fixed:
        for k, v in fixed.items(): p[k] = v
    return p, best[0]


def table(ev, geoms, params, label):
    h = predict(ev, geoms, *params)
    e = ev.copy(); e["h_pred"] = h
    print(f"    {label}: median pred/obs lead (mm) by type x width")
    for ty in ("straight", "corner", "mid_sinusoidal", "gentle_sinusoidal", "sharp_sinusoidal"):
        row = f"      {ty[:8]:<8}"
        for w in (0.01, 0.02, 0.03, 0.04, 0.05):
            g = e[(e["tunnel_type"] == ty) & (np.abs(e["width"] - w) < 1e-6)]
            row += f" | {np.median(g['h_pred'])*1000:4.0f}/{np.median(g['lead_onset'])*1000:4.0f}" if len(g) else " |    -/-  "
        print(row, flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--letters", nargs="*", default=["B", "A", "C"])
    ap.add_argument("--time-limit", type=float, default=120.0)
    a = ap.parse_args()
    bounds = [(0.1, 4.0), (0.0, 2.0), (0.0, 0.6), (0.0, 3.0)]   # D0, gamma, lam (m/rad), beta
    results = {}
    for L in a.letters:
        ev, geoms = build_events(L)
        print(f"== {L}: {len(ev)} events", flush=True)
        p_w, l_w = fit(ev, geoms, [1.2, 1.0, 0.0, 1.0], bounds, fixed={2: 0.0}, time_limit=a.time_limit)
        p_a, l_a = fit(ev, geoms, [max(p_w[0], .3), p_w[1], 0.05, 1.0], bounds, time_limit=2 * a.time_limit)
        print(f"  width-only : D0={p_w[0]:.2f} gamma={p_w[1]:.2f}                    log-loss {l_w:.4f}")
        print(f"  additive   : D0={p_a[0]:.2f} gamma={p_a[1]:.2f} lam={p_a[2]:.3f} beta={p_a[3]:.2f}  log-loss {l_a:.4f}", flush=True)
        table(ev, geoms, p_w, "width-only"); table(ev, geoms, p_a, "additive")
        results[L] = {"width_only": {"D0": p_w[0], "gamma": p_w[1], "loss": l_w},
                      "additive": {"D0": p_a[0], "gamma": p_a[1], "lam": p_a[2], "beta": p_a[3], "loss": l_a}}
    out = "budget_additive_fit_" + "_".join(a.letters[:1]) + f"_{len(a.letters)}p.json" if len(a.letters) > 3 else "budget_additive_fit.json"
    json.dump(results, open(HERE / "results" / out, "w"), indent=2, default=float)
    print("saved results/" + out); print("DONE")


if __name__ == "__main__":
    main()

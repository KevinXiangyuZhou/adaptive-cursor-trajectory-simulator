"""Can anything beat the static speed law? Anticipation kernels and dynamics.

Per-round full series of cursor speed v(t) and centerline curvature sampled at
several offsets AHEAD of the cursor. Models compared with grouped 5-fold CV
over trials (metric: R2 on log v):

  A  static law            log v ~ log W + log(k0+1) + log(kmax50+1)
  B  distributed lag       log v ~ log W + sum_j b_j log(k(s+off_j)+1)
                           (the regression LEARNS the anticipation kernel b_j)
  C  GBM on B's features   (nonlinear ceiling on the same information)
  D  target + lag dynamics v_hat(t+dt) = v_hat + dt/tau (v_target - v_hat),
                           v_target = exp(model B prediction), OPEN LOOP from
                           the round start (never sees measured v); tau chosen
                           on the training folds.

D is the mechanistic candidate: the static law is the set-point, a first-order
lag supplies the accel/decel dynamics the static models cannot express.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from human_gaze_lead import dense_grid, project_dense, PROC, SUBSAMPLE, gd, ld
from local_speed_law import kappa_profile, TYPES, V_MIN, TRIM_S

LETTERS = ["p01", "p02", "p03", "p04", "p06", "p07", "p08", "p10"]  # p06/p08 recollected 2026-09-03
BASE = SCRIPT_DIR / "human-gaze-lead-10p"
OFFS_MM = [0, 12, 25, 50, 75, 100, 150]   # lookahead offsets (mm)
DT = 0.005 * SUBSAMPLE                     # sample period after subsampling
TAUS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8]


def extract():
    rounds = []
    step = 0.001
    for L in LETTERS:
        s = gd.load_samples(L)
        cl = pd.read_csv(PROC / f"{L}_fixation_events_clean.csv")
        cl = cl[cl["keep"] == True]  # noqa: E712
        ev = gd.fixation_events(s)
        ev = ev[~ev["tunnel_type"].astype(str).str.contains("pointing")]
        _, geoms = ld.attach_geometry(s, ev)
        trials = cl.groupby("trial_id").first().reset_index()
        trials = trials[trials["tunnel_type"].astype(str).isin(TYPES)]
        for _, tr in trials.iterrows():
            tid = tr["trial_id"]; g = geoms.get((L, tid))
            if g is None:
                continue
            s_d, P_d = dense_grid(g, step)
            K_d = np.interp(s_d, g.s, kappa_profile(g))
            W = float(tr["width"])
            st = s[(s["trial_id"] == tid) & s["cursor_x"].notna() & s["speed"].notna()]
            for bid in sorted(st["block_id"].dropna().unique()):
                b = st[st["block_id"] == bid].iloc[::SUBSAMPLE]
                if len(b) < 10:
                    continue
                t = b["t"].to_numpy(); t = t - t[0]
                keep = (t > TRIM_S) & (t < t[-1] - TRIM_S)
                if keep.sum() < 20:
                    continue
                b = b[keep]
                s_c = project_dense(s_d, P_d, b["cursor_x"].to_numpy(),
                                    b["cursor_y"].to_numpy())
                idx = np.clip((s_c / step).astype(int), 0, len(s_d) - 1)
                feats = {f"k{o}": K_d[np.clip(idx + o, 0, len(s_d) - 1)]
                         for o in OFFS_MM}
                v = b["speed"].to_numpy()
                ok = v >= V_MIN
                if ok.sum() < 20:
                    continue
                rounds.append(dict(participant=L, trial_id=int(tid), block=int(bid),
                                   W=W, v=v[ok], **{k: f[ok] for k, f in feats.items()}))
        print(f"{L}: {sum(r['participant'] == L for r in rounds)} rounds", flush=True)
    return rounds


def main():
    rounds = extract()
    groups = np.array([f"{r['participant']}/{r['trial_id']}" for r in rounds])
    rng = np.random.default_rng(0)
    ug = rng.permutation(np.unique(groups))
    fold_of = {g: i % 5 for i, g in enumerate(ug)}
    fid = np.array([fold_of[g] for g in groups])

    def stack(rs, cols):
        X = np.column_stack(
            [np.concatenate([np.full(len(r["v"]), np.log(r["W"])) for r in rs])] +
            [np.concatenate([np.log(r[c] + 1.0) for r in rs]) for c in cols])
        y = np.concatenate([np.log(r["v"]) for r in rs])
        return np.column_stack([np.ones(len(y)), X]), y

    static_cols = ["k0", "k50"]
    lag_cols = [f"k{o}" for o in OFFS_MM]

    res = {m: [0.0, 0.0] for m in "ABCD"}
    kern = []
    for k in range(5):
        tr = [r for r, f in zip(rounds, fid) if f != k]
        te = [r for r, f in zip(rounds, fid) if f == k]
        ytr_mean = np.mean(np.concatenate([np.log(r["v"]) for r in tr]))

        for m, cols in [("A", static_cols), ("B", lag_cols)]:
            Xtr, ytr = stack(tr, cols)
            Xte, yte = stack(te, cols)
            beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
            res[m][0] += ((yte - Xte @ beta) ** 2).sum()
            res[m][1] += ((yte - ytr_mean) ** 2).sum()
            if m == "B":
                kern.append(beta[2:])
                beta_B = beta

        from sklearn.ensemble import HistGradientBoostingRegressor
        Xtr, ytr = stack(tr, lag_cols)
        Xte, yte = stack(te, lag_cols)
        gbm = HistGradientBoostingRegressor(max_iter=300, random_state=0)
        gbm.fit(Xtr[:, 1:], ytr)
        res["C"][0] += ((yte - gbm.predict(Xte[:, 1:])) ** 2).sum()
        res["C"][1] += ((yte - ytr_mean) ** 2).sum()

        # D: open-loop first-order lag toward model-B target, tau picked on train
        def lag_sse(rs, tau, beta):
            sse = n = 0.0
            for r in rs:
                X = np.column_stack(
                    [np.ones(len(r["v"])), np.full(len(r["v"]), np.log(r["W"]))] +
                    [np.log(r[c] + 1.0) for c in lag_cols])
                vt = np.exp(X @ beta)
                vh = np.empty_like(vt); vh[0] = vt[0]
                a = DT / tau
                for i in range(1, len(vt)):
                    vh[i] = vh[i - 1] + a * (vt[i - 1] - vh[i - 1])
                sse += ((np.log(r["v"]) - np.log(np.maximum(vh, 1e-4))) ** 2).sum()
                n += len(vt)
            return sse
        tau_best = min(TAUS, key=lambda tau: lag_sse(tr, tau, beta_B))
        yte_all = np.concatenate([np.log(r["v"]) for r in te])
        res["D"][0] += lag_sse(te, tau_best, beta_B)
        res["D"][1] += ((yte_all - ytr_mean) ** 2).sum()
        print(f"fold {k}: tau_best={tau_best}", flush=True)

    names = {"A": "static law (W, k0, k50)", "B": "distributed lag (learned kernel)",
             "C": "GBM on lag features", "D": "target + first-order lag (open loop)"}
    print("\ngrouped 5-fold CV R2 (log v):")
    for m in "ABCD":
        print(f"  {m} {names[m]:38s} {1 - res[m][0] / res[m][1]:.3f}")
    kern = np.mean(kern, axis=0)
    print("\nlearned anticipation kernel (coef on log(kappa+1) at offset mm):")
    for o, c in zip(OFFS_MM, kern):
        print(f"  +{o:3d} mm: {c:+.3f}")


if __name__ == "__main__":
    main()

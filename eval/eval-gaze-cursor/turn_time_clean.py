"""Re-derive the turning-time deadline constants on the CLEANED gaze events.

For every kept fixation: target arc position s_t = s_c + lead_corr; crossing time =
first sample time (within the same round) at which the cursor's arc position reaches s_t;
theta = turning angle of the centerline between s_c and s_t; W = local width.
LAD fit: T = T0 + tau * theta * (W_ref/W)^beta, beta on {0, 0.5, 1, 1.5}.

Usage: python turn_time_clean.py [--letters A B C ...]
Saves results/turn_time_clean.json.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import gaze_data as gd, lookahead_difficulty as ld

W_REF = 0.026


def lad(X, T):
    A = np.c_[np.ones(len(T)), X]; w = np.ones_like(T)
    for _ in range(25):
        coef = np.linalg.lstsq(A * w[:, None], T * w, rcond=None)[0]
        r = np.abs(T - A @ coef); w = 1 / np.sqrt(np.maximum(r, 0.02))
    return coef, float(np.mean(np.abs(T - A @ coef)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--letters", nargs="*", default=["A", "B", "C"])
    a = ap.parse_args()
    out = {}
    for L in a.letters:
        s = gd.load_samples(L)
        cl = pd.read_csv(HERE.parents[1] / "human_data" / "processed_gaze_events" / f"{L}_fixation_events_clean.csv")
        cl = cl[cl["keep"] == True]  # noqa: E712
        ev = gd.fixation_events(s)
        ev = ev[~ev["tunnel_type"].astype(str).str.contains("pointing")]
        ev, geoms = ld.attach_geometry(s, ev)
        ev = ev.merge(cl[["trial_id", "block_id", "fixation_id", "lead_corr"]],
                      on=["trial_id", "block_id", "fixation_id"], how="inner")
        rows = []
        for (tid, bid), grp in ev.groupby(["trial_id", "block_id"]):
            g = geoms.get((L, tid))
            if g is None: continue
            blk = s[(s["trial_id"] == tid) & (s["block_id"] == bid) & s["cursor_x"].notna()].sort_values("t")
            if len(blk) < 5: continue
            s_cur, _ = g.project(blk["cursor_x"].to_numpy(), blk["cursor_y"].to_numpy())
            s_mono = np.maximum.accumulate(s_cur)
            t_arr = blk["t"].to_numpy()
            for _, r in grp.iterrows():
                if not np.isfinite(r["lead_corr"]) or r["lead_corr"] <= 0.003: continue
                s_t = r["s_c"] + r["lead_corr"]
                if s_t > g.s_end - 1e-6: continue
                j = np.searchsorted(s_mono, s_t)
                if j >= len(t_arr): continue
                T = t_arr[j] - r["t_onset"]
                if not (0 < T < 1.5): continue
                th = float(np.interp(s_t, g.s, g.PHI) - np.interp(r["s_c"], g.s, g.PHI))
                W = float(np.clip(g.width_at(r["s_c"]), 1e-3, 1.0))
                rows.append((T, th, W))
        d = np.array(rows)
        if len(d) < 30:
            print(f"{L}: only {len(d)} crossing events — skipped"); continue
        T, th, W = d[:, 0], d[:, 1], d[:, 2]
        best = None
        for beta in (0.0, 0.5, 1.0, 1.5):
            coef, mad = lad(th * (W_REF / W) ** beta, T)
            if best is None or mad < best[2]:
                best = (beta, coef, mad)
        beta, coef, mad = best
        out[L] = {"T0": float(coef[0]), "tau": float(coef[1]), "beta": beta, "mad": mad, "n": int(len(d))}
        print(f"{L}: n={len(d)}  T = {coef[0]:.3f} + {coef[1]:.3f}*theta*(Wref/W)^{beta:g}   MAD {mad:.4f}", flush=True)
    json.dump(out, open(HERE / "results" / "turn_time_clean.json", "w"), indent=2)
    print("saved results/turn_time_clean.json"); print("DONE")


if __name__ == "__main__":
    main()

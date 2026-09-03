"""Grid figures of human signed gaze lead over the sinusoid tasks (no model runs).

Figure families per participant, built from the same per-round lead data
(steering tasks: gentle_sinusoidal, sharp_sinusoidal, the normal sinusoid
stored with tunnel_type "None", plus EXTRA_TYPES; columns are always rounds
1-3, round k in ROUND_COLORS[k-1] as in human_gaze_lead.py / the fixation
maps; continuous lead only, no event markers):

  {L}/lead_by_width/W{w}mm.png    one figure per width, rows = task type
                                  (gentle / sharp / normal)  -> 3x3
  {L}/lead_by_curvature/{type}.png one figure per task type (same curvature),
                                  rows = width (10..50 mm)   -> 5x3
  data/{L}_steering_lead.csv     the plotted steering lead series (long
                                  format: one row per plotted sample, with
                                  task type, width, trial, round, drift flag)
  data/{L}_pointing_lead.csv     the plotted pointing lead series (with
                                  distance, target radius, Fitts ID)
  {L}/lead_by_curvature/pointing.png unconstrained pointing: columns =
                                  distance (D/3, 2D/3, D), rows = difficulty
                                  rank (target radius large -> small), the 3
                                  rounds OVERLAID per panel. Lead here =
                                  (gaze - cursor) projected on the round's
                                  start -> target axis (no tunnel centerline
                                  exists); rounds must travel >= 60% of the
                                  nominal distance to count.

Every subplot of a figure shares the SAME X AND Y SCALE (set from the pooled
data of that figure), so lead magnitude and trial duration compare directly
across panels.

Lead is recomputed exactly as in human_gaze_lead.py: canonical drift
correction (gated per-block offsets for DRIFT_PARTICIPANTS, else the global
pointing bias) followed by the ABC-style projection (dense ~1 mm grid, global
nearest point, no forward constraint, no clamp).

Usage: python human_gaze_lead_width_grid.py [--letters p01 ...] [--out-dir human-gaze-lead-10p]
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from human_gaze_lead import (dense_grid, project_dense, PROC, SUBSAMPLE,
                             gd, gc, ld, ROUND_COLORS)

# p05/p06/p08/p09 sessions are deprecated (no per-participant folder here)
DEFAULT_LETTERS = ["p01", "p02", "p03", "p04", "p07", "p10"]
TYPES = [("gentle_sinusoidal", "gentle"),
         ("sharp_sinusoidal", "sharp"),
         ("None", "normal")]  # tunnel_type "None" = the normal sinusoid
# extra types that get a by-curvature (widths x rounds) figure but no row in
# the by-width grids
EXTRA_TYPES = [("straight", "straight"), ("corner", "corner")]
N_ROUNDS = 3


def shared_limits(series_list):
    """One x/y scale for a whole figure from its pooled (t, lead) series."""
    all_lead = np.concatenate([lead for _, lead in series_list])
    lo, hi = np.percentile(all_lead, [0.5, 99.5])
    pad = 0.06 * (hi - lo + 1e-9)
    ylim = (min(lo, 0.0) - pad, max(hi, 0.0) + pad)
    tmax = max(t.max() for t, _ in series_list)
    return (-0.02 * tmax, 1.02 * tmax), ylim


def render_grid(out_png, rows, cells, suptitle):
    """rows: list of (row_key, row_label); cells: {(row_key, col)} ->
    (t, lead, drift_corrected). One shared x/y scale."""
    xlim, ylim = shared_limits([(t, lead) for t, lead, _ in cells.values()])

    fig, axes = plt.subplots(len(rows), N_ROUNDS,
                             figsize=(4.2 * N_ROUNDS, 2.4 * len(rows)),
                             squeeze=False, sharex=True, sharey=True)
    n_corr = 0
    for r, (rk, rlabel) in enumerate(rows):
        for c in range(N_ROUNDS):
            ax = axes[r, c]
            cell = cells.get((rk, c))
            if cell is None:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", color="0.6")
            else:
                t, lead, corr = cell
                col = ROUND_COLORS[c % len(ROUND_COLORS)]
                ax.plot(t, lead, ".", ms=1.1, color=col, alpha=0.5)
                if corr:
                    ax.text(0.02, 0.03, "drift-corr", transform=ax.transAxes,
                            fontsize=7, color="0.45")
                    n_corr += 1
            ax.axhline(0.0, color="0.4", lw=0.8)
            ax.grid(alpha=0.25)
            if r == 0:
                ax.set_title(f"round {c + 1}", fontsize=10,
                             color=ROUND_COLORS[c % len(ROUND_COLORS)])
            if c == 0:
                ax.set_ylabel(f"{rlabel}\nsigned lead (m)")
            if r == len(rows) - 1:
                ax.set_xlabel("time since round start (s)")
    axes[0, 0].set_xlim(*xlim)
    axes[0, 0].set_ylim(*ylim)
    fig.suptitle(suptitle + f"\n({n_corr}/{len(cells)} rounds drift-corrected; "
                 f"ABC-style global projection)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def pointing_data(s, bias, off_by_block, min_travel=0.6):
    """Per-trial pointing lead: (gaze - cursor) projected on the round's
    start -> target axis. Returns {(dist, radius, trial_id)} -> list of
    (t, lead, drift_corrected) rounds (only rounds whose cursor travels at
    least min_travel * nominal distance)."""
    pt = s[s["tunnel_type"].astype(str) == "unconstrained_pointing"]
    out = {}
    for tid, g in pt.groupby("trial_id"):
        cj = g["condition_json"].dropna()
        if cj.empty:
            continue
        j = json.loads(cj.iloc[0])
        dist, rad = float(j["distance"]), float(j["targetRadius"])
        g = g[g["cursor_x"].notna() & g["gaze_task_x"].notna()]
        rounds = []
        for bid in sorted(g["block_id"].dropna().unique()):
            b = g[g["block_id"] == bid].sort_values("t")
            if len(b) < 20:
                continue
            P = b[["cursor_x", "cursor_y"]].to_numpy()
            p0, p1 = P[:3].mean(axis=0), P[-3:].mean(axis=0)
            if np.linalg.norm(p1 - p0) < min_travel * dist:
                continue
            u = (p1 - p0) / np.linalg.norm(p1 - p0)
            ox, oy = off_by_block.get(bid, (bias[0], bias[1]))
            G = np.column_stack([b["gaze_task_x"] - ox, b["gaze_task_y"] - oy])
            lead = (G - P) @ u
            rounds.append(((b["t"] - b["t"].min()).to_numpy(), lead,
                           bid in off_by_block))
            if len(rounds) == N_ROUNDS:
                break
        if rounds:
            out[(dist, rad, int(tid))] = rounds
    return out


def render_pointing(out_png, pcells, suptitle):
    """Columns = distance (ascending), rows = difficulty rank (radius large ->
    small within each distance), the rounds overlaid per panel."""
    dists = sorted({d for d, _, _ in pcells})
    cols = {d: sorted([k for k in pcells if k[0] == d], key=lambda k: -k[1])
            for d in dists}
    nrows = max(len(v) for v in cols.values())
    xlim, ylim = shared_limits([(t, lead) for rounds in pcells.values()
                                for t, lead, _ in rounds])

    fig, axes = plt.subplots(nrows, len(dists),
                             figsize=(4.2 * len(dists), 2.4 * nrows),
                             squeeze=False, sharex=True, sharey=True)
    n_corr = n_rounds = 0
    for c, d in enumerate(dists):
        for r in range(nrows):
            ax = axes[r, c]
            if r >= len(cols[d]):
                ax.set_visible(False)
                continue
            dist, rad, tid = cols[d][r]
            iD = np.log2(dist / (2 * rad) + 1)
            for ri, (t, lead, corr) in enumerate(pcells[(dist, rad, tid)]):
                col = ROUND_COLORS[ri % len(ROUND_COLORS)]
                ax.plot(t, lead, ".", ms=1.6, color=col, alpha=0.55)
                n_corr += corr; n_rounds += 1
            ax.axhline(0.0, color="0.4", lw=0.8)
            ax.grid(alpha=0.25)
            ax.set_title(f"R={rad*1000:g} mm  ID={iD:.2f}  (t{tid})", fontsize=9)
            if r == 0:
                ax.text(0.5, 1.28, f"distance {dist:.2f} m", transform=ax.transAxes,
                        ha="center", fontsize=11)
            if c == 0:
                ax.set_ylabel("signed lead (m)")
            if r == nrows - 1:
                ax.set_xlabel("time since round start (s)")
    axes[0, 0].set_xlim(*xlim)
    axes[0, 0].set_ylim(*ylim)
    fig.suptitle(suptitle + f"\n({n_corr}/{n_rounds} rounds drift-corrected; lead = "
                 f"gaze - cursor projected on the round's start->target axis)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--letters", nargs="*", default=DEFAULT_LETTERS)
    ap.add_argument("--out-dir", default=str(SCRIPT_DIR / "human-gaze-lead-10p"))
    a = ap.parse_args()
    out_dir = Path(a.out_dir)

    for L in a.letters:
        s = gd.load_samples(L)
        bias = gc.estimate_bias(s)
        drift = gc.estimate_block_drift(s) if L in gc.DRIFT_PARTICIPANTS else {}
        off_by_block = {b: (v[0], v[1]) for b, v in drift.items()
                        if np.hypot(v[0] - bias[0], v[1] - bias[1]) > gc.DRIFT_GATE_M}
        cl = pd.read_csv(PROC / f"{L}_fixation_events_clean.csv")
        cl = cl[cl["keep"] == True]  # noqa: E712
        ev = gd.fixation_events(s)
        ev = ev[~ev["tunnel_type"].astype(str).str.contains("pointing")]
        _, geoms = ld.attach_geometry(s, ev)

        trials = cl.groupby("trial_id").first().reset_index()
        trials["w_mm"] = (trials["width"] * 1000).round(1)
        trials["type_str"] = trials["tunnel_type"].astype(str)
        widths = sorted(set.intersection(*[set(trials.loc[trials["type_str"] == t, "w_mm"])
                                           for t, _ in TYPES]))

        # data[(type, w_mm, round)] = (t, lead, drift-corrected)
        data, tids = {}, {}
        for tt, _ in TYPES + EXTRA_TYPES:
            for w_mm in sorted(trials.loc[trials["type_str"] == tt, "w_mm"].unique()):
                sel = trials[(trials["type_str"] == tt) & (trials["w_mm"] == w_mm)]
                if sel.empty:
                    continue
                tid = sel.iloc[0]["trial_id"]
                g = geoms.get((L, tid))
                if g is None:
                    continue
                tids[(tt, w_mm)] = tid
                s_d, P_d = dense_grid(g)
                st = s[(s["trial_id"] == tid) & s["cursor_x"].notna() & s["gaze_task_x"].notna()]
                blocks = [b for b in sorted(st["block_id"].unique())
                          if (st["block_id"] == b).sum() >= 10 * SUBSAMPLE]
                for c, bid in enumerate(blocks[:N_ROUNDS]):
                    b = st[st["block_id"] == bid].iloc[::SUBSAMPLE]
                    ox, oy = off_by_block.get(bid, (bias[0], bias[1]))
                    s_c = project_dense(s_d, P_d, b["cursor_x"].to_numpy(), b["cursor_y"].to_numpy())
                    s_g = project_dense(s_d, P_d, (b["gaze_task_x"] - ox).to_numpy(),
                                        (b["gaze_task_y"] - oy).to_numpy())
                    t0 = b["t"].min()
                    data[(tt, w_mm, c)] = ((b["t"] - t0).to_numpy(), s_g - s_c,
                                           bid in off_by_block)

        # figures 1: one per width, rows = task type
        wdir = out_dir / L / "lead_by_width"
        wdir.mkdir(parents=True, exist_ok=True)
        for w_mm in widths:
            cells = {(tt, c): v for (tt, w, c), v in data.items() if w == w_mm}
            if not cells:
                continue
            rows = [(tt, f"{lab}  (t{int(tids[(tt, w_mm)])})" if (tt, w_mm) in tids else lab)
                    for tt, lab in TYPES]
            render_grid(wdir / f"W{w_mm:g}mm.png", rows, cells,
                        f"{L} — W={w_mm:g} mm: sinusoid tasks x rounds, one scale for all panels")
            print(f"{L}: by-width W={w_mm:g}mm -> {len(cells)} panels", flush=True)

        # figures 2: one per task type (same curvature), rows = width
        cdir = out_dir / L / "lead_by_curvature"
        cdir.mkdir(parents=True, exist_ok=True)
        for tt, lab in TYPES + EXTRA_TYPES:
            cells = {(w, c): v for (t2, w, c), v in data.items() if t2 == tt}
            if not cells:
                continue
            ws = sorted({w for w, _ in cells})
            rows = [(w, f"W={w:g} mm  (t{int(tids[(tt, w)])})" if (tt, w) in tids else f"W={w:g} mm")
                    for w in ws]
            kind = "sinusoid" if (tt, lab) in TYPES else "tunnel"
            render_grid(cdir / f"{lab}.png", rows, cells,
                        f"{L} — {lab} {kind}: widths x rounds, one scale for all panels")
            print(f"{L}: by-curvature {lab} -> {len(cells)} panels", flush=True)

        # figure 3: unconstrained pointing — distance x difficulty, rounds overlaid
        pcells = pointing_data(s, bias, off_by_block)

        # persist the plotted series (long format) under {out-dir}/data/
        ddir = out_dir / "data"
        ddir.mkdir(parents=True, exist_ok=True)
        lab_of = dict(TYPES + EXTRA_TYPES)
        rows_ = [pd.DataFrame({"participant": L, "tunnel_type": tt,
                               "type_label": lab_of[tt], "trial_id": tids[(tt, w)],
                               "width_mm": w, "round": c + 1, "drift_corrected": corr,
                               "t": t, "lead": lead})
                 for (tt, w, c), (t, lead, corr) in sorted(data.items())]
        pd.concat(rows_, ignore_index=True).to_csv(
            ddir / f"{L}_steering_lead.csv", index=False, float_format="%.6f")
        rows_ = [pd.DataFrame({"participant": L, "trial_id": tid,
                               "distance_m": dist, "target_radius_m": rad,
                               "fitts_id": np.log2(dist / (2 * rad) + 1),
                               "round": ri + 1, "drift_corrected": corr,
                               "t": t, "lead": lead})
                 for (dist, rad, tid), rounds in sorted(pcells.items())
                 for ri, (t, lead, corr) in enumerate(rounds)]
        if rows_:
            pd.concat(rows_, ignore_index=True).to_csv(
                ddir / f"{L}_pointing_lead.csv", index=False, float_format="%.6f")
        print(f"{L}: data -> {ddir}/{L}_steering_lead.csv, {L}_pointing_lead.csv", flush=True)
        if pcells:
            render_pointing(cdir / "pointing.png", pcells,
                            f"{L} — unconstrained pointing: distance x target size, "
                            f"3 rounds overlaid, one scale for all panels")
            print(f"{L}: by-curvature pointing -> {len(pcells)} panels", flush=True)
        print(f"{L}: done -> {out_dir / L}/", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()

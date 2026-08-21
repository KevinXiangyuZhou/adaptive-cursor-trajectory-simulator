"""Look-ahead-window difficulty analysis (the density view of the steering ID).

The steering ID is read as a local density along the tunnel centerline:

    id_W(s) = 1/W(s)       (steering-law width density)
    id_K(s) = |kappa(s)|   (curvature density; its integral = turning angle)

so integrating each over the whole tunnel gives ID_W and ID_K. The quantity of
interest is the difficulty inside the gaze look-ahead window: with the cursor
at arc-length s and gaze lead h (signed arc-length gaze - cursor, the CSV's
gaze_lead_signed at fixation onset),

    D_W(s, s+h) = int_s^{s+h} ds'/W(s')
    D_K(s, s+h) = int_s^{s+h} |kappa| ds'

Hypothesis: gaze lead shrinks when the integrated difficulty ahead is high.

Circularity note: integrating over the *actual* h window scales with h by
construction (D ~ density * h), so rho(h, D_actual) is mechanically biased
positive and is reported only as the naive number. The real tests:

  1. fixed-window: rho(h, D(s, s+h0)) for a fixed h0 grid; the hypothesis
     signature is negative rho. Events whose window does not fit before the
     tunnel end are dropped (end-of-trial leads shrink for goal reasons).
  2. two-segment tunnels, stratified by local width: within a stratum the
     local width is constant, so any rho(h, D_W_h0) is carried purely by the
     upcoming width change - width-ahead disentangled from width-here.
  3. corner trials: lead vs arc distance to the next 90-degree turn.
  4. budget constancy: is D(s, s+h_obs) more constant across events than when
     leads are shuffled across events? (permutation test on the CV)
  5. budget model (the horizon-prediction module): h_pred(s) solves
         int_s^{s+h} [1/W + lam*|kappa|] ds' = D0
     with (lam, D0) fitted per participant + pooled, benchmarked against the
     lead = a*width^b power law on the same events.

Geometry is rebuilt per trial from condition_json with the same generators the
experiment used (experiment.utils), and validated against the CSV's
json_local_width at the projected cursor position.

Outputs under results/: lookahead_summary.json + 4 figures.
Run: python3 lookahead_difficulty.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiment.environment import POINTING_Y_OFFSET  # noqa: E402
from experiment.utils import generateCornerPath, generateTunnelPath  # noqa: E402

import gaze_data as gd  # noqa: E402
import horizon_analysis as ha  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
H0_GRID = [0.02, 0.04, 0.06]   # fixed look-ahead windows (task units)
H0_MAIN = 0.04                 # headline window (~p75 of positive leads)
LAM_GRID = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]  # weight of |kappa| vs 1/W
D0_QUANTS = [0.35, 0.5, 0.65, 0.8, 0.9]  # budget candidates, quantiles of D_actual
N_BOOT = 500
N_PERM = 200
TWO_SEG = ("wide_to_narrow", "narrow_to_wide")
UNCON = ("unconstrained_pointing", "constrained_to_unconstrained")
FREE_W = 1e6      # effective width of free space: id_W = 1/W ~ 0
C2U_EXIT_X = 0.23  # tunnel exit of constrained_to_unconstrained (condition_json)
PATH_STEP = 0.002

COLORS = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green"}


# ------------------------------------------------------------------ geometry

class Geom:
    """Centerline polyline with arc length, width profile and the cumulative
    difficulty integrals IW(s) = int ds'/W and PHI(s) = int |kappa| ds'."""

    target_radius = None  # set for pointing/c2u tasks
    s_exit = None         # arc length of the tunnel exit (c2u only)

    def __init__(self, path: np.ndarray, width):
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        keep = np.concatenate([[True], seg > 1e-12])
        path = path[keep]
        self.path = path
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(seg)])
        n = len(path)
        self.W = np.full(n, float(width)) if np.isscalar(width) else np.asarray(width, float)

        invw = 1.0 / np.clip(self.W, 1e-6, None)
        self.IW = np.concatenate([[0.0], np.cumsum(0.5 * (invw[1:] + invw[:-1]) * seg)])

        d = np.diff(path, axis=0)
        dn = d / np.linalg.norm(d, axis=1, keepdims=True)
        ang = np.arccos(np.clip(np.sum(dn[:-1] * dn[1:], axis=1), -1.0, 1.0))
        self.PHI = np.zeros(n)
        if n > 2:
            self.PHI[1:n - 1] = np.cumsum(ang)
            self.PHI[n - 1] = self.PHI[n - 2]
        # turn points (sharp direction changes, e.g. 90-degree corners)
        self.turn_s = self.s[1:n - 1][ang > np.pi / 4] if n > 2 else np.array([])

    @property
    def s_end(self):
        return float(self.s[-1])

    def project(self, x, y):
        """Arc-length position and distance of point(s) projected on the path."""
        pts = np.column_stack([np.atleast_1d(x), np.atleast_1d(y)])
        P0 = self.path[:-1]
        d = np.diff(self.path, axis=0)
        L2 = np.sum(d * d, axis=1)
        s_out, dist_out = np.empty(len(pts)), np.empty(len(pts))
        for k, p in enumerate(pts):
            t = np.clip(np.sum((p - P0) * d, axis=1) / L2, 0.0, 1.0)
            proj = P0 + t[:, None] * d
            dd = np.sum((proj - p) ** 2, axis=1)
            i = int(np.argmin(dd))
            s_out[k] = self.s[i] + t[i] * np.sqrt(L2[i])
            dist_out[k] = np.sqrt(dd[i])
        return s_out, dist_out

    def window(self, s1, s2):
        """(D_W, D_K) integrated over [s1, s2] (clipped to the tunnel)."""
        s1 = np.clip(s1, 0.0, self.s_end)
        s2 = np.clip(s2, 0.0, self.s_end)
        dW = np.interp(s2, self.s, self.IW) - np.interp(s1, self.s, self.IW)
        dK = np.interp(s2, self.s, self.PHI) - np.interp(s1, self.s, self.PHI)
        return dW, dK

    def width_at(self, s):
        return np.interp(s, self.s, self.W)

    def solve_h(self, s1, lam, D0, cap_at_end=True):
        """Smallest h with int_{s1}^{s1+h} (1/W + lam|kappa|) = D0.

        cap_at_end=True clips the window at the tunnel end; False extends the
        tunnel virtually with its final width (ablation to separate the
        difficulty effect from the goal-proximity effect)."""
        IC = self.IW + lam * self.PHI
        target = np.interp(s1, self.s, IC) + D0
        s1 = np.atleast_1d(np.asarray(s1, float))
        target = np.atleast_1d(target)
        capped = target >= IC[-1]
        s2 = np.where(capped, self.s_end, np.interp(target, IC, self.s))
        if not cap_at_end:
            over = np.clip(target - IC[-1], 0.0, None)
            s2 = s2 + over * float(np.clip(self.W[-1], 1e-6, None))
        return s2 - s1, capped


def _dense_line(p0, p1, step=PATH_STEP):
    """Straight polyline from p0 to p1 with ~step spacing (endpoints exact)."""
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    n = max(2, int(np.ceil(np.linalg.norm(p1 - p0) / step)) + 1)
    t = np.linspace(0.0, 1.0, n)
    return p0[None, :] + t[:, None] * (p1 - p0)[None, :]


def build_geom(tunnel_type: str, cond: dict) -> Geom:
    if tunnel_type == "unconstrained_pointing":
        # straight free-space reach: start (0, 0.13) -> target; id_W ~ 0
        y_off = POINTING_Y_OFFSET.get(cond.get("targetPosition", "middle"), 0.0)
        path = _dense_line((0.0, 0.13), (cond["distance"], 0.13 + y_off))
        g = Geom(path, FREE_W)
        g.target_radius = float(cond.get("targetRadius", 0.01))
        return g
    if tunnel_type == "constrained_to_unconstrained":
        # horizontal tunnel (width segment1Width) to x=0.23, then free space
        # to the target; the direction change at the exit enters id_K
        exit_x = float(cond.get("transition_point", {}).get("taskX", C2U_EXIT_X))
        y_off = POINTING_Y_OFFSET.get(cond.get("targetPosition", "middle"), 0.0)
        seg1 = _dense_line((0.0, 0.13), (exit_x, 0.13))
        seg2 = _dense_line((exit_x, 0.13), (cond["distance"], 0.13 + y_off))
        path = np.vstack([seg1, seg2[1:]])
        W = np.full(len(path), FREE_W)
        W[:len(seg1)] = float(cond["segment1Width"])
        g = Geom(path, W)
        g.target_radius = float(cond.get("targetRadius", 0.01))
        g.s_exit = float(exit_x)
        return g
    if tunnel_type == "corner":
        path, w = generateCornerPath(
            width=cond.get("tunnelWidth", 0.03),
            num_corners=cond.get("numCorners", 2),
            corner_offset=cond.get("cornerOffset", 0.1),
        )
        return Geom(np.asarray(path, float), w)
    if tunnel_type in TWO_SEG:
        path, _ = generateTunnelPath(width=0.0, curvature=0.0)
        path = np.asarray(path, float)
        n = len(path)
        W = np.full(n, float(cond["segment1Width"]))
        W[n // 2:] = float(cond["segment2Width"])  # transition at taskX=0.23
        return Geom(path, W)
    # straight + sinusoidal families
    path, w = generateTunnelPath(
        width=cond.get("tunnelWidth", 0.02), curvature=cond.get("curvature", 0.0)
    )
    return Geom(np.asarray(path, float), w)


# ------------------------------------------------------------------ assembly

def attach_geometry(samples: pd.DataFrame, ev: pd.DataFrame):
    """Project every fixation-onset cursor position onto its trial centerline
    and compute window integrals. Returns (ev with new columns, geoms dict)."""
    cond_rows = (
        samples.dropna(subset=["condition_json"])
        .groupby(["participant", "trial_id"])[["tunnel_type", "condition_json"]]
        .first()
    )
    geoms, cache = {}, {}
    for (p, tid), row in cond_rows.iterrows():
        key = (row["tunnel_type"], row["condition_json"])
        if key not in cache:
            cache[key] = build_geom(row["tunnel_type"], json.loads(row["condition_json"]))
        geoms[(p, tid)] = cache[key]

    ev = ev.copy()
    for col in ("s_c", "proj_dist", "s_end", "W_geo", "d_next_turn",
                "D_W_act", "D_K_act"):
        ev[col] = np.nan
    for (p, tid), g in ev.groupby(["participant", "trial_id"]):
        geom = geoms.get((p, tid))
        if geom is None:
            continue
        s_c, dist = geom.project(g["cursor_x"].to_numpy(), g["cursor_y"].to_numpy())
        h = g["lead_onset"].to_numpy()
        dW, dK = geom.window(s_c, s_c + np.clip(h, 0.0, None))
        ev.loc[g.index, "s_c"] = s_c
        ev.loc[g.index, "proj_dist"] = dist
        ev.loc[g.index, "s_end"] = geom.s_end
        ev.loc[g.index, "W_geo"] = geom.width_at(s_c)
        ev.loc[g.index, "D_W_act"] = dW
        ev.loc[g.index, "D_K_act"] = dK
        if len(geom.turn_s):
            ahead = geom.turn_s[None, :] - s_c[:, None]
            ahead = np.where(ahead > 0, ahead, np.inf)
            ev.loc[g.index, "d_next_turn"] = np.min(ahead, axis=1)

    for h0 in H0_GRID:
        tag = f"{h0:g}"
        ev[f"fits_{tag}"] = ev["s_c"] + h0 <= ev["s_end"] + 1e-9
        dW = np.full(len(ev), np.nan)
        dK = np.full(len(ev), np.nan)
        for (p, tid), g in ev.groupby(["participant", "trial_id"]):
            geom = geoms.get((p, tid))
            if geom is None:
                continue
            s_c = g["s_c"].to_numpy()
            a, b = geom.window(s_c, s_c + h0)
            dW[ev.index.get_indexer(g.index)] = a
            dK[ev.index.get_indexer(g.index)] = b
        ev[f"D_W_{tag}"] = dW
        ev[f"D_K_{tag}"] = dK
    return ev, geoms


# ------------------------------------------------------------------- stats

def cluster_boot_rho(x, y, clusters, n_boot=N_BOOT, seed=0):
    """Spearman rho with a cluster (trial) bootstrap CI."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    cl = np.asarray(clusters)
    rho = ha.spearman(x, y)
    uniq = np.unique(cl)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(cl == c) for c in pick])
        if len(np.unique(y[idx])) > 1 and len(np.unique(x[idx])) > 1:
            boots.append(ha.spearman(x[idx], y[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"rho": float(rho), "ci95": [float(lo), float(hi)], "n": int(len(x))}


def partial_spearman(x, y, z):
    """Rank correlation of x and y after residualizing both on rank(z)."""
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    rz = pd.Series(z).rank().to_numpy()
    def resid(a):
        A = np.column_stack([rz, np.ones_like(rz)])
        return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    ex, ey = resid(rx), resid(ry)
    den = np.sqrt(np.sum(ex * ex) * np.sum(ey * ey))
    return float(np.sum(ex * ey) / den) if den > 0 else np.nan


def within_stratum_rho(df, strat_cols, xcol, ycol, ctrl_col=None, min_n=8):
    """Pooled within-stratum Spearman: rank inside each stratum, center, pool.
    With ctrl_col, ranks are first residualized on the control's rank (partial)."""
    xs, ys, per = [], [], {}
    for key, g in df.groupby(strat_cols):
        if len(g) < min_n:
            continue
        rx = g[xcol].rank().to_numpy()
        ry = g[ycol].rank().to_numpy()
        if np.std(rx) == 0 or np.std(ry) == 0:
            continue
        if ctrl_col is not None:
            rz = g[ctrl_col].rank().to_numpy()
            A = np.column_stack([rz, np.ones_like(rz)])
            rx = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
            ry = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
        xs.append(rx - rx.mean()); ys.append(ry - ry.mean())
        name = "_".join(str(k) for k in (key if isinstance(key, tuple) else (key,)))
        per[name] = {"rho": ha.spearman(g[xcol], g[ycol]), "n": int(len(g))}
        if ctrl_col is not None:
            per[name]["rho_partial"] = partial_spearman(g[xcol], g[ycol], g[ctrl_col])
    if not xs:
        return {"rho_pooled_within": np.nan, "strata": per}
    x = np.concatenate(xs); y = np.concatenate(ys)
    rho = float(np.sum(x * y) / np.sqrt(np.sum(x * x) * np.sum(y * y)))
    return {"rho_pooled_within": rho, "n_pooled": int(len(x)), "strata": per}


def budget_fit(ev_pos, geoms, lam_grid=LAM_GRID, cap_at_end=True):
    """Fit (lam, D0): h_pred solves int (1/W + lam|kappa|) = D0. In-sample,
    scored by Spearman(h_pred, h_obs) - same footing as the 2-param power law."""
    best = None
    h_obs = ev_pos["lead_onset"].to_numpy()
    for lam in lam_grid:
        D_act = ev_pos["D_W_act"].to_numpy() + lam * ev_pos["D_K_act"].to_numpy()
        for q in D0_QUANTS:
            D0 = float(np.quantile(D_act, q))
            h_pred = np.empty(len(ev_pos))
            capped = np.zeros(len(ev_pos), bool)
            for (p, tid), g in ev_pos.groupby(["participant", "trial_id"]):
                geom = geoms[(p, tid)]
                hp, cp = geom.solve_h(g["s_c"].to_numpy(), lam, D0, cap_at_end)
                loc = ev_pos.index.get_indexer(g.index)
                h_pred[loc] = hp
                capped[loc] = cp
            rho = ha.spearman(h_pred, h_obs)
            cand = {"lam": lam, "D0": D0, "D0_quantile": q, "rho_pred": float(rho),
                    "frac_capped": float(capped.mean())}
            if best is None or rho > best[0]["rho_pred"]:
                best = (cand, h_pred)
    return best


def budget_constancy(ev_pos, geoms, lam, n_perm=N_PERM, seed=0):
    """Is D(s, s+h_obs) more constant than with shuffled leads? One-sided
    permutation test on the coefficient of variation, per participant."""
    rng = np.random.default_rng(seed)
    out = {}
    for p, g in ev_pos.groupby("participant"):
        D_act = (g["D_W_act"] + lam * g["D_K_act"]).to_numpy()
        cv_act = float(np.std(D_act) / np.mean(D_act))
        h = g["lead_onset"].to_numpy()
        cvs = []
        for _ in range(n_perm):
            hp = rng.permutation(h)
            D = np.empty(len(g))
            for (pp, tid), gg in g.groupby(["participant", "trial_id"]):
                geom = geoms[(pp, tid)]
                loc = g.index.get_indexer(gg.index)
                s_c = gg["s_c"].to_numpy()
                dW, dK = geom.window(s_c, s_c + hp[loc])
                D[loc] = dW + lam * dK
            cvs.append(np.std(D) / np.mean(D))
        cvs = np.asarray(cvs)
        out[p] = {
            "cv_actual": cv_act,
            "cv_shuffled_median": float(np.median(cvs)),
            "cv_shuffled_ci95": [float(np.percentile(cvs, 2.5)),
                                 float(np.percentile(cvs, 97.5))],
            "p_perm": float((cvs <= cv_act).mean()),
            "D_median": float(np.median(D_act)),
            "n": int(len(g)),
        }
    return out


def unconstrained_analysis(ev_unc, ev_steer_pos, geoms, best):
    """Compare the unconstrained task families against steering under the
    density view. Free space has id_W ~ 0, so the steering-fitted budget
    model predicts (with NO refit) that the horizon runs to the goal in
    pointing, and 'opens up' before the tunnel exit in c2u."""
    ev_unc = ev_unc.copy()
    ev_unc["remaining"] = ev_unc["s_end"] - ev_unc["s_c"]
    ev_unc["h_pred"] = np.nan
    ev_unc["locked"] = False
    ev_unc["s_exit"] = np.nan
    ev_unc["W_tunnel"] = np.nan
    for (p, tid), g in ev_unc.groupby(["participant", "trial_id"]):
        geom = geoms[(p, tid)]
        hp, _ = geom.solve_h(g["s_c"].to_numpy(), best["lam"], best["D0"])
        ev_unc.loc[g.index, "h_pred"] = hp
        ev_unc.loc[g.index, "W_tunnel"] = float(geom.W[0])
        r = max(geom.target_radius or 0.01, 0.01)
        ev_unc.loc[g.index, "locked"] = (
            g["s_c"] + g["lead_onset"] >= geom.s_end - r)
        if geom.s_exit is not None:
            ev_unc.loc[g.index, "s_exit"] = geom.s_exit

    def family_stats(d, extra=lambda g: {}):
        st = {}
        for p, g in list(d.groupby("participant")) + [("pooled", d)]:
            gp = g[g["lead_onset"] > ha.MIN_LEAD]
            st[p] = {
                "n": int(len(g)),
                "frac_lead_positive": float((g["lead_onset"] > ha.MIN_LEAD).mean()),
                "frac_target_locked": float(g["locked"].mean()),
                "median_lead": float(gp["lead_onset"].median()),
                "median_remaining": float(gp["remaining"].median()),
                "median_lead_over_remaining": float(
                    (gp["lead_onset"] / gp["remaining"].clip(lower=1e-6)).median()),
                "rho_lead_remaining": ha.spearman(gp["lead_onset"], gp["remaining"]),
                "rho_fixed_th": ha.spearman(ha.FIXED_TH * gp["speed_onset"],
                                            gp["lead_onset"]),
                **extra(gp),
            }
        return st

    point = ev_unc[ev_unc["tunnel_type"] == "unconstrained_pointing"]
    c2u = ev_unc[ev_unc["tunnel_type"] == "constrained_to_unconstrained"]
    out = {
        "note": "budget h_pred uses the pooled steering fit (lam, D0) with no "
                "refit; free space contributes ~0 density so pointing h_pred "
                "= remaining distance to goal by construction",
        "pointing": family_stats(point),
        "c2u": family_stats(
            c2u, extra=lambda gp: {
                "rho_budget_pred": ha.spearman(gp["h_pred"], gp["lead_onset"])}),
    }
    # steering contrast on the same footing (locked threshold: half-width)
    st = {}
    for p, g in list(ev_steer_pos.groupby("participant")) + [("pooled", ev_steer_pos)]:
        rem = g["s_end"] - g["s_c"]
        locked = g["s_c"] + g["lead_onset"] >= g["s_end"] - np.maximum(
            0.5 * g["W_geo"], 0.01)
        st[p] = {
            "n": int(len(g)),
            "frac_target_locked": float(locked.mean()),
            "median_lead_over_remaining": float(
                (g["lead_onset"] / rem.clip(lower=1e-6)).median()),
            "rho_lead_remaining": ha.spearman(g["lead_onset"], rem),
        }
    out["steering_contrast"] = st
    return out, ev_unc


# ------------------------------------------------------------------- plots

def _binned_medians(x, y, n_bins=6):
    qs = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    qs = np.unique(qs)
    idx = np.clip(np.digitize(x, qs[1:-1]), 0, len(qs) - 2)
    bx, by, blo, bhi = [], [], [], []
    for b in range(len(qs) - 1):
        m = idx == b
        if m.sum() < 5:
            continue
        bx.append(np.median(x[m])); by.append(np.median(y[m]))
        blo.append(np.percentile(y[m], 25)); bhi.append(np.percentile(y[m], 75))
    return map(np.asarray, (bx, by, blo, bhi))


def plot_fixedwin(ev_ok, tag, outdir):
    cols = [(f"D_W_{tag}", "width density ahead  $\\int_s^{s+h_0}\\!ds'/W$"),
            (f"D_K_{tag}", "turning ahead  $\\int_s^{s+h_0}|\\kappa|\\,ds'$ (rad)"),
            ("D_sum", "combined  $D_W + D_K$")]
    d = ev_ok.copy()
    d["D_sum"] = d[f"D_W_{tag}"] + d[f"D_K_{tag}"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    for ax, (col, label) in zip(axes, cols):
        for p, g in d.groupby("participant"):
            bx, by, blo, bhi = _binned_medians(g[col].to_numpy(),
                                               g["lead_onset"].to_numpy())
            ax.plot(bx, by, "o-", color=COLORS[p], label=p, lw=2, ms=5)
            ax.fill_between(bx, blo, bhi, color=COLORS[p], alpha=0.12, lw=0)
        rho = ha.spearman(d[col], d["lead_onset"])
        ax.set_title(f"pooled Spearman $\\rho$ = {rho:.2f}", fontsize=10)
        ax.set_xlabel(label)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("gaze lead at fixation onset (task units)")
    axes[0].legend(title="participant", fontsize=9)
    fig.suptitle(f"Gaze lead vs integrated difficulty in a fixed look-ahead window "
                 f"(h0 = {tag}); bins = sextiles, band = IQR", fontsize=11)
    fig.tight_layout()
    fig.savefig(outdir / "lookahead_fixedwin_corr.png", dpi=150)
    plt.close(fig)


def plot_profiles(ev, geoms, outdir):
    """Three example tunnels: geometry, difficulty densities, and observed
    leads along the tunnel (events pooled over participants)."""
    picks = [("sharp_sinusoidal", "sharp sinusoidal, W=0.02"),
             ("corner", "corner, W=0.03"),
             ("wide_to_narrow", "wide-to-narrow 0.04->0.01")]
    sel = {}
    for tt, label in picks:
        cand = ev[(ev["tunnel_type"] == tt) & ev["s_c"].notna()]
        if tt == "sharp_sinusoidal":
            cand = cand[np.isclose(cand["W_geo"], 0.02, atol=5e-3)]
        if tt == "corner":
            cand = cand[np.isclose(cand["W_geo"], 0.03, atol=5e-3)]
        if tt == "wide_to_narrow":
            cand = cand[np.isclose(cand["width"].groupby(cand["trial_id"]).transform("max"), 0.04, atol=5e-3)]
        sel[tt] = cand
    fig, axes = plt.subplots(3, 3, figsize=(15, 9.5))
    for col, (tt, label) in enumerate(picks):
        cand = sel[tt]
        if len(cand) == 0:
            continue
        p0, tid0 = cand.iloc[0][["participant", "trial_id"]]
        geom = geoms[(p0, tid0)]

        ax = axes[0, col]
        ax.plot(geom.path[:, 0], geom.path[:, 1], "-", color="0.3", lw=1.2)
        n = np.column_stack([-np.gradient(geom.path[:, 1]), np.gradient(geom.path[:, 0])])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        for sgn in (1, -1):
            b = geom.path + sgn * 0.5 * geom.W[:, None] * n
            ax.plot(b[:, 0], b[:, 1], "-", color="0.75", lw=0.8)
        ax.set_aspect("equal")
        ax.set_title(label, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[1, col]
        invw = 1.0 / np.clip(geom.W, 1e-6, None)
        ax.plot(geom.s, invw, "-", color="tab:purple", lw=2, label="$id_W = 1/W$")
        ds = np.gradient(geom.s)
        kappa = np.gradient(np.unwrap(np.arctan2(*np.gradient(geom.path, axis=0).T[::-1]))) / (ds + 1e-12)
        ax.plot(geom.s, np.abs(kappa), "-", color="tab:red", lw=1.2, alpha=0.8,
                label="$id_K = |\\kappa|$")
        ax.set_yscale("log")
        ax.set_ylim(1e-1, 500)
        ax.set_ylabel("difficulty density (1/task-unit)" if col == 0 else "")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.25, lw=0.5)

        ax = axes[2, col]
        for p, g in cand.groupby("participant"):
            gpos = g[g["lead_onset"] > 0]
            ax.scatter(gpos["s_c"], gpos["lead_onset"], s=14, color=COLORS[p],
                       alpha=0.7, label=p)
        for sturn in geom.turn_s:
            for r in (1, 2):
                axes[r, col].axvline(sturn, color="0.8", lw=0.8, zorder=0)
        if tt == "wide_to_narrow":
            strans = geom.s[len(geom.s) // 2]
            for r in (1, 2):
                axes[r, col].axvline(strans, color="0.8", lw=0.8, zorder=0)
        ax.set_xlabel("cursor arc-length position s")
        ax.set_ylabel("gaze lead h (task units)" if col == 0 else "")
        if col == 0:
            ax.legend(fontsize=8, title="participant")
        ax.grid(alpha=0.25, lw=0.5)
        for r in (1, 2):
            axes[r, col].set_xlim(0, geom.s_end)
    fig.suptitle("The density view: tunnel geometry (top), difficulty densities along "
                 "the centerline (middle, log scale), observed gaze leads (bottom)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(outdir / "lookahead_density_profiles.png", dpi=150)
    plt.close(fig)


def plot_budget(const_stats, ev_pos, lam, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=False)
    for ax, (p, st) in zip(axes, sorted(const_stats.items())):
        g = ev_pos[ev_pos["participant"] == p]
        D = (g["D_W_act"] + lam * g["D_K_act"]).to_numpy()
        ax.hist(D, bins=30, color=COLORS[p], alpha=0.75)
        ax.axvline(st["D_median"], color="k", lw=1.2, ls="--",
                   label=f"median $D_0$ = {st['D_median']:.2f}")
        ax.set_title(
            f"{p}:  CV = {st['cv_actual']:.2f} vs shuffled "
            f"{st['cv_shuffled_median']:.2f}  (p = {st['p_perm']:.3f})", fontsize=10)
        ax.set_xlabel("difficulty in actual look-ahead window $D(s, s+h)$")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("fixation events")
    fig.suptitle(f"Budget constancy: realized window difficulty vs shuffled-lead "
                 f"control ($\\lambda$ = {lam:g})", fontsize=11)
    fig.tight_layout()
    fig.savefig(outdir / "lookahead_budget.png", dpi=150)
    plt.close(fig)


def plot_unconstrained(ev_unc, ev_steer_pos, stats, outdir):
    point = ev_unc[(ev_unc["tunnel_type"] == "unconstrained_pointing")
                   & (ev_unc["lead_onset"] > ha.MIN_LEAD)]
    c2u = ev_unc[(ev_unc["tunnel_type"] == "constrained_to_unconstrained")
                 & (ev_unc["lead_onset"] > ha.MIN_LEAD)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    ax = axes[0]
    rem_s = ev_steer_pos["s_end"] - ev_steer_pos["s_c"]
    ax.scatter(rem_s, ev_steer_pos["lead_onset"], s=6, color="0.75", alpha=0.4,
               label="steering (contrast)")
    for p, g in point.groupby("participant"):
        ax.scatter(g["remaining"], g["lead_onset"], s=14, color=COLORS[p],
                   alpha=0.75, label=p)
    lim = 0.5
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="lead = remaining (to goal)")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    rho_p = stats["pointing"]["pooled"]["rho_lead_remaining"]
    rho_s = stats["steering_contrast"]["pooled"]["rho_lead_remaining"]
    ax.set_title(f"pointing: $\\rho$(lead, remaining) = {rho_p:.2f}\n"
                 f"steering: {rho_s:.2f}", fontsize=10)
    ax.set_xlabel("remaining arc distance to goal")
    ax.set_ylabel("gaze lead at fixation onset (task units)")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    for p, g in c2u.groupby("participant"):
        ax.scatter(g["lead_onset"], g["h_pred"], s=14, color=COLORS[p],
                   alpha=0.7, label=p)
    lim = float(np.percentile(np.concatenate(
        [c2u["lead_onset"], c2u["h_pred"]]), 99)) if len(c2u) else 0.3
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    rho_b = stats["c2u"]["pooled"]["rho_budget_pred"]
    ax.set_title(f"c2u, steering-fitted budget model (no refit): "
                 f"$\\rho$ = {rho_b:.2f}", fontsize=10)
    ax.set_xlabel("observed lead at fixation onset")
    ax.set_ylabel("predicted lead")
    ax.legend(fontsize=8, title="participant")

    ax = axes[2]
    shades = plt.get_cmap("Purples")
    d = c2u.dropna(subset=["s_exit"]).copy()
    d["x"] = d["s_c"] - d["s_exit"]
    for i, w in enumerate(sorted(d["W_tunnel"].dropna().round(2).unique())):
        g = d[np.isclose(d["W_tunnel"], w, atol=5e-3)]
        if len(g) < 15:
            continue
        bx, by, blo, bhi = _binned_medians(g["x"].to_numpy(),
                                           g["lead_onset"].to_numpy(), n_bins=7)
        c = shades(0.45 + 0.2 * i)
        ax.plot(bx, by, "o-", color=c, lw=2, ms=4, label=f"tunnel W = {w:g}")
        ax.fill_between(bx, blo, bhi, color=c, alpha=0.15, lw=0)
    xs = np.linspace(-0.12, 0, 20)
    ax.plot(xs, -xs, "k:", lw=1.2, label="lead = dist to exit")
    ax.axvline(0.0, color="0.6", lw=1)
    ax.annotate("tunnel exit", (0.004, ax.get_ylim()[1] * 0.92), fontsize=8,
                color="0.4")
    ax.set_xlabel("cursor position relative to tunnel exit (s - s_exit)")
    ax.set_ylabel("gaze lead at fixation onset (task units)")
    ax.set_title("c2u: horizon opens up around the exit", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, lw=0.5)

    fig.suptitle("Unconstrained tasks vs steering: free space has ~zero difficulty "
                 "density, so the budget horizon runs to the goal", fontsize=11)
    fig.tight_layout()
    fig.savefig(outdir / "lookahead_unconstrained.png", dpi=150)
    plt.close(fig)


def plot_pred_vs_obs(ev_pos, h_pred, pl_params, best, outdir):
    h_obs = ev_pos["lead_onset"].to_numpy()
    pl_pred = ha.predict_lead(pl_params, ev_pos["width"])
    lim = float(np.percentile(np.concatenate([h_obs, h_pred, pl_pred]), 99))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True, sharey=True)
    for ax, yhat, name, rho in [
        (axes[0], h_pred,
         f"difficulty-budget model ($\\lambda$={best['lam']:g}, $D_0$={best['D0']:.2f})",
         best["rho_pred"]),
        (axes[1], pl_pred, "width power law lead = a·w^b (baseline)",
         ha.spearman(pl_pred, h_obs)),
    ]:
        for p, g in ev_pos.groupby("participant"):
            loc = ev_pos.index.get_indexer(g.index)
            ax.scatter(h_obs[loc], np.asarray(yhat)[loc], s=8, alpha=0.35,
                       color=COLORS[p], label=p)
        ax.plot([0, lim], [0, lim], "k--", lw=1)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("observed lead at fixation onset")
        ax.set_title(f"{name}\nSpearman $\\rho$ = {rho:.2f}", fontsize=10)
    axes[0].set_ylabel("predicted lead")
    axes[0].legend(title="participant", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "lookahead_pred_vs_obs.png", dpi=150)
    plt.close(fig)


# -------------------------------------------------------------------- main

def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    samples, events = gd.load_all()
    ev = ha.steering_events(events)
    ev, geoms = attach_geometry(samples, ev)
    ev = ev[ev["s_c"].notna()].copy()

    # -------- geometry validation against the CSV's own local width
    val = ev.dropna(subset=["W_geo", "width"])
    width_match = float((np.abs(val["W_geo"] - val["width"]) < 2e-3).mean())
    proj_p95 = float(ev["proj_dist"].quantile(0.95))
    print(f"[validate] local width match (geom vs CSV): {width_match:.1%}  "
          f"| cursor->centerline distance p95: {proj_p95:.4f}")

    ev_pos = ev[ev["lead_onset"] > ha.MIN_LEAD].copy()
    lead_q = ev_pos["lead_onset"].quantile([0.25, 0.5, 0.75]).to_dict()
    print(f"[info] positive-lead events: {len(ev_pos)}  lead quartiles: "
          + ", ".join(f"p{int(q*100)}={v:.3f}" for q, v in lead_q.items()))

    summary = {
        "n_events_analyzed": int(len(ev)),
        "n_positive_lead": int(len(ev_pos)),
        "lead_quartiles": {f"p{int(q*100)}": float(v) for q, v in lead_q.items()},
        "validation": {"width_match_frac": width_match, "proj_dist_p95": proj_p95},
        "naive_actual_window": {},
        "fixed_window": {},
        "two_segment_stratified": {},
        "corner_distance": {},
        "budget_constancy": {},
        "budget_model": {},
    }

    # -------- naive (circular) correlation, reported for completeness
    for p, g in list(ev_pos.groupby("participant")) + [("pooled", ev_pos)]:
        summary["naive_actual_window"][p] = {
            "rho_h_DW_actual": ha.spearman(g["lead_onset"], g["D_W_act"]),
            "rho_h_DK_actual": ha.spearman(g["lead_onset"], g["D_K_act"]),
        }

    # -------- test 1: fixed windows
    ev_pos["s_frac"] = ev_pos["s_c"] / ev_pos["s_end"]
    for h0 in H0_GRID:
        tag = f"{h0:g}"
        ok = ev_pos[ev_pos[f"fits_{tag}"]].copy()
        ok["D_sum"] = ok[f"D_W_{tag}"] + ok[f"D_K_{tag}"]
        cl = ok["participant"].astype(str) + "_" + ok["trial_id"].astype(str)
        entry = {
            "n": int(len(ok)),
            "pooled": {
                "D_W": cluster_boot_rho(ok[f"D_W_{tag}"], ok["lead_onset"], cl),
                "D_K": cluster_boot_rho(ok[f"D_K_{tag}"], ok["lead_onset"], cl),
                "D_sum": cluster_boot_rho(ok["D_sum"], ok["lead_onset"], cl),
                "local_invW_baseline": cluster_boot_rho(
                    1.0 / ok["W_geo"], ok["lead_onset"], cl),
            },
            # robustness: rank-partial out the position along the tunnel
            "pooled_partial_given_sfrac": {
                "D_W": partial_spearman(ok[f"D_W_{tag}"], ok["lead_onset"],
                                        ok["s_frac"]),
                "D_K": partial_spearman(ok[f"D_K_{tag}"], ok["lead_onset"],
                                        ok["s_frac"]),
                "D_sum": partial_spearman(ok["D_sum"], ok["lead_onset"],
                                          ok["s_frac"]),
            },
            "per_participant": {},
        }
        for p, g in ok.groupby("participant"):
            entry["per_participant"][p] = {
                "rho_D_W": ha.spearman(g[f"D_W_{tag}"], g["lead_onset"]),
                "rho_D_K": ha.spearman(g[f"D_K_{tag}"], g["lead_onset"]),
                "rho_D_sum": ha.spearman(g["D_sum"], g["lead_onset"]),
                "rho_local_invW": ha.spearman(1.0 / g["W_geo"], g["lead_onset"]),
                "n": int(len(g)),
            }
        summary["fixed_window"][tag] = entry
        if h0 == H0_MAIN:
            plot_fixedwin(ok, tag, RESULTS_DIR)

    # -------- test 2: two-segment tunnels, width-ahead vs width-here
    tag = f"{H0_MAIN:g}"
    ts = ev_pos[ev_pos["tunnel_type"].isin(TWO_SEG) & ev_pos[f"fits_{tag}"]].copy()
    ts["W_level"] = ts["W_geo"].round(2)
    summary["two_segment_stratified"] = {
        "h0": H0_MAIN,
        "n": int(len(ts)),
        "pooled_within_participant_x_width": within_stratum_rho(
            ts, ["participant", "W_level"], f"D_W_{tag}", "lead_onset"),
        # position along the tunnel is a strong confound here (start-transient
        # leads are long exactly where D_W ahead is high in narrow-first
        # tunnels), so the partial version is the meaningful one
        "pooled_within_partial_given_sfrac": within_stratum_rho(
            ts, ["participant", "W_level"], f"D_W_{tag}", "lead_onset",
            ctrl_col="s_frac"),
    }

    # -------- test 3: corners - lead vs distance to next turn
    co = ev_pos[(ev_pos["tunnel_type"] == "corner")
                & np.isfinite(ev_pos["d_next_turn"].astype(float))].copy()
    for p, g in list(co.groupby("participant")) + [("pooled", co)]:
        near = g[g["d_next_turn"] <= H0_MAIN]["lead_onset"]
        far = g[g["d_next_turn"] > H0_MAIN]["lead_onset"]
        summary["corner_distance"][p] = {
            "rho_h_dnext": ha.spearman(g["lead_onset"], g["d_next_turn"]),
            "rho_partial_given_sfrac": partial_spearman(
                g["lead_onset"], g["d_next_turn"], g["s_frac"]),
            "n": int(len(g)),
            "lead_median_corner_within_h0": float(near.median()) if len(near) else None,
            "lead_median_corner_beyond_h0": float(far.median()) if len(far) else None,
        }

    # -------- test 5: budget model (fit first; its lambda feeds test 4)
    best, h_pred = budget_fit(ev_pos, geoms)
    pl_params, _ = ha.fit_horizon_power(ev_pos)
    lam0_grid = [l for l in LAM_GRID if l == 0.0]
    best_w_only, _ = budget_fit(ev_pos, geoms, lam_grid=lam0_grid)
    best_nocap, _ = budget_fit(ev_pos, geoms, cap_at_end=False)
    summary["budget_model"]["pooled"] = {
        **best,
        "rho_power_law_same_events": float(
            ha.spearman(ha.predict_lead(pl_params, ev_pos["width"]),
                        ev_pos["lead_onset"])),
        "power_law_params": {k: pl_params[k] for k in ("a", "b")},
        "width_only_budget": best_w_only,
        "no_end_cap_ablation": best_nocap,
        "n": int(len(ev_pos)),
    }
    for p, g in ev_pos.groupby("participant"):
        b_p, hp_p = budget_fit(g, geoms)
        pl_p, _ = ha.fit_horizon_power(g)
        summary["budget_model"][p] = {
            **b_p,
            "rho_power_law_same_events": float(
                ha.spearman(ha.predict_lead(pl_p, g["width"]), g["lead_onset"])),
            "n": int(len(g)),
        }
    plot_pred_vs_obs(ev_pos, h_pred, pl_params, best, RESULTS_DIR)

    # -------- test 4: budget constancy with the fitted lambda
    const_stats = budget_constancy(ev_pos, geoms, best["lam"])
    summary["budget_constancy"] = {"lam": best["lam"], **const_stats}
    plot_budget(const_stats, ev_pos, best["lam"], RESULTS_DIR)

    plot_profiles(ev, geoms, RESULTS_DIR)

    # -------- unconstrained tasks (pointing, c2u) as the id_W ~ 0 limit
    ev_unc = events[
        events["tunnel_type"].isin(UNCON)
        & (events["speed_onset"] > ha.MIN_SPEED)
        & events["lead_onset"].notna()
    ].copy()
    ev_unc, geoms_unc = attach_geometry(samples, ev_unc)
    ev_unc = ev_unc[ev_unc["s_c"].notna()]
    unc_stats, ev_unc = unconstrained_analysis(ev_unc, ev_pos, geoms_unc, best)
    summary["unconstrained_comparison"] = unc_stats
    plot_unconstrained(ev_unc, ev_pos, unc_stats, RESULTS_DIR)

    with open(RESULTS_DIR / "lookahead_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print(json.dumps({k: summary[k] for k in
                      ("fixed_window", "two_segment_stratified", "corner_distance",
                       "budget_constancy", "budget_model",
                       "unconstrained_comparison")}, indent=2, default=float))
    print(f"\nwrote results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()

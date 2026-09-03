"""Generative saccade-fixate-catch-up cycle model — demonstration.

Finalized design:
  anchor   next gaze anchor from the width-only difficulty budget:
           int_{s_c}^{s_a} (W_ref/W)^GAMMA / W_ref ds = D0   (cap: path end,
           floor: v * T_MIN), pooled params from the fitted pipeline.
  trigger  saccade fires at cursor ARRIVAL at the anchor + latency TAU
           (the cursor keeps moving during the latency -> overrun, lead < 0).
  speed    the finalized local speed model: prior GAMSpeedModel (exact
           graveyard code) trained on clearance = W/2, kappa_local and
           kappa_ahead (max |kappa| over the next 50 mm) from all six
           participants' samples; RAW prediction (no floor/ceil clamp).

The cycle is simulated deterministically on the real task geometries (5
steering types x widths, p01's geometry set = the shared task battery) and the
emergent statistics are compared with the human ones measured this week:
hop-distance and catch-up-duration width exponents, medians by width, cycle
rate, and the within-width duration~curvature coupling.

Outputs (into human-gaze-lead-10p/):
  cycle_model_sawtooth.png   model sawtooth vs human rounds (sharp, 5 widths)
  cycle_model_laws.png       emergent hop/duration laws vs human medians
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import maximum_filter1d

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from human_gaze_lead import dense_grid, PROC, gd, ld
from local_speed_law import kappa_profile, runway_profile, TYPES
sys.path.insert(0, str(SCRIPT_DIR.parents[1] / "hcs_package" / "src"))
from hcs_package.speed_model import GAMSpeedModel

ARTIFACT = (SCRIPT_DIR.parents[1] / "hcs_package" / "src" / "hcs_package"
            / "models" / "gam_traversal_10p.pkl")

BASE = SCRIPT_DIR / "human-gaze-lead-10p"
GAMMA, W_REF, T_MIN = 0.66, 0.026, 0.10   # fitted budget params (A/B/C pipeline)
D0 = 1.0   # 10p-cohort refit (single-knob sweep vs pooled hop medians;
           # the A/B/C-pooled value 1.66 overshoots this cohort ~1.4x)
TAU = 0.19                                           # replan latency (s)
H_AHEAD = 0.05                                       # kappa lookahead (m)
STEP = 0.001
V_MIN_SIM = 0.01                                     # numerical speed floor


def train_speed_model():
    """Load the simulator's shipped traversal artifact (single source of
    truth; train_traversal_gam.py regenerates it from the 10p samples)."""
    return GAMSpeedModel.load(str(ARTIFACT))


def predict_speed(m, W, kappa, k_ahead, runway):
    return np.maximum(m.predict_speed_raw(W / 2, kappa, k_ahead, runway),
                      V_MIN_SIM)


def simulate(geom, speed_model):
    """One deterministic round; returns (t, lead) series and cycle events."""
    s_d, _ = dense_grid(geom, STEP)
    W_d = np.interp(s_d, geom.s, geom.W)
    K_d = np.interp(s_d, geom.s, kappa_profile(geom))
    win = int(H_AHEAD / STEP) + 1
    Ka_d = maximum_filter1d(K_d, size=win, origin=-(win // 2), mode="nearest")
    Dn_d = runway_profile(s_d, K_d)
    PHI_d = np.interp(s_d, geom.s, geom.PHI)
    v_d = predict_speed(speed_model, W_d, K_d, Ka_d, Dn_d)
    # cumulative time along the path and cumulative budget
    T_d = np.concatenate([[0.0], np.cumsum(STEP / v_d[:-1])])
    dens = ((W_REF / W_d) ** GAMMA) / W_REF
    B_d = np.concatenate([[0.0], np.cumsum(dens[:-1] * STEP)])

    def anchor_from(s):
        s_a = np.interp(np.interp(s, s_d, B_d) + D0, B_d, s_d)
        s_a = max(s_a, s + np.interp(s, s_d, v_d) * T_MIN)     # T_min floor
        return min(s_a, geom.s_end)

    events, t_series, lead_series = [], [], []
    s_c, t_now = 0.0, 0.0
    s_a = anchor_from(s_c)
    prev_anchor, t_sacc = 0.0, 0.0
    while s_c < geom.s_end - 1e-6 and t_now < 60:
        # cursor arrival time at the anchor, then latency TAU
        t_arr = t_now + (np.interp(s_a, s_d, T_d) - np.interp(s_c, s_d, T_d))
        t_next = t_arr + TAU
        # sample the sawtooth over this cycle
        ts = np.arange(t_now, t_next, 0.01)
        s_of_t = np.interp(np.interp(s_c, s_d, T_d) + (ts - t_now), T_d, s_d)
        t_series.append(ts); lead_series.append(s_a - s_of_t)
        s_end_cycle = float(s_of_t[-1]) if len(ts) else s_c
        events.append(dict(
            t=t_now, dur=t_next - t_now, hop=s_a - prev_anchor,
            s0=s_c, s1=s_end_cycle,
            dphi=float(np.interp(s_end_cycle, s_d, PHI_d) - np.interp(s_c, s_d, PHI_d))))
        # saccade: instant jump to the next anchor
        s_c, t_now = s_end_cycle, t_next
        prev_anchor = s_a
        if s_a >= geom.s_end - 1e-6:
            break
        s_a = anchor_from(s_c)
    return np.concatenate(t_series), np.concatenate(lead_series), pd.DataFrame(events)


def main():
    speed_model = train_speed_model()
    print("speed model trained (graveyard GAM, clearance+kappa+kappa_ahead, raw)")

    # task geometries: p01's battery (shared across participants)
    L = "p01"
    s = gd.load_samples(L)
    cl = pd.read_csv(PROC / f"{L}_fixation_events_clean.csv")
    cl = cl[cl["keep"] == True]  # noqa: E712
    ev = gd.fixation_events(s)
    ev = ev[~ev["tunnel_type"].astype(str).str.contains("pointing")]
    _, geoms = ld.attach_geometry(s, ev)
    trials = cl.groupby("trial_id").first().reset_index()
    trials = trials[trials["tunnel_type"].astype(str).isin(TYPES)]

    all_ev, traces = [], {}
    for _, tr in trials.iterrows():
        g = geoms.get((L, tr["trial_id"]))
        if g is None:
            continue
        t, lead, cev = simulate(g, speed_model)
        cev["type_label"] = TYPES[str(tr["tunnel_type"])]
        cev["width_mm"] = round(tr["width"] * 1000, 1)
        all_ev.append(cev)
        traces[(TYPES[str(tr["tunnel_type"])], round(tr["width"] * 1000, 1))] = (t, lead)
    m = pd.concat(all_ev, ignore_index=True)
    # interior cycles only (first hop starts from s=0, last is goal-capped)
    mi = m[(m["hop"] > 0) & (m["s1"] < m.groupby(["type_label", "width_mm"])["s1"]
                             .transform("max") - 1e-6)]

    hum = pd.read_csv(BASE / "data" / "saccade_events.csv")
    hum_cu = hum[(hum["catchup_s"] >= 0.05) & (hum["catchup_s"] <= 3.0)]

    print("\n=== emergent statistics: model vs human ===")
    b_m, _, _, _, _ = stats.linregress(np.log(mi["width_mm"]), np.log(mi["hop"] * 1000))
    b_d, _, _, _, _ = stats.linregress(np.log(mi["width_mm"]), np.log(mi["dur"]))
    print(f"hop-vs-width exponent:      model {b_m:+.2f}   human +0.55 (pooled +0.51)")
    print(f"duration-vs-width exponent: model {b_d:+.2f}   human -0.36 (pooled -0.32)")
    print("\nmedian hop (mm) by width:        " + "  ".join(
        f"W{w:g}: {mi.loc[mi.width_mm == w, 'hop'].median()*1000:.0f}/"
        f"{hum.loc[hum.width_mm == w, 'amp_m'].median()*1000:.0f}"
        for w in sorted(mi["width_mm"].unique())) + "   (model/human)")
    print("median catch-up (s) by width:    " + "  ".join(
        f"W{w:g}: {mi.loc[mi.width_mm == w, 'dur'].median():.2f}/"
        f"{hum_cu.loc[hum_cu.width_mm == w, 'catchup_s'].median():.2f}"
        for w in sorted(mi["width_mm"].unique())) + "   (model/human)")
    rate_m = len(m) / m.groupby(["type_label", "width_mm"])["t"].max().add(
        m.groupby(["type_label", "width_mm"])["dur"].last()).sum()
    print(f"\ncycle rate: model {rate_m:.2f} /s   human 1.86 /s")
    rhos = [stats.spearmanr(g_["dphi"], g_["dur"])[0]
            for w, g_ in mi.groupby("width_mm") if len(g_) > 10]
    print(f"within-width rho(dphi, dur): model {np.mean(rhos):.2f}   human 0.46-0.60")

    # --- figure 1: sawtooth vs human rounds (sharp sinusoid, all widths)
    hlead = pd.read_csv(BASE / "data" / f"{L}_steering_lead.csv")
    widths = sorted(mi["width_mm"].unique())
    fig, axes = plt.subplots(len(widths), 1, figsize=(10, 2.2 * len(widths)),
                             sharex=True, squeeze=False)
    for ax, w in zip(axes[:, 0], widths):
        h = hlead[(hlead["type_label"] == "sharp") & (hlead["width_mm"].round(1) == w)
                  & (hlead["round"] == 1)]
        ax.plot(h["t"], h["lead"], ".", ms=1.5, color="0.6",
                label=f"human {L} round 1" if w == widths[0] else None)
        t, lead = traces[("sharp", w)]
        ax.plot(t, lead, "-", lw=1.4, color="tab:red",
                label="cycle model" if w == widths[0] else None)
        ax.axhline(0, color="0.4", lw=0.7)
        ax.set_ylabel(f"W={w:g} mm\nlead (m)")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8, loc="upper right")
    axes[-1, 0].set_xlabel("time since round start (s)")
    fig.suptitle("Saccade-fixate-catch-up cycle: model (budget anchor + arrival"
                 f"+{TAU:g}s trigger + GAM speed) vs human — sharp sinusoid", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(BASE / "cycle_model_sawtooth.png", dpi=150)

    # --- figure 2: emergent laws vs human medians
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, (mcol, hdf, hcol, lab) in zip(axes, [
            ("hop", hum, "amp_m", "hop distance (mm)"),
            ("dur", hum_cu, "catchup_s", "catch-up duration (s)")]):
        wm = sorted(mi["width_mm"].unique())
        mv = [mi.loc[mi.width_mm == w, mcol].median() for w in wm]
        hv = [hdf.loc[hdf.width_mm == w, hcol].median() for w in wm]
        scale = 1000 if mcol == "hop" else 1
        ax.plot(wm, np.array(mv) * scale, "o-", color="tab:red", label="cycle model")
        ax.plot(wm, np.array(hv) * scale, "s--", color="0.3", label="human median")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks(wm); ax.set_xticklabels([f"{w:g}" for w in wm])
        ax.set_xlabel("tolerance W (mm)"); ax.set_ylabel(lab)
        ax.grid(alpha=0.3, which="both"); ax.legend()
    axes[0].set_title("emergent hop law"); axes[1].set_title("emergent catch-up law")
    fig.suptitle("Emergent cycle laws: deterministic model vs pooled human medians",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(BASE / "cycle_model_laws.png", dpi=150)
    print(f"\nfigures -> {BASE}/cycle_model_sawtooth.png, cycle_model_laws.png")


if __name__ == "__main__":
    main()

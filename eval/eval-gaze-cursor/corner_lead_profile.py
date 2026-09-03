"""Does the gaze lead (horizon amplitude) shrink around corners?

Corner tunnels (trials 6-10, W = 0.01..0.05, numCorners=2 -> four 90-degree
turns at s ~ 0.153 / 0.253 / 0.407 / 0.507, vertical legs 0.1 long).

For every fixation onset in a corner trial the cursor is projected on the
centerline (s_c) and located relative to the turns:
    d_next  = s_turn_next - s_c      (>0, arc distance to the upcoming turn)
    d_prev  = s_c - s_turn_prev      (>0, arc distance past the last turn)
    delta   = signed arc distance to the NEAREST turn (<0 approaching, >0 past)

Analyses
  1. lead profile vs delta (binned medians, bootstrap CI), raw and within-trial
     normalised (lead / trial median lead: removes the width + participant
     level so only the corner modulation remains).
  2. corner vs straight at equal width: median lead in straight tunnels vs the
     corner trials split into far-from-turn (|delta| > 0.04) and near-turn.
  3. clamp test: does the gaze anchor s_g = s_c + lead stop AT the turn
     (lead ~= d_next: anchor-at-landmark) or merely shrink? Fraction of
     anchors that cross the upcoming turn as a function of d_next, and the
     lead vs d_next scatter against the identity line.
  4. within-trial permutation test of the near-vs-far median difference.
  5. same profile for the per-fixation MAX lead (sawtooth peak) as a second
     amplitude measure, and for the continuous per-sample lead p90 envelope.

Run: python3 corner_lead_profile.py   -> results/corner_lead_profile.{png,json}
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gaze_data as gd
import horizon_analysis as ha
import lookahead_difficulty as ld

RESULTS_DIR = Path(__file__).resolve().parent / "results"
MIN_SPEED = ha.MIN_SPEED
NEAR = 0.04            # |delta| threshold for near-turn (= H0_MAIN)
EDGES = np.array([-0.12, -0.08, -0.04, -0.02, 0.0, 0.02, 0.04, 0.08, 0.12])
N_BOOT = 500
N_PERM = 1000
RNG = np.random.default_rng(0)
COLORS = ld.COLORS


def boot_median_ci(x, n_boot=N_BOOT):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return (np.nan, np.nan, np.nan)
    meds = [np.median(RNG.choice(x, len(x))) for _ in range(n_boot)]
    return (float(np.median(x)), float(np.percentile(meds, 2.5)),
            float(np.percentile(meds, 97.5)))


def load_events():
    frames, samples_all = [], []
    for p in gd.PARTICIPANTS:
        s = gd.load_samples(p)
        ev = gd.fixation_events(s)
        # per-fixation max lead (sawtooth peak) + per-fixation sample table
        d = s.dropna(subset=["fixation_id", "lead"]).copy()
        d["fixation_id"] = pd.to_numeric(d["fixation_id"], errors="coerce")
        mx = d.groupby(["block_id", "fixation_id"])["lead"].max().rename("lead_max")
        ev = ev.merge(mx.reset_index(), on=["block_id", "fixation_id"], how="left")
        frames.append(ev)
        samples_all.append(s)
    ev = pd.concat(frames, ignore_index=True)
    samples = pd.concat(samples_all, ignore_index=True)
    return samples, ev


def attach_corner_geometry(samples, ev):
    ev = ev[ev["tunnel_type"] == "corner"].copy()
    conds = (samples[samples["tunnel_type"] == "corner"]
             .dropna(subset=["condition_json"])
             .groupby(["participant", "trial_id"])["condition_json"].first())
    geoms = {k: ld.build_geom("corner", json.loads(v)) for k, v in conds.items()}
    for col in ("s_c", "s_end", "d_next", "d_prev", "delta", "turn_idx",
                "anchor_crosses_next"):
        ev[col] = np.nan
    for key, g in ev.groupby(["participant", "trial_id"]):
        geom = geoms[key]
        s_c, _ = geom.project(g["cursor_x"].to_numpy(), g["cursor_y"].to_numpy())
        turns = geom.turn_s
        diff = turns[None, :] - s_c[:, None]           # + = turn ahead
        ahead = np.where(diff > 0, diff, np.inf)
        behind = np.where(diff <= 0, -diff, np.inf)
        d_next = ahead.min(axis=1)
        d_prev = behind.min(axis=1)
        nearest = np.argmin(np.abs(diff), axis=1)
        delta = s_c - turns[nearest]
        s_g = s_c + g["lead_onset"].to_numpy()
        ev.loc[g.index, "s_c"] = s_c
        ev.loc[g.index, "s_end"] = geom.s_end
        ev.loc[g.index, "d_next"] = d_next
        ev.loc[g.index, "d_prev"] = d_prev
        ev.loc[g.index, "delta"] = delta
        ev.loc[g.index, "turn_idx"] = nearest
        ev.loc[g.index, "anchor_crosses_next"] = (
            np.isfinite(d_next) & (s_g > s_c + d_next + 0.005))
    return ev, geoms


def clean(ev):
    """Moving cursor, forward-looking fixation, not blink-corrupted."""
    m = ((ev["speed_onset"] > MIN_SPEED) & (ev["lead_onset"] > ha.MIN_LEAD)
         & ~ev["blink_corrupted"].astype(bool))
    return ev[m].copy()


def add_within_trial_norm(ev, col="lead_onset"):
    med = ev.groupby(["participant", "trial_id"])[col].transform("median")
    ev[col + "_norm"] = ev[col] / med
    return ev


def profile(ev, col, norm=False):
    """Binned medians of `col` (or its within-trial normalised version) vs delta."""
    val = col + ("_norm" if norm else "")
    out = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        g = ev[(ev["delta"] >= lo) & (ev["delta"] < hi)]
        m, l, u = boot_median_ci(g[val])
        out.append({"lo": float(lo), "hi": float(hi), "n": int(len(g)),
                    "median": m, "ci_lo": l, "ci_hi": u})
    return out


def near_far(ev, col="lead_onset"):
    near = ev[np.abs(ev["delta"]) <= NEAR][col]
    far = ev[np.abs(ev["delta"]) > NEAR][col]
    return near, far


def perm_test(ev, col="lead_onset", n_perm=N_PERM):
    """Within-(participant, trial) shuffle of lead across events; statistic =
    median(far) - median(near). One-sided p for shrink (observed > null)."""
    near, far = near_far(ev, col)
    obs = float(far.median() - near.median())
    is_near = (np.abs(ev["delta"]) <= NEAR).to_numpy()
    vals = ev[col].to_numpy().copy()
    groups = ev.groupby(["participant", "trial_id"]).indices
    null = np.empty(n_perm)
    for k in range(n_perm):
        v = vals.copy()
        for idx in groups.values():
            v[idx] = RNG.permutation(v[idx])
        null[k] = np.median(v[~is_near]) - np.median(v[is_near])
    p = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    return {"median_far": float(far.median()), "median_near": float(near.median()),
            "diff_far_minus_near": obs, "ratio_near_over_far": float(near.median() / far.median()),
            "n_near": int(len(near)), "n_far": int(len(far)), "p_perm": p}


def approach_vs_exit(ev, col="lead_onset"):
    """Split the near-turn events into approaching (delta<0) vs just past (>0)."""
    out = {}
    for name, m in (("approach_-0.04..0", (ev["delta"] >= -NEAR) & (ev["delta"] < 0)),
                    ("exit_0..0.04", (ev["delta"] >= 0) & (ev["delta"] <= NEAR)),
                    ("far_|delta|>0.04", np.abs(ev["delta"]) > NEAR)):
        m_, l, u = boot_median_ci(ev.loc[m, col])
        out[name] = {"n": int(m.sum()), "median": m_, "ci": [l, u]}
    return out


def straight_baseline(ev_all):
    st = ev_all[(ev_all["tunnel_type"] == "straight")
                & (ev_all["speed_onset"] > MIN_SPEED)
                & (ev_all["lead_onset"] > ha.MIN_LEAD)
                & ~ev_all["blink_corrupted"].astype(bool)].copy()
    st["W"] = st["width"].round(2)
    return st


def clamp_test(ev):
    """Fraction of anchors crossing the upcoming turn vs d_next; and how close
    lead sits to d_next when the turn is within reach."""
    bins = [0.0, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.12, 0.2]
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        g = ev[(ev["d_next"] >= lo) & (ev["d_next"] < hi)]
        if len(g) == 0:
            continue
        rows.append({"d_next_lo": lo, "d_next_hi": hi, "n": int(len(g)),
                     "frac_anchor_crosses_turn": float(g["anchor_crosses_next"].mean()),
                     "median_lead": float(g["lead_onset"].median()),
                     "median_lead_over_dnext": float((g["lead_onset"] / g["d_next"]).median())})
    return rows


def sample_envelope(samples, geoms):
    """Continuous view: per-sample lead p50/p90 vs delta over corner trials."""
    s = samples[(samples["tunnel_type"] == "corner") & samples["lead"].notna()
                & (samples["speed"] > MIN_SPEED)].copy()
    s["delta"] = np.nan
    for key, g in s.groupby(["participant", "trial_id"]):
        geom = geoms[key]
        # project on a coarse subsample for speed, then interpolate in time
        sub = g.iloc[::4]
        s_c, _ = geom.project(sub["cursor_x"].to_numpy(), sub["cursor_y"].to_numpy())
        s_c_full = np.interp(g["t"].to_numpy(), sub["t"].to_numpy(), s_c)
        diff = geom.turn_s[None, :] - s_c_full[:, None]
        nearest = np.argmin(np.abs(diff), axis=1)
        s.loc[g.index, "delta"] = s_c_full - geom.turn_s[nearest]
    prof = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        g = s[(s["delta"] >= lo) & (s["delta"] < hi)]["lead"]
        prof.append({"lo": float(lo), "hi": float(hi), "n": int(len(g)),
                     "p50": float(g.median()) if len(g) else np.nan,
                     "p90": float(g.quantile(0.9)) if len(g) else np.nan,
                     "frac_positive": float((g > 0).mean()) if len(g) else np.nan})
    return prof


def plot(ev, prof_by_p, prof_norm_by_p, st, clamp_rows, env, out_png):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    centers = 0.5 * (EDGES[:-1] + EDGES[1:])

    ax = axes[0, 0]
    for p, prof in prof_by_p.items():
        c = COLORS.get(p, "k")
        m = np.array([r["median"] for r in prof])
        lo = np.array([r["ci_lo"] for r in prof])
        hi = np.array([r["ci_hi"] for r in prof])
        ax.errorbar(centers, m, yerr=[m - lo, hi - m], fmt="o-", color=c,
                    lw=2 if p == "pooled" else 1, ms=5 if p == "pooled" else 3,
                    capsize=2, label=p, alpha=0.9 if p == "pooled" else 0.6)
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("signed arc distance to nearest turn  (<0 approaching, >0 past)")
    ax.set_ylabel("fixation-onset lead (task units)")
    ax.set_title("Raw onset lead around the turn")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for p, prof in prof_norm_by_p.items():
        c = COLORS.get(p, "k")
        m = np.array([r["median"] for r in prof])
        lo = np.array([r["ci_lo"] for r in prof])
        hi = np.array([r["ci_hi"] for r in prof])
        ax.errorbar(centers, m, yerr=[m - lo, hi - m], fmt="o-", color=c,
                    lw=2 if p == "pooled" else 1, ms=5 if p == "pooled" else 3,
                    capsize=2, label=p, alpha=0.9 if p == "pooled" else 0.6)
    ax.axhline(1, color="0.5", lw=0.8)
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("signed arc distance to nearest turn")
    ax.set_ylabel("lead / trial median lead")
    ax.set_title("Within-trial normalised (width & person removed)")

    ax = axes[0, 2]
    widths = sorted(ev["width"].round(2).unique())
    x = np.arange(len(widths))
    near_m, far_m, st_m = [], [], []
    for w in widths:
        g = ev[ev["width"].round(2) == w]
        near_m.append(g[np.abs(g["delta"]) <= NEAR]["lead_onset"].median())
        far_m.append(g[np.abs(g["delta"]) > NEAR]["lead_onset"].median())
        st_m.append(st[st["W"] == w]["lead_onset"].median())
    ax.bar(x - 0.27, st_m, 0.27, label="straight tunnel", color="0.6")
    ax.bar(x, far_m, 0.27, label=f"corner, |delta|>{NEAR}", color="tab:blue")
    ax.bar(x + 0.27, near_m, 0.27, label=f"corner, |delta|<={NEAR}", color="tab:red")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{w:g}" for w in widths])
    ax.set_xlabel("tunnel width")
    ax.set_ylabel("median onset lead")
    ax.set_title("Corner vs straight at equal width")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for p, g in ev.groupby("participant"):
        ax.scatter(g["d_next"], g["lead_onset"], s=10, alpha=0.5, color=COLORS[p], label=p)
    lim = 0.2
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="lead = d_next (anchor at turn)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, 0.12)
    ax.set_xlabel("arc distance to upcoming turn")
    ax.set_ylabel("onset lead")
    ax.set_title("Clamp test: lead vs distance to next turn")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    cx = [0.5 * (r["d_next_lo"] + r["d_next_hi"]) for r in clamp_rows]
    ax.plot(cx, [r["frac_anchor_crosses_turn"] for r in clamp_rows], "o-", color="tab:purple")
    ax.set_ylim(0, 1)
    ax.set_xlabel("arc distance to upcoming turn (bin centre)")
    ax.set_ylabel("fraction of anchors landing PAST the turn")
    ax.set_title("Does gaze look through the corner?")
    ax2 = ax.twinx()
    ax2.plot(cx, [r["median_lead"] for r in clamp_rows], "s--", color="tab:gray")
    ax2.set_ylabel("median lead", color="tab:gray")

    ax = axes[1, 2]
    ax.plot(centers, [r["p50"] for r in env], "o-", color="tab:blue", label="per-sample lead p50")
    ax.plot(centers, [r["p90"] for r in env], "s-", color="tab:red", label="per-sample lead p90 (envelope)")
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("signed arc distance to nearest turn")
    ax.set_ylabel("instantaneous lead")
    ax.set_title("Continuous sawtooth envelope (all samples, pooled)")
    ax.legend(fontsize=8)

    fig.suptitle("Gaze lead around 90-degree corners (A/B/C, corner trials 6-10)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    samples, ev_all = load_events()
    ev, geoms = attach_corner_geometry(samples, ev_all)
    n_raw = len(ev)
    ev = clean(ev)
    ev = add_within_trial_norm(ev, "lead_onset")
    ev = add_within_trial_norm(ev, "lead_max")
    print(f"corner fixation events: {n_raw} raw -> {len(ev)} moving/forward/no-blink")

    summary = {"n_events_raw": int(n_raw), "n_events_used": int(len(ev)),
               "near_threshold": NEAR, "bins": EDGES.tolist(),
               "turn_s": {f"{k[0]}_{k[1]}": g.turn_s.tolist() for k, g in list(geoms.items())[:1]},
               "profile_raw": {}, "profile_norm": {}, "profile_leadmax_norm": {},
               "near_far_perm": {}, "approach_vs_exit": {}, "corner_vs_straight_by_width": {},
               "clamp": {}, "sample_envelope": None,
               "rho_lead_vs_abs_delta": {}, "rho_lead_vs_dnext": {}}

    groups = list(ev.groupby("participant")) + [("pooled", ev)]
    prof_by_p, prof_norm_by_p = {}, {}
    for p, g in groups:
        prof_by_p[p] = summary["profile_raw"][p] = profile(g, "lead_onset")
        prof_norm_by_p[p] = summary["profile_norm"][p] = profile(g, "lead_onset", norm=True)
        summary["profile_leadmax_norm"][p] = profile(g, "lead_max", norm=True)
        summary["near_far_perm"][p] = {
            "onset": perm_test(g, "lead_onset"),
            "onset_norm": perm_test(g, "lead_onset_norm"),
            "leadmax_norm": perm_test(g, "lead_max_norm"),
        }
        summary["approach_vs_exit"][p] = approach_vs_exit(g, "lead_onset_norm")
        summary["clamp"][p] = clamp_test(g)
        summary["rho_lead_vs_abs_delta"][p] = ha.spearman(g["lead_onset_norm"], np.abs(g["delta"]))
        summary["rho_lead_vs_dnext"][p] = ha.spearman(g["lead_onset_norm"], g["d_next"])

    st = straight_baseline(ev_all)
    for w in sorted(ev["width"].round(2).unique()):
        g = ev[ev["width"].round(2) == w]
        near, far = near_far(g)
        sw = st[st["W"] == w]["lead_onset"]
        summary["corner_vs_straight_by_width"][f"{w:g}"] = {
            "straight_median": float(sw.median()) if len(sw) else None, "n_straight": int(len(sw)),
            "corner_far_median": float(far.median()), "n_far": int(len(far)),
            "corner_near_median": float(near.median()), "n_near": int(len(near)),
        }

    env = summary["sample_envelope"] = sample_envelope(samples, geoms)

    plot(ev, prof_by_p, prof_norm_by_p, st, summary["clamp"]["pooled"], env,
         RESULTS_DIR / "corner_lead_profile.png")
    with open(RESULTS_DIR / "corner_lead_profile.json", "w") as f:
        json.dump(summary, f, indent=1, default=float)

    # ---------------- console report
    print("\n== near (|delta|<=0.04) vs far, onset lead, within-trial permutation ==")
    for p in ("A", "B", "C", "pooled"):
        r = summary["near_far_perm"][p]["onset"]
        rn = summary["near_far_perm"][p]["onset_norm"]
        rm = summary["near_far_perm"][p]["leadmax_norm"]
        print(f"  {p:6s} raw  near {r['median_near']:.4f} vs far {r['median_far']:.4f} "
              f"(ratio {r['ratio_near_over_far']:.2f}, n {r['n_near']}/{r['n_far']}, p={r['p_perm']:.3f})"
              f" | norm ratio {rn['ratio_near_over_far']:.2f} p={rn['p_perm']:.3f}"
              f" | lead_max norm ratio {rm['ratio_near_over_far']:.2f} p={rm['p_perm']:.3f}")
    print("\n== normalised lead profile vs signed distance (pooled) ==")
    for r in summary["profile_norm"]["pooled"]:
        print(f"  [{r['lo']:+.2f},{r['hi']:+.2f})  n={r['n']:3d}  median {r['median']:.2f} "
              f"[{r['ci_lo']:.2f},{r['ci_hi']:.2f}]")
    print("\n== approach vs exit (normalised) ==")
    for p in ("A", "B", "C", "pooled"):
        d = summary["approach_vs_exit"][p]
        print(f"  {p:6s} " + "  ".join(f"{k}: {v['median']:.2f} (n={v['n']})" for k, v in d.items()))
    print("\n== corner vs straight at equal width (median onset lead) ==")
    for w, d in summary["corner_vs_straight_by_width"].items():
        print(f"  W={w}: straight {d['straight_median']} (n={d['n_straight']}) | corner far "
              f"{d['corner_far_median']:.4f} (n={d['n_far']}) | corner near {d['corner_near_median']:.4f} (n={d['n_near']})")
    print("\n== clamp test (pooled): anchors crossing the upcoming turn ==")
    for r in summary["clamp"]["pooled"]:
        print(f"  d_next [{r['d_next_lo']:.2f},{r['d_next_hi']:.2f}) n={r['n']:3d} "
              f"cross {r['frac_anchor_crosses_turn']:.2f}  lead {r['median_lead']:.4f}  lead/d_next {r['median_lead_over_dnext']:.2f}")
    print("\n== per-sample envelope (pooled) ==")
    for r in env:
        print(f"  [{r['lo']:+.2f},{r['hi']:+.2f}) n={r['n']:5d} p50 {r['p50']:.4f} p90 {r['p90']:.4f} pos {r['frac_positive']:.2f}")
    print("\nrho(norm lead, |delta|):", {k: round(v, 3) for k, v in summary["rho_lead_vs_abs_delta"].items()})
    print("rho(norm lead, d_next): ", {k: round(v, 3) for k, v in summary["rho_lead_vs_dnext"].items()})
    print(f"\nwrote {RESULTS_DIR / 'corner_lead_profile.png'} and .json")


if __name__ == "__main__":
    main()

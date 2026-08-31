"""Visualize the planner's generated reference paths for the eval-main steering tasks —
no model runs (max_steps=0: the simulator builds the reference path and returns before
the first solve, so no trajectory is generated).

For every steering tid of the participant: centerline, walls, and the Phase-0 reference
path the planner would track, plus per-task numbers (cut depth at high-curvature points,
minimum wall clearance; negative clearance = the reference path leaves the tunnel).

Usage: python viz_ref_paths.py --pid P170114 \
          --config ../eval-anchor-drive/results/P170114_anchor_persona_S9.json \
          --out-dir refpath-viz-S9-B
Outputs: <out-dir>/refpaths_<pid>.pdf (one page per task), refpaths_overview_<pid>.png,
         printed per-task table.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "eval-anchor-drive"))
import probe_anchor as pa           # noqa: E402  (sets sys.path / data dir)
import fit_speed_model as fsm       # noqa: E402
import run_eval as em               # noqa: E402
from hcs_package.reference_path import ReferencePath  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", default="P170114")
    ap.add_argument("--config", required=True, help="persona/config JSON (reference_path params are what matter)")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    out = HERE / (a.out_dir or f"refpath-viz-{a.pid}")
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.load(open(a.config)); cfg.pop("_description", None)
    cfg["add_noise"] = False; cfg["replan_latency_cv"] = 0.0
    print(f"{a.pid}: reference_path params = {json.dumps(cfg.get('reference_path', {}), indent=None)}")

    rounds, t2c, t2b = fsm.load_participant(a.pid)
    tids = [t for t in sorted(t2c) if t2b.get(t) == "steering" and t in rounds]
    rows, figs = [], []
    for tid in tids:
        cond = t2c[tid]
        tc, _cl = em.build_steering_task_config(cond)
        tc = dict(tc); tc["max_steps"] = 0          # build the path, run no model steps
        _traj, _spd, _dt, _diag, ref = pa._sim_with_diag(cfg, tc)
        cl_sp = ReferencePath(fsm._waypoints_m(tc), s=0.0, k=3)
        hw = float(cond["tunnelWidth"]) / 2.0

        s_cl = np.linspace(0.0, cl_sp.total_length, 1200)
        C = np.array([cl_sp(float(s)) for s in s_cl])
        T = cl_sp.tangents(s_cl); Nrm = np.column_stack([T[:, 1], -T[:, 0]])
        wl, wr = C + hw * Nrm, C - hw * Nrm
        R = np.array([ref(float(s)) for s in np.linspace(0.0, ref.total_length, 1200)])

        # signed offset of the reference path from the centerline + curvature there
        off, kap = [], []
        for p in R[::4]:
            th = cl_sp.find_closest_theta(np.asarray(p)); c = cl_sp(th); t = cl_sp.tangent(th)
            off.append((np.asarray(p) - c) @ np.array([t[1], -t[0]])); kap.append(abs(cl_sp.curvature(th)))
        off, kap = np.array(off), np.array(kap)
        spread = np.ptp(kap) > 1e-9
        hi = kap > np.percentile(kap, 85) if spread else np.zeros(len(kap), bool)
        lo = kap < np.percentile(kap, 15) if spread else np.ones(len(kap), bool)
        cut_mu = float(np.mean(np.abs(off[hi]))) * 1000 if hi.any() else 0.0
        cut_p90 = float(np.percentile(np.abs(off[hi]), 90)) * 1000 if hi.any() else 0.0
        ripple = float(np.percentile(np.abs(off[lo]), 95)) * 1000 if lo.any() else 0.0
        clear_min = float(hw - np.max(np.abs(off))) * 1000
        ttype = cond.get("tunnelType") or "sinusoidal"
        rows.append((tid, ttype, cond["tunnelWidth"] * 1000, cut_mu, cut_p90, clear_min))

        fig, ax = plt.subplots(figsize=(10, 5.5))
        # True corridor band: the tunnel is all points within hw of the centerline
        # (Minkowski sum with a disc) — draw as a thick round-join line so 90-degree
        # corners render correctly (naive +/-normal offsets self-intersect there).
        band, = ax.plot(C[:, 0], C[:, 1], "-", color="0.88", solid_capstyle="round",
                        solid_joinstyle="round", zorder=0, label="tunnel")
        ax.plot(C[:, 0], C[:, 1], "--", color="0.65", lw=1.0, label="centerline")
        ax.plot(R[:, 0], R[:, 1], "-", color="tab:red", lw=1.6, label="reference path")
        ax.plot(*R[0], "o", color="tab:green", ms=7, label="start")
        ax.plot(*R[-1], "s", color="tab:blue", ms=7, label="end")
        ax.set_aspect("equal")
        fig.canvas.draw()
        px_per_unit = float(np.hypot(*(ax.transData.transform((1.0, 0.0)) - ax.transData.transform((0.0, 0.0)))))
        band.set_linewidth(2.0 * hw * px_per_unit * 72.0 / fig.dpi)
        ax.legend(loc="best", fontsize=8)
        ax.set_title(f"tid {tid}  {ttype}  W={cond['tunnelWidth']*1000:.0f}mm | "
                     f"cut@high-kappa {cut_mu:.1f}/{cut_p90:.1f} mm | straight-leg ripple p95 {ripple:.1f} mm | "
                     f"min wall clearance {clear_min:.1f} mm", fontsize=10)
        figs.append(fig)
        print(f"  tid {tid:>3}  {ttype:<18} W={cond['tunnelWidth']*1000:3.0f}mm  "
              f"cut {cut_mu:4.1f}/{cut_p90:4.1f} mm  ripple {ripple:4.1f} mm  min clearance {clear_min:6.1f} mm"
              + ("   [REF PATH OUTSIDE WALLS]" if clear_min < 0 else ""), flush=True)

    pdf_path = out / f"refpaths_{a.pid}.pdf"
    with PdfPages(pdf_path) as pdf:
        for f in figs:
            pdf.savefig(f, bbox_inches="tight")
    n = len(figs); nc = 4; nr = int(np.ceil(n / nc))
    ov, axes = plt.subplots(nr, nc, figsize=(4.2 * nc, 2.6 * nr))
    for axo, f, (tid, ttype, w, cmu, c90, clr) in zip(np.ravel(axes), figs, rows):
        for ln in f.axes[0].get_lines():
            if ln.get_label() == "tunnel":
                continue
            axo.plot(ln.get_xdata(), ln.get_ydata(), ln.get_linestyle(), color=ln.get_color(),
                     lw=max(0.6, min(1.2, ln.get_linewidth() * 0.6)), marker=ln.get_marker(), ms=3)
        axo.set_aspect("equal"); axo.set_xticks([]); axo.set_yticks([])
        axo.set_title(f"t{tid} {ttype[:6]} {w:.0f}mm cut {cmu:.1f}", fontsize=7)
    for axo in np.ravel(axes)[n:]:
        axo.axis("off")
    ov.tight_layout(); ov.savefig(out / f"refpaths_overview_{a.pid}.png", dpi=160)
    for f in figs: plt.close(f)
    plt.close(ov)
    print(f"\nsaved {pdf_path} and refpaths_overview_{a.pid}.png ({n} steering tasks, 0 model steps run)")


if __name__ == "__main__":
    main()

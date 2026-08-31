"""Figure-3-style trajectory overlays (paper draft): participants A and B across
Sin W=50, Cor W=20, Cor W=40, Gen W=50 — human rounds (blue) vs model runs (orange)
over the tunnel corridor.

Usage: python fig3_overlays.py --a results/P105835_anchor_persona_S9e.json \
         --b results/P170114_anchor_persona_S9e.json --runs 5 --out results/fig3_overlays.png
"""
import argparse, json, sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import probe_anchor as pa, fit_speed_model as fsm
from hcs_package.reference_path import ReferencePath

CONDS = [("sinusoidal", 0.05, "Sin W=50"), ("corner", 0.02, "Cor W=20"),
         ("corner", 0.04, "Cor W=40"), ("gentle_sinusoidal", 0.05, "Gen W=50")]


def sim_one(args):
    cfg, tc, seed = args
    c = json.loads(json.dumps(cfg)); c["random_seed"] = seed
    traj, spd, dt, diag, ref = pa._sim_with_diag(c, tc)
    return np.asarray(traj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--runs", type=int, default=5); ap.add_argument("--out", default=str(HERE / "results" / "fig3_overlays.png"))
    args = ap.parse_args()
    rows = [("A", "P105835", args.a), ("B", "P170114", args.b)]
    fig, axes = plt.subplots(2, 4, figsize=(17, 4.6), gridspec_kw={"wspace": 0.04, "hspace": 0.25})
    jobs, meta = [], []
    data = {}
    for r, (letter, pid, cpath) in enumerate(rows):
        cfg = json.load(open(cpath)); cfg.pop("_description", None)
        rounds, t2c, t2b = fsm.load_participant(pid); tasks = fsm.build_tunnel_tasks(t2c, t2b)
        data[letter] = (rounds, t2c, t2b, tasks)
        for cidx, (ty, w, _lbl) in enumerate(CONDS):
            tid = next(t for t in tasks if t2b[t] == "steering" and abs(t2c[t]["tunnelWidth"] - w) < 1e-6
                       and (t2c[t].get("tunnelType") or "sinusoidal") == ty)
            for i in range(args.runs):
                jobs.append((cfg, tasks[tid][0], 1000 + i)); meta.append((r, cidx, tid))
    with Pool(8) as pool:
        results = pool.map(sim_one, jobs)

    panel = "abcdefgh"
    for (r, cidx, tid), traj in zip(meta, results):
        ax = axes[r][cidx]
        letter, pid, _ = rows[r]; rounds, t2c, t2b, tasks = data[letter]
        if not ax.lines:   # first hit on this panel: corridor + human rounds
            tc = tasks[tid][0]; hw = t2c[tid]["tunnelWidth"] / 2.0
            cl = ReferencePath(fsm._waypoints_m(tc), s=0.0, k=3)
            C = np.array([cl(float(s)) for s in np.linspace(0, cl.total_length, 900)])
            band, = ax.plot(C[:, 0], C[:, 1], "-", color="0.88", solid_capstyle="round",
                            solid_joinstyle="round", zorder=0, label="_tunnel")
            ax.set_aspect("equal")
            fig.canvas.draw()
            ppu = float(np.hypot(*(ax.transData.transform((1, 0)) - ax.transData.transform((0, 0)))))
            band.set_linewidth(2.0 * hw * ppu * 72.0 / fig.dpi)
            for h in rounds[tid]:
                htr = np.asarray(h["trajectory"])
                ax.plot(htr[:, 0], htr[:, 1], "-", color="tab:blue", lw=0.9, alpha=0.75, zorder=2)
        ax.plot(traj[:, 0], traj[:, 1], "-", color="tab:orange", lw=0.9, alpha=0.75, zorder=3)
        ty, w, lbl = CONDS[cidx]
        ax.set_title(f"({panel[r*4+cidx]}) Participant {letter}, {lbl}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.margins(0.03)
    fig.tight_layout()
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    print(f"saved {args.out}  (blue = human rounds, orange = {args.runs} model runs, noise on)")


if __name__ == "__main__":
    main()

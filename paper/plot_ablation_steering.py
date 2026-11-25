import sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plot_laws as pl
from r1_ablation_table import mt_by_tid, eval_dir

R = "results-cluster-10p/runs/"
RUNS = [("Ours (paper)", R + "mpcc-full-perpid-s42-20260909-1257-22c349c", "#D55E00", "^", "-"),
        ("Fixed lookahead + pace law", R + "mpcc-no_lookahead-perpid-s42-20260909-1259-22c349c", "#009E73", "D", "--"),
        ("Adaptive lookahead + fixed pace", R + "mpcc-no_pace-perpid-s42-20260909-1259-22c349c", "#CC79A7", "v", "-."),
        # both gaze mechanisms removed (the no_gaze run; it also replans every step)
        ("Fixed lookahead + fixed pace", R + "mpcc-no_gaze-perpid-s42-20260909-1300-22c349c", "#56B4E9", "P", (0, (5, 1.5, 1, 1.5, 1, 1.5))),
        ("Baseline (paper)", R + "baseline-full-perpid-s42-20260908-1800-c96338d", "#8C8C8C", "s", ":")]
geoms = pl.steering_geometry()
h, ids = mt_by_tid(eval_dir(RUNS[0][1]) / "Steering" / "steering_results.csv", "MT_s", True)
_, lam = pl.budget_id_data({"Human": h}, geoms)
tids = sorted(h, key=int)
def idk(t):
    w, L, phi = geoms[t]; return (L + lam * phi) / w
series = [("Human", h, "#0072B2", "o", "-")]
for lab, d, c, m, ls in RUNS:
    series.append((lab, mt_by_tid(eval_dir(d) / "Steering" / "steering_results.csv", "MT_s", False)[0], c, m, ls))

# usage: plot_ablation_steering.py OUT.png [--curvature-only]
CURV_ONLY = "--curvature-only" in sys.argv
PANELS = [(lambda t: ids[t], "ID = L / W", "Steering law"),
          (idk, rf"Curvature-aware ID$_k$ = (L + {lam:.2f}·Φ) / W", "Curvature-aware (Fig. 3a form)")]
if CURV_ONLY:
    PANELS = [(idk, rf"Curvature-aware ID$_k$ = (L + {lam:.2f}·Φ) / W", "")]
fig, axes = plt.subplots(1, len(PANELS), figsize=((3.9, 3.3) if CURV_ONLY else (7.2, 3.3)),
                         constrained_layout=True, squeeze=False)
axes = axes[0]
for ax, (xf, xlab, title) in zip(axes, PANELS):
    for lab, mt, c, m, ls in series:
        x = np.array([xf(t) for t in tids]); y = np.array([mt[t] for t in tids])
        a, b, r2 = pl.fit_line(x, y)
        ax.scatter(x, y, s=16, marker=m, color=c, edgecolor="white", linewidth=0.5, zorder=3, alpha=0.9)
        xx = np.linspace(0, x.max() * 1.03, 50)
        ax.plot(xx, a + b * xx, linestyle=ls, color=c, lw=1.4, zorder=2,
                label=f"{lab}: b={b:.3f}, R²={r2:.2f}")
    ax.set_xlabel(xlab)
    if title:
        ax.set_title(title, loc="left", fontsize=8.5)
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    ax.grid(True, color="#e6e6e6", lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=6.3, loc="upper left", handlelength=2.4)
axes[0].set_ylabel("Movement time (s), mean per condition")
fig.suptitle("Study 1 (8 participants, 25 tunnel conditions)" if CURV_ONLY else
             "Study 1 per-participant fits (8 participants, 25 tunnel conditions): lookahead vs. pace law ablations",
             fontsize=8.5, x=0.01, ha="left")
out = Path(sys.argv[1]); fig.savefig(out, dpi=200); fig.savefig(out.with_suffix(".pdf"))
print("saved", out, "lambda", lam)

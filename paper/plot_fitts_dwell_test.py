"""Fitts'-law test plots for the terminal-dwell end-condition change.

Compares the pooled-8 model's Fitts regression across termination variants —
first target entry (eval-main-pooled8-local) vs 0.12 s and 0.15 s of
consecutive in-target samples (eval-main-pooled8-dwell012/-dwell015) — same
fitted persona, no refit. Draws each variant in the paper style as one panel,
and prints per-condition human-model MT gaps plus within-distance-tier slopes
(D fixed, only R varying), the acceptance metric for radius sensitivity.

Usage:  python paper/plot_fitts_dwell_test.py
Writes: paper/figures/fitts_law_dwell_test.{pdf,png}
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_laws import (REPO, OUT_DIR, load_condition_means, draw_law)  # noqa: E402

VARIANTS = [  # (results dir, panel title, short tag)
    ("eval-main-pooled8-local", "first-entry termination", "before"),
    ("eval-main-pooled8-dwell012", "0.12 s in-target dwell", "d=0.12"),
    ("eval-main-pooled8-dwell015", "0.15 s in-target dwell", "d=0.15"),
]
TIERS = [("D=0.17", range(42, 47)), ("D=0.31", range(47, 52)),
         ("D=0.46", range(52, 57))]


def per_condition(csv_path):
    per = defaultdict(lambda: defaultdict(list))
    ids = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r.get("timed_out") == "True":
                continue
            per[int(r["tid"])][r["source"]].append(float(r["MT_kin_s"]))
            ids[int(r["tid"])] = float(r["ID"])
    return ({t: {s: float(np.mean(v)) for s, v in d.items()}
             for t, d in per.items()}, ids)


def main():
    dirs = [REPO / "results-cluster-10p" / d / "Fitts" for d, _, _ in VARIANTS]
    have = [d.exists() for d in dirs]

    n = sum(have)
    fig, axes = plt.subplots(1, n, figsize=(3.45 * n, 2.6), sharey=True)
    axes = np.atleast_1d(axes)
    k = 0
    for (d, title, _), ok in zip(VARIANTS, have):
        if not ok:
            print(f"skipping {d[0] if isinstance(d, tuple) else title}: no data")
            continue
        path = REPO / "results-cluster-10p" / d / "Fitts" / "fitts_results.csv"
        data = load_condition_means(path, "MT_kin_s")
        fits = draw_law(axes[k], data, "Index of difficulty (bits)", "ID")
        axes[k].set_title(title, fontsize=8)
        for s, (a, b, r2) in fits.items():
            print(f"{title} — {s}: MT = {a:.3f} + {b:.4f} ID, R2 = {r2:.3f}")
        if k > 0:
            axes[k].set_ylabel("")
        k += 1
    fig.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fitts_law_dwell_test.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    print(f"saved {OUT_DIR / 'fitts_law_dwell_test.pdf'}")

    # per-condition model MTs across variants + human reference
    conds = {}
    for (d, _, tag), ok in zip(VARIANTS, have):
        if ok:
            conds[tag], ids = per_condition(
                REPO / "results-cluster-10p" / d / "Fitts" / "fitts_results.csv")
    tags = list(conds)
    hdr = " ".join(f"{('m_' + t):>9}" for t in tags)
    print(f"\n{'tid':>4} {'MT_h':>6} {hdr}")
    base = conds[tags[0]]
    for t in sorted(base):
        cells = " ".join(f"{conds[tag].get(t, {}).get('Simulator', float('nan')):>9.3f}"
                         for tag in tags)
        print(f"{t:>4} {base[t]['Human']:>6.3f} {cells}")

    print("\nwithin-tier slope (s/bit), D fixed, only R varies:")
    print(f"{'tier':>8} {'Human':>7} " + " ".join(f"{t:>8}" for t in tags))
    for tier, tids in TIERS:
        x = np.array([ids[t] for t in tids])
        hy = np.array([base[t]["Human"] for t in tids])
        row = [f"{np.polyfit(x, hy, 1)[0]:>+7.3f}"]
        for tag in tags:
            y = np.array([conds[tag][t]["Simulator"] for t in tids])
            row.append(f"{np.polyfit(x, y, 1)[0]:>+8.3f}")
        print(f"{tier:>8} " + " ".join(row))


if __name__ == "__main__":
    main()

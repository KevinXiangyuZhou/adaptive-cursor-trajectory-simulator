"""Build a single-page HTML gallery of the per-trial eval-main plots in a
results directory, for quickly reviewing trajectories side by side.

Usage:
    python make_gallery.py results-gaze-fitted [more results dirs ...]

Writes {results_dir}/gallery.html with, per bucket -> participant -> trial:
trajectory plot, speed-vs-time and speed-vs-progress plots (relative links,
so the page works from the results dir itself), plus headline numbers from
each trial's results_summary.json (human/model CT, lateral RMSE, speed corr).
"""

import html
import json
import re
import sys
from pathlib import Path

BUCKETS = ["Steering", "Fitts", "ID4SCS/wide_to_narrow", "ID4SCS/narrow_to_wide",
           "ConstrainedToUnconstrained"]


def trial_sort_key(p):
    m = re.search(r"trial_(\d+)", p.name)
    return int(m.group(1)) if m else 0


def summarize(summary_path):
    try:
        d = json.load(open(summary_path))
    except Exception:
        return ""
    cond = d.get("condition", {}) or {}
    bits = [cond.get("description") or cond.get("tunnelType", "")]
    h = d.get("human", {}).get("completion_times") or []
    m = d.get("model", {}).get("completion_times") or []
    if h:
        bits.append(f"human CT {sum(h)/len(h):.2f}s (n={len(h)})")
    if m:
        bits.append(f"model CT {sum(m)/len(m):.2f}s (n={len(m)})")
    met = d.get("metrics", {}) or {}
    if met.get("lateral_rmse") is not None:
        bits.append(f"latRMSE {met['lateral_rmse']*1000:.1f}mm")
    if met.get("speed_corr") is not None:
        bits.append(f"spdCorr {met['speed_corr']:.2f}")
    return " · ".join(str(b) for b in bits if b)


def build(results_dir):
    root = Path(results_dir)
    out = [
        "<meta charset='utf-8'><title>%s</title>" % html.escape(root.name),
        "<style>body{font-family:sans-serif;margin:16px;background:#fafafa}"
        "h2{margin:28px 0 4px}h3{margin:18px 0 4px;color:#444}"
        ".trial{background:#fff;border:1px solid #ddd;border-radius:6px;"
        "padding:8px;margin:8px 0}.imgs{display:flex;gap:8px;flex-wrap:wrap}"
        ".imgs img{max-height:260px;border:1px solid #eee}"
        ".meta{color:#555;font-size:13px;margin:2px 0 6px}"
        "a{color:#06c}#toc a{margin-right:14px}</style>",
        f"<h1>{html.escape(root.name)} — per-trial plots</h1><div id='toc'></div>",
    ]
    toc = []
    for bucket in BUCKETS:
        bdir = root / bucket
        if not bdir.is_dir():
            continue
        anchor = bucket.replace("/", "_")
        parts = sorted(bdir.glob("participant_*"))
        if not parts:
            continue
        toc.append(f"<a href='#{anchor}'>{bucket}</a>")
        out.append(f"<h2 id='{anchor}'>{bucket}</h2>")
        for pdir in parts:
            out.append(f"<h3>{pdir.name}</h3>")
            for tdir in sorted([d for d in pdir.iterdir() if d.is_dir()],
                               key=trial_sort_key):
                imgs = sorted(tdir.glob("*.png"))
                if not imgs:
                    continue
                info = summarize(tdir / "results_summary.json")
                rel = lambda p: p.relative_to(root).as_posix()
                out.append(
                    f"<div class='trial'><b>{tdir.name}</b>"
                    f"<div class='meta'>{html.escape(info)}</div><div class='imgs'>"
                    + "".join(
                        f"<a href='{rel(i)}' target='_blank'><img src='{rel(i)}' "
                        f"loading='lazy' title='{i.name}'></a>"
                        for i in imgs)
                    + "</div></div>")
    out[3] = "<div id='toc'>" + " ".join(toc) + "</div>"
    dest = root / "gallery.html"
    dest.write_text("\n".join(out))
    print(f"Saved: {dest}")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        build(d)

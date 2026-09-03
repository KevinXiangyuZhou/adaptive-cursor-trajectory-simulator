"""Regenerate the processed gaze-cursor fixation dataset for all participants.

Output: human_data/processed_gaze_events/{letter}_fixation_events_clean.csv (one row per
steering fixation, raw and corrected leads, flags) + cleaning_summary.json.

The tables include the near-coincident MERGE pass (gaze_cleaning.merge_events):
consecutive fixations that re-settle on the same spot (micro-saccade within 10 mm /
0.1 s) are one functional fixation, folded into their first fragment (keep=False +
merged_into on the absorbed rows, union dwell on the survivor). THE MERGED, keep=True
EVENTS ARE THE CANONICAL DATASET FOR MODEL FITTING (budget/lead, turning-time,
intermittency constants); pass --no-merge only for A/B comparisons against the
pre-merge behaviour.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import gaze_data as gd, lookahead_difficulty as ld, gaze_cleaning as gc

OUT = HERE.parents[1] / "human_data" / "processed_gaze_events"
OUT.mkdir(parents=True, exist_ok=True)
COLS = ["participant", "trial_id", "block_id", "fixation_id", "tunnel_type", "width", "t_onset",
        "duration_s", "s_c", "cursor_x", "cursor_y", "lead_onset", "lead_corr", "gaze_task_x", "gaze_task_y",
        "gaze_x_corr", "gaze_y_corr", "gaze_path_dist", "blink_corrupted", "off_path",
        "regressive", "saccade_followed", "n_merged", "merged_into", "drift_corrected",
        "drift_x", "drift_y", "keep"]

ap = argparse.ArgumentParser()
ap.add_argument("--letters", nargs="*", default=["A", "B", "C"] + [f"p{i:02d}" for i in range(1, 11)])
ap.add_argument("--no-merge", action="store_true",
                help="skip the tracker-split merge pass (comparison only; merged is canonical)")
ap.add_argument("--no-drift", action="store_true",
                help="skip the per-block drift correction (global pointing bias only)")
ap.add_argument("--out-dir", default=str(OUT))
a = ap.parse_args()
out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

summary = {"_note": ("Merged events (keep=True) are the canonical fixation dataset for model "
                     "fitting; absorbed tracker-split fragments carry merged_into."
                     if not a.no_merge else "UNMERGED comparison run — do not fit models on this.")}
for L in a.letters:
    s = gd.load_samples(L)
    bias = gc.estimate_bias(s)
    drift = gc.estimate_block_drift(s) if (not a.no_drift and L in gc.DRIFT_PARTICIPANTS) else None
    ev = gd.fixation_events(s)
    ev = ev[~ev["tunnel_type"].astype(str).str.contains("pointing")]
    ev, geoms = ld.attach_geometry(s, ev)
    ev = ev.dropna(subset=["s_c"])
    out = gc.clean_events(ev, geoms, bias, drift=drift)
    n_kept_pre = int(out["keep"].sum())
    if not a.no_merge:
        out = gc.merge_events(out)
    out[[c for c in COLS if c in out.columns]].to_csv(out_dir / f"{L}_fixation_events_clean.csv", index=False)
    k = out[out["keep"]]
    n_absorbed = int(out["merged_into"].notna().sum()) if "merged_into" in out.columns else 0
    used = (out[out["drift_corrected"]]["block_id"].unique() if drift else [])
    drift_mag = ([float(np.hypot(*drift[b][:2])) * 1000 for b in used] if drift else [])
    summary[L] = {"bias_mm": [round(float(b) * 1000, 2) for b in bias], "n_fixations": int(len(out)),
                  "kept": int(out["keep"].sum()), "kept_before_merge": n_kept_pre,
                  "absorbed_fragments": n_absorbed,
                  "drift_blocks": int(len(used)),
                  "drift_median_mm": round(float(np.median(drift_mag)), 1) if drift_mag else None,
                  "dropped": {"blink": int(out["blink_corrupted"].sum()), "off_path": int(out["off_path"].sum()),
                               "regressive": int(out["regressive"].sum())},
                  "median_lead_raw_mm": round(float(np.nanmedian(out["lead_onset"])) * 1000, 1),
                  "median_lead_clean_mm": round(float(np.nanmedian(k["lead_corr"])) * 1000, 1) if len(k) else None,
                  "median_dwell_s": round(float(np.nanmedian(k["duration_s"])), 3) if len(k) else None}
    print(f"{L}: bias ({summary[L]['bias_mm'][0]:+.1f},{summary[L]['bias_mm'][1]:+.1f})mm | "
          f"kept {summary[L]['kept']}/{len(out)} (blink {summary[L]['dropped']['blink']}, "
          f"off-path {summary[L]['dropped']['off_path']}, regressive {summary[L]['dropped']['regressive']}"
          f", merged-away {n_absorbed}) | drift-corrected blocks {summary[L]['drift_blocks']} "
          f"(median |off| {summary[L]['drift_median_mm']}mm) | median lead raw {summary[L]['median_lead_raw_mm']:+.1f} -> "
          f"clean {summary[L]['median_lead_clean_mm']:+.1f} mm | dwell {summary[L]['median_dwell_s']}s", flush=True)
json.dump(summary, open(out_dir / "cleaning_summary.json", "w"), indent=2)
print("saved", out_dir / "cleaning_summary.json"); print("DONE")

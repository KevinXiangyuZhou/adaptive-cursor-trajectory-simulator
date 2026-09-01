"""Regenerate the processed gaze-cursor fixation dataset for all participants.

Output: human_data/processed_gaze_events/{letter}_fixation_events_clean.csv (one row per
steering fixation, raw and corrected leads, flags) + cleaning_summary.json.
"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import gaze_data as gd, lookahead_difficulty as ld, gaze_cleaning as gc

OUT = HERE.parents[1] / "human_data" / "processed_gaze_events"
OUT.mkdir(parents=True, exist_ok=True)
LETTERS = ["A", "B", "C"] + [f"p{i:02d}" for i in range(1, 11)]
COLS = ["participant", "trial_id", "block_id", "fixation_id", "tunnel_type", "width", "t_onset",
        "duration_s", "s_c", "lead_onset", "lead_corr", "gaze_task_x", "gaze_task_y",
        "gaze_x_corr", "gaze_y_corr", "gaze_path_dist", "blink_corrupted", "off_path",
        "regressive", "keep"]
summary = {}
for L in LETTERS:
    s = gd.load_samples(L)
    bias = gc.estimate_bias(s)
    ev = gd.fixation_events(s)
    ev = ev[~ev["tunnel_type"].astype(str).str.contains("pointing")]
    ev, geoms = ld.attach_geometry(s, ev)
    ev = ev.dropna(subset=["s_c"])
    out = gc.clean_events(ev, geoms, bias)
    out[[c for c in COLS if c in out.columns]].to_csv(OUT / f"{L}_fixation_events_clean.csv", index=False)
    k = out[out["keep"]]
    summary[L] = {"bias_mm": [round(float(b) * 1000, 2) for b in bias], "n_fixations": int(len(out)),
                  "kept": int(out["keep"].sum()),
                  "dropped": {"blink": int(out["blink_corrupted"].sum()), "off_path": int(out["off_path"].sum()),
                               "regressive": int(out["regressive"].sum())},
                  "median_lead_raw_mm": round(float(np.nanmedian(out["lead_onset"])) * 1000, 1),
                  "median_lead_clean_mm": round(float(np.nanmedian(k["lead_corr"])) * 1000, 1) if len(k) else None}
    print(f"{L}: bias ({summary[L]['bias_mm'][0]:+.1f},{summary[L]['bias_mm'][1]:+.1f})mm | "
          f"kept {summary[L]['kept']}/{len(out)} (blink {summary[L]['dropped']['blink']}, "
          f"off-path {summary[L]['dropped']['off_path']}, regressive {summary[L]['dropped']['regressive']}) | "
          f"median lead raw {summary[L]['median_lead_raw_mm']:+.1f} -> clean {summary[L]['median_lead_clean_mm']:+.1f} mm", flush=True)
json.dump(summary, open(OUT / "cleaning_summary.json", "w"), indent=2)
print("saved", OUT / "cleaning_summary.json"); print("DONE")

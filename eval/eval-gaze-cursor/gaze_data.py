"""Loader for the gaze-cursor CSVs in human_data/gaze_cursor_data/.

Each CSV is a 200 Hz gaze-clock table with the cursor resampled onto it and
precomputed gaze-lead metrics. Column headers embed parenthetical descriptions
(two contain newlines), so everything is parsed with pandas and renamed to
short keys here.

Two views:
  load_samples(letter)    -> per-sample DataFrame (in-trial rows only)
  fixation_events(samples)-> one row per (trial_id, fixation_id): the state at
                             fixation onset plus catch-up over the fixation.
The fixation-onset view is the behaviourally meaningful one for horizon
analysis: gaze saccades ahead to an anchor and dwells while the cursor closes
the gap, so instantaneous lead oscillates around zero while onset lead is the
anticipatory planning distance.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "human_data" / "gaze_cursor_data"

PARTICIPANTS = ["A", "B", "C"]

# long CSV header -> short column name (matched by prefix so the parenthetical
# descriptions don't have to be reproduced exactly)
_PREFIX_RENAMES = {
    "neon_time_s": "t",
    "gaze_transf_x_px": "gaze_x_px",
    "gaze_transf_y_px": "gaze_y_px",
    "fixation_id": "fixation_id",
    "cursor_traj_x": "cursor_x",
    "cursor_traj_y": "cursor_y",
    "cursor_screenvid_x_px": "cursor_px_x",
    "cursor_screenvid_y_px": "cursor_px_y",
    "cursor_speed": "speed",
    "gaze_cursor_dist_px": "dist_px",
    "gaze_cursor_signed_px": "signed_px",
    "gaze_lead_signed": "lead",
    "json_local_curvature": "curvature",
    "json_local_width": "width",
    "trial_id": "trial_id",
    "condition": "condition",
    "condition_json": "condition_json",
    "constrained": "constrained",
    "tunnelType": "tunnel_type",
    "in_trial": "in_trial",
}


def csv_path(letter: str) -> Path:
    matches = sorted(DATA_DIR.glob(f"{letter}_task_aligned_analysis.csv"))
    if not matches:
        raise FileNotFoundError(f"no gaze CSV for participant {letter} in {DATA_DIR}")
    return matches[0]


def load_samples(letter: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path(letter), low_memory=False)

    renames = {}
    for col in df.columns:
        for prefix, short in _PREFIX_RENAMES.items():
            if col == prefix or col.startswith(prefix + " "):
                renames[col] = short
                break
    df = df.rename(columns=renames)[list(dict.fromkeys(renames.values()))]

    df = df[df["in_trial"] == True].copy()  # noqa: E712
    for col in ("speed", "lead", "curvature", "width", "cursor_x", "cursor_y",
                "cursor_px_x", "cursor_px_y", "gaze_x_px", "gaze_y_px"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["trial_id"] = pd.to_numeric(df["trial_id"], errors="coerce").astype("Int64")
    df["participant"] = letter

    # rounds are not a CSV column: segment by gaps in the gaze clock within a
    # trial (a new round of the same trial re-enters in_trial after a gap)
    df = df.sort_values("t").reset_index(drop=True)
    new_block = (df["trial_id"] != df["trial_id"].shift(1)) | (df["t"].diff() > 0.5)
    df["block_id"] = new_block.cumsum()
    return df


def fixation_events(samples: pd.DataFrame) -> pd.DataFrame:
    d = samples.dropna(subset=["fixation_id", "lead"]).copy()
    d["fixation_id"] = pd.to_numeric(d["fixation_id"], errors="coerce")

    grouped = d.groupby(["block_id", "fixation_id"], sort=True)
    first = grouped.first()
    last = grouped.last()
    agg = grouped.agg(n_samples=("t", "size"), t_end=("t", "max"), t_start=("t", "min"))

    events = first[
        [
            "participant", "trial_id", "tunnel_type", "constrained", "condition",
            "lead", "speed", "width", "curvature", "cursor_x", "cursor_y", "t",
        ]
    ].rename(columns={"lead": "lead_onset", "speed": "speed_onset", "t": "t_onset"})
    events["lead_end"] = last["lead"]
    events["duration_s"] = (agg["t_end"] - agg["t_start"]).clip(lower=0.0)
    events["n_samples"] = agg["n_samples"]
    events["catch_up"] = events["lead_onset"] - events["lead_end"]
    # implied preview time at onset (guard the near-stationary samples)
    events["th_emp"] = events["lead_onset"] / events["speed_onset"].clip(lower=1e-3)
    return events.reset_index()


def load_all(letters=PARTICIPANTS):
    samples = pd.concat([load_samples(p) for p in letters], ignore_index=True)
    events = pd.concat(
        [fixation_events(samples[samples["participant"] == p]) for p in letters],
        ignore_index=True,
    )
    return samples, events

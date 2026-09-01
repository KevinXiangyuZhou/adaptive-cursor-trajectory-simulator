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
    "saccade_id": "saccade_id",
    "blink_id": "blink_id",
    "time_to_catch_s": "time_to_catch",
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
    # Prefer the re-exported "(1)" CSVs, which add saccade_id / blink_id /
    # time_to_catch_s on top of the original export. The 10-participant batch
    # (p01..p10) lives in human_data/task_aligned_all with the same schema.
    dirs = [DATA_DIR, DATA_DIR.parent / "task_aligned_all"]
    matches = []
    for d in dirs:
        matches += sorted(d.glob(f"{letter}_task_aligned_analysis*.csv"),
                          key=lambda p: (" (1)" not in p.name, p.name))
    if not matches:
        raise FileNotFoundError(f"no gaze CSV for participant {letter} in {dirs}")
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
                "cursor_px_x", "cursor_px_y", "gaze_x_px", "gaze_y_px",
                "saccade_id", "blink_id", "time_to_catch"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["trial_id"] = pd.to_numeric(df["trial_id"], errors="coerce").astype("Int64")
    df["participant"] = letter

    # rounds are not a CSV column: segment by gaps in the gaze clock within a
    # trial (a new round of the same trial re-enters in_trial after a gap)
    df = df.sort_values("t").reset_index(drop=True)
    new_block = (df["trial_id"] != df["trial_id"].shift(1)) | (df["t"].diff() > 0.5)
    df["block_id"] = new_block.cumsum()
    _add_gaze_task_coords(df)
    return df


def _add_gaze_task_coords(df: pd.DataFrame) -> None:
    """Add gaze_task_x / gaze_task_y: gaze screen-video pixels mapped into
    task units. The video mapping is a single uniform scale plus a per-axis
    offset (screenvid = cursor_scale * screen + b, screen = 1000 * task), so
    the scale is fit globally on cursor x (task units <-> screen-video px;
    x spans the whole tunnel, so the fit is well-conditioned, ~5850 px per
    task unit with <5 px residuals) and only the offsets are estimated per
    block via the median. This avoids degenerate per-block y fits in
    straight tunnels where cursor y barely moves."""
    df["gaze_task_x"] = np.nan
    df["gaze_task_y"] = np.nan
    pair_all = df.dropna(subset=["cursor_x", "cursor_px_x"])
    if len(pair_all) < 2 or np.ptp(pair_all["cursor_x"].to_numpy()) < 1e-3:
        return
    scale, _ = np.polyfit(pair_all["cursor_x"], pair_all["cursor_px_x"], 1)
    if abs(scale) < 1e-9:
        return
    for _, g in df.groupby("block_id", sort=False):
        for task_col, px_col, gaze_px_col, out_col in (
                ("cursor_x", "cursor_px_x", "gaze_x_px", "gaze_task_x"),
                ("cursor_y", "cursor_px_y", "gaze_y_px", "gaze_task_y")):
            pair = g.dropna(subset=[task_col, px_col])
            if pair.empty:
                continue
            offset = float((pair[px_col] - scale * pair[task_col]).median())
            df.loc[g.index, out_col] = (df.loc[g.index, gaze_px_col]
                                        - offset) / scale


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
    # median gaze position over the fixation, in task units (stable against
    # the per-sample tracker jitter) — NaN when the block's affine fit failed
    if "gaze_task_x" in d.columns:
        med = grouped[["gaze_task_x", "gaze_task_y"]].median()
        events["gaze_task_x"] = med["gaze_task_x"]
        events["gaze_task_y"] = med["gaze_task_y"]
    if "time_to_catch" in d.columns:
        events["time_to_catch"] = grouped["time_to_catch"].median()
        # onset value: the planning-event read of lead/v — the direct
        # empirical counterpart of the model's T_min floor (h >= v*T_min)
        events["ttc_onset"] = first["time_to_catch"]
    events = events.reset_index()

    # Blink filter: a fixation is blink-corrupted when a blink overlaps the
    # fixation itself or the saccade leading into it (previous fixation end ->
    # this onset). Blink samples carry blink_id but no fixation_id, so test
    # the raw per-sample table over each event's [previous end, end] window.
    events["blink_corrupted"] = False
    if "blink_id" in samples.columns:
        for bid, ev in events.groupby("block_id", sort=False):
            blink_t = samples.loc[(samples["block_id"] == bid)
                                  & samples["blink_id"].notna(), "t"].to_numpy()
            if not len(blink_t):
                continue
            ev = ev.sort_values("t_onset")
            t_end = (agg["t_end"].loc[bid].reindex(ev["fixation_id"]).to_numpy())
            win_start = np.concatenate([[ev["t_onset"].iloc[0] - 0.15], t_end[:-1]])
            corrupted = [bool(np.any((blink_t >= ws) & (blink_t <= we)))
                         for ws, we in zip(win_start, t_end)]
            events.loc[ev.index, "blink_corrupted"] = corrupted

    # Saccade-followed diagnostic: whether a detected saccade starts within
    # 0.10 s after the fixation ends. ~30% of fixation transitions have no
    # labelled saccade (small/undetected re-targets), so this is reported as
    # a robustness split, NOT used as a hard filter.
    events["saccade_followed"] = True
    if "saccade_id" in samples.columns:
        events["saccade_followed"] = False
        for bid, ev in events.groupby("block_id", sort=False):
            sac_t = samples.loc[(samples["block_id"] == bid)
                                & samples["saccade_id"].notna(), "t"].to_numpy()
            if not len(sac_t):
                continue
            t_end = (agg["t_end"].loc[bid].reindex(ev["fixation_id"]).to_numpy())
            followed = [bool(np.any((sac_t > te) & (sac_t <= te + 0.10)))
                        for te in t_end]
            events.loc[ev.index, "saccade_followed"] = followed
    return events


def load_all(letters=PARTICIPANTS):
    samples = pd.concat([load_samples(p) for p in letters], ignore_index=True)
    events = pd.concat(
        [fixation_events(samples[samples["participant"] == p]) for p in letters],
        ignore_index=True,
    )
    return samples, events

"""Corner-strategy phenotype from cursor JSONs: apex speed-dip ratio.

For every corner apex crossing, the minimum speed within +-0.015 task units of
the apex x, divided by the trial's median moving speed. Near 0 = full stop
("consecutive pointing" strategy); higher = carries speed through the turn
(corner-cutting strategy). Aug-2026 prolific cohort spans 0.00-0.41 median
with a cluster of full-stoppers; gaze participants A/B/C sit mid-sharp
(0.09-0.12) yet show chance-level gaze apex anchoring -- the strategy is
motor-side, not gaze-side, for that range.

Run: python3 corner_phenotype.py [data_dir ...]
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DIRS = [
    REPO / "human_data" / "aug-26-prolific",
    REPO / "human_data" / "gaze_cursor_data",
]
APEX_WINDOW_X = 0.015
NEAR_STOP_RATIO = 0.2
TUNNEL_END_X = 0.46


def trial_list(data: dict) -> list:
    return data.get("trialData") or [
        t for s in data.get("sessions", []) for t in s.get("trialData", [])
    ]


def apex_dips(trials: list) -> np.ndarray:
    dips = []
    for t in trials:
        cond = t.get("condition", {})
        if cond.get("tunnelType") != "corner":
            continue
        traj = np.array([[p["x"], p["y"]] for p in t.get("trajectory", [])], dtype=float)
        spd = np.array(t.get("speeds", []), dtype=float)
        n_c = cond.get("numCorners")
        if len(traj) < 10 or len(spd) < 10 or not n_c:
            continue
        m = min(len(traj), len(spd))
        traj, spd = traj[:m], spd[:m]
        seg = TUNNEL_END_X / (n_c + 1)
        moving = spd > 1e-4
        if not moving.any():
            continue
        ref = np.median(spd[moving])
        for i in range(1, n_c + 1):
            near = np.abs(traj[:, 0] - i * seg) < APEX_WINDOW_X
            if near.sum() >= 2 and ref > 0:
                dips.append(np.min(spd[near]) / ref)
    return np.asarray(dips)


def main(dirs):
    rows = []
    for d in dirs:
        for f in sorted(glob.glob(str(Path(d) / "*.json"))):
            if "index" in Path(f).name:
                continue
            data = json.load(open(f))
            dips = apex_dips(trial_list(data))
            if len(dips):
                pid = data.get("participantId", Path(f).stem)
                rows.append((pid, len(dips), float(np.median(dips)),
                             float(np.mean(dips < NEAR_STOP_RATIO))))
    rows.sort(key=lambda r: r[2])
    print("participant                 n_apex  dip_ratio_p50  frac_near_stop")
    for pid, n, p50, fs in rows:
        print(f"{pid:26s}  {n:5d}      {p50:.2f}           {fs:.2f}")
    return rows


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_DIRS)

"""Train the traversal-speed GAM artifacts shipped with the simulator.

Fits hcs_package.speed_model.GAMSpeedModel (v5 features: clearance = W/2,
|kappa| local, anticipatory kappa = max |kappa| over the next 50 mm, and
runway = arc distance to the next steering demand; v6 target: arc-binned
conditional-mean pace, so the deadline integral reproduces human traversal
times) on the arc-binned data
(human-gaze-lead-10p/data/local_pace_samples.csv, produced by
local_speed_law.py) and saves the pickles the gaze module loads for the plan
deadline t_plan = integral ds / v(s) = integral tau_hat(s) ds.

Per-participant artifacts (2026-09-03): alongside the pooled
gam_traversal_10p.pkl, one gam_traversal_{letter}.pkl is fitted per
participant in the pace data. A persona selects its own artifact with
    "speed_model": {"type": "gam_traversal", "path": "gam_traversal_p01.pkl"}
— a relative path is resolved against the package models/ directory
(cursor_simulator), so persona JSONs stay portable between machines. With no
"path" the pooled artifact is used, so existing configs are unaffected.

Usage: python train_traversal_gam.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(ROOT / "hcs_package" / "src"))
from hcs_package.speed_model import GAMSpeedModel

DATA = SCRIPT_DIR / "human-gaze-lead-10p" / "data" / "local_pace_samples.csv"
MODELS = ROOT / "hcs_package" / "src" / "hcs_package" / "models"


def fit_one(d, out_path, tag):
    m = GAMSpeedModel()
    m.fit((d["W_loc"] / 2).to_numpy(), d["k_loc"].to_numpy(),
          d["k_ahead"].to_numpy(), d["d_demand"].to_numpy(),
          d["tau"].to_numpy())
    m.save(str(out_path))
    # round-trip sanity + the fix-1 contrast: same narrow straight
    # geometry, short vs long runway; and a narrow-vs-wide curved probe
    m2 = GAMSpeedModel.load(str(out_path))
    z = np.array([0.0, 0.0])
    v_leg = m2.predict_speed_raw(np.array([0.005, 0.005]), z, z,
                                 np.array([0.05, 0.40]))
    v_cur = m2.predict_speed_raw(np.array([0.005, 0.025]),
                                 np.array([20.0, 20.0]),
                                 np.array([20.0, 20.0]),
                                 np.array([0.0, 0.0]))
    print(f"{tag:4s} n={len(d):6d}  straight W10 leg/tunnel: "
          f"{v_leg[0]:.3f}/{v_leg[1]:.3f} m/s  curved k20 W10/W50: "
          f"{v_cur[0]:.3f}/{v_cur[1]:.3f} m/s  -> {out_path.name}")


def main():
    d = pd.read_csv(DATA)
    MODELS.mkdir(parents=True, exist_ok=True)
    fit_one(d, MODELS / "gam_traversal_10p.pkl", "10p")
    for letter, dl in d.groupby("participant"):
        fit_one(dl, MODELS / f"gam_traversal_{letter}.pkl", letter)


if __name__ == "__main__":
    main()

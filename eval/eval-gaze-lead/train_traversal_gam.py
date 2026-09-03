"""Train the traversal-speed GAM artifact shipped with the simulator.

Fits hcs_package.speed_model.GAMSpeedModel (v5 features: clearance = W/2,
|kappa| local, anticipatory kappa = max |kappa| over the next 50 mm, and
runway = arc distance to the next steering demand; v6 target: arc-binned
conditional-mean pace, so the deadline integral reproduces human traversal
times) on the pooled 10-participant arc-binned data
(human-gaze-lead-10p/data/local_pace_samples.csv, produced by
local_speed_law.py) and saves the pickle the gaze module loads for the plan
deadline t_plan = integral ds / v(s) = integral tau_hat(s) ds.

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
OUT = ROOT / "hcs_package" / "src" / "hcs_package" / "models" / "gam_traversal_10p.pkl"


def main():
    d = pd.read_csv(DATA)
    m = GAMSpeedModel()
    m.fit((d["W_loc"] / 2).to_numpy(), d["k_loc"].to_numpy(),
          d["k_ahead"].to_numpy(), d["d_demand"].to_numpy(),
          d["tau"].to_numpy())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(OUT))
    # round-trip sanity: reload and spot-check the raw prediction
    m2 = GAMSpeedModel.load(str(OUT))
    v = m2.predict_speed_raw(np.array([0.005, 0.025]), np.array([1.0, 1.0]),
                             np.array([5.0, 5.0]), np.array([0.02, 0.02]))
    print(f"trained on {len(d)} samples -> {OUT}")
    print(f"sanity raw speeds (W=10mm/50mm, kappa 1, ahead 5, runway 20mm): "
          f"{np.round(v, 3)}")
    # the fix-1 contrast: same narrow straight geometry, short vs long runway
    z = np.array([0.0, 0.0])
    v_leg = m2.predict_speed_raw(np.array([0.005, 0.005]), z, z,
                                 np.array([0.05, 0.40]))
    print(f"straight W=10mm, runway 50mm (corner leg) vs 400mm (straight "
          f"tunnel): {np.round(v_leg, 3)}")


if __name__ == "__main__":
    main()

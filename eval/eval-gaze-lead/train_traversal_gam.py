"""Train the traversal-speed GAM artifact shipped with the simulator.

Fits hcs_package.speed_model.GAMSpeedModel (exact graveyard code, v4
features: clearance = W/2, |kappa| local, anticipatory kappa = max |kappa|
over the next 50 mm) on the pooled 10-participant per-sample data
(human-gaze-lead-10p/data/local_speed_samples.csv, produced by
local_speed_law.py) and saves the pickle the gaze module loads for the plan
deadline t_plan = integral ds / v(s).

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

DATA = SCRIPT_DIR / "human-gaze-lead-10p" / "data" / "local_speed_samples.csv"
OUT = ROOT / "hcs_package" / "src" / "hcs_package" / "models" / "gam_traversal_10p.pkl"


def main():
    d = pd.read_csv(DATA)
    m = GAMSpeedModel()
    m.fit((d["W_loc"] / 2).to_numpy(), d["k_loc"].to_numpy(),
          d["k_ahead"].to_numpy(), d["v"].to_numpy())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(OUT))
    # round-trip sanity: reload and spot-check the raw prediction
    m2 = GAMSpeedModel.load(str(OUT))
    v = m2.predict_speed_raw(np.array([0.005, 0.025]), np.array([1.0, 1.0]),
                             np.array([5.0, 5.0]))
    print(f"trained on {len(d)} samples -> {OUT}")
    print(f"sanity raw speeds (W=10mm/50mm, kappa 1, ahead 5): {np.round(v, 3)}")


if __name__ == "__main__":
    main()

"""Fit the CHI-26-EA baseline to one participant — same protocol as the
current model (eval/eval-anchor-drive/fit_anchor.py --model baseline).

This is a thin alias: the objective, data split, CMA-ES budget, held-out
summary and output layout are all fit_anchor's. Eight parameters are fitted:
the six EA planner weights (jerk, progress, wall, contour, lag,
desired_speed), the horizon Th, and the curvature attenuation constant
curvature_scale. Pointing runs through the same bypass corridor as the
current model. Output: $HCS_FIT_RESULTS_DIR/stages/base/{pid}_baseline_config_s{seed}.json.

Usage:
    python eval/baseline_fitting/fit_baseline.py --pid p01 --time-limit 23400 --popsize 12 --workers 12
    python eval/baseline_fitting/fit_baseline.py --pid p01 --quick --time-limit 120 --popsize 4 --workers 4 --seed 7
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval-anchor-drive"))
import fit_anchor  # noqa: E402

if __name__ == "__main__":
    if "--model" in sys.argv:
        raise SystemExit("fit_baseline.py fixes --model baseline; use fit_anchor.py to choose a model")
    sys.argv.insert(1, "--model"); sys.argv.insert(2, "baseline")
    fit_anchor.main()

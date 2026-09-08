"""ONE pooled CHI-26-EA baseline persona fitted jointly on all eight
participants — alias of eval/eval-anchor-drive/fit_anchor_pooled8.py
--model baseline (same pooled loss, (candidate x participant) parallelism,
held-out summary and output layout).
Output: $HCS_FIT_RESULTS_DIR/stages/pooled8/pooled8_baseline_config_s{seed}.json.

Usage:
    python eval/baseline_fitting/fit_baseline_pooled8.py --time-limit 18000 --workers 36
    python eval/baseline_fitting/fit_baseline_pooled8.py --quick --time-limit 120 --workers 8 --letters p01 p02 --skip-probe
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval-anchor-drive"))
import fit_anchor_pooled8  # noqa: E402

if __name__ == "__main__":
    if "--model" in sys.argv:
        raise SystemExit("fit_baseline_pooled8.py fixes --model baseline; use fit_anchor_pooled8.py to choose a model")
    sys.argv.insert(1, "--model"); sys.argv.insert(2, "baseline")
    fit_anchor_pooled8.main()

"""Probe: is the velocity-damping term (free_velocity * sum |v|^2) load-bearing,
or do the other terms (via-point deadline drive, acc_max hinge, turn-time
deadline, walls, jerk) already produce the slowing?  S15 design question.

Variants on the B S12 persona (width-only budget line): as fitted (0.227),
half damping, zero damping.  Quick probe, tunnel + pointing.
Run: python probe_no_damping.py [quick|full]
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_anchor as pa

base = json.load(open(HERE / "results" / "stages" / "S12" / "P170114_anchor_persona_S12.json"))
base.pop("_description", None)
FV = base["planner_weights"]["free_velocity"]

def run(label, fv, quick):
    ov = json.loads(json.dumps(base))
    ov["planner_weights"]["free_velocity"] = fv
    print(f"\n##### {label} (free_velocity={fv:.4g}, {'quick' if quick else 'full'})", flush=True)
    return pa.run_probe("P170114", "anchor", ov, quick=quick, n_workers=12,
                        buckets=("tunnel", "pointing"))

if __name__ == "__main__":
    quick = (sys.argv[1] if len(sys.argv) > 1 else "quick") == "quick"
    out = {}
    for label, fv in (("S12 as fitted", FV), ("half damping", FV / 2), ("no damping", 0.0)):
        r = run(label, fv, quick)
        out[label] = {"fv": fv, "tunnel": r["tunnel"], "pointing": r.get("pointing")}
    json.dump(out, open(HERE / "results" / "stages" / "S12" / "probe_no_damping_S12_B.json", "w"), default=float, indent=1)
    print("\nsaved results/stages/S12/probe_no_damping_S12_B.json")

"""Copy a fitted persona into a run's personas/ dir under the participant's
Prolific id (the name run_eval's --config-dir resolution expects), with the
fitting harness's deterministic settings undone.

    python cluster/stage_persona.py <fitted config> <personas dir>/<PID>.json <model>

Guards against personas saved noiseless (add_noise false; for mpcc also
replan latency cv 0) — the eval must run the stochastic model.
"""
import json
import sys

src, dst, model = sys.argv[1], sys.argv[2], sys.argv[3]
c = json.load(open(src))
c["add_noise"] = True
if model == "mpcc" and not float(c.get("replan_latency_cv", 0) or 0):
    c["replan_latency_cv"] = 0.89
c.setdefault("_fit", {})["staged_from"] = src
json.dump(c, open(dst, "w"), indent=2)
print(f"staged {src} -> {dst}")

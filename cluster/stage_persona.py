"""Copy a fitted persona into a run's personas/ dir under the participant's
Prolific id (the name run_eval's --config-dir resolution expects), with the
fitting harness's deterministic settings undone.

    python cluster/stage_persona.py <fitted config> <personas dir>/<PID>.json <model> [<kind>]

Guards against personas saved noiseless (add_noise false; for mpcc also
replan latency cv 0) — the eval must run the stochastic model. With <kind>
given, also reports which traversal GAM the persona carries and warns when a
perpid persona is about to eval on the pooled artifact (which collapses
cross-participant completion-time SD, 2026-09-09) or a pooled one on a
per-participant artifact.
"""
import json
import sys

src, dst, model = sys.argv[1], sys.argv[2], sys.argv[3]
kind = sys.argv[4] if len(sys.argv) > 4 else None
c = json.load(open(src))
c["add_noise"] = True
if model == "mpcc" and not float(c.get("replan_latency_cv", 0) or 0):
    c["replan_latency_cv"] = 0.89
if model == "mpcc" and kind:
    sm = c.get("speed_model")
    if isinstance(sm, dict) and sm.get("type") == "gam_traversal":
        path = sm.get("path")
        print(f"  traversal GAM: {path or 'pooled (gam_traversal_10p.pkl)'}")
        if kind != "pooled8" and not path:
            print(f"WARNING: {kind} persona {src} has no per-participant GAM "
                  f"path — eval will use the POOLED artifact", flush=True)
        if kind == "pooled8" and path:
            print(f"WARNING: pooled8 persona {src} pins {path} — eval should "
                  f"use the pooled artifact", flush=True)
c.setdefault("_fit", {})["staged_from"] = src
json.dump(c, open(dst, "w"), indent=2)
print(f"staged {src} -> {dst}")

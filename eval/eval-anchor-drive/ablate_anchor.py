"""Quick ablations of the fitted anchor persona: noiseless quick-subset probe
(tunnel + pointing losses) and a noise-on gaze-lead run (cycle rate, trigger
mix, onset lead and lead/speed by width/type) per variant.

Usage: python ablate_anchor.py --pid P170114 --persona results/P170114_anchor_persona_8h2.json
"""
import argparse, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "eval" / "eval-main"))
import probe_anchor as pa, run_eval as em

S1 = {"planner_weights": {"bend": 0.0}, "arrival_mode": "distance", "replan_deviation_frac": 0.3}
def _kap(D0): return {**S1, "budget": {"D0": D0, "T_min": 0.1, "gamma": 0.6, "W_ref": 0.026, "curvature_weighted": True}}
VARIANTS = {
    "S2":            {},
    "S2+leadfloor":  {"anchor_lead_floor": True},
    "S1":            S1,
    "S1+kappa0.3":   _kap(0.3),
    "S1+kappa0.6":   _kap(0.6),
    "S1+kappa1.2":   _kap(1.2),
    "fit":            {},
    "no-bend":        {"planner_weights": {"bend": 0.0}},
    "no-safety":      {"coast_safety": False},
    "bare":           {"planner_weights": {"bend": 0.0}, "coast_safety": False},
    "dist-arrival":   {"arrival_mode": "distance"},
    "bare+dist":      {"planner_weights": {"bend": 0.0}, "coast_safety": False, "arrival_mode": "distance"},
    "bare+dist+dev.3": {"planner_weights": {"bend": 0.0}, "coast_safety": False, "arrival_mode": "distance", "replan_deviation_frac": 0.3},
}
LETTER = {"P105835": "A", "P170114": "B", "P160254": "C"}


def gaze_stats(pid, persona_path, tag, noise=True):
    out = HERE / "results" / "ablation-gaze-lead" / tag
    subprocess.run([sys.executable, str(ROOT / "eval/eval-gaze-lead/model_gaze_lead.py"), "--letters", LETTER[pid],
                    "--noise", "on" if noise else "off", "--config", str(persona_path), "--out-dir", str(out)],
                   check=True, capture_output=True)
    ev = pd.read_csv(out / "model_lead_events.csv")
    t2c, _ = em.scan_conditions(ROOT / "human_data" / "gaze_cursor_data")
    st = ev[ev.bucket == "steering"].copy(); st["width"] = st.tid.map(lambda t: t2c[t]["tunnelWidth"]); st["ttype"] = st.tid.map(lambda t: t2c[t].get("tunnelType"))
    trig = st[st.trigger != "init"].trigger.value_counts(normalize=True)
    st = st[(st.lead_onset > 1e-4) & (st.speed_onset > 0.01) & (st.trigger != "init")]; st["th"] = st.lead_onset / st.speed_onset
    cyc = st.sort_values(["tid", "t"]).groupby("tid").t.diff().dropna().median()
    return {"cycle_s": cyc, "frac_arrival": float(trig.get("arrival+latency", 0)), "frac_dev": float(trig.get("deviation", 0)), "frac_exh": float(trig.get("exhausted", 0)),
            "lead_mm": {w: round(v * 1000, 1) for w, v in st.groupby("width").lead_onset.median().items()},
            "th_type": {k[:5]: round(v, 2) for k, v in st.groupby("ttype").th.median().items()}}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pid", default="P170114"); ap.add_argument("--persona", required=True)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS)); ap.add_argument("--workers", type=int, default=12); ap.add_argument("--no-gaze", action="store_true")
    a = ap.parse_args()
    base = json.load(open(a.persona)); base.pop("_description", None)
    rows = []
    for name in a.variants:
        ov = VARIANTS[name]
        cfg = json.loads(json.dumps(base))
        for k, v in ov.items():
            if k == "planner_weights": cfg["planner_weights"].update(v)
            elif k == "budget": cfg["budget"] = dict(v)
            else: cfg[k] = v
        # noiseless quick probe (same protocol as fitting)
        probe_ov = {k: v for k, v in cfg.items() if k not in ("speed_model", "reference_path", "_description")}
        r = pa.run_probe(a.pid, "anchor", override=probe_ov, quick=True, n_workers=a.workers, verbose=False)
        t, p = r["tunnel"], r.get("pointing") or {}
        row = {"variant": name, "tun_tr": t["train"]["loss"], "tun_te": t["test"]["loss"], "ctr": t["train"]["ct_ratio"],
               "types": {k[:5]: round(v["ct_ratio"], 2) for k, v in t["by_type"].items()},
               "pt_tr": (p.get("train") or {}).get("loss", np.nan), "pt_te": (p.get("test") or {}).get("loss", np.nan), "mtr": (p.get("train") or {}).get("mt_ratio", np.nan)}
        if not a.no_gaze:
            pp = HERE / "results" / "ablation-gaze-lead" / f"{name}.json"; pp.parent.mkdir(parents=True, exist_ok=True)
            json.dump(cfg, open(pp, "w"))
            row.update(gaze_stats(a.pid, pp, name))
        rows.append(row)
        print(f"{name:16s} tunnel {row['tun_tr']:6.2f}/{row['tun_te']:6.2f} CTr {row['ctr']:.2f} {row['types']} | pointing {row['pt_tr']:6.2f}/{row['pt_te']:6.2f} MTr {row['mtr']:.2f}"
              + (f" | gaze: cycle {row['cycle_s']:.2f}s arr/dev/exh {row['frac_arrival']:.2f}/{row['frac_dev']:.2f}/{row['frac_exh']:.2f} lead {row['lead_mm']} th {row['th_type']}" if not a.no_gaze else ""), flush=True)
    json.dump(rows, open(HERE / "results" / "ablation_summary.json", "w"), default=float, indent=1)


if __name__ == "__main__":
    main()

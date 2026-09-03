"""One-command validation of a fitted anchor persona for one participant:
stochastic persona -> eval-main (Fitts / steering law) -> gaze-lead run -> summary
table next to the aug-26 GAM persona and any earlier tagged personas.

Usage: python validate_persona.py --pid P170114 --tag S3 [--compare 8h2 S2]
Reads results/stages/{tag}/{pid}_anchor_config_{tag}_s42.json and the matching _anchor_fit_ record.
"""
import argparse, csv, json, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; R = HERE / "results"
STG = R / "stages"; EM = R / "eval-main"
sys.path.insert(0, str(ROOT / "eval" / "eval-main"))
import run_eval as em
LETTER = {"P105835": "A", "P170114": "B", "P160254": "C"}
CV = {"P105835": 0.8415, "P170114": 0.7633, "P160254": 1.0882}
HUMAN_TH = {"straight": 0.136, "corner": 0.178, "gentle_sinusoidal": 0.224, "mid_sinusoidal": 0.220, "sharp_sinusoidal": 0.254}


def laws(D):
    f = json.load(open(D / "Fitts" / "fitts_regression.json"))["aligned"]
    rows = [r for r in csv.DictReader(open(D / "Steering" / "steering_results.csv")) if r["timed_out"].lower() in ("false", "0", "")]
    out = {}
    for src in ("Human", "Simulator"):
        s = [r for r in rows if r["source"] == src]; ids = np.array([float(r["ID"]) for r in s]); mt = np.array([float(r["MT_s"]) for r in s])
        b = np.polyfit(ids, mt, 1); out[src] = (b, 1 - np.sum((mt - np.polyval(b, ids)) ** 2) / np.sum((mt - mt.mean()) ** 2))
    rat = [np.mean([float(r["MT_s"]) for r in rows if r["tid"] == t and r["source"] == "Simulator"]) / np.mean([float(r["MT_s"]) for r in rows if r["tid"] == t and r["source"] == "Human"]) for t in sorted({r["tid"] for r in rows})]
    return f, out, float(np.exp(np.mean(np.log(rat))))


def gaze(pid, folder):
    ev = pd.read_csv(folder / "model_lead_events.csv"); t2c, _ = em.scan_conditions(ROOT / "human_data" / "gaze_cursor_data")
    st = ev[ev.bucket == "steering"].copy(); st["width"] = st.tid.map(lambda t: t2c[t]["tunnelWidth"]); st["ttype"] = st.tid.map(lambda t: t2c[t].get("tunnelType"))
    trig = st[st.trigger != "init"].trigger.value_counts(normalize=True)
    st = st[(st.lead_onset > 1e-4) & (st.speed_onset > 0.01) & (st.trigger != "init")]; st["th"] = st.lead_onset / st.speed_onset
    cyc = st.sort_values(["tid", "t"]).groupby("tid").t.diff().dropna().median(); ns = st[st.ttype != "straight"]
    return cyc, trig, ns.groupby("width").lead_onset.median().mul(1000).round(0).to_dict(), ns.groupby("ttype").th.median().round(2).to_dict()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pid", default="P170114"); ap.add_argument("--tag", required=True)
    ap.add_argument("--compare", nargs="*", default=[]); ap.add_argument("--skip-runs", action="store_true")
    a = ap.parse_args(); pid, L = a.pid, LETTER[a.pid]
    stage_dir = STG / a.tag; stage_dir.mkdir(parents=True, exist_ok=True); EM.mkdir(exist_ok=True)
    persona = stage_dir / f"{pid}_anchor_persona_{a.tag}.json"
    if not a.skip_runs:
        c = json.load(open(stage_dir / f"{pid}_anchor_config_{a.tag}_s42.json")); c["add_noise"] = True; c["replan_latency_cv"] = CV[pid]; c["random_seed"] = 1000
        c["_description"] = f"anchor persona {a.tag} for {pid}, noise on"; json.dump(c, open(persona, "w"), indent=2)
        subprocess.run([sys.executable, str(ROOT / "eval/eval-main/run_eval.py"), "--config", str(persona), "--pid", pid, "--data-dir", str(ROOT / "human_data/gaze_cursor_data"),
                        "--results-dir", str(EM / f"eval-main-{a.tag}-{L}"), "--fresh-sim"], check=True, capture_output=True)
        misplaced = ROOT / "eval/eval-main" / "eval/eval-anchor-drive/results/eval-main" / f"eval-main-{a.tag}-{L}"
        if misplaced.exists():
            import shutil; shutil.rmtree(EM / f"eval-main-{a.tag}-{L}", ignore_errors=True); shutil.move(str(misplaced), str(EM / f"eval-main-{a.tag}-{L}"))
        subprocess.run([sys.executable, str(ROOT / "eval/eval-gaze-lead/model_gaze_lead.py"), "--letters", L, "--noise", "on", "--config", str(persona),
                        "--out-dir", f"model-gaze-lead-{a.tag}-{L}"], check=True, capture_output=True)
    # table
    entries = [("aug-26 GAM", {r["pid"]: r for r in json.load(open(R / "probes" / "gam_full.json"))}[pid], EM / f"eval-main-gam-{L}", None)]
    for t in list(a.compare) + [a.tag]:
        rec = json.load(open(STG / t / f"{pid}_anchor_fit_{t}_s42.json")); em_dir = EM / f"eval-main-{t}-{L}"
        entries.append((f"anchor {t}", rec, em_dir if em_dir.exists() else None, ROOT / "eval/eval-gaze-lead" / f"model-gaze-lead-{t}-{L}"))
    print(f"\n{pid} ({L}) — full trial sets, corrected geometry, same build")
    for name, rec, em_dir, gz in entries:
        t, p = rec["tunnel"], rec["pointing"]; tr, te = t["train"], t["test"]; ptr, pte = p["train"], p["test"]
        line = f"{name:12s} tunnel {tr['loss']:6.2f}/{te['loss']:6.2f} CTr {tr['ct_ratio']:.2f}/{te['ct_ratio']:.2f} spdcorr {tr['spd_corr']:.2f} | pointing {ptr['loss']:6.2f}/{pte['loss']:6.2f} MTr {ptr['mt_ratio']:.2f}/{pte['mt_ratio']:.2f}"
        if em_dir and em_dir.exists():
            f, law, rat = laws(em_dir)
            line += f" | Fitts b {f['model']['b_slope_s_per_bit']:.3f} R² {f['model']['r_squared']:.2f} (hum {f['human']['b_slope_s_per_bit']:.3f}) | steering b {law['Simulator'][0][0]:.3f} (hum {law['Human'][0][0]:.3f}) CTr {rat:.2f}"
        print(line)
        bt = t["by_type"]; print("   by type: " + "  ".join(f"{ty[:6]} {bt[ty]['ct_ratio']:.2f}" for ty in HUMAN_TH if ty in bt) + " | by width: " + " ".join(f"{float(w)*1000:.0f}:{v['ct_ratio']:.2f}" for w, v in t["by_width"].items()))
        if gz and gz.exists():
            cyc, trig, lead, th = gaze(pid, gz)
            print(f"   gaze: cycle {cyc:.2f}s (hum 0.38) arrival/early/exh {trig.get('arrival+latency',0):.2f}/{trig.get('deviation',0):.2f}/{trig.get('exhausted',0):.2f} | lead mm {lead} (hum 16/23/34/40/46) | lead/v {th} (hum .18/.22/.22/.25)")


if __name__ == "__main__":
    main()

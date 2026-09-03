"""Evaluate the pooled new-cohort anchor persona on each pool participant.

For each participant (noise ON, fitted pooled config):
  * tunnel: per-trial sim (one seed), completion-time ratio by width and type,
    human-variability-scaled tunnel loss (train/test widths);
  * pointing: MT ratio per radius, Fitts slope of sim vs human;
  * gaze rhythm: model replan cycle, trigger shares, model onset-lead medians
    by width vs the participant's canonical (corrected, merged) human leads.

Usage: python eval_pooled.py [--config results/pooled10_anchor_config_s42.json]
Saves results/pooled10_eval.json and prints the summary table.
"""
import argparse, copy, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_anchor as pa       # noqa: F401
import fit_speed_model as fsm
import run_eval as em
import strategy_stats as ss
from hcs_package.reference_path import ReferencePath
from fit_anchor_pooled import PID

PROC = HERE.parents[1] / "human_data" / "processed_gaze_events"
RESULTS = HERE / "results" / "pooled10"


def eval_participant(pid_short, cfg):
    rounds, t2c, t2b = fsm.load_participant(PID[pid_short])
    tasks = fsm.build_tunnel_tasks(t2c, t2b)
    tun_train, tun_test = fsm.split_tunnel(rounds, t2c, t2b)
    tun_all = {**{t: r for t, r in tun_train.items() if t2b[t] == "steering"},
               **{t: r for t, r in tun_test.items() if t2b[t] == "steering"}}
    pt_train, pt_test = fsm.split_pointing(rounds, t2c, t2b)
    pt_all = {**pt_train, **pt_test}
    fsm.compute_tunnel_scales({t: tun_all[t] for t in tun_all if t in tun_train}, tasks)
    fsm.compute_pointing_scales(pt_train if pt_train else pt_all)

    sim = fsm._make_sim(cfg)
    rows, leads, cycles, trig = [], [], [], {"arrival+latency": 0, "deviation": 0, "exhausted": 0}
    tl_train, tl_test = [], []
    strat = []      # corner strategy: (cut_model, cut_human, dip_model, dip_human)
    for tid in sorted(tun_all):
        tc, cl, hw = tasks[tid]
        ct_h = float(np.mean([(h["timestamps"][-1] - h["timestamps"][0]) / 1000.0 for h in tun_all[tid]]))
        tc = dict(tc); tc["max_steps"] = int(min(fsm.MAX_SIM_STEPS, max(60, 2.5 * ct_h / 0.05)))
        try:
            traj, spd, dt = fsm.run_single_sim(sim, tc)
        except Exception:
            rows.append((tid, t2c[tid].get("tunnelType"), t2c[tid]["tunnelWidth"], np.nan, ct_h, 0.0, np.nan)); continue
        comp = fsm._completion(traj, cl) if len(traj) >= 5 else 0.0
        ct_m = len(traj) * dt
        L = float(np.sum(np.linalg.norm(np.diff(np.asarray(cl, float), axis=0), axis=1)))
        rows.append((tid, t2c[tid].get("tunnelType"), t2c[tid]["tunnelWidth"], ct_m, ct_h, comp,
                     L / max(t2c[tid]["tunnelWidth"], 1e-6)))
        if comp >= 0.95:
            loss = float(np.mean([fsm.tunnel_loss(fsm.tunnel_metrics(traj, spd, h, cl, dt, hw))
                                  for h in tun_all[tid]]))
        else:
            loss = fsm.INCOMPLETE_PENALTY * (1.0 - min(comp, 1.0))
        (tl_train if tid in tun_train else tl_test).append(loss)
        # corner strategy: cut depth / apex dip, model vs human (eval-main style)
        if t2c[tid].get("tunnelType") == "corner" and comp >= 0.95:
            try:
                sp = ReferencePath(fsm._waypoints_m(tc), s=0.0, k=3)
                ks = np.array([abs(sp.curvature(float(x))) for x in np.linspace(0, sp.total_length, 1200)])
                thr = float(np.percentile(ks, 85))
                if thr >= 0.5:
                    m_stat = ss.stats(traj, spd, sp, thr)
                    h_stat = np.nanmean([ss.stats(h["trajectory"], h["speeds"], sp, thr)
                                          for h in tun_all[tid]], axis=0)
                    strat.append((t2c[tid]["tunnelWidth"], m_stat[0], h_stat[0],
                                  m_stat[2] if len(m_stat) > 2 else np.nan,
                                  h_stat[2] if len(h_stat) > 2 else np.nan))
            except Exception:
                pass
        di = sim.last_diagnostics or {}
        evs = di.get("replan_events", [])
        w = t2c[tid]["tunnelWidth"]
        for i, e in enumerate(evs):
            if e["trigger"] in trig:
                trig[e["trigger"]] += 1
            lead = e["anchor"] - e["theta"]
            if np.isfinite(lead) and lead > 0:
                leads.append((w, lead))
            if i + 1 < len(evs):
                cycles.append(evs[i + 1]["t"] - e["t"])
    tun = pd.DataFrame(rows, columns=["tid", "type", "W", "ct_m", "ct_h", "comp", "id_lw"])
    tun["ctr"] = tun["ct_m"] / tun["ct_h"]
    # steering law MT = a + b * (L/W): slope for model and human over completed trials
    ok = tun.dropna(subset=["ct_m", "id_lw"])
    b_model = b_human = None
    if len(ok) >= 6 and np.ptp(ok["id_lw"]) > 1:
        b_model = float(np.polyfit(ok["id_lw"], ok["ct_m"], 1)[0] * 100)   # s per 100 ID
        b_human = float(np.polyfit(ok["id_lw"], ok["ct_h"], 1)[0] * 100)

    # pointing
    mts, hmts, ids = [], [], []
    for tid in sorted(pt_all):
        for hp_ in fsm._human_pointing_profiles(pt_all[tid]):
            try:
                tc, _, _ = em.build_fitts_bypass_config(hp_["round"], hp_["R"], max_steps=90)
                traj, spd, dt = fsm.run_single_sim(sim, tc, target_radius=hp_["R"])
            except Exception:
                continue
            if len(traj) < 5 or len(traj) >= 90:
                continue
            mts.append(len(traj) * dt)
            hmts.append(float(hp_.get("mt_kin", np.nan)))
            D = float(np.hypot(hp_["center"][0] - hp_["start"][0], hp_["center"][1] - hp_["start"][1]))
            ids.append(np.log2(D / hp_["R"] + 1))

    # human leads (canonical, kept)
    clh = pd.read_csv(PROC / f"{pid_short}_fixation_events_clean.csv")
    clh = clh[clh["keep"] & (clh["lead_corr"] > 0.003)]
    h_leads = clh.groupby(np.round(clh["width"] * 1000))["lead_corr"].median() * 1000

    m_leads = pd.DataFrame(leads, columns=["W", "lead"])
    m_by_w = m_leads.groupby(np.round(m_leads["W"] * 1000))["lead"].median() * 1000
    n_tr = sum(trig.values())
    out = {
        "tunnel_loss_train": float(np.mean(tl_train)) if tl_train else None,
        "tunnel_loss_test": float(np.mean(tl_test)) if tl_test else None,
        "completion_rate": float((tun["comp"] >= 0.95).mean()),
        "ct_ratio_median": float(np.nanmedian(tun["ctr"])),
        "ct_ratio_by_width": {int(k * 1000): round(float(v), 2) for k, v in
                               tun.groupby(np.round(tun["W"] * 1000))["ctr"].median().items()},
        "ct_ratio_by_type": {str(k)[:14]: round(float(v), 2) for k, v in
                              tun.groupby("type")["ctr"].median().items()},
        "gaze_cycle_median_s": float(np.median(cycles)) if cycles else None,
        "trigger_shares": {k: round(v / max(n_tr, 1), 2) for k, v in trig.items()},
        "model_lead_by_width_mm": {int(k): round(float(v), 1) for k, v in m_by_w.items()},
        "human_lead_by_width_mm": {int(k): round(float(v), 1) for k, v in h_leads.items()},
        "steering_b_model_s_per100id": b_model,
        "steering_b_human_s_per100id": b_human,
        "corner_strategy": [{"W_mm": int(w * 1000), "cut_model_mm": round(cm, 1),
                              "cut_human_mm": round(chh, 1), "dip_model": round(dm, 2),
                              "dip_human": round(dh, 2)}
                             for w, cm, chh, dm, dh in sorted(strat)],
        "pointing_mt_mean_s": float(np.mean(mts)) if mts else None,
        "pointing_mt_ratio": (float(np.nanmedian(np.array(mts) / np.array(hmts)))
                               if mts and np.isfinite(hmts).any() else None),
        "fitts_b_model": (float(np.polyfit(ids, mts, 1)[0])
                           if len(mts) >= 4 and np.ptp(ids) > 0.5 else None),
        "n_pointing": len(mts),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(RESULTS / "pooled10_anchor_config_s42.json"))
    ap.add_argument("--pool", nargs="*", default=["p04", "p06", "p07", "p09", "p10"])
    ap.add_argument("--noise", choices=["on", "off"], default="on")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    cfg = copy.deepcopy(cfg)
    cfg["add_noise"] = a.noise == "on"
    res = {}
    for p in a.pool:
        res[p] = eval_participant(p, cfg)
        r = res[p]
        print(f"{p}: tunnel {r['tunnel_loss_train']:.2f}/{r['tunnel_loss_test']:.2f} "
              f"complete {r['completion_rate']:.0%} CTr {r['ct_ratio_median']:.2f} "
              f"| cycle {r['gaze_cycle_median_s'] or float('nan'):.2f}s trig {r['trigger_shares']} "
              f"| model leads {r['model_lead_by_width_mm']} | human {r['human_lead_by_width_mm']} "
              f"| pointing MT {r['pointing_mt_mean_s']} (n={r['n_pointing']})", flush=True)
        print(f"    CT by width {r['ct_ratio_by_width']} | by type {r['ct_ratio_by_type']}", flush=True)
        print(f"    steering b (s/100 ID): model {r['steering_b_model_s_per100id']} vs human "
              f"{r['steering_b_human_s_per100id']} | corners: {r['corner_strategy']}", flush=True)
    json.dump(res, open(RESULTS / "pooled10_eval.json", "w"), indent=2, default=float)
    print("saved results/pooled10/pooled10_eval.json"); print("DONE")


if __name__ == "__main__":
    main()

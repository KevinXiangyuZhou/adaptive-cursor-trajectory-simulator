"""Smoothness diagnostic: human vs persona (noise-on and noise-off) on the corner tunnels —
trajectories, signed lateral offset vs progress, speed vs time with fixation-replan markers,
plus high-frequency lateral RMS, lateral sign-change rate and peak acceleration.

Usage: python jiggle_diag.py --pid P170114 --persona results/P170114_anchor_persona_S12.json --out results/jiggle_diag_S12_B.png
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import probe_anchor as pa, fit_speed_model as fsm
from hcs_package.reference_path import ReferencePath


def profile(traj, sp):
    traj = np.asarray(traj, float); s = np.array([sp.find_closest_theta(p) for p in traj])
    o = np.array([(p - sp(th)) @ np.array([sp.tangent(th)[1], -sp.tangent(th)[0]]) for p, th in zip(traj, s)])
    return s, o


def wiggle(o, k=5):
    ma = np.convolve(o, np.ones(k) / k, mode='same'); hf = o - ma; d = np.diff(o)
    return float(np.sqrt(np.mean(hf[k:-k] ** 2))) * 1000, int(np.sum(np.sign(d[1:]) * np.sign(d[:-1]) < 0))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pid", default="P170114"); ap.add_argument("--persona", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--conds", nargs="*", default=["corner:0.04", "corner:0.02"])
    a = ap.parse_args()
    rounds, t2c, t2b = fsm.load_participant(a.pid); tasks = fsm.build_tunnel_tasks(t2c, t2b)
    persona = json.load(open(a.persona)); persona.pop('_description', None)
    fig, axes = plt.subplots(len(a.conds), 3, figsize=(20, 4.5 * len(a.conds)))
    axes = np.atleast_2d(axes)
    for row, cond in enumerate(a.conds):
        ty, w = cond.split(":"); w = float(w)
        tid = next((t for t in tasks if t2b[t] == 'steering' and abs(t2c[t]['tunnelWidth'] - w) < 1e-6 and (t2c[t].get('tunnelType') or 'sinusoidal') == ty), None)
        if tid is None or tid not in rounds:
            continue
        tc, cl, hw = tasks[tid]; sp = ReferencePath(fsm._waypoints_m(tc), s=0.0, k=3)
        ax0, ax1, ax2 = axes[row]
        for h in rounds[tid]:
            tr = np.asarray(h['trajectory']); s, o = profile(tr, sp); ts = (np.asarray(h['timestamps']) - h['timestamps'][0]) / 1000; v = np.asarray(h['speeds'])
            hfr, sc = wiggle(o); ax0.plot(tr[:, 0], tr[:, 1], '-', color='tab:blue', lw=1, alpha=.8); ax1.plot(s * 1000, o * 1000, '-', color='tab:blue', lw=1, alpha=.8); ax2.plot(ts, v, '-', color='tab:blue', lw=1, alpha=.8)
            print(f"{ty}{w*1000:.0f} human: hf-lateral RMS {hfr:.2f}mm sign-changes {sc/(sp.total_length*1000)*100:.1f}/100mm")
        for name, noise, col in (('noise-on', True, 'tab:orange'), ('noise-off', False, 'tab:green')):
            c = json.loads(json.dumps(persona)); c['add_noise'] = noise; c['random_seed'] = 1000
            if not noise: c['replan_latency_cv'] = 0.0
            traj, spd, dt, diag, ref = pa._sim_with_diag(c, tc); tr = np.asarray(traj); s, o = profile(tr, sp); t = np.arange(len(tr)) * dt; spd = np.asarray(spd)
            hfr, sc = wiggle(o); ev = diag['replan_events']; g = lambda e, k: e[k] if isinstance(e, dict) else getattr(e, k)
            fx = [g(e, 't') for e in ev if g(e, 'trigger') != 'motor']
            ax0.plot(tr[:, 0], tr[:, 1], '-', color=col, lw=1); ax1.plot(s * 1000, o * 1000, '-', color=col, lw=1); ax2.plot(t, spd, '-', color=col, lw=1)
            for tf in fx: ax2.axvline(tf, color=col, alpha=.15, lw=.8)
            acc = np.linalg.norm(np.diff(tr, 2, axis=0), axis=1) / dt ** 2 if len(tr) > 3 else np.array([0.0])
            print(f"{ty}{w*1000:.0f} model {name}: hf-lateral RMS {hfr:.2f}mm sign-changes {sc/(sp.total_length*1000)*100:.1f}/100mm max|a| {acc.max():.1f} p95|a| {np.percentile(acc,95):.1f} fixations {len(fx)} aborted {diag.get('aborted_breach')}")
        ref_pts = np.array([ref(float(x)) for x in np.linspace(0, ref.total_length, 600)]); sr, orf = profile(ref_pts, sp)
        ax0.plot(ref_pts[:, 0], ref_pts[:, 1], '--', color='0.4', lw=.8); ax1.plot(sr * 1000, orf * 1000, '--', color='0.4', lw=.8)
        ax0.set_aspect('equal'); ax0.set_title(f'{a.pid} {ty} W={w*1000:.0f}: blue human, orange noise-on, green noise-off, grey route')
        ax1.set_title('signed lateral offset vs progress (mm)'); ax1.axhline(hw * 1000, color='k', lw=.5); ax1.axhline(-hw * 1000, color='k', lw=.5)
        ax2.set_title('speed vs time (s); lines = fixation replans')
    fig.tight_layout(); fig.savefig(a.out, dpi=130); print('saved', a.out)


if __name__ == "__main__":
    main()

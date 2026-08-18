import sys,json,tempfile,warnings,numpy as np
warnings.filterwarnings("ignore")
ROOT='/Users/xiangyz/Desktop/adaptive-cursor-trajectory-simulator'
sys.path.insert(0,ROOT+'/hcs_package/src'); sys.path.insert(0,ROOT); sys.path.insert(0,ROOT+'/eval/eval-main')
from run_eval import _build_record, pointing_target_center, align_round, _compute_speeds
from experiment.environment import POINTING_Y_OFFSET
from hcs_package.cursor_simulator import CursorSimulator
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sim=CursorSimulator()
hd=json.load(open(ROOT+'/human_data/aug-26-prolific/participant_P6a0aa037f7816b7befeb15e6.json'))
td=hd['sessions'][0]['trialData']
TIDS=[57,62,66,71,74,83]
fig,axes=plt.subplots(2,len(TIDS),figsize=(4*len(TIDS),6))
print(f"{'tid':>4} {'W':>5} {'D':>4} {'R':>3} | human CT (2 rounds) | model CT | model timed_out | maxlat in tunnel(mm)")
for j,tid in enumerate(TIDS):
    rounds=[x for x in td if x['trial_id']==tid]
    hcts=[]; mcts=[]; tos=[]; lat=[]
    for x in rounds:
        c=x['condition']; traj=[[p['x'],p['y']] for p in x['trajectory']]; s=traj[0]; R=c['targetRadius']; W=c['segment1Width']
        tx=c['transition_point']['taskX']; ctr=[s[0]+c['distance'], s[1]+POINTING_Y_OFFSET[c['targetPosition']]]
        # waypoints (px in a 460x260 window ~ 1 px = 1 mm) : start -> transition -> target ; widths: W on segment 1, bypass after
        seg1=[[s[0]+ (tx-s[0])*k/8, s[1]] for k in range(9)]
        seg2=[[tx+(ctr[0]-tx)*k/8, s[1]+(ctr[1]-s[1])*k/8] for k in range(1,9)]
        pts=seg1+seg2; widths=[W]*len(seg1)+[10.0]*len(seg2)
        wp_px=[[p[0]/0.46*460, p[1]/0.26*260] for p in pts]
        task={"waypoints":wp_px,"screen_width":460,"screen_height":260,"target_radius":R,"max_steps":800,
              "constraints":{"coordinate_system":"normalized","default_margin":0.0,"regions":[{"constraint_type":"keep_in","geometry":{"type":"path","path":pts,"width":widths},"enabled":True}]}}
        tf=tempfile.NamedTemporaryFile('w',suffix='.json',delete=False); json.dump(task,tf); tf.close()
        tr=sim.generate_trajectory_with_waypoints(task_file=tf.name,target_radius=R,max_steps=800); rec=_build_record(tr,sim.interval,R,800)
        t=np.array(rec['trajectory']); hcts.append(x['completionTime']); mcts.append(rec['completion_time']); tos.append(rec['timed_out'])
        intun=t[t[:,0]<=tx]; lat.append(np.abs(intun[:,1]-s[1]).max()*1000 if len(intun) else 0)
        ax=axes[0][j]; ax.plot(np.array(traj)[:,0],np.array(traj)[:,1],'b-',alpha=.7); ax.plot(t[:,0],t[:,1],color='tab:orange',alpha=.8)
        ax.fill_between([s[0],tx],s[1]-W/2,s[1]+W/2,color='lightgray'); ax.add_patch(plt.Circle(ctr,R,color='salmon',alpha=.4)); ax.set_aspect('equal'); ax.set_title(f"t{tid} W={W} R={R*1000:.0f}mm",fontsize=9)
        ax2=axes[1][j]; ts=(np.array(x['timestamps'])-x['timestamps'][0])/1000; ax2.plot(ts,_compute_speeds(traj,x['timestamps']),'b-',alpha=.7); ax2.plot(np.arange(len(t))*0.05,np.convolve(rec['speeds'],np.ones(5)/5,'same'),color='tab:orange',alpha=.8); ax2.set_xlabel('t (s)'); ax2.set_ylabel('speed')
    print(f"{tid:>4} {W:5} {c['distance']:4.2f} {R*1000:3.0f} | {hcts[0]:.2f} {hcts[1]:.2f} | {mcts[0]:.2f} {mcts[1]:.2f} | {tos} | {lat[0]:.1f} {lat[1]:.1f}")
axes[0][0].legend(['human','model'],fontsize=8); fig.suptitle('constrained→unconstrained (tunnel W then free flight to target) — human vs model, P6a0aa0'); fig.tight_layout()
fig.savefig(ROOT+'/eval/eval-main/results/capability/c2u_probe.png',dpi=110); print('fig saved')

import sys,json,tempfile,warnings,numpy as np; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT='/Users/xiangyz/Desktop/adaptive-cursor-trajectory-simulator'
sys.path.insert(0,ROOT+'/hcs_package/src'); sys.path.insert(0,ROOT); sys.path.insert(0,ROOT+'/eval/eval-main')
from run_eval import build_fitts_bypass_config,_build_record,_compute_speeds
from hcs_package.cursor_simulator import CursorSimulator
def mksim(over):
    cfg=json.load(open(ROOT+'/hcs_package/src/hcs_package/user_configurations/office_worker.json')); cfg['speed_model']['path']=ROOT+'/hcs_package/src/hcs_package/user_configurations/population_gam.pkl'
    cfg['planner_weights'].update(over.pop('planner_weights',{})); cfg.update(over)
    tf=tempfile.NamedTemporaryFile('w',suffix='.json',delete=False); json.dump(cfg,tf); tf.close(); return CursorSimulator(tf.name)
sims={'MPCC office_worker (free-space LQR, well 1e-4)':mksim({}),'MPCC same, well off':mksim({'planner_weights':{'goal_precision':0.0}})}
hd=json.load(open(ROOT+'/human_data/aug-26-prolific/participant_P6a0aa037f7816b7befeb15e6.json'))
TIDS=[46,52,55]; fig,axes=plt.subplots(2,3,figsize=(15,7))
for j,tid in enumerate(TIDS):
    rounds=[x for x in hd['sessions'][0]['trialData'] if x['trial_id']==tid]
    axs,axp=axes[0][j],axes[1][j]
    for i,x in enumerate(rounds):
        traj=[[p['x'],p['y']] for p in x['trajectory']]; ts=x['timestamps']; R=x['condition']['targetRadius']
        t=(np.array(ts)-ts[0])/1000; sp=_compute_speeds(traj,ts)
        from run_eval import pointing_target_center
        s,e=np.array(traj[0]),np.array(pointing_target_center({'trajectory':traj,'condition':x['condition']})); D=np.linalg.norm(e-s); u=(e-s)/D
        axs.plot(t,sp,color='tab:blue',alpha=0.8,label='Human' if i==0 else None)
        axp.plot(t,(np.array(traj)-s)@u/D,color='tab:blue',alpha=0.8,label='Human' if i==0 else None)
        for (name,sim),col in zip(sims.items(),['tab:orange','tab:green']):
            tc,_,_=build_fitts_bypass_config({"trajectory":traj},R); t2=tempfile.NamedTemporaryFile('w',suffix='.json',delete=False); json.dump(tc,t2); t2.close()
            tr=sim.generate_trajectory_with_waypoints(task_file=t2.name,target_radius=R,max_steps=800); rec=_build_record(tr,sim.interval,R,800)
            mt=np.arange(len(rec['trajectory']))*sim.interval; msp=_compute_speeds(rec['trajectory'],list(mt*1000))
            axs.plot(mt,msp,color=col,alpha=0.8,label=name if i==0 else None)
            axp.plot(mt,(np.array(rec['trajectory'])-s)@u/D,color=col,alpha=0.8,label=name if i==0 else None)
    axs.set_title(f"t{tid}: {x['condition']['description']}",fontsize=9); axs.set_ylabel('speed (m/s), 5-pt smoothed'); axs.grid(alpha=.3)
    axp.axhline(1,color='k',lw=.5); axp.axhspan(1-R/D,1+R/D,color='salmon',alpha=.3); axp.set_ylabel('progress along start→target (1=target)'); axp.set_xlabel('time (s)'); axp.grid(alpha=.3)
axes[0][0].legend(fontsize=8); fig.suptitle('Pointing: human vs MPCC with free-space LQR objective (q=1, r=0.08, jerk=2e-6; 2 rounds each, noise on) — P6a0aa0',fontsize=10); fig.tight_layout()
fig.savefig(sys.argv[1],dpi=110); print('saved')

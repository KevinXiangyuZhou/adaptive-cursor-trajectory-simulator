"""Fittability test for the R-dependence of pointing MT.
Runs the model on R=5 / R=20 / R=25 mm conditions at 3 distances (tids 42,47,52 /
45,50,55 / 46,51,56) for a grid of fittable weights and reports the ALIGNED
MT_kin ratio (model/human population mean) per radius group and the fitted
Fitts slope on MT_kin. Usage: python pointing_sweep_kin.py out.json [configs.json]"""
import sys, json, tempfile, warnings, csv, collections
import numpy as np
warnings.filterwarnings("ignore")
ROOT='/Users/xiangyz/Desktop/adaptive-cursor-trajectory-simulator'
sys.path.insert(0,ROOT+'/hcs_package/src'); sys.path.insert(0,ROOT); sys.path.insert(0,ROOT+'/eval/eval-main')
OUT=sys.argv[1]
TIDS=[42,47,52,45,50,55,46,51,56]
RGROUP={42:5,47:5,52:5,45:20,50:20,55:20,46:25,51:25,56:25}
PID='P6a0aa037f7816b7befeb15e6'
BASE=ROOT+'/hcs_package/src/hcs_package/user_configurations/office_worker.json'
GAM=ROOT+'/hcs_package/src/hcs_package/user_configurations/population_gam.pkl'

def human_targets():
    rows=list(csv.DictReader(open(ROOT+'/eval/eval-main/results/Fitts/fitts_results.csv')))
    by=collections.defaultdict(list); ids={}
    for r in rows:
        if r['source']=='Human': by[int(r['tid'])].append(float(r['MT_kin_s'])); ids[int(r['tid'])]=float(r['ID'])
    return {t:float(np.mean(v)) for t,v in by.items()}, ids

def load_rounds():
    hd=json.load(open(f'{ROOT}/human_data/aug-26-prolific/participant_{PID}.json'))
    td=hd['sessions'][0]['trialData']
    return {tid:[x for x in td if x['trial_id']==tid] for tid in TIDS}

def run_config(args):
    name, over = args
    from run_eval import build_fitts_bypass_config, _build_record, pointing_target_center, align_round
    from hcs_package.cursor_simulator import CursorSimulator
    cfg=json.load(open(BASE)); cfg['speed_model']['path']=GAM
    cfg['planner_weights'].update(over.get('planner_weights',{}))
    for k,v in over.items():
        if k!='planner_weights': cfg[k]=v
    tf=tempfile.NamedTemporaryFile('w',suffix='.json',delete=False); json.dump(cfg,tf); tf.close()
    sim=CursorSimulator(tf.name); rounds=load_rounds(); res=[]
    for tid in TIDS:
        for x in rounds[tid]:
            traj=[[p['x'],p['y']] for p in x['trajectory']]; R=x['condition']['targetRadius']
            hr={"trajectory":traj,"condition":x['condition']}
            tc,cl,w=build_fitts_bypass_config(hr,R)
            t2=tempfile.NamedTemporaryFile('w',suffix='.json',delete=False); json.dump(tc,t2); t2.close()
            tr=sim.generate_trajectory_with_waypoints(task_file=t2.name,target_radius=R,max_steps=800)
            rec=_build_record(tr,sim.interval,R,800)
            ctr=pointing_target_center(hr); ts=[i*sim.interval for i in range(len(rec['trajectory']))]
            al=align_round(rec['trajectory'],ts,ctr,R)
            sp=np.convolve(rec['speeds'],np.ones(5)/5,'same')
            res.append(dict(tid=tid,R=R,mt_kin=al['mt_kin_s'],ct=rec['completion_time'],peak=float(sp.max()),timed_out=rec['timed_out']))
    return name, over, res

CONFIGS=[]
def add(name,**over): CONFIGS.append((name,over))
if len(sys.argv)>2:
    CONFIGS=[tuple(x) for x in json.load(open(sys.argv[2]))]
else:
    add('default (gp=1.5e-4, q/r=25, jerk=5e-6)')
    for gp in [0.0,5e-5,5e-4]:
        add(f'gp={gp}',planner_weights={'goal_precision':gp})
    for r in [0.02,0.01]:
        add(f'q/r={1/r:.0f}',planner_weights={'free_velocity':r})
        add(f'q/r={1/r:.0f} gp=0',planner_weights={'free_velocity':r,'goal_precision':0.0})
    add('jerk=2e-6',planner_weights={'jerk':2e-6})
    add('jerk=2e-6 gp=0',planner_weights={'jerk':2e-6,'goal_precision':0.0})
    add('jerk=2e-6 q/r=50 gp=0',planner_weights={'jerk':2e-6,'free_velocity':0.02,'goal_precision':0.0})

if __name__=='__main__':
    import multiprocessing as mp
    H,IDS=human_targets()
    with mp.Pool(10) as pool: results=pool.map(run_config, CONFIGS)
    json.dump({'human_mt_kin':H,'ids':IDS,'results':results},open(OUT,'w'))
    print(f"{'config':40s} {'R5':>5} {'R20':>5} {'R25':>5} | {'slope':>5} {'icpt':>5} | {'peak':>4} {'TO':>2}   (human slope on same tids: {np.polyfit([IDS[t] for t in TIDS],[H[t] for t in TIDS],1)[0]:.2f})")
    for name,over,res in results:
        g=collections.defaultdict(list)
        for x in res: g[RGROUP[x['tid']]].append(x['mt_kin']/H[x['tid']])
        ids=[IDS[x['tid']] for x in res]; mts=[x['mt_kin'] for x in res]; b,a=np.polyfit(ids,mts,1)
        print(f"{name:40s} {np.mean(g[5]):5.2f} {np.mean(g[20]):5.2f} {np.mean(g[25]):5.2f} | {b:5.2f} {a:5.2f} | {np.mean([x['peak'] for x in res]):4.2f} {sum(x['timed_out'] for x in res):2d}")

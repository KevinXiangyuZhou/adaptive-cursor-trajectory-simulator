"""Coarse capability sweep: can the current MPCC design reproduce human pointing?
Varies jerk / progress / goal_precision / Th / speed-ceiling; measures CT vs
population-mean human CT, peak speed, path ratio, timeouts on 5 conditions."""
import sys, os, json, tempfile, itertools, warnings, csv, collections
import numpy as np
warnings.filterwarnings("ignore")
ROOT='/Users/xiangyz/Desktop/adaptive-cursor-trajectory-simulator'
sys.path.insert(0,ROOT+'/hcs_package/src'); sys.path.insert(0,ROOT); sys.path.insert(0,ROOT+'/eval/eval-main')
OUT=sys.argv[1]
TIDS=[43,46,50,52,55]
PID='P6a0aa037f7816b7befeb15e6'
BASE=ROOT+'/hcs_package/src/hcs_package/user_configurations/office_worker.json'
GAM=ROOT+'/hcs_package/src/hcs_package/user_configurations/population_gam.pkl'

def human_targets():
    rows=list(csv.DictReader(open(ROOT+'/eval/eval-main/results/Fitts/fitts_results.csv')))
    by=collections.defaultdict(list); pk=collections.defaultdict(list)
    for r in rows:
        if r['source']=='Human': by[int(r['tid'])].append(float(r['MT_s']))
    return {t:np.mean(v) for t,v in by.items()}

def load_rounds():
    hd=json.load(open(f'{ROOT}/human_data/aug-26-prolific/participant_{PID}.json'))
    td=hd['sessions'][0]['trialData']
    out={}
    for tid in TIDS:
        out[tid]=[([[p['x'],p['y']] for p in x['trajectory']], x['condition']['targetRadius'], x['completionTime']) for x in td if x['trial_id']==tid]
    return out

def run_config(args):
    name, over = args
    from run_eval import build_fitts_bypass_config, _build_record
    from hcs_package.cursor_simulator import CursorSimulator
    cfg=json.load(open(BASE)); cfg['speed_model']['path']=GAM
    pw=over.get('planner_weights',{}); cfg['planner_weights'].update(pw)
    for k,v in over.items():
        if k not in ('planner_weights','ceil'): cfg[k]=v
    tf=tempfile.NamedTemporaryFile('w',suffix='.json',delete=False); json.dump(cfg,tf); tf.close()
    sim=CursorSimulator(tf.name)
    if 'ceil' in over: sim.speed_model.ceil=over['ceil']
    rounds=load_rounds(); res=[]
    for tid in TIDS:
        for traj,R,hct in rounds[tid]:
            tc,cl,w=build_fitts_bypass_config({"trajectory":traj},R)
            t2=tempfile.NamedTemporaryFile('w',suffix='.json',delete=False); json.dump(tc,t2); t2.close()
            tr=sim.generate_trajectory_with_waypoints(task_file=t2.name,target_radius=R,max_steps=800)
            rec=_build_record(tr,sim.interval,R,800)
            t=np.array(rec['trajectory']); s,e=np.array(traj[0]),np.array(traj[-1]); D=np.linalg.norm(e-s)
            n=np.array([-(e-s)[1],(e-s)[0]])/D
            # bell-shapedness: fraction of time at >50% peak (human ~0.35), n speed peaks
            sp=np.array(rec['speeds'])
            sm=np.convolve(sp,np.ones(5)/5,mode='same')
            de=np.linalg.norm(t-e,axis=1); outside=np.where(de>2*R)[0]; t2R=(len(t)-outside[-1]-1)*0.05 if len(outside) else 0.0
            res.append(dict(tid=tid,R=R,D=D,hct=hct,mct=rec['completion_time'],peak=float(sm.max()),t2R=t2R,
                            path_ratio=rec['path_length']/D,maxlat=float(np.abs((t-s)@n).max()),
                            timed_out=rec['timed_out'],t_peak_frac=float(np.argmax(sp)/len(sp))))
    return name, over, res

CONFIGS=[]
def add(name,**over): CONFIGS.append((name,over))
add('baseline')
for j in [1e-6,2e-7]:
    add(f'jerk={j}',planner_weights={'jerk':j})
for p in [3e-6,3e-5]:
    add(f'progress={p}',planner_weights={'progress':p})
for j,p in itertools.product([1e-6,2e-7],[3e-6,3e-5]):
    add(f'jerk={j},progress={p}',planner_weights={'jerk':j,'progress':p})
for g in [3e-5,0.0]:
    add(f'gp={g}',planner_weights={'goal_precision':g})
    add(f'jerk=2e-7,progress=3e-5,gp={g}',planner_weights={'jerk':2e-7,'progress':3e-5,'goal_precision':g})
add('Th=0.6',Th=0.6)
add('jerk=2e-7,progress=3e-5,Th=0.6',planner_weights={'jerk':2e-7,'progress':3e-5},Th=0.6)
add('ceil=1.0',ceil=1.0)
add('jerk=2e-7,progress=3e-5,ceil=1.0',planner_weights={'jerk':2e-7,'progress':3e-5},ceil=1.0)
add('jerk=2e-7,progress=3e-5,ceil=1.0,Th=0.6',planner_weights={'jerk':2e-7,'progress':3e-5},ceil=1.0,Th=0.6)
add('jerk=2e-7,progress=3e-5,ceil=1.0,gp=3e-5',planner_weights={'jerk':2e-7,'progress':3e-5,'goal_precision':3e-5},ceil=1.0)

if len(sys.argv) > 2:   # optional: JSON list of [name, overrides] replaces the built-in grid
    CONFIGS = [tuple(x) for x in json.load(open(sys.argv[2]))]

if __name__=='__main__':
    import multiprocessing as mp
    H=human_targets()
    with mp.Pool(10) as pool:
        results=pool.map(run_config, CONFIGS)
    json.dump({'human_pop_ct':H,'results':results},open(OUT,'w'))
    print(f"{'config':45s} {'CTratio':>7} {'|logr|':>6} {'peak':>5} {'pathR':>5} {'lat':>6} {'tpk':>4} {'t2R':>5} {'TO':>2}")
    for name,over,res in results:
        r=np.array([x['mct']/H[x['tid']] for x in res]); 
        print(f"{name:45s} {r.mean():7.2f} {np.abs(np.log(r)).mean():6.2f} {np.mean([x['peak'] for x in res]):5.2f} {np.mean([x['path_ratio'] for x in res]):5.2f} {np.mean([x['maxlat'] for x in res]):6.3f} {np.mean([x['t_peak_frac'] for x in res]):4.2f} {np.mean([x['t2R'] for x in res]):5.2f} {sum(x['timed_out'] for x in res):2d}")

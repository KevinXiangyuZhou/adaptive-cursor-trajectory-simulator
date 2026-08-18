"""Design iteration for the goal-precision term.
For each config: all 15 pointing conditions x 2 rounds (P6a0aa0 geometry, noise on).
Reports, per target radius, aligned MT_kin ratio vs population human, endpoint
depth (|end - centre| / R, median), re-entries into the target, plus Fitts
slope, peak speed and timeouts. Usage: python pointing_sweep_precision.py out.json [configs.json]"""
import sys, json, tempfile, warnings, csv, collections
import numpy as np
warnings.filterwarnings("ignore")
ROOT='/Users/xiangyz/Desktop/adaptive-cursor-trajectory-simulator'
sys.path.insert(0,ROOT+'/hcs_package/src'); sys.path.insert(0,ROOT); sys.path.insert(0,ROOT+'/eval/eval-main')
OUT=sys.argv[1]
TIDS=list(range(42,57))
PIDS=['P6a0aa037f7816b7befeb15e6','P698082f6dfbf6cd0a3d6584b','P69f1b200d44045f2d287e0ad']
BASE=ROOT+'/hcs_package/src/hcs_package/user_configurations/office_worker.json'
GAM=ROOT+'/hcs_package/src/hcs_package/user_configurations/population_gam.pkl'
BASE_OVER={'planner_weights':{}}   # persona defaults unless overridden per config

def human_targets():
    rows=list(csv.DictReader(open(ROOT+'/eval/eval-main/results/Fitts/fitts_results.csv')))
    by=collections.defaultdict(list); ids={}
    for r in rows:
        if r['source']=='Human': by[int(r['tid'])].append(float(r['MT_kin_s'])); ids[int(r['tid'])]=float(r['ID'])
    return {t:float(np.mean(v)) for t,v in by.items()}, ids

def load_rounds():
    out={tid:[] for tid in TIDS}
    for pid in PIDS:
        td=json.load(open(f'{ROOT}/human_data/aug-26-prolific/participant_{pid}.json'))['sessions'][0]['trialData']
        for tid in TIDS: out[tid]+= [x for x in td if x['trial_id']==tid]
    return out

def run_config(args):
    name, over = args
    from run_eval import build_fitts_bypass_config, _build_record, pointing_target_center, align_round
    from hcs_package.cursor_simulator import CursorSimulator
    cfg=json.load(open(BASE)); cfg['speed_model']['path']=GAM
    cfg['planner_weights'].update(BASE_OVER['planner_weights']); cfg['planner_weights'].update(over.get('planner_weights',{}))
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
            ctr=np.array(pointing_target_center(hr)); t=np.array(rec['trajectory']); ts=[i*sim.interval for i in range(len(t))]
            al=align_round(rec['trajectory'],ts,ctr,R)
            d=np.linalg.norm(t-ctr,axis=1); inside=d<=R; entries=int(np.sum((~inside[:-1])&inside[1:]))
            sp=np.convolve(rec['speeds'],np.ones(5)/5,'same')
            n_kin=max(int(round(al['final_entry_s']/sim.interval)),1)
            res.append(dict(tid=tid,R=R,mt_kin=al['mt_kin_s'],ct=rec['completion_time'],peak=float(sp.max()),tpk=float(np.argmax(sp[:n_kin+1])/n_kin),
                            end_over_R=float(d[-1]/R),entries=entries,timed_out=rec['timed_out'],
                            path_ratio=rec['path_length']/np.linalg.norm(ctr-t[0])))
    return name, over, res

CONFIGS=[]
def add(name,**over): CONFIGS.append((name,over))
if len(sys.argv)>2:
    CONFIGS=[tuple(x) for x in json.load(open(sys.argv[2]))]
else:
    add('persona default (LQR r=.08 w=2e-6 well 1e-4)')
    add('well off',planner_weights={'goal_precision':0.0})
    add('well 3e-4',planner_weights={'goal_precision':3e-4})
    add('r=.04',planner_weights={'free_velocity':0.04})
    add('r=.16',planner_weights={'free_velocity':0.16})
    add('jerk 8e-6',planner_weights={'jerk':8e-6})

if __name__=='__main__':
    import multiprocessing as mp
    H,IDS=human_targets()
    with mp.Pool(min(10,len(CONFIGS))) as pool: results=pool.map(run_config, CONFIGS)
    json.dump({'human_mt_kin':H,'ids':IDS,'results':results},open(OUT,'w'))
    HEND={5:0.65,10:0.39,15:0.42,20:0.31,25:0.28}   # human endpoint/R medians (population)
    Rs=[5,10,15,20,25]
    print("MT_kin ratio by R (human=1) | endpoint/R median by R (human: "+" ".join(f"{HEND[r]:.2f}" for r in Rs)+") | slope icpt | peak tpk entries pathR TO")
    print(f"{'config':30s} "+" ".join(f"{'R'+str(r):>5}" for r in Rs)+" | "+" ".join(f"{'R'+str(r):>5}" for r in Rs)+" |")
    for name,over,res in results:
        g=collections.defaultdict(list); e=collections.defaultdict(list)
        for x in res: g[round(x['R']*1000)].append(x['mt_kin']/H[x['tid']]); e[round(x['R']*1000)].append(x['end_over_R'])
        ids=[IDS[x['tid']] for x in res]; mts=[x['mt_kin'] for x in res]; b,a=np.polyfit(ids,mts,1)
        mt_err=np.mean([abs(np.log(np.mean(g[r]))) for r in Rs]); end_err=np.mean([abs(np.median(e[r])-HEND[r]) for r in Rs])
        print(f"{name:30s} "+" ".join(f"{np.mean(g[r]):5.2f}" for r in Rs)+" | "+" ".join(f"{np.median(e[r]):5.2f}" for r in Rs)+f" | {b:5.2f} {a:5.2f} | {np.mean([x['peak'] for x in res]):4.2f} {np.mean([x['tpk'] for x in res]):4.2f} {np.mean([x['entries'] for x in res]):4.2f} {np.mean([x['path_ratio'] for x in res]):5.2f} {sum(x['timed_out'] for x in res):2d} | MTerr {mt_err:.3f} ENDerr {end_err:.3f}")
    print("(human slope 0.30, intercept -0.12; peak ~0.64; tpk 0.31 [R5 0.24 .. R25 0.38]; entries 1.09; pathR 1.05)")
    print("tpk by R:")
    for name,over,res in results:
        tt=collections.defaultdict(list); en=collections.defaultdict(list)
        for x in res: tt[round(x['R']*1000)].append(x['tpk']); en[round(x['R']*1000)].append(x['entries'])
        print(f"  {name:32s} tpk "+" ".join(f"{np.mean(tt[r]):.2f}" for r in Rs)+"   entries "+" ".join(f"{np.mean(en[r]):.2f}" for r in Rs))

import sys,json,tempfile,warnings,numpy as np,collections
warnings.filterwarnings("ignore")
ROOT='/Users/xiangyz/Desktop/adaptive-cursor-trajectory-simulator'
sys.path.insert(0,ROOT+'/hcs_package/src'); sys.path.insert(0,ROOT); sys.path.insert(0,ROOT+'/eval/eval-main')
from run_eval import scan_conditions, HUMAN_DATA_DIR, build_steering_task_config, load_trials_by_participant, run_tunnel_simulator
import stats as law_stats
OUT=sys.argv[1]; CONFIGS=json.load(open(sys.argv[2]))
PIDS=['P6a0aa037f7816b7befeb15e6','P698082f6dfbf6cd0a3d6584b','P69f1b200d44045f2d287e0ad','P5e84bd7c34d5d9072128fac1']
tid_to_condition, tid_to_bucket = scan_conditions(HUMAN_DATA_DIR)
STIDS=sorted(t for t,b in tid_to_bucket.items() if b=='steering')
hum=load_trials_by_participant(STIDS)
def job(args):
    name,over,pid,tid=args
    from hcs_package.cursor_simulator import CursorSimulator
    cfg=json.load(open(ROOT+'/hcs_package/src/hcs_package/user_configurations/office_worker.json')); cfg['speed_model']['path']=ROOT+'/hcs_package/src/hcs_package/user_configurations/population_gam.pkl'
    cfg['planner_weights'].update(over.get('planner_weights',{}))
    for k,v in over.items():
        if k!='planner_weights': cfg[k]=v
    tf=tempfile.NamedTemporaryFile('w',suffix='.json',delete=False); json.dump(cfg,tf); tf.close(); sim=CursorSimulator(tf.name)
    task,cl=build_steering_task_config(tid_to_condition[tid]); n=len(hum[pid][tid])
    recs=run_tunnel_simulator(sim,task,task['target_radius'],n)
    return name,pid,tid,[r['completion_time'] for r in recs],[r['timed_out'] for r in recs]
if __name__=='__main__':
    import multiprocessing as mp
    jobs=[(name,over,pid,tid) for name,over in CONFIGS for pid in PIDS for tid in STIDS]
    with mp.Pool(12) as pool: res=pool.map(job,jobs)
    json.dump(res,open(OUT,'w'))
    for name,_ in CONFIGS:
        rows_h=[];rows_m=[];rat=[];to=0
        for n_,pid,tid,cts,tos in res:
            if n_!=name: continue
            task,cl=build_steering_task_config(tid_to_condition[tid]); ID=law_stats.centerline_arc_length(cl)/tid_to_condition[tid]['tunnelWidth']
            h=[hum[pid][tid][k]['completion_time'] for k in hum[pid][tid]]
            rows_h+= [(ID,x) for x in h]; rows_m+=[(ID,x) for x in cts]; rat.append(np.mean(cts)/np.mean(h)); to+=sum(tos)
        bh=np.polyfit(*zip(*rows_h),1); bm=np.polyfit(*zip(*rows_m),1)
        print(f"{name:34s} CT ratio {np.mean(rat):.2f} | steering law model {bm[0]:.3f}*ID+{bm[1]:.2f}  human {bh[0]:.3f}*ID+{bh[1]:.2f} | timeouts {to}")

"""S2 vs S3 (S2 + gaze memory + consumed corners + enforced coast-safety + acc_max) probes."""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_anchor as pa
base = json.load(open(HERE / 'results' / 'stages' / 'S2' / 'P170114_anchor_config_S2_s42.json')); base.pop('_description', None)
S3 = {'anchor_memory': True, 'corner_consume': True, 'planner_weights': {'safety': 5e5, 'acc_max': 4.0}}
def run(label, extra, quick, buckets=("tunnel", "pointing"), workers=12):
    ov = json.loads(json.dumps(base)); ov.update({k: v for k, v in extra.items() if k != 'planner_weights'})
    if 'planner_weights' in extra: ov['planner_weights'] = {**base['planner_weights'], **extra['planner_weights']}
    print(f"\n##### {label} ({'quick' if quick else 'full'})", flush=True)
    return pa.run_probe('P170114', 'anchor', ov, quick=quick, n_workers=workers, buckets=buckets)
if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('quick', 'all'):
        run("S2 (as fitted)", {}, True)
        run("S3", S3, True)
        run("S3 acc_max 6", {**S3, 'planner_weights': {**S3['planner_weights'], 'acc_max': 6.0}}, True)
    if which in ('full', 'all'):
        r = run("S3", S3, False)
        json.dump({'tunnel': r['tunnel'], 'pointing': r.get('pointing')}, open(HERE / 'results' / 'probes' / 's3_full_summary.json', 'w'), default=float)

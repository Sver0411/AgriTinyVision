"""Replay an idealized upload barrier followed by a download over paired Mahimahi traces."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRACE_NAMES = ('ATT-LTE-driving', 'TMobile-LTE-driving', 'Verizon-EVDO-driving')
COMMIT = 'f1346c38c415abfc98626dd82450f49a57033603'
PHASES = (0, 1, 2, 3, 4)  # Eighths of min(last uplink timestamp, last downlink timestamp).
HORIZONS = (1, 10, 100, 1353)
CAPACITY = 1500


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_table(path):
    with path.open(encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))


def save_table(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def finish(trace,ready,volume,side):
    if volume==0: return ready
    if volume<0: raise ValueError('Negative payload')
    first=int(np.searchsorted(trace,ready,side=side)); index=first+(volume+CAPACITY-1)//CAPACITY-1
    if index>=len(trace): raise ValueError('Transfer extends beyond original recording')
    return int(trace[index])


def times(up,down,start,upload,download):
    barrier=finish(up,start,upload,'left')
    # Millisecond observations cannot resolve the relative order of opposite-direction slots.
    done=finish(down,barrier,download,'right' if upload else 'left')
    return barrier-start, done-barrier, done-start


def fixed_rate_check():
    t=np.arange(10,101,10,dtype=np.int64)
    tests=[(0,1500,1500,(10,10,20)),(0,3001,1501,(30,20,50)),(10,1500,0,(0,0,0)),(0,0,1500,(0,10,10))]
    for start,u,d,expected in tests:
        assert times(t,t,start,u,d)==expected
    assert finish(np.array([1,1,4]),1,3000,'left')==1
    try: finish(t,100,1501,'left')
    except ValueError: pass
    else: raise AssertionError('Out-of-trace transfer accepted')
    return dict(slot_bytes=CAPACITY,slot_every_ms=10,start_ms=0,upload_bytes=3001,download_bytes=1501,
                expected_upload_ms=30,expected_download_ms=20,observed_total_ms=times(t,t,0,3001,1501)[2])


def workloads(root):
    results=root/'results'; pairs=load_table(results/'edge_adaptation_metrics.csv')
    by_id={(int(r['seed']),int(r['budget']),r['method']):r for r in pairs}
    jpg={r['source_path']:int(r['jpeg_bytes']) for r in load_table(results/'fgvc_backend.csv') if int(r['long_edge'])==224}
    with np.load(results/'edge_adaptation_predictions.npz',allow_pickle=False) as f:
        test=list(f['test_paths']); seeds=f['seeds'].tolist(); pool=f['train_paths'].copy()
    if len(seeds)!=20 or len(test)!=1353 or len(set(test))!=1353: raise ValueError('Unexpected evaluation split')
    query={h:sum(jpg[p]+8 for p in test[:h]) for h in HORIZONS};out=[]
    for i,seed in enumerate(seeds):
        for budget in (32,128):
            subset=pool[i,:budget]; cal_images=sum(jpg[p] for p in subset)
            edge=by_id[seed,budget,'local_int8'];remote=by_id[seed,budget,'dino']
            edge_up=int(edge['calibration_upload_bytes']);head=int(edge['head_downlink_bytes']);dino_train=int(remote['calibration_upload_bytes'])
            review=int(edge['remote_review_additional_record_bytes']);assert edge_up==budget*(9+4+1280) and head==1294
            assert dino_train==cal_images+9*budget and review==cal_images+8*budget
            for labeling in ('local','remote_review'):
                for h in HORIZONS:
                    for method,u,d,row in [('local_int8',edge_up+(review if labeling=='remote_review' else 0),head,edge),('dino',dino_train+query[h],12*h,remote)]:
                        if h==len(test):
                            field='local_label_total_bytes' if labeling=='local' else 'remote_label_total_bytes'
                            assert u+d==int(row[field])
                        out.append(dict(seed=seed,budget=budget,labeling=labeling,horizon=h,method=method,upload_bytes=u,download_bytes=d))
    return out,test


def main(args):
    synthetic=fixed_rate_check();work,test=workloads(args.root);rows=[];sources={};summary=[]
    for name in TRACE_NAMES:
        traces=[];sources[name]={}
        for direction in ('up','down'):
            file=args.traces/(name+'.'+direction);t=np.loadtxt(file,dtype=np.int64,ndmin=1)
            assert len(t)>0 and np.all(np.diff(t)>=0) and t[0]>=0
            traces.append(t);sources[name][direction]=dict(sha256=digest(file),file_bytes=file.stat().st_size,slots=len(t),first_ms=int(t[0]),last_ms=int(t[-1]),
                url=f'https://raw.githubusercontent.com/ravinet/mahimahi/{COMMIT}/traces/{file.name}')
        up,down=traces;end=int(min(up[-1],down[-1]));group={}
        for phase in PHASES:
            start=(phase*end)//8
            for w in work:
                ut,dt,tt=times(up,down,start,w['upload_bytes'],w['download_bytes'])
                if start+tt>end:raise ValueError('Transfer outside common observed span')
                rows.append(dict(trace=name,phase_eighths=phase,start_ms=start,**w,upload_ms=ut,download_ms=dt,total_ms=tt))
                key=(w['budget'],w['labeling'],w['horizon']);pair=group.setdefault(key,{}).setdefault((phase,w['seed']),{})
                pair[w['method']]=tt
        for (budget,labeling,h),values in group.items():
            a=np.asarray([v['local_int8'] for v in values.values()]);b=np.asarray([v['dino'] for v in values.values()]); assert len(a)==100
            result=dict(trace=name,budget=budget,labeling=labeling,horizon=h,paired_seed_phases=len(a))
            for label,v in [('edge',a),('dino',b),('paired_dino_minus_edge',b-a)]:
                for stat,val in [('min',v.min()),('p50',np.quantile(v,.5)),('p95',np.quantile(v,.95)),('max',v.max())]:result[label+'_'+stat+'_s']=float(val)/1000.
            result.update(edge_faster=int(np.sum(a<b)),ties=int(np.sum(a==b)),dino_faster=int(np.sum(a>b)));summary.append(result)
    assert len(rows)==9600 and len(summary)==48
    args.out.mkdir(parents=True,exist_ok=True);save_table(args.out/'mahimahi_runs.csv',rows);save_table(args.out/'mahimahi_summary.csv',summary)
    meta=dict(upstream_repo='https://github.com/ravinet/mahimahi',upstream_revision=COMMIT,
              upstream_license='GPL-3.0-or-later per debian/copyright Files: *; repository COPYING is GNU GPL v3',
              sources=sources,synthetic_check=synthetic,rows=len(rows),summary_groups=len(summary),seeds=20,phases_eighths=list(PHASES),
              horizons=list(HORIZONS),budgets=[32,128],trace_wraps=0,all_runs_inside_common_observed_span=True,
              test_order_sha256=hashlib.sha256(('\n'.join(test)+'\n').encode()).hexdigest(),
              workload_source_sha256={n:digest(args.root/'results'/n) for n in ('fgvc_backend.csv','edge_adaptation_metrics.csv','edge_adaptation_predictions.npz')})
    (args.out/'mahimahi_manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(f'{len(rows)} replays, {len(summary)} paired summary groups; fixed-rate check passed.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--traces',type=Path,default=ROOT/'data/raw/mahimahi_traces');p.add_argument('--out',type=Path,default=ROOT/'results');main(p.parse_args())

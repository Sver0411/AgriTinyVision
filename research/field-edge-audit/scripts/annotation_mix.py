"""Price acquisition of missing camera-side labels and compare to the same JPEG oracle."""
import hashlib
import numpy as np
from joint_budget import table,write,OUT

def main():
    oracle={(int(r['seed']),int(r['budget'])):r for r in table('head_vs_oracle.csv')}
    costs={r['source_path']:int(r['jpeg_bytes'])+8 for r in table('fgvc_backend.csv') if r['long_edge']=='224'}
    caps={'cornell':2285396,'kashmir':945647,'latvia':2697020}
    old=table('joint_budget_summary.csv')
    for site in caps:
        r=next(r for r in old if r['site']==site and r['cap_percent']=='25')
        assert caps[site]==int(r['cap_bytes'])
    with np.load(OUT/'edge_adaptation_predictions.npz') as z:
        paths=z['train_paths']
    rows=[];breaks=[]
    for seed,path in enumerate(paths):
        for labels in (32,128):
            pkg=oracle[seed,labels]
            base=int(pkg['observed_update_bytes_local_review']); bound=int(pkg['oracle_with_balanced_head_bytes'])
            records=sorted(path[:labels],key=lambda p:hashlib.sha256(('remote-review-v1/'+p).encode()).digest())
            sums=base+np.r_[0,np.cumsum([costs[p] for p in records])]
            assert sums[-1]==int(pkg['observed_update_bytes_remote_review'])
            def last(fit):
                return int(np.flatnonzero(fit)[-1]) if fit.any() else -1
            for remote,byte in enumerate(sums):
                rows.append(dict(seed=seed,labels=labels,remote_images=remote,camera_labels=labels-remote,
                                 setup_bytes=int(byte),oracle_min_bytes=bound,strictly_cheaper_than_oracle=int(byte<bound),
                                 **{s+'_fits':int(byte<=b) for s,b in caps.items()}))
            breaks.append(dict(seed=seed,labels=labels,last_remote_count_beating_oracle=last(sums<bound),
                               **{s+'_max_remote_count':last(sums<=b) for s,b in caps.items()}))
    write('annotation_mix.csv',rows);write('annotation_boundaries.csv',breaks)
    for labels in (32,128):
        q=[b['last_remote_count_beating_oracle'] for b in breaks if b['labels']==labels]
        print(labels,'remote labels still cheaper than oracle (range/median)',min(q),np.median(q),max(q))
        for f in [0,.25,.5,.75,1]:
            arr=[r for r in rows if r['labels']==labels and r['remote_images']==int(labels*f)]
            print(f,int(round(np.mean([r['setup_bytes'] for r in arr]))),sum(r['strictly_cheaper_than_oracle'] for r in arr),
                  sum(r['kashmir_fits'] for r in arr))


if __name__=='__main__':
    main()

"""Pair frozen outputs on identical source records and compute fixed-pool deployment tradeoffs."""
import csv
import hashlib
import json
from pathlib import Path
import argparse

import numpy as np
from certify import correct_sign_power, interval

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results'


def read(name):
    with (OUT/name).open() as f:
        return list(csv.DictReader(f))


def write(name, rows):
    with (OUT/name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def confusion(y, pred):
    tn,fp,fn,tp = np.bincount(2*y+pred, minlength=4)
    sen=tp/(tp+fn) if tp+fn else None
    spec=tn/(tn+fp) if tn+fp else None
    return dict(tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),correct=int(tn+tp),
                accuracy=float((tn+tp)/len(y)),sensitivity=sen,specificity=spec,
                balanced_accuracy=(sen+spec)/2 if sen is not None and spec is not None else None)


def orchard_groups(edge):
    names=sorted(edge)
    hashes=[int(edge[p]['phash'],16) for p in names]
    parent=list(range(len(names)))
    def find(i):
        while i!=parent[i]:
            i,parent[i]=parent[i],parent[parent[i]]
        return i
    for j,h in enumerate(hashes):
        for i in range(j):
            if (h^hashes[i]).bit_count()<=4:
                parent[find(j)]=find(i)
    groups={}
    for i,name in enumerate(names):
        groups.setdefault(find(i),[]).append(name)
    labels=lambda names:{edge[p]['y'] for p in names}
    return [p[0] for p in groups.values() if len(labels(p))==1], [
        dict(representative=g[0],members='|'.join(g),size=len(g),conflicting=int(len(labels(g))>1))
        for g in groups.values() if len(g)>1]


def rank_selection(order, size, budget):
    selected=np.zeros(len(size),dtype=bool)
    for i in order:
        if size[i] <= budget:
            selected[i]=True
            budget-=size[i]
    return selected


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mobile-only',action='store_true',help='Run before the independent DINO extension is ready')
    mobile_only=parser.parse_args().mobile_only
    dino={} if mobile_only else {(r['dataset'],r['source_path']):r for r in read('specialist_predictions.csv')}
    balanced={} if mobile_only else {(r['dataset'],r['source_path']):r for r in read('specialist_balanced.csv')}
    edge_doc={r['source_path']:{**r,'y':int(r['y'])} for r in read('plantdoc_logits.csv') if int(r['training_crop'])}
    edge_fg={r['source_path']:{**r,'y':int(r['true_diseased'])} for r in read('fgvc_edge.csv')}
    board={r['source_path']:{**r,'binary_pred':int(float(r['board_p_suspect']) >= float(np.float32(.30))),
                            'p_suspect':r['board_p_suspect']} for r in edge_doc.values() if r['in_public_hardware_60']=='1'}
    pred_doc=read('backend_predictions.csv'); pred_fg=read('fgvc_backend.csv')
    assert len(edge_doc)==131 and len(board)==60 and len(edge_fg)==1821
    kept,dups=orchard_groups(edge_fg)
    if dups:
        write('orchard_duplicate_groups.csv',dups)
    public_components={r['test_component'] for r in board.values()}
    cohorts={
      'Board60':(board,pred_doc),
      'PlantDoc131':(edge_doc,pred_doc),
      'PlantDocClean':({p:r for p,r in edge_doc.items() if r['keep_dedup']=='1'},pred_doc),
      'PlantDocNewClean':({p:r for p,r in edge_doc.items() if r['keep_dedup']=='1' and r['test_component'] not in public_components},pred_doc),
      'FGVC1821':(edge_fg,pred_fg),
      'FGVCDedup':({p:edge_fg[p] for p in kept},pred_fg),
    }
    outcomes=[]; routing=[]; audits=[]; crops=[]
    for cohort,(edge,backend) in cohorts.items():
        ids=sorted(edge)
        N=len(ids)
        y=np.array([edge[p]['y'] for p in ids])
        e=np.array([int(edge[p]['binary_pred']) for p in ids])
        prob=np.array([float(edge[p]['p_suspect']) for p in ids])
        names=np.array([edge[p].get('crop','apple') for p in ids])
        for label,pred in [('local',e),('always_alert',np.ones(N,dtype=int))]:
            outcomes.append(dict(cohort=cohort,system=label,n=N,repairs=int(np.sum((e!=y)&(pred==y))),regressions=int(np.sum((e==y)&(pred!=y))),
                             both_wrong=int(np.sum((e!=y)&(pred!=y))),bytes=0,**confusion(y,pred)))
        predmap={(row['source_path'],int(row['long_edge'])):row for row in backend}
        reference_bytes=sum(int(predmap[p,224]['jpeg_bytes']) for p in ids)
        for side in (224,112):
            rows=[predmap[p,side] for p in ids]
            assert all(str(a['y'])==str(b) and a['source_sha256']==edge[p]['file_sha256'] for a,b,p in zip(rows,y,ids))
            size=np.array([int(r['jpeg_bytes']) for r in rows])
            main=np.array([int(r['pred_diseased']) for r in rows])
            conditional=np.array([int(float(r['p_diseased_known_crop'])>=0.5) for r in rows])
            for label,pred in [(f'server{side}',main),(f'crop_known{side}',conditional)]:
                repairs=int(np.sum((e!=y)&(pred==y))); reg=int(np.sum((e==y)&(pred!=y)))
                outcomes.append(dict(cohort=cohort,system=label,n=N,repairs=repairs,regressions=reg,both_wrong=int(np.sum((e!=y)&(pred!=y))),bytes=int(sum(size)),**confusion(y,pred)))
            variants=[('mobile',main)]
            if side==224 and not mobile_only:
                matched=[dino[(r['dataset'],r['source_path'])] for r in rows]
                assert all(a['jpeg_sha256']==b['jpeg_sha256'] and int(a['y'])==int(b['y']) for a,b in zip(rows,matched))
                specialist=np.array([int(b['pred_diseased']) for b in matched])
                outcomes.append(dict(cohort=cohort,system='specialist224',n=N,
                    repairs=int(np.sum((e!=y)&(specialist==y))), regressions=int(np.sum((e==y)&(specialist!=y))),
                    both_wrong=int(np.sum((e!=y)&(specialist!=y))),bytes=int(sum(size)),**confusion(y,specialist)))
                variants.append(('specialist',specialist))
                weighted=np.array([int(balanced[(r['dataset'],r['source_path'])]['pred_diseased']) for r in rows])
                outcomes.append(dict(cohort=cohort,system='balanced224',n=N,repairs=int(np.sum((e!=y)&(weighted==y))),
                    regressions=int(np.sum((e==y)&(weighted!=y))),both_wrong=int(np.sum((e!=y)&(weighted!=y))),
                    bytes=int(sum(size)),**confusion(y,weighted)))
                variants.append(('balanced',weighted))
            for backend_name,main in variants:
                fixes=(e!=y)&(main==y); harm=(e==y)&(main!=y); dis=fixes|harm
                D=int(dis.sum()); R=int(fixes.sum())
                assert np.array_equal(dis,(main!=e))
                for name in sorted(set(names)):
                    mask=names==name
                    for label,pred in [('local',e),(f'{backend_name}{side}',main)]:
                        if label=='local' and (side!=224 or backend_name!='mobile'):
                            continue
                        crops.append(dict(cohort=cohort,crop=name,model=label,n=int(mask.sum()),**confusion(y[mask],pred[mask])))
                rng=np.random.default_rng(20261127)
                discordant=np.flatnonzero(dis)
                for b in (10,25,50):
                    for design in ('disagreement','uniform'):
                        candidate=discordant if design=='disagreement' else np.arange(N)
                        idx=rng.choice(candidate,size=min(b,len(candidate)),replace=False)
                        n=int(dis[idx].sum());x=int(fixes[idx].sum())
                        low,high=interval(D,n,x)
                        audits.append(dict(cohort=cohort,backend=backend_name,side=side,budget=b,design=design,N=N,D=D,R=R,
                            labels_revealed=len(idx),labelled_disagreements=n,labelled_repairs=x,
                            net_correct=int(2*R-D),ci_low_count=2*low-D,ci_high_count=2*high-D,
                            sign_power=correct_sign_power(N,D,R,b,design),shadow_bytes=int(sum(size))))
                for frac in (.05,.10,.25,.50,1.):
                    B=int(reference_bytes*frac)
                    ordered={'uncertainty':np.argsort(abs(prob-.30),kind='stable'),
                             'smallest':np.argsort(size,kind='stable'),
                             'oracle':np.flatnonzero(fixes)[np.argsort(size[fixes],kind='stable')]}
                    # Paired budget sweeps reuse the same fixed distribution of random orders.
                    for seed in range(30):
                        ordered[f'random_{seed}']=np.random.default_rng(seed).permutation(N)
                    for label,order in ordered.items():
                        chosen=rank_selection(order,size,B); final=np.where(chosen,main,e)
                        rep=int(sum(fixes&chosen)); reg=int(sum(harm&chosen))
                        assert int((final==y).sum()-(e==y).sum())==rep-reg
                        routing.append(dict(cohort=cohort,backend=backend_name,side=side,budget_fraction=frac,policy=label,
                                            budget_bytes=B,used_bytes=int(size[chosen].sum()),calls=int(chosen.sum()),
                                            repairs=rep,regressions=reg,net_correct=rep-reg,**confusion(y,final)))
    for name,rows in [('upload_outcomes.csv',outcomes),('upload_routing.csv',routing),('audit_power.csv',audits),('upload_by_crop.csv',crops)]:
        write(name,rows)


if __name__=='__main__':
    main()

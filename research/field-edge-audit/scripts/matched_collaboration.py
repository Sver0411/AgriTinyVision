"""Compare single-score operating points with the frozen two-direction edge/DINO rule."""
import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import struct

import numpy as np

from directional_audit import select_fixed_grid
from directional_policy import CALIBRATION, EDGE_CUTOFF, REPLY, REQUEST, decide, pack, probabilities_from_logits, select_queries, unpack
from edge_adaptation_split import ROOT, RESULTS, table, write_table
from edge_adaptation_wire import normalized

HEAD = ROOT / 'data/features/dino_plantdoc_balanced_head.npz'
DINO_CAL = struct.Struct('<4sIB')
THRESHOLD = struct.Struct('<f')


@dataclass
class Data:
    paths: list
    y: np.ndarray
    e: np.ndarray
    d: np.ndarray
    sizes: np.ndarray


@dataclass
class Method:
    name: str
    active: str
    threshold: float | None
    tier: str
    fit_upload: int
    update_bytes: int


def cornell(split):
    ids=[r['source_path'] for r in table(RESULTS/'edge_adaptation_split.csv') if r['status']==split]
    e={r['source_path']:r for r in table(RESULTS/'fgvc_edge.csv')}
    d={r['source_path']:r for r in table(RESULTS/'specialist_balanced.csv') if r['feature_dataset']=='fgvc'}
    sizes={r['source_path']:int(r['jpeg_bytes']) for r in table(RESULTS/'fgvc_backend.csv') if r['long_edge']=='224' and r['jpeg_quality']=='75'}
    y=np.array([int(e[p]['true_diseased']) for p in ids]);edge=np.array([float(e[p]['p_suspect']) for p in ids],np.float32)
    assert np.array_equal(y,[int(d[p]['y']) for p in ids]) and np.array_equal(edge>=EDGE_CUTOFF,[int(e[p]['binary_pred']) for p in ids])
    return Data(ids,y,edge,probabilities_from_logits([float(d[p]['score']) for p in ids]),np.array([sizes[p] for p in ids]))


def kashmir():
    rows=table(RESULTS/'transfer_manifest.csv');ids=[r['source_path'] for r in rows]
    with np.load(ROOT/'data/features/kashmir.npz',allow_pickle=False) as x,np.load(HEAD,allow_pickle=False) as h:
        if list(x['source_path'])!=ids:raise ValueError('Kashmir identities differ')
        margin=normalized(x['dino'])@h['coef'].ravel()+h['intercept'].item()
    e=probabilities_from_logits([float(r['edge_score_logit']) for r in rows]); assert np.array_equal(e>=EDGE_CUTOFF,[int(r['edge_pred']) for r in rows])
    return Data(ids,np.array([int(r['y']) for r in rows]),e,probabilities_from_logits(margin),np.array([int(r['jpeg_bytes']) for r in rows]))


def same_data(left,right):
    return list(left.paths)==list(right.paths) and all(np.array_equal(getattr(left,k),getattr(right,k)) for k in ('y','e','d','sizes'))


def stored_scores():
    path=RESULTS/'matched_collaboration_scores.csv'
    if sha256(path.read_bytes()).hexdigest()!='58a4a70919ac97302e9933205b6ec916e27911087e56caa8772ae77b4c5b73b0':
        raise ValueError('Cornell/Kashmir score cache mismatch')
    rows=table(path);out={}
    for site,number in [('calibration',450),('cornell',1353),('kashmir',384)]:
        r=[x for x in rows if x['cohort']==site];assert len(r)==number
        out[site]=Data([x['source_path'] for x in r],np.array([int(x['y']) for x in r]),
            np.array([float(x['edge_probability']) for x in r],np.float32),np.array([float(x['dino_probability']) for x in r],np.float32),
            np.array([int(x['jpeg_bytes']) for x in r]))
    return out


def latvia(portable):
    with np.load(RESULTS/'latvia_decisions.npz',allow_pickle=False) as z:
        compact=Data(list(z['source_path']),z['y'],z['edge_probability'],z['remote_probability'],z['jpeg_bytes'])
        source_hashes=list(z['source_sha256'])
    manifest=table(ROOT/'data/latvia_source/manifest.csv')
    if compact.paths!=[r['source_path'] for r in manifest] or source_hashes!=[r['source_sha256'] for r in manifest] or not np.array_equal(compact.y,[int(r['y']) for r in manifest]):
        raise ValueError('Latvian portable scores do not match the pinned image audit')
    full=ROOT/'data/features/latvia.npz'
    if full.exists() and not portable:
        with np.load(full,allow_pickle=False) as z:
            assert z['source_head_sha256'].item()==sha256(HEAD.read_bytes()).hexdigest()
            # Reproduce the released device-score emulation's float32 two-logit softmax.
            logits=z['logits']; exp=np.exp(logits-logits.max(axis=1,keepdims=True)); edge=exp[:,1]/exp.sum(axis=1)
            original=Data(list(z['source_path']),z['y'],edge,probabilities_from_logits(z['dino_balanced_margin']),z['jpeg_bytes'])
        if not same_data(compact,original):raise ValueError('Latvian packed and full scores differ')
    return compact


def confusion(y,pred):
    tp=int(np.count_nonzero(pred & (y==1))); tn=int(np.count_nonzero(~pred & (y==0)))
    fp=int(np.count_nonzero(pred & (y==0))); fn=int(np.count_nonzero(~pred & (y==1)))
    return dict(tp=tp,tn=tn,fp=fp,fn=fn,sensitivity=tp/(tp+fn),specificity=tn/(tn+fp),balanced_accuracy=(tp/(tp+fn)+tn/(tn+fp))/2)


def select_single(cal,source,base,sweep):
    scores=cal.e if source=='edge' else cal.d; original=cal.e>=EDGE_CUTOFF
    thresholds=np.unique(np.r_[np.float32(0),scores,np.nextafter(np.float32(1),np.float32(np.inf))]).astype(np.float32)
    candidates=[]
    for cutoff in thresholds:
        pred=scores>=cutoff; stats=confusion(cal.y,pred)
        row=dict(score=source,cutoff=float(cutoff),fp=stats['fp'],fn=stats['fn'],retractions=int((original&~pred).sum()))
        row['half_fn']=int(row['fn']<=base['fn']//2);row['both_strict']=int(row['fp']<base['fp'] and row['fn']<base['fn']);candidates.append(row)
    assert (candidates[0]['fn'],candidates[-1]['fp'])==(0,0)
    sweep.extend(candidates)
    preferred=[r for r in candidates if r['half_fn'] and r['both_strict']]; strict=[r for r in candidates if r['both_strict']]
    if preferred:
        primary=min(preferred,key=lambda r:(r['fp'],r['retractions'],r['fn'],r['cutoff']));tier='half_fn_and_strict_fp'
    elif strict:
        primary=min(strict,key=lambda r:(r['fn'],r['fp'],r['retractions'],r['cutoff']));tier='both_counts_strict'
    else:
        primary=None;tier='fallback_published_edge'
    np_cut=min((r for r in candidates if r['half_fn']),key=lambda r:(r['fp'],r['fn'],r['retractions'],r['cutoff']))
    fit_up=CALIBRATION.size*len(scores) if source=='edge' else DINO_CAL.size*len(scores)+int(cal.sizes.sum())
    if source=='dino':
        for idx,(yi,edge_pred) in enumerate(zip(cal.y,original,strict=True)):
            code=int(yi)|(int(edge_pred)<<1);wire=DINO_CAL.pack(b'DCL1',idx,code);assert DINO_CAL.unpack(wire)==(b'DCL1',idx,code)
    wire_cut=THRESHOLD.pack(np.nan if primary is None else primary['cutoff']);decoded=THRESHOLD.unpack(wire_cut)[0]
    if primary is not None:assert decoded==primary['cutoff']
    else:assert np.isnan(decoded)
    return [Method(source+'_target',source if primary else 'edge',float(EDGE_CUTOFF) if primary is None else decoded,tier,fit_up,len(wire_cut)),
            Method(source+'_np',source,THRESHOLD.unpack(THRESHOLD.pack(np_cut['cutoff']))[0],'half_fn_only_min_fp',fit_up,THRESHOLD.size)]


def evaluate(data,method,rule):
    if method.active=='joint':
        ask=select_queries(data.e,rule);pred=decide(data.e,np.where(ask,data.d,np.nan),rule)
    else:
        ask=np.full(len(data.y),method.active=='dino');scores=data.d if method.active=='dino' else data.e
        pred=scores>=np.float32(method.threshold)
    up=np.where(ask,data.sizes+REQUEST.size,0);down=ask.astype(np.int64)*REPLY.size
    return pred,ask,up,down


def run(args):
    compact=stored_scores()
    if not args.portable and (ROOT/'data/features/kashmir.npz').exists() and HEAD.exists():
        for site,full in [('calibration',cornell('label_pool')),('cornell',cornell('test')),('kashmir',kashmir())]:
            if not same_data(compact[site],full):raise ValueError('Full and packed scores differ for '+site)
    cal=compact['calibration']; assert len(cal.y)==450;meta=json.loads((RESULTS/'directional_rule.json').read_bytes());blob=(RESULTS/'directional_rule.bin').read_bytes();rule=unpack(blob)
    assert sha256(blob).hexdigest()==meta['binary_sha256'] and sha256(('\n'.join(cal.paths)+'\n').encode()).hexdigest()==meta['fit_paths_sha256']
    base=confusion(cal.y,cal.e>=EDGE_CUTOFF);assert base['fp']==meta['baseline']['fp'] and base['fn']==meta['baseline']['fn']
    _,_,_,_,selected,tier=select_fixed_grid(cal.y,cal.e,cal.d,cal.sizes);assert pack(selected[1])==blob and tier==meta['selection_tier']
    sweeps=[];methods=[Method('edge_published','edge',float(EDGE_CUTOFF),'published',0,0), Method('dino_fixed','dino',0.5,'fixed',0,0)]
    for src in ('edge','dino'):methods.extend(select_single(cal,src,base,sweeps))
    methods.append(Method('directional','joint',None,tier,meta['calibration_upload_bytes'],len(blob)))
    assert meta['calibration_upload_bytes']==int(cal.sizes.sum())+CALIBRATION.size*len(cal.y)
    outcomes=[];decisions=[];calibrations=[];paired=[];mechanism=[]
    for m in methods:
        pred,ask,_,_=evaluate(cal,m,rule);calibrations.append(dict(method=m.name,effective_inference=m.active,selection_tier=m.tier,cutoff=m.threshold,
            cal_n=len(cal.y),cal_half_fn_limit=base['fn']//2,cal_queries_if_deployed=int(ask.sum()),**confusion(cal.y,pred),calibration_upload_bytes=m.fit_upload,config_down_bytes=m.update_bytes))
    args.out.mkdir(parents=True,exist_ok=True);write_table(args.out/'matched_collaboration_calibration.csv',calibrations);write_table(args.out/'matched_collaboration_sweep.csv',sweeps)
    sites=[('cornell',compact['cornell']),('kashmir',compact['kashmir'])]
    if not args.omit_latvia:sites.append(('latvia',latvia(args.portable)))
    for site,data in sites:
        assert len(data.y)=={'cornell':1353,'kashmir':384,'latvia':1190}[site] and len(data.paths)==len(set(data.paths)); predictions={}
        for m in methods:
            p,ask,up,down=evaluate(data,m,rule);predictions[m.name]=p; stats=confusion(data.y,p);n=len(data.paths)
            outcomes.append(dict(cohort=site,n=n,method=m.name,effective_inference=m.active,selection_tier=m.tier,**stats,
                n_queried=int(ask.sum()),calibration_upload_bytes=m.fit_upload,config_down_bytes=m.update_bytes,
                infer_upload_bytes=int(up.sum()),infer_reply_bytes=int(down.sum()),inference_app_bytes=int(up.sum()+down.sum()),
                inference_bytes_per_image=float((up+down).mean()),total_app_bytes=int(m.fit_upload+m.update_bytes+up.sum()+down.sum())))
            for i,path in enumerate(data.paths):decisions.append(dict(cohort=site,source_path=path,reference=int(data.y[i]),method=m.name,edge_p=float(data.e[i]),
                stored_dino_p=float(data.d[i]),decision=int(p[i]),remote_queried=int(ask[i]),inference_upload_bytes=int(up[i]),inference_reply_bytes=int(down[i])))
        joint=predictions['directional'];js=confusion(data.y,joint)
        # A descriptive check of representability reads model decisions, never reference labels.
        # Any threshold on one score assigns all tied scores the same class.
        band=(data.d>=np.float32(rule.retract_cutoff)) & (data.d<np.float32(rule.rescue_cutoff))
        gate=select_queries(data.e,rule)
        if np.all(gate):
            expected=(data.d>=np.float32(rule.rescue_cutoff)) | (band & (data.e>=EDGE_CUTOFF))
            assert np.array_equal(expected,joint)
        diagnostic=dict(cohort=site,total=n,remote_intermediate=int(band.sum()),
                        intermediate_joint_positive=int((band & joint).sum()),intermediate_joint_negative=int((band & ~joint).sum()))
        for source,scores in [('edge',data.e),('dino',data.d)]:
            diagnostic['minimum_disagree_any_'+source+'_cutoff']=min([int(joint.sum())]+[int(np.sum((scores>=t)!=joint)) for t in np.unique(scores)])
        mechanism.append(diagnostic)
        for m in methods:
            if m.name=='directional':continue
            p=predictions[m.name];s=confusion(data.y,p);diff=dict(fp_change_joint_minus_control=js['fp']-s['fp'],fn_change_joint_minus_control=js['fn']-s['fn'])
            paired.append(dict(cohort=site,control=m.name,**diff,both_errors_lower=int(diff['fp_change_joint_minus_control']<0 and diff['fn_change_joint_minus_control']<0),
                balanced_accuracy_gain=js['balanced_accuracy']-s['balanced_accuracy'],joint_only_correct=int(np.sum((joint==data.y)&(p!=data.y))),
                control_only_correct=int(np.sum((p==data.y)&(joint!=data.y))),identical_decisions=int(np.sum(joint==p))))
    write_table(args.out/'matched_collaboration_outcomes.csv',outcomes);write_table(args.out/'matched_collaboration_predictions.csv',decisions);write_table(args.out/'matched_collaboration_pairs.csv',paired)
    write_table(args.out/'matched_collaboration_mechanism.csv',mechanism)
    print('cohort method FP FN BA queried; calibration thresholds',[(m.name,m.threshold,m.tier) for m in methods]);
    for r in outcomes:print(r['cohort'],r['method'],r['fp'],r['fn'],round(r['balanced_accuracy'],4),r['n_queried'])

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--portable',action='store_true',help='Use packed scores and source manifests without loading embeddings');parser.add_argument('--out',type=Path,default=RESULTS);parser.add_argument('--omit-latvia',action='store_true',help='Stage a preliminary two-cohort set; the default run requires all three cohorts')
    run(parser.parse_args())

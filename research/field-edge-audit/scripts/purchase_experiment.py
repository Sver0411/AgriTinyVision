"""Target supervision acquisition, prospective controller selection and paired stream replay."""
import argparse
import csv
import ctypes
import hashlib
import os
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from acquisition_baselines import METHODS, bbsd_soft, download_byte_count, edge_fit, fit_router, reference_scores
from edge_adaptation_wire import predict_from_head, quantized
from purchase_cohorts import load
from purchase_policy import choose_action

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'purchase'
PILOT = 16
FULL_RATE = 6800
ACTION_BYTES = 8
REQUEST_BYTES = 32
SITES = ('cornell', 'kashmir', 'latvia')
BACKBONES = ('v2', 'v2grid', 'v4')
FRONT_HALF = 2
BASE = 'edgeboost_rank'


def array(x, dtype):
    return np.ascontiguousarray(x, dtype=dtype)


@lru_cache(maxsize=1)
def library():
    source = ROOT/'c_reference'/'purchase_stream.c'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    directory = Path(tempfile.gettempdir()) / 'crop-camera-stream'
    directory.mkdir(exist_ok=True)
    shared = directory / (digest + '.so')
    if not shared.exists():
        tmp = directory / f'{digest}.{os.getpid()}.so'
        subprocess.run(['cc','-std=c99','-shared','-fPIC','-O2',str(source),'-o',str(tmp)],check=True)
        tmp.replace(shared)
    lib = ctypes.CDLL(str(shared))
    c_int, c_double = ctypes.c_int, ctypes.c_double
    pointer = lambda d: np.ctypeslib.ndpointer(dtype=d,ndim=1,flags='C_CONTIGUOUS')
    dbl, byte, cost, info = map(pointer,(np.float64,np.uint8,np.int32,np.int64))
    lib.past_percentile.argtypes = [c_int,dbl,dbl]
    lib.past_percentile.restype = None
    lib.replay.argtypes = [c_int]*6+[c_double,cost,byte,byte,dbl,dbl,byte,byte,byte,info]
    lib.replay.restype = c_int
    return lib


def ranks(priority):
    priority = array(priority,np.float64)
    out = np.empty_like(priority)
    library().past_percentile(len(out),priority,out)
    return out


@dataclass(frozen=True)
class Plan:
    method: str
    model: str
    fit_n: int
    labels: int
    setup: int
    guard: bool = False
    action: str = ''
    reason: str = ''
    fallback: bool = False


def keep(method, guard=False, reason='', fallback=False):
    return Plan(method,'source',0,0,0,guard,'freeze_guard' if guard else 'freeze',reason,fallback)


def pilot_evidence(x,y,source,cloud,sizes):
    # Label-independent held-out positions. The two nested fits see identical validation positions.
    n=len(y)
    small=np.empty(n); large=np.empty(n)
    for fold in range(4):
        val=np.arange(n)%4==fold
        tr=np.flatnonzero(~val)
        for dest,train in ((small,tr[::2]),(large,tr)):
            head=edge_fit(x[train],y[train])
            dest[val]=predict_from_head('local_int8',head,x[val])
    return small,large


def pilot_outcome(source, frozen_priority, margin, cloud, sizes, cutoff, setup, cap, n, mode, guard):
    """Price the candidate on paid past observations; never see evaluation scores or sizes."""
    original=source>=cutoff
    if margin is None:
        local=original
        priority=frozen_priority
    else:
        local=margin>=0
        priority=-np.abs(margin)
    pred=(local|original).copy() if guard else local.copy()
    if setup>cap:
        return original.copy()
    reserve_fraction=setup/cap if mode=='earned' else 0
    n_before=min(len(pred),int(np.floor(len(pred)*reserve_fraction)))
    pred[:n_before]=original[:n_before]
    usable=np.arange(len(pred))>=n_before
    if guard:
        usable &= ~original
    limit=(cap-setup)*len(pred)/n
    # This is a historical pilot forecast. Evaluation uses the causal controller below.
    for i in np.argsort(-priority,kind='stable'):
        if usable[i] and sizes[i]<=limit:
            pred[i]=cloud[i]
            limit-=sizes[i]
    return pred


class TrainingPanel:
    def __init__(self, d, indices, seed, reference):
        self.seed=seed
        self.x=quantized(d['cal_x'][indices])[0]
        self._bank_labels=d['cal_y']
        self._bank_cloud=d['cal_cloud_p']
        self.indices=indices
        self.p=d['cal_p'][indices]
        self.sizes=d['cal_size'][indices].astype(int)+20
        self.width=self.x.shape[1]
        self.head=self.width+14
        receipts=d['cal_size'][indices].astype(int)+self.width+37
        self.fees=np.r_[0,np.cumsum(receipts)]
        self.cutoff=float(np.float32(d['cutoff']))
        self.shift=bbsd_soft(reference,self.p[:PILOT])[0]
        self.fitted={}
        self._pilot_evidence=None

    def paid_training_prefix(self,n):
        # Each policy is billed for exactly its admitted prefix. Fits can be cached
        # across counterfactual policies, but are never inputs to a pre-purchase quote.
        ids=self.indices[:n]
        return self._bank_labels[ids],self._bank_cloud[ids]>=.5

    def fit(self, kind, n):
        key=(kind,n)
        if key not in self.fitted:
            y,cloud=self.paid_training_prefix(n)
            model=fit_router(kind,self.x[:n],y,cloud,self.seed)
            if model.download_bytes!=download_byte_count(kind,self.width):
                raise ValueError('Quoted and measured download bytes disagree')
            self.fitted[key]=model
        return self.fitted[key]

    def bill(self,n,model,staged=False):
        if not n:
            return 0
        # A staged expansion downloads an initial shadow classifier and a final classifier.
        shadow=self.head if staged and n>PILOT else 0
        return int(self.fees[n])+REQUEST_BYTES+ACTION_BYTES+download_byte_count(model,self.width)+shadow

    def audit(self,n,cap,mode,limit,rule='protected',force_guard=False):
        pilot=int(self.fees[PILOT])+REQUEST_BYTES+ACTION_BYTES+self.head
        method='adaptive_guard' if force_guard else ('adaptive' if rule=='protected' else 'empirical')
        if FRONT_HALF*pilot>cap:
            return keep(method,force_guard,'pilot_quote_exceeds_half')
        audit_y,cloud=self.paid_training_prefix(PILOT)
        if self._pilot_evidence is None:
            self._pilot_evidence=pilot_evidence(self.x[:PILOT],audit_y,self.p[:PILOT],cloud,self.sizes[:PILOT])
        small_margin,large_margin=self._pilot_evidence
        p=self.p[:PILOT]
        original=p>=self.cutoff
        future={'freeze':0,'probe':0,'expand':int(self.fees[limit]-self.fees[PILOT])+self.head}
        if not force_guard:
            future.update({k+'_guard':v for k,v in tuple(future.items())})
        preds={}
        for action,fee in future.items():
            margin=None if action.startswith('freeze') else large_margin
            preds[action]=pilot_outcome(p,-abs(p-self.cutoff),margin,cloud,self.sizes[:PILOT],
                                        self.cutoff,pilot+fee,cap,n,mode,force_guard or action.endswith('_guard'))
        small=small_margin>=0;large=large_margin>=0
        learning={'plain':(small|original,large|original) if force_guard else (small,large),
                  'guard':(small|original,large|original)}
        choice=choose_action(audit_y,preds,cap-pilot,future,rule=rule,learning_pair=learning,max_labels=limit)
        action=choice.action
        frozen=action.startswith('freeze')
        labels=limit if action.startswith('expand') else PILOT
        return Plan(method,'source' if frozen else BASE,0 if frozen else labels,labels,
                    pilot+future[action],force_guard or action.endswith('_guard'),action,choice.reason)

    def plans(self,n,cap,mode,limit):
        main=self.audit(n,cap,mode,limit)
        protected=self.audit(n,cap,mode,limit,force_guard=True)
        result=[keep('native'),keep('freeze'),keep('freeze_guard',True),main,protected,
                self.audit(n,cap,mode,limit,'paired_erm')]
        def fixed(name,kind,num,guard=False,staged=False,allow=True):
            if not allow or not num:
                return keep(name,guard,'no_purchase')
            # Actual download sizes are architecture constants for these ports. Avoid
            # making affordability depend on fitting inaccessible labels.
            price=self.bill(num,kind,staged)
            if price>cap:
                return keep(name,guard,'hard_cap',True)
            return Plan(name,kind,num,num,price,guard,'fit_'+str(num),'fixed')
        result.append(fixed('probe16',BASE,PILOT))
        result.append(fixed('bbsd',BASE,limit,allow=self.shift))
        for kind in METHODS:
            result.append(fixed('always_'+kind,kind,limit))
        result.append(fixed('always_rank_guard',BASE,limit,True))
        for name,base,guard in [('matched_rank',main,False),('matched_guard',main,True),
                                ('guard_matched_rank',protected,False),('guard_matched_guard',protected,True)]:
            pl=fixed(name,BASE,base.labels,guard,staged=True)
            assert pl.setup==base.setup and pl.labels==base.labels
            result.append(pl)
        for kind in METHODS:
            if kind!=BASE:
                # Includes the actual shadow-head cost and charges the published router.
                pl=fixed('matched_'+kind,kind,main.labels,staged=True)
                if pl.fallback and main.labels:
                    # The same labels and base head are already funded. Decline only the unaffordable router.
                    pl=Plan(pl.method,BASE,main.labels,main.labels,main.setup,False,'fallback_to_matched_rank','router_fee',True)
                result.append(pl)
        return result


def stream_inputs(d):
    return {'q':quantized(d['test_x'])[0],'source':d['test_p']>=np.float32(d['cutoff']),
            'urgency':-np.abs(d['test_p'].astype(float)-float(np.float32(d['cutoff']))),
            'size':d['test_size']+20,'cloud':d['test_cloud_p']>=.5,'orders':d['test_orders']}


class Replay:
    def __init__(self,inputs,panel):
        self.inputs=inputs
        self.panel=panel
        self.states={}
        self.scores={}
        self.fixed={}
        self.inverse=[np.argsort(order) for order in inputs['orders']]

    def state(self,model,num,order_id):
        key=(model,num)
        ikey=(key,order_id)
        if ikey in self.states:
            return self.states[ikey]
        if key not in self.scores:
            if model=='source':
                local=self.inputs['source'];priority=self.inputs['urgency']
            else:
                z,priority=self.panel.fit(model,num).score(self.inputs['q'])
                local=z>=0
            self.scores[key]=local,priority
        local,priority=self.scores[key]
        idx=self.inputs['orders'][order_id]
        p=array(priority[idx],np.float64)
        src=array(self.inputs['source'][idx],np.uint8)
        args=(array(self.inputs['size'][idx],np.int32),src,array(local[idx],np.uint8),
              p,ranks(p),array(self.inputs['cloud'][idx],np.uint8))
        self.states[ikey]=args
        return args

    def run(self,plan,order_id,mode,rate):
        args=self.state(plan.model,plan.fit_n,order_id)
        n=len(args[0]);grant=n*rate
        isfixed=plan.model=='edgeboost_fixed' or plan.method=='native'
        threshold=float('inf')
        if plan.model=='edgeboost_fixed':
            key=(plan.fit_n,grant-plan.setup,n)
            if key not in self.fixed:
                panel_sizes=self.panel.sizes[:plan.fit_n]
                funded=np.clip((grant-plan.setup)/(n*panel_sizes.mean()),0,1)
                self.fixed[key]=self.panel.fit(plan.model,plan.fit_n).fixed_cutoff(panel_sizes,funded)
            threshold=self.fixed[key]
        pred=np.empty(n,np.uint8);queries=np.empty_like(pred);stats=np.zeros(4,np.int64)
        err=library().replay(n,rate,int(mode=='earned'),plan.setup,int(plan.guard),int(isfixed),
                             threshold,*args,pred,queries,stats)
        if err:
            raise ValueError('Failed funding-prefix check')
        inverse=self.inverse[order_id]
        return pred[inverse],queries[inverse],stats


def write(path,rows):
    with path.open('w',newline='') as file:
        w=csv.DictWriter(file,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)


def run_site(site,backbone,out,seeds):
    d=load(site,backbone)
    inputs=stream_inputs(d)
    reference=reference_scores(backbone)
    n=len(inputs['source'])
    rows=[];pred_masks=[];query_masks=[]
    for seed in range(seeds):
        panel=TrainingPanel(d,d['cal_draws'][seed],seed,reference)
        replay=Replay(inputs,panel)
        for limit in (32,128):
            for pct in (10,25,50):
                rate=FULL_RATE*pct//100
                cap=n*rate
                for mode in ('prepaid','earned'):
                    for plan in panel.plans(n,cap,mode,limit):
                        for order_id in range(len(inputs['orders'])):
                            pred,ask,stats=replay.run(plan,order_id,mode,rate)
                            # The evaluator alone opens test references after the stream returns.
                            y=d['test_y'].astype(bool)
                            fp=int(np.sum(pred & ~y));fn=int(np.sum(~pred.astype(bool) & y))
                            n_pos=int(y.sum());n_neg=n-n_pos
                            review,calls,start,deficit=map(int,stats)
                            if plan.guard and np.any(inputs['source'] & ~pred.astype(bool)):
                                raise ValueError('Retained-alert policy withdrew a source alert')
                            assert review==int(inputs['size'][ask.astype(bool)].sum())
                            assert int(ask.sum())==calls and plan.setup+review<=cap and not deficit
                            rows.append(dict(site=site,backbone=backbone,seed=seed,order=order_id,label_cap=limit,
                                             cap_pct=pct,grant_mode=mode,method=plan.method,fp=fp,fn=fn,
                                             ba=1-(fp/n_neg+fn/n_pos)/2,accuracy=1-(fp+fn)/n,
                                             labels=plan.labels,setup_bytes=plan.setup,review_bytes=review,
                                             total_bytes=plan.setup+review,queries=calls,install_frame=start,
                                             n_neg=n_neg,n_pos=n_pos,mask_row=len(rows),cap_bytes=cap,
                                             action=plan.action,reason=plan.reason,guard=int(plan.guard),
                                             fallback=int(plan.fallback),prefix_overdraft=deficit))
                            pred_masks.append(np.packbits(pred));query_masks.append(np.packbits(ask))
        if (seed+1)%5==0: print(site,backbone,seed+1,flush=True)
    out.mkdir(exist_ok=True,parents=True)
    write(out/f'runs_{site}_{backbone}.csv',rows)
    np.savez_compressed(out/f'masks_{site}_{backbone}.npz',pred=pred_masks,query=query_masks,
                        test_id=d['test_id'],test_group=d['test_group'])
    return len(rows)


def merge(out):
    paths=[out/f'runs_{s}_{b}.csv' for s in SITES for b in BACKBONES]
    groups=defaultdict(list)
    tmp=out/'runs.tmp'
    with tmp.open('wb') as combined:
        for i,path in enumerate(paths):
            with path.open('rb') as f:
                header=f.readline()
                if i==0:combined.write(header)
                for line in f:combined.write(line)
            with path.open(newline='') as f:
                for row in csv.DictReader(f):
                    keys=('site','backbone','label_cap','cap_pct','grant_mode','method')
                    groups[tuple(row[k] for k in keys)].append(row)
    tmp.replace(out/'runs.csv')
    records=[]
    for key,rows in groups.items():
        record=dict(zip(keys,key));record['runs']=len(rows)
        for k in ('fp','fn','ba','accuracy','labels','setup_bytes','review_bytes','total_bytes','queries',
                  'install_frame','n_neg','n_pos','fallback','prefix_overdraft'):
            record[k]=sum(float(r[k]) for r in rows)/len(rows)
        records.append(record)
    write(out/'summary.csv',records)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--site',choices=SITES);p.add_argument('--backbone',choices=BACKBONES)
    p.add_argument('--seeds',type=int,default=20);p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--merge',action='store_true');a=p.parse_args()
    if a.merge:
        merge(a.out)
    elif a.site and a.backbone:
        print(run_site(a.site,a.backbone,a.out,a.seeds),flush=True)
    else:
        p.error('Pass one site and backbone, or --merge')

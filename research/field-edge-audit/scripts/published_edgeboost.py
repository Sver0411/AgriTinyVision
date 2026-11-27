"""Reproduce the published EdgeBoost gate and compare it under the orchard byte protocol."""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import re
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.model_selection import StratifiedKFold

from edge_adaptation import fit
from edge_adaptation_wire import normalized, quantized, pack_head, predict_from_head
from evaluate_upload import rank_selection
from joint_budget import dataset, table, write, OUT, FEATURES

REV = '26abd91857989a542ef67c7fd59f619ee563f458'
ASSET_HASHES = {
    'offload.py': 'ca695f2223cee72536900d1b737cc1fd4b3c2c33ef69c2c8a9e2f0fd45121652',
    'cifar_test_labels.npy': '0e906797bd0c9ea20884b650168267425783fb958c63fc8edd87e0c2d41f1371',
    'mobnetv3cal_cifar100_predictions.npy': '5323239c86c389d387daaa3e8c071d158e4f9a3aa84e224da1a93385784ae4b8',
    'efficientnetv2_predictions.npy': '428eb6c0ca2462313b5290b8ef396413ef18feb55d217cb4580e4da25bd525e5',
}
THRESHOLDS = [.04, .087, .298, .59, .72, .83, .93]


def official_check(directory):
    import urllib.request
    directory.mkdir(parents=True, exist_ok=True)
    for name, digest in ASSET_HASHES.items():
        path = directory / name
        if not path.exists():
            source = f'https://raw.githubusercontent.com/ds-kiel/EdgeBoost/{REV}/{name}'
            with urllib.request.urlopen(source, timeout=90) as response:
                path.write_bytes(response.read())
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Upstream asset mismatch: {name}')
    spec = importlib.util.spec_from_file_location('upstream_edgeboost', directory / 'offload.py')
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    y, p, server = [np.load(directory / k) for k in list(ASSET_HASHES)[1:]]
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        upstream.offload(y, p, p, server)
    output = captured.getvalue()
    correct = [float(v) for v in re.findall(r'Combined approach accuracy on test set: ([.\d]+)%', output)]
    upload = [float(v) for v in re.findall(r'% of uncertain predictions at network 1: ([.\d]+)%', output)]
    ordered = np.sort(p, axis=1)
    confidence = ordered[:, -1] - ordered[:, -2]
    observations = []
    for i, cutoff in enumerate(THRESHOLDS):
        ask = confidence < cutoff
        prediction = np.where(ask, server.argmax(1), p.argmax(1))
        assert round(100*(prediction == y).mean(),1) == correct[i]
        assert round(int(ask.sum())/len(ask)*100,2) == upload[i]
        observations.append(dict(threshold=cutoff, count=len(y), queried=int(ask.sum()),
                                 correct=int((prediction==y).sum()), upstream_acc_pct=correct[i],
                                 upstream_upload_pct=upload[i]))
    write('published_edgeboost_author_replay.csv',observations)


def calibration():
    labels = {r['source_path']:int(r['y']) for r in table('edge_adaptation_split.csv')}
    sizes = {r['source_path']:int(r['jpeg_bytes']) + 20 for r in table('fgvc_backend.csv') if r['long_edge']=='224'}
    with np.load(FEATURES/'edge_fgvc.npz') as z:
        lookup = {p:i for i,p in enumerate(z['source_path'])}
        features = quantized(z['activation'])[0]
    with np.load(OUT/'edge_adaptation_predictions.npz') as z:
        draws = z['train_paths']
    params, rows = [], []
    for seed, ids in enumerate(draws):
        yy = np.array([labels[p] for p in ids])
        xx = features[[lookup[p] for p in ids]]
        oof = np.zeros(len(ids))
        for train, held in StratifiedKFold(n_splits=4,shuffle=True,random_state=seed).split(xx,yy):
            w,b = fit(normalized(xx[train]),yy[train])
            blob = pack_head('local_int8',w,b)
            oof[held] = predict_from_head('local_int8',blob,xx[held])
        def nll(log_temperature):
            scaled = oof / np.exp(log_temperature)
            return float((np.logaddexp(0,scaled) - yy*scaled).mean())
        result = minimize_scalar(nll,bounds=(-6,6),method='bounded')
        if not result.success:
            raise RuntimeError(result.message)
        temperature = float(np.float32(np.exp(result.x)))
        conf = np.tanh(abs(oof)/(2*temperature))
        cal_cost = np.array([sizes[p] for p in ids])
        gates = {}
        for pct in [10,25,50]:
            candidates = np.unique(np.r_[np.float32(0),np.nextafter(conf.astype(np.float32),np.float32(np.inf)),np.float32(1.000001)])
            # Nearest byte fraction on the calibration population, irrespective of test class/size.
            fractions = np.array([cal_cost[conf<t].sum()/cal_cost.sum() for t in candidates])
            cutoff = float(np.float32(candidates[np.argmin(abs(fractions-pct/100))]))
            gates[pct]=cutoff
        rows.append(dict(seed=seed,n=len(ids),temperature=temperature,uncalibrated_nll=nll(0),
                         calibrated_nll=nll(np.log(temperature)),**{f'threshold_{p}':v for p,v in gates.items()}))
        params.append((temperature,gates))
    write('published_edgeboost_calibration.csv',rows)
    return params


def main(author_dir):
    if author_dir:
        official_check(author_dir)
    parameters = calibration()
    fees = {(int(r['seed']),place):int(r[f'{place}_label_total_bytes'])
            for r in table('edge_adaptation_metrics.csv') if r['method']=='local_int8' and r['budget']=='128'
            for place in ['local','remote']}
    rows = []
    for site in ['cornell','kashmir','latvia']:
        ids, y, edge_p, expert, jpeg, scores = dataset(site)
        costs = jpeg + 20
        preds, masks = [], []
        anonymous_ids = np.array([hashlib.sha256(('edgeboost-order-v1/' + p).encode()).hexdigest() for p in ids])
        frozen_order = np.lexsort((anonymous_ids,abs(edge_p-np.float32(.3))))
        for seed,(temp,gates) in enumerate(parameters):
            score = scores[seed]
            calibrated_margin = np.tanh(abs(score)/(2*temp))
            order = np.lexsort((anonymous_ids,calibrated_margin))
            assert np.array_equal(order,np.lexsort((anonymous_ids,abs(score))))
            for pct in [10,25,50]:
                cap = int(costs.sum())*pct//100
                for place in ['local','remote','prepaid']:
                    nominal = 0 if place=='prepaid' else fees[seed,place]
                    for mode in ['budget_rank','fixed_gate']:
                        extra = 0 if mode=='budget_rank' else 8+(128*4 if place=='local' else 0)
                        setup = nominal+extra if place!='prepaid' else 0
                        if setup>cap:
                            if nominal<=cap:
                                used=nominal
                                ask=rank_selection(order,costs,cap-used)
                                local=score>=0
                                state='adapted_rank_fallback'
                            else:
                                used=0
                                ask=rank_selection(frozen_order,costs,cap)
                                local=edge_p>=np.float32(.3)
                                state='source_fallback'
                        else:
                            used=setup
                            local=score>=0
                            if mode=='budget_rank':
                                chosen=order
                            else:
                                eligible=calibrated_margin<gates[pct]
                                chosen=np.flatnonzero(eligible)[np.argsort(anonymous_ids[eligible],kind='stable')]
                            ask=rank_selection(chosen,costs,cap-used)
                            state='full_policy'
                        pred=np.where(ask,expert,local)
                        fp=int(((y==0)&pred).sum()); fn=int(((y==1)&~pred).sum())
                        rows.append(dict(site=site, seed=seed, cap_percent=pct, review_location=place,
                                         method='edgeboost_'+mode, deployment=state, cap_bytes=cap,setup_bytes=used,
                                         total_bytes=used+int(costs[ask].sum()), queried=int(ask.sum()),fp=fp,fn=fn,
                                         balanced_accuracy=1-.5*(fp/(y==0).sum()+fn/(y==1).sum()),
                                         accuracy=float(np.mean(pred==y))))
                        preds.append(pred);masks.append(ask)
        np.savez_compressed(OUT/f'published_edgeboost_{site}.npz',source_path=ids,y=y,expert=expert,
                            application_bytes=costs,predictions=np.asarray(preds,dtype=np.uint8),queries=np.asarray(masks,dtype=np.uint8))
    write('published_edgeboost_runs.csv',rows)
    old={(r['site'],int(r['seed']),int(r['cap_percent']),r['review_location']):r for r in table('joint_budget_runs.csv') if r['method']=='joint'}
    checked=0
    for r in rows:
        if r['method']=='edgeboost_budget_rank' and r['review_location']!='prepaid':
            ref=old[r['site'],r['seed'],r['cap_percent'],r['review_location']]
            assert all(int(ref[k])==r[k] for k in ['fp','fn','total_bytes','setup_bytes'])
            checked+=1
    audit = []
    historical = table('joint_budget_runs.csv')
    for site in ('cornell','kashmir','latvia'):
        earlier = [r for r in historical if r['site']==site]
        keyed = {(int(r['seed']),int(r['cap_percent']),r['review_location']): i for i,r in enumerate(earlier) if r['method']=='joint'}
        now = [r for r in rows if r['site']==site]
        with np.load(OUT/f'published_edgeboost_{site}.npz') as eb, np.load(OUT/f'joint_budget_{site}.npz') as old:
            assert np.array_equal(eb['source_path'],old['source_path'])
            e_p,e_q = eb['predictions'],eb['queries']
            o_p,o_q = old['predictions'],old['queries']
            for i,r in enumerate(now):
                key=(r['seed'],r['cap_percent'],r['review_location'])
                if r['method']!='edgeboost_budget_rank' or key not in keyed:
                    continue
                j=keyed[key]
                dq=int(np.count_nonzero(e_q[i]!=o_q[j]));dp=int(np.count_nonzero(e_p[i]!=o_p[j]))
                assert dq==dp==0
                audit.append(dict(site=site,seed=key[0],cap_percent=key[1],review_location=key[2],
                                  n=len(e_p[i]),changed_queries=dq,changed_decisions=dp))
    write('published_edgeboost_parity.csv', audit)
    print('Exact per-image pairing in',checked,'cold settings and',sum(r['n'] for r in audit),'paired image decisions.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--author-cache',type=Path,help='fetch and replay pinned author code and probability arrays')
    main(parser.parse_args().author_cache)

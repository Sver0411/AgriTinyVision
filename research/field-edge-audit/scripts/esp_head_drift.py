"""Compare frozen-head transfer and refitting under the shipped-scale host approximation."""
import zipfile
import numpy as np
from scipy.stats import spearmanr
from edge_adaptation import fit
from edge_adaptation_wire import normalized,quantized,predict_from_head,pack_head
from evaluate_upload import rank_selection
from joint_budget import ROOT,OUT,FEATURES,dataset,table,write

def aligned(file, names, key):
    with np.load(file) as z:
        lookup={p:i for i,p in enumerate(z['source_path'])}
        return z[key][[lookup[p] for p in names]]


def main():
    cache=ROOT/'data/raw/esp-grid-cache'
    with np.load(OUT/'edge_adaptation_predictions.npz') as z:
        draws=z['train_paths']
    truth={r['source_path']:int(r['y']) for r in table('edge_adaptation_split.csv')}
    refit=[]
    for names in draws:
        q=quantized(aligned(cache/'cornell.npz',names,'integer').astype(np.float32)/32)[0]
        refit.append(pack_head('local_int8',*fit(normalized(q),np.array([truth[p] for p in names]))))
    m=table('edge_adaptation_metrics.csv')
    fees={(int(r['seed']),place):int(r[f'{place}_label_total_bytes']) for r in m
          if r['method']=='local_int8' and r['budget']=='128' for place in ('local','remote')}
    rows=[]
    with zipfile.ZipFile(OUT/'edge_adaptation_heads.zip') as heads:
        for site in ('cornell','kashmir','latvia'):
            names,y,edge_p,server,jpeg,ref=dataset(site)
            codes=aligned(cache/f'{site}.npz',names,'integer')
            approx=codes.astype(np.float32)/32
            xq=quantized(approx)[0]
            if site=='cornell':
                fp32=aligned(FEATURES/'edge_fgvc.npz',names,'activation')
            else:
                fp32=aligned(FEATURES/f'{site}.npz',names,'edge')
            similarities=np.sum(normalized(approx)*normalized(fp32),axis=1)
            base_scores=aligned(cache/f'{site}.npz',names,'prob')
            costs=jpeg+20;cap=int(costs.sum())//4
            collected=[]
            for seed in range(20):
                blob=heads.read(f's{seed:02d}-b128/local_int8.bin')
                candidates={'FP32_reference':ref[seed], 'FP32_head_grid_backbone':predict_from_head('local_int8',blob,xq),
                            'grid_head_grid_backbone':predict_from_head('local_int8',refit[seed],xq)}
                for where in ('local','remote'):
                    setup=fees[seed,where]
                    for mode,score in candidates.items():
                        original_score=ref[seed]
                        if setup<=cap:
                            ask=rank_selection(np.lexsort((names,abs(score))),costs,cap-setup)
                            previous=rank_selection(np.lexsort((names,abs(original_score))),costs,cap-setup)
                            edge=score>=0
                        else:
                            prob=edge_p if mode=='FP32_reference' else base_scores
                            ask=rank_selection(np.lexsort((names,abs(prob-.3))),costs,cap)
                            previous=rank_selection(np.lexsort((names,abs(edge_p-.3))),costs,cap)
                            edge=prob>=.3
                        pred=np.where(ask,server,edge)
                        fp=int(((y==0)&pred).sum());fn=int(((y==1)&~pred).sum())
                        rows.append(dict(site=site,seed=seed,mode=mode,reviewer=where,n=len(y),
                                         feature_cosine_median=float(np.median(similarities)),
                                         feature_cosine_p05=float(np.quantile(similarities,.05)),
                                         unadapted_decision_agreement=float(np.mean((base_scores>=.3)==(edge_p>=.3))),
                                         adapted_decision_agreement=float(np.mean((score>=0)==(original_score>=0))),
                                         margin_spearman=float(spearmanr(score,original_score).statistic),
                                         routing_jaccard=float((ask&previous).sum()/(ask|previous).sum()),
                                         fp=fp,fn=fn,balanced_accuracy=1-.5*(fp/(y==0).sum()+fn/(y==1).sum()),
                                         accuracy=float((pred==y).mean()),actual_setup=setup if setup<=cap else 0,
                                         application_bytes=int(costs[ask].sum())+(setup if setup<=cap else 0)))
                        collected.append((pred,ask))
            np.savez_compressed(OUT/f'esp_head_drift_{site}.npz',source_path=names,feature_cosine=similarities,
                                predictions=np.packbits([x[0] for x in collected],axis=1),queries=np.packbits([x[1] for x in collected],axis=1))
    write('esp_head_drift.csv',rows)
    summary=[]
    for site in ('cornell','kashmir','latvia'):
        for mode in ('FP32_reference','FP32_head_grid_backbone','grid_head_grid_backbone'):
            for where in ('local','remote'):
                rr=[r for r in rows if r['site']==site and r['mode']==mode and r['reviewer']==where]
                summary.append(dict(site=site,mode=mode,reviewer=where,**{k:float(np.mean([r[k] for r in rr])) for k in (
                    'feature_cosine_median','feature_cosine_p05','unadapted_decision_agreement','adapted_decision_agreement',
                    'margin_spearman','routing_jaccard','balanced_accuracy','accuracy','application_bytes')}))
    write('esp_head_drift_summary.csv',summary)


if __name__=='__main__':
    main()

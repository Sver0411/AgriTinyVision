"""Replay Cornell-fitted heads on untouched Kashmir images; compute a JPEG-oracle cost bound on Cornell."""
import csv
import json
import zipfile
from collections import Counter
from pathlib import Path
import numpy as np

from edge_adaptation_wire import normalized, quantized, predict_from_head
from evaluate_upload import confusion, read, write, ROOT, OUT
from oracle_cost import minimum_bytes


def main():
    hold=[r for r in read('edge_adaptation_split.csv') if r['status']=='test']
    edge_main={r['source_path']:r for r in read('fgvc_edge.csv')}
    cloud={(r['source_path'],int(r['long_edge'])):r for r in read('fgvc_backend.csv')}
    spec={r['source_path']:r for r in read('specialist_predictions.csv') if r['dataset']=='FGVC7'}
    paths=[r['source_path'] for r in hold]
    y=np.array([int(r['y']) for r in hold]); edge=np.array([int(edge_main[p]['binary_pred']) for p in paths])
    menu=[(np.array([int(cloud[p,side]['pred_diseased']) for p in paths]), np.array([int(cloud[p,side]['jpeg_bytes']) for p in paths])) for side in (112,224)]
    menu.append((np.array([int(spec[p]['pred_diseased']) for p in paths]),menu[-1][1]))
    balanced={r['source_path']:r for r in read('specialist_balanced.csv') if r['dataset']=='FGVC7'}
    expanded=menu+[(np.array([int(balanced[p]['pred_diseased']) for p in paths]),menu[-1][1])]
    metrics=read('edge_adaptation_metrics.csv')
    bounds=[]
    for r in metrics:
        if r['method']!='local_int8':continue
        cost=minimum_bytes(y,edge,menu,int(r['true_negative']),int(r['true_positive']))
        extended_cost=minimum_bytes(y,edge,expanded,int(r['true_negative']),int(r['true_positive']))
        bounds.append(dict(oracle_with_balanced_head_bytes=extended_cost if extended_cost is not None else 'unreachable',seed=r['seed'],budget=r['budget'],n=len(y),target_balanced_accuracy=r['balanced_accuracy'],
                           oracle_min_jpeg_bytes=cost if cost is not None else 'unreachable',
                           observed_update_bytes_local_review=r['local_label_total_bytes'],
                           observed_update_bytes_remote_review=r['remote_label_total_bytes']))
    assert len(bounds)==60
    write('head_vs_oracle.csv',bounds)
    km=read('transfer_manifest.csv')
    ys=np.array([int(r['y']) for r in km])
    with np.load(ROOT/'data/features/kashmir.npz',allow_pickle=False) as z:
        assert list(z['source_path'])==[r['source_path'] for r in km]
        q,_=quantized(z['edge'])
        features=dict(local_int8=q,local_fp32=normalized(z['edge']),dino=normalized(z['dino']),
                      scalar=np.array([[float(r['edge_score_logit'])] for r in km],dtype=np.float32))
    # Fixed-pool cross-source fingerprints. Matching an earlier asset would invalidate its independence.
    with (ROOT/'data/features/source_fingerprints.csv').open() as f:refs=[(r['file_sha256'],int(r['phash_hex'],16)) for r in csv.DictReader(f)]
    refs.extend((r['file_sha256'],int(r['phash'],16)) for r in edge_main.values())
    external_hits=sum(any(s==r['source_sha256'] or (h^int(r['phash'],16)).bit_count()<=4 for s,h in refs) for r in km)
    within_pairs=sum((int(a['phash'],16)^int(b['phash'],16)).bit_count()<=4 for i,a in enumerate(km) for b in km[:i])
    (OUT/'transfer_screen.json').write_text(json.dumps(dict(n=len(km),reference_rows=len(refs),cross_hits=external_hits,within_pairs=within_pairs,
                        unique_file_hashes=len({r['source_sha256'] for r in km})),indent=2)+'\n')
    with np.load(ROOT/'data/features/dino_plantdoc_head.npz',allow_pickle=False) as h:
        plantdoc_dino=(features['dino']@h['coef'][0]+h['intercept'][0]>=0).astype(int)
    preds=[]; lines=[]
    def record(method,nlab,seed,pred):
        lines.append(dict(method=method,budget=nlab,seed=seed,n=len(ys),**confusion(ys,pred),
                          alternaria_recall=float(pred[[r['source_label']=='Alternaria' for r in km]].mean()),
                          mosaic_recall=float(pred[[r['source_label']=='Apple_Mosaic' for r in km]].mean())))
        preds.append(pred)
    record('frozen',0,-1,np.array([int(r['edge_pred']) for r in km]))
    record('mobile224',0,-1,np.array([int(float(r['mobile_server_p'])>=.5) for r in km]))
    record('plantdoc_dino',0,-1,plantdoc_dino)
    with np.load(ROOT/'data/features/dino_plantdoc_balanced_head.npz',allow_pickle=False) as h:
        weighted=(features['dino']@h['coef'].reshape(-1)+h['intercept'].reshape(-1)[0]>=0).astype(int)
    record('plantdoc_balanced',0,-1,weighted)
    with zipfile.ZipFile(OUT/'edge_adaptation_heads.zip') as archive:
        for m in metrics:
            method=m['method']
            if method not in features:continue
            b=int(m['budget']);s=int(m['seed'])
            blob=archive.read(f's{s:02d}-b{b:03d}/{method}.bin')
            pred=(predict_from_head(method,blob,features[method])>=0).astype(int)
            record(method,b,s,pred)
    assert len(lines)==244
    write('transfer_outcomes.csv',lines)
    np.savez_compressed(OUT/'transfer_predictions.npz',source_path=np.array([r['source_path'] for r in km]),
                        method=np.array([r['method'] for r in lines]), budget=[r['budget'] for r in lines],
                        seeds=[r['seed'] for r in lines], y=ys,pred=np.array(preds))


if __name__=='__main__':
    main()

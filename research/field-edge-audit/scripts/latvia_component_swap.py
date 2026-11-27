"""Cross the source and early-Latvia heads while holding the later-date review budget fixed."""

import csv
import zipfile
import numpy as np
from sklearn.metrics import roc_auc_score

from edge_adaptation import fit
from edge_adaptation_wire import normalized, pack_head, predict_from_head, quantized
from evaluate_upload import rank_selection
from joint_budget import OUT, FEATURES, table, write


def metrics(y, pred):
    fp=int(((y==0)&pred).sum())
    fn=int(((y==1)&~pred).sum())
    sens=1-fn/(y==1).sum()
    spec=1-fp/(y==0).sum()
    return dict(fp=fp,fn=fn,sensitivity=sens,specificity=spec,balanced_accuracy=(sens+spec)/2,
                accuracy=float(np.mean(pred==y)))


def main():
    draw_rows=table('latvia_failure_train_ids.csv')
    dates={r['source_path']:r for r in table('latvia_local_date_split.csv')}
    with np.load(FEATURES/'latvia.npz') as z:
        paths=z['source_path']; labels=z['y']; jpeg=z['jpeg_bytes']
        q=quantized(z['edge'])[0]; dino=normalized(z['dino'])
        source_server=z['dino_balanced_margin'].reshape(-1)
    lookup={p:i for i,p in enumerate(paths)}
    test=np.array([i for i,p in enumerate(paths) if dates[p]['status']=='test'])
    assert len(test)==579 and len(set(paths))==1190
    y=labels[test]
    cost=jpeg[test]+20
    cap=int(cost.sum())//4
    rows=[]; saved=[]; asks=[]; scores=[]; setups=[]
    with zipfile.ZipFile(OUT/'edge_adaptation_heads.zip') as heads:
        for seed in range(20):
            early=sorted([r for r in draw_rows if int(r['seed'])==seed],key=lambda r:int(r['draw_position']))
            train=np.array([lookup[r['source_path']] for r in early])
            assert len(set(train))==128 and not set(test).intersection(train)
            assert max(dates[p]['capture_date'] for p in paths[train]) < min(dates[p]['capture_date'] for p in paths[test])
            edge_head=pack_head('local_int8',*fit(normalized(q[train]),labels[train]))
            edge={
                'cornell':predict_from_head('local_int8',heads.read(f's{seed:02d}-b128/local_int8.bin'),q[test]),
                'early_latvia':predict_from_head('local_int8',edge_head,q[test]),
            }
            w,b=fit(dino[train],labels[train])
            cloud_blob=pack_head('local_fp32',w,b)
            server={'plantdoc':source_server[test],
                    'early_latvia':predict_from_head('local_fp32',cloud_blob,dino[test])}
            # Fit with image label on site: one labelled JPEG record on the cloud.
            cloud_setup=int((jpeg[train]+9).sum())
            local_setup=128*1293+len(edge_head)
            setups.append(dict(seed=seed,n_local_labels=128,local_head_bytes=local_setup,
                               cloud_head_bytes_on_server=len(cloud_blob),labelled_jpeg_upload=cloud_setup,
                               dual_setup_bytes=local_setup+int((jpeg[train]+8).sum()),inference_cap=cap))
            for edge_name,local in edge.items():
                ask=rank_selection(np.lexsort((paths[test],abs(local))),cost,cap)
                for server_name,cloud in server.items():
                    pred=np.where(ask,cloud>=0,local>=0)
                    rows.append(dict(seed=seed,edge=edge_name,cloud=server_name,n=len(test),queried=int(ask.sum()),
                                     inference_bytes=int(cost[ask].sum()),cap_bytes=cap,
                                     **metrics(y,pred),cloud_auroc=float(roc_auc_score(y,cloud)),
                                     edge_auroc=float(roc_auc_score(y,local)),
                                     **{'full_cloud_'+k:v for k,v in metrics(y,cloud>=0).items()}))
                    saved.append(pred);asks.append(ask);scores.append(np.stack([local,cloud]))
    write('latvia_component_swap.csv',rows)
    write('latvia_component_setup.csv',setups)
    np.savez_compressed(OUT/'latvia_component_cases.npz',source_path=paths[test],y=y,application_bytes=cost,
                        predictions=np.array(saved,dtype=np.uint8),queries=np.array(asks,dtype=np.uint8),scores=np.array(scores))
    summary=[]
    for edge in ['cornell','early_latvia']:
        for cloud in ['plantdoc','early_latvia']:
            rr=[r for r in rows if r['edge']==edge and r['cloud']==cloud]
            summary.append(dict(edge=edge,cloud=cloud,n_images=len(test),seeds=len(rr),
                                **{k:float(np.mean([r[k] for r in rr])) for k in (
                                    'fp','fn','balanced_accuracy','accuracy','sensitivity','specificity','queried','inference_bytes',
                                    'cloud_auroc','edge_auroc','full_cloud_balanced_accuracy','full_cloud_sensitivity','full_cloud_specificity')}))
    write('latvia_component_summary.csv',summary)
    for r in summary:
        print(r['edge'],r['cloud'],r['balanced_accuracy'],r['fp'],r['fn'],r['cloud_auroc'],r['full_cloud_balanced_accuracy'])


if __name__=='__main__':
    main()

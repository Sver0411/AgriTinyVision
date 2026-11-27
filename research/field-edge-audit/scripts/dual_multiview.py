"""Fixed three-view remote diagnosis and one calibration-selected threshold for both errors."""
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import numpy as np
import onnxruntime as ort
from PIL import Image,ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from edge_model import ROOT
from field_specialist import checked_model, MODEL_DIR
from server_backend import jpeg_transport
from edge_adaptation_wire import normalized

FEAT=ROOT/'data/features'
OUT=ROOT/'results'


def table(p):
    with p.open() as f:return list(csv.DictReader(f))


def views(payload,cfg):
    with Image.open(io.BytesIO(payload)) as image:
        im=image.convert('RGB')
    w,h=im.size;m=min(w,h)
    shots=[im,im.crop((0,0,m,m)),im.crop((w-m,h-m,w,h))]
    x=np.stack([np.asarray(i.resize((224,224),Image.Resampling.BICUBIC),dtype=np.float32) for i in shots])*np.float32(cfg['rescale_factor'])
    return ((x-np.array(cfg['image_mean'],dtype=np.float32))/np.array(cfg['image_std'],dtype=np.float32)).transpose(0,3,1,2)


def extract(path):
    import pyarrow.parquet as pq
    cfg=checked_model()
    options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
    server=ort.InferenceSession(str(MODEL_DIR/'model.onnx'),options,providers=['CPUExecutionProvider'])
    fgvc=table(OUT/'edge_adaptation_split.csv')
    ids=[r['source_path'] for r in fgvc if r['status'] in ('label_pool','test')]
    kash=table(OUT/'transfer_manifest.csv')
    ks={r['image']['path']:r['image']['bytes'] for r in pq.read_table(path).to_pylist()}
    for name,names in [('cornell',ids),('kashmir',[r['source_path'] for r in kash])]:
        features=[];sizes=[];fingerprints=[]
        for idx in range(0,len(names),4):
            tensors=[]
            for p in names[idx:idx+4]:
                content=(ROOT/'data/raw/fgvc7'/p).read_bytes() if name=='cornell' else ks[p]
                with Image.open(io.BytesIO(content)) as f:img=ImageOps.exif_transpose(f).convert('RGB')
                payload=jpeg_transport(img,448)
                sizes.append(len(payload));fingerprints.append(hashlib.sha256(payload).hexdigest())
                tensors.append(views(payload,cfg))
            out=server.run(['last_hidden_state'],{'pixel_values':np.ascontiguousarray(np.concatenate(tensors))})[0][:,0,:].reshape(len(tensors),3,384)
            features.extend(np.concatenate([out[:,0,:],out[:,1:].mean(axis=1),out[:,1:].max(axis=1)],axis=1))
            if idx%240==0: print(name,idx,flush=True)
        np.savez_compressed(FEAT/f'dual_mv_{name}.npz',source_path=names,features=np.array(features),jpeg_bytes=sizes,jpeg_sha256=fingerprints)


def counts(y,p):
    return int(((p==1)&(y==0)).sum()),int(((p==0)&(y==1)).sum())


def fit_evaluate():
    splits={r['source_path']:r for r in table(OUT/'edge_adaptation_split.csv')}
    edge={r['source_path']:int(r['binary_pred']) for r in table(OUT/'fgvc_edge.csv')}
    kash={r['source_path']:r for r in table(OUT/'transfer_manifest.csv')}
    with np.load(FEAT/'dual_mv_cornell.npz') as z:
        names=z['source_path'];x=normalized(z['features']);sizes=z['jpeg_bytes']
    bank=sorted([i for i,p in enumerate(names) if splits[p]['status']=='label_pool'],
                key=lambda i:hashlib.sha256(('multi-v1/'+names[i]).encode()).hexdigest())
    fit,cal=bank[:300],bank[300:]
    lab=lambda ids:np.array([int(splits[names[i]]['y']) for i in ids])
    std=StandardScaler().fit(x[fit])
    model=LogisticRegression(C=1,class_weight='balanced',max_iter=3000).fit(std.transform(x[fit]),lab(fit))
    score=model.decision_function(std.transform(x));base=np.array([edge[p] for p in names]);y=lab(cal)
    fp0,fn0=counts(y,base[cal]);best=None
    for t in np.r_[np.unique(score[cal]),np.nextafter(max(score[cal]),np.inf)]:
        fp,fn=counts(y,(score[cal]>=t).astype(int))
        reductions=((fp0-fp)/max(1,fp0),(fn0-fn)/max(1,fn0))
        key=(min(reductions),sum(reductions),-float(t))
        if best is None or key>best[0]:best=key,float(t)
    t=best[1];fallback=best[0][0]<=0
    policy=dict(threshold=t,uses_frozen_edge=fallback,calibration_fpfn=[fp0,fn0],candidate_calibration_fpfn=counts(y,score[cal]>=t),
                n_fit=len(fit),n_cal=len(cal),calibration_jpeg_bytes=int(sum(sizes[bank])),fit_ids_sha256=hashlib.sha256('\n'.join(names[fit]).encode()).hexdigest())
    np.savez_compressed(FEAT/'dual_mv_head.npz',coef=model.coef_[0],intercept=model.intercept_,std_mean=std.mean_,std_scale=std.scale_)
    (OUT/'dual_multiview_rule.json').write_text(json.dumps(policy,indent=2)+'\n')
    test=[i for i,p in enumerate(names) if splits[p]['status']=='test']
    with np.load(FEAT/'dual_mv_kashmir.npz') as z:
        kn=z['source_path'];ks=model.decision_function(std.transform(normalized(z['features'])));ky=np.array([int(kash[p]['y']) for p in kn]);ke=np.array([int(kash[p]['edge_pred']) for p in kn]);kb=z['jpeg_bytes']
    groups=[('Cornell1353',names[test],score[test],lab(test),base[test],sizes[test]),('Kashmir384',kn,ks,ky,ke,kb)]
    rows=[];summary=[]
    for group,ids,scores,y,old,sz in groups:
        p=old if fallback else (scores>=t).astype(int)
        fp,fn=counts(y,p);op,on=counts(y,old)
        summary.append(dict(cohort=group,n=len(y),original_fp=op,original_fn=on,new_fp=fp,new_fn=fn,both_lower=fp<op and fn<on,jpeg_bytes=int(sum(sz)) if not fallback else 0))
        rows.extend(dict(cohort=group,source_path=i,y=int(a),original=int(o),remote_score=float(s),decision=int(d),jpeg_bytes=int(b)) for i,a,o,s,d,b in zip(ids,y,old,scores,p,sz))
    with (OUT/'dual_multiview_predictions.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    (OUT/'dual_multiview_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(summary)


if __name__=='__main__':
    if len(sys.argv)>1:extract(Path(sys.argv[1]))
    fit_evaluate()

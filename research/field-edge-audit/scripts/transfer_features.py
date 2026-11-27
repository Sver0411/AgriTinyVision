"""Collect fixed out-of-source inputs and run the same exported representations on each."""
import csv
import hashlib
import io
import json
import shutil
import urllib.request
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import pyarrow.parquet as pq
from onnx import helper, TensorProto
from PIL import Image,ImageOps
from scipy.fft import dctn
from edge_model import ROOT, EdgeModel
from server_backend import jpeg_transport, model_tensor, WEIGHT_HASH
from field_specialist import tensor, checked_model, MODEL_DIR

REV='38e4c0600bbf962f2379bf1574b8049c801e089d'
URL='https://huggingface.co/datasets/Project-AgML/apple_leaf_disease_classification/resolve/'+REV+'/data/train-00000-of-00001.parquet'
CLASS_NAMES=('Alternaria','Apple_Mosaic','Healthy')


def session(graph):
    config=ort.SessionOptions()
    config.intra_op_num_threads=2;config.inter_op_num_threads=1
    return ort.InferenceSession(graph,config,providers=['CPUExecutionProvider'])


def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('parquet',type=Path)
    path=ap.parse_args().parquet
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        with urllib.request.urlopen(URL,timeout=180) as src, path.open('wb') as dst:
            shutil.copyfileobj(src,dst)
    data=pq.read_table(path).to_pylist()
    if len(data)!=5612:raise ValueError('unexpected pinned data')
    rank=lambda r:hashlib.sha256(('field-edge-v1/'+r['image']['path']).encode()).hexdigest()
    rows=[r for cl in range(3) for r in sorted((v for v in data if v['label']==cl),key=rank)[:128]]
    if len({v['image']['path'] for v in rows})!=384:raise ValueError('duplicate IDs')
    raw=ROOT/'data/raw'
    mobile=EdgeModel()
    graph=onnx.load(raw/'tinyvision.onnx')
    final=graph.graph.node[-1]
    graph.graph.output.append(helper.make_tensor_value_info(final.input[0],TensorProto.FLOAT,[1,1280]))
    edge_session=session(graph.SerializeToString())
    crop_model=raw/'backend-model/model.onnx'
    if hashlib.sha256(crop_model.read_bytes()).hexdigest()!=WEIGHT_HASH:raise ValueError('MobileNet conversion mismatch')
    remote=session(str(crop_model))
    cfg=json.loads((crop_model.parent/'config.json').read_text())
    healthy=np.array([cfg['id2label'][str(i)].startswith('Healthy ') for i in range(38)])
    cfg_dino=checked_model()
    dino=session(str(MODEL_DIR/'model.onnx'))
    tmp=raw/'transfer_jpegs';tmp.mkdir(exist_ok=True)
    edge_values=[]; dino_values=[]; manifest=[]
    for r in rows:
        content=r['image']['bytes']
        with Image.open(io.BytesIO(content)) as f:
            im=ImageOps.exif_transpose(f).convert('RGB')
        raw_image=np.array(im.resize((160,160),Image.Resampling.BILINEAR),dtype=np.uint8)
        x=((raw_image.astype(np.float32)/255.-mobile.mean)/mobile.std).transpose(2,0,1).copy()[None]
        logits,act=edge_session.run(None,{mobile.input_name:x})
        edge_values.append(act[0])
        jpeg=jpeg_transport(im,224);h=hashlib.sha256(jpeg).hexdigest();file=tmp/(h+'.jpg')
        file.write_bytes(jpeg)
        with Image.open(io.BytesIO(jpeg)) as jp:
            rx=model_tensor(jp.convert('RGB'))[None].copy()
        z=remote.run(None,{remote.get_inputs()[0].name:rx})[0].flatten()
        w=np.exp(z-z.max());p=float(w[~healthy].sum()/w.sum())
        small=np.asarray(im.convert('L').resize((32,32)),dtype=np.float64)
        spectrum=dctn(small,norm='ortho')[:8,:8].flatten()
        name=r['image']['path']
        row=dict(source_path=name,y=int(r['label']!=2),source_label=CLASS_NAMES[r['label']],
                 source_sha256=hashlib.sha256(content).hexdigest(),phash=np.packbits(spectrum>np.median(spectrum[1:])).tobytes().hex(),
                 jpeg_sha256=h,jpeg_bytes=len(jpeg),edge_score_logit=float(logits[0,1]-logits[0,0]),edge_pred=int(logits[0,1]-logits[0,0]>=np.log(.3/.7)),mobile_server_p=p)
        dt=tensor(file,row,cfg_dino)[None]
        dino_values.append(dino.run(['last_hidden_state'],{'pixel_values':dt})[0][0,0])
        manifest.append(row)
    with (ROOT/'results/transfer_manifest.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(manifest[0]));writer.writeheader();writer.writerows(manifest)
    np.savez_compressed(ROOT/'data/features/kashmir.npz',source_path=[r['source_path'] for r in manifest],edge=np.array(edge_values),dino=np.array(dino_values))
    record=dict(revision=REV,url=URL,parquet_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),parquet_rows=len(data),sampled=len(rows),
                sampling='Lowest SHA256(field-edge-v1/+filename) within each of three source classes, 128 each')
    (ROOT/'data/kashmir_source.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':
    main()

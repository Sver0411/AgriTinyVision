"""Host arithmetic replay with the released ESP-PPQ tensor scales and S3 rounding.

The crop model remains immutable. The host graph exposes its pooled activation.
This does not invoke the ESP-DL kernels or measure MCU latency or energy.
"""
import argparse
import csv
import hashlib
import io
import json
import re
from pathlib import Path
import zipfile

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort
from PIL import Image, ImageOps

from edge_model import ROOT, MODEL_SHA256
from joint_budget import write, table, OUT, FEATURES

CONFIG_HASH='b10bc87a05c8efadb7afd7abcb8f23e79c0c1f7d04c168a7f0b92d1c3e9634bd'


def graph(path):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=CONFIG_HASH:
        raise ValueError('Wrong ESP-PPQ export')
    q=json.loads(path.read_text()); source=ROOT/'data/raw/tinyvision.onnx'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==MODEL_SHA256
    model=onnx.load(source); g=model.graph
    weights={x.name:x for x in g.initializer}
    sequence=[]; constants={}
    def constant(x):
        if x not in constants:
            name='emul_const_'+str(len(constants))
            constants[x]=name
            g.initializer.append(numpy_helper.from_array(np.array(x,dtype=np.float32),name))
        return constants[x]
    def scale_of(config):
        return q['values'][str(config['dominator'])]['scale']
    def grid(value,destination,config):
        scale=constant(scale_of(config))
        for op,inputs,out in [('Div',[value,scale],destination+'_scaled'),
                              ('Add',[destination+'_scaled',constant(.5)],destination+'_shift'),
                              ('Floor',[destination+'_shift'],destination+'_round'),
                              ('Clip',[destination+'_round',constant(config['quant_min']),constant(config['quant_max'])],destination+'_clamp'),
                              ('Mul',[destination+'_clamp',scale],destination)]:
            sequence.append(helper.make_node(op,inputs,[out]))
    grid('image','image_quant',q['configs'][g.node[0].name]['image'])
    treated=set()
    for node in g.node:
        node=onnx.NodeProto.FromString(node.SerializeToString())
        cfg=q['configs'].get(node.name,{})
        for i,v in enumerate(node.input):
            if v=='image':
                node.input[i]='image_quant'
            elif v in weights and v in cfg and v not in treated:
                spec=cfg[v]; scale=scale_of(spec); orig=numpy_helper.to_array(weights[v]).astype(np.float64)
                val=np.clip(np.floor(orig/scale+.5),spec['quant_min'],spec['quant_max'])*scale
                weights[v].CopyFrom(numpy_helper.from_array(val.astype(np.float32),v))
                treated.add(v)
        raw=node.output[0]
        # OVERLAPPED values belong to a fused operator in the compiler graph.
        active=raw in cfg and cfg[raw]['state']=='ACTIVATED'
        if active:
            node.output[0]=raw+'_before_grid'
        sequence.append(node)
        if active:
            grid(node.output[0],raw,cfg[raw])
    assert len(treated)==106
    del g.node[:];g.node.extend(sequence)
    g.output.append(helper.make_tensor_value_info('/Flatten_output_0', TensorProto.FLOAT,[1,1280]))
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    return ort.InferenceSession(model.SerializeToString(),options,providers=['CPUExecutionProvider'])


def prepare(im,meta):
    rgb=np.asarray(ImageOps.exif_transpose(im).convert('RGB').resize((160,160),Image.Resampling.BILINEAR),dtype=np.uint8)
    data=(rgb.astype(np.float32)/np.float32(255)-np.array(meta['mean'],np.float32))/np.array(meta['std'],np.float32)
    return np.ascontiguousarray(data.transpose(2,0,1)[None]),hashlib.sha256(rgb.tobytes()).hexdigest()


def dataset_images(site):
    if site=='board':
        with (ROOT/'data/board_pairs.csv').open() as f:rows=list(csv.DictReader(f))
        for r in rows:
            with Image.open(ROOT/'data/raw/plantdoc'/r['source_id']) as im:
                yield r['source_id'],im,r['rgb_sha256'],float(r['board_int8_score'])
    elif site=='cornell':
        for r in table('fgvc_edge.csv'):
            with Image.open(ROOT/'data/raw/fgvc7'/r['source_path']) as im:
                yield r['source_path'],im,r['rgb160_sha256'],None
    elif site=='latvia':
        with np.load(FEATURES/'latvia.npz') as z,zipfile.ZipFile(ROOT/'data/raw/latvia.zip') as source:
            for p,d in zip(z['source_path'],z['rgb160_sha256'],strict=True):
                with Image.open(io.BytesIO(source.read(str(p)))) as im:
                    yield p,im,d,None
    elif site=='kashmir':
        import pyarrow.parquet as pq
        images={r['image']['path']:r['image']['bytes'] for r in pq.read_table(ROOT/'data/raw/kashmir.parquet').to_pylist()}
        for r in table('transfer_manifest.csv'):
            p=r['source_path']; content=images[p]
            assert hashlib.sha256(content).hexdigest()==r['source_sha256']
            with Image.open(io.BytesIO(content)) as im:
                yield p,im,None,None


def main(directory,sites):
    session=graph(directory/'tinyvision.json')
    meta=json.loads((ROOT/'data/raw/tinyvision-metadata.json').read_text())['preprocessing']
    fixture=(directory/'tinyvision.info').read_text()
    sections=re.split(r'test (?:inputs|outputs) value:',fixture)
    if len(sections)!=3:
        raise ValueError('Invalid embedded ESP-PPQ fixture')
    def values(s):
        numbers=s.split('array([',1)[1].split(']',1)[0]
        return np.fromstring(numbers,sep=',',dtype=np.int32)
    input_data=values(sections[1]).reshape(1,160,160,3).transpose(0,3,1,2).astype(np.float32)/64
    expect=values(sections[2])[:2].tolist()
    actual=(session.run(None,{'image':input_data})[0].ravel()*16).astype(int).tolist()
    print('embedded fixture output: published',expect,'host',actual,flush=True)
    out_cache=ROOT/'data/raw/esp-grid-cache';out_cache.mkdir(exist_ok=True)
    for site in sites:
        paths=[];activations=[];scores=[];board=[]
        for p,im,d,board_score in dataset_images(site):
            x,digest=prepare(im,meta)
            if d is not None and d!=digest:
                raise ValueError('Image mismatch: '+str(p))
            logits,hidden=session.run(None,{'image':x})
            if not np.all(hidden*32==np.rint(hidden*32)):
                raise ValueError('Pooled features did not match the published 2^-5 grid')
            prob=float(1/(1+np.exp(-np.diff(logits.ravel()).item())))
            paths.append(p);activations.append(np.rint(hidden[0]*32).astype(np.int8));scores.append(prob)
            if board_score is not None:
                board.append(dict(source_path=p,board_score=board_score,replay_score=prob,
                                  hard_agrees=int((prob>=.3)==(board_score>=.3)),absolute_probability_error=abs(prob-board_score)))
        np.savez_compressed(out_cache/f'{site}.npz',source_path=paths,integer=activations,prob=scores)
        if board:
            write('esp_grid_board.csv',board)
            z=dict(embedded_fixture_expected_q=expect,embedded_fixture_observed_q=actual,
                   board_n=len(board),board_hard_agreements=sum(r['hard_agrees'] for r in board),
                   median_abs_score_error=float(np.median([r['absolute_probability_error'] for r in board])),
                   max_abs_score_error=max(r['absolute_probability_error'] for r in board),config_sha256=CONFIG_HASH)
            (OUT/'esp_grid_anchor.json').write_text(json.dumps(z,indent=2)+'\n')
            print(z,flush=True)
        print('Cached',site,len(paths),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('release_dir',type=Path)
    parser.add_argument('sites',nargs='*',default=['board','cornell','kashmir','latvia'])
    args=parser.parse_args();main(args.release_dir,args.sites)

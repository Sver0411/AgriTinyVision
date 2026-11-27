"""Offline PlantVillage inference on transmitted JPEG bytes, using a frozen ONNX checkpoint."""
import argparse
import csv
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import subprocess

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
REV = '5467f6012d78d1c446145d5f582da6096f852ae8'
MODEL_REV = '95110a1ffcc188d1b5a36524d52e303e40c2403d'
WEIGHT_HASH = 'c209f96f0a87a265b0ff6765c2e84d48038a7d8b8692d7629445b4513c344eb7'
TARGETS = {'apple', 'bell_pepper', 'potato', 'tomato'}
CROP_CLASSES = {'apple': range(0,4), 'bell_pepper': range(18,20), 'potato': range(20,23), 'tomato': range(28,38)}
HEALTHY_DIRS = {'apple leaf','bell_pepper leaf','blueberry leaf','cherry leaf','peach leaf','raspberry leaf','soyabean leaf','strawberry leaf','tomato leaf','grape leaf'}


def plantdoc_images(directory):
    """Read Git objects directly to preserve case-distinct source filenames on macOS."""
    files = subprocess.check_output(['git','-C',str(directory),'ls-tree','-r','-z',REV,'--','test','train'])
    proc = subprocess.Popen(['git','-C',str(directory),'cat-file','--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        for entry in files.split(b'\0'):
            if not entry:
                continue
            head, name = entry.split(b'\t',1)
            source_path = name.decode('utf-8')
            split, label, _ = source_path.split('/',2)
            crop = label.lower().split()[0]
            if split != 'test' and crop not in TARGETS:
                continue
            if Path(source_path).suffix.lower() not in {'.jpg','.jpeg','.png'}:
                continue
            oid = head.split()[-1]
            proc.stdin.write(oid+b'\n'); proc.stdin.flush()
            returned, kind, length = proc.stdout.readline().split()
            assert returned == oid and kind == b'blob'
            encoded = proc.stdout.read(int(length)); assert proc.stdout.read(1) == b'\n'
            yield dict(dataset='PlantDoc',source_path=source_path,split=split,source_label=label,crop=crop,
                       target_crop=int(crop in TARGETS),y=int(label.lower() not in HEALTHY_DIRS)), encoded
    finally:
        proc.stdin.close(); proc.stdout.close(); proc.wait()


def manifest_images(manifest, root):
    """Manifest columns: source_path, split, source_label, crop, y (0 healthy; 1 diseased), dataset."""
    with manifest.open(newline='',encoding='utf-8') as f:
        for row in csv.DictReader(f):
            info = {k:row[k] for k in ('dataset','source_path','split','source_label','crop','y')}
            info['target_crop'] = int(info['crop'] in TARGETS)
            if int(info['y']) not in (0,1):
                raise ValueError(f'Unsupported health label: {info}')
            yield info,(root/info['source_path']).read_bytes()


def fgvc_images(root):
    """Keep Cornell's original one-hot diagnosis for the 2020 labeled train split."""
    classes=('healthy','multiple_diseases','rust','scab')
    with (root/'train.csv').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            labels=[c for c in classes if int(row[c])==1]
            if len(labels)!=1: raise ValueError(f'Invalid diagnosis for {row["image_id"]}')
            label=labels[0]; name='images/'+row['image_id']+'.jpg'
            yield dict(dataset='FGVC7',source_path=name,split='train',source_label=label,crop='apple',target_crop=1,y=int(label!='healthy')), (root/name).read_bytes()


def model_tensor(image):
    # Exact processor configuration: bilinear shorter edge 256; centered 224 crop; RGB [-1,1].
    w,h = image.size
    new_size = (256, int(256*h/w)) if w<=h else (int(256*w/h),256)
    image = image.resize(new_size, Image.Resampling.BILINEAR)
    x,y=(new_size[0]-224)//2,(new_size[1]-224)//2
    value=np.asarray(image.crop((x,y,x+224,y+224)),dtype=np.float32)/np.float32(255.)
    return ((value-np.float32(.5))/np.float32(.5)).transpose(2,0,1)


def jpeg_transport(image, edge):
    image=image.copy(); image.thumbnail((edge,edge),Image.Resampling.LANCZOS)
    buffer=BytesIO(); image.save(buffer,'JPEG',quality=75,subsampling=2,optimize=False,progressive=False)
    return buffer.getvalue()


def run(args):
    model=args.model_dir/'model.onnx'
    if sha256(model.read_bytes()).hexdigest()!=WEIGHT_HASH:
        raise ValueError('Unknown model revision')
    config=json.loads((args.model_dir/'config.json').read_text())
    labels=[config['id2label'][str(k)] for k in range(38)]; healthy=np.asarray([name.startswith('Healthy ') for name in labels])
    options=ort.SessionOptions(); options.intra_op_num_threads=3; options.inter_op_num_threads=1
    runtime=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
    args.csv.parent.mkdir(parents=True,exist_ok=True); args.payloads.mkdir(parents=True,exist_ok=True)
    if args.fgvc:
        images=fgvc_images(args.images)
    elif args.manifest:
        images=manifest_images(args.manifest,args.images)
    else:
        images=plantdoc_images(args.images)
    rows=[]; tensors=[]; writer=None; count=0
    def flush(out):
        nonlocal writer,count
        logits=runtime.run(['logits'], {'pixel_values':np.stack(tensors)})[0].astype(np.float64)
        exp=np.exp(logits-logits.max(axis=1,keepdims=True)); probabilities=exp/exp.sum(axis=1,keepdims=True)
        if not np.isfinite(probabilities).all():
            raise ValueError('Nonfinite posterior')
        for row,z,p in zip(rows,logits,probabilities,strict=True):
            top=int(p.argmax()); score=float(p[~healthy].sum())
            ids=CROP_CLASSES.get(row['crop']); conditioned=''
            if ids is not None:
                ids=list(ids); e=np.exp(z[ids]-np.max(z[ids])); conditioned=float(e[~healthy[ids]].sum()/e.sum())
            row.update(p_healthy=1-score,p_diseased=score,pred_diseased=int(score>=.5),top1_id=top,top1_label=labels[top],p_top1=float(p[top]),
                       p_diseased_known_crop=conditioned)
            if writer is None:
                writer=csv.DictWriter(out,fieldnames=list(row)); writer.writeheader()
            writer.writerow(row);count+=1
        tensors.clear();rows.clear()
    with args.csv.open('w',encoding='utf-8',newline='') as out:
        for info,original in images:
            with Image.open(BytesIO(original)) as src:
                image=ImageOps.exif_transpose(src).convert('RGB')
            original_hash=sha256(original).hexdigest()
            for edge in (224,112):
                encoded=jpeg_transport(image,edge); digest=sha256(encoded).hexdigest(); file=args.payloads / (digest+'.jpg')
                if not file.exists(): file.write_bytes(encoded)
                with Image.open(BytesIO(encoded)) as payload:
                    rows.append(dict(**info,source_sha256=original_hash,original_width=image.width,original_height=image.height,
                                     long_edge=edge,jpeg_quality=75,payload_width=payload.width,payload_height=payload.height,
                                     jpeg_bytes=len(encoded),jpeg_sha256=digest))
                    tensors.append(model_tensor(payload.convert('RGB')))
                if len(rows)>=args.batch:flush(out)
        if rows:flush(out)
    print(f'{count} decoded-JPEG predictions written to {args.csv}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fgvc',action='store_true',help='Read Cornell Plant Pathology 2020 train.csv and images/*.jpg')
    parser.add_argument('--images',type=Path,default=ROOT/'data/raw/plantdoc'); parser.add_argument('--manifest',type=Path)
    parser.add_argument('--model-dir',type=Path,default=ROOT/'data/raw/backend-model'); parser.add_argument('--batch',type=int,default=12)
    parser.add_argument('--csv',type=Path,default=ROOT/'results/backend_predictions.csv'); parser.add_argument('--payloads',type=Path,default=ROOT/'data/backend_jpegs')
    run(parser.parse_args())

"""Audit saved JPEG lengths, posterior consistency, and paired offline health predictions."""
from collections import Counter
from pathlib import Path
import argparse
import csv
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def metrics(rows):
    y = np.array([int(r['y']) for r in rows]); s=np.array([float(r['p_diseased']) for r in rows]); h=s>=0.5
    y1,y0 = s[y==1],s[y==0]; pairs=y1[:,None]-y0[None,:]
    costs=np.array([int(r['jpeg_bytes']) for r in rows]); n=len(rows)
    return dict(n=n,healthy=int((y==0).sum()),diseased=int(y.sum()),tn=int(((y==0)&~h).sum()),fp=int(((y==0)&h).sum()),
                fn=int(((y==1)&~h).sum()),tp=int(((y==1)&h).sum()),accuracy=float((h==y).mean()),
                auroc=float((pairs>0).mean()+0.5*(pairs==0).mean()),jpeg_bytes_total=int(costs.sum()),
                jpeg_bytes_mean=float(costs.mean()),jpeg_bytes_median=float(np.median(costs)))


def verify(rows, payloads):
    pairs=Counter()
    for r in rows:
        key=(r['dataset'],r['source_path']); pairs[key]+=1
        payload=(payloads/(r['jpeg_sha256']+'.jpg')).read_bytes()
        assert hashlib.sha256(payload).hexdigest()==r['jpeg_sha256'] and len(payload)==int(r['jpeg_bytes'])
        assert float(r['p_healthy'])+float(r['p_diseased']) == 1.0 and 0.0<=float(r['p_diseased'])<=1.0
        assert int(r['pred_diseased']) == int(float(r['p_diseased'])>=0.5)
    assert set(pairs.values())=={2}


def main(root):
    plantdoc=read_csv(root/'results/backend_predictions.csv');fgvc=read_csv(root/'results/fgvc_backend.csv');board=read_csv(root/'data/board_pairs.csv')
    verify(plantdoc,root/'data/backend_jpegs');verify(fgvc,root/'data/fgvc_backend_jpegs')
    out={'definition':'y=1 denotes disease; prediction uses unconditional 38-class disease mass >= 0.5', 'groups':{},'matched_board_60':{}}
    for size in ['224','112']:
        test=[r for r in plantdoc if r['split']=='test' and r['long_edge']==size]
        for name,rows in [('plantdoc_official_test_all',test),('plantdoc_official_test_target4',[r for r in test if int(r['target_crop'])]),('fgvc7_labeled',[r for r in fgvc if r['long_edge']==size])]:
            out['groups'].setdefault(name,{})[size]=metrics(rows)
        by_path={r['source_path']:r for r in test};joint=Counter()
        for b in board:
            r=by_path[b['source_id']];assert r['source_sha256']==b['source_file_sha256']; disease=int(b['true_health']=='diseased');assert int(r['y'])==disease
            local=int(b['board_int8_binary']=='SUSPECT')==disease;remote=int(r['pred_diseased'])==disease
            joint[('both_correct' if remote else 'board_only_correct') if local else ('backend_only_correct' if remote else 'both_wrong')]+=1
        out['matched_board_60'][size]=dict(joint)
    (root/'results/backend_audit.json').write_text(json.dumps(out,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);main(p.parse_args().root)

"""Extract original competition IDs from a versioned mirror and check against separate labels."""
import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = 'https://api.kaggle.com/v1/datasets/download/piantic/plantpathology-apple-dataset?datasetVersionNumber=1'
LABELS = 'https://api.kaggle.com/v1/datasets/download/jagadeeshthogata/plant-pathology-2020-fgvc7/train.csv'
CLASSES = ('healthy', 'multiple_diseases', 'rust', 'scab')


def read_url(url):
    req = urllib.request.Request(url, headers={'User-Agent':'kaggle-api-v1.7.4.5','Accept':'application/json'})
    return urllib.request.urlopen(req, timeout=180)


def labels(content):
    rows = csv.DictReader(io.StringIO(content.decode('utf-8-sig')))
    out = {}
    for row in rows:
        name = row['image_id']
        if re.fullmatch(r'Train_[0-9]+', name):
            if name in out:
                raise ValueError('duplicate image ID')
            values = tuple(int(row[k]) for k in CLASSES)
            if sum(values) != 1 or set(values) - {0,1}:
                raise ValueError('invalid source label')
            out[name] = values
    if set(out) != {f'Train_{i}' for i in range(1821)}:
        raise ValueError('incomplete release ID set')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('archive', type=Path, help='path for a local, reusable Kaggle zip')
    ap.add_argument('--label-file', type=Path)
    args = ap.parse_args()
    if not args.archive.exists():
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        temp = args.archive.with_suffix('.part')
        with read_url(ARCHIVE) as src, temp.open('wb') as dst:
            shutil.copyfileobj(src, dst)
        temp.replace(args.archive)
    ref = args.label_file.read_bytes() if args.label_file else read_url(LABELS).read()
    official_ids = labels(ref)
    out = ROOT / 'data/raw/fgvc7'
    (out/'images').mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as z:
        label_member = next(p for p in z.namelist() if Path(p).name == 'train.csv')
        if labels(z.read(label_member)) != official_ids:
            raise ValueError('the reference and mirror class labels differ')
        members = {Path(p).stem:p for p in z.namelist() if re.fullmatch(r'Train_[0-9]+\.jpg', Path(p).name)}
        if set(members) != set(official_ids):
            raise ValueError('image set and primary class labels differ')
        for name,p in members.items():
            (out/'images'/f'{name}.jpg').write_bytes(z.read(p))
    (out/'train.csv').write_bytes(ref)
    src = dict(competition='https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7',
               archive=ARCHIVE, label_crosscheck=LABELS, n=len(official_ids),
               archive_sha256=hashlib.file_digest(args.archive.open('rb'),'sha256').hexdigest(),
               labels_sha256=hashlib.sha256(ref).hexdigest())
    (ROOT/'data/fgvc_source.json').write_text(json.dumps(src,indent=2)+'\n')


if __name__ == '__main__':
    main()

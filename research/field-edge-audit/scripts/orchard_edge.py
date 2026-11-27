"""Apply frozen FP32 reference to release-labelled orchard images and record fingerprints."""
import csv
import hashlib
from edge_model import EdgeModel, ROOT, read_rgb
from PIL import Image
import numpy as np
from scipy.fft import dctn


def main():
    root = ROOT / 'data/raw/fgvc7'
    model = EdgeModel()
    rows = []
    for record in csv.DictReader((root/'train.csv').open()):
        path = 'images/' + record['image_id'] + '.jpg'
        image = read_rgb(root / path)
        small = np.asarray(image.convert('L').resize((32,32)), dtype=float)
        spectrum = dctn(small, type=2, norm='ortho')[:8,:8].flatten()
        hash_bits = np.packbits(spectrum > np.median(spectrum[1:]))
        row = dict(source_path=path, source_label=next(k for k,v in record.items() if k!='image_id' and v=='1'),
                   true_diseased=1-int(record['healthy']), file_sha256=hashlib.sha256((root/path).read_bytes()).hexdigest(),
                   phash=hash_bits.tobytes().hex(), **model.run_rgb(image))
        rows.append(row)
    out = ROOT/'results/fgvc_edge.csv'
    with out.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__=='__main__':
    main()

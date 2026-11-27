"""Extract and validate the pinned MobileNetV4 graph on the original image bytes."""
import argparse
import csv
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
from PIL import Image, ImageOps
import pyarrow.parquet as pq

from field_specialist import tensor_from_image
from server_backend import plantdoc_images

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
CACHE = RAW / "second_backbone"
REVISION = "3ba07f12712fa58fd6b3d661f9909c9e332c5005"
BASE = f"https://huggingface.co/onnx-community/mobilenetv4_conv_small.e2400_r224_in1k/resolve/{REVISION}/"
HASHES = {
    "onnx/model.onnx": "ca5b8dbf490f54e83c5d72f787821565447ea94347935f6b834cf0a4af0e8d37",
    "preprocessor_config.json": "21bb2bccbd790ded22ca9ac77758578bae4d90d8fc578a612097bd38e9d40b2f",
    "config.json": "ae99506e66f8e19bc73809e5b4c3ae31fc7c2bca53752dc7cb89c1e47831a335",
}


def checked_model(threads):
    CACHE.mkdir(parents=True, exist_ok=True)
    for name, expected in HASHES.items():
        path = CACHE / Path(name).name
        if not path.exists():
            with urlopen(BASE + name, timeout=90) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"Downloaded file mismatch: {name}")
            path.write_bytes(data)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Cached file mismatch: {path}")
    model = onnx.load(CACHE / "model.onnx")
    last = model.graph.node[-1]
    if last.op_type != "Gemm" or list(last.input[1:]) != ["classifier.weight", "classifier.bias"]:
        raise ValueError("Unknown final classifier")
    params = {x.name: numpy_helper.to_array(x) for x in model.graph.initializer if x.name in last.input}
    weights, bias = params[last.input[1]], params[last.input[2]]
    hidden = last.input[0]
    model.graph.output.append(helper.make_tensor_value_info(hidden, TensorProto.FLOAT, ["batch", weights.shape[1]]))
    config = ort.SessionOptions()
    config.intra_op_num_threads = threads
    config.inter_op_num_threads = 1
    session = ort.InferenceSession(model.SerializeToString(), config, providers=["CPUExecutionProvider"])
    return session, hidden, weights, bias, json.loads((CACHE / "preprocessor_config.json").read_text())


def csv_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def records(site):
    if site in ("plantdoc", "cornell"):
        root = RAW / ("plantdoc" if site == "plantdoc" else "fgvc7")
        data = ROOT / "data/features"
        with np.load(data / ("dino_plantdoc.npz" if site == "plantdoc" else "dino_fgvc.npz")) as saved:
            paths, labels, digests = (saved[key] for key in ("source_path", "y", "source_sha256"))
        if site == "plantdoc":
            with np.load(data / "dino_plantdoc_balanced_head.npz") as saved:
                wanted = set(saved["fit_keys"])
            index = {path: i for i, path in enumerate(paths) if path in wanted}
            for record, raw in plantdoc_images(root):
                path = record["source_path"]
                if path in index:
                    i = index[path]
                    yield path, labels[i], digests[i], raw
            return
        else:
            inds = range(len(paths))
        for i in inds:
            yield paths[i], labels[i], digests[i], (root / paths[i]).read_bytes()
    elif site == "kashmir":
        rows = csv_rows(ROOT / "results/transfer_manifest.csv")
        wanted = {row["source_path"] for row in rows}
        subset = {}
        for batch in pq.ParquetFile(RAW / "kashmir.parquet").iter_batches(batch_size=128, columns=["image"]):
            for row in batch.to_pylist():
                im = row["image"]
                if im["path"] in wanted:
                    subset[im["path"]] = im["bytes"]
        for row in rows:
            yield row["source_path"], int(row["y"]), row["source_sha256"], subset[row["source_path"]]
    else:
        rows = csv_rows(ROOT / "data/latvia_source/manifest.csv")
        with ZipFile(RAW / "latvia.zip") as archive:
            for row in rows:
                yield row["source_path"], int(row["y"]), row["source_sha256"], archive.read(row["source_path"])


def preprocess(row, cfg):
    path, label, digest, raw = row
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"Original-file hash mismatch: {path}")
    with Image.open(io.BytesIO(raw)) as im:
        data = tensor_from_image(ImageOps.exif_transpose(im).convert("RGB"), cfg)
    return path, label, digest, data


def extract(site, model, batch_size, workers):
    session, hidden, weights, bias, cfg = model
    cache = CACHE / (site + ".npz")
    expected_n = {"plantdoc": 1237, "cornell": 1821, "kashmir": 384, "latvia": 1190}[site]
    if cache.exists():
        with np.load(cache) as data:
            assert data["features"].shape == (expected_n, 1280) and data["model_sha256"] == HASHES["onnx/model.onnx"]
        print(f"{site}: validated existing cache", flush=True)
        return
    paths, labels, digests, features = [], [], [], []
    max_error = 0.0
    rows = iter(records(site))
    from itertools import islice
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while raw_batch := list(islice(rows, batch_size)):
            batch = list(pool.map(lambda row: preprocess(row, cfg), raw_batch))
            tensors = np.stack([row[3] for row in batch])
            logits, emb = session.run([session.get_outputs()[0].name, hidden], {session.get_inputs()[0].name: tensors})
            product = emb @ weights.T + bias
            error = float(np.max(np.abs(product - logits)))
            if error > 1e-4 or not np.isfinite(emb).all():
                raise ValueError(f"Feature verification failed: {site}, {error}")
            max_error = max(max_error, error)
            features.append(emb)
            paths.extend(row[0] for row in batch)
            labels.extend(row[1] for row in batch)
            digests.extend(row[2] for row in batch)
            if len(paths) // 256 > (len(paths)-len(batch)) // 256:
                print(f"{site} {len(paths)}/{expected_n}", flush=True)
    assert len(set(paths)) == expected_n
    temp = cache.with_suffix(".temp.npz")
    np.savez_compressed(temp, source_path=paths, y=np.array(labels, dtype=np.int8), source_sha256=digests,
                        features=np.concatenate(features), model_sha256=HASHES["onnx/model.onnx"],
                        max_reconstruction_error=max_error)
    temp.replace(cache)
    print(f"{site}: {len(paths)} images, maximum ImageNet logit reconstruction error {max_error:.8g}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", nargs="+", choices=("plantdoc", "cornell", "kashmir", "latvia"),
                        default=("plantdoc", "cornell", "kashmir", "latvia"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=12)
    args = parser.parse_args()
    model = checked_model(args.threads)
    for site in args.sites:
        extract(site, model, args.batch_size, args.workers)


if __name__ == "__main__":
    main()

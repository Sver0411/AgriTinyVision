"""Extract verified pre-classifier activations from the frozen, full-precision release."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

from edge_model import MODEL_SHA256, ROOT, read_rgb


def extract(output):
    model_path = ROOT / "data/raw/tinyvision.onnx"
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != MODEL_SHA256:
        raise ValueError("Unexpected released graph")
    m = onnx.load(model_path)
    last = m.graph.node[-1]
    if last.op_type != "Gemm" or any(
        a.name == "transB" and a.i != 1 for a in last.attribute
    ):
        raise ValueError("Expected released linear head")
    init = {t.name: t for t in m.graph.initializer}
    w, b = (numpy_helper.to_array(init[name]) for name in last.input[1:])
    if w.shape != (2, 1280) or b.shape != (2,):
        raise ValueError("Unexpected head tensor dimensions")
    feature_name = last.input[0]
    m.graph.output.append(
        helper.make_tensor_value_info(feature_name, TensorProto.FLOAT, [1, 1280])
    )
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = 1
    opt.inter_op_num_threads = 1
    sess = ort.InferenceSession(
        m.SerializeToString(), opt, providers=["CPUExecutionProvider"]
    )
    input_name = sess.get_inputs()[0].name
    prep = json.loads((ROOT / "data/raw/tinyvision-metadata.json").read_text())[
        "preprocessing"
    ]
    mean = np.asarray(prep["mean"], dtype=np.float32)
    std = np.asarray(prep["std"], dtype=np.float32)
    base = ROOT / "data/raw/fgvc7"
    with (base / "train.csv").open() as f:
        source = list(csv.DictReader(f))
    paths = []
    source_hash = []
    pixel_hash = []
    acts = []
    logits = []
    for i, row in enumerate(source):
        pg = f"images/{row['image_id']}.jpg"
        path = base / pg
        image = read_rgb(path)
        rgb = np.asarray(
            image.resize((160, 160), Image.Resampling.BILINEAR), dtype=np.uint8
        )
        arr = (rgb.astype(np.float32) / np.float32(255) - mean) / std
        inp = np.ascontiguousarray(arr.transpose(2, 0, 1)[None])
        raw, feature = sess.run([last.output[0], feature_name], {input_name: inp})
        paths.append(pg)
        source_hash.append(hashlib.sha256(path.read_bytes()).hexdigest())
        pixel_hash.append(hashlib.sha256(rgb.tobytes()).hexdigest())
        logits.append(raw[0])
        acts.append(feature[0])
        if (i + 1) % 400 == 0:
            print(f"Extracted {i + 1}/{len(source)}", flush=True)
    X = np.stack(acts)
    out = np.stack(logits)
    reconstructed = X @ w.T + b
    err = float(np.max(np.abs(out - reconstructed)))
    if err >= 1e-4 or not np.all(np.isfinite(X)):
        raise ValueError(f"Final-head reconstruction error {err:g}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        source_path=np.asarray(paths),
        source_sha256=np.asarray(source_hash),
        rgb160_sha256=np.asarray(pixel_hash),
        activation=X,
        logits=out,
        model_sha256=MODEL_SHA256,
        tensor_name=feature_name,
        logit_reconstruction_max_abs=err,
        original_head_weights=w,
        original_head_bias=b,
    )
    print(
        f"{len(X)} x {X.shape[1]} original activations, maximum logit error {err:g}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "data/features/edge_fgvc.npz"
    )
    extract(parser.parse_args().out)

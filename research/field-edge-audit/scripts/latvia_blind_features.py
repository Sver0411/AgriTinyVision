"""Extract blind Latvia features directly from the ZIP, then replay one frozen fusion rule."""

import argparse
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import time
from zipfile import ZipFile, is_zipfile
import zlib

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto
from PIL import Image, ImageOps
from scipy.fft import dctn

from dual_fusion_portable import local_decision, remote_decision
from dual_multiview import views
from edge_adaptation_wire import IMAGE, REPLY, normalized
from edge_model import EdgeModel, MODEL_SHA256, ROOT
from field_specialist import FEATURES, RESULTS, MODEL_DIR, WEIGHT_SHA, checked_model, digest_file, tensor_from_image
from server_backend import jpeg_transport


OUTPUT = FEATURES / "latvia.npz"
DEPLOYMENT_SHA = "0a9f30fc451671206572872672188885155d6bfdef55a0b4f76e81a4f3e74464"


def open_archive(path, central):
    if is_zipfile(path):
        return ZipFile(path)
    if central is None:
        raise ValueError("Incomplete ZIP; provide the collector's --central central.zip for streaming mode")
    # Optional during collection. The published reproduction uses Python's ZipFile.
    module_path = central.parent / "growing_zip.py"
    spec = importlib.util.spec_from_file_location("growing_zip", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GrowingZip(path, central)


def session(source, threads):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(source, options, providers=["CPUExecutionProvider"])


def hardware_graph():
    mobile = EdgeModel()
    graph = onnx.load(ROOT / "data/raw/tinyvision.onnx")
    pooled = graph.graph.node[-1].input[0]
    graph.graph.output.append(helper.make_tensor_value_info(pooled, TensorProto.FLOAT, [1, 1280]))
    return mobile, session(graph.SerializeToString(), 1)


def decode(payload):
    with Image.open(io.BytesIO(payload)) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def prepare(original, mobile, edge, cfg):
    image = decode(original)
    rgb = np.array(image.resize((160, 160), Image.Resampling.BILINEAR), dtype=np.uint8)
    v = (rgb.astype(np.float32) / np.float32(255) - mobile.mean) / mobile.std
    logits, features = edge.run(None, {mobile.input_name: np.ascontiguousarray(v.transpose(2, 0, 1)[None])})
    small = np.asarray(image.convert("L").resize((32, 32)), dtype=np.float64)
    spectrum = dctn(small, norm="ortho")[:8, :8].ravel()
    phash = np.packbits(spectrum > np.median(spectrum[1:])).tobytes().hex()
    jpeg, large = jpeg_transport(image, 224), jpeg_transport(image, 448)
    with Image.open(io.BytesIO(jpeg)) as encoded:
        primary = tensor_from_image(encoded.convert("RGB"), cfg)
    four = np.concatenate((primary[None], views(large, cfg)), axis=0)
    row = dict(source_sha256=hashlib.sha256(original).hexdigest(), phash=phash, logits=logits[0],
               edge=features[0], rgb160_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
               jpeg_bytes=len(jpeg), jpeg_sha256=hashlib.sha256(jpeg).hexdigest(),
               jpeg448_bytes=len(large), jpeg448_sha256=hashlib.sha256(large).hexdigest())
    return row, four


def extract(path, index, central, batch_size, threads):
    with index.open(newline="") as src:
        indexed = list(csv.DictReader(src))
    paths = [r["source_path"] for r in indexed]
    if len(set(paths)) != 1190 or len(paths) != len(set(paths)):
        raise ValueError("Expected the complete fixed 1,190 unique archive paths")
    cfg = checked_model()
    dino = session(str(MODEL_DIR / "model.onnx"), threads)
    mobile, edge = hardware_graph()
    records = []
    start = time.perf_counter()
    with open_archive(path, central) as archive:
        entries = [e.filename for e in archive.infolist() if not e.is_dir()]
        if set(entries) != set(paths) or len(entries) != len(paths):
            raise ValueError("Download directory and frozen index differ")
        for offset in range(0, len(paths), batch_size):
            inputs = []
            for i in range(offset, min(offset + batch_size, len(paths))):
                original = archive.read(paths[i])
                if (len(original) != int(indexed[i]["source_bytes"])
                        or zlib.crc32(original) != int(indexed[i]["crc32"], 16)):
                    raise ValueError("Latvia source bytes/CRC differ")
                record, four = prepare(original, mobile, edge, cfg)
                inputs.append(four)
                records.append(record)
            batch = np.ascontiguousarray(np.concatenate(inputs))
            cls = dino.run(["last_hidden_state"], {"pixel_values": batch})[0][:, 0].copy()
            if cls.shape != (4 * len(inputs), 384) or not np.isfinite(cls).all():
                raise ValueError("Invalid multiview tensor")
            for idx, four in enumerate(cls.reshape(len(inputs), 4, 384)):
                records[offset + idx]["dino"] = four[0].copy()
                # 448px full view, elementwise mean and elementwise max of two corner crops.
                records[offset + idx]["multiview"] = np.concatenate((four[1], four[2:].mean(axis=0), four[2:].max(axis=0)))
            if offset == 0 or (offset // batch_size + 1) % 20 == 0 or offset + batch_size >= len(paths):
                print(f"Blind Latvia {len(records)}/{len(paths)} images, {time.perf_counter()-start:.1f} s", flush=True)
    arrays = {k: np.asarray([r[k] for r in records]) for k in records[0]}
    source_head = FEATURES / "dino_plantdoc_balanced_head.npz"
    with np.load(source_head, allow_pickle=False) as h:
        score = normalized(arrays["dino"]) @ h["coef"][0] + h["intercept"].item()
    double_logits = arrays["logits"].astype(np.float64)
    edge_margin = double_logits[:, 1] - double_logits[:, 0]
    arrays.update(source_path=np.asarray(paths), y=np.asarray([int(r["y"]) for r in indexed], dtype=np.uint8),
                  source_label=np.asarray([r["source_label"] for r in indexed]), edge_margin=edge_margin,
                  dino_balanced_margin=score, edge_model_sha256=MODEL_SHA256, dino_model_sha256=WEIGHT_SHA,
                  source_head_sha256=digest_file(source_head), blind_index_sha256=digest_file(index),
                  jpeg_quality=np.int32(75), elapsed_seconds=time.perf_counter() - start)
    temporary = OUTPUT.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(OUTPUT)
    print(OUTPUT, digest_file(OUTPUT), flush=True)


def score_frozen():
    path = RESULTS / "dual_fusion_deployment.json"
    if digest_file(path) != DEPLOYMENT_SHA:
        raise ValueError("Preselected deployment changed")
    rule = json.loads(path.read_bytes())
    if (rule["seed"], rule["budget"]) != (0, 450):
        raise ValueError("Preselected seed and annotation budget changed")
    with np.load(OUTPUT, allow_pickle=False) as z:
        if z["source_head_sha256"].item() != rule["source_head_sha256"]:
            raise ValueError("Source DINO head changed after fusion freeze")
        paths, edge, dino = z["source_path"], z["edge_margin"], z["dino_balanced_margin"]
        if len(paths) != 1190 or len(set(paths)) != len(paths):
            raise ValueError("Incomplete blind predictions")
        sizes, y = z["jpeg_bytes"], z["y"]
    original = edge >= np.log(0.3 / 0.7)
    decisions = [local_decision(rule, e) for e in edge]
    upload = np.asarray([d is None for d in decisions])
    pred = np.asarray([remote_decision(rule, a, b) if v is None else v for a, b, v in zip(edge, dino, decisions)])
    result = dict(cohort="Latvia", n=len(y), budget=450, seed=0, frozen_deployment_sha256=DEPLOYMENT_SHA,
                  source_features_sha256=digest_file(OUTPUT), healthy=int(sum(y == 0)), diseased=int(sum(y == 1)),
                  original_fp=int(sum((y == 0) & original)), original_fn=int(sum((y == 1) & ~original)),
                  fusion_fp=int(sum((y == 0) & pred)), fusion_fn=int(sum((y == 1) & ~pred)),
                  images_uploaded=int(sum(upload)), inference_jpeg_bytes=int(sizes[upload].sum()),
                  inference_application_bytes=int(sizes[upload].sum() + int(sum(upload)) * (IMAGE.size + REPLY.size)))
    result["both_strict"] = result["fusion_fp"] < result["original_fp"] and result["fusion_fn"] < result["original_fn"]
    (RESULTS / "dual_fusion_latvia.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(RESULTS / "dual_fusion_latvia_predictions.npz", source_path=paths, y=y, original=original,
                        fusion=pred, upload=upload, frozen_deployment_sha256=DEPLOYMENT_SHA)
    print(json.dumps(result, indent=2), flush=True)


def verify(manifest_path, archive_provenance):
    """Compare the collector's independent ZIP traversal with our feature rows."""
    from PIL import __version__ as pillow_version

    with manifest_path.open(newline="") as src:
        reference = {r["source_path"]: r for r in csv.DictReader(src)}
    archive = json.loads(archive_provenance.read_text())
    with np.load(OUTPUT, allow_pickle=False) as data:
        names = data["source_path"]
        if len(names) != 1190 or set(names) != set(reference):
            raise ValueError("Source-manifest and feature indices disagree")
        rows = [reference[p] for p in names]
        for field in ("source_sha256", "y", "source_label"):
            if not np.array_equal(data[field].astype(str), [r[field] for r in rows]):
                raise ValueError("Independent source identifiers disagree in " + field)
        widths = {"edge": 1280, "dino": 384, "multiview": 1152, "logits": 2}
        for field, width in widths.items():
            v = data[field]
            if v.shape != (1190, width) or v.dtype != np.float32 or not np.isfinite(v).all():
                raise ValueError("Invalid extracted representation in " + field)
        distances = [(int(x, 16) ^ int(r["phash"], 16)).bit_count() for x, r in zip(data["phash"], rows)]
        facts = dict(n=len(names), archive_bytes=archive["archive_bytes"], archive_sha256=archive["archive_sha256"],
                     manifest_sha256=digest_file(manifest_path), feature_sha256=digest_file(OUTPUT),
                     exact_source_sha256_matches=len(names), phash_exact_matches=distances.count(0),
                     max_phash_bit_difference=max(distances), marginal_shapes={k: [len(names), v] for k,v in widths.items()},
                     source_model_head_sha256=data["source_head_sha256"].item(), edge_model_sha256=MODEL_SHA256,
                     dino_model_sha256=WEIGHT_SHA, numpy=np.__version__, pillow=pillow_version, onnxruntime=ort.__version__,
                     index_sha256=data["blind_index_sha256"].item(), runtime_s=data["elapsed_seconds"].item())
    if facts["manifest_sha256"] != archive["manifest_sha256"]:
        raise ValueError("Collector's manifest signature differs")
    (RESULTS / "latvia_feature_provenance.json").write_text(json.dumps(facts, indent=2) + "\n")
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    prep = sub.add_parser("extract")
    prep.add_argument("archive", type=Path)
    prep.add_argument("index", type=Path)
    prep.add_argument("--central", type=Path)
    prep.add_argument("--batch", type=int, default=3)
    prep.add_argument("--threads", type=int, default=4)
    sub.add_parser("score")
    check = sub.add_parser("verify")
    check.add_argument("manifest", type=Path)
    check.add_argument("provenance", type=Path)
    args = parser.parse_args()
    if args.stage == "extract":
        extract(args.archive, args.index, args.central, args.batch, args.threads)
    elif args.stage == "score":
        score_frozen()
    else:
        verify(args.manifest, args.provenance)

"""Offline ONNX FP32 vs a recorded MCU result for the identical RGB bytes."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from agritinyvision.pc_camera.image_preprocess import load_image_bgr, prepare_bgr
from agritinyvision.pc_camera.metrics import write_json


def compare(run, image, bundle):
    import onnxruntime as ort

    run, bundle = Path(run), Path(bundle)
    meta = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
    model_meta = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
    if meta["device"]["model_sha256"] != model_meta["artifact_sha256"]["tinyvision.espdl"]:
        raise ValueError("Recorded MCU model differs from reference bundle")
    reference = bundle / "tinyvision.onnx"
    if hashlib.sha256(reference.read_bytes()).hexdigest() != model_meta["artifact_sha256"]["tinyvision.onnx"]:
        raise ValueError("Reference ONNX hash differs from metadata")
    config = meta["config"]
    rgb = prepare_bgr(load_image_bgr(image), config.get("rotation", 0), config.get("mirror", False))
    digest = hashlib.sha256(rgb.tobytes()).hexdigest()
    with (run / "frame_results.csv").open(newline="", encoding="utf-8") as f:
        rows = [
            row
            for row in csv.DictReader(f)
            if not row["error"] and row.get("rgb_sha256") == digest and row.get("suspect_probability")
        ]
    if not rows:
        raise ValueError("Run has no inferred frame with the identical RGB SHA256")
    pre = model_meta["preprocessing"]
    x = (rgb.astype(np.float32) / 255 - np.array(pre["mean"], np.float32)) / np.array(pre["std"], np.float32)
    x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
    session = ort.InferenceSession(str(reference), providers=["CPUExecutionProvider"])
    logits = np.asarray(session.run(None, {session.get_inputs()[0].name: x})[0], dtype=np.float32).reshape(2)
    exps = np.exp(logits - logits.max())
    probability = float(exps[1] / exps.sum())
    report = {
        "purpose": "independent deployment consistency comparison",
        "evidence": meta["evidence"],
        "rgb_sha256": digest,
        "model_sha256": meta["device"]["model_sha256"],
        "onnx_sha256": model_meta["artifact_sha256"]["tinyvision.onnx"],
        "fp32_suspect_probability": probability,
        "device_results": [
            {
                "sequence": int(row["sequence"]),
                "suspect_probability": float(row["suspect_probability"]),
                "difference_from_fp32": float(row["suspect_probability"]) - probability,
            }
            for row in rows
        ],
        "decision_threshold": meta["device"]["decision_threshold"],
    }
    write_json(run / "reference_comparison.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--bundle", default="releases/v0.1.0")
    args = p.parse_args()
    print(json.dumps(compare(args.run, args.image, args.bundle), indent=2))

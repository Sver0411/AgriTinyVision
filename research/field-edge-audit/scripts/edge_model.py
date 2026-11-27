"""Frozen AgriTinyVision v0.1.0 FP32 host inference on the original RGB input."""
import hashlib
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA256 = "350c32d88d19eb16dca7d905036d2c99a716a3938b91f52715349e43ee6ab909"


def read_rgb(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


class EdgeModel:
    def __init__(self, model=None, metadata=None):
        model = Path(model or ROOT / "data/raw/tinyvision.onnx")
        metadata = Path(metadata or ROOT / "data/raw/tinyvision-metadata.json")
        self.metadata = json.loads(metadata.read_text())
        digest = hashlib.sha256(model.read_bytes()).hexdigest()
        if digest != MODEL_SHA256 or digest != self.metadata["artifact_sha256"]["tinyvision.onnx"]:
            raise ValueError("Frozen release digest mismatch")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        p = self.metadata["preprocessing"]
        self.mean = np.asarray(p["mean"], dtype=np.float32)
        self.std = np.asarray(p["std"], dtype=np.float32)

    def run_rgb(self, image):
        if not isinstance(image, Image.Image) or image.mode != "RGB":
            raise ValueError("Supply a PIL RGB image with its orientation already resolved")
        if min(image.size) < 32:
            raise ValueError("AgriTinyVision requires an original image of at least 32 by 32")
        rgb = np.array(image.resize((160, 160), Image.Resampling.BILINEAR), dtype=np.uint8)
        data = (rgb.astype(np.float32) / np.float32(255) - self.mean) / self.std
        model_input = np.ascontiguousarray(data.transpose(2, 0, 1)[None])
        start = perf_counter_ns()
        scores = self.session.run(None, {self.input_name: model_input})[0].reshape(2)
        elapsed = (perf_counter_ns() - start) / 1e6
        weights = np.exp(scores - np.max(scores))
        p = float(weights[1] / np.sum(weights))
        a = rgb.astype(np.float32)
        gray = a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114
        lap = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1]
        brightness, contrast, blur = float(gray.mean()), float(gray.std()), float(lap.var())
        quality = self.metadata["quality_gate"]
        passed = brightness >= quality["brightness_min"] and brightness <= quality["brightness_max"]
        passed = passed and contrast >= quality["contrast_min"] and blur >= quality["blur_min"]
        state = "NORMAL" if np.float32(p) < np.float32(0.2) else "SUSPECT" if np.float32(p) >= np.float32(0.4) else "UNKNOWN"
        return {
            "rgb160_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(), "normal_logit": float(scores[0]),
            "suspect_logit": float(scores[1]), "p_suspect": p, "binary_pred": int(np.float32(p) >= np.float32(0.3)),
            "quality_passed": int(passed), "state": state if passed else "UNKNOWN",
            "brightness": brightness, "contrast": contrast, "laplacian_variance": blur,
            "host_inference_ms": elapsed,
        }

    def run(self, path):
        return self.run_rgb(read_rgb(path))

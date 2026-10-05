import math

import numpy as np

from .data.dataset import open_rgb
from .utils import load_yaml

QUALITY_GATE_VERSION = "laplacian-gray-v1"


def quality_gate(image, config):
    try:
        im = open_rgb(image) if isinstance(image, (str, bytes)) or hasattr(image, "__fspath__") else image
        if im.mode != "RGB" or min(im.size) < config["min_dimension"]:
            return {"passed": False, "reasons": ["invalid_format_or_size"]}
        # Metrics always computed at the same resolution, before model normalization.
        a = np.asarray(im.resize((config["metric_size"], config["metric_size"])), dtype=np.float32)
        gray = a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114
        lap = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1]
        values = {
            "brightness": float(gray.mean()),
            "contrast": float(gray.std()),
            "laplacian_variance": float(lap.var()),
        }
        reasons = []
        if values["laplacian_variance"] < config["blur_min"]:
            reasons.append("blur")
        if not config["brightness_min"] <= values["brightness"] <= config["brightness_max"]:
            reasons.append("exposure")
        if values["contrast"] < config["contrast_min"]:
            reasons.append("low_contrast")
        return {"passed": not reasons, "reasons": reasons, "metrics": values, "version": QUALITY_GATE_VERSION}
    except Exception as e:
        return {"passed": False, "reasons": ["invalid_input"], "error": str(e)}


def decide(suspect_probability, threshold, quality, min_confidence):
    if not quality["passed"]:
        return {"state": "UNKNOWN", "reason": "quality_gate", "quality": quality}
    p = float(suspect_probability)
    if not math.isfinite(p) or not 0 <= p <= 1:
        return {"state": "UNKNOWN", "reason": "invalid_model_output"}
    confidence = max(p, 1 - p)
    if confidence < min_confidence:
        return {"state": "UNKNOWN", "reason": "insufficient_confidence", "confidence": confidence}
    return {
        "state": "SUSPECT" if p >= threshold else "NORMAL",
        "suspect_probability": p,
        "confidence": confidence,
        "decision_threshold": threshold,
    }


def load_quality(path="configs/quality.yaml"):
    c = load_yaml(path)
    if (
        not 0.5 <= c["min_confidence"] <= 1
        or not 0 <= c["brightness_min"] < c["brightness_max"] <= 255
        or c["contrast_min"] < 0
        or c["blur_min"] < 0
        or c["metric_size"] < 3
        or c["min_dimension"] < 1
    ):
        raise ValueError("Invalid quality configuration")
    return c

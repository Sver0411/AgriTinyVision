import math
from decimal import Decimal

import numpy as np

from .data.dataset import open_rgb
from .utils import load_yaml

QUALITY_GATE_VERSION = "laplacian-gray-v1"
ABSTENTION_POLICY = "threshold-centered-v1"


def abstention_bounds(threshold, margin):
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Invalid decision threshold")
    if not math.isfinite(margin) or not 0 <= margin <= min(threshold, 1 - threshold):
        raise ValueError("Invalid abstention margin")
    # Decimal configuration arithmetic avoids 0.30 - 0.10 falling below 0.20.
    # Export these same float32 constants to firmware; do not recompute them there.
    t, m = Decimal(str(threshold)), Decimal(str(margin))
    return float(np.float32(float(t - m))), float(np.float32(float(t + m)))


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


def decide(suspect_probability, threshold, quality, abstention_margin):
    if not quality["passed"]:
        return {"state": "UNKNOWN", "reason": "quality_gate", "quality": quality}
    try:
        p = float(suspect_probability)
    except (TypeError, ValueError):
        return {"state": "UNKNOWN", "reason": "invalid_model_output"}
    if not math.isfinite(p) or not 0 <= p <= 1:
        return {"state": "UNKNOWN", "reason": "invalid_model_output"}
    lower, upper = abstention_bounds(threshold, abstention_margin)
    decision_p = float(np.float32(p))
    state = "NORMAL" if decision_p < lower else "UNKNOWN" if decision_p < upper else "SUSPECT"
    return {
        "state": state,
        "reason": "decision_boundary" if state == "UNKNOWN" else "threshold_interval",
        "suspect_probability": p,
        "decision_threshold": threshold,
        "abstention_margin": abstention_margin,
        "normal_upper_bound": lower,
        "suspect_lower_bound": upper,
        "abstention_policy": ABSTENTION_POLICY,
    }


def load_quality(path="configs/quality.yaml"):
    c = load_yaml(path)
    if c.get("version") != QUALITY_GATE_VERSION:
        raise ValueError("Unsupported quality-gate version")
    if (
        "min_confidence" in c
        or not math.isfinite(c["abstention_margin"])
        or not 0 <= c["abstention_margin"] < 0.5
        or not 0 <= c["brightness_min"] < c["brightness_max"] <= 255
        or c["contrast_min"] < 0
        or c["blur_min"] < 0
        or c["metric_size"] < 3
        or c["min_dimension"] < 1
    ):
        raise ValueError("Invalid quality configuration")
    return c

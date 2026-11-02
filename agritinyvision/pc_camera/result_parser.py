"""Validate device telemetry before displaying or recording it."""

import json
import math

from .protocol import ProtocolError

STATES = {"NORMAL", "SUSPECT", "UNKNOWN"}
TIMINGS = ("quality_us", "preprocess_us", "inference_us", "total_us")
MEMORY = ("free_internal", "free_psram", "min_free_internal", "min_free_psram")


def parse_json(payload):
    try:
        value = json.loads(payload)
    except (ValueError, UnicodeError) as exc:
        raise ProtocolError("invalid JSON response") from exc
    if not isinstance(value, dict):
        raise ProtocolError("response must be an object")
    return value


def number(value, name, low=0, high=float("inf"), nullable=False):
    if value is None and nullable:
        return
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ProtocolError(f"invalid {name}")
    if not low <= value <= high:
        raise ProtocolError(f"out-of-range {name}")


def parse_device(payload):
    d = parse_json(payload)
    for key in (
        "boot_id",
        "firmware_version",
        "model_version",
        "model_sha256",
        "idf_version",
        "esp_dl_version",
    ):
        if not isinstance(d.get(key), str) or not d[key]:
            raise ProtocolError(f"missing device {key}")
    if (d.get("chip"), d.get("input_shape"), d.get("output_shape"), d.get("dtype")) != (
        "ESP32-S3",
        [1, 160, 160, 3],
        [1, 2],
        "int8",
    ):
        raise ProtocolError("incompatible device/model tensors")
    if not isinstance(d.get("model_loaded"), bool):
        raise ProtocolError("missing model load status")
    if len(d["model_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in d["model_sha256"]):
        raise ProtocolError("invalid model hash")
    for key in ("input_exponent", "output_exponent"):
        number(d.get(key), key, -32, 32)
        if type(d[key]) is not int:
            raise ProtocolError(f"integer required for {key}")
    number(d.get("decision_threshold"), "decision_threshold", 0, 1)
    number(d.get("normal_upper_bound"), "normal_upper_bound", 0, 1)
    number(d.get("suspect_lower_bound"), "suspect_lower_bound", 0, 1)
    if not d["normal_upper_bound"] <= d["decision_threshold"] <= d["suspect_lower_bound"]:
        raise ProtocolError("invalid decision boundaries")
    return d


def parse_result(payload, sequence, device):
    r = parse_json(payload)
    if r.get("sequence") != sequence or r.get("boot_id") != device["boot_id"]:
        raise ProtocolError("sequence mismatch or device restart")
    if (
        r.get("model_version") != device["model_version"]
        or r.get("firmware_version") != device["firmware_version"]
    ):
        raise ProtocolError("device version changed")
    if r.get("state") not in STATES or not isinstance(r.get("reason"), str):
        raise ProtocolError("invalid inference state")
    if r.get("quality_passed") is not None and not isinstance(r["quality_passed"], bool):
        raise ProtocolError("invalid quality gate")
    for key in TIMINGS + MEMORY:
        number(r.get(key), key, nullable=key in ("preprocess_us", "inference_us", "quality_us"))
    for key in ("normal_probability", "suspect_probability", "confidence"):
        number(r.get(key), key, 0, 1, nullable=True)
    number(r.get("decision_threshold"), "decision_threshold", 0, 1)
    if abs(r["decision_threshold"] - device["decision_threshold"]) > 1e-6:
        raise ProtocolError("decision threshold changed")
    p, n, confidence = (r[k] for k in ("suspect_probability", "normal_probability", "confidence"))
    if (p is None) != (n is None):
        raise ProtocolError("partial probability output")
    if r.get("quality_passed") is False and any(r.get(k) is not None for k in
            ("normal_probability", "suspect_probability", "confidence", "preprocess_us", "inference_us")):
        raise ProtocolError("quality rejection must skip normalization and inference")
    if p is None or n is None:
        if r["state"] != "UNKNOWN" or confidence is not None:
            raise ProtocolError("missing probability requires UNKNOWN/null confidence")
    else:
        if abs(p + n - 1) > 1e-5 or r["quality_passed"] is not True:
            raise ProtocolError("invalid probability/quality combination")
        expected = (
            "NORMAL"
            if p < device["normal_upper_bound"]
            else "UNKNOWN"
            if p < device["suspect_lower_bound"]
            else "SUSPECT"
        )
        if r["state"] != expected:
            raise ProtocolError("state disagrees with frozen decision policy")
        assigned = n if expected == "NORMAL" else p if expected == "SUSPECT" else None
        if (
            assigned is None
            and confidence is not None
            or assigned is not None
            and (confidence is None or abs(confidence - assigned) > 1e-5)
        ):
            raise ProtocolError("confidence must describe the assigned class")
    if r["state"] != "UNKNOWN" and r["inference_us"] is None:
        raise ProtocolError("classification requires device inference timing")
    return r

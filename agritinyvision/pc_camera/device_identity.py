"""Bind a device handshake to a hash-verified, frozen deployment contract."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .deployment import deployment_contract
from .protocol import ProtocolError

IDENTITY_FIELDS = (
    "model_sha256", "model_version", "deployment_sha256", "input_shape", "output_shape",
    "dtype", "input_exponent", "output_exponent", "decision_threshold", "normal_upper_bound",
    "suspect_lower_bound", "abstention_policy", "quality_gate_version",
)


def load_expected_identity(bundle="releases/v0.1.0", deployment=None, quality="configs/quality.yaml"):
    contract = deployment_contract(bundle, quality)
    if deployment:
        saved = json.loads(Path(deployment).read_text(encoding="utf-8"))
        if saved != contract:
            raise ValueError("部署元数据与正式模型包/质量配置不一致")
    return {
        "model_sha256": contract["model_sha256"],
        "model_version": contract["model_version"],
        "deployment_sha256": contract["deployment_sha256"],
        "input_shape": contract["tensors"]["input"]["shape"],
        "output_shape": contract["tensors"]["output"]["shape"],
        "dtype": "int8",
        "input_exponent": contract["tensors"]["input"]["exponent"],
        "output_exponent": contract["tensors"]["output"]["exponent"],
        "decision_threshold": float(np.float32(contract["decision_threshold"])),
        "normal_upper_bound": contract["normal_upper_bound"],
        "suspect_lower_bound": contract["suspect_lower_bound"],
        "abstention_policy": contract["abstention_policy"],
        "quality_gate_version": contract["quality_gate"]["version"],
    }


def verify_device_identity(device, expected):
    checks = {key: key in device and device[key] == expected[key] for key in IDENTITY_FIELDS}
    # Firmware serializes float32 constants; compare against the same representation exactly.
    for key in ("decision_threshold", "normal_upper_bound", "suspect_lower_bound"):
        checks[key] = isinstance(device.get(key), (float, int)) and not isinstance(device.get(key), bool) and (
            float(np.float32(device[key])) == float(np.float32(expected[key]))
        )
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        details = "; ".join(f"{k}: expected={expected[k]!r}, actual={device.get(k)!r}" for k in failed)
        raise ProtocolError("设备模型身份校验失败，拒绝推理：" + details)
    return {
        "status": "PASS", "checked_at": datetime.now(timezone.utc).isoformat(),
        "expected": {k: expected[k] for k in IDENTITY_FIELDS},
        "actual": {k: device[k] for k in IDENTITY_FIELDS}, "checks": checks,
    }

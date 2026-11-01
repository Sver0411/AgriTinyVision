"""Inspect the real EDL2 FlatBuffer and emit a hash-bound camera deployment contract."""

import hashlib
import json
import re
import shutil
import struct
from pathlib import Path

import flatbuffers.number_types as nt
from flatbuffers.table import Table

from agritinyvision.quality import abstention_bounds, load_quality


def inspect_espdl(path):
    """EDL2 schema fields from ESP-DL fbs_loader/espdl.fbs (no graph execution)."""
    data = Path(path).read_bytes()
    if len(data) < 32 or data[:4] != b"EDL2":
        raise ValueError("Expected real unencrypted EDL2 model")
    mode, size = struct.unpack_from("<II", data, 4)
    if mode != 0 or not 4 <= size <= len(data) - 16:
        raise ValueError("Invalid EDL2 model length/encryption")
    data = data[16 : 16 + size]

    def table(parent, slot):
        offset = parent.Offset(4 + slot * 2)
        if not offset:
            raise ValueError(f"Missing FlatBuffer field {slot}")
        return Table(data, parent.Indirect(parent.Pos + offset))

    def integer(parent, slot, dtype=nt.Int32Flags):
        offset = parent.Offset(4 + slot * 2)
        return parent.Get(dtype, parent.Pos + offset) if offset else 0

    def vector(parent, slot):
        offset = parent.Offset(4 + slot * 2)
        if not offset:
            raise ValueError("Missing FlatBuffer vector")
        length = parent.VectorLen(offset)
        if not 1 <= length <= 8:
            raise ValueError("Invalid IO vector length")
        return parent.Vector(offset), length

    try:
        model = Table(data, struct.unpack_from("<I", data)[0])
        graph = table(model, 7)  # Model.graph

        def tensor(slot):
            pos, length = vector(graph, slot)
            if length != 1:
                raise ValueError("Exactly one model input/output required")
            value = Table(data, graph.Indirect(pos))
            type_info = table(value, 1)
            if integer(type_info, 0, nt.Uint8Flags) != 1:
                raise ValueError("Expected TensorTypeAndShape union")
            tensor_type = table(type_info, 1)
            shape_table = table(tensor_type, 1)
            dims, count = vector(shape_table, 0)
            shape = []
            for i in range(count):
                dim = Table(data, shape_table.Indirect(dims + i * 4))
                shape.append(integer(table(dim, 0), 1, nt.Int64Flags))
            exp_pos, exp_count = vector(value, 3)
            if exp_count != 1:
                raise ValueError("Model IO must use scalar exponents")
            return {
                "shape": shape,
                "dtype": integer(tensor_type, 0),
                "exponent": struct.unpack_from("<q", data, exp_pos)[0],
            }

        return {"input": tensor(4), "output": tensor(5)}
    except (IndexError, struct.error, TypeError) as exc:
        raise ValueError("Invalid model FlatBuffer") from exc


def prepare(bundle, destination="firmware_example/main", quality_path="configs/quality.yaml"):
    bundle, destination = Path(bundle), Path(destination)
    source = bundle / "tinyvision.espdl"
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata_path = bundle / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["artifact_sha256"]["tinyvision.espdl"] != sha:
        raise ValueError("Model SHA256 disagrees with metadata")
    if metadata["classes"] != ["NORMAL", "SUSPECT"] or metadata["target"] != "esp32s3":
        raise ValueError("Expected frozen ESP32-S3 two-class model")
    if (metadata["input_width"], metadata["input_height"], metadata["channels"]) != (160, 160, 3):
        raise ValueError("Camera protocol requires the 160 model")
    pre = metadata["preprocessing"]
    if pre["mean"] != [0.485, 0.456, 0.406] or pre["std"] != [0.229, 0.224, 0.225]:
        raise ValueError("ImageNet preprocessing contract differs")
    if pre["color"] != "RGB" or pre["resize"] != "direct square bilinear, antialias":
        raise ValueError("RGB bilinear antialias resize contract differs")
    tensors = inspect_espdl(source)
    # TensorDataType.INT8 = 3 in the converter's actual FlatBuffer schema.
    if tensors["input"]["shape"] != [1, 160, 160, 3] or tensors["output"]["shape"] != [1, 2]:
        raise ValueError("Actual model tensor shapes differ")
    if any(t["dtype"] != 3 for t in tensors.values()):
        raise ValueError("Actual model tensors must be INT8")
    info_path = bundle / "tinyvision.info"
    info = info_path.read_text(encoding="utf-8")
    for name, shape, tensor in (
        ("image", "1x160x160x3", tensors["input"]),
        ("logits", "1x2", tensors["output"]),
    ):
        expression = rf"%{name}\[INT8, {shape}\], exponents: \[(-?\d+)\]"
        match = re.search(expression, info)
        if not match or int(match[1]) != tensor["exponent"]:
            raise ValueError(f"tinyvision.info disagrees with binary {name}")
    quality = load_quality(quality_path)
    if quality["metric_size"] != 160:
        raise ValueError("Quality metric_size must be 160")
    threshold = metadata["decision_threshold"]
    lower, upper = abstention_bounds(threshold, quality["abstention_margin"])
    config = {
        "version": "camera-deployment-v1",
        "model_version": metadata["version"],
        "model_sha256": sha,
        "model_bytes": source.stat().st_size,
        "tensors": tensors,
        "preprocessing": pre,
        "decision_threshold": threshold,
        "normal_upper_bound": lower,
        "suspect_lower_bound": upper,
        "quality_gate": quality,
        "abstention_policy": "threshold-centered-v1",
        "source_metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "source_info_sha256": hashlib.sha256(info_path.read_bytes()).hexdigest(),
        "quality_config_sha256": hashlib.sha256(Path(quality_path).read_bytes()).hexdigest(),
        "policy_source": "current integration contract + configs/quality.yaml; release bundle preserved",
    }
    deployment_sha = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config["deployment_sha256"] = deployment_sha
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "tinyvision.espdl"
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    header = [
        "#pragma once",
        f'#define CAMERA_DEPLOYMENT_SHA256 "{deployment_sha}"',
        f'#define CAMERA_MODEL_SHA256 "{sha}"',
        f"#define CAMERA_MODEL_VERSION {json.dumps(metadata['version'])}",
        f"#define CAMERA_MODEL_BYTES {source.stat().st_size}",
        f"#define CAMERA_INPUT_EXPONENT {tensors['input']['exponent']}",
        f"#define CAMERA_OUTPUT_EXPONENT {tensors['output']['exponent']}",
    ]
    constants = {
        "THRESHOLD": threshold,
        "NORMAL_BOUND": lower,
        "SUSPECT_BOUND": upper,
        "BLUR_MIN": quality["blur_min"],
        "BRIGHTNESS_MIN": quality["brightness_min"],
        "BRIGHTNESS_MAX": quality["brightness_max"],
        "CONTRAST_MIN": quality["contrast_min"],
    }
    for name, value in constants.items():
        header.append(f"#define CAMERA_{name} {float(value):.17e}f")
    for name in ("mean", "std"):
        header.append(
            f"static constexpr float CAMERA_{name.upper()}[3] = {{"
            + ",".join(f"{v:.17e}f" for v in pre[name])
            + "};"
        )
    (destination / "camera_model_config.h").write_text("\n".join(header) + "\n", encoding="utf-8")
    (destination / "camera_deployment.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config

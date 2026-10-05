import re
import shutil
from pathlib import Path

from jsonschema import validate

from . import CLASSES
from .data.transforms import MEAN, STD
from .quality import QUALITY_GATE_VERSION, load_quality
from .utils import commit, now, read_json, sha256, write_json


def validate_metadata(metadata, schema="docs/metadata.schema.json"):
    validate(metadata, read_json(schema))


def release(checkpoint, onnx_path, metrics_path, espdl_path=None, version="0.1.0", output=None):
    from .train import load_checkpoint

    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("Expected numeric semantic version")
    _, cp = load_checkpoint(checkpoint)
    c = cp["config"]
    out = Path(output or f"releases/v{version}")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Refusing to overwrite an existing release")
    validation = read_json(str(onnx_path) + ".validation.json")
    if not validation["passed"] or validation["sha256"] != sha256(onnx_path):
        raise ValueError("ONNX numerical validation missing or stale")
    test = read_json(metrics_path)
    if (
        test.get("split") != "test"
        or test.get("threshold_source") != "validation"
        or test["decision_threshold"] != cp["threshold"]
        or test.get("model_sha256") != sha256(checkpoint)
    ):
        raise ValueError("Release needs final test metrics matching the exact checkpoint and threshold")
    quant_report = None
    if espdl_path:
        p = Path(espdl_path)
        quant_report = read_json(str(p) + ".quantization.json")
        if (
            not quant_report["status"].startswith("SUCCESS")
            or quant_report["target"] != "esp32s3"
            or quant_report["sha256"] != sha256(p)
            or quant_report["source_sha256"] != sha256(onnx_path)
        ):
            raise ValueError("ESP-DL conversion report missing, failed or mismatched")
    env = read_json(Path(checkpoint).parent / "environment.json")
    artifacts = {"tinyvision.onnx": sha256(onnx_path)}
    if espdl_path:
        artifacts["tinyvision.espdl"] = sha256(espdl_path)
    metadata = {
        "model_name": "AgriTinyVision",
        "version": version,
        "git_commit": commit(),
        "training_git_commit": env.get("git_commit"),
        "input_width": c["input_size"],
        "input_height": c["input_size"],
        "channels": 3,
        "classes": list(CLASSES),
        "decision_threshold": cp["threshold"],
        "quality_gate_version": QUALITY_GATE_VERSION,
        "quality_gate": load_quality(),
        "quantization": "int8" if espdl_path else "fp32",
        "target": "esp32s3",
        "dataset_version": env["dataset_version"],
        "dataset_hash": env["dataset_hash"],
        "train_split_hash": env["train_split_hash"],
        "created_at": now(),
        "sha256": artifacts.get("tinyvision.espdl", artifacts["tinyvision.onnx"]),
        "artifact_sha256": artifacts,
        "espdl_export": "SUCCESS" if espdl_path else "PENDING",
        "hardware_status": "PENDING HARDWARE VALIDATION",
        "pretrained_weights": env["pretrained_weights"],
        "preprocessing": {
            "color": "RGB",
            "resize": "direct square bilinear, antialias",
            "range": "0..1",
            "mean": MEAN,
            "std": STD,
            "onnx_layout": "NCHW",
            "onnx_input_type": "float32",
            "output": "two logits; apply stable softmax externally",
            "positive_index": 1,
            "espdl_layout": "inspect exported .info; ESP-DL Conv uses NHWC",
            "espdl_quantization": "read actual tensor exponent/dtype; never reuse ONNX float pointer",
        },
    }
    validate_metadata(metadata)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_path, out / "tinyvision.onnx")
    if espdl_path:
        shutil.copy2(espdl_path, out / "tinyvision.espdl")
        for suffix in (".info", ".json"):
            extra = Path(espdl_path).with_suffix(suffix)
            if extra.exists():
                shutil.copy2(extra, out / ("tinyvision" + suffix))
    write_json(out / "metadata.json", metadata)
    write_json(out / "metrics.json", {"fp32": test, "espdl_int8_simulation": quant_report})
    (out / "labels.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
    (out / "README.md").write_text(
        f"# AgriTinyVision v{version}\n\nESP-DL export: {metadata['espdl_export']}\n\nPENDING HARDWARE VALIDATION\n\nPublic grouped subset; field effectiveness is unverified. See metadata for preprocessing and thresholds.\n\nThe .espdl binary targets ESP32-S3 only. Host quantized simulation is not MCU evidence.\n",
        encoding="utf-8",
    )
    return out

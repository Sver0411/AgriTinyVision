from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from .train import load_checkpoint
from .utils import load_yaml, sha256, write_json

OPERATOR_SOURCE = "https://github.com/espressif/esp-dl/blob/master/operator_support_state.md"
# Reviewed subset for torchvision MobileNetV2, not a blanket compatibility guarantee.
SUPPORTED = {
    "Conv",
    "Clip",
    "Relu",
    "Add",
    "GlobalAveragePool",
    "Flatten",
    "Gemm",
    "Reshape",
    "Constant",
    "Identity",
}


def inspect_operators(path):
    graph = onnx.load(path)
    ops = Counter(n.op_type for n in graph.graph.node)
    unsupported = sorted(set(ops) - SUPPORTED)
    grouped_conv_issues = []
    for node in graph.graph.node:
        if node.op_type == "Conv":
            attributes = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            group = attributes.get("group", 1)
            weight = next((w for w in graph.graph.initializer if w.name == node.input[1]), None)
            if group != 1 and (weight is None or weight.dims[1] != 1):
                grouped_conv_issues.append(node.name)
    return {
        "operators": dict(ops),
        "unsupported": unsupported,
        "grouped_conv_issues": grouped_conv_issues,
        "source": OPERATOR_SOURCE,
        "target": "esp32s3",
        "opset": graph.opset_import[0].version,
        "static_review_passed": not unsupported and not grouped_conv_issues,
        "deployment_status": "PENDING ESP-PPQ conversion and hardware validation",
    }


def export_model(model, size, path, config=None):
    torch.set_num_threads(4)
    c = config or load_yaml("configs/export.yaml")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    x = torch.randn(1, 3, size, size, generator=torch.Generator().manual_seed(41))
    torch.onnx.export(
        model,
        x,
        str(path),
        input_names=["image"],
        output_names=["logits"],
        opset_version=c["opset"],
        dynamo=False,
    )
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    checks = []
    for sample in (x, torch.zeros_like(x), torch.ones_like(x), x * 0.3):
        with torch.inference_mode():
            a = model(sample).numpy()
        b = session.run(None, {"image": sample.numpy()})[0]
        np.testing.assert_allclose(a, b, atol=c["validation_atol"], rtol=c["validation_rtol"])
        checks.append(float(np.max(np.abs(a - b))))
    report = {
        "passed": True,
        "max_absolute_error": max(checks),
        "inputs_checked": len(checks),
        "atol": c["validation_atol"],
        "rtol": c["validation_rtol"],
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "compatibility": inspect_operators(path),
    }
    write_json(str(path) + ".validation.json", report)
    return report


def export(checkpoint, output=None):
    model, cp = load_checkpoint(checkpoint)
    out = output or str(Path(checkpoint).parent / "tinyvision.onnx")
    report = export_model(model, cp["config"]["input_size"], out)
    report["checkpoint_sha256"] = sha256(checkpoint)
    write_json(str(out) + ".validation.json", report)
    print(report, flush=True)
    return out


def validate_checkpoint_onnx(checkpoint, path):
    """Bind an existing reference ONNX to its exact trained checkpoint, without rewriting it."""
    from .utils import read_json

    torch.set_num_threads(4)
    model, cp = load_checkpoint(checkpoint)
    size = cp["config"]["input_size"]
    c = load_yaml("configs/export.yaml")
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = torch.randn(1, 3, size, size, generator=torch.Generator().manual_seed(41))
    errors = []
    for sample in (x, torch.zeros_like(x), torch.ones_like(x), x * 0.3):
        with torch.inference_mode():
            expected = model(sample).numpy()
        actual = session.run(None, {"image": sample.numpy()})[0]
        np.testing.assert_allclose(expected, actual, atol=c["validation_atol"], rtol=c["validation_rtol"])
        errors.append(float(np.max(np.abs(expected - actual))))
    report = read_json(str(path) + ".validation.json")
    if report["sha256"] != sha256(path):
        raise ValueError("Existing validation report is stale")
    report.update(checkpoint_sha256=sha256(checkpoint), checkpoint_binding_max_absolute_error=max(errors))
    write_json(str(path) + ".validation.json", report)
    return report


def preflight(config_path, output="reports/operator_preflight.onnx"):
    from .models.mobilenetv2 import build
    from .utils import validate_config

    c = validate_config(load_yaml(config_path))
    # Structural dry run before spending GPU time. Random weights are never a trained artifact.
    report = export_model(build(c, download_weights=False), c["input_size"], output)
    report["purpose"] = "random-weight structural preflight only"
    write_json("reports/operator_preflight.json", report)
    if not report["compatibility"]["static_review_passed"]:
        raise RuntimeError("Operator compatibility preflight failed")
    return report

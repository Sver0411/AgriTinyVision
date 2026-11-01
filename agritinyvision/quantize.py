import importlib.metadata
import random
import shutil
import traceback
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from .data.dataset import ManifestDataset
from .data.transforms import preprocessing
from .export_onnx import inspect_operators
from .threshold import metrics
from .train import load_checkpoint
from .utils import load_yaml, now, read_json, sha256, write_json


def validate_quantization_report(model_path, checkpoint, threshold):
    report = read_json(str(model_path) + ".quantization.json")
    validation = report.get("validation_metrics", {})
    if (
        not report.get("status", "").startswith("SUCCESS")
        or report.get("sha256") != sha256(model_path)
        or report.get("checkpoint_sha256") != sha256(checkpoint)
        or report.get("quantization_selection_split") != "validation"
        or report.get("decision_threshold_source") != "validation"
        or validation.get("split") != "val"
        or validation.get("threshold_source") != "validation"
        or validation.get("decision_threshold") != threshold
        or validation.get("model_sha256") != sha256(model_path)
        or "test_metrics" in report
    ):
        raise ValueError(
            "Quantized candidate needs matching validation-only provenance; test selection is forbidden"
        )
    return report


def predict_quantized(model_path, loader, dev, report):
    """Read an already frozen artifact; never recalibrate while evaluating test."""
    labels, scores = [], []
    if report["backend"] == "espdl":
        from esp_ppq import TorchExecutor
        from esp_ppq.api import load_native_graph

        native = Path(str(model_path) + ".native")
        if report.get("simulation_graph_sha256") != sha256(native):
            raise ValueError("Quantized simulation graph missing or changed")
        # Native PPQ files use pickle. Load only locally produced trusted graphs.
        executor = TorchExecutor(graph=load_native_graph(str(native)), device=str(dev))
        for x, y in loader:
            with torch.inference_mode():
                result = executor.forward(x.to(dev))[0]
                scores.extend(result.float().softmax(1)[:, 1].cpu().tolist())
                labels.extend(y.tolist())
        backend = "ESP-PPQ quantized graph simulation; not ESP32 hardware"
    else:
        import onnxruntime as ort

        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        for x, y in loader:
            logits = session.run(None, {session.get_inputs()[0].name: x.numpy()})[0]
            scores.extend(torch.from_numpy(logits).float().softmax(1)[:, 1].tolist())
            labels.extend(y.tolist())
        backend = "ONNX Runtime INT8 CPU; desktop only"
    return labels, scores, backend


def calibration_indices(dataset, count, seed):
    # Balanced sampling from train only, without augmentations or val/test contamination.
    buckets = {}
    for i, row in enumerate(dataset.rows):
        buckets.setdefault(row["source_label"], []).append(i)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    indices = []
    while len(indices) < min(count, len(dataset)):
        for bucket in buckets.values():
            if bucket and len(indices) < count:
                indices.append(bucket.pop())
    return indices


def quantize(model_path, checkpoint, backend="espdl", output=None, allow_cpu=False, evaluation_split="val"):
    if evaluation_split != "val":
        raise ValueError(
            "Quantization development can evaluate only val; use evaluate --final for frozen test"
        )
    _, cp = load_checkpoint(checkpoint)
    numerical = read_json(str(model_path) + ".validation.json")
    if (
        not numerical.get("passed")
        or numerical.get("sha256") != sha256(model_path)
        or numerical.get("checkpoint_sha256") != sha256(checkpoint)
    ):
        raise ValueError("Quantization needs ONNX numerical validation matching the frozen checkpoint")
    c = cp["config"]
    if sha256(c["manifest"]) != cp["manifest_sha256"]:
        raise ValueError("Calibration manifest differs from training")
    ec = load_yaml("configs/export.yaml")
    suffix = ".espdl" if backend == "espdl" else ".int8.onnx"
    out = Path(output or str(Path(model_path).with_suffix(suffix)))
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise ValueError(f"Refusing to overwrite quantized artifact: {out}")
    dataset = ManifestDataset(c["manifest"], c["dataset_root"], "train", preprocessing(c["input_size"]))
    indices = calibration_indices(dataset, ec["calibration_samples"], c["seed"])
    loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False, num_workers=0)
    report = {
        "created_at": now(),
        "backend": backend,
        "target": "esp32s3" if backend == "espdl" else "desktop ONNX Runtime only",
        "source_sha256": sha256(model_path),
        "checkpoint_sha256": sha256(checkpoint),
        "decision_threshold_source": "validation",
        "quantization_selection_split": "validation",
        "calibration_split": "train",
        "calibration_samples": len(indices),
        "calibration_paths": [dataset.rows[i]["path"] for i in indices],
        "quant_type": ec["quant_type"],
        "hardware_status": "PENDING HARDWARE VALIDATION",
    }
    try:
        if backend == "onnx":
            from onnxruntime.quantization import (
                CalibrationDataReader,
                QuantFormat,
                QuantType,
                quantize_static,
            )

            class Reader(CalibrationDataReader):
                def __init__(self):
                    self.iterator = iter(loader)

                def get_next(self):
                    batch = next(self.iterator, None)
                    return {"image": batch[0].numpy()} if batch else None

            quantize_static(
                str(model_path),
                str(out),
                Reader(),
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                per_channel=False,
                extra_options={"ActivationSymmetric": True, "WeightSymmetric": True},
                op_types_to_quantize=["Conv", "Gemm"],
            )
            report["warning"] = (
                "ORT QDQ INT8 is a desktop comparison artifact, not an ESP-DL deployment model"
            )
            report["status"] = "SUCCESS"
        else:
            from .doctor import device

            dev = device(allow_cpu)
            from esp_ppq import QuantizationSettingFactory
            from esp_ppq.api import espdl_quantize_onnx, export_ppq_graph
            from esp_ppq.core import TargetPlatform

            review = inspect_operators(model_path)
            if not review["static_review_passed"]:
                raise ValueError(f"Unsupported operators: {review}")
            report["esp_ppq_version"] = importlib.metadata.version("esp-ppq")
            setting = QuantizationSettingFactory.espdl_setting()
            setting.quantize_activation_setting.calib_algorithm = "percentile"
            # Converter can simplify in place: preserve the numerically validated FP32 source.
            working = out.with_suffix(".working.onnx")
            shutil.copy2(model_path, working)
            graph = espdl_quantize_onnx(
                onnx_import_file=str(working),
                espdl_export_file=str(out),
                calib_dataloader=loader,
                calib_steps=len(indices),
                input_shape=[1, 3, c["input_size"], c["input_size"]],
                target="esp32s3",
                quant_type=ec["quant_type"],
                setting=setting,
                device=str(dev),
                collate_fn=lambda batch: batch[0].to(dev),
                error_report=False,
                export_test_values=True,
                verbose=0,
            )
            if not out.is_file() or out.stat().st_size == 0:
                raise RuntimeError("ESP-PPQ returned without a nonempty .espdl artifact")
            native = Path(str(out) + ".native")
            if native.exists():
                raise ValueError("Refusing to overwrite a saved quantized graph")
            export_ppq_graph(graph, TargetPlatform.NATIVE, str(native))
            report["simulation_graph_sha256"] = sha256(native)
            report["status"] = "SUCCESS (conversion and host simulation)"
        report.update(sha256=sha256(out), bytes=out.stat().st_size)
        validation = ManifestDataset(c["manifest"], c["dataset_root"], "val", preprocessing(c["input_size"]))
        labels, scores, evaluation_backend = predict_quantized(
            out,
            DataLoader(validation, batch_size=1, shuffle=False),
            dev if backend == "espdl" else "cpu",
            report,
        )
        report["validation_metrics"] = metrics(labels, scores, cp["threshold"])
        report["validation_metrics"].update(
            split="val", threshold_source="validation", backend=evaluation_backend, model_sha256=sha256(out)
        )
        write_json(str(out) + ".quantization.json", report)
        print(report, flush=True)
        return out
    except Exception as e:
        report.update(
            status="FAILED",
            error=f"{type(e).__name__}: {e}",
            traceback=traceback.format_exc(),
            artifact_present=out.exists(),
            export_status="PENDING",
        )
        write_json(str(out) + ".quantization.json", report)
        raise

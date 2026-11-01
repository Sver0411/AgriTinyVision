import time
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from .data.dataset import ManifestDataset
from .data.transforms import preprocessing
from .doctor import device
from .quality import decide, load_quality, quality_gate
from .threshold import metrics
from .train import load_checkpoint, predict
from .utils import read_json, sha256, write_json


def validate_final_request(checkpoint, model_path, final, frozen_checkpoint_sha256, frozen_model_sha256):
    if not final:
        if frozen_checkpoint_sha256 or frozen_model_sha256:
            raise ValueError("Frozen hashes require explicit --final mode")
        return
    if frozen_checkpoint_sha256 != sha256(checkpoint):
        raise ValueError("Final test requires the exact frozen checkpoint SHA256")
    if model_path and frozen_model_sha256 != sha256(model_path):
        raise ValueError("Final test requires the exact frozen model SHA256")


def evaluate(
    checkpoint,
    model_path=None,
    allow_cpu=False,
    output=None,
    final=False,
    frozen_checkpoint_sha256=None,
    frozen_model_sha256=None,
):
    validate_final_request(checkpoint, model_path, final, frozen_checkpoint_sha256, frozen_model_sha256)
    split = "test" if final else "val"
    dev = device(allow_cpu)
    model, cp = load_checkpoint(checkpoint, dev)
    c = cp["config"]
    if sha256(c["manifest"]) != cp["manifest_sha256"]:
        raise ValueError("Evaluation manifest differs from training")
    out = Path(
        output
        or (
            str(model_path) + f".{split}_metrics.json"
            if model_path
            else Path(checkpoint).parent / f"{split}_metrics.json"
        )
    )
    if out.exists():
        raise ValueError(f"Evaluation already exists: {out}. Do not repeatedly consult test metrics to tune.")
    quant_report = None
    if model_path and Path(str(model_path) + ".quantization.json").is_file():
        from .quantize import validate_quantization_report

        quant_report = validate_quantization_report(model_path, checkpoint, cp["threshold"])
    elif model_path and Path(model_path).suffix == ".espdl":
        raise ValueError("ESP-DL evaluation requires matching validation-only quantization report")
    elif model_path:
        validation = read_json(str(model_path) + ".validation.json")
        if (
            not validation.get("passed")
            or validation.get("sha256") != sha256(model_path)
            or validation.get("checkpoint_sha256") != sha256(checkpoint)
        ):
            raise ValueError("FP32 ONNX evaluation requires numerical validation bound to the checkpoint")
    dataset = ManifestDataset(c["manifest"], c["dataset_root"], split, preprocessing(c["input_size"]))
    loader = DataLoader(
        dataset, batch_size=c["batch_size"] if not model_path else 1, shuffle=False, num_workers=c["workers"]
    )
    started = time.perf_counter()
    if quant_report:
        from .quantize import predict_quantized

        labels, scores, backend = predict_quantized(model_path, loader, dev, quant_report)
    elif model_path:
        import onnxruntime as ort

        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        labels, scores = [], []
        for x, y in loader:
            logits = session.run(None, {session.get_inputs()[0].name: x.numpy()})[0]
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
            scores.extend(probs[:, 1].tolist())
            labels.extend(y.tolist())
        backend = "ONNX Runtime CPU; desktop only"
    else:
        labels, scores = predict(model, loader, dev)
        backend = f"PyTorch {dev}"
    threshold = cp["threshold"]  # Frozen validation-selected threshold; never retuned on test.
    result = metrics(labels, scores, threshold)
    result.update(
        split=split,
        threshold_source="validation",
        decision_threshold_source="validation",
        backend=backend,
        model_sha256=sha256(model_path or checkpoint),
        seconds=time.perf_counter() - started,
        evaluation_role="final evaluation; no test-set tuning"
        if final
        else "validation; candidate development",
        final_metrics_split="test" if final else None,
        frozen_checkpoint_sha256=frozen_checkpoint_sha256,
        frozen_model_sha256=frozen_model_sha256 if model_path else frozen_checkpoint_sha256,
    )
    if quant_report:
        result.update(
            quantization_selection_split="validation",
            quantization_report_sha256=sha256(str(model_path) + ".quantization.json"),
            checkpoint_sha256=sha256(checkpoint),
        )
    q = load_quality()
    states, reasons = [], {}
    for row, score in zip(dataset.rows, scores):
        gate = quality_gate(Path(c["dataset_root"]) / row["path"], q)
        answer = decide(score, threshold, gate, q["abstention_margin"])
        states.append(answer["state"])
        for reason in gate.get("reasons", []):
            reasons[reason] = reasons.get(reason, 0) + 1
    accepted = [i for i, state in enumerate(states) if state != "UNKNOWN"]
    system_fn = sum(y == 1 and state == "NORMAL" for y, state in zip(labels, states))
    result["system_gate"] = {
        "coverage": len(accepted) / len(states),
        "unknown_count": states.count("UNKNOWN"),
        "quality_reasons": reasons,
        "normal_count": states.count("NORMAL"),
        "suspect_count": states.count("SUSPECT"),
        "suspect_as_normal": system_fn,
        "suspect_unknown": sum(y == 1 and state == "UNKNOWN" for y, state in zip(labels, states)),
        "accepted_metrics": metrics([labels[i] for i in accepted], [scores[i] for i in accepted], threshold)
        if accepted
        else None,
        "quality_config": q,
        "ood_detection": "not implemented",
        "plant_content_detection": "not implemented",
    }
    write_json(out, result)
    print(result, flush=True)
    return result

"""Run frozen desktop references on the exact saved device RGB inputs."""

import hashlib
import json
import time
from pathlib import Path

from PIL import Image

from agritinyvision.pc_camera.device_identity import load_expected_identity, verify_device_identity
from agritinyvision.pc_camera.metrics import write_json
from agritinyvision.quality import load_quality, quality_gate
from agritinyvision.quantization_analysis import (
    FrozenHostReference,
    OnnxReference,
    comparison_summary,
    frozen_state,
    saturation_analysis,
)
from agritinyvision.utils import sha256

from .capture import now, read_json, read_samples, verified_sample


def compare_batch(batch, bundle="releases/v0.1.0", host_model=None, host_report=None):
    batch = Path(batch)
    metadata = read_json(batch / "batch.json")
    expected = load_expected_identity(bundle)
    verify_device_identity(metadata["device"], expected)
    fp32 = OnnxReference(bundle)
    host = None
    if host_model:
        report = read_json(host_report)
        if sha256(host_model) != expected["model_sha256"] or report["source_sha256"] != fp32.sha256:
            raise ValueError("Host and FP32 artifacts must match the batch deployment")
        host = FrozenHostReference(host_model, report)
    rows = []
    for sample in read_samples(batch):
        meta, rgb = verified_sample(batch, sample)
        digest = hashlib.sha256(rgb.tobytes()).hexdigest()
        prediction = read_json(batch / sample["prediction"]) if (batch / sample["prediction"]).exists() else {"status": "pending"}
        q = quality_gate(Image.fromarray(rgb), load_quality())
        row = {"sample_id": sample["sample_id"], "id": sample["sample_id"], "rgb_sha256": digest,
               "scenario": meta["context"].get("scene", "UNRECORDED"), "errors": {}, "quality": q,
               "saturation": saturation_analysis(rgb, expected["input_exponent"]), "device": None, "host": None}
        for name, runner in (("fp32", fp32), ("host", host)):
            if runner is None:
                continue
            started = time.perf_counter()
            p = runner(rgb)
            row[name] = {"probabilities": p, "state": frozen_state(p[1], q, expected),
                         "wall_ms": (time.perf_counter() - started) * 1000}
        if prediction["status"] == "ok":
            r = prediction["result"]
            if r["rgb_sha256"] != digest:
                raise ValueError("Prediction input hash mismatch: " + sample["sample_id"])
            p = None if r["suspect_probability"] is None else [r["normal_probability"], r["suspect_probability"]]
            row["device"] = {"probabilities": p, "state": r["state"], "wall_ms": r["end_to_end_ms"],
                             "inference_us": r["inference_us"], "quality_passed": r["quality_passed"]}
        else:
            row["errors"]["device"] = prediction.get("error", prediction["status"])
        for first, second in (("fp32", "host"), ("fp32", "device"), ("host", "device")):
            a, b = row.get(first), row.get(second)
            row[first + "_minus_" + second] = [x - y for x, y in zip(a["probabilities"], b["probabilities"])] \
                if a and b and a["probabilities"] is not None and b["probabilities"] is not None else None
        rows.append(row)
    result = {"created_at": now(), "evidence": metadata["evidence"], "expected_identity": expected,
              "onnx_sha256": fp32.sha256, "host": "MEASURED" if host else "PENDING: original frozen native graph required",
              "summary": comparison_summary(rows), "samples": rows}
    write_json(batch / "reports/INFERENCE_PARITY.json", result)
    lines = ["# Inference parity", "", "RGB hashes identify the saved bytes used for each device result.", "",
             "Host: " + result["host"], "", "```json", json.dumps(result["summary"], indent=2), "```", "",
             "| Sample | FP32 SUSPECT | Host SUSPECT | Device SUSPECT | States (FP32 / host / device) |",
             "| --- | ---: | ---: | ---: | --- |"]
    for row in rows:
        values = [row.get(k) for k in ("fp32", "host", "device")]
        ps = [str(v["probabilities"][1]) if v and v["probabilities"] is not None else "null" for v in values]
        states = [v["state"] if v else "PENDING" for v in values]
        lines.append(f"| {row['sample_id']} | {' | '.join(ps)} | {' / '.join(states)} |")
    (batch / "reports/INFERENCE_PARITY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result

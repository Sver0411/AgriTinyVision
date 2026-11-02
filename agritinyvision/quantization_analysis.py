"""Read-only FP32 / frozen PPQ / physical-device comparison of identical RGB bytes."""

import csv
import hashlib
import json
import platform
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from .image_contract import normalize_rgb
from .pc_camera.device_identity import load_expected_identity
from .pc_camera.image_preprocess import load_image_bgr, prepare_bgr
from .pc_camera.metrics import RunRecorder, write_json
from .quality import load_quality, quality_gate
from .utils import sha256


def saturation_analysis(rgb, exponent):
    normalized = normalize_rgb(rgb)
    rounded = np.floor(normalized * np.float32(2.0 ** -exponent) + np.float32(0.5))
    clipped = (rounded < -128) | (rounded > 127)
    channels = {}
    for c, name in enumerate("RGB"):
        raw, norm = rgb[..., c], normalized[..., c]
        channels[name] = {
            "raw_min": int(raw.min()), "raw_max": int(raw.max()), "raw_mean": float(raw.mean()),
            "normalized_min": float(norm.min()), "normalized_max": float(norm.max()),
            "normalized_percentiles": np.percentile(norm, [1, 50, 99]).tolist(),
            "saturated_values": int(clipped[..., c].sum()), "saturated_fraction": float(clipped[..., c].mean()),
        }
    return {"saturated_values": int(clipped.sum()), "saturated_fraction": float(clipped.mean()),
            "input_exponent": exponent, "channels": channels}


def probabilities(logits):
    values = np.asarray(logits, dtype=np.float32).reshape(-1)
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError("Expected two finite logits")
    p = np.exp(values - values.max())
    return (p / p.sum()).tolist()


class OnnxReference:
    def __init__(self, bundle):
        import onnxruntime as ort

        bundle = Path(bundle)
        meta = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
        model = bundle / "tinyvision.onnx"
        if sha256(model) != meta["artifact_sha256"]["tinyvision.onnx"]:
            raise ValueError("ONNX SHA256 differs from frozen metadata")
        self.session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        self.sha256 = sha256(model)

    def __call__(self, rgb):
        x = np.ascontiguousarray(normalize_rgb(rgb).transpose(2, 0, 1)[None])
        return probabilities(self.session.run(None, {self.session.get_inputs()[0].name: x})[0])


class FrozenHostReference:
    def __init__(self, model, report):
        from .quantize import load_frozen_executor

        self.executor = load_frozen_executor(model, report)

    def __call__(self, rgb):
        import torch

        x = torch.from_numpy(np.ascontiguousarray(normalize_rgb(rgb).transpose(2, 0, 1)[None]))
        with torch.inference_mode():
            logits = self.executor.forward(x)[0].float().cpu().numpy()
        return probabilities(logits)


def frozen_state(p, quality, identity):
    if not quality["passed"]:
        return "UNKNOWN"
    p = float(np.float32(p))
    return "NORMAL" if p < identity["normal_upper_bound"] else "UNKNOWN" if p < identity["suspect_lower_bound"] else "SUSPECT"


def comparison_summary(rows):
    summary = {"samples": len(rows), "scenarios": dict(Counter(r["scenario"] for r in rows))}
    for first, second in (("fp32", "host"), ("fp32", "device"), ("host", "device")):
        pair = [r for r in rows if r.get(first) and r.get(second)]
        predicted = [r for r in pair if r[first]["probabilities"] is not None
                     and r[second]["probabilities"] is not None]
        differences = [abs(r[first]["probabilities"][1] - r[second]["probabilities"][1]) for r in predicted]
        summary[first + "_vs_" + second] = {
            "state_pairs": len(pair), "probability_pairs": len(predicted),
            "state_agreement": sum(r[first]["state"] == r[second]["state"] for r in pair) / len(pair)
            if pair else "PENDING",
            "suspect_probability_mae": float(np.mean(differences)) if differences else "PENDING",
            "suspect_probability_max_error": max(differences) if differences else "PENDING",
        }
    triples = [r for r in rows if all(r.get(k) for k in ("fp32", "host", "device"))]
    summary["three_way_state_agreement"] = {
        "count": len(triples), "rate": sum(r["fp32"]["state"] == r["host"]["state"] == r["device"]["state"]
                                            for r in triples) / len(triples) if triples else "PENDING"}
    fractions = [r["saturation"]["saturated_fraction"] for r in rows]
    summary["saturation_fraction"] = {"p50": float(np.percentile(fractions, 50)),
                                      "p95": float(np.percentile(fractions, 95))} if fractions else "PENDING"
    summary["failures"] = [{"id": r["id"], "errors": r["errors"]} for r in rows if r["errors"]]
    return summary


def analyze(manifest, bundle, output, client=None, host_model=None, host_report=None):
    """No calibration, parameter selection or artifact modification occurs here."""
    manifest, bundle, output = Path(manifest), Path(bundle), Path(output)
    identity = load_expected_identity(bundle)
    spec = json.loads(manifest.read_text(encoding="utf-8"))
    samples = spec["samples"]
    if not samples:
        raise ValueError("Manifest needs samples")
    # Validate source hashes before creating a run or opening the device.
    for sample in samples:
        source = (manifest.parent / sample["path"]).resolve()
        if sha256(source) != sample["source_sha256"]:
            raise ValueError("Image SHA256 mismatch: " + sample["id"])
    fp32 = OnnxReference(bundle)
    host = None
    host_status = "PENDING: frozen native graph and hash-bound quantization report required"
    if host_model and host_report:
        if sha256(host_model) != identity["model_sha256"]:
            raise ValueError("Host model differs from deployment")
        report = json.loads(Path(host_report).read_text(encoding="utf-8"))
        if report["source_sha256"] != fp32.sha256:
            raise ValueError("Host source ONNX differs from FP32 reference")
        host = FrozenHostReference(host_model, report)
        host_status = "MEASURED: ESP-PPQ frozen quantized graph simulation"
    output.mkdir(parents=True, exist_ok=False)
    device = None
    if client:
        device = client.connect()
        if device["identity_validation"]["expected"] != identity:
            client.close()
            raise ValueError("Device client expected a different deployment")
    quality_config = load_quality()
    rows = []
    recorder = RunRecorder(output / "device_runs", {"manifest_sha256": sha256(manifest),
                           "purpose": "controlled quantization comparison"}, device, evidence=client.evidence) if client else None
    run_started = last_result = time.perf_counter()
    try:
        for sample in samples:
            rgb = prepare_bgr(load_image_bgr(manifest.parent / sample["path"]))
            digest = hashlib.sha256(rgb.tobytes()).hexdigest()
            q = quality_gate(Image.fromarray(rgb), quality_config)
            row = {k: sample.get(k) for k in ("id", "scenario", "source", "source_label", "source_split", "parent_sha256",
                                             "annotation_valid", "transform")}
            row.update(rgb_sha256=digest, quality=q,
                       saturation=saturation_analysis(rgb, identity["input_exponent"]), errors={})
            for name, runner in (("fp32", fp32), ("host", host), ("device", client)):
                row[name] = None
                if runner is None:
                    continue
                started = time.perf_counter()
                try:
                    if name == "device":
                        r = client.infer(rgb.tobytes())
                        # CRC-checked frame/ACK and matching sequence bind the sent RGB to this response.
                        p = None if r["suspect_probability"] is None else [r["normal_probability"], r["suspect_probability"]]
                        state = r["state"]
                        if r["quality_passed"] != q["passed"]:
                            raise ValueError("Device/reference quality decision differs")
                        if r.get("saturated_values") is not None and r["saturated_values"] != row["saturation"]["saturated_values"]:
                            raise ValueError("Device/reference saturation differs")
                        row["device_telemetry"] = r
                        now = time.perf_counter()
                        r = {**r, "rgb_sha256": digest, "serial_roundtrip_ms": r["end_to_end_ms"],
                             "actual_fps": 1 / (now - last_result)}
                        recorder.record(r, now - run_started)
                        last_result = now
                    else:
                        p = runner(rgb)
                        state = frozen_state(p[1], q, identity)
                    row[name] = {"probabilities": p, "state": state,
                                 "wall_ms": (time.perf_counter() - started) * 1000}
                except Exception as exc:
                    row["errors"][name] = str(exc)
                    if name == "device":
                        recorder.error(str(exc), client.sequence)
            for first, second in (("fp32", "host"), ("fp32", "device"), ("host", "device")):
                a, b = row[first], row[second]
                row[first + "_minus_" + second] = [x - y for x, y in zip(a["probabilities"], b["probabilities"])] \
                    if a and b and a["probabilities"] is not None and b["probabilities"] is not None else None
            rows.append(row)
    finally:
        if client:
            client.close()
        if recorder:
            recorder.close()
    summary = comparison_summary(rows)
    metadata = {"created_at": datetime.now(timezone.utc).isoformat(), "manifest_sha256": sha256(manifest),
                "expected_identity": identity, "device": device, "onnx_sha256": fp32.sha256,
                "host": host_status, "device_status": "MEASURED: " + client.evidence if client else "PENDING",
                "python": platform.python_version(), "purpose": "deployment error analysis; no threshold selection"}
    from .pc_camera.metrics import environment_metadata

    metadata["project"] = environment_metadata()
    metadata["analysis_source_sha256"] = sha256(__file__)
    from importlib.metadata import PackageNotFoundError, version

    metadata["dependencies"] = {}
    for package in ("numpy", "Pillow", "torch", "onnxruntime", "esp-ppq", "pyserial"):
        try:
            metadata["dependencies"][package] = version(package)
        except PackageNotFoundError:
            metadata["dependencies"][package] = None
    metadata["host_graph_sha256"] = report.get("simulation_graph_sha256") if host else None
    metadata["device_run"] = recorder.path.relative_to(output).as_posix() if recorder else None
    write_json(output / "comparison.json", {"metadata": metadata, "summary": summary, "samples": rows})
    write_json(output / "summary.json", summary)
    fields = ["id", "scenario", "rgb_sha256", "saturated_values", "saturated_fraction"]
    fields += [f"{b}_{k}" for b in ("fp32", "host", "device") for k in ("normal", "suspect", "state", "wall_ms")]
    fields += ["errors"]
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for r in rows:
            flat = {k: r[k] for k in ("id", "scenario", "rgb_sha256")}
            flat.update({k: r["saturation"][k] for k in ("saturated_values", "saturated_fraction")})
            for b in ("fp32", "host", "device"):
                v = r[b]
                flat[b + "_normal"] = v["probabilities"][0] if v and v["probabilities"] is not None else None
                flat[b + "_suspect"] = v["probabilities"][1] if v and v["probabilities"] is not None else None
                for k in ("state", "wall_ms"):
                    flat[b + "_" + k] = v[k] if v else None
            flat["errors"] = json.dumps(r["errors"], ensure_ascii=False)
            writer.writerow(flat)
    lines = ["# Quantization comparison", "", f"Host: {host_status}",
             f"Device: {metadata['device_status']}", "", "```json", json.dumps(summary, indent=2), "```", "",
             "Quality-rejected device frames have null probabilities and no model inference. FP32/host raw outputs",
             "are retained for analysis, with the same quality gate applied to reported states.",
             "These controlled scenarios measure deployment differences, not independent field accuracy."]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    plot_analysis(output, rows)
    return {"metadata": metadata, "summary": summary}


def plot_analysis(output, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["scenario"] for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for i, channel in enumerate("RGB"):
        axes[0].bar(x + (i - 1) * 0.25, [r["saturation"]["channels"][channel]["saturated_fraction"] * 100
                                       for r in rows], width=0.25, label=channel)
    axes[0].set_ylabel("Clipped input values (%)")
    axes[0].legend()
    for name in ("fp32", "host", "device"):
        if not any(r.get(name) for r in rows):
            continue
        axes[1].plot(x, [r[name]["probabilities"][1] if r.get(name) and r[name]["probabilities"] is not None
                        else float("nan") for r in rows], marker="o", label=name)
    axes[1].set_ylabel("SUSPECT probability")
    axes[1].set_xticks(x, names, rotation=35, ha="right")
    axes[1].legend()
    if not any(r.get("host") for r in rows):
        axes[0].set_title("Host simulation: PENDING (frozen graph unavailable)")
    fig.tight_layout()
    fig.savefig(Path(output) / "saturation_comparison.png", dpi=140)
    plt.close(fig)

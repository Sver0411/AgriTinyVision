"""Bounded experiment export; public preparation is explicit and excludes raw pictures."""

import csv
import io
import json
import math
import re
import zipfile
from pathlib import Path

from .metrics import MEMORY, generate_report, write_json

RUN_FILES = {"run_metadata.json", "frame_results.csv", "device_metrics.csv", "summary.json",
             "latency_report.json", "errors.log", "HARDWARE_REPORT.md"}
PLOT_FILES = {"latency_distribution.png", "fps_over_time.png", "memory_usage.png", "transfer_vs_inference.png"}


def checked_file(path, root):
    path, root = Path(path), Path(root).resolve()
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_relative_to(root) or not path.is_file():
        raise ValueError("Experiment file escapes allowed directory or is a link")
    return resolved


def experiment_zip(folder, allowed_root):
    folder, root = Path(folder), Path(allowed_root).resolve()
    if folder.is_symlink() or not folder.resolve().is_relative_to(root):
        raise ValueError("Experiment directory escapes configured output root")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(RUN_FILES):
            path = folder / name
            if path.exists():
                archive.write(checked_file(path, folder), name)
        plots = folder / "plots"
        if plots.exists() and (plots.is_symlink() or not plots.resolve().is_relative_to(folder.resolve())):
            raise ValueError("Plot directory is a link or escapes experiment")
        for name in sorted(PLOT_FILES):
            path = plots / name
            if path.exists():
                archive.write(checked_file(path, folder), "plots/" + name)
    buffer.seek(0)
    return buffer


def public_experiment(source, destination):
    """Create a new sanitized directory; never publish or alter the original run."""
    source, destination = Path(source), Path(destination)
    meta = json.loads(checked_file(source / "run_metadata.json", source).read_text(encoding="utf-8"))
    device = meta.get("device", {})
    clean_device = {}
    for key in ("firmware_version", "model_version", "idf_version", "esp_dl_version"):
        value = device.get(key)
        if isinstance(value, str) and re.fullmatch(r"v?\d+(?:\.\d+)+(?:[-+][A-Za-z0-9.]+)?", value):
            clean_device[key] = value
    for key in ("model_sha256", "deployment_sha256", "firmware_sha256"):
        value = device.get(key)
        if isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value):
            clean_device[key] = value
    for key in ("input_exponent", "output_exponent", "decision_threshold", "normal_upper_bound",
                "suspect_lower_bound", "model_bytes", "app_image_bytes", "app_partition_bytes", "flash_chip_bytes"):
        value = device.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            clean_device[key] = value
    verification = device.get("identity_validation", {})
    if verification.get("status") == "PASS":
        clean_device["identity_validation"] = {"status": "PASS", "checks": {
            k: v for k, v in verification.get("checks", {}).items()
            if k in {"model_sha256", "model_version", "deployment_sha256", "input_shape", "output_shape",
                     "dtype", "input_exponent", "output_exponent", "decision_threshold", "normal_upper_bound",
                     "suspect_lower_bound", "abstention_policy", "quality_gate_version"} and type(v) is bool}}
    clean_config = {k: v for k, v in meta.get("config", {}).items()
                    if k in {"camera", "baudrate", "fps", "rotation", "mirror", "timeout"}
                    and isinstance(v, (bool, int, float)) and math.isfinite(v)}
    pc = meta.get("pc", {})
    def safe_version(value):
        return value if isinstance(value, str) and re.fullmatch(r"\d+(?:\.\d+)+(?:[A-Za-z0-9.+-]*)", value) else None

    project = meta.get("project", {})
    clean = {"schema_version": "public-experiment-v1", "evidence": meta["evidence"] if meta["evidence"] in
             {"hardware", "synthetic-web-unit-test", "synthetic-unit-test"} else "unspecified",
             "started_at": None, "finished_at": None, "duration_s": meta.get("duration_s", 0),
             "config": clean_config, "device": clean_device,
             "pc": {"platform": "redacted", "python": safe_version(pc.get("python")),
                    "dependencies": {k: safe_version(v) for k, v in pc.get("dependencies", {}).items()
                                     if k in {"numpy", "Pillow", "opencv-python", "pyserial", "matplotlib",
                                              "Flask", "torch", "onnxruntime", "esp-ppq"}}},
             "project": {"project_version": safe_version(project.get("project_version")),
                         "git_commit": project.get("git_commit") if re.fullmatch(r"[0-9a-f]{40}", str(project.get("git_commit"))) else None,
                         "git_dirty": project.get("git_dirty") if type(project.get("git_dirty")) is bool else None},
             "redactions": ["device identifiers", "local paths", "private configuration", "raw error text", "photos"]}
    for key in ("started_at", "finished_at"):
        v = meta.get(key)
        clean[key] = v if isinstance(v, str) and re.fullmatch(r"[\dT:+.Z-]+", v) else None
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "run_metadata.json", clean)
    from .metrics import FIELDS

    with checked_file(source / "frame_results.csv", source).open(encoding="utf-8", newline="") as f:
        original = list(csv.DictReader(f))
    rows = []
    for row in original:
        safe = {key: "" for key in FIELDS}
        for key in FIELDS:
            value = row.get(key, "")
            if key in ("state", "quality_passed"):
                safe[key] = value if value in ("NORMAL", "SUSPECT", "UNKNOWN", "True", "False") else ""
            elif key == "error":
                safe[key] = "recorded_error" if value else ""
            elif key == "rgb_sha256":
                safe[key] = value if re.fullmatch("[0-9a-f]{64}", value) else ""
            elif key in ("timestamp", "boot_id", "reason"):
                continue
            else:
                try:
                    numeric = float(value)
                    if math.isfinite(numeric):
                        safe[key] = numeric
                except (ValueError, TypeError):
                    pass
        rows.append(safe)
    for name, fields in (("frame_results.csv", FIELDS),
                         ("device_metrics.csv", ("sequence", "timestamp", "quality_us", "preprocess_us",
                                                 "inference_us", "total_us", *MEMORY))):
        with (destination / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    (destination / "errors.log").write_text(f"recorded_error_count={sum(bool(r['error']) for r in rows)}\n", encoding="utf-8")
    generate_report(destination)
    return destination

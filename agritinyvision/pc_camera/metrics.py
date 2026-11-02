"""Durable per-frame experiment records and reports from measured device responses."""

import csv
import hashlib
import json
import platform
import subprocess
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from agritinyvision import __version__

from .result_parser import MEMORY, TIMINGS

FIELDS = (
    "timestamp",
    "rgb_sha256",
    "elapsed_s",
    "sequence",
    "state",
    "reason",
    "normal_probability",
    "suspect_probability",
    "confidence",
    "decision_threshold",
    "quality_passed",
    "brightness",
    "contrast",
    "laplacian_variance",
    "transfer_ms",
    "pc_capture_ms",
    "pc_preprocess_ms",
    "serial_roundtrip_ms",
    *TIMINGS,
    "end_to_end_ms",
    "actual_fps",
    *MEMORY,
    "saturated_values",
    "saturated_fraction",
    "boot_id",
    "error",
)


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def environment_metadata():
    root = Path(__file__).resolve().parents[2]
    git = ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root)]
    try:
        commit = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output([*git, "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"project_version": __version__, "git_commit": commit, "git_dirty": dirty}


class RunRecorder:
    def __init__(self, root="runs/hardware", config=None, device=None, evidence="hardware"):
        now = datetime.now(timezone.utc)
        self.start_monotonic = time.perf_counter()
        self.path = Path(root) / now.astimezone(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
        self.path.mkdir(parents=True, exist_ok=False)
        self.metadata = {
            "schema_version": "experiment-v3",
            "project": environment_metadata(),
            "started_at": now.isoformat(),
            "evidence": evidence,
            "pc": {"platform": platform.platform(), "python": platform.python_version()},
            "config": config or {},
            "device": device or {},
        }
        dependencies = {}
        for package in ("numpy", "Pillow", "opencv-python", "pyserial", "matplotlib", "Flask", "torch", "onnxruntime", "esp-ppq"):
            try:
                dependencies[package] = version(package)
            except PackageNotFoundError:
                dependencies[package] = None
        self.metadata["pc"]["dependencies"] = dependencies
        module_root = Path(__file__).resolve().parent
        self.metadata["pc"]["source_sha256"] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(module_root.glob("*.py"))
        }
        write_json(self.path / "run_metadata.json", self.metadata)
        self.files = []
        self.writers = []
        for filename, fields in (
            ("frame_results.csv", FIELDS),
            ("device_metrics.csv", ("timestamp", "sequence", *TIMINGS, *MEMORY)),
        ):
            f = (self.path / filename).open("w", newline="", encoding="utf-8")
            writer = csv.DictWriter(f, fields, extrasaction="ignore")
            writer.writeheader()
            self.files.append(f)
            self.writers.append(writer)
        self.errors = (self.path / "errors.log").open("a", encoding="utf-8")
        self.error_count = 0
        self.closed = False

    def record(self, result, elapsed_s):
        row = {
            **result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": elapsed_s,
            "error": "",
        }
        if result.get("saturated_values") is not None:
            row["saturated_fraction"] = result["saturated_values"] / (160 * 160 * 3)
        for writer, f in zip(self.writers, self.files):
            writer.writerow(row)
            f.flush()

    def error(self, message, sequence=None):
        self.error_count += 1
        stamp = datetime.now(timezone.utc).isoformat()
        self.errors.write(f"{stamp} seq={sequence} {message}\n")
        self.errors.flush()
        self.writers[0].writerow(
            {"timestamp": stamp, "sequence": sequence, "state": "", "error": message}
        )
        self.files[0].flush()

    def close(self, plots=True):
        if not self.closed:
            self.metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
            self.metadata["duration_s"] = time.perf_counter() - self.start_monotonic
            write_json(self.path / "run_metadata.json", self.metadata)
            for f in self.files:
                f.close()
            self.errors.close()
            self.closed = True
        return generate_report(self.path, plots=plots)


def generate_report(path, plots=True):
    path = Path(path)
    meta = json.loads((path / "run_metadata.json").read_text(encoding="utf-8"))
    with (path / "frame_results.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    good = [r for r in rows if not r["error"]]

    def values(key):
        return [float(r[key]) for r in good if r.get(key) not in ("", None)]

    latency = {}
    for key in ("pc_capture_ms", "pc_preprocess_ms", "serial_roundtrip_ms", "transfer_ms", "quality_us",
                "preprocess_us", "inference_us", "total_us", "end_to_end_ms"):
        v = values(key)
        latency[key] = (
            {
                "count": len(v),
                "p50": float(np.percentile(v, 50)),
                "p95": float(np.percentile(v, 95)),
                "min": min(v),
                "max": max(v),
            }
            if v
            else "PENDING"
        )
    memory = {}
    for key in MEMORY:
        v = values(key)
        memory[key] = {"first": v[0], "last": v[-1], "minimum": min(v), "maximum": max(v)} if v else "PENDING"
    elapsed = values("elapsed_s")
    duration = max(meta.get("duration_s", 0), max(elapsed) if elapsed else 0)
    summary = {
        "evidence": meta["evidence"],
        "frames": len(good),
        "attempts": len(rows),
        "success_rate": len(good) / len(rows) if rows else "PENDING",
        "states": dict(Counter(r["state"] for r in good)),
        "accepted_classifications": sum(r["state"] in ("NORMAL", "SUSPECT") for r in good),
        "quality_rejections": sum(r["quality_passed"] == "False" for r in good),
        "errors": dict(Counter(r["error"] for r in rows if r["error"])),
        "duration_s": duration,
        "actual_fps": len(good) / duration if duration else "PENDING",
        "boot_ids": sorted({r["boot_id"] for r in good}),
        "memory": memory,
        "hardware_validation": "RECORDED" if good and meta["evidence"] == "hardware" else "PENDING",
    }
    saturation = values("saturated_fraction")
    summary["saturation_fraction"] = {
        "count": len(saturation), "p50": float(np.percentile(saturation, 50)),
        "p95": float(np.percentile(saturation, 95)), "maximum": max(saturation),
    } if saturation else "PENDING"
    window = max(1, len(good) // 5)
    summary["latency_trend"] = {}
    for key in ("inference_us", "end_to_end_ms"):
        v = values(key)
        summary["latency_trend"][key] = {"first_window_p50": float(np.median(v[:window])),
                                         "last_window_p50": float(np.median(v[-window:])),
                                         "window_frames": min(window, len(v))} if v else "PENDING"
    write_json(path / "latency_report.json", latency)
    write_json(path / "summary.json", summary)
    device = meta["device"]

    def metric(name):
        value = latency[name]
        return "PENDING" if isinstance(value, str) else f"{value['p50']:.3f} / {value['p95']:.3f}"

    report = [
        "# AgriTinyVision 硬件实验报告",
        "",
        f"实验时间：{meta['started_at']}",
        f"证据来源：{meta['evidence']}；硬件记录状态：{summary['hardware_validation']}",
        "",
        "| 项目 | 记录 |",
        "| --- | --- |",
        f"| PC 环境 | {meta['pc']['platform']} / Python {meta['pc']['python']} |",
    ]
    for key in (
        "chip",
        "board",
        "idf_version",
        "esp_dl_version",
        "firmware_version",
        "firmware_sha256",
        "model_version",
        "model_sha256",
        "model_bytes",
        "model_load_us",
        "app_image_bytes",
        "app_partition_bytes",
        "flash_chip_bytes",
        "boot_id",
        "reset_reason",
    ):
        report.append(f"| {key} | {device.get(key, 'PENDING')} |")
    report += [
        f"| 输入配置 | {json.dumps(meta['config'], ensure_ascii=False)} |",
        f"| 实验帧数 / 请求数 | {len(good)} / {len(rows)} |",
        f"| 协议响应成功率 | {summary['success_rate']} |",
        f"| 实际 FPS | {summary['actual_fps']} |",
        f"| 连续运行秒数 | {duration:.3f} |",
        f"| 推理 P50 / P95 (us) | {metric('inference_us')} |",
        f"| 传输至 ACK P50 / P95 (ms) | {metric('transfer_ms')} |",
        f"| 端到端 P50 / P95 (ms) | {metric('end_to_end_ms')} |",
        "",
        "## 内存及异常",
        "",
        "```json",
        json.dumps({"memory": memory, "errors": summary["errors"]}, indent=2),
        "```",
        "",
        "## 实验范围",
        "",
        "传输时间从 PC 开始发送到收到 MCU CRC 验收 ACK；包含链路及 ACK 往返。"
        "推理时间仅围绕 MCU model->run() 测量。空值表示该帧没有执行该步骤。",
        "Internal RAM 与 PSRAM 记录的是空闲量及设备启动以来最低空闲量；"
        "最低空闲量包含系统和通信缓冲，不能单独归因于模型。",
        "模型文件、app image、app 分区容量和 Flash 芯片容量分别记录。固件 ELF 大小明细由 idf.py size 输出。",
        "实验仅覆盖本目录记录的设备、画面和时段；现场准确率与功耗使用独立实验评估。",
        "",
    ]
    (path / "HARDWARE_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    if plots and good:
        _plots(path, good)
    return summary


def _plots(path, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = path / "plots"
    folder.mkdir(exist_ok=True)

    def series(key, scale=1):
        return [float(r[key]) / scale if r.get(key) else float("nan") for r in rows]

    def save(name, ylabel):
        plt.ylabel(ylabel)
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(folder / name, dpi=140)
        plt.close()

    plt.figure(figsize=(7, 4))
    v = [float(r["inference_us"]) / 1000 for r in rows if r.get("inference_us")]
    if v:
        plt.hist(v, bins=min(30, max(1, len(v))))
    plt.xlabel("MCU inference (ms)")
    save("latency_distribution.png", "Frames")
    plt.figure(figsize=(7, 4))
    plt.plot(series("elapsed_s"), series("actual_fps"))
    plt.xlabel("Elapsed (s)")
    save("fps_over_time.png", "Completed FPS")
    plt.figure(figsize=(7, 4))
    plt.plot(series("elapsed_s"), series("free_internal", 1024), label="Internal RAM free")
    plt.plot(series("elapsed_s"), series("free_psram", 1024), label="PSRAM free")
    plt.legend()
    plt.xlabel("Elapsed (s)")
    save("memory_usage.png", "KiB")
    plt.figure(figsize=(7, 4))
    plt.plot(series("elapsed_s"), series("transfer_ms"), label="Transfer to ACK")
    plt.plot(series("elapsed_s"), series("inference_us", 1000), label="MCU inference")
    plt.legend()
    plt.xlabel("Elapsed (s)")
    save("transfer_vs_inference.png", "ms")

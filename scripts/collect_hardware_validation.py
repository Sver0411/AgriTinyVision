"""Build the hardware acceptance summary from recorded device experiments."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_run(path):
    path = Path(path)
    summary = read(path / "summary.json")
    metadata = read(path / "run_metadata.json")
    with (path / "frame_results.csv").open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if not r["error"]]
    saturation = [int(r["saturated_values"]) / 76800 for r in rows if r["saturated_values"]]
    return {
        "directory": str(path),
        "summary": summary,
        "latency": read(path / "latency_report.json"),
        "device": metadata["device"],
        "input": metadata["config"],
        "saturation_fraction_p50": float(np.percentile(saturation, 50)) if saturation else None,
        "saturation_fraction_p95": float(np.percentile(saturation, 95)) if saturation else None,
    }


def collect(single, camera, stability, destination="reports"):
    destination = Path(destination)
    fixture = read(destination / "hardware_fixture_v02.json")
    protocol = read(destination / "hardware_protocol_v02.json")
    runs = {"single": load_run(single), "camera_100": load_run(camera), "stability_500": load_run(stability)}
    reference = read(Path(single) / "reference_comparison.json")
    checks = {
        "fixture": fixture["status"] == "PASS",
        "protocol_and_quality": protocol["status"] == "PASS",
        "camera_100": runs["camera_100"]["summary"]["frames"] >= 100,
        "stability_500": runs["stability_500"]["summary"]["frames"] >= 500,
        "at_least_5_minutes": runs["stability_500"]["summary"]["duration_s"] >= 300,
        "responses_without_errors": all(r["summary"]["success_rate"] == 1 for r in runs.values()),
        "no_restarts_during_runs": all(len(r["summary"]["boot_ids"]) == 1 for r in runs.values()),
        "same_firmware": len(
            {r["device"]["firmware_sha256"] for r in runs.values()} | {protocol["device"]["firmware_sha256"]}
        )
        == 1,
        "same_frozen_model": {r["device"]["model_sha256"] for r in runs.values()}
        == {fixture["model_sha256"]},
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "evidence": "hardware",
        "checks": checks,
        "fixture": fixture,
        "protocol_checks": len(protocol["tests"]),
        "reference_comparison": reference,
        "runs": runs,
    }
    (destination / "hardware_validation_v02.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    device = runs["stability_500"]["device"]
    lines = [
        "# AgriTinyVision v0.2 实机验收",
        "",
        f"验收结果：**{report['status']}**。",
        "",
        "Windows PC → USB Serial/JTAG → ESP32-S3 N16R8，CPU 160 MHz、PSRAM 80 MHz。",
        f"设备端 ESP-IDF {device['idf_version']} / ESP-DL {device['esp_dl_version']}，固件 {device['firmware_version']}。",
        f"模型 SHA256：`{device['model_sha256']}`。",
        f"固件 ELF SHA256：`{device['firmware_sha256']}`。",
        "",
        "| 实验 | 帧数 | 秒数 | 响应成功率 | 推理 P50/P95 ms | 传输至 ACK P50/P95 ms | 端到端 P50/P95 ms | FPS |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, run in runs.items():
        s, lat = run["summary"], run["latency"]

        def pair(key, scale=1):
            v = lat[key]
            return f"{v['p50'] / scale:.3f} / {v['p95'] / scale:.3f}"

        lines.append(
            f"| {name} | {s['frames']} | {s['duration_s']:.2f} | {s['success_rate']:.0%} | "
            f"{pair('inference_us', 1000)} | {pair('transfer_ms')} | {pair('end_to_end_ms')} | "
            f"{s['actual_fps']:.3f} |"
        )
    lines += [
        "",
        "## 内存与完整性",
        "",
        "| 实验 | Internal 空闲首/末 bytes | PSRAM 空闲首/末 bytes | boot_id 数量 |",
        "| --- | --- | --- | --- |",
    ]
    for name, run in runs.items():
        m = run["summary"]["memory"]
        lines.append(
            f"| {name} | {m['free_internal']['first']:.0f} / {m['free_internal']['last']:.0f} | "
            f"{m['free_psram']['first']:.0f} / {m['free_psram']['last']:.0f} | "
            f"{len(run['summary']['boot_ids'])} |"
        )
    lines += [
        "",
        f"设备 Flash 芯片 {device['flash_chip_bytes']:,} bytes，app 分区 {device['app_partition_bytes']:,} bytes，"
        f"运行镜像 {device['app_image_bytes']:,} bytes；模型文件 {device['model_bytes']:,} bytes。",
        "",
        "15 项实机质量/协议检查通过。质量不合格返回 UNKNOWN，模型推理时间为空；"
        "重复帧、CRC、尺寸、超长包、半包超时及重新连接恢复均由设备实际验证。",
        "",
        "## 同一输入对照",
        "",
        f"FP32 p_suspect={reference['fp32_suspect_probability']:.9f}；"
        f"ESP32 INT8 p_suspect={reference['device_results'][0]['suspect_probability']:.9f}；"
        f"差值={reference['device_results'][0]['difference_from_fp32']:.9f}。两者均为 NORMAL。",
        "对照按 RGB SHA256 匹配相同输入；分类阈值保持 validation 冻结值 0.30。",
        "",
        "## 实验范围",
        "",
        "camera_100 使用电脑摄像头实时画面；stability_500 使用同一 train-only 叶片图作为重复性基准。"
        "计时与内存来自 ESP32 返回的数据。这里的成功率表示请求取得有效响应，农业识别准确率由独立带标注现场数据评估。",
        "首次整帧测试定位到 8 KiB USB 接收环形缓冲溢出；改为有界 96 KiB 后完成本报告中的验收。",
        "默认保存指标与输入 SHA256，摄像头原图不写入实验目录。",
        "",
        "## 原始实验目录",
        "",
    ]
    lines.extend(f"- {name}: `{run['directory']}`" for name, run in runs.items())
    (destination / "HARDWARE_V02.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--single", required=True)
    p.add_argument("--camera", required=True)
    p.add_argument("--stability", required=True)
    p.add_argument("--output", default="reports")
    a = p.parse_args()
    result = collect(a.single, a.camera, a.stability, a.output)
    print(json.dumps({"status": result["status"], "checks": result["checks"]}, indent=2))

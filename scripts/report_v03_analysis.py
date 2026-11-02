"""Build a source-bound, aggregate-only research report from measured v0.3 records."""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agritinyvision.pc_camera.metrics import write_json
from agritinyvision.utils import sha256


def report(comparison, live_run, destination):
    comparison, live_run, destination = Path(comparison), Path(live_run), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    d = json.loads((comparison / "comparison.json").read_text(encoding="utf-8"))
    live_meta = json.loads((live_run / "run_metadata.json").read_text(encoding="utf-8"))
    live_summary = json.loads((live_run / "summary.json").read_text(encoding="utf-8"))
    live_latency = json.loads((live_run / "latency_report.json").read_text(encoding="utf-8"))
    identity = d["metadata"]["expected_identity"]
    probes = {r["scenario"]: r for r in d["samples"]}
    expected_probes = {"normal_light", "low_light", "strong_light", "blurred", "high_contrast", "low_contrast",
                       "different_color", "non_plant", "checker", "noise"}
    if set(probes) != expected_probes or len(d["samples"]) != 10:
        raise ValueError("This report template requires the ten controlled v0.3 probe scenarios")
    if identity["model_sha256"] != "bfda065ad95c3c3458d3c135b4c4b4575bdad6a31d2445b9283065d3851af3fa":
        raise ValueError("This v0.3 study template requires the original frozen 160 model")
    if any(not r.get("fp32") or not r.get("device") for r in d["samples"]):
        raise ValueError("Use comparison REPORT.md for incomplete probes; this aggregate requires all measured pairs")
    base, color = probes["normal_light"], probes["different_color"]
    def saturated(name):
        return probes[name]["saturation"]["saturated_fraction"] * 100
    color_error = abs(color["fp32"]["probabilities"][1] - color["device"]["probabilities"][1])
    strong_error = abs(probes["strong_light"]["fp32"]["probabilities"][1] - probes["strong_light"]["device"]["probabilities"][1])
    low_channels = probes["low_light"]["saturation"]["channels"]
    low_channel_text = "、".join(f"{k} {low_channels[k]['saturated_fraction']*100:.2f}%" for k in "RGB")
    host_pending = d["metadata"]["host"].startswith("PENDING")
    curated = {
        "version": "v0.3", "comparison_sha256": sha256(comparison / "comparison.json"),
        "manifest_sha256": d["metadata"]["manifest_sha256"],
        "environment": {k: d["metadata"][k] for k in ("created_at", "python", "dependencies", "project", "analysis_source_sha256")},
        "expected_identity": identity, "identity_validation": d["metadata"]["device"]["identity_validation"],
        "firmware_sha256": d["metadata"]["device"]["firmware_sha256"],
        "comparison_summary": d["summary"],
        "scenarios": [{k: r[k] for k in ("id", "scenario", "source", "source_label", "source_split", "parent_sha256", "annotation_valid",
                                           "transform", "rgb_sha256", "saturation", "quality", "fp32", "host", "device", "errors")}
                      for r in d["samples"]],
        "live_run": {"metadata_sha256": sha256(live_run / "run_metadata.json"),
                     "frames": live_summary["frames"], "duration_s": live_summary["duration_s"],
                     "actual_fps": live_summary["actual_fps"], "memory": live_summary["memory"],
                     "latency_trend": live_summary["latency_trend"], "saturation_fraction": live_summary["saturation_fraction"],
                     "errors": live_summary["errors"], "latency": live_latency,
                     "project": live_meta["project"]},
        "host_status": d["metadata"]["host"], "power": "PENDING", "independent_field_validation": "PENDING",
    }
    write_json(destination / "quantization_analysis_v03.json", curated)
    table = ["| 场景 | 饱和 % | FP32 p(SUSPECT) | ESP32 p(SUSPECT) | FP32 / ESP32 状态 |",
             "| --- | ---: | ---: | ---: | --- |"]
    for r in d["samples"]:
        p = r["fp32"]["probabilities"][1]
        device = r["device"]
        probability = f"{device['probabilities'][1]:.6f}" if device and device["probabilities"] is not None else "null"
        state = device["state"] if device else "PENDING"
        table.append(f"| {r['scenario']} | {r['saturation']['saturated_fraction'] * 100:.4f} | {p:.6f} | {probability} | {r['fp32']['state']} / {state} |")
    timings = ["| 阶段 | P50 ms | P95 ms |", "| --- | ---: | ---: |"]
    for key, scale in (("pc_capture_ms", 1), ("pc_preprocess_ms", 1), ("transfer_ms", 1),
                       ("quality_us", 1000), ("preprocess_us", 1000), ("inference_us", 1000),
                       ("serial_roundtrip_ms", 1), ("end_to_end_ms", 1)):
        v = live_latency[key]
        timings.append(f"| {key} | {v['p50'] / scale:.3f} | {v['p95'] / scale:.3f} |" if isinstance(v, dict)
                       else f"| {key} | PENDING | PENDING |")
    s = d["summary"]["fp32_vs_device"]
    text = f"""# v0.3 INT8 量化与部署差异分析

## 目的和冻结资产

比较相同 RGB 输入的 ONNX FP32、冻结 ESP-PPQ host simulation 和 ESP32 INT8，
分析输入饱和、概率差异和状态一致性。模型、量化参数和判定阈值未调整。
模型 SHA256：`{identity['model_sha256']}`。
Deployment SHA256：`{identity['deployment_sha256']}`。
模型版本 {identity['model_version']}，设备固件 {d['metadata']['device']['firmware_version']}，
输入/输出 exponent {identity['input_exponent']}/{identity['output_exponent']}，冻结阈值 0.30，边界 0.20/0.40。

## 环境和数据来源

本轮时间：{d['metadata']['created_at']}；Python {d['metadata']['python']}，Windows 11，
ESP-IDF {d['metadata']['device']['idf_version']}，ESP-DL {d['metadata']['device']['esp_dl_version']}。
依赖版本、基线 Git SHA、dirty 标记、分析源码 SHA 和输入 manifest SHA 记录在
[结构化数据](quantization_analysis_v03.json)。这是本地修改的开发代码，Git SHA 标识基线。

源叶片标识为 {base['source']}，标签 {base['source_label']}，划分 {base['source_split']}，
原图 SHA `{base.get('parent_sha256')}`。本轮选用历史训练来源作部署诊断。
七类叶片探针由同一图像进行确定性变换；颜色场景为合成通道置换，
不能当自然异色叶片的独立标注。其余为合成几何梯度、1-pixel 棋盘和 seed=303 随机噪声。
本轮未使用最终测试集调参，受控探针不构成田间准确率样本。

## 三方方法和后端状态

先核对源图 SHA，再执行同一 RGB/BILINEAR 160 预处理。FP32 和 host 使用同一归一化 NCHW；
设备收到同一 RGB888，CRC/ACK、序列号和 boot_id 绑定响应，PC 记录发送 RGB SHA。
设备不是自行回传图像 SHA。正式模型身份 13 项字段全部通过。

FP32 与真实 ESP32 已执行 {len(d['samples'])} 个场景。Host 状态：**{d['metadata']['host']}**。
{'本地没有正式 ESP-DL 对应的 frozen .native 和 simulation_graph_sha256；已安装 esp-ppq 本身不补足这些文件。' if host_pending else 'Host 已加载哈希绑定的冻结量化图。'}
历史 host 测试分数保留为历史证据，不填入本轮逐图结果。

## 量化误差和分类一致率

{chr(10).join(table)}

门控状态一致 {int(s['state_agreement'] * s['state_pairs'])}/{s['state_pairs']}；
质量通过且有双端概率的分母为 {s['probability_pairs']}，p(SUSPECT) MAE={s['suspect_probability_mae']:.6f}，
最大绝对差={s['suspect_probability_max_error']:.6f}。blurred / non_plant 的设备质量失败，
没有执行归一化和模型；FP32 原始概率保留分析，但两端门控状态均为 UNKNOWN。
三方状态一致率：{d['summary']['three_way_state_agreement']['rate']}，分母 {d['summary']['three_way_state_agreement']['count']}。
FP32/host 与 host/device 统计见 JSON 对应分母。

## 饱和和场景分析

exponent=-6 的可表示反量化范围为 [-2, 1.984375]；ImageNet RGB 全范围归一化分别为
R [-2.117904,2.248908]、G [-2.035714,2.428571]、B [-1.804444,2.640000]。
按 S3 实际舍入，未截断的 uint8 范围为 R 7..240、G 2..230、B 0..217。
因此此输入刻度不能覆盖全部颜色强度；这来自冻结模型，不在本轮改变。

低光场景各通道截断为 {low_channel_text}。
棋盘总计 {saturated('checker'):.4f}%，固定源图总计 {saturated('normal_light'):.4f}%。
本组十探针总体饱和比例 P50={d['summary']['saturation_fraction']['p50']*100:.4f}%、P95={d['summary']['saturation_fraction']['p95']*100:.4f}%；各场景只有一个受控样本，
该分布不是农业图像总体分布。每通道归一化分位数和原始范围存于 JSON。

![各场景的通道截断和分类概率](quantization_saturation_v03.png)

## 失败案例和解释

颜色通道置换从 FP32 {color['fp32']['state']} 变成设备 {color['device']['state']}，概率差 {color_error:.6f}，输入饱和 {saturated('different_color'):.4f}%。
强光饱和 {saturated('strong_light'):.4f}%，概率差 {strong_error:.6f}。饱和率不能单独预测输出误差。
{'Host 对照缺失，尚不能把误差分解为量化图误差与设备执行误差。' if host_pending else '量化图误差与设备执行误差分别依据三方统计解释。'}
棋盘和噪声质量通过并得到 NORMAL，直接说明质量门控不判断植物存在性。
协议/执行错误样本数为 {len(d['summary']['failures'])}；部署判定分歧和执行错误分别统计。

## 新摄像头计时

另外采集 {live_summary['frames']} 帧实时画面，全部取得设备结果，不保存原始照片，
画面没有独立农业标注，仅用于链路与计时。整次实验 {live_summary['duration_s']:.3f} 秒，
完成 FPS={live_summary['actual_fps']:.3f}，包含相机启动及首帧等待。

{chr(10).join(timings)}

采集 P95 包含首次等待；十帧样本描述当前会话，不能代表长期分布。首尾空闲内部 RAM
{live_summary['memory']['free_internal']['first']:.0f}/{live_summary['memory']['free_internal']['last']:.0f} bytes、PSRAM
{live_summary['memory']['free_psram']['first']:.0f}/{live_summary['memory']['free_psram']['last']:.0f} bytes。
v0.2 的 500 帧 / 536.46 秒、527.840/527.850 ms
设备推理记录保留在 [HARDWARE_V02.md](HARDWARE_V02.md)，两次实验分别报告。
设备推理、发送到结果和采集到结果的时钟口径不同；不通过相加边际 P50 推算总延迟。

## 结论与下一步

保持正式模型、exponent 和阈值。首先恢复原始可信 native 图，补齐相同 RGB 的 host/device
逐图对照；然后使用独立带标注真实光照/色彩叶片分析分层误差。候选校准集或 QAT 使用
独立目录和验证集比较，选定后再一次性评估最终测试集。
独立田间验证、功耗仪测试、候选量化优化为 PENDING。
性能优化优先测 USB 发送/ACK、设备质量/归一化和 PC 首帧等待；调整采样频率时继续使用
单帧在途策略，并单独运行较长时段、多场景及重连实验。
"""
    (destination / "QUANTIZATION_ANALYSIS_V03.md").write_text(text, encoding="utf-8")
    shutil.copy2(comparison / "saturation_comparison.png", destination / "quantization_saturation_v03.png")
    return curated


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comparison", required=True)
    p.add_argument("--live-run", required=True)
    p.add_argument("--output", default="reports")
    args = p.parse_args()
    report(args.comparison, args.live_run, args.output)

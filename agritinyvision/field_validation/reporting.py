"""Batch reports with explicit denominators and separate prediction/annotation evidence."""

import csv
import json
from pathlib import Path

import numpy as np

from agritinyvision.pc_camera.metrics import write_json
from agritinyvision.utils import sha256

from .annotations import labels
from .capture import read_json
from .evaluation import STATES, evaluate_records, load_records, reference

REPORT_FILES = {"EXPERIMENT_SUMMARY.md", "DATASET_STATISTICS.json", "INFERENCE_RESULTS.csv",
                "QUALITY_GATE_REPORT.json", "THREE_STATE_REPORT.json", "INFERENCE_PARITY.json",
                "INFERENCE_PARITY.md", "CONFUSION_MATRIX.png", "SCENARIO_COMPARISON.png",
                "UNKNOWN_DISTRIBUTION.png", "SCENARIO_STATISTICS.json"}


def text(value):
    return "PENDING" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)


def generate_batch_report(batch, annotation_source="manual", min_scenario_samples=5):
    batch = Path(batch)
    records = load_records(batch, annotation_source)
    result = evaluate_records(records, min_scenario_samples)
    out = batch / "reports"
    meta = read_json(batch / "batch.json")
    statistics = {**result["dataset"], "annotation_source": annotation_source,
                  "manual_reviews": len(labels(batch)), "source_labels": len(labels(batch, "source")),
                  "source_kind": meta["source_kind"], "context": meta["context"],
                  "scene_support_threshold": min_scenario_samples,
                  "batch_sha256": sha256(batch / "batch.json"), "manifest_sha256": sha256(batch / "manifest.jsonl"),
                  "annotation_sha256": sha256(batch / f"annotations/{annotation_source}.jsonl")
                  if (batch / f"annotations/{annotation_source}.jsonl").exists() else None}
    write_json(out / "DATASET_STATISTICS.json", statistics)
    write_json(out / "QUALITY_GATE_REPORT.json", result["quality"])
    write_json(out / "THREE_STATE_REPORT.json", result["three_state"])
    write_json(out / "SCENARIO_STATISTICS.json", result["scenarios"])
    parity_file = out / "INFERENCE_PARITY.json"
    if not parity_file.exists():
        write_json(parity_file, {"status": "PENDING", "reason": "Run frozen desktop references on saved inputs"})
    parity = read_json(parity_file)
    backend_classification = {}
    parity_rows = {r["sample_id"]: r for r in parity.get("samples", [])}
    for name in ("fp32", "host"):
        engine_rows = []
        for row in records:
            counterpart = parity_rows.get(row["sample_id"], {})
            prediction = counterpart.get(name)
            if prediction and prediction["probabilities"] is not None:
                engine_rows.append({**row, "prediction": {"status": "ok"}, "result": {
                    "state": prediction["state"], "suspect_probability": prediction["probabilities"][1],
                    "quality_passed": counterpart["quality"]["passed"]}})
        backend_classification[name] = evaluate_records(engine_rows)["three_state"] if engine_rows else None
    write_json(out / "THREE_STATE_REPORT.json", {**result["three_state"], "backend_classification": backend_classification})
    result["backend_classification"] = backend_classification
    fields = ["sample_id", "rgb_sha256", "label_origin", "reference_label", "human_quality", "device_status",
              "normal_probability", "suspect_probability", "state", "quality_passed", "reason", "inference_us",
              "end_to_end_ms", "saturated_fraction", "excluded_from_evaluation"]
    with (out / "INFERENCE_RESULTS.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for row in records:
            r = row["result"] or {}
            writer.writerow({"sample_id": row["sample_id"], "rgb_sha256": row["metadata"]["rgb_sha256"],
                "label_origin": row["annotation"].get("label_origin"), "reference_label": reference(row),
                "human_quality": row["annotation"].get("image_quality"), "device_status": row["prediction"]["status"],
                **{key: r.get(key) for key in fields if key in r},
                "saturated_fraction": r["saturated_values"] / 76800 if r.get("saturated_values") is not None else None,
                "excluded_from_evaluation": bool(row["metadata"]["context"].get("excluded_from_evaluation"))})
    valid = [r["result"] for r in records if r["result"]]
    latencies = [r["inference_us"] / 1000 for r in valid if r["inference_us"] is not None]
    memory = {k: {"minimum": min(r[k] for r in valid), "maximum": max(r[k] for r in valid)}
              for k in ("free_internal", "free_psram") if valid}
    s = result["three_state"]
    report = ["# 农业图像批次评价", "", f"批次：{meta['batch_id']}；开始：{meta['started_at']}",
              f"来源：{meta['source_kind']}；设备证据：{meta['evidence']}；评价标签：{annotation_source}",
              f"Git：{meta['project']['git_commit']}；dirty={meta['project']['git_dirty']}",
              f"固件 / 模型：{meta['device']['firmware_version']} / {meta['device']['model_version']}",
              f"模型 SHA：`{meta['device']['model_sha256']}`", f"固件 SHA：`{meta['device'].get('firmware_sha256')}`", "",
              "| 项目 | 数值 |", "| --- | ---: |",
              f"| 总图片 / 成功 / 排除 | {len(records)} / {result['dataset']['successful']} / {result['dataset']['excluded']} |",
              f"| 独立参考标签 / 人工复核 | {s['reference_samples']} / {statistics['manual_reviews']} |",
              f"| 概率可用二分类 Accuracy / n | {text(s['binary_probability_cohort']['accuracy'])} / {s['binary_probability_cohort']['n']} |",
              f"| 接受覆盖率 | {text(s['accepted_coverage'])} |", f"| 接受输入准确率 | {text(s['accepted']['accuracy'])} |",
              f"| 接受输入错误率 | {text(s['accepted_error_rate'])} |", f"| UNKNOWN 拒绝率 | {text(s['unknown_rejection_rate'])} |",
              f"| SUSPECT→NORMAL / SUSPECT→UNKNOWN | {text(s['suspect_missed_as_normal_rate'])} / {text(s['suspect_deferred_unknown_rate'])} |",
              f"| 正常叶片误报率 | {text(s['normal_false_alarm_rate'])} |",
              f"| 人工质量评价 | {result['quality']['status']}，n={result['quality']['n']} |",
              f"| 设备推理 P50 / P95 ms | {text(float(np.percentile(latencies,50)) if latencies else None)} / {text(float(np.percentile(latencies,95)) if latencies else None)} |", "",
              "## 指标口径", "", "UNKNOWN 计为拒绝，不计为正确分类。二分类使用冻结 0.30；系统使用冻结 0.20/0.40。",
              "概率可用二分类、质量通过、已接受与人工 USABLE 的分母分别保存，不能相互替代。",
              "质量拒绝设备没有执行模型，其反事实分类未知。概率区间拒绝可对照同一输入强制二分类。", "",
              "## UNKNOWN", "", "```json", json.dumps(s["unknown_analysis"], indent=2, ensure_ascii=False), "```", "",
              "## 内存", "", "```json", json.dumps(memory, indent=2), "```", "",
              "## 推理环境对照", "", "```json", json.dumps(parity.get("summary", parity), indent=2, ensure_ascii=False), "```", "",
              "## 失败与延后判断样本", "", "| Sample | 参考 | 设备 |", "| --- | --- | --- |"]
    report += [f"| {r['sample_id']} | {r['truth']} | {r['state']} |" for r in result["failures"]]
    report += ["", "## 实验范围", "", "来源标签是独立数据集作者的类别或显式人工标注；预测不生成真实标签。",
               "人工质量/光照/距离未记录时相应评价为 PENDING 或 UNRECORDED；小场景只报告计数与样本不足。",
               "公开互联网农业图像与本地田间摄像头采集分别记录；诊断性子集指标不外推到完整田间总体。",
               "植物存在性、完整 OOD 和具体病害诊断不由质量门控提供。",
               "执行错误单列，不转成 UNKNOWN。分组划分结果单独保存，组不足时不制造独立验证/测试集合。", ""]
    (out / "EXPERIMENT_SUMMARY.md").write_text("\n".join(report), encoding="utf-8")
    plots(out, result)
    return result


def plots(out, result):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.array(result["three_state"]["three_state_matrix"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(matrix, cmap="Greens", vmin=0, vmax=max(1, int(matrix.max())))
    for (i, j), count in np.ndenumerate(matrix):
        ax.text(j, i, str(count), ha="center", va="center", color="white" if count > matrix.max() / 2 else "black")
    ax.set(xticks=range(3), xticklabels=STATES, yticks=range(2), yticklabels=STATES[:2], xlabel="Device prediction", ylabel="Reference label")
    fig.tight_layout()
    fig.savefig(out / "CONFUSION_MATRIX.png", dpi=140)
    plt.close(fig)
    scenarios = result["scenarios"]["crop_type"]
    names = list(scenarios)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, key in enumerate(("accepted_coverage", "accepted_error_rate", "unknown_rejection_rate")):
        ax.bar(np.arange(len(names)) + (i - 1) * .25,
               [scenarios[n][key] if scenarios[n][key] is not None else np.nan for n in names], width=.25,
               label=("Coverage", "Accepted error", "UNKNOWN rate")[i])
    ax.set_xticks(range(len(names)), [n + (" (small n)" if scenarios[n]["support"] == "insufficient" else "") for n in names])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction (see JSON denominators)")
    if not names:
        ax.text(.5, .5, "PENDING: no samples", ha="center", transform=ax.transAxes)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5, 1.15), ncol=3)
    fig.tight_layout()
    fig.savefig(out / "SCENARIO_COMPARISON.png", dpi=140)
    plt.close(fig)
    unknown = result["three_state"]["unknown_analysis"]
    fig, ax = plt.subplots(figsize=(6, 4))
    names = ["quality_rejected", "probability_rejected", "other_rejected"]
    ax.bar(["Quality", "Probability", "Other"], [unknown[n] for n in names])
    ax.set_ylabel("UNKNOWN labelled samples")
    fig.tight_layout()
    fig.savefig(out / "UNKNOWN_DISTRIBUTION.png", dpi=140)
    plt.close(fig)

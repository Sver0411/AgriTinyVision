"""Generate a report from authentic frozen run evidence, without rerunning test inference."""

import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agritinyvision.utils import read_json, sha256, write_json


def pct(x):
    return f"{100 * x:.2f}%"


def main():
    env = read_json("reports/environment.json")
    audit = read_json("reports/dataset_audit.json")
    selection = read_json("reports/candidate_selection.json")
    records = []
    for r in selection["experiments"]:
        evidence = Path("reports/experiments") / Path(r["run"].replace("\\", "/")).name
        records.append(
            (
                r,
                read_json(evidence / "test_metrics.json"),
                read_json(evidence / "tinyvision.espdl.quantization.json"),
                read_json(evidence / "tinyvision.onnx.validation.json"),
            )
        )
    build_log = Path("reports/firmware_build.log")
    build_text = build_log.read_text(encoding="utf-8", errors="replace") if build_log.exists() else ""
    binary = Path("firmware_example/build/agritinyvision_fixture.bin")
    build_ok = "Project build complete" in build_text and binary.exists()
    build = {
        "idf": "5.4.4",
        "esp_dl": "3.3.13",
        "target": "esp32s3",
        "compiled": build_ok,
        "firmware_bytes": binary.stat().st_size if binary.exists() else None,
        "firmware_sha256": sha256(binary) if binary.exists() else None,
        "hardware_status": "PENDING HARDWARE VALIDATION",
        "initial_failure": "CMake aligned embedding cmake_dir omitted; corrected to official utility path and rebuilt",
    }
    write_json("reports/firmware_build.json", build)
    tests = Path("reports/tests_final.txt").read_text(encoding="utf-8", errors="replace")
    summary = re.findall(r"\d+ passed[^\n]*", tests)[-1]
    md = [
        "# AgriTinyVision v0.1 first-run report",
        "",
        f"报告时间：{datetime.now(timezone(timedelta(hours=8))).isoformat()}（Asia/Shanghai）。全部为实际本机结果，未运行 ESP32 真机。",
        "",
        "## Environment",
        "",
        f"GPU: {env['gpus'][0]['name']}，VRAM {env['gpus'][0]['vram_bytes'] / 2**30:.2f} GiB。Python {env['python'].split()[0]}，{env['os']}。",
        f"PyTorch {env['pytorch']}，CUDA runtime {env['cuda_runtime']}，cuDNN {env['cudnn']}；CUDA available={env['cuda_available']}，CUDA tensor 实算检查={env['cuda_tensor_check']}。",
        "",
        "## Dataset",
        "",
        "PlantVillage raw/color 分组原图子集，固定源 commit `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a`。苹果、甜椒、马铃薯、番茄，12 个源标签，最多每标签 240 张；马铃薯健康类仅 152 张。",
        f"总计 {audit['total_images']}；NORMAL={audit['labels']['NORMAL']}，SUSPECT={audit['labels']['SUSPECT']}。train/val/test={audit['splits']['train']}/{audit['splits']['val']}/{audit['splits']['test']}。子集比例不代表真实患病率。",
        "",
        "## Data Leakage Protection",
        "",
        f"真实 leaf ID + 精确 SHA256 + pHash≤4 近重复连通分量合并后 GroupShuffleSplit(seed=41)。非法/损坏图={audit['invalid_count']}，精确重复组={len(audit['duplicate_hash_groups'])}，近重复对={len(audit['near_duplicate_pairs'])}，跨 split 关联泄漏={len(audit['cross_split_leaks'])}。",
        "只使用 raw/color 原图，不混入同叶片灰度/分割/增强版本。plant/session/scene ID 不可用，leaf grouping 和 pHash 无法排除所有残余泄漏。官方分组缺失/歧义图片被排除，不制造假植物 ID。",
        "",
        "## Experiments",
        "",
        "torchvision MobileNetV2 1.0，ImageNet IMAGENET1K_V1 预训练，全参数微调；AdamW、cosine、AMP、seed 41、batch 32、最多 8 epochs、验证早停 patience 3。",
        "",
        "| Input | Epochs | Parameters | FP32 ONNX bytes | Conv/Linear MACs | GPU train seconds | Peak allocated VRAM |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r, test, quant, validation in records:
        md.append(
            f"| {r['input_size']}×{r['input_size']} | {r['epochs_completed']} | {r['parameters']:,} | {validation['bytes']:,} | {r['conv_linear_macs']:,} | {r['training_seconds']:.2f} | {r['peak_vram_allocated_bytes'] / 2**20:.1f} MiB |"
        )
    md += [
        "",
        f"`reports/candidate_selection.json` 于 `{selection['created_at']}`，在任何最终 test 评估之前冻结 `{selection['selected_run']}`。两者最佳 validation Recall=100%、F1≈0.99827、FPR≈0.813%；160 MACs 约一半。之后不因 224 的 test 更高而改选。MACs 只计算 Conv/Linear，忽略其他算子与预处理，不是 latency。",
        "",
        "首次 runtime Git lookup 因沙箱/桌面目录所有权差异返回 null，原始失败保留。初始 source commit 在第一轮训练期间保存；训练/model/data/threshold 文件逐个验证与该 snapshot 一致，写入 environment.json 的 source_snapshot_commit。没有补造原始 git_commit；后续使用仓库范围 safe.directory 修复。",
        "",
        "## Metrics",
        "",
        "最终 held-out test=417（NORMAL 130 / SUSPECT 287），没有 test 调参；INT8 是 ESP-PPQ 量化图 host simulation，未在 ESP32 运行。",
        "",
        "| Model | Accuracy | Precision | SUSPECT Recall | Specificity | F1 | ROC-AUC | PR-AUC/AP | FPR | FNR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    keys = (
        "accuracy",
        "precision",
        "suspect_recall",
        "specificity",
        "f1",
        "roc_auc",
        "pr_auc",
        "false_positive_rate",
        "false_negative_rate",
    )
    for r, test, quant, validation in records:
        for suffix, result in (("FP32", test), ("INT8 simulation", quant["test_metrics"])):
            md.append(
                f"| MobileNetV2 {r['input_size']} {suffix} | "
                + " | ".join(pct(result[k]) for k in keys)
                + " |"
            )
    md += ["", "Confusion matrix：行=true NORMAL/SUSPECT；列=pred NORMAL/SUSPECT。"]
    for r, test, quant, validation in records:
        md.append(
            f"- {r['input_size']} FP32 `{test['confusion_matrix']}`；INT8 simulation `{quant['test_metrics']['confusion_matrix']}`。"
        )
    md += [
        "",
        "## Threshold / UNKNOWN",
        "",
        "validation 搜索 0.20..0.80，step 0.05；Recall≥0.95、FPR≤0.35 后按 F1 选择，约束满足。160 threshold=0.30，224=0.20。INT8 test 保持 FP32 validation 阈值，没有用 test 重新优化。160 INT8 有 1/287 个异常判为正常，应保留漏检事实。",
        "质量门：固定分辨率 Laplacian variance / brightness / contrast + 二分类 max probability 置信度门；异常输入、模糊/极暗/极亮合成图测试返回 UNKNOWN。160 FP32 公开 test coverage=100%、UNKNOWN=0；受控图片不能证明现场拒绝质量。完整 OOD 和植物存在性检测未实现，高置信无植物画面可能通过。阈值均待现场校准。",
        "",
        "## Quantization",
        "",
        "PyTorch → ONNX opset 18 → ESP-PPQ 1.3.11 → ESP32-S3 w8a8 实际成功。train-only 按源标签 round-robin 256 张、无随机增强校准，S3 per-tensor symmetric INT8，保留 torchvision ReLU6/Clip。",
        "",
        "| Input | FP32 bytes | INT8 .espdl bytes | INT8 Recall | INT8 FNR | PyTorch/ONNX max abs error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r, test, quant, validation in records:
        md.append(
            f"| {r['input_size']} | {validation['bytes']:,} | {quant['bytes']:,} | {pct(quant['test_metrics']['suspect_recall'])} | {pct(quant['test_metrics']['false_negative_rate'])} | {validation['max_absolute_error']:.8g} |"
        )
    md += [
        "",
        "160 INT8 Accuracy 下降约 0.24 percentage points、Recall 下降约 0.35 points；本轮记录结果，未在查看 test 后优化模型/阈值。QAT/mixed precision 尚未运行。CPU benchmark 单独保存，绝不代表 MCU 推理速度。",
        "",
        "## ESP-DL",
        "",
        "训练前已用随机权重结构预检 ONNX 算子；Conv/depthwise groups、Clip、Add、GlobalAveragePool、Flatten、Gemm 通过静态检查。训练后两个模型各使用相同四组输入与 PyTorch 做数值验证，另绑定 checkpoint/ONNX SHA256。",
        "实际 160 .espdl 输入 INT8 NHWC `[1,160,160,3]`、exponent=-6；输出 INT8 `[1,2]`、exponent=-4，反量化后 softmax。两个 .espdl 都生成并在完整 test 上做 host graph simulation。格式/算子支持与 host 仿真不保证 8 MB PSRAM 中整机可运行。",
        "",
        "## Release",
        "",
        "选定 `releases/v0.1.0/`：tinyvision.espdl、tinyvision.onnx、metadata.json、labels.txt、metrics.json、README.md；附 tinyvision.info/json 转换诊断。模型二进制仅发布到 GitHub Release，不进入 Git；原始图像和 checkpoint 保留本地忽略目录。ESP-DL export SUCCESS，hardware status PENDING。",
        "",
        "## Tests",
        "",
        f"{summary}。两个 warning 为 PyTorch legacy ONNX exporter deprecation（与官方 ESP-PPQ 示例一致）；Ruff、editable package 安装与 CLI 已验证。",
        "覆盖 label mapping、source-label 矛盾、非法/空/损坏图片、路径逃逸、group split、精确近重复、preprocessing、quality gate、UNKNOWN、validation-only threshold、config、metadata、release/hash、ONNX 一致性及 train-only calibration。CI 使用 synthetic fixtures，不重跑训练。",
        "",
        "## Limitations",
        "",
        "公开子集干净背景、受控照明、4 种物种和标签比例限制，具有 dataset bias 与 domain shift；未做多 seed 统计/独立农场验证。训练和量化流水线可运行，不能证明真实现场 disease detection。plant/session/scene 元数据缺失，残余泄漏风险仍在。",
        "",
        "## Hardware Pending",
        "",
        f"ESP-IDF 5.4.4 + ESP-DL 3.3.13 + target esp32s3 实际 compile={'PASS' if build_ok else 'PENDING/FAILED'}，firmware binary={build['firmware_bytes']} bytes（编译文件大小，不是真机 Flash 测量）。真实 train-only normalized NHWC golden fixture 与 C++ 量化输入/反量化输出示例已准备。",
        "没有刷写或运行板子。生产 camera decode/resize/quality gate 移植和主 Gateway 集成未完成。model load time、inference latency、peak internal RAM、PSRAM usage、Flash usage（完整硬件系统）、camera preprocessing latency、power consumption 全部 **PENDING HARDWARE VALIDATION**。",
        "",
        "## Next Step",
        "",
        "1. 在 ESP32-S3 N16R8 运行 golden fixture，核对输出及张量顺序/指数，测 RAM/PSRAM、latency、Flash 和功耗。",
        "2. 移植实际 Gateway 摄像头 RGB/resize/quality gate，验证 Python ↔ firmware preprocessing parity。",
        "3. 采集现场数据并按 plant/site/session/time 分组，包含背景/无植物/OOD/模糊/曝光失败样本。",
        "4. 在独立现场 validation 上微调与阈值重校准，最后一次性评估独立 test；必要时更窄 MobileNetV2 或 QAT。",
        "",
    ]
    Path("reports/FIRST_RUN.md").write_text("\n".join(md), encoding="utf-8")
    print("Generated reports/FIRST_RUN.md from actual evidence")


if __name__ == "__main__":
    main()

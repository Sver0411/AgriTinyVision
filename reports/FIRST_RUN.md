# AgriTinyVision v0.1 first-run report

报告时间：2026-10-06T01:23:57.904043+08:00（Asia/Shanghai）。全部为实际本机结果，未运行 ESP32 真机。

## Environment

GPU: NVIDIA GeForce RTX 4060 Laptop GPU，VRAM 8.00 GiB。Python 3.12.15，Windows-11-10.0.22621-SP0。
PyTorch 2.11.0+cu128，CUDA runtime 12.8，cuDNN 91900；CUDA available=True，CUDA tensor 实算检查=True。

## Dataset

PlantVillage raw/color 分组原图子集，固定源 commit `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a`。苹果、甜椒、马铃薯、番茄，12 个源标签，最多每标签 240 张；马铃薯健康类仅 152 张。
总计 2792；NORMAL=872，SUSPECT=1920。train/val/test=1964/411/417。子集比例不代表真实患病率。

## Data Leakage Protection

真实 leaf ID + 精确 SHA256 + pHash≤4 近重复连通分量合并后 GroupShuffleSplit(seed=41)。非法/损坏图=0，精确重复组=0，近重复对=1，跨 split 关联泄漏=0。
只使用 raw/color 原图，不混入同叶片灰度/分割/增强版本。plant/session/scene ID 不可用，leaf grouping 和 pHash 无法排除所有残余泄漏。官方分组缺失/歧义图片被排除，不制造假植物 ID。

## Experiments

torchvision MobileNetV2 1.0，ImageNet IMAGENET1K_V1 预训练，全参数微调；AdamW、cosine、AMP、seed 41、batch 32、最多 8 epochs、验证早停 patience 3。

| Input | Epochs | Parameters | FP32 ONNX bytes | Conv/Linear MACs | GPU train seconds | Peak allocated VRAM |
|---|---:|---:|---:|---:|---:|---:|
| 160×160 | 8 | 2,226,434 | 8,878,502 | 152,805,760 | 130.34 | 772.3 MiB |
| 224×224 | 7 | 2,226,434 | 8,878,502 | 299,496,832 | 126.74 | 1368.5 MiB |

`reports/candidate_selection.json` 于 `2026-10-05T17:05:49.443203+00:00`，在任何最终 test 评估之前冻结 `runs\baseline160`。两者最佳 validation Recall=100%、F1≈0.99827、FPR≈0.813%；160 MACs 约一半。之后不因 224 的 test 更高而改选。MACs 只计算 Conv/Linear，忽略其他算子与预处理，不是 latency。

首次 runtime Git lookup 因沙箱/桌面目录所有权差异返回 null，原始失败保留。初始 source commit 在第一轮训练期间保存；训练/model/data/threshold 文件逐个验证与该 snapshot 一致，写入 environment.json 的 source_snapshot_commit。没有补造原始 git_commit；后续使用仓库范围 safe.directory 修复。

## Metrics

最终 held-out test=417（NORMAL 130 / SUSPECT 287），没有 test 调参；INT8 是 ESP-PPQ 量化图 host simulation，未在 ESP32 运行。

| Model | Accuracy | Precision | SUSPECT Recall | Specificity | F1 | ROC-AUC | PR-AUC/AP | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MobileNetV2 160 FP32 | 99.76% | 99.65% | 100.00% | 99.23% | 99.83% | 99.99% | 100.00% | 0.77% | 0.00% |
| MobileNetV2 160 INT8 simulation | 99.52% | 99.65% | 99.65% | 99.23% | 99.65% | 99.99% | 100.00% | 0.77% | 0.35% |
| MobileNetV2 224 FP32 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% | 0.00% |
| MobileNetV2 224 INT8 simulation | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% | 0.00% |

Confusion matrix：行=true NORMAL/SUSPECT；列=pred NORMAL/SUSPECT。
- 160 FP32 `[[129, 1], [0, 287]]`；INT8 simulation `[[129, 1], [1, 286]]`。
- 224 FP32 `[[130, 0], [0, 287]]`；INT8 simulation `[[130, 0], [0, 287]]`。

## Threshold / UNKNOWN

validation 搜索 0.20..0.80，step 0.05；Recall≥0.95、FPR≤0.35 后按 F1 选择，约束满足。160 threshold=0.30，224=0.20。INT8 test 保持 FP32 validation 阈值，没有用 test 重新优化。160 INT8 有 1/287 个异常判为正常，应保留漏检事实。
质量门：固定分辨率 Laplacian variance / brightness / contrast + 二分类 max probability 置信度门；异常输入、模糊/极暗/极亮合成图测试返回 UNKNOWN。160 FP32 公开 test coverage=100%、UNKNOWN=0；受控图片不能证明现场拒绝质量。完整 OOD 和植物存在性检测未实现，高置信无植物画面可能通过。阈值均待现场校准。

## Quantization

PyTorch → ONNX opset 18 → ESP-PPQ 1.3.11 → ESP32-S3 w8a8 实际成功。train-only 按源标签 round-robin 256 张、无随机增强校准，S3 per-tensor symmetric INT8，保留 torchvision ReLU6/Clip。

| Input | FP32 bytes | INT8 .espdl bytes | INT8 Recall | INT8 FNR | PyTorch/ONNX max abs error |
|---|---:|---:|---:|---:|---:|
| 160 | 8,878,502 | 2,438,752 | 99.65% | 0.35% | 1.0490417e-05 |
| 224 | 8,878,502 | 2,511,296 | 100.00% | 0.00% | 4.8577785e-06 |

160 INT8 Accuracy 下降约 0.24 percentage points、Recall 下降约 0.35 points；本轮记录结果，未在查看 test 后优化模型/阈值。QAT/mixed precision 尚未运行。CPU benchmark 单独保存，绝不代表 MCU 推理速度。

## ESP-DL

训练前已用随机权重结构预检 ONNX 算子；Conv/depthwise groups、Clip、Add、GlobalAveragePool、Flatten、Gemm 通过静态检查。训练后两个模型各使用相同四组输入与 PyTorch 做数值验证，另绑定 checkpoint/ONNX SHA256。
实际 160 .espdl 输入 INT8 NHWC `[1,160,160,3]`、exponent=-6；输出 INT8 `[1,2]`、exponent=-4，反量化后 softmax。两个 .espdl 都生成并在完整 test 上做 host graph simulation。格式/算子支持与 host 仿真不保证 8 MB PSRAM 中整机可运行。

## Release

选定 `releases/v0.1.0/`：tinyvision.espdl、tinyvision.onnx、metadata.json、labels.txt、metrics.json、README.md；附 tinyvision.info/json 转换诊断。模型二进制仅发布到 GitHub Release，不进入 Git；原始图像和 checkpoint 保留本地忽略目录。ESP-DL export SUCCESS，hardware status PENDING。

## Tests

36 passed, 2 warnings in 5.22s。两个 warning 为 PyTorch legacy ONNX exporter deprecation（与官方 ESP-PPQ 示例一致）；Ruff、editable package 安装与 CLI 已验证。
覆盖 label mapping、source-label 矛盾、非法/空/损坏图片、路径逃逸、group split、精确近重复、preprocessing、quality gate、UNKNOWN、validation-only threshold、config、metadata、release/hash、ONNX 一致性及 train-only calibration。CI 使用 synthetic fixtures，不重跑训练。

## Limitations

公开子集干净背景、受控照明、4 种物种和标签比例限制，具有 dataset bias 与 domain shift；未做多 seed 统计/独立农场验证。训练和量化流水线可运行，不能证明真实现场 disease detection。plant/session/scene 元数据缺失，残余泄漏风险仍在。

## Hardware Pending

ESP-IDF 5.4.4 + ESP-DL 3.3.13 + target esp32s3 实际 compile=PASS，firmware binary=3838640 bytes（编译文件大小，不是真机 Flash 测量）。真实 train-only normalized NHWC golden fixture 与 C++ 量化输入/反量化输出示例已准备。
没有刷写或运行板子。生产 camera decode/resize/quality gate 移植和主 Gateway 集成未完成。model load time、inference latency、peak internal RAM、PSRAM usage、Flash usage（完整硬件系统）、camera preprocessing latency、power consumption 全部 **PENDING HARDWARE VALIDATION**。

## Next Step

1. 在 ESP32-S3 N16R8 运行 golden fixture，核对输出及张量顺序/指数，测 RAM/PSRAM、latency、Flash 和功耗。
2. 移植实际 Gateway 摄像头 RGB/resize/quality gate，验证 Python ↔ firmware preprocessing parity。
3. 采集现场数据并按 plant/site/session/time 分组，包含背景/无植物/OOD/模糊/曝光失败样本。
4. 在独立现场 validation 上微调与阈值重校准，最后一次性评估独立 test；必要时更窄 MobileNetV2 或 QAT。

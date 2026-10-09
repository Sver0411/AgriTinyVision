<h1 align="center">AgriTinyVision</h1>

<p align="center"><strong>面向边缘研究的轻量作物图像异常筛查</strong></p>

<p align="center">图像筛查 · 不确定性拒绝 · INT8 导出</p>

---

<p align="center">
  <a href="#project-overview">概览</a> ·
  <a href="#方法与推理状态">方法</a> ·
  <a href="#主要结果与当前状态">验证状态</a> ·
  <a href="#relationship-to-smart-agriculture-edge-ai">主项目关联</a> ·
  <a href="README.md">English</a>
</p>

---

<a id="project-overview"></a>

AgriTinyVision 是面向 ESP32-S3 的轻量作物图像异常筛查研究流程。它使用 PyTorch 训练紧凑分类器，导出 ONNX，再将 INT8 候选转换为 ESP-DL。研究问题是图像是否值得进一步检查，不是疾病诊断。

## 方法与推理状态

```text
作物图像 → 质量门控 → 轻量模型 → 置信度 / 决策阈值
              │                     │
              └── 拒绝 → UNKNOWN ←──┘
                    接受 → NORMAL / SUSPECT
```

| 结果 | 含义 |
| --- | --- |
| `NORMAL` | 输入通过门控，被分类为正常 |
| `SUSPECT` | 输入通过门控，提示进一步检查 |
| `UNKNOWN` | 输入被拒绝、不可读、质量不足、输出无效或置信度不足 |

只有 NORMAL 和 SUSPECT 是训练类别。UNKNOWN 是拒绝或不确定结果，**不是第三个训练类别**。质量门控检查清晰度、亮度、对比度、尺寸和解码；置信度使用最大 softmax 概率。这些暂定门控需要田间校准，不是完整的分布外检测器；非植物或陌生图像仍可能得到高置信度预测。

基线为 torchvision MobileNetV2 1.0，使用 ImageNet `IMAGENET1K_V1` 权重并全参数微调。160×160、224×224 共享划分、seed 和训练预算。阈值在验证集选择，候选与阈值在最终测试前冻结。二分类指标与覆盖率、UNKNOWN 比例及通过门控输入的指标分别报告。

## 主要结果与当前状态

首次完整流程使用 RTX 4060 Laptop GPU。v0.1 选择 MobileNetV2 160×160：两种尺寸验证结果相近，而 160×160 的卷积/线性 MACs 约为一半。后续测试分数没有改变选择。

| 模型 | 测试准确率 | SUSPECT 召回率 | FNR | 产物或评估方式 |
| --- | ---: | ---: | ---: | --- |
| MobileNetV2 160 FP32 | 99.76% | 100.00% | 0.00% | 8,878,502 字节 ONNX |
| MobileNetV2 160 INT8 | 99.52% | 99.65% | 0.35% | 2,438,752 字节 ESP-DL 模型；主机仿真 |
| MobileNetV2 224 FP32 | 100.00% | 100.00% | 0.00% | 8,878,502 字节 ONNX |
| MobileNetV2 224 INT8 | 100.00% | 100.00% | 0.00% | 2,511,296 字节 ESP-DL 模型；主机仿真 |

这些分数来自背景受控的整理后 PlantVillage 子集，不是田间画面。INT8 结果是**主机量化图仿真**，不是 MCU 测量。

| 阶段 | 证据 |
| --- | --- |
| PyTorch 训练 / 冻结测试 | 桌面 GPU 已完成 |
| ONNX 导出 | 与 PyTorch 至少四个相同输入的数值核验 |
| ESP-PPQ → ESP-DL INT8 | `esp32s3` w8a8 转换；完整测试集主机图仿真 |
| ESP-IDF 固件示例 | 编译成功，**未运行板端推理** |
| 板端内存、延迟、相机路径、功耗 | 等待硬件验证 |

详细结果见[首次运行报告](reports/FIRST_RUN.md)。目标 ESP32-S3 N16R8 配有 16 MB Flash、8 MB PSRAM；转换和算子检查不证明完整模型能够装入并运行。

## 数据与评估范围

实验使用[官方 PlantVillage 数据集](https://github.com/spMohanty/PlantVillage-Dataset)的 2,792 张 raw/color 图像，覆盖苹果、甜椒、马铃薯、番茄及 12 个源标签，每标签最多 240 张。这不是完整数据集，排除灰度与分割变体。新标签须在 [label_mapping.yaml](configs/label_mapping.yaml)显式映射。

Manifest 记录源/二元标签、叶片 ID、哈希和划分。叶片 ID、完全重复和近重复关联图像先分组，再确定性划分。缺少可靠植株、拍摄会话和场景 ID，意味着不能排除所有泄漏。原始图像不进入 Git。详见[数据指南](data/README.md)和[审计](reports/dataset_audit.md)。

## 快速开始与复现

需要 Python 3.10+，流程支持 Windows/Linux。训练和 GPU 评估默认使用 CUDA，需要 CPU 时显式传入 `--allow-cpu`。先安装匹配平台的 PyTorch wheel，再安装项目依赖。首次运行使用 Windows 11、Python 3.12.15、PyTorch 2.11.0+cu128。

```bash
python -m venv .venv
source .venv/bin/activate
# Windows activation: .venv\Scripts\activate
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[dev]'
python -m pip install esp-ppq
python -m agritinyvision.doctor --output reports/environment.json
```

复现锁定环境时先安装适合的 CUDA wheel，再使用 [requirements-lock.txt](requirements-lock.txt)。其他平台选择兼容 wheel，并重新运行环境检查。

准备、审计数据并训练两种基线尺寸：

```bash
python scripts/download_plantvillage.py
python -m agritinyvision prepare-data
python -m agritinyvision audit
python -m agritinyvision preflight --config configs/train_mobilenetv2_160.yaml
python -m agritinyvision train --config configs/train_mobilenetv2_160.yaml --output runs/baseline160
python -m agritinyvision train --config configs/train_mobilenetv2_224.yaml --output runs/baseline224
```

开展新实验时，在独立实验副本中将[已发布的选择记录](reports/candidate_selection.json)保存在输出路径之外，再于测试前运行 `python scripts/select_candidate.py`。该命令拒绝覆盖现有选择记录，也拒绝在已有测试报告后重新选择。不用测试分数调整训练、阈值或门控。评估、推理、导出并量化所选候选：

```bash
python -m agritinyvision evaluate --checkpoint runs/baseline160/best.pt
python -m agritinyvision infer --checkpoint runs/baseline160/best.pt --image /path/to/leaf.jpg
python -m agritinyvision export --checkpoint runs/baseline160/best.pt --format onnx
python -m agritinyvision quantize --checkpoint runs/baseline160/best.pt --model runs/baseline160/tinyvision.onnx
python -m agritinyvision release \
  --checkpoint runs/baseline160/best.pt \
  --onnx runs/baseline160/tinyvision.onnx \
  --metrics runs/baseline160/test_metrics.json \
  --espdl runs/baseline160/tinyvision.espdl
```

转换路径为 PyTorch → ONNX → ESP-PPQ → ESP-DL。ONNX 使用 batch 1、NCHW FP32、opset 18、两个 logits。INT8 校准使用 256 张分层训练图像，无随机增强；完整测试图仿真保留 FP32 验证阈值。静态算子检查依据 [ESP-DL 支持列表](https://github.com/espressif/esp-dl/blob/master/operator_support_state.md)。

可选 `quantize --backend onnx` 属于桌面 ONNX Runtime 比较，`benchmark` 也仅用于桌面。断点续训使用 `train --resume runs/baseline160/last.pt --config configs/train_mobilenetv2_160.yaml`，要求配置与 manifest 一致。更多流程见[首次运行报告](reports/FIRST_RUN.md)及 CLI `--help`。

发布核验模型哈希、冻结指标、ONNX 数值检查、转换报告与元数据。失败保留原始错误；没有 ESP-DL 的研究包标记导出 PENDING，不伪造模型。Checkpoint/二进制不进入 Git，拟以 Release 资产发布。[集成契约](docs/INTEGRATION_CONTRACT.md)和[固件指南](firmware_example/README.md)定义预处理、张量布局和硬件验收。可选检查：`python -m pytest -q` 和 `python -m ruff check agritinyvision scripts tests`。

## Relationship to Smart-Agriculture-Edge-AI

[Smart-Agriculture-Edge-AI](https://github.com/Sver0411/Smart-Agriculture-Edge-AI) 是关联系统项目和未来视觉集成目标。AgriTinyVision 的相机/模型路径尚未接入主系统实际板端视觉控制闭环；网关控制、LoRa、接管和同步属于独立系统工作。

## 局限与后续工作

板端推理、模型加载时间、内部 RAM/PSRAM、总 Flash、相机预处理延迟及功耗均未验证。PlantVillage 偏差与域迁移意味着高测试分数不能解释为田间疾病识别效果。仍需田间相机数据、独立验证、阈值校准及板端测量；细粒度诊断、分割和严重度估计不在 v0.1 范围。

## 许可证

代码采用 [MIT License](LICENSE)。数据集图像、预训练权重及其他外部材料仍受各自条款约束，仓库代码许可证不会重新授权这些材料。

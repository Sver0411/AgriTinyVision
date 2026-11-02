<h1 align="center">AgriTinyVision v0.3</h1>

---

<p align="center">
  <a href="#project-overview">概览</a> ·
  <a href="#pc-摄像头-web-控制台">Web 控制台</a> ·
  <a href="#方法与推理状态">方法</a> ·
  <a href="#主要结果与当前状态">验证状态</a> ·
  <a href="#relationship-to-smart-agriculture-edge-ai">主项目关联</a> ·
  <a href="README.md">English</a>
</p>

---

<a id="project-overview"></a>

AgriTinyVision 包含轻量作物图像分类器的训练、导出和 ESP32-S3 实机部署。电脑摄像头采集图像，通过 USB 交给 ESP32-S3 N16R8 运行已冻结的 MobileNetV2 160 INT8 模型，由本地 Web 控制台显示结果、设备耗时和内存，并保存实验报告。

## PC 摄像头 Web 控制台

环境：Windows 11、Python 3.10+、电脑摄像头、ESP32-S3 N16R8 和 USB 数据线。固件沿用 ESP-IDF 5.4.4、ESP-DL 3.3.13。

将已有模型包放入 `releases/v0.1.0`，在仓库根目录运行：

```powershell
python -m pip install -e '.[camera,web,dev]'
python scripts/run_pc_camera.py --list-ports
python scripts/prepare_camera_firmware.py --bundle releases/v0.1.0
# ESP-IDF 5.4.4 PowerShell
.\scripts\build_firmware.ps1 -Mode Camera -Transport USBJTAG
# 确认目标端口后手动安装固件
python "$env:IDF_PATH\tools\idf.py" -C firmware_example -B "$PWD\build-camera" -p COM3 flash
# 项目 Python 环境：启动 Web 控制台
python scripts/run_pc_camera.py --port COM3 --camera 0 --fps 1
```

程序会在电脑浏览器打开 `http://127.0.0.1:8765`。依次点击“连接相机 → 连接 ESP32 → 单帧测试 → 开始推理”。页面提供分类概率、质量拒绝原因、各阶段耗时、延迟趋势、内存、实验报告保存和 ZIP 下载。

电脑负责解码、方向处理、RGB 转换及 PIL 双线性抗锯齿缩放。ESP32 负责质量检查、ImageNet 归一化、按实际 exponent 量化、ESP-DL 推理和状态判定。浏览器 JPEG 预览与 RGB888 模型输入分开处理；ONNX 对照由独立离线脚本执行。

详见 [Windows 设置](docs/PC_CAMERA_SETUP.md)、[USB 协议](docs/USB_PROTOCOL.md)、[硬件部署](docs/HARDWARE_DEPLOYMENT.md)及[实机报告](reports/HARDWARE_V02.md)。

## v0.3 可靠性与对照实验

Web 控制台和命令行现会根据真实冻结模型包核对设备的模型 SHA256、版本、Deployment SHA256、张量形状/类型/exponent、质量契约及决策边界。任一不匹配时，发送图像前关闭连接。现有 v0.2 固件已提供这些字段，固件二进制和模型保持一致。`--bundle` 选择本次实验模型包，`--deployment` 可额外校验生成的部署 JSON。

受控图像使用同一 RGB SHA256 对照 ONNX FP32、冻结 ESP-PPQ 仿真和 ESP32，各后端独立记录概率及耗时。运行方式：

```powershell
python scripts/prepare_analysis_scenes.py --image PATH_TO_TRAIN_LEAF --source DATASET_ID --label NORMAL --split train --output runs/v03/my-scenes
python scripts/quantization_analysis.py --manifest runs/v03/my-scenes/manifest.json --bundle releases/v0.1.0 --port COM3 --output runs/v03/my-comparison
# 取得原始可信、哈希绑定的冻结 PPQ 图后添加：
# --host-model PATH_TO_FROZEN.espdl --host-report PATH_TO_FROZEN.espdl.quantization.json
python scripts/analyze_performance.py --run RUN_DIRECTORY
python scripts/export_experiment.py --run RUN_DIRECTORY --public-output NEW_SANITIZED_DIRECTORY
python -m pytest -q --strict-markers
# 显式选择本地模型文件和物理设备检查：
$env:ATV_HARDWARE_PORT='COM3'
python -m pytest -q -m 'hardware or local_artifact'
```

Host 模型需要相邻 `.native` 图和匹配的 `simulation_graph_sha256`；仅加载本地产生的可信图。原图缺失时保留 PENDING，对照工具不重新校准或修改正式模型。本轮十类探针覆盖源叶片的光照、对比度、模糊和颜色变化，以及合成非植物图、棋盘和噪声。源叶片来自训练划分，实验评估部署差异。结果及颜色置换场景的判定分歧见[量化分析](reports/QUANTIZATION_ANALYSIS_V03.md)。

| v0.3 验证项 | 证据 |
| --- | --- |
| 模型身份和错误模型拒绝 | COM3 物理设备检查及软件不匹配测试 |
| FP32 与设备对照 | 十类受控探针；八组概率，十组门控状态中九组一致 |
| 本轮探针的冻结 PPQ 对照 | PENDING：原模型 native 图缺失 |
| Web 安全 | 本地同站会话、Host/Origin 检查、CSRF 令牌和有边界的导出；非回环绑定需要独立口令认证 |
| 可复现记录 | 项目/Git/源码哈希、设备身份、依赖、配置、开始结束时间和分阶段耗时 |
| 软件 / 硬件证据 | 见 [v0.3 验收](docs/V03_VALIDATION.md)；模拟串口和物理设备分开记录 |
| 独立田间准确率 / 功耗实测 | PENDING：独立带标注田间数据 / 外部功耗仪 |

本地访问继续使用 `127.0.0.1`；状态、预览和实验下载均要求控制台同站会话。非回环绑定需设置 `ATV_WEB_PASSWORD`（至少 16 字符），浏览器登录用户名为 `agri`。公开数据准备工具创建单独的脱敏目录，不自动上传。详见 [v0.3 指南](docs/V03_RELIABILITY.md)。

## 方法与推理状态

[采集与标注工作台](http://127.0.0.1:8765/field)增加按需保存的无损图像批次、独立人工标签、CSV/ZIP 导出和冻结三状态评价。首轮外部 PlantDoc test 子集含 60 张来源标注图片：设备二分类准确率 88.33%，接受覆盖率 95.00%，接受准确率 89.47%，UNKNOWN 为 3/60；参考正常图片误报 5/12。指标依据数据集作者标签，描述固定子集。人工质量复核和独立 Field 分组划分为 PENDING。详见[操作指南](docs/FIELD_VALIDATION.md)、[实测结果](docs/FIELD_VALIDATION_RESULTS.md)和[汇总报告](reports/field_validation/README.md)。

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

只有 NORMAL 和 SUSPECT 是训练类别，UNKNOWN 是拒绝状态。沿用 validation 冻结阈值 0.30，当前 threshold-centered 策略导出 float32 的 0.20/0.40 边界：低于下界为 NORMAL，中间区间为 UNKNOWN，达到上界为 SUSPECT。质量失败优先返回 UNKNOWN。置信度表示已分配类别的概率，UNKNOWN 使用 null。规则见[集成契约](docs/INTEGRATION_CONTRACT.md)。

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
| ESP-IDF 固件 | Fixture、USB Serial/JTAG、UART 分支编译通过 |
| ESP32 真实推理 | golden Fixture 及 15 项质量/协议检查通过 |
| PC 摄像头链路 | 实时摄像头 100/100 帧取得有效设备结果 |
| 连续运行 | 固定叶片 500/500 帧，持续 536.46 秒，无重启或首尾空闲内存下降 |
| v0.2 软件回归 | 138 项测试通过（历史基线） |

训练与主机仿真结果保留在[首次运行报告](reports/FIRST_RUN.md)。[v0.2 实机报告](reports/HARDWARE_V02.md)记录真实设备数据：推理 P50/P95 为 527.840/527.850 ms，内部 RAM 空闲 286,447 bytes，PSRAM 空闲 5,046,852 bytes，app 镜像为 3,574,960 bytes。模型文件、app 镜像及分区容量分别统计。

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

独立带标注田间验证和外部仪器功耗实测为 PENDING。PlantVillage 分数对应受控数据集，536.46 秒固定图片实验对应该输入和时段。质量门控检查模糊、亮度和对比度；植物存在性与完整 OOD 检测需要独立方法。量化优化作为单独候选研究，v0.3 保留冻结 exponent 和阈值。

## 许可证

代码采用 [MIT License](LICENSE)。数据集图像、预训练权重及其他外部材料仍受各自条款约束，仓库代码许可证不会重新授权这些材料。

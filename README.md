# AgriTinyVision

`Sver0411/Smart-Agriculture-Edge-AI` 的配套视觉模型项目。唯一长期职责：使用 RTX GPU 训练、评估、压缩和导出可部署到 Gateway A **ESP32-S3 N16R8（16 MB Flash / 8 MB PSRAM）** 的 Tiny Vision 模型。

**Project North Star — Do Not Drift**：本项目不是高精度农业病害诊断平台、YOLO 实时检测项目、Web Dashboard、云端农业管理系统或视频监控系统。第一版只回答：植物在视觉上是否有值得进一步检查的异常？

```text
农业图像 → Tiny Vision → NORMAL / SUSPECT / UNKNOWN
Camera → Image Quality Gate → TinyVision → Gateway Decision Engine
```

模型只有 `NORMAL`、`SUSPECT` 两个训练类别。`UNKNOWN` 是推理状态：质量失败、输入异常或置信度不足时拒绝判断。没有互联网时，为 Gateway A 提供最低限度的视觉异常分流能力。水泵、风扇、LoRa、Gateway failover、云同步属于主项目；高精度病害分类、分割、严重程度和趋势识别属于未来 Server Vision，不在 v0.1 实现。

## 当前状态

工程及真实首轮结果见 [reports/FIRST_RUN.md](reports/FIRST_RUN.md)。任何缺失的实验或硬件测量都明确标记 PENDING；不使用空模型伪装成功。公开数据结果不能证明真实农场病害检测有效。

## 环境与安装

Python 3.10+，Windows/Linux。独立虚拟环境，CUDA 版 PyTorch 与 GPU 驱动必须匹配。首轮使用 Python 3.12、PyTorch 2.11.0+cu128；可复現版本会保存在 `requirements-lock.txt`。

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate ; Linux: source .venv/bin/activate
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[dev]'
python -m pip install esp-ppq
python -m agritinyvision.doctor --output reports/environment.json
```

训练和 GPU 评估默认要求 CUDA，不会静默退回 CPU。确有需要时显式传 `--allow-cpu`。ONNX Runtime benchmark 固定为 **Desktop CPU**，绝不表示 MCU 速度。

## 数据、审计及训练

[PlantVillage 官方源](https://github.com/spMohanty/PlantVillage-Dataset) 的 raw/color 原图。下载脚本固定 commit、使用官方叶片映射，只选映射明确的图片；不使用 grayscale/segmented 等同叶片变体。首轮子集为苹果、甜椒、马铃薯、番茄 12 个源标签，最高每源标签 240 张，总计 2,792 张；并非完整 PlantVillage 数据集。映射在 `configs/label_mapping.yaml`，新标签必须显式映射，不能默认当作病害。

```bash
python scripts/download_plantvillage.py
python -m agritinyvision prepare-data
python -m agritinyvision audit
python -m agritinyvision preflight --config configs/train_mobilenetv2_160.yaml
python -m agritinyvision train --config configs/train_mobilenetv2_160.yaml --output runs/baseline160
python -m agritinyvision train --config configs/train_mobilenetv2_224.yaml --output runs/baseline224
```

脚本只下载原图到 Git 忽略的 `data/raw/`。`data/manifests/` 保留路径、source label、二分类映射、真实 leaf ID、SHA256 和 split。审计检测路径越界、缺失/空/损坏图片、非法标签、SHA256 重复、64-bit DCT pHash 近重复风险（距离 ≤4），把 leaf group 与重复/近重复连通分量合并后做确定性 GroupShuffleSplit。不是简单随机切图片。真实 plant ID、拍摄序列/场景 ID 不可用；leaf ID 和 pHash 不能彻底消除所有泄漏，限制必须保留。

主 baseline 是 torchvision **MobileNetV2 1.0，ImageNet IMAGENET1K_V1 预训练**，微调所有参数；160 与 224 比较使用相同 split/seed/epoch 预算。配置 `pretrained: false` 支持从头训练；较窄的 MobileNetV2 需从头训练，torchvision 没有对应预训练权重。增强只作用 train：适度随机裁剪、翻转、小旋转、亮度/对比度/饱和度变化。val/test/calibration 无随机增强，统一 RGB square bilinear resize → 0..1 → ImageNet mean/std。

每次实验保存 config、environment、history、best/last checkpoint、val metrics、阈值搜索；包括 git commit、数据版本/hash、train split hash、seed、optimizer/scheduler/AMP、GPU、CUDA、训练时长和峰值 VRAM。AdamW / SGD，cosine scheduler，验证集早停。`train --resume runs/baseline160/last.pt --config ...` 恢复 optimizer/scheduler/scaler 和 RNG，要求 config 与 manifest 完全相同。确定性为 best effort；不支持的 deterministic kernel 会警告。只加载可信来源的 PyTorch checkpoint。

## 评估与 UNKNOWN

```bash
python -m agritinyvision evaluate --checkpoint runs/baseline160/best.pt
python -m agritinyvision infer --checkpoint runs/baseline160/best.pt --image /path/to/leaf.jpg
```

在 **validation** 上搜索 0.20..0.80，步长 0.05；优先满足 SUSPECT Recall ≥0.95 且 FPR ≤0.35，再选 F1。达不到约束时明确记录 constraints_met=false，并以召回优先降级。checkpoint 选择基于该规则，不以最高 accuracy 为唯一标准。test 只做冻结模型的最终评估；不得根据 test 反复调整训练、阈值或质量门。evaluate 拒绝覆盖已有最终报告。

报告 Accuracy、Precision、SUSPECT Recall、Specificity、F1、ROC-AUC、PR-AUC（average precision）、Confusion Matrix、FPR、**FNR**。二分类 raw metrics 与经过质量门后的系统覆盖率、UNKNOWN、accepted metrics 分开报告，不能通过拒绝难例夸大整体准确率。

独立质量门 (`configs/quality.yaml`) 在固定 160×160 灰度上计算 Laplacian variance、brightness、contrast；读取失败/小尺寸也返回 UNKNOWN。通过后，softmax 最大概率低于 min_confidence 时返回 UNKNOWN，否则按 SUSPECT 概率和验证阈值映射。这些默认质量/置信度阈值只是暂定启发式，尚未做现场校准；**不是完整 OOD detection，也不能保证画面包含植物**。无植物、未知域可能高置信通过，Gateway 不得把这种结果作为病害确诊。

## 导出、量化、发布

```bash
python -m agritinyvision export --checkpoint runs/baseline160/best.pt --format onnx
python -m agritinyvision quantize --checkpoint runs/baseline160/best.pt --model runs/baseline160/tinyvision.onnx
# 可选 Desktop ORT INT8 比较，不能用于 MCU 部署
python -m agritinyvision quantize --checkpoint runs/baseline160/best.pt --model runs/baseline160/tinyvision.onnx --backend onnx
python -m agritinyvision evaluate --checkpoint runs/baseline160/best.pt --model runs/baseline160/tinyvision.int8.onnx
python -m agritinyvision benchmark --model runs/baseline160/tinyvision.onnx
python -m agritinyvision release --checkpoint runs/baseline160/best.pt --onnx runs/baseline160/tinyvision.onnx --metrics runs/baseline160/test_metrics.json --espdl runs/baseline160/tinyvision.espdl
```

ONNX 固定 batch=1、NCHW、float32、opset 18、输出 two logits；至少四个相同输入与 PyTorch 做数值比对。预检核查 [ESP-DL 官方支持算子及限制](https://github.com/espressif/esp-dl/blob/master/operator_support_state.md)，并不等于已证实整图在 8 MB PSRAM 中可运行。保留 torchvision ReLU6（ONNX Clip），不在导出时静默改为 ReLU。

官方路线 [PyTorch → ONNX → ESP-PPQ → .espdl](https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/how_to_quantize_model.html)，target=`esp32s3`，默认 `w8a8`，只用 train 的分层 round-robin 256 张图校准。量化失败保留真实 traceback；量化成功后在完整 test 上做 ESP-PPQ graph simulation，保持 FP32 validation 选定阈值。任何 host simulation 不等于 ESP32 真机。INT8 损失大时依次考虑更多代表性校准数据、QAT、mixed precision，必须用 validation 诊断，不能用 test 做优化搜索。

release 校验模型 hash、最终 test metrics、ONNX 数值报告、ESP-DL 转换报告、metadata schema。未成功生成 .espdl 时可省略 `--espdl` 制作 **ESP-DL export: PENDING** 的研究包；不会生成假的空文件。模型文件和完整 checkpoint 不进 Git，最终模型只作为 GitHub Release artifact 发布。见 [集成契约](docs/INTEGRATION_CONTRACT.md)。

## 硬件及真实农业边界

`firmware_example/` 准备 ESP-IDF / ESP-DL / ESP32-S3 集成。model load time、inference latency、internal RAM、PSRAM、Flash、camera preprocessing latency、power 全部 **PENDING HARDWARE VALIDATION**，不可引用 PC benchmark 代替。

公开病害数据常有干净背景、受控光照、物种限制、dataset bias 和 domain shift。首轮证明的是训练/量化/导出流水线，不证明现场病害识别。未来必须用 Gateway A 实际摄像头现场数据微调、独立验证及阈值重校准。

## 测试

```bash
python -m pytest -q
python -m ruff check agritinyvision scripts tests
```

CI 使用合成小图测试，不跑完整训练、不下载农业数据。代码 MIT；外部图像及预训练权重适用各自条款，仓库代码许可证不重新许可这些数据。

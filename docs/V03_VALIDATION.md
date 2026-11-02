# v0.3 验收记录

本轮基于 main / `3ae13554d3716269eaf488f1cf31f751817060dc`，2026-11-02 完成本地代码增强和当前资源范围内的验证。
没有提交或推送远端，没有改动分支历史，没有重新训练、量化或刷写设备。

## v0.2 → v0.3

| 阶段 | 改动 | 验收 |
| --- | --- | --- |
| P0 身份 | 正式包真哈希和 FlatBuffer 张量生成预期身份；连接强制比对模型/版本/部署/质量/决策 | 软件 16 项身份测试；物理设备身份通过、错误期望哈希拒绝；另 1 项真实模型元数据篡改检查 |
| P0 量化 | 同一 RGB 的 FP32 / frozen host / ESP32 框架；来源、每通道分布、饱和、概率差、一致率、计时 | 框架 8 项软件测试；10 类 FP32 / 真设备探针；host 原 native 图缺失，PENDING |
| P0 图像 | 共享 ImageNet 常量；显式 RGB/BGR、方向和镜像；质量缩放统一 BILINEAR | 新增 52 项预处理测试；原 11 项主机 C++ 核心测试；实机质量拒绝不执行模型 |
| P0 表述 | 双语 README、来源和口径说明、历史与新实测分开 | 保留历史报告；记录颜色置换判定分歧、田间/功耗状态 |
| P0 CI | 安装 camera,web,dev；Ruff、完整 pytest、JUnit artifact | 本地等价检查通过；远端 Actions 本轮 PENDING，按要求不推送触发 |
| P1 记录 | 项目/Git/source hashes、依赖、身份、UTC 时间、配置、阶段计时、饱和、首尾趋势 | 软件报告生成测试；真实 10 场景实验子目录和 10 帧摄像头记录包含六类必需文件 |
| P1 导出 | 报告再生成、单独脱敏目录、白名单 ZIP、无照片 | 自动测试和真实 10 帧脱敏运行；无绝对路径/MAC，原记录保留 |
| P1 Web | 读取会话、Host/Origin、CSRF、非回环独立密码、路径/软链接边界 | 8 项安全测试 + 原 6 项 Web 测试；真实 HTTP 未认证 401、会话读取 200、操作 202 |
| P2 性能 | 采集/PC 预处理/串口/设备分阶段，按帧残差、内存和趋势分析 | 真实 10 帧计时、性能分析 JSON/MD；保持正式模型和默认频率 |

全量新增、修改、删除文件见 [V03_CHANGESET.md](V03_CHANGESET.md) 和 [JSON 清单](../reports/changeset_v03.json)。

## 测试证据

| 类别 | 结果 | 记录 |
| --- | --- | --- |
| Ruff 全量 | All checks passed | [ruff_v03.txt](../reports/ruff_v03.txt) |
| 软件（含 HTTP 和模拟串口 / C++ 核心） | 222 passed，0 skipped，4 deselected | [文本](../reports/tests_v03_software.txt)、[JUnit](../reports/tests_v03_software.xml) |
| 显式本地资源检查 | 4 passed：2 项物理设备 + 2 项本地模型文件，0 skipped | [文本](../reports/tests_v03_hardware.txt)、[JUnit](../reports/tests_v03_hardware.xml) |
| 真设备对照 | 10/10 场景返回结果；10 组门控状态、8 组模型概率，9/10 状态一致 | [量化报告](../reports/QUANTIZATION_ANALYSIS_V03.md)、[JSON](../reports/quantization_analysis_v03.json) |
| 真摄像头链路 | 10/10 帧返回有效结果，原始画面不保存 | 量化报告的独立摄像头计时部分 |
| 实际 Web HTTP | 401 / 200 / 202 按访问条件返回 | [web_runtime_v03.json](../reports/web_runtime_v03.json) |
| GitHub Actions | PENDING：workflow 已修改，本轮没有远端运行 | `.github/workflows/tests.yml` |

软件测试的两条 warning 来自 PyTorch 原有 ONNX legacy exporter 弃用提示。四项本地资源
测试由 marker 显式排除在普通 CI 外；这四项已另行实际运行，未使用 skip 掩盖失败。
软件序列和 C++ 核心执行只作为软件 / 模拟硬件证据，不作为设备时延实测。
JUnit 中的本机 hostname 和文本报告的仓库绝对路径已脱敏，测试数量与结果保持原值。
既有 compare_device_reference.py 复用统一 FP32 后端，在历史输入的临时副本上验证得到
相同概率 0.0025958786718547344，原历史目录未写入。运行中的页面正常加载，浏览器脚本错误为零。

常规命令：

```powershell
python -m pip install -e '.[camera,web,dev]'
python -m ruff check agritinyvision scripts tests
python -m pytest -q --strict-markers
$env:ATV_HARDWARE_PORT='COM3'
python -m pytest -q -m 'hardware or local_artifact'
```

## 模型和历史资产

- 模型 SHA：`bfda065ad95c3c3458d3c135b4c4b4575bdad6a31d2445b9283065d3851af3fa`。
- ONNX SHA：`350c32d88d19eb16dca7d905036d2c99a716a3938b91f52715349e43ee6ab909`。
- Deployment SHA：`f32296c1196da61aff1430481e66c9dd938db57e8d98565c926f415dc7dfc167`。
- 设备仍运行 firmware 0.2.0，模型 0.1.0；项目源码为 0.3.0。ELF SHA 为 `bab257fda4b57c0f922cc86a8ec76e9037135208c3ba7adda93957eb6711f622`。
- 阈值 0.30、float32 下上界 0.20/0.40、exponent -6/-4 保留。v0.2 历史结果和配置无 diff。

本轮身份通过 13 项字段比对；故意传入错误哈希的真实 HELLO 被拒绝，连接关闭且没有发送 FRAME。
质量失败的黑 / 灰 / 白实机输入均返回 UNKNOWN，归一化、推理计时和概率为 null。

## 复现本轮分析

以下是本地生成的目录标识，不自动公开；原图来源见原数据 manifest 和分析报告。

```powershell
python scripts/prepare_analysis_scenes.py --image "data/raw/plantvillage_color/Apple___healthy/7a13d83d-7511-41c5-9fd5-9fd18d3cf3b1___RS_HL 5685.JPG" --source PlantVillage-7f7ecc7e-Apple-healthy --label NORMAL --split train --output runs/v03/NEW_SCENES
python scripts/quantization_analysis.py --manifest runs/v03/NEW_SCENES/manifest.json --bundle releases/v0.1.0 --port COM3 --output runs/v03/NEW_COMPARISON
```

有原始可信 frozen graph 时，将模型放在与其 `.native` 相邻的路径，并增加
`--host-model ... --host-report ...`。report 同时绑定 ESP-DL、原 ONNX 和 native SHA；
不能拿重新校准的图替代历史模型的图。

本轮最终分析来自 `runs/v03/comparison_release_candidate`；摄像头记录来自
`runs/v03/live_timing/20261102_221655_439261`。输入 manifest、模型和源码 SHA 写入汇总 JSON，
源码哈希与执行时文件核对一致；新实验与 v0.2 目录分开。

```powershell
python scripts/analyze_performance.py --run runs/v03/live_timing/20261102_221655_439261
python scripts/report_v03_analysis.py --comparison runs/v03/comparison_release_candidate --live-run runs/v03/live_timing/20261102_221655_439261 --output reports
python scripts/export_experiment.py --run runs/v03/live_timing/20261102_221655_439261 --public-output runs/v03/NEW_PUBLIC_REVIEW
```

报告模板针对本研究的冻结 160 模型和十场景，模型/场景不满足时拒绝生成概括性结论；
其他实验仍可使用通用 comparison.json / REPORT.md 和设备实验报告。

## 验证边界和后续实验

| 项目 | 当前状态 | 所需资源 |
| --- | --- | --- |
| 本轮逐图 host/device 误差和三方一致率 | PENDING | 正式历史 frozen native 图及摘要报告 |
| 自然异色叶片、独立田间准确率 | PENDING | 具备来源 / 独立划分 / 标注的真实摄像头数据 |
| 外部功耗实测 | PENDING | 电流/功率记录仪与同步工作负载 |
| 量化优化 | 分析已完成；候选优化 PENDING | 代表性 train 校准样本、独立 val，候选单独导出 |
| v0.3 多时段多场景长期可靠性 | PENDING | 独立较长设备实验；v0.2 的固定图 536.46 秒记录保留其原范围 |
| v0.3 远端 CI | PENDING | 用户确认后推送触发 |

下一步首先补齐原 frozen graph，分开测量 FP32→host 和 host→device 的概率误差；
颜色置换的大误差与低输入饱和并存，不能把全部误差归因于截断。随后收集独立带标注
光照/颜色叶片，验证 OOD / 植物存在性的方法。候选校准集或 QAT 使用独立目录与 val，
不反复以最终测试集选择参数。性能先分析传输和质量/归一化残差，再评估复制、缓冲和
采样频率；设备推理耗时与完整响应分别记录。

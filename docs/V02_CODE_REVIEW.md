# v0.2 代码审查

审查日期：2026-11-02。

| 项目 | 审查结果 |
| --- | --- |
| 固件入口 | `firmware_example/main/app_main.cpp` 是原 golden Fixture；`camera_main.cpp` 是 PC RGB 输入模式，通过 Kconfig 切换 |
| 模型 | 本地 `releases/v0.1.0/tinyvision.espdl`，2,438,752 bytes |
| SHA256 | `bfda065ad95c3c3458d3c135b4c4b4575bdad6a31d2445b9283065d3851af3fa`，与 metadata 一致 |
| 实际模型输入 | FlatBuffer 与 tinyvision.info 均为 INT8 NHWC `[1,160,160,3]`，exponent=-6 |
| 实际模型输出 | INT8 `[1,2]`，exponent=-4，NORMAL / SUSPECT 顺序 |
| ESP-DL | 本地组件 3.3.13；沿用真实 `dl::Model` 构造器、`get_inputs()`、`get_outputs()`、`run()`、`dl::quantize` 和 `dl::dequantize` |
| 环境 | ESP-IDF 5.4.4；16 MB Flash、Octal 80 MHz PSRAM，原 6 MB app 分区 |
| USB | 原 sdkconfig：UART0 主控制台，USB Serial/JTAG 辅助控制台；本机 COM3 枚举为 VID:PID 303A:1001，读取得到 ESP32-S3 rev0.2、8 MB PSRAM、USB Serial/JTAG |
| 预处理 | PIL RGB direct square bilinear antialias → 设备 ImageNet mean/std → 实际 exponent 量化 |
| 质量门 | 当前 `configs/quality.yaml`，160×160、灰度系数 0.299/0.587/0.114、四邻域 Laplacian、排除边界 |
| 决策规则 | validation 冻结阈值 0.30；当前契约 margin=0.10，导出 float32 0.20/0.40 边界 |

历史 Release metadata 的 `min_confidence=0.6` 属于原发布策略。摄像头部署使用当前集成契约的 threshold-centered-v1，独立生成 `camera_deployment.json` 与配置头，并记录来源 SHA256。历史 Release、模型、Fixture、数据划分和训练评估流程保留。

PC 实时模块只导入采集、协议和记录代码。参考 ONNX 推理位于独立的离线对照脚本，使用 RGB SHA256 匹配同一输入。

硬件推理验收状态统一记录在 [HARDWARE_VALIDATION.md](HARDWARE_VALIDATION.md)。

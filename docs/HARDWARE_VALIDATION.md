# 硬件验收记录

日期：2026-11-02。目标：ESP32-S3 N16R8，MobileNetV2 160 INT8。

| 层次 | 状态 | 证据 |
| --- | --- | --- |
| 软件回归 | PASS | `reports/tests_v02.txt`，含 Python、Web 与共享 C++ 核心测试 |
| 摄像头固件编译 | PASS | `reports/firmware_camera_build.json` |
| Fixture 固件编译 | PASS | `reports/firmware_fixture_v02_build.json` |
| USB 设备识别 | PASS | COM3、VID:PID 303A:1001、ESP32-S3 rev0.2、USB Serial/JTAG、8 MB PSRAM |
| 设备端模型与数值验收 | PASS | `reports/hardware_fixture_v02.json`、`reports/hardware_protocol_v02.json` |
| 摄像头连续运行验收 | PASS | 实时摄像头 100/100 帧成功 |
| 长时间运行验收 | PASS | 固定叶片 500/500 帧成功，持续 536.46 秒 |

开发板已按用户授权直接安装 AgriTinyVision 固件，依次运行原 Fixture 和摄像头模式。当前固件运行 USB Serial/JTAG 图像推理服务。

完整验收结果：[HARDWARE_V02.md](../reports/HARDWARE_V02.md)；机器可读记录：`reports/hardware_validation_v02.json`。

## 验收顺序

1. 使用原 Fixture 记录模型加载、同一 golden 输入的概率、Python 参考差异及内存。
2. 刷写 Camera 模式，HELLO_ACK 核对 boot_id、model_loaded、模型 SHA256、版本、张量属性与 Flash 信息。
3. 对固定图像运行 `--headless --image IMAGE --frames 1`；再执行独立 FP32 对照脚本，按 RGB SHA256 核对输入一致。
4. 分别运行 100 帧、500 帧及 `--frames 0 --duration 300`。
5. 以低照度、过曝、遮挡和模糊画面检查质量门及 UNKNOWN；检查拒绝帧 inference_us=null。
6. 检查错误日志、boot_id、延迟 P50/P95、最低空闲量与首末空闲量变化。拔插串口后重新连接并测试单帧。

## 硬件测量表

| 指标 | 当前记录 |
| --- | --- |
| Fixture 模型加载 / 首次推理 | 463.037 / 537.003 ms |
| 长测质量门 / 预处理 / 推理 P50 | 见报告 latency；推理 527.840 ms |
| 长测推理 P95 | 527.850 ms |
| 长测传输至 ACK P50 / P95 | 353.899 / 357.332 ms |
| 长测端到端 P50 / P95 | 1107.175 / 1114.935 ms |
| Internal RAM 当前空闲首 / 末 | 286,447 / 286,447 bytes |
| PSRAM 当前空闲首 / 末 | 5,046,852 / 5,046,852 bytes |
| Flash 芯片 / app 镜像 | 16,777,216 / 3,574,960 bytes |
| 长测实际 FPS | 0.932；500 帧，536.46 秒，单一 boot_id，无错误 |
| 同图 FP32 / ESP32 INT8 p_suspect | 0.002595879 / 0.000552779，均为 NORMAL |

每次实验自动生成 CSV、JSON、四张 PNG 和 Markdown 报告。构建产物大小保存在构建记录中；设备性能表由对应硬件实验填入。

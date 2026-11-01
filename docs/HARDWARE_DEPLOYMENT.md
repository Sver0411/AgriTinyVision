# ESP32-S3 部署

目标为 ESP32-S3 N16R8（16 MB Flash、8 MB Octal PSRAM），ESP-IDF 5.4.4 与 ESP-DL 3.3.13。使用 PC 摄像头，通过 USB 将 RGB888 图像发送给开发板。

## 模式与构建

`CONFIG_ATV_CAMERA_MODE=y` 编译 `camera_main.cpp`；关闭时编译原 `app_main.cpp` golden Fixture。两种入口均使用同一个真实 `tinyvision.espdl`，Fixture 的输入和参考值继续由原脚本生成。

Camera 默认 USB Serial/JTAG，支持在 `menuconfig → AgriTinyVision` 选择 UART0 Bridge、调整 UART 波特率。现有主/辅助控制台配置保留；Camera 应用通过对应驱动收发协议包，并关闭运行时文字日志，启动文字由 PC 的 magic 同步过滤。USB CDC 占用原生 USB PHY 的配置会在选用 Serial/JTAG 时给出构建提示。

Windows 操作及手动刷写命令见 [PC_CAMERA_SETUP.md](PC_CAMERA_SETUP.md)。

## 模型与部署配置

准备脚本首先验证 Release metadata 中的模型 SHA256，然后读取真实 EDL2 FlatBuffer 的输入输出，核对 `.info`。当前模型：

| 项目 | 值 |
| --- | --- |
| 文件大小 | 2,438,752 bytes |
| SHA256 | bfda065ad95c3c3458d3c135b4c4b4575bdad6a31d2445b9283065d3851af3fa |
| 输入 | INT8 NHWC `[1,160,160,3]`，exponent=-6 |
| 输出 | INT8 `[1,2]`，exponent=-4 |
| 类别 | NORMAL、SUSPECT |
| 阈值 | validation 选择的 0.30 |
| 拒绝规则 | 当前契约 threshold-centered-v1；margin=0.10，使用导出的 float32 边界 |

`camera_deployment.json` 记录真实模型、metadata、`.info` 和质量配置的 SHA256。配置头由该过程生成；设备启动重新计算嵌入模型 SHA256，再通过实际 ESP-DL 张量验证形状、类型和 exponent。

## 内存与预处理

原生 USB 驱动使用有界 96 KiB 接收环形缓冲，能够容纳单次 76,832-byte 完整包；大块 malloc 依照现有配置使用 PSRAM。UART0 接收环形缓冲为 8 KiB。设备另外从 PSRAM 分配固定 76,800-byte RGB 缓冲，再加载 ESP-DL 模型。收包临时块 4 KiB，控制结果上限 4 KiB。质量门按像素计算灰度与统计量，不额外分配整张 float 灰度图。

PSRAM 分配失败时，设备保留固定控制包缓冲用于握手和 model_error 诊断。只有张量数据有效且符合部署配置时才接受推理。模型参数和激活的实际分配由 ESP-DL 管理；HELLO 和逐帧结果分别记录 Internal RAM / PSRAM 当前与最低空闲量，实机长测用于确认整体内存余量。

PC：摄像头解码 → 方向和可选镜像 → BGR 转 RGB → PIL direct square bilinear antialias 160×160。ESP32：质量检查 → 像素除以 255 → ImageNet Mean/Std → NHWC INT8。

量化使用真实输入 exponent 和 `dl::quantize<int8_t>`，S3 使用 half-up 舍入并截断到 [-128,127]。设备返回 saturated_values；亮度极端值经过 ImageNet 归一化后可能超出 exponent=-6 的 INT8 范围，保留原模型和量化规则并记录数量。输出通过真实 exponent 反量化，减去最大 logit 后执行 Softmax。

Confidence 表示已分配类别的概率，UNKNOWN 使用 null。最终状态由既有阈值拒绝区间决定，不以 max(p,1-p) 替换决策规则。

## Flash 与构建数据

分区表保留 factory app `0x10000` 起、容量 `0x600000`。构建记录分别保存模型 bytes、app bin bytes、app 分区容量、bin SHA256、ELF SHA256 和 `idf.py size` 明细。设备握手另报实际 Flash 芯片容量与运行分区信息。

`.espdl`、`.onnx`、二进制 Fixture、build 目录与实验数据均沿用 Git 忽略规则；源码、测试和小型审查记录进入版本管理。

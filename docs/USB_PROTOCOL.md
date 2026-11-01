# ATV2 USB 串口协议 v1

所有整数使用 Little Endian。CRC 为 CRC-32/ISO-HDLC（IEEE，多项式反射值 0xEDB88320，init/xorout 0xFFFFFFFF），与 `zlib.crc32` 一致。`123456789` 的校验值为 0xCBF43926。

## 包头

| Offset | bytes | 内容 |
| --- | --- | --- |
| 0 | 4 | ASCII `ATV2` |
| 4 | 1 | 协议版本 1 |
| 5 | 1 | 消息类型 |
| 6 | 2 | Flags=0 |
| 8 | 4 | Sequence ID，1..0xFFFFFFFF |
| 12 | 2 | Width，FRAME=160，其他=0 |
| 14 | 2 | Height，FRAME=160，其他=0 |
| 16 | 1 | Pixel Format，FRAME=1（RGB888），其他=0 |
| 17 | 3 | Reserved=0 |
| 20 | 4 | Payload Length |
| 24 | 4 | Payload CRC32 |
| 28 | 4 | Header CRC32，计算 bytes 0..27 |
| 32 | N | Payload |

FRAME 固定 76,800 bytes RGB888，逐行、行内逐像素 RGB；完整包 76,832 bytes。不压缩、不分块。控制包最多 4,096 bytes；JSON 仅用于小型设备信息、结果和错误对象。

## 消息和会话

| 值 | 类型 | Payload |
| --- | --- | --- |
| 1 | HELLO | 空；开始新会话 |
| 2 | HELLO_ACK | UTF-8 JSON 设备信息 |
| 3 | FRAME | RGB888 |
| 4 | FRAME_ACK | 空；MCU 已接收并通过 CRC |
| 5 | INFER_RESULT | UTF-8 JSON 设备推理结果 |
| 6 | ERROR | UTF-8 JSON：state=UNKNOWN、code、message、boot_id |
| 7 | PING | 空，请求与回包使用同一序号 |

```text
PC                        ESP32
HELLO(seq=1) ------------>
             <------------ HELLO_ACK(seq=1, boot_id, model_loaded)
FRAME(seq=2) ------------>
             <------------ FRAME_ACK(seq=2)
                           Quality → Normalize → INT8 → model->run()
             <------------ INFER_RESULT(seq=2)
```

一帧完成后再发送下一帧。序号严格递增；设备拒绝已消费或旧的 FRAME，不重复推理。HELLO 重建会话及序号窗口；设备启动生成新的 boot_id。接收端同时核对包头序号、结果序号和 boot_id。

PC 默认请求超时 30 秒；设备包超时 15 秒、字节间空闲超时 2 秒。包头 CRC 通过后才信任长度，随后验证格式、尺寸和 payload CRC。错误包通过 magic 搜索恢复同步；设备接收缓冲固定容量，超限包不会分配内存。USB Serial/JTAG 使用 96 KiB 环形缓冲承接完整单帧突发，避免驱动环形缓冲满后丢字节。PC 在完整性错误、序号异常、重启或超时后关闭当前连接，重新握手。设备在半包超时后要求 HELLO。

## HELLO_ACK

含 chip、board（目标配置）、transport、uart_baudrate、firmware_version、firmware_sha256、firmware_hash_kind、model_version、model_sha256、model_bytes、model_loaded、model_error、model_load_us、input_shape、output_shape、dtype、input_exponent、output_exponent、decision_threshold、normal_upper_bound、suspect_lower_bound、abstention_policy、quality_gate_version、confidence_definition、idf_version、esp_dl_version、boot_id、reset_reason，以及 Flash 和内存信息。

firmware_sha256 是 ESP-IDF 内嵌的 app ELF SHA256；构建记录另外保存 `.bin` SHA256。deployment_sha256 绑定模型、metadata 和质量/预处理策略；PC 运行记录同时保存依赖版本及模块源码 SHA256。board 字段表示本固件的 N16R8 目标配置。Flash 芯片容量由驱动读取，app_image_bytes 来自镜像元数据，app_partition_bytes 来自分区表。

## INFER_RESULT

| 字段 | 类型与含义 |
| --- | --- |
| sequence、boot_id | uint32 / string；请求和设备会话身份 |
| state、reason | NORMAL / SUSPECT / UNKNOWN；状态原因 |
| normal_probability、suspect_probability | number 或 null；反量化 logits 后稳定 Softmax |
| confidence | 被分配类别的概率；UNKNOWN=null |
| decision_threshold | 已冻结阈值 |
| quality_passed | bool |
| brightness、contrast、laplacian_variance | 设备图像质量指标 |
| quality_us、preprocess_us、inference_us、total_us | 微秒；未执行步骤=null |
| saturated_values | 量化前舍入值超出 [-128,127] 的元素数；未执行量化=null |
| free_internal、free_psram | 当前可字节寻址空闲量，bytes |
| min_free_internal、min_free_psram | 启动以来最低空闲量，bytes |
| model_version、firmware_version | string |

质量失败时直接 UNKNOWN，概率、Confidence 和推理耗时为空。有效概率沿用 threshold-centered-v1；UNKNOWN 为拒绝状态，不是训练类别。

## 时间口径

transfer_ms：PC 开始发送到收到 CRC 验收 ACK，包含链路及 ACK 往返；不是单向物理 USB 时间。

inference_us：设备 `model->run()` 前后计时。total_us：ACK 后开始处理到生成结果字段，含质量、预处理、模型和结果组装，不含串口上传及结果发送。end_to_end_ms：PC 开始发送到收到并校验结果。FPS 按连续完成结果的间隔计算，保留真实序号，不通过丢弃结果提高显示值。

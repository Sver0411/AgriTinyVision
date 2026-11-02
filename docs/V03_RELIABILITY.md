# v0.3 实验与访问约定

## 模型身份

`device_identity.load_expected_identity` 只读校验正式包：读取 metadata、实际 ESP-DL
FlatBuffer 的输入输出张量及 exponent、info 与质量配置，按现有 camera-deployment-v1
规则计算 Deployment SHA256。指定 `--deployment` 时还要求部署 JSON 的全部内容匹配。
连接后的 HELLO_ACK 逐字段比对模型哈希、版本、部署哈希、形状、dtype、exponent、
冻结决策和质量版本，结果写入 run_metadata.json 的 device.identity_validation。

本轮保持 ATV2 包格式 / wire version 1。v0.2 camera 固件已返回全部字段，无需刷写。
缺字段固件会报出字段名与预期/实际值并关闭串口；用该包对应的 camera 固件重新连接。
模型版本不匹配也拒绝正式推理。哈希检查使用实际文件摘要与设备报告值比对。

## 图像处理

| 步骤 | 接口约定 |
| --- | --- |
| 文件解码 | Pillow EXIF transpose → RGB，load_image_bgr 显式返回 BGR |
| 摄像头 | OpenCV uint8 H×W×3 BGR，宽高均至少 32 |
| 转色 | prepare_bgr / prepare_rgb(color_order=...) 显式声明，不猜通道 |
| 方向 | 顺时针 0/90/180/270°；随后水平镜像；均在缩放前完成 |
| 缩放 | uint8 PIL BILINEAR，直接方形 160×160，与 torchvision PIL Resize(antialias=True) 一致 |
| 模型 / 质量输入 | 相同的 76,800 个 RGB888 bytes；设备不再缩放 |
| 质量计算 | 160×160；灰度 0.299R+0.587G+0.114B，亮度均值、总体标准差；内圈四邻域 Laplacian 总体方差 |
| 归一化 | float32(pixel/255−ImageNet mean)/std，mean=(.485,.456,.406)，std=(.229,.224,.225) |
| 量化 | 读取真实输入 exponent -6；S3 floor(x×2^-exponent+.5) 后截断到 [-128,127]；按通道统计截断 |
| 张量 | ESP-DL INT8 NHWC [1,160,160,3] → [1,2]；输出 exponent -4；稳定 softmax |
| 状态 | p<下界 NORMAL，下界≤p<上界 UNKNOWN，其余 SUSPECT；质量失败优先 UNKNOWN，跳过归一化和模型 |

Python 质量门控原先使用 Pillow 默认缩放插值；v0.3 显式使用 BILINEAR，解决非 160
输入和摄像头路径的不一致。v0.2 设备已接收 160 RGB，因此原设备报告保持其原值。
质量门控衡量清晰度、亮度和对比度；棋盘或噪声可能通过，不代表植物存在性或完整 OOD 判断。

## 三方对照

运行命令见中英文 README。manifest 每项包含 id、场景、源图 SHA、来源/标签/划分、
变换说明和标签有效性。源图摘要在初始化后端前核对；所有后端接收同一预处理结果。
设备端绑定依赖该帧 CRC 校验 ACK、序列号及 boot_id，RGB SHA 由 PC 对发送字节计算，
当前协议没有设备重算 RGB SHA 的字段。

FP32 / host 的原始概率保留用于误差分析；三方显示状态应用相同质量规则和冻结边界。
质量拒绝的设备帧没有概率；它们参与门控状态一致率，不参与概率误差分母。JSON 保留
每个后端错误及两类概率差；CSV、摘要、Markdown、PNG 与独立 device_runs 实验目录一起保存。
host graph 的模型、源 ONNX 和 native 摘要都必须匹配，缺失时以 PENDING 单独显示。
native 是 pickle，仅接受可信本地文件。本轮没有重新校准、训练或选择任何候选参数。

## 实验记录和导出

RunRecorder 记录项目版本、Git SHA/dirty、模块源码 SHA、依赖版本、Python、设备固件/
模型/部署身份、ESP-IDF/ESP-DL、配置变更、UTC 开始结束时间。项目源码有本地修改时
dirty=true；Git SHA 指向基线而源码 SHA 标识实际代码。运行目录自动生成，避免覆盖历史。

每次设备实验输出 run_metadata.json、frame_results.csv、device_metrics.csv、
latency_report.json、summary.json 和 errors.log，并可生成 Markdown、PNG。
保存中的实验有开始时间，正常关闭补充结束时间；异常中断保留已刷新到磁盘的逐帧数据。

`export_experiment.py --run ... --public-output ...` 创建新脱敏目录：不复制照片；去除设备
MAC/boot ID、用户名、原路径、私有配置、逐帧原错误字符串，保留模型哈希、数值参数、
版本及统计。脚本不公开文件或推送 GitHub。脱敏后的错误按 recorded_error 计数，失去原始
诊断细节是有意的。控制台 ZIP 是本地完整实验导出，公开前使用脱敏工具。

## Web

默认绑定 127.0.0.1。首次打开首页建立 HttpOnly、SameSite=Strict 的签名会话；读取状态、
串口、预览、ZIP 也需要此会话。操作另需页面令牌。Host / 连接对端限定回环地址，Origin
必须同源。静态资源和操作保留现有浏览器流程。

非回环绑定时独立口令认证覆盖所有路由，首页本身也需认证。环境变量 ATV_WEB_PASSWORD
至少 16 字符，用户名 agri；口令不进入 CLI 参数、首页或实验配置。在远程网络中使用
HTTPS 反向代理或受保护隧道传输认证信息。ZIP 仅允许配置输出根中的当前实验，文件和
图表使用白名单，禁止链接逃逸；请求参数不接受文件路径。

## 测试和性能

CI 安装 `.[camera,web,dev]` 并运行 Ruff 和完整软件测试。普通运行显式 deselect
hardware/local_artifact；`-m 'hardware or local_artifact'` 选择另行准备的真实设备和本地模型。
hardware 测试必须通过 ATV_HARDWARE_PORT 明确指定端口；选中但缺资源时失败。
模拟串口与主机 C++ 核心测试不产生实机性能结论。

pc_capture_ms / pc_preprocess_ms 是 PC 时钟；transfer_ms 是开始发送至 CRC ACK；
serial_roundtrip_ms 是开始发送至结果解析；quality_us / preprocess_us / inference_us /
total_us 为设备时钟。v0.3 end_to_end_ms 从本帧采集开始（固定图片则从预处理开始）至结果
解析，包括图像等待和网页服务中的调度时间。v0.2 end_to_end_ms 是发送至结果，保留历史定义。
固定图片的采集耗时为空。不要把 model->run 时间或几个独立 P50 之和当总响应。

analyze_performance.py 按帧计算串口往返减设备总耗时的残差，包含传输、序列化与 PC
接收，报告首尾延迟窗口及内存。优化顺序依据测量分析 USB 缓冲/发送、RGB 数据复制、
设备质量/归一化、内存和采样频率；单帧队列维持有界。本阶段不改变模型或正式协议性能参数。

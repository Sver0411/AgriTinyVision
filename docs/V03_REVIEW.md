# v0.3 实施审查

基线：`3ae13554d3716269eaf488f1cf31f751817060dc`，分支 main，开始时工作区干净。
本轮在本地实施；提交、远端更新和公开实验数据另行确认。

## 架构和研究资产

训练 / 数据划分 / ONNX / 量化 / release 位于 agritinyvision 对应模块；PC 链路由
camera → image_preprocess → SerialClient → ESP-DL 构成，InferenceController、RunRecorder
负责异步执行和记录，CameraService / Flask 提供桌面 Web。固件 camera_core.hpp 为可在主机
编译的纯 C++ 协议、质量和归一化核心，camera_main.cpp 为 ESP-IDF 设备实现。

v0.2 有 138 项软件测试、15 项实机协议检查、100 帧摄像头记录、500 帧固定叶片记录。
历史文件和 runs、正式模型二进制不覆盖。模型 0.1.0 的 SHA 为
`bfda065ad95c3c3458d3c135b4c4b4575bdad6a31d2445b9283065d3851af3fa`；
阈值 0.30，float32 UNKNOWN 边界 0.20/0.40，输入/输出 exponent -6/-4。

## 修改和验收矩阵

| 优先级 / 问题 | 文件和方案 | 风险 | 验收 |
| --- | --- | --- | --- |
| P0 模型身份未绑定实验 | 新增 pc_camera/device_identity.py；deployment.py 提供只读构造；serial_client.py、cli.py、web_service.py 默认绑定正式包与部署 JSON | 旧固件字段不全需明确拒绝；v0.2 已提供字段，无需改协议 | 真哈希篡改、版本、部署哈希、shape/dtype/exponent、决策和质量契约不匹配拒绝；拒绝后不发 FRAME；验证结果入元数据 |
| P0 三方对照 | 新增 quantization_analysis.py 和 scripts/quantization_analysis.py、场景准备脚本；复用 prepare_bgr / 现有 PPQ frozen graph 加载逻辑 | 正式 native 图可能缺失，禁止重新校准冒充原图；质量拒绝帧无设备概率 | 相同 RGB SHA、ONNX/ESP-DL/native 图哈希绑定；场景、通道饱和、差异、一致率、耗时和失败分别记录 |
| P0 预处理 | image_preprocess.py 显式 RGB/BGR 接口；quality.py 显式双线性；新增测试与契约 | 质量门控缩放会影响非 160 输入；已部署 160 输入不变 | 尺寸、旋转、镜像、灰度极值、非法输入；torchvision 一致、原 C++ 测试通过；实机质量失败不运行模型 |
| P0 科研表述 | README.md、README.zh-CN.md、docs/V03_VALIDATION.md、reports/QUANTIZATION_ANALYSIS_V03.md | 历史 / 新实验混淆 | 双语一致，准确区分 FP32、host、设备、受控图、田间数据和功耗 |
| P0 CI | tests.yml、pyproject.toml、测试标记 | Flask 缺失跳过；真实模型二进制不在 Git | 安装 camera,web,dev；Ruff；含 HTTP 的完整 pytest；artifact / hardware 显式选择，不把本地执行称为 Actions 成功 |
| P1 可复现性 | metrics.py、controller、camera、CLI；新增 experiment_export.py、导出脚本 | 端到端耗时与设备耗时混淆；泄露用户数据 | 版本、Git dirty/source hashes、身份、环境、配置、开始结束时间；通道饱和和性能统计；公开包仅白名单字段，去设备身份和绝对路径，拒绝软链接 |
| P1 Web 安全 | web.py、app.js、tests/test_web_security.py | DNS rebinding、CSRF、预览泄漏、ZIP 软链接穿越 | loopback Host / Origin 限制、同源会话保护读接口；非本地绑定强口令认证；路径边界和软链接测试；原本地流程保留 |
| P2 性能 | PC 采集、预处理和总耗时记录，性能分析脚本/报告 | 将推理时间当总响应；无依据优化模型 | 各阶段单位和时钟独立；首尾内存、延迟趋势；仅分析，不改正式模型 |

## 执行顺序

1. 身份校验 → 定向测试。
2. 三方分析及预处理 → 合成图单元测试、原 C++ 核心、实际可用后端实验。
3. CI 与双语研究说明 → 完整软件回归。
4. 元数据、导出、安全 → HTTP / 脱敏 / 边界测试。
5. 实际设备检查、性能统计和最终验收记录。

环境初查：Python 环境存在 torch、esp_ppq、Flask、flatbuffers；系统枚举到 COM3。
正式 .espdl / ONNX 存在；尚未定位到正式模型对应的 .native，仅发现合成烟测图。
后续先查 frozen graph 来源；若无法取得绑定原 ESP-DL 的图，host 栏保留 PENDING，
不重新量化正式模型。田间带标注图和外部功耗仪属于独立验收输入。

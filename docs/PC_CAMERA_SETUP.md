# Windows 11 摄像头与 ESP32 推理

使用 ESP32-S3 N16R8、USB 数据线和 PC 摄像头。电脑进行采集、方向处理、BGR→RGB 和 PIL 双线性抗锯齿缩放，ESP32 执行质量检查、归一化、INT8 量化及 ESP-DL 推理。

## 1. Python 依赖

在仓库根目录打开 PowerShell，使用已有虚拟环境：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e '.[camera,web,dev]'
python scripts/run_pc_camera.py --check-camera --camera 0
python scripts/run_pc_camera.py --list-ports
```

摄像头被其他程序占用时先关闭对应程序。在 Windows“隐私和安全性 → 相机”中允许桌面应用访问摄像头。编号可通过 `--camera 1` 切换。

## 2. 准备真实模型

模型包目录应包含 `tinyvision.espdl`、`tinyvision.info`、`metadata.json`。对照实验另需同包 `tinyvision.onnx`。

```powershell
python scripts/prepare_camera_firmware.py --bundle releases/v0.1.0
```

脚本核对模型 SHA256，直接读取二进制张量形状、INT8 类型和 exponent，并与 `.info` 交叉核对。输出配置头、`camera_deployment.json` 和固件嵌入的真实模型。

## 3. 编译固件

打开 ESP-IDF 5.4.4 PowerShell，或先运行安装目录的 `export.ps1`。本机安装示例：

```powershell
. D:\DevTools\ESP-IDF\v5.4.4\export.ps1
Set-Location D:\File\AgriTinyVision
.\scripts\build_firmware.ps1 -Mode Camera -Transport USBJTAG
```

构建产物在 `build-camera/`。使用 USB-UART Bridge 时改为 `-Transport UART`；该模式 UART0 默认 921600，PC 的 `--baudrate` 与固件保持一致。原生 USB Serial/JTAG 由 USB 驱动调度，波特率参数不会改变物理 USB 速率。

原 Fixture 可独立构建：

```powershell
.\scripts\build_firmware.ps1 -Mode Fixture
```

该模式使用现有 `fixture_input.bin`、`fixture_config.h`。需要重新准备 train-only golden Fixture 时执行原有 `python scripts/prepare_firmware.py`。

## 4. 刷写

先根据 `--list-ports` 确认连接的目标板。关闭串口监视器，再手动执行（以下以 COM3 为例）：

```powershell
python "$env:IDF_PATH\tools\idf.py" -C firmware_example -B "$PWD\build-camera" -p COM3 flash
```

刷写会安装新固件；代码构建脚本只执行 build 和 size。若下载模式无法进入，按住 BOOT、短按 RESET，再松开 BOOT 后重试。对带两个 USB 插座的开发板，将数据线插到所选接口对应的插座。

## 5. Web 控制台

回到项目 Python 环境：

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/run_pc_camera.py --port COM3 --camera 0 --baudrate 921600 --fps 1
```

程序自动打开电脑浏览器，地址为 `http://127.0.0.1:8765`。可直接在页面选择 COM 端口、波特率、摄像头编号、处理频率和图像方向。

依次点击“连接相机 → 连接 ESP32 → 单帧测试”。确认设备信息、概率和 MCU 推理时间后点击“开始推理”。“暂停推理”停止后续请求，当前帧正常完成并记录。“保存报告”生成图表和 Markdown，“下载实验 ZIP”导出当前实验；“清空显示”清理页面日志。

Web 页面复用本机 Python 服务访问摄像头和串口，预览 JPEG 仅用于浏览器显示；发给 ESP32 的图像继续使用原始采集帧生成 RGB888。多个页面共享一个设备会话，单帧请求按序完成。页面针对电脑浏览器设计。

使用其他 HTTP 端口或手动打开浏览器：

```powershell
python scripts/run_pc_camera.py --port COM3 --http-port 8766 --no-browser
```

页面显示实际采集尺寸和帧号、串口及模型信息、NORMAL/SUSPECT/UNKNOWN、分配类别的 Confidence、质量拒绝原因、各阶段时间及延迟趋势。UNKNOWN 的 Confidence 使用空值；状态由已有阈值拒绝策略决定。运行中修改方向或频率会写入 metadata 的配置变更记录。

可用 `--rotation 90` 调整顺时针方向，或显式使用 `--mirror`；配置会进入实验记录。默认方向 0、不镜像、direct square resize。

## 6. 自动实验

```powershell
python scripts/run_pc_camera.py --port COM3 --headless --frames 1
python scripts/run_pc_camera.py --port COM3 --headless --frames 100 --fps 1
python scripts/run_pc_camera.py --port COM3 --headless --frames 500 --fps 1
python scripts/run_pc_camera.py --port COM3 --headless --frames 0 --duration 300 --fps 1
```

每次生成独立的 `runs/hardware/YYYYMMDD_HHMMSS_ffffff/`，包含 metadata、逐帧 CSV、设备指标 CSV、summary、latency_report、errors.log、四张图和 HARDWARE_REPORT.md。默认保存指标和 RGB SHA256，不保存摄像头原图。

固定图像用于重复设备测试和独立 FP32 对照：

```powershell
python scripts/run_pc_camera.py --port COM3 --headless --image path/to/leaf.jpg --frames 1
python scripts/compare_device_reference.py --run runs/hardware/RUN_DIRECTORY --image path/to/leaf.jpg --bundle releases/v0.1.0
python scripts/run_pc_camera.py --report runs/hardware/RUN_DIRECTORY
```

对照脚本按 RGB SHA256 匹配相同输入，单独运行 ONNX FP32 并写入 `reference_comparison.json`。

## 7. 错误处理

| 提示 | 操作 |
| --- | --- |
| Cannot open camera / capture failed | 检查相机权限、编号及占用，重新 Connect Camera |
| Serial access denied | 关闭 IDF monitor 或其他串口程序，再连接 |
| timeout / sequence mismatch / device restart | 检查 USB，重新 Connect ESP32 建立新会话 |
| payload_crc / frame_format | 检查线缆和接口配置，重新连接后测试单帧 |
| Model not loaded | 根据设备 model_error 检查模型配置、SHA256、张量和 PSRAM |

硬件测量与验收记录见 [HARDWARE_VALIDATION.md](HARDWARE_VALIDATION.md)。

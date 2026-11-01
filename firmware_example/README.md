# ESP32-S3 N16R8 firmware

ESP-IDF 5.4.4 / ESP-DL 3.3.13. Two Kconfig entries share the frozen MobileNetV2 160 INT8 model:

- Camera: `CONFIG_ATV_CAMERA_MODE=y`, native USB Serial/JTAG or UART0 RGB888 input, device quality gate, normalization, quantization and ESP-DL inference.
- Fixture: default mode, the retained train-image golden input and FP32 reference comparison.

Windows build and usage: [PC_CAMERA_SETUP.md](../docs/PC_CAMERA_SETUP.md). Protocol: [USB_PROTOCOL.md](../docs/USB_PROTOCOL.md). Build and hardware acceptance evidence: [HARDWARE_VALIDATION.md](../docs/HARDWARE_VALIDATION.md).

```bash
python scripts/prepare_firmware.py
cd firmware_example
idf.py set-target esp32s3
idf.py build
# When the user is ready to test on their actual S3 N16R8:
idf.py -p PORT flash monitor
```

Model and golden fixture are generated from a real successful ESP32-S3 conversion and a train calibration image, never placeholder binaries. `fixture_input.bin` is normalized NHWC float32; the C++ example quantizes it using the actual model input exponent and dequantizes output logits before softmax/threshold logic. Check .info against expected INT8 [1,160,160,3] input and two output logits. Generated artifacts are ignored by Git; reproduce them with the script after obtaining the actual release model/training run.

The correctness patch preserves this model and input. Python exports float32 state boundaries into fixture_config.h: p<0.20 NORMAL, 0.20≤p<0.40 UNKNOWN, p≥0.40 SUSPECT at threshold=0.30 and margin=0.10. Quality failure or an invalid probability always yields UNKNOWN. The firmware compares the exported constants directly, without max(p,1-p) confidence or recomputing threshold +/- margin. Margin is provisional and needs field validation recalibration. New fixture/build evidence is recorded separately from v0.1.0 history.

Camera mode uses PC decode/orientation/bilinear-antialias resize and receives 160×160 RGB888. Quality checks, ImageNet normalization and quantization execute on the MCU. Its generated configuration binds the real model SHA256, metadata and quality policy. Fixture mode keeps its original preprocessed input and Python-generated quality/state constants.

Actual ESP32-S3 measurements, artifact identities and run directories are recorded in [HARDWARE_V02.md](../reports/HARDWARE_V02.md). Camera 100-frame and fixed-leaf 500-frame runs both returned all requested results; the 500-frame run lasted 536.46 seconds with stable first/last free-memory counters.

Based on [Espressif model-run API](https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/how_to_run_model.html) and aligned .espdl embedding utilities. No code in Smart-Agriculture-Edge-AI is changed.

Acceptance includes the golden Fixture, Camera single-frame, fixed-image reference comparison, 100/500-frame and sustained experiments. Hardware inference acceptance: **PASS**.

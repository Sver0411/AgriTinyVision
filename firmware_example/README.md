# ESP32-S3 N16R8 fixture inference example

**PENDING HARDWARE VALIDATION**. This example is a preprocessed image fixture, not a camera/Gateway application. It is prepared against ESP-IDF 5.4.4 and ESP-DL 3.3.13; actual compile status is recorded in reports/FIRST_RUN.md.

```bash
python scripts/prepare_firmware.py
cd firmware_example
idf.py set-target esp32s3
idf.py build
# When the user is ready to test on their actual S3 N16R8:
idf.py -p PORT flash monitor
```

Model and golden fixture are generated from a real successful ESP32-S3 conversion and a train calibration image, never placeholder binaries. `fixture_input.bin` is normalized NHWC float32; the C++ example quantizes it using the actual model input exponent and dequantizes output logits before softmax/threshold logic. Check .info against expected INT8 [1,160,160,3] input and two output logits. Generated artifacts are ignored by Git; reproduce them with the script after obtaining the actual release model/training run.

Production camera decode, orientation, bilinear antialias resize, the quality gate itself, metadata loading, integrity verification and Gateway decision routing are not implemented in this fixture. Quality-pass and thresholds are generated from Python for the fixture only. A successful firmware build does not establish preprocessing parity, memory fit or agricultural accuracy. No automatic flashing is done by this project.

Measure on the actual board: model load time, repeated warmed inference latency (median/p95), peak internal RAM and PSRAM attributable to model plus camera, Flash use, camera preprocessing latency and power with external measurement. Free-heap counters alone are not peak model RAM. Log correct artifact SHA256, component versions, board/clock/PSRAM config and fixture output agreement first. All measurements remain PENDING until real hardware runs.

Based on [Espressif model-run API](https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/how_to_run_model.html) and aligned .espdl embedding utilities. No code in Smart-Agriculture-Edge-AI is changed.

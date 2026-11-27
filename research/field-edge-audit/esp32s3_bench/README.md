# ESP32-S3 On-Device Benchmark for BASA

This standalone ESP-IDF v5.4+ project measures the exact on-device latency, SRAM footprint, and bit-exact numerical parity of the BASA 1,294-byte `EQ01` INT8 classification head, 1,280-D activation quantization, and 64-frame rolling percentile streaming gate on an **ESP32-S3 N16R8** (matching the `AgriTinyVision` v0.2 hardware setup).

## Quick Start (ESP-IDF v5.4+)

```bash
cd esp32s3_bench
idf.py set-target esp32s3
idf.py build flash monitor
```

Wait ~2 seconds after boot. Copy the single line starting with `BENCH_JSON: {...}` from the serial monitor and save it to `results/basa/esp32s3_hardware_bench.json` (or paste it back into the paper's hardware table).

## Host Sanity Check (No Board Needed)

```bash
cc -O2 -std=c99 -Wall -Wextra esp32s3_bench/main/basa_bench_main.c -o /tmp/basa_host_bench -lm && /tmp/basa_host_bench
```

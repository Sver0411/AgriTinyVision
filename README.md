<h1 align="center">AgriTinyVision v0.3</h1>

---

<p align="center">
  <a href="#project-overview">Overview</a> ·
  <a href="#pc-camera-web-console">Web Console</a> ·
  <a href="#method-and-inference-states">Method</a> ·
  <a href="#key-results-and-current-status">Status</a> ·
  <a href="#relationship-to-smart-agriculture-edge-ai">Related System</a> ·
  <a href="README.zh-CN.md">简体中文</a>
</p>

---

<a id="project-overview"></a>

AgriTinyVision trains compact crop-image classifiers and runs the frozen MobileNetV2 160 INT8 model on ESP32-S3 N16R8. A PC camera supplies images over USB; a local Web console displays the device's predictions, timing and memory measurements, and saves experiment reports.

## PC camera Web console

Requirements: Windows 11, Python 3.10+, a PC camera, ESP32-S3 N16R8 and a USB data cable. Firmware uses ESP-IDF 5.4.4 and ESP-DL 3.3.13.

Place the existing model bundle in `releases/v0.1.0`, then run from the checkout:

```powershell
python -m pip install -e '.[camera,web,dev]'
python scripts/run_pc_camera.py --list-ports
python scripts/prepare_camera_firmware.py --bundle releases/v0.1.0
# ESP-IDF 5.4.4 PowerShell
.\scripts\build_firmware.ps1 -Mode Camera -Transport USBJTAG
# Install manually on the selected port
python "$env:IDF_PATH\tools\idf.py" -C firmware_example -B "$PWD\build-camera" -p COM3 flash
# Project Python environment: start the Web console
python scripts/run_pc_camera.py --port COM3 --camera 0 --fps 1
```

The application opens `http://127.0.0.1:8765` in the computer's browser. Connect the camera and ESP32, run a single frame, then start continuous inference. The page includes probabilities, quality rejection reasons, device timing, latency history, memory and experiment ZIP download.

The PC decodes, orients and resizes RGB images with the existing PIL bilinear-antialias path. ESP32 performs quality checks, ImageNet normalization, quantization using the actual tensor exponent, ESP-DL inference and state decisions. Browser JPEG previews are separate from the RGB888 model input. ONNX comparison runs through a separate offline script.

See [Windows setup](docs/PC_CAMERA_SETUP.md), [USB protocol](docs/USB_PROTOCOL.md), [deployment](docs/HARDWARE_DEPLOYMENT.md) and [hardware results](reports/HARDWARE_V02.md).

## v0.3 reliability and comparison workflow

The Web console and headless runner now verify the connected model against the actual frozen bundle: model SHA256, version, deployment SHA256, tensor shapes/types/exponents, quality contract and decision bounds. A mismatch closes the connection before any frame is sent. The existing v0.2 firmware supplies these fields; its binary and the model remain unchanged. Use `--bundle` to select the experiment bundle and optional `--deployment` to additionally validate a generated deployment JSON.

Controlled image probes share one RGB SHA256 across ONNX FP32, frozen ESP-PPQ simulation and ESP32. Each backend records its own probabilities and timings. Prepare and run a comparison with:

```powershell
python scripts/prepare_analysis_scenes.py --image PATH_TO_TRAIN_LEAF --source DATASET_ID --label NORMAL --split train --output runs/v03/my-scenes
python scripts/quantization_analysis.py --manifest runs/v03/my-scenes/manifest.json --bundle releases/v0.1.0 --port COM3 --output runs/v03/my-comparison
# Add the original trusted, hash-bound frozen PPQ graph when available:
# --host-model PATH_TO_FROZEN.espdl --host-report PATH_TO_FROZEN.espdl.quantization.json
python scripts/analyze_performance.py --run RUN_DIRECTORY
python scripts/export_experiment.py --run RUN_DIRECTORY --public-output NEW_SANITIZED_DIRECTORY
python -m pytest -q --strict-markers
# Explicit local artifact and physical-device checks:
$env:ATV_HARDWARE_PORT='COM3'
python -m pytest -q -m 'hardware or local_artifact'
```

The host model requires its adjacent `.native` graph and a matching `simulation_graph_sha256`; only load a trusted locally produced graph. Missing original graphs remain PENDING; comparison never recalibrates or modifies the official model. The ten v0.3 probes cover source-leaf lighting/contrast/blur/color variations plus synthetic non-plant, checker and noise images. Their source leaf belongs to the training split. They measure deployment differences, not field accuracy. Results and the color-permutation disagreement are documented in [quantization analysis](reports/QUANTIZATION_ANALYSIS_V03.md).

| v0.3 validation | Evidence |
| --- | --- |
| Model identity and wrong-model refusal | Physical COM3 checks and software mismatch tests |
| FP32 versus device | Ten controlled probes; eight probability pairs, nine of ten gated states agree |
| Frozen PPQ comparison for these probes | PENDING: original model's native graph is absent |
| Web security | Local same-site session, Host/Origin checks, CSRF token and bounded exports; non-loopback binding requires independent password authentication |
| Reproducibility | Project/Git/source hashes, device identity, dependencies, configuration, start/end time and stage timings |
| Software / hardware evidence | See [v0.3 validation](docs/V03_VALIDATION.md); simulated transports are kept separate from physical checks |
| Independent field accuracy / power measurement | PENDING: independent labeled field data / external power instrument |

Local browsing remains at `127.0.0.1`; status, preview and experiment downloads require the same-site console session. Non-loopback binding requires `ATV_WEB_PASSWORD` (at least 16 characters), with browser login username `agri`. The public-data preparation tool creates a separate sanitized directory and never uploads it. See [v0.3 guide](docs/V03_RELIABILITY.md).

## Method and inference states

The [collection and annotation workspace](http://127.0.0.1:8765/field) adds opt-in lossless photo batches, independent human labels, CSV/ZIP export and frozen three-state evaluation. A first external PlantDoc test subset contains 60 source-labelled images: device binary accuracy 88.33%, accepted coverage 95.00%, accepted accuracy 89.47%, UNKNOWN 3/60. Normal-reference false alarms are 5/12. These figures use dataset-author labels and describe this fixed subset. Human quality review and independent grouped field splits remain PENDING. See the [operation guide](docs/FIELD_VALIDATION.md), [results](docs/FIELD_VALIDATION_RESULTS.md) and [aggregate reports](reports/field_validation/README.md).

```text
Plant image → quality gate → compact model → confidence / decision threshold
                  │                         │
                  └── rejected → UNKNOWN ←──┘
                           accepted → NORMAL / SUSPECT
```

| Outcome | Meaning |
| --- | --- |
| `NORMAL` | Accepted input classified as normal |
| `SUSPECT` | Accepted input flagged for further inspection |
| `UNKNOWN` | Rejected/unreadable/poor-quality input, invalid output, or insufficient confidence |

Only NORMAL and SUSPECT are trained classes. UNKNOWN is a rejection state. The frozen validation threshold is 0.30, with the current threshold-centered policy exporting float32 bounds of 0.20 and 0.40: lower probabilities yield NORMAL, the middle interval yields UNKNOWN, and upper probabilities yield SUSPECT. Quality failure takes priority. Confidence is the assigned class probability; UNKNOWN has null confidence. Details are in the [integration contract](docs/INTEGRATION_CONTRACT.md).

The baseline is torchvision MobileNetV2 1.0 with ImageNet `IMAGENET1K_V1` weights and full fine-tuning. The 160×160 and 224×224 variants share split, seed, and epoch budget. Thresholds are selected on validation data; candidate and threshold are frozen before final test evaluation. Binary metrics are reported separately from coverage, UNKNOWN rate, and accepted-input metrics.

## Key results and current status

The first pipeline run used an RTX 4060 Laptop GPU. The selected v0.1 candidate is MobileNetV2 160×160: validation results were similar at both sizes, while 160×160 uses roughly half the convolution/linear MACs. Later test scores did not change that selection.

| Model | Test accuracy | SUSPECT recall | FNR | Artifact or evaluation |
| --- | ---: | ---: | ---: | --- |
| MobileNetV2 160 FP32 | 99.76% | 100.00% | 0.00% | 8,878,502-byte ONNX |
| MobileNetV2 160 INT8 | 99.52% | 99.65% | 0.35% | 2,438,752-byte ESP-DL model; host simulation |
| MobileNetV2 224 FP32 | 100.00% | 100.00% | 0.00% | 8,878,502-byte ONNX |
| MobileNetV2 224 INT8 | 100.00% | 100.00% | 0.00% | 2,511,296-byte ESP-DL model; host simulation |

These scores come from a curated PlantVillage subset with controlled backgrounds, not field footage. INT8 results are **host-side quantized graph simulation**, not MCU measurements.

| Stage | Evidence |
| --- | --- |
| PyTorch training / frozen test evaluation | Completed on desktop GPU |
| ONNX export | Numerical checks against at least four identical PyTorch inputs |
| ESP-PPQ → ESP-DL INT8 | Converted for `esp32s3` w8a8; full-test host graph simulation |
| ESP-IDF firmware | Fixture, USB Serial/JTAG and UART branches compiled |
| Real ESP32 inference | Golden Fixture and 15 quality/protocol checks passed |
| PC camera path | 100/100 live-camera frames returned valid device results |
| Sustained inference | 500/500 fixed-leaf frames in 536.46 seconds, no restarts or first/last free-memory decrease |
| v0.2 software regression | 138 tests passed (historical baseline) |

The [first-run report](reports/FIRST_RUN.md) retains the training and host-simulation results. The [v0.2 hardware report](reports/HARDWARE_V02.md) records actual ESP32-S3 measurements: 527.840/527.850 ms inference P50/P95, 286,447 bytes free internal RAM, 5,046,852 bytes free PSRAM and a 3,574,960-byte application image. Model file size, application image size and partition capacity are recorded separately.

## Data and evaluation scope

The experiment uses 2,792 raw/color images from the [official PlantVillage dataset](https://github.com/spMohanty/PlantVillage-Dataset): apples, bell peppers, potatoes, and tomatoes, 12 source labels, capped at 240 images per label. This is not the full dataset; grayscale/segmented variants are excluded. New labels require explicit mapping in [label_mapping.yaml](configs/label_mapping.yaml).

Manifests record source/binary labels, leaf IDs, hashes, and splits. Leaf-ID and exact/near-duplicate groups are kept together before deterministic splitting. Missing plant/session/scene IDs mean these controls cannot rule out every kind of leakage. Raw images are ignored by Git. See the [data guide](data/README.md) and [audit](reports/dataset_audit.md).

## Quick start and reproduction

Python 3.10+; the workflow supports Windows/Linux. Training and GPU evaluation use CUDA by default; use explicit `--allow-cpu` when needed. Install a compatible PyTorch wheel first, then project dependencies. The first run used Python 3.12.15 and PyTorch 2.11.0+cu128 on Windows 11.

```bash
python -m venv .venv
source .venv/bin/activate
# Windows activation: .venv\Scripts\activate
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[dev]'
python -m pip install esp-ppq
python -m agritinyvision.doctor --output reports/environment.json
```

For the locked environment, install the appropriate CUDA wheel then [requirements-lock.txt](requirements-lock.txt). On other platforms choose a compatible wheel and rerun the environment check.

Prepare, audit, and train both baseline sizes:

```bash
python scripts/download_plantvillage.py
python -m agritinyvision prepare-data
python -m agritinyvision audit
python -m agritinyvision preflight --config configs/train_mobilenetv2_160.yaml
python -m agritinyvision train --config configs/train_mobilenetv2_160.yaml --output runs/baseline160
python -m agritinyvision train --config configs/train_mobilenetv2_224.yaml --output runs/baseline224
```

For a new experiment, preserve [the published selection](reports/candidate_selection.json) outside the output path in a separate experiment copy, then run `python scripts/select_candidate.py` before testing. The command refuses to overwrite an existing selection or select after test reports exist. Do not use test scores to tune training, thresholds, or gates. Evaluate, infer, export, and quantize the selected candidate:

```bash
python -m agritinyvision evaluate --checkpoint runs/baseline160/best.pt
python -m agritinyvision infer --checkpoint runs/baseline160/best.pt --image /path/to/leaf.jpg
python -m agritinyvision export --checkpoint runs/baseline160/best.pt --format onnx
python -m agritinyvision quantize --checkpoint runs/baseline160/best.pt --model runs/baseline160/tinyvision.onnx
python -m agritinyvision release \
  --checkpoint runs/baseline160/best.pt \
  --onnx runs/baseline160/tinyvision.onnx \
  --metrics runs/baseline160/test_metrics.json \
  --espdl runs/baseline160/tinyvision.espdl
```

The path is PyTorch → ONNX → ESP-PPQ → ESP-DL. ONNX uses batch 1, NCHW FP32, opset 18, and two logits. INT8 calibration uses 256 stratified training images without random augmentation; full-test graph simulation retains the FP32 validation threshold. Static checks use the [ESP-DL operator support list](https://github.com/espressif/esp-dl/blob/master/operator_support_state.md).

Optional `quantize --backend onnx` is a desktop ONNX Runtime comparison; `benchmark` is also desktop-only. Resume uses `train --resume runs/baseline160/last.pt --config configs/train_mobilenetv2_160.yaml` and requires matching config/manifests. Further workflow details are in the [first-run report](reports/FIRST_RUN.md) and CLI `--help`.

Release validation checks model hashes, frozen metrics, ONNX checks, conversion reports, and metadata. Failed conversion preserves its error; research bundles without ESP-DL mark export PENDING and never fabricate a model. Checkpoints/binaries are excluded from Git and intended for Release assets. The [integration contract](docs/INTEGRATION_CONTRACT.md) and [firmware guide](firmware_example/README.md) define preprocessing, tensor layout, and hardware acceptance. Optional checks: `python -m pytest -q` and `python -m ruff check agritinyvision scripts tests`.

## Relationship to Smart-Agriculture-Edge-AI

[Smart-Agriculture-Edge-AI](https://github.com/Sver0411/Smart-Agriculture-Edge-AI) is the related system project and a future visual-integration target. AgriTinyVision's camera/model path is not connected to a physical main-system visual control loop; gateway control, LoRa, failover, and synchronization are separate system work.

## Limitations and future work

Independent labeled field validation and external-instrument power measurements are PENDING. The PlantVillage scores describe the controlled dataset; the 536.46-second fixed-image run describes that input and duration. Quality gates check blur, brightness and contrast rather than plant presence or complete OOD coverage. Quantization optimization remains a separate candidate study; v0.3 retains the frozen exponent and thresholds.

## License

The code is licensed under the [MIT License](LICENSE). Dataset images, pretrained weights, and other external materials remain subject to their respective terms; the repository's code license does not relicense those materials.

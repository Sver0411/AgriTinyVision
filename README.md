<h1 align="center">AgriTinyVision</h1>

<p align="center"><strong>Lightweight Crop-Image Anomaly Screening for Edge Research</strong></p>

<p align="center">Image Triage · Uncertainty Rejection · INT8 Export</p>

---

<p align="center">
  <a href="#project-overview">Overview</a> ·
  <a href="#method-and-inference-states">Method</a> ·
  <a href="#key-results-and-current-status">Status</a> ·
  <a href="#relationship-to-smart-agriculture-edge-ai">Related System</a> ·
  <a href="README.zh-CN.md">简体中文</a>
</p>

---

<a id="project-overview"></a>

AgriTinyVision is a lightweight crop-image anomaly-screening research pipeline targeting ESP32-S3. It trains compact classifiers in PyTorch, exports ONNX, and converts INT8 candidates to ESP-DL. The question is whether an image warrants closer inspection; this is not a disease diagnosis system.

## Method and inference states

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

Only NORMAL and SUSPECT are trained classes. UNKNOWN is a rejection or uncertainty outcome, **not a third training class**. Quality checks use sharpness, brightness, contrast, image size, and decoding; confidence uses maximum softmax probability. These provisional gates need field calibration and are not a complete out-of-distribution detector. Plant-free or unfamiliar images can still receive confident predictions.

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
| ESP-IDF firmware example | Build successful; **no board inference run** |
| Device memory, latency, camera path, power | Pending hardware validation |

The [first-run report](reports/FIRST_RUN.md) retains detailed results. The target ESP32-S3 N16R8 has 16 MB Flash and 8 MB PSRAM; conversion and operator checks do not prove the complete model fits or runs there.

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

Board inference, model load time, internal RAM/PSRAM, total Flash, camera preprocessing latency, and power remain unvalidated. PlantVillage bias and domain shift prevent interpreting high test scores as farm disease-recognition performance. Field camera data, independent validation, threshold calibration, and board measurements are required; fine-grained diagnosis, segmentation, and severity estimation are outside v0.1.

## License

The code is licensed under the [MIT License](LICENSE). Dataset images, pretrained weights, and other external materials remain subject to their respective terms; the repository's code license does not relicense those materials.

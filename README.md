# AgriTinyVision

**A small vision model pipeline for offline crop anomaly triage on ESP32-S3.**

AgriTinyVision is the companion model project to [Smart-Agriculture-Edge-AI](https://github.com/Sver0411/Smart-Agriculture-Edge-AI). It uses an RTX GPU to train and evaluate compact image classifiers, then quantizes and packages a candidate for Gateway A, an ESP32-S3 N16R8 with 16 MB of flash and 8 MB of PSRAM.

The v0.1 goal is to answer one narrow question: **does this plant image show something that deserves a closer look?** It is not a disease diagnosis system.

```text
Plant image → image-quality gate → Tiny Vision model → Gateway decision engine
                                      NORMAL / SUSPECT
                           Input or quality failure → UNKNOWN
```

NORMAL and SUSPECT are the model's two trained classes. UNKNOWN is an inference outcome for unreadable or poor-quality input, invalid output, or insufficient confidence. It is not a third class. The gateway can use the result to request another image or further inspection.

This repository focuses on the model pipeline. Gateway actuators, pumps, fans, LoRa, failover, and cloud synchronization belong to the companion project. Fine-grained disease classification, segmentation, severity estimation, and trend analysis are future server-side work and are outside the v0.1 scope.

## Project status

The first end-to-end run was completed on an RTX 4060 Laptop GPU. It includes 160×160 and 224×224 baselines, a frozen test evaluation, PyTorch-to-ONNX numerical checks, ESP32-S3 INT8 conversion to ESP-DL format, host-side quantized graph simulation, and a successful ESP-IDF firmware example build.

**The model has not yet been run on an ESP32 board.** Device memory, latency, camera preprocessing, and power measurements remain pending hardware validation. The [first-run report](reports/FIRST_RUN.md) records the experiments, results, and limitations.

The selected v0.1 candidate is the 160×160 MobileNetV2. It was frozen before final test evaluation: both input sizes had similar validation results, while the 160×160 model uses roughly half the convolution and linear MACs. The later test results are reported below for transparency; they did not change the selection.

| Model | Test accuracy | SUSPECT recall | FNR | Artifact or evaluation |
| --- | ---: | ---: | ---: | --- |
| MobileNetV2 160 FP32 | 99.76% | 100.00% | 0.00% | 8,878,502-byte ONNX |
| MobileNetV2 160 INT8 | 99.52% | 99.65% | 0.35% | 2,438,752-byte ESP-DL model; host simulation |
| MobileNetV2 224 FP32 | 100.00% | 100.00% | 0.00% | 8,878,502-byte ONNX |
| MobileNetV2 224 INT8 | 100.00% | 100.00% | 0.00% | 2,511,296-byte ESP-DL model; host simulation |

These figures come from a curated PlantVillage subset, not field footage. They measure this training and export pipeline; they do not establish real-world disease-detection performance.

## Environment and installation

Python 3.10 or newer is required. The workflow supports Windows and Linux. Training and GPU evaluation use CUDA by default and do not silently fall back to CPU. If CPU execution is necessary, pass the explicit --allow-cpu option. The ONNX Runtime benchmark runs on a desktop CPU and says nothing about ESP32 inference speed.

The first run used Python 3.12.15, PyTorch 2.11.0+cu128, and an RTX 4060 Laptop GPU on Windows 11. Install a PyTorch build that matches your operating system, CUDA runtime, and driver.

~~~bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux
source .venv/bin/activate

python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[dev]'
python -m pip install esp-ppq
python -m agritinyvision.doctor --output reports/environment.json
~~~

To reproduce the locked environment, install the appropriate CUDA PyTorch wheel first, then install the versions in requirements-lock.txt. On a different platform, choose a compatible PyTorch wheel and rerun the environment check; one machine's environment report does not validate another machine.

## Data preparation and audit

The first experiment uses a capped subset of raw/color images from the [official PlantVillage dataset](https://github.com/spMohanty/PlantVillage-Dataset). It covers apples, bell peppers, potatoes, and tomatoes across 12 source labels, with at most 240 images per source label: 2,792 images total. This is not the full PlantVillage dataset. Grayscale and segmented variants are excluded. The source commit and explicit source-label mapping are recorded in the repository.

~~~bash
python scripts/download_plantvillage.py
python -m agritinyvision prepare-data
python -m agritinyvision audit
~~~

The downloader places original images in data/raw/, which is ignored by Git. Manifests in data/manifests/ record file paths, source labels, binary labels, leaf IDs, SHA-256 hashes, and split assignments. The audit checks for path traversal, missing or corrupt files, invalid labels, exact duplicates, and perceptual near-duplicates.

Splitting is group-based, not a random split of individual images. Images connected by a leaf ID or duplicate/near-duplicate relationship are grouped before a deterministic GroupShuffleSplit. The dataset does not provide reliable plant, capture-session, or scene IDs, and leaf IDs plus perceptual hashes cannot rule out every form of leakage. That limitation remains.

New source labels must be mapped explicitly in configs/label_mapping.yaml. Unmapped labels are not automatically treated as diseased.

## Training

The baseline is torchvision MobileNetV2 1.0 with ImageNet IMAGENET1K_V1 pretrained weights and full-parameter fine-tuning. The 160×160 and 224×224 runs use the same split, seed, and epoch budget. A pretrained: false setting supports training from scratch; narrower MobileNetV2 variants must be trained from scratch because torchvision does not provide matching pretrained weights.

Augmentation is applied only to training images: moderate random crops, horizontal flips, small rotations, and brightness, contrast, and saturation changes. Validation, test, and calibration images use deterministic RGB square resizing, scaling to 0–1, and ImageNet mean/std normalization.

~~~bash
python -m agritinyvision preflight --config configs/train_mobilenetv2_160.yaml
python -m agritinyvision train --config configs/train_mobilenetv2_160.yaml --output runs/baseline160
python -m agritinyvision train --config configs/train_mobilenetv2_224.yaml --output runs/baseline224

# Freeze the candidate before any final test evaluation.
python scripts/select_candidate.py
~~~

Each run stores its configuration, environment, training history, best and last checkpoints, validation metrics, and threshold search. Provenance includes the source snapshot, data and training-split hashes, seed, optimizer, scheduler, AMP settings, GPU, CUDA version, training time, and peak VRAM.

Training supports AdamW or SGD with a cosine scheduler and validation-based early stopping. Resume a run with:

~~~bash
python -m agritinyvision train --resume runs/baseline160/last.pt --config configs/train_mobilenetv2_160.yaml
~~~

Resume requires the configuration and data manifest to match. Deterministic execution is best-effort; unsupported deterministic kernels produce a warning. Load PyTorch checkpoints only from sources you trust.

## Evaluation and inference

~~~bash
python -m agritinyvision evaluate --checkpoint runs/baseline160/best.pt
python -m agritinyvision infer --checkpoint runs/baseline160/best.pt --image /path/to/leaf.jpg
~~~

The SUSPECT decision threshold is selected on the validation set by searching 0.20–0.80 in 0.05 increments. The selection first seeks SUSPECT recall of at least 0.95 and a false-positive rate no greater than 0.35, then chooses the best F1 among thresholds that meet those constraints. If no threshold meets both, the run records that fact and falls back with recall as the priority. Candidate selection does not use accuracy as its only criterion.

After the candidate and threshold are frozen, the test set is used for final evaluation only. Do not use test results to tune training, thresholds, or the quality gate. The evaluate command refuses to overwrite an existing final report.

Reports include accuracy, precision, SUSPECT recall, specificity, F1, ROC-AUC, PR-AUC (average precision), confusion matrix, false-positive rate (FPR), and false-negative rate (FNR). Raw binary-classification metrics are reported separately from system coverage, UNKNOWN rate, and metrics on accepted inputs. This prevents a system from appearing more accurate merely by rejecting difficult images.

The independent quality gate in configs/quality.yaml checks Laplacian variance, brightness, and contrast on a fixed 160×160 grayscale image. Decode failures and images that are too small also return UNKNOWN. After the quality check, the confidence gate returns UNKNOWN when the model's maximum softmax probability is below min_confidence; otherwise the SUSPECT probability is compared with the frozen validation threshold.

These quality and confidence thresholds are provisional heuristics that still need field calibration. They are not a complete out-of-distribution detector and cannot confirm that an image contains a plant. An unfamiliar or plant-free image may still receive a high-confidence prediction. Gateway software must not treat that prediction as a disease diagnosis.

## Export, quantization, and release

~~~bash
python -m agritinyvision export --checkpoint runs/baseline160/best.pt --format onnx

python -m agritinyvision quantize \
  --checkpoint runs/baseline160/best.pt \
  --model runs/baseline160/tinyvision.onnx

# Optional desktop ONNX Runtime INT8 comparison; this is not an MCU deployment.
python -m agritinyvision quantize \
  --checkpoint runs/baseline160/best.pt \
  --model runs/baseline160/tinyvision.onnx \
  --backend onnx

python -m agritinyvision evaluate \
  --checkpoint runs/baseline160/best.pt \
  --model runs/baseline160/tinyvision.int8.onnx

python -m agritinyvision benchmark --model runs/baseline160/tinyvision.onnx

python -m agritinyvision release \
  --checkpoint runs/baseline160/best.pt \
  --onnx runs/baseline160/tinyvision.onnx \
  --metrics runs/baseline160/test_metrics.json \
  --espdl runs/baseline160/tinyvision.espdl
~~~

The ONNX reference has batch size 1, NCHW float32 input, opset 18, and two logits. Export validation compares at least four identical inputs against PyTorch. A static preflight checks operators against the [ESP-DL support list](https://github.com/espressif/esp-dl/blob/master/operator_support_state.md); this does not prove the full model will fit in 8 MB of PSRAM. The export preserves torchvision's ReLU6 behavior as ONNX Clip.

The deployment conversion path is PyTorch → ONNX → ESP-PPQ → ESP-DL. It targets esp32s3 with w8a8 quantization. Calibration uses 256 stratified, round-robin training images without random augmentation. Quantized graph simulation runs across the full test set, while retaining the threshold chosen on FP32 validation. A successful conversion or host-side simulation still does not confirm that the model runs on the physical board.

If conversion fails, the original error is preserved. If no ESP-DL file was produced, the release command can create a research bundle with ESP-DL export marked PENDING; it never creates a placeholder model. Release validation checks model hashes, frozen test metrics, ONNX numerical checks, the ESP-DL conversion report, and the metadata schema.

Full checkpoints and model binaries are excluded from Git. Model artifacts are intended to be distributed as GitHub Release assets. See the [Gateway A integration contract](docs/INTEGRATION_CONTRACT.md) for tensor layout, preprocessing, inference states, and hardware acceptance criteria.

## Hardware and field limitations

firmware_example/ prepares the ESP-IDF, ESP-DL, and ESP32-S3 integration. The firmware example compiles, but it has not been run on hardware. Model load time, inference latency, internal RAM, PSRAM, system flash usage, camera preprocessing latency, and power consumption are all **PENDING HARDWARE VALIDATION**. A desktop benchmark cannot stand in for these measurements.

PlantVillage images have clean backgrounds and controlled capture conditions and cover only four crops in this subset. Dataset bias and domain shift limit what the reported scores can tell us. The first run validates the training, quantization, and export workflow; it does not validate disease recognition on a farm. Field camera data, independent validation, and threshold recalibration are required before making field-performance claims.

## Tests

~~~bash
python -m pytest -q
python -m ruff check agritinyvision scripts tests
~~~

CI uses small synthetic images. It does not download crop data or run a full training job. The first-run report records 36 passing tests, along with the environment and firmware build results.

## License

The code is licensed under the MIT License. Dataset images, pretrained weights, and other external materials remain subject to their respective terms; the repository's code license does not relicense those materials.

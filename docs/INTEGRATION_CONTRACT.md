# AgriTinyVision ↔ Gateway A integration contract (v0.1)

Producer: AgriTinyVision. Consumer: `Sver0411/Smart-Agriculture-Edge-AI`, Gateway A, ESP32-S3 N16R8. This repository never changes the consumer system or implements its actuator, LoRa, failover or cloud logic.

## Release bundle

| File | Contract |
| --- | --- |
| tinyvision.espdl | Actual ESP-PPQ export for esp32s3, w8a8; absent when conversion PENDING |
| tinyvision.onnx | Fixed batch 1 NCHW float32 reference, two logits; not MCU model |
| metadata.json | Version, hashes, binary classes, threshold, preprocessing, quality gate and provenance |
| labels.txt | Exactly NORMAL then SUSPECT, one per line |
| metrics.json | Frozen-threshold test metrics; FP32 and quantized host simulation distinguished |
| README.md | Explicit ESP-DL export and hardware validation status |
| tinyvision.info / tinyvision.json | Converter tensor layout, exponent/dtype and quantization diagnostics if produced |

Machine-readable metadata schema: [metadata.schema.json](metadata.schema.json). SHA256 identifies the .espdl if present, otherwise the ONNX. `artifact_sha256` binds each model independently. A bundle without a SUCCESS conversion report is never a deployment-ready bundle; successful conversion is still **PENDING HARDWARE VALIDATION**.

## Input and numerical processing

1. Read and validate image; respect orientation, decode RGB. The quality gate applies to the image before model normalization.
2. Quality metrics use a fixed 160×160 image and gray = 0.299R + 0.587G + 0.114B. Laplacian uses a four-neighbor kernel without boundary pixels. Compare variance, brightness mean, contrast std against metadata configuration. Small images / decode failure → UNKNOWN.
3. Resize directly to metadata width × height using bilinear antialias; no center crop. Divide RGB by 255, then subtract mean [0.485,0.456,0.406] and divide std [0.229,0.224,0.225]. Firmware resize differences must be checked with golden vectors, never assumed equivalent.
4. ONNX reference uses float32 [1,3,H,W]. **ESP-DL requires the tensor layout and quantized dtype/exponent actually exported** (usually NHWC for Conv). Inspect .info / model tensor shape and use the model input's exponent. Do not pass a normalized ONNX NCHW float pointer to an INT8 NHWC input.
5. Output is two logits in labels.txt order. Dequantize actual output dtype/exponent; stable softmax → p_suspect. Ensure finite values.

## State logic

```text
input/decode/quality fail → UNKNOWN
otherwise run model
invalid model output → UNKNOWN
max(p_normal, p_suspect) < min_confidence → UNKNOWN
otherwise p_suspect >= decision_threshold → SUSPECT
otherwise → NORMAL
```

UNKNOWN is never a third trained class. Decision threshold is selected only on validation, then frozen before any test evaluation. The confidence gate uses binary maximum probability, not calibrated certainty or proof of plant presence. Simple blur/brightness/contrast cannot detect arbitrary OOD or all no-plant content; those capabilities are not implemented in v0.1. A high-confidence out-of-domain input may pass. UNKNOWN must be routed to inspection/retry by the Gateway decision engine; never convert it silently to NORMAL.

Suggested result envelope (not an actuator command):

```json
{"state":"UNKNOWN","reason":"quality_gate","model_version":"0.1.0","model_sha256":"...","suspect_probability":null}
```

Allowed state: NORMAL / SUSPECT / UNKNOWN. For valid model output include p_suspect, decision_threshold, maximum binary confidence and quality reasons. Gateway chooses whether to capture again, queue/upload, or request GPU inspection.

## Hardware acceptance gate

Model load time, inference latency, peak internal RAM, PSRAM, Flash, camera preprocessing latency and power: **PENDING HARDWARE VALIDATION**. Require correct model hash, tensor dimensions/order, preprocessing parity, output agreement on exported test vectors, and actual memory fit while camera/Gateway services run. Host benchmark/simulation and operator listing do not certify deployment. No hard field-accuracy claim until independent Gateway camera data validation and threshold recalibration.

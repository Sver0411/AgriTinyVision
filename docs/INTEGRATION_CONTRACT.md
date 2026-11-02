# AgriTinyVision ↔ Gateway A integration contract (v0.2)

## PC camera deployment

PC captures BGR, applies configured orientation, converts RGB and resizes with the existing PIL bilinear-antialias path to 160×160. The ESP32 receives RGB888 using [ATV2](USB_PROTOCOL.md), performs quality checks, ImageNet normalization, quantization with the actual tensor exponent, ESP-DL inference and the state policy below. Camera mode and the retained golden Fixture are separate Kconfig build entries.

`camera_deployment.json` binds the real model and source metadata/quality configuration. Historical v0.1.0 metadata remains intact; camera deployment uses the current threshold-centered-v1 policy and exports the same float32 bounds as the Fixture. Confidence is the probability of the assigned class; UNKNOWN has null confidence. It does not alter the state policy.

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

Machine-readable metadata schema: [metadata.schema.json](metadata.schema.json). SHA256 identifies the .espdl if present, otherwise the ONNX. `artifact_sha256` binds each model independently. Conversion and physical validation are separate evidence: the frozen 160 model has the v0.2 hardware results linked below; other candidates require their own device checks.

This contract includes the Final Correctness Patch. The original v0.1.0 release and its old metadata/metrics remain immutable historical evidence. They do not report coverage under the corrected abstention rule. Newly generated bundles require `decision_threshold_source=validation`, `quantization_selection_split=validation` (or not_applicable for FP32-only research), and `final_metrics_split=test`. Conversion evaluates val only; explicit final evaluation requires the frozen checkpoint SHA256 and, for ONNX/INT8, the frozen artifact SHA256. Quantized final reports must bind the validation-only conversion report hash and the exact checkpoint/artifact. Test-based quantization selection or missing provenance fails release validation.

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
p_suspect < normal_upper_bound → NORMAL
normal_upper_bound <= p_suspect < suspect_lower_bound → UNKNOWN
p_suspect >= suspect_lower_bound → SUSPECT
```

Invalid output includes NaN, Inf and values outside [0,1]. Quality failure has priority regardless of probability. `normal_upper_bound = decision_threshold - abstention_margin`, `suspect_lower_bound = decision_threshold + abstention_margin`. With threshold=0.30 and margin=0.10 the conceptual intervals are [0,0.20) NORMAL, [0.20,0.40) UNKNOWN, [0.40,1] SUSPECT. States are monotonic with p_suspect. Do not use max(p,1-p) to decide or describe confidence in the assigned state.

The producer computes bounds from decimal configuration values and exports their float32 representations in metadata and the fixture header. Both Python and firmware compare float32 probabilities to these **same exported constants**. Firmware must not independently compute float32 threshold +/- margin, which can shift the lower boundary by one ULP. The actual metadata bounds are approximately 0.20000000298 and 0.40000000596. Validate the original probability before casting; this convention only standardizes state decisions, not all Python/MCU numerical inference.

UNKNOWN is never a third trained class. Decision threshold is selected only on validation and remains 0.30 for the frozen 160 candidate. Margin=0.10 is a provisional engineering default, not test-tuned. The quality gate measures blur, brightness and contrast; plant presence and complete OOD detection require separate methods. An out-of-domain input may pass. UNKNOWN must be routed to inspection/retry by the Gateway decision engine; never convert it silently to NORMAL.

Suggested result envelope (not an actuator command):

```json
{"state":"UNKNOWN","reason":"quality_gate","model_version":"0.1.0","model_sha256":"...","suspect_probability":null}
```

Allowed state: NORMAL / SUSPECT / UNKNOWN. For valid model output include p_suspect, decision_threshold, abstention_margin, exported state boundaries and quality reasons. Gateway chooses whether to capture again, queue/upload, or request GPU inspection.

## Camera preprocessing boundary

v0.2 keeps direct square bilinear antialias resize: the checkpoint, ONNX, .espdl reference and golden Fixture use this preprocessing. A non-square camera frame changes geometry when resized to 160×160; use independent field validation to assess that input domain.

Alternative center/ROI crop or letterbox policies can be compared on independent camera validation data. A preprocessing change requires new parity and domain validation; the frozen baseline retains its existing resize policy.

## Hardware acceptance gate

The v0.2 PC-assisted ESP32-S3 path passed real golden-Fixture, quality/protocol, camera 100-frame and fixed-image 500-frame validation. The 500-frame run lasted 536.46 seconds with unchanged first/last free-memory counters. Artifact hashes, tensor checks, load/inference timings and Flash measurements are recorded in [HARDWARE_V02.md](../reports/HARDWARE_V02.md). Free-memory counters describe the complete device session. Independent labeled field data is used for agricultural accuracy evaluation.

The first acceptance step is **same tinyvision.espdl → ESP32-S3 N16R8 → same golden input → model->run(), with no camera**. Record load success, prediction state, p_suspect, Python reference difference, warmed inference latency, internal RAM, PSRAM and Flash. Then run the PC camera single-frame and sustained experiments described in [HARDWARE_VALIDATION.md](HARDWARE_VALIDATION.md).

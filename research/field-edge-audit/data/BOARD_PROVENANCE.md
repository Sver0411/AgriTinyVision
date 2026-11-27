# Historical ESP32-S3 and FP32 image pairs

`board_pairs.csv` contains 60 historical predictions on paired images from the public AgriTinyVision hardware study. Each row is one original PlantDoc test image. No images are redistributed. The revision of AgriTinyVision is `16e7a2fdec43c02258f22e85b085ac269345e124`. The underlying PlantDoc release is `5467f6012d78d1c446145d5f582da6096f852ae8` and the PlantDoc repository specifies CC BY 4.0. [Original evaluation and selection procedure](https://github.com/Sver0411/AgriTinyVision/blob/16e7a2fdec43c02258f22e85b085ac269345e124/docs/FIELD_VALIDATION_RESULTS.md) · [PlantDoc dataset](https://github.com/pratikkayal/PlantDoc-Dataset/tree/5467f6012d78d1c446145d5f582da6096f852ae8) · [Dataset paper](https://arxiv.org/abs/1911.10317)

## What was paired

The original study ran the frozen MobileNetV2 160×160 ESP-DL INT8 model on an ESP32-S3 and the FP32 ONNX model on matching saved RGB inputs. The extraction joins source and inference records by sample ID and verifies the model-input pixel hash across four reports. The FP32 score is the probability of SUSPECT. The original JSON uses the separate key `host` for a quantized desktop simulator whose results were unavailable on all 60 images. [Hardware/FP32 parity records](https://github.com/Sver0411/AgriTinyVision/blob/16e7a2fdec43c02258f22e85b085ac269345e124/reports/field_validation/INFERENCE_PARITY.json) · [Independent device CSV](https://github.com/Sver0411/AgriTinyVision/blob/16e7a2fdec43c02258f22e85b085ac269345e124/reports/field_validation/INFERENCE_RESULTS.csv)

The binary decision uses the frozen float32 cutoff closest to 0.30. All 60 images passed the frozen input-quality gate. The three-state decision then yields NORMAL below the float32 0.20 cutoff, UNKNOWN from that cutoff up to the float32 0.40 cutoff, and SUSPECT from the latter cutoff upwards. All 60 crop health references came from dataset-author source classes. The `true_health` values healthy/diseased denote those mapped classes. There was no independent human quality or disease review. [Source manifest and labels](https://github.com/Sver0411/AgriTinyVision/blob/16e7a2fdec43c02258f22e85b085ac269345e124/reports/field_validation/SOURCE_MANIFEST.json)

## Splits and identifiers

The original study used a predetermined seed of 41 to select four examples from each of 15 PlantDoc test source classes. These comprise 12 apple, eight bell pepper, eight potato and 32 tomato images. There are 12 NORMAL and 48 SUSPECT labels. Every row belongs to the original PlantDoc `test` partition. Source plant and session identifiers are unavailable, and the upstream grouped split has a single conservative source group. Consequently all 60 rows retain the value `unassigned` for the separate downstream audit split. The upstream duplicate screen found zero exact/file/pixel or pHash-within-four matches against the specified 2,792 PlantVillage images. [Grouped split](https://github.com/Sver0411/AgriTinyVision/blob/16e7a2fdec43c02258f22e85b085ac269345e124/reports/field_validation/FIELD_SPLIT.json) · [Duplicate screen](https://github.com/Sver0411/AgriTinyVision/blob/16e7a2fdec43c02258f22e85b085ac269345e124/reports/field_validation/SOURCE_AUDIT.json)

| CSV column | Meaning |
| --- | --- |
| `image_id` | Original AgriTinyVision sample ID |
| `source_class`, `common_species`, `source_id` | PlantDoc class name, common crop name, and path within the pinned PlantDoc tree |
| `source_file_sha256`, `source_git_blob_sha1` | Source image file identifiers |
| `source_rgb_sha256`, `rgb_sha256` | Hashes of decoded original RGB and the matching 160×160 model input, respectively |
| `host_fp32_score`, `board_int8_score` | Frozen FP32 and measured ESP32-S3 SUSPECT probabilities |
| `*_binary`, `*_state` | Forced binary and original three-state decision outputs |
| `board_int8_inference_us`, `board_end_to_end_ms`, `host_fp32_inference_wall_ms` | Original instrumentation. Board end-to-end time includes its PC/device pipeline. A cloud round trip was not measured |
| `input_saturated_fraction`, `brightness`, `contrast`, `laplacian_variance` | Input saturation and image-quality measurements recorded by the original parity analysis |

## Reproduction

```bash
python3 field_pairs.py --prepare --json results/board_audit.json
```

This downloads four small SHA-256-verified reports from the pinned AgriTinyVision revision. To rebuild offline, add `--source /path/to/AgriTinyVision`. To audit the checked-in CSV without contacting GitHub, omit `--prepare`.

At the frozen binary cutoffs the two backends are both correct on 53 images, both wrong on six, and FP32 alone correct on one potato late-blight image. Therefore even an oracle routing to either recorded backend can reach only 54 correct binary decisions on this set. The signed three-state outcome differs. All three board UNKNOWN decisions become correct labels under FP32, and one correct board NORMAL becomes UNKNOWN under FP32. A prespecified policy that calls FP32 only on board UNKNOWN resolves three omissions and yields 54 correct, six wrong, and zero deferred at three calls. These figures describe the fixed 60-image evidence. Learning a routing policy requires a separate development cohort.

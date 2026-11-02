# 本阶段文件清单

本清单相对开始时的本地 v0.3 工作区；先前未提交改动保持原有基础。

新增 47，修改 8，删除 0。不包含已忽略的照片、模型、批次和缓存。

开始时的 54 份版本管理范围内历史报告保持原字节；本轮报告使用新文件名或 field_validation 子目录。

架构、操作和验收见 [FIELD_VALIDATION_REVIEW.md](FIELD_VALIDATION_REVIEW.md)、[FIELD_VALIDATION.md](FIELD_VALIDATION.md)、[FIELD_VALIDATION_RESULTS.md](FIELD_VALIDATION_RESULTS.md)。

## 新增

| 文件 |
| --- |
| `agritinyvision/field_validation/__init__.py` |
| `agritinyvision/field_validation/annotations.py` |
| `agritinyvision/field_validation/capture.py` |
| `agritinyvision/field_validation/evaluation.py` |
| `agritinyvision/field_validation/parity.py` |
| `agritinyvision/field_validation/reporting.py` |
| `agritinyvision/field_validation/split.py` |
| `agritinyvision/field_validation/web.py` |
| `agritinyvision/pc_camera/web_static/field.css` |
| `agritinyvision/pc_camera/web_static/field.html` |
| `agritinyvision/pc_camera/web_static/field.js` |
| `docs/FIELD_CHANGESET.md` |
| `docs/FIELD_VALIDATION.md` |
| `docs/FIELD_VALIDATION_RESULTS.md` |
| `docs/FIELD_VALIDATION_REVIEW.md` |
| `reports/field_changeset.json` |
| `reports/field_validation/CONFUSION_MATRIX.png` |
| `reports/field_validation/DATASET_STATISTICS.json` |
| `reports/field_validation/EXPERIMENT_SUMMARY.md` |
| `reports/field_validation/FIELD_SPLIT.json` |
| `reports/field_validation/INFERENCE_PARITY.json` |
| `reports/field_validation/INFERENCE_PARITY.md` |
| `reports/field_validation/INFERENCE_RESULTS.csv` |
| `reports/field_validation/QUALITY_GATE_REPORT.json` |
| `reports/field_validation/README.md` |
| `reports/field_validation/SCENARIO_COMPARISON.png` |
| `reports/field_validation/SCENARIO_STATISTICS.json` |
| `reports/field_validation/SOURCE_AUDIT.json` |
| `reports/field_validation/SOURCE_MANIFEST.json` |
| `reports/field_validation/THREE_STATE_REPORT.json` |
| `reports/field_validation/UNKNOWN_DISTRIBUTION.png` |
| `reports/ruff_field.txt` |
| `reports/tests_field_hardware.txt` |
| `reports/tests_field_hardware.xml` |
| `reports/tests_field_software.txt` |
| `reports/tests_field_software.xml` |
| `reports/web_field_runtime.json` |
| `scripts/collect_field_report.py` |
| `scripts/download_field_data.py` |
| `scripts/field_validation.py` |
| `scripts/run_field_dataset.py` |
| `tests/test_field_capture.py` |
| `tests/test_field_download.py` |
| `tests/test_field_evaluation.py` |
| `tests/test_field_hardware.py` |
| `tests/test_field_parity.py` |
| `tests/test_field_web.py` |

## 修改

| 文件 |
| --- |
| `README.md` |
| `README.zh-CN.md` |
| `agritinyvision/pc_camera/cli.py` |
| `agritinyvision/pc_camera/inference_controller.py` |
| `agritinyvision/pc_camera/metrics.py` |
| `agritinyvision/pc_camera/web.py` |
| `agritinyvision/pc_camera/web_service.py` |
| `agritinyvision/pc_camera/web_static/index.html` |

## 删除

无。

# v0.3 文件清单

完整功能映射、风险和验收见 [审查报告](V03_REVIEW.md) 与 [验收记录](V03_VALIDATION.md)。

新增 29 个文件；修改 23 个文件；删除 0 个文件。仅列版本管理范围，运行目录、模型二进制及缓存继续忽略。

## 新增

| 文件 |
| --- |
| `agritinyvision/image_contract.py` |
| `agritinyvision/pc_camera/device_identity.py` |
| `agritinyvision/pc_camera/experiment_export.py` |
| `agritinyvision/quantization_analysis.py` |
| `docs/V03_CHANGESET.md` |
| `docs/V03_RELIABILITY.md` |
| `docs/V03_REVIEW.md` |
| `docs/V03_VALIDATION.md` |
| `reports/QUANTIZATION_ANALYSIS_V03.md` |
| `reports/changeset_v03.json` |
| `reports/quantization_analysis_v03.json` |
| `reports/quantization_saturation_v03.png` |
| `reports/ruff_v03.txt` |
| `reports/tests_v03_hardware.txt` |
| `reports/tests_v03_hardware.xml` |
| `reports/tests_v03_software.txt` |
| `reports/tests_v03_software.xml` |
| `reports/web_runtime_v03.json` |
| `scripts/analyze_performance.py` |
| `scripts/export_experiment.py` |
| `scripts/prepare_analysis_scenes.py` |
| `scripts/quantization_analysis.py` |
| `scripts/report_v03_analysis.py` |
| `tests/test_device_hardware.py` |
| `tests/test_device_identity.py` |
| `tests/test_experiment_v03.py` |
| `tests/test_preprocessing_v03.py` |
| `tests/test_quantization_analysis.py` |
| `tests/test_web_security.py` |

## 修改

| 文件 |
| --- |
| `.github/workflows/tests.yml` |
| `README.md` |
| `README.zh-CN.md` |
| `agritinyvision/__init__.py` |
| `agritinyvision/data/transforms.py` |
| `agritinyvision/pc_camera/cli.py` |
| `agritinyvision/pc_camera/deployment.py` |
| `agritinyvision/pc_camera/image_preprocess.py` |
| `agritinyvision/pc_camera/inference_controller.py` |
| `agritinyvision/pc_camera/metrics.py` |
| `agritinyvision/pc_camera/result_parser.py` |
| `agritinyvision/pc_camera/serial_client.py` |
| `agritinyvision/pc_camera/web.py` |
| `agritinyvision/pc_camera/web_service.py` |
| `agritinyvision/quality.py` |
| `agritinyvision/quantize.py` |
| `docs/INTEGRATION_CONTRACT.md` |
| `docs/PC_CAMERA_SETUP.md` |
| `pyproject.toml` |
| `scripts/compare_device_reference.py` |
| `tests/test_firmware_core.py` |
| `tests/test_pc_camera.py` |
| `tests/test_web_camera.py` |

## 删除

无。

import json

import pytest
from test_pc_camera import device, result

from agritinyvision.pc_camera.metrics import RunRecorder
from scripts.analyze_performance import performance_report


def test_environment_stage_units_saturation_and_performance_report(tmp_path):
    recorder = RunRecorder(tmp_path, {"camera": 0, "fps": 1.0}, device(), evidence="synthetic-unit-test")
    for i in range(5):
        r = result(seq=i + 2)
        r.update(pc_capture_ms=5, pc_preprocess_ms=2, transfer_ms=10, serial_roundtrip_ms=12,
                 end_to_end_ms=19, saturated_values=768, actual_fps=1)
        recorder.record(r, i + 1)
    summary = recorder.close(plots=False)
    assert summary["saturation_fraction"]["p50"] == 0.01
    meta = json.loads((recorder.path / "run_metadata.json").read_text())
    assert meta["project"]["project_version"] == "0.3.0"
    assert len(meta["project"]["git_commit"]) == 40
    assert "Flask" in meta["pc"]["dependencies"]
    assert "finished_at" in meta and "started_at" in meta
    perf = performance_report(recorder.path)
    assert perf["serial_minus_device_total_ms"]["p50"] == pytest.approx(11.88)
    assert perf["latency"]["pc_capture_ms"]["p50"] == 5
    assert perf["latency"]["inference_us"]["p50"] == 100
    assert perf["latency"]["end_to_end_ms"]["p50"] == 19
    assert (recorder.path / "PERFORMANCE.md").exists()

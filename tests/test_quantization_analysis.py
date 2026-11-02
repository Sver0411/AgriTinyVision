import json

import numpy as np
import pytest
from PIL import Image

from agritinyvision.quantization_analysis import comparison_summary, probabilities, saturation_analysis
from scripts.prepare_analysis_scenes import prepare_scenes


def test_per_channel_saturation_and_real_rounding():
    white = saturation_analysis(np.full((160, 160, 3), 255, np.uint8), -6)
    assert white["saturated_values"] == 76800
    black = saturation_analysis(np.zeros((160, 160, 3), np.uint8), -6)
    assert black["channels"]["B"]["saturated_values"] == 0
    assert black["channels"]["R"]["saturated_values"] == 25600
    assert black["channels"]["G"]["saturated_values"] == 25600
    assert black["saturated_values"] == 51200


def test_pair_denominators_and_quality_rejection():
    rows = [{"id": "rejected", "scenario": "low_light", "errors": {},
             "saturation": {"saturated_fraction": 0.2},
             "fp32": {"probabilities": [0.9, 0.1], "state": "UNKNOWN"}, "host": None,
             "device": {"probabilities": None, "state": "UNKNOWN"}}]
    s = comparison_summary(rows)
    assert s["fp32_vs_device"]["state_agreement"] == 1
    assert s["fp32_vs_device"]["probability_pairs"] == 0
    assert s["host_vs_device"]["state_agreement"] == "PENDING"
    assert s["three_way_state_agreement"]["count"] == 0


def test_scene_manifest_records_sources_without_field_claims(tmp_path):
    image = tmp_path / "leaf.png"
    Image.fromarray(np.random.default_rng(6).integers(0, 256, (200, 320, 3), dtype=np.uint8)).save(image)
    spec = prepare_scenes(image, tmp_path / "scenes", "synthetic-test-source", "NORMAL", "train")
    assert len(spec["samples"]) == 10
    assert sum(s["annotation_valid"] for s in spec["samples"]) == 1
    assert len({s["scenario"] for s in spec["samples"]}) == 10
    assert all(not s["path"].startswith(str(tmp_path)) for s in spec["samples"])
    assert json.loads((tmp_path / "scenes/manifest.json").read_text())["version"] == "analysis-scenes-v1"


def test_stable_softmax_and_invalid_logits():
    np.testing.assert_allclose(probabilities([10000, 10000]), [0.5, 0.5])
    with pytest.raises(ValueError):
        probabilities([float("nan"), 1])


def test_modified_native_graph_is_rejected_before_loading_pickle(tmp_path):
    from agritinyvision.quantize import load_frozen_executor
    from agritinyvision.utils import sha256

    model = tmp_path / "frozen.espdl"
    model.write_bytes(b"synthetic unit test artifact")
    native = tmp_path / "frozen.espdl.native"
    native.write_bytes(b"not a pickle and must not be loaded")
    with pytest.raises(ValueError, match="simulation graph"):
        load_frozen_executor(model, {"sha256": sha256(model), "simulation_graph_sha256": "f" * 64})


@pytest.mark.simulated_hardware
def test_analysis_executes_and_exports_simulated_transport_without_hardware_claim(tmp_path, monkeypatch):
    from test_pc_camera import FakeSerial, device

    import agritinyvision.quantization_analysis as module
    from agritinyvision.pc_camera.device_identity import IDENTITY_FIELDS
    from agritinyvision.pc_camera.serial_client import SerialClient
    from agritinyvision.utils import sha256

    image = tmp_path / "noise.png"
    Image.fromarray(np.random.default_rng(303).integers(0, 256, (160, 160, 3), dtype=np.uint8)).save(image)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"samples": [{"id": "test", "scenario": "noise", "path": image.name,
                                                "source_sha256": sha256(image)}]}))
    expected = {k: device()[k] for k in IDENTITY_FIELDS}
    monkeypatch.setattr(module, "load_expected_identity", lambda *a: expected)

    class Reference:
        sha256 = "c" * 64

        def __init__(self, bundle):
            pass

        def __call__(self, rgb):
            return [0.9, 0.1]

    monkeypatch.setattr(module, "OnnxReference", Reference)
    client = SerialClient("TEST", serial_factory=FakeSerial, expected_identity=expected)
    result = module.analyze(manifest, tmp_path, tmp_path / "result", client)
    assert result["metadata"]["device_status"] == "MEASURED: simulated_hardware"
    assert result["summary"]["fp32_vs_device"]["state_agreement"] == 0
    assert result["summary"]["fp32_vs_device"]["probability_pairs"] == 1
    folder = tmp_path / "result" / result["metadata"]["device_run"]
    for file in ("run_metadata.json", "frame_results.csv", "device_metrics.csv", "latency_report.json", "summary.json", "errors.log"):
        assert (folder / file).exists()
    assert json.loads((folder / "summary.json").read_text())["hardware_validation"] == "PENDING"
    assert (tmp_path / "result/saturation_comparison.png").exists()


def test_analysis_source_hash_rejection_precedes_backend(tmp_path):
    import agritinyvision.quantization_analysis as module
    from agritinyvision.quantization_analysis import analyze

    image = tmp_path / "input.png"
    image.write_bytes(b"changed source")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"samples": [{"id": "tampered", "path": image.name,
                                                "source_sha256": "a" * 64}]}))
    original = module.load_expected_identity
    try:
        module.load_expected_identity = lambda *a: {}
        with pytest.raises(ValueError, match="SHA256 mismatch"):
            analyze(manifest, tmp_path, tmp_path / "result")
    finally:
        module.load_expected_identity = original
    assert not (tmp_path / "result").exists()


def test_fp32_and_host_receive_identical_normalized_tensor():
    import torch

    from agritinyvision.quantization_analysis import FrozenHostReference, OnnxReference

    inputs = {}

    class Session:
        def get_inputs(self):
            return [type("Input", (), {"name": "image"})()]

        def run(self, _, feeds):
            inputs["fp32"] = feeds["image"]
            return [np.array([[0.75, -0.25]], np.float32)]

    class Executor:
        def forward(self, x):
            inputs["host"] = x.numpy()
            return [torch.tensor([[0.75, -0.25]])]

    fp32 = OnnxReference.__new__(OnnxReference)
    fp32.session = Session()
    host = FrozenHostReference.__new__(FrozenHostReference)
    host.executor = Executor()
    rgb = np.random.default_rng(22).integers(0, 256, (160, 160, 3), dtype=np.uint8)
    assert fp32(rgb) == host(rgb)
    assert inputs["fp32"].shape == (1, 3, 160, 160)
    np.testing.assert_array_equal(inputs["fp32"], inputs["host"])

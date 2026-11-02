import numpy as np
import pytest
from test_field_evaluation import make_batch
from test_pc_camera import device

from agritinyvision.field_validation.capture import read_json
from agritinyvision.field_validation.parity import compare_batch


def test_saved_rgb_integrity_and_shared_frozen_backend(tmp_path, monkeypatch):
    import agritinyvision.field_validation.parity as module

    batch, sid = make_batch(tmp_path, 7, "one")
    monkeypatch.setattr(module, "load_expected_identity", lambda *a: device())

    class Reference:
        sha256 = "c" * 64

        def __init__(self, bundle):
            pass

        def __call__(self, rgb):
            assert rgb.shape == (160, 160, 3) and rgb.dtype == np.uint8
            return [.8, .2]

    monkeypatch.setattr(module, "OnnxReference", Reference)
    data = compare_batch(batch.path, tmp_path)
    assert data["host"].startswith("PENDING")
    assert data["samples"][0]["fp32"]["state"] == "UNKNOWN"
    assert data["summary"]["fp32_vs_device"]["state_agreement"] == 0
    assert (batch.path / "reports/INFERENCE_PARITY.md").exists()
    meta = read_json(batch.path / f"metadata/{sid}.json")
    assert meta["rgb_sha256"] == data["samples"][0]["rgb_sha256"]
    (batch.path / f"images/model_input/{sid}.png").write_bytes(b"changed file")
    with pytest.raises((ValueError, OSError)):
        compare_batch(batch.path, tmp_path)

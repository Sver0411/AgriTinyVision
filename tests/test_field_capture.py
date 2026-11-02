import hashlib
import json

import numpy as np
import pytest
from PIL import Image
from test_pc_camera import device, result
from test_web_camera import service  # noqa: F401

from agritinyvision.field_validation.annotations import annotation_csv, labels, save_annotation
from agritinyvision.field_validation.capture import FieldBatch, read_json, read_samples, verified_sample
from agritinyvision.pc_camera.image_preprocess import prepare_bgr


@pytest.fixture
def batch(tmp_path):
    return FieldBatch(tmp_path, device(), {"camera": 0}, {"plant_id": "plant-1"}, "simulated_hardware")


def sample(batch):
    bgr = np.random.default_rng(13).integers(0, 256, (240, 320, 3), dtype=np.uint8)
    rgb = prepare_bgr(bgr, 90, True)
    sid = batch.begin(bgr, rgb, {"rotation": 90, "mirror": True})
    return sid, bgr, rgb


def test_lossless_original_and_actual_input_are_bound(batch):
    sid, bgr, rgb = sample(batch)
    batch.complete(sid, result())
    s = read_samples(batch.path)[0]
    meta = read_json(batch.path / s["metadata"])
    np.testing.assert_array_equal(np.asarray(Image.open(batch.path / s["original"])), bgr[..., ::-1])
    actual = np.asarray(Image.open(batch.path / s["model_input"]))
    np.testing.assert_array_equal(actual, rgb)
    assert meta["rgb_sha256"] == hashlib.sha256(actual.tobytes()).hexdigest()
    assert meta["original_size"] == [320, 240] and batch.completed == 1
    assert read_json(batch.path / s["prediction"])["result"]["state"] == "SUSPECT"
    _, verified = verified_sample(batch.path, s)
    np.testing.assert_array_equal(verified, rgb)


def test_changed_original_or_direction_breaks_pair_integrity(batch):
    sid, _, _ = sample(batch)
    s = read_samples(batch.path)[0]
    path = batch.path / s["metadata"]
    metadata = read_json(path)
    metadata["camera_config"]["mirror"] = False
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="reproduce"):
        verified_sample(batch.path, s)


def test_transport_error_remains_error_not_unknown(batch):
    sid, _, _ = sample(batch)
    batch.error(sid, OSError("serial unplugged"))
    p = read_json(batch.path / f"predictions/{sid}.json")
    assert p == {"status": "error", "error": "serial unplugged"}
    assert batch.completed == 0


def annotation():
    return {"plant_presence": "YES", "image_quality": "USABLE", "visual_label": "NORMAL",
            "judgment_certain": True, "expert_confirmed": False, "annotator": "independent-reviewer"}


def test_labels_save_progress_and_never_copy_prediction(batch):
    sid, _, _ = sample(batch)
    batch.complete(sid, result(p=0.9))
    assert labels(batch.path) == {}
    save_annotation(batch.path, sid, annotation())
    assert labels(batch.path)[sid]["visual_label"] == "NORMAL"
    revised = {**annotation(), "visual_label": "UNCERTAIN", "judgment_certain": False}
    save_annotation(batch.path, sid, revised)
    assert len(labels(batch.path)) == 1
    assert "UNCERTAIN" in annotation_csv(batch.path)
    assert read_json(batch.path / f"predictions/{sid}.json")["result"]["state"] == "SUSPECT"


def test_disease_label_requires_evidence(batch):
    sid, _, _ = sample(batch)
    with pytest.raises(ValueError, match="依据"):
        save_annotation(batch.path, sid, {**annotation(), "disease_label": "disease"})


def test_collection_is_opt_in_and_interval_can_pause_resume(service, tmp_path):  # noqa: F811
    service.config["field_output"] = str(tmp_path / "field")
    service._execute("connect_camera", {})
    service._tick()
    service._execute("connect_device", {})
    service._execute("single", {})
    service.controller.worker.join(3)
    service._tick()
    assert not (tmp_path / "field").exists()
    service._execute("field_create", {"context": {"plant_id": "p1"}, "interval_s": 5})
    service._execute("field_capture", {})
    service.controller.worker.join(3)
    service._tick()
    assert service.snapshot()["field"]["completed"] == 1
    service._execute("field_start", {})
    assert service.field_running
    service._execute("field_pause", {})
    assert not service.field_running
    service._execute("field_start", {})
    assert service.field_running
    service._execute("field_pause", {})
    assert len(read_samples(service.field_batch.path)) == 1

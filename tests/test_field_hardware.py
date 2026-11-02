"""Physical-device capture pipeline using a generated probe, not an agricultural validation image."""

import hashlib
import os
import queue

import numpy as np
import pytest
from PIL import Image

from agritinyvision.field_validation.capture import FieldBatch, read_json, read_samples, verified_sample
from agritinyvision.pc_camera.inference_controller import InferenceController
from agritinyvision.pc_camera.metrics import RunRecorder
from agritinyvision.pc_camera.serial_client import SerialClient

pytestmark = pytest.mark.hardware


def test_physical_capture_pipeline_saves_exact_transmitted_rgb(tmp_path):
    port = os.environ.get("ATV_HARDWARE_PORT")
    if not port:
        pytest.fail("Set ATV_HARDWARE_PORT to select the physical board")
    client = SerialClient(port)
    recorder = None
    try:
        d = client.connect()
        batch = FieldBatch(tmp_path / "batches", d, {}, {"source_kind": "generated_hardware_probe"})
        recorder = RunRecorder(tmp_path / "runs", {}, d)
        controller = InferenceController(client, recorder, rotation=90, mirror=True)
        bgr = np.random.default_rng(59).integers(0, 256, (240, 320, 3), dtype=np.uint8)
        assert controller.submit(bgr, field_batch=batch)
        controller.worker.join(30)
        try:
            kind, value = controller.events.get_nowait()
        except queue.Empty:
            pytest.fail("Physical pipeline did not return within 30 seconds")
        assert kind == "result", value
        sample = read_samples(batch.path)[0]
        with Image.open(batch.path / sample["model_input"]) as image:
            rgb_hash = hashlib.sha256(image.tobytes()).hexdigest()
        prediction = read_json(batch.path / sample["prediction"])
        metadata, _ = verified_sample(batch.path, sample)
        assert metadata["camera_config"]["rotation"] == 90 and metadata["camera_config"]["mirror"] is True
        assert rgb_hash == value["rgb_sha256"] == prediction["result"]["rgb_sha256"]
        assert batch.completed == 1 and prediction["status"] == "ok"
        batch.finish()
    finally:
        client.close()
        if recorder:
            recorder.close(plots=False)

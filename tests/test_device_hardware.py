"""Opt-in tests sending real frames to the selected physical board; never run in regular CI."""

import os

import numpy as np
import pytest

from agritinyvision.pc_camera.device_identity import load_expected_identity
from agritinyvision.pc_camera.protocol import ProtocolError
from agritinyvision.pc_camera.serial_client import SerialClient

pytestmark = pytest.mark.hardware


@pytest.fixture
def port():
    selected = os.environ.get("ATV_HARDWARE_PORT")
    if not selected:
        pytest.fail("Set ATV_HARDWARE_PORT to explicitly select the physical ESP32")
    return selected


def test_actual_board_identity_and_quality_skips_model(port):
    client = SerialClient(port)
    try:
        d = client.connect()
        assert d["identity_validation"]["status"] == "PASS"
        for value in (0, 128, 255):
            r = client.infer(np.full((160, 160, 3), value, np.uint8).tobytes())
            assert r["state"] == "UNKNOWN" and not r["quality_passed"]
            assert r["inference_us"] is None and r["preprocess_us"] is None
            assert r["suspect_probability"] is None
    finally:
        client.close()


def test_actual_wrong_expected_model_refused(port):
    expected = load_expected_identity()
    expected["model_sha256"] = "f" * 64
    client = SerialClient(port, expected_identity=expected)
    try:
        with pytest.raises(ProtocolError, match="model_sha256"):
            client.connect()
        assert client.serial is None and client.sequence == 1
    finally:
        client.close()

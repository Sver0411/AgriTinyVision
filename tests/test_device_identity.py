import copy
import json

import pytest
from test_pc_camera import FakeSerial, device, result

from agritinyvision.pc_camera.device_identity import load_expected_identity, verify_device_identity
from agritinyvision.pc_camera.protocol import ProtocolError
from agritinyvision.pc_camera.result_parser import parse_result
from agritinyvision.pc_camera.serial_client import DeviceError, SerialClient


@pytest.mark.parametrize("key,value", [
    ("model_sha256", "f" * 64), ("deployment_sha256", "e" * 64), ("model_version", "0.9.0"),
    ("input_shape", [1, 3, 160, 160]), ("output_shape", [2, 1]), ("dtype", "float32"),
    ("input_exponent", -5), ("output_exponent", -3), ("decision_threshold", 0.5),
    ("normal_upper_bound", 0.19), ("suspect_lower_bound", 0.41), ("quality_gate_version", "other"),
])
def test_real_field_comparison_rejects_wrong_identity(key, value):
    expected = device()
    wrong = copy.deepcopy(expected)
    wrong[key] = value
    with pytest.raises(ProtocolError, match=key):
        verify_device_identity(wrong, expected)


def test_missing_v02_identity_field_requires_compatible_firmware():
    older = device()
    del older["deployment_sha256"]
    with pytest.raises(ProtocolError, match="deployment_sha256"):
        verify_device_identity(older, device())
    assert verify_device_identity(device(), device())["status"] == "PASS"


@pytest.mark.simulated_hardware
def test_wrong_model_closes_transport_before_frame():
    expected = device()
    expected["model_sha256"] = "f" * 64
    transport = FakeSerial()
    client = SerialClient("TEST", serial_factory=lambda: transport, expected_identity=expected)
    with pytest.raises(ProtocolError, match="model_sha256"):
        client.connect()
    assert not transport.is_open and not transport.seen
    with pytest.raises(DeviceError):
        client.infer(bytes(76800))


@pytest.mark.local_artifact
def test_frozen_bundle_identity_and_metadata_tamper(tmp_path):
    from agritinyvision.pc_camera.deployment import deployment_contract

    expected = load_expected_identity(deployment="firmware_example/main/camera_deployment.json")
    assert expected["model_sha256"] == "bfda065ad95c3c3458d3c135b4c4b4575bdad6a31d2445b9283065d3851af3fa"
    config = deployment_contract("releases/v0.1.0")
    config["tensors"]["input"]["exponent"] = -5
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="元数据"):
        load_expected_identity(deployment=path)


def test_quality_rejection_cannot_claim_model_execution():
    r = result(quality=False)
    r["inference_us"] = 100
    with pytest.raises(ProtocolError, match="skip"):
        parse_result(json.dumps(r).encode(), 2, device())


def test_bundle_file_is_hashed_instead_of_trusting_declared_digest(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "tinyvision.espdl").write_bytes(b"modified model bytes")
    (bundle / "metadata.json").write_text(json.dumps({"artifact_sha256": {"tinyvision.espdl": "a" * 64}}))
    with pytest.raises(ValueError, match="SHA256 disagrees"):
        load_expected_identity(bundle)

import json
import zlib

import numpy as np
import pytest
from PIL import Image

from agritinyvision.data.transforms import preprocessing
from agritinyvision.pc_camera.camera import Camera
from agritinyvision.pc_camera.image_preprocess import prepare_bgr
from agritinyvision.pc_camera.inference_controller import InferenceController
from agritinyvision.pc_camera.metrics import RunRecorder
from agritinyvision.pc_camera.protocol import (
    FRAME_BYTES,
    HEADER,
    Decoder,
    MessageType,
    Packet,
    ProtocolError,
    frame_packet,
    json_packet,
)
from agritinyvision.pc_camera.result_parser import parse_device, parse_result
from agritinyvision.pc_camera.serial_client import DeviceError, SerialClient


def device():
    return {
        "chip": "ESP32-S3",
        "boot_id": "test-boot",
        "firmware_version": "0.2.0",
        "model_version": "test-only",
        "model_sha256": "a" * 64,
        "idf_version": "5.4.4",
        "esp_dl_version": "3.3.13",
        "input_shape": [1, 160, 160, 3],
        "output_shape": [1, 2],
        "input_exponent": -6,
        "output_exponent": -4,
        "dtype": "int8",
        "model_loaded": True,
        "decision_threshold": 0.3,
        "normal_upper_bound": float(np.float32(0.2)),
        "suspect_lower_bound": float(np.float32(0.4)),
        "transport": "synthetic test transport",
        "deployment_sha256": "b" * 64,
        "abstention_policy": "threshold-centered-v1",
        "quality_gate_version": "laplacian-gray-v1",
    }


def result(seq=2, p=0.9, quality=True):
    state = "UNKNOWN" if not quality or 0.2 <= p < 0.4 else "NORMAL" if p < 0.2 else "SUSPECT"
    return {
        "sequence": seq,
        "boot_id": "test-boot",
        "firmware_version": "0.2.0",
        "model_version": "test-only",
        "state": state,
        "reason": "test",
        "normal_probability": 1 - p if quality else None,
        "suspect_probability": p if quality else None,
        "confidence": None if state == "UNKNOWN" else p if state == "SUSPECT" else 1 - p,
        "quality_passed": quality,
        "decision_threshold": 0.3,
        "quality_us": 8,
        "preprocess_us": 10 if quality else None,
        "inference_us": 100 if quality else None,
        "total_us": 120,
        "free_internal": 20000,
        "free_psram": 2000000,
        "min_free_internal": 19000,
        "min_free_psram": 1900000,
    }


def test_wire_crc_fragmentation_and_boot_logs():
    assert zlib.crc32(b"123456789") == 0xCBF43926
    packet = frame_packet(0x12345678, bytes(i % 256 for i in range(FRAME_BYTES)))
    data = b"ESP-ROM startup\n" + packet.encode()
    d = Decoder()
    packets = []
    for i in range(0, len(data), 37):
        packets.extend(d.feed(data[i : i + 37]))
    assert packets == [packet]
    assert len(packet.encode()) == 76832
    assert packet.encode()[8:12] == b"\x78\x56\x34\x12"


@pytest.mark.parametrize(
    "field,value", [("width", 159), ("height", 161), ("pixel_format", 0), ("payload", b"x"), ("sequence", 0)]
)
def test_invalid_frame_fields(field, value):
    kwargs = dict(
        kind=MessageType.FRAME, sequence=1, width=160, height=160, pixel_format=1, payload=bytes(FRAME_BYTES)
    )
    kwargs[field] = value
    with pytest.raises(ProtocolError):
        Packet(**kwargs).encode()


@pytest.mark.parametrize("offset", [4, 8, 20, 28, 40])
def test_corrupt_frame_recovers_to_next_header(offset):
    broken = bytearray(frame_packet(1, bytes(FRAME_BYTES)).encode())
    broken[offset] ^= 128
    good = Packet(MessageType.PING, 2)
    decoder = Decoder()
    assert decoder.feed(broken + good.encode()) == [good]
    assert decoder.errors


def test_large_noise_and_untrusted_length_stay_bounded():
    d = Decoder()
    h = bytearray(frame_packet(1, bytes(FRAME_BYTES)).encode()[:32])
    h[20:24] = (0xFFFFFFFF).to_bytes(4, "little")
    h[28:32] = zlib.crc32(h[:28]).to_bytes(4, "little")
    assert d.feed(h + b"x" * 1000000 + Packet(MessageType.PING, 3).encode()) == [Packet(MessageType.PING, 3)]
    assert len(d.buffer) <= 3


def test_partial_frame_timeout_and_recovery():
    clock = [0]
    d = Decoder(timeout=2, clock=lambda: clock[0])
    d.feed(frame_packet(1, bytes(FRAME_BYTES)).encode()[:200])
    clock[0] = 3
    assert d.feed(Packet(MessageType.PING, 2).encode()) == [Packet(MessageType.PING, 2)]
    assert d.errors == ["packet_timeout"]


@pytest.mark.parametrize("p,quality", [(0.1, True), (0.3, True), (0.9, True), (0.9, False)])
def test_result_unknown_and_confidence(p, quality):
    r = result(p=p, quality=quality)
    assert parse_result(json.dumps(r).encode(), 2, device()) == r


@pytest.mark.parametrize(
    "key,value",
    [
        ("sequence", 3),
        ("boot_id", "reboot"),
        ("state", "BAD"),
        ("confidence", 0.1),
        ("normal_probability", 0.7),
        ("inference_us", -1),
        ("free_internal", float("nan")),
        ("decision_threshold", 0.5),
    ],
)
def test_invalid_device_result(key, value):
    r = result()
    r[key] = value
    with pytest.raises(ProtocolError):
        parse_result(json.dumps(r).encode(), 2, device())


def test_device_handshake_contract():
    d = device()
    assert parse_device(json.dumps(d).encode()) == d
    d["input_shape"] = [1, 3, 160, 160]
    with pytest.raises(ProtocolError):
        parse_device(json.dumps(d).encode())


class FakeSerial:
    def __init__(self, fault=None):
        self.is_open = False
        self.buffer = bytearray()
        self.fault = fault
        self.seen = set()

    def open(self):
        self.is_open = True

    def close(self):
        self.is_open = False

    def reset_input_buffer(self):
        self.buffer.clear()

    def flush(self):
        pass

    def write(self, data):
        p = Decoder().feed(data)[0]
        if p.kind == MessageType.HELLO:
            self.buffer.extend(json_packet(MessageType.HELLO_ACK, p.sequence, device()).encode())
        elif p.kind == MessageType.FRAME:
            if self.fault == "disconnect":
                self.close()
                raise OSError("Serial unplugged")
            if self.fault == "timeout":
                return len(data)
            seq = p.sequence + (self.fault == "sequence")
            if seq in self.seen:
                self.buffer.extend(
                    json_packet(
                        MessageType.ERROR, seq, {"code": "duplicate", "message": "Already consumed"}
                    ).encode()
                )
            else:
                self.seen.add(seq)
                self.buffer.extend(Packet(MessageType.FRAME_ACK, seq).encode())
                self.buffer.extend(json_packet(MessageType.INFER_RESULT, seq, result(seq)).encode())
        return len(data)

    def read(self, size):
        data = bytes(self.buffer[:size])
        del self.buffer[:size]
        return data


def test_serial_single_request_and_ack_timings():
    client = SerialClient("TEST", serial_factory=FakeSerial, expected_identity=device())
    client.connect()
    r = client.infer(bytes(FRAME_BYTES))
    assert r["sequence"] == 2 and r["state"] == "SUSPECT"
    assert r["end_to_end_ms"] >= r["transfer_ms"] >= 0
    assert r["inference_us"] == 100  # Never overwrite MCU timing with PC elapsed time.
    client.close()


def test_model_not_loaded_refuses_frames(monkeypatch):
    original = device()
    original.update(model_loaded=False, model_error="rgb_psram_allocation")
    monkeypatch.setitem(globals(), "device", lambda: original)
    client = SerialClient("TEST", serial_factory=FakeSerial, expected_identity=device())
    with pytest.raises(DeviceError, match="rgb_psram_allocation"):
        client.connect()
    assert client.serial is None and client.device is None


@pytest.mark.parametrize(
    "fault,exception", [("sequence", ProtocolError), ("timeout", TimeoutError), ("disconnect", OSError)]
)
def test_serial_fault_requires_new_handshake(fault, exception):
    client = SerialClient("TEST", timeout=0.03, serial_factory=lambda: FakeSerial(fault), expected_identity=device())
    client.connect()
    with pytest.raises(exception):
        client.infer(bytes(FRAME_BYTES))
    assert client.device is None and client.serial is None
    with pytest.raises(DeviceError):
        client.infer(bytes(FRAME_BYTES))


def test_duplicate_sequence_never_generates_another_result():
    client = SerialClient("TEST", serial_factory=FakeSerial, expected_identity=device())
    client.connect()
    client.infer(bytes(FRAME_BYTES))
    client.sequence -= 1
    with pytest.raises(DeviceError, match="duplicate"):
        client.infer(bytes(FRAME_BYTES))


def test_camera_bgr_resize_matches_training_contract():
    rng = np.random.default_rng(91)
    frame = rng.integers(0, 256, (245, 321, 3), dtype=np.uint8)
    rgb = prepare_bgr(frame)
    x = preprocessing(160)(Image.fromarray(frame[:, :, ::-1])).numpy()
    normalized = (rgb.astype(np.float32) / 255 - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
        [0.229, 0.224, 0.225], np.float32
    )
    np.testing.assert_allclose(normalized.transpose(2, 0, 1), x, rtol=0, atol=0)
    assert rgb.shape == (160, 160, 3) and rgb.dtype == np.uint8


def test_orientation_and_invalid_camera_input():
    a = np.zeros((160, 160, 3), dtype=np.uint8)
    a[:80, :80, 2] = 255
    assert prepare_bgr(a, rotation=90)[0, -1, 0] == 255
    assert prepare_bgr(a, mirror=True)[0, -1, 0] == 255
    with pytest.raises(ValueError):
        prepare_bgr(np.zeros((10, 10, 3), dtype=np.uint8))


def test_camera_disconnect_releases_capture():
    class Capture:
        released = False

        def isOpened(self):
            return not self.released

        def read(self):
            return False, None

        def release(self):
            self.released = True

    c = Capture()
    camera = Camera(capture_factory=lambda index: c)
    camera.connect()
    with pytest.raises(RuntimeError, match="capture failed"):
        camera.read()
    assert c.released and not camera.connected


def test_metrics_real_schema_and_pending_report(tmp_path):
    recorder = RunRecorder(tmp_path, {"fps": 1}, device(), evidence="synthetic-unit-test")
    recorder.record({**result(), "transfer_ms": 5, "end_to_end_ms": 20, "actual_fps": 1}, 1)
    recorder.error("CRC error", 3)
    summary = recorder.close(plots=False)
    assert summary["frames"] == 1 and summary["attempts"] == 2 and summary["success_rate"] == 0.5
    assert summary["hardware_validation"] == "PENDING"
    assert (recorder.path / "device_metrics.csv").read_text().count("\n") == 2
    assert json.loads((recorder.path / "latency_report.json").read_text())["inference_us"]["p50"] == 100
    assert "PENDING" in (recorder.path / "HARDWARE_REPORT.md").read_text(encoding="utf-8")


def test_controller_uses_only_one_inflight_frame(tmp_path):
    client = SerialClient("TEST", serial_factory=FakeSerial, expected_identity=device())
    client.connect()
    recorder = RunRecorder(tmp_path, evidence="synthetic-unit-test")
    controller = InferenceController(client, recorder)
    assert controller.submit(np.zeros((160, 160, 3), np.uint8))
    controller.worker.join(3)
    assert not controller.busy
    assert not controller.submit(np.zeros((160, 160, 3), np.uint8))  # Result must be consumed first.
    kind, value = controller.events.get(timeout=1)
    assert kind == "result" and value["sequence"] == 2
    recorder.close(plots=False)
    client.close()


def test_camera_runtime_never_imports_reference_inference():
    import subprocess
    import sys

    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import agritinyvision.pc_camera.cli; "
            "assert 'torch' not in sys.modules; assert 'onnxruntime' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert process.returncode == 0, process.stderr


def test_header_size():
    assert HEADER.size == 32

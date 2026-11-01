"""Run quality/parity and malformed-packet checks against an actual ESP32."""

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from agritinyvision.pc_camera.image_preprocess import load_image_bgr, prepare_bgr
from agritinyvision.pc_camera.metrics import write_json
from agritinyvision.pc_camera.protocol import MessageType, frame_packet
from agritinyvision.pc_camera.serial_client import DeviceError, SerialClient
from agritinyvision.quality import load_quality, quality_gate


def validate(port, image, output):
    client = SerialClient(port)
    report = {"evidence": "hardware", "port": port, "tests": []}
    try:
        report["device"] = client.connect()
        bundle = json.loads(Path("releases/v0.1.0/metadata.json").read_text(encoding="utf-8"))
        assert report["device"]["model_sha256"] == bundle["artifact_sha256"]["tinyvision.espdl"]
        client.ping()
        report["tests"].append({"name": "handshake_hash_tensors_ping", "status": "PASS"})
        yy, xx = np.indices((160, 160))
        patterns = {
            "black": np.zeros((160, 160, 3), np.uint8),
            "white": np.full((160, 160, 3), 255, np.uint8),
            "flat": np.full((160, 160, 3), 128, np.uint8),
            "gradient": np.repeat((xx * 255 / 159).astype(np.uint8)[:, :, None], 3, axis=2),
            "checker": np.repeat((((xx + yy) % 2) * 255).astype(np.uint8)[:, :, None], 3, axis=2),
            "noise": np.random.default_rng(51).integers(0, 256, (160, 160, 3), dtype=np.uint8),
            "train_fixture": prepare_bgr(load_image_bgr(image)),
        }
        for name, rgb in patterns.items():
            result = client.infer(rgb.tobytes())
            expected = quality_gate(Image.fromarray(rgb), load_quality())
            assert result["quality_passed"] == expected["passed"], name
            for key, value in expected["metrics"].items():
                assert np.isclose(result[key], value, rtol=2e-6, atol=0.02), (name, key, result[key], value)
            if not expected["passed"]:
                assert result["state"] == "UNKNOWN" and result["inference_us"] is None
                assert result["suspect_probability"] is None and result["confidence"] is None
            else:
                normalized = (
                    rgb.astype(np.float32) / 255 - np.array([0.485, 0.456, 0.406], np.float32)
                ) / np.array([0.229, 0.224, 0.225], np.float32)
                raw = np.floor(normalized * 2.0 ** -report["device"]["input_exponent"] + 0.5)
                saturated = int(np.count_nonzero((raw < -128) | (raw > 127)))
                assert result["saturated_values"] == saturated
            report["tests"].append(
                {"name": "quality_" + name, "status": "PASS", "result": result, "python_quality": expected}
            )
            print(f"PASS quality_{name}: {result['state']}", flush=True)

        rgb = patterns["train_fixture"].tobytes()
        fixture_p = next(
            t["result"]["suspect_probability"]
            for t in report["tests"]
            if t["name"] == "quality_train_fixture"
        )
        fixture = json.loads(Path("reports/hardware_fixture_v02.json").read_text())
        assert abs(fixture_p - fixture["suspect_probability_logged"]) < 1e-6
        report["tests"].append({"name": "retained_fixture_output_parity", "status": "PASS"})

        for name in ("duplicate", "payload_crc", "dimensions", "oversized", "partial_timeout"):
            client.connect()
            if name == "duplicate":
                client.infer(rgb)
                seq = client.sequence
            else:
                seq = client._next()
            packet = bytearray(frame_packet(seq, rgb).encode())
            code = {
                "duplicate": "duplicate_or_old_sequence",
                "payload_crc": "payload_crc",
                "dimensions": "frame_format",
                "oversized": "frame_format",
                "partial_timeout": "timeout",
            }[name]
            if name == "payload_crc":
                packet[-1] ^= 1
            elif name in ("dimensions", "oversized"):
                if name == "dimensions":
                    packet[12:14] = (159).to_bytes(2, "little")
                else:
                    packet[20:24] = (0xFFFFFFFF).to_bytes(4, "little")
                packet[28:32] = zlib.crc32(packet[:28]).to_bytes(4, "little")
                packet = packet[:32]
            elif name == "partial_timeout":
                packet = packet[:132]
            client._write(packet)
            try:
                client._receive(seq, MessageType.INFER_RESULT, time.perf_counter() + 20)
                raise AssertionError(f"{name}: invalid frame accepted")
            except DeviceError as exc:
                assert f"ESP32 {code}:" in str(exc), str(exc)
                report["tests"].append({"name": name, "status": "PASS", "device_error": str(exc)})
            print(f"PASS {name}", flush=True)
        client.connect()
        result = client.infer(rgb)
        report["tests"].append({"name": "reconnect_recovery", "status": "PASS", "result": result})
        report["status"] = "PASS"
        return report
    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = str(exc)
        raise
    finally:
        client.close()
        write_json(output, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="reports/hardware_protocol_v02.json")
    args = parser.parse_args()
    validate(args.port, args.image, args.output)

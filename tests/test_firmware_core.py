"""Execute the same C++ parser/quality/normalization code as the MCU, without ESP hardware."""

import os
import shutil
import subprocess
import zlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from agritinyvision.pc_camera.protocol import FRAME_BYTES, MessageType, Packet, frame_packet
from agritinyvision.quality import load_quality, quality_gate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def core_executable(tmp_path_factory):
    compiler = os.environ.get("ATV_HOST_CXX") or shutil.which("g++") or shutil.which("clang++")
    zig = ROOT / "tools/host-compiler/ziglang/zig.exe"
    command = [compiler] if compiler else [str(zig), "c++"] if zig.exists() else None
    if command is None:
        pytest.skip("Install g++/clang++ or set ATV_HOST_CXX for native firmware core tests")
    output = tmp_path_factory.mktemp("native") / ("camera-core.exe" if os.name == "nt" else "camera-core")
    subprocess.run(
        [
            *command,
            "-std=c++17",
            "-O2",
            "-ffp-contract=off",
            "-I",
            str(ROOT / "firmware_example/main"),
            str(ROOT / "tests/firmware_core_harness.cpp"),
            "-o",
            str(output),
        ],
        check=True,
        timeout=180,
    )
    return output


def run_core(executable, tmp_path, mode, data):
    path = tmp_path / "input.bin"
    path.write_bytes(data)
    return subprocess.check_output([str(executable), mode, str(path)], text=True, timeout=10)


@pytest.mark.parametrize("pattern", ["black", "white", "flat", "gradient", "checker", "noise"])
def test_cpp_quality_matches_python(core_executable, tmp_path, pattern):
    yy, xx = np.indices((160, 160))
    if pattern == "noise":
        rgb = np.random.default_rng(51).integers(0, 256, (160, 160, 3), dtype=np.uint8)
    else:
        a = {
            "black": np.zeros((160, 160)),
            "white": np.full((160, 160), 255),
            "flat": np.full((160, 160), 128),
            "gradient": xx * 255 / 159,
            "checker": ((xx + yy) % 2) * 255,
        }[pattern]
        rgb = np.repeat(a.astype(np.uint8)[:, :, None], 3, axis=2)
    expected = quality_gate(Image.fromarray(rgb), load_quality())
    raw = run_core(core_executable, tmp_path, "quality", rgb.tobytes()).split()
    actual = list(map(float, raw[:3]))
    np.testing.assert_allclose(actual, list(expected["metrics"].values()), rtol=2e-6, atol=0.02)
    assert bool(int(raw[4])) == expected["passed"]


def test_cpp_normalization_and_saturation(core_executable, tmp_path):
    rgb = np.repeat(np.arange(256, dtype=np.uint8), 3).reshape(256, 3)
    output = run_core(core_executable, tmp_path, "normalize", rgb.tobytes())
    actual = np.fromstring(output, sep="\n").reshape(256, 3)
    expected = (rgb.astype(np.float32) / 255 - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
        [0.229, 0.224, 0.225], np.float32
    )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=6e-9)
    # S3 rounds half up; exponent is verified from the frozen binary, not chosen to avoid saturation.
    raw = np.floor(expected * 64 + 0.5)
    assert np.count_nonzero((raw < -128) | (raw > 127)) > 0


def test_cpp_protocol_cross_language_and_bounds(core_executable, tmp_path):
    frame = frame_packet(2, bytes(i % 256 for i in range(FRAME_BYTES))).encode()
    corrupt = bytearray(frame)
    corrupt[-1] ^= 1
    noise = b"boot log\n" + b"ATV2" * 100
    data = noise + bytes(corrupt) + Packet(MessageType.HELLO, 3).encode() + frame
    output = run_core(core_executable, tmp_path, "protocol", data)
    assert output.count("packet") == 2
    assert "packet 1 3 0" in output and "packet 3 2 76800" in output
    assert "error 4" in output  # BAD_CRC
    partial = run_core(core_executable, tmp_path, "protocol", frame[:100])
    assert partial.strip() == "timeout"


@pytest.mark.parametrize("offset,value,size", [(12, 159, 2), (16, 2, 1), (20, 0xFFFFFFFF, 4)])
def test_cpp_untrusted_header_never_writes_out_of_bounds(core_executable, tmp_path, offset, value, size):
    header = bytearray(frame_packet(2, bytes(FRAME_BYTES)).encode()[:32])
    header[offset:offset + size] = value.to_bytes(size, "little")
    header[28:32] = zlib.crc32(header[:28]).to_bytes(4, "little")
    output = run_core(core_executable, tmp_path, "protocol", header + Packet(MessageType.HELLO, 3).encode())
    assert "error 3" in output and "packet 1 3 0" in output


def test_actual_local_model_inspection(tmp_path):
    from agritinyvision.pc_camera.deployment import prepare

    bundle = ROOT / "releases/v0.1.0"
    if not (bundle / "tinyvision.espdl").exists():
        pytest.skip("Local release artifacts are supplied outside Git")
    config = prepare(bundle, tmp_path, ROOT / "configs/quality.yaml")
    assert config["tensors"]["input"] == {"shape": [1, 160, 160, 3], "dtype": 3, "exponent": -6}
    assert config["tensors"]["output"] == {"shape": [1, 2], "dtype": 3, "exponent": -4}
    assert config["decision_threshold"] == 0.3
    assert "min_confidence" not in config["quality_gate"]

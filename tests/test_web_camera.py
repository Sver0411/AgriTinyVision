import io
import json
import queue
import re
import zipfile

import numpy as np
import pytest
from test_pc_camera import FakeSerial

from agritinyvision.pc_camera.camera import Camera
from agritinyvision.pc_camera.metrics import RunRecorder
from agritinyvision.pc_camera.serial_client import SerialClient
from agritinyvision.pc_camera.web_service import CameraService


class TestRecorder(RunRecorder):
    __test__ = False

    def __init__(self, *args):
        super().__init__(*args, evidence="synthetic-web-unit-test")

    def close(self, plots=False):
        return super().close(plots=False)


class Capture:
    def __init__(self, index):
        self.open = True

    def isOpened(self):
        return self.open

    def read(self):
        return True, np.zeros((160, 160, 3), dtype=np.uint8)

    def release(self):
        self.open = False


@pytest.fixture
def service(tmp_path):
    config = {
        "camera": 0,
        "port": "TEST",
        "baudrate": 921600,
        "fps": 1.0,
        "rotation": 0,
        "mirror": False,
        "timeout": 0.1,
        "output": str(tmp_path),
    }
    s = CameraService(
        config, Camera(capture_factory=Capture), SerialClient("TEST", serial_factory=FakeSerial), TestRecorder
    )
    yield s
    s.close()


def test_web_service_uses_existing_device_path(service):
    service._execute("connect_camera", {})
    service._tick()
    service._execute("connect_device", {})
    service._execute("single", {})
    service.controller.worker.join(3)
    service._tick()
    s = service.snapshot()
    assert s["camera_connected"] and s["device_connected"]
    assert s["result"]["sequence"] == 2 and s["result"]["inference_us"] == 100
    assert s["completed"] == 1 and s["camera_size"] == [160, 160]
    assert len(s["history"]) == 1 and service.preview_jpeg.startswith(b"\xff\xd8")
    meta = json.loads((service.recorder.path / "run_metadata.json").read_text())
    assert meta["evidence"] == "synthetic-web-unit-test"
    service._execute("disconnect_device", {})
    assert service.snapshot()["result"] is None


def test_web_session_controls_and_configuration(service):
    with pytest.raises(ValueError, match="先连接"):
        service._execute("start", {})
    service._configure({"fps": 2, "rotation": 90})
    assert service.config["fps"] == 2 and service.config["rotation"] == 90
    with pytest.raises(ValueError):
        service._configure({"fps": float("nan")})
    service._execute("connect_camera", {})
    with pytest.raises(ValueError, match="断开摄像头"):
        service._configure({"camera": 1})
    service._execute("stop", {})
    service._execute("clear_logs", {})
    assert not service.snapshot()["logs"] and not service.running
    service._execute("disconnect_camera", {})
    assert not service.camera_connected and service.preview_jpeg is None


def test_web_command_queue_is_bounded(service):
    for _ in range(16):
        service.enqueue("stop")
    with pytest.raises(queue.Full):
        service.enqueue("stop")
    with pytest.raises(ValueError):
        service.enqueue("flash_firmware")


@pytest.fixture
def http(service):
    pytest.importorskip("flask")
    from agritinyvision.pc_camera.web import create_app

    app = create_app(service)
    app.testing = True
    client = app.test_client()
    html = client.get("/").get_data(as_text=True)
    token = re.search(r'name="atv-token" content="([^"]+)"', html)[1]
    return client, {"X-ATV-Token": token}


def test_web_http_actions_require_page_token(http, service):
    client, headers = http
    assert client.post("/api/action/stop", json={}).status_code == 403
    assert client.post("/api/action/stop", json=[], headers=headers).status_code == 400
    assert client.post("/api/action/invalid", json={}, headers=headers).status_code == 400
    assert client.post("/api/action/stop", json={}, headers=headers).status_code == 202
    assert service.commands.qsize() == 1
    assert client.get("/api/status").json["completed"] == 0
    assert client.get("/api/preview").status_code == 204


def test_web_assets_and_preview(http, service):
    client, _ = http
    assert "设备工作台" in client.get("/").get_data(as_text=True)
    assert "__TOKEN__" not in client.get("/").get_data(as_text=True)
    assert client.get("/assets/app.js").status_code == 200
    assert client.get("/assets/style.css").status_code == 200
    assert "blob:" in client.get("/").headers["Content-Security-Policy"]
    service._execute("connect_camera", {})
    service._tick()
    preview = client.get("/api/preview")
    assert preview.status_code == 200 and preview.mimetype == "image/jpeg"
    assert preview.data.startswith(b"\xff\xd8")


def test_web_download_contains_only_experiment_files(http, service):
    client, _ = http
    assert client.get("/api/export").status_code == 409
    service._execute("connect_device", {})
    service.recorder.close(plots=False)
    (service.recorder.path / "unrelated.txt").write_text("exclude")
    response = client.get("/api/export")
    assert response.status_code == 200 and response.mimetype == "application/zip"
    with zipfile.ZipFile(io.BytesIO(response.data)) as archive:
        assert "frame_results.csv" in archive.namelist()
        assert "HARDWARE_REPORT.md" in archive.namelist()
        assert "unrelated.txt" not in archive.namelist()

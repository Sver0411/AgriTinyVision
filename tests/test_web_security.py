import base64
import io
import json
import re
import zipfile

import pytest
from test_web_camera import service  # noqa: F401

from agritinyvision.pc_camera.experiment_export import experiment_zip, public_experiment
from agritinyvision.pc_camera.web import create_app


@pytest.mark.parametrize("route", ["/api/status", "/api/ports", "/api/preview", "/api/export"])
def test_read_interfaces_require_same_site_session(service, route):  # noqa: F811
    client = create_app(service).test_client()
    assert client.get(route).status_code == 401
    response = client.get("/")
    assert "HttpOnly" in response.headers["Set-Cookie"]
    assert "SameSite=Strict" in response.headers["Set-Cookie"]
    assert client.get(route, headers={"Origin": "http://evil.example"}).status_code == 403


def test_dns_rebinding_and_nonlocal_peer_rejected(service):  # noqa: F811
    client = create_app(service).test_client()
    assert client.get("/", headers={"Host": "evil.example:8765"}).status_code == 403
    assert client.get("/", environ_base={"REMOTE_ADDR": "192.168.1.50"}).status_code == 403
    assert client.get("/assets/../../README.md").status_code == 404


def test_network_binding_requires_independent_password(service):  # noqa: F811
    with pytest.raises(ValueError, match="ATV_WEB_PASSWORD"):
        create_app(service, "0.0.0.0")
    with pytest.raises(ValueError):
        create_app(service, "0.0.0.0", "weak")
    password = "local-test-secret-1234"
    client = create_app(service, "0.0.0.0", password).test_client()
    assert client.get("/").status_code == 401
    auth = "Basic " + base64.b64encode(("agri:" + password).encode()).decode()
    page = client.get("/", headers={"Authorization": auth})
    assert page.status_code == 200 and password not in page.get_data(as_text=True)
    token = re.search(r'name="atv-token" content="([^"]+)"', page.get_data(as_text=True))[1]
    assert client.get("/api/status").status_code == 401
    assert client.get("/api/status", headers={"Authorization": auth}).status_code == 200
    assert client.post("/api/action/stop", json={}, headers={"Authorization": auth, "X-ATV-Token": token,
                       "Origin": "http://evil.example"}).status_code == 403


def test_export_root_and_links(service, tmp_path):  # noqa: F811
    service._execute("connect_device", {})
    service.recorder.close(plots=False)
    folder = service.recorder.path
    with pytest.raises(ValueError):
        experiment_zip(folder, tmp_path / "other")
    target = tmp_path / "private.txt"
    target.write_text("private secret")
    summary = folder / "summary.json"
    summary.unlink()
    try:
        summary.symlink_to(target)
    except OSError:
        # Windows may prohibit symlink creation; root escape still tested above.
        summary.write_text("{}")
    else:
        with pytest.raises(ValueError, match="link"):
            experiment_zip(folder, tmp_path)


def test_public_export_removes_device_and_private_paths(service, tmp_path):  # noqa: F811
    service._execute("connect_device", {})
    recorder = service.recorder
    recorder.metadata["config"]["image"] = r"C:\Users\private-user\photo.jpg"
    recorder.metadata["device"]["mac"] = "30:30:f9:3d:a2:40"
    recorder.metadata["device"]["model_sha256"] = "a" * 64
    recorder.error(r"C:\Users\private-user\photo.jpg failed on 30:30:f9:3d:a2:40")
    recorder.close(plots=False)
    target = public_experiment(recorder.path, tmp_path / "public")
    raw = experiment_zip(target, tmp_path).getvalue()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        content = "\n".join(archive.read(n).decode("utf-8") for n in archive.namelist() if not n.endswith(".png"))
        assert "private-user" not in content and "30:30:f9:3d:a2:40" not in content
        meta = json.loads(archive.read("run_metadata.json"))
        assert meta["device"]["model_sha256"] == "a" * 64
        assert "boot_id" not in meta["device"]
    assert "private-user" in (recorder.path / "run_metadata.json").read_text()

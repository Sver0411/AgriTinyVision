import io
import json
import re
import zipfile

import numpy as np
import pytest
from test_field_capture import annotation
from test_pc_camera import result
from test_web_camera import service  # noqa: F401

from agritinyvision.field_validation.capture import FieldBatch, read_json, read_samples
from agritinyvision.field_validation.web import batch_zip
from agritinyvision.pc_camera.web import create_app


@pytest.fixture
def field_http(service, tmp_path):  # noqa: F811
    service.config["field_output"] = str(tmp_path / "batches")
    service._execute("connect_device", {})
    batch = FieldBatch(tmp_path / "batches", service.client.device, {}, {"plant_id": "plant"}, "simulated_hardware")
    rgb = np.random.default_rng(8).integers(0, 256, (160, 160, 3), dtype=np.uint8)
    sid = batch.begin(rgb[..., ::-1], rgb)
    r = result(p=.9)
    r.update(rgb_sha256=read_json(batch.path / f"metadata/{sid}.json")["rgb_sha256"], end_to_end_ms=4,
             saturated_values=50)
    batch.complete(sid, r)
    app = create_app(service)
    app.testing = True
    client = app.test_client()
    page = client.get("/field").get_data(as_text=True)
    token = re.search(r'name="atv-token" content="([^"]+)"', page)[1]
    return client, {"X-ATV-Token": token}, batch, sid


def test_annotation_roundtrip_images_csv_and_independent_predictions(field_http):
    client, headers, batch, sid = field_http
    prefix = "/api/field/" + batch.path.name
    assert client.get("/api/field/batches").json[0]["samples"] == 1
    assert client.get(prefix).json["samples"][0]["annotated"] is False
    assert client.get(prefix + f"/image/{sid}/model_input").mimetype == "image/png"
    assert client.post(prefix + f"/annotation/{sid}", json=annotation()).status_code == 403
    assert client.post(prefix + f"/annotation/{sid}", json=annotation(), headers=headers).status_code == 200
    detail = client.get(prefix + f"/sample/{sid}").json
    assert detail["annotation"]["visual_label"] == "NORMAL" and detail["prediction"]["result"]["state"] == "SUSPECT"
    assert "NORMAL" in client.get(prefix + "/annotations.csv").get_data(as_text=True)
    report = client.post(prefix + "/report", json={"labels": "manual"}, headers=headers)
    assert report.status_code == 200 and report.json["three_state"]["accepted"]["accuracy"] == 0


def test_batch_download_only_contains_declared_images_and_reports(field_http):
    client, _, batch, sid = field_http
    (batch.path / "private.txt").write_text("not part of batch")
    response = client.get(f"/api/field/{batch.path.name}/export")
    with zipfile.ZipFile(io.BytesIO(response.data)) as archive:
        assert f"images/original/{sid}.png" in archive.namelist()
        assert f"images/model_input/{sid}.png" in archive.namelist()
        assert "private.txt" not in archive.namelist()


def test_collection_routes_require_session_and_reject_path_inputs(field_http, service):  # noqa: F811
    client, _, batch, sid = field_http
    untrusted = create_app(service).test_client()
    assert untrusted.get("/api/field/batches").status_code == 401
    assert client.get("/api/field/../../README.md").status_code == 404
    assert client.get(f"/api/field/{batch.path.name}/image/{sid}/metadata").status_code == 404
    assert client.get(f"/api/field/{batch.path.name}/sample/not-a-sample").status_code == 404
    assert client.get("/api/field/batches", headers={"Origin": "http://evil.example"}).status_code == 403


def test_batch_symlink_cannot_escape_export(field_http, tmp_path):
    _, _, batch, _ = field_http
    sample = read_samples(batch.path)[0]
    metadata = batch.path / sample["metadata"]
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"private": "data"}))
    metadata.unlink()
    try:
        metadata.symlink_to(outside)
    except OSError:
        metadata.write_text("{}")
    else:
        with pytest.raises(ValueError, match="link"):
            batch_zip(batch.path)

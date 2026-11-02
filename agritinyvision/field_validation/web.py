"""Local annotation routes, using the existing console session and CSRF boundary."""

import io
import re
import zipfile
from pathlib import Path

from flask import Response, abort, jsonify, request, send_file

from agritinyvision.pc_camera.experiment_export import checked_file

from .annotations import annotation_csv, labels, save_annotation
from .capture import read_json, read_samples
from .reporting import REPORT_FILES, generate_batch_report


def batch_zip(batch):
    names = {"batch.json", "manifest.jsonl", "annotations/manual.jsonl", "annotations/source.jsonl"}
    for sample in read_samples(batch):
        names.update(sample[k] for k in ("original", "model_input", "metadata", "prediction"))
    names.update("reports/" + name for name in REPORT_FILES | {"SOURCE_AUDIT.json", "FIELD_SPLIT.json"})
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(names):
            file = batch / name
            if file.exists():
                archive.write(checked_file(file, batch), name)
    output.seek(0)
    return output


def register_routes(app, service, assets, token):
    def root():
        return Path(service.config.get("field_output", "runs/field_validation"))

    def batch_folder(batch_id):
        if not re.fullmatch(r"\d{8}_\d{6}_[0-9a-f]{8}", batch_id):
            abort(404)
        folder = root() / batch_id
        if not folder.is_dir() or folder.is_symlink() or not folder.resolve().is_relative_to(root().resolve()):
            abort(404)
        return folder

    def find_sample(folder, sample_id):
        for sample in read_samples(folder):
            if sample["sample_id"] == sample_id:
                return sample
        abort(404)

    @app.get("/field")
    def field_page():
        from flask import session

        session["authorized"] = True
        return Response((assets / "field.html").read_text(encoding="utf-8").replace("__TOKEN__", token), mimetype="text/html")

    @app.get("/api/field/batches")
    def batches():
        result = []
        for path in sorted(root().glob("*/batch.json"), reverse=True):
            folder = batch_folder(path.parent.name)
            meta = read_json(checked_file(folder / "batch.json", folder))
            result.append({"batch_id": folder.name, "source_kind": meta["source_kind"],
                           "samples": len(read_samples(folder)), "annotated": len(labels(folder))})
        return jsonify(result)

    @app.get("/api/field/<batch_id>")
    def batch_detail(batch_id):
        folder = batch_folder(batch_id)
        samples = read_samples(folder)
        manual = labels(folder)
        return jsonify(metadata=read_json(checked_file(folder / "batch.json", folder)),
                       samples=[{"sample_id": s["sample_id"], "annotated": s["sample_id"] in manual} for s in samples])

    @app.get("/api/field/<batch_id>/sample/<sample_id>")
    def sample_detail(batch_id, sample_id):
        folder = batch_folder(batch_id)
        sample = find_sample(folder, sample_id)
        pred = folder / sample["prediction"]
        return jsonify(metadata=read_json(checked_file(folder / sample["metadata"], folder)),
                       prediction=read_json(checked_file(pred, folder)) if pred.exists() else {"status": "pending"},
                       annotation=labels(folder).get(sample_id), source_annotation=labels(folder, "source").get(sample_id))

    @app.get("/api/field/<batch_id>/image/<sample_id>/<kind>")
    def image(batch_id, sample_id, kind):
        if kind not in ("original", "model_input"):
            abort(404)
        folder = batch_folder(batch_id)
        sample = find_sample(folder, sample_id)
        return send_file(checked_file(folder / sample[kind], folder), mimetype="image/png")

    @app.post("/api/field/<batch_id>/annotation/<sample_id>")
    def annotate(batch_id, sample_id):
        folder = batch_folder(batch_id)
        payload = request.get_json()
        if not isinstance(payload, dict):
            return jsonify(error="标注应为 JSON 对象"), 400
        try:
            saved = save_annotation(folder, sample_id, payload)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        return jsonify(saved)

    @app.get("/api/field/<batch_id>/annotations.csv")
    def annotations(batch_id):
        return Response("\ufeff" + annotation_csv(batch_folder(batch_id)), mimetype="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="annotations.csv"'})

    @app.get("/api/field/<batch_id>/export")
    def field_export(batch_id):
        folder = batch_folder(batch_id)
        return send_file(batch_zip(folder), mimetype="application/zip", as_attachment=True,
                         download_name=batch_id + ".zip")

    @app.post("/api/field/<batch_id>/report")
    def report(batch_id):
        folder = batch_folder(batch_id)
        payload = request.get_json()
        if not isinstance(payload, dict) or payload.get("labels") not in ("manual", "source"):
            return jsonify(error="选择 manual 或 source 标签"), 400
        if service.busy and service.field_batch and service.field_batch.path.resolve() == folder.resolve():
            return jsonify(error="等待当前帧完成后生成报告"), 409
        return jsonify(generate_batch_report(folder, payload["labels"]))

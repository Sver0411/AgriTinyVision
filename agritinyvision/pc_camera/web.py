"""Local HTTP interface; USB and model execution stay on the existing device path."""

import hmac
import io
import queue
import secrets
import threading
import time
import webbrowser
import zipfile
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_file
from werkzeug.serving import WSGIRequestHandler, make_server

from .web_service import CameraService

ASSETS = Path(__file__).with_name("web_static")


class RequestHandler(WSGIRequestHandler):
    def log_request(self, code="-", size="-"):
        if str(code).startswith("5"):
            super().log_request(code, size)


def create_app(service):
    app = Flask(__name__, static_folder=str(ASSETS), static_url_path="/assets")
    app.config["MAX_CONTENT_LENGTH"] = 4096
    token = secrets.token_urlsafe(24)

    @app.after_request
    def response_headers(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data: blob:; frame-ancestors 'none'"
        )
        return response

    @app.get("/")
    def index():
        return Response(
            (ASSETS / "index.html").read_text(encoding="utf-8").replace("__TOKEN__", token),
            mimetype="text/html",
        )

    @app.get("/api/status")
    def status():
        return jsonify(service.snapshot())

    @app.get("/api/ports")
    def ports():
        from serial.tools.list_ports import comports

        return jsonify([{"port": p.device, "description": p.description} for p in comports()])

    @app.post("/api/action/<action>")
    def action(action):
        if not hmac.compare_digest(request.headers.get("X-ATV-Token", ""), token):
            return jsonify(error="请刷新当前页面后重试"), 403
        payload = request.get_json()
        if not isinstance(payload, dict):
            return jsonify(error="配置必须是 JSON 对象"), 400
        try:
            service.enqueue(action, payload)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        except queue.Full:
            return jsonify(error="操作队列繁忙，请稍候"), 409
        return jsonify(accepted=True), 202

    @app.get("/api/preview")
    def preview():
        with service.lock:
            data = service.preview_jpeg
        return Response(data, mimetype="image/jpeg") if data else ("", 204)

    @app.get("/api/export")
    def export():
        snapshot = service.snapshot()
        if not snapshot["run_directory"]:
            return jsonify(error="连接 ESP32 并记录实验后可下载"), 409
        folder = Path(snapshot["run_directory"])
        names = {
            "run_metadata.json",
            "frame_results.csv",
            "device_metrics.csv",
            "summary.json",
            "latency_report.json",
            "errors.log",
            "HARDWARE_REPORT.md",
        }
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for file in sorted(folder.iterdir()):
                if file.is_file() and file.name in names:
                    archive.write(file, file.name)
            for file in sorted((folder / "plots").glob("*.png")):
                archive.write(file, "plots/" + file.name)
        buffer.seek(0)
        return send_file(
            buffer, mimetype="application/zip", as_attachment=True, download_name=folder.name + ".zip"
        )

    return app


def serve(args):
    service = CameraService(vars(args))
    app = create_app(service)
    server = make_server(args.host, args.http_port, app, threaded=True, request_handler=RequestHandler)
    url = f"http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{server.server_port}"
    service.start()
    print(f"AgriTinyVision Web: {url}", flush=True)
    if not args.no_browser:

        def open_browser():
            time.sleep(0.4)
            webbrowser.open(url)

        threading.Thread(target=open_browser, daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.close()

"""Local HTTP interface; USB and model execution stay on the existing device path."""

import hmac
import ipaddress
import os
import queue
import secrets
import threading
import time
import webbrowser
from pathlib import Path
from urllib.parse import urlsplit

from flask import Flask, Response, jsonify, request, send_file, session
from werkzeug.serving import WSGIRequestHandler, make_server

from .experiment_export import experiment_zip
from .web_service import CameraService

ASSETS = Path(__file__).with_name("web_static")


class RequestHandler(WSGIRequestHandler):
    def log_request(self, code="-", size="-"):
        if str(code).startswith("5"):
            super().log_request(code, size)


def is_loopback(host):
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def create_app(service, host="127.0.0.1", password=None):
    local = is_loopback(host)
    if not local and (not isinstance(password, str) or len(password) < 16):
        raise ValueError("非本地绑定需要 ATV_WEB_PASSWORD 环境变量（至少 16 个字符），用户名 agri")
    app = Flask(__name__, static_folder=str(ASSETS), static_url_path="/assets")
    app.config["MAX_CONTENT_LENGTH"] = 4096
    app.config.update(SECRET_KEY=secrets.token_bytes(32), SESSION_COOKIE_HTTPONLY=True,
                      SESSION_COOKIE_SAMESITE="Strict", SESSION_COOKIE_NAME="atv_session")
    token = secrets.token_urlsafe(24)

    @app.before_request
    def access_control():
        hostname = urlsplit(request.host_url).hostname or ""
        if local and (not is_loopback(hostname) or not is_loopback(request.remote_addr or "")):
            return jsonify(error="仅允许本机访问"), 403
        if not local:
            auth = request.authorization
            if (not auth or auth.type.lower() != "basic" or auth.username != "agri"
                    or not hmac.compare_digest((auth.password or "").encode(), password.encode())):
                return Response("Authentication required", 401, {"WWW-Authenticate": 'Basic realm="AgriTinyVision"'})
        origin = request.headers.get("Origin")
        if origin and origin != request.host_url.rstrip("/"):
            return jsonify(error="拒绝跨站请求"), 403
        if request.path.startswith("/api/") and not session.get("authorized"):
            return jsonify(error="请先打开控制台首页"), 401
        if request.path.startswith("/api/") and request.method == "POST" and not hmac.compare_digest(
                request.headers.get("X-ATV-Token", "").encode(), token.encode()):
            return jsonify(error="请刷新当前页面后重试"), 403

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
        session["authorized"] = True
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
        try:
            buffer = experiment_zip(folder, service.config.get("output", "runs/hardware"))
        except (ValueError, OSError):
            return jsonify(error="实验文件路径无效"), 403
        return send_file(
            buffer, mimetype="application/zip", as_attachment=True, download_name=folder.name + ".zip"
        )

    from agritinyvision.field_validation.web import register_routes

    register_routes(app, service, ASSETS, token)
    return app


def serve(args):
    service = CameraService(vars(args))
    app = create_app(service, args.host, os.environ.get("ATV_WEB_PASSWORD"))
    server = make_server(args.host, args.http_port, app, threaded=True, request_handler=RequestHandler)
    display_host = "127.0.0.1" if args.host == "0.0.0.0" else f"[{args.host}]" if ":" in args.host else args.host
    url = f"http://{display_host}:{server.server_port}"
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

"""One owner for the camera/serial session shared by every browser view."""

import copy
import io
import queue
import threading
import time
from collections import deque
from datetime import datetime, timezone

from PIL import Image

from agritinyvision.field_validation.capture import FieldBatch

from .camera import Camera
from .inference_controller import InferenceController
from .metrics import RunRecorder, generate_report, write_json
from .serial_client import SerialClient

ACTIONS = {
    "connect_camera",
    "disconnect_camera",
    "connect_device",
    "disconnect_device",
    "start",
    "stop",
    "single",
    "save",
    "clear_logs",
    "configure",
    "field_create", "field_capture", "field_start", "field_pause", "field_finish",
}


class CameraService:
    def __init__(self, config, camera=None, client=None, recorder_factory=RunRecorder):
        self.config = dict(config)
        self.camera = camera or Camera(config.get("camera", 0))
        self.client = client or SerialClient(
            config.get("port") or "", config.get("baudrate", 921600), config.get("timeout", 30),
            bundle=config.get("bundle", "releases/v0.1.0"), deployment=config.get("deployment")
        )
        self.recorder_factory = recorder_factory
        self.recorder = self.controller = self.frame = None
        self.preview_jpeg = None
        self.lock = threading.RLock()
        self.commands = queue.Queue(maxsize=16)
        self.halt = threading.Event()
        self.thread = None
        self.camera_connected = self.device_connected = self.connecting = self.running = False
        self.device_info = self.last_result = None
        self.camera_size = None
        self.capture_count = self.completed = 0
        self.history, self.logs = deque(maxlen=90), deque(maxlen=200)
        self.error = ""
        self.next_inference = 0
        self.capture_ms = self.capture_started = None
        self.field_batch = None
        self.field_running = False
        self.field_interval = 5.0
        self.next_field = 0
        self.captured_at = None

    def start(self):
        self.thread = threading.Thread(target=self._loop, name="camera-service", daemon=True)
        self.thread.start()

    def enqueue(self, action, payload=None):
        if action not in ACTIONS:
            raise ValueError("未知操作")
        self.commands.put_nowait((action, payload or {}))

    @property
    def busy(self):
        controller = self.controller
        return controller is not None and controller.busy

    def _update(self, **values):
        with self.lock:
            for key, value in values.items():
                setattr(self, key, value)

    def _log(self, text, level="info"):
        with self.lock:
            self.logs.append(
                {"timestamp": datetime.now(timezone.utc).isoformat(), "message": text, "level": level}
            )

    def snapshot(self):
        with self.lock:
            recorder = self.recorder
            return copy.deepcopy(
                {
                    "camera_connected": self.camera_connected,
                    "device_connected": self.device_connected,
                    "connecting": self.connecting,
                    "running": self.running,
                    "busy": self.busy,
                    "camera_size": self.camera_size,
                    "capture_count": self.capture_count,
                    "completed": self.completed,
                    "device": self.device_info,
                    "result": self.last_result,
                    "error": self.error,
                    "history": list(self.history),
                    "logs": list(self.logs),
                    "config": self.config,
                    "run_directory": str(recorder.path) if recorder else None,
                    "field": {"batch_id": self.field_batch.path.name if self.field_batch else None,
                              "captures": self.field_batch.count if self.field_batch else 0,
                              "completed": self.field_batch.completed if self.field_batch else 0,
                              "running": self.field_running, "interval_s": self.field_interval},
                }
            )

    def _configure(self, changes):
        allowed = {"camera", "port", "baudrate", "fps", "rotation", "mirror"}
        if any(key not in allowed for key in changes):
            raise ValueError("不支持的配置项")
        config = {**self.config, **changes}
        if (
            not isinstance(config.get("camera"), int)
            or not 0 <= config["camera"] <= 20
            or not isinstance(config.get("port") or "", str)
            or not isinstance(config.get("baudrate"), int)
            or not 9600 <= config["baudrate"] <= 3000000
            or not isinstance(config.get("fps"), (float, int))
            or not 0 < config["fps"] <= 30
            or config.get("rotation") not in (0, 90, 180, 270)
            or not isinstance(config.get("mirror"), bool)
        ):
            raise ValueError("检查摄像头编号、串口、波特率和处理频率")
        if self.camera_connected and config["camera"] != self.config["camera"]:
            raise ValueError("切换编号前请断开摄像头")
        if self.device_connected and any(config[k] != self.config[k] for k in ("port", "baudrate")):
            raise ValueError("切换串口前请断开 ESP32")
        if self.busy and any(config[k] != self.config[k] for k in ("rotation", "mirror")):
            raise ValueError("等待当前帧完成后调整图像方向")
        changed = {key: value for key, value in changes.items() if value != self.config.get(key)}
        self._update(config=config)
        if changed and self.recorder and not self.recorder.closed:
            self.recorder.metadata.setdefault("config_changes", []).append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "after_sequence": self.client.sequence,
                    "values": changed,
                }
            )
            write_json(self.recorder.path / "run_metadata.json", self.recorder.metadata)
        if self.controller:
            self.controller.rotation, self.controller.mirror = config["rotation"], config["mirror"]

    def _require_ready(self):
        if not self.camera_connected or not self.device_connected or self.frame is None:
            raise ValueError("请先连接摄像头和 ESP32")

    def _execute(self, action, payload):
        if action.startswith("field_"):
            self._field_execute(action, payload)
            return
        if payload:
            self._configure(payload)
        if action == "connect_camera":
            self.camera.index = self.config["camera"]
            self.camera.connect()
            self._update(camera_connected=True, error="")
            self._log(f"摄像头 {self.camera.index} 已连接")
        elif action == "disconnect_camera":
            self.camera.disconnect()
            self._update(
                camera_connected=False, running=False, field_running=False, frame=None, preview_jpeg=None, last_result=None
            )
            self._log("摄像头已断开")
        elif action == "connect_device":
            if self.busy:
                raise ValueError("等待当前帧完成后连接设备")
            if not self.config.get("port"):
                raise ValueError("请选择 COM 端口")
            self._update(connecting=True, running=False, device_connected=False)
            try:
                if self.field_batch:
                    self.field_batch.finish()
                self._update(field_batch=None, field_running=False)
                if self.recorder:
                    self.recorder.close()
                self._update(controller=None, recorder=None)
                self.client.port, self.client.baudrate = self.config["port"], self.config["baudrate"]
                device = self.client.connect()
                self.recorder = self.recorder_factory(
                    self.config.get("output", "runs/hardware"), self.config, device
                )
                self.controller = InferenceController(
                    self.client, self.recorder, self.config["rotation"], self.config["mirror"]
                )
                self._update(
                    device_info=device, device_connected=True, completed=0, last_result=None, error=""
                )
                with self.lock:
                    self.history.clear()
                self._log(f"{self.config['port']} 已连接 · ESP-DL 模型已加载")
            finally:
                self._update(connecting=False)
        elif action == "disconnect_device":
            self.client.close()
            self._update(device_connected=False, running=False, field_running=False, last_result=None)
            self._log("ESP32 已断开")
        elif action == "start":
            self._require_ready()
            self._update(running=True, field_running=False, error="")
            self.next_inference = 0
            self._log("开始连续推理")
        elif action == "stop":
            self._update(running=False, field_running=False)
            self._log("已暂停，当前帧将完成并记录")
        elif action == "single":
            self._require_ready()
            self._update(field_running=False)
            if not self.controller.submit(self.frame, self.capture_ms, self.capture_started):
                raise ValueError("当前帧处理中，请稍候")
            self._update(error="")
        elif action == "save":
            self._update(running=False)
            if self.busy:
                raise ValueError("等待当前帧完成后保存报告")
            if self.recorder is None:
                raise ValueError("连接 ESP32 后即可记录实验")
            generate_report(self.recorder.path)
            self._log(f"报告已保存：{self.recorder.path}")
        elif action == "clear_logs":
            with self.lock:
                self.logs.clear()

    def _tick(self):
        if self.camera_connected:
            self.capture_started = time.perf_counter()
            self.captured_at = datetime.now(timezone.utc).isoformat()
            frame = self.camera.read()
            self.capture_ms = (time.perf_counter() - self.capture_started) * 1000
            h, w = frame.shape[:2]
            preview = Image.fromarray(frame[:, :, ::-1])
            preview.thumbnail((800, 600))
            encoded = io.BytesIO()
            preview.save(encoded, format="JPEG", quality=82)
            self._update(
                frame=frame,
                capture_count=self.camera.frame_count,
                camera_size=[w, h],
                preview_jpeg=encoded.getvalue(),
            )
        if self.controller:
            try:
                kind, value = self.controller.events.get_nowait()
                if kind == "error":
                    self._update(error=value, running=False, field_running=False, device_connected=False, last_result=None)
                    self._log(value, "error")
                else:
                    with self.lock:
                        self.completed += 1
                        self.history.append(
                            {
                                "sequence": value["sequence"],
                                "inference_ms": value["inference_us"] / 1000
                                if value["inference_us"] is not None
                                else None,
                                "end_to_end_ms": value["end_to_end_ms"],
                            }
                        )
                        if self.camera_connected and self.device_connected:
                            self.last_result = value
                    self._log(f"#{value['sequence']} {value['state']} · {value['reason']}")
            except queue.Empty:
                pass
        now = time.perf_counter()
        if self.field_running and now >= self.next_field and not self.busy:
            if self._field_submit():
                self.next_field = now + self.field_interval
        if self.running and now >= self.next_inference and self.controller and not self.busy:
            if self.controller.submit(self.frame, self.capture_ms, self.capture_started):
                self.next_inference = now + 1 / self.config["fps"]

    def _loop(self):
        while not self.halt.is_set():
            try:
                action, payload = self.commands.get_nowait()
                try:
                    self._execute(action, payload)
                except Exception as exc:
                    self._update(error=str(exc))
                    self._log(str(exc), "error")
            except queue.Empty:
                pass
            try:
                self._tick()
            except Exception as exc:
                self.camera.disconnect()
                self._update(
                    camera_connected=False,
                    running=False,
                    field_running=False,
                    frame=None,
                    preview_jpeg=None,
                    last_result=None,
                    error=str(exc),
                )
                self._log(str(exc), "error")
                if self.recorder:
                    self.recorder.error(str(exc))
            self.halt.wait(0.08)

    def close(self):
        self.halt.set()
        if self.thread:
            self.thread.join(self.config.get("timeout", 30) + 2)
        if self.controller and self.controller.worker:
            self.controller.worker.join(self.config.get("timeout", 30) + 2)
        self.client.close()
        self.camera.disconnect()
        if self.recorder:
            self.recorder.close()
        if self.field_batch:
            self.field_batch.finish()

    def _field_submit(self):
        return self.controller.submit(self.frame, self.capture_ms, self.capture_started, self.field_batch,
                                      dict(self.config), self.captured_at)

    def _field_execute(self, action, payload):
        if action == "field_pause":
            self._update(field_running=False)
            return
        if action == "field_finish":
            self._update(field_running=False)
            if self.busy:
                raise ValueError("等待当前帧完成后结束批次")
            if self.field_batch:
                self.field_batch.finish()
            self._update(field_batch=None)
            return
        self._require_ready()
        if action == "field_create":
            if self.busy:
                raise ValueError("等待当前帧完成后创建批次")
            context = payload.get("context")
            keys = {"plant_id", "session_id", "crop_type", "lighting", "distance", "angle", "background",
                    "environment", "abnormality_severity", "scene"}
            if not isinstance(context, dict) or set(context) - keys or any(
                    not isinstance(v, str) or len(v) > 200 for v in context.values()):
                raise ValueError("采集条件必须是场景文本，每项至多 200 字符")
            interval = payload.get("interval_s")
            if isinstance(interval, bool) or not isinstance(interval, (int, float)) or not 0.2 <= interval <= 3600:
                raise ValueError("采集间隔应为 0.2～3600 秒")
            if self.field_batch:
                self.field_batch.finish()
            batch = FieldBatch(self.config.get("field_output", "runs/field_validation"), self.device_info,
                               self.config, context, self.client.evidence)
            self._update(field_batch=batch, field_interval=interval, field_running=False, running=False)
            self._log("采集批次已创建：" + batch.path.name)
            return
        if self.field_batch is None:
            raise ValueError("请先创建采集批次")
        if action == "field_start":
            self._update(field_running=True, running=False)
            self.next_field = 0
        elif action == "field_capture":
            self._update(running=False)
            if not self._field_submit():
                raise ValueError("当前帧处理中，请稍候")

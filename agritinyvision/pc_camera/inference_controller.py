import hashlib
import queue
import threading
import time

from .image_preprocess import prepare_bgr
from .protocol import ProtocolError
from .serial_client import DeviceError


class InferenceController:
    """One worker and one in-flight request; UI never accumulates camera frames."""

    def __init__(self, client, recorder, rotation=0, mirror=False):
        self.client, self.recorder = client, recorder
        self.rotation, self.mirror = rotation, mirror
        self.events = queue.Queue(maxsize=1)
        self.worker = None
        self.start = time.perf_counter()
        self.last_completed = self.start
        self.completed = 0

    @property
    def busy(self):
        return self.worker is not None and self.worker.is_alive()

    def submit(self, frame, capture_ms=None, capture_started=None, field_batch=None, capture_config=None,
               captured_at=None):
        if self.busy or not self.events.empty():
            return False
        # Snapshot belongs to exactly one sequence; capture can continue while the MCU runs.
        snapshot = frame.copy()
        self.worker = threading.Thread(target=self._infer, args=(snapshot, capture_ms, capture_started,
                                       field_batch, capture_config, captured_at), daemon=True)
        self.worker.start()
        return True

    def _infer(self, frame, capture_ms=None, capture_started=None, field_batch=None, capture_config=None,
               captured_at=None):
        sample_id = None
        try:
            started = time.perf_counter()
            rgb = prepare_bgr(frame, self.rotation, self.mirror)
            preprocessing_ms = (time.perf_counter() - started) * 1000
            if field_batch:
                config = {**(capture_config or {}), "rotation": self.rotation, "mirror": self.mirror}
                sample_id = field_batch.begin(frame, rgb, config, captured_at)
            result = self.client.infer(rgb.tobytes())
            result["rgb_sha256"] = hashlib.sha256(rgb.tobytes()).hexdigest()
            now = time.perf_counter()
            result["serial_roundtrip_ms"] = result["end_to_end_ms"]
            result["pc_capture_ms"] = capture_ms
            result["pc_preprocess_ms"] = preprocessing_ms
            result["end_to_end_ms"] = (now - (capture_started or started)) * 1000
            self.completed += 1
            result["actual_fps"] = 1 / (now - self.last_completed)
            self.last_completed = now
            self.recorder.record(result, now - self.start)
            if field_batch:
                field_batch.complete(sample_id, result)
                result["field_sample_id"] = sample_id
            self.events.put(("result", result))
        except (OSError, ValueError, TimeoutError, ProtocolError, DeviceError) as exc:
            if field_batch and sample_id:
                field_batch.error(sample_id, exc)
            self.recorder.error(str(exc), self.client.sequence)
            self.events.put(("error", str(exc)))

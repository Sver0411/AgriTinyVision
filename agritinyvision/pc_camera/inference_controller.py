import hashlib
import queue
import threading
import time

from .image_preprocess import prepare_bgr


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

    def submit(self, frame):
        if self.busy or not self.events.empty():
            return False
        # Snapshot belongs to exactly one sequence; capture can continue while the MCU runs.
        snapshot = frame.copy()
        self.worker = threading.Thread(target=self._infer, args=(snapshot,), daemon=True)
        self.worker.start()
        return True

    def _infer(self, frame):
        try:
            rgb = prepare_bgr(frame, self.rotation, self.mirror)
            result = self.client.infer(rgb.tobytes())
            result["rgb_sha256"] = hashlib.sha256(rgb.tobytes()).hexdigest()
            now = time.perf_counter()
            self.completed += 1
            result["actual_fps"] = 1 / (now - self.last_completed)
            self.last_completed = now
            self.recorder.record(result, now - self.start)
            self.events.put(("result", result))
        except Exception as exc:
            self.recorder.error(str(exc), self.client.sequence)
            self.events.put(("error", str(exc)))

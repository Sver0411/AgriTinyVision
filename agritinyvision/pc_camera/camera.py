class Camera:
    def __init__(self, index=0, capture_factory=None):
        self.index = index
        self.capture_factory = capture_factory
        self.capture = None
        self.frame_count = 0

    @property
    def connected(self):
        return self.capture is not None and self.capture.isOpened()

    def connect(self):
        self.disconnect()
        if self.capture_factory is None:
            import cv2

            self.capture_factory = cv2.VideoCapture
        self.capture = self.capture_factory(self.index)
        if not self.connected:
            self.disconnect()
            raise RuntimeError(f"Cannot open camera {self.index}")

    def disconnect(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def read(self):
        if not self.connected:
            raise RuntimeError("Camera disconnected")
        ok, frame = self.capture.read()
        if not ok or frame is None:
            self.disconnect()
            raise RuntimeError("Camera disconnected or capture failed")
        self.frame_count += 1
        return frame

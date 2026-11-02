"""One outstanding frame, verified receive ACK, then the MCU inference result."""

import threading
import time

from .device_identity import load_expected_identity, verify_device_identity
from .protocol import Decoder, MessageType, Packet, ProtocolError, frame_packet
from .result_parser import parse_device, parse_json, parse_result


class DeviceError(RuntimeError):
    pass


class SerialClient:
    def __init__(self, port, baudrate=921600, timeout=30.0, serial_factory=None, expected_identity=None,
                 bundle="releases/v0.1.0", deployment=None):
        self.port, self.baudrate, self.timeout = port, baudrate, timeout
        self.serial_factory = serial_factory
        self.evidence = "hardware" if serial_factory is None else "simulated_hardware"
        self.serial = None
        self.device = None
        self.sequence = 0
        self.decoder = Decoder()
        self.pending = []
        self.lock = threading.Lock()
        self.expected_identity = expected_identity
        self.bundle, self.deployment = bundle, deployment

    def close(self):
        transport, self.serial = self.serial, None
        self.device = None
        if transport is not None:
            transport.close()

    def _next(self):
        if self.sequence == 0xFFFFFFFF:
            raise DeviceError("Sequence exhausted; reconnect ESP32")
        self.sequence += 1
        return self.sequence

    def connect(self):
        with self.lock:
            self.close()
            expected = self.expected_identity or load_expected_identity(self.bundle, self.deployment)
            if self.serial_factory is None:
                import serial

                self.serial_factory = serial.Serial
            s = self.serial_factory()
            s.port, s.baudrate, s.timeout, s.write_timeout = self.port, self.baudrate, 0.1, 5
            s.dtr = s.rts = False
            self.serial = s
            try:
                s.open()
                s.reset_input_buffer()
                self.sequence = 0
                self.decoder.reset()
                self.pending.clear()
                seq = self._next()
                self._write(Packet(MessageType.HELLO, seq).encode())
                packet = self._receive(seq, MessageType.HELLO_ACK, time.perf_counter() + self.timeout)
                self.device = parse_device(packet.payload)
                if not self.device["model_loaded"]:
                    raise DeviceError("Model not loaded: " + str(self.device.get("model_error")))
                self.device["identity_validation"] = verify_device_identity(self.device, expected)
                return self.device
            except Exception:
                self.close()
                raise

    def _write(self, data):
        if self.serial is None or not self.serial.is_open:
            raise DeviceError("ESP32 disconnected")
        if self.serial.write(data) != len(data):
            raise DeviceError("Incomplete serial write")
        self.serial.flush()

    def _receive(self, sequence, kind, deadline):
        while time.perf_counter() < deadline:
            if self.pending:
                packet = self.pending.pop(0)
                if packet.sequence != sequence:
                    raise ProtocolError("sequence mismatch; reconnect ESP32")
                if packet.kind == MessageType.ERROR:
                    error = parse_json(packet.payload)
                    raise DeviceError(f"ESP32 {error.get('code')}: {error.get('message')}")
                if packet.kind != kind:
                    raise ProtocolError(f"expected {kind.name}, received {packet.kind.name}")
                return packet
            if self.serial is None or not self.serial.is_open:
                raise DeviceError("ESP32 disconnected")
            self.pending.extend(self.decoder.feed(self.serial.read(4096)))
            if self.decoder.errors:
                raise ProtocolError("serial packet: " + ", ".join(self.decoder.errors))
        raise TimeoutError(f"ESP32 timeout waiting for {kind.name}; reconnect device")

    def infer(self, rgb):
        with self.lock:
            if self.device is None:
                raise DeviceError("Connect ESP32 before inference")
            seq = self._next()
            # Construct and validate before sending anything to the device.
            data = frame_packet(seq, rgb).encode()
            start = time.perf_counter()
            deadline = start + self.timeout
            try:
                self._write(data)
                self._receive(seq, MessageType.FRAME_ACK, deadline)
                acknowledged = time.perf_counter()
                packet = self._receive(seq, MessageType.INFER_RESULT, deadline)
                result = parse_result(packet.payload, seq, self.device)
                end = time.perf_counter()
            except Exception:
                # A timeout may leave an old result in flight. A new handshake is required.
                self.close()
                raise
            return {
                **result,
                "transfer_ms": (acknowledged - start) * 1000,
                "end_to_end_ms": (end - start) * 1000,
            }

    def ping(self):
        with self.lock:
            seq = self._next()
            self._write(Packet(MessageType.PING, seq).encode())
            return self._receive(seq, MessageType.PING, time.perf_counter() + self.timeout)

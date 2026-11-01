"""ATV2 wire protocol. All integers are little endian; CRC32 is IEEE/zlib."""

import json
import struct
import time
import zlib
from dataclasses import dataclass
from enum import IntEnum

MAGIC = b"ATV2"
VERSION = 1
WIDTH = HEIGHT = 160
FRAME_BYTES = WIDTH * HEIGHT * 3
CONTROL_BYTES = 4096
HEADER = struct.Struct("<4sBBHIHHB3xIII")


class MessageType(IntEnum):
    HELLO = 1
    HELLO_ACK = 2
    FRAME = 3
    FRAME_ACK = 4
    INFER_RESULT = 5
    ERROR = 6
    PING = 7


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class Packet:
    kind: MessageType
    sequence: int
    payload: bytes = b""
    width: int = 0
    height: int = 0
    pixel_format: int = 0

    def encode(self):
        validate_fields(
            self.kind, self.sequence, self.width, self.height, self.pixel_format, len(self.payload)
        )
        header = HEADER.pack(
            MAGIC,
            VERSION,
            self.kind,
            0,
            self.sequence,
            self.width,
            self.height,
            self.pixel_format,
            len(self.payload),
            zlib.crc32(self.payload),
            0,
        )
        return header[:28] + struct.pack("<I", zlib.crc32(header[:28])) + self.payload


def validate_fields(kind, sequence, width, height, pixel_format, length):
    if not 1 <= sequence <= 0xFFFFFFFF:
        raise ProtocolError("sequence must be 1..4294967295")
    if kind == MessageType.FRAME:
        if (width, height, pixel_format, length) != (WIDTH, HEIGHT, 1, FRAME_BYTES):
            raise ProtocolError("FRAME requires 160x160 RGB888 / 76800 bytes")
    elif width or height or pixel_format or length > CONTROL_BYTES:
        raise ProtocolError("invalid control packet dimensions or length")
    if kind in (MessageType.HELLO, MessageType.FRAME_ACK, MessageType.PING) and length:
        raise ProtocolError("request/ack payload must be empty")


def frame_packet(sequence, rgb):
    return Packet(MessageType.FRAME, sequence, bytes(rgb), WIDTH, HEIGHT, 1)


def json_packet(kind, sequence, value):
    return Packet(kind, sequence, json.dumps(value, allow_nan=False, separators=(",", ":")).encode())


class Decoder:
    """Bounded streaming decoder; keeps sync across boot logs, damaged headers and CRC failures."""

    def __init__(self, timeout=15.0, clock=time.monotonic):
        self.buffer = bytearray()
        self.timeout = timeout
        self.clock = clock
        self.started = None
        self.errors = []

    def reset(self):
        self.buffer.clear()
        self.started = None

    def feed(self, data=b""):
        now = self.clock()
        self.errors = []
        if self.started is not None and now - self.started > self.timeout:
            self.reset()
            self.errors.append("packet_timeout")
        packets = []
        # Consume input in small slices even when a caller supplies megabytes of noise.
        for offset in range(0, max(1, len(data)), 1024):
            self.buffer.extend(data[offset : offset + 1024])
            while True:
                index = self.buffer.find(MAGIC)
                if index < 0:
                    self.buffer[:] = self.buffer[-3:]
                    self.started = None
                    break
                if index:
                    del self.buffer[:index]
                if self.started is None:
                    self.started = now
                if len(self.buffer) < HEADER.size:
                    break
                raw = self.buffer[: HEADER.size]
                _, version, kind, flags, seq, w, h, fmt, length, crc, hcrc = HEADER.unpack(raw)
                try:
                    if version != VERSION or flags or raw[17:20] != b"\0\0\0":
                        raise ProtocolError("header_version_or_flags")
                    if zlib.crc32(raw[:28]) != hcrc:
                        raise ProtocolError("header_crc")
                    kind = MessageType(kind)
                    validate_fields(kind, seq, w, h, fmt, length)
                except ValueError as exc:
                    self.errors.append(str(exc))
                    del self.buffer[0]
                    self.started = None
                    continue
                if len(self.buffer) < HEADER.size + length:
                    break
                payload = bytes(self.buffer[HEADER.size : HEADER.size + length])
                del self.buffer[: HEADER.size + length]
                self.started = None
                if zlib.crc32(payload) != crc:
                    self.errors.append("payload_crc")
                    continue
                packets.append(Packet(kind, seq, payload, w, h, fmt))
        return packets

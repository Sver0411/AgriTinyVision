"""Measured calibration packets, portable binary heads and exact INT8 inference."""

import struct

import numpy as np

TRAIN = struct.Struct("<4sIB")
IMAGE = struct.Struct("<4sI")
REPLY = struct.Struct("<4sIf")
F32 = struct.Struct("<4sHf")
I8 = struct.Struct("<4sHff")
SCALAR = struct.Struct("<f")
MAGIC = {
    "scalar": b"TXSC",
    "local_fp32": b"TXFP",
    "local_int8": b"TXQ8",
    "dino": b"TXJP",
}


def normalized(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.where(norms > 0, norms, 1.0)


def quantized(x):
    """Per last-axis scale, round-to-nearest-even, symmetric signed range."""
    x = np.asarray(x, dtype=np.float32)
    maximum = np.max(np.abs(x), axis=-1, keepdims=True)
    scale = np.where(maximum > 0, maximum / np.float32(127), np.float32(1)).astype(
        np.float32
    )
    q = np.clip(np.rint(x / scale), -127, 127).astype(np.int8)
    return q, scale.squeeze(-1)


def training_record(method, ident, label, x):
    header = TRAIN.pack(MAGIC[method], ident, label)
    if method == "local_int8":
        q, scale = quantized(x)
        payload = SCALAR.pack(scale) + q.tobytes()
    elif method == "dino":
        payload = x
    else:
        payload = np.asarray(x, dtype="<f4").tobytes()
    return header + payload


def training_input(method, record, ident, label, width):
    magic, uid, y = TRAIN.unpack_from(record)
    if (magic, uid, y) != (MAGIC[method], ident, label):
        raise ValueError("Training record metadata mismatch")
    payload = record[TRAIN.size :]
    if method == "local_int8":
        if len(payload) != SCALAR.size + width:
            raise ValueError("Malformed quantized record")
        (scale,) = SCALAR.unpack_from(payload)
        code = np.frombuffer(payload, np.int8, offset=SCALAR.size)
        # Retain physical scale when decoding; do not add a float32 product rounding.
        return code.astype(np.float64) * scale
    if method == "dino":
        return payload
    if len(payload) != 4 * width:
        raise ValueError("Malformed float record")
    return np.frombuffer(payload, "<f4")


def pack_head(method, weights, bias):
    weights = np.asarray(weights, np.float32)
    if method == "local_int8":
        q, scale = quantized(weights)
        return I8.pack(b"EQ01", len(q), scale, bias) + q.tobytes()
    return F32.pack(b"EF01", len(weights), bias) + weights.astype("<f4").tobytes()


def predict_from_head(method, blob, features):
    if method == "local_int8":
        magic, d, scale, bias = I8.unpack_from(blob)
        if magic != b"EQ01" or len(blob) != I8.size + d:
            raise ValueError("Malformed INT8 head")
        w = np.frombuffer(blob, dtype=np.int8, offset=I8.size).astype(np.int32)
        q = np.asarray(features, np.int32)
        dot = q @ w
        norm = np.sqrt(np.einsum("ij,ij->i", q, q, dtype=np.int32))
        return dot.astype(np.float64) * scale / np.where(norm > 0, norm, 1.0) + bias
    magic, d, bias = F32.unpack_from(blob)
    if magic != b"EF01" or len(blob) != F32.size + 4 * d:
        raise ValueError("Malformed FP32 head")
    w = np.frombuffer(blob, "<f4", offset=F32.size).astype(np.float64)
    return np.asarray(features, np.float64) @ w + bias

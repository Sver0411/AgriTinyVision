import platform
import time
from pathlib import Path

import numpy as np

from .utils import sha256, write_json


def benchmark(model_path, iterations=100, output=None):
    import onnxruntime as ort

    if iterations < 1:
        raise ValueError("iterations must be positive")
    settings = ort.SessionOptions()
    settings.intra_op_num_threads = 4
    session = ort.InferenceSession(str(model_path), sess_options=settings, providers=["CPUExecutionProvider"])
    shape = session.get_inputs()[0].shape
    x = np.random.default_rng(41).standard_normal(shape).astype(np.float32)
    inputs = {session.get_inputs()[0].name: x}
    for _ in range(10):
        session.run(None, inputs)
    durations = []
    for _ in range(iterations):
        start = time.perf_counter()
        session.run(None, inputs)
        durations.append((time.perf_counter() - start) * 1000)
    result = {
        "benchmark_scope": "DESKTOP CPU ONLY; never ESP32 latency",
        "hardware_status": "PENDING HARDWARE VALIDATION",
        "os": platform.platform(),
        "processor": platform.processor(),
        "onnxruntime": ort.__version__,
        "provider": "CPUExecutionProvider",
        "threads": 4,
        "iterations": iterations,
        "median_ms": float(np.median(durations)),
        "p95_ms": float(np.percentile(durations, 95)),
        "model_bytes": Path(model_path).stat().st_size,
        "sha256": sha256(model_path),
    }
    write_json(output or str(model_path) + ".benchmark.json", result)
    print(result)
    return result


def compute(model, size):
    """Analytical Conv/Linear multiply-accumulates; excludes pooling/activation/preprocessing."""
    import torch
    from torch import nn

    torch.set_num_threads(4)

    count = [0]

    def hook(module, inputs, output):
        if isinstance(module, nn.Conv2d):
            count[0] += output.numel() * (module.in_channels // module.groups) * np.prod(module.kernel_size)
        elif isinstance(module, nn.Linear):
            count[0] += output.numel() * module.in_features

    handles = [
        m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))
    ]
    with torch.inference_mode():
        model.cpu().eval()(torch.zeros(1, 3, size, size))
    for handle in handles:
        handle.remove()
    return int(count[0])

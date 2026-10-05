import importlib.metadata
import json
import platform
import sys

from .utils import now, write_json


def diagnose():
    result = {"created_at": now(), "python": sys.version, "os": platform.platform(), "cuda_available": False}
    try:
        import torch

        result.update(
            pytorch=torch.__version__,
            cuda_runtime=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(),
            cuda_available=torch.cuda.is_available(),
            gpu_count=torch.cuda.device_count(),
        )
        result["gpus"] = [
            {
                "name": torch.cuda.get_device_name(i),
                "vram_bytes": torch.cuda.get_device_properties(i).total_memory,
            }
            for i in range(torch.cuda.device_count())
        ]
        if torch.cuda.is_available():
            x = torch.arange(16, device="cuda", dtype=torch.float32)
            result["cuda_tensor_check"] = float((x * x).sum().cpu()) == 1240.0
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    result["packages"] = {p.metadata["Name"]: p.version for p in importlib.metadata.distributions()}
    return result


def device(allow_cpu=False):
    import torch

    if torch.cuda.is_available():
        print(f"Using device: {torch.cuda.get_device_name(0)}", flush=True)
        return torch.device("cuda")
    if not allow_cpu:
        raise RuntimeError("CUDA unavailable. Run doctor; CPU requires explicit --allow-cpu.")
    print("Using device: CPU (explicit fallback)", flush=True)
    return torch.device("cpu")


def main(output=None):
    r = diagnose()
    print(json.dumps(r, indent=2))
    if output:
        write_json(output, r)
    return r


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--output")
    main(p.parse_args().output)

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        value = yaml.safe_load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def commit():
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={Path.cwd().as_posix()}", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def validate_config(c):
    required = {
        "model",
        "input_size",
        "seed",
        "manifest",
        "dataset_root",
        "epochs",
        "batch_size",
        "optimizer",
        "learning_rate",
        "augmentation",
        "threshold",
        "pretrained",
        "width_mult",
        "workers",
        "patience",
        "weight_decay",
        "scheduler",
        "amp",
        "deterministic",
    }
    missing = required - c.keys()
    if missing:
        raise ValueError(f"Missing config fields: {sorted(missing)}")
    if c["model"] != "mobilenet_v2" or c["input_size"] not in (160, 192, 224):
        raise ValueError("v0.1 supports MobileNetV2 with 160/192/224 RGB inputs")
    if c["width_mult"] not in (0.35, 0.5, 1.0):
        raise ValueError("Supported width_mult: 0.35, 0.5, 1.0")
    if c["pretrained"] and c["width_mult"] != 1.0:
        raise ValueError("torchvision ImageNet weights require width_mult=1.0")
    if c["optimizer"] not in ("adamw", "sgd"):
        raise ValueError("Unsupported optimizer")
    if c["scheduler"] != "cosine" or c["weight_decay"] < 0 or c["patience"] < 1:
        raise ValueError("Invalid scheduler, weight decay or early stopping patience")
    if any(not isinstance(c[k], bool) for k in ("pretrained", "amp", "deterministic")):
        raise ValueError("pretrained, amp and deterministic must be YAML booleans")
    if c["epochs"] < 1 or c["batch_size"] < 1 or c["learning_rate"] <= 0 or c["workers"] < 0:
        raise ValueError("Invalid training scalar")
    t = c["threshold"]
    if not 0 < t["min"] <= t["max"] < 1 or t["step"] <= 0:
        raise ValueError("Invalid threshold search interval")
    for key in ("target_recall", "max_fpr"):
        if not 0 <= t[key] <= 1:
            raise ValueError(f"Invalid {key}")
    a = c["augmentation"]
    if not 0 < a["crop_min"] <= 1 or not 0 <= a["rotation"] <= 20:
        raise ValueError("Augmentation exceeds mild field-image bounds")
    if not 0 <= a["horizontal_flip"] <= 1:
        raise ValueError("Invalid horizontal_flip probability")
    if any(not 0 <= a[k] <= 0.3 for k in ("brightness", "contrast", "saturation")):
        raise ValueError("Color augmentation must be mild")
    return c

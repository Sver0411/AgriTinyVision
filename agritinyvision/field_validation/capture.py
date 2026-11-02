"""Lossless image pairs and plain-file batches; ordinary inference saves no photos."""

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from agritinyvision.pc_camera.experiment_export import checked_file
from agritinyvision.pc_camera.image_preprocess import prepare_rgb
from agritinyvision.pc_camera.metrics import environment_metadata, write_json
from agritinyvision.utils import sha256


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_samples(batch):
    with (Path(batch) / "manifest.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sample_files(batch, sample):
    batch = Path(batch)
    return {k: checked_file(batch / sample[k], batch) for k in ("metadata", "original", "model_input")}


def verified_sample(batch, sample):
    """Validate the external saved pair once when entering an analysis pipeline."""
    files = sample_files(batch, sample)
    metadata = read_json(files["metadata"])
    arrays = {}
    for key, file_hash, pixel_hash in (("original", "original_file_sha256", "original_rgb_sha256"),
                                       ("model_input", "model_input_file_sha256", "rgb_sha256")):
        if sha256(files[key]) != metadata[file_hash]:
            raise ValueError("Saved image file hash mismatch: " + sample["sample_id"])
        with Image.open(files[key]) as image:
            arrays[key] = np.asarray(image.convert("RGB")).copy()
        if hashlib.sha256(arrays[key].tobytes()).hexdigest() != metadata[pixel_hash]:
            raise ValueError("Saved pixel hash mismatch: " + sample["sample_id"])
    config = metadata["camera_config"]
    reproduced = prepare_rgb(arrays["original"], rotation=config.get("rotation", 0), mirror=config.get("mirror", False))
    if not np.array_equal(reproduced, arrays["model_input"]):
        raise ValueError("Original image and recorded preprocessing do not reproduce model input")
    return metadata, arrays["model_input"]


class FieldBatch:
    def __init__(self, root, device, config, context, evidence="hardware"):
        batch_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
        self.path = Path(root) / batch_id
        self.path.mkdir(parents=True)
        for folder in ("images/original", "images/model_input", "metadata", "predictions", "annotations", "reports"):
            (self.path / folder).mkdir(parents=True)
        self.device, self.config, self.context = dict(device), dict(config), dict(context)
        self.count = self.completed = 0
        self.metadata = {"schema": "field-batch-v1", "batch_id": batch_id, "started_at": now(),
                         "source_kind": context.get("source_kind", "pc_camera"), "evidence": evidence,
                         "project": environment_metadata(), "device": self.device, "config": self.config,
                         "context": self.context}
        write_json(self.path / "batch.json", self.metadata)
        (self.path / "manifest.jsonl").touch()

    def begin(self, bgr, rgb, config=None, captured_at=None, context=None):
        sample_id = uuid.uuid4().hex
        original = Image.fromarray(bgr[..., ::-1])
        paths = {"sample_id": sample_id, "original": f"images/original/{sample_id}.png",
                 "model_input": f"images/model_input/{sample_id}.png",
                 "metadata": f"metadata/{sample_id}.json", "prediction": f"predictions/{sample_id}.json"}
        original.save(self.path / paths["original"])
        Image.fromarray(rgb).save(self.path / paths["model_input"])
        metadata = {"sample_id": sample_id, "captured_at": captured_at or now(),
                    "original_size": list(original.size), "camera_config": dict(config or self.config),
                    "context": {**self.context, **(context or {})},
                    "original_file_sha256": sha256(self.path / paths["original"]),
                    "original_rgb_sha256": hashlib.sha256(original.tobytes()).hexdigest(),
                    "model_input_file_sha256": sha256(self.path / paths["model_input"]),
                    "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                    "model_sha256": self.device["model_sha256"],
                    "firmware_sha256": self.device.get("firmware_sha256"),
                    "model_version": self.device["model_version"], "firmware_version": self.device["firmware_version"]}
        write_json(self.path / paths["metadata"], metadata)
        with (self.path / "manifest.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(paths) + "\n")
        self.count += 1
        return sample_id

    def complete(self, sample_id, result):
        write_json(self.path / f"predictions/{sample_id}.json", {"status": "ok", "result": result})
        self.completed += 1

    def error(self, sample_id, error):
        # An execution failure is separate from the model's UNKNOWN decision.
        write_json(self.path / f"predictions/{sample_id}.json", {"status": "error", "error": str(error)})

    def finish(self):
        self.metadata.update(finished_at=now(), captures=self.count, completed=self.completed)
        write_json(self.path / "batch.json", self.metadata)

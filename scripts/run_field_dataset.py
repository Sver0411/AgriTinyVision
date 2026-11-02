"""Replay an externally sourced, preselected image manifest on the physical frozen deployment."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageOps

from agritinyvision.data.audit import perceptual_hash
from agritinyvision.field_validation.capture import FieldBatch, read_json
from agritinyvision.pc_camera.image_preprocess import load_image_bgr, prepare_bgr
from agritinyvision.pc_camera.metrics import write_json
from agritinyvision.pc_camera.protocol import ProtocolError
from agritinyvision.pc_camera.serial_client import DeviceError, SerialClient
from agritinyvision.utils import sha256


def fingerprints(manifest, root, cache):
    manifest, root, cache = Path(manifest), Path(root), Path(cache)
    digest = sha256(manifest)
    if cache.exists():
        saved = read_json(cache)
        if saved["manifest_sha256"] == digest:
            return saved["samples"]
    rows = []
    for row in read_json(manifest)["samples"]:
        with Image.open(root / row["path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            rows.append({"path": row["path"], "split": row["split"], "sha256": sha256(root / row["path"]),
                         "rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(), "phash": perceptual_hash(image)})
    write_json(cache, {"manifest_sha256": digest, "samples": rows})
    return rows


def run(manifest, port, output, reference_manifest, reference_root):
    manifest = Path(manifest)
    source = read_json(manifest)
    known = fingerprints(reference_manifest, reference_root, Path(".cache/field_reference_fingerprints.json"))
    client = SerialClient(port)
    device = client.connect()
    context = {"source_kind": "public_dataset", "dataset": source["dataset"],
               "source_group": source["repository"] + "@" + source["revision"],
               "repository": source["repository"], "revision": source["revision"],
               "source_manifest_sha256": sha256(manifest), "license": source["license"]}
    batch = FieldBatch(output, device, {"port": port, "rotation": 0, "mirror": False}, context, client.evidence)
    print("Batch:", batch.path, flush=True)
    overlap = []
    try:
        for index, sample in enumerate(source["samples"]):
            image = manifest.parent / sample["path"]
            if sha256(image) != sample["sha256"]:
                raise ValueError("Source file hash differs: " + sample["path"])
            bgr = load_image_bgr(image)
            rgb = prepare_bgr(bgr)
            original_rgb = np.ascontiguousarray(bgr[..., ::-1])
            original_hash = hashlib.sha256(original_rgb.tobytes()).hexdigest()
            phash = perceptual_hash(Image.fromarray(original_rgb))
            matches = [r for r in known if r["sha256"] == sample["sha256"] or r["rgb_sha256"] == original_hash
                       or (phash ^ r["phash"]).bit_count() <= 4]
            per_image = {**sample, "source_kind": "public_dataset", "crop_type": sample["crop_type"],
                         "excluded_from_evaluation": bool(matches), "overlap_reference": matches}
            sid = batch.begin(bgr, rgb, context=per_image)
            label = {"sample_id": sid, "plant_presence": "", "visual_label": sample["label"],
                     "judgment_certain": None, "reference_eligible": True, "image_quality": "", "crop_type": sample["crop_type"],
                     "label_origin": "dataset_author", "annotator": "PlantDoc dataset authors",
                     "source_label": sample["source_label"], "source_url": sample["source_url"],
                     "expert_confirmed": False,
                     "notes": "Binary mapping of author taxonomy; human quality, scene and certainty not re-reviewed"}
            with (batch.path / "annotations/source.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(label, ensure_ascii=False) + "\n")
            if matches:
                write_json(batch.path / f"predictions/{sid}.json", {"status": "excluded", "reason": "reference overlap"})
                overlap.append({"sample_id": sid, "source": sample["source_path"], "matches": matches})
            else:
                try:
                    result = client.infer(rgb.tobytes())
                except (OSError, TimeoutError, ProtocolError, DeviceError) as exc:
                    batch.error(sid, exc)
                    raise
                result["rgb_sha256"] = hashlib.sha256(rgb.tobytes()).hexdigest()
                result["serial_roundtrip_ms"] = result["end_to_end_ms"]
                batch.complete(sid, result)
            print(f"{index+1}/{len(source['samples'])} {sample['source_label']} " + ("excluded" if matches else result["state"]), flush=True)
    finally:
        client.close()
        batch.finish()
    write_json(batch.path / "reports/SOURCE_AUDIT.json", {"reference_manifest_sha256": sha256(reference_manifest),
               "reference_samples": len(known), "threshold": "exact file/RGB or pHash Hamming<=4",
               "overlap": overlap, "selected_samples": len(source["samples"]),
               "provenance": context, "independence": "No source plant/session IDs; conservative batch group"})
    return batch.path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--port", required=True)
    p.add_argument("--output", default="runs/field_validation")
    p.add_argument("--reference-manifest", default="data/manifests/plantvillage_split.json")
    p.add_argument("--reference-root", default="data/raw/plantvillage_color")
    a = p.parse_args()
    run(a.manifest, a.port, a.output, a.reference_manifest, a.reference_root)

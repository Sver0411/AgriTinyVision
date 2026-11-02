"""Conservative connected groups for plants, sessions, batches and duplicate images."""

import random
from collections import Counter
from pathlib import Path

from PIL import Image

from agritinyvision.data.audit import perceptual_hash
from agritinyvision.pc_camera.metrics import write_json

from .annotations import labels
from .capture import read_json, read_samples, sample_files


def split_batches(batches, output, seed=41, ratios=(.2, .4, .4)):
    rows, hashes = [], []
    for batch in map(Path, batches):
        meta = read_json(batch / "batch.json")
        annotated = labels(batch)
        for sample in read_samples(batch):
            files = sample_files(batch, sample)
            m = read_json(files["metadata"])
            c = {**m["context"], **{k: v for k, v in annotated.get(sample["sample_id"], {}).items() if v != ""}}
            rows.append({"sample_id": sample["sample_id"], "batch_id": meta["batch_id"],
                         "plant_id": c.get("plant_id"), "session_id": c.get("session_id"),
                         "source_group": c.get("source_group"), "rgb_sha256": m["rgb_sha256"],
                         "original_rgb_sha256": m["original_rgb_sha256"], "split": None})
            with Image.open(files["original"]) as im:
                hashes.append(perceptual_hash(im))
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(j)] = find(i)

    seen = {}
    for i, row in enumerate(rows):
        for key in ("batch_id", "plant_id", "session_id", "source_group", "rgb_sha256", "original_rgb_sha256"):
            value = row[key]
            if value:
                marker = key, value
                if marker in seen:
                    union(i, seen[marker])
                seen[marker] = i
    near_pairs = []
    for i in range(len(rows)):
        for j in range(i):
            if (hashes[i] ^ hashes[j]).bit_count() <= 4:
                union(i, j)
                near_pairs.append([rows[j]["sample_id"], rows[i]["sample_id"]])
    groups = {}
    for i in range(len(rows)):
        groups.setdefault(find(i), []).append(i)
    names = ("field_train", "field_validation", "field_test")
    active = [i for i, v in enumerate(ratios) if v > 0]
    status = "READY" if len(groups) >= len(active) else "PENDING: insufficient independent groups"
    if status == "READY":
        ordered = sorted(groups.values(), key=lambda ids: min(rows[i]["sample_id"] for i in ids))
        random.Random(seed).shuffle(ordered)
        counts = [0, 0, 0]
        for group in ordered:
            chosen = min(active, key=lambda i: counts[i] / ratios[i])
            for index in group:
                rows[index]["split"] = names[chosen]
            counts[chosen] += len(group)
    result = {"seed": seed, "ratios": list(ratios), "status": status, "independent_groups": len(groups),
              "policy": "batch/session/plant/source + exact RGB + connected pHash Hamming<=4",
              "counts": dict(Counter(r["split"] or "unassigned" for r in rows)), "near_pairs": near_pairs, "samples": rows}
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new split file to preserve the recorded partition")
    write_json(output, result)
    return result

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.fft import dctn

from .. import CLASSES
from ..utils import now, read_json, sha256, write_json
from .dataset import open_rgb, safe_path


def perceptual_hash(im):
    gray = np.asarray(im.convert("L").resize((32, 32)), dtype=float)
    values = dctn(gray, norm="ortho")[:8, :8].flatten()
    bits = values > np.median(values[1:])
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def audit(manifest, root, output="reports/dataset_audit", near_distance=4):
    rows = read_json(manifest)["samples"] if isinstance(manifest, (str, Path)) else manifest["samples"]
    errors, sizes, hashes, phashes = [], Counter(), defaultdict(list), []
    for i, r in enumerate(rows):
        try:
            if r.get("label") not in CLASSES:
                raise ValueError("invalid_label")
            if not r.get("group_id"):
                raise ValueError("missing_group_id")
            p = safe_path(root, r["path"])
            if not p.is_file():
                raise ValueError("invalid_path")
            if not p.stat().st_size:
                raise ValueError("zero_byte_image")
            im = open_rgb(p)
            im.load()
            sizes[f"{im.width}x{im.height}"] += 1
            digest = sha256(p)
            if r.get("sha256") and digest != r["sha256"]:
                raise ValueError("content_hash_mismatch")
            r["sha256"] = digest
            hashes[digest].append(i)
            phashes.append((i, perceptual_hash(im)))
        except Exception as e:
            errors.append({"index": i, "path": r.get("path"), "error": str(e)})
    exact = [v for v in hashes.values() if len(v) > 1]
    near = []
    # Full pairwise review: fine for v0.1 subset; no risk-hiding candidate subsampling.
    for pos, (i, h) in enumerate(phashes):
        for j, other in phashes[pos + 1 :]:
            if (h ^ other).bit_count() <= near_distance and rows[i]["sha256"] != rows[j]["sha256"]:
                near.append([i, j])
    leaks = []
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r.get("group_id")].append(i)
    for kind, sets in (("group", groups.values()), ("exact_duplicate", exact), ("near_duplicate", near)):
        for indices in sets:
            splits = {rows[i].get("split") for i in indices} - {None}
            if len(splits) > 1:
                leaks.append({"kind": kind, "indices": list(indices), "splits": sorted(splits)})
    labels = Counter(r.get("label", "INVALID") for r in rows)
    splits = Counter(r.get("split", "unsplit") for r in rows)
    result = {
        "created_at": now(),
        "total_images": len(rows),
        "labels": dict(labels),
        "class_ratios": {k: v / max(1, len(rows)) for k, v in labels.items()},
        "splits": dict(splits),
        "split_labels": {
            s: dict(Counter(r["label"] for r in rows if r.get("split", "unsplit") == s)) for s in splits
        },
        "dimensions": dict(sizes),
        "invalid_images": errors,
        "invalid_count": len(errors),
        "duplicate_hash_groups": exact,
        "near_duplicate_pairs": near,
        "near_hash": "64-bit DCT pHash",
        "near_distance": near_distance,
        "cross_split_leaks": leaks,
        "passed": not errors and not leaks,
        "limitation": "pHash is a heuristic; leaf IDs do not guarantee independent plant/scene/capture-session IDs.",
    }
    if output:
        write_json(str(output) + ".json", result)
        Path(str(output) + ".md").write_text(
            "# Dataset audit\n\n"
            + "\n".join(
                f"- {k}: {v}"
                for k, v in result.items()
                if k
                not in (
                    "near_duplicate_pairs",
                    "duplicate_hash_groups",
                    "invalid_images",
                    "cross_split_leaks",
                )
            )
            + "\n\nDetailed offending indices are in the JSON report.\n",
            encoding="utf-8",
        )
    return result, rows

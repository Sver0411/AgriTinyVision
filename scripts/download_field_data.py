"""Download a preselected external PlantDoc test subset with original labels and attribution."""

import argparse
import hashlib
import json
import random
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agritinyvision.pc_camera.metrics import write_json
from agritinyvision.utils import sha256

REVISION = "5467f6012d78d1c446145d5f582da6096f852ae8"
REPOSITORY = "https://github.com/pratikkayal/PlantDoc-Dataset"
CLASSES = {
    "Apple leaf": ("Apple", "NORMAL"), "Apple Scab Leaf": ("Apple", "SUSPECT"),
    "Apple rust leaf": ("Apple", "SUSPECT"), "Bell_pepper leaf": ("Bell pepper", "NORMAL"),
    "Bell_pepper leaf spot": ("Bell pepper", "SUSPECT"),
    "Potato leaf early blight": ("Potato", "SUSPECT"), "Potato leaf late blight": ("Potato", "SUSPECT"),
    "Tomato leaf": ("Tomato", "NORMAL"), "Tomato Early blight leaf": ("Tomato", "SUSPECT"),
    "Tomato Septoria leaf spot": ("Tomato", "SUSPECT"), "Tomato leaf bacterial spot": ("Tomato", "SUSPECT"),
    "Tomato leaf late blight": ("Tomato", "SUSPECT"), "Tomato leaf mosaic virus": ("Tomato", "SUSPECT"),
    "Tomato leaf yellow virus": ("Tomato", "SUSPECT"), "Tomato mold leaf": ("Tomato", "SUSPECT"),
}


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "AgriTinyVision-field-validation"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def download(output, per_class=4, seed=41):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    raw = f"https://raw.githubusercontent.com/pratikkayal/PlantDoc-Dataset/{REVISION}/"
    tree = json.loads(fetch(f"https://api.github.com/repos/pratikkayal/PlantDoc-Dataset/git/trees/{REVISION}?recursive=1"))
    if tree["truncated"]:
        raise ValueError("Incomplete source tree")
    buckets = defaultdict(list)
    for item in tree["tree"]:
        parts = item["path"].split("/")
        if item["type"] == "blob" and len(parts) == 3 and parts[0] == "test" and parts[1] in CLASSES:
            buckets[parts[1]].append(item)
    if set(buckets) != set(CLASSES):
        raise ValueError("Source class list differs from the fixed selection protocol")
    rng = random.Random(seed)
    chosen = []
    for category in sorted(buckets):
        items = sorted(buckets[category], key=lambda item: item["path"])
        rng.shuffle(items)
        chosen.extend(items[:per_class])
    write_json(output / "selection.json", {"revision": REVISION, "seed": seed, "per_class": per_class,
                                          "selection_before_inference": [i["path"] for i in chosen]})

    def save(item):
        source = item["path"]
        filename = hashlib.sha256(source.encode()).hexdigest()[:20] + Path(source).suffix.lower()
        url = raw + urllib.parse.quote(source, safe="/")
        content = fetch(url)
        blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        if blob != item["sha"]:
            raise ValueError("Git blob mismatch: " + source)
        (output / filename).write_bytes(content)
        crop, label = CLASSES[source.split("/")[1]]
        return {"path": filename, "sha256": sha256(output / filename), "source_url": url,
                "source_path": source, "source_label": source.split("/")[1], "label": label, "crop_type": crop,
                "git_blob_sha1": blob, "source_split": "test", "source_group": "PlantDoc-" + REVISION}

    with ThreadPoolExecutor(max_workers=4) as pool:
        samples = list(pool.map(save, chosen))
    (output / "LICENSE.txt").write_bytes(fetch(raw + "LICENSE.txt"))
    (output / "ATTRIBUTION.md").write_text(
        "# PlantDoc\n\nSingh, Jain, Jain, Kayal, Kumawat and Batra (2020), "
        "PlantDoc: A Dataset for Visual Plant Disease Detection.\n\n"
        + REPOSITORY + "\n\nRevision: " + REVISION + "\n\nRepository license: CC-BY-4.0. "
        "Original internet-image provenance and class labels are retained per sample.\n", encoding="utf-8")
    manifest = {"dataset": "PlantDoc external test subset", "repository": REPOSITORY, "revision": REVISION,
                "license": "CC-BY-4.0", "seed": seed, "per_class": per_class, "samples": samples,
                "label_basis": "independent dataset-author category; no project model predictions used",
                "quality_labels": "not supplied", "plant_session_ids": "not supplied"}
    write_json(output / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--per-class", type=int, default=4)
    p.add_argument("--seed", type=int, default=41)
    a = p.parse_args()
    if not 1 <= a.per_class <= 20:
        p.error("per-class must be between 1 and 20")
    print("Downloaded", len(download(a.output, a.per_class, a.seed)["samples"]), "source-labeled images")

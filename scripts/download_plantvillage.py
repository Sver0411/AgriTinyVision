"""Pinned raw-color subset with original leaf-group metadata, never augmented variants."""

import argparse
import hashlib
import json
import random
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "spMohanty/PlantVillage-Dataset"


def fetch(url):
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "AgriTinyVision/0.1"})
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataset.yaml")
    args = parser.parse_args()
    import yaml

    c = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    root = Path(c["root"])
    root.mkdir(parents=True, exist_ok=True)
    cache = Path("data/cache")
    cache.mkdir(parents=True, exist_ok=True)
    pin_path = cache / "source_revision.json"
    if c.get("revision"):
        revision = c["revision"]
    elif pin_path.exists():
        revision = json.loads(pin_path.read_text())["revision"]
    else:
        revision = json.loads(fetch(f"https://api.github.com/repos/{REPO}/commits/master"))["sha"]
        pin_path.write_text(json.dumps({"revision": revision}))
    raw = f"https://raw.githubusercontent.com/{REPO}/{revision}/"
    group_path = cache / f"leaf-map-{revision}.json"
    if not group_path.exists():
        group_path.write_bytes(fetch(raw + "leaf_grouping/leaf-map.json"))
    grouping = json.loads(group_path.read_text())
    print("Pinned dataset revision:", revision, "group map entries:", len(grouping), flush=True)
    listing_path = cache / f"color-directories-{revision}.json"
    if not listing_path.exists():
        listing = json.loads(fetch(f"https://api.github.com/repos/{REPO}/contents/raw?ref={revision}"))
        color_sha = next(r["sha"] for r in listing if r["name"] == "color")
        listing_path.write_bytes(fetch(f"https://api.github.com/repos/{REPO}/git/trees/{color_sha}"))
    tree = json.loads(listing_path.read_text())
    if tree.get("truncated"):
        raise RuntimeError("GitHub tree truncated; refusing incomplete dataset enumeration")
    mapping = yaml.safe_load(Path(c["label_mapping"]).read_text())["mapping"]
    candidates, unmatched = [], 0
    rng = random.Random(c["seed"])
    for label in c["source_classes"]:
        if label not in mapping:
            raise ValueError(f"Unmapped source class: {label}")
        class_sha = next(r["sha"] for r in tree["tree"] if r["path"] == label)
        class_cache = cache / f"{class_sha}.json"
        if not class_cache.exists():
            class_cache.write_bytes(fetch(f"https://api.github.com/repos/{REPO}/git/trees/{class_sha}"))
        class_tree = json.loads(class_cache.read_text())
        if class_tree.get("truncated"):
            raise RuntimeError(f"Truncated class listing: {label}")
        files = sorted(
            label + "/" + r["path"]
            for r in class_tree["tree"]
            if r["type"] == "blob" and r["path"].lower().endswith((".jpg", ".jpeg", ".png"))
        )
        rng.shuffle(files)
        selected = 0
        for path in files:
            filename = Path(path).name
            identifier = filename.replace("_final_masked", "").split("___")[-1].split("copy")[0]
            identifier = (
                identifier.replace(".jpg", "")
                .replace(".JPG", "")
                .replace(".png", "")
                .replace(".PNG", "")
                .lower()
                .strip()
            )
            suggestions = grouping.get(identifier, [])
            matched = [v for v in suggestions if v.split(":::")[0] == label]
            if len(matched) != 1:
                unmatched += 1
                continue
            value = matched[0]
            candidates.append(
                {
                    "path": path,
                    "source_label": label,
                    "label": mapping[label],
                    "group_id": value,
                    "source_group_id": value,
                }
            )
            selected += 1
            if selected >= c["max_per_class"]:
                break
        if not selected:
            raise RuntimeError(f"No leaf-grouped images available for {label}")
        print(label, selected, flush=True)

    def download(row):
        path = root / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            content = fetch(raw + "raw/color/" + urllib.parse.quote(row["path"]))
            path.write_bytes(content)
        content = path.read_bytes()
        row["sha256"] = hashlib.sha256(content).hexdigest()
        return row

    samples = []
    with ThreadPoolExecutor(max_workers=c["download_workers"]) as pool:
        for row in pool.map(download, candidates):
            samples.append(row)
            if len(samples) % 200 == 0:
                print(f"Downloaded {len(samples)}/{len(candidates)}", flush=True)
    manifest = {
        "dataset": "PlantVillage raw/color grouped subset",
        "version": revision,
        "source": f"https://github.com/{REPO}/tree/{revision}",
        "download_config": c,
        "grouping_sha256": hashlib.sha256(group_path.read_bytes()).hexdigest(),
        "unmatched_candidates_skipped": unmatched,
        "group_limitations": "Original leaf identity only; plant and capture-session IDs unavailable.",
        "samples": samples,
    }
    out = Path(c["raw_manifest"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Manifest:", out, flush=True)


if __name__ == "__main__":
    main()

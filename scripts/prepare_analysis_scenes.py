"""Create labeled-source controlled variations and synthetic probes locally, without tuning."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from agritinyvision.pc_camera.metrics import write_json
from agritinyvision.utils import sha256


def prepare_scenes(image, output, source, label, split):
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=False)
    with Image.open(image) as im:
        leaf = ImageOps.exif_transpose(im).convert("RGB")
    yy, xx = np.indices((160, 160))
    variations = {
        "normal_light": (leaf, "identity"),
        "low_light": (ImageEnhance.Brightness(leaf).enhance(0.25), "brightness x0.25"),
        "strong_light": (ImageEnhance.Brightness(leaf).enhance(1.8), "brightness x1.8"),
        "blurred": (leaf.filter(ImageFilter.GaussianBlur(5)), "Gaussian radius 5 original pixels"),
        "high_contrast": (ImageEnhance.Contrast(leaf).enhance(2), "contrast x2"),
        "low_contrast": (ImageEnhance.Contrast(leaf).enhance(0.2), "contrast x0.2"),
        "different_color": (Image.fromarray(np.asarray(leaf)[..., [1, 2, 0]]), "synthetic channel permutation GBR"),
        "non_plant": (Image.fromarray(np.stack((xx, yy, (xx + yy) % 160), axis=-1).astype(np.uint8)), "synthetic geometric gradient"),
        "checker": (Image.fromarray(np.repeat((((xx + yy) % 2) * 255)[..., None], 3, axis=2).astype(np.uint8)), "1-pixel checker"),
        "noise": (Image.fromarray(np.random.default_rng(303).integers(0, 256, (160, 160, 3), dtype=np.uint8)), "numpy seed 303 uniform uint8"),
    }
    samples = []
    for name, (im, transform) in variations.items():
        path = folder / (name + ".png")
        im.save(path)
        synthetic = name in ("non_plant", "checker", "noise")
        samples.append({"id": name, "scenario": name, "path": path.name, "source_sha256": sha256(path),
                        "parent_sha256": None if synthetic else sha256(image),
                        "source": "synthetic probe" if synthetic else source,
                        "source_label": None if synthetic else label, "source_split": "synthetic" if synthetic else split,
                        "annotation_valid": name == "normal_light", "transform": transform})
    result = {"version": "analysis-scenes-v1", "purpose": "controlled deployment probes, not field evaluation",
              "samples": samples}
    write_json(folder / "manifest.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--source", required=True, help="dataset/source identifier, not private path")
    p.add_argument("--label", required=True)
    p.add_argument("--split", required=True, choices=("train", "val", "independent"))
    a = p.parse_args()
    print(json.dumps(prepare_scenes(a.image, a.output, a.source, a.label, a.split)["samples"], indent=2))

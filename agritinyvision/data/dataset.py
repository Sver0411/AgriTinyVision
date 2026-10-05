from pathlib import Path

from PIL import Image, ImageOps
from torch.utils.data import Dataset

from .. import CLASSES
from ..utils import read_json


def map_label(source, mapping):
    if source not in mapping or mapping[source] not in CLASSES:
        raise ValueError(f"Unmapped or invalid source label: {source}")
    return mapping[source]


def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Image path escapes dataset root: {relative}")
    return path


def open_rgb(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


class ManifestDataset(Dataset):
    def __init__(self, manifest, root, split, transform):
        self.rows = [r for r in read_json(manifest)["samples"] if r["split"] == split]
        if not self.rows or {r["label"] for r in self.rows} != set(CLASSES):
            raise ValueError(f"{split} must contain both NORMAL and SUSPECT")
        self.root, self.transform = root, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return self.transform(open_rgb(safe_path(self.root, r["path"]))), CLASSES.index(r["label"])

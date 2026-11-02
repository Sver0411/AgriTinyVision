"""Only decode/orientation/BGR→RGB/bilinear-antialias resize on the PC."""

import numpy as np
from PIL import Image, ImageOps


def prepare_bgr(frame, rotation=0, mirror=False):
    return prepare_rgb(frame, "BGR", rotation, mirror)


def prepare_rgb(frame, color_order="RGB", rotation=0, mirror=False):
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        raise ValueError("Image must be uint8 RGB/BGR")
    if frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 32:
        raise ValueError("Camera image must have 3 channels and dimensions >=32")
    if rotation not in (0, 90, 180, 270):
        raise ValueError("rotation must be 0, 90, 180 or 270")
    if color_order not in ("RGB", "BGR") or not isinstance(mirror, bool):
        raise ValueError("Specify RGB/BGR color order and boolean mirror")
    rgb = Image.fromarray(frame[:, :, ::-1] if color_order == "BGR" else frame)
    if rotation:
        rgb = rgb.rotate(-rotation, expand=True)
    if mirror:
        rgb = ImageOps.mirror(rgb)
    # PIL is the same uint8 bilinear path used by torchvision's PIL Resize.
    return np.asarray(rgb.resize((160, 160), Image.Resampling.BILINEAR)).copy()


def load_image_bgr(path):
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB"))[:, :, ::-1].copy()

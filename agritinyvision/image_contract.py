"""Frozen RGB / ImageNet input contract shared by reference analysis and training."""

import numpy as np

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SIZE = 160


def normalize_rgb(rgb):
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.shape != (SIZE, SIZE, 3):
        raise ValueError("Expected 160x160 uint8 RGB")
    return (rgb.astype(np.float32) / np.float32(255) - np.array(MEAN, np.float32)) / np.array(STD, np.float32)

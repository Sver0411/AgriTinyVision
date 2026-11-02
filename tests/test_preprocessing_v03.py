import numpy as np
import pytest
from PIL import Image, ImageFilter

from agritinyvision.data.transforms import preprocessing
from agritinyvision.image_contract import normalize_rgb
from agritinyvision.pc_camera.image_preprocess import prepare_bgr, prepare_rgb
from agritinyvision.quality import load_quality, quality_gate


@pytest.mark.parametrize("shape", [(32, 64), (245, 321), (480, 640), (160, 160), (1000, 33)])
@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("mirror", [False, True])
def test_rgb_bgr_orientation_and_training_match(shape, rotation, mirror):
    a = np.random.default_rng(3).integers(0, 256, (*shape, 3), dtype=np.uint8)
    rgb = prepare_rgb(a, rotation=rotation, mirror=mirror)
    np.testing.assert_array_equal(rgb, prepare_bgr(a[..., ::-1], rotation, mirror))
    oriented = np.rot90(a, -(rotation // 90))
    if mirror:
        oriented = oriented[:, ::-1]
    expected = preprocessing(160)(Image.fromarray(oriented)).numpy()
    np.testing.assert_array_equal(normalize_rgb(rgb).transpose(2, 0, 1), expected)


@pytest.mark.parametrize("frame", [None, np.zeros((160, 160)), np.zeros((160, 160, 4), np.uint8),
                                  np.zeros((31, 160, 3), np.uint8), np.zeros((160, 160, 3), np.float32)])
def test_invalid_inputs(frame):
    with pytest.raises(ValueError):
        prepare_rgb(frame)


def test_quality_uses_same_bilinear_metrics_resolution():
    a = np.random.default_rng(2).integers(0, 256, (340, 420, 3), dtype=np.uint8)
    direct = quality_gate(Image.fromarray(a), load_quality())
    resized = quality_gate(Image.fromarray(prepare_rgb(a)), load_quality())
    assert direct == resized


@pytest.mark.parametrize("value", [0, 20, 128, 240, 255])
def test_constant_extreme_brightness_rejected(value):
    q = quality_gate(Image.fromarray(np.full((160, 160, 3), value, np.uint8)), load_quality())
    assert not q["passed"] and "low_contrast" in q["reasons"]


def test_blurred_image_rejected():
    a = np.random.default_rng(2).integers(0, 256, (160, 160, 3), dtype=np.uint8)
    q = quality_gate(Image.fromarray(a).filter(ImageFilter.GaussianBlur(10)), load_quality())
    assert not q["passed"] and "blur" in q["reasons"]

import pytest
from PIL import Image

from agritinyvision.data.audit import audit
from agritinyvision.utils import load_yaml, validate_config


def test_source_mapping_mismatch_is_rejected(tmp_path):
    Image.new("RGB", (40, 40)).save(tmp_path / "image.png")
    m = {
        "samples": [
            {"path": "image.png", "group_id": "1", "source_label": "Tomato___healthy", "label": "SUSPECT"}
        ]
    }
    r, _ = audit(m, tmp_path, output=None)
    assert r["invalid_count"] == 1
    assert r["invalid_images"][0]["error"] == "source_label_mapping_mismatch"


@pytest.mark.parametrize(
    "field,value", [("amp", "true"), ("scheduler", "unknown"), ("patience", 0), ("weight_decay", -1)]
)
def test_scalar_and_boolean_config_validation(field, value):
    c = load_yaml("configs/train_mobilenetv2_160.yaml")
    c[field] = value
    with pytest.raises(ValueError):
        validate_config(c)

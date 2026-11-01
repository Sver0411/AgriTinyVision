import copy
import json

import numpy as np
import pytest
from PIL import Image

from agritinyvision.data.audit import audit
from agritinyvision.data.dataset import map_label, open_rgb, safe_path
from agritinyvision.data.split import split_groups
from agritinyvision.data.transforms import preprocessing
from agritinyvision.quality import decide, load_quality, quality_gate
from agritinyvision.threshold import metrics, tune
from agritinyvision.utils import load_yaml, sha256, validate_config, write_json


@pytest.fixture
def tiny_data(tmp_path):
    rng = np.random.default_rng(41)
    samples = []
    for i in range(24):
        name = f"{i}.png"
        Image.fromarray(rng.integers(0, 256, (40, 48, 3), dtype=np.uint8)).save(tmp_path / name)
        samples.append(
            {"path": name, "label": "NORMAL" if i % 2 == 0 else "SUSPECT", "group_id": str(i // 2)}
        )
    return tmp_path, {"samples": samples}


def test_mapping_explicit_only():
    mapping = load_yaml("configs/label_mapping.yaml")["mapping"]
    assert map_label("Tomato___healthy", mapping) == "NORMAL"
    assert map_label("Tomato___Late_blight", mapping) == "SUSPECT"
    assert set(mapping.values()) == {"NORMAL", "SUSPECT"}
    with pytest.raises(ValueError):
        map_label("unseen", mapping)


@pytest.mark.parametrize("path", ["../escape.png", "../../evil.jpg"])
def test_path_escape(tmp_path, path):
    with pytest.raises(ValueError):
        safe_path(tmp_path, path)


def test_audit_invalid_inputs(tiny_data):
    root, m = tiny_data
    (root / "empty.png").touch()
    (root / "broken.png").write_text("not an image")
    for path, label in [
        ("empty.png", "NORMAL"),
        ("broken.png", "NORMAL"),
        ("absent.png", "NORMAL"),
        ("0.png", "UNKNOWN"),
    ]:
        m["samples"].append({"path": path, "label": label, "group_id": path})
    report, _ = audit(m, root, output=None)
    assert report["invalid_count"] == 4
    assert not report["passed"]


def test_duplicates_and_near_duplicates(tiny_data):
    root, m = tiny_data
    original = root / "0.png"
    (root / "duplicate.png").write_bytes(original.read_bytes())
    a = np.asarray(open_rgb(original)).copy()
    a[0, 0, 0] ^= 1
    Image.fromarray(a).save(root / "near.png")
    m["samples"] += [
        {"path": "duplicate.png", "label": "NORMAL", "group_id": "different"},
        {"path": "near.png", "label": "NORMAL", "group_id": "near"},
    ]
    report, rows = audit(m, root, output=None)
    assert [0, 24] in report["duplicate_hash_groups"]
    assert [0, 25] in report["near_duplicate_pairs"]
    split_groups(rows, report)
    assert rows[0]["split"] == rows[24]["split"] == rows[25]["split"]
    final, _ = audit({"samples": rows}, root, output=None)
    assert final["passed"]


def test_group_split_reproducible(tiny_data):
    root, m = tiny_data
    report, rows = audit(m, root, output=None)
    a = split_groups(copy.deepcopy(rows), report, seed=41)
    b = split_groups(copy.deepcopy(rows), report, seed=41)
    assert a == b
    assert {r["split"] for r in a} == {"train", "val", "test"}
    for group in {r["group_id"] for r in a}:
        assert len({r["split"] for r in a if r["group_id"] == group}) == 1


def test_leakage_is_a_failure(tiny_data):
    root, m = tiny_data
    m["samples"][0]["split"] = "train"
    m["samples"][1]["split"] = "test"
    report, _ = audit(m, root, output=None)
    assert not report["passed"]
    assert report["cross_split_leaks"][0]["kind"] == "group"


def test_content_hash_tamper(tiny_data):
    root, m = tiny_data
    m["samples"][0]["sha256"] = "0" * 64
    report, _ = audit(m, root, output=None)
    assert report["invalid_images"][0]["error"] == "content_hash_mismatch"


@pytest.mark.parametrize("size", [160, 224])
def test_preprocessing_rgb_and_normalization(size):
    im = Image.new("RGB", (80, 40), (255, 0, 0))
    x = preprocessing(size)(im)
    assert x.shape == (3, size, size)
    assert x[0].mean().item() == pytest.approx((1 - 0.485) / 0.229, abs=1e-5)
    assert x[1].mean().item() == pytest.approx(-0.456 / 0.224, abs=1e-5)


@pytest.mark.parametrize(
    "color,reason", [((0, 0, 0), "exposure"), ((255, 255, 255), "exposure"), ((128, 128, 128), "blur")]
)
def test_quality_unknown(color, reason):
    gate = quality_gate(Image.new("RGB", (160, 160), color), load_quality())
    assert reason in gate["reasons"]
    assert decide(0.99, 0.3, gate, 0.1)["state"] == "UNKNOWN"


def test_quality_valid_and_missing(tmp_path):
    im = Image.fromarray(np.random.default_rng(41).integers(20, 230, (160, 160, 3), dtype=np.uint8))
    assert quality_gate(im, load_quality())["passed"]
    assert not quality_gate(tmp_path / "missing.jpg", load_quality())["passed"]


@pytest.mark.parametrize(
    "p,state",
    [(0.01, "NORMAL"), (0.9, "SUSPECT"), (0.3, "UNKNOWN"), (float("nan"), "UNKNOWN"), (1.1, "UNKNOWN")],
)
def test_threshold_states(p, state):
    assert decide(p, 0.3, {"passed": True}, 0.1)["state"] == state


def test_metrics_and_validation_only_threshold():
    y, p = [0, 0, 1, 1], [0.1, 0.4, 0.3, 0.9]
    result = metrics(y, p, 0.5)
    assert result["confusion_matrix"] == [[2, 0], [1, 1]]
    assert result["suspect_recall"] == 0.5
    assert result["false_negative_rate"] == 0.5
    settings = load_yaml("configs/train_mobilenetv2_160.yaml")["threshold"]
    found = tune(y, p, settings)
    assert found["selected"]["suspect_recall"] == 1.0
    with pytest.raises(ValueError, match="validation"):
        tune(y, p, settings, split="test")


@pytest.mark.parametrize(
    "field,value",
    [("input_size", 512), ("epochs", 0), ("batch_size", -1), ("optimizer", "magic"), ("width_mult", 0.35)],
)
def test_config_invalid(field, value):
    c = load_yaml("configs/train_mobilenetv2_160.yaml")
    c[field] = value
    with pytest.raises(ValueError):
        validate_config(c)


def test_configs_valid():
    for size in (160, 224):
        assert validate_config(load_yaml(f"configs/train_mobilenetv2_{size}.yaml"))["input_size"] == size


def test_onnx_numerical_consistency(tmp_path):
    from agritinyvision.export_onnx import export_model
    from agritinyvision.models.mobilenetv2 import build

    c = load_yaml("configs/train_mobilenetv2_160.yaml")
    c.update(width_mult=0.35, pretrained=False)
    result = export_model(build(c), 160, tmp_path / "fixture.onnx")
    assert result["passed"] and result["inputs_checked"] == 4
    assert result["compatibility"]["static_review_passed"]


def release_fixture(tmp_path, monkeypatch):
    import agritinyvision.train as train_module

    cp = tmp_path / "best.pt"
    cp.write_bytes(b"Synthetic test fixture only")
    model = tmp_path / "fixture.onnx"
    model.write_bytes(b"Synthetic ONNX transport fixture only")
    write_json(
        str(model) + ".validation.json",
        {"passed": True, "sha256": sha256(model), "checkpoint_sha256": sha256(cp)},
    )
    config = {"input_size": 160}
    monkeypatch.setattr(
        train_module, "load_checkpoint", lambda *args, **kwargs: (None, {"config": config, "threshold": 0.3})
    )
    write_json(
        tmp_path / "environment.json",
        {
            "git_commit": None,
            "dataset_version": "synthetic-test",
            "dataset_hash": "a" * 64,
            "train_split_hash": "b" * 64,
            "pretrained_weights": None,
        },
    )
    test = tmp_path / "test.json"
    write_json(
        test,
        {
            "split": "test",
            "threshold_source": "validation",
            "decision_threshold": 0.3,
            "model_sha256": sha256(cp),
            "decision_threshold_source": "validation",
            "final_metrics_split": "test",
            "frozen_checkpoint_sha256": sha256(cp),
            "frozen_model_sha256": sha256(cp),
            "system_gate": {"quality_config": load_quality()},
        },
    )
    return cp, model, test


def test_release_pending_has_no_fake_espdl(tmp_path, monkeypatch):
    from agritinyvision.release import release, validate_metadata

    cp, model, test = release_fixture(tmp_path, monkeypatch)
    out = release(cp, model, test, output=tmp_path / "release")
    assert not (out / "tinyvision.espdl").exists()
    metadata = json.loads((out / "metadata.json").read_text())
    validate_metadata(metadata)
    assert metadata["espdl_export"] == "PENDING"
    assert metadata["decision_threshold_source"] == "validation"
    assert metadata["final_metrics_split"] == "test"
    assert metadata["quality_gate"]["abstention_margin"] == 0.1
    assert (out / "labels.txt").read_text().splitlines() == ["NORMAL", "SUSPECT"]
    metadata["classes"] = ["NORMAL", "SUSPECT", "UNKNOWN"]
    with pytest.raises(Exception):
        validate_metadata(metadata)


def test_release_rejects_tampered_validation(tmp_path, monkeypatch):
    from agritinyvision.release import release

    cp, model, test = release_fixture(tmp_path, monkeypatch)
    model.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="stale"):
        release(cp, model, test, output=tmp_path / "release")


def test_release_rejects_wrong_checkpoint_metrics(tmp_path, monkeypatch):
    from agritinyvision.release import release

    cp, model, test = release_fixture(tmp_path, monkeypatch)
    m = json.loads(test.read_text())
    m["model_sha256"] = "0" * 64
    write_json(test, m)
    with pytest.raises(ValueError):
        release(cp, model, test, output=tmp_path / "release")


def test_calibration_only_train_round_robin():
    from agritinyvision.quantize import calibration_indices

    class FakeDataset:
        rows = [{"source_label": "a", "split": "train"}] * 5 + [{"source_label": "b", "split": "train"}] * 5

        def __len__(self):
            return len(self.rows)

    d = FakeDataset()
    result = calibration_indices(d, 6, 41)
    assert len(result) == len(set(result)) == 6
    assert sum(i < 5 for i in result) == 3

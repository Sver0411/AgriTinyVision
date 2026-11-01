import copy
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from test_pipeline import release_fixture

from agritinyvision.data.audit import audit
from agritinyvision.evaluate import validate_final_request
from agritinyvision.quality import abstention_bounds, decide, load_quality
from agritinyvision.quantize import quantize, validate_quantization_report
from agritinyvision.utils import sha256, write_json


@pytest.mark.parametrize(
    "p,state",
    [
        (0.05, "NORMAL"),
        (0.19, "NORMAL"),
        (0.20, "UNKNOWN"),
        (0.21, "UNKNOWN"),
        (0.30, "UNKNOWN"),
        (0.35, "UNKNOWN"),
        (0.39, "UNKNOWN"),
        (0.40, "SUSPECT"),
        (0.41, "SUSPECT"),
        (0.45, "SUSPECT"),
        (0.90, "SUSPECT"),
    ],
)
def test_threshold_centered_intervals(p, state):
    result = decide(p, 0.30, {"passed": True}, 0.10)
    assert result["state"] == state
    assert "confidence" not in result


def test_dense_monotonicity_and_float32_boundaries():
    order = {"NORMAL": 0, "UNKNOWN": 1, "SUSPECT": 2}
    states = [order[decide(p, 0.3, {"passed": True}, 0.1)["state"]] for p in np.linspace(0, 1, 10001)]
    assert states == sorted(states) and set(states) == {0, 1, 2}
    lower, upper = map(np.float32, abstention_bounds(0.3, 0.1))
    assert decide(np.nextafter(lower, np.float32(0)), 0.3, {"passed": True}, 0.1)["state"] == "NORMAL"
    assert decide(lower, 0.3, {"passed": True}, 0.1)["state"] == "UNKNOWN"
    assert decide(np.nextafter(upper, np.float32(0)), 0.3, {"passed": True}, 0.1)["state"] == "UNKNOWN"
    assert decide(upper, 0.3, {"passed": True}, 0.1)["state"] == "SUSPECT"


@pytest.mark.parametrize("p", [float("nan"), float("inf"), -float("inf"), -0.01, 1.01, None, "bad"])
def test_invalid_probability_is_unknown(p):
    assert decide(p, 0.3, {"passed": True}, 0.1)["reason"] == "invalid_model_output"


@pytest.mark.parametrize("p", [0, 0.3, 1, float("nan"), None])
def test_quality_failure_has_priority(p):
    assert decide(p, 0.3, {"passed": False}, 0.1)["reason"] == "quality_gate"


@pytest.mark.parametrize(
    "threshold,margin", [(0.3, -0.1), (0.3, 0.4), (float("nan"), 0.1), (0.3, float("inf"))]
)
def test_invalid_abstention_configuration(threshold, margin):
    with pytest.raises(ValueError):
        abstention_bounds(threshold, margin)


def test_old_confidence_config_is_rejected(tmp_path):
    config = load_quality()
    config["min_confidence"] = 0.6
    path = tmp_path / "quality.yaml"
    write_json(path, config)  # JSON is also valid YAML.
    with pytest.raises(ValueError):
        load_quality(path)


def test_quantization_test_is_refused_before_any_file_or_data_access():
    with pytest.raises(ValueError, match="only val"):
        quantize("absent.onnx", "absent.pt", evaluation_split="test")


def test_quantization_default_reads_only_train_and_val(tmp_path, monkeypatch):
    import onnxruntime.quantization as ortq
    import torch

    import agritinyvision.quantize as module

    manifest = tmp_path / "manifest.json"
    manifest.write_text("synthetic only")
    source, checkpoint = tmp_path / "model.onnx", tmp_path / "best.pt"
    source.write_bytes(b"synthetic transport")
    checkpoint.write_bytes(b"synthetic checkpoint")
    write_json(
        str(source) + ".validation.json",
        {
            "passed": True,
            "sha256": sha256(source),
            "checkpoint_sha256": sha256(checkpoint),
        },
    )
    config = {"manifest": str(manifest), "dataset_root": str(tmp_path), "input_size": 160, "seed": 41}
    monkeypatch.setattr(
        module,
        "load_checkpoint",
        lambda *a: (
            None,
            {
                "config": config,
                "manifest_sha256": sha256(manifest),
                "threshold": 0.3,
            },
        ),
    )
    visited = []

    class Dataset:
        def __init__(self, manifest, root, split, transform):
            visited.append(split)
            assert split in {"train", "val"}
            self.rows = [{"source_label": "A", "path": "a.png"}, {"source_label": "B", "path": "b.png"}]

        def __len__(self):
            return 2

        def __getitem__(self, i):
            return torch.zeros(3, 160, 160), i

    monkeypatch.setattr(module, "ManifestDataset", Dataset)
    monkeypatch.setattr(
        ortq, "quantize_static", lambda source, out, *a, **k: Path(out).write_bytes(b"quantized transport")
    )
    monkeypatch.setattr(module, "predict_quantized", lambda *a: ([0, 1], [0.1, 0.9], "synthetic scores only"))
    out = quantize(source, checkpoint, backend="onnx")
    report = json.loads(Path(str(out) + ".quantization.json").read_text())
    assert visited == ["train", "val"]
    assert report["validation_metrics"]["split"] == "val"
    assert report["quantization_selection_split"] == "validation"
    assert "test_metrics" not in report
    validate_quantization_report(out, checkpoint, 0.3)


def test_final_test_requires_explicit_matching_frozen_hashes(tmp_path):
    checkpoint, model = tmp_path / "best.pt", tmp_path / "model.espdl"
    checkpoint.write_bytes(b"cp")
    model.write_bytes(b"model")
    with pytest.raises(ValueError, match="checkpoint"):
        validate_final_request(checkpoint, model, True, None, None)
    with pytest.raises(ValueError, match="model"):
        validate_final_request(checkpoint, model, True, sha256(checkpoint), "0" * 64)
    validate_final_request(checkpoint, model, True, sha256(checkpoint), sha256(model))
    with pytest.raises(ValueError, match="explicit"):
        validate_final_request(checkpoint, model, False, sha256(checkpoint), sha256(model))


def quantized_release_fixture(tmp_path, monkeypatch):
    cp, onnx, test = release_fixture(tmp_path, monkeypatch)
    artifact = tmp_path / "model.espdl"
    artifact.write_bytes(b"Synthetic ESP-DL transport fixture only")
    report = {
        "backend": "espdl",
        "status": "SUCCESS",
        "target": "esp32s3",
        "sha256": sha256(artifact),
        "source_sha256": sha256(onnx),
        "checkpoint_sha256": sha256(cp),
        "decision_threshold_source": "validation",
        "quantization_selection_split": "validation",
        "validation_metrics": {
            "split": "val",
            "threshold_source": "validation",
            "decision_threshold": 0.3,
            "model_sha256": sha256(artifact),
        },
    }
    write_json(str(artifact) + ".quantization.json", report)
    final = {
        "split": "test",
        "final_metrics_split": "test",
        "decision_threshold_source": "validation",
        "threshold_source": "validation",
        "decision_threshold": 0.3,
        "quantization_selection_split": "validation",
        "model_sha256": sha256(artifact),
        "checkpoint_sha256": sha256(cp),
        "frozen_checkpoint_sha256": sha256(cp),
        "frozen_model_sha256": sha256(artifact),
        "quantization_report_sha256": sha256(str(artifact) + ".quantization.json"),
        "system_gate": {"quality_config": load_quality()},
    }
    final_path = tmp_path / "int8_test.json"
    write_json(final_path, final)
    return cp, onnx, test, artifact, final_path


def test_release_records_new_provenance_and_intervals(tmp_path, monkeypatch):
    from agritinyvision.release import release, validate_metadata

    cp, onnx, test, artifact, final_path = quantized_release_fixture(tmp_path, monkeypatch)
    out = release(cp, onnx, test, artifact, output=tmp_path / "release", quantized_metrics_path=final_path)
    metadata = json.loads((out / "metadata.json").read_text())
    assert metadata["decision_threshold_source"] == "validation"
    assert metadata["quantization_selection_split"] == "validation"
    assert metadata["final_metrics_split"] == "test"
    assert metadata["state_logic"]["normal_upper_bound"] == float(np.float32(0.2))
    assert metadata["state_logic"]["suspect_lower_bound"] == float(np.float32(0.4))
    metadata["state_logic"]["normal_upper_bound"] = 0.25
    with pytest.raises(ValueError, match="boundaries"):
        validate_metadata(metadata)


@pytest.mark.parametrize(
    "field,value",
    [
        ("quantization_selection_split", "test"),
        ("model_sha256", "0" * 64),
        ("quantization_report_sha256", "0" * 64),
        ("decision_threshold", 0.5),
        ("final_metrics_split", "val"),
        ("frozen_checkpoint_sha256", "0" * 64),
    ],
)
def test_release_rejects_bad_quantized_final_provenance(tmp_path, monkeypatch, field, value):
    from agritinyvision.release import release

    cp, onnx, test, artifact, final_path = quantized_release_fixture(tmp_path, monkeypatch)
    final = json.loads(final_path.read_text())
    final[field] = value
    write_json(final_path, final)
    with pytest.raises(ValueError, match="Quantized final test"):
        release(cp, onnx, test, artifact, output=tmp_path / "release", quantized_metrics_path=final_path)


@pytest.mark.parametrize(
    "kind", ["test_selection", "historical_test_report", "artifact_changed", "no_final_metrics"]
)
def test_release_rejects_quantization_selection_or_missing_evidence(tmp_path, monkeypatch, kind):
    from agritinyvision.release import release

    cp, onnx, test, artifact, final_path = quantized_release_fixture(tmp_path, monkeypatch)
    path = str(artifact) + ".quantization.json"
    report = json.loads(Path(path).read_text())
    if kind == "test_selection":
        report["quantization_selection_split"] = "test"
    elif kind == "historical_test_report":
        report["test_metrics"] = {"split": "test"}
    elif kind == "artifact_changed":
        artifact.write_bytes(b"changed")
    else:
        final_path = None
    write_json(path, report)
    with pytest.raises(ValueError):
        release(cp, onnx, test, artifact, output=tmp_path / "release", quantized_metrics_path=final_path)


def test_source_abc_counts_and_missing_class_failure(tmp_path, monkeypatch):
    import agritinyvision.data.audit as module

    monkeypatch.setattr(
        module, "load_yaml", lambda *a: {"mapping": {"A": "NORMAL", "B": "SUSPECT", "C": "SUSPECT"}}
    )
    rows = []
    rng = np.random.default_rng(41)
    for split in ("train", "val", "test"):
        for source in ("A", "B", "C"):
            name = f"{split}_{source}.png"
            Image.fromarray(rng.integers(0, 256, (40, 48, 3), dtype=np.uint8)).save(tmp_path / name)
            rows.append(
                {
                    "path": name,
                    "label": "NORMAL" if source == "A" else "SUSPECT",
                    "source_label": source,
                    "split": split,
                    "group_id": name,
                }
            )
    report, _ = audit({"samples": rows}, tmp_path, output=tmp_path / "audit")
    assert report["passed"]
    assert report["split_source_labels"] == {s: {"A": 1, "B": 1, "C": 1} for s in ("train", "val", "test")}
    assert "| C | 1 | 1 | 1 |" in (tmp_path / "audit.md").read_text()
    incomplete = copy.deepcopy(rows[:-1])
    report, _ = audit({"samples": incomplete}, tmp_path, output=None)
    assert not report["passed"]
    assert report["missing_split_source_labels"] == [{"split": "test", "source_label": "C"}]
    assert report["warnings"]


def test_evaluate_default_validation_and_explicit_frozen_final(tmp_path, monkeypatch):
    import agritinyvision.evaluate as module

    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("synthetic manifest")
    config = {
        "manifest": str(manifest),
        "dataset_root": str(tmp_path),
        "input_size": 160,
        "batch_size": 2,
        "workers": 0,
    }
    monkeypatch.setattr(module, "device", lambda *a: "cpu")
    monkeypatch.setattr(
        module,
        "load_checkpoint",
        lambda *a: (
            None,
            {
                "config": config,
                "manifest_sha256": sha256(manifest),
                "threshold": 0.3,
            },
        ),
    )
    visited = []
    rng = np.random.default_rng(41)
    for i in range(2):
        Image.fromarray(rng.integers(20, 230, (160, 160, 3), dtype=np.uint8)).save(tmp_path / f"{i}.png")

    class Dataset:
        def __init__(self, manifest, root, split, transform):
            visited.append(split)
            self.rows = [{"path": "0.png"}, {"path": "1.png"}]

    monkeypatch.setattr(module, "ManifestDataset", Dataset)
    monkeypatch.setattr(module, "predict", lambda *a: ([0, 1], [0.1, 0.35]))
    validation = module.evaluate(checkpoint)
    assert validation["split"] == "val" and visited == ["val"]
    final = module.evaluate(checkpoint, final=True, frozen_checkpoint_sha256=sha256(checkpoint))
    assert visited == ["val", "test"]
    assert final["final_metrics_split"] == "test"
    assert final["system_gate"]["unknown_count"] == 1
    assert (tmp_path / "val_metrics.json").is_file() and (tmp_path / "test_metrics.json").is_file()
    with pytest.raises(ValueError, match="already exists"):
        module.evaluate(checkpoint, final=True, frozen_checkpoint_sha256=sha256(checkpoint))

import copy
import hashlib

import numpy as np
import pytest
from test_pc_camera import device, result

from agritinyvision.field_validation.annotations import save_annotation
from agritinyvision.field_validation.capture import FieldBatch
from agritinyvision.field_validation.evaluation import binary_metrics, evaluate_records
from agritinyvision.field_validation.reporting import REPORT_FILES, generate_batch_report
from agritinyvision.field_validation.split import split_batches


def row(truth, state, p, quality=True, human_quality="USABLE"):
    return {"sample_id": f"{truth}-{state}-{p}", "metadata": {"context": {"crop_type": "Apple"}},
            "prediction": {"status": "ok"}, "annotation": {"plant_presence": "YES", "visual_label": truth,
            "judgment_certain": True, "image_quality": human_quality, "label_origin": "manual"},
            "result": {"state": state, "suspect_probability": p, "quality_passed": quality}}


def cohort():
    return [row("NORMAL", "NORMAL", .1), row("NORMAL", "SUSPECT", .8, human_quality="BLUR"),
            row("SUSPECT", "NORMAL", .1), row("SUSPECT", "SUSPECT", .9),
            row("NORMAL", "UNKNOWN", .25), row("SUSPECT", "UNKNOWN", .25),
            row("SUSPECT", "UNKNOWN", None, False, "TOO_DARK")]


def test_exact_binary_and_abstention_denominators():
    evaluated = evaluate_records(cohort())
    s = evaluated["three_state"]
    assert s["three_state_matrix"] == [[1, 1, 1], [1, 1, 2]]
    assert s["binary_probability_cohort"]["confusion_matrix"] == [[2, 1], [2, 1]]
    assert s["accepted"]["accuracy"] == .5
    assert s["accepted_coverage"] == pytest.approx(4 / 7)
    assert s["unknown_rejection_rate"] == pytest.approx(3 / 7)
    assert s["system_correct_fraction"] == pytest.approx(2 / 7)
    assert s["suspect_missed_as_normal_rate"] == .25
    assert s["suspect_deferred_unknown_rate"] == .5
    assert s["unknown_analysis"]["boundary_forced_binary_errors_intercepted"] == 1
    assert s["unknown_analysis"]["boundary_forced_binary_correct_withheld"] == 1


def test_quality_reference_is_independent_and_missing_labels_are_pending():
    q = evaluate_records(cohort())["quality"]
    assert q["confusion_matrix"] == [[5, 0], [1, 1]]
    assert q["unusable_false_acceptance_rate"] == .5
    rows = cohort()
    for r in rows:
        r["annotation"]["image_quality"] = ""
    q = evaluate_records(rows)["quality"]
    assert q["status"] == "PENDING" and q["n"] == 0
    assert q["unusable_false_acceptance_rate"] is None


def test_non_target_uncertain_and_execution_errors_are_not_classification_truth():
    rows = cohort()
    non_target = row("NOT_APPLICABLE", "NORMAL", .1)
    non_target["annotation"]["plant_presence"] = "NO"
    uncertain = row("NORMAL", "NORMAL", .1)
    uncertain["annotation"]["judgment_certain"] = False
    error = copy.deepcopy(uncertain)
    error.update(sample_id="error", prediction={"status": "error", "error": "USB unplugged"}, result=None)
    excluded = copy.deepcopy(rows[0])
    excluded["metadata"]["context"]["excluded_from_evaluation"] = True
    r = evaluate_records(rows + [non_target, uncertain, error, excluded])
    assert r["three_state"]["reference_samples"] == 7
    assert r["three_state"]["non_target"] == {"n": 1, "accepted_fraction": 1}
    assert r["dataset"]["excluded"] == 1
    assert len(r["execution_errors"]) == 1
    assert r["scenarios"]["lighting"]["UNRECORDED"]["samples"] == 9
    assert r["scenarios"]["lighting"]["UNRECORDED"]["support"] == "unrecorded"


def test_empty_denominators_are_unmeasured():
    s = evaluate_records([])
    assert binary_metrics([])["accuracy"] is None
    assert s["three_state"]["accepted_coverage"] is None
    assert s["quality"]["status"] == "PENDING"


def test_author_class_labels_do_not_invent_manual_dimensions():
    source = row("NORMAL", "NORMAL", .1)
    source["annotation"] = {"label_origin": "dataset_author", "reference_eligible": True,
                            "visual_label": "NORMAL", "plant_presence": "", "judgment_certain": None,
                            "image_quality": ""}
    r = evaluate_records([source])
    assert r["three_state"]["binary_probability_cohort"]["accuracy"] == 1
    assert r["quality"]["status"] == "PENDING"
    assert r["three_state"]["valid_input_coverage"]["n"] == 0


def make_batch(root, seed, plant):
    batch = FieldBatch(root, device(), {}, {"plant_id": plant}, "simulated_hardware")
    rgb = np.random.default_rng(seed).integers(0, 256, (160, 160, 3), dtype=np.uint8)
    sid = batch.begin(rgb[..., ::-1], rgb)
    r = result(p=.9)
    r.update(rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(), end_to_end_ms=4, saturated_values=100)
    batch.complete(sid, r)
    return batch, sid


def test_reports_generated_without_inventing_manual_or_parity_labels(tmp_path):
    batch, sid = make_batch(tmp_path, 1, "plant-a")
    first = generate_batch_report(batch.path)
    assert first["dataset"]["labelled"] == 0
    save_annotation(batch.path, sid, {"plant_presence": "YES", "image_quality": "USABLE", "visual_label": "NORMAL",
                    "judgment_certain": True, "expert_confirmed": False})
    second = generate_batch_report(batch.path)
    assert second["three_state"]["accepted"]["accuracy"] == 0
    expected = REPORT_FILES - {"INFERENCE_PARITY.md"}
    assert expected <= {p.name for p in (batch.path / "reports").iterdir()}


def test_grouped_split_no_adjacent_or_duplicate_leakage_and_fixed_seed(tmp_path):
    a, _ = make_batch(tmp_path, 1, "plant-a")
    b, _ = make_batch(tmp_path, 2, "plant-b")
    c, _ = make_batch(tmp_path, 3, "plant-c")
    first = split_batches([a.path, b.path, c.path], tmp_path / "split1.json")
    second = split_batches([a.path, b.path, c.path], tmp_path / "split2.json")
    assert first == second and first["status"] == "READY"
    assert len(first["counts"]) == 3
    with pytest.raises(FileExistsError):
        split_batches([a.path], tmp_path / "split1.json")
    duplicate, _ = make_batch(tmp_path, 1, "different-plant-name")
    merged = split_batches([a.path, duplicate.path], tmp_path / "merged.json")
    assert merged["independent_groups"] == 1 and merged["counts"] == {"unassigned": 2}
    assert all(r["split"] is None for r in merged["samples"])

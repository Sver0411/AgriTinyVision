"""Binary, selective three-state and independently labelled quality metrics."""

from collections import Counter
from pathlib import Path

import numpy as np

from .annotations import QUALITY, labels
from .capture import read_json, read_samples, verified_sample

STATES = ("NORMAL", "SUSPECT", "UNKNOWN")
SCENARIOS = ("lighting", "distance", "angle", "background", "crop_type", "abnormality_severity")


def fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def binary_metrics(pairs):
    matrix = [[0, 0], [0, 0]]
    for truth, predicted in pairs:
        matrix[int(truth == "SUSPECT")][int(predicted == "SUSPECT")] += 1
    (tn, fp), (fn, tp) = matrix
    return {"n": len(pairs), "confusion_matrix": matrix, "axes": ["NORMAL", "SUSPECT"],
            "accuracy": fraction(tp + tn, len(pairs)), "precision": fraction(tp, tp + fp),
            "recall": fraction(tp, tp + fn), "f1": fraction(2 * tp, 2 * tp + fp + fn),
            "specificity": fraction(tn, tn + fp), "false_positive_rate": fraction(fp, tn + fp),
            "false_negative_rate": fraction(fn, tp + fn)}


def load_records(batch, annotation_source="manual"):
    batch = Path(batch)
    assigned = labels(batch, annotation_source)
    records = []
    for sample in read_samples(batch):
        metadata, _ = verified_sample(batch, sample)
        predfile = batch / sample["prediction"]
        prediction = read_json(predfile) if predfile.exists() else {"status": "pending"}
        r = prediction.get("result")
        if r and r["rgb_sha256"] != metadata["rgb_sha256"]:
            raise ValueError("Prediction and input RGB hash differ: " + sample["sample_id"])
        records.append({"sample_id": sample["sample_id"], "metadata": metadata, "prediction": prediction,
                        "annotation": assigned.get(sample["sample_id"], {}), "result": r})
    return records


def reference(row):
    a = row["annotation"]
    if a.get("label_origin") == "dataset_author" and a.get("reference_eligible") is True:
        return a.get("visual_label") if a.get("visual_label") in STATES[:2] else None
    return a.get("visual_label") if a.get("plant_presence") == "YES" and a.get("judgment_certain") is True \
        and a.get("visual_label") in STATES[:2] else None


def classification(rows):
    labelled = [r for r in rows if reference(r)]
    raw = [(reference(r), "SUSPECT" if np.float32(r["result"]["suspect_probability"]) >= np.float32(.30)
            else "NORMAL") for r in labelled if r["result"]["suspect_probability"] is not None]
    accepted = [r for r in labelled if r["result"]["state"] in STATES[:2]]
    matrix = [[0, 0, 0], [0, 0, 0]]
    for r in labelled:
        matrix[int(reference(r) == "SUSPECT")][STATES.index(r["result"]["state"])] += 1
    n, ns, nn = len(labelled), sum(matrix[1]), sum(matrix[0])
    rejected = [r for r in labelled if r["result"]["state"] == "UNKNOWN"]
    usable = [r for r in labelled if r["annotation"].get("image_quality") == "USABLE"]
    boundary = [r for r in rejected if r["result"]["quality_passed"] and r["result"]["suspect_probability"] is not None]
    forced_boundary = [(reference(r), "SUSPECT" if np.float32(r["result"]["suspect_probability"]) >= np.float32(.3)
                        else "NORMAL") for r in boundary]
    quality_pairs = [(reference(r), "SUSPECT" if np.float32(r["result"]["suspect_probability"]) >= np.float32(.3)
                      else "NORMAL") for r in labelled if r["result"]["quality_passed"] and r["result"]["suspect_probability"] is not None]
    return {"reference_samples": n, "binary_probability_cohort": binary_metrics(raw),
            "quality_passed_binary": binary_metrics(quality_pairs),
            "accepted": binary_metrics([(reference(r), r["result"]["state"]) for r in accepted]),
            "three_state_matrix": matrix, "truth_axes": list(STATES[:2]), "prediction_axes": list(STATES),
            "accepted_coverage": fraction(len(accepted), n), "unknown_rejection_rate": fraction(len(rejected), n),
            "system_correct_fraction": fraction(matrix[0][0] + matrix[1][1], n),
            "accepted_error_rate": fraction(sum(reference(r) != r["result"]["state"] for r in accepted), len(accepted)),
            "suspect_missed_as_normal_rate": fraction(matrix[1][0], ns),
            "suspect_deferred_unknown_rate": fraction(matrix[1][2], ns),
            "suspect_not_flagged_rate": fraction(matrix[1][0] + matrix[1][2], ns),
            "normal_false_alarm_rate": fraction(matrix[0][1], nn),
            "valid_input_coverage": {"n": len(usable), "rate": fraction(sum(r["result"]["state"] != "UNKNOWN" for r in usable), len(usable))},
            "unknown_analysis": {"count": len(rejected),
                "quality_rejected": sum(r["result"]["quality_passed"] is False for r in rejected),
                "probability_rejected": len(boundary),
                "other_rejected": sum(r["result"]["quality_passed"] is not False and r not in boundary for r in rejected),
                "boundary_forced_binary_errors_intercepted": sum(a != b for a, b in forced_boundary),
                "boundary_forced_binary_correct_withheld": sum(a == b for a, b in forced_boundary),
                "forced_binary_error_rate_probability_cohort": 1 - binary_metrics(raw)["accuracy"] if raw else None,
                "quality_rejection_counterfactual": "not observed: device model was not executed"}}


def quality_metrics(rows):
    reviewed = [r for r in rows if r["annotation"].get("image_quality") in QUALITY]
    matrix = [[0, 0], [0, 0]]  # human usable/unusable × gate pass/reject
    for r in reviewed:
        matrix[int(r["annotation"]["image_quality"] != "USABLE")][int(not r["result"]["quality_passed"])] += 1
    per_kind = {}
    for label in QUALITY:
        chosen = [r for r in reviewed if r["annotation"]["image_quality"] == label]
        per_kind[label] = {"n": len(chosen), "rejected": sum(not r["result"]["quality_passed"] for r in chosen),
                           "rejection_rate": fraction(sum(not r["result"]["quality_passed"] for r in chosen), len(chosen))}
    return {"status": "MEASURED" if reviewed else "PENDING", "n": len(reviewed), "confusion_matrix": matrix,
            "human_axes": ["USABLE", "UNUSABLE"], "gate_axes": ["PASS", "REJECT"], "by_quality": per_kind,
            "usable_false_rejection_rate": fraction(matrix[0][1], sum(matrix[0])),
            "unusable_false_acceptance_rate": fraction(matrix[1][0], sum(matrix[1]))}


def evaluate_records(records, min_scenario_samples=5):
    excluded = [r for r in records if r["metadata"]["context"].get("excluded_from_evaluation")]
    rows = [r for r in records if r["prediction"]["status"] == "ok" and r not in excluded]
    result = classification(rows)
    result["states_all_successful"] = {s: {"count": sum(r["result"]["state"] == s for r in rows),
                                          "fraction": fraction(sum(r["result"]["state"] == s for r in rows), len(rows))} for s in STATES}
    non_target = [r for r in rows if r["annotation"].get("plant_presence") == "NO"]
    result["non_target"] = {"n": len(non_target), "accepted_fraction": fraction(
        sum(r["result"]["state"] != "UNKNOWN" for r in non_target), len(non_target))}
    scenarios = {}
    for dimension in SCENARIOS:
        buckets = {}
        for row in rows:
            value = row["annotation"].get(dimension) or row["metadata"]["context"].get(dimension) or "UNRECORDED"
            buckets.setdefault(value, []).append(row)
        scenarios[dimension] = {value: {"samples": len(group), "support": "unrecorded" if value == "UNRECORDED" else
                                        "meets_minimum_count" if sum(bool(reference(r)) for r in group) >= min_scenario_samples else "insufficient",
                                        **classification(group)} for value, group in sorted(buckets.items())}
    return {"dataset": {"samples": len(records), "successful": len(rows), "excluded": len(excluded),
                        "prediction_status": dict(Counter(r["prediction"]["status"] for r in records)),
                        "labelled": sum(bool(reference(r)) for r in rows),
                        "annotation_origins": dict(Counter(r["annotation"].get("label_origin", "unlabelled") for r in records))},
            "three_state": result, "quality": quality_metrics(rows), "scenarios": scenarios,
            "failures": [{"sample_id": r["sample_id"], "truth": reference(r), "state": r["result"]["state"]}
                         for r in rows if reference(r) and r["result"]["state"] != reference(r)],
            "execution_errors": [{"sample_id": r["sample_id"], "error": r["prediction"].get("error")}
                                 for r in records if r["prediction"]["status"] == "error"]}

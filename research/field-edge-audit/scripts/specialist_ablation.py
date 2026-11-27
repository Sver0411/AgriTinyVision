"""Test source-only class reweighting and target scalar calibration on fixed DINO features."""

import csv
import json

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

from edge_adaptation import fit, score_row
from edge_adaptation_split import write_table
from edge_adaptation_wire import pack_head, predict_from_head
from field_specialist import FEATURES, RESULTS, TARGETS, digest_file, score_test_cohorts


def fit_source_balanced():
    source_head = FEATURES / "dino_plantdoc_head.npz"
    with np.load(source_head, allow_pickle=False) as original:
        fit_keys = original["fit_keys"]
    with np.load(FEATURES / "dino_plantdoc.npz", allow_pickle=False) as source:
        lookup = {str(path): i for i, path in enumerate(source["source_path"])}
        ids = np.array([lookup[path] for path in fit_keys])
        if (len(ids) != 1237 or len(set(ids)) != len(ids)
                or not np.all(source["split"][ids] == "train")
                or not np.isin(source["crop"][ids], sorted(TARGETS)).all()
                or len(set(source["jpeg_sha256"][ids])) != len(ids)):
            raise ValueError("Original source training IDs are inconsistent")
        head = LogisticRegression(C=1.0, solver="lbfgs", tol=1e-8, max_iter=3000,
                                  class_weight="balanced")
        labels = source["y"][ids]
        head.fit(normalize(source["cls"][ids].astype(np.float64)), labels)
    np.savez_compressed(FEATURES / "dino_plantdoc_balanced_head.npz", coef=head.coef_,
                        intercept=head.intercept_, classes=head.classes_, fit_keys=fit_keys,
                        fit_class_counts=np.bincount(labels), original_head_sha256=digest_file(source_head))

    rows = []
    for name in ("plantdoc", "fgvc"):
        with np.load(FEATURES / f"dino_{name}.npz", allow_pickle=False) as archive:
            cols = {key: archive[key] for key in ("dataset", "source_path", "split", "crop", "y")}
            scores = head.decision_function(normalize(archive["cls"].astype(np.float64)))
            for i, p in enumerate(expit(scores)):
                row = {k: v[i].item() for k, v in cols.items()}
                rows.append(dict(**row, feature_dataset=name, score=float(scores[i]),
                                 pred_diseased=int(p >= 0.5), p_diseased=float(p)))
    write_table(RESULTS / "specialist_balanced.csv", rows)
    cohorts = score_test_cohorts(rows)
    (RESULTS / "specialist_balanced_cohorts.json").write_text(json.dumps(cohorts, indent=2) + "\n")
    return cohorts


def target_calibration():
    with np.load(FEATURES / "dino_plantdoc_head.npz", allow_pickle=False) as locked:
        w, bias = locked["coef"][0], locked["intercept"].item()
    with np.load(FEATURES / "dino_fgvc.npz", allow_pickle=False) as features:
        lookup = {str(path): i for i, path in enumerate(features["source_path"])}
        raw = normalize(features["cls"].astype(np.float64)) @ w + bias
        y = features["y"]
    with (RESULTS / "specialist_predictions.csv").open() as file:
        source_scores = {r["source_path"]: float(r["score"]) for r in csv.DictReader(file)
                         if r["feature_dataset"] == "fgvc"}
    if (set(source_scores) != set(lookup)
            or not np.allclose(raw, [source_scores[p] for p in lookup], rtol=0, atol=1e-12)):
        raise ValueError("Locked source-only logit reconstruction failed")

    split_path = RESULTS / "edge_adaptation_predictions.npz"
    with np.load(split_path, allow_pickle=False) as split:
        seeds, budgets = split["seeds"], split["budgets"]
        train, test = split["train_paths"], split["test_paths"]
        test_y = split["test_y"]
        full = split["pred"][:, :, list(split["methods"]).index("dino")]
        pool = set(split["label_pool_paths"])
    if set(test) & pool or any(len(set(draw)) != len(draw) or not set(draw) <= pool for draw in train):
        raise ValueError("Target calibration and test identities overlap or duplicate")
    test_ids = np.array([lookup[path] for path in test])
    if not np.array_equal(test_y, y[test_ids]):
        raise ValueError("Held-out labels and features disagree")

    # Match scalar packet precision, optimizer, fallback, and head decoding of the
    # other target adaptations. The source head and its feature direction are frozen.
    scores = raw.astype(np.float32).reshape(-1, 1)
    with (RESULTS / "edge_adaptation_metrics.csv").open() as file:
        full_metrics = {(int(r["seed"]), int(r["budget"])): r for r in csv.DictReader(file)
                        if r["method"] == "dino"}
    rows = []
    logits = np.empty((len(seeds), len(budgets), len(test)), dtype=np.float64)
    for s, seed in enumerate(seeds):
        for b, budget in enumerate(budgets):
            ids = np.array([lookup[path] for path in train[s, :budget]])
            weights, intercept = fit(scores[ids], y[ids])
            packed = pack_head("scalar", weights, intercept)
            logits[s, b] = predict_from_head("scalar", packed, scores[test_ids])
            one = score_row("scalar", test_y, logits[s, b] >= 0)
            whole = score_row("dino", test_y, full[s, b])
            row = dict(seed=int(seed), budget=int(budget), n_test=len(test),
                       diseased_labels_in_draw=int(y[ids].sum()),
                       scalar_weight=float(np.float32(weights[0])),
                       scalar_intercept=float(np.float32(intercept)))
            for key in ("balanced_accuracy", "accuracy", "sensitivity", "specificity"):
                if not np.isclose(whole[key], float(full_metrics[int(seed), int(budget)][key]), atol=1e-15, rtol=0):
                    raise ValueError("Original 384-dimensional prediction scores failed to reproduce")
                row["scalar_" + key] = one[key]
                row["full_" + key] = whole[key]
            rows.append(row)
    write_table(RESULTS / "specialist_scalar.csv", rows)
    np.savez_compressed(RESULTS / "specialist_scalar_predictions.npz", seeds=seeds, budgets=budgets,
                        test_paths=test, test_y=test_y, train_paths=train, logit=logits,
                        pred=(logits >= 0).astype(np.uint8), split_archive_sha256=digest_file(split_path))
    means = {}
    for budget in budgets:
        subset = [r for r in rows if r["budget"] == budget]
        means[int(budget)] = {k: float(np.mean([r[k] for r in subset])) for k in (
            "scalar_balanced_accuracy", "full_balanced_accuracy", "scalar_accuracy", "full_accuracy")}
    with (RESULTS / "specialist_balanced.csv").open() as file:
        frozen = {r["source_path"]: int(r["pred_diseased"]) for r in csv.DictReader(file)
                  if r["feature_dataset"] == "fgvc"}
    frozen_pred = np.array([frozen[path] for path in test])
    frozen_scores = score_row("source_balanced", test_y, frozen_pred)
    summary = {
        "test_images": len(test),
        "source_balanced_zero_target_labels": {
            k: float(frozen_scores[k]) for k in ("accuracy", "balanced_accuracy", "sensitivity", "specificity")
        },
        "mean_over_label_draws": means,
    }
    (RESULTS / "specialist_scalar_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps({"source_balanced": fit_source_balanced(), "target": target_calibration()}, indent=2))

"""Freeze a two-score Cornell fusion rule and evaluate its two errors against AgriTinyVision.

Run `python scripts/dual_fusion.py fit` before `python scripts/dual_fusion.py evaluate`.
Only Cornell candidate-pool labels enter fit. The binary edge decision uses the
publisher's 0.30 probability cutoff. All comparison decisions use exact scores.
"""

import argparse
import csv
import hashlib
import json
import struct

import numpy as np
from sklearn.linear_model import LogisticRegression

from edge_adaptation_split import ROOT, RESULTS, write_table
from edge_adaptation_wire import IMAGE, REPLY, TRAIN, normalized
from dual_fusion_portable import local_decision, remote_decision


RULES = RESULTS / "dual_fusion_rules.json"
FEATURES = ROOT / "data/features"
EDGE_CUTOFF = float(np.log(0.3 / 0.7))
BUDGETS = (128, 450)
SEEDS = tuple(range(20))
FIT_MARGIN = struct.Struct("<d")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cornell(paths):
    """Read feature and label values only for the requested paths."""
    wanted = set(paths)
    with (RESULTS / "fgvc_edge.csv").open() as src:
        rows = {r["source_path"]: r for r in csv.DictReader(src) if r["source_path"] in wanted}
    with (RESULTS / "specialist_balanced.csv").open() as src:
        dino = {r["source_path"]: float(r["score"]) for r in csv.DictReader(src)
                if r["feature_dataset"] == "fgvc" and r["source_path"] in wanted}
    with np.load(FEATURES / "dino_fgvc.npz", allow_pickle=False) as src:
        size = dict(zip(src["source_path"], src["jpeg_bytes"], strict=True))
    x = np.array([(float(rows[p]["suspect_logit"]) - float(rows[p]["normal_logit"]), dino[p]) for p in paths])
    y = np.array([int(rows[p]["true_diseased"]) for p in paths])
    original = np.array([int(rows[p]["binary_pred"]) for p in paths], dtype=bool)
    if not np.array_equal(original, x[:, 0] >= EDGE_CUTOFF):
        raise ValueError("Cornell original edge score does not reproduce released decisions")
    return x, y, original, np.array([size[p] for p in paths])


def load_kashmir():
    with (RESULTS / "transfer_manifest.csv").open() as src:
        rows = list(csv.DictReader(src))
    with np.load(FEATURES / "dino_plantdoc_balanced_head.npz", allow_pickle=False) as head:
        w, bias = head["coef"][0], head["intercept"].item()
    paths = [r["source_path"] for r in rows]
    with np.load(FEATURES / "kashmir.npz", allow_pickle=False) as src:
        if not np.array_equal(src["source_path"], paths):
            raise ValueError("Kashmir score and JPEG order mismatch")
        dino = normalized(src["dino"]) @ w + bias
    x = np.column_stack(([float(r["edge_score_logit"]) for r in rows], dino))
    y = np.array([int(r["y"]) for r in rows])
    original = np.array([int(r["edge_pred"]) for r in rows], dtype=bool)
    if not np.array_equal(original, x[:, 0] >= EDGE_CUTOFF):
        raise ValueError("Kashmir edge decisions changed")
    return paths, x, y, original, np.array([int(r["jpeg_bytes"]) for r in rows])


def score_columns(rule, x):
    bounded = np.clip(x, rule["lo"], rule["hi"])
    return ((bounded - rule["mean"]) / rule["scale"]) * rule["w"]


def candidate_cutoffs(values):
    levels = np.unique(values)
    return np.r_[np.nextafter(levels[0], -np.inf), (levels[:-1] + levels[1:]) / 2,
                 np.nextafter(levels[-1], np.inf)]


def false_counts(y, decision):
    return int(np.sum((y == 0) & decision)), int(np.sum((y == 1) & ~decision))


def fit_rule(x, y, original, fit_ids, threshold_ids):
    xf, yf = x[fit_ids], y[fit_ids]
    if len(set(yf)) < 2:
        return dict(mode="original", reason="single-class fusion fit")
    lo, hi = np.quantile(xf, (0.01, 0.99), axis=0)
    bounded = np.clip(xf, lo, hi)
    mean, scale = bounded.mean(axis=0), bounded.std(axis=0)
    scale[scale == 0] = 1
    head = LogisticRegression(C=1.0, solver="lbfgs", class_weight="balanced", max_iter=3000, tol=1e-8)
    head.fit((bounded - mean) / scale, yf)
    # The intercept is absorbed by the held-out threshold.
    rule = dict(lo=lo.tolist(), hi=hi.tolist(), mean=mean.tolist(), scale=scale.tolist(),
                w=head.coef_[0].tolist())
    yc = y[threshold_ids]
    fp0, fn0 = false_counts(yc, original[threshold_ids])
    rule.update(calibration_fp_original=fp0, calibration_fn_original=fn0)
    signed = score_columns(rule, x[threshold_ids]).sum(axis=1)
    # Fixed denominators when there are no observed baseline errors in a class.
    # Integer scaling compares both relative improvements exactly.
    best_key = (0, 0, 0)
    best = None
    for threshold in candidate_cutoffs(signed):
        fp, fn = false_counts(yc, signed >= threshold)
        gain_fp = (fp0 - fp) * max(1, fn0)
        gain_fn = (fn0 - fn) * max(1, fp0)
        key = (min(gain_fp, gain_fn), gain_fp + gain_fn, gain_fn)
        if key > best_key:
            best_key = key
            best = (float(threshold), fp, fn)
    if best is None:
        return dict(**rule, mode="original", reason="original wins held-out comparison",
                    calibration_fp=fp0, calibration_fn=fn0)
    return dict(**rule, mode="fusion", threshold=best[0], calibration_fp=best[1], calibration_fn=best[2])


def fit():
    archive = RESULTS / "edge_adaptation_predictions.npz"
    with np.load(archive, allow_pickle=False) as src:
        pool = src["label_pool_paths"]
        previous = src["train_paths"]
    if len(set(pool)) != 450 or previous.shape != (20, 128):
        raise ValueError("Unexpected prior Cornell annotation design")
    x, y, baseline, _ = load_cornell(pool)
    if any(FIT_MARGIN.unpack(FIT_MARGIN.pack(v))[0] != v for v in x[:, 0]):
        raise ValueError("Device margin failed calibration-packet round trip")
    rules = []
    for seed in SEEDS:
        permutation = np.random.default_rng(20261127 + seed).permutation(len(pool))
        if not np.array_equal(pool[permutation[:128]], previous[seed]):
            raise ValueError("128-label set differs from the paired annotation experiment")
        for budget in BUDGETS:
            fit_ids = permutation[:budget // 4]
            threshold_ids = permutation[budget // 4:budget]
            rule = fit_rule(x, y, baseline, fit_ids, threshold_ids)
            rules.append(dict(seed=seed, budget=budget, fit_paths=pool[fit_ids].tolist(),
                              threshold_paths=pool[threshold_ids].tolist(), **rule))
    archive = dict(protocol="source-balanced DINO and released edge logits; 1%-99% fitting-set clipping; "
                   "first floor(B/4) labels for balanced regression; the rest select held-out threshold "
                   "by maximized minimum relative FP/FN reduction, then sum, then FN gain; "
                   "the original edge wins an exact tie; lowest threshold wins further fusion ties",
                   budgets=BUDGETS, seeds=SEEDS, source_head_sha256=digest(FEATURES / "dino_plantdoc_balanced_head.npz"),
                   annotation_pool_paths_sha256=hashlib.sha256(("\n".join(pool) + "\n").encode()).hexdigest(),
                   rules=rules)
    RULES.write_text(json.dumps(archive, separators=(",", ":"), allow_nan=False) + "\n")
    print("Frozen rules:", len(rules), digest(RULES))


def decisions(rule, x, baseline):
    if rule["mode"] == "original":
        return baseline.copy(), np.zeros(len(x), dtype=bool)
    transformed = score_columns(rule, x)
    dino_endpoints = score_columns(rule, np.array([rule["lo"], rule["hi"]]))[:, 1]
    lower = transformed[:, 0] + dino_endpoints.min()
    upper = transformed[:, 0] + dino_endpoints.max()
    threshold = rule["threshold"]
    request = (lower < threshold) & (upper >= threshold)
    output = lower >= threshold
    output[request] = transformed[request].sum(axis=1) >= threshold
    if not np.array_equal(output, transformed.sum(axis=1) >= threshold):
        raise ValueError("Bandwidth gate altered full fusion output")
    return output, request


def evaluate():
    frozen = json.loads(RULES.read_text())
    if frozen["source_head_sha256"] != digest(FEATURES / "dino_plantdoc_balanced_head.npz"):
        raise ValueError("Source model changed since threshold freeze")
    with np.load(RESULTS / "edge_adaptation_predictions.npz", allow_pickle=False) as src:
        cornell = list(src["test_paths"])
        pool = list(src["label_pool_paths"])
    if set(cornell) & set(pool):
        raise ValueError("Cornell test and calibration pool intersect")
    _, _, _, pool_bytes = load_cornell(pool)
    calibration_size = dict(zip(pool, pool_bytes, strict=True))
    reports, predictions = [], {}
    for name, (paths, x, y, baseline, sizes) in (
        ("cornell", (cornell, *load_cornell(cornell))),
        ("kashmir", load_kashmir()),
    ):
        fp0, fn0 = false_counts(y, baseline)
        preds, masks = [], []
        for rule in frozen["rules"]:
            draw_paths = rule["fit_paths"] + rule["threshold_paths"]
            if len(set(draw_paths)) != rule["budget"] or not set(draw_paths).issubset(pool):
                raise ValueError("Calibration identities changed")
            pred, request = decisions(rule, x, baseline)
            local = [local_decision(rule, edge) for edge in x[:, 0]]
            actual_pred = [remote_decision(rule, e, d) if v is None else v
                           for v, (e, d) in zip(local, x, strict=True)]
            if not (np.array_equal(request, [v is None for v in local])
                    and np.array_equal(pred, actual_pred)):
                raise ValueError("Portable gate or float32 score reply altered an evaluated decision")
            fp, fn = false_counts(y, pred)
            preds.append(pred)
            masks.append(request)
            payload = int(sizes[request].sum())
            calibration_jpeg = int(sum(calibration_size[p] for p in draw_paths))
            # The device holds its edge margin; remote replies with a DINO score.
            # No edge margin needs to ride an inference request, only training records.
            reports.append(dict(cohort=name, seed=rule["seed"], budget=rule["budget"], mode=rule["mode"],
                                n=len(y), healthy=int(sum(y == 0)), diseased=int(sum(y == 1)),
                                original_fp=fp0, original_fn=fn0, fusion_fp=fp, fusion_fn=fn,
                                strictly_both_improved=int(fp < fp0 and fn < fn0),
                                specificity=1 - fp / sum(y == 0), sensitivity=1 - fn / sum(y == 1),
                                images_uploaded=int(request.sum()), jpeg_bytes=payload,
                                inference_wire_bytes=payload + int(request.sum()) * (IMAGE.size + REPLY.size),
                                calibration_jpeg_bytes=calibration_jpeg,
                                calibration_wire_bytes=calibration_jpeg + rule["budget"] * (TRAIN.size + FIT_MARGIN.size)))
        predictions[name + "_path"] = np.asarray(paths)
        predictions[name + "_y"] = y
        predictions[name + "_original"] = baseline
        predictions[name + "_fusion"] = np.asarray(preds, dtype=np.uint8)
        predictions[name + "_uploaded"] = np.asarray(masks, dtype=np.uint8)
    write_table(RESULTS / "dual_fusion_metrics.csv", reports)
    np.savez_compressed(RESULTS / "dual_fusion_predictions.npz", **predictions,
                        seeds=[r["seed"] for r in frozen["rules"]], budgets=[r["budget"] for r in frozen["rules"]],
                        frozen_rules_sha256=digest(RULES))
    summary = []
    for cohort in ("cornell", "kashmir"):
        for budget in BUDGETS:
            sub = [r for r in reports if r["cohort"] == cohort and r["budget"] == budget]
            summary.append(dict(cohort=cohort, budget=budget, seeds=len(sub), original_fp=sub[0]["original_fp"],
                                original_fn=sub[0]["original_fn"], both_strict=sum(r["strictly_both_improved"] for r in sub),
                                strict_fp=sum(r["fusion_fp"] < r["original_fp"] for r in sub),
                                strict_fn=sum(r["fusion_fn"] < r["original_fn"] for r in sub),
                                original_fallback=sum(r["mode"] == "original" for r in sub),
                                **{k + "_mean": float(np.mean([r[k] for r in sub])) for k in (
                                    "fusion_fp", "fusion_fn", "specificity", "sensitivity", "images_uploaded",
                                    "inference_wire_bytes", "calibration_wire_bytes")}))
    write_table(RESULTS / "dual_fusion_summary.csv", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fit", "evaluate"))
    if parser.parse_args().stage == "fit":
        fit()
    else:
        evaluate()

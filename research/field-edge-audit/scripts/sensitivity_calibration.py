"""Evaluate an empirical 95% sensitivity target using 64 disjoint calibration labels."""

import argparse
import json
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import numpy as np
from sklearn.metrics import roc_auc_score

from edge_adaptation import (
    BUDGETS,
    METHODS,
    SEEDS,
    fit,
    load_features,
    prepare_records,
    score_row,
)
from edge_adaptation_split import RESULTS, ROOT, split, write_table
from edge_adaptation_wire import normalized, pack_head, predict_from_head, quantized

MODEL_NAMES = ("frozen", "scalar", "local_int8", "dino")


def choose_cutoff(scores, labels):
    pos = np.asarray(scores)[np.asarray(labels) == 1]
    if not len(pos):
        return -np.inf
    keep = (19 * len(pos) + 19) // 20
    return float(np.sort(pos)[-keep])


def means(records, by, metrics):
    keys = sorted({tuple(r[k] for k in by) for r in records})
    out = []
    for key in keys:
        selected = [r for r in records if tuple(r[k] for k in by) == key]
        row = dict(zip(by, key))
        row["n_draws"] = len(selected)
        for metric in metrics:
            values = np.asarray([r[metric] for r in selected])
            row[metric + "_mean"] = float(values.mean())
            row[metric + "_min"] = float(values.min())
            row[metric + "_max"] = float(values.max())
        out.append(row)
    return out


def original_auc(paths, test, y, margin):
    out = []
    with np.load(
        RESULTS / "edge_adaptation_predictions.npz", allow_pickle=False
    ) as trace:
        if not np.array_equal(trace["test_paths"], paths[test]) or not np.array_equal(
            trace["test_y"], y[test]
        ):
            raise ValueError("Original result uses a different holdout")
        if (
            list(trace["budgets"]) != list(BUDGETS)
            or list(trace["methods"]) != list(METHODS)
            or list(trace["seeds"]) != list(SEEDS)
        ):
            raise ValueError("Trace axes differ")
        z = trace["score"]
        trains = trace["train_paths"]
        out.append(
            dict(
                seed=-1,
                budget=0,
                method="frozen",
                auroc=roc_auc_score(y[test], margin[test, 0]),
            )
        )
        for s in SEEDS:
            for bi, b in enumerate(BUDGETS):
                for mi, method in enumerate(METHODS[1:], start=1):
                    out.append(
                        dict(
                            seed=s,
                            budget=b,
                            method=method,
                            auroc=roc_auc_score(y[test], z[s, bi, mi]),
                        )
                    )
    write_table(RESULTS / "edge_adaptation_auroc.csv", out)
    write_table(
        RESULTS / "edge_adaptation_auroc_summary.csv",
        means(out, ["budget", "method"], ["auroc"]),
    )
    return trains


def run(jpeg_dir):
    rows, _, available, test = split()
    paths = np.asarray([r["source_path"] for r in rows])
    index = {p: i for i, p in enumerate(paths)}
    x, margin, cls, jpegs, _ = load_features(rows, jpeg_dir)
    y = np.asarray([int(r["true_diseased"]) for r in rows], np.uint8)
    uid = np.asarray([int(Path(p).stem.removeprefix("Train_")) for p in paths])
    trains = original_auc(paths, test, y, margin)
    train, _ = prepare_records(available, x, margin, cls, jpegs, y, uid)
    all_x = {"scalar": margin, "local_int8": quantized(x)[0], "dino": normalized(cls)}
    cands = {absolute: rel for rel, absolute in enumerate(available)}
    out = []
    test_score = np.zeros((len(SEEDS), len(MODEL_NAMES), len(test)))
    calibration_score = np.zeros((len(SEEDS), len(MODEL_NAMES), 64))
    test_pred = np.zeros((len(SEEDS), len(MODEL_NAMES), 2, len(test)), dtype=np.uint8)
    cutoffs = np.zeros((len(SEEDS), len(MODEL_NAMES), 2))
    archive = RESULTS / "sensitivity64_heads.zip"
    # Acquire no test labels while fitting or calibrating.
    with ZipFile(archive, "w") as f:
        for s in SEEDS:
            ab = np.asarray([index[p] for p in trains[s]])
            if len(set(ab)) != 128 or not set(ab).issubset(cands):
                raise ValueError("Incomplete or leaking 128-label draw")
            rows_train = np.asarray([cands[i] for i in ab[:64]])
            for method in MODEL_NAMES[1:]:
                w, b = fit(train[method][rows_train], y[ab[:64]])
                blob = pack_head(method, w, b)
                f.writestr(ZipInfo(f"s{s:02d}/{method}.bin"), blob)
    recorded_frozen = np.asarray([int(r["binary_pred"]) for r in rows])
    for s in SEEDS:
        ab = np.asarray([index[p] for p in trains[s]])
        c = ab[64:]
        scored = np.concatenate((c, test))
        with ZipFile(archive) as f:
            for mi, method in enumerate(MODEL_NAMES):
                if method == "frozen":
                    scores = margin[scored, 0].astype(np.float64)
                    native = float(np.log(0.3 / (1 - 0.3)))
                    if not np.array_equal(scores >= native, recorded_frozen[scored]):
                        raise ValueError(
                            "Native frozen margin decisions differ from released threshold"
                        )
                else:
                    blob = f.read(f"s{s:02d}/{method}.bin")
                    scores = predict_from_head(method, blob, all_x[method][scored])
                    native = 0.0
                zc, zt = scores[:64], scores[64:]
                test_score[s, mi] = zt
                calibration_score[s, mi] = zc
                positives = int(y[c].sum())
                target = choose_cutoff(zc, y[c])
                for op, (name, threshold) in enumerate(
                    (("native", native), ("target95", target))
                ):
                    selection = zc >= threshold
                    tp_cal = int(np.count_nonzero(selection & (y[c] == 1)))
                    actual = (tp_cal / positives) if positives else None
                    if (
                        name == "target95"
                        and positives
                        and 20 * tp_cal < 19 * positives
                    ):
                        raise ValueError("Positive calibration constraint violated")
                    pred = zt >= threshold
                    test_pred[s, mi, op] = pred
                    cutoffs[s, mi, op] = threshold
                    out.append(
                        dict(
                            seed=s,
                            **score_row(method, y[test], pred),
                            operating_point=name,
                            n_fit=0 if method == "frozen" else 64,
                            n_calibrate=64 if name == "target95" else 0,
                            n_calibration_positive=positives,
                            n_calibration_true_positive=tp_cal,
                            calibration_sensitivity=actual,
                            empty_positive_fallback=int(
                                positives == 0 and name == "target95"
                            ),
                            threshold=threshold,
                            auroc=float(roc_auc_score(y[test], zt)),
                            fit_single_class=int(
                                method != "frozen" and len(np.unique(y[ab[:64]])) == 1
                            ),
                        )
                    )
    write_table(RESULTS / "sensitivity64_metrics.csv", out)
    summary = means(
        out,
        ["method", "operating_point"],
        ["sensitivity", "specificity", "balanced_accuracy", "auroc", "accuracy"],
    )
    for r in summary:
        items = [
            v
            for v in out
            if v["method"] == r["method"]
            and v["operating_point"] == r["operating_point"]
        ]
        r["n_test_sensitivity_ge_95"] = sum(
            int(v["sensitivity"] >= 0.95) for v in items
        )
    write_table(RESULTS / "sensitivity64_summary.csv", summary)
    np.savez_compressed(
        RESULTS / "sensitivity64_predictions.npz",
        test_paths=paths[test],
        test_y=y[test],
        fit_paths=trains[:, :64],
        calibration_paths=trains[:, 64:],
        methods=MODEL_NAMES,
        operating_points=("native", "target95"),
        seeds=SEEDS,
        test_score=test_score,
        calibration_score=calibration_score,
        pred=test_pred,
        threshold=cutoffs,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jpeg-dir", type=Path, default=ROOT / "data/fgvc_backend_jpegs")
    run(ap.parse_args().jpeg_dir)

"""Reproduce label-paired FGVC7 refits, actual traffic and all decoded-head predictions."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import zipfile

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression

from edge_adaptation_split import RESULTS, ROOT, split, write_table
from edge_adaptation_wire import (
    IMAGE,
    REPLY,
    TRAIN,
    normalized,
    pack_head,
    predict_from_head,
    quantized,
    training_input,
    training_record,
)

METHODS = ("frozen", "scalar", "local_int8", "local_fp32", "dino")
BUDGETS = (8, 32, 128)
SEEDS = tuple(range(20))


def digest(value):
    return hashlib.sha256(value).hexdigest()


def load_features(rows, jpeg_dir):
    paths = [r["source_path"] for r in rows]
    with np.load(ROOT / "data/features/edge_fgvc.npz", allow_pickle=False) as data:
        if paths != list(data["source_path"]) or [
            r["file_sha256"] for r in rows
        ] != list(data["source_sha256"]):
            raise ValueError("Released image and local feature identities differ")
        if [r["rgb160_sha256"] for r in rows] != list(data["rgb160_sha256"]):
            raise ValueError("Local image tensor identities differ")
        x = data["activation"]
        logits = data["logits"]
        maxerr = float(data["logit_reconstruction_max_abs"])
        if (
            data["model_sha256"].item()
            != "350c32d88d19eb16dca7d905036d2c99a716a3938b91f52715349e43ee6ab909"
        ):
            raise ValueError("Unexpected local graph")
    logref = np.asarray(
        [(float(r["normal_logit"]), float(r["suspect_logit"])) for r in rows],
        np.float32,
    )
    if not np.allclose(logref, logits, atol=1e-4, rtol=0):
        raise ValueError("Exported activations changed the original scores")
    with np.load(ROOT / "data/features/dino_fgvc.npz", allow_pickle=False) as data:
        idx = {str(path): i for i, path in enumerate(data["source_path"])}
        if set(idx) != set(paths) or len(data["source_path"]) != len(paths):
            raise ValueError("DINO identity mismatch")
        order = np.asarray([idx[path] for path in paths])
        for field, expected in (
            ("source_sha256", [r["file_sha256"] for r in rows]),
            ("y", [int(r["true_diseased"]) for r in rows]),
            ("long_edge", np.full(len(rows), 224)),
            ("jpeg_quality", np.full(len(rows), 75)),
        ):
            if not np.array_equal(data[field][order], expected):
                raise ValueError("DINO row alignment mismatch in " + field)
        if (
            data["model_sha256"].item()
            != "83141175ec78b4ff9a2bb58a4c7c264ba0054d1c2e122e5a8114b79a8d4179ea"
        ):
            raise ValueError("Unexpected DINO checkpoint")
        cls = data["cls"][order]
        checksums = data["jpeg_sha256"][order]
        byte_lengths = data["jpeg_bytes"][order]
    jpegs = []
    for sha, expected in zip(checksums, byte_lengths):
        encoded = (jpeg_dir / (sha + ".jpg")).read_bytes()
        if len(encoded) != expected or digest(encoded) != sha:
            raise ValueError(
                "The DINO feature's JPEG differs from communication payload"
            )
        jpegs.append(encoded)
    if x.shape != (len(rows), 1280) or cls.shape != (len(rows), 384):
        raise ValueError("Unexpected activations")
    return x, logits[:, 1:2] - logits[:, 0:1], cls, jpegs, maxerr


def fit(x, y):
    if len(np.unique(y)) == 1:
        return np.zeros(x.shape[1]), (1.0 if y[0] else -1.0)
    model = LogisticRegression(
        C=1.0,
        solver="liblinear",
        class_weight="balanced",
        max_iter=3000,
        tol=1e-8,
        random_state=0,
    )
    model.fit(x, y)
    return model.coef_[0], model.intercept_[0]


def score_row(method, y, pred):
    pos = y == 1
    neg = ~pos
    tp = int(pred[pos].sum())
    tn = int((pred[neg] == 0).sum())
    sens, spec = tp / np.sum(pos), tn / np.sum(neg)
    return dict(
        method=method,
        true_positive=tp,
        true_negative=tn,
        false_positive=int(neg.sum()) - tn,
        false_negative=int(pos.sum()) - tp,
        accuracy=float(np.mean(y == pred)),
        sensitivity=sens,
        specificity=spec,
        balanced_accuracy=float((sens + spec) / 2),
    )


def prepare_records(available, x, margin, cls, jpegs, y, uids):
    values = {"scalar": margin, "local_fp32": x, "local_int8": x, "dino": jpegs}
    train, sizes = {}, {}
    # Expose only candidate-pool labels to any supervised calibration routine.
    for method in METHODS[1:]:
        feats = []
        lens = []
        for i in available:
            value = values[method][i]
            packet = training_record(method, uids[i], y[i], value)
            # For a colocated remote reviewer, the on-device diagnosis is unavailable.
            # A missing-label placeholder preserves the wire layout. The server joins
            # the human diagnosis to the same stable image ID after review.
            unlabelled = training_record(method, uids[i], 255, value)
            if (
                len(unlabelled) != len(packet)
                or unlabelled[TRAIN.size :] != packet[TRAIN.size :]
            ):
                raise ValueError("Remote-review protocol changed feature bytes")
            decoded = training_input(method, packet, uids[i], y[i], len(value))
            if method == "dino":
                if decoded != jpegs[i]:
                    raise ValueError(
                        "DINO calibration input did not survive the wire round trip"
                    )
                decoded = cls[i]
            feats.append(decoded)
            lens.append(len(packet))
        train[method] = (
            np.asarray(feats) if method == "scalar" else normalized(np.asarray(feats))
        )
        sizes[method] = np.asarray(lens, dtype=np.int64)
    return train, sizes


def run(upload_dir):
    rows, split_rows, available, test = split()
    write_table(RESULTS / "edge_adaptation_split.csv", split_rows)
    x, margin, cls, jpegs, maxerr = load_features(rows, upload_dir)
    y = np.asarray([int(r["true_diseased"]) for r in rows], np.uint8)
    paths = np.asarray([r["source_path"] for r in rows])
    uids = np.asarray([int(Path(p).stem.removeprefix("Train_")) for p in paths])
    train, sizes = prepare_records(available, x, margin, cls, jpegs, y, uids)
    qtest, _ = quantized(x[test])
    inputs = dict(
        scalar=margin[test],
        local_int8=qtest,
        local_fp32=normalized(x[test]),
        dino=normalized(cls[test]),
    )
    frozen = np.asarray([float(rows[i]["p_suspect"]) for i in test], dtype=np.float32)
    frozen_labels = np.asarray([int(rows[i]["binary_pred"]) for i in test])
    if not np.array_equal(frozen >= np.float32(0.3), frozen_labels):
        raise ValueError("Frozen reference label mismatch")
    permutations = np.asarray(
        [
            np.random.default_rng(20261127 + s).permutation(len(available))[
                : max(BUDGETS)
            ]
            for s in SEEDS
        ]
    )
    train_paths = paths[available[permutations]]
    test_jpeg = sum(len(jpegs[i]) for i in test)
    test_upload = sum(len(IMAGE.pack(b"IMG1", uids[i]) + jpegs[i]) for i in test)
    test_reply = len(test) * len(REPLY.pack(b"PRD1", 0, 0))
    pred = np.empty((len(SEEDS), len(BUDGETS), len(METHODS), len(test)), dtype=np.uint8)
    scores = np.empty(pred.shape, dtype=np.float64)
    metrics = []
    packfile = RESULTS / "edge_adaptation_heads.zip"
    local_fp = {}
    # ZIP stores only released binary heads. Network cost is the member body.
    with zipfile.ZipFile(packfile, "w", compression=zipfile.ZIP_STORED) as archive:
        for s, perm in zip(SEEDS, permutations):
            for b in BUDGETS:
                subset = perm[:b]
                labels = y[available[subset]]
                for method in METHODS[1:]:
                    w, bias = fit(train[method][subset], labels)
                    blob = pack_head(method, w, bias)
                    archive.writestr(f"s{s:02d}-b{b:03d}/{method}.bin", blob)
                    # Independent float evaluation of the same stored precision, with
                    # separately reconstructed coefficients and normalized integer inputs.
                    if method == "local_int8":
                        qw, ws = quantized(w)
                        wref = qw.astype(np.float64) * ws
                        test_x = normalized(qtest)
                    else:
                        wref = w.astype(np.float32).astype(np.float64)
                        test_x = inputs[method].astype(np.float64)
                    local_fp[s, b, method] = test_x @ wref + np.float32(bias)
    max_delta = 0.0
    with zipfile.ZipFile(packfile) as archive:
        if archive.testzip() is not None:
            raise ValueError("Model archive CRC failed")
        for s, perm in zip(SEEDS, permutations):
            for bi, b in enumerate(BUDGETS):
                subset = perm[:b]
                absolute = available[subset]
                labels = y[absolute]
                jbytes = sum(len(jpegs[i]) for i in absolute)
                image_record_bytes = sum(
                    len(IMAGE.pack(b"IMG1", uids[i]) + jpegs[i]) for i in absolute
                )
                for mi, method in enumerate(METHODS):
                    local, remote = method != "frozen", method == "dino"
                    if method == "frozen":
                        z = frozen.astype(np.float64)
                        decision = frozen_labels
                        size = training = extra = 0
                    else:
                        blob = archive.read(f"s{s:02d}-b{b:03d}/{method}.bin")
                        size = len(blob)
                        training = int(sizes[method][subset].sum())
                        z = predict_from_head(method, blob, inputs[method])
                        decision = z >= 0
                        reference = local_fp[s, b, method]
                        delta = float(np.max(np.abs(reference - z)))
                        max_delta = max(max_delta, delta)
                        if delta > 1e-9 or not np.array_equal(decision, reference >= 0):
                            raise ValueError(
                                f"Downloaded head changed expected decisions for {s, b, method}: {delta}"
                            )
                        extra = 0 if remote else image_record_bytes
                    pred[s, bi, mi] = decision
                    scores[s, bi, mi] = z
                    up = test_upload if remote else 0
                    down = test_reply if remote else 0
                    update = size if local and not remote else 0
                    local_total = training + update + up + down
                    metrics.append(
                        dict(
                            seed=s,
                            budget=b,
                            **score_row(method, y[test], decision),
                            n_test=len(test),
                            labels_used=b if local else 0,
                            diseased_labels_in_draw=int(sum(int(t) for t in labels)),
                            unique_classes_in_draw=int(len(np.unique(labels))),
                            written_head_bytes=size,
                            calibration_upload_bytes=training,
                            calibration_header_bytes=b * TRAIN.size if local else 0,
                            calibration_jpeg_bytes=jbytes if remote else 0,
                            calibration_feature_payload_bytes=training - b * TRAIN.size
                            if local and not remote
                            else 0,
                            head_downlink_bytes=update,
                            test_upload_jpeg_bytes=test_jpeg if remote else 0,
                            test_upload_record_bytes=up,
                            test_score_reply_bytes=down,
                            local_label_total_bytes=local_total,
                            remote_review_additional_jpeg_bytes=jbytes
                            if local and not remote
                            else 0,
                            remote_review_additional_record_bytes=extra,
                            remote_label_total_bytes=local_total + extra,
                        )
                    )
    write_table(RESULTS / "edge_adaptation_metrics.csv", metrics)
    np.savez_compressed(
        RESULTS / "edge_adaptation_predictions.npz",
        test_paths=paths[test],
        test_y=y[test],
        label_pool_paths=paths[available],
        train_paths=train_paths,
        methods=METHODS,
        budgets=BUDGETS,
        seeds=SEEDS,
        score=scores,
        pred=pred,
    )
    aggregated = []
    for b in BUDGETS:
        for method in METHODS:
            ss = [r for r in metrics if r["budget"] == b and r["method"] == method]
            ag = dict(budget=b, method=method)
            for key in (
                "accuracy",
                "balanced_accuracy",
                "sensitivity",
                "specificity",
                "calibration_upload_bytes",
                "head_downlink_bytes",
                "test_upload_jpeg_bytes",
                "test_upload_record_bytes",
                "test_score_reply_bytes",
                "remote_review_additional_record_bytes",
                "local_label_total_bytes",
                "remote_label_total_bytes",
            ):
                v = np.asarray([r[key] for r in ss])
                ag.update(
                    {
                        key + "_mean": float(np.mean(v)),
                        key + "_min": float(np.min(v)),
                        key + "_max": float(np.max(v)),
                    }
                )
            aggregated.append(ag)
    write_table(RESULTS / "edge_adaptation_summary.csv", aggregated)
    info = dict(
        n_sources=len(rows),
        status_count=dict(Counter(r["status"] for r in split_rows)),
        test_healthy=int(np.sum(y[test] == 0)),
        test_diseased=int(np.sum(y[test] == 1)),
        pool_healthy=int(np.sum(y[available] == 0)),
        pool_diseased=int(np.sum(y[available] == 1)),
        n_packed_heads=len(local_fp),
        n_decoded_decisions=int(len(local_fp) * len(test)),
        max_decoded_score_delta=max_delta,
        n_single_class_draws=sum(
            int(r["unique_classes_in_draw"] == 1)
            for r in metrics
            if r["method"] == "local_int8"
        ),
        max_original_head_reconstruction_logit_error=maxerr,
        frozen_score_unit="p_suspect; threshold float32 0.30",
        trained_score_unit="signed logit; threshold 0",
        sklearn=sklearn.__version__,
        numpy=np.__version__,
        python=platform.python_version(),
        features_sha256={
            p.name: digest(p.read_bytes())
            for p in (ROOT / "data/features").glob("*_fgvc.npz")
        },
        split_sha256=digest((RESULTS / "edge_adaptation_split.csv").read_bytes()),
    )
    (RESULTS / "edge_adaptation_manifest.json").write_text(
        json.dumps(info, indent=2) + "\n"
    )
    for ag in aggregated:
        print(
            ag["budget"],
            ag["method"],
            f"BA {ag['balanced_accuracy_mean']:.4f}",
            f"accuracy {ag['accuracy_mean']:.4f}",
            f"wire {ag['local_label_total_bytes_mean']:.1f} / {ag['remote_label_total_bytes_mean']:.1f}",
        )
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jpeg-dir", type=Path, default=ROOT / "data/fgvc_backend_jpegs"
    )
    run(parser.parse_args().jpeg_dir)

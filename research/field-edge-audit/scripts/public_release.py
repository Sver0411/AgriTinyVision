"""Export the public artifact and restore FGVC7 reference columns from an authorized train.csv."""

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from build_artifact import ROOT, inputs

CORNELL = re.compile(r"^(?:images/)?Train_[0-9]+(?:\.jpg)?$")
LABEL_KEYS = {"y", "reference", "true_diseased", "source_label"}
EXCLUDED = {
    "data/features/dino_fgvc.npz",
    "data/features/edge_fgvc.npz",
    "data/features/dual_mv_cornell.npz",
}
INDEX = "PUBLIC_DATA_INDEX.json"


RUN_CACHE = {
    "joint_budget_runs.csv": ("joint_budget_cornell.npz", "predictions"),
    "published_cascade_runs.csv": ("published_cascade_cornell.npz", "pred"),
    "published_gatekeeper_runs.csv": ("published_gatekeeper_cornell.npz", "pred"),
    "published_defer_runs.csv": (
        "published_defer_decisions_cornell.npz",
        "predictions",
    ),
    "published_edgeboost_runs.csv": ("published_edgeboost_cornell.npz", "predictions"),
    "streaming_runs.csv": ("streaming_decisions_cornell.npz", "predictions"),
    "streaming_published_runs.csv": ("streaming_published_cornell.npz", "predictions"),
    "esp_head_drift.csv": ("esp_head_drift_cornell.npz", "predictions"),
}
MEASURES = set(
    (
        "fp fn tp tn true_positive true_negative false_positive false_negative accuracy balanced_accuracy "
        "sensitivity specificity corrected spoiled proposed_balanced_accuracy proposed_accuracy paired_delta_ba_pp "
        "paired_delta_acc_pp proposed_only_correct router_only_correct corrected_over_proposed spoiled_over_proposed"
    ).split()
)
REPLAY = {
    "matched_collaboration_sweep.csv",
    "second_backbone_runs.csv",
    "rural_deadline_runs.csv",
    "upload_routing.csv",
    "dual_existing.csv",
}
FLOW_FIELD = {
    "upload_routing.csv": {
        "used_bytes",
        "calls",
        "repairs",
        "regressions",
        "net_correct",
        "correct",
    },
    "dual_existing.csv": {
        "locked_threshold",
        "fp",
        "fn",
        "delta_fp",
        "delta_fn",
        "test_oracle_threshold",
        "test_oracle_fn",
        "paired_dual",
        "strict_dual",
        "test_oracle_dual",
    },
}
ALL_CORNELL = {
    "edge_adaptation_metrics.csv",
    "sensitivity64_metrics.csv",
    "specialist_scalar.csv",
}
REPLAY.update(ALL_CORNELL | {"published_defer_default.csv"})
SENSITIVITY_CACHE = "sensitivity64_predictions.npz"


def cornell_row(row):
    return any(
        "cornell" in row.get(k, "").lower() or "fgvc" in row.get(k, "").lower()
        for k in ("site", "cohort", "dataset")
    )


def fine_export(src, dst):
    is_sweep = src.name == "matched_collaboration_sweep.csv"
    if src.name not in RUN_CACHE and src.name not in REPLAY:
        return None
    with src.open(newline="") as f:
        data = csv.DictReader(f)
        fields = data.fieldnames
        rows = list(data)
    if is_sweep:
        keys = {"fp", "fn", "both_strict", "half_fn"}
    elif src.name == "dual_existing.csv":
        keys = set(fields) - {
            "cohort",
            "policy",
            "method",
            "fit_labels",
            "cal_labels",
            "seed",
            "threshold",
            "n",
            "original_fp",
            "original_fn",
        }
    elif src.name == "specialist_scalar.csv":
        keys = {
            k
            for k in fields
            if k.startswith(("scalar_", "full_"))
            and k not in ("scalar_weight", "scalar_intercept")
        }
    elif src.name == "sensitivity64_metrics.csv":
        keys = MEASURES | {
            "threshold",
            "n_calibration_true_positive",
            "calibration_sensitivity",
        }
    else:
        keys = MEASURES | FLOW_FIELD.get(src.name, set())
    keys = sorted(keys.intersection(fields))
    affected = 0
    for row in rows:
        if is_sweep or src.name in ALL_CORNELL or cornell_row(row):
            affected += 1
            for key in keys:
                row[key] = ""
    if not affected or not keys:
        return None
    with dst.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return dict(
        fields=keys,
        n=affected,
        mechanism="threshold enumeration"
        if is_sweep
        else "repeated paired reference evaluations",
    )


def read_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    tmp = path.with_suffix(".rejoining")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def values(y, pred):
    pred = np.asarray(pred, dtype=bool)
    pos = y == 1
    fp = int(np.count_nonzero(pred & ~pos))
    fn = int(np.count_nonzero(~pred & pos))
    return dict(
        fp=fp,
        fn=fn,
        accuracy=float((pred == y).mean()),
        balanced_accuracy=float((pred[pos].mean() + (~pred[~pos]).mean()) / 2),
    )


def restore_fine(root, index):
    out = root / "results"
    setup = {
        (int(r["seed"]), loc): int(r[f"{loc}_label_total_bytes"])
        for r in read_rows(out / "edge_adaptation_metrics.csv")
        if r["method"] == "local_int8" and r["budget"] == "128"
        for loc in ("local", "remote")
    }
    from published_cascade import uncertainty_control

    with np.load(
        out / "published_cascade_deployment_margins.npz", allow_pickle=False
    ) as z:
        names, e, expert, jpeg, margins = [
            z["cornell_" + k]
            for k in ("paths", "edge_p", "expert", "jpeg", "local_margin")
        ]
    costs = jpeg + 20
    baselines = {}
    for name, spec in index["fine"].items():
        path = root / name
        if path.name not in RUN_CACHE:
            continue
        archive, field = RUN_CACHE[path.name]
        rows = read_rows(path)
        selected = [r for r in rows if cornell_row(r)]
        with np.load(path.parent / archive, allow_pickle=False) as z:
            ids = z["source_path"]
            p = z[field]
            if p.shape[1] != len(ids):
                p = np.unpackbits(p, axis=1, count=len(ids))
            if "y" in z.files:
                y = z["y"]
            else:
                all_labels = {
                    r["source_path"]: int(r["true_diseased"])
                    for r in read_rows(out / "fgvc_edge.csv")
                }
                y = np.array([all_labels[k] for k in ids])
        if len(selected) != len(p):
            raise ValueError("Run and prediction counts differ in " + name)
        for row, pred in zip(selected, p, strict=True):
            fv = values(y, pred)
            if "proposed_accuracy" in spec["fields"]:
                key = (
                    int(row["seed"]),
                    row["review_location"],
                    row["accounting"],
                    int(row["cap_bytes"]),
                )
                if key not in baselines:
                    seed, loc, stage, cap = key
                    control = uncertainty_control(
                        names,
                        e,
                        margins[seed],
                        costs,
                        cap,
                        setup[seed, loc],
                        stage == "cold",
                    )
                    baselines[key] = np.where(control.queries, expert, control.edge)
                if not np.array_equal(ids, names):
                    raise ValueError("Incompatible baseline ordering")
                b = baselines[key]
                bm = values(y, b)
                yes = int(((pred == y) & (b != y)).sum())
                no = int(((pred != y) & (b == y)).sum())
                fv.update(
                    proposed_accuracy=bm["accuracy"],
                    proposed_balanced_accuracy=bm["balanced_accuracy"],
                    paired_delta_ba_pp=100
                    * (fv["balanced_accuracy"] - bm["balanced_accuracy"]),
                    paired_delta_acc_pp=100 * (yes - no) / len(y),
                    proposed_only_correct=no,
                    router_only_correct=yes,
                    corrected_over_proposed=yes,
                    spoiled_over_proposed=no,
                )
            for k in spec["fields"]:
                row[k] = fv[k]
        write_rows(path, rows)
    for name in index["fine"]:
        path = root / name
        if path.name not in REPLAY:
            continue
        if path.name == "matched_collaboration_sweep.csv":
            from matched_collaboration import Data, select_single

            rows = [
                r
                for r in read_rows(out / "matched_collaboration_scores.csv")
                if r["cohort"] == "calibration"
            ]
            d = Data(
                [r["source_path"] for r in rows],
                np.array([int(r["y"]) for r in rows]),
                np.array([float(r["edge_probability"]) for r in rows], np.float32),
                np.array([float(r["dino_probability"]) for r in rows], np.float32),
                np.array([int(r["jpeg_bytes"]) for r in rows]),
            )
            sweep = []
            for src in ("edge", "dino"):
                select_single(d, src, values(d.y, d.e >= np.float32(0.30)), sweep)
            write_rows(path, sweep)
        elif path.name in ALL_CORNELL or path.name == "published_defer_default.csv":
            from edge_adaptation import score_row

            if path.name == "published_defer_default.csv":
                with np.load(
                    path.parent / "published_defer_scores_cornell.npz"
                ) as cache:
                    ref = cache["y"]
                    mi = {v: i for i, v in enumerate(cache["methods"])}
                    predictions = np.where(
                        cache["priority"] > 0, cache["expert"], cache["local"]
                    )
                rows = read_rows(path)
                for r in rows:
                    if cornell_row(r):
                        metrics = values(
                            ref, predictions[int(r["seed"]), mi[r["method"]]]
                        )
                        for k in index["fine"][name]["fields"]:
                            r[k] = metrics[k]
                write_rows(path, rows)
                continue
            cache_names = {
                "edge_adaptation_metrics.csv": "edge_adaptation_predictions.npz",
                "specialist_scalar.csv": "specialist_scalar_predictions.npz",
                "sensitivity64_metrics.csv": SENSITIVITY_CACHE,
            }
            rows = read_rows(path)
            with np.load(out / cache_names[path.name]) as cache:
                pred = cache["pred"]
                y = cache["test_y"]
                si = {v: i for i, v in enumerate(cache["seeds"])}
                if "budgets" in cache:
                    bi = {v: i for i, v in enumerate(cache["budgets"])}
                if "methods" in cache:
                    mi = {v: i for i, v in enumerate(cache["methods"])}
                if path.name == "sensitivity64_metrics.csv":
                    op = {v: i for i, v in enumerate(cache["operating_points"])}
                    th = cache["threshold"]
                    scores = cache["calibration_score"]
                    ids = cache["calibration_paths"]
                    labels = {
                        r["source_path"]: int(r["true_diseased"])
                        for r in read_rows(out / "fgvc_edge.csv")
                    }
                if path.name == "specialist_scalar.csv":
                    with np.load(out / "edge_adaptation_predictions.npz") as edge:
                        full_pred = edge["pred"][
                            :, :, edge["methods"].tolist().index("dino")
                        ]
                for r in rows:
                    i = si[int(r["seed"])]
                    if path.name == "sensitivity64_metrics.csv":
                        j = mi[r["method"]]
                        q = op[r["operating_point"]]
                        l = np.array([labels[id] for id in ids[i]])
                        metrics = score_row(r["method"], y, pred[i, j, q])
                        cut = th[i, j, q]
                        count = int(np.count_nonzero((scores[i, j] >= cut) & (l == 1)))
                        metrics.update(
                            threshold=cut,
                            n_calibration_true_positive=count,
                            calibration_sensitivity=count / l.sum() if l.sum() else "",
                        )
                    elif path.name == "specialist_scalar.csv":
                        j = bi[int(r["budget"])]
                        metrics = {}
                        for name2, p in [
                            ("full", full_pred[i, j]),
                            ("scalar", pred[i, j]),
                        ]:
                            metrics.update(
                                {
                                    name2 + "_" + k: v
                                    for k, v in score_row(name2, y, p).items()
                                }
                            )
                    else:
                        metrics = score_row(
                            r["method"],
                            y,
                            pred[i, bi[int(r["budget"])], mi[r["method"]]],
                        )
                    for k in index["fine"][name]["fields"]:
                        r[k] = metrics[k]
            write_rows(path, rows)
        elif path.name == "second_backbone_runs.csv":
            from second_backbone import run_rows

            write_rows(path, list(run_rows(out)))
        elif path.name == "rural_deadline_runs.csv":
            import rural_link

            profiles = read_rows(out / "rural_measured_kpis.csv")
            for p in profiles:
                for k in ("up_mbps", "down_mbps", "rtt_ms", "distance_m"):
                    p[k] = float(p[k])
            from unittest.mock import patch

            with patch.object(rural_link, "field_profiles", return_value=profiles):
                rural_link.main(None)
        elif path.name == "upload_routing.csv":
            subprocess.run(
                [sys.executable, str(root / "scripts/evaluate_upload.py")],
                cwd=root,
                check=True,
                stdout=subprocess.DEVNULL,
            )
        elif path.name == "dual_existing.csv":
            subprocess.run(
                [sys.executable, str(root / "scripts/dual_existing.py")],
                cwd=root,
                check=True,
                stdout=subprocess.DEVNULL,
            )


def reference_path(fieldnames):
    for name in ("source_path", "path", "image_id", "source_id"):
        if name in fieldnames:
            return name
    return None


def csv_export(src, dst):
    with src.open(newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        id_key = reference_path(fields)
        keys = sorted(LABEL_KEYS.intersection(fields))
        if not id_key or not keys:
            shutil.copyfile(src, dst)
            return None
        rows = list(reader)
    changed = 0
    for row in rows:
        if CORNELL.fullmatch(row[id_key]):
            for key in keys:
                row[key] = ""
            changed += 1
    if not changed:
        shutil.copyfile(src, dst)
        return None
    with dst.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return dict(id_key=id_key, fields=keys, n=changed)


def npz_export(src, dst):
    with np.load(src, allow_pickle=False) as z:
        drop = {}
        for key in z.files:
            if key not in (
                "y",
                "test_y",
                "cornell_y",
                "labels",
                "training_y",
                "train_y",
                "cal_y",
            ):
                continue
            preferred = {
                "test_y": ["test_paths"],
                "cornell_y": ["cornell_paths", "cornell_path"],
                "training_y": ["training_paths"],
                "train_y": ["train_paths"],
                "cal_y": ["cal_paths", "paths"],
            }
            options = preferred.get(key, []) + [
                "source_path",
                "paths",
                "test_paths",
                "train_paths",
                "source_paths",
                "fit_paths",
                "ids",
            ]
            for opt in options:
                if opt in z.files and z[opt].shape == z[key].shape:
                    sample = z[opt].flat[0]
                    if isinstance(sample, (str, np.str_)) and CORNELL.fullmatch(
                        str(sample)
                    ):
                        drop[key] = dict(ids=opt, dtype=z[key].dtype.str)
                        break
        if not drop:
            shutil.copyfile(src, dst)
            return None
        retained = {k: z[k] for k in z.files if k not in drop}
    if src.name == SENSITIVITY_CACHE:
        # The empirical target is itself an observed positive calibration score.
        # Its test decision boundary can identify the same positive image.
        retained["native_threshold"] = retained.pop("threshold")[:, :, 0]
        retained["native_pred"] = retained.pop("pred")[:, :, 0]
    np.savez_compressed(dst, **retained)
    return drop


def export(target):
    target = target.resolve()
    if target == ROOT or target.is_relative_to(ROOT):
        raise ValueError("Use a distinct public staging directory")
    if target.exists() and any(target.iterdir()):
        raise ValueError(
            "Export into a new or empty directory to prevent stale private files from surviving"
        )
    if (ROOT / INDEX).exists():
        raise ValueError(
            "Export from the original private workspace, not a redacted or partially restored staging copy"
        )
    index = dict(
        version=2,
        csv={},
        npz={},
        fine={},
        order_statistics={},
        local_feature_caches=sorted(EXCLUDED | {"data/features/latvia.npz"}),
    )
    for path in inputs():
        name = path.relative_to(ROOT).as_posix()
        if name in EXCLUDED or name == INDEX:
            continue
        dst = target / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".csv":
            fine = fine_export(path, dst)
            if fine:
                index["fine"][name] = fine
            else:
                rule = csv_export(path, dst)
                if rule:
                    index["csv"][name] = rule
        elif path.suffix == ".npz":
            rule = npz_export(path, dst)
            if rule:
                index["npz"][name] = rule
            if path.name == SENSITIVITY_CACHE:
                index["order_statistics"][name] = dict(
                    fields=["pred", "threshold"],
                    withheld_operating_point="target95",
                    retained=["native_pred", "native_threshold"],
                    reason="Exact positive-class calibration order statistic",
                )
        else:
            shutil.copyfile(path, dst)
    (target / INDEX).write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(
        f"Public export: {len(index['csv'])} identity-linked CSV joins, {len(index['npz'])} array joins and {len(index['fine'])} fine-grained evaluation tables."
    )


def restore(root, official):
    index = json.loads((root / INDEX).read_text())
    if index.get("version") not in (1, 2):
        raise ValueError("Unrecognized public data index")
    for section in ("csv", "npz", "fine", "order_statistics"):
        for name in index.get(section, {}):
            path = root / name
            if (
                Path(name).is_absolute()
                or ".." in Path(name).parts
                or path.resolve() != path
                or not path.is_file()
            ):
                raise ValueError("Invalid indexed result path: " + name)
    with official.open(newline="", encoding="utf-8-sig") as f:
        labels = {}
        for r in csv.DictReader(f):
            classes = ("healthy", "multiple_diseases", "rust", "scab")
            if any(r[k] not in ("0", "1") for k in classes):
                raise ValueError("Expected one-hot labels")
            category = [k for k in classes if r[k] == "1"]
            if len(category) != 1 or r["image_id"] in labels:
                raise ValueError("Invalid input label table")
            labels[r["image_id"]] = category[0]
    if set(labels) != {f"Train_{i}" for i in range(1821)}:
        raise ValueError("Wrong FGVC7 cohort")
    # Semantic fingerprint independent of platform-specific newlines.
    signature = hashlib.sha256(
        "\n".join(f"{k},{v}" for k, v in sorted(labels.items())).encode()
    ).hexdigest()
    if signature != OFFICIAL_LABEL_DIGEST:
        raise ValueError("The supplied labels differ from the benchmark")
    for name, spec in index["csv"].items():
        path = root / name
        with path.open(newline="") as f:
            data = csv.DictReader(f)
            fields = data.fieldnames
            rows = list(data)
        count = 0
        for row in rows:
            source = row[spec["id_key"]]
            if not CORNELL.fullmatch(source):
                continue
            label = labels[Path(source).stem]
            for field in spec["fields"]:
                row[field] = (
                    label if field == "source_label" else str(int(label != "healthy"))
                )
            count += 1
        if count != spec["n"]:
            raise ValueError("Unexpected Cornell row count in " + name)
        tmp = path.with_suffix(".joining")
        with tmp.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        tmp.replace(path)
    for name, rules in index["npz"].items():
        path = root / name
        with np.load(path, allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        for key, spec in rules.items():
            ids = data[spec["ids"]]
            data[key] = np.array(
                [labels[Path(p).stem] != "healthy" for p in ids.flat],
                dtype=spec["dtype"],
            ).reshape(ids.shape)
        tmp = path.with_suffix(".joining.npz")
        np.savez_compressed(tmp, **data)
        tmp.replace(path)
    for name in index.get("order_statistics", {}):
        path = root / name
        with np.load(path, allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        if "native_threshold" not in data and {"pred", "threshold"}.issubset(data):
            continue
        native = data.pop("native_threshold")
        native_pred = data.pop("native_pred")
        thresholds = np.empty((*native.shape, 2), dtype=native.dtype)
        thresholds[:, :, 0] = native
        predictions = np.empty(
            (*native.shape, 2, native_pred.shape[-1]), dtype=native_pred.dtype
        )
        predictions[:, :, 0] = native_pred
        from sensitivity_calibration import choose_cutoff

        for i, ids in enumerate(data["calibration_paths"]):
            y = np.array([labels[Path(p).stem] != "healthy" for p in ids])
            for j in range(native.shape[1]):
                threshold = choose_cutoff(data["calibration_score"][i, j], y)
                thresholds[i, j, 1] = threshold
                predictions[i, j, 1] = data["test_score"][i, j] >= threshold
        data.update(pred=predictions, threshold=thresholds)
        tmp = path.with_suffix(".joining.npz")
        np.savez_compressed(tmp, **data)
        tmp.replace(path)
    if index.get("fine"):
        if root.resolve() == ROOT.resolve():
            restore_fine(root, index)
        else:
            subprocess.run(
                [
                    sys.executable,
                    str(root / "scripts/public_release.py"),
                    "rebuild-fine",
                    "--root",
                    str(root),
                ],
                cwd=root,
                check=True,
            )
    print(
        "Joined authorized Cornell references and reconstructed local evaluation statistics."
    )


OFFICIAL_LABEL_DIGEST = (
    "842be6d511da3577246ef219edfe265961879ce68896924be04cb8926889872c"
)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    m = p.add_subparsers(dest="command", required=True)
    e = m.add_parser("export")
    e.add_argument("target", type=Path)
    h = m.add_parser("restore")
    h.add_argument("official_csv", type=Path)
    h.add_argument("--root", type=Path, default=ROOT)
    j = m.add_parser("rebuild-fine")
    j.add_argument("--root", type=Path, default=ROOT)
    a = p.parse_args()
    if a.command == "export":
        export(a.target)
    elif a.command == "restore":
        restore(a.root.resolve(), a.official_csv)
    else:
        restore_fine(a.root.resolve(), json.loads((a.root / INDEX).read_text()))

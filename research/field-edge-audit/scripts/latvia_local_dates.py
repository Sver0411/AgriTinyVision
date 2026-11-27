"""Calibrate the sealed directional-rule family on early Latvian EXIF capture days."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from directional_audit import HEAD_FILE, digest, evaluate_scores, select_fixed_grid
from directional_policy import (
    CALIBRATION,
    EDGE_CUTOFF,
    Rule,
    decide,
    pack,
    probabilities_from_logits,
    select_queries,
    unpack,
)
from edge_adaptation_split import ROOT, RESULTS, table, write_table

FEATURES = ROOT / "data/features/latvia.npz"
PROTOCOL = ROOT / "protocol/LATVIA_LOCAL_DATES.md"
LOCAL_JSON = RESULTS / "latvia_local_directional_rule.json"
LOCAL_BIN = RESULTS / "latvia_local_directional_rule.bin"
SPLIT = RESULTS / "latvia_local_date_split.csv"
SUPPORT = RESULTS / "latvia_local_support.json"


def capture_date(text):
    try:
        return datetime.strptime(text.strip(), "%Y:%m:%d %H:%M:%S").date().isoformat()
    except ValueError:
        return ""


def components(rows):
    """Include unknown-date records in the graph to retain transitive paths."""
    roots = list(range(len(rows)))
    hashes = [int(r["phash"], 16) for r in rows]

    def find(i):
        while roots[i] != i:
            roots[i] = roots[roots[i]]
            i = roots[i]
        return i

    for i, h in enumerate(hashes):
        for j in range(i):
            if (h ^ hashes[j]).bit_count() <= 4 or rows[i]["source_sha256"] == rows[j][
                "source_sha256"
            ]:
                a, b = find(i), find(j)
                roots[max(a, b)] = min(a, b)
    result = defaultdict(list)
    for i in range(len(rows)):
        result[find(i)].append(i)
    return list(result.values())


def assign(rows):
    dates = [capture_date(r["exif_datetime_original"]) for r in rows]
    labels = [int(r["y"]) for r in rows]
    if any(label not in (0, 1) for label in labels):
        raise ValueError("Nonbinary scab reference")
    info = [None] * len(rows)
    for members in components(rows):
        observed = {dates[i] for i in members if dates[i]}
        conflict = len({labels[i] for i in members}) > 1
        cross = len(observed) > 1
        representative = min(rows[i]["source_path"] for i in members)
        for i in members:
            reason = (
                "cross_date_and_class_conflict"
                if cross and conflict
                else "cross_date"
                if cross
                else "conflicting_reference"
                if conflict
                else "missing_original_date"
                if not dates[i]
                else ""
            )
            info[i] = dict(
                source_path=rows[i]["source_path"],
                source_sha256=rows[i]["source_sha256"],
                phash=rows[i]["phash"],
                capture_date=dates[i],
                y=labels[i],
                component_representative=representative,
                component_size=len(members),
                status=reason,
            )
    eligible = sorted({r["capture_date"] for r in info if not r["status"]})
    cut = (len(eligible) + 1) // 3
    early = set(eligible[:cut])
    for r in info:
        if not r["status"]:
            r["status"] = "calibration" if r["capture_date"] in early else "test"
    supported = len(eligible) >= 3
    reasons = []
    if not supported:
        reasons.append("fewer_than_three_capture_days")
    for pool in ("calibration", "test"):
        if {r["y"] for r in info if r["status"] == pool} != {0, 1}:
            reasons.append(pool + "_missing_one_or_both_classes")
    return info, eligible, cut, reasons


def prepare(manifest):
    source = table(manifest)
    if len(source) != 1190 or len({r["source_path"] for r in source}) != 1190:
        raise ValueError("Incomplete Latvia manifest")
    if "exif_datetime_original" not in source[0]:
        raise ValueError("Final camera-time join required")
    info, dates, cut, reasons = assign(source)
    statuses = Counter(r["status"] for r in info)
    groups = defaultdict(list)
    for r in info:
        groups[r["capture_date"]].append(r)
    date_rows = []
    for d, rr in sorted(groups.items()):
        date_rows.append(
            dict(
                capture_date=d or "unknown",
                official_total=len(rr),
                official_non_scab=sum(r["y"] == 0 for r in rr),
                official_scab=sum(r["y"] == 1 for r in rr),
                excluded=sum(r["status"] not in ("calibration", "test") for r in rr),
                retained_calibration=sum(r["status"] == "calibration" for r in rr),
                retained_test=sum(r["status"] == "test" for r in rr),
                retained_non_scab=sum(
                    r["status"] in ("calibration", "test") and r["y"] == 0 for r in rr
                ),
                retained_scab=sum(
                    r["status"] in ("calibration", "test") and r["y"] == 1 for r in rr
                ),
            )
        )
    write_table(SPLIT, info)
    write_table(RESULTS / "latvia_local_date_support.csv", date_rows)
    record = dict(
        protocol_sha256=digest(PROTOCOL.read_bytes()),
        official_manifest_sha256=digest(manifest.read_bytes()),
        source_count=len(info),
        status_count=dict(statuses),
        eligible_dates=dates,
        calibration_dates=dates[:cut],
        test_dates=dates[cut:],
        supported=not reasons,
        reason=reasons,
        split_file_sha256=digest(SPLIT.read_bytes()),
        distinct_exact_files=len({r["source_sha256"] for r in info}),
        components=len({r["component_representative"] for r in info}),
        dated_count=sum(bool(r["capture_date"]) for r in info),
        calibration_labels=dict(
            Counter(r["y"] for r in info if r["status"] == "calibration")
        ),
        test_labels=dict(Counter(r["y"] for r in info if r["status"] == "test")),
    )
    SUPPORT.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


def read_pool(kind):
    support = json.loads(SUPPORT.read_text())
    if not support["supported"]:
        raise ValueError("Insufficient site/date support: " + str(support["reason"]))
    if support["split_file_sha256"] != digest(SPLIT.read_bytes()) or support[
        "protocol_sha256"
    ] != digest(PROTOCOL.read_bytes()):
        raise ValueError("The locked date cohort changed")
    selected = [r for r in table(SPLIT) if r["status"] == kind]
    with np.load(FEATURES, allow_pickle=False) as z:
        lookup = {p: i for i, p in enumerate(z["source_path"])}
        if len(lookup) != 1190:
            raise ValueError("Incomplete extract")
        idx = np.array([lookup[r["source_path"]] for r in selected])
        if not np.array_equal(
            z["source_sha256"][idx], [r["source_sha256"] for r in selected]
        ):
            raise ValueError("Source photo changed")
        if not np.array_equal(z["phash"][idx], [r["phash"] for r in selected]):
            raise ValueError("DCT hashes differ")
        if not np.array_equal(z["y"][idx], [int(r["y"]) for r in selected]):
            raise ValueError("Reference labels differ")
        if str(z["source_head_sha256"].item()) != digest(HEAD_FILE.read_bytes()):
            raise ValueError("Wrong source DINO head")
        if int(z["jpeg_quality"].item()) != 75:
            raise ValueError("Incorrect JPEG quality")
        # Match released float32 two-logit softmax. Detect any disagreement with the
        # equivalent signed-margin decision before including the records.
        logits = z["logits"][idx].astype(np.float32)
        ex = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        e = (ex[:, 1] / ex.sum(axis=1)).astype(np.float32)
        if not np.array_equal(
            e >= EDGE_CUTOFF,
            probabilities_from_logits(z["edge_margin"][idx]) >= EDGE_CUTOFF,
        ):
            raise ValueError("Released edge hard decision mismatch")
        d = probabilities_from_logits(z["dino_balanced_margin"][idx])
        sizes = z["jpeg_bytes"][idx].astype(np.int64)
        if np.any(sizes <= 0):
            raise ValueError("Unmeasured JPEG length")
        y = np.array([int(r["y"]) for r in selected], dtype=np.uint8)
    return selected, y, e, d, sizes


def seal_file(path, data):
    if path.exists() and path.read_bytes() != data:
        raise ValueError("A different sealed artifact exists: " + str(path))
    path.write_bytes(data)


def fit():
    rows, y, e, d, sizes = read_pool("calibration")
    base, scored, primary, fallback, selected, tier = select_fixed_grid(y, e, d, sizes)
    idx, rule, fitted = selected
    blob = pack(rule)
    cal_bytes = int(sizes.sum()) + len(y) * CALIBRATION.size
    metadata = dict(
        protocol_sha256=digest(PROTOCOL.read_bytes()),
        split_sha256=digest(SPLIT.read_bytes()),
        cornell_rule_sha256=digest((RESULTS / "directional_rule.bin").read_bytes()),
        source_balanced_head_sha256=digest(HEAD_FILE.read_bytes()),
        rule=asdict(rule),
        binary_sha256=digest(blob),
        grid_index=idx,
        grid_n=len(scored),
        n_primary=len(primary),
        n_strict=len(fallback),
        selection_tier=tier,
        calibration_labels=len(y),
        calibration_jpeg_bytes=int(sizes.sum()),
        calibration_upload_bytes=cal_bytes,
        deployment_update_bytes=len(blob) if rule.rescue or rule.retract else 0,
        baseline=base,
        selected=fitted,
        calibration_paths_sha256=digest(
            ("\n".join(r["source_path"] for r in rows) + "\n").encode()
        ),
    )
    seal_file(LOCAL_BIN, blob)
    seal_file(LOCAL_JSON, (json.dumps(metadata, indent=2) + "\n").encode())
    print(json.dumps(metadata, indent=2))


def run_test():
    metadata = json.loads(LOCAL_JSON.read_text())
    blob = LOCAL_BIN.read_bytes()
    if metadata["binary_sha256"] != digest(blob) or metadata["split_sha256"] != digest(
        SPLIT.read_bytes()
    ):
        raise ValueError("Latvia local rule or split mismatch")
    cornell = (RESULTS / "directional_rule.bin").read_bytes()
    if digest(cornell) != metadata["cornell_rule_sha256"]:
        raise ValueError("Cornell rule changed")
    rows, y, e, d, sizes = read_pool("test")
    dates = np.array([r["capture_date"] for r in rows])
    blocks = [("ALL", np.ones(len(y), bool))] + [
        (date, dates == date) for date in sorted(set(dates))
    ]
    report = []
    predictions = []
    for method, rule in (
        ("frozen", Rule()),
        ("cornell_transport", unpack(cornell)),
        ("latvia_local", unpack(blob)),
    ):
        pred = decide(e, d, rule)
        ask = select_queries(e, rule)
        for i, r in enumerate(rows):
            predictions.append(
                dict(
                    method=method,
                    source_path=r["source_path"],
                    date=dates[i],
                    y=int(y[i]),
                    edge=int(e[i] >= EDGE_CUTOFF),
                    prediction=int(pred[i]),
                    queried=int(ask[i]),
                    query_jpeg_bytes=int(sizes[i]) if ask[i] else 0,
                )
            )
        for day, m in blocks:
            base = evaluate_scores(y[m], e[m], d[m], sizes[m], Rule())
            out = evaluate_scores(y[m], e[m], d[m], sizes[m], rule)
            npos = int(np.count_nonzero(y[m] == 1))
            nneg = int(np.count_nonzero(y[m] == 0))
            report.append(
                dict(
                    method=method,
                    date=day,
                    n=npos + nneg,
                    scab=npos,
                    non_scab=nneg,
                    tn=nneg - out["fp"],
                    tp=npos - out["fn"],
                    **out,
                    baseline_fp=base["fp"],
                    baseline_fn=base["fn"],
                    delta_fp=out["fp"] - base["fp"],
                    delta_fn=out["fn"] - base["fn"],
                    strict_dual=int(out["fp"] < base["fp"] and out["fn"] < base["fn"]),
                    sensitivity=(npos - out["fn"]) / npos if npos else "",
                    specificity=(nneg - out["fp"]) / nneg if nneg else "",
                )
            )
    write_table(RESULTS / "latvia_local_directional_outcomes.csv", report)
    write_table(RESULTS / "latvia_local_directional_predictions.csv", predictions)
    print(json.dumps([r for r in report if r["date"] == "ALL"], indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("prepare", "fit", "test"))
    ap.add_argument("--manifest", type=Path, default=ROOT / "data/latvia_source/manifest.csv")
    args = ap.parse_args()
    if args.stage == "prepare":
        prepare(args.manifest)
    elif args.stage == "fit":
        fit()
    else:
        run_test()

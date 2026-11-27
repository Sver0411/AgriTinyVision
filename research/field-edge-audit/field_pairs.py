"""Rebuild and audit the original 60-image PlantDoc paired deployment cohort.

python field_pairs.py --prepare [--source /path/to/pinned/AgriTinyVision]
python field_pairs.py --json results/board_audit.json
"""

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
UPSTREAM = "https://github.com/Sver0411/AgriTinyVision"
REVISION = "16e7a2fdec43c02258f22e85b085ac269345e124"
PLANTDOC = "5467f6012d78d1c446145d5f582da6096f852ae8"
DIRECTORY = "reports/field_validation"
BINARY = 0.30000001192092896
LOWER = 0.20000000298023224
UPPER = 0.4000000059604645
FILES = {
    "SOURCE_MANIFEST.json": "1be05b4d0c05cd31ce32ccacb2e043060bcbf8976b7405f9733ba938cff973c7",
    "INFERENCE_PARITY.json": "1d79ee3c835f53064d3a6bec465ae634f436b733c5f1d786ab55b8779cdf46df",
    "FIELD_SPLIT.json": "e1398af101da13841ce9f729d08657e0732eafc2808e9cadda3933b2009c7685",
    "INFERENCE_RESULTS.csv": "561568a65c2899d7785813b1d16fbc27f08b10c3b1fde01df1761c8899f22d33",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_report(name, source):
    if source:
        data = (source / DIRECTORY / name).read_bytes()
    else:
        url = f"https://raw.githubusercontent.com/Sver0411/AgriTinyVision/{REVISION}/{DIRECTORY}/{name}"
        with urllib.request.urlopen(url, timeout=30) as reply:
            data = reply.read()
    require(
        hashlib.sha256(data).hexdigest() == FILES[name],
        "Upstream report SHA-256 mismatch: " + name,
    )
    if name.endswith(".csv"):
        return list(csv.DictReader(io.StringIO(data.decode())))
    return json.loads(data)


def by_id(items):
    indexed = {r["sample_id"]: r for r in items}
    require(len(indexed) == len(items), "Duplicate sample ID")
    return indexed


def binary(p):
    return "SUSPECT" if p >= BINARY else "NORMAL"


def prepare(csv_path, source):
    manifest = read_report("SOURCE_MANIFEST.json", source)
    parity = read_report("INFERENCE_PARITY.json", source)
    splits = read_report("FIELD_SPLIT.json", source)
    measurements = by_id(read_report("INFERENCE_RESULTS.csv", source))
    ms = by_id(manifest["samples"])
    ps = by_id(parity["samples"])
    ss = by_id(splits["samples"])
    require(
        parity["evidence"] == "hardware" and manifest["source"]["revision"] == PLANTDOC,
        "Source identity",
    )
    require(
        splits["counts"] == {"unassigned": 60} and splits["independent_groups"] == 1,
        "Unexpected independent groups",
    )
    require(
        len(ms) == 60 and ms.keys() == ps.keys() == ss.keys() == measurements.keys(),
        "Paired IDs differ",
    )
    contract = parity["expected_identity"]
    require(
        (BINARY, LOWER, UPPER)
        == (
            contract["decision_threshold"],
            contract["normal_upper_bound"],
            contract["suspect_lower_bound"],
        ),
        "Decision cutoffs changed",
    )
    rows = []
    for key, ref in ms.items():
        pair = ps[key]
        geo = ss[key]
        raw = measurements[key]
        require(
            ref["rgb_sha256"]
            == pair["rgb_sha256"]
            == raw["rgb_sha256"]
            == geo["rgb_sha256"],
            "Unequal RGB fingerprints: " + key,
        )
        require(
            ref["label"] == raw["reference_label"]
            and raw["label_origin"] == "dataset_author",
            "True label provenance",
        )
        require(
            pair["host"] is None
            and pair["errors"] == {}
            and raw["device_status"] == "ok",
            "Missing on-board/FP32 pair",
        )
        require(
            not ref["excluded_from_evaluation"]
            and raw["excluded_from_evaluation"] == "False",
            "Unexpected exclusion",
        )
        require(
            geo["split"] is None and ref["source_split"] == "test", "Split provenance"
        )
        h, b = pair["fp32"], pair["device"]
        hscore, bscore = h["probabilities"][1], b["probabilities"][1]
        require(
            raw["state"] == b["state"]
            and float(raw["suspect_probability"]) == bscore
            and b["quality_passed"],
            "Inconsistent on-device prediction",
        )
        for pred, score in [(h, hscore), (b, bscore)]:
            require(math.isfinite(score) and 0 <= score <= 1, "Invalid score")
            expected = (
                "NORMAL"
                if score < LOWER
                else "SUSPECT"
                if score >= UPPER
                else "UNKNOWN"
            )
            require(pred["state"] == expected, "Frozen three-state disagreement")
        stats = pair["quality"]["metrics"]
        rows.append(
            {
                "image_id": key,
                "common_species": ref["crop_type"].lower(),
                "true_health": {"NORMAL": "healthy", "SUSPECT": "diseased"}[
                    ref["label"]
                ],
                "reference_label": ref["label"],
                "source_class": ref["source_label"],
                "source_id": ref["source_path"],
                "source_file_sha256": ref["sha256"],
                "source_git_blob_sha1": ref["git_blob_sha1"],
                "source_rgb_sha256": geo["original_rgb_sha256"],
                "rgb_sha256": ref["rgb_sha256"],
                "source_split": ref["source_split"],
                "audit_split": "unassigned",
                "batch_id": geo["batch_id"],
                "plant_id": geo["plant_id"] or "",
                "session_id": geo["session_id"] or "",
                "host_fp32_score": hscore,
                "board_int8_score": bscore,
                "host_fp32_binary": binary(hscore),
                "board_int8_binary": binary(bscore),
                "host_fp32_state": h["state"],
                "board_int8_state": b["state"],
                "host_fp32_inference_wall_ms": h["wall_ms"],
                "board_int8_inference_us": b["inference_us"],
                "board_end_to_end_ms": b["wall_ms"],
                "quality_gate_passed": int(b["quality_passed"]),
                "input_saturated_fraction": pair["saturation"]["saturated_fraction"],
                "brightness": stats["brightness"],
                "contrast": stats["contrast"],
                "laplacian_variance": stats["laplacian_variance"],
                "source_url": ref["source_url"],
            }
        )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def action_outcomes(truth, actions):
    counts = Counter(
        "correct" if action == y else "deferred" if action == "UNKNOWN" else "wrong"
        for y, action in zip(truth, actions, strict=True)
    )
    accepted = counts["correct"] + counts["wrong"]
    return {
        "n": len(truth),
        "correct": counts["correct"],
        "wrong": counts["wrong"],
        "deferred": counts["deferred"],
        "coverage": accepted / len(truth),
        "accuracy_among_accepted": counts["correct"] / accepted if accepted else None,
        "correct_fraction_all": counts["correct"] / len(truth),
    }


def matrix(truth, actions, states):
    return {
        y: {
            s: sum(yy == y and a == s for yy, a in zip(truth, actions, strict=True))
            for s in states
        }
        for y in ("NORMAL", "SUSPECT")
    }


def summary(csv_path):
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    require(
        len(rows) == 60 and len({r["image_id"] for r in rows}) == 60,
        "Expected the complete 60-image cohort",
    )
    truth = [r["reference_label"] for r in rows]
    board = [r["board_int8_binary"] for r in rows]
    host = [r["host_fp32_binary"] for r in rows]
    bs = [r["board_int8_state"] for r in rows]
    hs = [r["host_fp32_state"] for r in rows]
    cases = Counter()
    split_buckets = Counter()
    discord = []
    for r, y, e, h, es, ss in zip(rows, truth, board, host, bs, hs, strict=True):
        cases[(e == y, h == y)] += 1
        if e != h or es != ss:
            discord.append(
                {
                    k: r[k]
                    for k in (
                        "image_id",
                        "source_class",
                        "reference_label",
                        "host_fp32_score",
                        "board_int8_score",
                        "host_fp32_binary",
                        "board_int8_binary",
                        "host_fp32_state",
                        "board_int8_state",
                    )
                }
            )
        split_buckets[r["audit_split"]] += 1
    switch = [h if e == "UNKNOWN" else e for e, h in zip(bs, hs, strict=True)]
    bd = {
        i: sum(r["common_species"] == i for r in rows)
        for i in sorted({r["common_species"] for r in rows})
    }
    diffs = [
        abs(float(r["host_fp32_score"]) - float(r["board_int8_score"])) for r in rows
    ]
    nplus = cases[(False, True)]
    nminus = cases[(True, False)]
    nchange = nplus + nminus
    exact_p = min(
        1.0,
        2
        * sum(math.comb(nchange, k) for k in range(min(nplus, nminus) + 1))
        / 2**nchange,
    )
    return {
        "source_revision": REVISION,
        "plantdoc_revision": PLANTDOC,
        "n": len(rows),
        "crop_counts": bd,
        "class_counts": dict(sorted(Counter(r["source_class"] for r in rows).items())),
        "reference_counts": dict(Counter(truth)),
        "source_groups": 1,
        "source_partitions": dict(Counter(r["source_split"] for r in rows)),
        "audit_splits": dict(split_buckets),
        "binary_disagreement_n": nchange,
        "three_state_disagreement_n": sum(a != b for a, b in zip(bs, hs, strict=True)),
        "board_binary": matrix(truth, board, ("NORMAL", "SUSPECT")),
        "fp32_binary": matrix(truth, host, ("NORMAL", "SUSPECT")),
        "paired_binary_outcomes": {
            "both_correct": cases[(True, True)],
            "board_only_correct": nminus,
            "fp32_only_correct": nplus,
            "both_wrong": cases[(False, False)],
        },
        "mcnemar_two_sided_exact_descriptive": exact_p,
        "binary_oracle_correct": cases[(True, True)] + nplus + nminus,
        "binary_oracle_ceiling_gain_by_budget_queries": [
            min(k, nplus) for k in range(len(rows) + 1)
        ],
        "frozen_board": action_outcomes(truth, bs),
        "frozen_fp32": action_outcomes(truth, hs),
        "route_board_unknown_to_fp32": {
            "calls": bs.count("UNKNOWN"),
            **action_outcomes(truth, switch),
        },
        "three_state_board": matrix(truth, bs, ("NORMAL", "UNKNOWN", "SUSPECT")),
        "three_state_fp32": matrix(truth, hs, ("NORMAL", "UNKNOWN", "SUSPECT")),
        "mean_absolute_suspect_score_difference": statistics.mean(diffs),
        "maximum_absolute_suspect_score_difference": max(diffs),
        "discordant_samples": discord,
        "interpretation": "The historical external cohort has one conservative source group. Oracle ceilings describe these paired outcomes only. Calls model routing to the recorded FP32 model; no network transfer or external expert is measured.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Rebuild the paired CSV from SHA-256-verified reports",
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="Existing pinned AgriTinyVision checkout; otherwise fetch only four reports",
    )
    parser.add_argument("--csv", type=Path, default=HERE / "data/board_pairs.csv")
    parser.add_argument(
        "--json", type=Path, help="Write the audit as JSON; default is standard output"
    )
    opt = parser.parse_args()
    if opt.prepare:
        prepare(opt.csv, opt.source)
    out = json.dumps(summary(opt.csv), indent=2, allow_nan=False) + "\n"
    if opt.json:
        opt.json.parent.mkdir(parents=True, exist_ok=True)
        opt.json.write_text(out)
    else:
        print(out, end="")


if __name__ == "__main__":
    main()

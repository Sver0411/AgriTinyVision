"""Select a single two-direction policy on 450 labels, then replay the sealed binary."""

import argparse
import csv
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from directional_policy import (
    Rule,
    REQUEST,
    REPLY,
    CALIBRATION,
    EDGE_CUTOFF,
    probabilities_from_logits,
    select_queries,
    decide,
    pack,
    unpack,
)
from edge_adaptation_split import RESULTS, ROOT, table, write_table
from edge_adaptation_wire import normalized

JSON_FILE = RESULTS / "directional_rule.json"
BIN_FILE = RESULTS / "directional_rule.bin"
HEAD_FILE = ROOT / "data/features/dino_plantdoc_balanced_head.npz"
RESCUE_EDGE = (0.0, 0.05, 0.20)
RETRACT_EDGE = (0.40, 0.70, 0.95, 1.0)
RESCUE_DINO = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
RETRACT_DINO = (0.50, 0.40, 0.30, 0.20, 0.10, 0.05)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verified_payload(path, expected_size):
    data = path.read_bytes()
    if len(data) != int(expected_size) or digest(data) != path.stem:
        raise ValueError("Payload identity failure: " + str(path))
    return data


def cornell(which):
    paths = [
        r["source_path"]
        for r in table(RESULTS / "edge_adaptation_split.csv")
        if r["status"] == which
    ]
    ids = set(paths)
    if len(paths) != len(ids) or len(paths) != {"label_pool": 450, "test": 1353}[which]:
        raise ValueError("Wrong pool")

    def subset(file, additional=lambda r: True):
        with (RESULTS / file).open() as stream:
            return {
                r["source_path"]: r
                for r in csv.DictReader(stream)
                if r["source_path"] in ids and additional(r)
            }

    edge = subset("fgvc_edge.csv")
    remote = subset("specialist_balanced.csv", lambda r: r["feature_dataset"] == "fgvc")
    payload = subset(
        "fgvc_backend.csv",
        lambda r: r["long_edge"] == "224" and r["jpeg_quality"] == "75",
    )
    if not ids == set(edge) == set(remote) == set(payload):
        raise ValueError("Pool IDs mismatch")
    with (
        np.load(HEAD_FILE, allow_pickle=False) as head,
        np.load(ROOT / "data/features/dino_fgvc.npz", allow_pickle=False) as feats,
    ):
        lookup = {p: i for i, p in enumerate(feats["source_path"])}
        raw = normalized(feats["cls"][[lookup[p] for p in paths]]) @ head[
            "coef"
        ].ravel() + float(head["intercept"].item())
    if not np.allclose(
        raw, [float(remote[p]["score"]) for p in paths], rtol=0, atol=1e-12
    ):
        raise ValueError("Source balanced logit mismatch")
    e = np.array([float(edge[p]["p_suspect"]) for p in paths], np.float32)
    y = np.array([int(edge[p]["true_diseased"]) for p in paths], np.uint8)
    d = probabilities_from_logits(raw)
    if not np.array_equal(
        e >= EDGE_CUTOFF, [int(edge[p]["binary_pred"]) for p in paths]
    ) or not np.array_equal(y, [int(remote[p]["y"]) for p in paths]):
        raise ValueError("Inconsistent original decisions or labels")
    j = [
        verified_payload(
            ROOT / "data/fgvc_backend_jpegs" / (payload[p]["jpeg_sha256"] + ".jpg"),
            payload[p]["jpeg_bytes"],
        )
        for p in paths
    ]
    return SimpleNamespace(
        ids=paths, numid=[int(Path(p).stem[6:]) for p in paths], y=y, e=e, d=d, j=j
    )


def kashmir():
    rr = table(RESULTS / "transfer_manifest.csv")
    ids = [r["source_path"] for r in rr]
    with (
        np.load(ROOT / "data/features/kashmir.npz", allow_pickle=False) as f,
        np.load(HEAD_FILE, allow_pickle=False) as h,
    ):
        if list(f["source_path"]) != ids:
            raise ValueError("Kashmir feature order")
        raw = normalized(f["dino"]) @ h["coef"].ravel() + float(h["intercept"].item())
    e = probabilities_from_logits([float(r["edge_score_logit"]) for r in rr])
    if not np.array_equal(e >= EDGE_CUTOFF, [int(r["edge_pred"]) for r in rr]):
        raise ValueError("Kashmir baseline mismatch")
    j = [
        verified_payload(
            ROOT / "data/raw/transfer_jpegs" / (r["jpeg_sha256"] + ".jpg"),
            r["jpeg_bytes"],
        )
        for r in rr
    ]
    return SimpleNamespace(
        ids=ids,
        numid=list(range(len(rr))),
        e=e,
        d=probabilities_from_logits(raw),
        y=np.array([int(r["y"]) for r in rr], np.uint8),
        j=j,
    )


def counts(y, edge, pred, ask, sizes):
    fp = int(np.count_nonzero(pred & (y == 0)))
    fn = int(np.count_nonzero(~pred & (y == 1)))
    rescued = (~edge) & pred
    retracted = edge & (~pred)
    up = int(sizes[ask].sum()) + int(ask.sum()) * REQUEST.size
    down = int(ask.sum()) * REPLY.size
    return dict(
        fp=fp,
        fn=fn,
        true_rescues=int(np.count_nonzero(rescued & (y == 1))),
        false_rescues=int(np.count_nonzero(rescued & (y == 0))),
        true_retractions=int(np.count_nonzero(retracted & (y == 0))),
        false_retractions=int(np.count_nonzero(retracted & (y == 1))),
        retractions=int(retracted.sum()),
        query_count=int(ask.sum()),
        jpeg_bytes=int(sizes[ask].sum()),
        inference_upload_bytes=up,
        inference_reply_bytes=down,
        inference_total_bytes=up + down,
    )


def candidate_rules():
    rescue = [None] + list(itertools.product(RESCUE_EDGE, RESCUE_DINO))
    retract = [None] + list(itertools.product(RETRACT_EDGE, RETRACT_DINO))
    for aa, bb in itertools.product(rescue, retract):
        lo, rc = (0.0, 0.5) if aa is None else aa
        hi, cc = (1.0, 0.5) if bb is None else bb
        yield unpack(pack(Rule(aa is not None, bb is not None, lo, hi, rc, cc)))


def evaluate_scores(y, e, d, sizes, rule):
    edge = e >= EDGE_CUTOFF
    ask = select_queries(e, rule)
    pred = decide(e, d, rule)
    return counts(y, edge, pred, ask, sizes)


def evaluate(data, rule):
    sizes = np.fromiter((len(j) for j in data.j), dtype=np.int64)
    return evaluate_scores(data.y, data.e, data.d, sizes, rule)


def select_fixed_grid(y, e, d, sizes):
    """Use the same calibrated candidates and preference order in both local studies."""
    base = evaluate_scores(y, e, d, sizes, Rule())
    scored = []
    for idx, rule in enumerate(candidate_rules()):
        r = evaluate_scores(y, e, d, sizes, rule)
        scored.append((idx, rule, r))
    if len(scored) != 475:
        raise ValueError("Grid changed")
    feasible = [
        t
        for t in scored
        if t[2]["fn"] <= base["fn"] // 2
        and t[2]["fp"] < base["fp"]
        and t[2]["fn"] < base["fn"]
    ]
    fallback = [
        t for t in scored if t[2]["fn"] < base["fn"] and t[2]["fp"] < base["fp"]
    ]

    def key(t, columns):
        return tuple(t[2][k] for k in columns) + (t[0],)

    if feasible:
        row = min(
            feasible,
            key=lambda t: key(t, ("fp", "retractions", "fn", "inference_total_bytes")),
        )
        tier = "half_fn_and_strict_fp"
    elif fallback:
        row = min(
            fallback,
            key=lambda t: key(t, ("fn", "fp", "retractions", "inference_total_bytes")),
        )
        tier = "both_counts_strict"
    else:
        row = scored[0]
        tier = "original"
    return base, scored, feasible, fallback, row, tier


def fit():
    data = cornell("label_pool")
    calibration_bytes = 0
    for i in range(len(data.e)):
        rec = CALIBRATION.pack(b"CAL1", data.numid[i], data.e[i], data.y[i]) + data.j[i]
        if (
            CALIBRATION.unpack_from(rec)
            != (b"CAL1", data.numid[i], data.e[i], data.y[i])
            or rec[CALIBRATION.size :] != data.j[i]
        ):
            raise ValueError("Calibration request changed bytes")
        calibration_bytes += len(rec)
    sizes = np.fromiter((len(j) for j in data.j), dtype=np.int64)
    base, scored, feasible, fallback, row, tier = select_fixed_grid(
        data.y, data.e, data.d, sizes
    )
    idx, rule, stats = row
    blob = pack(rule)
    metadata = dict(
        protocol="protocol/DUAL_DIRECTIONAL.md",
        protocol_sha256=digest((ROOT / "protocol/DUAL_DIRECTIONAL.md").read_bytes()),
        rule=asdict(rule),
        binary_sha256=digest(blob),
        source_balanced_head_sha256=digest(HEAD_FILE.read_bytes()),
        source_model="PlantDoc source-balanced binary DINO, without Cornell category training",
        selection_tier=tier,
        grid_index=idx,
        grid_n=len(scored),
        n_primary=len(feasible),
        n_strict=len(fallback),
        edge_rescue_lower=RESCUE_EDGE,
        edge_retract_upper=RETRACT_EDGE,
        remote_rescue_min=RESCUE_DINO,
        remote_retract_max=RETRACT_DINO,
        fit_n=len(data.ids),
        fit_paths_sha256=digest(("\n".join(data.ids) + "\n").encode()),
        fit_healthy=int(sum(data.y == 0)),
        fit_diseased=int(sum(data.y == 1)),
        baseline=base,
        selected=stats,
        calibration_upload_bytes=calibration_bytes,
        calibration_jpeg_bytes=sum(map(len, data.j)),
        rule_file_bytes=len(blob),
        deployment_update_bytes=len(blob) if rule.rescue or rule.retract else 0,
    )
    # Refuse to overwrite a different sealed policy. Replays of the identical selection are allowed.
    if BIN_FILE.exists() and BIN_FILE.read_bytes() != blob:
        raise ValueError("Existing sealed rule differs")
    if JSON_FILE.exists() and json.loads(JSON_FILE.read_text()) != json.loads(
        json.dumps(metadata)
    ):
        raise ValueError("Existing sealed metadata differs")
    BIN_FILE.write_bytes(blob)
    JSON_FILE.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def replay():
    metadata = json.loads(JSON_FILE.read_text())
    blob = BIN_FILE.read_bytes()
    if (
        digest(blob) != metadata["binary_sha256"]
        or digest(HEAD_FILE.read_bytes()) != metadata["source_balanced_head_sha256"]
    ):
        raise ValueError("Sealed model mismatch")
    rule = unpack(blob)
    rows = []
    predictions = []
    for name, data in (("cornell", cornell("test")), ("kashmir", kashmir())):
        query = select_queries(data.e, rule)
        replies = np.full(len(query), np.nan, dtype=np.float32)
        tx = rx = 0
        for i in np.flatnonzero(query):
            frame = REQUEST.pack(b"IMG1", data.numid[i]) + data.j[i]
            reply = REPLY.pack(b"PRD1", data.numid[i], data.d[i])
            if (
                REQUEST.unpack_from(frame) != (b"IMG1", data.numid[i])
                or frame[REQUEST.size :] != data.j[i]
            ):
                raise ValueError("Corrupt image")
            _, idx, p = REPLY.unpack(reply)
            if idx != data.numid[i]:
                raise ValueError("Corrupt reply")
            replies[i] = p
            tx += len(frame)
            rx += len(reply)
        pred = decide(data.e, replies, rule)
        base = evaluate(data, Rule())
        out = evaluate(data, rule)
        if (
            tx != out["inference_upload_bytes"]
            or rx != out["inference_reply_bytes"]
            or not np.array_equal(pred, decide(data.e, data.d, rule))
        ):
            raise ValueError("Remote traffic replay mismatch")
        out.update(
            cohort=name,
            n=len(data.y),
            orig_fp=base["fp"],
            orig_fn=base["fn"],
            delta_fp=out["fp"] - base["fp"],
            delta_fn=out["fn"] - base["fn"],
            strict_dual=int(out["fp"] < base["fp"] and out["fn"] < base["fn"]),
            healthy=int(sum(data.y == 0)),
            diseased=int(sum(data.y == 1)),
            calibration_nonrecurring_bytes=metadata["calibration_upload_bytes"],
            update_nonrecurring_bytes=metadata["deployment_update_bytes"],
        )
        rows.append(out)
        predictions.extend(
            dict(
                cohort=name,
                path=p,
                reference=int(data.y[i]),
                edge_p=float(data.e[i]),
                queried=int(query[i]),
                reply_p=float(replies[i]) if query[i] else "",
                original=int(data.e[i] >= EDGE_CUTOFF),
                decision=int(pred[i]),
                payload_jpeg_bytes=len(data.j[i]) if query[i] else 0,
            )
            for i, p in enumerate(data.ids)
        )
        print(name, out)
    write_table(RESULTS / "directional_outcomes.csv", rows)
    write_table(RESULTS / "directional_predictions.csv", predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["fit", "replay"])
    fit() if parser.parse_args().stage == "fit" else replay()

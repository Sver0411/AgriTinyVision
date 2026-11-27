"""Isolate the effect of early local supervision on a fixed future-date test pool."""

import argparse

import numpy as np

from edge_adaptation import fit
from edge_adaptation_wire import (
    normalized,
    pack_head,
    predict_from_head,
    quantized,
    training_record,
    training_input,
)
from evaluate_upload import rank_selection
from latvia_failure import (
    ROOT,
    RESULTS,
    SETUP,
    MARGINS,
    read,
    save,
    describe,
    group_metrics,
    random_masks,
)

CACHE = RESULTS / "latvia_failure_local_scores.npz"


def chronological_draws(paths, splits):
    ids = {p: i for i, p in enumerate(paths)}
    if len(ids) != len(paths) or set(ids) != set(splits):
        raise ValueError("Feature identifiers differ from date split")
    early = sorted(
        p
        for p, r in splits.items()
        if r["status"] == "calibration" and p == r["component_representative"]
    )
    eligible = np.array([ids[p] for p in early])
    draws = np.array(
        [
            eligible[
                np.random.default_rng(20261127 + s).permutation(len(eligible))[:128]
            ]
            for s in range(20)
        ]
    )
    later = np.array([i for i, p in enumerate(paths) if splits[p]["status"] == "test"])
    early_groups = {
        splits[paths[i]]["component_representative"] for i in draws.flatten()
    }
    late_groups = {splits[paths[i]]["component_representative"] for i in later}
    if (len(early), len(later)) != (590, 579) or early_groups & late_groups:
        raise ValueError("Chronological split or image-component disjointness changed")
    if max(splits[p]["capture_date"] for p in early) >= min(
        splits[paths[i]]["capture_date"] for i in later
    ):
        raise ValueError("Dates crossed")
    return draws, later


def refit():
    with np.load(ROOT / "data/features/latvia.npz") as z:
        paths, x, y = z["source_path"], z["edge"], z["y"]
    with np.load(RESULTS / "latvia_decisions.npz") as z:
        if not np.array_equal(paths, z["source_path"]) or not np.array_equal(y, z["y"]):
            raise ValueError("Portable predictions differ from original features")
        probability, expert, size = (
            z["edge_probability"],
            z["remote_probability"],
            z["jpeg_bytes"],
        )
    splits = {r["source_path"]: r for r in read("latvia_local_date_split.csv")}
    draws, test = chronological_draws(paths, splits)
    with np.load(MARGINS) as z:
        assert np.array_equal(z["latvia_paths"], paths)
        transferred = z["latvia_scores"][:, test]
    qtest, _ = quantized(x[test])
    local, ids = [], []
    for seed, draw in enumerate(draws):
        records = [training_record("local_int8", int(i), int(y[i]), x[i]) for i in draw]
        features = normalized(
            [
                training_input("local_int8", packet, int(i), int(y[i]), x.shape[1])
                for i, packet in zip(draw, records, strict=True)
            ]
        )
        weights, bias = fit(features, y[draw])
        blob = pack_head("local_int8", weights, bias)
        assert len(blob) + sum(len(packet) for packet in records) == SETUP
        local.append(predict_from_head("local_int8", blob, qtest))
        ids.extend(
            dict(seed=seed, draw_position=j, source_path=paths[i])
            for j, i in enumerate(draw)
        )
    save("latvia_failure_train_ids.csv", ids)
    np.savez_compressed(
        CACHE,
        source_path=paths[test],
        y=y[test],
        edge_probability=probability[test],
        expert_probability=expert[test],
        application_bytes=size[test] + 20,
        transfer_margin=transferred,
        local_margin=np.array(local),
        training_paths=paths[draws],
    )


def analyze():
    groups = []
    with np.load(CACHE, allow_pickle=False) as z:
        paths, y, ep, expert, costs = (
            z[k]
            for k in (
                "source_path",
                "y",
                "edge_probability",
                "expert_probability",
                "application_bytes",
            )
        )
        cap = int(costs.sum()) // 4
        remaining = cap - SETUP
        random = random_masks(paths, costs, remaining)
        for origin, key in [
            ("cornell", "transfer_margin"),
            ("latvia_early", "local_margin"),
        ]:
            for seed, score in enumerate(z[key]):
                q = rank_selection(np.lexsort((paths, abs(score))), costs, remaining)
                perf = group_metrics(
                    np.ones(len(paths), bool),
                    y,
                    score,
                    ep >= np.float32(0.3),
                    expert,
                    q,
                    random,
                )
                groups.append(
                    dict(
                        origin=origin,
                        seed=seed,
                        cap_bytes=cap,
                        setup_bytes=SETUP,
                        total_bytes=SETUP + int(costs[q].sum()),
                        **perf,
                    )
                )
    save("latvia_failure_local_runs.csv", groups)
    save("latvia_failure_local_summary.csv", describe(groups, ("origin",)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refit",
        action="store_true",
        help="Regenerate the local-site heads and prediction cache using the original feature vectors.",
    )
    args = parser.parse_args()
    if args.refit:
        refit()
    analyze()

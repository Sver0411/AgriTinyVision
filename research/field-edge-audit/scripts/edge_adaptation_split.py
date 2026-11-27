"""Freeze a path-hashed, perceptual-component FGVC7 holdout before fitting any heads."""

import csv
import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def table(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_table(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def hashed(purpose, path):
    return hashlib.sha256(("FGVC7/" + purpose + "/v1|" + path).encode()).hexdigest()


def split():
    rows = table(RESULTS / "fgvc_edge.csv")
    paths = [r["source_path"] for r in rows]
    if len(rows) != 1821 or len(set(paths)) != len(rows):
        raise ValueError("Expected 1821 unique Cornell paths")
    cross = {r["orchard_path"] for r in table(RESULTS / "orchard_overlap.csv")}
    if not cross.issubset(set(paths)):
        raise ValueError("Unexpected orchard overlap path")
    phash = [int(r["phash"], 16) for r in rows]
    roots = list(range(len(rows)))

    def root(i):
        while roots[i] != i:
            roots[i] = roots[roots[i]]
            i = roots[i]
        return i

    for i, h in enumerate(phash):
        for j in range(i):
            if (h ^ phash[j]).bit_count() <= 4 or rows[i]["file_sha256"] == rows[j][
                "file_sha256"
            ]:
                ri, rj = root(i), root(j)
                roots[max(ri, rj)] = min(ri, rj)
    groups = {}
    for i in range(len(rows)):
        groups.setdefault(root(i), []).append(i)
    ids = []
    attrs = [None] * len(rows)
    for group in groups.values():
        labels = {int(rows[i]["true_diseased"]) for i in group}
        reason = ""
        if any(paths[i] in cross for i in group):
            reason = "cross_corpus"
        elif len(labels) != 1:
            reason = "conflicting_labels"
        rep = min(group, key=lambda i: (hashed("representative", paths[i]), paths[i]))
        if not reason:
            ids.append(rep)
        for i in group:
            attrs[i] = dict(
                source_path=paths[i],
                y=rows[i]["true_diseased"],
                source_label=rows[i]["source_label"],
                phash=rows[i]["phash"],
                group_representative=paths[rep],
                group_size=len(group),
                status=reason or ("eligible" if i == rep else "same_component"),
                split_hash=hashed("split", paths[rep]),
            )
    ids.sort(key=lambda i: (hashed("split", paths[i]), paths[i]))
    cutoff = len(ids) // 4
    pools = np.asarray(ids[:cutoff]), np.asarray(ids[cutoff:])
    for poolname, pool in zip(("label_pool", "test"), pools):
        for i in pool:
            attrs[i]["status"] = poolname
    return rows, attrs, pools[0], pools[1]


if __name__ == "__main__":
    rows, split_rows, label, test = split()
    write_table(RESULTS / "edge_adaptation_split.csv", split_rows)
    from collections import Counter

    print(
        Counter(r["status"] for r in split_rows),
        "candidates / test",
        len(label),
        len(test),
    )

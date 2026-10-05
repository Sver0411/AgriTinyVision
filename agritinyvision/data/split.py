from sklearn.model_selection import GroupShuffleSplit

from .. import CLASSES
from ..utils import object_hash, write_json


def split_groups(rows, report, seed=41, ratios=(0.7, 0.15, 0.15)):
    if report["invalid_count"]:
        raise ValueError("Audit failed; repair invalid images/labels before splitting")
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(j)] = find(i)

    seen = {}
    for i, r in enumerate(rows):
        if r["group_id"] in seen:
            union(i, seen[r["group_id"]])
        seen[r["group_id"]] = i
    for cluster in report["duplicate_hash_groups"] + report["near_duplicate_pairs"]:
        for j in cluster[1:]:
            union(cluster[0], j)
    groups = [find(i) for i in range(len(rows))]
    trainval, test = next(
        GroupShuffleSplit(n_splits=1, test_size=ratios[2], random_state=seed).split(rows, groups=groups)
    )
    tr, va = next(
        GroupShuffleSplit(
            n_splits=1, test_size=ratios[1] / (ratios[0] + ratios[1]), random_state=seed + 1
        ).split(trainval, groups=[groups[i] for i in trainval])
    )
    partitions = {"train": trainval[tr], "val": trainval[va], "test": test}
    for name, indices in partitions.items():
        if {rows[int(i)]["label"] for i in indices} != set(CLASSES):
            raise ValueError(f"Group split {name} lacks a binary class; need more independent groups")
        for i in indices:
            rows[int(i)]["split"] = name
            rows[int(i)]["leakage_group"] = str(groups[int(i)])
    return rows


def prepare(manifest, root, output, seed=41):
    from ..utils import read_json
    from .audit import audit

    m = read_json(manifest)
    report, rows = audit(m, root, "reports/dataset_audit_before_split")
    m["samples"] = split_groups(rows, report, seed)
    m["split_method"] = (
        "leaf/source groups + exact SHA256 + connected pHash <=4 components; GroupShuffleSplit"
    )
    m["seed"] = seed
    m["dataset_hash"] = object_hash(sorted((r["path"], r["sha256"], r["label"], r["group_id"]) for r in rows))
    m["train_split_hash"] = object_hash(sorted(r["path"] for r in rows if r["split"] == "train"))
    write_json(output, m)
    final, _ = audit(output, root)
    if not final["passed"]:
        raise ValueError("Cross-split leakage detected after splitting")
    return final

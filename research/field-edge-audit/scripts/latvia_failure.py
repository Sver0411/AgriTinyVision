"""Diagnose the locked 25%-budget router using image-paired actions and exact upper bounds."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from evaluate_upload import rank_selection
from joint_budget import dataset

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SETUP = 166798
SITES = ("cornell", "kashmir", "latvia")
MARGINS = RESULTS / "latvia_failure_margins.npz"


def load_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read(name):
    return load_csv(RESULTS / name)


def save(name, rows):
    with (RESULTS / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate_margins():
    arrays = {}
    for site in SITES:
        paths, _, _, _, _, scores = dataset(site)
        arrays[site + "_paths"] = paths
        arrays[site + "_scores"] = scores
    np.savez_compressed(MARGINS, **arrays)


def oracle_selection(y, edge, expert, costs, remaining):
    """Maximize corrected true-negative/n0 + corrected true-positive/n1 within the cap."""
    n0, n1 = np.bincount(y, minlength=2)
    if min(n0, n1) == 0:
        raise ValueError("Balanced accuracy needs both classes")
    fixable = (edge != y) & (expert == y)
    order, prefix = [], []
    for label in (0, 1):
        idx = np.flatnonzero(fixable & (y == label))
        idx = idx[np.argsort(costs[idx], kind="stable")]
        order.append(idx)
        prefix.append(np.r_[0, np.cumsum(costs[idx], dtype=np.int64)])
    best = (0, 0, 0, 0)
    for a, cost0 in enumerate(prefix[0]):
        if cost0 > remaining:
            break
        b = int(np.searchsorted(prefix[1], remaining - cost0, side="right") - 1)
        consumed = int(cost0 + prefix[1][b])
        candidate = (int(a * n1 + b * n0), -consumed, a, b)
        best = max(best, candidate)
    selected = np.zeros(len(y), dtype=bool)
    _, _, a, b = best
    selected[order[0][:a]] = True
    selected[order[1][:b]] = True
    return selected


def auc(y, value):
    return float(roc_auc_score(y, value)) if np.unique(y).size == 2 else np.nan


def average_if_defined(x):
    return float(np.mean(x)) if np.size(x) else np.nan


def ba(y, preds):
    pred = np.asarray(preds)
    if y.min() == y.max():
        return np.full(pred.shape[:-1], np.nan)
    return (pred[..., y == 1].mean(axis=-1) + 1 - pred[..., y == 0].mean(axis=-1)) / 2


def group_metrics(group, y, score, ep, exp_score, q, random_q):
    y = y[group]
    edge, expert = score[group] >= 0, exp_score[group] >= 0.5
    original = ep[group]
    q, random_q = q[group], random_q[:, group]
    joint = np.where(q, expert, edge)
    random_predictions = np.where(random_q, expert, edge)
    error = edge != y
    correction = error & (expert == y)
    regression = ~error & (expert != y)
    disagree = edge != expert
    rank = -np.abs(score[group])
    class_weights = np.array([1 / max(1, np.count_nonzero(y == k)) for k in (0, 1)])[y]
    weighted_error_auc = (
        roc_auc_score(error, rank, sample_weight=class_weights)
        if np.unique(error).size == 2
        else np.nan
    )
    report = dict(
        n=len(y),
        n_pos=int(y.sum()),
        n_neg=int(len(y) - y.sum()),
        sent=int(q.sum()),
        updated_score_auc=auc(y, score[group]),
        expert_score_auc=auc(y, exp_score[group]),
        uncertainty_error_auc=auc(error, -np.abs(score[group])),
        uncertainty_repair_auc=auc(correction, rank),
        uncertainty_error_auc_equal_class_weight=weighted_error_auc,
        uncertainty_error_auc_on_positive=auc(error[y == 1], rank[y == 1]),
        uncertainty_error_auc_on_negative=auc(error[y == 0], rank[y == 0]),
        uncertainty_value_auc_on_disagreements=auc(
            correction[disagree], rank[disagree]
        ),
        original_ba=ba(y, original),
        updated_ba=ba(y, edge),
        expert_ba=ba(y, expert),
        joint_ba=ba(y, joint),
        random_ba_mean=ba(y, random_predictions).mean(),
        joint_accuracy=float((joint == y).mean()),
        updated_accuracy=float((edge == y).mean()),
        original_accuracy=float((original == y).mean()),
        queried_edge_accuracy=average_if_defined((edge == y)[q]),
        queried_expert_accuracy=average_if_defined((expert == y)[q]),
    )
    for label, err in [(0, "fp"), (1, "fn")]:
        target = y == label
        report.update(
            {
                err + "_original": int(((original != y) & target).sum()),
                err + "_updated": int((error & target).sum()),
                err + "_expert": int(((expert != y) & target).sum()),
                err + "_joint": int(((joint != y) & target).sum()),
                err + "_random_mean": float(
                    np.count_nonzero((random_predictions != y) & target, axis=1).mean()
                ),
                err + "_correctable": int((correction & target).sum()),
                err + "_spoilable": int((regression & target).sum()),
                err + "_corrected_on_upload": int((q & correction & target).sum()),
                err + "_introduced_on_upload": int((q & regression & target).sum()),
            }
        )
        assert (
            report[err + "_updated"]
            - report[err + "_corrected_on_upload"]
            + report[err + "_introduced_on_upload"]
            == report[err + "_joint"]
        )
    return report


def group_definitions(site, paths):
    groups = [("all", "all", np.ones(len(paths), bool))]
    if site == "cornell":
        label = {r["source_path"]: r["source_label"] for r in read("fgvc_edge.csv")}
        groups.append(
            (
                "reference",
                "healthy_or_scab",
                np.array([label[p] in ("healthy", "scab") for p in paths]),
            )
        )
    if site == "latvia":
        manifest = {
            r["source_path"]: r
            for r in load_csv(ROOT / "data/latvia_source/manifest.csv")
        }
        split = {r["source_path"]: r for r in read("latvia_local_date_split.csv")}
        dates = np.array([split[p]["capture_date"] for p in paths])
        models = np.array(
            [manifest[p]["exif_make"] + " " + manifest[p]["exif_model"] for p in paths]
        )
        groups.append(
            (
                "deduplication",
                "representatives",
                np.array([p == split[p]["component_representative"] for p in paths]),
            )
        )
        for name, values in [("date", dates), ("device", models)]:
            for key in np.unique(values):
                groups.append((name, str(key), values == key))
                groups.append(("omit_" + name, str(key), values != key))
    return groups


def random_masks(paths, costs, remaining):
    canonical = np.argsort(paths)
    return np.array(
        [
            rank_selection(
                np.random.default_rng(21000 + i).permutation(canonical),
                costs,
                remaining,
            )
            for i in range(20)
        ]
    )


def describe(rows, groups):
    selected = defaultdict(list)
    for row in rows:
        selected[tuple(row[k] for k in groups)].append(row)
    output = []
    for keys, sub in selected.items():
        out = dict(zip(groups, keys, strict=True))
        for k in sub[0]:
            if k in groups or k == "seed":
                continue
            a = np.array([r[k] for r in sub], dtype=float)
            out[k] = float(np.nanmean(a)) if np.isfinite(a).any() else ""
        output.append(out)
    return output


def stratified_discrimination(paths, y, scores, remote):
    """Condition disease-score AUC on each acquisition stratum by counting comparable pairs."""
    manifest = {
        r["source_path"]: r for r in load_csv(ROOT / "data/latvia_source/manifest.csv")
    }
    days = np.array([manifest[p]["exif_datetime_original"][:10] for p in paths])
    devices = np.array(
        [manifest[p]["exif_make"] + " " + manifest[p]["exif_model"] for p in paths]
    )
    crossings = np.array(
        [d + "|" + model for d, model in zip(days, devices, strict=True)]
    )
    all_pairs = int(np.count_nonzero(y == 0)) * int(np.count_nonzero(y == 1))
    rows = []
    for name, group in [
        ("pooled", np.zeros(len(paths))),
        ("date", days),
        ("device", devices),
        ("date_and_device", crossings),
    ]:
        strata = []
        for value in np.unique(group):
            idx = group == value
            size = int(idx.sum())
            positives = int(y[idx].sum())
            pairs = positives * (size - positives)
            if pairs:
                strata.append((idx, pairs))
        weights = [w for _, w in strata]
        remote_auc = np.average(
            [auc(y[idx], remote[idx]) for idx, _ in strata], weights=weights
        )
        for seed, m in enumerate(scores):
            edge_auc = np.average(
                [auc(y[idx], m[idx]) for idx, _ in strata], weights=weights
            )
            rows.append(
                dict(
                    grouping=name,
                    seed=seed,
                    all_strata=len(np.unique(group)),
                    two_class_strata=len(strata),
                    images_in_two_class_strata=sum(int(idx.sum()) for idx, _ in strata),
                    retained_pairs=sum(weights),
                    pair_fraction=sum(weights) / all_pairs,
                    updated_score_auc=edge_auc,
                    expert_score_auc=remote_auc,
                )
            )
    save("latvia_failure_stratified_auc.csv", describe(rows, ("grouping",)))


def audit():
    records = read("joint_budget_runs.csv")
    cloud = {
        (r["cohort"], r["source_path"]): float(r["dino_probability"])
        for r in read("matched_collaboration_scores.csv")
    }
    group_rows, controls = [], []
    with np.load(MARGINS, allow_pickle=False) as margins:
        for site in SITES:
            with np.load(RESULTS / f"joint_budget_{site}.npz") as z:
                paths, y, ep, expert = (
                    z[k] for k in ("source_path", "y", "original", "expert")
                )
                costs, predictions, queries = (
                    z[k] for k in ("application_bytes", "predictions", "queries")
                )
            score = margins[site + "_scores"]
            assert np.array_equal(paths, margins[site + "_paths"])
            if site == "latvia":
                with np.load(RESULTS / "latvia_decisions.npz") as z:
                    assert np.array_equal(paths, z["source_path"])
                    remote_score = z["remote_probability"]
            else:
                remote_score = np.array([cloud[site, p] for p in paths])
            assert np.array_equal(expert, remote_score >= 0.5)
            if site == "latvia":
                stratified_discrimination(paths, y, score, remote_score)
            rows = [r for r in records if r["site"] == site]
            lookup = {
                (int(r["seed"]), r["method"]): i
                for i, r in enumerate(rows)
                if int(r["cap_percent"]) == 25 and r["review_location"] == "local"
            }
            cap = int(costs.sum()) // 4
            available = cap - SETUP
            random_q = random_masks(paths, costs, available)
            site_groups = group_definitions(site, paths)
            for s, m in enumerate(score):
                idx = lookup[s, "joint"]
                assert (
                    int(rows[idx]["setup_bytes"]) == SETUP
                    and int(rows[idx]["cap_bytes"]) == cap
                )
                edge, q = m >= 0, queries[idx].astype(bool)
                assert np.array_equal(edge, predictions[lookup[s, "update_only"]])
                assert np.array_equal(
                    q, rank_selection(np.lexsort((paths, abs(m))), costs, available)
                )
                observed = np.where(q, expert, edge)
                assert np.array_equal(observed, predictions[idx])
                for category, key, ix in site_groups:
                    group_rows.append(
                        dict(
                            site=site,
                            group_type=category,
                            group=key,
                            seed=s,
                            **group_metrics(ix, y, m, ep, remote_score, q, random_q),
                        )
                    )
                oracle = oracle_selection(y, edge, expert, costs, available)
                assert int(costs[oracle].sum()) <= available
                policies = [
                    ("actual", observed, q),
                    ("perfect_expert_on_actual_queries", np.where(q, y, edge), q),
                    (
                        "optimal_router_fixed_expert",
                        np.where(oracle, expert, edge),
                        oracle,
                    ),
                    (
                        "unlimited_oracle_existing_experts",
                        np.where(edge == y, edge, expert),
                        (edge != y) & (expert == y),
                    ),
                ]
                for method, pred, actions in policies:
                    controls.append(
                        dict(
                            site=site,
                            seed=s,
                            method=method,
                            n=len(y),
                            setup_bytes=SETUP,
                            cap_bytes=cap,
                            selected=int(actions.sum()),
                            total_bytes=SETUP + int(costs[actions].sum()),
                            fp=int(np.count_nonzero(pred[y == 0])),
                            fn=int(np.count_nonzero(~pred[y == 1].astype(bool))),
                            balanced_accuracy=ba(y, pred),
                            accuracy=float((y == pred).mean()),
                        )
                    )
    save("latvia_failure_groups_runs.csv", group_rows)
    grouped = describe(group_rows, ("site", "group_type", "group"))
    save("latvia_failure_groups.csv", grouped)
    standardizations = []
    for r in grouped:
        if r["group_type"] not in ("all", "reference"):
            continue
        for prevalence in (0.5, 976 / 1353, 983 / 1190):
            outcomes = {}
            for method in ("original", "updated", "joint", "random_mean"):
                sensitivity = 1 - r["fn_" + method] / r["n_pos"]
                specificity = 1 - r["fp_" + method] / r["n_neg"]
                outcomes[method + "_accuracy"] = (
                    prevalence * sensitivity + (1 - prevalence) * specificity
                )
            standardizations.append(
                dict(
                    site=r["site"],
                    group=r["group"],
                    specified_positive_prevalence=prevalence,
                    observed_positive_prevalence=r["n_pos"] / r["n"],
                    **outcomes,
                )
            )
    save("latvia_failure_prevalence.csv", standardizations)
    save("latvia_failure_oracle_runs.csv", controls)
    save("latvia_failure_oracle.csv", describe(controls, ("site", "method")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild-margins",
        action="store_true",
        help="Read the local latent vectors and verify the archived margins.",
    )
    args = parser.parse_args()
    if args.rebuild_margins:
        generate_margins()
    audit()

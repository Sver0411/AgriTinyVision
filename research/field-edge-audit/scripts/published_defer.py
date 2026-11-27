"""Fit and budget-replay published learned deferral rules on the matched experiment."""

import argparse
import csv
import hashlib
import zipfile
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from edge_adaptation_wire import pack_head, predict_from_head, normalized, quantized
from evaluate_upload import rank_selection

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FEATURES = ROOT / "data/features"
SITES = ("cornell", "kashmir", "latvia")
BASELINE = "confidence_reference"
LOSSES = ("ova", "compare_confidence", "mao_two_stage")
VARIANTS = ("ordinary", "balanced")
METHODS = (BASELINE,) + tuple(f"{loss}_{variant}" for loss in LOSSES for variant in VARIANTS)
BUDGETS = (10, 25, 50)
SEEDS = range(20)


def table(name):
    with (RESULTS / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def train_weights(label, variant):
    if variant == "ordinary":
        return np.ones(len(label))
    frequencies = np.bincount(label, minlength=2)
    if not np.all(frequencies):
        return np.ones(len(label))
    return len(label) / (2 * frequencies[label])


def binary_head(x, target, weight):
    keep = weight > 0
    if not np.any(keep):
        return pack_head("local_int8", np.zeros(x.shape[1]), 0.)
    if len(np.unique(target[keep])) == 1:
        return pack_head("local_int8", np.zeros(x.shape[1]), 1. if target[keep][0] else -1.)
    model = LogisticRegression(C=1, solver="liblinear", max_iter=3000, tol=1e-8, random_state=0)
    model.fit(x[keep], target[keep], sample_weight=weight[keep])
    return pack_head("local_int8", model.coef_[0], model.intercept_[0])


def mao_examples(x, label, oof_decision, expert, weight):
    local_cost = (oof_decision != label).astype(float)
    remote_cost = (expert != label).astype(float)
    return (np.concatenate((x, x)), np.repeat([1, 0], len(x)),
            np.concatenate((weight * local_cost, weight * remote_cost)))


def score_actions(name, class_logit, router_logit):
    """Return local class and a strictly local priority score (larger is more urgent)."""
    local = class_logit >= 0
    if name == BASELINE:
        priority = -np.abs(class_logit)
    elif name.startswith("ova_"):
        priority = router_logit - np.abs(class_logit)
    elif name.startswith("compare_confidence_"):
        priority = expit(router_logit) - expit(np.abs(class_logit))
    elif name.startswith("mao_two_stage_"):
        priority = router_logit
    else:
        raise ValueError(name)
    return local, priority


def fit_draw(seed, x, q, label, expert, original_head, feedback_count=128):
    heads = {}
    for variant in VARIANTS:
        w = train_weights(label, variant)
        class_head = binary_head(x, label, w)
        if variant == "balanced":
            if class_head != original_head:
                raise ValueError(f"Refitted seed {seed} differs from archived INT8 head")
            heads["reference"] = original_head
        heads["class_" + variant] = class_head
        feedback = slice(0, feedback_count)
        feedback_weight = train_weights(label[feedback], variant)
        expert_correct = (expert[feedback] == label[feedback]).astype(int)
        heads["correctness_" + variant] = binary_head(x[feedback], expert_correct, feedback_weight)
        oof = np.empty(len(label), dtype=bool)
        splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=20261127 + seed)
        for train, val in splitter.split(x, label):
            fold_head = binary_head(x[train], label[train], train_weights(label[train], variant))
            oof[val] = predict_from_head("local_int8", fold_head, q[val]) >= 0
        cost_examples = mao_examples(x[feedback], label[feedback], oof[feedback],
                                     expert[feedback], feedback_weight)
        heads["two_stage_" + variant] = binary_head(*cost_examples)
    return heads


def aligned(archive_paths, desired):
    if len(set(archive_paths)) != len(archive_paths):
        raise ValueError("Duplicate feature identities")
    mapping = {p:i for i,p in enumerate(archive_paths)}
    return np.array([mapping[p] for p in desired], dtype=int)


def setup_rows():
    return {int(r["seed"]):r for r in table("edge_adaptation_metrics.csv")
            if r["method"] == "local_int8" and r["budget"] == "128"}


def fit(output, feedback_count):
    with np.load(RESULTS / "edge_adaptation_predictions.npz", allow_pickle=False) as data:
        train_paths = data["train_paths"].copy()
        pool = set(data["label_pool_paths"])
        if any(not set(draw).issubset(pool) or len(set(draw)) != 128 for draw in train_paths):
            raise ValueError("Training IDs must be the 128 labeled calibration cases")
        if not pool.isdisjoint(data["test_paths"]):
            raise ValueError("Holdout leakage")
    with np.load(FEATURES / "edge_fgvc.npz", allow_pickle=False) as data:
        features = data["activation"].copy()
        edge_paths = data["source_path"].copy()
    source = {r["source_path"]:r for r in table("specialist_balanced.csv") if r["feature_dataset"] == "fgvc"}
    cost = setup_rows()
    jpeg_sizes = {r["source_path"]:int(r["jpeg_bytes"]) for r in table("fgvc_backend.csv")
                  if r["long_edge"] == "224" and r["jpeg_quality"] == "75"}
    trained = []
    metadata = []
    with zipfile.ZipFile(RESULTS / "edge_adaptation_heads.zip") as reference, zipfile.ZipFile(
            output / "published_defer_heads.zip", "w", compression=zipfile.ZIP_STORED) as archive:
        for seed in SEEDS:
            ids = train_paths[seed]
            rows = [source[p] for p in ids]
            label = np.array([int(r["y"]) for r in rows])
            expert = np.array([int(r["pred_diseased"]) for r in rows])
            q, scales = quantized(features[aligned(edge_paths, ids)])
            x = normalized(q.astype(np.float64) * scales[:, None])
            original = reference.read(f"s{seed:02d}-b128/local_int8.bin")
            heads = fit_draw(seed, x, q, label, expert, original, feedback_count)
            classifier_bytes = len(original)
            if any(len(h) != classifier_bytes for h in heads.values()):
                raise ValueError("Unexpected packaged head size")
            for tag, blob in heads.items():
                archive.writestr(f"s{seed:02d}/{tag}.bin", blob)
            trained.append(heads)
            row = cost[seed]
            training_features = int(row["calibration_upload_bytes"])
            training_images = int(row["remote_review_additional_record_bytes"])
            if training_features + training_images + classifier_bytes != int(row["remote_label_total_bytes"]):
                raise ValueError("Training traffic mismatch")
            calibration_jpegs = sum(jpeg_sizes[p] + 8 for p in ids[:feedback_count])
            if sum(jpeg_sizes[p] + 8 for p in ids) != training_images:
                raise ValueError("Calibration-image transport disagrees with the original protocol")
            metadata.append(dict(seed=seed, n_feedback=feedback_count, labelling_pool_sha256=hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                                 n_labels=len(ids), target_positives=int(label.sum()), cloud_correct=int((expert==label).sum()),
                                 activation_upload_bytes=training_features, image_upload_bytes=training_images,
                                 single_head_bytes=classifier_bytes, learned_install_bytes=2*classifier_bytes,
                                 learned_setup_bytes=training_features+calibration_jpegs+2*classifier_bytes,
                                 expert_feedback_image_bytes=calibration_jpegs,
                                 learned_remote_setup_bytes=training_features+training_images+2*classifier_bytes,
                                 confidence_local_setup=int(row["local_label_total_bytes"]),
                                 confidence_remote_setup=int(row["remote_label_total_bytes"])))
    write_csv(output / "published_defer_training.csv", metadata)
    return trained


def site_data(site):
    with np.load(RESULTS / f"joint_budget_{site}.npz", allow_pickle=False) as ref:
        paths, y, original, expert, costs = (ref[k].copy() for k in ("source_path","y","original","expert","application_bytes"))
    if site == "cornell":
        filename, key = "edge_fgvc.npz", "activation"
    else:
        filename, key = f"{site}.npz", "edge"
    with np.load(FEATURES / filename, allow_pickle=False) as archive:
        q, _ = quantized(archive[key][aligned(archive["source_path"], paths)])
    if site == "latvia":
        with np.load(RESULTS / "latvia_decisions.npz", allow_pickle=False) as z:
            edge_p = z["edge_probability"][aligned(z["source_path"], paths)]
    else:
        by_path = {r["path"]:np.float32(r["edge_p"]) for r in table("directional_predictions.csv") if r["cohort"] == site}
        edge_p = np.array([by_path[p] for p in paths])
    if not np.array_equal(edge_p >= np.float32(.3), original):
        raise ValueError("Unmodified edge and fallback probabilities disagree")
    return paths, y, original, expert, costs, q, edge_p


def opaque_tiebreak(site, paths):
    # In Kashmir and Latvia the released paths contain the diagnosis.
    # Hash photo identity first. Distinguish exact copies with an opaque secondary key.
    if site == "cornell":
        hashes = {r["source_path"]:r["file_sha256"] for r in table("fgvc_edge.csv")}
    elif site == "kashmir":
        hashes = {r["source_path"]:r["source_sha256"] for r in table("transfer_manifest.csv")}
    else:
        with np.load(RESULTS / "latvia_decisions.npz", allow_pickle=False) as z:
            hashes = dict(zip(z["source_path"],z["source_sha256"],strict=True))
    salt = b"field-edge-opaque-tie-v1/"
    return np.array([hashlib.sha256(salt + bytes.fromhex(str(hashes[p]))).hexdigest()
                     + hashlib.sha256(salt + str(p).encode()).hexdigest() for p in paths])


def make_scores(site, trained, output):
    paths, y, original, expert, costs, q, edge_p = site_data(site)
    local = np.empty((len(SEEDS), len(METHODS), len(y)), dtype=np.uint8)
    scores = np.empty(local.shape, dtype=np.float64)
    for seed, heads in enumerate(trained):
        linear = {k:predict_from_head("local_int8", v, q) for k,v in heads.items()}
        local[seed,0], scores[seed,0] = score_actions(BASELINE, linear["reference"], None)
        for i,name in enumerate(METHODS[1:], start=1):
            kind, variant = name.rsplit("_",1)
            router = "two_stage_" if kind == "mao_two_stage" else "correctness_"
            local[seed,i], scores[seed,i] = score_actions(name, linear["class_"+variant], linear[router+variant])
    np.savez_compressed(output / f"published_defer_scores_{site}.npz", source_path=paths, y=y,
                        original=original, expert=expert, costs=costs, original_priority=-abs(edge_p-np.float32(.3)),
                        methods=np.array(METHODS), local=local, priority=scores, tie_key=opaque_tiebreak(site, paths))


def count_outcome(y, p):
    fp = int(np.count_nonzero((y == 0) & p))
    fn = int(np.count_nonzero((y == 1) & ~p))
    return dict(fp=fp, fn=fn, accuracy=float(np.mean(y==p)),
                balanced_accuracy=float(1 - .5 * (fp/(y==0).sum() + fn/(y==1).sum())))


def select_model(name, learned_cost, confidence_cost, budget, prepaid):
    desired = confidence_cost if name == BASELINE else learned_cost
    if prepaid or desired <= budget:
        return name, 0 if prepaid else desired, desired
    if confidence_cost <= budget:
        return BASELINE, confidence_cost, desired
    return "published_edge", 0, desired


def replay_site(site, training, source, output):
    with np.load(source / f"published_defer_scores_{site}.npz", allow_pickle=False) as z:
        ids, y, old, expert, costs = (z[k] for k in ("source_path","y","original","expert","costs"))
        if tuple(z["methods"]) != METHODS:
            raise ValueError("Unrecognized priority score axes")
        local, score, old_score = z["local"].astype(bool), z["priority"], z["original_priority"]
        tie_key = z["tie_key"]
    reports, diagnostics = [], []
    predictions, queried = [], []
    ref_i = METHODS.index(BASELINE)
    original_order = np.lexsort((tie_key, -old_score))
    for seed in SEEDS:
        for mi, method in enumerate(METHODS):
            candidate_order = np.lexsort((tie_key, -score[seed,mi]))
            if method != BASELINE:
                default_query = score[seed,mi] > 0
                p = np.where(default_query, expert, local[seed,mi]).astype(bool)
                diagnostics.append(dict(site=site, seed=seed, method=method, n=len(y), queried=int(default_query.sum()),
                                        inference_bytes=int(costs[default_query].sum()), **count_outcome(y,p)))
            fallback_order = np.lexsort((tie_key, -score[seed,ref_i]))
            for place in ("local","remote"):
                reference_setup = int(training[seed][f"confidence_{place}_setup"])
                router_setup = int(training[seed]["learned_setup_bytes"])
                if place == "remote":
                    router_setup = int(training[seed].get("learned_remote_setup_bytes", router_setup))
                for prepaid in (False,True):
                    mode = "prepaid" if prepaid else "charged"
                    for pct in BUDGETS:
                        cap = int(costs.sum())*pct//100
                        active, charged, desired = select_model(method, router_setup, reference_setup, cap, prepaid)
                        if active == "published_edge":
                            order, local_pred = original_order, old
                        elif active == BASELINE:
                            order, local_pred = fallback_order, local[seed,ref_i]
                        else:
                            order, local_pred = candidate_order, local[seed,mi]
                        mask = rank_selection(order, costs, cap - charged)
                        p = np.where(mask,expert,local_pred).astype(bool)
                        used = int(costs[mask].sum())
                        if charged + used > cap:
                            raise ValueError("Byte ceiling exceeded")
                        row = dict(site=site, seed=seed, method=method, reviewer=place, accounting=mode,
                                   cap_percent=pct, cap_bytes=cap, active_method=active, desired_setup_bytes=desired,
                                   actual_setup_bytes=charged, setup_fits=int(desired<=cap), n=len(y), queried=int(mask.sum()),
                                   inference_bytes=used, total_bytes=charged+used, row_index=len(reports), **count_outcome(y,p))
                        reports.append(row)
                        predictions.append(p)
                        queried.append(mask)
    np.savez_compressed(output / f"published_defer_decisions_{site}.npz", source_path=ids, y=y,
                        predictions=np.packbits(predictions,axis=1), queries=np.packbits(queried,axis=1))
    return reports, diagnostics


def summarize(rows):
    reference = {(r["site"],r["seed"],r["reviewer"],r["accounting"],r["cap_percent"]):r for r in rows if r["method"]==BASELINE}
    keys = ("site","method","reviewer","accounting","cap_percent")
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
    summary = []
    for key, group in groups.items():
        delta = np.array([r["balanced_accuracy"] - reference[r["site"],r["seed"],r["reviewer"],r["accounting"],r["cap_percent"]]["balanced_accuracy"] for r in group])
        summary.append(dict(**dict(zip(keys,key)), n_seeds=len(group), installed=sum(r["active_method"]==r["method"] for r in group),
                            adapted_confidence_fallback=sum(r["active_method"]==BASELINE and r["method"]!=BASELINE for r in group),
                            frozen_fallback=sum(r["active_method"]=="published_edge" for r in group),
                            mean_ba=np.mean([r["balanced_accuracy"] for r in group]), mean_accuracy=np.mean([r["accuracy"] for r in group]),
                            mean_fp=np.mean([r["fp"] for r in group]), mean_fn=np.mean([r["fn"] for r in group]),
                            mean_queries=np.mean([r["queried"] for r in group]), mean_bytes=np.mean([r["total_bytes"] for r in group]),
                            mean_gain_over_confidence_pp=100*delta.mean(), wins_against_confidence=int((delta>1e-12).sum()),
                            ties_against_confidence=int((abs(delta)<=1e-12).sum()), losses_against_confidence=int((delta< -1e-12).sum())))
    return summary


def run(output, cached, feedback_count=128):
    output.mkdir(parents=True,exist_ok=True)
    if cached is None:
        trained = fit(output, feedback_count)
        for site in SITES:
            make_scores(site, trained, output)
        cached = output
    with (cached / "published_defer_training.csv").open(newline="") as f:
        training = {int(r["seed"]):r for r in csv.DictReader(f)}
    runs, diagnostics = [], []
    for site in SITES:
        rows, default = replay_site(site, training, cached, output)
        runs.extend(rows)
        diagnostics.extend(default)
    write_csv(output / "published_defer_runs.csv", runs)
    write_csv(output / "published_defer_default.csv", diagnostics)
    write_csv(output / "published_defer_summary.csv", summarize(runs))
    legacy = {(r["site"],int(r["seed"]),r["review_location"],int(r["cap_percent"])):r for r in table("joint_budget_runs.csv") if r["method"]=="joint"}
    checked = 0
    for row in runs:
        if row["method"] == BASELINE and row["accounting"] == "charged":
            ref = legacy[row["site"],row["seed"],row["reviewer"],row["cap_percent"]]
            assert (row["fp"],row["fn"],row["total_bytes"]) == (int(ref["fp"]),int(ref["fn"]),int(ref["total_bytes"]))
            checked += 1
    print(f"Verified all {checked} original reference runs and replayed {len(runs)} budgeted policy runs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RESULTS)
    parser.add_argument("--replay", type=Path, help="Rebuild from released scalar priority scores without loading image features.")
    parser.add_argument("--feedback-count", type=int, default=128, choices=(8, 32, 64, 128),
                        help="Expert calibration queries; the full 128 ground-truth labels still fit the local classifier.")
    args = parser.parse_args()
    if args.feedback_count != 128 and args.out == RESULTS and args.replay is None:
        args.out = RESULTS / f"published_defer_partial_{args.feedback_count}"
    run(args.out, args.replay, args.feedback_count)

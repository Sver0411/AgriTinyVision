"""Reimplement Diff-01 and Diff-Prob from Jitkrittum et al. (NeurIPS 2023)."""

import argparse
import csv
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import struct
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from scipy.special import expit, xlogy
from sklearn.model_selection import StratifiedKFold

from directional_policy import EDGE_CUTOFF, REQUEST, REPLY
from edge_adaptation import fit
from edge_adaptation_wire import normalized, pack_head, predict_from_head, quantized
from evaluate_upload import rank_selection
from joint_budget import dataset, table


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FEATURES = ROOT / "data/features"
PROTOCOL_HASH = "47bd7dfae399af461ef056f70ded0bf7558085bb05773757788000b365b3470a"
WIDTHS = (5, 64, 16, 1)
SHAPES = tuple(shape for ni, no in zip(WIDTHS, WIDTHS[1:]) for shape in ((ni, no), (no,)))
HEADER = struct.Struct("<4s4H")
REGULARIZER = .001
STAGES = ("cold", "predeployed")
RANKS = ("budget", "positive")
METHODS = tuple(f"{target}_{mode}" for target in ("diff01", "diffprob")
                for mode in ("paper20", "steps800", "balanced800"))


def csv_write(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def probability_features(local_p):
    local_p = np.asarray(local_p, dtype=np.float64)
    if not np.all((local_p >= 0) & (local_p <= 1)):
        raise ValueError("Expected probabilities in [0,1]")
    p = np.stack((1 - local_p, local_p), axis=1)
    entropy = -xlogy(p, p).sum(axis=1)
    top = np.sort(p, axis=1)[:, ::-1]
    cls = np.eye(2)[np.argmax(p, axis=1)]
    return np.column_stack((entropy, top, cls))


def targets(y, edge_prob, remote_prob):
    y = np.asarray(y)
    hard = ((remote_prob >= .5) == y).astype(float) - ((edge_prob >= .5) == y)
    soft = (2.0 * y - 1.0) * (remote_prob - edge_prob)
    return {"diff01": hard, "diffprob": soft}


def initialize(seed):
    rng = np.random.default_rng(91431 + int(seed))
    params = []
    for ni, no in zip(WIDTHS, WIDTHS[1:]):
        limit = np.sqrt(6 / (ni + no))
        params.extend([rng.uniform(-limit, limit, (ni, no)), np.zeros(no)])
    return params


def forward(x, weights):
    a = np.maximum(x @ weights[0] + weights[1], 0)
    b = np.maximum(a @ weights[2] + weights[3], 0)
    pred = (b @ weights[4] + weights[5]).ravel()
    return a, b, pred


def objective_gradients(x, target, sample_weights, weights):
    a, b, pred = forward(x, weights)
    error = pred - target
    loss = np.mean(sample_weights * error**2) + REGULARIZER * sum(np.sum(w**2) for w in weights[::2])
    delta = (2 * sample_weights * error / len(x))[:, None]
    delta_b = (delta @ weights[4].T) * (b > 0)
    delta_a = (delta_b @ weights[2].T) * (a > 0)
    gradients = [x.T @ delta_a, delta_a.sum(axis=0), a.T @ delta_b, delta_b.sum(axis=0),
                 b.T @ delta, delta.sum(axis=0)]
    for i in (0, 2, 4):
        gradients[i] += 2 * REGULARIZER * weights[i]
    return loss, gradients


def pack(weights):
    body = b"".join(np.asarray(w, dtype="<f4").tobytes() for w in weights)
    return HEADER.pack(b"JCP1", *WIDTHS) + body


def unpack(blob):
    magic, *sizes = HEADER.unpack_from(blob)
    if magic != b"JCP1" or tuple(sizes) != WIDTHS or len(blob) != HEADER.size + 4*sum(np.prod(s) for s in SHAPES):
        raise ValueError("Invalid router parameter file")
    offset = HEADER.size
    weights = []
    for shape in SHAPES:
        size = int(np.prod(shape))
        arr = np.frombuffer(blob, dtype="<f4", count=size, offset=offset).reshape(shape)
        weights.append(arr.astype(np.float64))
        offset += 4 * size
    return weights


def train_routes(x, target, y, seed, balanced):
    weights = initialize(seed)
    first = [np.zeros_like(w) for w in weights]
    second = [np.zeros_like(w) for w in weights]
    sample_weights = np.ones(len(y))
    if balanced:
        sample_weights = len(y) / (2 * np.bincount(y.astype(int))[y.astype(int)])
    saved = {}
    for step in range(1, 801):
        _, grads = objective_gradients(x, target, sample_weights, weights)
        for k, grad in enumerate(grads):
            first[k] = .9 * first[k] + .1 * grad
            second[k] = .999 * second[k] + .001 * grad * grad
            weights[k] -= .0007 * (first[k] / (1 - .9**step)) / (np.sqrt(second[k] / (1 - .999**step)) + 1e-8)
        if step in (20, 800):
            saved[step] = pack(weights)
    return saved


def fit_portable(seed, paths, tables, vectors):
    idx = np.array([vectors["lookup"][p] for p in paths])
    y = np.array([int(tables[p]["y"]) for p in paths])
    remote = np.array([float(tables[p]["dino_probability"]) for p in paths])
    fold_id = np.full(len(y), -1, dtype=np.int8)
    local_p = np.zeros(len(y))
    quantized_x = vectors["q"][idx]
    normalized_x = normalized(quantized_x)
    for fold, (train, val) in enumerate(StratifiedKFold(n_splits=4, shuffle=True, random_state=3071+seed).split(idx, y)):
        assert not set(train).intersection(val)
        head = pack_head("local_int8", *fit(normalized_x[train], y[train]))
        local_p[val] = expit(predict_from_head("local_int8", head, quantized_x[val]))
        fold_id[val] = fold
    assert (fold_id >= 0).all()
    x = probability_features(local_p)
    weights = {}
    for target_name, target in targets(y, local_p, remote).items():
        ordinary = train_routes(x, target, y, seed, balanced=False)
        balanced = train_routes(x, target, y, seed, balanced=True)
        weights[target_name + "_paper20"] = ordinary[20]
        weights[target_name + "_steps800"] = ordinary[800]
        weights[target_name + "_balanced800"] = balanced[800]
    return weights, (fold_id, local_p, remote, y)


def train(out):
    calibration = {r["source_path"]: r for r in table("matched_collaboration_scores.csv") if r["cohort"] == "calibration"}
    with np.load(RESULTS/"edge_adaptation_predictions.npz", allow_pickle=False) as z:
        train_paths = z["train_paths"]
        assert not set(train_paths.ravel()).intersection(z["test_paths"])
    with np.load(FEATURES/"edge_fgvc.npz", allow_pickle=False) as z:
        vectors = {"lookup": {p:i for i,p in enumerate(z["source_path"])}, "q": quantized(z["activation"])[0]}
    cache = {}
    folds, crossfit_probs, remote_probs, labels = [], [], [], []
    with ZipFile(out/"published_cascade_routers.zip", "w", compression=ZIP_DEFLATED) as archive:
        for seed, paths in enumerate(train_paths):
            if len(set(paths)) != 128 or not set(paths).issubset(calibration):
                raise ValueError("Invalid calibration draw")
            weights, (fold, local_p, remote, y) = fit_portable(seed, paths, calibration, vectors)
            folds.append(fold)
            crossfit_probs.append(local_p)
            remote_probs.append(remote)
            labels.append(y)
            for name, blob in weights.items():
                archive.writestr(f"s{seed:02d}/{name}.bin", blob)
                cache[seed, name] = blob
    np.savez_compressed(out/"published_cascade_crossfit.npz", paths=train_paths, fold=folds,
                        oof_local_probability=crossfit_probs, remote_probability=remote_probs, y=labels)
    return cache


def source_inputs(out, refresh):
    """Only scalar deployment margins are needed when the Latvian embedding file is unavailable."""
    path = out / "published_cascade_deployment_margins.npz"
    fallback = RESULTS / path.name
    if not refresh and (path.exists() or fallback.exists()):
        with np.load(path if path.exists() else fallback, allow_pickle=False) as z:
            return {k:z[k] for k in z}
    cache = {}
    for site in ("cornell", "kashmir", "latvia"):
        ids, y, e, expert, jpeg, local = dataset(site)
        for label, value in zip(("paths","y","edge_p","expert","jpeg","local_margin"),
                                (ids,y,e,expert,jpeg,local), strict=True):
            cache[site + "_" + label] = value
    np.savez_compressed(path, **cache)
    return cache


def budget_order(score, paths, costs, remaining, eligibility):
    keep = np.flatnonzero(score > 0) if eligibility == "positive" else np.arange(len(score))
    order = keep[np.lexsort((paths[keep], -score[keep]))]
    return rank_selection(order, costs, remaining)


def ba(y, pred):
    return .5 * ((~pred[y == 0]).mean() + pred[y == 1].mean())


@dataclass
class Decisions:
    edge: np.ndarray
    queries: np.ndarray
    setup: int
    tier: str


def uncertainty_control(paths, e, margin, costs, cap, setup, is_cold):
    charged = setup if is_cold else 0
    if charged <= cap:
        pred, score, tier = margin >= 0, -np.abs(margin), "adapted_uncertainty"
    else:
        pred, score, tier, charged = e >= EDGE_CUTOFF, -np.abs(e-EDGE_CUTOFF), "original_uncertainty", 0
    return Decisions(pred, budget_order(score, paths, costs, cap-charged, "budget"), charged, tier)


def route_sites(blobs, data, out):
    metrics = {(int(r["seed"]), loc):int(r[f"{loc}_label_total_bytes"])
               for r in table("edge_adaptation_metrics.csv")
               if r["method"] == "local_int8" and r["budget"]=="128" for loc in ("local","remote")}
    all_runs = []
    for site in ("cornell", "kashmir", "latvia"):
        paths, y, edge_p, expert, jpeg, local = (data[f"{site}_{k}"] for k in
                                                  ("paths","y","edge_p","expert","jpeg","local_margin"))
        expert = np.asarray(expert,dtype=bool)
        costs = jpeg.astype(np.int64) + REQUEST.size + REPLY.size
        predictions, masks, score_archive = [], [], []
        location = len(all_runs)
        for seed, margin in enumerate(local):
            x = probability_features(expit(margin))
            for method in METHODS:
                blob = blobs[seed,method]
                score = forward(x,unpack(blob))[-1]
                score_archive.append(score)
                min_setup = metrics[seed,"remote"] + len(blob)
                for pct in (10, 25, 50):
                    cap = int(costs.sum()) * pct // 100
                    for loc in ("local","remote"):
                        for stage in STAGES:
                            cold = stage == "cold"
                            ref = uncertainty_control(paths, edge_p, margin, costs, cap, metrics[seed,loc], cold)
                            benchmark = np.where(ref.queries, expert, ref.edge)
                            for eligibility in RANKS:
                                router_charge = min_setup if cold else 0
                                if router_charge <= cap:
                                    mask = budget_order(score, paths, costs, cap-router_charge, eligibility)
                                    effective = Decisions(margin >= 0, mask, router_charge, "learned_router")
                                else:
                                    effective = ref
                                pred = np.where(effective.queries, expert, effective.edge)
                                fp = int(((y==0)&pred).sum())
                                fn = int(((y==1)&~pred).sum())
                                bacc = ba(y,pred)
                                total = effective.setup + int(costs[effective.queries].sum())
                                assert total <= cap
                                all_runs.append(dict(site=site, seed=seed, method=method, eligibility=eligibility,
                                    cap_percent=pct, cap_bytes=cap, review_location=loc, accounting=stage,
                                    deployment=effective.tier, full_setup_bytes=min_setup, setup_bytes=effective.setup,
                                    inferred_bytes=total-effective.setup,total_bytes=total,
                                    query_count=int(effective.queries.sum()), fp=fp, fn=fn,
                                    accuracy=float((pred==y).mean()),balanced_accuracy=float(bacc),
                                    proposed_balanced_accuracy=float(ba(y,benchmark)), proposed_accuracy=float((benchmark==y).mean()),
                                    paired_delta_ba_pp=float(100*(bacc-ba(y,benchmark))),
                                    paired_delta_acc_pp=float(100*np.mean((pred==y).astype(float)-(benchmark==y))),
                                    proposed_only_correct=int(((pred!=y)&(benchmark==y)).sum()),
                                    router_only_correct=int(((pred==y)&(benchmark!=y)).sum())))
                                predictions.append(np.packbits(pred))
                                masks.append(np.packbits(effective.queries))
        np.savez_compressed(out/f"published_cascade_{site}.npz", source_path=paths,y=y,expert=expert,
                            application_bytes=costs, methods=METHODS,
                            scores=np.asarray(score_archive).reshape(20,len(METHODS),len(y)),
                            pred=np.array(predictions), queries=np.array(masks),
                            run_first_index=location)
    return all_runs


def summaries(rows):
    from itertools import groupby
    keys = ("site","cap_percent","review_location","accounting","eligibility","method")
    by_key = lambda r:tuple(r[k] for k in keys)
    result=[]
    for key, group in groupby(sorted(rows,key=by_key),key=by_key):
        vals = list(group)
        assert len(vals)==20
        deltas=np.array([v["paired_delta_ba_pp"] for v in vals])
        entry=dict(zip(keys,key,strict=True))
        for k in ("balanced_accuracy","accuracy","proposed_balanced_accuracy","proposed_accuracy","total_bytes",
                  "setup_bytes","full_setup_bytes","query_count","fp","fn","paired_delta_ba_pp","paired_delta_acc_pp"):
            entry["mean_"+k]=float(np.mean([v[k] for v in vals]))
        entry.update(seeds=20,active_router_seeds=sum(v["deployment"]=="learned_router" for v in vals),
                     better_draws=int((deltas>1e-10).sum()),tied_draws=int((abs(deltas)<=1e-10).sum()),
                     min_paired_delta_ba_pp=float(deltas.min()), max_paired_delta_ba_pp=float(deltas.max()))
        result.append(entry)
    return result


def verify(out):
    """Recompute all counts and byte totals from the packed query/decision rows."""
    with (out/"published_cascade_runs.csv").open() as f:
        all_rows=list(csv.DictReader(f))
    expected={(r["site"], int(r["seed"]), int(r["cap_percent"]), r["review_location"]):r
              for r in table("joint_budget_runs.csv") if r["method"]=="joint"}
    margin=source_inputs(out,False)
    with ZipFile(out/"published_cascade_routers.zip") as archive:
        if archive.testzip() is not None or len(archive.namelist())!=120:
            raise ValueError("Router archive failed integrity check")
        for site in ("cornell","kashmir","latvia"):
            with np.load(out/f"published_cascade_{site}.npz",allow_pickle=False) as z:
                paths=z["source_path"]
                y,serv,costs=z["y"],z["expert"].astype(bool),z["application_bytes"]
                masks=np.unpackbits(z["queries"],axis=1,count=len(y)).astype(bool)
                pred=np.unpackbits(z["pred"],axis=1,count=len(y)).astype(bool)
                scores=z["scores"]
                offset=int(z["run_first_index"])
                rows=all_rows[offset:offset+len(pred)]
                assert len(rows)==2880 and all(r["site"]==site for r in rows)
                m=margin[site+"_local_margin"]
                e=margin[site+"_edge_p"]
                assert np.array_equal(paths,margin[site+"_paths"])
                for seed in range(20):
                    x=probability_features(expit(m[seed]))
                    for k,name in enumerate(METHODS):
                        blob=archive.read(f"s{seed:02d}/{name}.bin")
                        recalculated=forward(x,unpack(blob))[-1]
                        np.testing.assert_allclose(recalculated,scores[seed,k],rtol=1e-10,atol=1e-12)
                for j,r in enumerate(rows):
                    seed,pct=int(r["seed"]),int(r["cap_percent"])
                    fallback=(r["deployment"]=="original_uncertainty")
                    start=(e>=EDGE_CUTOFF) if fallback else (m[seed]>=0)
                    assert np.array_equal(np.where(masks[j],serv,start), pred[j])
                    assert int(masks[j].sum())==int(r["query_count"])
                    assert int(costs[masks[j]].sum())+int(r["setup_bytes"])==int(r["total_bytes"])<=int(r["cap_bytes"])
                    assert int(((y==0)&pred[j]).sum())==int(r["fp"])
                    assert int(((y==1)&~pred[j]).sum())==int(r["fn"])
                    assert np.isclose(ba(y,pred[j]),float(r["balanced_accuracy"]))
                    if r["accounting"]=="cold":
                        original=expected[site,seed,pct,r["review_location"]]
                        assert np.isclose(float(original["balanced_accuracy"]),float(r["proposed_balanced_accuracy"]))
    with np.load(out/"published_cascade_crossfit.npz",allow_pickle=False) as z:
        for seed,fold in enumerate(z["fold"]):
            assert np.array_equal(np.bincount(fold),[32,32,32,32])
            assert len(set(z["paths"][seed]))==128
        assert np.isfinite(z["oof_local_probability"]).all()
    print("Verified 120 packed networks, cross-fit assignments, 8,640 per-image policies, every cap, and the original paired comparison.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=RESULTS)
    parser.add_argument("--refresh-inputs",action="store_true")
    parser.add_argument("--verify-only",action="store_true")
    args=parser.parse_args()
    if sha256((ROOT/"protocol/PUBLISHED_CASCADE.md").read_bytes()).hexdigest()!=PROTOCOL_HASH:
        raise ValueError("Protocol digest mismatch")
    args.out.mkdir(parents=True,exist_ok=True)
    if args.verify_only:
        verify(args.out)
        return
    inputs=source_inputs(args.out,args.refresh_inputs)
    blobs=train(args.out)
    runs=route_sites(blobs,inputs,args.out)
    csv_write(args.out/"published_cascade_runs.csv",runs)
    csv_write(args.out/"published_cascade_summary.csv",summaries(runs))
    print(f"{len(blobs)} trained routers, {len(runs)} budget-constrained deployments")
    verify(args.out)


if __name__=="__main__":
    main()

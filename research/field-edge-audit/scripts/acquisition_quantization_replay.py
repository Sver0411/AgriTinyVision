"""Isolate backbone precision in the archived Cornell-128 experiments."""
import csv
from collections import defaultdict
from hashlib import sha256
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from acquisition_baselines import METHODS, fit_probability_router, fit_router
from edge_adaptation_wire import quantized
from evaluate_upload import rank_selection
from published_defer import aligned, count_outcome, write_csv
from streaming_published import past_percentile, serve
from streaming_review import grant_args, original_urgency

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results"
FEATURE = ROOT / "data/features"
GRID = ROOT / "data/raw/esp-grid-cache"
STATES = (("fp32", "fp32"), ("fp32", "grid"), ("grid", "grid"))
EXPERT_METHODS = (*METHODS, "diffprob")
POLICIES = ("frozen", "uncertainty", *EXPERT_METHODS)
LEARN_EXPERT = ("ova", "compare_confidence", "mao", "diff01", "diffprob")
ARCHIVE = {
    "frozen": "frozen_history", "uncertainty": "uncertainty_history", "edgeboost_rank": "edgeboost_history",
    "edgeboost_fixed": "edgeboost_fixed", "ova": "ova_history", "compare_confidence": "compare_confidence_history",
    "mao": "mao_history",
}


def tab(name):
    with (RESULT / name).open(newline="") as f:
        return list(csv.DictReader(f))


def features(site):
    loc = FEATURE / ("edge_fgvc.npz" if site == "cornell" else site+".npz")
    with np.load(loc) as f:
        ids = f["source_path"].copy()
        q = quantized(f["activation"] if site=="cornell" else f["edge"])[0]
    with np.load(GRID / (site+".npz")) as f:
        which = aligned(f["source_path"], ids)
        g = quantized(f["integer"][which].astype(np.float32)/32)[0]
        prob = f["prob"][which]
    return ids, {"fp32": q, "grid": g}, prob


def fit_all():
    ids, xs, _ = features("cornell")
    data = {r["source_path"]:r for r in tab("specialist_balanced.csv") if r["feature_dataset"]=="fgvc"}
    sizes = {r["source_path"]:int(r["jpeg_bytes"])+20 for r in tab("fgvc_backend.csv") if r["long_edge"]=="224"}
    cloud_prob = {r["source_path"]:float(r["dino_probability"]) for r in tab("matched_collaboration_scores.csv")
                  if r["cohort"]=="calibration"}
    with np.load(RESULT/"edge_adaptation_predictions.npz") as f:
        draws = f["train_paths"]
        assert draws.shape == (20,128) and not set(draws.ravel()) & set(f["test_paths"])
    fitted = {}
    for kind in xs:
        for seed, paths in enumerate(draws):
            q = xs[kind][aligned(ids, paths)]
            y = np.array([int(data[p]["y"]) for p in paths])
            c = np.array([int(data[p]["pred_diseased"]) for p in paths], bool)
            models = {m:fit_router(m,q,y,c,seed) for m in METHODS}
            models["diffprob"] = fit_probability_router(q,y,[cloud_prob[p] for p in paths],seed)
            lengths = np.array([sizes[p] for p in paths])
            cutoff = models["edgeboost_fixed"].fixed_cutoff(lengths,.25)
            fitted[kind,seed] = models, cutoff, lengths
    return fitted


def run_site(site, fitted, reports, historical):
    ids, xs, grid_p = features(site)
    with np.load(RESULT/f"streaming_inputs_{site}.npz") as f:
        paths, y, cloud, costs, orders, tie = [f[k].copy() for k in ("paths","y","server","costs","orders","batch_tie")]
        old_edge, old_urgent = f["original"].astype(bool), f["urgencies"].copy()
    with np.load(RESULT/f"streaming_published_{site}.npz") as f:
        assert np.array_equal(f["source_path"], paths)
        old_predictions, old_queries = f["predictions"], f["queries"]
    which=aligned(ids,paths)
    xs = {k: v[which] for k,v in xs.items()}
    old_by = {"fp32": (old_edge, old_urgent), "grid":(grid_p[which]>=np.float32(.3),original_urgency(grid_p[which]))}
    edgeboost_ids = np.array([sha256(("edgeboost-order-v1/"+str(p)).encode()).hexdigest() for p in paths])
    n=len(y)
    pred_rows, query_rows = [], []
    for train, test in STATES:
        for seed in range(20):
            models, cutoff, cal = fitted[train,seed]
            scores = {m:models[m].score(xs[test]) for m in EXPERT_METHODS}
            candidates = {m:(z>=0, rank) for m,(z,rank) in scores.items()}
            candidates["uncertainty"] = candidates["edgeboost_rank"]
            candidates["frozen"] = old_by[test]
            numer = int(cal[:32].sum())
            plans = [("online_remote", seq, order, grant_args({"numerator":numer},25,n)[1]) for seq,order in enumerate(orders)]
            plans += [(k, 0, None, int(costs.sum())//4) for k in ("batch_local","batch_remote","batch_prepaid")]
            for scenario, seq, order, cap in plans:
                is_online = order is not None
                prepaid = scenario == "batch_prepaid"
                remote = scenario in ("batch_remote","online_remote")
                records = len(cal)*(xs[test].shape[1]+13)
                jpeg_fee = int((cal-12).sum())
                single = 0 if prepaid else records + (jpeg_fee if remote else 0) + models["edgeboost_rank"].download_bytes
                for method in POLICIES:
                    if method == "frozen" or prepaid:
                        setup = 0
                    else:
                        model = models["edgeboost_rank" if method=="uncertainty" else method]
                        extra = 4*len(cal) if method=="edgeboost_fixed" and not remote else 0
                        setup = records + (jpeg_fee if remote or method in LEARN_EXPERT else 0) + model.download_bytes + extra
                    if setup <= cap:
                        active = method
                    elif single <= cap and (method in LEARN_EXPERT or not is_online):
                        active,setup = "uncertainty",single
                    else:
                        active,setup = "frozen",0
                    edge, priority = candidates[active]
                    fixed = active=="edgeboost_fixed"
                    if is_online:
                        stream_priority=priority[order] if fixed else past_percentile(priority[order])
                        sel,_=serve(stream_priority,costs[order],n,cap-setup,numer/32,cutoff if fixed else None)
                        ask=np.zeros(n,bool)
                        ask[order]=sel
                    else:
                        if fixed:
                            ix=np.argsort(edgeboost_ids)
                            selection=ix[priority[ix]>cutoff]
                        else:
                            selection=np.lexsort((tie,-priority))
                        ask=rank_selection(selection,costs,cap-setup)
                    pred=np.where(ask,cloud,edge).astype(bool)
                    actual = setup + int(costs[ask].sum())
                    assert actual<=cap
                    metric=count_outcome(y,pred)
                    if train==test=="fp32" and is_online and method in ARCHIVE:
                        row=historical[site,seed,seq,ARCHIVE[method]]
                        j=int(row["row_index"])
                        if not (np.array_equal(np.packbits(ask),old_queries[j]) and
                                np.array_equal(np.packbits(pred),old_predictions[j]) and actual==int(row["total_bytes"])):
                            raise ValueError(f"Historical FP32 disagreement {site,seed,seq,method}")
                    reports.append(dict(site=site,training_features=train,serving_features=test,scenario=scenario,
                                        seed=seed,sequence=seq,method=method,actual_policy=active,cap_bytes=cap,
                                        setup_bytes=setup,total_bytes=actual,queried=int(ask.sum()),**metric,
                                        query_sha256=sha256(np.packbits(ask).tobytes()).hexdigest(),
                                        predictions_sha256=sha256(np.packbits(pred).tobytes()).hexdigest()))
                    if site=="cornell":
                        pred_rows.append(np.packbits(pred))
                        query_rows.append(np.packbits(ask))
    if site=="cornell":
        local_rows = [r for r in reports if r["site"]=="cornell"]
        assert len(local_rows)==len(pred_rows)==len(query_rows)
        for record, pred_bits, ask_bits in zip(local_rows, pred_rows, query_rows, strict=True):
            assert sha256(pred_bits.tobytes()).hexdigest()==record["predictions_sha256"]
            assert sha256(ask_bits.tobytes()).hexdigest()==record["query_sha256"]
        np.savez_compressed(RESULT / "acquisition_quantization_cornell_masks.npz",
                            source_path=paths, pred=np.stack(pred_rows), query=np.stack(query_rows))



def summaries(rows):
    groups=defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in ("site","training_features","serving_features","scenario","method"))].append(r)
    out=[]
    for key,g in groups.items():
        counts={k:float(np.mean([r[k] for r in g])) for k in ("fp","fn","accuracy","balanced_accuracy","total_bytes","queried","setup_bytes")}
        out.append(dict(zip(("site","training_features","serving_features","scenario","method"),key),runs=len(g),**counts))
    # Use a unique set for rankings: margin and EdgeBoost rank are a proven duplicate.
    ranks=defaultdict(list)
    for r in out:
        tag=(r["site"],r["training_features"],r["serving_features"],r["scenario"])
        if r["method"]!="uncertainty":ranks[tag].append(r)
    for tag,group in ranks.items():
        for r,rank in zip(group,rankdata([-r["balanced_accuracy"] for r in group],method="min"),strict=True):
            r["rank_vs_unique_methods"]=int(rank)
        margin=next(r for r in group if r["method"]=="edgeboost_rank")
        copy=next(r for r in out if tuple(r[k] for k in ("site","training_features","serving_features","scenario"))==tag and r["method"]=="uncertainty")
        assert np.isclose(copy["balanced_accuracy"],margin["balanced_accuracy"]);copy["rank_vs_unique_methods"]=margin["rank_vs_unique_methods"]
    return out


def paired_changes(rows):
    # Average image order within each draw before counting the signs. Repeated
    # held-out cohorts do not supply independent field replications.
    by_seed=defaultdict(list)
    for r in rows:
        key=tuple(r[k] for k in ("site","scenario","training_features","serving_features","method","seed"))
        by_seed[key].append(r["balanced_accuracy"])
    by_seed={k:float(np.mean(v)) for k,v in by_seed.items()}
    out=[]
    unique=[m for m in POLICIES if m!="uncertainty"]
    for site in ("cornell","kashmir","latvia"):
        for scenario in ("online_remote","batch_local","batch_remote","batch_prepaid"):
            for next_state in STATES[1:]:
                for first, second in combinations(unique,2):
                    deltas=[]
                    for state in (STATES[0],next_state):
                        deltas.append(np.array([by_seed[(site,scenario,*state,first,seed)]-by_seed[(site,scenario,*state,second,seed)]
                                                    for seed in range(20)])*100)
                    before,after=deltas
                    sign=lambda x:np.sign(np.where(abs(x)>1e-10,x,0.))
                    out.append(dict(site=site,scenario=scenario,training_features=next_state[0],serving_features=next_state[1],
                                    first=first,second=second,old_delta_ba_pp=before.mean(),new_delta_ba_pp=after.mean(),
                                    order_reversed=int(sign(before.mean())*sign(after.mean())<0),
                                    before_first_wins=int((sign(before)>0).sum()),after_first_wins=int((sign(after)>0).sum()),
                                    paired_draws_with_reversal=int((sign(before)*sign(after)<0).sum()),draws=20))
    return out


def verify_offline(summaries):
    tags={"uncertainty":"confidence_reference","edgeboost_rank":"edgeboost_budget_rank",
          "edgeboost_fixed":"edgeboost_fixed_gate","ova":"ova_balanced","compare_confidence":"compare_confidence_balanced",
          "mao":"mao_two_stage_balanced","diff01":"diff01_paper20","diffprob":"diffprob_paper20",
          "gatekeeper":"gk_a05_t100","ce":"ce_t100_bal"}
    lookup={r["method"]:r for r in tab("methods_comparison.csv")
            if r["edge_model"]=="AgriTinyVision_MobileNetV2"
            and (not r["implementation"].startswith("linear_router_") or r["implementation"]=="linear_router_128_expert_calibration_photos")}
    count=0
    for r in summaries:
        if r["training_features"]!=r["serving_features"] or r["serving_features"]!="fp32" or r["method"] not in tags:
            continue
        cond={"batch_local":"cold_local","batch_remote":"cold_remote","batch_prepaid":"predeployed"}.get(r["scenario"])
        if cond:
            ref=float(lookup[tags[r["method"]]][r["site"]+"_"+cond+"_BA_pct"])
            if abs(100*r["balanced_accuracy"]-ref)>1e-8:
                raise ValueError(f"Historical offline mean mismatch for {r['site']}, {r['method']}, {cond}: {ref}, {r['balanced_accuracy']}")
            count+=1
    return count


def main():
    fitted = fit_all()
    gold = {(r["site"],int(r["seed"]),int(r["sequence"]),r["method"]):r for r in tab("streaming_published_runs.csv")
            if r["cap_percent"]=="25" and r["reviewer"]=="remote"}
    reports=[]
    for site in ("cornell","kashmir","latvia"):
        run_site(site,fitted,reports,gold)
    out=summaries(reports)
    print("Verified offline mean comparisons against earlier 25% publication table:",verify_offline(out))
    write_csv(RESULT/"acquisition_quantization_replay_runs.csv",reports)
    write_csv(RESULT/"acquisition_quantization_paired.csv",paired_changes(reports))
    write_csv(RESULT/"acquisition_quantization_replay_summary.csv",out)
    print("Verified",len(gold),"archived FP32 selection masks, predictions and byte accounts.")
    print(len(reports),"paired evaluations,",len(out),"summaries.")
    for r in out:
        if r["scenario"]=="online_remote" and r["rank_vs_unique_methods"]==1:
            print(r["site"], r["training_features"], r["serving_features"], r["method"],round(100*r["balanced_accuracy"],2))


if __name__=="__main__":
    main()

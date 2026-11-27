"""Head-only, equal-byte transfer of the published Gatekeeper classification objective."""

import argparse
import csv
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from scipy.special import expit, xlogy

from directional_policy import EDGE_CUTOFF, REQUEST, REPLY
from edge_adaptation_wire import I8, normalized, pack_head, quantized
from joint_budget import table
from published_cascade import ROOT, RESULTS, FEATURES, STAGES, ba, csv_write, uncertainty_control
from evaluate_upload import rank_selection

PROTOCOL_HASH = "a43c7fdd89e5cf2354834a76952be4eaf026a06762ea464a7dbfa1517b503648"
# Name, alpha (None is cross-entropy), optimizer horizon, and class balancing.
ARMS = (("gk_a05_t100", .5, 100, False),
        ("gk_a01_t100", .1, 100, False), ("gk_a03_t100", .3, 100, False),
        ("gk_a07_t100", .7, 100, False), ("gk_a09_t100", .9, 100, False),
        ("gk_a05_t20", .5, 20, False), ("gk_a05_t400", .5, 400, False),
        ("gk_a05_t100_bal", .5, 100, True),
        ("ce_t100", None, 100, False), ("ce_t100_bal", None, 100, True))
METHODS = tuple(arm[0] for arm in ARMS)


def objective(z, y, alpha, sample_weight):
    p = expit(z)
    ce = np.logaddexp(0, z) - y*z
    if alpha is None:
        return np.mean(sample_weight * ce), sample_weight * (p-y) / len(y)
    true = (z >= 0) == y
    kl = np.log(2.0) + xlogy(p, p) + xlogy(1-p, 1-p)
    per_row = np.where(true, alpha*ce, (1-alpha)*kl)
    gradient = np.where(true, alpha*(p-y), (1-alpha)*z*p*(1-p))
    return np.mean(sample_weight * per_row), sample_weight*gradient/len(y)


def decode_head(blob):
    magic, dim, scale, bias = I8.unpack_from(blob)
    if magic != b"EQ01" or len(blob)!=dim+I8.size:
        raise ValueError("Bad original decision head")
    return np.r_[np.frombuffer(blob, np.int8, offset=I8.size).astype(float)*scale, bias]


def optimize(x, y, start, alpha, horizon, balanced):
    w=start.copy()
    first=np.zeros_like(w)
    second=np.zeros_like(w)
    weights=len(y)/(2*np.bincount(y)[y]) if balanced else np.ones(len(y))
    for t in range(1,horizon+1):
        _, derivative=objective(x@w,y,alpha,weights)
        g=x.T@derivative
        first=.9*first+.1*g
        second=.999*second+.001*g*g
        w-=.001*(first/(1-.9**t))/(np.sqrt(second/(1-.999**t))+1e-8)
    return pack_head("local_int8",w[:-1],w[-1])


def fitting_data():
    with np.load(RESULTS/"edge_adaptation_predictions.npz") as z:
        per_seed=z["train_paths"]
        assert len(set(z["test_paths"]).intersection(per_seed.ravel()))==0
    with np.load(FEATURES/"edge_fgvc.npz") as z:
        by_path={p:i for i,p in enumerate(z["source_path"])}
        q,_=quantized(z["activation"])
    labels={r["source_path"]:int(r["y"]) for r in table("edge_adaptation_split.csv") if r["status"]=="label_pool"}
    return per_seed,by_path,q,labels


def train(out):
    per_seed,by_path,q,labels=fitting_data()
    weights={}
    training=[]
    with ZipFile(RESULTS/"edge_adaptation_heads.zip") as base, ZipFile(out/"published_gatekeeper_heads.zip","w",compression=ZIP_DEFLATED) as archive:
        for seed,paths in enumerate(per_seed):
            y=np.array([labels[p] for p in paths])
            assert len(set(paths))==len(y)==128
            x0=normalized(q[[by_path[p] for p in paths]])
            x=np.column_stack((x0,np.ones(len(paths))))
            original=base.read(f"s{seed:02d}-b128/local_int8.bin")
            start=decode_head(original)
            for name,alpha,steps,balanced in ARMS:
                blob=optimize(x,y,start,alpha,steps,balanced)
                assert len(blob)==len(original)==1294
                archive.writestr(f"s{seed:02d}/{name}.bin",blob)
                weights[seed,name]=blob
                samplew=len(y)/(2*np.bincount(y)[y]) if balanced else np.ones(len(y))
                end=decode_head(blob)
                training.append(dict(seed=seed,method=name,alpha=alpha,steps=steps,class_balanced=int(balanced),
                                     setup_head_bytes=len(blob), training_ids_sha256=sha256(("\n".join(paths)+"\n").encode()).hexdigest(),
                                     initial_head_sha256=sha256(original).hexdigest(),final_head_sha256=sha256(blob).hexdigest(),
                                     initial_correct=int(((x@start>=0)==y).sum()),final_correct=int(((x@end>=0)==y).sum()),
                                     initial_objective=objective(x@start,y,alpha,samplew)[0],final_objective=objective(x@end,y,alpha,samplew)[0]))
    csv_write(out/"published_gatekeeper_fit.csv",training)
    return weights


def inference(activations,weights):
    """Use integer dot products and normalized INT8 rows, as in deployed EQ01."""
    q,_=quantized(activations)
    q=q.astype(np.int32)
    norm=np.sqrt(np.einsum("ij,ij->i",q,q,dtype=np.int32))
    norm=np.where(norm>0,norm,1.)
    codes=[]; scales=[]; biases=[]
    for seed in range(20):
        for m in METHODS:
            blob=weights[seed,m]
            magic,dim,scale,bias=I8.unpack_from(blob)
            assert magic==b"EQ01" and len(blob)==I8.size+dim and q.shape[1]==dim
            codes.append(np.frombuffer(blob,np.int8,offset=I8.size))
            scales.append(scale);biases.append(bias)
    dot=np.asarray(codes,dtype=np.int32)@q.T
    return (dot*np.array(scales)[:,None]/norm+np.array(biases)[:,None]).reshape(20,len(METHODS),len(q))


def evaluate(out,weights):
    with np.load(RESULTS/"published_cascade_deployment_margins.npz",allow_pickle=False) as z:
        compact={k:z[k] for k in z}
    setup={(int(r["seed"]),loc):int(r[f"{loc}_label_total_bytes"]) for r in table("edge_adaptation_metrics.csv")
           if r["method"]=="local_int8" and r["budget"]=="128" for loc in ("local","remote")}
    rows=[]
    for site in ("cornell","kashmir","latvia"):
        paths,y,e,expert,jpeg,original_score=(compact[site+"_"+k] for k in ("paths","y","edge_p","expert","jpeg","local_margin"))
        costs=jpeg.astype(np.int64)+REQUEST.size+REPLY.size
        feature_path=FEATURES/("edge_fgvc.npz" if site=="cornell" else f"{site}.npz")
        archive_name="published_gatekeeper_"+site+".npz"
        if feature_path.exists():
            with np.load(feature_path,allow_pickle=False) as z:
                idx={p:i for i,p in enumerate(z["source_path"])}
                acts=z["activation" if site=="cornell" else "edge"][[idx[p] for p in paths]]
            scores=inference(acts,weights)
        else:
            if site!="latvia":
                raise FileNotFoundError(feature_path)
            # The portable artifact carries signed margins and model identities for the restricted archive.
            with np.load(RESULTS/archive_name,allow_pickle=False) as z:
                np.testing.assert_array_equal(z["source_path"],paths)
                expected=[sha256(weights[s,m]).hexdigest() for s in range(20) for m in METHODS]
                np.testing.assert_array_equal(z["model_sha256"],expected)
                scores=z["signed_margin"]
        start=len(rows); decisions=[]; queries=[]
        for seed in range(20):
            for j,method in enumerate(METHODS):
                m=scores[seed,j]
                order=np.lexsort((paths,np.abs(m)))
                adapted=m>=0
                for pct in (10,25,50):
                    cap=int(costs.sum())*pct//100
                    for loc in ("local","remote"):
                        for stage in STAGES:
                            cold=stage=="cold"
                            ref=uncertainty_control(paths,e,original_score[seed],costs,cap,setup[seed,loc],cold)
                            benchmark=np.where(ref.queries,expert,ref.edge)
                            if ref.tier=="original_uncertainty":
                                pred,asked,edge=benchmark,ref.queries,ref.edge
                            else:
                                asked=rank_selection(order,costs,cap-ref.setup)
                                edge=adapted
                                pred=np.where(asked,expert,edge)
                            size=ref.setup+int(costs[asked].sum())
                            fp=int(((y==0)&pred).sum());fn=int(((y==1)&~pred).sum())
                            rows.append(dict(site=site,seed=seed,method=method,eligibility="budget",cap_percent=pct,
                                cap_bytes=cap,review_location=loc,accounting=stage,
                                deployment="gatekeeper" if ref.tier=="adapted_uncertainty" else ref.tier,
                                full_setup_bytes=setup[seed,loc],setup_bytes=ref.setup,total_bytes=size,
                                query_count=int(asked.sum()),fp=fp,fn=fn,accuracy=float(np.mean(pred==y)),
                                balanced_accuracy=ba(y,pred),proposed_accuracy=float(np.mean(benchmark==y)),
                                proposed_balanced_accuracy=ba(y,benchmark),paired_delta_ba_pp=100*(ba(y,pred)-ba(y,benchmark)),
                                paired_delta_acc_pp=100*np.mean((pred==y).astype(float)-(benchmark==y)),
                                corrected_over_proposed=int(((pred==y)&(benchmark!=y)).sum()),
                                spoiled_over_proposed=int(((pred!=y)&(benchmark==y)).sum())))
                            decisions.append(np.packbits(pred));queries.append(np.packbits(asked))
        np.savez_compressed(out/archive_name,source_path=paths,y=y,expert=expert,application_bytes=costs,
                            signed_margin=scores,methods=METHODS,queries=np.asarray(queries),pred=np.asarray(decisions),
                            model_sha256=[sha256(weights[s,m]).hexdigest() for s in range(20) for m in METHODS],
                            run_first_index=start)
    csv_write(out/"published_gatekeeper_runs.csv",rows)
    keys=("site","method","cap_percent","review_location","accounting")
    from itertools import groupby
    key=lambda row:tuple(row[k] for k in keys)
    grouped=[]
    for k,g in groupby(sorted(rows,key=key),key=key):
        group=list(g)
        assert len(group)==20
        record=dict(zip(keys,k))
        for name in ("balanced_accuracy","accuracy","proposed_accuracy","proposed_balanced_accuracy",
                     "paired_delta_ba_pp","paired_delta_acc_pp","query_count","total_bytes","setup_bytes","fp","fn"):
            record["mean_"+name]=float(np.mean([r[name] for r in group]))
        delta=np.array([r["paired_delta_ba_pp"] for r in group])
        record.update(active_head_draws=sum(r["deployment"]=="gatekeeper" for r in group),
                      better_draws=int((delta>1e-10).sum()),tied_draws=int((abs(delta)<=1e-10).sum()),
                      minimum_delta_ba_pp=float(delta.min()),maximum_delta_ba_pp=float(delta.max()))
        grouped.append(record)
    csv_write(out/"published_gatekeeper_summary.csv",grouped)


def verify(out):
    with (out/"published_gatekeeper_runs.csv").open() as f:
        rows=list(csv.DictReader(f))
    reference={(r["site"],int(r["seed"]),int(r["cap_percent"]),r["review_location"]):r for r in table("joint_budget_runs.csv") if r["method"]=="joint"}
    with ZipFile(out/"published_gatekeeper_heads.zip") as heads, np.load(RESULTS/"published_cascade_deployment_margins.npz") as compact:
        assert heads.testzip() is None and len(heads.namelist())==20*len(METHODS)
        hashes=[sha256(heads.read(f"s{s:02d}/{m}.bin")).hexdigest() for s in range(20) for m in METHODS]
        for site in ("cornell","kashmir","latvia"):
            with np.load(out/f"published_gatekeeper_{site}.npz",allow_pickle=False) as z:
                np.testing.assert_array_equal(z["model_sha256"],hashes)
                y,expert,cost=z["y"],z["expert"],z["application_bytes"]
                ask=np.unpackbits(z["queries"],axis=1,count=len(y)).astype(bool)
                pred=np.unpackbits(z["pred"],axis=1,count=len(y)).astype(bool)
                score=z["signed_margin"]
                e=compact[site+"_edge_p"]
                block=rows[int(z["run_first_index"]):int(z["run_first_index"])+len(pred)]
                for i,r in enumerate(block):
                    assert r["site"]==site
                    seed,pct=int(r["seed"]),int(r["cap_percent"])
                    m=score[seed,METHODS.index(r["method"])]
                    fallback=r["deployment"]=="original_uncertainty"
                    edge=(e>=EDGE_CUTOFF) if fallback else m>=0
                    assert np.array_equal(pred[i],np.where(ask[i],expert,edge))
                    assert int(cost[ask[i]].sum())+int(r["setup_bytes"])==int(r["total_bytes"])<=int(r["cap_bytes"])
                    assert int(((y==0)&pred[i]).sum())==int(r["fp"]) and int(((y==1)&~pred[i]).sum())==int(r["fn"])
                    assert np.isclose(ba(y,pred[i]),float(r["balanced_accuracy"]))
                    if r["accounting"]=="cold":
                        old=reference[site,seed,pct,r["review_location"]]
                        assert np.isclose(float(r["proposed_balanced_accuracy"]),float(old["balanced_accuracy"]))
                        assert int(r["setup_bytes"])==int(old["setup_bytes"])
    print(f"Verified {len(hashes)} head binaries and {len(rows)} equal-setup byte-capped deployments.")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=RESULTS)
    p.add_argument("--verify-only",action="store_true")
    args=p.parse_args()
    if sha256((ROOT/"protocol/PUBLISHED_GATEKEEPER.md").read_bytes()).hexdigest()!=PROTOCOL_HASH:
        raise ValueError("Gatekeeper protocol digest mismatch")
    args.out.mkdir(parents=True,exist_ok=True)
    if not args.verify_only:
        evaluate(args.out,train(args.out))
    verify(args.out)


if __name__=="__main__":
    main()

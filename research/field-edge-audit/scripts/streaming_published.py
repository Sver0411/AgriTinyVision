"""One-frame matched streaming comparisons for published routing scores."""

import argparse
from bisect import bisect_left, bisect_right, insort
from collections import deque
import csv
from pathlib import Path

import numpy as np

from published_defer import count_outcome, write_csv, aligned
from streaming_review import RESULTS, SITES, CAPS, SEEDS, grant_args

METHODS=("frozen_history", "uncertainty_history","edgeboost_history","compare_confidence_history",
         "ova_history","mao_history","edgeboost_fixed")
LEARNED={"compare_confidence_history":"compare_confidence_balanced","ova_history":"ova_balanced",
         "mao_history":"mao_two_stage_balanced"}
WINDOW=64
BURN=16


def past_percentile(scores):
    """One new scalar enters after its rank among the preceding window is computed."""
    fifo=deque()
    ordered=[]
    result=[]
    for s in scores:
        if len(ordered)<BURN:
            result.append(-1.)
        else:
            result.append((bisect_left(ordered,s)+bisect_right(ordered,s))/(2*len(ordered)))
        if len(fifo)==WINDOW:
            old=fifo.popleft()
            del ordered[bisect_left(ordered,old)]
        fifo.append(s)
        insort(ordered,s)
    return result


def serve(priorities, lengths, planned_n, budget, mean, fixed_gate=None):
    """Consume scalar and size streams without access to future scores or labels."""
    balance=budget
    selected=[]
    largest=0
    for t,(priority,nbytes) in enumerate(zip(priorities,lengths,strict=True)):
        if fixed_gate is None:
            target=min(1.,balance/max(1,planned_n-t)/max(1.,mean))
            send=priority >= 1.-target and priority>=0
        else:
            send=priority>fixed_gate
        if send and balance>=20:
            largest=max(largest,nbytes-20)
            mean+=.05*(nbytes-mean)
            send=nbytes<=balance
        else:
            send=False
        if send:balance-=nbytes
        selected.append(send)
    return np.asarray(selected,bool),largest


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    with (RESULTS/"published_defer_training.csv").open() as fp:
        charges={int(r["seed"]):r for r in csv.DictReader(fp)}
    with (RESULTS/"published_edgeboost_calibration.csv").open() as fp:
        edgeboost={int(r["seed"]):r for r in csv.DictReader(fp)}
    with np.load(RESULTS/"streaming_calibration.npz") as z:
        nums=[int(z[f"s{s}_b128_numerator"]) for s in SEEDS]
    reports=[]
    for site in SITES:
        with np.load(RESULTS/f"streaming_inputs_{site}.npz") as a, np.load(RESULTS/f"published_defer_scores_{site}.npz") as p:
            ids=a["paths"]
            take=aligned(p["source_path"],ids)
            cloud=a["server"].astype(bool)
            old=a["original"].astype(bool)
            orig_rank=a["urgencies"]
            orders=a["orders"]
            lengths=a["costs"].astype(np.int64)
            margin=a["margins"][1]
            y=a["y"]
            source_methods=list(p["methods"])
            scores=p["priority"][:,:,take]
            local=p["local"][:,:,take].astype(bool)
            if not np.array_equal(cloud,p["expert"][take]) or not np.array_equal(y,p["y"][take]):
                raise ValueError("Mismatched expert or labels")
        n=len(ids)
        decisions=[]
        masks=[]
        site_rows=[]
        for seed in SEEDS:
            temperature=float(edgeboost[seed]["temperature"])
            confidence=-abs(margin[seed])
            # tanh is monotone; score arguments remain finite and avoid saturation.
            eb_margin=np.tanh(np.abs(margin[seed])/(2*temperature))
            edge_priority=-eb_margin
            dep={"frozen_history":(orig_rank,old),"uncertainty_history":(confidence,margin[seed]>=0),
                 "edgeboost_history":(edge_priority,margin[seed]>=0),"edgeboost_fixed":(edge_priority,margin[seed]>=0)}
            for tag,m in LEARNED.items():
                idx=source_methods.index(m)
                dep[tag]=(scores[seed,idx],local[seed,idx])
            for sequence,order in enumerate(orders):
                l=lengths[order].tolist()
                frozen_stream=past_percentile(orig_rank[order].tolist())
                priorities={}
                for name,(u,_) in dep.items():
                    priorities[name]=u[order].tolist() if name=="edgeboost_fixed" else past_percentile(u[order].tolist())
                if not np.array_equal(priorities["uncertainty_history"],priorities["edgeboost_history"]):
                    raise ValueError("Rounding altered EdgeBoost ranking in the evaluated score range")
                for reviewer in ("local","remote"):
                    reference_cost=int(charges[seed][f"confidence_{reviewer}_setup"])
                    remote_router_cost=int(charges[seed]["learned_remote_setup_bytes" if reviewer=="remote" else "learned_setup_bytes"])
                    for cap_pct in CAPS:
                        _,cap=grant_args({"numerator":nums[seed]},cap_pct,n)
                        baseline_query=None
                        for name in METHODS:
                            desired=0 if name=="frozen_history" else remote_router_cost if name in LEARNED else reference_cost
                            # No temperature needs to be transmitted in rank-only mode.
                            # The original fixed operating point transmits T and threshold.
                            if name=="edgeboost_fixed":desired+=8
                            if desired<=cap:
                                active,setup=name,desired
                            elif name in LEARNED and reference_cost<=cap:
                                active,setup="uncertainty_history",reference_cost
                            else:
                                active,setup="frozen_history",0
                            rank=frozen_stream if active=="frozen_history" else priorities[active]
                            local_class=dep[active][1]
                            fixed=-float(edgeboost[seed][f"threshold_{cap_pct}"]) if active=="edgeboost_fixed" else None
                            asks,peak=serve(rank,l,n,cap-setup,nums[seed]/32,fixed)
                            if name=="uncertainty_history":baseline_query=asks
                            if name=="edgeboost_history" and not np.array_equal(asks,baseline_query):
                                raise ValueError("Monotone EdgeBoost score lost query parity")
                            query=np.zeros(n,bool)
                            query[order]=asks
                            pred=np.where(query,cloud,local_class)
                            spent=int(lengths[query].sum())+setup
                            if spent>cap:raise ValueError("Overspent")
                            row=dict(site=site,seed=seed,sequence=sequence,reviewer=reviewer,cap_percent=cap_pct,
                                     cap_bytes=cap,method=name,active=active,desired_setup_bytes=desired,actual_setup_bytes=setup,
                                     n=n,queried=int(query.sum()),total_bytes=spent,peak_buffered_jpeg_bytes=peak,
                                     ranking_window=0 if active=="edgeboost_fixed" else WINDOW,rank_burnin=0 if active=="edgeboost_fixed" else BURN,
                                     row_index=len(site_rows),**count_outcome(y,pred))
                            site_rows.append(row)
                            masks.append(np.packbits(query))
                            decisions.append(np.packbits(pred))
        np.savez_compressed(out/f"streaming_published_{site}.npz",source_path=ids,y=y,orders=orders,
                            predictions=decisions,queries=masks)
        reports.extend(site_rows)
        print(f"{site}: {len(site_rows)} streaming comparison rows",flush=True)
    write_csv(out/"streaming_published_runs.csv",reports)
    keys=("site","reviewer","cap_percent","method")
    groups={}
    for r in reports:groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
    summary=[]
    for key,g in groups.items():
        by_order=[np.mean([r["balanced_accuracy"] for r in g if r["sequence"]==i]) for i in sorted({r["sequence"] for r in g})]
        summary.append(dict(**dict(zip(keys,key)),n_seeds=20,n_orders=len(by_order),mean_ba=float(np.mean(by_order)),
                            min_order_mean_ba=min(by_order),max_order_mean_ba=max(by_order),
                            mean_accuracy=float(np.mean([r["accuracy"] for r in g])),mean_fp=np.mean([r["fp"] for r in g]),
                            mean_fn=np.mean([r["fn"] for r in g]),mean_queries=np.mean([r["queried"] for r in g]),
                            mean_total_bytes=np.mean([r["total_bytes"] for r in g]),max_peak_jpeg_bytes=max(r["peak_buffered_jpeg_bytes"] for r in g),
                            installed_fraction=np.mean([r["method"]==r["active"] for r in g])))
    write_csv(out/"streaming_published_summary.csv",summary)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=RESULTS)
    run(parser.parse_args().out)

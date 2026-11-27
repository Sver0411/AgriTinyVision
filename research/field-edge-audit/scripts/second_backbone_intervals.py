"""Conditional sensitivity intervals for paired, frozen MobileNetV4 operating points.

Run after the locked experiment. These resamples hold the observed calibration
draws, route selection, byte sizes and expert outputs fixed in the original pool.
"""
import csv

import numpy as np

from evaluate_upload import rank_selection
from published_cascade import csv_write
from second_backbone import HEADS, RESULTS, SITES, read_csv


def interval(y, delta, dates, seed=2749, rounds=2000):
    rng = np.random.default_rng(seed)
    estimate = .5*(delta[y==0].mean(axis=0)+delta[y==1].mean(axis=0))
    out = []
    if dates is None:
        groups = [np.flatnonzero(y==label) for label in (0,1)]
        for _ in range(rounds):
            out.append(.5*sum(delta[rng.choice(g,len(g),replace=True)].mean(axis=0) for g in groups))
        unit = "image_stratified_within_class"
    else:
        days = np.unique(dates)
        # Work on sufficient statistics; each sampled date contributes every one of its pictures.
        class_n = np.array([[(y[dates==d]==label).sum() for label in (0,1)] for d in days])
        class_sum = np.array([[delta[(dates==d)&(y==label)].sum(axis=0) for label in (0,1)] for d in days])
        for weight in rng.multinomial(len(days),np.ones(len(days))/len(days),size=rounds):
            n = weight@class_n
            if (n>0).all():
                sums = np.einsum("g,gcm->cm",weight,class_sum)
                out.append((sums/n[:,None]).mean(axis=0))
        unit = f"whole_capture_date_{len(days)}_dates"
    lo,hi = np.quantile(out,[.025,.975],axis=0)
    return estimate,lo,hi,len(out),unit


def main():
    heads = read_csv(RESULTS/"second_backbone_fit.csv")
    date = {r["source_path"]:r["capture_date"] for r in read_csv(RESULTS/"latvia_local_date_split.csv")}
    rows = []
    for site in SITES:
        with np.load(RESULTS/f"second_backbone_{site}.npz") as z:
            paths,y,exp,source,margins,costs = (z[k] for k in ("paths","y","expert","source_margin","margins","application_bytes"))
        cap = int(costs.sum())//4
        dates = np.array([date[p] for p in paths]) if site.startswith("latvia") else None
        canonical = np.argsort(paths)
        random = [np.random.default_rng(21000+j).permutation(canonical) for j in range(20)]
        source_order=np.lexsort((paths,abs(source)))
        source_queries=rank_selection(source_order,costs,cap)
        methods=("source_only","source_confidence","target_only","target_random","gatekeeper_confidence","ce_balanced_confidence")
        for loc in ("local","remote"):
            per_seed=[]
            for s,row in enumerate(heads):
                paid=int(row[loc])
                if paid>cap:
                    local, remain = [source]*len(HEADS),cap
                else:
                    local, remain = margins[s],cap-paid
                selected=[rank_selection(np.lexsort((paths,abs(m))),costs,remain) for m in local]
                reviewed=[np.where(queries,exp,m>=0) for queries,m in zip(selected,local,strict=True)]
                random_correct=np.mean([np.where(rank_selection(order,costs,remain),exp,local[0]>=0)==y for order in random],axis=0)
                original=source>=0
                controls=[original==y,np.where(source_queries,exp,original)==y,(local[0]>=0)==y,random_correct,reviewed[1]==y,reviewed[2]==y]
                per_seed.append(np.asarray(reviewed[0]==y,dtype=float)[:,None]-np.column_stack(controls))
            est,lo,hi,valid,unit=interval(y,np.mean(per_seed,axis=0),dates)
            for j,name in enumerate(methods):
                rows.append(dict(site=site,cap_percent=25,review_location=loc,accounting="cold",target="target_confidence",
                                 comparator=name,mean_difference_ba_pp=100*est[j],conditional_lower95_pp=100*lo[j],
                                 conditional_upper95_pp=100*hi[j],unit=unit,requested_resamples=2000,
                                 valid_resamples=valid,calibration_draws_held_fixed=20))
    # Compare every point estimate with the budget-run table generated on the locked path.
    saved={(r["site"],r["review_location"],r["comparator"]):float(r["mean_target_confidence_minus_comparator_pp"])
           for r in read_csv(RESULTS/"second_backbone_paired.csv") if r["cap_percent"]=="25" and r["accounting"]=="cold"}
    for r in rows:
        assert np.isclose(saved[r["site"],r["review_location"],r["comparator"]],r["mean_difference_ba_pp"],atol=1e-10)
    csv_write(RESULTS/"second_backbone_intervals.csv",rows)
    print(f"{len(rows)} secondary conditional intervals checked against the main paired run table.")


if __name__=="__main__":
    main()

"""Audit Latvian error changes on locked acquisition runs, without fitting a policy."""
import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from purchase_cohorts import ROOT, load

BACKBONES = ("v2", "v2grid", "v4")
METHODS = ("native", "freeze", "freeze_guard", "adaptive", "adaptive_guard",
           "matched_rank", "matched_guard", "guard_matched_rank", "guard_matched_guard",
           "always_rank_guard", "always_ova", "always_edgeboost_rank")
TREATMENTS = ("adaptive", "adaptive_guard", "freeze_guard")


def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def save(path, rows):
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    temp = path.with_suffix(".tmp.csv")
    with temp.open("w", newline="") as f:
        w = csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    temp.replace(path)


def masks_and_groups(ids, actual_groups):
    indexed = {}
    for r in read(ROOT/"results/latvia_local_date_split.csv"):
        if r["status"] != "test":
            continue
        digest = bytes.fromhex(r["source_sha256"])
        path_hash = hashlib.sha256(r["source_path"].encode()).digest()
        uid = hashlib.sha256(b"crop-purchase-id-v1/" + digest + path_hash).hexdigest()
        indexed[uid] = r
    assert set(indexed)==set(ids)
    rows = [indexed[i] for i in ids]
    days = np.array([r["capture_date"] for r in rows])
    assert np.array_equal(actual_groups, days)
    first_for_digest = {}
    for i in np.argsort(ids):
        first_for_digest.setdefault(rows[i]["source_sha256"],i)
    exact = np.isin(np.arange(len(ids)),list(first_for_digest.values()))
    grouped = np.array([r["source_path"]==r["component_representative"] for r in rows])
    assert len(rows)==579 and exact.sum()==578 and grouped.sum()==558
    subgroups = {"all": np.ones(len(ids),dtype=bool), "exact_content":exact,
                 "phash_component":grouped}
    subgroups.update({"date_"+d: days==d for d in np.unique(days)})
    subgroups.update({"omit_"+d: days!=d for d in np.unique(days)})
    return subgroups


def summarize(y, pred, query):
    fp = pred[:,~y].sum(axis=1); fn = (~pred[:,y]).sum(axis=1)
    tp = y.sum()-fn; tn=(~y).sum()-fp
    sens=float(tp.mean()/y.sum()) if y.any() else ""
    spec=float(tn.mean()/(~y).sum()) if (~y).any() else ""
    den=tp+fp
    prec=np.divide(tp,den,out=np.zeros(len(den),float),where=den>0)
    return dict(n=len(y),pos=int(y.sum()),neg=int((~y).sum()),runs=len(pred),
                fp=float(fp.mean()),fn=float(fn.mean()),tp=float(tp.mean()),tn=float(tn.mean()),
                fp_min=int(fp.min()),fp_max=int(fp.max()),fn_min=int(fn.min()),fn_max=int(fn.max()),
                sensitivity=sens,specificity=spec,ba=(sens+spec)/2 if sens!="" and spec!="" else "",
                accuracy=float((tp+tn).mean()/len(y)),alerts=float(pred.sum(1).mean()),
                precision_if_any_alert=float(prec[den>0].mean()) if (den>0).any() else "",
                queries=float(query.sum(1).mean()))


def comparison(y, treat, base):
    up=treat & ~base; down=base & ~treat
    rescued=up[:,y].sum(1); spoiled=down[:,y].sum(1)
    extra=up[:,~y].sum(1); cleared=down[:,~y].sum(1)
    fn0=(~base[:,y]).sum(1); pred_neg=(~base).sum(1)
    tp0=base[:,y].sum(1); pred_pos=base.sum(1)
    sh_up=np.divide(up.sum(1)*fn0,pred_neg,out=np.zeros(len(base),float),where=pred_neg>0)
    sh_down=np.divide(down.sum(1)*tp0,pred_pos,out=np.zeros(len(base),float),where=pred_pos>0)
    hits=rescued-spoiled; fp=extra-cleared; sham=sh_up-sh_down
    assert np.array_equal((treat.sum(1)-base.sum(1)),hits+fp)
    return dict(n=len(y),n_pos=int(y.sum()),n_neg=int((~y).sum()),runs=len(treat),
                rescued_fn=rescued.mean(),introduced_fn=spoiled.mean(),
                corrected_fp=cleared.mean(),introduced_fp=extra.mean(),
                net_fn=(spoiled-rescued).mean(),net_fp=fp.mean(),net_alerts=(hits+fp).mean(),
                sham_expected_net_tp=sham.mean(),observed_minus_sham_tp=(hits-sham).mean(),
                proportion_runs_both_errors_strictly_lower=((hits>0)&(fp<0)).mean(),
                fn_cost_in_fp_units_at_tie=(fp.mean()/hits.mean() if hits.mean()>0 and fp.mean()>=0 else ""))


def audit_one(out, backbone):
    data=load("latvia",backbone)
    y=data["test_y"].astype(bool)
    groups=masks_and_groups(data["test_id"],data["test_group"])
    native=data["test_p"]>=float(data["cutoff"])
    native_fp=int((native&~y).sum()); native_fn=int((~native&y).sum())
    ceiling=dict(backbone=backbone,images=len(y),positives=int(y.sum()),negatives=int((~y).sum()),
                 native_fp=native_fp,native_fn=native_fn,guard_min_fp=native_fp,guard_max_fn=native_fn,
                 guard_specificity_ceiling=1-native_fp/(~y).sum(),
                 guard_ba_ceiling_with_perfect_sensitivity=1-native_fp/(2*(~y).sum()))
    meta_grid=defaultdict(lambda:defaultdict(list))
    for r in read(out/("runs_latvia_"+backbone+".csv")):
        if int(r["cap_pct"])==25 and r["method"] in METHODS:
            meta_grid[int(r["label_cap"]),r["grant_mode"]][r["method"]].append(r)
    if set(meta_grid)!={(b,m) for b in (32,128) for m in ("earned","prepaid")}:
        raise ValueError(f"Missing primary cells for {backbone}: {meta_grid.keys()}")
    with np.load(out/("masks_latvia_"+backbone+".npz")) as z:
        assert np.array_equal(data["test_id"],z["test_id"])
        assert np.array_equal(data["test_group"],z["test_group"])
        pm,qm=z["pred"],z["query"]
    state_rows=[]; change_rows=[]; route_rows=[]
    for (cap,mode),by_method in sorted(meta_grid.items()):
        meta=dict(backbone=backbone,label_cap=cap,cap_pct=25,grant_mode=mode)
        results={}
        if set(METHODS)!=set(by_method):
            raise ValueError(f"{backbone}: missing expected methods {set(METHODS)-set(by_method)}")
        run_keys=None
        for method,rr in by_method.items():
            rr=sorted(rr,key=lambda r:(int(r["seed"]),int(r["order"])))
            keys=[(int(r["seed"]),int(r["order"])) for r in rr]
            if run_keys is None: run_keys=keys
            assert keys==run_keys and len(set(keys))==20
            ii=[int(r["mask_row"]) for r in rr]
            pred=np.unpackbits(pm[ii],axis=1,count=len(y)).astype(bool)
            query=np.unpackbits(qm[ii],axis=1,count=len(y)).astype(bool)
            assert np.array_equal(pred[:,~y].sum(1),[int(r["fp"]) for r in rr])
            assert np.array_equal((~pred[:,y]).sum(1),[int(r["fn"]) for r in rr])
            assert np.array_equal(query.sum(1),[int(r["queries"]) for r in rr])
            if method=="native":
                assert np.all(pred==native) and not query.any()
            if method.endswith("_guard"):
                assert np.all(pred[:,native]) and not query[:,native].any(),method
            results[method]=(pred,query)
            for group,keep in groups.items():
                state_rows.append(dict(**meta,method=method,subset=group,
                                       **summarize(y[keep],pred[:,keep],query[:,keep])))
        for t in TREATMENTS:
            tp,tq=results[t]
            for name in METHODS:
                if t==name or (t=="freeze_guard" and name not in ("native","freeze")):
                    continue
                bp,bq=results[name]
                for group,keep in groups.items():
                    change_rows.append(dict(**meta,treatment=t,baseline=name,subset=group,
                                         **comparison(y[keep],tp[:,keep],bp[:,keep])))
                for t_req,b_req in ((False,False),(False,True),(True,False),(True,True)):
                    k=(tq==t_req)&(bq==b_req)
                    up=k&tp&~bp; down=k&~tp&bp
                    route_rows.append(dict(**meta,treatment=t,baseline=name,
                                           treatment_queried=int(t_req),baseline_queried=int(b_req),
                                           exposures=float(k.sum(1).mean()),
                                           rescued_fn=float(up[:,y].sum(1).mean()),introduced_fn=float(down[:,y].sum(1).mean()),
                                           corrected_fp=float(down[:,~y].sum(1).mean()),introduced_fp=float(up[:,~y].sum(1).mean())))
    return ceiling,state_rows,change_rows,route_rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results",type=Path,default=ROOT/"results/purchase")
    p.add_argument("--backbones",nargs="+",choices=BACKBONES,default=BACKBONES)
    args=p.parse_args()
    ceilings=[]; states=[]; changes=[]; routes=[]
    for b in args.backbones:
        c,s,d,r=audit_one(args.results,b)
        ceilings.append(c);states.extend(s);changes.extend(d);routes.extend(r)
    # A partial rerun uses a suffix so it cannot overwrite a completed full audit.
    tag="" if list(args.backbones)==list(BACKBONES) else "_"+"_".join(args.backbones)
    for name,rows in (("guard_ceiling",ceilings),("operating_subsets",states),("paired_subsets",changes),
                      ("query_decomposition",routes)):
        save(args.results/("latvia_"+name+tag+".csv"),rows)
    focus_pairs = {("freeze_guard", "native"), ("adaptive_guard", "native"), ("adaptive_guard", "freeze_guard"),
                   ("adaptive", "freeze"), ("adaptive_guard", "guard_matched_guard")}
    primary = [r for r in changes if r["subset"] == "all" and r["label_cap"] == 32
               and (r["treatment"], r["baseline"]) in focus_pairs]
    save(args.results / ("latvia_recall_primary" + tag + ".csv"), primary)
    print(f"{len(ceilings)} backbones, {len(states)} operating rows, {len(changes)} paired subsets, {len(routes)} query strata")


if __name__=="__main__":
    main()

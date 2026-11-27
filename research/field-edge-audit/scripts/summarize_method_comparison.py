"""Produce the manuscript's matched 25% cap comparison directly from run files."""
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
SITES = ("cornell","kashmir","latvia","latvia_later")
STAGES = ("cold_local","cold_remote","predeployed")


def rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def summarize():
    cells = defaultdict(dict)
    def add(model,method,role,site,stage,ba,active):
        key = (model,method,role)
        assert (site,stage) not in cells[key]
        cells[key][site,stage] = (float(ba)*100,int(active))
    name = "AgriTinyVision_MobileNetV2"
    for n in (128,32,8,64):
        directory = OUT if n==128 else OUT/f"published_defer_partial_{n}"
        for r in rows(directory/"published_defer_summary.csv"):
            if r["cap_percent"]!="25" or (r["accounting"]=="prepaid" and r["reviewer"]=="remote"):
                continue
            method = r["method"]
            if n!=128 and method=="confidence_reference":
                continue
            role = "adapted_reference" if method=="confidence_reference" else f"linear_router_{n}_expert_calibration_photos"
            stage = "predeployed" if r["accounting"]=="prepaid" else "cold_"+r["reviewer"]
            add(name,method,role,r["site"],stage,r["mean_ba"],r["installed"])
    rates = defaultdict(list)
    active = defaultdict(int)
    for r in rows(OUT/"published_edgeboost_runs.csv"):
        if r["cap_percent"]=="25":
            stage = "predeployed" if r["review_location"]=="prepaid" else "cold_"+r["review_location"]
            key = (r["method"],r["site"],stage)
            rates[key].append(float(r["balanced_accuracy"]))
            active[key] += r["deployment"]=="full_policy"
    for (m,site,stage),v in rates.items():
        assert len(v)==20
        add(name,m,"binary_temperature_margin",site,stage,mean(v),active[m,site,stage])
    for kind in ("cascade","gatekeeper"):
        for r in rows(OUT/f"published_{kind}_summary.csv"):
            if r["cap_percent"]!="25" or r.get("eligibility","budget")!="budget" or (r["accounting"]=="predeployed" and r["review_location"]=="remote"):
                continue
            m = r["method"]
            role = "cross_fitted_conditional_gain" if kind=="cascade" else "head_only_continuation"
            stage = "predeployed" if r["accounting"]=="predeployed" else "cold_"+r["review_location"]
            add(name,m,role,r["site"],stage,r["mean_balanced_accuracy"],r["active_router_seeds" if kind=="cascade" else "active_head_draws"])
    for r in rows(OUT/"second_backbone_summary.csv"):
        if r["cap_percent"]!="25" or (r["accounting"]=="predeployed" and r["review_location"]=="remote"):
            continue
        stage = "predeployed" if r["accounting"]=="predeployed" else "cold_"+r["review_location"]
        add("MobileNetV4_Conv_Small",r["method"],"second_edge_model",r["site"],stage,r["mean_balanced_accuracy"],r["active_head_draws"])
    entries=[]
    for (backbone,method,role),values in cells.items():
        record=dict(edge_model=backbone,method=method,implementation=role,cap_percent=25,calibration_draws=20)
        for site in SITES:
            for stage in STAGES:
                value=values.get((site,stage))
                record[f"{site}_{stage}_BA_pct"]="" if value is None else value[0]
                record[f"{site}_{stage}_active_draws"]="" if value is None else value[1]
        entries.append(record)
    return entries


if __name__=="__main__":
    entries=summarize()
    with (OUT/"methods_comparison.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=entries[0]);writer.writeheader();writer.writerows(entries)
    print(f"{len(entries)} matched operating-point rows.")

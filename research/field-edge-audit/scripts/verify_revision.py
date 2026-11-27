"""Verify reported aggregate values using only public summary CSV/JSON files.

Uses the Python standard library. No image, model, per-image evaluation
reference, feature matrix, network request, fitting routine or output file is used.
The aggregate checks cannot reconstruct per-image query identities. The batch
identity claim is checked against the separately released per-condition parity
counts. For online ranking we independently compare every available summary field.
"""
import argparse
import csv
import json
import math
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[1] / "results"
SITES = ("cornell","kashmir","latvia")
FILES = (
    "streaming_summary.csv","streaming_published_summary.csv","streaming_payload_audit.csv",
    "joint_budget_summary.csv","published_edgeboost_parity.csv","published_edgeboost_author_replay.csv",
    "methods_comparison.csv","esp_grid_anchor.json","esp_head_drift_summary.csv",
    "mahimahi_summary.csv","rural_measured_kpis.csv","rural_deadline_summary.csv",
    "second_backbone_fit.csv","second_backbone_feature_audit.csv","second_backbone_summary.csv",
    "second_backbone_intervals.csv","latvia_failure_groups.csv","latvia_failure_stratified_auc.csv",
    "latvia_failure_oracle.csv","latvia_failure_transport.csv","latvia_component_summary.csv",
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def close(value, want, tolerance=1e-8):
    require(math.isclose(float(value),want,rel_tol=0,abs_tol=tolerance), f"Observed {value}, expected {want} ± {tolerance}")


def find(rows, **filters):
    matches = [r for r in rows if all(r[k]==str(v) for k,v in filters.items())]
    require(len(matches)==1,f"Expected one row for {filters}, found {len(matches)}")
    return matches[0]


def percent(row, field, expected):
    # The manuscript reports most percentages to two decimals.
    close(100*float(row[field]),expected,.0050001)


def verify_stream(t):
    main=t["streaming_summary"]
    policies={
        (32,"updated_only"):(77.35,77.29,51.12), (32,"updated_fixed"):(82.06,80.09,53.35),
        (32,"updated_paced"):(83.03,80.44,53.16), (128,"updated_only"):(80.01,67.58,52.11),
        (128,"updated_fixed"):(83.51,73.93,52.80), (128,"updated_paced"):(84.52,73.32,53.54),
        (128,"updated_credit"):(78.16,67.58,45.35),
    }
    for (n,method),expected in policies.items():
        for site,ba in zip(SITES,expected):
            r=find(main,site=site,budget_labels=n,reviewer="remote",cap_percent=25,method=method)
            percent(r,"mean_ba",ba)
            require(float(r["mean_setup_bytes"])<=float(r["mean_total_bytes"])<=float(r["mean_cap_bytes"]),"Online budget violated")
            if method=="updated_paced":
                close(r["update_activation_fraction"],int(n==32 or site!="kashmir"))
    for site,cap,fraction in zip(SITES,(2273844,645348,1999907),(24.87,17.06,18.54)):
        r=find(main,site=site,budget_labels=128,reviewer="remote",cap_percent=25,method="updated_paced")
        close(r["mean_cap_bytes"],cap,.500001)
        percent(r,"mean_cap_fraction_of_full_target",fraction)
    close(find(main,site="latvia",budget_labels=128,reviewer="remote",cap_percent=25,method="updated_paced")["mean_fn"],496.4)
    payloads=t["streaming_payload_audit"]
    require(sum(int(r["verified_payloads"]) for r in payloads)==2927 and
            all(r["verified_payloads"]==r["sha256_and_lengths_matched"] for r in payloads),"Payload audit incomplete")
    require(max(int(r["all_images_single_frame_max_jpeg_bytes"]) for r in payloads)==13205,"Incorrect maximum JPEG size")
    learned=t["streaming_published_summary"]
    expected={"frozen_history":(72.33,73.13,50.26),"uncertainty_history":(84.47,73.13,53.26),
              "edgeboost_history":(84.47,73.13,53.26),"compare_confidence_history":(84.84,73.13,53.37),
              "ova_history":(85.00,73.13,53.57),"mao_history":(83.48,73.13,53.32),"edgeboost_fixed":(84.55,73.13,54.00)}
    for m,panel in expected.items():
        for site,ba in zip(SITES,panel):
            percent(find(learned,site=site,method=m,reviewer="remote",cap_percent=25),"mean_ba",ba)
    compared=0
    for r in learned:
        if r["method"]=="uncertainty_history":
            other=find(learned,site=r["site"],method="edgeboost_history",reviewer=r["reviewer"],cap_percent=r["cap_percent"])
            for k in r:
                if k!="method":
                    require(r[k]==other[k],f"Online parity summary differs at {k}")
            compared+=int(r["n_seeds"])*int(r["n_orders"])
    require(compared==1320,"Wrong number of online paired score comparisons")


def verify_batch(t):
    table=t["joint_budget_summary"]
    for site,l,r in zip(SITES,(87.06,84.28,54.74),(84.76,76.37,53.70)):
        for location,want in (("local",l),("remote",r)):
            row=find(table,site=site,cap_percent=25,review_location=location)
            percent(row,"joint_mean_ba",want)
            require(float(row["joint_mean_total_bytes"])<=int(row["cap_bytes"]),"Batch budget violated")
    require(int(find(table,site="kashmir",cap_percent=25,review_location="remote")["seeds_feasible"])==0,"Wrong fallback count")
    parity=t["published_edgeboost_parity"]
    keys={(r["site"],r["seed"],r["cap_percent"],r["review_location"]) for r in parity}
    require(len(keys)==len(parity)==360 and all(int(r["changed_queries"])==int(r["changed_decisions"])==0 for r in parity),
            "Batch EdgeBoost query or decision parity failed")
    cifar=t["published_edgeboost_author_replay"]
    require(len(cifar)==7 and all(int(r["count"])==8000 for r in cifar),"Original CIFAR gate audit failed")
    for r in cifar:
        close(100*int(r["queried"])/8000,float(r["upstream_upload_pct"]),.00501)
        close(100*int(r["correct"])/8000,float(r["upstream_acc_pct"]),.05001)
    methods=t["methods_comparison"]
    require(len(methods)==50,"Incomplete published comparator table")
    def port(method,role):
        return find(methods,edge_model="AgriTinyVision_MobileNetV2",method=method,implementation=role)
    ref=port("confidence_reference","adapted_reference")
    rank=port("edgeboost_budget_rank","binary_temperature_margin")
    for site in SITES:
        for stage in ("cold_local","cold_remote","predeployed"):
            key=f"{site}_{stage}_BA_pct"
            close(ref[key],float(rank[key]))
        require(float(port("ce_t100_bal","head_only_continuation")[site+"_cold_local_BA_pct"])>float(ref[site+"_cold_local_BA_pct"]),
                "Same-price CE control claimed ordering failed")
    for method,expected in (("ova_balanced",(87.43,84.98,55.70)),("compare_confidence_balanced",(87.54,85.11,55.22)),
                            ("mao_two_stage_balanced",(84.58,82.27,53.81))):
        row=port(method,"linear_router_128_expert_calibration_photos")
        for site,x in zip(SITES,expected):
            close(row[site+"_predeployed_BA_pct"],x,.005001)


def verify_device_transport(t):
    grid=t["esp_grid_anchor"]
    require((grid["board_n"],grid["board_hard_agreements"])==(60,59),"Board anchor counts differ")
    require(grid["embedded_fixture_expected_q"]==[58,-59] and grid["embedded_fixture_observed_q"]==[56,-62],"Fixture codes differ")
    close(grid["median_abs_score_error"],.000172,.0000005)
    close(grid["max_abs_score_error"],.204,.0005)
    for i,site in enumerate(SITES):
        for mode,ba in (("FP32_reference",(84.76,76.37,53.70)),("FP32_head_grid_backbone",(78.86,77.73,50.39)),
                        ("grid_head_grid_backbone",(81.60,77.73,53.26))):
            r=find(t["esp_head_drift_summary"],site=site,mode=mode,reviewer="remote")
            percent(r,"balanced_accuracy",ba[i])
            close(r["feature_cosine_median"],(.894,.908,.918)[i],.0005)
    measured=t["rural_measured_kpis"]
    require(len(measured)==9,"Incomplete forest KPI series")
    for k,bounds in (("up_mbps",(8,179)),("down_mbps",(133.4,624)),("rtt_ms",(12,27))):
        vals=[float(r[k]) for r in measured];close(min(vals),bounds[0]);close(max(vals),bounds[1])
    link=t["rural_deadline_summary"]
    snapshots=[r for r in link if r["series"]=="COMMECT_forestry_snapshot" and r["site"]=="cornell" and
               r["reviewer"]=="remote" and r["deadline_s"]=="2.0"]
    require(len({r["profile"] for r in snapshots})==9,"Forest snapshot coverage differs")
    for r in snapshots:
        close(r["requested"],166.4);close(r["timely"],166.4)
    for seconds,n,ba in ((2,0,80.01),(30,85,82.36)):
        r=find(link,site="cornell",reviewer="remote",profile="constrained_loss_outage",deadline_s=f"{seconds}.0",
               cold_connection="0",assumed_cloud_seconds="0.1")
        close(r["timely"],n);percent(r,"balanced_accuracy",ba)
        close(float(r["requested"])-float(r["timely"]),float(r["fallback_count"]))
    traces=t["mahimahi_summary"]
    for horizon,outcome in (("1",(600,80,520)),("1353",(1200,0,0))):
        cohort=[r for r in traces if r["horizon"]==horizon]
        counts=tuple(sum(int(r[k]) for r in cohort) for k in ("edge_faster","ties","dino_faster"))
        require(counts==outcome,f"Trace paired counts differ at horizon {horizon}")
        r=find(cohort,trace="ATT-LTE-driving",budget=128,labeling="remote_review")
        close(r["edge_p50_s"],8.294);close(r["dino_p50_s"],6.879 if horizon=="1" else 87.656)


def verify_v4(t):
    features=t["second_backbone_feature_audit"]
    require({r["site"]:int(r["images"]) for r in features}==dict(plantdoc=1237,cornell=1821,kashmir=384,latvia=1190),
            "Second-backbone sample counts differ")
    require(all(int(r["feature_dimension"])==1280 and float(r["max_imagenet_logit_reconstruction_error"])<6.9e-5
                for r in features),"Second-backbone logit reconstruction failed")
    fits=t["second_backbone_fit"]
    require(len(fits)==20 and len({r["sorted_source_ids_sha256"] for r in fits})==20,"Calibration identity audit differs")
    for r in fits:
        close(r["feature_bytes"],128*1293);close(r["head_bytes"],1294);close(r["local"],128*1293+1294)
        close(r["remote"],float(r["local"])+float(r["remote_jpeg_and_id_bytes"]))
    v4=t["second_backbone_summary"]
    for i,site in enumerate((*SITES,"latvia_later")):
        for method,vals in (("source_confidence",(80.53,83.59,66.58,68.11)),("target_confidence",(84.65,76.27,59.40,61.82)),
                            ("target_random",(81.52,74.38,58.98,61.04)),("ce_balanced_confidence",(84.89,77.91,59.82,62.19))):
            r=find(v4,site=site,method=method,review_location="local",cap_percent=25,accounting="cold")
            percent(r,"mean_balanced_accuracy",vals[i])
        r=find(t["second_backbone_intervals"],site=site,comparator="source_confidence",review_location="local")
        for key,expected in (("mean_difference_ba_pp",(4.11,-7.32,-7.18,-6.29)),("conditional_lower95_pp",(1.90,-11.62,-11.44,-10.70)),
                             ("conditional_upper95_pp",(6.37,-2.98,.37,1.63))):
            close(r[key],expected[i],.005001)


def verify_latvia(t):
    f=find(t["latvia_failure_groups"],site="latvia",group_type="all",group="all")
    for k,w in (("n",1190),("n_pos",983),("n_neg",207),("fn_original",118),("fn_updated",471.8),("fn_joint",536.35),
                ("fn_corrected_on_upload",25.50),("fn_introduced_on_upload",90.05),
                ("fp_corrected_on_upload",25.95),("fp_introduced_on_upload",1.50)):
        close(f[k],w)
    close(float(f["fn_updated"])-float(f["fn_corrected_on_upload"])+float(f["fn_introduced_on_upload"]),float(f["fn_joint"]))
    close(100*(float(f["fn_updated"])-float(f["fn_original"]))/(float(f["fn_joint"])-float(f["fn_original"])),84.6,.05)
    percent(f,"queried_edge_accuracy",53.66);percent(f,"queried_expert_accuracy",38.92)
    close(f["uncertainty_error_auc_on_positive"],.368,.0005)
    same=find(t["latvia_failure_stratified_auc"],grouping="date_and_device")
    close(same["two_class_strata"],11);close(same["updated_score_auc"],.512,.0005);close(same["expert_score_auc"],.666,.0005)
    percent(same,"pair_fraction",8.38)
    oracle=find(t["latvia_failure_oracle"],site="latvia",method="optimal_router_fixed_expert")
    percent(oracle,"balanced_accuracy",79.95)
    transport=t["latvia_failure_transport"]
    for method,auc,n,ba in (("jpeg224_center",.696,271.8,54.74),("jpeg448_center",.743,90.5,53.30),
                            ("original_center",.757,1,52.14)):
        full=find(transport,method=method,comparison="full_expert")
        close(full["expert_auc"],auc,.0005)
        limit=find(transport,method=method,comparison="same_byte_ceiling")
        close(limit["queries"],n);percent(limit,"balanced_accuracy",ba)
    close(find(transport,method="original_center",comparison="full_expert")["fn"],637)
    for e,s,fp,fn,b in (("cornell","plantdoc",23.8,267.8,57.06),("cornell","early_latvia",29.7,216.65,58.01),
                         ("early_latvia","plantdoc",37.35,201.4,54.20),("early_latvia","early_latvia",42.35,146.7,56.12)):
        r=find(t["latvia_component_summary"],edge=e,cloud=s)
        close(r["n_images"],579);close(r["seeds"],20);close(r["fp"],fp);close(r["fn"],fn)
        close(float(r["balanced_accuracy"]),1-.5*(fp/72+fn/507));percent(r,"balanced_accuracy",b)
        percent(r,"full_cloud_balanced_accuracy",63.09 if s=="plantdoc" else 68.11)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results",type=Path,default=DEFAULT,help="Directory containing published aggregate files")
    ap.add_argument("--list-inputs",action="store_true",help="Print the minimal aggregate-file inventory and exit")
    args=ap.parse_args()
    if args.list_inputs:
        print("\n".join(FILES));return
    tables={}
    for name in FILES:
        path=args.results/name
        with path.open(newline="") as f:
            tables[path.stem]=json.load(f) if path.suffix==".json" else list(csv.DictReader(f))
    for title,check in (("one-frame operation, nominal quota and online score parity",verify_stream),
                        ("charged/prepaid methods, byte ceilings and batch EdgeBoost parity",verify_batch),
                        ("ESP approximation, measured and synthetic links, trace service times",verify_device_transport),
                        ("second backbone, packed setup prices and conditional intervals",verify_v4),
                        ("Latvian missed cases, routing, transport and local-head interventions",verify_latvia)):
        check(tables);print("PASS",title)
    print(f"Verified five groups of manuscript claims from {len(FILES)} public aggregate files.")


if __name__=="__main__":
    main()

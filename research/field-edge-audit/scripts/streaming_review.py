"""Single-pass selective review with declared byte grants and causal state."""

import argparse
import csv
import hashlib
import struct
import zipfile
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold

from edge_adaptation_wire import normalized, predict_from_head, quantized
from evaluate_upload import rank_selection
from published_defer import aligned, binary_head, count_outcome, opaque_tiebreak, table, train_weights, write_csv

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FEATURES = ROOT / "data/features"
GATE = struct.Struct("<4sff")
BUDGETS = (32, 128)
CAPS = (10, 25, 50)
SITES = ("cornell", "kashmir", "latvia")
SEEDS = range(20)
METHODS = ("frozen_only", "updated_only", "frozen_batch", "updated_batch", "frozen_fixed", "updated_fixed",
           "frozen_paced", "updated_paced", "frozen_credit", "updated_credit")


def urgency(score):
    return 2. * expit(-np.abs(score))


def original_urgency(p):
    return 1. - np.abs(p.astype(np.float64) - float(np.float32(.30))) / .70


def calibration_inputs():
    with np.load(RESULTS / "edge_adaptation_predictions.npz", allow_pickle=False) as f:
        train_paths = f["train_paths"]
    with np.load(FEATURES / "edge_fgvc.npz", allow_pickle=False) as f:
        feature_paths = f["source_path"]
        activations = f["activation"]
    records = {r["source_path"]:r for r in table("fgvc_edge.csv")}
    jp = {r["source_path"]:int(r["jpeg_bytes"]) + 20 for r in table("fgvc_backend.csv")
          if r["long_edge"]=="224" and r["jpeg_quality"]=="75"}
    charges = {(int(r["seed"]),int(r["budget"])):r for r in table("edge_adaptation_metrics.csv")
               if r["method"]=="local_int8" and int(r["budget"]) in BUDGETS}
    results = {}
    with zipfile.ZipFile(RESULTS / "edge_adaptation_heads.zip") as heads:
        for seed, ids in enumerate(train_paths):
            for b in BUDGETS:
                paths = ids[:b]
                q, scales = quantized(activations[aligned(feature_paths,paths)])
                x = normalized(q.astype(float)*scales[:,None])
                targets = np.array([int(records[p]["true_diseased"]) for p in paths])
                full = binary_head(x,targets,train_weights(targets,"balanced"))
                if full != heads.read(f"s{seed:02d}-b{b:03d}/local_int8.bin"):
                    raise ValueError("Local refit does not reproduce the deployed head")
                folds = min(4,int(np.bincount(targets,minlength=2).min()))
                if folds < 2:
                    raise ValueError("Insufficient labels for honest gate calibration")
                oof = np.empty(len(paths))
                for tr,va in StratifiedKFold(folds,shuffle=True,random_state=20261127+seed).split(x,targets):
                    fitted = binary_head(x[tr], targets[tr], train_weights(targets[tr],"balanced"))
                    oof[va] = predict_from_head("local_int8",fitted,q[va])
                sizes = np.array([jp[p] for p in paths])
                edge = np.array([float(records[p]["p_suspect"]) for p in paths],np.float32)
                r = charges[seed,b]
                results[seed,b] = dict(costs=sizes, numerator=int(sizes[:32].sum()),
                                       frozen=original_urgency(edge), updated=urgency(oof),
                                       local=int(r["local_label_total_bytes"]), remote=int(r["remote_label_total_bytes"]))
    return results


def identities(site, paths):
    if site=="cornell":
        data = {r["source_path"]:r["file_sha256"] for r in table("fgvc_edge.csv")}
        timestamp = None
    elif site=="kashmir":
        data = {r["source_path"]:r["source_sha256"] for r in table("transfer_manifest.csv")}
        timestamp = None
    else:
        with (ROOT / "data/latvia_source/manifest.csv").open(newline="") as f:
            meta = {r["source_path"]:r for r in csv.DictReader(f)}
        data = {p:r["source_sha256"] for p,r in meta.items()}
        timestamp = np.array([meta[p]["exif_datetime_original"] for p in paths])
        if len(timestamp)!=1190 or np.any(timestamp==""):
            raise ValueError("Incomplete original chronology")
    digest = [bytes.fromhex(data[p]) for p in paths]
    orders = []
    for salt in range(1 if timestamp is not None else 5):
        key = f"streaming-order-v1/{salt}/".encode()
        tie = np.array([hashlib.sha256(key+h).hexdigest()+hashlib.sha256(key+str(p).encode()).hexdigest()
                        for p,h in zip(paths,digest,strict=True)])
        orders.append(np.argsort(tie) if timestamp is None else np.lexsort((tie,timestamp)))
    return np.asarray(orders)


def evaluation_inputs(site):
    with np.load(RESULTS/f"joint_budget_{site}.npz") as base:
        ids,y,costs,origin,server = (base[k] for k in ("source_path","y","application_bytes","original","expert"))
    if site=="cornell":
        with np.load(RESULTS/"edge_adaptation_predictions.npz") as z:
            method = list(z["methods"]).index("local_int8")
            order = aligned(z["test_paths"],ids)
            scores = np.asarray([z["score"][:,list(z["budgets"]).index(b),method,:][:,order] for b in BUDGETS])
    else:
        with np.load(FEATURES/f"{site}.npz") as z:
            q,_ = quantized(z["edge"][aligned(z["source_path"],ids)])
        with zipfile.ZipFile(RESULTS/"edge_adaptation_heads.zip") as archive:
            scores = np.asarray([[predict_from_head("local_int8",archive.read(f"s{s:02d}-b{b:03d}/local_int8.bin"),q)
                                 for s in SEEDS] for b in BUDGETS])
    if site=="latvia":
        with np.load(RESULTS/"latvia_decisions.npz") as z:
            p = z["edge_probability"][aligned(z["source_path"],ids)]
    else:
        by_path = {r["path"]:float(r["edge_p"]) for r in table("directional_predictions.csv") if r["cohort"]==site}
        p = np.array([by_path[path] for path in ids],np.float32)
    if not np.array_equal(p>=np.float32(.3),origin):
        raise ValueError("Released original decision differs")
    orders = identities(site,ids)
    return ids,y,server,costs,origin,scores,original_urgency(p),orders,opaque_tiebreak(site,ids)


def cutoff(cal_scores, lengths, expected_bytes):
    order = np.argsort(-cal_scores,kind="stable")
    running = np.cumsum(lengths[order]) / len(lengths)
    thresholds = np.r_[np.nextafter(np.float32(1),np.float32(np.inf)),cal_scores[order]]
    # A threshold cannot split identical urgencies.
    valid = np.r_[True,cal_scores[order][1:]!=cal_scores[order][:-1],True]
    means = np.r_[0.,running]
    keep = np.flatnonzero(valid)
    nearest = keep[int(np.argmin(np.abs(means[keep]-max(0.,expected_bytes))))]
    return float(np.float32(thresholds[nearest]))


def scan(urgencies, sizes, mode, setup, initial_threshold, calibration_mean,
         rate_numerator, planned_n=None, grant=None):
    """Time-ordered iterator inputs. The credit policy has neither horizon nor grant."""
    rate = rate_numerator / 3200.
    initial = mode != "credit"
    if initial and (planned_n is None or grant is None or setup>grant):
        raise ValueError("Unaffordable installation")
    threshold = initial_threshold
    expected_cost = calibration_mean
    balance = (grant-setup) if initial else 0
    installed_at = 1 if initial else 0
    spent = setup if initial else 0
    maximum_payload = 0
    selected = []
    last_earned = 0
    for t,(u,cost) in enumerate(zip(urgencies,sizes,strict=True),start=1):
        if mode=="credit":
            cumulative_credit = (t*rate_numerator)//3200
            balance += cumulative_credit-last_earned
            last_earned = cumulative_credit
            if not installed_at:
                if balance<setup:
                    selected.append(False)
                    continue
                balance -= setup
                spent += setup
                installed_at=t
            balance=min(balance,4*calibration_mean)
            fraction=min(1.,rate/max(1.,expected_cost))
        else:
            fraction=min(1.,balance/(max(1,planned_n-t+1)*max(1.,expected_cost)))
        passes = u>=threshold
        paid = False
        if passes and balance>=20:
            maximum_payload=max(maximum_payload,cost-20)
            if cost<=balance:
                balance-=cost
                spent+=cost
                paid=True
            expected_cost+=.05*(cost-expected_cost)
        selected.append(paid)
        if mode!="fixed":
            threshold=max(0.,min(1.,threshold+.05*(float(passes)-fraction)))
        if mode=="credit" and spent>last_earned:
            raise ValueError("Prefix rate violated")
    return np.asarray(selected,bool),installed_at,spent,maximum_payload


def grant_args(cal, fraction, n):
    numerator = fraction * cal["numerator"]
    return numerator,(n*numerator)//3200


def assess_site(site, arrays, encoded, output):
    ids,y,server,costs,original,margin,frozen_priority,orders,batch_tie = arrays
    n=len(ids)
    results=[]
    masks=[]
    predictions=[]
    for bi,b in enumerate(BUDGETS):
        for s in SEEDS:
            cal=encoded[s,b]
            mean=float(np.float32(cal["numerator"]/32))
            updated=margin[bi,s]>=0
            updated_priority=urgency(margin[bi,s])
            for reviewer in ("local","remote"):
                for pct in CAPS:
                    numerator,cap=grant_args(cal,pct,n)
                    rate=numerator/3200
                    for method in METHODS:
                        category,mode=method.split("_")
                        wants_update=category=="updated"
                        gate=0 if mode in ("only","batch") else GATE.size
                        desired=(cal[reviewer] if wants_update else 0)+gate
                        # In the no-horizon arm, eventual affordability is unknown.
                        uses_update=wants_update and (mode=="credit" or desired<=cap)
                        setup=(cal[reviewer] if uses_update else 0)+gate
                        priority=updated_priority if uses_update else frozen_priority
                        local=updated if uses_update else original
                        cal_priorities=cal["updated" if uses_update else "frozen"]
                        target_bytes=rate if mode=="credit" else (cap-setup)/n
                        gate_cost=cutoff(cal_priorities,cal["costs"],target_bytes)
                        if mode=="batch":
                            # Tie priority from a content-hash permutation, never a label-bearing path.
                            sorted_indices=np.lexsort((batch_tie,-priority))
                            batch_mask=rank_selection(sorted_indices,costs,cap-setup)
                        for sequence,order in enumerate(orders):
                            if mode=="only":
                                selected=np.zeros(n,bool)
                                final=local
                                install,spent,peak,peak_frames=1,setup,0,0
                            elif mode=="batch":
                                selected=batch_mask
                                final=np.where(selected,server,local)
                                install=1
                                spent=setup+int(costs[selected].sum())
                                peak=int(costs.sum())-20*n
                                peak_frames=n
                            else:
                                scores_in_order=priority[order].tolist()
                                sizes_in_order=costs[order].tolist()
                                fixed=(None,None) if mode=="credit" else (n,cap)
                                q,install,spent,peak=scan(scores_in_order,sizes_in_order,mode,setup,
                                                         gate_cost,mean,numerator,*fixed)
                                selected=np.zeros(n,bool)
                                selected[order]=q
                                active=local.copy()
                                if mode=="credit" and uses_update:
                                    pre=order[:install-1] if install else order
                                    active[pre]=original[pre]
                                final=np.where(selected,server,active)
                                peak_frames=int(peak>0)
                            charged=setup if install else 0
                            if spent != charged+int(costs[selected].sum()) or spent>cap:
                                raise ValueError("Causal replay failed byte accounting")
                            first_ordinal=(install if uses_update else 0)
                            row=dict(site=site,budget_labels=b,seed=s,reviewer=reviewer,cap_percent=pct,
                                     source_bytes_per_frame=rate,target_full_bytes=int(costs.sum()),cap_bytes=cap,
                                     sequence=sequence,method=method,update_installed=int(uses_update and bool(install)),
                                     update_start_ordinal=first_ordinal,gate_start_ordinal=install,desired_setup_bytes=desired,
                                     actual_setup_bytes=charged,n=n,queried=int(selected.sum()),total_bytes=spent,
                                     remaining_bytes=cap-spent,peak_buffered_jpeg_bytes=int(peak),peak_frames=peak_frames,
                                     gate_threshold=gate_cost if mode not in ("batch","only") else "",row_index=len(results),
                                     **count_outcome(y,final.astype(bool)))
                            results.append(row)
                            masks.append(np.packbits(selected))
                            predictions.append(np.packbits(final))
    np.savez_compressed(output/f"streaming_decisions_{site}.npz",source_path=ids,y=y,orders=orders,
                        predictions=predictions,queries=masks)
    return results


def summarize(all_rows):
    keys=("site","budget_labels","reviewer","cap_percent","method")
    grouped={}
    for r in all_rows:
        grouped.setdefault(tuple(r[k] for k in keys),[]).append(r)
    results=[]
    for key,rows in grouped.items():
        seq_means=[]
        for j in sorted({r["sequence"] for r in rows}):
            seq_means.append(np.mean([r["balanced_accuracy"] for r in rows if r["sequence"]==j]))
        vals=lambda k:np.array([r[k] for r in rows])
        nseed=len({r["seed"] for r in rows})
        results.append(dict(**dict(zip(keys,key)),n_seeds=nseed,n_orders=len(seq_means),
                            mean_ba=float(vals("balanced_accuracy").mean()),min_order_mean_ba=min(seq_means),max_order_mean_ba=max(seq_means),
                            mean_accuracy=float(vals("accuracy").mean()),mean_fp=float(vals("fp").mean()),mean_fn=float(vals("fn").mean()),
                            mean_queries=float(vals("queried").mean()),mean_cap_bytes=float(vals("cap_bytes").mean()),
                            mean_total_bytes=float(vals("total_bytes").mean()),mean_setup_bytes=float(vals("actual_setup_bytes").mean()),
                            mean_spent_fraction_of_cap=float((vals("total_bytes")/vals("cap_bytes")).mean()),
                            mean_cap_fraction_of_full_target=float((vals("cap_bytes")/vals("target_full_bytes")).mean()),
                            update_activation_fraction=float(vals("update_installed").mean()),
                            mean_first_updated_frame_among_installed=float(vals("update_start_ordinal")[vals("update_installed")==1].mean())
                            if vals("update_installed").any() else "",
                            max_buffered_frames=int(vals("peak_frames").max()),max_buffered_jpeg_bytes=int(vals("peak_buffered_jpeg_bytes").max())))
    return results


def run(out, replay):
    out.mkdir(parents=True,exist_ok=True)
    if replay is None:
        cal=calibration_inputs()
        contents={}
        for (s,b),fields in cal.items():
            for key,value in fields.items():
                contents[f"s{s}_b{b}_{key}"]=value
        np.savez_compressed(out/"streaming_calibration.npz",**contents)
    else:
        cal={}
        with np.load(replay/"streaming_calibration.npz",allow_pickle=False) as archive:
            for s in SEEDS:
                for b in BUDGETS:
                    fields={}
                    for k in ("frozen","updated","costs","numerator","local","remote"):
                        a=archive[f"s{s}_b{b}_{k}"]
                        fields[k]=int(a) if a.shape==() else a
                    cal[s,b]=fields
    all_rows=[]
    for site in SITES:
        names=("paths","y","server","costs","original","margins","urgencies","orders","batch_tie")
        if replay is None:
            arrays=evaluation_inputs(site)
            np.savez_compressed(out/f"streaming_inputs_{site}.npz",**dict(zip(names,arrays,strict=True)))
        else:
            with np.load(replay/f"streaming_inputs_{site}.npz",allow_pickle=False) as archive:
                arrays=tuple(archive[k] for k in names)
        rs=assess_site(site,arrays,cal,out)
        all_rows.extend(rs)
        print(f"{site}: finished {len(rs)} byte-capped single-pass, component and batch replay rows",flush=True)
    write_csv(out/"streaming_runs.csv",all_rows)
    write_csv(out/"streaming_summary.csv",summarize(all_rows))


def verify_payloads(out):
    """Re-read Cornell/Kashmir bytes and re-encode Latvia to validate actual payload sizes."""
    from io import BytesIO
    from contextlib import nullcontext
    from PIL import Image, ImageOps
    from server_backend import jpeg_transport

    report=[]
    for site in SITES:
        with np.load(RESULTS/f"streaming_inputs_{site}.npz") as inp:
            paths=inp["paths"]
            expected=inp["costs"]-20
        if site=="cornell":
            hashes={r["source_path"]:r["jpeg_sha256"] for r in table("fgvc_backend.csv")
                    if r["long_edge"]=="224" and r["jpeg_quality"]=="75"}
            folder=ROOT/"data/fgvc_backend_jpegs"
        elif site=="kashmir":
            hashes={r["source_path"]:r["jpeg_sha256"] for r in table("transfer_manifest.csv")}
            folder=ROOT/"data/raw/transfer_jpegs"
        else:
            with np.load(FEATURES/"latvia.npz") as f:
                hashes=dict(zip(f["source_path"],f["jpeg_sha256"],strict=True))
        context=zipfile.ZipFile(ROOT/"data/raw/latvia.zip") if site=="latvia" else nullcontext(None)
        observed=[]
        with context as archive:
            for path,sz in zip(paths,expected,strict=True):
                if site=="latvia":
                    with Image.open(BytesIO(archive.read(path))) as image:
                        payload=jpeg_transport(ImageOps.exif_transpose(image).convert("RGB"),224)
                else:
                    payload=(folder/(str(hashes[path])+".jpg")).read_bytes()
                if len(payload)!=sz or hashlib.sha256(payload).hexdigest()!=hashes[path]:
                    raise ValueError(f"Payload mismatch in {site}: {path}")
                observed.append(len(payload))
                del payload
        row=dict(site=site,verified_payloads=len(observed),sha256_and_lengths_matched=len(observed),
                 all_images_single_frame_max_jpeg_bytes=max(observed),
                 all_images_batch_jpeg_bytes=sum(observed))
        report.append(row)
        print(row,flush=True)
    write_csv(out/"streaming_payload_audit.csv",report)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=RESULTS)
    parser.add_argument("--replay",type=Path,help="Use the released small scalar score and ordering archives.")
    parser.add_argument("--verify-payloads",action="store_true",help="Revalidate actual JPEG bytes using the original local data caches.")
    args=parser.parse_args()
    if args.verify_payloads:
        verify_payloads(args.out)
    else:
        run(args.out,args.replay)

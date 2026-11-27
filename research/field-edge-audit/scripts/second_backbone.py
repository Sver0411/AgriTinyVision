"""Train and audit the second edge backbone with the original spending constraints."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from directional_policy import REQUEST, REPLY
from edge_adaptation import fit
from edge_adaptation_wire import normalized, pack_head, predict_from_head, quantized, training_input, training_record
from evaluate_upload import rank_selection
from published_cascade import csv_write
from published_gatekeeper import decode_head, optimize

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CACHE = ROOT / "data/raw/second_backbone"
HEADS = ("target", "gatekeeper", "ce_balanced")
POLICIES = ("source_only", "source_confidence", "target_only", "target_confidence", "gatekeeper_confidence",
            "ce_balanced_confidence", "target_random")
PROTOCOL_SHA256 = "cf4f3ee50af0f60cbac1dffd58a6fde70581c1c21ea42cd9ce806814e693c139"
SITES = ("cornell", "kashmir", "latvia", "latvia_later")


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def read_cache(site):
    with np.load(CACHE / (site + ".npz")) as z:
        return {key: z[key] for key in z}


def measure(y, pred):
    fp = int(np.sum((y == 0) & pred))
    fn = int(np.sum((y == 1) & ~pred))
    n0 = int(np.sum(y == 0))
    n1 = len(y) - n0
    tpr, tnr = 1-fn/n1, 1-fp/n0
    return dict(fp=fp, fn=fn, sensitivity=tpr, specificity=tnr,
                accuracy=1-(fp+fn)/len(y), balanced_accuracy=(tnr+tpr)/2)


def encoded_head(x, y):
    return pack_head("local_int8", *fit(x, y))


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train(out):
    with np.load(RESULTS / "edge_adaptation_predictions.npz") as z:
        train_paths, test_paths = z["train_paths"], z["test_paths"]
    assert train_paths.shape == (20, 128) and not (set(train_paths.ravel()) & set(test_paths))
    plant = read_cache("plantdoc")
    with np.load(ROOT / "data/features/dino_plantdoc_balanced_head.npz") as saved:
        assert set(saved["fit_keys"]) == set(plant["source_path"])
    q, _ = quantized(plant["features"])
    source = encoded_head(normalized(q), plant["y"])
    heads = {"source": source}
    cornell = read_cache("cornell")
    mapping = {p: i for i, p in enumerate(cornell["source_path"])}
    with np.load(ROOT / "data/features/dino_fgvc.npz") as saved:
        jpeg = dict(zip(saved["source_path"], saved["jpeg_bytes"], strict=True))
    fit_records = []
    for seed, paths in enumerate(train_paths):
        ix = [mapping[p] for p in paths]
        y = cornell["y"][ix]
        x = cornell["features"][ix]
        packets = [training_record("local_int8", i, label, vec) for i, label, vec in zip(ix, y, x, strict=True)]
        x_wire = normalized([training_input("local_int8", packet, i, label, x.shape[1])
                             for i, label, packet in zip(ix, y, packets, strict=True)])
        initial = encoded_head(x_wire, y)
        heads[seed, "target"] = initial
        design = np.column_stack((x_wire, np.ones(len(y))))
        for name, alpha, balanced in (("gatekeeper", .5, False), ("ce_balanced", None, True)):
            heads[seed, name] = optimize(design, y, decode_head(initial), alpha, 100, balanced)
        assert all(len(heads[seed, name]) == len(initial) for name in HEADS)
        local = sum(map(len, packets)) + len(initial)
        remote_add = sum(int(jpeg[p])+8 for p in paths)
        fit_records.append(dict(seed=seed, labels=len(paths), diseased_labels=int(y.sum()),
                                sorted_source_ids_sha256=hashlib.sha256(("\n".join(sorted(paths))+"\n").encode()).hexdigest(),
                                feature_bytes=sum(map(len, packets)), head_bytes=len(initial), local=local,
                                remote_jpeg_and_id_bytes=remote_add, remote=local+remote_add))
    with ZipFile(out / "second_backbone_heads.zip", "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("source.bin", source)
        for (seed, name), blob in [(key, val) for key, val in heads.items() if key != "source"]:
            archive.writestr(f"s{seed:02d}/{name}.bin", blob)
    csv_write(out / "second_backbone_fit.csv", fit_records)
    return heads, fit_records


def prepare(out, heads):
    with np.load(RESULTS / "published_cascade_deployment_margins.npz") as z:
        ref = {k: z[k] for k in z if not k.endswith(("_local_margin", "_edge_p"))}
    with np.load(RESULTS / "edge_adaptation_predictions.npz") as z:
        assert set(z["test_paths"]) == set(ref["cornell_paths"])
    later = {r["source_path"] for r in read_csv(RESULTS / "latvia_local_date_split.csv") if r["status"] == "test"}
    assert len(later) == 579
    source_records, source_images, origins = [], [], []
    for site in ("plantdoc", "cornell", "kashmir", "latvia"):
        data = read_cache(site)
        paths, y, features = (data[key] for key in ("source_path", "y", "features"))
        q, _ = quantized(features)
        source = predict_from_head("local_int8", heads["source"], q)
        source_records.append(dict(site=site, cohort="source_fit" if site=="plantdoc" else "full_collection",
                                   n=len(y), diseased=int(y.sum()), **measure(y, source >= 0)))
        origins.append(dict(site=site, images=len(y), feature_dimension=features.shape[1],
                            cache_sha256=file_digest(CACHE/(site+".npz")), model_sha256=str(data["model_sha256"]), max_imagenet_logit_reconstruction_error=float(data["max_reconstruction_error"])))
        source_images.extend(dict(site=site, source_path=p, y=int(t), source_sha256=h, source_margin=m)
                             for p,t,h,m in zip(paths,y,data["source_sha256"],source,strict=True))
        if site == "plantdoc":
            continue
        by_path = {p: i for i, p in enumerate(paths)}
        eval_paths, eval_y, expert, jpeg = (ref[f"{site}_{key}"] for key in ("paths","y","expert","jpeg"))
        idx = [by_path[p] for p in eval_paths]
        assert np.array_equal(y[idx], eval_y)
        margin = np.array([[predict_from_head("local_int8", heads[seed,name], q[idx]) for name in HEADS] for seed in range(20)])
        costs = jpeg.astype(np.int64) + REQUEST.size + REPLY.size
        for label, keep in ((site,np.ones(len(idx),dtype=bool)), ("latvia_later",np.isin(eval_paths,list(later)))):
            if label == "latvia_later" and site != "latvia":
                continue
            if label=="latvia_later":
                assert keep.sum()==len(later)
            np.savez_compressed(out / f"second_backbone_{label}.npz", paths=eval_paths[keep], y=eval_y[keep],
                                source_margin=source[idx][keep], margins=margin[:,:,keep], expert=expert[keep],
                                application_bytes=costs[keep], head_order=HEADS,
                                model_sha256=[hashlib.sha256(heads[seed,name]).hexdigest() for seed in range(20) for name in HEADS])
            for group, pred in (("evaluation_source",source[idx][keep]>=0), ("evaluation_full_remote_reference",expert[keep])):
                source_records.append(dict(site=label,cohort=group,n=int(keep.sum()),diseased=int(eval_y[keep].sum()),**measure(eval_y[keep],pred)))
    csv_write(out / "second_backbone_reference.csv", source_records)
    csv_write(out / "second_backbone_source_images.csv", source_images)
    csv_write(out / "second_backbone_feature_audit.csv", origins)


def run_rows(out):
    setup = read_csv(out / "second_backbone_fit.csv")
    for site in SITES:
        with np.load(out / f"second_backbone_{site}.npz") as z:
            source, margins, costs, paths, expert, y = (z[k] for k in ("source_margin","margins","application_bytes","paths","expert","y"))
            assert list(z["head_order"]) == list(HEADS)
        allbytes = int(costs.sum())
        canonical = np.argsort(paths)
        fixed_random = [np.random.default_rng(21000+j).permutation(canonical) for j in range(20)]
        source_order = np.lexsort((paths,np.abs(source)))
        no_queries = np.zeros(len(y),dtype=bool)
        for seed, fit_row in enumerate(setup):
            assert int(fit_row["seed"]) == seed
            by_head = dict(source=source, **dict(zip(HEADS,margins[seed],strict=True)))
            orders = {name:np.lexsort((paths,np.abs(score))) for name,score in by_head.items()}
            for pct in (10,25,50):
                cap = (allbytes*pct)//100
                cache = {}
                for loc in ("local","remote"):
                    for stage in ("cold","predeployed"):
                        for method in POLICIES:
                            wants_update = not method.startswith("source_")
                            price = int(fit_row[loc]) if wants_update and stage == "cold" else 0
                            active = wants_update and price<=cap
                            paid = price if active else 0
                            name = "source"
                            if active:
                                name = {"gatekeeper_confidence":"gatekeeper", "ce_balanced_confidence":"ce_balanced"}.get(method,"target")
                            edge = by_head[name]>=0
                            route = method.split("_")[-1]
                            for draw in (range(20) if route=="random" else [-1]):
                                order = fixed_random[draw] if route=="random" else orders[name]
                                key = (cap-paid,route,"any" if route=="random" else name,draw)
                                if key not in cache:
                                    cache[key] = no_queries if route=="only" else rank_selection(order,costs,cap-paid)
                                ask = cache[key]
                                pred = np.where(ask,expert,edge)
                                corrected = int(np.sum(ask & (edge!=y) & (expert==y)))
                                spoiled = int(np.sum(ask & (edge==y) & (expert!=y)))
                                total = paid + int(costs[ask].sum())
                                assert total<=cap
                                yield dict(site=site,seed=seed,cap_percent=pct,review_location=loc,accounting=stage,
                                           method=method,random_order=draw,head=name,queries=int(ask.sum()),cap_bytes=cap,
                                           setup_bytes=paid,total_bytes=total,corrected=corrected,spoiled=spoiled,**measure(y,pred))


def summarize(rows):
    groups = defaultdict(list)
    keys = ("site","cap_percent","review_location","accounting","method")
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    summary = []
    for key, group in groups.items():
        d = dict(zip(keys,key,strict=True))
        means = ("queries","setup_bytes","total_bytes","corrected","spoiled","fp","fn","sensitivity","specificity","accuracy","balanced_accuracy")
        d.update({f"mean_{metric}":float(np.mean([float(r[metric]) for r in group])) for metric in means})
        d.update(calibration_draws=len(set(r["seed"] for r in group)),realizations=len(group),
                 active_head_draws=len({r["seed"] for r in group if r["head"]!="source"}))
        summary.append(d)
    return summary


def paired_summaries(rows):
    """Keep calibration draws paired; first average the 20 random orders within draw."""
    dimensions = ("site", "cap_percent", "review_location", "accounting")
    grouped = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = tuple(r[k] for k in dimensions)
        grouped[key][r["seed"],r["method"]].append(r["balanced_accuracy"])
    results = []
    for fields, cells in grouped.items():
        levels = {m:np.array([np.mean(cells[s,m]) for s in range(20)]) for m in POLICIES}
        comparators = {m:levels[m] for m in POLICIES if m!="target_confidence"}
        comparators["better_single_action"] = np.maximum(levels["target_only"],levels["source_confidence"])
        for comparator, scores in comparators.items():
            delta = 100*(levels["target_confidence"]-scores)
            results.append(dict(zip(dimensions,fields),comparator=comparator,
                                mean_target_confidence_minus_comparator_pp=float(delta.mean()),
                                min_delta_pp=float(delta.min()),max_delta_pp=float(delta.max()),
                                positive_draws=int((delta>1e-10).sum()),tied_draws=int((abs(delta)<=1e-10).sum()),
                                negative_draws=int((delta < -1e-10).sum())))
    return results


def verify(out):
    setup = read_csv(out / "second_backbone_fit.csv")
    source_rows = read_csv(out / "second_backbone_source_images.csv")
    assert len(source_rows) == 4632
    source_ids = {r["source_path"] for r in source_rows if r["site"]=="plantdoc"}
    with np.load(ROOT/"data/features/dino_plantdoc_balanced_head.npz") as ref:
        assert source_ids==set(ref["fit_keys"]) and all(p.startswith("train/") for p in source_ids)
    for label, filename in (("plantdoc","dino_plantdoc"),("cornell","dino_fgvc")):
        with np.load(ROOT/f"data/features/{filename}.npz") as z:
            hashes=dict(zip(z["source_path"],z["source_sha256"],strict=True))
            assert all(hashes[r["source_path"]]==r["source_sha256"] for r in source_rows if r["site"]==label)
    for label, path in (("latvia",ROOT/"data/latvia_source/manifest.csv"),("kashmir",RESULTS/"transfer_manifest.csv")):
        ids={r["source_path"]:(r["source_sha256"],r["y"]) for r in read_csv(path)}
        assert {(r["source_path"],r["source_sha256"],r["y"]) for r in source_rows if r["site"]==label}=={(p,*value) for p,value in ids.items()}
    with np.load(RESULTS/"edge_adaptation_predictions.npz") as ref:
        for row, paths in zip(setup,ref["train_paths"],strict=True):
            digest=hashlib.sha256(("\n".join(sorted(paths))+"\n").encode()).hexdigest()
            assert digest==row["sorted_source_ids_sha256"] and not (set(paths)&set(ref["test_paths"]))
    with ZipFile(out / "second_backbone_heads.zip") as heads:
        assert len(heads.namelist())==61
        for site in SITES:
            with np.load(out / f"second_backbone_{site}.npz") as z:
                expected=[hashlib.sha256(heads.read(f"s{s:02d}/{m}.bin")).hexdigest() for s in range(20) for m in HEADS]
                np.testing.assert_array_equal(z["model_sha256"], expected)
                with np.load(RESULTS / "published_cascade_deployment_margins.npz") as ref:
                    origin=site.replace("_later","")
                    index={p:i for i,p in enumerate(ref[origin+"_paths"])}
                    idx=[index[p] for p in z["paths"]]
                    for a,b in (("expert","expert"),("y","y")):
                        np.testing.assert_array_equal(z[a],ref[origin+"_"+b][idx])
                    np.testing.assert_array_equal(z["application_bytes"],ref[origin+"_jpeg"][idx]+20)
        for r in setup:
            assert int(r["head_bytes"])==len(heads.read("source.bin"))==1294 and int(r["feature_bytes"])==128*1293
    assert file_digest(ROOT/"protocol/SECOND_BACKBONE.md")==PROTOCOL_SHA256
    gold=read_csv(out / "second_backbone_runs.csv")
    count=0
    replay=list(run_rows(out))
    for actual,expected in zip(replay,gold,strict=True):
        assert all(str(v)==expected[k] for k,v in actual.items())
        count+=1
    for metric, table in (("summary",summarize(replay)),("paired",paired_summaries(replay))):
        saved=read_csv(out/f"second_backbone_{metric}.csv")
        for actual,expected in zip(table,saved,strict=True):
            assert all(str(v)==expected[k] for k,v in actual.items())
    assert count==24960
    print(f"Replayed {count} per-budget classification results across all four panels and 61 serialized heads.")
    return count


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=RESULTS)
    parser.add_argument("--verify-only",action="store_true")
    args=parser.parse_args()
    out=args.output
    if args.verify_only:
        verify(out)
        return
    out.mkdir(exist_ok=True,parents=True)
    heads,_=train(out)
    prepare(out,heads)
    rows=list(run_rows(out))
    csv_write(out/"second_backbone_runs.csv",rows)
    csv_write(out/"second_backbone_summary.csv",summarize(rows))
    csv_write(out/"second_backbone_paired.csv",paired_summaries(rows))
    n=verify(out)
    metadata=dict(model="MobileNetV4-Conv-Small e2400 r224 FP32",scope="CPU graph plus protocol-byte replay",
                  protocol_sha256=file_digest(ROOT/"protocol/SECOND_BACKBONE.md"),
                  heads=61,run_rows=n,source_n=1237,calibration_images_per_draw=128,calibration_draws=20,
                  confidence_cutoff=.5,expert_jpeg_long_edge=224,expert_jpeg_quality=75)
    (out/"second_backbone_meta.json").write_text(json.dumps(metadata,indent=2)+"\n")


if __name__=="__main__":
    main()

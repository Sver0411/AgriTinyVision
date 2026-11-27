"""Evaluate the frozen AgriTinyVision host model on every official PlantDoc test image.

The held-out cohort excludes both development-set overlap and within-test duplicate
components. Hardware outputs are joined only for the 60 archived physical records.
"""
import csv
from collections import Counter
import hashlib
import json
from io import BytesIO
from pathlib import Path
import platform
import statistics
import subprocess
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime
from PIL import Image, ImageOps, __version__ as pillow_version
from scipy.fft import dctn
from edge_model import EdgeModel, MODEL_SHA256, ROOT, read_rgb


RAW = ROOT / "data/raw"
AGRI = RAW / "agritinyvision"
DOC = RAW / "plantdoc"
PV = RAW / "plantvillage/raw/color"
RESULTS = ROOT / "results"
REVISIONS = {"agritinyvision": "16e7a2fdec43c02258f22e85b085ac269345e124",
             "plantdoc": "5467f6012d78d1c446145d5f582da6096f852ae8",
             "plantvillage": "7f7ecc7e1eaca78107e3affe7cb5abd9427e139a"}
HEALTHY = {"Apple leaf","Bell_pepper leaf","Blueberry leaf","Cherry leaf","Peach leaf","Raspberry leaf",
           "Soyabean leaf","Strawberry leaf","Tomato leaf","grape leaf"}
DISEASE = {"Apple Scab Leaf","Apple rust leaf","Bell_pepper leaf spot","Corn Gray leaf spot",
           "Corn leaf blight","Corn rust leaf","Potato leaf early blight","Potato leaf late blight",
           "Squash Powdery mildew leaf","Tomato Early blight leaf","Tomato Septoria leaf spot",
           "Tomato leaf bacterial spot","Tomato leaf late blight","Tomato leaf mosaic virus",
           "Tomato leaf yellow virus","Tomato mold leaf","grape leaf black rot"}
SEEN_CROPS = {"apple","bell_pepper","potato","tomato"}


def read_json(path):
    return json.loads(path.read_text())


def fingerprint(task):
    collection, filename, absolute_path, expected_digest, blob_id = task
    data = absolute_path.read_bytes() if absolute_path.is_file() else b""
    if blob_id:
        blob_hash = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if blob_hash != blob_id:
            # Some official train files differ only in case and cannot coexist on
            # a case-insensitive checkout. Read the pinned object in those cases.
            data = subprocess.check_output(["git","-C",str(DOC),"cat-file","blob",blob_id])
    digest = hashlib.sha256(data).hexdigest()
    if expected_digest and digest != expected_digest:
        raise ValueError(f"Published SHA-256 mismatch for {collection}/{filename}")
    with Image.open(BytesIO(data)) as encoded:
        image = ImageOps.exif_transpose(encoded).convert("RGB")
    gray = np.asarray(image.convert("L").resize((32, 32)), dtype=np.float64)
    dct = dctn(gray, norm="ortho")[:8, :8].flatten()
    phash = int.from_bytes(np.packbits(dct > np.median(dct[1:])).tobytes(), "big")
    return {"collection": collection, "path": filename, "width": image.width, "height": image.height,
            "file_sha256": digest, "pixels_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            "phash": phash}


def matching(a, b):
    distance = (a["phash"] ^ b["phash"]).bit_count()
    same_pixels = (a["width"],a["height"],a["pixels_sha256"]) == (b["width"],b["height"],b["pixels_sha256"])
    if a["file_sha256"] == b["file_sha256"]:
        return "file", distance
    if same_pixels:
        return "pixels", distance
    if distance <= 4:
        return "perceptual", distance
    return None


def write_csv(path, records, fields):
    with path.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def main():
    start = perf_counter()
    RESULTS.mkdir(parents=True, exist_ok=True)
    for repo, revision in REVISIONS.items():
        actual = subprocess.check_output(["git","-C",str(RAW/repo),"rev-parse","HEAD"],text=True).strip()
        if actual != revision:
            raise ValueError(f"Unpinned revision {repo} {actual}")
    refs = AGRI / "reports/field_validation"
    archive = {x["source_path"]:x for x in read_json(refs/"SOURCE_MANIFEST.json")["samples"]}
    paired = {x["sample_id"]:x for x in read_json(refs/"INFERENCE_PARITY.json")["samples"]}
    training = read_json(AGRI / "data/manifests/plantvillage_split.json")["samples"]
    assert len(training)==2792 and len(archive)==len(paired)==60
    tree = subprocess.check_output(["git","-C",str(DOC),"ls-tree","-r","-z","HEAD","--","test","train"])
    tasks = []
    for entry in tree.rstrip(b"\0").split(b"\0"):
        oid, name = entry.split(b"\t",1)
        filename = name.decode("utf-8")
        cohort = filename.split("/",1)[0]
        tasks.append((f"plantdoc_{cohort}",filename,DOC/filename,None,oid.split()[2].decode()))
    tasks += [(f"plantvillage_{row['split']}",row["path"],PV/row["path"],row["sha256"],None) for row in training]
    with ThreadPoolExecutor(max_workers=4) as pool:
        info = list(pool.map(fingerprint,tasks))
    fingerprints_dir = ROOT / "data/features"
    fingerprints_dir.mkdir(parents=True, exist_ok=True)
    for row in info:
        row["phash_hex"] = f"{row['phash']:016x}"
    write_csv(fingerprints_dir/"source_fingerprints.csv",info,
              ["collection","path","width","height","file_sha256","pixels_sha256","phash_hex"])
    ref = [x for x in info if x["collection"]!="plantdoc_test"]
    test = [x for x in info if x["collection"]=="plantdoc_test"]
    if len(test)!=236 or len(ref)!=2342+2792:
        raise ValueError("Incomplete pinned raw data")
    nparts = len(test)
    parent = list(range(nparts))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    exclusions = [set() for _ in test]
    pairs = []
    for i, item in enumerate(test):
        for other in ref:
            found = matching(item,other)
            if found:
                mode,gap=found
                exclusions[i].add(other["collection"])
                pairs.append({"left":item["path"],"right_source":other["collection"],"right":other["path"],"match":mode,"hamming":gap})
        for j in range(i):
            found = matching(item,test[j])
            if found:
                mode,gap=found
                parent[root(i)] = root(j)
                pairs.append({"left":item["path"],"right_source":"plantdoc_test","right":test[j]["path"],"match":mode,"hamming":gap})
    for i in range(nparts):
        root(i)
    edge = EdgeModel()
    records = []
    for i,item in enumerate(test):
        path = item["path"]
        cls = Path(path).parent.name
        if cls not in HEALTHY|DISEASE:raise ValueError(f"Unknown source class {cls}")
        crop=cls.split()[0].lower()
        binary=int(cls in DISEASE)
        comp=root(i)
        group_matches=set().union(*(exclusions[j] for j in range(nparts) if root(j)==comp))
        member_labels={int(Path(test[j]["path"]).parent.name in DISEASE) for j in range(nparts) if root(j)==comp}
        reasons=sorted(group_matches)
        if len(member_labels)!=1:reasons.append("conflicting_test_labels")
        if i!=comp:reasons.append("within_test_copy")
        base = {"source_path":path,"source_class":cls,"crop":crop,"training_crop":int(crop in SEEN_CROPS),
                "y":binary,"official_test":1,"keep_dedup":int(not reasons),"dedup_reasons":"|".join(reasons),
                "test_component":test[comp]["path"],"in_public_hardware_60":int(path in archive),
                "width":item["width"],"height":item["height"],"file_sha256":item["file_sha256"],
                "phash_hex":f"{item['phash']:016x}"}
        base.update(edge.run(DOC/path))
        base.update({"sample_id":"","board_p_suspect":"","board_state":"","board_inference_us":"",
                     "published_fp32_p_suspect":"","published_fp32_state":""})
        if path in archive:
            entry=archive[path];published=paired[entry["sample_id"]]
            if entry["sha256"]!=base["file_sha256"] or entry["rgb_sha256"]!=base["rgb160_sha256"]:
                raise ValueError(f"Published inputs mismatch: {path}")
            if (entry["label"]=="SUSPECT")!=binary:raise ValueError(f"Published label mismatch: {path}")
            d,f=published["device"],published["fp32"]
            base.update(sample_id=entry["sample_id"],board_p_suspect=d["probabilities"][1],
                        board_state=d["state"],board_inference_us=d["inference_us"],
                        published_fp32_p_suspect=f["probabilities"][1],published_fp32_state=f["state"])
        records.append(base)
    write_csv(RESULTS/"plantdoc_overlap.csv",pairs,["left","right_source","right","match","hamming"])
    write_csv(RESULTS/"plantdoc_logits.csv",records,list(records[0]))
    public=[r for r in records if r["in_public_hardware_60"]]
    def describe(cohort):
        ac=[r for r in cohort if r["state"]!="UNKNOWN"]
        return {"n":len(cohort),"healthy":sum(not r["y"] for r in cohort),"suspect":sum(r["y"] for r in cohort),
                "binary_correct":sum(r["binary_pred"]==r["y"] for r in cohort),
                "accepted":len(ac),"accepted_correct":sum((r["state"]=="SUSPECT")==r["y"] for r in ac),
                "quality_rejected":sum(not r["quality_passed"] for r in cohort)}
    cohorts={"official_test":records,"public_hardware_60":public,
             "dedup_test":[r for r in records if r["keep_dedup"]],
             "dedup_disjoint_from_60":[r for r in records if r["keep_dedup"] and not r["in_public_hardware_60"]
                                        and not any(t["in_public_hardware_60"] and r["test_component"]==t["test_component"] for t in records)],
             "dedup_four_crops":[r for r in records if r["keep_dedup"] and r["training_crop"]]}
    summary={"model_sha256":MODEL_SHA256,"commits":REVISIONS,"runtime":{"python":platform.python_version(),
              "onnxruntime":onnxruntime.__version__,"pillow":pillow_version,"numpy":np.__version__,
              "host":platform.machine(),"provider":"CPUExecutionProvider","intra_threads":1,"inter_threads":1},
             "rows_per_reference_source":dict(Counter(x['collection'] for x in ref)),
             "cohorts":{s:describe(rows) for s,rows in cohorts.items()},
             "source_classes":len({r['source_class'] for r in records}),
             "pair_matches_by_reference_source":dict(Counter(x["right_source"] for x in pairs)),
             "max_archived_fp32_probability_difference":max(abs(r["p_suspect"]-r["published_fp32_p_suspect"]) for r in public),
             "archived_fp32_state_disagreements":sum(r["state"]!=r["published_fp32_state"] for r in public),
             "fp32_vs_board_state_disagreements":sum(r["state"]!=r["board_state"] for r in public),
             "host_inference_median_ms":statistics.median(r['host_inference_ms'] for r in records),
             "analysis_seconds":perf_counter()-start}
    (RESULTS/"plantdoc_inference_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()

"""Frozen DINOv2-small plus a PlantDoc-trained linear binary disease head.

Only the archived 224-pixel, quality-75 JPEG is supplied to the network. A
separate .npz for each corpus retains the row-aligned raw CLS embeddings.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import time
from urllib.request import urlopen

import numpy as np
import onnxruntime as ort
from PIL import Image
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score, average_precision_score


ROOT = Path(__file__).resolve().parents[1]
REV = "c2bb04a51fab207c420665f1946016107bffc701"
WEIGHT_SHA = "83141175ec78b4ff9a2bb58a4c7c264ba0054d1c2e122e5a8114b79a8d4179ea"
PREPROC_SHA = "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828"
TARGETS = {"apple", "bell_pepper", "potato", "tomato"}
FIELDS = ("dataset", "source_path", "split", "source_label", "crop", "y", "source_sha256",
          "jpeg_sha256", "jpeg_bytes", "jpeg_quality", "long_edge")
INT_FIELDS = {"y", "jpeg_bytes", "jpeg_quality", "long_edge"}
FEATURES = ROOT / "data/features"
RESULTS = ROOT / "results"
MODEL_DIR = ROOT / "data/raw/dino"


def digest_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def checked_model():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for file,remote,digest in (("model.onnx","onnx/model.onnx",WEIGHT_SHA),
                               ("preprocessor_config.json","preprocessor_config.json",PREPROC_SHA)):
        target=MODEL_DIR/file
        if not target.exists():
            with urlopen(f"https://huggingface.co/Xenova/dinov2-small/resolve/{REV}/{remote}",timeout=120) as response:
                target.write_bytes(response.read())
        if digest_file(target)!=digest:
            raise ValueError(f"Unexpected upstream file bytes: {target}")
    return json.loads((MODEL_DIR/"preprocessor_config.json").read_text())


def tensor(path, row, cfg):
    if digest_file(path)!=row["jpeg_sha256"] or path.stat().st_size!=int(row["jpeg_bytes"]):
        raise ValueError(f"JPEG bytes differ from measured payload: {row['source_path']}")
    with Image.open(path) as encoded:
        image = encoded.convert("RGB")
    return tensor_from_image(image, cfg)


def tensor_from_image(image, cfg):
    """Apply the original 224px DINO resize, center crop and normalization."""
    w,h=image.size
    size=cfg["size"]["shortest_edge"]
    ww,hh=(size,int(size*h/w)) if w<=h else (int(size*w/h),size)
    image=image.resize((ww,hh),Image.Resampling(cfg["resample"]))
    cw,ch=cfg["crop_size"]["width"],cfg["crop_size"]["height"]
    left,top=(ww-cw)//2,(hh-ch)//2
    pixels=np.asarray(image.crop((left,top,left+cw,top+ch)),dtype=np.float32)
    pixels=pixels*np.float32(cfg["rescale_factor"])
    pixels=(pixels-np.asarray(cfg["image_mean"],dtype=np.float32))/np.asarray(cfg["image_std"],dtype=np.float32)
    return np.ascontiguousarray(pixels.transpose(2,0,1))


def embed(name, csvpath, jpegdir, batch_size, threads):
    cfg=checked_model()
    with csvpath.open(newline="") as stream:
        rows=[r for r in csv.DictReader(stream) if int(r["long_edge"])==224 and
              (r["split"]!="train" or name=="fgvc" or r["crop"] in TARGETS)]
    expected=1478 if name=="plantdoc" else 1821
    if len(rows)!=expected or len({r['source_path'] for r in rows})!=expected:
        raise ValueError(f"Incomplete or repeated {name} inputs: got {len(rows)}, expected {expected}")
    if any(int(row["jpeg_quality"]) != 75 for row in rows):
        raise ValueError("Expected 224-pixel JPEGs encoded at quality 75")
    options=ort.SessionOptions()
    options.intra_op_num_threads=threads
    options.inter_op_num_threads=1
    session=ort.InferenceSession(str(MODEL_DIR/"model.onnx"),options,providers=["CPUExecutionProvider"])
    embeddings=[]
    begin=time.perf_counter()
    for offset in range(0,len(rows),batch_size):
        batch=rows[offset:offset+batch_size]
        images=np.stack([tensor(jpegdir/(r["jpeg_sha256"]+".jpg"),r,cfg) for r in batch])
        hidden=session.run(["last_hidden_state"],{"pixel_values":images})[0]
        cls=hidden[:,0,:].copy()
        if cls.shape!=(len(batch),384) or not np.isfinite(cls).all():
            raise ValueError("Invalid DINOv2 class token")
        embeddings.append(cls)
        if offset%(batch_size*20)==0:
            print(f"{name} {offset+len(batch)}/{len(rows)} in {time.perf_counter()-begin:.1f}s",flush=True)
    arrays={key:np.asarray([r[key] for r in rows],dtype=np.int32 if key in INT_FIELDS else str) for key in FIELDS}
    arrays.update(cls=np.concatenate(embeddings),model_sha256=np.asarray(WEIGHT_SHA),preprocessing_sha256=np.asarray(PREPROC_SHA),
                  model_revision=np.asarray(REV),elapsed_seconds=np.asarray(time.perf_counter()-begin),
                  input_manifest_sha256=np.asarray(digest_file(csvpath)))
    FEATURES.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(FEATURES/f"dino_{name}.npz",**arrays)


def score_test_cohorts(records):
    with (RESULTS/"plantdoc_logits.csv").open() as src:
        plant={r["source_path"]:r for r in csv.DictReader(src)}
    with (RESULTS/"fgvc_edge.csv").open() as src:
        apple={r["source_path"]:r for r in csv.DictReader(src)}
    tests=[r for r in records if r["feature_dataset"]=="plantdoc" and r["split"]=="test"]
    ext=[r for r in records if r["feature_dataset"]=="fgvc"]
    groups={"plantdoc_all":(tests,plant),
            "plantdoc_dedup_four_crops":([r for r in tests if r["crop"] in TARGETS and plant[r["source_path"]]["keep_dedup"]=="1"],plant),
            "plantdoc_dedup_all":([r for r in tests if plant[r["source_path"]]["keep_dedup"]=="1"],plant),
            "cornell_all_external":(ext,apple)}
    report={}
    for name,(subset,lookup) in groups.items():
        y=np.asarray([r["y"] for r in subset])
        p=np.asarray([r["p_diseased"] for r in subset])
        pred=(p>=.5)
        other=np.asarray([int(lookup[r["source_path"]]["binary_pred"]) for r in subset])
        diagnosis_key="y" if lookup is plant else "true_diseased"
        if not np.array_equal(y,[int(lookup[r["source_path"]][diagnosis_key]) for r in subset]):
            raise ValueError("Mismatched labels in paired comparison")
        true_neg=int(np.sum((y==0)&(~pred))); true_pos=int(np.sum((y==1)&pred))
        report[name]={"n":len(y),"true_negative":true_neg,"true_positive":true_pos,
                      "false_positive":int(np.sum((y==0)&pred)),"false_negative":int(np.sum((y==1)&(~pred))),
                      "accuracy":float(np.mean(pred==y)),"balanced_accuracy":float((true_pos/sum(y==1)+true_neg/sum(y==0))/2),
                      "roc_auc":float(roc_auc_score(y,p)),"average_precision":float(average_precision_score(y,p)),
                      "corrected_edge_error":int(np.sum((pred==y)&(other!=y))),
                      "introduced_edge_error":int(np.sum((pred!=y)&(other==y)))}
    return report


def fit_predict():
    trainfile=FEATURES/"dino_plantdoc.npz"
    with np.load(trainfile,allow_pickle=False) as train:
        rows=np.flatnonzero((train["split"]=="train")&np.isin(train["crop"],sorted(TARGETS)))
        if len(rows)!=1242:
            raise ValueError("PlantDoc official training input is incomplete")
        # Collapse duplicate transmitted inputs using only the development split. Rows
        # with contradictory training labels are discarded as a whole group.
        groups={}
        for i in rows:
            groups.setdefault(str(train["jpeg_sha256"][i]),[]).append(int(i))
        ids=[indices[0] for _,indices in sorted(groups.items()) if len(set(train["y"][indices]))==1]
        fit_keys=[str(train["source_path"][i]) for i in ids]
        clf=LogisticRegression(C=1.0,solver="lbfgs",max_iter=3000,tol=1e-8)
        clf.fit(normalize(train["cls"][ids].astype(np.float64)),train["y"][ids])
        if list(clf.classes_)!=[0,1]:
            raise ValueError("Expected both training classes")
    records=[]
    RESULTS.mkdir(parents=True,exist_ok=True)
    summary={"source_model":"https://huggingface.co/facebook/dinov2-small",
             "converted_model":"https://huggingface.co/Xenova/dinov2-small/tree/"+REV,
             "model_sha256":WEIGHT_SHA,"preprocessing_sha256":PREPROC_SHA,"embedding":"raw CLS, last_hidden_state[:,0,:]",
             "classification":"L2-normalized float64 raw CLS, sklearn L2 logistic regression, C=1, fixed threshold p>=0.5",
             "fit_source":"PlantDoc official train, apple/bell_pepper/potato/tomato",
             "training_images_before_payload_dedup":len(rows),"fit_rows":len(ids),
             "fit_source_paths_sha256":hashlib.sha256(("\n".join(fit_keys)+"\n").encode()).hexdigest(),
             "sklearn":sklearn_version,"onnxruntime":ort.__version__,"python":platform.python_version(),
             "coef_sha256":hashlib.sha256(clf.coef_.tobytes()+clf.intercept_.tobytes()).hexdigest(),
             "n_iter":int(clf.n_iter_[0]),"data":{}}
    np.savez_compressed(FEATURES/"dino_plantdoc_head.npz",coef=clf.coef_,intercept=clf.intercept_,classes=clf.classes_,
                        fit_keys=np.asarray(fit_keys))
    for corpus in ("plantdoc","fgvc"):
        archive=FEATURES/f"dino_{corpus}.npz"
        with np.load(archive,allow_pickle=False) as z:
            arrays={k:z[k] for k in FIELDS}
            scores=clf.decision_function(normalize(z["cls"].astype(np.float64)))
            probs=1/(1+np.exp(-scores))
            for i,(logit,p) in enumerate(zip(scores,probs,strict=True)):
                r={key:vals[i].item() for key,vals in arrays.items()}
                r.update(backend="dino_v2_small_plantdoc_linear",score=float(logit),p_diseased=float(p),
                         pred_diseased=int(p>=.5),feature_row=i,feature_dataset=corpus)
                records.append(r)
            summary["data"][corpus]={"n":len(scores),"feature_dim":int(z["cls"].shape[1]),"npz_sha256":digest_file(archive),
                                      "seconds":float(z["elapsed_seconds"])}
    with (RESULTS/"specialist_predictions.csv").open("w",newline="") as out:
        writer=csv.DictWriter(out,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)
    (RESULTS/"specialist_cohorts.json").write_text(json.dumps(score_test_cohorts(records),indent=2)+"\n")
    (RESULTS/"specialist_provenance.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["plantdoc","fgvc","head"])
    parser.add_argument("--manifest",type=Path)
    parser.add_argument("--jpegs",type=Path)
    parser.add_argument("--batch",type=int,default=8)
    parser.add_argument("--threads",type=int,default=3)
    args=parser.parse_args()
    if args.stage=="head":
        fit_predict()
    else:
        plant=args.stage=="plantdoc"
        manifest=args.manifest or RESULTS/("backend_predictions.csv" if plant else "fgvc_backend.csv")
        jpegs=args.jpegs or ROOT/"data"/("backend_jpegs" if plant else "fgvc_backend_jpegs")
        embed(args.stage,manifest,jpegs,args.batch,args.threads)


if __name__=="__main__":
    main()

"""Prepare matched target-label purchasing panels using locally licensed research inputs.

Private arrays are written only below data/raw. Only counts, hashes and split checks
are exported. Decisions must receive bought calibration labels through their ledger;
test_y and test_cloud_p are for scored/deferred outcomes respectively.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from directional_policy import probabilities_from_logits
from edge_adaptation_wire import normalized, predict_from_head, quantized

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"
CACHE = DATA / "raw/purchase_cohorts"
SITES = ("cornell", "kashmir", "latvia")


def rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z}


def sha(content):
    return hashlib.sha256(content).hexdigest()


def aligned(names, target):
    lookup = {p: i for i, p in enumerate(names)}
    if len(lookup) != len(names):
        raise ValueError("Duplicate source identifier")
    return np.array([lookup[p] for p in target])


def cloud(x):
    h = load_npz(DATA / "features/dino_plantdoc_balanced_head.npz")
    return probabilities_from_logits(normalized(x) @ h["coef"][0] + h["intercept"].item())


def v4_p(x):
    with ZipFile(OUT / "second_backbone_heads.zip") as z:
        return probabilities_from_logits(predict_from_head("local_int8", z.read("source.bin"), quantized(x)[0]))


def source_v4(site, names, y, digests):
    z = load_npz(DATA / "raw/second_backbone" / (site + ".npz"))
    i = aligned(z["source_path"], names)
    assert np.array_equal(z["y"][i], y) and np.array_equal(z["source_sha256"][i], digests)
    return z["features"][i]


def near_distance(a, b):
    left = [int(v, 16) for v in a]
    right = [int(v, 16) for v in b]
    return min((x ^ y).bit_count() for x in left for y in right)


def cornell():
    ref = rows(OUT / "edge_adaptation_split.csv")
    data = load_npz(DATA / "features/dino_fgvc.npz")
    edge = load_npz(DATA / "features/edge_fgvc.npz")
    frozen = {r["source_path"]: r for r in rows(OUT / "fgvc_edge.csv")}
    wanted = [r for r in ref if r["status"] in ("label_pool", "test")]
    cal = np.array([r["status"] == "label_pool" for r in wanted])
    names = np.array([r["source_path"] for r in wanted])
    assert all(r["group_representative"] == r["source_path"] for r in wanted)
    i = aligned(data["source_path"], names)
    j = aligned(edge["source_path"], names)
    y, digest = data["y"][i], data["source_sha256"][i]
    assert np.array_equal(edge["source_sha256"][j], digest)
    assert np.array_equal(y, np.array([int(r["y"]) for r in wanted]))
    return dict(names=names, digest=digest, y=y, cal=cal, size=data["jpeg_bytes"][i],
                dates=None, stamps=None, phash=np.array([r["phash"] for r in wanted]),
                v2=edge["activation"][j], v2p=np.array([float(frozen[p]["p_suspect"]) for p in names], np.float32),
                v4=source_v4("cornell", names, y, digest), cp=cloud(data["cls"][i]))


def latvia():
    data = load_npz(DATA / "features/latvia.npz")
    split = rows(OUT / "latvia_local_date_split.csv")
    metadata = {r["source_path"]: r for r in rows(DATA / "latvia_source/manifest.csv")}
    selected = [r for r in split if r["status"] == "test" or r["source_path"] == r["component_representative"]]
    cal = np.array([r["status"] == "calibration" for r in selected])
    assert not ({r["component_representative"] for r in selected if r["status"] == "calibration"}
                & {r["component_representative"] for r in selected if r["status"] == "test"})
    names = np.array([r["source_path"] for r in selected])
    i = aligned(data["source_path"], names)
    y, digest = data["y"][i], data["source_sha256"][i]
    assert np.array_equal(digest, [r["source_sha256"] for r in selected])
    assert np.array_equal(y, [int(r["y"]) for r in selected])
    recorded = load_npz(OUT / "latvia_decisions.npz")
    j = aligned(recorded["source_path"], names)
    return dict(names=names, digest=digest, y=y, cal=cal, size=data["jpeg_bytes"][i],
                dates=np.array([r["capture_date"] for r in selected]),
                stamps=np.array([metadata[p]["exif_datetime_original"] for p in names]),
                phash=data["phash"][i], v2=data["edge"][i], v2p=recorded["edge_probability"][j],
                v4=source_v4("latvia", names, y, digest), cp=recorded["remote_probability"][j])


def kashmir_bank():
    """Select without reading labels, then decode chosen photos on all frozen models."""
    from PIL import Image
    import pyarrow.parquet as pq
    from scipy.fft import dctn
    from esp_quantized_replay import graph as grid_graph, prepare as prepare_grid
    from field_specialist import checked_model as dino_config, tensor_from_image, MODEL_DIR
    from latvia_blind_features import decode, hardware_graph, session
    from second_backbone_features import checked_model as model_v4
    from server_backend import jpeg_transport

    path = DATA / "raw/kashmir.parquet"
    source = json.loads((DATA / "kashmir_source.json").read_text())
    assert sha(path.read_bytes()) == source["parquet_sha256"]
    old = rows(OUT / "transfer_manifest.csv")
    test_paths = {r["source_path"] for r in old}
    # Screen the original test cohort, both non-Kashmir sites and original model-training sources.
    original = rows(DATA / "features/source_fingerprints.csv")
    others = [(r["file_sha256"], r["phash_hex"]) for r in original]
    others += [(r["file_sha256"], r["phash"]) for r in rows(OUT / "fgvc_edge.csv")]
    others += [(r["source_sha256"], r["phash"]) for r in rows(DATA / "latvia_source/manifest.csv")]
    others += [(r["source_sha256"], r["phash"]) for r in old]
    seen_digest = {h for h, _ in others}
    seen_phash = [int(p, 16) for _, p in others]
    images = pq.read_table(path, columns=["image"])["image"].to_pylist()
    assert len(images) == 5612
    ranked = [(sha(b"buy-bank-kashmir-v1/" + hashlib.sha256(r["bytes"]).digest()), i)
              for i, r in enumerate(images) if r["path"] not in test_paths]
    selected, distance_rejected, exact_rejected = [], 0, 0
    for _, i in sorted(ranked):
        f = images[i]
        digest = sha(f["bytes"])
        if digest in seen_digest:
            exact_rejected += 1
            continue
        image = decode(f["bytes"])
        gray = np.asarray(image.convert("L").resize((32, 32)), dtype=np.float64)
        spectrum = dctn(gray,norm="ortho")[:8,:8].ravel()
        phash = np.packbits(spectrum > np.median(spectrum[1:])).tobytes().hex()
        number = int(phash, 16)
        if any((number ^ v).bit_count() <= 5 for v in seen_phash):
            distance_rejected += 1
            continue
        selected.append((i, digest, phash))
        seen_digest.add(digest)
        seen_phash.append(number)
        if len(selected) == 256:
            break
    assert len(selected) == 256
    # The publisher's labels are joined only after the irrevocable choice of all bank identities.
    y = pq.read_table(path, columns=["label"])["label"].to_numpy()
    mobile, edge = hardware_graph()
    four, hidden, fw, bias, fourth = model_v4(2)
    dino = session(str(MODEL_DIR / "model.onnx"), 2)
    cfg = dino_config()
    grid = grid_graph(DATA/"raw/esp-ppq/tinyvision.json")
    grid_meta = json.loads((DATA/"raw/tinyvision-metadata.json").read_text())["preprocessing"]
    payload = {k: [] for k in ("names","digest","y","size","phash","v2","v2p","v4","dino","jpeg_sha256","v2grid","v2gridp")}
    test_index = {r["path"]: i for i,r in enumerate(images) if r["path"] in test_paths}
    # Re-extract 4 original reference images to check scales, model, JPEG and cache alignment.
    anchors = [(test_index[r["source_path"]],r["source_sha256"],r["phash"]) for r in old[:4]]
    for i, digest, phash in selected+anchors:
        f = images[i]
        image = decode(f["bytes"])
        rgb = np.asarray(image.resize((160,160),Image.Resampling.BILINEAR), np.uint8)
        inp = np.ascontiguousarray(((rgb.astype(np.float32)/255-mobile.mean)/mobile.std).transpose(2,0,1)[None])
        logits, e = edge.run(None,{mobile.input_name:inp})
        ex = np.exp(logits[0] - logits[0].max())
        vp = tensor_from_image(image, fourth)[None]
        log4, x4 = four.run([four.get_outputs()[0].name,hidden],{four.get_inputs()[0].name:vp})
        assert np.max(np.abs(x4 @ fw.T + bias - log4)) < 1e-4
        jpeg = jpeg_transport(image, 224)
        with Image.open(io.BytesIO(jpeg)) as im:
            dx = tensor_from_image(im.convert("RGB"),cfg)[None]
        d = dino.run(["last_hidden_state"],{"pixel_values": dx})[0][0,0]
        gx, _ = prepare_grid(image,grid_meta)
        gl, gh = grid.run(None,{"image":gx})
        assert np.all(gh * 32 == np.rint(gh * 32))
        gcode = np.rint(gh[0]*32).astype(np.int8)
        gp = float(1/(1+np.exp(-np.diff(gl.ravel()).item())))
        values = (f["path"],digest,int(y[i]!=2),len(jpeg),phash,e[0],ex[1]/ex.sum(),x4[0],d,sha(jpeg),gcode,gp)
        for k,v in zip(payload,values,strict=True):
            payload[k].append(v)
    payload = {k: np.array(v) for k,v in payload.items()}
    x = load_npz(DATA/"features/kashmir.npz")
    i = aligned(x["source_path"],payload["names"][-4:])
    assert np.max(np.abs(payload["v2"][-4:]-x["edge"][i])) < 1e-5
    assert np.max(np.abs(payload["dino"][-4:]-x["dino"][i])) < 1e-3
    ref = source_v4("kashmir",payload["names"][-4:],payload["y"][-4:],payload["digest"][-4:])
    assert np.max(np.abs(payload["v4"][-4:]-ref)) < 1e-3
    orig_grid = load_npz(DATA/"raw/esp-grid-cache/kashmir.npz")
    gi = aligned(orig_grid["source_path"],payload["names"][-4:])
    assert np.array_equal(orig_grid["integer"][gi],payload["v2grid"][-4:])
    assert np.array_equal(orig_grid["prob"][gi],payload["v2gridp"][-4:])
    assert list(payload["jpeg_sha256"][-4:])==[r["jpeg_sha256"] for r in old[:4]]
    return {**{k:v[:-4] for k,v in payload.items()},
            "selection_exact_rejected":np.array(exact_rejected),"selection_phash_rejected":np.array(distance_rejected)}


def kashmir():
    cache = CACHE/"kashmir_extra.npz"
    if not cache.exists():
        bank = kashmir_bank()
        tmp = cache.with_suffix(".new.npz")
        np.savez_compressed(tmp, **bank)
        tmp.replace(cache)
    extra = load_npz(cache)
    originals = rows(OUT/"transfer_manifest.csv")
    old = load_npz(DATA/"features/kashmir.npz")
    names = np.array([r["source_path"] for r in originals])
    i = aligned(old["source_path"], names)
    digest = np.array([r["source_sha256"] for r in originals])
    y = np.array([int(r["y"]) for r in originals])
    directional = {r["path"]:r for r in rows(OUT/"directional_predictions.csv") if r["cohort"]=="kashmir"}
    ext = dict(names=names,digest=digest,y=y,phash=[r["phash"] for r in originals],
               size=[int(r["jpeg_bytes"]) for r in originals],v2=old["edge"][i],
               v2p=np.array([float(directional[p]["edge_p"]) for p in names],np.float32),
               v4=source_v4("kashmir", names, y, digest),dino=old["dino"][i])
    gr = load_npz(DATA/"raw/esp-grid-cache/kashmir.npz")
    gi = aligned(gr["source_path"],names)
    ext.update(v2grid=gr["integer"][gi],v2gridp=gr["prob"][gi])
    joined = {k:np.concatenate((extra[k],ext[k])) for k in ext}
    cp = cloud(joined.pop("dino"))
    return {**joined,"cp":cp, "cal":np.r_[np.ones(256,bool),np.zeros(384,bool)],
            "stamps":None,"dates":None, "selection_exact_rejected":int(extra["selection_exact_rejected"]),
            "selection_phash_rejected":int(extra["selection_phash_rejected"])}


def build(site):
    data = {"cornell":cornell,"latvia":latvia,"kashmir":kashmir}[site]()
    cal, digest, names = data["cal"], data["digest"], data["names"]
    # Content is primary. A second opaque hash distinguishes different archive records for byte-identical images.
    uids = np.array([sha(b"crop-purchase-id-v1/" + bytes.fromhex(h) + hashlib.sha256(str(p).encode()).digest())
                     for h,p in zip(digest,names,strict=True)])
    assert len(set(uids)) == len(uids)
    assert set(digest[cal]).isdisjoint(digest[~cal])
    distance = near_distance(data["phash"][cal], data["phash"][~cal])
    assert distance > (5 if site=="kashmir" else 4)
    order_idx = lambda arr, salt: np.argsort([sha(salt+bytes.fromhex(k)) for k in arr],kind="stable")
    draws = np.stack([order_idx(uids[cal], f"purchase-paid-bank-v1/{seed}/".encode())[:128] for seed in range(20)])
    if site == "latvia":
        assert max(data["dates"][cal]) < min(data["dates"][~cal])
        orders = np.lexsort((uids[~cal], data["stamps"][~cal]))[None]
        groups = data["dates"][~cal]
    else:
        orders = np.stack([order_idx(uids[~cal],f"purchase-eval-v1/{k}/".encode()) for k in range(5)])
        groups = uids[~cal]
    if site != "kashmir":
        gr = load_npz(DATA/"raw/esp-grid-cache"/(site+".npz"))
        gi = aligned(gr["source_path"],names)
        data.update(v2grid=gr["integer"][gi],v2gridp=gr["prob"][gi])
    for backbone in ("v2","v4","v2grid"):
        x = data[backbone]
        frozen = v4_p(x) if backbone=="v4" else data[backbone+"p"]
        arrays = {}
        for label, mask in (("cal",cal),("test",~cal)):
            for suffix, vec in (("x",x),("id",uids),("p",frozen),("cloud_p",data["cp"]),
                                ("size",data["size"]),("y",np.asarray(data["y"],np.int8))):
                arrays[label+"_"+suffix] = vec[mask]
        z = load_npz(OUT/f"second_backbone_{site}.npz")
        mask = np.isin(z["paths"], names[~cal])
        dest = aligned(names[~cal], z["paths"][mask])
        assert np.array_equal((arrays["test_cloud_p"][dest]>=.5),z["expert"][mask])
        assert np.array_equal(arrays["test_size"][dest]+20,z["application_bytes"][mask])
        assert np.array_equal(arrays["test_y"][dest],z["y"][mask])
        if backbone == "v4":
            assert np.array_equal(frozen[~cal][dest]>=.5,z["source_margin"][mask]>=0)
        elif backbone == "v2":
            comparison = load_npz(OUT/f"joint_budget_{site}.npz")
            keep = np.isin(comparison["source_path"], names[~cal])
            orig = aligned(names[~cal],comparison["source_path"][keep])
            assert np.array_equal(frozen[~cal][orig]>=.30, comparison["original"][keep])
        fn = CACHE/f"{site}_{backbone}.npz"
        tmp=fn.with_suffix(".new.npz")
        np.savez_compressed(tmp, **arrays, cal_draws=draws,test_orders=orders,test_order=orders[0],
                            test_group=groups, cutoff=.5 if backbone=="v4" else .3)
        tmp.replace(fn)
    assert (sum(cal),sum(~cal))=={"cornell":(450,1353),"kashmir":(256,384),"latvia":(590,579)}[site]
    info = dict(site=site,calibration_candidates=int(cal.sum()),test_images=int((~cal).sum()),
                test_positives=int(data["y"][~cal].sum()),test_distinct_content=len(set(digest[~cal])),
                split_exact_content_overlap=0,min_split_phash_distance=distance,eval_orders=len(orders),
                cal_id_digest=sha("".join(sorted(uids[cal])).encode()),
                test_id_digest=sha("".join(sorted(uids[~cal])).encode()),
                target_label_draws_digest=sha("".join(uids[cal][draws].flat).encode()),
                test_order_digest=sha("".join(uids[~cal][orders].flat).encode()),
                v2_and_v4_feature_width=int(data["v2"].shape[1]),test_jpeg_bytes=int(data["size"][~cal].sum()),
                cal_mean_jpeg_bytes=float(data["size"][cal].mean()))
    if site=="latvia":
        info.update(cal_days=sorted(set(data["dates"][cal])),test_days=sorted(set(data["dates"][~cal])))
    if site=="kashmir":
        info.update(selection_exact_rejected=data["selection_exact_rejected"],
                    selection_phash_rejected=data["selection_phash_rejected"],anchor_images_replayed=4)
    (CACHE/f"{site}_audit.json").write_text(json.dumps(info,indent=2)+"\n")
    print(site,cal.sum(),(~cal).sum(),distance,flush=True)


def load(site, backbone):
    if site not in SITES or backbone not in ("v2","v4","v2grid"):
        raise ValueError("Unknown site or backbone")
    return load_npz(CACHE/f"{site}_{backbone}.npz")


def verify():
    for site in SITES:
        original = load(site,"v2")
        n, m = len(original["cal_id"]), len(original["test_id"])
        assert {(450,1353),(256,384),(590,579)}.issuperset({(n,m)})
        assert set(original["cal_id"]).isdisjoint(original["test_id"])
        assert len(np.unique(original["test_id"])) == m
        assert original["cal_draws"].shape == (20,128)
        assert np.all((original["cal_draws"] >= 0) & (original["cal_draws"] < n))
        assert all(len(np.unique(draw)) == 128 for draw in original["cal_draws"])
        assert np.array_equal(original["test_order"], original["test_orders"][0])
        assert all(np.array_equal(np.sort(order),np.arange(m)) for order in original["test_orders"])
        for variant in ("v2","v4","v2grid"):
            data = load(site,variant)
            assert data["cutoff"] == (.5 if variant == "v4" else .3)
            for split,l in (("cal",n),("test",m)):
                assert data[f"{split}_x"].shape == (l,1280)
                assert np.isfinite(data[f"{split}_x"]).all()
                prob=data[f"{split}_p"]
                assert np.all((prob>=0)&(prob<=1))
                assert np.isfinite(data[f"{split}_cloud_p"]).all()
                assert data[f"{split}_x"].dtype == (np.int8 if variant == "v2grid" else np.float32)
            for key in original:
                if key not in {"cal_x","test_x","cal_p","test_p","cutoff"}:
                    assert np.array_equal(original[key],data[key]), (site,variant,key)
        print(f"{site}: three feature versions, {n} disjoint candidates, {m} matched evaluations, 20 panels pass")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sites",nargs="*",metavar="SITE",help="cornell, kashmir or latvia")
    parser.add_argument("--verify-only",action="store_true")
    args=parser.parse_args()
    if args.verify_only:
        verify()
        return
    CACHE.mkdir(parents=True,exist_ok=True)
    for site in args.sites or SITES:
        build(site)
    if all((CACHE/f"{s}_audit.json").exists() for s in SITES):
        info=[json.loads((CACHE/f"{s}_audit.json").read_text()) for s in SITES]
        (OUT/"purchase_cohort_audit.json").write_text(json.dumps(info,indent=2)+"\n")


if __name__=="__main__":
    main()

"""Audit simultaneous FP/FN reductions of already locked deployment policies."""

from collections import defaultdict
from zipfile import ZipFile

import numpy as np

from edge_adaptation_split import ROOT, RESULTS, table, write_table
from edge_adaptation_wire import normalized, quantized, predict_from_head


def error_counts(y, pred):
    pred = np.asarray(pred, bool)
    return int(np.sum(pred & (y == 0))), int(np.sum(~pred & (y == 1)))


def oracle_on_locked_scores(y, scores, fp_cap):
    """Lexicographically minimize (FN,FP) under FP<=cap at an inclusive threshold.

    Only full groups of tied scores may be admitted. All-negative is also feasible.
    This use of the evaluation labels cannot set a deployable threshold.
    """
    order = np.argsort(-scores, kind="stable")
    z, truth = scores[order], y[order]
    last = np.r_[z[:-1] != z[1:], True]
    tps = np.r_[0, np.cumsum(truth == 1)[last]]
    fps = np.r_[0, np.cumsum(truth == 0)[last]]
    fns = int(truth.sum()) - tps
    return min((int(fn), int(fp)) for fn, fp in zip(fns, fps) if fp <= fp_cap)


def audit():
    with np.load(RESULTS / "edge_adaptation_predictions.npz", allow_pickle=False) as v:
        cy, cp, cscore, cpred = v["test_y"], v["test_paths"], v["score"], v["pred"]
        methods, budgets, seeds = v["methods"], v["budgets"], v["seeds"]
    edge = {r["source_path"]: r for r in table(RESULTS / "fgvc_edge.csv")}
    c_raw = np.array(
        [float(edge[p]["suspect_logit"]) - float(edge[p]["normal_logit"]) for p in cp],
        dtype=np.float32,
    )
    c_base = np.array([int(edge[p]["binary_pred"]) for p in cp])
    krows = table(RESULTS / "transfer_manifest.csv")
    ky = np.array([int(r["y"]) for r in krows])
    kp = np.array([r["source_path"] for r in krows])
    k_raw = np.array([float(r["edge_score_logit"]) for r in krows], dtype=np.float32)
    k_base = np.array([int(r["edge_pred"]) for r in krows])
    with np.load(ROOT / "data/features/kashmir.npz", allow_pickle=False) as z:
        if not np.array_equal(kp, z["source_path"]):
            raise ValueError("Kashmir path mismatch")
        kfeatures = dict(
            local_int8=quantized(z["edge"])[0],
            local_fp32=normalized(z["edge"]),
            dino=normalized(z["dino"]),
            scalar=k_raw[:, None],
        )
    with np.load(RESULTS / "transfer_predictions.npz", allow_pickle=False) as v:
        if not np.array_equal(ky, v["y"]) or not np.array_equal(kp, v["source_path"]):
            raise ValueError("Kashmir replay order mismatch")
        kprev = {
            (m, int(b), int(s)): p
            for m, b, s, p in zip(v["method"], v["budget"], v["seeds"], v["pred"])
        }
    cy, ky = np.asarray(cy), np.asarray(ky)
    if error_counts(cy, c_base) != (215, 34) or error_counts(ky, k_base) != (81, 4):
        raise ValueError("Pool or original decisions changed")
    results = []

    def add(name, method, fit, cal, seed, threshold, zs, expected=(None, None)):
        current = []
        for cohort, y, base, z, exp in zip(
            ("cornell", "kashmir"), (cy, ky), (c_base, k_base), zs, expected
        ):
            if len(z) != len(y) or not np.isfinite(z).all():
                raise ValueError("Invalid scores")
            dec = z >= threshold
            if exp is not None and not np.array_equal(dec, exp):
                raise ValueError(
                    f"Decision replay mismatch {cohort, method, fit, seed}"
                )
            fp, fn = error_counts(y, dec)
            bfp, bfn = error_counts(y, base)
            ofn, ofp = oracle_on_locked_scores(y, z, bfp - 1)
            current.append(
                dict(
                    cohort=cohort,
                    policy=name,
                    method=method,
                    fit_labels=fit,
                    cal_labels=cal,
                    seed=seed,
                    locked_threshold=float(threshold),
                    n=len(y),
                    baseline_fp=bfp,
                    baseline_fn=bfn,
                    fp=fp,
                    fn=fn,
                    delta_fp=fp - bfp,
                    delta_fn=fn - bfn,
                    strict_dual=int(fp < bfp and fn < bfn),
                    test_oracle_strict_fp_cap=bfp - 1,
                    test_oracle_min_fn=ofn,
                    test_oracle_fp_at_min_fn=ofp,
                    test_oracle_dual_possible=int(ofn < bfn),
                )
            )
        pair = int(all(r["strict_dual"] for r in current))
        for r in current:
            r["locked_dual_on_both_cohorts"] = pair
        results.extend(current)

    add(
        "native",
        "frozen",
        0,
        0,
        -1,
        np.log(0.3 / 0.7),
        (c_raw, k_raw),
        (c_base, k_base),
    )
    with ZipFile(RESULTS / "edge_adaptation_heads.zip") as f:
        for s in seeds:
            for bidx, b in enumerate(budgets):
                for midx, m in enumerate(methods[1:], start=1):
                    blob = f.read(f"s{s:02d}-b{b:03d}/{m}.bin")
                    kscr = predict_from_head(m, blob, kfeatures[m])
                    add(
                        "native",
                        m,
                        int(b),
                        0,
                        int(s),
                        0,
                        (cscore[s, bidx, midx], kscr),
                        (cpred[s, bidx, midx], kprev[m, int(b), int(s)]),
                    )
    with (
        np.load(RESULTS / "sensitivity64_predictions.npz", allow_pickle=False) as v,
        ZipFile(RESULTS / "sensitivity64_heads.zip") as f,
    ):
        if not np.array_equal(v["test_paths"], cp):
            raise ValueError("Cornell sensitivity order mismatch")
        zc, cutoff, decisions = v["test_score"], v["threshold"], v["pred"]
        for s in seeds:
            for j, m in enumerate(v["methods"]):
                kscr = (
                    k_raw
                    if m == "frozen"
                    else predict_from_head(m, f.read(f"s{s:02d}/{m}.bin"), kfeatures[m])
                )
                for op, name in enumerate(v["operating_points"]):
                    if m == "frozen" and name == "native":
                        continue
                    add(
                        name,
                        m,
                        0 if m == "frozen" else 64,
                        64 if name == "target95" else 0,
                        int(s),
                        cutoff[s, j, op],
                        (zc[s, j], kscr),
                        (decisions[s, j, op], None),
                    )

    # Source-only and prior binary-label remote comparators already existed at the audit cutoff.
    c_mobile = {
        r["source_path"]: r
        for r in table(RESULTS / "fgvc_backend.csv")
        if r["long_edge"] == "224"
    }
    add(
        "native",
        "mobile224",
        0,
        0,
        -1,
        0.5,
        (
            np.array([float(c_mobile[p]["p_diseased"]) for p in cp]),
            np.array([float(r["mobile_server_p"]) for r in krows]),
        ),
        (
            np.array([int(c_mobile[p]["pred_diseased"]) for p in cp]),
            kprev["mobile224", 0, -1],
        ),
    )
    source = {}
    for m, file, headname in (
        ("plantdoc_dino", "specialist_predictions.csv", "dino_plantdoc_head.npz"),
        (
            "plantdoc_balanced",
            "specialist_balanced.csv",
            "dino_plantdoc_balanced_head.npz",
        ),
    ):
        c = {
            r["source_path"]: r
            for r in table(RESULTS / file)
            if r["dataset"] == "FGVC7"
        }
        with np.load(ROOT / "data/features" / headname, allow_pickle=False) as h:
            ks = kfeatures["dino"] @ h["coef"].ravel() + float(h["intercept"].item())
        cs = np.array([float(c[p]["score"]) for p in cp])
        source[m] = (cs, ks)
        add(
            "native",
            m,
            0,
            0,
            -1,
            0,
            (cs, ks),
            (np.array([int(c[p]["pred_diseased"]) for p in cp]), kprev[m, 0, -1]),
        )
    with np.load(
        RESULTS / "specialist_scalar_predictions.npz", allow_pickle=False
    ) as v:
        if not np.array_equal(v["test_paths"], cp):
            raise ValueError("Source-scalar input order mismatch")
        source_scores = v["logit"]
        source_pred = v["pred"]
    for r in table(RESULTS / "specialist_scalar.csv"):
        s, b = int(r["seed"]), int(r["budget"])
        a = float(r["scalar_weight"])
        h = float(r["scalar_intercept"])
        zz = tuple(
            np.asarray(z, dtype=np.float32).astype(np.float64) * a + h
            for z in source["plantdoc_dino"]
        )
        offset = list(budgets).index(b)
        if not np.allclose(zz[0], source_scores[s, offset], atol=1e-6, rtol=0):
            raise ValueError("Source scalar scores mismatch")
        add(
            "native",
            "plantdoc_dino_scalar",
            b,
            0,
            s,
            0,
            zz,
            (source_pred[s, offset], None),
        )
    write_table(RESULTS / "dual_existing.csv", results)
    groups = defaultdict(list)
    for r in results:
        groups[
            (r["cohort"], r["policy"], r["method"], r["fit_labels"], r["cal_labels"])
        ].append(r)
    agg = []
    for key, rr in sorted(groups.items()):
        cohort, policy, method, fit, cal = key
        agg.append(
            dict(
                cohort=cohort,
                policy=policy,
                method=method,
                fit_labels=fit,
                cal_labels=cal,
                n_policies=len(rr),
                n_strict_dual=sum(r["strict_dual"] for r in rr),
                n_paired_dual=sum(r["locked_dual_on_both_cohorts"] for r in rr),
                n_test_oracle_dual=sum(r["test_oracle_dual_possible"] for r in rr),
                fp_mean=sum(r["fp"] for r in rr) / len(rr),
                fn_mean=sum(r["fn"] for r in rr) / len(rr),
                winning_seeds=" ".join(str(r["seed"]) for r in rr if r["strict_dual"]),
            )
        )
    write_table(RESULTS / "dual_existing_summary.csv", agg)
    for r in agg:
        if r["n_strict_dual"]:
            print(r)
    print("paired configurations", len(results) // 2)


if __name__ == "__main__":
    audit()

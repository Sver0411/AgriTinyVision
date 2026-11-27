"""Budget-Amortized Source-Anchored Adaptation (BASA) for bandwidth-limited edge cameras.

Strictly causal and leak-free:
1. Horizon-amortized supervision budgeting selects k from window bytes B, grant mode,
   and unlabelled reservoir JPEG lengths only.
2. Quantile-stratified acquisition selects k candidates across uniform quantiles of
   unlabelled source probabilities p_0 (no labels accessed prior to purchase).
3. Anisotropic Source-Anchored Regularization (ASAR) solves the convex logistic
   objective with an orthogonal shrinkage penalty around the factory source head w_0,
   selecting lambda in {inf, 25, 5, 1} (simplest first, Occam's razor) via stratified
   out-of-fold discrimination on the k purchased labels alone.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from edge_adaptation_wire import I8, normalized, pack_head, predict_from_head, quantized
from published_defer import train_weights
from purchase_cohorts import load
from purchase_experiment import (
    ACTION_BYTES,
    BACKBONES,
    FULL_RATE,
    REQUEST_BYTES,
    SITES,
    array,
    library,
    ranks,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "basa"
LAMBDA_GRID = (float("inf"), 25.0, 5.0, 1.0)
RHO_PREPAID = 0.40
RHO_EARNED = 0.25
MIN_LABELS = 16
OCCAM_MARGIN = 0.005


@dataclass(frozen=True)
class BASAPlan:
    k: int
    setup_bytes: int
    selected_local: tuple[int, ...]
    chosen_lambda: float
    head_blob: bytes
    fallback_frozen: bool = False


def load_factory_source_heads(root: Path = ROOT) -> dict[str, np.ndarray]:
    """Extract L2-normalized factory source directions w_0 for V2/V2-grid and V4."""
    model = onnx.load(str(root / "data/raw/tinyvision.onnx"))
    inits = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    w_v2 = (inits["classifier.1.weight"][1] - inits["classifier.1.weight"][0]).astype(np.float64)
    w_v2 /= np.linalg.norm(w_v2)
    with zipfile.ZipFile(root / "results/second_backbone_heads.zip") as zf:
        blob = zf.read("source.bin")
        _, _, scale, _ = I8.unpack_from(blob)
        w_v4 = np.frombuffer(blob, dtype=np.int8, offset=I8.size).astype(np.float64) * scale
        w_v4 /= np.linalg.norm(w_v4)
    return {"v2": w_v2, "v2grid": w_v2, "v4": w_v4}


def amortized_label_count(
    cap_bytes: int,
    label_cap: int,
    reservoir_receipts: np.ndarray,
    fixed_overhead: int,
    grant_mode: str,
    rho_prepaid: float = RHO_PREPAID,
    rho_earned: float = RHO_EARNED,
    min_labels: int = MIN_LABELS,
) -> int:
    """Compute horizon-amortized paid label budget k using only unlabelled reservoir sizes."""
    if grant_mode not in ("prepaid", "earned"):
        raise ValueError(f"Unsupported grant mode: {grant_mode}")
    rho = rho_prepaid if grant_mode == "prepaid" else rho_earned
    mean_receipt = float(np.mean(reservoir_receipts))
    if mean_receipt <= 0 or cap_bytes <= fixed_overhead:
        return 0
    raw_k = int(np.floor((rho * cap_bytes - fixed_overhead) / mean_receipt))
    # Cap setup at most 50% of window bytes even when clamping up to min_labels
    max_half = int(np.floor((0.5 * cap_bytes - fixed_overhead) / mean_receipt))
    if max_half < 8:
        return 0
    k = max(min_labels, min(label_cap, raw_k))
    k = min(k, label_cap, max_half)
    return max(0, k)


def select_quantile_stratified(source_probs: np.ndarray, k: int) -> np.ndarray:
    """Label-free quantile-stratified selection from unlabelled reservoir scores."""
    m = len(source_probs)
    if k <= 0:
        return np.empty(0, dtype=int)
    if k >= m:
        return np.arange(m, dtype=int)
    order = np.argsort(source_probs, kind="stable")
    bins = np.array_split(order, k)
    return np.array([int(b[len(b) // 2]) for b in bins], dtype=int)


def fit_asar_head(x_norm: np.ndarray, y: np.ndarray, w0: np.ndarray, lam: float) -> bytes:
    """Fit Anisotropic Source-Anchored Regularized (ASAR) logistic head in EQ01 INT8 format.

    Objective: sum_i omega_i * ell(y_i, w^T x_i + b) + 0.5 * (||Pi_0 w||^2 + lam * ||Pi_perp w||^2)
    where Pi_0 projects onto the unit source head w0 and Pi_perp = I - Pi_0.
    At lam = 1, reduces identically to standard isotropic L2 LogisticRegression(C=1).
    At lam = inf, reduces to 2-parameter Platt recalibration along w0.
    """
    y = np.asarray(y, dtype=np.int32)
    w = train_weights(y, "balanced")
    if len(np.unique(y)) < 2:
        return pack_head("local_int8", np.zeros(x_norm.shape[1]), 1.0 if y[0] else -1.0)
    z0 = x_norm @ w0
    if np.isinf(lam):
        model = LogisticRegression(C=10.0, solver="liblinear", max_iter=3000, tol=1e-8, random_state=0)
        model.fit(z0[:, None], y, sample_weight=w)
        a = max(0.0, float(model.coef_[0, 0]))
        return pack_head("local_int8", a * w0, float(model.intercept_[0]))
    if lam == 1.0:
        model = LogisticRegression(C=1.0, solver="liblinear", max_iter=3000, tol=1e-8, random_state=0)
        model.fit(x_norm, y, sample_weight=w)
        return pack_head("local_int8", model.coef_[0], float(model.intercept_[0]))
    inv_sqrt = 1.0 / np.sqrt(lam)
    proj = np.outer(z0, w0)
    x_scaled = proj + inv_sqrt * (x_norm - proj)
    model = LogisticRegression(C=1.0, solver="liblinear", max_iter=3000, tol=1e-8, random_state=0)
    model.fit(x_scaled, y, sample_weight=w)
    u = model.coef_[0]
    u_par = float(np.dot(u, w0)) * w0
    w_final = u_par + inv_sqrt * (u - u_par)
    return pack_head("local_int8", w_final, float(model.intercept_[0]))


def select_asar_head(
    q_cal: np.ndarray,
    y_cal: np.ndarray,
    w0: np.ndarray,
    seed: int,
    grid: tuple[float, ...] = LAMBDA_GRID,
    occam_margin: float = OCCAM_MARGIN,
) -> tuple[bytes, float]:
    """Select shrinkage strength lambda via temperature-invariant stratified OOF BA+AUROC."""
    x_norm = normalized(q_cal)
    y_cal = np.asarray(y_cal, dtype=np.int32)
    if len(np.unique(y_cal)) < 2:
        return pack_head("local_int8", np.zeros(q_cal.shape[1]), 1.0 if y_cal[0] else -1.0), float("inf")
    if len(grid) == 1:
        return fit_asar_head(x_norm, y_cal, w0, grid[0]), float(grid[0])
    minority = int(np.bincount(y_cal, minlength=2).min())
    if minority < 2:
        return fit_asar_head(x_norm, y_cal, w0, float("inf")), float("inf")
    n_splits = min(4, minority)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(skf.split(x_norm, y_cal))
    best_lam = float(grid[0])
    best_score = -1.0
    for lam in grid:
        oof = np.empty(len(y_cal), dtype=np.float64)
        for tr, va in folds:
            head = fit_asar_head(x_norm[tr], y_cal[tr], w0, lam)
            oof[va] = predict_from_head("local_int8", head, q_cal[va])
        ba = float(balanced_accuracy_score(y_cal, oof >= 0.0))
        auc = float(roc_auc_score(y_cal, oof))
        score = 0.5 * ba + 0.5 * auc
        # Simplest-first Occam's razor: move to a less anchored head only if OOF improves by > margin
        if best_score < 0.0 or score > best_score + occam_margin:
            best_score = score
            best_lam = float(lam)
    return fit_asar_head(x_norm, y_cal, w0, best_lam), best_lam


def build_basa_plan(
    d: dict,
    seed: int,
    cap_bytes: int,
    label_cap: int,
    grant_mode: str,
    w0: np.ndarray,
    *,
    use_amortization: bool = True,
    use_stratified: bool = True,
    use_asar: bool = True,
) -> BASAPlan:
    """Build a strictly causal BASA acquisition and adaptation plan for one seed."""
    res_idx = d["cal_draws"][seed][:label_cap]
    p_res = d["cal_p"][res_idx]
    sz_res = d["cal_size"][res_idx].astype(int)
    width = int(d["cal_x"].shape[1])
    receipts_res = sz_res + width + 37
    head_bytes = width + 14
    fixed_overhead = REQUEST_BYTES + ACTION_BYTES + head_bytes

    if use_amortization:
        k = amortized_label_count(cap_bytes, label_cap, receipts_res, fixed_overhead, grant_mode)
    else:
        k = label_cap

    if k <= 0:
        return BASAPlan(0, 0, (), float("inf"), b"", fallback_frozen=True)

    def pick(count: int) -> np.ndarray:
        if use_stratified:
            return select_quantile_stratified(p_res, count)
        return np.arange(count, dtype=int)

    sel_local = pick(k)
    setup = int(receipts_res[sel_local].sum()) + fixed_overhead
    if not use_amortization and setup > cap_bytes:
        return BASAPlan(0, 0, (), float("inf"), b"", fallback_frozen=True)
    while setup > cap_bytes and k > 8:
        k -= 1
        sel_local = pick(k)
        setup = int(receipts_res[sel_local].sum()) + fixed_overhead
    if setup > cap_bytes:
        return BASAPlan(0, 0, (), float("inf"), b"", fallback_frozen=True)

    chosen_ids = res_idx[sel_local]
    q_cal = quantized(d["cal_x"][chosen_ids])[0]
    y_cal = d["cal_y"][chosen_ids]
    grid = LAMBDA_GRID if use_asar else (1.0,)
    head_blob, chosen_lam = select_asar_head(q_cal, y_cal, w0, seed, grid=grid)
    return BASAPlan(
        k=int(k),
        setup_bytes=int(setup),
        selected_local=tuple(int(i) for i in sel_local),
        chosen_lambda=float(chosen_lam),
        head_blob=head_blob,
        fallback_frozen=False,
    )


VARIANTS = {
    "basa": dict(use_amortization=True, use_stratified=True, use_asar=True, guard=False),
    "basa_guard": dict(use_amortization=True, use_stratified=True, use_asar=True, guard=True),
    "basa_no_asar": dict(use_amortization=True, use_stratified=True, use_asar=False, guard=False),
    "basa_no_strat": dict(use_amortization=True, use_stratified=False, use_asar=True, guard=False),
    "basa_no_amort": dict(use_amortization=False, use_stratified=True, use_asar=True, guard=False),
}


def bootstrap_paired_ci(
    diff_by_seed_order: np.ndarray,
    group_ids: np.ndarray,
    n_boot: int = 1000,
    seed: int = 20261127,
) -> tuple[float, float, float]:
    """Conditional paired 95% bootstrap interval over image/day groups (matches repo protocol)."""
    # diff_by_seed_order: shape (R, N_test) per-image balanced-accuracy contribution difference
    # Group by unique groups (classes on Cornell/Kashmir handled externally; here day/item groups)
    rng = np.random.default_rng(seed)
    mean_diff = float(diff_by_seed_order.mean())
    uniq, inv = np.unique(group_ids, return_inverse=True)
    g = len(uniq)
    group_means = np.zeros((diff_by_seed_order.shape[0], g), dtype=np.float64)
    counts = np.bincount(inv)
    for idx in range(g):
        group_means[:, idx] = diff_by_seed_order[:, inv == idx].sum(axis=1)
    total_n = diff_by_seed_order.shape[1]
    boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        draw = rng.integers(0, g, size=g)
        denom = counts[draw].sum()
        boot[b] = group_means[:, draw].sum() / (diff_by_seed_order.shape[0] * denom) if denom else mean_diff
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return mean_diff, float(lo), float(hi)


def run_benchmark(out_dir: Path = OUT, seeds: int = 20, pcts: tuple[int, ...] = (10, 25, 50)) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    w0_map = load_factory_source_heads(ROOT)
    lib = library()

    all_rows = []
    summary_rows = []
    paired_ci_rows = []

    # Load existing baseline summary for direct side-by-side audit
    with (ROOT / "results/purchase/summary.csv").open(newline="") as f:
        base_summary = {
            (r["site"], r["backbone"], int(r["label_cap"]), int(r["cap_pct"]), r["grant_mode"], r["method"]): r
            for r in csv.DictReader(f)
        }

    for site in SITES:
        for bb in BACKBONES:
            d = load(site, bb)
            n = len(d["test_y"])
            q_test = quantized(d["test_x"])[0]
            src_test = (d["test_p"] >= np.float32(d["cutoff"])).astype(np.uint8)
            src_urg = -np.abs(d["test_p"].astype(float) - float(np.float32(d["cutoff"])))
            size_test = (d["test_size"] + 20).astype(np.int32)
            cloud_test = (d["test_cloud_p"] >= 0.5).astype(np.uint8)
            y_test = d["test_y"].astype(bool)
            n_pos = int(y_test.sum())
            n_neg = n - n_pos
            orders = d["test_orders"]
            inv_orders = [np.argsort(o) for o in orders]
            w0 = w0_map[bb]

            # Load saved baseline masks for exact paired bootstrap CIs at 25%
            masks_path = ROOT / "results/purchase" / f"masks_{site}_{bb}.npz"
            runs_path = ROOT / "results/purchase" / f"runs_{site}_{bb}.csv"
            base_pred_masks = None
            base_run_index = {}
            if masks_path.exists() and runs_path.exists():
                with np.load(masks_path, allow_pickle=False) as mz:
                    base_pred_masks = mz["pred"]
                with runs_path.open(newline="") as rf:
                    for rr in csv.DictReader(rf):
                        if rr["cap_pct"] == "25" and rr["method"] in (
                            "freeze",
                            "adaptive",
                            "always_edgeboost_rank",
                            "always_ova",
                            "always_ce",
                            "probe16",
                        ):
                            key = (
                                int(rr["seed"]),
                                int(rr["order"]),
                                int(rr["label_cap"]),
                                rr["grant_mode"],
                                rr["method"],
                            )
                            base_run_index[key] = int(rr["mask_row"])

            for pct in pcts:
                rate = FULL_RATE * pct // 100
                cap_bytes = n * rate
                for label_cap in (32, 128):
                    for mode in ("prepaid", "earned"):
                        basa_preds_cell = []
                        basa_keys_cell = []
                        for vname, cfg in VARIANTS.items():
                            # Run ablations & guard at primary 25%; run core basa at all pcts
                            if pct != 25 and vname != "basa":
                                continue
                            cell_rows = []
                            for seed in range(seeds):
                                plan = build_basa_plan(
                                    d,
                                    seed,
                                    cap_bytes,
                                    label_cap,
                                    mode,
                                    w0,
                                    use_amortization=cfg["use_amortization"],
                                    use_stratified=cfg["use_stratified"],
                                    use_asar=cfg["use_asar"],
                                )
                                if plan.fallback_frozen:
                                    local_pred = src_test
                                    prio = src_urg
                                else:
                                    z = predict_from_head("local_int8", plan.head_blob, q_test)
                                    local_pred = (z >= 0.0).astype(np.uint8)
                                    prio = -np.abs(z)
                                for oi, idx in enumerate(orders):
                                    p_ord = array(prio[idx], np.float64)
                                    args = (
                                        array(size_test[idx], np.int32),
                                        array(src_test[idx], np.uint8),
                                        array(local_pred[idx], np.uint8),
                                        p_ord,
                                        ranks(p_ord),
                                        array(cloud_test[idx], np.uint8),
                                    )
                                    pred = np.empty(n, np.uint8)
                                    ask = np.empty_like(pred)
                                    stats = np.zeros(4, np.int64)
                                    err = lib.replay(
                                        n,
                                        rate,
                                        int(mode == "earned"),
                                        plan.setup_bytes,
                                        int(cfg["guard"]),
                                        0,
                                        float("inf"),
                                        *args,
                                        pred,
                                        ask,
                                        stats,
                                    )
                                    if err != 0:
                                        raise RuntimeError(f"Replay error {err} on {site}/{bb}/{vname}")
                                    pred_orig = pred[inv_orders[oi]].astype(bool)
                                    ask_orig = ask[inv_orders[oi]].astype(bool)
                                    if cfg["guard"] and np.any(src_test.astype(bool) & ~pred_orig):
                                        raise RuntimeError("Recall guard withdrew a native alert")
                                    review, calls, start, deficit = map(int, stats)
                                    assert review == int(size_test[ask_orig].sum())
                                    assert int(ask_orig.sum()) == calls
                                    assert plan.setup_bytes + review <= cap_bytes and deficit == 0
                                    fp = int(np.sum(pred_orig & ~y_test))
                                    fn = int(np.sum(~pred_orig & y_test))
                                    ba = 1.0 - (fp / n_neg + fn / n_pos) / 2.0
                                    acc = 1.0 - (fp + fn) / n
                                    row = dict(
                                        site=site,
                                        backbone=bb,
                                        label_cap=label_cap,
                                        cap_pct=pct,
                                        grant_mode=mode,
                                        method=vname,
                                        seed=seed,
                                        order=oi,
                                        fp=fp,
                                        fn=fn,
                                        ba=ba,
                                        accuracy=acc,
                                        labels=plan.k,
                                        setup_bytes=plan.setup_bytes,
                                        review_bytes=review,
                                        total_bytes=plan.setup_bytes + review,
                                        queries=calls,
                                        install_frame=start,
                                        chosen_lambda="inf" if np.isinf(plan.chosen_lambda) else f"{plan.chosen_lambda:g}",
                                        is_anchored=int(plan.chosen_lambda > 1.0),
                                        is_platt=int(np.isinf(plan.chosen_lambda)),
                                        fallback=int(plan.fallback_frozen),
                                        prefix_overdraft=deficit,
                                    )
                                    cell_rows.append(row)
                                    all_rows.append(row)
                                    if pct == 25 and vname == "basa":
                                        basa_preds_cell.append(pred_orig)
                                        basa_keys_cell.append((seed, oi))

                            agg = dict(
                                site=site,
                                backbone=bb,
                                label_cap=label_cap,
                                cap_pct=pct,
                                grant_mode=mode,
                                method=vname,
                                runs=len(cell_rows),
                            )
                            for field in (
                                "fp",
                                "fn",
                                "ba",
                                "accuracy",
                                "labels",
                                "setup_bytes",
                                "review_bytes",
                                "total_bytes",
                                "queries",
                                "install_frame",
                                "is_anchored",
                                "is_platt",
                                "fallback",
                                "prefix_overdraft",
                            ):
                                agg[field] = sum(float(r[field]) for r in cell_rows) / len(cell_rows)
                            summary_rows.append(agg)

                        # Compute paired 95% bootstrap intervals for BASA vs key baselines at 25%
                        if pct == 25 and base_pred_masks is not None:
                            basa_mat = np.stack(basa_preds_cell)  # (R, N)
                            # Per-sample BA weight: +1/(2*n_pos) for correct pos, +1/(2*n_neg) for correct neg
                            sample_w = np.where(y_test, 0.5 / n_pos, 0.5 / n_neg)
                            groups = d["test_group"]
                            rng = np.random.default_rng(20261127)
                            for comp in ("freeze", "adaptive", "always_edgeboost_rank", "always_ova", "always_ce", "probe16"):
                                comp_rows = [
                                    np.unpackbits(base_pred_masks[base_run_index[(s, o, label_cap, mode, comp)]])[:n].astype(bool)
                                    for (s, o) in basa_keys_cell
                                ]
                                comp_mat = np.stack(comp_rows)
                                delta_correct = (basa_mat == y_test[None, :]).astype(np.float64) - (
                                    comp_mat == y_test[None, :]
                                ).astype(np.float64)
                                # Exact mean BA diff in percentage points
                                mean_pp = float(np.mean(delta_correct @ sample_w) * 100.0)
                                # Stratified within-class bootstrap for Cornell/Kashmir, day-cluster for Latvia
                                boot = np.empty(1000, dtype=np.float64)
                                if site == "latvia":
                                    uniq_days, inv_days = np.unique(groups, return_inverse=True)
                                    g_cnt = len(uniq_days)
                                    # Pre-aggregate TP/TN deltas and pos/neg counts per day
                                    pos_mask = y_test
                                    neg_mask = ~y_test
                                    day_dp = np.stack([delta_correct[:, (inv_days == i) & pos_mask].sum(axis=1) for i in range(g_cnt)], axis=1)
                                    day_dn = np.stack([delta_correct[:, (inv_days == i) & neg_mask].sum(axis=1) for i in range(g_cnt)], axis=1)
                                    day_np = np.array([((inv_days == i) & pos_mask).sum() for i in range(g_cnt)])
                                    day_nn = np.array([((inv_days == i) & neg_mask).sum() for i in range(g_cnt)])
                                    for b in range(1000):
                                        dr = rng.integers(0, g_cnt, size=g_cnt)
                                        sp = day_np[dr].sum()
                                        sn = day_nn[dr].sum()
                                        term_p = day_dp[:, dr].sum(axis=1) / sp if sp else 0.0
                                        term_n = day_dn[:, dr].sum(axis=1) / sn if sn else 0.0
                                        boot[b] = float(np.mean(0.5 * (term_p + term_n)) * 100.0)
                                else:
                                    pos_idx = np.flatnonzero(y_test)
                                    neg_idx = np.flatnonzero(~y_test)
                                    dc_pos = delta_correct[:, pos_idx].mean(axis=0)
                                    dc_neg = delta_correct[:, neg_idx].mean(axis=0)
                                    for b in range(1000):
                                        dp = dc_pos[rng.integers(0, len(pos_idx), size=len(pos_idx))].mean()
                                        dn = dc_neg[rng.integers(0, len(neg_idx), size=len(neg_idx))].mean()
                                        boot[b] = 50.0 * (dp + dn)
                                lo, hi = np.percentile(boot, [2.5, 97.5])
                                paired_ci_rows.append(
                                    dict(
                                        site=site,
                                        backbone=bb,
                                        label_cap=label_cap,
                                        grant_mode=mode,
                                        comparator=comp,
                                        basa_ba_pct=100.0 * float(np.mean((basa_mat == y_test[None, :]) @ sample_w)),
                                        comp_ba_pct=100.0 * float(np.mean((comp_mat == y_test[None, :]) @ sample_w)),
                                        mean_diff_pp=mean_pp,
                                        ci95_lo_pp=float(lo),
                                        ci95_hi_pp=float(hi),
                                    )
                                )
            print(f"Finished {site} / {bb}", flush=True)

    def write_csv(path: Path, rows: list[dict]) -> None:
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write_csv(out_dir / "runs.csv", all_rows)
    write_csv(out_dir / "summary.csv", summary_rows)
    write_csv(out_dir / "paired_ci.csv", paired_ci_rows)

    # Print executive summary
    print("\n=== BASA EXECUTIVE SUMMARY (36 primary cells, 25% byte rate) ===")
    for comp in ("adaptive", "freeze", "always_edgeboost_rank", "always_ova", "always_ce", "probe16"):
        sub = [r for r in paired_ci_rows if r["comparator"] == comp]
        wins = sum(1 for r in sub if r["mean_diff_pp"] > 1e-9)
        sig_wins = sum(1 for r in sub if r["ci95_lo_pp"] > 0)
        sig_loss = sum(1 for r in sub if r["ci95_hi_pp"] < 0)
        mean_d = np.mean([r["mean_diff_pp"] for r in sub])
        print(
            f"vs {comp:22s}: wins={wins:2d}/36 (sig+={sig_wins:2d}, sig-={sig_loss:2d}), mean={mean_d:+.2f} pp"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()
    run_benchmark(args.out, args.seeds)

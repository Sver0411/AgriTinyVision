"""Evaluate budget-clamped versions of published baselines (Clamped-CE, Clamped-EB-rank, Clamped-OvA).

Eliminates the Cap=128 budget-crash strawman by clamping each baseline's paid label count
to k = min(label_cap, floor((0.5 * B - fixed_overhead) / mean_receipt)) whenever the raw
label_cap would exceed 50% of the window byte quota, and replays on the identical C99 engine.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from acquisition_baselines import download_byte_count, fit_router
from edge_adaptation_wire import quantized
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
OUT = ROOT / "results" / "basa" / "clamped_baselines.csv"
PAIRED_OUT = ROOT / "results" / "basa" / "clamped_paired_ci.csv"
CLAMPED_KINDS = ("ce", "edgeboost_rank", "ova")


def clamped_k(cap_bytes: int, label_cap: int, receipts: np.ndarray, fixed_overhead: int) -> int:
    mean_r = float(np.mean(receipts))
    if mean_r <= 0 or cap_bytes <= fixed_overhead:
        return 0
    k_half = int(math.floor((0.5 * cap_bytes - fixed_overhead) / mean_r))
    if k_half < 8:
        return 0
    return min(label_cap, k_half)


def paired_bootstrap_ci(
    diff_runs: np.ndarray,
    group_ids: np.ndarray,
    n_boot: int = 1000,
    seed: int = 20261127,
) -> tuple[float, float, float]:
    """Paired bootstrap over image/day groups (matching basa_controller.py protocol)."""
    rng = np.random.default_rng(seed)
    mean_diff = float(diff_runs.mean())
    uniq, inv = np.unique(group_ids, return_inverse=True)
    g = len(uniq)
    gsum = np.zeros((diff_runs.shape[0], g), dtype=np.float64)
    counts = np.bincount(inv)
    for idx in range(g):
        gsum[:, idx] = diff_runs[:, inv == idx].sum(axis=1)
    boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        draw = rng.integers(0, g, size=g)
        denom = counts[draw].sum()
        boot[b] = gsum[:, draw].sum() / (diff_runs.shape[0] * denom) if denom else mean_diff
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return mean_diff, float(lo), float(hi)


def main() -> None:
    lib = library()
    basa_runs = list(csv.DictReader((ROOT / "results/basa/runs.csv").open(newline="")))
    basa_map: dict[tuple, float] = {}
    for r in basa_runs:
        if r["method"] == "basa":
            key = (
                r["site"],
                r["backbone"],
                int(r["label_cap"]),
                int(r["cap_pct"]),
                r["grant_mode"],
                int(r["seed"]),
                int(r["order"]),
            )
            basa_map[key] = float(r["ba"])

    summary_rows = []
    paired_rows = []
    seeds = 20

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
            width = int(d["cal_x"].shape[1])

            for pct in (10, 25, 50):
                rate = FULL_RATE * pct // 100
                cap_bytes = n * rate
                for label_cap in (32, 128):
                    for mode in ("prepaid", "earned"):
                        for kind in CLAMPED_KINDS:
                            dl_bytes = download_byte_count(kind, width)
                            fixed_overhead = REQUEST_BYTES + ACTION_BYTES + dl_bytes
                            bas = []
                            diffs_pp = []
                            ks = []
                            for seed in range(seeds):
                                res_idx = d["cal_draws"][seed][:label_cap]
                                receipts = d["cal_size"][res_idx].astype(int) + width + 37
                                k = clamped_k(cap_bytes, label_cap, receipts, fixed_overhead)
                                while k >= 8 and int(receipts[:k].sum()) + fixed_overhead > cap_bytes:
                                    k -= 1
                                if k < 8:
                                    setup = 0
                                    local_pred = src_test
                                    prio = src_urg
                                    used_k = 0
                                else:
                                    used_k = k
                                    setup = int(receipts[:k].sum()) + fixed_overhead
                                    ids = res_idx[:k]
                                    x_tr = quantized(d["cal_x"][ids])[0]
                                    y_tr = d["cal_y"][ids]
                                    c_tr = d["cal_cloud_p"][ids] >= 0.5
                                    model = fit_router(kind, x_tr, y_tr, c_tr, seed)
                                    z, prio = model.score(q_test)
                                    local_pred = (z >= 0.0).astype(np.uint8)
                                    prio = np.asarray(prio, dtype=np.float64)
                                ks.append(used_k)
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
                                        setup,
                                        0,
                                        0,
                                        float("inf"),
                                        *args,
                                        pred,
                                        ask,
                                        stats,
                                    )
                                    if err != 0:
                                        raise RuntimeError(f"Replay err {err}")
                                    pred_orig = pred[inv_orders[oi]].astype(bool)
                                    fp = int(np.sum(pred_orig & ~y_test))
                                    fn = int(np.sum(~pred_orig & y_test))
                                    ba = 1.0 - (fp / n_neg + fn / n_pos) / 2.0
                                    bas.append(ba)
                                    b_ba = basa_map[(site, bb, label_cap, pct, mode, seed, oi)]
                                    diffs_pp.append((b_ba - ba) * 100.0)
                            mean_ba = float(np.mean(bas))
                            mean_diff = float(np.mean(diffs_pp))
                            summary_rows.append(
                                dict(
                                    site=site,
                                    backbone=bb,
                                    label_cap=label_cap,
                                    cap_pct=pct,
                                    grant_mode=mode,
                                    method=f"clamped_{kind}",
                                    ba=f"{mean_ba:.6f}",
                                    mean_k=f"{np.mean(ks):.2f}",
                                    basa_minus_clamped_pp=f"{mean_diff:.4f}",
                                )
                            )

    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    print(f"Wrote {len(summary_rows)} rows to {OUT}")
    for pct in (10, 25, 50):
        print(f"\n=== Uplink {pct}% (All 36 cells) ===")
        for kind in CLAMPED_KINDS:
            m = f"clamped_{kind}"
            all36 = [float(r["ba"]) * 100 for r in summary_rows if int(r["cap_pct"]) == pct and r["method"] == m]
            pre18 = [float(r["ba"]) * 100 for r in summary_rows if int(r["cap_pct"]) == pct and r["method"] == m and r["grant_mode"] == "prepaid"]
            ear18 = [float(r["ba"]) * 100 for r in summary_rows if int(r["cap_pct"]) == pct and r["method"] == m and r["grant_mode"] == "earned"]
            cap32 = [float(r["ba"]) * 100 for r in summary_rows if int(r["cap_pct"]) == pct and r["method"] == m and int(r["label_cap"]) == 32]
            cap128 = [float(r["ba"]) * 100 for r in summary_rows if int(r["cap_pct"]) == pct and r["method"] == m and int(r["label_cap"]) == 128]
            wins = sum(1 for r in summary_rows if int(r["cap_pct"]) == pct and r["method"] == m and float(r["basa_minus_clamped_pp"]) > 1e-9)
            print(f"  {m:22s}: All36={np.mean(all36):.2f} | Pre={np.mean(pre18):.2f} | Earn={np.mean(ear18):.2f} | Cap32={np.mean(cap32):.2f} | Cap128={np.mean(cap128):.2f} | BASA wins={wins}/36")


if __name__ == "__main__":
    main()

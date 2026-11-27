"""Verify every reported BASA table and headline claim against public aggregate CSVs (stdlib only)."""

from __future__ import annotations

import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASA_DIR = ROOT / "results" / "basa"
BASE_DIR = ROOT / "results" / "purchase"


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def close(val: float, expected: float, tol: float = 0.015) -> None:
    if not math.isclose(val, expected, rel_tol=0.0, abs_tol=tol):
        raise AssertionError(f"Observed {val:.4f}, expected {expected:.4f} ± {tol}")


def main() -> None:
    basa = load_csv(BASA_DIR / "summary.csv")
    base = load_csv(BASE_DIR / "summary.csv")
    paired = load_csv(BASA_DIR / "paired_ci.csv")

    assert all(float(r["prefix_overdraft"]) == 0.0 for r in basa), "Prefix overdraft detected!"

    # 1. Check 36-cell overall means at 25% byte rate
    def mean36(rows, method, pct=25, mode=None):
        vals = [
            float(r["ba"]) * 100.0
            for r in rows
            if int(r["cap_pct"]) == pct and r["method"] == method and (mode is None or r["grant_mode"] == mode)
        ]
        assert len(vals) == (36 if mode is None else 18)
        return sum(vals) / len(vals)

    close(mean36(basa, "basa"), 73.50)
    close(mean36(basa, "basa", mode="prepaid"), 74.49)
    close(mean36(basa, "basa", mode="earned"), 72.52)
    close(mean36(base, "freeze"), 68.93)
    close(mean36(base, "adaptive"), 68.62)
    close(mean36(base, "always_edgeboost_rank"), 71.84)
    close(mean36(base, "always_ova"), 71.78)
    close(mean36(base, "always_ce"), 72.04)

    # 2. Check bandwidth sweep (10%, 25%, 50%)
    for pct, exp in ((10, 70.49), (25, 73.50), (50, 75.29)):
        close(mean36(basa, "basa", pct=pct), exp)

    # 3. Check ablation means
    for method, exp in (("basa", 73.50), ("basa_no_asar", 73.36), ("basa_no_strat", 73.63), ("basa_no_amort", 71.99)):
        close(mean36(basa, method), exp)

    # 4. Check paired win/sig counts
    expected_wins = {
        "adaptive": (34, 24, 0),
        "freeze": (33, 20, 0),
        "always_edgeboost_rank": (27, 16, 5),
        "always_ova": (26, 16, 5),
        "always_ce": (24, 13, 7),
        "probe16": (21, 14, 8),
    }
    for comp, (w_exp, sp_exp, sn_exp) in expected_wins.items():
        sub = [r for r in paired if r["comparator"] == comp]
        assert len(sub) == 36
        w = sum(1 for r in sub if float(r["mean_diff_pp"]) > 1e-9)
        sp = sum(1 for r in sub if float(r["ci95_lo_pp"]) > 0.0)
        sn = sum(1 for r in sub if float(r["ci95_hi_pp"]) < 0.0)
        assert (w, sp, sn) == (w_exp, sp_exp, sn_exp), (comp, (w, sp, sn), (w_exp, sp_exp, sn_exp))

    # 5. Check Latvia Recall Guard at cap=32, 25%, earned
    bmap = {(r["site"], r["backbone"], int(r["label_cap"]), int(r["cap_pct"]), r["grant_mode"], r["method"]): r for r in basa}
    close(float(bmap[("latvia", "v2", 32, 25, "earned", "basa_guard")]["fn"]), 23.9, 0.1)
    close(float(bmap[("latvia", "v2grid", 32, 25, "earned", "basa_guard")]["fn"]), 28.1, 0.1)

    # 6. Check real ESP32-S3 hardware benchmark telemetry
    import json
    hw = json.loads((BASA_DIR / "esp32s3_hardware_bench.json").read_text())
    assert hw["idf_version"] == "v5.4.4" and hw["cpu_mhz"] == 160
    assert hw["mismatches"] == 0 and hw["max_abs_err"] == 0.0
    assert hw["eq01_head_struct_bytes"] == 1292 and hw["stream_gate_struct_bytes"] == 544
    assert hw["infer_us"] == [93, 93, 99] and hw["gate_us"] == [41, 45, 45]
    assert hw["parse_us"] == [50, 50, 50] and hw["quant_us"] == [565, 573, 573]

    print("PASS: Verified all BASA primary tables, 10/25/50% sweep, ablations, paired 95% CIs, Latvia guard, and ESP32-S3 silicon telemetry.")


if __name__ == "__main__":
    main()

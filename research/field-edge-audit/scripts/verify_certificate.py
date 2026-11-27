"""Independent integer/rational audit for a fixed-pool binary shadow-inference design.

Run python3 scripts/verify_certificate.py [--max-n 9] [--compare-primary].
Only the optional last stage imports certify.py; all reference calculations and
exhaustive checks are computed first. The null has no positive/negative sign.
"""

import argparse
import importlib.util
import json
import math
from collections import Counter
from fractions import Fraction
from functools import lru_cache
from itertools import combinations
from pathlib import Path

ALPHA = Fraction(1, 20)


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def choose(total, count):
    return math.comb(total, count) if 0 <= count <= total else 0


def possible(d, r, n, x):
    return choose(r, x) * choose(d - r, n - x)


@lru_cache(None)
def reference_interval(d, n, x, alpha=ALPHA):
    """Invert both exact discrete tails on integer candidate counts R."""
    if not (0 <= x <= n <= d and 0 < alpha < 1):
        raise ValueError("Invalid population, sample, observation or level")
    retained = []
    normalization = choose(d, n)
    for r in range(x, d - n + x + 1):
        lt = sum(possible(d, r, n, t) for t in range(x + 1))
        gt = sum(possible(d, r, n, t) for t in range(x, n + 1))
        # Avoid floating-point rounding when a tail is exactly alpha/2.
        if 2 * alpha.denominator * min(lt, gt) > alpha.numerator * normalization:
            retained.append(r)
    require(
        retained and retained == list(range(retained[0], retained[-1] + 1)),
        "Non-contiguous confidence set",
    )
    return retained[0], retained[-1]


def direction(d, interval):
    lo, hi = interval
    return 1 if 2 * lo > d else -1 if 2 * hi < d else 0


def allocation(n, d, r, b, design):
    """Return exact rational mass on the observed pair (sampled disagreements, reversals to correct)."""
    if not (n >= 1 and 0 <= r <= d <= n and 0 <= b <= n):
        raise ValueError("Invalid finite population or budget")
    if design == "disagreement":
        k = min(b, d)
        return {
            (k, x): Fraction(ways, choose(d, k))
            for x in range(k + 1)
            if (ways := possible(d, r, k, x))
        }
    if design != "uniform":
        raise ValueError(design)
    result = {}
    for k in range(max(0, b - (n - d)), min(b, d) + 1):
        p_k = Fraction(choose(d, k) * choose(n - d, b - k), choose(n, b))
        for x in range(k + 1):
            if ways := possible(d, r, k, x):
                result[k, x] = p_k * Fraction(ways, choose(d, k))
    return result


def exact_performance(d, r, probabilities):
    require(sum(probabilities.values()) == 1, "Distribution does not sum to one")
    coverage = positive = negative = Fraction(0)
    for (k, x), p in probabilities.items():
        ci = reference_interval(d, k, x)
        if ci[0] <= r <= ci[1]:
            coverage += p
        sign = direction(d, ci)
        if sign > 0:
            positive += p
        if sign < 0:
            negative += p
    correct = positive if 2 * r > d else negative if 2 * r < d else Fraction(0)
    return coverage, correct, positive, negative


def all_draws(n, d, r):
    """Count every subset of physical frame indices, without probability formulas."""
    disagreements = (1 << d) - 1
    corrections = (1 << r) - 1
    full = [Counter() for _ in range(n + 1)]
    direct = [Counter() for _ in range(d + 1)]
    for mask in range(1 << n):
        b = mask.bit_count()
        k = (mask & disagreements).bit_count()
        x = (mask & corrections).bit_count()
        full[b][k, x] += 1
        if b == k:
            direct[b][k, x] += 1
    return full, direct


def compare_probability(analytic, enumerated):
    denominator = sum(enumerated.values())
    require(len(analytic) == len(enumerated), "Wrong observation support")
    require(
        all(analytic[key] * denominator == value for key, value in enumerated.items()),
        "Exact law differs from subset enumeration",
    )


def exhaustive(max_n):
    worst = {design: (Fraction(1), None) for design in ("disagreement", "uniform")}
    comparisons = 0
    for n in range(1, max_n + 1):
        for d in range(n + 1):
            for r in range(d + 1):
                full, direct = all_draws(n, d, r)
                for b in range(n + 1):
                    for design, actual in (
                        ("uniform", full[b]),
                        ("disagreement", direct[min(b, d)]),
                    ):
                        distribution = allocation(n, d, r, b, design)
                        compare_probability(distribution, actual)
                        coverage, correct, pos, neg = exact_performance(
                            d, r, distribution
                        )
                        require(
                            coverage >= 1 - ALPHA,
                            f"Undercoverage {(n, d, r, b, design, coverage)}",
                        )
                        if 2 * r < d:
                            require(
                                pos <= ALPHA / 2,
                                "False-positive direction too frequent",
                            )
                        if 2 * r > d:
                            require(
                                neg <= ALPHA / 2,
                                "False-negative direction too frequent",
                            )
                        if 2 * r == d:
                            require(
                                pos + neg <= ALPHA and correct == 0, "Null directivity"
                            )
                        if coverage < worst[design][0]:
                            worst[design] = coverage, (n, d, r, b)
                        comparisons += 1
    return comparisons, {
        k: {"min_coverage": str(c), "at_N_D_R_b": a} for k, (c, a) in worst.items()
    }


def every_layout(max_n=6):
    """Exhaustively verify permutation invariance across every ternary layout."""
    cases = 0
    for n in range(1, max_n + 1):
        ids = range(n)
        for d in range(n + 1):
            for r in range(d + 1):
                base, _ = all_draws(n, d, r)
                for placement in combinations(ids, d):
                    dmask = sum(1 << i for i in placement)
                    for positions in combinations(placement, r):
                        rmask = sum(1 << i for i in positions)
                        hist = [Counter() for _ in range(n + 1)]
                        for draw in range(1 << n):
                            hist[draw.bit_count()][
                                (draw & dmask).bit_count(), (draw & rmask).bit_count()
                            ] += 1
                        require(hist == base, "Frame ordering changed the law")
                        cases += 1
    return cases


def base_regressions():
    require(reference_interval(0, 0, 0) == (0, 0), "No disagreements")
    require(reference_interval(7, 0, 0) == (0, 7), "No labels")
    require(reference_interval(7, 7, 3) == (3, 3), "Full audit")
    require(reference_interval(40, 1, 1) == (2, 40), "Exact alpha/2 tie: lower")
    require(reference_interval(40, 1, 0) == (0, 38), "Exact alpha/2 tie: upper")
    for design, expected in [
        ("disagreement", Fraction(1)),
        ("uniform", Fraction(1, 60)),
    ]:
        require(
            exact_performance(1, 1, allocation(60, 1, 1, 1, design))[1] == expected,
            "One-correctable-frame example",
        )
    # Identical reported scores may yield different correctness. The proof
    # requires the two deployed predictions (and target truth) to be binary.


def compare_primary(max_n, filename):
    spec = importlib.util.spec_from_file_location("primary_certificate", filename)
    primary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(primary)
    intervals = powers = 0
    for d in range(max_n + 1):
        for k in range(d + 1):
            for x in range(k + 1):
                ref = reference_interval(d, k, x)
                got = tuple(primary.interval(d, k, x, alpha=0.05))
                require(ref == got, f"Primary interval mismatch {(d, k, x, ref, got)}")
                intervals += 1
    for x in (0, 1):
        require(
            tuple(primary.interval(40, 1, x, alpha=0.05))
            == reference_interval(40, 1, x),
            f"Primary interval fails exact tail tie {(40, 1, x)}",
        )
    for n in range(1, max_n + 1):
        for d in range(n + 1):
            for r in range(d + 1):
                for b in range(n + 1):
                    for design in ("uniform", "disagreement"):
                        ref = exact_performance(d, r, allocation(n, d, r, b, design))[1]
                        got = primary.correct_sign_power(n, d, r, b, design=design)
                        require(
                            abs(float(ref) - got) < 1e-10,
                            f"Primary power mismatch {(n, d, r, b, design, float(ref), got)}",
                        )
                        powers += 1
    return {"intervals": intervals + 2, "powers": powers}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-n", type=int, default=10)
    parser.add_argument("--compare-primary", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if not 1 <= args.max_n <= 15:
        parser.error("--max-n must be between 1 and 15")
    base_regressions()
    cells, worst = exhaustive(args.max_n)
    layout_n = min(args.max_n, 6)
    result = {
        "max_population": args.max_n,
        "confidence_level": "19/20",
        "tail_rule": "retain each one-sided tail strictly above 1/40",
        "population_budget_design_cells": cells,
        "worst_observed": worst,
        "all_ternary_layouts_up_to": layout_n,
        "permutation_cases": every_layout(layout_n),
        "real_hardware_one_label_positive_power": {
            "disagreement": "1",
            "uniform": "1/60",
        },
    }
    if args.compare_primary:
        result["primary_agreement"] = compare_primary(
            args.max_n, Path(__file__).with_name("certify.py")
        )
    data = json.dumps(result, indent=2) + "\n"
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(data)
    print(data, end="")


if __name__ == "__main__":
    main()

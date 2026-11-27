"""Exact randomization probabilities for thinning an already fixed review policy.

Each frame eligible under the reviewed policy is independently sent with known
probability q. Images skipped by the randomizer keep the released edge decision.
Disease labels and model outputs are held fixed throughout the calculation.
"""

import csv
from pathlib import Path

import numpy as np
from scipy.stats import binom

from joint_error_evidence import reconstruct

RESULTS = Path(__file__).resolve().parents[1] / 'results'
QUERY_PROBABILITIES = (0.0, 0.10, 0.25, 0.50, 1.0)


def strict_reduction_probability(fixed, introduced, query_probability):
    if query_probability == 0:
        return 0.0
    if query_probability == 1:
        return float(fixed > introduced)
    counts = np.arange(int(fixed) + 1)
    return float(binom.pmf(counts, fixed, query_probability) @ binom.cdf(counts - 1, introduced, query_probability))


def analyze(rows):
    for r in rows:
        for q in QUERY_PROBABILITIES:
            probability = {}
            expected = {}
            for error in ('fp', 'fn'):
                fixed, introduced = r[f'{error}_fixed'], r[f'{error}_introduced']
                probability[error] = strict_reduction_probability(fixed, introduced, q)
                expected[error] = r[f'{error}_before'] + q * (introduced - fixed)
            yield dict(site=r['site'], rule=r['rule'], independent_query_probability=q,
                       n=r['n'], expected_queried_photos=q * r['photos_queried'],
                       expected_inference_application_bytes=q * r['inference_application_bytes'],
                       one_time_calibration_and_install_bytes=r['setup_application_bytes'],
                       expected_fp=expected['fp'], expected_fn=expected['fn'],
                       p_strict_fp_reduction=probability['fp'], p_strict_fn_reduction=probability['fn'],
                       p_strict_joint_reduction=probability['fp'] * probability['fn'])


def main():
    rows = list(analyze(reconstruct()))
    with (RESULTS / 'random_query_budget.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()

"""Exact small-score validation of the tie-preserving diagnostic threshold envelope."""

import itertools
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from dual_existing import error_counts, oracle_on_locked_scores


class DualAuditTest(unittest.TestCase):
    def test_oracle_against_individual_cutoffs(self):
        scores = np.array([1.0, 0.0, 2.0, 1.0, 2.0])
        for labels in itertools.product((0, 1), repeat=len(scores)):
            y = np.array(labels)
            rows = [
                error_counts(y, scores >= t)
                for t in (-np.inf, 0, 0.5, 1, 1.5, 2, np.inf)
            ]
            for max_fp in range(1 + sum(y == 0)):
                expected = min((fn, fp) for fp, fn in rows if fp <= max_fp)
                self.assertEqual(oracle_on_locked_scores(y, scores, max_fp), expected)

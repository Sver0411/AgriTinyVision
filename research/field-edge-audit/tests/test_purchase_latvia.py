import sys
import unittest
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from purchase_latvia import comparison, summarize


class AlertAuditTests(unittest.TestCase):
    def test_sham_matches_uniform_addition_and_withdrawal(self):
        y = np.array([True, False, True, True, False, False, True])
        source = np.array([False, False, True, True, True, False, False])
        changed = np.array([True, True, False, True, True, False, False])
        obs = comparison(y, changed[None], source[None])
        # Add two different alerts and withdraw one, uniformly in their eligible sets.
        possible = []
        for additions in combinations(np.flatnonzero(~source), 2):
            for withdrawals in combinations(np.flatnonzero(source), 1):
                possible.append(y[list(additions)].sum() - y[list(withdrawals)].sum())
        self.assertAlmostEqual(float(np.mean(possible)), obs["sham_expected_net_tp"])
        self.assertEqual(0,obs["net_fn"])
        self.assertEqual(1,obs["net_fp"])

    def test_single_class_date_does_not_impute_specificity(self):
        y = np.ones(3, dtype=bool)
        pred = np.array([[False, True, True]])
        row = summarize(y, pred, pred)
        self.assertEqual(1,row["fn"])
        self.assertEqual("",row["specificity"])
        self.assertEqual("",row["ba"])
        self.assertAlmostEqual(2/3,row["sensitivity"])


if __name__ == "__main__":
    unittest.main()

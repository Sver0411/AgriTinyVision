"""Boundary, wire and audited population checks for the locked directional rule."""

import csv
import json
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from directional_policy import Rule, CONFIG, pack, unpack, decide, select_queries


class DirectionalPolicyTest(unittest.TestCase):
    def test_boundaries_and_absent_reply(self):
        r = unpack(pack(Rule(True, True, 0.05, 0.95, 0.6, 0.3)))
        e = np.array([0, 0.05, 0.20, 0.30, 0.95, 1])
        d = np.array([np.nan, 0.6, 0.59, 0.3, 0.29, np.nan])
        self.assertEqual(CONFIG.size, 22)
        np.testing.assert_array_equal(select_queries(e, r), [0, 1, 1, 1, 1, 0])
        np.testing.assert_array_equal(decide(e, d, r), [0, 1, 0, 1, 0, 1])
        d[1] = np.nan
        with self.assertRaises(ValueError):
            decide(e, d, r)

    def test_operation_signs_and_transferred_blob(self):
        with (ROOT / "results/directional_predictions.csv").open() as f:
            pred = list(csv.DictReader(f))
        with (ROOT / "results/directional_outcomes.csv").open() as f:
            reports = list(csv.DictReader(f))
        with (ROOT / "results/directional_rule.json").open() as f:
            r = json.load(f)
        binary = (ROOT / "results/directional_rule.bin").read_bytes()
        self.assertEqual(unpack(binary), Rule(**r["rule"]))
        for report in reports:
            selected = [v for v in pred if v["cohort"] == report["cohort"]]
            # Scalar replay of both independent actions without importing the deployment function.
            for v in selected:
                old = bool(int(v["original"]))
                ask = bool(int(v["queried"]))
                if ask:
                    p = float(v["reply_p"])
                    expected = (
                        p >= r["rule"]["rescue_cutoff"]
                        if not old
                        else p >= r["rule"]["retract_cutoff"]
                    )
                else:
                    expected = old
                self.assertEqual(bool(int(v["decision"])), expected)
            self.assertEqual(
                int(report["orig_fp"])
                + int(report["false_rescues"])
                - int(report["true_retractions"]),
                int(report["fp"]),
            )
            self.assertEqual(
                int(report["orig_fn"])
                - int(report["true_rescues"])
                + int(report["false_retractions"]),
                int(report["fn"]),
            )

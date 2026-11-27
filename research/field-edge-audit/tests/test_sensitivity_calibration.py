"""Regression checks for defining and reproducing the empirical recall operating point."""

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from sensitivity_calibration import choose_cutoff


class SensitivityTests(unittest.TestCase):
    def test_rank_tie_and_missing(self):
        self.assertTrue(np.isneginf(choose_cutoff([1, 8], [0, 0])))
        self.assertEqual(choose_cutoff([2, 9, 9], [1, 0, 0]), 2)
        scores = np.array(
            [1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        )
        threshold = choose_cutoff(scores, np.ones(len(scores)))
        self.assertEqual(threshold, 1)
        self.assertLess(np.mean(scores > threshold), 0.95)
        self.assertEqual(choose_cutoff(np.arange(20), np.ones(20)), 1)

    def test_saved_decisions_and_thresholds(self):
        import csv

        with (ROOT / "results/fgvc_edge.csv").open() as f:
            ref = {r["source_path"]: int(r["true_diseased"]) for r in csv.DictReader(f)}
        with (ROOT / "results/sensitivity64_metrics.csv").open() as f:
            report = {
                (int(r["seed"]), r["method"], r["operating_point"]): r
                for r in csv.DictReader(f)
            }
        with np.load(
            ROOT / "results/sensitivity64_predictions.npz", allow_pickle=False
        ) as trace:
            test = trace["test_paths"]
            labels = trace["test_y"]
            npos = int(labels.sum())
            nneg = len(labels) - npos
            for s in trace["seeds"]:
                fit = set(trace["fit_paths"][s])
                cal = trace["calibration_paths"][s]
                self.assertFalse(set(cal) & fit or set(test) & (set(cal) | fit))
                pos = np.array([ref[p] for p in cal]) == 1
                for j, method in enumerate(trace["methods"]):
                    scores = trace["test_score"][s, j]
                    auc = (
                        rankdata(scores)[labels == 1].sum() - npos * (npos + 1) / 2
                    ) / (npos * nneg)
                    for k, op in enumerate(trace["operating_points"]):
                        row = report[s, method, op]
                        cutoff = trace["threshold"][s, j, k]
                        if op == "target95":
                            positive_scores = trace["calibration_score"][s, j][pos]
                            keep = sum(positive_scores >= cutoff)
                            self.assertGreaterEqual(
                                20 * keep, 19 * len(positive_scores)
                            )
                            self.assertLess(
                                20 * sum(positive_scores > cutoff),
                                19 * len(positive_scores),
                            )
                        np.testing.assert_array_equal(
                            scores >= cutoff, trace["pred"][s, j, k]
                        )
                        self.assertAlmostEqual(auc, float(row["auroc"]), places=12)
            # The fitted 64-label scalar has positive slope for every observed seed;
            # empirical recall thresholding therefore gives the same test decisions.
            np.testing.assert_array_equal(
                trace["pred"][:, 0, 1], trace["pred"][:, 1, 1]
            )

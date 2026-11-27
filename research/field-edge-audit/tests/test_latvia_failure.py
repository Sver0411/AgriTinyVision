"""Check the finite-batch oracle independently of the observed field outcomes."""

from itertools import product
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from latvia_failure import ba, oracle_selection, read


class FailureOracleTest(unittest.TestCase):
    def test_routing_bounds_and_matched_archived_values(self):
        rows = read("latvia_failure_oracle_runs.csv")
        observed = {(r["site"], int(r["seed"]), r["method"]): r for r in rows}
        groups = {
            (r["site"], r["group_type"], r["group"]): r
            for r in read("latvia_failure_groups.csv")
        }
        previous = [
            r
            for r in read("joint_budget_ablation.csv")
            if r["cap_percent"] == "25" and r["review_location"] == "local"
        ]
        for reference in previous:
            site = reference["site"]
            r = groups[site, "all", "all"]
            self.assertAlmostEqual(
                float(r["joint_ba"]), float(reference["selective_review_mean_ba"])
            )
            self.assertAlmostEqual(
                float(r["random_ba_mean"]), float(reference["uniform_review_mean_ba"])
            )
            for seed in range(20):
                actual, bound, expert, unlimited = [
                    observed[site, seed, k]
                    for k in (
                        "actual",
                        "optimal_router_fixed_expert",
                        "perfect_expert_on_actual_queries",
                        "unlimited_oracle_existing_experts",
                    )
                ]
                self.assertLessEqual(int(bound["total_bytes"]), int(bound["cap_bytes"]))
                self.assertLessEqual(
                    float(actual["balanced_accuracy"]),
                    float(bound["balanced_accuracy"]) + 1e-12,
                )
                self.assertLessEqual(
                    float(actual["balanced_accuracy"]),
                    float(expert["balanced_accuracy"]) + 1e-12,
                )
                self.assertLessEqual(
                    float(bound["balanced_accuracy"]),
                    float(unlimited["balanced_accuracy"]) + 1e-12,
                )

    def test_agrees_with_every_subset(self):
        rng = np.random.default_rng(719)
        for n in (3, 5, 8):
            for _ in range(7):
                y = np.r_[0, 1, rng.integers(2, size=n - 2)]
                edge, remote = rng.integers(2, size=(2, n))
                costs = rng.integers(1, 8, size=n)
                all_queries = np.array(list(product((False, True), repeat=n)))
                traffic = all_queries @ costs
                all_ba = ba(y, np.where(all_queries, remote, edge))
                for cap in range(int(costs.sum()) + 1):
                    q = oracle_selection(y, edge, remote, costs, cap)
                    self.assertLessEqual(costs[q].sum(), cap)
                    self.assertAlmostEqual(
                        ba(y, np.where(q, remote, edge)), all_ba[traffic <= cap].max()
                    )
                    self.assertTrue(np.all((edge[q] != y[q]) & (remote[q] == y[q])))


if __name__ == "__main__":
    unittest.main()

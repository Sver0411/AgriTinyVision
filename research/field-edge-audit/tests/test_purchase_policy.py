"""Decision-only examples independent of any orchard ground truth."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from purchase_policy import choose_action


class PurchaseDecisionTests(unittest.TestCase):
    def setUp(self):
        self.y = np.repeat([False, True], 8)
        self.wrong = ~self.y

    def test_learning_evidence_can_pay_for_expansion(self):
        outcomes = {"freeze": self.wrong, "probe": self.y, "expand": self.y}
        costs = {"freeze": 0, "probe": 200, "expand": 500}
        pick = choose_action(self.y, outcomes, 500, costs, learning_pair=(self.wrong, self.y))
        self.assertEqual("expand", pick.action)
        no_future_gain = choose_action(self.y, outcomes, 500, costs, learning_pair=(self.y, self.y))
        self.assertEqual("probe", no_future_gain.action)
        short = choose_action(self.y, outcomes, 499, costs, learning_pair=(self.wrong, self.y))
        self.assertEqual("probe", short.action)
        erm = choose_action(self.y, outcomes, 500, costs, learning_pair=(self.wrong, self.y), rule="paired_erm")
        self.assertEqual("probe", erm.action)

    def test_no_data_or_no_effect_keeps_reference(self):
        costs = {"freeze": 0, "probe": 5}
        self.assertEqual("freeze", choose_action(self.y, {"freeze": self.y, "probe": self.y}, 50, costs).action)
        only = np.ones(16)
        pick = choose_action(only, {"freeze": only, "probe": only}, 50, costs)
        self.assertEqual("insufficient_class_support", pick.reason)

    def test_sensitivity_and_balance_are_distinct(self):
        frozen = np.ones(16, bool)
        candidate = self.y.copy()
        candidate[-1] = False
        outcomes = {"freeze": frozen, "probe": candidate}
        costs = {"freeze": 0, "probe": 5}
        self.assertEqual("freeze", choose_action(self.y, outcomes, 5, costs).action)
        self.assertEqual("probe", choose_action(self.y, outcomes, 5, costs, rule="without_recall_guard").action)
        guarded = frozen | candidate
        self.assertFalse(np.any(self.y & frozen & ~guarded))

    def test_guard_has_separate_learning_evidence(self):
        outcomes = {"freeze": self.wrong, "probe_guard": self.y, "expand_guard": self.y}
        costs = {"freeze": 0, "probe_guard": 3, "expand_guard": 7}
        only_plain = choose_action(self.y, outcomes, 7, costs, learning_pair=(self.wrong, self.y))
        self.assertEqual("probe_guard", only_plain.action)
        guarded = choose_action(self.y, outcomes, 7, costs, learning_pair={"guard": (self.wrong, self.y)})
        self.assertEqual("expand_guard", guarded.action)

    def test_invalid_inputs_fail(self):
        with self.assertRaises(ValueError):
            choose_action(self.y, {"freeze": self.y, "probe": self.y}, -1, {"freeze": 0, "probe": 3})
        with self.assertRaises(ValueError):
            choose_action(self.y, {"freeze": self.y, "probe": self.y[:-1]}, 5, {"freeze": 0, "probe": 3})

"""Regression tests for the finite-population binary sign certificate."""

import sys
import unittest
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from verify_certificate import allocation, exact_performance, reference_interval


class FinitePopulationTests(unittest.TestCase):
    def test_zero_audit_and_census(self):
        self.assertEqual(reference_interval(0, 0, 0), (0, 0))
        self.assertEqual(reference_interval(5, 0, 0), (0, 5))
        self.assertEqual(reference_interval(5, 5, 2), (2, 2))

    def test_exact_endpoint_tie(self):
        self.assertEqual(reference_interval(40, 1, 1), (2, 40))
        self.assertEqual(reference_interval(40, 1, 0), (0, 38))

    def test_one_repair_hardware_case(self):
        targeted = allocation(60, 1, 1, 1, "disagreement")
        uniform = allocation(60, 1, 1, 1, "uniform")
        self.assertEqual(targeted, {(1, 1): Fraction(1)})
        self.assertEqual(uniform, {(0, 0): Fraction(59, 60), (1, 1): Fraction(1, 60)})

    def test_label_swap_symmetry_and_null(self):
        for d in range(1, 9):
            for n in range(d + 1):
                for x in range(n + 1):
                    lo, hi = reference_interval(d, n, x)
                    self.assertEqual(reference_interval(d, n, n - x), (d - hi, d - lo))
        out = exact_performance(6, 3, allocation(9, 6, 3, 9, "uniform"))
        self.assertEqual(out, (1, 0, 0, 0))

    def test_fixed_equal_tail_power_need_not_be_monotone(self):
        targeted = allocation(9, 8, 5, 5, "disagreement")
        uniform = allocation(9, 8, 5, 5, "uniform")
        self.assertEqual(exact_performance(8, 5, targeted)[1], Fraction(1, 56))
        self.assertEqual(exact_performance(8, 5, uniform)[1], Fraction(1, 21))


if __name__ == "__main__":
    unittest.main()

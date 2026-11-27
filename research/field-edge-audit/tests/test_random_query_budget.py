import itertools
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from random_query_budget import strict_reduction_probability


class QueryBudgetTest(unittest.TestCase):
    def test_matches_enumerated_query_subsets(self):
        for fixed, added in itertools.product(range(5), repeat=2):
            for q in (0, 0.1, 0.25, 0.5, 1):
                exact = 0
                for subset in itertools.product([0, 1], repeat=fixed + added):
                    if sum(subset[:fixed]) > sum(subset[fixed:]):
                        k = sum(subset)
                        exact += q**k * (1 - q)**(len(subset) - k)
                self.assertAlmostEqual(strict_reduction_probability(fixed, added, q), exact)


if __name__ == '__main__':
    unittest.main()

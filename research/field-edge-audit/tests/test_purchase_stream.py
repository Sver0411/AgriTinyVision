"""Regression tests for no-borrowing replay and label-independent quotes."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from purchase_experiment import ACTION_BYTES, REQUEST_BYTES, TrainingPanel, library, ranks


class PurchaseStreamTests(unittest.TestCase):
    def run_stream(self, first_future_changed=False):
        n = 80
        sizes = np.full(n, 6000, np.int32)
        source = np.zeros(n, np.uint8)
        local = np.ones(n, np.uint8)
        remote = np.ones(n, np.uint8)
        priority = np.arange(n, dtype=np.float64)
        if first_future_changed:
            sizes[59:] *= 4
            remote[59:] = 0
            priority[59:] = -priority[59:]
        pred, asks = np.empty(n,np.uint8), np.empty(n,np.uint8)
        stats = np.zeros(4,np.int64)
        status = library().replay(n,1700,1,34000,0,0,np.inf,sizes,source,local,priority,
                                  ranks(priority),remote,pred,asks,stats)
        self.assertEqual(status, 0)
        return pred, asks, stats, sizes

    def test_no_borrowing_and_prefix_independence(self):
        pred, asks, stats, sizes = self.run_stream()
        modified_pred, modified_asks, _, _ = self.run_stream(True)
        self.assertEqual(stats[2], 20)
        np.testing.assert_array_equal(pred[:19],0)
        np.testing.assert_array_equal(asks[:19],0)
        np.testing.assert_array_equal(pred[:59],modified_pred[:59])
        np.testing.assert_array_equal(asks[:59],modified_asks[:59])
        spent = np.cumsum(asks.astype(np.int64)*sizes)
        spent[19:] += 34000
        self.assertTrue(np.all(spent <= np.arange(1,81)*1700))

    def test_unaffordable_quote_reads_no_label(self):
        indices = np.arange(128)
        panel = TrainingPanel({
            'cal_x': np.zeros((128,4),np.float32),
            'cal_y': None, 'cal_cloud_p': None,
            'cal_size': np.full(128,6000), 'cal_p': np.full(128,.2), 'cutoff': .3,
        }, indices, 0, np.full(121,.2))
        self.assertEqual(panel.bill(32,'edgeboost_rank'),
                         32*(6000+4+37)+REQUEST_BYTES+ACTION_BYTES+4+14)
        with patch.object(panel,'paid_training_prefix',side_effect=AssertionError):
            action = panel.audit(30,51000,'earned',32)
        self.assertEqual(action.labels,0)
        self.assertEqual(action.setup,0)


if __name__ == '__main__':
    unittest.main()

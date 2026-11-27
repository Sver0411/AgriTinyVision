"""Unit tests for the BASA (Budget-Amortized Source-Anchored Adaptation) controller."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from basa_controller import (  # noqa: E402
    amortized_label_count,
    fit_asar_head,
    select_asar_head,
    select_quantile_stratified,
)
from edge_adaptation_wire import I8, normalized, predict_from_head, quantized  # noqa: E402
from published_defer import binary_head, train_weights  # noqa: E402


class BASAControllerTests(unittest.TestCase):
    def test_amortized_label_count_scales_with_horizon_and_mode(self):
        receipts = np.full(128, 10000, dtype=int)
        fixed = 1334
        # Short horizon (Kashmir-like): 384 * 1700 = 652,800
        k_short_pre = amortized_label_count(652_800, 32, receipts, fixed, "prepaid")
        k_short_earn = amortized_label_count(652_800, 32, receipts, fixed, "earned")
        self.assertGreaterEqual(k_short_pre, k_short_earn)
        self.assertEqual(k_short_earn, 16)
        # Long horizon (Cornell-like): 1353 * 1700 = 2,300,100
        k_long_earn = amortized_label_count(2_300_100, 128, receipts, fixed, "earned")
        self.assertGreater(k_long_earn, 32)
        self.assertLess(k_long_earn, 128)
        # Tiny budget that cannot afford even 8 labels returns 0
        self.assertEqual(amortized_label_count(50_000, 32, receipts, fixed, "earned"), 0)

    def test_quantile_stratified_is_deterministic_and_covers_tails(self):
        probs = np.linspace(0.01, 0.99, 100)
        rng = np.random.default_rng(7)
        perm = rng.permutation(100)
        p_shuffled = probs[perm]
        sel = select_quantile_stratified(p_shuffled, 16)
        self.assertEqual(len(sel), 16)
        self.assertEqual(len(set(sel.tolist())), 16)
        chosen_p = np.sort(p_shuffled[sel])
        self.assertLess(chosen_p[0], 0.10)
        self.assertGreater(chosen_p[-1], 0.90)

    def test_asar_lambda_one_matches_isotropic_binary_head(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(32, 64)).astype(np.float32)
        q, _ = quantized(x)
        x_norm = normalized(q)
        y = np.array([0] * 12 + [1] * 20, dtype=np.int32)
        w0 = rng.normal(size=64)
        w0 /= np.linalg.norm(w0)
        h_asar = fit_asar_head(x_norm, y, w0, 1.0)
        h_std = binary_head(x_norm, y, train_weights(y, "balanced"))
        self.assertEqual(h_asar, h_std)
        self.assertEqual(len(h_asar), I8.size + 64)

    def test_asar_prefers_platt_when_source_head_is_near_perfect(self):
        rng = np.random.default_rng(123)
        d = 128
        w0 = np.zeros(d)
        w0[0] = 1.0
        # 24 samples: signal lies purely along w0, 127 orthogonal dims are pure noise
        x = rng.normal(scale=0.5, size=(24, d)).astype(np.float32)
        y = np.array([0] * 12 + [1] * 12, dtype=np.int32)
        x[:12, 0] = -2.0
        x[12:, 0] = +2.0
        q, _ = quantized(x)
        head, chosen_lam = select_asar_head(q, y, w0, seed=0)
        self.assertTrue(np.isinf(chosen_lam))
        scores = predict_from_head("local_int8", head, q)
        self.assertEqual(int(np.sum((scores >= 0) == y)), 24)


if __name__ == "__main__":
    unittest.main()

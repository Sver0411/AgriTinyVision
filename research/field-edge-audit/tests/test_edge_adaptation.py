"""Wire, sign and split regressions for the target-labelled comparison."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from edge_adaptation import fit
from edge_adaptation_wire import (
    pack_head,
    predict_from_head,
    quantized,
    training_input,
    training_record,
)


class EdgeAdaptationWireTests(unittest.TestCase):
    def test_signed_quantization_and_ties(self):
        x = np.asarray([[0, 0, 0, 0], [-127, -0.5, 0.5, 1.5]], np.float32)
        q, scale = quantized(x)
        np.testing.assert_array_equal(q, [[0, 0, 0, 0], [-127, 0, 0, 2]])
        np.testing.assert_array_equal(scale, [1, 1])

    def test_packet_identity_and_cost(self):
        x = np.linspace(-1, 1, 1280, dtype=np.float32)
        q, scale = quantized(x)
        packet = training_record("local_int8", 7, 0, x)
        self.assertEqual(len(packet), 1293)
        decoded = training_input("local_int8", packet, 7, 0, len(x))
        np.testing.assert_array_equal(decoded, q.astype(np.float64) * scale)
        with self.assertRaises(ValueError):
            training_input("local_int8", packet, 8, 0, len(x))
        with self.assertRaises(ValueError):
            training_input("local_int8", packet[:-1], 7, 0, len(x))

    def test_single_class_and_zero_activation(self):
        x = np.zeros((3, 1280))
        for label in (0, 1):
            w, b = fit(x, np.full(3, label))
            raw = pack_head("local_int8", w, b)
            self.assertEqual(len(raw), 1294)
            score = predict_from_head("local_int8", raw, x)
            self.assertTrue(np.array_equal(score >= 0, np.full(3, bool(label))))

    def test_quantized_direction_and_float_payload(self):
        x = np.asarray([[0, 0], [127, 0], [0, -127]], np.int8)
        head = pack_head("local_int8", [1.0, 2.0], -0.25)
        q, scale = quantized([1.0, 2.0])
        expected = [-0.25, q[0] * float(scale) - 0.25, -q[1] * float(scale) - 0.25]
        np.testing.assert_allclose(
            predict_from_head("local_int8", head, x), expected, atol=1e-7
        )
        scalar = pack_head("scalar", [1.25], -0.5)
        np.testing.assert_array_equal(
            predict_from_head("scalar", scalar, [[0], [2]]), [-0.5, 2.0]
        )
        self.assertEqual(len(scalar), 14)

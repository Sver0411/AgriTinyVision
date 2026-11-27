"""Regression checks for the NeurIPS cascade reimplementation."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from published_cascade import (budget_order, forward, initialize, objective_gradients,
                               pack, probability_features, targets, uncertainty_control, unpack)


class PublishedCascadeTests(unittest.TestCase):
    def test_binary_features_and_targets(self):
        probs = np.array([0., .2, .5, .8, 1.])
        x = probability_features(probs)
        self.assertEqual(x.shape, (5,5))
        np.testing.assert_array_equal(x[:,3:], np.eye(2)[[0,0,0,1,1]])
        self.assertTrue(np.isfinite(x).all())
        np.testing.assert_allclose(x[[0,-1],0], 0)
        self.assertAlmostEqual(x[2,0],np.log(2))
        z=targets(np.array([0,1,1,0]),np.array([.2,.2,.9,.8]),np.array([.6,.8,.7,.1]))
        np.testing.assert_array_equal(z["diff01"],[-1,1,0,1])
        np.testing.assert_allclose(z["diffprob"],[-.4,.6,-.2,.7])

    def test_mlp_derivative_and_wire(self):
        x=probability_features(np.array([.13,.33,.72,.95]))
        target=np.array([1.,0,-1,1]); weights=initialize(1)
        weights[1][:]=.01
        weights[3][:]=.01
        class_weights=np.array([.75,1.25,.75,1.25])
        _,g=objective_gradients(x,target,class_weights,weights)
        for i, idx in [(0,(2,3)),(1,(8,)),(2,(23,5)),(3,(7,)),(4,(11,0)),(5,(0,))]:
            h=1e-6
            weights[i][idx]+=h
            plus=objective_gradients(x,target,class_weights,weights)[0]
            weights[i][idx]-=2*h
            minus=objective_gradients(x,target,class_weights,weights)[0]
            weights[i][idx]+=h
            self.assertAlmostEqual(float(g[i][idx]), (plus-minus)/(2*h),places=7)
        w32=[w.astype(np.float32).astype(float) for w in weights]
        self.assertEqual(len(pack(w32)),len(pack(unpack(pack(w32)))))
        np.testing.assert_array_equal(forward(x,w32)[-1],forward(x,unpack(pack(w32)))[-1])
        with self.assertRaises(ValueError):unpack(pack(w32)[:-1])

    def test_nonpositive_budget_and_head_fallback(self):
        paths=np.array(["b","a","c"]); costs=np.array([5,3,2])
        score=np.array([.6,.6,-.1])
        np.testing.assert_array_equal(budget_order(score,paths,costs,5,"budget"),[False,True,True])
        np.testing.assert_array_equal(budget_order(score,paths,costs,5,"positive"),[False,True,False])
        e=np.array([.1,.4,.7]); margin=np.array([-.2,0.,.3])
        self.assertEqual(uncertainty_control(paths,e,margin,costs,5,6,True).tier,"original_uncertainty")
        self.assertEqual(uncertainty_control(paths,e,margin,costs,5,6,False).tier,"adapted_uncertainty")


if __name__=="__main__":
    unittest.main()

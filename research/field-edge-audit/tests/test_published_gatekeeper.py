"""Check Gatekeeper's true-label gradients and serialized binary decisions."""

import sys
from pathlib import Path
import unittest

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))

from edge_adaptation_wire import pack_head,predict_from_head,quantized
from published_gatekeeper import METHODS,decode_head,inference,objective


class GatekeeperTests(unittest.TestCase):
    def test_piecewise_gradient_and_correctness_masks(self):
        z=np.array([-.7,1.3,.8,-.4]); labels=np.array([0,1,0,1])
        weight=np.array([.6,1.4,.6,1.4])
        for alpha in (.1,.5,.9,None):
            loss,grad=objective(z,labels,alpha,weight)
            self.assertTrue(np.isfinite(loss) and np.isfinite(grad).all())
            for i in range(len(z)):
                perturb=np.eye(4)[i]*1e-6
                diff=(objective(z+perturb,labels,alpha,weight)[0]-objective(z-perturb,labels,alpha,weight)[0])/(2e-6)
                self.assertAlmostEqual(grad[i],diff,places=8)
        _,grad=objective(z,labels,.5,weight)
        self.assertTrue(np.all(np.sign(grad)==[1,-1,1,-1]))
        self.assertTrue(np.isfinite(objective(np.array([-900.,900.]),np.array([1,0]),.5,np.ones(2))[0]))

    def test_integer_scoring_matches_published_wire_decoder(self):
        acts=np.array([[1.,.3,-.2],[0,0,0],[.2,.6,.8]])
        blob=pack_head("local_int8",np.array([-.6,.1,.8]),.13)
        w=decode_head(blob)
        self.assertEqual(w.shape,(4,))
        scores=inference(acts,{(s,m):blob for s in range(20) for m in METHODS})
        expected=predict_from_head("local_int8",blob,quantized(acts)[0])
        np.testing.assert_allclose(scores[0,0],expected,rtol=1e-12,atol=1e-12)
        self.assertTrue(np.array_equal(scores[-1,-1]>=0,expected>=0))


if __name__=="__main__":
    unittest.main()

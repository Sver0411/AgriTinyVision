"""Reconstruct paired rates and check the sampling units."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from purchase_inference import CoupledResamples, paired_differences, score_arrays


class PurchaseInferenceTests(unittest.TestCase):
    def test_coupled_image_intervals_for_deterministic_improvement(self):
        y=np.repeat([False,True],10)
        treatment=np.stack([y,y])
        alternative=~treatment
        rs=CoupledResamples("cornell",y,np.arange(len(y)),nboot=42)
        result=rs.intervals(paired_differences(y,treatment,alternative))
        np.testing.assert_allclose(result["ba_pp"],(100,100))
        np.testing.assert_allclose(result["fpr_pp"],(-100,-100))
        np.testing.assert_allclose(result["fnr_pp"],(-100,-100))

    def test_cluster_resampling_and_packed_masks(self):
        dates=np.array([f"2020-05-{n:02d}" for n in range(1,10)])
        y=np.array([False] + [True]*8)
        rs=CoupledResamples("latvia",y,dates,nboot=400)
        self.assertLess(rs.valid.sum(),400)
        self.assertGreater(rs.valid.sum(),0)
        np.testing.assert_allclose(rs.intervals(np.zeros(len(y)))["ba_pp"],(0,0))
        pred=np.packbits(np.array([y,~y]),axis=1)
        rows=[dict(seed=1,order=0,mask_row=1,fp=1,fn=8,ba=0),dict(seed=0,order=0,mask_row=0,fp=0,fn=0,ba=1)]
        _,a,ba=score_arrays(rows,pred,y)
        np.testing.assert_array_equal(a[0],y)
        np.testing.assert_array_equal(ba,[1,0])

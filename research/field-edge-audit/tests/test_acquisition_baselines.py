import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from acquisition_baselines import METHODS, bbsd_soft, download_byte_count, fit_router, reference_scores
from edge_adaptation_wire import predict_from_head


class AcquisitionBaselines(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(72)
        self.q = rng.integers(-127, 128, (30, 10), dtype=np.int8)
        self.y = np.r_[np.zeros(11, dtype=int), np.ones(19, dtype=int)]
        self.expert = rng.random(30) > .4

    def test_formula_cost_and_single_pass_predictions(self):
        for method in METHODS:
            model = fit_router(method, self.q[:20], self.y[:20], self.expert[:20], seed=7)
            logit, urgent = model.score(self.q)
            prefix = model.score(self.q[:19])
            for complete, part in zip((logit, urgent), prefix):
                np.testing.assert_allclose(complete[:19], part, rtol=1e-13, atol=1e-13)
            minimum = self.q.shape[1] + 14
            expected = minimum + ({"ova":minimum, "compare_confidence":minimum, "mao":minimum,
                                    "diff01":5776, "edgeboost_fixed":8}.get(method,0))
            self.assertEqual(model.download_bytes, expected)
            self.assertEqual(model.download_bytes, download_byte_count(method,self.q.shape[1]))
            if method in ("ova", "compare_confidence"):
                c = predict_from_head("local_int8", model.correctness, self.q)
                expected = c-abs(logit) if method=="ova" else expit(c)-expit(abs(logit))
                np.testing.assert_array_equal(urgent, expected)
            if method=="edgeboost_fixed":
                self.assertGreater(model.temperature, 0.)
                for fraction in (0., .25, 1.):
                    cutoff = model.fixed_cutoff(np.arange(1,21),fraction)
                    accepted = -model.oof_confidence > cutoff
                    if fraction==0.:self.assertFalse(accepted.any())
                    if fraction==1.:self.assertTrue(accepted.all())

    def test_binary_shift_detector_matches_author_rule_and_checks_invalid_values(self):
        src = np.linspace(.01,.99,121)
        for target in (src, np.linspace(.98,1.,16)):
            got, p, statistic = bbsd_soft(src,target)
            raw = ks_2samp(src,target)
            self.assertEqual(got, raw.pvalue < .025)
            self.assertAlmostEqual(statistic, raw.statistic)
            self.assertAlmostEqual(p,min(1.,2*raw.pvalue))
        with self.assertRaises(ValueError): bbsd_soft(src,[float("nan")])
        for backbone in ("v2","v2grid","v4"):
            p = reference_scores(backbone)
            self.assertEqual(p.shape,(121,))
            self.assertTrue(np.all((p>=0)&(p<=1)))


if __name__=="__main__":
    unittest.main()

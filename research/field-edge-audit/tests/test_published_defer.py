"""Checks for the fixed-expert ports and their released paired replay artifacts."""

import csv
import sys
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import expit

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from published_defer import BASELINE, RESULTS, SITES, mao_examples, score_actions, select_model


class PublishedDeferralTests(unittest.TestCase):
    def test_binary_ova_and_compare_confidence_scores(self):
        local = np.array([-2., .3, 1.])
        expert = np.array([1.2, -.2, 2.])
        classes = np.stack([-local, local], axis=1)
        expected = classes.argmax(axis=1)
        pred, ova = score_actions("ova_balanced", local, expert)
        np.testing.assert_array_equal(pred, expected)
        np.testing.assert_array_equal(ova, expert - classes.max(axis=1))
        _, cc = score_actions("compare_confidence_balanced", local, expert)
        probability = expit(local)
        np.testing.assert_allclose(cc, expit(expert) - np.maximum(probability,1-probability))
        np.testing.assert_array_equal(ova > 0, cc > 0)

    def test_mao_logistic_loss_reduction(self):
        label = np.array([0,0,1,1])
        local = np.array([0,1,1,0])
        expert = np.array([0,0,0,0])
        score = np.array([[.6],[-1],[2],[0.]])
        weight = np.array([1.,1.5,2.,.5])
        x,t,w = mao_examples(score,label,local,expert,weight)
        equivalent = np.sum(w*np.logaddexp(0, (1-2*t)*x[:,0]))
        expected = np.sum(weight*((local!=label)*np.logaddexp(0,-score[:,0])+
                                  (expert!=label)*np.logaddexp(0,score[:,0])))
        self.assertAlmostEqual(equivalent,expected)

    def test_installation_fallback(self):
        self.assertEqual(select_model("ova_balanced", 250,100, 250,False),("ova_balanced",250,250))
        self.assertEqual(select_model("ova_balanced", 250,100, 150,False),(BASELINE,100,250))
        self.assertEqual(select_model("ova_balanced", 250,100, 70,False),("published_edge",0,250))
        self.assertEqual(select_model("ova_balanced", 250,100, 70,True),("ova_balanced",0,250))

    def test_released_per_image_counts_and_costs(self):
        for direction in [RESULTS, *(RESULTS / f"published_defer_partial_{n}" for n in (8,32,64))]:
            groups = defaultdict(list)
            with (direction/"published_defer_runs.csv").open(newline="") as f:
                for r in csv.DictReader(f):
                    groups[r["site"]].append(r)
            for site in SITES:
                rows = groups[site]
                with np.load(direction/f"published_defer_decisions_{site}.npz") as decisions, np.load(direction/f"published_defer_scores_{site}.npz") as scores:
                    y = decisions["y"]
                    np.testing.assert_array_equal(scores["source_path"],decisions["source_path"])
                    self.assertEqual(len(rows),1680)
                    self.assertEqual(len(np.unique(scores["tie_key"])),len(y))
                    p = np.unpackbits(decisions["predictions"],axis=1,count=len(y))
                    query = np.unpackbits(decisions["queries"],axis=1,count=len(y))
                    np.testing.assert_array_equal((p[:,y==0]==1).sum(axis=1),[int(r["fp"]) for r in rows])
                    np.testing.assert_array_equal((p[:,y==1]==0).sum(axis=1),[int(r["fn"]) for r in rows])
                    np.testing.assert_array_equal(query @ scores["costs"].astype(np.int64),[int(r["inference_bytes"]) for r in rows])
                    self.assertTrue(all(int(r["total_bytes"])==int(r["actual_setup_bytes"])+int(r["inference_bytes"])
                                        and int(r["total_bytes"])<=int(r["cap_bytes"]) for r in rows))


if __name__ == "__main__":
    unittest.main()

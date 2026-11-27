"""Check causal prefixes, no-advance-credit operation and released one-pass masks."""

import csv
import sys
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from streaming_review import RESULTS, SITES, scan
from streaming_published import past_percentile, serve


class StreamingTests(unittest.TestCase):
    def test_same_prefix_never_depends_on_suffix(self):
        early=[.1,.9,.2,.3,.8,.6,.55,.91]*5
        costs=[105,180,145,210,80,150,140,75]*5
        for kind in ("fixed","paced","credit"):
            args=(kind,106,.6,140.,100*3200,80 if kind!="credit" else None,6150 if kind!="credit" else None)
            first=scan(early+[0.01]*40,costs+[45]*40,*args)
            other=scan(early+[.999]*40,costs+[870]*40,*args)
            np.testing.assert_array_equal(first[0][:40],other[0][:40])
            if kind=="credit":
                for k in range(1,41):
                    _,_,spent,_=scan(early[:k],costs[:k],*args)
                    self.assertLessEqual(spent,k*100)

    def test_percentiles_use_only_earlier_frames(self):
        s=np.random.default_rng(55).normal(size=120)
        ranks=past_percentile(s)
        expected=[]
        for i,score in enumerate(s):
            h=s[max(0,i-64):i]
            expected.append(-1. if len(h)<16 else (sum(h<score)+sum(h<=score))/(2*len(h)))
        np.testing.assert_array_equal(ranks,expected)
        q1,_=serve(ranks,[100]*120,120,4500,100.)
        changed=past_percentile(np.r_[s[:43], np.zeros(77)])
        q2,_=serve(changed,[100]*43+[20]*77,120,4500,100.)
        np.testing.assert_array_equal(q1[:43],q2[:43])

    def test_released_cases_and_edgeboost_query_parity(self):
        for prefix in ("streaming", "streaming_published"):
            groups=defaultdict(list)
            with (RESULTS/f"{prefix}_runs.csv").open(newline="") as f:
                for row in csv.DictReader(f):
                    groups[row["site"]].append(row)
            for site in SITES:
                decisions=(f"streaming_decisions_{site}.npz" if prefix=="streaming" else f"streaming_published_{site}.npz")
                with np.load(RESULTS/decisions) as z, np.load(RESULTS/f"streaming_inputs_{site}.npz") as inp:
                    y=z["y"]
                    q=np.unpackbits(z["queries"],axis=1,count=len(y))
                    p=np.unpackbits(z["predictions"],axis=1,count=len(y))
                    costs=inp["costs"]
                    by={}
                    for i,row in enumerate(groups[site]):
                        self.assertEqual(i,int(row["row_index"]))
                        self.assertEqual(int(row["fp"]),((p[i]==1)&(y==0)).sum())
                        self.assertEqual(int(row["fn"]),((p[i]==0)&(y==1)).sum())
                        spent=int(row["actual_setup_bytes"])+int(costs[q[i].astype(bool)].sum())
                        self.assertEqual(spent,int(row["total_bytes"]))
                        self.assertLessEqual(spent,int(row["cap_bytes"]))
                        if prefix=="streaming_published":
                            k=(row["reviewer"],row["seed"],row["cap_percent"],row["sequence"])
                            if row["method"]=="uncertainty_history":by[k]=i
                            if row["method"]=="edgeboost_history":
                                np.testing.assert_array_equal(p[by[k]],p[i])
                                np.testing.assert_array_equal(q[by[k]],q[i])


if __name__=="__main__":
    unittest.main()

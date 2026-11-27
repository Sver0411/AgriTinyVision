"""Audit equal-budget collaboration and ablate the order in which reviews are sent."""

import csv
from pathlib import Path

import numpy as np

from evaluate_upload import rank_selection

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'


def balanced(reference, predictions):
    err = predictions != reference
    return 1 - (err[..., reference == 0].mean(axis=-1) + err[..., reference == 1].mean(axis=-1)) / 2


def audit():
    with (RESULTS / 'joint_budget_runs.csv').open() as f:
        all_rows = list(csv.DictReader(f))
    summary = []
    for site in ('cornell', 'kashmir', 'latvia'):
        rows = [r for r in all_rows if r['site'] == site]
        with np.load(RESULTS / f'joint_budget_{site}.npz') as z:
            preds, queries = z['predictions'].astype(bool), z['queries'].astype(bool)
            expert, y, costs = z['expert'], z['y'], z['application_bytes']
            lookup = {}
            for i,r in enumerate(rows):
                pred = preds[i]
                assert int(((y==0)&pred).sum()) == int(r['fp'])
                assert int(((y==1)&~pred).sum()) == int(r['fn'])
                assert np.isclose(balanced(y,pred),float(r['balanced_accuracy']))
                assert int(costs[queries[i]].sum()) + int(r['setup_bytes']) == int(r['total_bytes']) <= int(r['cap_bytes'])
                lookup[int(r['cap_percent']),r['review_location'],int(r['seed']),r['method']] = i
            # Fixed random orders are independent of score, seed, disease status and byte length.
            # Start with canonical identifiers so that file ordering differences cannot change the draws.
            canonical = np.argsort(z['source_path'])
            orders = [np.random.default_rng(21000 + j).permutation(canonical) for j in range(20)]
            for pct in (10,25,50):
                for loc in ('local','remote'):
                    differences = []
                    powers = []
                    repair, regression, remote_ba, local_ba = [], [], [], []
                    for seed in range(20):
                        j = lookup[pct,loc,seed,'joint']
                        row = rows[j]
                        original = preds[lookup[pct,loc,seed,'update_only']]
                        mask = queries[j]
                        assert np.array_equal(preds[j], np.where(mask,expert,original))
                        remaining = int(row['cap_bytes'])-int(row['setup_bytes'])
                        random_preds = np.asarray([np.where(rank_selection(order,costs,remaining),expert,original)
                                                   for order in orders])
                        random_ba = balanced(y, random_preds)
                        joint_ba = float(row['balanced_accuracy'])
                        differences.append(joint_ba - random_ba.mean())
                        powers.append(joint_ba)
                        repair.append(int((mask & (original!=y) & (expert==y)).sum()))
                        regression.append(int((mask & (original==y) & (expert!=y)).sum()))
                        remote_ba.append(random_ba.mean())
                        local_ba.append(float(balanced(y, original)))
                    summary.append(dict(site=site, cap_percent=pct, review_location=loc, calibration_draws=20,
                                        random_orders_per_draw=20, mean_before_review_ba=float(np.mean(local_ba)),
                                        uniform_review_mean_ba=float(np.mean(remote_ba)), selective_review_mean_ba=float(np.mean(powers)),
                                        selection_gain_pp=float(100*np.mean(differences)),
                                        seeds_beating_mean_uniform=int((np.array(differences)>1e-12).sum()),
                                        mean_corrected_on_review=float(np.mean(repair)), mean_spoiled_on_review=float(np.mean(regression))))
    with (RESULTS / 'joint_budget_ablation.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)


if __name__ == '__main__':
    audit()

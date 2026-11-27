"""Separate local operating point from score discrimination on the same images."""
from pathlib import Path
import csv

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from acquisition_baselines import edge_fit
from edge_adaptation_wire import predict_from_head, quantized
from purchase_cohorts import load


def select_cutoff(y, score, source_cutoff):
    if np.unique(y).size < 2:
        return source_cutoff
    fpr, tpr, limits = roc_curve(y, score, drop_intermediate=False)
    youden = tpr - fpr
    tied = limits[np.isclose(youden, youden.max(), rtol=0, atol=1e-12)]
    return float(min(tied, key=lambda t: (abs(t-source_cutoff), t)))


def metrics(y, score, cutoff):
    prediction = score >= cutoff
    fp = int((prediction & (y == 0)).sum())
    fn = int((~prediction & (y == 1)).sum())
    ba = 1. - (fp / (y == 0).sum() + fn / (y == 1).sum()) / 2.
    return dict(fp=fp, fn=fn, ba=ba, auc=roc_auc_score(y, score))


def diagnose(site, backbone):
    data = load(site, backbone)
    y, source = data['test_y'], data['test_p']
    cutoff = float(np.float32(data['cutoff']))
    pre = dict(site=site, backbone=backbone, n=len(y), n_positive=int(y.sum()))
    for rule, score, threshold in (
        ('native', source, cutoff),
        ('test_oracle', source, select_cutoff(y, source, cutoff)),
        ('cloud', data['test_cloud_p'], .5),
    ):
        yield dict(**pre, rule=rule, labels=0, draws=1, **metrics(y, score, threshold))
    q = quantized(data['test_x'])[0]
    for n in (16, 32, 128):
        results = {'recalibrate_threshold': []}
        if n == 32:
            results['replace_head'] = []
        for draw in data['cal_draws'][:, :n]:
            local_y = data['cal_y'][draw]
            new_cutoff = select_cutoff(local_y, data['cal_p'][draw], cutoff)
            results['recalibrate_threshold'].append(metrics(y, source, new_cutoff))
            if n == 32:
                x = quantized(data['cal_x'][draw])[0]
                fitted = edge_fit(x, local_y)
                margin = predict_from_head('local_int8', fitted, q)
                results['replace_head'].append(metrics(y, margin, 0.))
        for rule, records in results.items():
            yield dict(**pre, rule=rule, labels=n, draws=len(records),
                       **{k: float(np.mean([item[k] for item in records])) for k in records[0]})


def main():
    out = Path(__file__).resolve().parents[1] / 'results/purchase/backbone_diagnostic.csv'
    rows = [row for site in ('cornell', 'kashmir', 'latvia') for backbone in ('v2', 'v2grid', 'v4')
            for row in diagnose(site, backbone)]
    with out.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()

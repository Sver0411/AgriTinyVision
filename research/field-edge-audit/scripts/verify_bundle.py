"""Check the relationships that support the principal claims without downloading images."""
import csv
import json
import sys
import zipfile
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'results'


def table(name):
    with (RESULT/name).open() as f:return list(csv.DictReader(f))


def main():
    outcome=table('upload_outcomes.csv')
    for c in {r['cohort'] for r in outcome}:
        rows=[r for r in outcome if r['cohort']==c];base=next(r for r in rows if r['system']=='local')
        for r in rows:
            assert int(r['correct'])-int(base['correct'])==int(r['repairs'])-int(r['regressions'])
            assert sum(int(r[k]) for k in ['tn','fp','fn','tp'])==int(r['n'])
    with np.load(RESULT/'edge_adaptation_predictions.npz',allow_pickle=False) as z:
        y=z['test_y']; ps=z['pred']
        assert ps.shape==(20,3,5,1353) and len(set(z['test_paths']).intersection(z['train_paths'].flatten()))==0
        for r in table('edge_adaptation_metrics.csv'):
            k=np.where(z['methods']==r['method'])[0][0];b=np.where(z['budgets']==int(r['budget']))[0][0]
            p=ps[int(r['seed']),b,k]
            assert int(((p==1)&(y==1)).sum())==int(r['true_positive'])
            assert int(((p==0)&(y==0)).sum())==int(r['true_negative'])
    with zipfile.ZipFile(RESULT/'edge_adaptation_heads.zip') as archive:
        assert len(archive.namelist())==240
        assert all(len(archive.read(p))==1294 for p in archive.namelist() if p.endswith('/local_int8.bin'))
    with np.load(RESULT/'transfer_predictions.npz',allow_pickle=False) as z:
        assert z['pred'].shape==(244,384)
        for r,p in zip(table('transfer_outcomes.csv'),z['pred'],strict=True):
            assert int((p==z['y']).sum())==int(r['correct'])
    assert len(table('head_vs_oracle.csv'))==60
    assert len(table('orchard_overlap.csv'))==0
    from joint_error_evidence import by_source_label, reconstruct
    from random_query_budget import analyze
    joint = reconstruct()
    for filename, records in [('joint_error_evidence.csv', joint), ('joint_error_by_label.csv', by_source_label()),
                              ('random_query_budget.csv', list(analyze(joint)))]:
        expected = table(filename)
        for row, saved in zip(records, expected, strict=True):
            assert set(row) == set(saved)
            for key, value in row.items():
                if isinstance(value, float):
                    assert np.isclose(value, float(saved[key]), rtol=1e-12, atol=1e-15)
                else:
                    assert str(value) == saved[key]
    runs = table('joint_budget_runs.csv')
    joint_summary = table('joint_budget_summary.csv')
    assert len(runs) == 1440 and len(joint_summary) == 18
    for site in ('cornell', 'kashmir', 'latvia'):
        subset = [r for r in runs if r['site'] == site]
        with np.load(RESULT/f'joint_budget_{site}.npz',allow_pickle=False) as z:
            assert len(subset) == len(z['predictions']) == len(z['queries']) == 480
            y, pred, ask = z['y'], z['predictions'].astype(bool), z['queries'].astype(bool)
            fp = pred[:, y == 0].sum(axis=1)
            fn = (~pred[:, y == 1]).sum(axis=1)
            assert fp.tolist() == [int(r['fp']) for r in subset]
            assert fn.tolist() == [int(r['fn']) for r in subset]
            new_bytes = ask @ z['application_bytes'].astype(np.int64)
            assert new_bytes.tolist() == [int(r['total_bytes']) - int(r['setup_bytes']) for r in subset]
            assert all(int(r['total_bytes']) <= int(r['cap_bytes']) for r in subset)
        for summary in [s for s in joint_summary if s['site'] == site]:
            selected = [r for r in subset if r['review_location'] == summary['review_location']
                        and r['cap_percent'] == summary['cap_percent']]
            scores = {}
            for method in ('joint','update_only','review_only','frozen'):
                scores[method] = np.array([float(r['balanced_accuracy']) for r in selected if r['method'] == method])
                assert np.isclose(scores[method].mean(), float(summary[f'{method}_mean_ba']))
            gains = scores['joint'] - np.maximum(scores['update_only'], scores['review_only'])
            assert np.isclose(100 * gains.mean(), float(summary['joint_gain_vs_stronger_component_pp']))
            assert np.count_nonzero(gains > 1e-12) == int(summary['joint_beats_both_draws'])
    matched = {}
    for r in table('matched_collaboration_predictions.csv'):
        group = matched.setdefault((r['cohort'], r['method']), {})
        assert r['source_path'] not in group
        group[r['source_path']] = (int(r['reference']), int(r['decision']),
                                   int(r['inference_upload_bytes']) + int(r['inference_reply_bytes']))
    for r in table('matched_collaboration_outcomes.csv'):
        t, h, size = np.array(list(matched.pop((r['cohort'],r['method'])).values())).T
        assert len(t) == int(r['n'])
        assert ((t == 0) & (h == 1)).sum() == int(r['fp'])
        assert ((t == 1) & (h == 0)).sum() == int(r['fn'])
        assert int(size.sum()) == int(r['inference_app_bytes'])
    assert not matched
    print('Verified 240 heads, 244 transfer rows, 60 oracle bounds, 6 paired-error comparisons, 30 random-query budgets and 1,440 byte-capped policy runs.')


if __name__=='__main__':
    main()

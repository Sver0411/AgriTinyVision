"""Reconstruct the directional-policy error and traffic tables from per-image decisions.

Input is scalar score or decision data. No photographs, feature extractors or
network access are required. Writes results/joint_error_evidence.csv.
"""

import csv
import json
from pathlib import Path

import numpy as np

from directional_policy import EDGE_CUTOFF, REPLY, REQUEST, decide, select_queries, unpack

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'


def read_csv(name):
    with (RESULTS / name).open() as f:
        return list(csv.DictReader(f))


def measure(site, method, reference, original, prediction, queries, payload, calibration_n, setup_bytes):
    reference, original, prediction, queries = [
        np.asarray(x, dtype=bool) for x in (reference, original, prediction, queries)
    ]
    payload = np.asarray(payload, dtype=np.int64)
    out = dict(site=site, rule=method, n=len(reference),
               negative_reference=int(sum(~reference)), positive_reference=int(sum(reference)),
               calibration_images=calibration_n, setup_application_bytes=setup_bytes,
               photos_queried=int(queries.sum()),
               inference_application_bytes=int(payload[queries].sum() + queries.sum() * (REQUEST.size + REPLY.size)))
    for error, group in [('fp', ~reference), ('fn', reference)]:
        before = group & (original != reference)
        after = group & (prediction != reference)
        fixed, added = (before & ~after).sum(), (after & ~before).sum()
        assert before.sum() - fixed + added == after.sum()
        out.update({f'{error}_before': int(before.sum()), f'{error}_after': int(after.sum()),
                    f'{error}_fixed': int(fixed), f'{error}_introduced': int(added),
                    f'{error}_rate_before': float(before.sum() / group.sum()),
                    f'{error}_rate_after': float(after.sum() / group.sum())})
    out['both_strictly_lower'] = int(out['fp_after'] < out['fp_before'] and out['fn_after'] < out['fn_before'])
    return out


def reconstruct():
    with (RESULTS / 'directional_rule.json').open() as f:
        training = json.load(f)
    setup = training['calibration_upload_bytes'] + training['deployment_update_bytes']
    rule = unpack((RESULTS / 'directional_rule.bin').read_bytes())
    records = read_csv('directional_predictions.csv')
    rows = []
    for cohort in ('cornell', 'kashmir'):
        group = [r for r in records if r['cohort'] == cohort]
        col = lambda name, dtype: np.array([r[name] for r in group], dtype=dtype)
        edge = col('edge_p', np.float32)
        remote = col('reply_p', np.float32)
        pred = decide(edge, remote, rule)
        assert np.array_equal(pred, col('decision', int))
        rows.append(measure(cohort, 'cornell_directional', col('reference', int), edge >= EDGE_CUTOFF,
                            pred, select_queries(edge, rule), col('payload_jpeg_bytes', int),
                            training['fit_n'], setup))
    with np.load(RESULTS / 'latvia_decisions.npz', allow_pickle=False) as predictions:
        ids = predictions['source_path']
        reference, edge, remote = (predictions[key] for key in ('y', 'edge_probability', 'remote_probability'))
        labels = list(predictions['policies'])
        per_rule = predictions['pred']
        pred = decide(edge, remote, rule)
        assert np.array_equal(pred, per_rule[labels.index('fixed_asymmetric')])
        if len(set(ids)) != 1190 or tuple(np.bincount(reference)) != (207, 983):
            raise ValueError('Latvian source inventory differs')
        cohorts = {'LatviaAll': np.arange(len(ids)), 'LatviaScreened': predictions['representatives']}
        lookup = {(r['cohort'],r['policy']): r for r in read_csv('latvia_dual_outcomes.csv')}
        for cohort, indices in cohorts.items():
            ref = reference[indices]
            for name, policy_cases in zip(labels, per_rule, strict=True):
                rec = lookup[cohort,name]
                decisions = policy_cases[indices]
                assert ((decisions == 1) & (ref == 0)).sum() == int(rec['new_fp'])
                assert ((decisions == 0) & (ref == 1)).sum() == int(rec['new_fn'])
            rows.append(measure(cohort, 'cornell_directional', ref, edge[indices] >= EDGE_CUTOFF, pred[indices],
                                select_queries(edge[indices], rule), predictions['jpeg_bytes'][indices],
                                training['fit_n'], setup))
        with (RESULTS / 'latvia_local_directional_rule.json').open() as f:
            local_training = json.load(f)
        local_rule = unpack((RESULTS / 'latvia_local_directional_rule.bin').read_bytes())
        source_index = {p: i for i, p in enumerate(ids)}
        split = read_csv('latvia_local_date_split.csv')
        test = np.array([source_index[r['source_path']] for r in split if r['status'] == 'test'])
        early_dates = {r['capture_date'] for r in split if r['status'] == 'calibration'}
        late_dates = {r['capture_date'] for r in split if r['status'] == 'test'}
        assert max(early_dates) < min(late_dates) and len(test) == 579
        local_cases = [r for r in read_csv('latvia_local_directional_predictions.csv') if r['method'] == 'latvia_local']
        local_cases = {r['source_path']: int(r['prediction']) for r in local_cases}
        for name, policy, cal_count, setup_cost in [
            ('cornell_directional', rule, training['fit_n'], setup),
            ('latvia_date_calibrated', local_rule, local_training['calibration_labels'],
             local_training['calibration_upload_bytes'] + local_training['deployment_update_bytes'])
        ]:
            pred = decide(edge[test], remote[test], policy)
            if name == 'latvia_date_calibrated':
                assert np.array_equal(pred, [local_cases[p] for p in ids[test]])
            rows.append(measure('LatviaLaterDates', name, reference[test], edge[test] >= EDGE_CUTOFF, pred,
                                select_queries(edge[test], policy), predictions['jpeg_bytes'][test], cal_count, setup_cost))
    return rows


def by_source_label():
    cases = read_csv('directional_predictions.csv')
    subgroups = {}
    for cohort, table in [('cornell', 'fgvc_edge.csv'), ('kashmir', 'transfer_manifest.csv')]:
        by_path = {r['source_path']: r['source_label'] for r in read_csv(table)}
        for r in cases:
            if r['cohort'] == cohort:
                key = cohort, by_path[r['path']]
                subgroups.setdefault(key, []).append((int(r['reference']), int(r['original']), int(r['decision'])))
    with np.load(RESULTS / 'latvia_decisions.npz', allow_pickle=False) as result:
        policies = list(result['policies'])
        frozen, pred = result['pred'][[policies.index('published_edge'), policies.index('fixed_asymmetric')]]
        for truth, original, decision in zip(result['y'], frozen, pred, strict=True):
            subgroups.setdefault(('LatviaAll', ('Publisher-negative', 'Scab')[int(truth)]), []).append(
                (truth, original, decision))
    rows = []
    for (cohort, label), cases in subgroups.items():
        ref, edge, new = np.asarray(cases).T
        assert len(set(ref)) == 1
        old_error, new_error = edge != ref, new != ref
        rows.append(dict(site=cohort, source_label=label, error_type='fn' if ref[0] else 'fp', n=len(ref),
                         before=int(old_error.sum()), after=int(new_error.sum()),
                         fixed=int((old_error & ~new_error).sum()), introduced=int((~old_error & new_error).sum())))
    return rows


def main():
    detail = by_source_label()
    with (RESULTS / 'joint_error_by_label.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(detail[0]))
        w.writeheader()
        w.writerows(detail)
    rows = reconstruct()
    with (RESULTS / 'joint_error_evidence.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r['site'], r['rule'], f"FP {r['fp_before']} -> {r['fp_after']}, FN {r['fn_before']} -> {r['fn_after']}")


if __name__ == '__main__':
    main()

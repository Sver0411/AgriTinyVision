"""Compare edge updating, routing and their combination under one total byte cap."""

import argparse
import csv
import hashlib
import zipfile
from pathlib import Path

import numpy as np

from directional_policy import EDGE_CUTOFF, REPLY, REQUEST, probabilities_from_logits
from edge_adaptation_wire import predict_from_head, quantized
from evaluate_upload import rank_selection

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
FEATURES = ROOT / 'data/features'
FROZEN_PROTOCOL = 'cd365441a6da896618e2c28212a562593cca3a21babe097dfd3881fd165428f4'


def table(filename):
    with (OUT / filename).open() as f:
        return list(csv.DictReader(f))


def write(filename, rows):
    with (OUT / filename).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dataset(site):
    if site == 'cornell':
        with np.load(OUT / 'edge_adaptation_predictions.npz') as z:
            paths, y = z['test_paths'], z['test_y']
            local = z['score'][:, list(z['budgets']).index(128), list(z['methods']).index('local_int8'), :]
            assert np.array_equal(z['seeds'], np.arange(20))
        rows = {r['path']: r for r in table('directional_predictions.csv') if r['cohort'] == 'cornell'}
        edge_p = np.array([float(rows[p]['edge_p']) for p in paths], dtype=np.float32)
        expert = np.array([float(rows[p]['reply_p']) >= .5 for p in paths])
        sizes = np.array([int(rows[p]['payload_jpeg_bytes']) for p in paths])
        return paths, y, edge_p, expert, sizes, local
    with np.load(FEATURES / f'{site}.npz') as z:
        paths, acts = z['source_path'], z['edge']
        if site == 'latvia':
            y, logits, sizes = z['y'], z['logits'], z['jpeg_bytes']
            exp = np.exp(logits - logits.max(axis=1, keepdims=True))
            edge_p = exp[:, 1] / exp.sum(axis=1)
            expert = probabilities_from_logits(z['dino_balanced_margin'].reshape(-1)) >= .5
        else:
            rows = {r['path']: r for r in table('directional_predictions.csv') if r['cohort'] == 'kashmir'}
            y = np.array([int(rows[p]['reference']) for p in paths])
            edge_p = np.array([float(rows[p]['edge_p']) for p in paths], dtype=np.float32)
            expert = np.array([float(rows[p]['reply_p']) >= .5 for p in paths])
            sizes = np.array([int(rows[p]['payload_jpeg_bytes']) for p in paths])
    q, _ = quantized(acts)
    with zipfile.ZipFile(OUT / 'edge_adaptation_heads.zip') as archive:
        local = np.array([predict_from_head('local_int8', archive.read(f's{s:02d}-b128/local_int8.bin'), q)
                          for s in range(20)])
    return paths, y, edge_p, expert, sizes, local


def run_site(site, setup_lookup):
    paths, y, edge_p, expert, jpeg, local = dataset(site)
    assert len(set(paths)) == len(y) and local.shape == (20, len(paths))
    costs = jpeg.astype(np.int64) + REQUEST.size + REPLY.size
    edge = edge_p >= EDGE_CUTOFF
    edge_order = np.lexsort((paths, np.abs(edge_p - EDGE_CUTOFF)))
    no_query = np.zeros(len(y), dtype=bool)
    reports, summary, predictions, queries = [], [], [], []
    for pct in (10, 25, 50):
        budget = int(costs.sum()) * pct // 100
        frozen_review = rank_selection(edge_order, costs, budget)
        for label_location in ('local', 'remote'):
            for seed in range(20):
                setup = setup_lookup[seed,label_location]
                feasible = setup <= budget
                score = local[seed]
                adapted = score >= 0 if feasible else edge
                if feasible:
                    order = np.lexsort((paths, abs(score)))
                    jointly_reviewed = rank_selection(order, costs, budget - setup)
                else:
                    jointly_reviewed = frozen_review
                cases = (
                    ('frozen', edge, no_query, 0),
                    ('update_only', adapted, no_query, setup if feasible else 0),
                    ('review_only', edge, frozen_review, 0),
                    ('joint', adapted, jointly_reviewed, setup if feasible else 0),
                )
                for method, base, queried, charged_setup in cases:
                    pred = np.where(queried, expert, base)
                    fp = int(np.count_nonzero((y == 0) & pred))
                    fn = int(np.count_nonzero((y == 1) & ~pred))
                    ba = 1.0 - .5 * (fp / np.count_nonzero(y == 0) + fn / np.count_nonzero(y == 1))
                    total = int(charged_setup + costs[queried].sum())
                    assert total <= budget
                    reports.append(dict(site=site, cap_percent=pct, cap_bytes=budget, review_location=label_location,
                                        seed=seed, method=method, n=len(y), setup_fits_cap=int(feasible), setup_bytes=charged_setup,
                                        review_frames=int(queried.sum()), total_bytes=total, fp=fp, fn=fn, balanced_accuracy=float(ba)))
                    predictions.append(pred)
                    queries.append(queried)
            rows = [r for r in reports if r['cap_percent']==pct and r['review_location']==label_location]
            grouped = {m:[r for r in rows if r['method']==m] for m in ('frozen','update_only','review_only','joint')}
            ba = {m:np.array([r['balanced_accuracy'] for r in g]) for m,g in grouped.items()}
            gain = ba['joint'] - np.maximum(ba['update_only'],ba['review_only'])
            summary.append(dict(site=site, cap_percent=pct, cap_bytes=budget, review_location=label_location,
                                seeds_feasible=sum(r['setup_fits_cap'] for r in grouped['joint']),
                                **{f'{m}_mean_ba':float(x.mean()) for m,x in ba.items()},
                                joint_gain_vs_stronger_component_pp=float(100 * gain.mean()),
                                joint_beats_both_draws=int((gain > 1e-12).sum()),
                                joint_mean_total_bytes=float(np.mean([r['total_bytes'] for r in grouped['joint']])),
                                joint_mean_review_frames=float(np.mean([r['review_frames'] for r in grouped['joint']]))))
    np.savez_compressed(OUT / f'joint_budget_{site}.npz', source_path=paths, y=y, original=edge, expert=expert,
                        application_bytes=costs, predictions=np.array(predictions, dtype=np.uint8),
                        queries=np.array(queries, dtype=np.uint8))
    return reports, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sites', nargs='*', default=['cornell','kashmir','latvia'])
    args = parser.parse_args()
    assert hashlib.sha256((ROOT / 'protocol/JOINT_BUDGET.txt').read_bytes()).hexdigest() == FROZEN_PROTOCOL
    lookup = {}
    for row in table('edge_adaptation_metrics.csv'):
        if row['method'] == 'local_int8' and row['budget'] == '128':
            for where in ('local', 'remote'):
                lookup[int(row['seed']), where] = int(row[f'{where}_label_total_bytes'])
    details, summaries = [], []
    for site in args.sites:
        rows, summary = run_site(site, lookup)
        details.extend(rows)
        summaries.extend(summary)
    write('joint_budget_runs.csv', details)
    write('joint_budget_summary.csv', summaries)
    for r in summaries:
        if r['cap_percent'] == 25:
            print(r)


if __name__ == '__main__':
    main()

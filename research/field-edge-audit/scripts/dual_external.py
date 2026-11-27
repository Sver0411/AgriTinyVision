"""Replay the policies named before inspecting predictions on the Latvian archive."""

import csv
import json
import zipfile
from collections import defaultdict

import numpy as np

from directional_policy import decide, probabilities_from_logits, unpack
from dual_fusion_portable import local_decision, remote_decision
from edge_adaptation_wire import normalized, predict_from_head, quantized
from edge_model import ROOT


def pair_counts(reference, original, prediction):
    result = {}
    for label, error in [(0, 'fp'), (1, 'fn')]:
        selected = reference == label
        old_error = selected & (original != reference)
        new_error = selected & (prediction != reference)
        result.update({
            f'n_class{label}': int(selected.sum()),
            f'original_{error}': int(old_error.sum()),
            f'new_{error}': int(new_error.sum()),
            f'fixed_{error}': int((old_error & ~new_error).sum()),
            f'introduced_{error}': int((~old_error & new_error).sum()),
        })
    result['both_strict'] = int(result['new_fp'] < result['original_fp']
                                and result['new_fn'] < result['original_fn'])
    return result


def representative_indices(names, reference, filehash, phash):
    """Select a stable representative from each reference-consistent component."""
    parent = list(range(len(reference)))

    def root(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    hashes = [int(s, 16) if isinstance(s, (str, np.str_)) else int(s) for s in phash]
    for j, h in enumerate(hashes):
        for i in range(j):
            if filehash[i] == filehash[j] or (h ^ hashes[i]).bit_count() <= 4:
                parent[root(j)] = root(i)
    groups = defaultdict(list)
    for i in range(len(reference)):
        groups[root(i)].append(i)
    return np.array([min(g, key=lambda k: names[k]) for g in groups.values()
                     if len(set(reference[g])) == 1], dtype=int)


def run():
    results = ROOT / 'results'
    features = ROOT / 'data/features'
    with np.load(features / 'latvia.npz', allow_pickle=False) as a:
        names = a['source_path']
        reference = a['y'].astype(int)
        edge_margin = a['edge_margin'].reshape(-1)
        remote_margin = a['dino_balanced_margin'].reshape(-1)
        activations = a['edge']
        multiview = a['multiview']
        filehash, phash = a['source_sha256'], a['phash']
        jpeg_bytes, jpeg448_bytes = a['jpeg_bytes'], a['jpeg448_bytes']
        logits = a['logits']
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        edge_probability = exp[:, 1] / exp.sum(axis=1)
    assert len(names) == 1190 and list(np.bincount(reference)) == [207, 983]
    original = edge_probability >= np.float32(.30)
    remote_probability = probabilities_from_logits(remote_margin)
    predictions = {'published_edge': original, 'source_balanced_dino': remote_margin >= 0}

    quantized_features, _ = quantized(activations)
    with zipfile.ZipFile(results / 'sensitivity64_heads.zip') as heads, np.load(results / 'sensitivity64_predictions.npz') as c:
        method = list(c['methods']).index('local_int8')
        operating_point = list(c['operating_points']).index('target95')
        for j, seed in enumerate(c['seeds']):
            cutoff = c['threshold'][j, method, operating_point]
            blob = heads.read(f's{seed:02d}/local_int8.bin')
            score = predict_from_head('local_int8', blob, quantized_features)
            predictions[f'int8_high_sensitivity_{seed:02d}'] = score >= cutoff

    fusion = json.loads((results / 'dual_fusion_deployment.json').read_text())
    fused = []
    for edge, remote in zip(edge_margin, remote_margin, strict=True):
        local = local_decision(fusion, float(edge))
        fused.append(remote_decision(fusion, float(edge), float(remote)) if local is None else local)
    predictions['fixed_fusion_450_seed0'] = np.array(fused)

    rule = json.loads((results / 'dual_multiview_rule.json').read_text())
    with np.load(features / 'dual_mv_head.npz') as head:
        x = (normalized(multiview) - head['std_mean']) / head['std_scale']
        score = x @ head['coef'] + head['intercept'][0]
    predictions['fixed_multiview'] = original if rule['uses_frozen_edge'] else score >= rule['threshold']

    asymmetric = unpack((results / 'directional_rule.bin').read_bytes())
    predictions['fixed_asymmetric'] = decide(edge_probability, remote_probability, asymmetric)
    reps = representative_indices(names, reference, filehash, phash)
    rows = []
    for policy, p in predictions.items():
        for cohort, idx in [('LatviaAll', np.arange(len(reference))), ('LatviaScreened', reps)]:
            counts = pair_counts(reference[idx], original[idx], p[idx])
            rows.append(dict(cohort=cohort, policy=policy, **counts))
    with (results / 'latvia_dual_outcomes.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(results / 'latvia_decisions.npz', source_path=names, source_sha256=filehash,
                        y=reference, policies=list(predictions), pred=np.array(list(predictions.values()), dtype=np.uint8),
                        representatives=reps, edge_probability=edge_probability, remote_probability=remote_probability,
                        jpeg_bytes=jpeg_bytes, jpeg448_bytes=jpeg448_bytes)
    print(f'Latvia: {len(names)} official images, {len(reps)} representatives, {len(predictions)} policies')


if __name__ == '__main__':
    run()

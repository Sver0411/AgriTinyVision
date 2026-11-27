"""Check the primary acquisition, precision and threshold claims from public aggregates."""
import csv
from pathlib import Path

RESULT = Path(__file__).resolve().parents[1] / 'results'


def rows(path):
    with (RESULT/path).open(newline='') as f:
        return list(csv.DictReader(f))


def main():
    data = rows('purchase/summary.csv')
    assert sum(int(r['runs']) for r in data) == 221760
    assert all(float(r['prefix_overdraft']) == 0 for r in data)
    fields = 'site backbone label_cap grant_mode'.split()
    primary = {tuple(r[k] for k in fields)+(r['method'],): r for r in data if r['cap_pct'] == '25'}
    for baseline, wins in [('matched_rank',8),('always_edgeboost_rank',4),('freeze',13)]:
        assert sum(float(row['ba']) > float(primary[key[:-1]+(baseline,)]['ba'])+1e-12
                   for key,row in primary.items() if key[-1] == 'adaptive') == wins
    for backbone, expected in [('v2',(31.,67.,32.,67.)),('v2grid',(45.,67.,47.,67.)),
                               ('v4',(206.,12.,196.05,15.05))]:
        actual = tuple(float(primary['latvia',backbone,'32','earned',method][metric])
                       for method in ('freeze_guard','adaptive_guard') for metric in ('fn','fp'))
        assert actual == expected, (backbone, actual)
    rank = rows('acquisition_quantization_paired.csv')
    for site, train, expected in [('cornell','fp32',6),('cornell','grid',0),
                                 ('latvia','fp32',13),('latvia','grid',10)]:
        group = [r for r in rank if r['site'] == site and r['scenario'] == 'online_remote'
                 and r['training_features'] == train and r['serving_features'] == 'grid']
        assert len(group) == 45 and sum(int(r['order_reversed']) for r in group) == expected
    score = {(r['site'],r['backbone'],r['rule'],r['labels']):r
             for r in rows('purchase/backbone_diagnostic.csv')}
    for (site,backbone,rule,n), row in score.items():
        if rule == 'recalibrate_threshold':
            assert abs(float(row['auc'])-float(score[site,backbone,'native','0']['auc'])) < 1e-12
    assert round(float(score['latvia','v2','native','0']['auc']),3) == .449
    assert round(float(score['latvia','v4','native','0']['auc']),3) == .791
    assert round(float(score['latvia','v4','replace_head','32']['auc']),3) == .622
    print('Verified 221,760 runs in the aggregates, 36 primary cells, alert trade-offs and precision rankings.')


if __name__ == '__main__':
    main()

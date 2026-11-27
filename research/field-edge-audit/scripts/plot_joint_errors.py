"""Plot class-specific rate changes and exact routing randomization probabilities."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from joint_error_evidence import ROOT, reconstruct
from random_query_budget import strict_reduction_probability

STYLES = {
    'cornell': ('Cornell', '#005f73', 'o', '-'),
    'kashmir': ('Kashmir', '#ab4b22', 's', '--'),
    'LatviaAll': ('Latvia (scab)', '#59428b', '^', ':'),
}


def main():
    rows = [r for r in reconstruct() if r['rule'] == 'cornell_directional' and r['site'] in STYLES]
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':8.5, 'axes.spines.top':False,
                         'axes.spines.right':False, 'pdf.fonttype':42})
    fig, (effects, chance) = plt.subplots(1, 2, figsize=(7.7, 2.68))
    fig.subplots_adjust(left=.09, bottom=.26, right=.99, top=.88, wspace=.4)
    xs = [100 * (r['fp_rate_after'] - r['fp_rate_before']) for r in rows]
    ys = [100 * (r['fn_rate_after'] - r['fn_rate_before']) for r in rows]
    xlo, xhi = min(-2, min(xs)) - 3, max(2, max(xs)) + 3
    ylo, yhi = min(-2, min(ys)) - 1, max(2, max(ys)) + 1
    effects.fill_between([xlo, 0], ylo, 0, color='#ddf2ed')
    effects.axhline(0, color='#78878a', lw=.8)
    effects.axvline(0, color='#78878a', lw=.8)
    handles = []
    for x, y, r in zip(xs, ys, rows, strict=True):
        name, color, marker, linestyle = STYLES[r['site']]
        effects.plot(x, y, marker=marker, ms=6, color=color, zorder=3)
        q = np.linspace(0, 1, 201)
        probability = [strict_reduction_probability(r['fp_fixed'], r['fp_introduced'], t)
                       * strict_reduction_probability(r['fn_fixed'], r['fn_introduced'], t) for t in q]
        line, = chance.plot(q, probability, linestyle=linestyle, color=color, lw=1.8,
                            marker=marker, markevery=[50], ms=4, label=name)
        handles.append(line)
    effects.set(xlim=(xlo, xhi), ylim=(ylo, yhi), xlabel='Change in false-positive rate (pp)',
                ylabel='Change in false-negative rate (pp)')
    effects.set_title('A   Change with full review', loc='left', fontsize=9, weight='bold')
    chance.set(xlim=(0, 1), ylim=(-.035, 1.04), xlabel='Probability of uploading a photo',
               ylabel='Probability both counts fall')
    chance.set_title('B   Independent random uploads', loc='left', fontsize=9, weight='bold')
    chance.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    chance.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    chance.grid(axis='y', color='#d8dfdf', lw=.6)
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, -.01), ncol=3, frameon=False)
    fig.savefig(ROOT / 'paper/figures/joint_errors.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()

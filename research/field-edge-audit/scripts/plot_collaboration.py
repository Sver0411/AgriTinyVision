"""Plot the budget and selection ablations using their verified cohort summaries."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]


def rows(file):
    with (ROOT / 'results' / file).open() as f:
        return {(r['site'], int(r['cap_percent'])): r for r in csv.DictReader(f) if r['review_location'] == 'local'}


def main():
    main = rows('joint_budget_summary.csv')
    ablation = rows('joint_budget_ablation.csv')
    fig, axes = plt.subplots(1, 3, figsize=(7.75, 2.55), sharey=True)
    plt.rcParams['pdf.fonttype'] = 42
    specs = [('Updated edge + selective review','joint_mean_ba','#005f73','o','-'),
             ('Updated edge + random review','uniform_review_mean_ba','#946130','^','--'),
             ('Edge update only','update_only_mean_ba','#515b63','s',':'),
             ('Original edge + selective review','review_only_mean_ba','#785c93','D','-.')]
    for ax, (site,title) in zip(axes, [('cornell','Cornell'),('kashmir','Kashmir'),('latvia','Latvia (scab)')]):
        for label,key,color,marker,style in specs:
            data = ablation if key == 'uniform_review_mean_ba' else main
            ax.plot([10,25,50], [100*float(data[site,p][key]) for p in [10,25,50]],
                    linestyle=style, marker=marker, color=color, label=label, linewidth=1.5, markersize=3.7)
        ax.set_title(title, fontsize=9, loc='left', weight='bold')
        ax.set_xticks([10,25,50],['10%','25%','50%'],fontsize=8)
        ax.tick_params(axis='y',labelsize=8)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y', color='#e3e7e5', lw=.65)
        ax.set_ylim(43,92)
        ax.yaxis.set_major_locator(MaxNLocator(5))
    axes[0].set_ylabel('Balanced accuracy (%)',fontsize=8)
    fig.subplots_adjust(top=.88,bottom=.35,left=.075,right=.995,wspace=.15)
    fig.text(.53,.165,'Total application bytes as a share of full remote review',ha='center',fontsize=8)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center',ncol=2,frameon=False,fontsize=7.5)
    fig.savefig(ROOT / 'paper/figures/collaboration.pdf')
    plt.close(fig)


if __name__=='__main__':
    main()

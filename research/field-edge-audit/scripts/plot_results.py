"""Produce vector figures from the experiment tables."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'paper/figures'
OUT.mkdir(parents=True,exist_ok=True)


def read(file):
    with (ROOT/'results'/file).open() as f:return list(csv.DictReader(f))


def group(rows,method,b):
    return [r for r in rows if r['method']==method and int(r['budget'])==b]


def main():
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    data=read('edge_adaptation_metrics.csv');transfer=read('transfer_outcomes.csv')
    colors={'frozen':'#64717A','scalar':'#A66328','local_int8':'#006D72','dino':'#445DB2','balanced':'#39333D'}
    names={'scalar':'Edge score only','local_int8':'Edge INT8 head','dino':'Remote DINO head'}
    fig,axes=plt.subplots(1,2,figsize=(8.1,2.72),sharey=True,layout='constrained')
    for ax,rows,title in zip(axes,(data,transfer),('Cornell  ·  held-out images','Kashmir  ·  transferred heads')):
        ax.set_title(title,loc='left',fontweight='bold',fontsize=10)
        baseline=next(float(r['balanced_accuracy']) for r in rows if r['method']=='frozen')*100
        if rows is data:
            source_rows=read('specialist_balanced.csv'); ids={r['source_path'] for r in read('edge_adaptation_split.csv') if r['status']=='test'}
            u=[r for r in source_rows if r['dataset']=='FGVC7' and r['source_path'] in ids]
            source=50*sum(np.mean([v['pred_diseased']==v['y'] for v in u if v['y']==c]) for c in ('0','1'))
        else:
            source=float(next(r['balanced_accuracy'] for r in rows if r['method']=='plantdoc_balanced'))*100
        ax.axhline(baseline,linestyle=':',color=colors['frozen'],label='Frozen edge')
        ax.axhline(source,ls='--',color=colors['balanced'],lw=1,label='Source-balanced remote')
        for m in ('scalar','local_int8','dino'):
            v=[np.array([float(row['balanced_accuracy'])*100 for row in group(rows,m,b)]) for b in (8,32,128)]
            means=[a.mean() for a in v];lower=[np.quantile(a,.25) for a in v];upper=[np.quantile(a,.75) for a in v]
            x=np.arange(3)
            ax.fill_between(x,lower,upper,alpha=.09,color=colors[m],linewidth=0)
            ax.plot(x,means,'o-',color=colors[m],label=names[m],lw=1.65,ms=4)
        ax.set_xticks([0,1,2],['8','32','128']);ax.set_xlabel('Cornell labels used for adaptation')
        ax.grid(axis='y',alpha=.16);ax.set_ylim(48,93)
    axes[0].set_ylabel('Balanced accuracy (%)')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=7.5,frameon=False)
    fig.savefig(OUT/'adaptation.pdf');plt.close(fig)
    dat=read('head_vs_oracle.csv')
    fig,axes=plt.subplots(1,2,figsize=(8.1,2.65),layout='constrained',sharey=True)
    for ax,b in zip(axes,[32,128]):
        a=[r for r in dat if int(r['budget'])==b]
        a.sort(key=lambda r:float(r['oracle_with_balanced_head_bytes']))
        xs=np.arange(1,len(a)+1)
        for key,label,col,style in [('oracle_with_balanced_head_bytes','Ideal JPEG router','#445DB2','o-'),('observed_update_bytes_local_review','Head, labels available locally','#006D72','-'),('observed_update_bytes_remote_review','Head, remote image review','#A66328','.-')]:
            ax.plot(xs,[float(r[key])/1024 for r in a],style,lw=1.3,ms=3,color=col,label=label)
        ax.set_title(f'{b} calibration labels',loc='left',fontweight='bold',fontsize=10)
        ax.set_xlabel('Calibration draws, sorted by oracle cost');ax.set_xticks([1,5,10,15,20]);ax.set_yscale('log');ax.grid(axis='y',alpha=.2,which='both')
    axes[0].set_ylabel('Total communication (KiB)')
    h,l=axes[0].get_legend_handles_labels()
    fig.legend(h,l,loc='outside lower center',ncol=3,fontsize=8,frameon=False)
    fig.savefig(OUT/'oracle.pdf');plt.close(fig)


if __name__=='__main__':
    main()

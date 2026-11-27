"""Plot a matched-cohort precision sensitivity of the historical streaming benchmark."""
import csv
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results"
STATES = (("fp32", "fp32"), ("fp32", "grid"), ("grid", "grid"))
FOCUS = {
    "ce": ("Balanced CE", "#784c87", "o", "-"),
    "ova": ("OvA", "#007974", "s", "-"),
    "edgeboost_fixed": ("EdgeBoost fixed gate", "#335d9d", "D", "-"),
    "mao": ("Mao two-stage", "#a54921", "^", "-"),
    "edgeboost_rank": ("Uncertainty / EdgeBoost rank", "#505359", "v", "--"),
}


def table(name):
    with (RESULT / name).open(newline="") as f:
        return list(csv.DictReader(f))


def main():
    rows = [r for r in table("acquisition_quantization_replay_summary.csv") if r["scenario"]=="online_remote" and r["method"]!="uncertainty"]
    reversals = [r for r in table("acquisition_quantization_paired.csv") if r["scenario"]=="online_remote"]
    by = {(r["site"],r["training_features"],r["serving_features"],r["method"]):r for r in rows}
    methods = sorted({r["method"] for r in rows})
    assert len(methods)==10
    mpl.rcParams.update({"font.family":"DejaVu Sans", "font.size":9.8, "pdf.fonttype":42,
                         "axes.spines.right":False,"axes.spines.top":False,"axes.spines.bottom":False})
    fig, axes = plt.subplots(1,2,figsize=(9.3,4.65),sharey=True)
    fig.subplots_adjust(left=.08,right=.97,top=.77,bottom=.26,wspace=.18)
    fig.text(.06,.94,"Camera precision changes the ordering of review policies",
             ha="left",va="center",fontsize=15.3,weight="semibold",color="#20292d")
    fig.text(.06,.884,"Remote annotation for 128 Cornell images  ·  25% source-planned byte quota  ·  one-frame streaming",
             ha="left",va="center",fontsize=9.4,color="#45565e")
    for ax, site, title in zip(axes,("cornell","latvia"),("Cornell · 1,353 images","Latvia · 1,190 images"),strict=True):
        ax.set_title(title,loc="left",fontweight="bold",fontsize=11,pad=7)
        ax.set(axisbelow=True, xlim=(-.1,2.10), ylim=(10.5,.48), yticks=[1,2,3,4,5,6,7,8,9,10])
        ax.grid(axis="y",lw=.55,color="#dfE3E5")
        ax.spines["left"].set_color("#a6b0b5")
        ax.set_xticks([0,1,2],["FP32 fit\nFP32 serving","FP32 fit\nGrid serving","Grid fit\nGrid serving"])
        ax.tick_params(axis="both",length=0,pad=5,labelsize=8.5)
        for name in methods:
            sequence = [int(by[site,*state,name]["rank_vs_unique_methods"]) for state in STATES]
            if name in FOCUS:
                _,color,marker,style = FOCUS[name]
                ax.plot(sequence,linestyle=style,linewidth=1.8,color=color,marker=marker,
                        markersize=5.6,markeredgecolor="white",markeredgewidth=.55,zorder=3)
            else:
                ax.plot(sequence,color="#c8ced0",linewidth=.9,zorder=1)
        counts = []
        for state in STATES[1:]:
            subset=[r for r in reversals if (r["site"],r["training_features"],r["serving_features"])==(site,*state)]
            counts.append(sum(int(r["order_reversed"]) for r in subset))
            assert len(subset)==45
        ax.text(.5,-.29,f"Pairs reversed vs. FP32   {counts[0]}/45 on grid   ·   {counts[1]}/45 after refitting",
                ha="center",fontsize=8.2,transform=ax.transAxes,color="#46545b")
    axes[0].set_ylabel("Rank by mean balanced accuracy",fontsize=9,labelpad=8)
    handles = [Line2D([],[],color=color,marker=marker,ls=style,lw=1.8,ms=5,label=label)
               for label,color,marker,style in FOCUS.values()]
    fig.legend(handles=handles,loc="lower left",bbox_to_anchor=(.06,.05),frameon=False,ncol=5,
               columnspacing=1.05,handlelength=1.9,handletextpad=.4,fontsize=8.7)
    fig.text(.06,.026,"Ranks use all ten distinct policies on the same finite cohorts and 20 calibration draws. Grid uses exported-scale CPU arithmetic.",
             fontsize=8.15,color="#46545b")
    path=RESULT/"purchase/precision_ranks.pdf"
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


if __name__=="__main__":
    main()

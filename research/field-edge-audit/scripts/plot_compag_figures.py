"""Generate publication-quality figures for the COMPAG (Elsevier) BASA manuscript."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "figures"

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.2,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 7.4,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

PALETTE = {
    "basa": "#0B6E75",
    "freeze": "#6C757D",
    "adaptive": "#C85A32",
    "eb": "#3B75AF",
    "ova": "#8E6BBF",
    "ce": "#D19A28",
    "clamped_ce": "#B87D14",
}


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def plot_architecture() -> None:
    """Figure 1: BASA system and unified single-ledger protocol diagram."""
    fig, ax = plt.subplots(figsize=(7.2, 2.55))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 36)
    ax.axis("off")

    def box(x, y, w, h, title, body, fc="#F5F9FA", ec="#0B6E75", lw=1.05, title_color="#094D52"):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.3,rounding_size=1.1",
            facecolor=fc, edgecolor=ec, linewidth=lw,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h - 1.5, title, ha="center", va="top",
                fontsize=7.0, fontweight="bold", color=title_color)
        ax.text(x + w / 2, y + h / 2 - 1.4, body, ha="center", va="center",
                fontsize=6.25, color="#22303A", linespacing=1.28)

    def arrow(x1, y1, x2, y2, label="", dy=1.4, color="#334E58", ls="-"):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>,head_width=0.26,head_length=0.42",
                            color=color, lw=0.95, linestyle=ls),
        )
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, label, ha="center", va="center",
                    fontsize=6.0, color="#334E58")

    ax.add_patch(patches.FancyBboxPatch((0.6, 1.2), 62.6, 33.4, boxstyle="round,pad=0.3,rounding_size=1.3",
                                        facecolor="#FAFCFC", edgecolor="#B8CED2", lw=0.8, ls="--"))
    ax.text(2.0, 33.4, "Edge Camera (ESP32-S3, 1.8 KB BASA state)", fontsize=6.9, fontweight="bold", color="#0B6E75")

    ax.add_patch(patches.FancyBboxPatch((69.2, 1.2), 30.2, 33.4, boxstyle="round,pad=0.3,rounding_size=1.3",
                                        facecolor="#FFFDF8", edgecolor="#DFCAA0", lw=0.8, ls="--"))
    ax.text(70.5, 33.4, "Cloud Service & Annotator", fontsize=6.9, fontweight="bold", color="#8A5E14")

    box(2.0, 17.8, 26.8, 12.6,
        "1. Horizon-Amortized Ledger",
        "Quota B = N·r (prepaid / earned)\nLabel cap k ≤ ρB / c_cal\n(ρ = 0.40 prepaid, 0.25 earned)",
        fc="#EEF7F8", ec="#0B6E75")

    box(33.2, 17.8, 28.6, 12.6,
        "2. Quantile-Stratified Selection",
        "Unlabelled local reservoir\nUniform quantiles of source p0(x)\nLabel-free · tail-variance guard",
        fc="#EEF7F8", ec="#0B6E75")

    box(70.6, 17.8, 27.4, 12.6,
        "3. ASAR Convex Fitting",
        "Human labels on k JPEGs\nSource-anchored prior w0\nOOF CV over λ ∈ {∞, 25, 5, 1}",
        fc="#FFF8EB", ec="#C28B28", title_color="#7A520B")

    box(70.6, 2.8, 27.4, 10.6,
        "Frozen Cloud Expert",
        "PlantDoc-balanced DINOv2\n224-px JPEG + 20 B header",
        fc="#FFF8EB", ec="#C28B28", title_color="#7A520B")

    box(2.0, 2.8, 59.8, 10.6,
        "4. On-Device INT8 Inference & Causal Single-Frame Gate",
        "Install 1,294-B EQ01 head once funded · Maintain 64-frame rolling urgency\n"
        "Defer frame t when π_t ≥ 1 − rem / ((N−t)·ŝ) and bytes fit · Optional Recall Guard",
        fc="#F2F6FA", ec="#3B75AF", title_color="#234C75")

    arrow(29.1, 24.1, 32.9, 24.1, "k", dy=1.6)
    arrow(62.1, 25.8, 70.3, 25.8, "k JPEGs\n+ int8 x", dy=2.7)
    arrow(70.3, 20.8, 62.1, 20.8, "1,294-B\nEQ01", dy=-2.7, color="#0B6E75")
    ax.annotate("", xy=(47.5, 13.7), xytext=(47.5, 17.5),
                arrowprops=dict(arrowstyle="-|>,head_width=0.26,head_length=0.42", color="#0B6E75", lw=0.95))
    arrow(62.1, 9.6, 70.3, 9.6, "JPEG", dy=1.5, color="#3B75AF")
    arrow(70.3, 6.0, 62.1, 6.0, "label", dy=-1.6, color="#3B75AF", ls="--")

    fig.tight_layout(pad=0.15)
    fig.savefig(FIG_DIR / "compag_basa_arch.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_main_comparison() -> None:
    """Figure 2: Earned-credit BA comparison across 9 site×backbone cells, including Clamped-CE."""
    basa = {(r["site"], r["backbone"], int(r["label_cap"]), int(r["cap_pct"]), r["grant_mode"], r["method"]): float(r["ba"]) * 100
            for r in load_csv(ROOT / "results/basa/summary.csv")}
    base = {(r["site"], r["backbone"], int(r["label_cap"]), int(r["cap_pct"]), r["grant_mode"], r["method"]): float(r["ba"]) * 100
            for r in load_csv(ROOT / "results/purchase/summary.csv")}
    clamp = {(r["site"], r["backbone"], int(r["label_cap"]), int(r["cap_pct"]), r["grant_mode"], r["method"]): float(r["ba"]) * 100
             for r in load_csv(ROOT / "results/basa/clamped_baselines.csv")}

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.15), sharey="col")
    sites = [("cornell", "Cornell (N=1,353)"), ("kashmir", "Kashmir (N=384)"), ("latvia", "Latvia (N=579)")]
    bbs = [("v2", "V2"), ("v2grid", "V2-grid"), ("v4", "V4")]
    methods = [
        ("Frozen", "freeze", PALETTE["freeze"], base),
        ("Pilot-Audit-16", "adaptive", PALETTE["adaptive"], base),
        ("Always-EB", "always_edgeboost_rank", PALETTE["eb"], base),
        ("Always-OvA", "always_ova", PALETTE["ova"], base),
        ("Clamped-CE", "clamped_ce", PALETTE["clamped_ce"], clamp),
        ("BASA (Ours)", "basa", PALETTE["basa"], basa),
    ]
    caps = [32, 128]

    x = np.arange(len(bbs))
    width = 0.135
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * width

    for ri, cap in enumerate(caps):
        for ci, (site, site_title) in enumerate(sites):
            ax = axes[ri, ci]
            for mi, (label, mkey, color, src) in enumerate(methods):
                vals = [src[(site, bb, cap, 25, "earned", mkey)] for bb, _ in bbs]
                edge = "#073F44" if mkey == "basa" else "none"
                lw = 0.9 if mkey == "basa" else 0.0
                bars = ax.bar(x + offsets[mi], vals, width=width * 0.92, label=label if (ri == 0 and ci == 0) else None,
                              color=color, edgecolor=edge, linewidth=lw, zorder=3)
                if mkey == "basa":
                    for bar, v in zip(bars, vals):
                        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.7, f"{v:.1f}",
                                ha="center", va="bottom", fontsize=6.1, fontweight="bold", color="#094D52")
            ax.set_xticks(x)
            ax.set_xticklabels([lbl for _, lbl in bbs])
            ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.65, zorder=0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if ri == 0:
                ax.set_title(site_title, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"Cap={cap}\nMean BA (%)")
            ymin = 42 if site == "latvia" else 68
            ymax = 76 if site == "latvia" else 90
            ax.set_ylim(ymin, ymax)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.94], h_pad=0.9, w_pad=0.6)
    fig.savefig(FIG_DIR / "compag_main_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_ablation_and_sweep() -> None:
    """Figure 3: (a) Component ablation + ASAR anchor rates; (b) Bandwidth sweep including Clamped-CE."""
    basa_rows = load_csv(ROOT / "results/basa/summary.csv")
    base_rows = load_csv(ROOT / "results/purchase/summary.csv")
    clamp_rows = load_csv(ROOT / "results/basa/clamped_baselines.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.7))

    bbs = [("v2", "V2 (FP32)\n[λ>1: 20%]"), ("v2grid", "V2-grid (INT8)\n[λ>1: 25%]"), ("v4", "MobileNetV4\n[λ>1: 87%]")]
    ablations = [
        ("Full BASA", "basa", PALETTE["basa"]),
        ("w/o ASAR Prior", "basa_no_asar", "#E07A5F"),
        ("w/o Stratified Sel.", "basa_no_strat", "#F2CC8F"),
        ("w/o Amortization", "basa_no_amort", "#81B29A"),
    ]
    x = np.arange(len(bbs))
    w = 0.19
    offs = (np.arange(len(ablations)) - 1.5) * w
    for ai, (lbl, mkey, col) in enumerate(ablations):
        means = []
        for bb, _ in bbs:
            vals = [float(r["ba"]) * 100 for r in basa_rows if r["backbone"] == bb and r["cap_pct"] == "25" and r["method"] == mkey]
            means.append(np.mean(vals))
        bars = ax1.bar(x + offs[ai], means, width=w * 0.92, label=lbl, color=col,
                       edgecolor="#073F44" if mkey == "basa" else "none", linewidth=0.8 if mkey == "basa" else 0, zorder=3)
        for bar, v in zip(bars, means):
            ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.35, f"{v:.1f}",
                     ha="center", va="bottom", fontsize=6.1, color="#22303A",
                     fontweight="bold" if mkey == "basa" else "normal")
    ax1.set_xticks(x)
    ax1.set_xticklabels([b for _, b in bbs], fontsize=7.3)
    ax1.set_ylim(66, 82.8)
    ax1.set_ylabel("Mean BA across 3 sites (%)")
    ax1.set_title("(a) Component ablation & ASAR anchoring", fontweight="bold")
    ax1.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.65, zorder=0)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#DDDDDD", fontsize=6.3)

    pcts = [10, 25, 50]
    curves = [
        ("BASA (Ours)", "basa", PALETTE["basa"], "o", "-", basa_rows),
        ("Clamped-CE", "clamped_ce", PALETTE["clamped_ce"], "P", "-", clamp_rows),
        ("Always-EB", "always_edgeboost_rank", PALETTE["eb"], "s", "--", base_rows),
        ("Always-OvA", "always_ova", PALETTE["ova"], "^", "-.", base_rows),
        ("Pilot-Audit-16", "adaptive", PALETTE["adaptive"], "D", ":", base_rows),
        ("Frozen", "freeze", PALETTE["freeze"], "v", "--", base_rows),
    ]
    for lbl, mkey, col, mk, ls, src in curves:
        ys = []
        for p in pcts:
            vals = [float(r["ba"]) * 100 for r in src if int(r["cap_pct"]) == p and r["method"] == mkey]
            ys.append(np.mean(vals))
        ax2.plot(pcts, ys, label=lbl, color=col, marker=mk, linestyle=ls,
                 linewidth=2.0 if mkey == "basa" else (1.45 if mkey == "clamped_ce" else 1.15),
                 markersize=5.0 if mkey == "basa" else 4.0)
        if mkey == "basa":
            offsets_y = [0.65, 0.65, -1.25]
            for px, py, dy in zip(pcts, ys, offsets_y):
                ax2.text(px, py + dy, f"{py:.2f}", ha="center", va="bottom" if dy > 0 else "top",
                         fontsize=6.7, fontweight="bold", color="#094D52")
    ax2.set_xticks(pcts)
    ax2.set_xticklabels(["10%\n(680 B/f)", "25%\n(1,700 B/f)", "50%\n(3,400 B/f)"])
    ax2.set_ylim(63, 78.5)
    ax2.set_ylabel("Mean BA across 36 settings (%)")
    ax2.set_title("(b) Performance across uplink byte rates", fontweight="bold")
    ax2.grid(True, linestyle=":", linewidth=0.5, alpha=0.65)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="lower right", ncol=2, frameon=True, facecolor="white", framealpha=0.92, edgecolor="#DDDDDD", fontsize=6.0, columnspacing=0.7)

    fig.tight_layout(w_pad=1.2)
    fig.savefig(FIG_DIR / "compag_ablation_and_sweep.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_precision_ranks() -> None:
    """Figure 4: Quantization rank reversals with explicit cohort scope labels."""
    rows = [r for r in load_csv(ROOT / "results/acquisition_quantization_replay_summary.csv")
            if r["scenario"] == "online_remote" and r["method"] != "uncertainty"]
    reversals = [r for r in load_csv(ROOT / "results/acquisition_quantization_paired.csv")
                 if r["scenario"] == "online_remote"]
    by = {(r["site"], r["training_features"], r["serving_features"], r["method"]): r for r in rows}
    methods = sorted({r["method"] for r in rows})
    states = (("fp32", "fp32"), ("fp32", "grid"), ("grid", "grid"))
    focus = {
        "ce": ("Balanced CE", "#784c87", "o", "-"),
        "ova": ("OvA", "#007974", "s", "-"),
        "edgeboost_fixed": ("EdgeBoost fixed gate", "#335d9d", "D", "-"),
        "mao": ("Mao two-stage", "#a54921", "^", "-"),
        "edgeboost_rank": ("Uncertainty / EdgeBoost rank", "#505359", "v", "--"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.65), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.80, bottom=0.25, wspace=0.14)
    fig.text(0.06, 0.94, "ESP32-S3 INT8 quantization alters router rankings",
             ha="left", va="center", fontsize=11.0, weight="bold", color="#20292d")
    fig.text(0.06, 0.885, "128 Cornell labels transferred to Cornell eval (1,353) and full Latvia archive (1,190) · 25% byte quota",
             ha="left", va="center", fontsize=7.6, color="#45565e")

    titles = ("Cornell eval split (1,353 images)", "Latvia full archive (1,190 images)")
    for ax, site, title in zip(axes, ("cornell", "latvia"), titles):
        ax.set_title(title, loc="left", fontweight="bold", fontsize=8.8, pad=6)
        ax.set(axisbelow=True, xlim=(-0.12, 2.12), ylim=(10.5, 0.5), yticks=list(range(1, 11)))
        ax.grid(axis="y", lw=0.5, color="#dfe3e5")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_color("#a6b0b5")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["FP32 fit\nFP32 serving", "FP32 fit\nGrid serving", "Grid fit\nGrid serving"], fontsize=7.6)
        ax.tick_params(axis="both", length=0, pad=4)
        for name in methods:
            seq = [int(by[site, *st, name]["rank_vs_unique_methods"]) for st in states]
            if name in focus:
                _, color, marker, style = focus[name]
                ax.plot(seq, linestyle=style, linewidth=1.65, color=color, marker=marker,
                        markersize=4.8, markeredgecolor="white", markeredgewidth=0.5, zorder=3)
            else:
                ax.plot(seq, color="#c8ced0", linewidth=0.85, zorder=1)
        counts = []
        for st in states[1:]:
            sub = [r for r in reversals if (r["site"], r["training_features"], r["serving_features"]) == (site, *st)]
            counts.append(sum(int(r["order_reversed"]) for r in sub))
        ax.text(0.5, -0.27, f"Pairs reversed vs. FP32: {counts[0]}/45 on grid · {counts[1]}/45 after refitting",
                ha="center", fontsize=7.4, transform=ax.transAxes, color="#46545b")

    axes[0].set_ylabel("Rank by mean balanced accuracy", fontsize=8.2, labelpad=6)
    handles = [Line2D([], [], color=c, marker=m, ls=s, lw=1.6, ms=4.5, label=l) for l, c, m, s in focus.values()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.52, 0.01), frameon=False, ncol=5,
               columnspacing=0.9, handlelength=1.6, handletextpad=0.35, fontsize=7.3)
    fig.savefig(FIG_DIR / "compag_precision_ranks.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_architecture()
    plot_main_comparison()
    plot_ablation_and_sweep()
    plot_precision_ranks()
    print("Generated all 4 COMPAG figures in", FIG_DIR)

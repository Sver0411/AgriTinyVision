"""Plot the score, error composition and equal-budget controls in the external failure analysis."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from latvia_failure import RESULTS, read


def main():
    group_rows = read("latvia_failure_groups.csv")
    rows = [
        next(r for r in group_rows if r["site"] == site and r["group"] == "all")
        for site in ("cornell", "kashmir", "latvia")
    ]
    target = rows[-1]
    oracle = {
        r["method"]: float(r["balanced_accuracy"])
        for r in read("latvia_failure_oracle.csv")
        if r["site"] == "latvia"
    }
    plt.rcParams.update(
        {
            "font.size": 8.4,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axs = plt.subplots(1, 3, figsize=(9.1, 2.8))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.26, top=0.84, wspace=0.44)
    colors = ["#216879", "#9e4f36", "#65558a"]
    x = np.arange(3)
    labels = ["Cornell", "Kashmir", "Latvia"]
    for i, r in enumerate(rows):
        axs[0].plot(
            [i - 0.11, i + 0.11],
            [
                float(r["uncertainty_error_auc_on_positive"]),
                float(r["uncertainty_error_auc_on_negative"]),
            ],
            color="#9ba4a7",
            lw=0.9,
            zorder=1,
        )
        axs[0].scatter(
            i - 0.11,
            float(r["uncertainty_error_auc_on_positive"]),
            color=colors[1],
            s=26,
            marker="o",
            label="Positive" if i == 0 else None,
        )
        axs[0].scatter(
            i + 0.11,
            float(r["uncertainty_error_auc_on_negative"]),
            color=colors[0],
            s=27,
            marker="s",
            label="Negative" if i == 0 else None,
        )
    axs[0].axhline(0.5, ls=":", lw=1, color="#656a70")
    axs[0].set(
        xticks=x, xticklabels=labels, ylim=(0.25, 0.96), ylabel="Error-detection AUROC"
    )
    axs[0].legend(
        loc="upper left",
        bbox_to_anchor=(-0.05, -0.18),
        ncol=2,
        fontsize=7,
        frameon=False,
    )
    axs[0].set_title(
        "A   Edge uncertainty ranking", loc="left", fontsize=9, weight="bold"
    )
    xn = np.arange(3)
    steps = ["Original", "Update", "Joint"]
    for err, cls, marker, color in [
        ("fp", "n_neg", "s", colors[0]),
        ("fn", "n_pos", "o", colors[1]),
    ]:
        vals = [
            100 * float(target[f"{err}_{m}"]) / float(target[cls])
            for m in ("original", "updated", "joint")
        ]
        axs[1].plot(
            xn,
            vals,
            lw=1.6,
            marker=marker,
            ms=4,
            color=color,
            label="False alarms" if err == "fp" else "Misses",
        )
    axs[1].set(
        xticks=xn,
        xticklabels=steps,
        ylim=(0, 100),
        ylabel="Class-specific error rate (%)",
    )
    axs[1].set_title("B   Latvian error rates", loc="left", fontsize=9, weight="bold")
    axs[1].legend(
        loc="upper left",
        bbox_to_anchor=(-0.16, -0.18),
        ncol=2,
        fontsize=7,
        frameon=False,
    )
    policy = [
        "actual",
        "perfect_expert_on_actual_queries",
        "optimal_router_fixed_expert",
    ]
    vals = [100 * oracle[p] for p in policy]
    axs[2].bar(xn, vals, width=0.64, color=[colors[1], colors[0], colors[2]])
    for idx, v in enumerate(vals):
        axs[2].text(idx, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    axs[2].set(
        xticks=xn,
        xticklabels=["Actual", "Perfect\nexpert", "Oracle\nrouter"],
        ylim=(0, 92),
        ylabel="Balanced accuracy (%)",
    )
    axs[2].set_title("C   Same byte ceiling", loc="left", fontsize=9, weight="bold")
    for ax in axs:
        ax.grid(axis="y", lw=0.45, color="#d5dadd")
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=8)
    fig.savefig(RESULTS / "latvia_failure_diagnostic.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()

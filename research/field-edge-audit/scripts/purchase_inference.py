"""Paired conditional inference for the paid-update replay. Reads evaluated runs only."""
import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SITES = ("cornell", "kashmir", "latvia")
BACKBONES = ("v2", "v2grid", "v4")
GRANTS = ("prepaid", "earned")
NBOOT = 2000
PAIRS = {
    "adaptive": ("freeze", "always_edgeboost_rank", "always_ova", "matched_rank", "matched_guard",
                 "matched_ova", "matched_edgeboost_fixed", "matched_compare_confidence", "matched_mao",
                 "matched_diff01", "matched_gatekeeper", "matched_ce", "bbsd", "empirical"),
    "adaptive_guard": ("freeze", "freeze_guard", "always_rank_guard", "matched_guard", "adaptive",
                        "guard_matched_rank", "guard_matched_guard", "native"),
}
PRIMARY = ("adaptive", "matched_rank")


def read_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_differences(test_y, selected, alternative):
    """Average per-image difference in probability of calling positive across matched runs."""
    if selected.shape != alternative.shape or selected.shape[1] != len(test_y):
        raise ValueError("Mismatched paired predictions")
    return selected.mean(axis=0) - alternative.mean(axis=0)


class CoupledResamples:
    """All actions for a site share the same bootstrap samples of validation units."""

    def __init__(self, site, y, groups, nboot=NBOOT):
        self.y = np.asarray(y, dtype=bool)
        n0 = np.count_nonzero(~self.y)
        n1 = np.count_nonzero(self.y)
        if min(n0, n1) == 0:
            raise ValueError("Both classes must occur")
        self.groups = groups
        self.stratified = site != "latvia"
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(("purchase-ci-v1/" + site).encode()).digest()[:8], "little"))
        if self.stratified:
            self.w0 = rng.multinomial(n0, np.full(n0, 1 / n0), size=nboot).astype(np.float64)
            self.w1 = rng.multinomial(n1, np.full(n1, 1 / n1), size=nboot).astype(np.float64)
            self.den0 = np.full(nboot, n0)
            self.den1 = np.full(nboot, n1)
            self.units = len(y)
        else:
            gs, self.idx = np.unique(groups, return_inverse=True)
            if len(gs) != 9 or any(len(str(g)) != 10 for g in gs):
                raise ValueError("Expected nine held-out Latvian calendar dates")
            self.units = len(gs)
            self.w = rng.multinomial(len(gs), np.full(len(gs), 1 / len(gs)), size=nboot).astype(np.float64)
            self.den0 = self.w @ np.bincount(self.idx, weights=~self.y, minlength=self.units)
            self.den1 = self.w @ np.bincount(self.idx, weights=self.y, minlength=self.units)
        self.valid = (self.den0 > 0) & (self.den1 > 0)

    def intervals(self, delta_positive_calls):
        y = self.y
        if self.stratified:
            fp = self.w0 @ delta_positive_calls[~y]
            fn = -(self.w1 @ delta_positive_calls[y])
        else:
            fp = self.w @ np.bincount(self.idx, weights=(~y) * delta_positive_calls, minlength=self.units)
            fn = -(self.w @ np.bincount(self.idx, weights=y * delta_positive_calls, minlength=self.units))
        ok = self.valid
        fpr = 100 * fp[ok] / self.den0[ok]
        fnr = 100 * fn[ok] / self.den1[ok]
        return {tag: np.quantile(v, (0.025, 0.975)) for tag, v in (
            ("ba_pp", -(fpr + fnr) / 2), ("fpr_pp", fpr), ("fnr_pp", fnr))}


def unit_keys(rows):
    keys = [(int(r["seed"]), int(r["order"])) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate seed/order")
    return keys


def score_arrays(rows, packed, y):
    recs = sorted(rows, key=lambda r: (int(r["seed"]), int(r["order"])))
    matrix = np.unpackbits(packed[[int(r["mask_row"]) for r in recs]], axis=1, count=len(y)).astype(bool)
    fp = matrix[:, ~y].sum(axis=1)
    fn = (~matrix[:, y]).sum(axis=1)
    ba = 1 - 0.5 * (fp / (~y).sum() + fn / y.sum())
    if not np.array_equal(fp, [int(r["fp"]) for r in recs]) or not np.array_equal(fn, [int(r["fn"]) for r in recs]):
        raise ValueError("Saved masks disagree with FP or FN")
    if not np.allclose(ba, [float(r["ba"]) for r in recs], rtol=0, atol=2e-7):
        raise ValueError("Saved masks disagree with balanced accuracy")
    return recs, matrix, ba


def operating_row(meta, name, runs, matrix, ba):
    result = dict(**meta, method=name, n_runs=len(runs), n_seeds=len({r["seed"] for r in runs}),
                  ba=float(ba.mean()))
    for field in ("fp", "fn", "labels", "setup_bytes", "review_bytes", "total_bytes",
                  "queries", "install_frame"):
        vals = [float(r[field]) for r in runs if r.get(field) not in ("", None, "None")]
        if field == "install_frame":
            vals = [v for v in vals if v >= 0]
        result["mean_" + field] = float(np.mean(vals)) if vals else ""
    result["n_neg"] = int(runs[0]["n_neg"])
    result["n_pos"] = int(runs[0]["n_pos"])
    if result["n_neg"] + result["n_pos"] != matrix.shape[1]:
        raise ValueError("Published reference counts disagree with the masks")
    return result


def analyze_site(site, backbone, records, out):
    from purchase_cohorts import load
    data = load(site, backbone)
    y = np.asarray(data["test_y"], dtype=bool)
    with np.load(out / ("masks_" + site + "_" + backbone + ".npz"), allow_pickle=False) as f:
        for name in ("test_id", "test_group"):
            if not np.array_equal(data[name], f[name]):
                raise ValueError("Test identities differ in mask bundle and licensed input")
        masks = f["pred"]
    resamples = CoupledResamples(site, y, data["test_group"])
    groups = defaultdict(lambda: defaultdict(list))
    for r in records:
        if (r["site"], r["backbone"]) != (site, backbone):
            raise ValueError("Unexpected site/backbone")
        if int(r["cap_pct"]) == 25:
            groups[(int(r["label_cap"]), r["grant_mode"])][r["method"]].append(r)
    operating = []
    pairs = []
    needed = set(PAIRS).union(*map(set, PAIRS.values()))
    if set(groups) != {(limit, mode) for limit in (32, 128) for mode in GRANTS}:
        raise ValueError(f"Incomplete spending cells in {site}/{backbone}: {list(groups)}")
    for (label_cap, mode), methods in sorted(groups.items()):
        meta = dict(site=site, backbone=backbone, label_cap=label_cap, cap_pct=25, grant_mode=mode)
        if not needed.issubset(methods):
            raise ValueError(f"Missing methods in {meta}: {sorted(needed-set(methods))}")
        scores = {}
        for m in methods:
            runs, mat, ba = score_arrays(methods[m], masks, y)
            operating.append(operating_row(meta, m, runs, mat, ba))
            if m in needed:
                scores[m] = runs, mat, ba
        for treatment, controls in PAIRS.items():
            ra, ma, ba = scores[treatment]
            for control in controls:
                rb, mb, bb = scores[control]
                if unit_keys(ra) != unit_keys(rb):
                    raise ValueError("Different draws/orders between paired policies")
                delta = paired_differences(y, ma, mb)
                d_fp = delta[~y].sum()
                d_fn = -delta[y].sum()
                d_ba = 100 * (ba - bb).mean()
                d_fpr, d_fnr = 100 * d_fp / (~y).sum(), 100 * d_fn / y.sum()
                if not np.isclose(d_ba, -(d_fpr + d_fnr) / 2):
                    raise ValueError("Pairwise BA decomposition failed")
                equal_label_rows = sum(int(a["labels"]) == int(b["labels"]) for a, b in zip(ra, rb, strict=True))
                equal_cost_rows = sum(int(a["setup_bytes"]) == int(b["setup_bytes"]) for a, b in zip(ra, rb, strict=True))
                if (treatment == "adaptive" and control.startswith("matched_")) or (treatment == "adaptive_guard" and control.startswith("guard_matched_")):
                    if equal_label_rows != len(ra):
                        raise ValueError("Label-matched controls must purchase the same prefix in each run")
                limits = resamples.intervals(delta)
                seeds = np.array([int(r["seed"]) for r in ra])
                seed_diff = np.array([(ba[seeds == s] - bb[seeds == s]).mean() for s in np.unique(seeds)])
                row = dict(**meta, treatment=treatment, control=control, n_images=len(y), n_neg=(~y).sum(), n_pos=y.sum(),
                           n_runs=len(ra), n_seeds=len(seed_diff), equal_label_count_runs=equal_label_rows,
                           equal_setup_cost_runs=equal_cost_rows,
                           mean_labels_treatment=np.mean([int(r["labels"]) for r in ra]),
                           mean_labels_control=np.mean([int(r["labels"]) for r in rb]), higher_ba_seeds=int((seed_diff>1e-12).sum()),
                           equal_ba_seeds=int((np.abs(seed_diff)<=1e-12).sum()), ba_treatment=ba.mean(), ba_control=bb.mean(),
                           delta_ba_pp=d_ba, mean_fp_difference=d_fp, mean_fn_difference=d_fn,
                           delta_fpr_pp=d_fpr, delta_fnr_pp=d_fnr, resampling_unit="image_within_class" if resamples.stratified else "capture_date",
                           n_units=resamples.units, n_valid_bootstraps=int(resamples.valid.sum()), n_bootstraps=NBOOT)
                for field, (lo, hi) in limits.items():
                    row[field+"_lo"] = lo
                    row[field+"_hi"] = hi
                pairs.append(row)
    return operating, pairs


def plot_effects(pairs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"pdf.fonttype": 42, "axes.spines.right": False, "axes.spines.top": False, "font.size": 9.5})
    fig, axes = plt.subplots(3, 3, figsize=(10.2, 6.45), sharex=True, sharey="row")
    colors = {"prepaid": "#176978", "earned": "#a74d34"}
    offsets = {"prepaid": -.08, "earned": .08}
    symbols = {"prepaid": "o", "earned": "s"}
    labels = {"prepaid": "All bytes granted up front", "earned": "Bytes accrue per arrival"}
    channels = [("ba_pp", "Balanced accuracy\nΔ percentage points"), ("fnr_pp", "False-negative rate\nΔ percentage points"),
                ("fpr_pp", "False-positive rate\nΔ percentage points")]
    indexed = {(r["site"],r["backbone"],r["grant_mode"]):r for r in pairs if r["label_cap"]==32 and (r["treatment"],r["control"])==PRIMARY}
    for row, (metric, ylabel) in enumerate(channels):
        for col, site in enumerate(SITES):
            ax = axes[row, col]
            ax.axhline(0, color="#52646b", lw=.8, ls="--")
            ax.grid(axis="y", lw=.45, color="#d8dddf")
            for mode in GRANTS:
                x=[]; m=[]; lo=[]; hi=[]
                for bi,b in enumerate(BACKBONES):
                    r=indexed[site,b,mode]
                    x.append(bi+offsets[mode]);m.append(r["delta_"+metric])
                    lo.append(r[metric+"_lo"]);hi.append(r[metric+"_hi"])
                ax.vlines(x, lo, hi, colors=colors[mode], linewidth=1.3)
                ax.scatter(x, m, color=colors[mode], s=24, marker=symbols[mode], zorder=3, label=labels[mode])
            ax.set(xlim=(-.45,2.45), xticks=[0,1,2], xticklabels=["V2","S3 grid*","V4"])
            if row == 0: ax.set_title(site.title(), weight="bold", loc="left", fontsize=11)
            if col == 0: ax.set_ylabel(ylabel)
    fig.legend(*axes[0,0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.56,.985), ncol=2, frameon=False)
    fig.text(.095,.036,"Paired differences relative to the label-matched rank control. Bars are conditional 95% percentile intervals.",fontsize=9)
    fig.text(.095,.009,"*S3 grid uses released scales on the host. Latvia resamples nine capture dates; the other two sites resample images within class.",fontsize=8.3)
    fig.subplots_adjust(top=.89,bottom=.145,left=.105,right=.985,wspace=.16,hspace=.22)
    fig.savefig(out / "paired_effects.pdf",bbox_inches="tight",pad_inches=.09)
    plt.close(fig)


def plot_ledger(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(10.0,4.15))
    ax.set_xlim(0,10);ax.set_ylim(0,4.4);ax.set_axis_off()
    def box(x,y,w,h,text,fill="#eff5f5",edge="#336d72"):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.04,rounding_size=.08",
                                   edgecolor=edge,facecolor=fill,linewidth=1.1))
        ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=9.4,color="#1d3039")
    def arrow(a,b):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=13,color="#58696c",lw=1.1))
    box(.1,1.55,1.61,1.0,"Byte ledger\nUp-front or\nper-arrival credit")
    box(2.03,1.55,1.75,1.0,"Affordable pilot\nBuy 16 target\nlabels")
    box(4.1,1.55,1.76,1.0,"Paired audit\nCompare gains,\nrecall and prices")
    for y,t in ((3.13,"Keep source head"),(1.78,"Install pilot head"),(.43,"Buy more labels\nFit expanded head")):
        box(6.19,y,1.63,.73,t,fill="#f9f5ea",edge="#a48546")
        arrow((5.9,2.05),(6.15,y+.365));arrow((7.86,y+.365),(8.24,2.05))
    box(8.28,1.55,1.59,1.0,"Online review\nSpend balance\nframe by frame")
    arrow((1.75,2.05),(1.99,2.05));arrow((3.82,2.05),(4.06,2.05))
    ax.text(2.85,.64,"Optional source-alert guard retains\noriginal positive calls after review.",ha="center",va="center",fontsize=8.7,color="#42545a")
    ax.plot([.90,.90,6.00],[2.59,3.42,3.42],color="#76898c",lw=1,ls="--")
    arrow((6.00,3.42),(6.15,3.42))
    ax.text(3.47,3.55,"Pilot quote exceeds half the window budget",ha="center",fontsize=8.6,color="#42545a")
    ax.text(.12,4.15,"Paying for a classifier update",weight="bold",fontsize=14,color="#1d3039")
    ax.text(.12,.02,"Calibration photos, exchanged features, returned configuration and review transactions share the same byte balance.",fontsize=9)
    fig.savefig(out / "purchase_decision_flow.pdf",bbox_inches="tight",pad_inches=.1)
    plt.close(fig)


def main(out):
    out.mkdir(parents=True,exist_ok=True)
    combined=out/"runs.csv"
    by_cell=defaultdict(list)
    if combined.exists():
        for r in read_rows(combined): by_cell[r["site"],r["backbone"]].append(r)
    operating, pairs=[],[]
    for site in SITES:
        for backbone in BACKBONES:
            records=by_cell[(site,backbone)] if combined.exists() else read_rows(out/f"runs_{site}_{backbone}.csv")
            a,b=analyze_site(site,backbone,records,out)
            operating.extend(a);pairs.extend(b)
    write_rows(out/"operating_summary.csv",operating)
    write_rows(out/"paired_summary.csv",pairs)
    plot_effects(pairs,out);plot_ledger(out)
    print(f"Verified primary masks and wrote {len(operating)} method cells and {len(pairs)} paired comparisons.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ROOT/"results/purchase")
    main(parser.parse_args().out)

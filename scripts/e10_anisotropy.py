from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.attribution import corr_mean, isoscore, uniformity
from src.encode import MODELS, load_emb
from src.plotting import MODEL_COLORS, RES_DIR, figure, legend_outside, save

METRICS = {"isoscore": "IsoScore (higher = more isotropic)", "uniformity": "Uniformity loss (lower = more uniform)", "corr_mean": "Mean |inter-dim correlation|"}
TARGETS = {"rel50_last": "last", "rel50_random": "random"}


def main() -> None:
    summary = pd.read_csv(RES_DIR / "e1_retrieval_summary.csv")
    at50 = summary[(summary.frac - 0.5).abs() < 1e-9].set_index(["model", "method"]).rel_mean / 100
    rows = []
    for model in MODELS:
        x = load_emb(model, "simplewiki", "all")
        rows.append(
            {
                "model": model,
                "isoscore": isoscore(x),
                "uniformity": uniformity(x),
                "corr_mean": corr_mean(x),
                **{k: float(at50[(model, m)]) for k, m in TARGETS.items()},
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(RES_DIR / "e10_anisotropy.csv", index=False)
    corr = []
    for metric in METRICS:
        for target in TARGETS:
            p = pearsonr(df[metric], df[target])
            s = spearmanr(df[metric], df[target])
            corr.append({"metric": metric, "target": target, "n": len(df), "pearson": p.statistic, "pearson_p": p.pvalue, "spearman": s.statistic, "spearman_p": s.pvalue})
    cdf = pd.DataFrame(corr)
    cdf.to_csv(RES_DIR / "e10_correlations.csv", index=False)
    print(df.round(4).to_string())
    print(cdf.round(3).to_string())
    fig, axes = figure(3, 1, sharey=True)
    for ax, (metric, name) in zip(axes, METRICS.items()):
        for _, r in df.iterrows():
            ax.scatter(r[metric], r.rel50_last * 100, s=140, color=MODEL_COLORS[r.model], label=MODELS[r.model].label)
            ax.scatter(r[metric], r.rel50_random * 100, s=140, facecolors="none", edgecolors=MODEL_COLORS[r.model], linewidths=2.5)
        rl = cdf[(cdf.metric == metric) & (cdf.target == "rel50_last")].pearson.iloc[0]
        rr = cdf[(cdf.metric == metric) & (cdf.target == "rel50_random")].pearson.iloc[0]
        ax.set_xlabel(f"{name}\nPearson r: last {rl:.2f}, random {rr:.2f}", fontsize=16)
    axes[0].set_ylabel("Relative nDCG@10 at 50% removed (%)")
    handles, names = axes[0].get_legend_handles_labels()
    handles.append(axes[0].scatter([], [], s=140, color="black"))
    handles.append(axes[0].scatter([], [], s=140, facecolors="none", edgecolors="black", linewidths=2.5))
    names += ["Filled: last-K", "Hollow: random-K"]
    legend_outside(axes[2], handles=handles, labels=names)
    save(fig, "fig_anisotropy_vs_drop")


if __name__ == "__main__":
    main()

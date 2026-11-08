from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from joblib import Parallel, delayed

from src.encode import CORE, MODELS, load_emb
from src.evaluate import classification_suite, retrieval_suite
from src.metrics import device, n_jobs
from src.paper import PAPER_OF, TABLE2
from src.plotting import MODEL_COLORS, RES_DIR, figure, figure_legend, save
from src.truncate import pca_apply, pca_fit


def pca_for(model: str) -> tuple[np.ndarray, np.ndarray]:
    return pca_fit(load_emb(model, "allnli", "all"), MODELS[model].dim // 2)


def retrieval_pca(model: str) -> float:
    mean, comps = pca_for(model)
    m, c = torch.from_numpy(mean).to(device()), torch.from_numpy(comps).to(device())
    rel = []
    for ev in retrieval_suite(model).values():
        full = ev.ndcg().mean()
        rel.append(ev.ndcg(q=(ev.q - m) @ c, d=(ev.d - m) @ c).mean() / full)
    return 100 * float(np.mean(rel))


def classification_pca(model: str) -> float:
    mean, comps = pca_for(model)
    rel = [
        ev.accuracy(transform=lambda x: pca_apply(x, mean, comps)).mean() / ev.accuracy().mean()
        for ev in classification_suite(model).values()
    ]
    return 100 * float(np.mean(rel))


def pick(df: pd.DataFrame, model: str) -> pd.Series:
    return df[(df.model == model) & (df.method == "random") & (df.frac == 0.5)].iloc[0]


def main() -> None:
    models = list(MODELS)
    r = pd.read_csv(RES_DIR / "e1_retrieval_summary.csv")
    c = pd.read_csv(RES_DIR / "e1_classification_summary.csv")
    cls = Parallel(n_jobs=min(n_jobs(), len(models)), prefer="threads")(delayed(classification_pca)(m) for m in models)
    rows = []
    for model, cls_pca in zip(models, cls):
        paper = TABLE2.get(PAPER_OF.get(model, ""), {})
        rows.append(
            {
                "model": model,
                "beir_trunc": pick(r, model).rel_mean,
                "beir_trunc_std": pick(r, model).rel_std,
                "beir_pca": retrieval_pca(model),
                "mteb_trunc": pick(c, model).rel_mean,
                "mteb_trunc_std": pick(c, model).rel_std,
                "mteb_pca": cls_pca,
                **{f"paper_{k}": v for k, v in paper.items()},
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(RES_DIR / "e3_pca.csv", index=False)
    plot_df = df[df.model.isin(CORE)].reset_index(drop=True)
    plot_df.to_csv(RES_DIR / "fig_pca_vs_random.csv", index=False)
    fig, axes = figure(ncols=2, sharey=True)
    x = np.arange(len(plot_df))
    for ax, bench, label in zip(axes, ["beir", "mteb"], ["NanoBEIR nDCG@10", "MTEB accuracy"]):
        for off, kind, hatch in [(-0.2, "trunc", ""), (0.2, "pca", "//")]:
            ax.bar(x + off, plot_df[f"{bench}_{kind}"], 0.38, color=[MODEL_COLORS[m] for m in plot_df.model], hatch=hatch, edgecolor="black")
            ax.scatter(x + off, plot_df[f"paper_{bench}_{kind}"], marker="*", s=300, color="black", zorder=5)
        ax.set_xticks(x, [MODELS[m].label.replace(" ", "\n", 1) for m in plot_df.model])
        ax.set_xlabel(label)
        ax.set_ylim(60, 105)
    axes[0].set_ylabel("Relative performance at 50% (%)")
    handles = [
        Patch(facecolor="white", edgecolor="black", label="Random 50%"),
        Patch(facecolor="white", edgecolor="black", hatch="//", label="PCA 50%"),
        Line2D([], [], marker="*", color="black", ls="", markersize=16, label="Paper (Table 2)"),
    ]
    save(fig, "fig_pca_vs_random", rect=figure_legend(fig, handles, [h.get_label() for h in handles], right=0.78))
    print(df.round(1).to_string())


if __name__ == "__main__":
    main()

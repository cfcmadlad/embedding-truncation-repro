from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from src.attribution import mean_query_embedding, nanobeir_full, nanobeir_scores, outlier_dims
from src.encode import CORE, MODELS
from src.paper import PAPER_OF, TABLE1
from src.plotting import MODEL_COLORS, RES_DIR, figure, legend_outside, save
from src.truncate import SEEDS, keep_without


def main() -> None:
    rows = []
    for model in CORE:
        dim = MODELS[model].dim
        out = outlier_dims(mean_query_embedding(model))
        full = nanobeir_full(model).mean()
        without = nanobeir_scores(model, keep_without(dim, out)).mean()
        pool = np.setdiff1d(np.arange(dim), out)
        rand = np.array(
            [nanobeir_scores(model, keep_without(dim, np.random.default_rng(s).choice(pool, len(out), replace=False))).mean() for s in SEEDS]
        )
        p = TABLE1[PAPER_OF[model]]
        rows.append(
            {
                "model": model,
                "n_outliers": len(out),
                "outlier_dims": " ".join(map(str, out)),
                "ndcg_full": full,
                "ndcg_without_outliers": without,
                "ndcg_random_mean": rand.mean(),
                "ndcg_random_std": rand.std(),
                "delta_outliers": without - full,
                "delta_random": rand.mean() - full,
                "paper_model": PAPER_OF[model],
                "paper_n_outliers": p["n_outliers"],
                "paper_without_outliers": p["without_outliers"],
                "paper_random": p["random"],
                "paper_random_std": p["random_std"],
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(RES_DIR / "e6_outliers.csv", index=False)
    print(df.drop(columns="outlier_dims").to_string())
    fig, ax = figure()
    x = np.arange(len(df))
    w = 0.38
    ax.bar(x - w / 2, df.delta_outliers * 1e3, w, color=[MODEL_COLORS[m] for m in df.model], label="Outlier dims removed")
    ax.bar(
        x + w / 2,
        df.delta_random * 1e3,
        w,
        yerr=df.ndcg_random_std * 1e3,
        capsize=6,
        color=[MODEL_COLORS[m] for m in df.model],
        alpha=0.4,
        hatch="//",
        label="Same number of random non-outlier dims removed (10 seeds)",
    )
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x, [f"{MODELS[m].label}\n({n} outliers)" for m, n in zip(df.model, df.n_outliers)])
    ax.set_ylabel("Change in nDCG@10 (x 0.001)")
    handles = [
        Patch(facecolor="#555555", label="Outlier dims removed"),
        Patch(facecolor="#bbbbbb", hatch="//", label="Same number of random\nnon-outlier dims removed\n(mean and std, 10 seeds)"),
    ]
    legend_outside(ax, handles=handles)
    save(fig, "fig_outliers")


if __name__ == "__main__":
    main()

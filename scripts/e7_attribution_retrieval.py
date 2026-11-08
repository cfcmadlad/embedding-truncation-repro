from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.attribution import labels, per_dataset_deltas
from src.data import NANOBEIR
from src.encode import CORE, MODELS
from src.plotting import DEGRADING, IMPROVING, RES_DIR, figure, legend_outside, save


def main() -> None:
    rows, per_ds = [], []
    for model in CORE:
        full, deltas = per_dataset_deltas(model)
        dim = deltas.shape[1]
        ndcg_full = float(full.mean())
        delta = deltas.mean(0)
        rows.append(
            pd.DataFrame(
                {
                    "model": model,
                    "dim": np.arange(dim),
                    "ndcg_full": ndcg_full,
                    "ndcg_without": ndcg_full + delta,
                    "delta": delta,
                    "label": labels(delta),
                }
            )
        )
        per_ds.append(
            pd.DataFrame(
                {
                    "model": model,
                    "dataset": np.repeat(NANOBEIR, dim),
                    "dim": np.tile(np.arange(dim), len(NANOBEIR)),
                    "delta": deltas.ravel(),
                }
            )
        )
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(RES_DIR / "e7_attribution_retrieval.csv", index=False)
    pd.concat(per_ds, ignore_index=True).to_csv(RES_DIR / "e7_attribution_retrieval_per_dataset.csv", index=False)
    counts = df.groupby(["model", "label"]).size().unstack(fill_value=0).reindex(columns=["degrading", "improving", "tie"], fill_value=0)
    counts.to_csv(RES_DIR / "e7_attribution_retrieval_counts.csv")
    print(counts)
    for model in CORE:
        sub = df[df.model == model]
        fig, ax = figure()
        for lab, color in (("degrading", DEGRADING), ("improving", IMPROVING), ("tie", "#2ca02c")):
            s = sub[sub.label == lab]
            if len(s):
                ax.scatter(s.dim, s.ndcg_without, s=14, color=color, label=f"{lab.capitalize()}: {len(s)}")
        ax.axhline(sub.ndcg_full.iloc[0], color="#d62728", lw=2.5, label="Full embedding")
        ax.set_xlabel(f"Removed dimension index (of {MODELS[model].dim})")
        ax.set_ylabel("Mean NanoBEIR nDCG@10")
        legend_outside(ax, markerscale=2.5)
        save(fig, f"fig_attribution_{model}")


if __name__ == "__main__":
    main()

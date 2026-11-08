from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.attribution import TIE_TOL
from src.data import NANOBEIR
from src.encode import CORE, MODELS
from src.plotting import MODEL_COLORS, RES_DIR, figure, legend_outside, save


def main() -> None:
    per = pd.read_csv(RES_DIR / "e7_attribution_retrieval_per_dataset.csv")
    n_ds = len(NANOBEIR)
    rows = []
    for model in CORE:
        sub = per[per.model == model]
        counts = (sub.delta > TIE_TOL).groupby(sub.dim).sum().to_numpy()
        hist = np.bincount(counts, minlength=n_ds + 1)
        for k, h in enumerate(hist):
            rows.append({"model": model, "n_datasets": k, "n_dims": int(h), "pct_dims": 100 * h / MODELS[model].dim})
    df = pd.DataFrame(rows)
    df.to_csv(RES_DIR / "e9_shared_degrading.csv", index=False)
    print(df.pivot(index="n_datasets", columns="model", values="pct_dims").round(1).to_string())
    fig, ax = figure()
    w = 0.8 / len(CORE)
    for i, model in enumerate(CORE):
        s = df[df.model == model]
        ax.bar(s.n_datasets + (i - (len(CORE) - 1) / 2) * w, s.pct_dims, w, color=MODEL_COLORS[model], label=f"{MODELS[model].label} (D={MODELS[model].dim})")
    ax.set_xticks(range(n_ds + 1))
    ax.set_xlabel("Number of NanoBEIR datasets on which the dimension is degrading")
    ax.set_ylabel("Share of dimensions (%)")
    legend_outside(ax)
    save(fig, "fig_shared_degrading")


if __name__ == "__main__":
    main()

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.attribution import nanobeir_full, nanobeir_relative
from src.encode import CORE, MODELS
from src.plotting import DEGRADING, IMPROVING, LAST_K, RES_DIR, figure, legend_outside, save
from src.truncate import keep_last_k, keep_without

STEP = 0.05
MAX_LAST = 0.9


def main() -> None:
    attr = pd.read_csv(RES_DIR / "e7_attribution_retrieval.csv")
    rows = []
    for model in CORE:
        dim = MODELS[model].dim
        full = nanobeir_full(model)
        sub = attr[attr.model == model]
        orders = {
            "degrading": sub[sub.label == "degrading"].sort_values("delta", ascending=False).dim.to_numpy(),
            "improving": sub[sub.label == "improving"].sort_values("delta").dim.to_numpy(),
        }
        for method in ("degrading", "improving", "last"):
            rows.append({"model": model, "method": method, "frac": 0.0, "n_removed": 0, "rel": 1.0})
            for j in range(1, int(round(MAX_LAST / STEP)) + 1):
                n = int(round(j * STEP * dim))
                if method == "last":
                    keep = keep_last_k(dim, j * STEP)
                elif n <= len(orders[method]):
                    keep = keep_without(dim, orders[method][:n])
                else:
                    break
                rows.append({"model": model, "method": method, "frac": round(j * STEP, 2), "n_removed": n, "rel": nanobeir_relative(model, keep, full)})
    df = pd.DataFrame(rows)
    df.to_csv(RES_DIR / "e8_degrading_removal.csv", index=False)
    print(df.groupby(["model", "method"]).rel.agg(["max", "min", "count"]))
    fig, axes = figure(2, 2, sharex=True, sharey=True)
    for ax, model in zip(axes.ravel(), CORE):
        sub = df[df.model == model]
        for method, color, name in (("degrading", DEGRADING, "Degrading"), ("improving", IMPROVING, "Improving"), ("last", LAST_K, "Last-K")):
            s = sub[sub.method == method]
            ax.plot(s.frac * 100, s.rel * 100, "o-", color=color, label=name, ms=5)
        ax.text(0.03, 0.06, MODELS[model].label, transform=ax.transAxes, fontsize=16)
    for ax in axes[1]:
        ax.set_xlabel("Dimensions removed (%)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Relative nDCG@10 (%)")
    axes[0, 0].set_ylim(np.floor(df.rel.min() * 100) - 2, np.ceil(df.rel.max() * 100) + 2)
    legend_outside(axes[0, 1], bbox_to_anchor=(1.02, 0.0))
    save(fig, "fig_degrading_removal")


if __name__ == "__main__":
    main()

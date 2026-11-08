from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.encode import MODELS, MRL_PAIR
from src.plotting import MODEL_COLORS, RES_DIR, figure, figure_legend, save


def main() -> None:
    r = pd.read_csv(RES_DIR / "e1_retrieval_summary.csv").assign(task="retrieval")
    c = pd.read_csv(RES_DIR / "e1_classification_summary.csv").assign(task="classification")
    df = pd.concat([r, c], ignore_index=True)
    df = df[df.model.isin(MRL_PAIR)][["task", "model", "method", "frac", "rel_mean", "rel_std"]]
    df.to_csv(RES_DIR / "e4_mrl.csv", index=False)
    df.to_csv(RES_DIR / "fig_mrl.csv", index=False)
    fig, axes = figure(ncols=2, sharey=True)
    for ax, task in zip(axes, ["retrieval", "classification"]):
        for model in MRL_PAIR:
            for method, ls in [("last", "-"), ("random", "--")]:
                s = df[(df.task == task) & (df.model == model) & (df.method == method)].sort_values("frac")
                ax.errorbar(100 * s.frac, s.rel_mean, yerr=s.rel_std, color=MODEL_COLORS[model], ls=ls, marker="o", capsize=3,
                            label=f"{MODELS[model].label}, {'last' if method == 'last' else 'random'}-K")
        ax.set_xlabel(f"K% removed (%), {'NanoBEIR' if task == 'retrieval' else 'MTEB'}")
        ax.set_xticks(range(0, 100, 20))
    axes[0].set_ylabel("Relative performance (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    save(fig, "fig_mrl", rect=figure_legend(fig, handles, labels, right=0.68))
    print(df[df.frac.isin([0.5, 0.8, 0.9])].round(2).to_string())


if __name__ == "__main__":
    main()

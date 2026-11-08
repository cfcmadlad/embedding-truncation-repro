from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.encode import CORE, MODELS
from src.evaluate import retrieval_suite
from src.metrics import spearman_rows
from src.paper import FIG8_SPEARMAN, PAPER_OF
from src.plotting import MODEL_COLORS, RES_DIR, figure, legend_outside, save
from src.truncate import FRACTIONS, keep_last_k


def main() -> None:
    rows = []
    for model in MODELS:
        for name, ev in retrieval_suite(model).items():
            full = ev.scores()
            for frac in FRACTIONS[1:]:
                rho = spearman_rows(full, ev.scores(keep_last_k(MODELS[model].dim, frac)))
                rows.append({"model": model, "dataset": name, "frac": frac, "spearman": float(rho.mean())})
    df = pd.DataFrame(rows)
    df.to_csv(RES_DIR / "e2_spearman.csv", index=False)
    summ = df.groupby(["model", "frac"])["spearman"].agg(["mean", "std"]).reset_index()
    summ.columns = ["model", "frac", "spearman_mean", "spearman_std_over_datasets"]
    summ["paper"] = [FIG8_SPEARMAN.get(PAPER_OF.get(m, ""), {}).get(int(round(100 * f)), float("nan")) for m, f in zip(summ.model, summ.frac)]
    summ.to_csv(RES_DIR / "e2_spearman_summary.csv", index=False)
    plot_df = summ[summ.model.isin(CORE)]
    plot_df.to_csv(RES_DIR / "fig_spearman.csv", index=False)
    fig, ax = figure()
    for model in CORE:
        s = plot_df[plot_df.model == model]
        ax.plot(100 * s.frac, s.spearman_mean, color=MODEL_COLORS[model], marker="o", label=MODELS[model].label)
        p = s.dropna(subset=["paper"])
        ax.scatter(100 * p.frac, p.paper, color=MODEL_COLORS[model], marker="*", s=300, edgecolor="black", zorder=5)
    ax.scatter([], [], color="white", marker="*", s=300, edgecolor="black", label="Paper (Fig 8)")
    ax.set_xlabel("Last K% removed (%)")
    ax.set_ylabel("Mean Spearman, full vs truncated ranking")
    ax.set_xticks(range(10, 100, 10))
    legend_outside(ax)
    save(fig, "fig_spearman")
    print(summ[summ.frac.isin([0.5, 0.9])].round(3).to_string())


if __name__ == "__main__":
    main()

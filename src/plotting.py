from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import ROOT

FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"

MODEL_COLORS = {
    "minilm": "#8c564b",
    "mpnet": "#d62728",
    "contriever": "#ff7f0e",
    "e5": "#1f77b4",
    "mrl": "#2ca02c",
    "nonmrl": "#9467bd",
}
DEGRADING = "#1f77b4"
IMPROVING = "#ff7f0e"
LAST_K = "#7f7f7f"
PAPER = "#000000"

plt.rcParams.update(
    {
        "font.size": 16,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 2.5,
        "lines.markersize": 8,
    }
)


def figure(ncols: int = 1, nrows: int = 1, **kw):
    return plt.subplots(nrows, ncols, figsize=(12, 6.75), **kw)


def legend_outside(ax_or_fig, **kw) -> None:
    kw.setdefault("loc", "center left")
    kw.setdefault("bbox_to_anchor", (1.01, 0.5))
    kw.setdefault("frameon", False)
    ax_or_fig.legend(**kw)


def save(fig, name: str, rect: tuple[float, float, float, float] = (0, 0, 1, 1)) -> Path:
    FIG_DIR.mkdir(exist_ok=True)
    fig.tight_layout(rect=rect)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=200)
    fig.savefig(FIG_DIR / f"{name}.pdf")
    plt.close(fig)
    return FIG_DIR / f"{name}.png"

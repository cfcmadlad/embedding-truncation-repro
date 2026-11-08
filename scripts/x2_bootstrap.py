from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D

from src.encode import CORE, MODELS
from src.evaluate import retrieval_suite
from src.metrics import n_jobs
from src.plotting import DEGRADING, IMPROVING, LAST_K, RES_DIR, figure, save
from src.theory import loo_deltas, stratified_bootstrap_weights

N_BOOT = 1000
TOL = 1e-9


def run_model(model: str) -> tuple[dict, pd.DataFrame]:
    suite = retrieval_suite(model)
    pairs = [loo_deltas(ev) for ev in suite.values()]
    deltas = [p[0] for p in pairs]
    full = [p[1] for p in pairs]
    groups = np.concatenate([np.full(len(f), i) for i, f in enumerate(full)])
    pooled = np.concatenate(deltas, axis=1).astype(np.float64)
    point = np.mean([d.mean(1) for d in deltas], axis=0)
    w = stratified_bootstrap_weights(groups, N_BOOT, 0)
    boot = w @ pooled.T
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    label = np.where(point > TOL, "degrading", np.where(point < -TOL, "improving", "tie"))
    full_cat = np.concatenate(full)
    se_ds = np.array([f.std(ddof=1) / np.sqrt(len(f)) for f in full])
    se_mean = float(np.sqrt((se_ds**2).sum()) / len(full))
    full_boot = w @ full_cat
    order = np.argsort(point, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    dims = pd.DataFrame(
        {
            "model": model,
            "dim": np.arange(len(point)),
            "rank": rank,
            "delta": point,
            "ci_lo": lo,
            "ci_hi": hi,
            "boot_se": boot.std(0, ddof=1),
            "label": label,
            "significant": (lo > 0) | (hi < 0),
        }
    )
    summary = {
        "model": model,
        "dims": len(point),
        "full_ndcg": float(np.mean([f.mean() for f in full])),
        "n_degrading": int((label == "degrading").sum()),
        "n_improving": int((label == "improving").sum()),
        "n_tie": int((label == "tie").sum()),
        "sig_degrading": int((lo > 0).sum()),
        "sig_improving": int((hi < 0).sum()),
        "expected_by_chance_per_side": 0.025 * len(point),
        "median_abs_delta": float(np.median(np.abs(point))),
        "max_abs_delta": float(np.abs(point).max()),
        "median_boot_se_delta": float(np.median(dims.boot_se)),
        "se_nanobeir_mean": se_mean,
        "se_nanobeir_mean_bootstrap": float(full_boot.std(ddof=1)),
        "se_dataset_median": float(np.median(se_ds)),
    }
    return summary, dims


def plot(dims: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig, axes = figure(2, 2, sharex=False, sharey=False)
    for ax, model in zip(axes.flat, CORE):
        d = dims[dims.model == model].sort_values("rank")
        se = summary.set_index("model").loc[model, "se_nanobeir_mean"]
        ax.fill_between(d["rank"], d.ci_lo, d.ci_hi, color=LAST_K, alpha=0.35, lw=0)
        colors = np.where(d.label == "degrading", DEGRADING, np.where(d.label == "improving", IMPROVING, LAST_K))
        ax.scatter(d["rank"], d.delta, c=colors, s=6)
        sig = d[d.significant]
        ax.scatter(sig["rank"], sig.delta, facecolors="none", edgecolors="black", s=40, lw=1)
        ax.axhline(0, color="black", lw=1)
        ax.axhline(se, color="black", ls="--", lw=1.5)
        ax.axhline(-se, color="black", ls="--", lw=1.5)
        ax.text(0.03, 0.85, MODELS[model].label, transform=ax.transAxes, fontsize=16)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    fig.supxlabel("Dimension rank by delta", fontsize=18)
    fig.supylabel("nDCG@10 change", fontsize=18)
    handles = [
        Line2D([], [], color=DEGRADING, marker="o", ls="", label="Degrading dim"),
        Line2D([], [], color=IMPROVING, marker="o", ls="", label="Improving dim"),
        Line2D([], [], color=LAST_K, lw=8, alpha=0.35, label="95% bootstrap CI"),
        Line2D([], [], color="black", marker="o", mfc="none", ls="", label="CI excludes 0"),
        Line2D([], [], color="black", ls="--", label="SE of NanoBEIR nDCG@10"),
    ]
    axes[0, 1].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    save(fig, "fig_bootstrap")


def main() -> None:
    torch.set_num_threads(n_jobs())
    rows, frames = [], []
    for model in CORE:
        s, d = run_model(model)
        rows.append(s)
        frames.append(d)
    summary = pd.DataFrame(rows)
    dims = pd.concat(frames, ignore_index=True)
    summary.to_csv(RES_DIR / "x2_bootstrap.csv", index=False)
    dims.to_csv(RES_DIR / "x2_bootstrap_dims.csv", index=False)
    plot(dims, summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

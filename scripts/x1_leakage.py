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
from src.plotting import DEGRADING, LAST_K, RES_DIR, figure, save
from src.theory import loo_deltas
from src.truncate import keep_last_k, keep_without

SPLITS = range(5)
STEP = 0.05
TOL = 1e-9


def split_indices(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    perm = np.random.default_rng(seed).permutation(n)
    return np.sort(perm[: n // 2]), np.sort(perm[n // 2 :])


def run_model(model: str) -> tuple[list[dict], list[dict]]:
    suite = retrieval_suite(model)
    dim = MODELS[model].dim
    pairs = {k: loo_deltas(ev) for k, ev in suite.items()}
    deltas = {k: v[0] for k, v in pairs.items()}
    full = {k: v[1] for k, v in pairs.items()}
    rows, corr = [], []
    for seed in SPLITS:
        splits = {k: split_indices(len(full[k]), seed + 1000 * i) for i, k in enumerate(suite)}
        da = {k: deltas[k][:, splits[k][0]].mean(1) for k in suite}
        db = {k: deltas[k][:, splits[k][1]].mean(1) for k in suite}
        avg_a = np.mean(list(da.values()), axis=0)
        avg_b = np.mean(list(db.values()), axis=0)
        corr.append({"model": model, "split": seed, "level": "average", "pearson": float(np.corrcoef(avg_a, avg_b)[0, 1])})
        for k in suite:
            ok = np.std(da[k]) > 0 and np.std(db[k]) > 0
            corr.append({"model": model, "split": seed, "level": k, "pearson": float(np.corrcoef(da[k], db[k])[0, 1]) if ok else np.nan})
        degrading = np.flatnonzero(avg_a > TOL)
        order = degrading[np.argsort(-avg_a[degrading], kind="stable")]
        base_a = {k: full[k][splits[k][0]].mean() for k in suite}
        base_b = {k: full[k][splits[k][1]].mean() for k in suite}
        fracs = [0.0] + [round(STEP * i, 2) for i in range(1, 20) if int(round(STEP * i * dim)) <= len(order)]
        for frac in fracs:
            n_rem = int(round(frac * dim))
            keep = keep_without(dim, order[:n_rem])
            keep_last = keep_last_k(dim, frac)
            rel_a, rel_b, rel_last = [], [], []
            for k, ev in suite.items():
                per_q = ev.ndcg(keep)
                last_q = ev.ndcg(keep_last)
                a, b = splits[k]
                rel_a.append(per_q[a].mean() / base_a[k])
                rel_b.append(per_q[b].mean() / base_b[k])
                rel_last.append(last_q[b].mean() / base_b[k])
            rows.append(
                {
                    "model": model,
                    "split": seed,
                    "frac": frac,
                    "n_removed": n_rem,
                    "n_degrading_a": len(order),
                    "rel_in_sample": 100 * float(np.mean(rel_a)),
                    "rel_held_out": 100 * float(np.mean(rel_b)),
                    "rel_last_k_held_out": 100 * float(np.mean(rel_last)),
                }
            )
    return rows, corr


def plot(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["model", "frac"])
        .agg(
            n_splits=("split", "count"),
            in_sample=("rel_in_sample", "mean"),
            in_sample_std=("rel_in_sample", "std"),
            held_out=("rel_held_out", "mean"),
            held_out_std=("rel_held_out", "std"),
            last_k=("rel_last_k_held_out", "mean"),
        )
        .reset_index()
    )
    agg = agg[agg.n_splits == len(SPLITS)]
    fig, axes = figure(2, 2, sharex=True, sharey=True)
    for ax, model in zip(axes.flat, CORE):
        a = agg[agg.model == model]
        x = 100 * a.frac
        ax.axhline(100, color="black", lw=1)
        ax.plot(x, a.last_k, color=LAST_K, marker="o")
        ax.errorbar(x, a.in_sample, yerr=a.in_sample_std, color=DEGRADING, marker="o", capsize=3)
        ax.errorbar(x, a.held_out, yerr=a.held_out_std, color=DEGRADING, marker="s", ls="--", mfc="white", capsize=3)
        ax.text(0.03, 0.06, MODELS[model].label, transform=ax.transAxes, fontsize=16)
    fig.supxlabel("Removed dimensions (%)", fontsize=18)
    fig.supylabel("Relative nDCG@10 (%)", fontsize=18)
    handles = [
        Line2D([], [], color=DEGRADING, marker="o", label="Degrading, same queries"),
        Line2D([], [], color=DEGRADING, marker="s", ls="--", mfc="white", label="Degrading, held-out queries"),
        Line2D([], [], color=LAST_K, marker="o", label="Last-K, held-out queries"),
    ]
    axes[0, 1].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    save(fig, "fig_leakage")
    return agg


def main() -> None:
    torch.set_num_threads(n_jobs())
    rows, corr = [], []
    for model in CORE:
        r, c = run_model(model)
        rows += r
        corr += c
    df = pd.DataFrame(rows)
    df.to_csv(RES_DIR / "x1_leakage.csv", index=False)
    cdf = pd.DataFrame(corr)
    cdf.to_csv(RES_DIR / "x1_delta_correlation.csv", index=False)
    agg = plot(df)
    agg.to_csv(RES_DIR / "x1_leakage_plot.csv", index=False)
    summary = []
    for model in CORE:
        a = agg[agg.model == model]
        peak = a.loc[a.in_sample.idxmax()]
        c = cdf[cdf.model == model]
        avg = c[c.level == "average"].pearson
        per = c[c.level != "average"].groupby("split").pearson.mean()
        summary.append(
            {
                "model": model,
                "peak_frac": peak.frac,
                "peak_in_sample": peak.in_sample,
                "held_out_at_peak": peak.held_out,
                "max_held_out": a.held_out.max(),
                "corr_average_mean": avg.mean(),
                "corr_average_std": avg.std(),
                "corr_per_dataset_mean": per.mean(),
                "corr_per_dataset_std": per.std(),
            }
        )
    pd.DataFrame(summary).to_csv(RES_DIR / "x1_summary.csv", index=False)
    print(pd.DataFrame(summary).round(3).to_string(index=False))


if __name__ == "__main__":
    main()

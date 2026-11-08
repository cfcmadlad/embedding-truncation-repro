from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.data import CLASSIFICATION
from src.encode import CORE, MODELS
from src.evaluate import classification_suite, retrieval_suite
from src.metrics import n_jobs
from src.paper import FIG1_RANDOM_AVG
from src.plotting import MODEL_COLORS, PAPER, RES_DIR, figure, figure_legend, save
from src.truncate import FRACTIONS, SEEDS, keep_last_k, keep_random


def settings(dim: int):
    for frac in FRACTIONS:
        yield "last", frac, -1, keep_last_k(dim, frac)
        for seed in SEEDS:
            yield "random", frac, seed, keep_random(dim, frac, seed)


def run_retrieval(models: list[str]) -> pd.DataFrame:
    rows = []
    for model in models:
        dim = MODELS[model].dim
        for name, ev in retrieval_suite(model).items():
            for method, frac, seed, keep in settings(dim):
                rows.append({"model": model, "dataset": name, "method": method, "frac": frac, "seed": seed, "ndcg": float(ev.ndcg(keep).mean())})
    return pd.DataFrame(rows)


def _cls_job(model: str, task: str, method: str, frac: float, seed: int, keep: np.ndarray) -> dict:
    acc = classification_suite(model)[task].accuracy(keep).mean()
    return {"model": model, "task": task, "method": method, "frac": frac, "seed": seed, "accuracy": float(acc)}


def run_classification(models: list[str]) -> pd.DataFrame:
    tasks = list(CLASSIFICATION)
    jobs = [
        delayed(_cls_job)(model, task, method, frac, seed, keep)
        for model in models
        for task in tasks
        for method, frac, seed, keep in settings(MODELS[model].dim)
        if not (method == "random" and frac == 0.0 and seed > 0)
    ]
    df = pd.DataFrame(Parallel(n_jobs=n_jobs(), prefer="threads")(jobs))
    zero = df[(df.method == "random") & (df.frac == 0.0)]
    return pd.concat([df] + [zero.assign(seed=s) for s in SEEDS[1:]], ignore_index=True)


def merge(new: pd.DataFrame, path: Path) -> pd.DataFrame:
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[~old.model.isin(new.model.unique())], new], ignore_index=True)
    new = new.sort_values(["model", new.columns[1], "method", "frac", "seed"]).reset_index(drop=True)
    new.to_csv(path, index=False)
    return new


def summarise(df: pd.DataFrame, key: str, metric: str, out_name: str) -> pd.DataFrame:
    full = df[(df.method == "last") & (df.frac == 0.0)].set_index(["model", key])[metric]
    df = df.join(full.rename("full"), on=["model", key])
    df["rel"] = 100 * df[metric] / df["full"]
    per_seed = df.groupby(["model", "method", "frac", "seed"]).agg(rel=("rel", "mean"), score=(metric, "mean")).reset_index()
    out = per_seed.groupby(["model", "method", "frac"]).agg(
        rel_mean=("rel", "mean"), rel_std=("rel", "std"), **{out_name: ("score", "mean")}
    )
    return out.fillna(0.0).reset_index()


def plot(summary: pd.DataFrame, task: str, models: list[str], name: str) -> None:
    fig, axes = figure(ncols=2, sharey=True)
    xs = [100 * f for f in FRACTIONS]
    for ax, method in zip(axes, ["last", "random"]):
        for model in models:
            s = summary[(summary.model == model) & (summary.method == method)].sort_values("frac")
            ax.errorbar(100 * s.frac, s.rel_mean, yerr=s.rel_std, color=MODEL_COLORS[model], marker="o", capsize=3, label=MODELS[model].label)
        if method == "random":
            ax.plot(xs, FIG1_RANDOM_AVG[task], color=PAPER, ls="--", marker="x", label="Paper avg, Fig 1")
        ax.set_xlabel(f"{'Last' if method == 'last' else 'Random'} K% removed (%)")
        ax.set_xticks(range(0, 100, 10))
    axes[0].set_ylabel(f"Relative {'nDCG@10' if task == 'retrieval' else 'accuracy'} (%)")
    handles, labels = axes[1].get_legend_handles_labels()
    save(fig, name, rect=figure_legend(fig, handles, labels))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["retrieval", "classification", "both"], default="both")
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    args = ap.parse_args()
    models = args.models
    if args.part in ("retrieval", "both"):
        df = merge(run_retrieval(models), RES_DIR / "e1_retrieval.csv")
        summ = summarise(df, "dataset", "ndcg", "ndcg_mean")
        summ.to_csv(RES_DIR / "e1_retrieval_summary.csv", index=False)
        summ[summ.model.isin(CORE)].to_csv(RES_DIR / "fig_trunc_retrieval.csv", index=False)
        if set(CORE) <= set(summ.model):
            plot(summ, "retrieval", CORE, "fig_trunc_retrieval")
    if args.part in ("classification", "both"):
        df = merge(run_classification(models), RES_DIR / "e1_classification.csv")
        summ = summarise(df, "task", "accuracy", "acc_mean")
        summ.to_csv(RES_DIR / "e1_classification_summary.csv", index=False)
        summ[summ.model.isin(CORE)].to_csv(RES_DIR / "fig_trunc_classification.csv", index=False)
        if set(CORE) <= set(summ.model):
            plot(summ, "classification", CORE, "fig_trunc_classification")


if __name__ == "__main__":
    main()

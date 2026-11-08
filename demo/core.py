from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TRUNBED_DEVICE", "cpu")
os.environ.setdefault("TRUNBED_ENCODE_DEVICE", "cpu")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch

from src.data import CLASSIFICATION, NANOBEIR, load_retrieval
from src.data import CACHE
from src.encode import MODELS, encode_texts, load_emb, load_model
from src.evaluate import ClassificationEval, RetrievalEval
from src.metrics import spearman_rows
from src.truncate import keep_last_k, keep_random, keep_without, n_removed, pca_apply, pca_fit

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
METHODS = ["last-K", "random-K", "PCA", "remove degrading", "remove improving"]


def read_csv(name: str) -> pd.DataFrame | None:
    path = RESULTS / name
    return pd.read_csv(path) if path.exists() else None


@lru_cache(maxsize=None)
def retrieval(model: str, dataset: str) -> RetrievalEval:
    return RetrievalEval(model, dataset)


@lru_cache(maxsize=None)
def classification(model: str, task: str) -> ClassificationEval:
    return ClassificationEval(model, task)


@lru_cache(maxsize=None)
def texts(dataset: str):
    return load_retrieval(dataset)


@lru_cache(maxsize=None)
def pca_full(model: str) -> tuple[np.ndarray, np.ndarray]:
    path = CACHE / "pca" / f"{model}.npz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        mean, comps = pca_fit(load_emb(model, "allnli", "all"), MODELS[model].dim)
        np.savez(path, mean=mean, comps=comps)
    z = np.load(path)
    return z["mean"], z["comps"]


def pca(model: str, m: int) -> tuple[np.ndarray, np.ndarray]:
    mean, comps = pca_full(model)
    return mean, comps[:, :m]


def warm_encoder(model: str) -> None:
    load_model(model)


@lru_cache(maxsize=None)
def attribution_order(model: str, label: str) -> np.ndarray | None:
    df = read_csv("e7_attribution_retrieval.csv")
    if df is None:
        return None
    df = df[(df.model == model) & (df.label == label)]
    return df.reindex(df.delta.abs().sort_values(ascending=False).index).dim.to_numpy()


def selection(model: str, method: str, frac: float, seed: int):
    dim = MODELS[model].dim
    removed = n_removed(dim, frac)
    if method == "last-K":
        return keep_last_k(dim, frac), None, removed
    if method == "random-K":
        return keep_random(dim, frac, seed), None, removed
    if method == "PCA":
        mean, comps = pca(model, dim - removed)
        return None, (mean, comps), removed
    order = attribution_order(model, "degrading" if method == "remove degrading" else "improving")
    if order is None:
        raise FileNotFoundError("results/e7_attribution_retrieval.csv")
    drop = order[:removed]
    return keep_without(dim, drop), None, len(drop)


def retrieval_scores(model: str, dataset: str, method: str, frac: float, seed: int):
    ev = retrieval(model, dataset)
    full = ev.scores()
    keep, proj, removed = selection(model, method, frac, seed)
    if proj is None:
        trunc = ev.scores(keep)
    else:
        q = torch.from_numpy(pca_apply(ev.q.numpy(), *proj))
        d = torch.from_numpy(pca_apply(ev.d.numpy(), *proj))
        trunc = ev.scores(None, q, d)
    return ev, full, trunc, removed


def retrieval_result(model: str, dataset: str, method: str, frac: float, seed: int) -> dict:
    ev, full, trunc, removed = retrieval_scores(model, dataset, method, frac, seed)
    nf = float(ev.ndcg_scores(full).mean())
    nt = float(ev.ndcg_scores(trunc).mean())
    return {
        "metric": "nDCG@10",
        "full": nf,
        "truncated": nt,
        "relative": 100 * nt / nf,
        "spearman": float(spearman_rows(full, trunc).mean()),
        "removed": removed,
    }


def classification_result(model: str, task: str, method: str, frac: float, seed: int) -> dict:
    df = read_csv("e1_classification.csv")
    pre = None
    if df is not None and method in ("last-K", "random-K"):
        key = "last" if method == "last-K" else "random"
        sub = df[(df.model == model) & (df.task == task)]
        full = sub[(sub.frac == 0) & (sub.method == "last")].accuracy
        hit = sub[(sub.method == key) & np.isclose(sub.frac, frac) & ((sub.seed == seed) | (key == "last"))].accuracy
        if len(full) and len(hit):
            pre = (float(full.iloc[0]), float(hit.iloc[0]))
    dim = MODELS[model].dim
    removed = n_removed(dim, frac)
    if pre is None:
        ev = classification(model, task)
        keep, proj, removed = selection(model, method, frac, seed)
        nf = float(ev.accuracy().mean())
        nt = float(ev.accuracy(keep, (lambda x: pca_apply(x, *proj)) if proj else None).mean())
        pre = (nf, nt)
    return {"metric": "accuracy", "full": pre[0], "truncated": pre[1], "relative": 100 * pre[1] / pre[0], "spearman": None, "removed": removed}


def evaluate(model: str, dataset: str, method: str, frac: float, seed: int) -> dict:
    if dataset in CLASSIFICATION:
        return classification_result(model, dataset, method, frac, seed)
    return retrieval_result(model, dataset, method, frac, seed)


def top_docs(model: str, dataset: str, method: str, frac: float, seed: int, query: str | None, qi: int | None, k: int = 5):
    ev, full, trunc, _ = retrieval_scores(model, dataset, method, frac, seed)
    if query is not None:
        spec = MODELS[model]
        qv = torch.from_numpy(encode_texts(model, [query], spec.query_prefix))
        keep, proj, _ = selection(model, method, frac, seed)
        f = ev.scores(None, qv, ev.d)[0]
        if proj is None:
            t = ev.scores(keep, qv, ev.d)[0]
        else:
            t = ev.scores(None, torch.from_numpy(pca_apply(qv.numpy(), *proj)), torch.from_numpy(pca_apply(ev.d.numpy(), *proj)))[0]
        rel = torch.zeros_like(f)
    else:
        f, t = full[qi].clone(), trunc[qi].clone()
        f[ev.exclude[qi]] = float("-inf")
        t[ev.exclude[qi]] = float("-inf")
        rel = ev.rel[qi]
    rank_full = torch.argsort(torch.argsort(f, descending=True))
    rank_trunc = torch.argsort(torch.argsort(t, descending=True))
    ds = texts(dataset)

    def rows(scores, other_rank):
        idx = torch.topk(scores, k).indices.tolist()
        return [
            {"doc": ds.docs[j][:300], "score": float(scores[j]), "relevant": bool(rel[j] > 0), "other_rank": int(other_rank[j]) + 1}
            for j in idx
        ]

    return rows(f, rank_trunc), rows(t, rank_full)


DATASETS = NANOBEIR + list(CLASSIFICATION)

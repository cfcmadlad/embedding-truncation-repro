from __future__ import annotations

import csv
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.data import Retrieval, load_classification, load_retrieval
from src.encode import MODELS, encode_texts
from src.evaluate import RetrievalEval
from src.metrics import logreg_accuracy, mteb_subsamples
from src.paper import FIG1_RANDOM_AVG, PAPER_OF, TABLE2
from src.truncate import FRACTIONS, keep_last_k, keep_random

RETRIEVAL_FIELDS = ["model", "dataset", "docs", "removed_pct", "full_ndcg10", "last_k", "random_k_mean", "random_k_std", "random_k_relative_pct", "paper_fig1_avg_pct"]
CLASSIFICATION_FIELDS = ["model", "task", "train_texts", "test_texts", "experiments", "removed_pct", "full_acc", "last_k", "random_k_mean", "random_k_std", "random_k_relative_pct", "paper_fig1_avg_pct"]
CLASSIFICATION_FRACS = [0.0, 0.3, 0.5, 0.7, 0.9]
RETRIEVAL_BAR = 90.0
CLASSIFICATION_BAR = 95.0


def subset(ds: Retrieval, max_docs: int | None) -> Retrieval:
    if not max_docs or max_docs >= len(ds.docs):
        return ds
    judged = np.flatnonzero(ds.qrels.sum(0) > 0)
    rest = np.setdiff1d(np.arange(len(ds.docs)), judged)[: max(0, max_docs - len(judged))]
    keep = np.sort(np.concatenate([judged, rest]))
    return Retrieval(ds.name, ds.query_ids, ds.queries, [ds.doc_ids[i] for i in keep], [ds.docs[i] for i in keep], ds.qrels[:, keep])


def retrieval_sweep(ev: RetrievalEval, dim: int, fracs: list[float] = FRACTIONS, seeds: list[int] = list(range(10))) -> list[dict]:
    full = ev.ndcg_mean()
    rows = []
    for frac in fracs:
        last = ev.ndcg_mean(keep_last_k(dim, frac))
        rand = np.array([ev.ndcg_mean(keep_random(dim, frac, s)) for s in seeds])
        rows.append(
            {
                "removed_pct": int(round(frac * 100)),
                "full": full,
                "last_k": last,
                "random_mean": float(rand.mean()),
                "random_std": float(rand.std()),
                "random_rel": 100 * float(rand.mean()) / full,
                "seed0": float(rand[0]),
            }
        )
    return rows


def remove_vs_zero(ev: RetrievalEval, dim: int, keep: np.ndarray) -> tuple[float, float, float]:
    mask = torch.zeros(dim, dtype=ev.q.dtype, device=ev.q.device)
    mask[torch.as_tensor(keep, device=ev.q.device)] = 1
    removed = ev.scores(keep)
    zeroed = ev.scores(None, ev.q * mask, ev.d * mask)
    gap = float((removed - zeroed).abs().max())
    return float(ev.ndcg_scores(removed).mean()), float(ev.ndcg_scores(zeroed).mean()), gap


@dataclass
class ClassificationPlan:
    task: str
    subsamples: list[np.ndarray]
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_texts: list[str]
    test_texts: list[str]
    y_train: np.ndarray
    y_test: np.ndarray
    local: list[np.ndarray]


def classification_plan(task: str, experiments: int, test_n: int) -> ClassificationPlan:
    ds = load_classification(task)
    subsamples = mteb_subsamples(ds.train_labels, ds.samples_per_label, experiments)
    train_idx = np.unique(np.concatenate(subsamples))
    remap = np.full(len(ds.train_labels), -1)
    remap[train_idx] = np.arange(len(train_idx))
    n_test = len(ds.test_labels)
    test_idx = np.sort(np.random.default_rng(0).choice(n_test, min(test_n, n_test), replace=False))
    return ClassificationPlan(
        task,
        subsamples,
        train_idx,
        test_idx,
        [ds.train_texts[i] for i in train_idx],
        [ds.test_texts[i] for i in test_idx],
        ds.train_labels[train_idx],
        ds.test_labels[test_idx],
        [remap[s] for s in subsamples],
    )


def classification_sweep(plan: ClassificationPlan, x_train: np.ndarray, x_test: np.ndarray, dim: int, seeds: list[int], fracs: list[float] = CLASSIFICATION_FRACS, workers: int = 1) -> list[dict]:
    def acc(keep: np.ndarray) -> float:
        return float(logreg_accuracy(x_train[:, keep], plan.y_train, x_test[:, keep], plan.y_test, plan.local).mean())

    jobs = {(frac, kind): keep_last_k(dim, frac) if kind == "last" else keep_random(dim, frac, kind) for frac in fracs for kind in (["last"] + seeds if frac else ["last"])}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        done = dict(zip(jobs, pool.map(acc, jobs.values())))
    full = done[(fracs[0], "last")]
    rows = []
    for frac in fracs:
        rand = np.array([done[(frac, s)] for s in seeds]) if frac else np.array([full])
        rows.append(
            {
                "removed_pct": int(round(frac * 100)),
                "full": full,
                "last_k": done[(frac, "last")],
                "random_mean": float(rand.mean()),
                "random_std": float(rand.std()),
                "random_rel": 100 * float(rand.mean()) / full,
            }
        )
    return rows


def encode_timed(model: str, texts: list[str], prefix: str) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    out = encode_texts(model, texts, prefix)
    return out, time.perf_counter() - start


def paper_avg(task: str, removed_pct: int) -> float:
    return FIG1_RANDOM_AVG[task][removed_pct // 10]


def paper_row(model: str, task: str) -> float | None:
    name = PAPER_OF.get(model)
    return TABLE2[name]["beir_trunc" if task == "retrieval" else "mteb_trunc"] if name in TABLE2 else None


def write_csv(path: Path, fields: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(fields)
        w.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def render(rows: list[dict], task: str) -> list[str]:
    out = [f"{'removed':>8}{'last-K':>10}{'random-K':>11}{'std':>8}{'kept (%)':>10}{'paper avg (%)':>15}"]
    for r in rows:
        out.append(f"{r['removed_pct']:>7}%{r['last_k']:>10.4f}{r['random_mean']:>11.4f}{r['random_std']:>8.4f}{r['random_rel']:>10.1f}{paper_avg(task, r['removed_pct']):>15.1f}")
    return out


def verdict(retrieval_rel: float, classification_rel: float | None, model: str) -> list[str]:
    ref = paper_row(model, "retrieval")
    cref = paper_row(model, "classification")
    lines = [f"retrieval       random 50% keeps {retrieval_rel:.1f}% of nDCG@10 (paper bar: over {RETRIEVAL_BAR:.0f}%" + (f", paper {PAPER_OF[model]}: {ref}%)" if ref else ")")]
    ok = retrieval_rel > RETRIEVAL_BAR
    if classification_rel is not None:
        lines.append(f"classification  random 50% keeps {classification_rel:.1f}% of accuracy (paper bar: over {CLASSIFICATION_BAR:.0f}%" + (f", paper {PAPER_OF[model]}: {cref}%)" if cref else ")"))
        ok = ok and classification_rel > CLASSIFICATION_BAR
    lines.append("VERDICT  " + ("the paper's claim holds in this live run." if ok else "at least one bar was missed in this live run; see the numbers above."))
    return lines


def replay_lines(retrieval_csv: Path, classification_csv: Path | None = None, reason: str = "live encoding failed") -> list[str]:
    rows = read_csv(retrieval_csv)
    lines = [
        f"STORED RESULT, NOT RECOMPUTED ({reason}). This is the csv written by an earlier live run:",
        f"file       {retrieval_csv.name}",
        f"model      {rows[0]['model']}, dataset {rows[0]['dataset']}, {rows[0]['docs']} documents, full nDCG@10 {float(rows[0]['full_ndcg10']):.4f}",
    ]
    parsed = [
        {
            "removed_pct": int(r["removed_pct"]),
            "last_k": float(r["last_k"]),
            "random_mean": float(r["random_k_mean"]),
            "random_std": float(r["random_k_std"]),
            "random_rel": float(r["random_k_relative_pct"]),
        }
        for r in rows
    ]
    lines += render(parsed, "retrieval")
    cls_rel = None
    if classification_csv is not None and classification_csv.exists():
        crows = read_csv(classification_csv)
        lines.append(f"stored classification run: {crows[0]['task']}, {crows[0]['train_texts']} training texts")
        cparsed = [
            {
                "removed_pct": int(r["removed_pct"]),
                "last_k": float(r["last_k"]),
                "random_mean": float(r["random_k_mean"]),
                "random_std": float(r["random_k_std"]),
                "random_rel": float(r["random_k_relative_pct"]),
            }
            for r in crows
        ]
        lines += render(cparsed, "classification")
        cls_rel = next(r["random_rel"] for r in cparsed if r["removed_pct"] == 50)
    rel = next(r["random_rel"] for r in parsed if r["removed_pct"] == 50)
    lines += verdict(rel, cls_rel, rows[0]["model"])
    lines.append("(stored numbers, not a live measurement)")
    return lines

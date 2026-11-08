from __future__ import annotations

import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression


def device() -> torch.device:
    want = os.environ.get("TRUNBED_DEVICE", "cpu")
    return torch.device(want if want == "cpu" or torch.cuda.is_available() else "cpu")


def n_jobs() -> int:
    return int(os.environ.get("TRUNBED_JOBS", "8"))


def similarity(q: torch.Tensor, d: torch.Tensor, sim: str) -> torch.Tensor:
    if sim == "cos":
        q = torch.nn.functional.normalize(q, dim=-1)
        d = torch.nn.functional.normalize(d, dim=-1)
    return q @ d.transpose(-1, -2)


def discounts(k: int, dev: torch.device) -> torch.Tensor:
    return 1.0 / torch.log2(torch.arange(2, k + 2, device=dev, dtype=torch.float32))


def idcg(rel: torch.Tensor, k: int = 10) -> torch.Tensor:
    top = torch.topk(rel, min(k, rel.shape[-1]), dim=-1).values
    return (top * discounts(top.shape[-1], rel.device)).sum(-1)


def ndcg_at_k(
    scores: torch.Tensor,
    rel: torch.Tensor,
    k: int = 10,
    exclude: torch.Tensor | None = None,
    ideal: torch.Tensor | None = None,
) -> torch.Tensor:
    if exclude is not None:
        scores = scores.masked_fill(exclude, float("-inf"))
    idx = torch.topk(scores, k, dim=-1).indices
    gains = torch.gather(rel.expand(*scores.shape[:-2], *rel.shape), -1, idx)
    dcg = (gains * discounts(k, scores.device)).sum(-1)
    ideal = idcg(rel, k) if ideal is None else ideal
    return dcg / ideal.clamp_min(1e-12)


def ranks(x: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(x, dim=-1)
    r = torch.empty_like(order, dtype=torch.float32)
    r.scatter_(-1, order, torch.arange(x.shape[-1], device=x.device, dtype=torch.float32).expand_as(order).contiguous())
    return r


def spearman_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ra = ranks(a)
    rb = ranks(b)
    ra = ra - ra.mean(-1, keepdim=True)
    rb = rb - rb.mean(-1, keepdim=True)
    return (ra * rb).sum(-1) / (ra.norm(dim=-1) * rb.norm(dim=-1)).clamp_min(1e-12)


def mteb_subsamples(labels: np.ndarray, per_label: int, n_experiments: int = 10, seed: int = 42) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    idxs = np.arange(len(labels))
    out = []
    for _ in range(n_experiments):
        rng.shuffle(idxs)
        counts: dict[int, int] = {}
        keep = []
        for i in idxs:
            c = counts.get(labels[i], 0)
            if c < per_label:
                keep.append(i)
                counts[labels[i]] = c + 1
        out.append(np.asarray(keep))
    return out


def logreg_accuracy(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    subsamples: list[np.ndarray],
) -> np.ndarray:
    accs = []
    for idx in subsamples:
        clf = LogisticRegression(max_iter=100, random_state=42)
        clf.fit(x_train[idx], y_train[idx])
        accs.append(float((clf.predict(x_test) == y_test).mean()))
    return np.asarray(accs)

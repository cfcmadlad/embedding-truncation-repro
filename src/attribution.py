from __future__ import annotations

import numpy as np
import torch

from src.data import CACHE, NANOBEIR
from src.encode import load_emb
from src.evaluate import RetrievalEval, retrieval_suite
from src.metrics import idcg, ndcg_at_k

ATTR_DIR = CACHE / "attr"
TIE_TOL = 1e-9


def loo_ndcg_per_query(ev: RetrievalEval, query_idx: np.ndarray | None = None, chunk: int = 32) -> np.ndarray:
    sel = slice(None) if query_idx is None else torch.as_tensor(query_idx, device=ev.q.device)
    q, rel, exclude = ev.q[sel], ev.rel[sel], ev.exclude[sel]
    d = ev.d
    ideal = idcg(rel)
    dots = q @ d.T
    qn2 = (q * q).sum(-1)
    dn2 = (d * d).sum(-1)
    out = np.empty((q.shape[1], q.shape[0]), dtype=np.float32)
    for start in range(0, q.shape[1], chunk):
        c = torch.arange(start, min(start + chunk, q.shape[1]), device=q.device)
        qc = q[:, c].T
        dc = d[:, c].T
        num = dots.unsqueeze(0) - qc.unsqueeze(-1) * dc.unsqueeze(1)
        if ev.sim == "cos":
            den = torch.sqrt((qn2.unsqueeze(0) - qc * qc).clamp_min(1e-12)).unsqueeze(-1) * torch.sqrt(
                (dn2.unsqueeze(0) - dc * dc).clamp_min(1e-12)
            ).unsqueeze(1)
            num = num / den
        out[start : start + len(c)] = ndcg_at_k(num, rel, 10, exclude, ideal).cpu().numpy()
    return out


def loo_cached(model: str, name: str) -> np.ndarray:
    path = ATTR_DIR / model / f"{name}.npy"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, loo_ndcg_per_query(retrieval_suite(model)[name]))
    return np.load(path)


def nanobeir_full(model: str) -> np.ndarray:
    return np.array([retrieval_suite(model)[n].ndcg().mean() for n in NANOBEIR])


def nanobeir_scores(model: str, keep: np.ndarray | None = None) -> np.ndarray:
    return np.array([retrieval_suite(model)[n].ndcg(keep).mean() for n in NANOBEIR])


def nanobeir_relative(model: str, keep: np.ndarray | None, full: np.ndarray) -> float:
    return float(np.mean(nanobeir_scores(model, keep) / full))


def per_dataset_deltas(model: str) -> tuple[np.ndarray, np.ndarray]:
    full = nanobeir_full(model)
    without = np.stack([loo_cached(model, n).mean(1) for n in NANOBEIR])
    return full, without - full[:, None]


def labels(delta: np.ndarray) -> np.ndarray:
    return np.where(np.abs(delta) < TIE_TOL, "tie", np.where(delta > 0, "degrading", "improving"))


def mean_query_embedding(model: str) -> np.ndarray:
    return np.concatenate([load_emb(model, n, "queries") for n in NANOBEIR]).mean(0)


def outlier_dims(v: np.ndarray, z: float = 3.0) -> np.ndarray:
    return np.flatnonzero(np.abs(v - v.mean()) > z * v.std())


def isoscore(x: np.ndarray) -> float:
    x = x.astype(np.float64) - x.mean(0)
    _, s, _ = np.linalg.svd(x, full_matrices=False)
    var = np.zeros(x.shape[1])
    var[: len(s)] = s**2 / (x.shape[0] - 1)
    n = x.shape[1]
    sig = np.sqrt(n) * var / np.linalg.norm(var)
    defect = np.linalg.norm(sig - 1) / np.sqrt(2 * (n - np.sqrt(n)))
    phi = (n - defect**2 * (n - np.sqrt(n))) ** 2 / n**2
    return float((n * phi - 1) / (n - 1))


def uniformity(x: np.ndarray, t: float = 2.0, chunk: int = 1000) -> float:
    z = torch.nn.functional.normalize(torch.from_numpy(x).double(), dim=-1)
    n = z.shape[0]
    total = torch.zeros((), dtype=torch.float64)
    for s in range(0, n, chunk):
        sq = (2 - 2 * z[s : s + chunk] @ z.T).clamp_min(0)
        e = torch.exp(-t * sq)
        idx = torch.arange(s, min(s + chunk, n))
        e[idx - s, idx] = 0
        total += e.sum()
    return float(torch.log(total / (n * (n - 1))))


def corr_mean(x: np.ndarray) -> float:
    c = np.corrcoef(x.astype(np.float64), rowvar=False)
    off = ~np.eye(c.shape[0], dtype=bool)
    return float(np.nanmean(np.abs(c[off])))

from __future__ import annotations

import math

import numpy as np
import torch

from src.attribution import loo_cached
from src.evaluate import RetrievalEval


def loo_deltas(ev: RetrievalEval, noise: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    full = ev.ndcg()
    d = loo_cached(ev.model, ev.name) - full[None, :]
    d[np.abs(d) < noise] = 0.0
    return d, full


def stratified_bootstrap_weights(groups: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = np.unique(groups)
    w = np.zeros((n_boot, len(groups)), dtype=np.float64)
    for g in labels:
        idx = np.flatnonzero(groups == g)
        counts = rng.multinomial(len(idx), np.full(len(idx), 1.0 / len(idx)), size=n_boot)
        w[:, idx] = counts / (len(idx) * len(labels))
    return w


def pair_moments(q: torch.Tensor, docs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    s = torch.einsum("qn,qsn->qs", q, docs)
    w = q * q
    u = torch.einsum("qsn,qn->qs", docs * docs, w)
    g = torch.bmm(docs * w.unsqueeze(1), docs.transpose(1, 2))
    delta = s.unsqueeze(2) - s.unsqueeze(1)
    s2 = u.unsqueeze(2) + u.unsqueeze(1) - 2 * g
    return delta, s2


def swap_probability(delta: torch.Tensor, s2: torch.Tensor, n: int, m: int) -> torch.Tensor:
    var_r = (s2 / n - (delta / n) ** 2).clamp_min(0)
    sd = torch.sqrt(n * n * var_r / m * (n - m) / (n - 1))
    z = delta.abs() / sd.clamp_min(1e-30)
    return 0.5 * torch.erfc(z / math.sqrt(2))


def upper_mean(x: torch.Tensor) -> torch.Tensor:
    s = x.shape[-1]
    iu = torch.triu_indices(s, s, 1)
    return x[..., iu[0], iu[1]].mean(-1)


def predicted_tau(delta: torch.Tensor, s2: torch.Tensor, n: int, m: int) -> torch.Tensor:
    return 1 - 2 * upper_mean(swap_probability(delta, s2, n, m))


def observed_tau(full: torch.Tensor, trunc: torch.Tensor) -> torch.Tensor:
    a = torch.sign(full.unsqueeze(2) - full.unsqueeze(1))
    b = torch.sign(trunc.unsqueeze(2) - trunc.unsqueeze(1))
    return upper_mean(a * b)


def copula_spearman(tau: np.ndarray | float) -> np.ndarray:
    return 6 / np.pi * np.arcsin(np.sin(np.pi * np.asarray(tau) / 2) / 2)


def contribution_energy(q: torch.Tensor, docs: torch.Tensor) -> torch.Tensor:
    return (q * q * 2 * docs.var(dim=1, unbiased=True)).mean(0)


def participation_ratio(c: np.ndarray) -> float:
    c = np.asarray(c, dtype=np.float64)
    return float(c.sum() ** 2 / (len(c) * (c * c).sum()))

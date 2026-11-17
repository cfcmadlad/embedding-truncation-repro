from __future__ import annotations

import numpy as np

from src.data import Retrieval


def synthetic(n_q: int = 12, n_d: int = 80, dim: int = 384, seed: int = 0):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n_q, dim)).astype(np.float32)
    d = rng.normal(size=(n_d, dim)).astype(np.float32)
    for i in range(n_q):
        d[i * 3] = q[i] + 0.5 * rng.normal(size=dim).astype(np.float32)
    qrels = np.zeros((n_q, n_d), dtype=np.float32)
    for i in range(n_q):
        qrels[i, i * 3] = 1
    ds = Retrieval("synthetic", [f"q{i}" for i in range(n_q)], [f"query {i}" for i in range(n_q)], [f"d{j}" for j in range(n_d)], [f"doc {j}" for j in range(n_d)], qrels)
    return q, d, ds

from __future__ import annotations

import numpy as np

FRACTIONS = [round(0.1 * i, 1) for i in range(10)]
SEEDS = list(range(10))


def n_removed(dim: int, frac: float) -> int:
    return int(round(frac * dim))


def keep_last_k(dim: int, frac: float) -> np.ndarray:
    return np.arange(dim - n_removed(dim, frac))


def keep_random(dim: int, frac: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(dim, dim - n_removed(dim, frac), replace=False))


def keep_without(dim: int, removed: np.ndarray) -> np.ndarray:
    mask = np.ones(dim, dtype=bool)
    mask[np.asarray(removed, dtype=int)] = False
    return np.flatnonzero(mask)


def pca_fit(x: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(0)
    _, _, vt = np.linalg.svd(x - mean, full_matrices=False)
    return mean.astype(np.float32), vt[:m].T.astype(np.float32)


def pca_apply(x: np.ndarray, mean: np.ndarray, comps: np.ndarray) -> np.ndarray:
    return (x - mean) @ comps

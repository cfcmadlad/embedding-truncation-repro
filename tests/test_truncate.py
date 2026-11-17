from __future__ import annotations

import numpy as np

from src.truncate import FRACTIONS, SEEDS, keep_last_k, keep_random, keep_without, n_removed, pca_apply, pca_fit


def test_sizes():
    for dim in (384, 768):
        for frac in FRACTIONS:
            assert len(keep_last_k(dim, frac)) == dim - n_removed(dim, frac)
            assert len(keep_random(dim, frac, 0)) == dim - n_removed(dim, frac)


def test_last_k_is_a_prefix():
    assert keep_last_k(10, 0.3).tolist() == list(range(7))


def test_random_k_deterministic_sorted_unique_and_seed_dependent():
    a = keep_random(768, 0.5, 3)
    assert (a == keep_random(768, 0.5, 3)).all()
    assert (np.diff(a) > 0).all()
    assert a.min() >= 0 and a.max() < 768
    assert not (a == keep_random(768, 0.5, 4)).all()


def test_ten_seeds_give_ten_different_subsets():
    subsets = {tuple(keep_random(384, 0.5, s)) for s in SEEDS}
    assert len(SEEDS) == 10 and len(subsets) == 10


def test_keep_without():
    assert keep_without(6, np.array([1, 4])).tolist() == [0, 2, 3, 5]


def test_pca_shapes_and_orthonormal_components():
    x = np.random.default_rng(0).normal(size=(500, 20)).astype(np.float32)
    mean, comps = pca_fit(x, 10)
    assert pca_apply(x, mean, comps).shape == (500, 10)
    assert np.allclose(comps.T @ comps, np.eye(10), atol=1e-4)

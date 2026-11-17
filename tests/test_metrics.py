from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.stats import spearmanr

from src.metrics import idcg, mteb_subsamples, ndcg_at_k, spearman_rows


def test_ndcg_hand_computed():
    scores = torch.tensor([[0.9, 0.8, 0.7, 0.1]])
    rel = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
    got = float(ndcg_at_k(scores, rel, 10)[0])
    dcg = 1 / np.log2(3) + 1 / np.log2(5)
    ideal = 1 + 1 / np.log2(3)
    assert got == pytest.approx(dcg / ideal, rel=1e-6)


def test_ndcg_graded_gains_and_perfect_ranking():
    rel = torch.tensor([[3.0, 0.0, 1.0, 2.0, 0.0]])
    scores = torch.tensor([[5.0, 0.0, 3.0, 4.0, 1.0]])
    assert float(ndcg_at_k(scores, rel, 10)[0]) == pytest.approx(1.0, rel=1e-6)
    assert float(idcg(rel)[0]) == pytest.approx(3 + 2 / np.log2(3) + 1 / np.log2(4), rel=1e-6)


def test_ndcg_excludes_identical_id():
    scores = torch.tensor([[9.0, 1.0, 0.5]])
    rel = torch.tensor([[0.0, 1.0, 0.0]])
    exclude = torch.tensor([[True, False, False]])
    assert float(ndcg_at_k(scores, rel, 10, exclude)[0]) == pytest.approx(1.0)
    assert float(ndcg_at_k(scores, rel, 10)[0]) < 1.0


def test_ndcg_matches_pytrec_eval():
    pytrec_eval = pytest.importorskip("pytrec_eval")
    rng = np.random.default_rng(1)
    n_q, n_d = 20, 200
    scores = rng.normal(size=(n_q, n_d)).astype(np.float32)
    rel = (rng.random((n_q, n_d)) < 0.03).astype(np.float32) * rng.integers(1, 3, size=(n_q, n_d))
    rel[:, 0] = 1
    ours = ndcg_at_k(torch.from_numpy(scores), torch.from_numpy(rel), 10).numpy()
    run = {f"q{i}": {f"d{j}": float(scores[i, j]) for j in range(n_d)} for i in range(n_q)}
    qrels = {f"q{i}": {f"d{j}": int(rel[i, j]) for j in np.flatnonzero(rel[i])} for i in range(n_q)}
    ref = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10"}).evaluate(run)
    theirs = np.array([ref[f"q{i}"]["ndcg_cut_10"] for i in range(n_q)])
    assert np.abs(ours - theirs).max() < 1e-5


def test_spearman_matches_scipy():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(5, 300))
    b = a + rng.normal(size=(5, 300))
    got = spearman_rows(torch.from_numpy(a), torch.from_numpy(b)).numpy()
    ref = np.array([spearmanr(a[i], b[i]).statistic for i in range(5)])
    assert np.abs(got - ref).max() < 1e-5


def test_mteb_subsamples_balanced_and_deterministic():
    labels = np.repeat(np.arange(5), 40)
    a = mteb_subsamples(labels, 8, 3)
    b = mteb_subsamples(labels, 8, 3)
    assert all((x == y).all() for x, y in zip(a, b))
    for idx in a:
        assert np.bincount(labels[idx]).tolist() == [8] * 5
    assert not (a[0] == a[1]).all()

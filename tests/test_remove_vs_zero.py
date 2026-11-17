from __future__ import annotations

import numpy as np
import pytest
import torch

from src import live
from src.evaluate import RetrievalEval
from src.metrics import similarity
from src.truncate import keep_random
from tests.helpers import synthetic


@pytest.mark.parametrize("sim", ["cos", "dot"])
def test_scores_identical(sim):
    rng = np.random.default_rng(0)
    q = torch.from_numpy(rng.normal(size=(7, 64)))
    d = torch.from_numpy(rng.normal(size=(30, 64)))
    keep = keep_random(64, 0.5, 1)
    mask = torch.zeros(64, dtype=q.dtype)
    mask[keep] = 1
    removed = similarity(q[:, keep], d[:, keep], sim)
    zeroed = similarity(q * mask, d * mask, sim)
    assert (removed - zeroed).abs().max() < 1e-12


@pytest.mark.parametrize("model", ["minilm", "contriever"])
def test_live_helper_reports_zero_gap(model):
    q, d, ds = synthetic(dim=384 if model == "minilm" else 768)
    ev = RetrievalEval(model, "synthetic", q.astype(np.float64), d.astype(np.float64), ds)
    dim = q.shape[1]
    removed, zeroed, gap = live.remove_vs_zero(ev, dim, keep_random(dim, 0.5, 0))
    assert removed == pytest.approx(zeroed, abs=1e-9)
    assert gap < 1e-9


def test_sweep_shape_and_baseline():
    q, d, ds = synthetic()
    ev = RetrievalEval("minilm", "synthetic", q, d, ds)
    rows = live.retrieval_sweep(ev, 384, [0.0, 0.5], [0, 1, 2])
    assert rows[0]["random_rel"] == pytest.approx(100.0)
    assert rows[0]["full"] == pytest.approx(ev.ndcg_mean())
    assert rows[1]["random_std"] >= 0

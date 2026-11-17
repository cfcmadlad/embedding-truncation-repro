from __future__ import annotations

import numpy as np

from src import live
from tests.helpers import synthetic


def test_subset_keeps_every_judged_document_and_aligns_qrels():
    _, _, ds = synthetic()
    judged = set(np.flatnonzero(ds.qrels.sum(0) > 0).tolist())
    small = live.subset(ds, 30)
    assert len(small.docs) == 30
    kept = {ds.doc_ids.index(i) for i in small.doc_ids}
    assert judged <= kept
    assert small.qrels.shape == (len(ds.queries), 30)
    assert small.qrels.sum() == ds.qrels.sum()
    for j, doc_id in enumerate(small.doc_ids):
        assert (small.qrels[:, j] == ds.qrels[:, ds.doc_ids.index(doc_id)]).all()


def test_subset_never_drops_judged_even_below_their_count():
    _, _, ds = synthetic()
    small = live.subset(ds, 5)
    assert small.qrels.sum() == ds.qrels.sum()


def test_subset_noop_when_large_or_unset():
    _, _, ds = synthetic()
    assert live.subset(ds, None) is ds
    assert live.subset(ds, 10_000) is ds


def test_subset_is_deterministic():
    _, _, ds = synthetic()
    assert live.subset(ds, 40).doc_ids == live.subset(ds, 40).doc_ids

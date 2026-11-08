from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

from src.data import CLASSIFICATION, NANOBEIR, load_classification, load_retrieval
from src.encode import MODELS, load_emb
from src.metrics import device, idcg, logreg_accuracy, mteb_subsamples, ndcg_at_k, similarity


class RetrievalEval:
    def __init__(self, model: str, name: str):
        dev = device()
        ds = load_retrieval(name)
        self.model = model
        self.name = name
        self.sim = MODELS[model].sim
        self.q = torch.from_numpy(load_emb(model, name, "queries")).to(dev)
        self.d = torch.from_numpy(load_emb(model, name, "docs")).to(dev)
        self.rel = torch.from_numpy(ds.qrels).to(dev)
        self.exclude = torch.from_numpy(ds.same_id_mask).to(dev)
        self.ideal = idcg(self.rel)

    def scores(self, keep: np.ndarray | None = None, q: torch.Tensor | None = None, d: torch.Tensor | None = None) -> torch.Tensor:
        q = self.q if q is None else q
        d = self.d if d is None else d
        if keep is not None:
            k = torch.as_tensor(keep, device=q.device, dtype=torch.long)
            q = q.index_select(-1, k)
            d = d.index_select(-1, k)
        return similarity(q, d, self.sim)

    def ndcg_scores(self, scores: torch.Tensor, k: int = 10) -> torch.Tensor:
        return ndcg_at_k(scores, self.rel, k, self.exclude, self.ideal)

    def ndcg(self, keep: np.ndarray | None = None, q: torch.Tensor | None = None, d: torch.Tensor | None = None) -> np.ndarray:
        return self.ndcg_scores(self.scores(keep, q, d)).cpu().numpy()


@lru_cache(maxsize=None)
def retrieval_suite(model: str) -> dict[str, RetrievalEval]:
    return {name: RetrievalEval(model, name) for name in NANOBEIR}


class ClassificationEval:
    def __init__(self, model: str, name: str):
        ds = load_classification(name)
        self.model = model
        self.name = name
        self.subsamples = mteb_subsamples(ds.train_labels, ds.samples_per_label)
        used = np.unique(np.concatenate(self.subsamples))
        remap = np.full(len(ds.train_labels), -1)
        remap[used] = np.arange(len(used))
        self.x_train = load_emb(model, name, "train")[used]
        self.y_train = ds.train_labels[used]
        self.local = [remap[s] for s in self.subsamples]
        self.x_test = load_emb(model, name, "test")
        self.y_test = ds.test_labels

    def accuracy(self, keep: np.ndarray | None = None, transform=None) -> np.ndarray:
        xtr, xte = self.x_train, self.x_test
        if transform is not None:
            xtr, xte = transform(xtr), transform(xte)
        if keep is not None:
            xtr, xte = xtr[:, keep], xte[:, keep]
        return logreg_accuracy(xtr, self.y_train, xte, self.y_test, self.local)


@lru_cache(maxsize=None)
def classification_suite(model: str) -> dict[str, ClassificationEval]:
    return {name: ClassificationEval(model, name) for name in CLASSIFICATION}


def mean_relative(trunc: dict[str, float], full: dict[str, float]) -> float:
    return float(np.mean([trunc[k] / full[k] for k in full]))

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data import CACHE

EMB_DIR = CACHE / "emb"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    dim: int
    sim: str
    query_prefix: str
    doc_prefix: str
    label: str
    paper_label: str


MODELS = {
    m.key: m
    for m in [
        ModelSpec("minilm", "sentence-transformers/paraphrase-MiniLM-L3-v2", 384, "cos", "", "", "Para-MiniLM-L3", "Para-MiniLM"),
        ModelSpec("mpnet", "sentence-transformers/all-mpnet-base-v2", 768, "cos", "", "", "MPNet base", "MPNet"),
        ModelSpec("contriever", "facebook/contriever", 768, "dot", "", "", "Contriever", "Contriever"),
        ModelSpec("e5", "intfloat/e5-base-v2", 768, "cos", "query: ", "passage: ", "E5 base v2", "E5-large"),
        ModelSpec("mrl", "tomaarsen/mpnet-base-nli-matryoshka", 768, "cos", "", "", "MPNet NLI MRL", ""),
        ModelSpec("nonmrl", "tomaarsen/mpnet-base-nli", 768, "cos", "", "", "MPNet NLI", ""),
    ]
}
CORE = ["minilm", "mpnet", "contriever", "e5"]
MRL_PAIR = ["mrl", "nonmrl"]


def emb_path(model: str, dataset: str, part: str) -> Path:
    return EMB_DIR / model / f"{dataset}__{part}.npy"


def load_emb(model: str, dataset: str, part: str) -> np.ndarray:
    return np.load(emb_path(model, dataset, part), mmap_mode=None).astype(np.float32, copy=False)


def has_emb(model: str, dataset: str, part: str) -> bool:
    return emb_path(model, dataset, part).exists()


_LOADED: dict[str, object] = {}


def load_model(key: str, device: str | None = None):
    import torch
    from sentence_transformers import SentenceTransformer

    if key not in _LOADED:
        device = device or os.environ.get("TRUNBED_ENCODE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        model = SentenceTransformer(MODELS[key].hf_id, device=device)
        model.max_seq_length = min(model.max_seq_length or 512, 512)
        if device == "cuda":
            model.half()
        else:
            torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
        _LOADED[key] = model
    return _LOADED[key]


def encode_texts(key: str, texts: list[str], prefix: str = "", batch_size: int | None = None) -> np.ndarray:
    model = load_model(key)
    duty = float(os.environ.get("TRUNBED_DUTY", "1"))
    batch_size = batch_size or int(os.environ.get("TRUNBED_BATCH", "128"))
    chunk = len(texts) if duty >= 1 else 8 * batch_size
    parts = []
    for start in range(0, len(texts), chunk):
        t0 = time.perf_counter()
        parts.append(
            model.encode(
                [prefix + t for t in texts[start : start + chunk]],
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
        )
        if duty < 1:
            time.sleep((time.perf_counter() - t0) * (1 - duty) / duty)
    return np.concatenate(parts).astype(np.float32)


def encode_and_cache(key: str, dataset: str, part: str, texts: list[str], prefix: str) -> float:
    path = emb_path(key, dataset, part)
    if path.exists():
        return 0.0
    path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    emb = encode_texts(key, texts, prefix)
    elapsed = time.perf_counter() - start
    np.save(path, emb)
    model = load_model(key)
    meta = {
        "model": MODELS[key].hf_id,
        "dataset": dataset,
        "part": part,
        "n": int(emb.shape[0]),
        "dim": int(emb.shape[1]),
        "normalisation": "none applied at encode time; model modules: "
        + ",".join(type(m).__name__ for m in model),
        "prefix": prefix,
        "max_seq_length": int(model.max_seq_length),
        "dtype_encode": str(next(model.parameters()).dtype),
        "device": str(model.device),
        "seconds": round(elapsed, 2),
        "gpu_duty_cycle": float(os.environ.get("TRUNBED_DUTY", "1")),
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    return elapsed

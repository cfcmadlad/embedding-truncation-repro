from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
TEXT_DIR = CACHE / "text"

NANOBEIR = [
    "NanoArguAna",
    "NanoClimateFEVER",
    "NanoDBPedia",
    "NanoFEVER",
    "NanoFiQA2018",
    "NanoHotpotQA",
    "NanoMSMARCO",
    "NanoNFCorpus",
    "NanoNQ",
    "NanoQuoraRetrieval",
    "NanoSCIDOCS",
    "NanoSciFact",
    "NanoTouche2020",
]

CLASSIFICATION = {
    "Banking77": ("mteb/banking77", None, 8),
    "Emotion": ("mteb/emotion", None, 16),
    "TweetSentimentExtraction": ("mteb/tweet_sentiment_extraction", None, 32),
    "MTOPIntent": ("mteb/mtop_intent", "en", 8),
}

SENTENCE_SETS = {"allnli": 20000, "simplewiki": 10000}


@dataclass
class Retrieval:
    name: str
    query_ids: list[str]
    queries: list[str]
    doc_ids: list[str]
    docs: list[str]
    qrels: np.ndarray

    @property
    def same_id_mask(self) -> np.ndarray:
        pos = {d: j for j, d in enumerate(self.doc_ids)}
        mask = np.zeros(self.qrels.shape, dtype=bool)
        for i, q in enumerate(self.query_ids):
            if q in pos:
                mask[i, pos[q]] = True
        return mask


@dataclass
class Classification:
    name: str
    train_texts: list[str]
    train_labels: np.ndarray
    test_texts: list[str]
    test_labels: np.ndarray
    samples_per_label: int


def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fetch_retrieval(name: str) -> dict:
    from datasets import load_dataset

    repo = f"zeta-alpha-ai/{name}"
    corpus = load_dataset(repo, "corpus", split="train")
    queries = load_dataset(repo, "queries", split="train")
    qrels = load_dataset(repo, "qrels", split="train")
    rel_col = "score" if "score" in qrels.column_names else None
    triples = [
        [str(r["query-id"]), str(r["corpus-id"]), float(r[rel_col]) if rel_col else 1.0]
        for r in qrels
    ]
    judged = {t[0] for t in triples}
    q = [(str(r["_id"]), r["text"]) for r in queries if str(r["_id"]) in judged]
    return {
        "query_ids": [a for a, _ in q],
        "queries": [b for _, b in q],
        "doc_ids": [str(x) for x in corpus["_id"]],
        "docs": list(corpus["text"]),
        "qrels": triples,
    }


def load_retrieval(name: str) -> Retrieval:
    path = TEXT_DIR / f"{name}.json"
    if not path.exists():
        _write(path, _fetch_retrieval(name))
    raw = _read(path)
    qpos = {q: i for i, q in enumerate(raw["query_ids"])}
    dpos = {d: j for j, d in enumerate(raw["doc_ids"])}
    rel = np.zeros((len(qpos), len(dpos)), dtype=np.float32)
    for q, d, s in raw["qrels"]:
        if q in qpos and d in dpos:
            rel[qpos[q], dpos[d]] = s
    return Retrieval(name, raw["query_ids"], raw["queries"], raw["doc_ids"], raw["docs"], rel)


def _fetch_classification(name: str) -> dict:
    from datasets import load_dataset

    repo, lang, _ = CLASSIFICATION[name]
    if lang:
        files = {s: f"hf://datasets/{repo}/{lang}/{s}-00000-of-00001.parquet" for s in ("train", "test")}
        ds = load_dataset("parquet", data_files=files)
    else:
        ds = load_dataset(repo)
    out = {}
    for split in ("train", "test"):
        labels = ds[split]["label"]
        out[split] = {"texts": list(ds[split]["text"]), "labels": [int(x) for x in labels]}
    return out


def load_classification(name: str) -> Classification:
    path = TEXT_DIR / f"{name}.json"
    if not path.exists():
        _write(path, _fetch_classification(name))
    raw = _read(path)
    return Classification(
        name,
        raw["train"]["texts"],
        np.asarray(raw["train"]["labels"]),
        raw["test"]["texts"],
        np.asarray(raw["test"]["labels"]),
        CLASSIFICATION[name][2],
    )


def _fetch_sentences(name: str) -> list[str]:
    from datasets import load_dataset

    rng = np.random.default_rng(0)
    if name == "allnli":
        ds = load_dataset("sentence-transformers/all-nli", "pair", split="train")
        pool = sorted(set(ds["anchor"]) | set(ds["positive"]))
    else:
        ds = load_dataset("sentence-transformers/simple-wiki", split="train")
        pool = sorted(set(ds["text"]))
    idx = rng.choice(len(pool), SENTENCE_SETS[name], replace=False)
    return [pool[i] for i in np.sort(idx)]


def load_sentences(name: str) -> list[str]:
    path = TEXT_DIR / f"{name}.json"
    if not path.exists():
        _write(path, {"texts": _fetch_sentences(name)})
    return _read(path)["texts"]

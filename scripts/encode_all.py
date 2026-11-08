from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import CLASSIFICATION, NANOBEIR, SENTENCE_SETS, load_classification, load_retrieval, load_sentences
from src.encode import MODELS, encode_and_cache


def jobs():
    for name in NANOBEIR:
        ds = load_retrieval(name)
        yield name, "queries", ds.queries, "q"
        yield name, "docs", ds.docs, "d"
    for name in CLASSIFICATION:
        ds = load_classification(name)
        yield name, "train", ds.train_texts, "q"
        yield name, "test", ds.test_texts, "q"
    for name in SENTENCE_SETS:
        yield name, "all", load_sentences(name), "q"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--texts-only", action="store_true")
    args = ap.parse_args()
    items = list(jobs())
    if args.texts_only:
        return
    for key in args.models:
        spec = MODELS[key]
        total = sum(
            encode_and_cache(key, dataset, part, texts, spec.query_prefix if kind == "q" else spec.doc_prefix)
            for dataset, part, texts, kind in items
        )
        print(key, round(total, 1), flush=True)

if __name__ == "__main__":
    main()

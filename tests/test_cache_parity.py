from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src import live
from src.encode import has_emb
from src.evaluate import RetrievalEval

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not has_emb("minilm", "NanoSciFact", "docs"), reason="full cache not present")
def test_live_sweep_reproduces_stored_e1_value_from_cache():
    ev = RetrievalEval("minilm", "NanoSciFact")
    rows = live.retrieval_sweep(ev, 384, [0.5], [0])
    with (ROOT / "results" / "e1_retrieval.csv").open(newline="", encoding="utf-8") as f:
        stored = next(float(r["ndcg"]) for r in csv.DictReader(f) if r["model"] == "minilm" and r["dataset"] == "NanoSciFact" and r["method"] == "random" and r["frac"] == "0.5" and r["seed"] == "0")
    assert rows[0]["seed0"] == pytest.approx(stored, abs=1e-6)

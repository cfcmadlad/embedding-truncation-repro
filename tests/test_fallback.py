from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from src import live

ROOT = Path(__file__).resolve().parents[1]
HEADER = "model,dataset,docs,removed_pct,full_ndcg10,last_k,random_k_mean,random_k_std,random_k_relative_pct,paper_fig1_avg_pct"


def write_stored(folder: Path) -> None:
    rows = [HEADER]
    for pct, rel in [(0, 100.0), (50, 94.2), (90, 48.7)]:
        rows.append(f"minilm,NanoSciFact,2919,{pct},0.4713,0.4500,0.4438,0.0168,{rel},{live.paper_avg('retrieval', pct)}")
    (folder / "live_small_minilm_NanoSciFact.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def load_script():
    spec = importlib.util.spec_from_file_location("live_small_under_test", ROOT / "scripts" / "live_small.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_replay_lines_are_labelled_stored(tmp_path):
    write_stored(tmp_path)
    text = "\n".join(live.replay_lines(tmp_path / "live_small_minilm_NanoSciFact.csv"))
    assert "STORED RESULT, NOT RECOMPUTED" in text
    assert "94.2" in text and "0.4713" in text
    assert "stored numbers, not a live measurement" in text


def test_encoding_failure_prints_stored_csv(tmp_path, monkeypatch, capsys):
    write_stored(tmp_path)
    module = load_script()
    monkeypatch.setattr(module, "RESULTS", tmp_path)

    def boom(*a, **k):
        raise OSError("model files missing")

    monkeypatch.setattr(module, "load_model", boom)
    monkeypatch.setattr(sys, "argv", ["live_small.py", "--fast"])
    module.main()
    out = capsys.readouterr().out
    assert "live run failed: OSError: model files missing" in out
    assert "STORED RESULT, NOT RECOMPUTED" in out
    assert "VERDICT" in out


def test_replay_without_stored_file_exits_nonzero(tmp_path, monkeypatch):
    module = load_script()
    monkeypatch.setattr(module, "RESULTS", tmp_path)
    monkeypatch.setattr(sys, "argv", ["live_small.py", "--replay"])
    with pytest.raises(SystemExit) as e:
        module.main()
    assert e.value.code == 2

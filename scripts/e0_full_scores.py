from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytrec_eval

from src.data import load_retrieval
from src.encode import EMB_DIR, MODELS
from src.evaluate import classification_suite, retrieval_suite
from src.plotting import RES_DIR


def pytrec_check(model: str, name: str) -> dict:
    ev = retrieval_suite(model)[name]
    ds = load_retrieval(name)
    s = ev.scores().masked_fill(ev.exclude, float("-inf")).cpu().numpy()
    top = np.argsort(-s, axis=1)[:, :100]
    run = {qid: {ds.doc_ids[j]: float(s[i, j]) for j in top[i]} for i, qid in enumerate(ds.query_ids)}
    qrels = {
        qid: {ds.doc_ids[j]: int(ds.qrels[i, j]) for j in np.flatnonzero(ds.qrels[i])}
        for i, qid in enumerate(ds.query_ids)
    }
    ref = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10"}).evaluate(run)
    ours = ev.ndcg()
    theirs = np.array([ref[q]["ndcg_cut_10"] for q in ds.query_ids])
    return {"model": model, "dataset": name, "ours": float(ours.mean()), "pytrec_eval": float(theirs.mean()), "max_abs_diff": float(np.abs(ours - theirs).max())}


def encoding_times() -> pd.DataFrame:
    metas = [json.loads(p.read_text()) for p in EMB_DIR.glob("*/*.json")]
    df = pd.DataFrame(metas)
    df["model_key"] = df["model"].map({s.hf_id: k for k, s in MODELS.items()})
    return df.groupby(["model_key", "model", "device", "dtype_encode", "gpu_duty_cycle"]).agg(texts=("n", "sum"), seconds=("seconds", "sum")).reset_index()


def main() -> None:
    encoding_times().to_csv(RES_DIR / "e0_encoding_times.csv", index=False)
    rows = []
    for model in MODELS:
        for name, ev in retrieval_suite(model).items():
            rows.append({"model": model, "task": "retrieval", "dataset": name, "score": float(ev.ndcg().mean())})
        for name, ev in classification_suite(model).items():
            rows.append({"model": model, "task": "classification", "dataset": name, "score": float(ev.accuracy().mean())})
    df = pd.DataFrame(rows)
    df.to_csv(RES_DIR / "e0_full_scores.csv", index=False)
    summary = df.groupby(["model", "task"])["score"].mean().unstack()
    summary.to_csv(RES_DIR / "e0_full_scores_summary.csv")
    checks = [pytrec_check(m, n) for m in ("mpnet", "contriever") for n in ("NanoSciFact", "NanoArguAna")]
    (RES_DIR / "e0_pytrec_check.json").write_text(json.dumps(checks, indent=1))
    print(summary.round(4))
    print(pd.DataFrame(checks))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo"))

import core
from src.encode import MODELS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mpnet", choices=list(MODELS))
    ap.add_argument("--dataset", default="NanoSciFact", choices=core.DATASETS)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--method", default="random", choices=["last", "random", "pca", "degrading", "improving"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    method = {"last": "last-K", "random": "random-K", "pca": "PCA", "degrading": "remove degrading", "improving": "remove improving"}[args.method]
    start = time.perf_counter()
    res = core.evaluate(args.model, args.dataset, method, round(args.k / 100, 2), args.seed)
    print(f"model      {args.model} ({MODELS[args.model].hf_id})")
    print(f"dataset    {args.dataset}")
    print(f"method     {method}, K={args.k}%, seed={args.seed}, removed {res['removed']}/{MODELS[args.model].dim}")
    print(f"full       {res['metric']} {res['full']:.4f}")
    print(f"truncated  {res['metric']} {res['truncated']:.4f}")
    print(f"relative   {res['relative']:.2f}%")
    if res["spearman"] is not None:
        print(f"spearman   {res['spearman']:.4f}")
    print(f"seconds    {time.perf_counter() - start:.2f}")


if __name__ == "__main__":
    main()

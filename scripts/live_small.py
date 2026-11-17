from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.runtime import force_cpu, physical_cores, setup_console, threads

BOOT = time.perf_counter()
setup_console()
force_cpu()
if "--replay" not in sys.argv:
    print("starting: loading libraries and the encoder, about 15 s, then the live run", flush=True)

import numpy as np
import torch

from src import live
from src.data import CLASSIFICATION, NANOBEIR, load_retrieval
from src.encode import MODELS, has_emb, load_emb, load_model
from src.evaluate import RetrievalEval
from src.truncate import FRACTIONS, SEEDS, keep_random

RESULTS = ROOT / "results"
PROFILES = {
    "fast": {"max_docs": 400, "experiments": 1, "test_n": 300, "seeds": 3, "fracs": [0.0, 0.5, 0.9]},
    "full": {"max_docs": None, "experiments": 2, "test_n": 1000, "seeds": 5, "fracs": live.CLASSIFICATION_FRACS},
}


def stored_e1(model: str, dataset: str, frac: float, seed: int) -> float | None:
    path = RESULTS / "e1_retrieval.csv"
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["model"] == model and r["dataset"] == dataset and r["method"] == "random" and abs(float(r["frac"]) - frac) < 1e-9 and int(r["seed"]) == seed:
                return float(r["ndcg"])
    return None


def retrieval_part(args: argparse.Namespace, profile: dict, spec) -> float:
    full_ds = load_retrieval(args.dataset)
    ds = live.subset(full_ds, args.max_docs or profile["max_docs"])
    cut = len(ds.docs) < len(full_ds.docs)
    note = f"  [corpus cut, all {int((ds.qrels.sum(0) > 0).sum())} judged documents kept]" if cut else ""
    t = time.perf_counter()
    load_model(args.model)
    print(f"startup    imports and encoder load {time.perf_counter() - BOOT:.1f} s (model load alone {time.perf_counter() - t:.1f} s)")
    print(f"dataset    {args.dataset}: {len(ds.queries)} queries, {len(ds.docs)} of {len(full_ds.docs)} documents{note}")
    q, qs = live.encode_timed(args.model, ds.queries, spec.query_prefix)
    d, dsec = live.encode_timed(args.model, ds.docs, spec.doc_prefix)
    n = len(q) + len(d)
    print(f"encoded    {n} texts live in {qs + dsec:.1f} s ({n / (qs + dsec):.0f} texts per second, cpu, {torch.get_num_threads()} threads)")
    if not cut and has_emb(args.model, args.dataset, "docs"):
        gap = float(np.abs(d - load_emb(args.model, args.dataset, "docs")).max())
        print(f"cache      live document embeddings differ from the cached ones behind RESULTS.md by at most {gap:.1e}")
    ev = RetrievalEval(args.model, args.dataset, q, d, ds)
    rows = live.retrieval_sweep(ev, spec.dim, FRACTIONS, SEEDS)
    print(f"full       nDCG@10 {rows[0]['full']:.4f}")
    print("reference  'paper avg' is Figure 1 of the paper: random K, average of its 6 encoders over BEIR (read off the plot)\n")
    print("\n".join(live.render(rows, "retrieval")))
    half = keep_random(spec.dim, 0.5, 0)
    removed, zeroed, gap = live.remove_vs_zero(ev, spec.dim, half)
    print(f"\nremove vs zero   the same {spec.dim - len(half)} dimensions removed or zeroed: nDCG@10 {removed:.4f} and {zeroed:.4f}, largest score difference {gap:.1e}")
    stored = None if cut else stored_e1(args.model, args.dataset, 0.5, 0)
    if stored is not None:
        print(f"stored     random 50% seed 0 in results/e1_retrieval.csv (cached embeddings): {stored:.4f}; this live run: {rows[5]['seed0']:.4f}")
    path = RESULTS / f"live_small_{args.model}_{args.dataset}{'_fast' if cut else ''}.csv"
    live.write_csv(path, live.RETRIEVAL_FIELDS, [[args.model, args.dataset, len(ds.docs), r["removed_pct"], r["full"], r["last_k"], r["random_mean"], r["random_std"], r["random_rel"], live.paper_avg("retrieval", r["removed_pct"])] for r in rows])
    print(f"written    {path.relative_to(ROOT)}")
    return rows[5]["random_rel"]


def classification_part(args: argparse.Namespace, profile: dict, spec) -> float:
    plan = live.classification_plan(args.task, profile["experiments"], profile["test_n"])
    print(f"\ntask       {args.task}: {len(plan.train_texts)} training texts ({profile['experiments']} MTEB subsample(s) of {CLASSIFICATION[args.task][2]} per label), {len(plan.test_texts)} test texts")
    x_train, ts = live.encode_timed(args.model, plan.train_texts, spec.query_prefix)
    x_test, es = live.encode_timed(args.model, plan.test_texts, spec.query_prefix)
    print(f"encoded    {len(x_train) + len(x_test)} texts live in {ts + es:.1f} s")
    start = time.perf_counter()
    rows = live.classification_sweep(plan, x_train, x_test, spec.dim, list(range(profile["seeds"])), profile["fracs"], min(4, threads()))
    print(f"full       accuracy {rows[0]['full']:.4f} (logistic regression, {time.perf_counter() - start:.1f} s for the sweep)")
    print("reference  'paper avg' is Figure 1 of the paper: random K, average of its 6 encoders over 12 MTEB tasks (read off the plot)\n")
    print("\n".join(live.render(rows, "classification")))
    suffix = "_fast" if args.fast else ""
    path = RESULTS / f"live_small_cls_{args.model}_{args.task}{suffix}.csv"
    live.write_csv(path, live.CLASSIFICATION_FIELDS, [[args.model, args.task, len(plan.train_texts), len(plan.test_texts), profile["experiments"], r["removed_pct"], r["full"], r["last_k"], r["random_mean"], r["random_std"], r["random_rel"], live.paper_avg("classification", r["removed_pct"])] for r in rows])
    print(f"written    {path.relative_to(ROOT)}")
    return next(r["random_rel"] for r in rows if r["removed_pct"] == 50)


def replay(args: argparse.Namespace, reason: str) -> None:
    retrieval = RESULTS / f"live_small_{args.model}_{args.dataset}.csv"
    if not retrieval.exists():
        print(f"no stored result named {retrieval.name} in results; run the live script once on a machine where it works and keep the csv")
        sys.exit(2)
    print("\n".join(live.replay_lines(retrieval, RESULTS / f"live_small_cls_{args.model}_{args.task}.csv", reason)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="minilm", choices=list(MODELS))
    ap.add_argument("--dataset", default="NanoSciFact", choices=NANOBEIR)
    ap.add_argument("--task", default="Banking77", choices=list(CLASSIFICATION))
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--no-classification", action="store_true")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--replay", action="store_true")
    args = ap.parse_args()
    if args.replay:
        replay(args, "replay requested")
        return
    if not args.fast and not args.full and not args.max_docs and physical_cores() < 4:
        args.fast = True
        print(f"note       only {physical_cores()} CPU cores found, so --fast is used to stay near one minute (add --full to override)")
    if args.threads:
        os.environ["TRUNBED_THREADS"] = str(args.threads)
    torch.set_num_threads(threads())
    profile = PROFILES["fast" if args.fast else "full"]
    spec = MODELS[args.model]
    start = time.perf_counter()
    print(f"model      {spec.hf_id} ({spec.dim} dimensions, {spec.sim} similarity)")
    try:
        retrieval_rel = retrieval_part(args, profile, spec)
        classification_rel = None if args.no_classification else classification_part(args, profile, spec)
    except Exception as e:
        print(f"\nlive run failed: {type(e).__name__}: {e}\n")
        replay(args, "live encoding failed")
        return
    print("\n" + "\n".join(live.verdict(retrieval_rel, classification_rel, args.model)))
    print(f"total      {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()

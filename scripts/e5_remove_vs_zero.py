from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.encode import MODELS, load_emb
from src.plotting import RES_DIR


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a @ b.T


def compare(q: np.ndarray, d: np.ndarray, dims: np.ndarray) -> dict:
    keep = np.setdiff1d(np.arange(q.shape[1]), dims)
    qz, dz = q.copy(), d.copy()
    qz[:, dims] = 0
    dz[:, dims] = 0
    return {
        "removed": len(dims),
        "max_abs_diff_dot": float(np.abs(q[:, keep] @ d[:, keep].T - qz @ dz.T).max()),
        "max_abs_diff_cos": float(np.abs(cosine(q[:, keep], d[:, keep]) - cosine(qz, dz)).max()),
    }


def main() -> None:
    rng = np.random.default_rng(0)
    out = {"synthetic": [], "cached": []}
    q = rng.standard_normal((64, 768)).astype(np.float64)
    d = rng.standard_normal((512, 768)).astype(np.float64)
    for n in (1, 10, 384, 700):
        out["synthetic"].append(compare(q, d, rng.choice(768, n, replace=False)))
    for model in MODELS:
        q = load_emb(model, "NanoSciFact", "queries").astype(np.float64)
        d = load_emb(model, "NanoSciFact", "docs").astype(np.float64)
        dim = q.shape[1]
        for n in (1, dim // 2):
            out["cached"].append({"model": model, **compare(q, d, rng.choice(dim, n, replace=False))})
    worst = max(r[k] for part in out.values() for r in part for k in ("max_abs_diff_dot", "max_abs_diff_cos"))
    out["worst_abs_diff"] = worst
    out["identical_within_1e-10"] = bool(worst < 1e-10)
    (RES_DIR / "e5_remove_vs_zero.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"worst_abs_diff": worst, "identical": out["identical_within_1e-10"]}))


if __name__ == "__main__":
    main()

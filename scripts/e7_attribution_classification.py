from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from src.attribution import labels
from src.encode import MODELS
from src.evaluate import ClassificationEval
from src.metrics import n_jobs
from src.plotting import RES_DIR

PAIRS = [("minilm", "Banking77"), ("minilm", "Emotion"), ("mpnet", "Banking77"), ("mpnet", "Emotion")]


def run_dims(ce: ClassificationEval, dims: np.ndarray) -> np.ndarray:
    full = np.arange(ce.x_train.shape[1])
    return np.array([ce.accuracy(np.delete(full, d)).mean() for d in dims])


def main() -> None:
    threadpool_limits(1)
    out = RES_DIR / "e7_attribution_classification.csv"
    frames = []
    for model, task in PAIRS:
        start = time.perf_counter()
        ce = ClassificationEval(model, task)
        acc_full = float(ce.accuracy().mean())
        dims = np.arange(MODELS[model].dim)
        chunks = np.array_split(dims, n_jobs() * 4)
        acc = np.concatenate(Parallel(n_jobs=n_jobs(), prefer="threads")(delayed(run_dims)(ce, c) for c in chunks))
        delta = acc - acc_full
        frames.append(
            pd.DataFrame({"model": model, "task": task, "dim": dims, "acc_full": acc_full, "acc_without": acc, "delta": delta, "label": labels(delta)})
        )
        pd.concat(frames, ignore_index=True).to_csv(out, index=False)
        print(model, task, round(time.perf_counter() - start, 1), pd.Series(labels(delta)).value_counts().to_dict(), flush=True)


if __name__ == "__main__":
    main()

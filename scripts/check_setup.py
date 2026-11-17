from __future__ import annotations

import importlib
import platform
import shutil
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.runtime import force_cpu, setup_console

setup_console()
force_cpu()

REQUIRED = ["numpy", "torch", "sklearn", "scipy", "pandas", "sentence_transformers", "transformers", "psutil"]
OPTIONAL = {"streamlit": "only for the Streamlit demo", "matplotlib": "only for the Streamlit demo", "pytest": "only for the test suite", "pytrec_eval": "only for the nDCG cross-check"}
FAILS: list[str] = []


def report(ok: bool, label: str, detail: str = "", required: bool = True) -> None:
    tag = "ok" if ok else ("FAIL" if required else "absent")
    print(f"{tag:<7}{label:<34}{detail}")
    if not ok and required:
        FAILS.append(label)


def check_python() -> None:
    v = sys.version_info
    report((3, 10) <= v[:2] < (3, 14), "python", f"{platform.python_version()} on {platform.system()} {platform.machine()}; needs 3.10 to 3.13")
    report(True, "console encoding", f"{sys.stdout.encoding}; path {'contains spaces (handled)' if ' ' in str(ROOT) else 'has no spaces'}")


def check_packages() -> None:
    for name in REQUIRED:
        try:
            module = importlib.import_module(name)
            report(True, f"package {name}", getattr(module, "__version__", ""))
        except Exception as e:
            report(False, f"package {name}", f"{type(e).__name__}: install with pip install -r requirements-demo.txt")
    for name, why in OPTIONAL.items():
        try:
            module = importlib.import_module(name)
            report(True, f"package {name}", getattr(module, "__version__", ""), False)
        except Exception:
            report(False, f"package {name}", why, False)


def check_files() -> None:
    model = ROOT / "lite" / "models" / "minilm"
    weights = list(model.glob("*.safetensors")) + list(model.glob("*.bin"))
    report((model / "modules.json").exists() and bool(weights), "model files lite/models/minilm", "" if weights else "run python scripts/fetch_lite.py")
    for name in ("NanoSciFact", "Banking77"):
        found = [d for d in (ROOT / "cache" / "text", ROOT / "lite" / "data") if (d / f"{name}.json").exists()]
        report(bool(found), f"data {name}.json", str(found[0].relative_to(ROOT)) if found else "run python scripts/fetch_lite.py")
    stored = ROOT / "results" / "live_small_minilm_NanoSciFact.csv"
    report(stored.exists(), "stored replay csv", stored.name)
    report((ROOT / "cache" / "emb" / "READY").exists(), "full cache (Streamlit demo)", "present" if (ROOT / "cache" / "emb" / "READY").exists() else "not needed for the live run", False)
    report(shutil.disk_usage(ROOT).free > 2e9, "free disk space", f"{shutil.disk_usage(ROOT).free / 1e9:.0f} GB")
    with socket.socket() as s:
        report(s.connect_ex(("127.0.0.1", 8501)) != 0, "port 8501 free (Streamlit)", "", False)


def smoke() -> None:
    import numpy as np
    import torch

    from src import live
    from src.encode import MODELS, encode_texts
    from src.evaluate import RetrievalEval
    from src.truncate import keep_random
    from tests.helpers import synthetic

    start = time.perf_counter()
    try:
        out = encode_texts("minilm", ["a cat sits on the mat", "dogs bark loudly", "random truncation of embeddings"], "")
        good = out.shape == (3, MODELS["minilm"].dim) and bool(np.isfinite(out).all())
        q, d, ds = synthetic(dim=MODELS["minilm"].dim)
        ev = RetrievalEval("minilm", "synthetic", q, d, ds)
        removed, zeroed, gap = live.remove_vs_zero(ev, MODELS["minilm"].dim, keep_random(MODELS["minilm"].dim, 0.5, 0))
        good = good and abs(removed - zeroed) < 1e-6 and torch.get_num_threads() >= 1
        seconds = time.perf_counter() - start
        report(good, "smoke run", f"encode 3 texts, nDCG, remove vs zero in {seconds:.1f} s (imports excluded)")
    except Exception as e:
        report(False, "smoke run", f"{type(e).__name__}: {e}")


def main() -> None:
    check_python()
    check_packages()
    check_files()
    smoke()
    print()
    print("RESULT   all required checks ok" if not FAILS else f"RESULT   {len(FAILS)} FAIL: " + ", ".join(FAILS))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()

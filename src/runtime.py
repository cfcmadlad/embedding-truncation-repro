from __future__ import annotations

import os
import sys


def physical_cores() -> int:
    try:
        import psutil

        n = psutil.cpu_count(logical=False)
    except Exception:
        n = None
    return max(1, n or (os.cpu_count() or 2) // 2)


def threads() -> int:
    return max(1, int(os.environ.get("TRUNBED_THREADS") or os.environ.get("TRUNBED_JOBS") or physical_cores()))


def setup_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def force_cpu() -> None:
    os.environ["TRUNBED_DEVICE"] = "cpu"
    os.environ["TRUNBED_ENCODE_DEVICE"] = "cpu"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")

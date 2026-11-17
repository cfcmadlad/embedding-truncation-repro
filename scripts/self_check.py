from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.runtime import physical_cores, setup_console

setup_console()

STEPS = [
    ("check", ["scripts/check_setup.py"], 120),
    ("fast live run, 1 thread", ["scripts/live_small.py", "--fast", "--threads", "1"], 300),
    ("full live run", ["scripts/live_small.py"], 600),
    ("replay of the stored result", ["scripts/live_small.py", "--replay"], 60),
]


def machine() -> list[str]:
    try:
        import psutil

        ram = f"{psutil.virtual_memory().total / 1e9:.0f} GB"
    except Exception:
        ram = "unknown"
    return [
        f"date      {datetime.now():%Y-%m-%d %H:%M}",
        f"host      {socket.gethostname()}",
        f"system    {platform.platform()}",
        f"python    {platform.python_version()} ({sys.executable})",
        f"cpu       {platform.processor() or platform.machine()}, {os.cpu_count()} logical, {physical_cores()} physical cores, {ram} RAM",
        f"path      {ROOT} (spaces: {' ' in str(ROOT)})",
    ]


def run(name: str, args: list[str], timeout: int) -> tuple[str, float, int]:
    env = dict(os.environ, PYTHONUTF8="1", CUDA_VISIBLE_DEVICES="-1")
    start = time.perf_counter()
    try:
        proc = subprocess.run([sys.executable, *args], cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
        return proc.stdout + proc.stderr, time.perf_counter() - start, proc.returncode
    except subprocess.TimeoutExpired:
        return f"TIMEOUT after {timeout} s", time.perf_counter() - start, -1


def main() -> None:
    lines = ["SELF CHECK REPORT", *machine(), ""]
    summary = []
    for name, args, timeout in STEPS:
        print(f"running: {name} ...", flush=True)
        out, seconds, code = run(name, args, timeout)
        status = "ok" if code == 0 else f"exit code {code}"
        summary.append(f"{name:<32}{seconds:6.1f} s  {status}")
        lines += [f"===== {name} ({seconds:.1f} s, {status})", out.strip(), ""]
        print(f"  {seconds:.1f} s, {status}", flush=True)
    lines = lines[:1] + ["SUMMARY", *summary, ""] + lines[1:]
    path = ROOT / f"selfcheck_report_{socket.gethostname()}.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(summary))
    print(f"\nreport written to {path.name}; send this file back")


if __name__ == "__main__":
    main()

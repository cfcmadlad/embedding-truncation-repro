from __future__ import annotations

import argparse
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://github.com/cfcmadlad/embedding-truncation-repro/releases/download/cache-v1"
PARTS = ["cache_part1.zip", "cache_part2.zip"]


def download(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url) as resp, dest.open("wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while chunk := resp.read(1 << 22):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r{dest.name}: {done / 1e9:.2f} / {total / 1e9:.2f} GB", end="", flush=True)
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    ready = ROOT / "cache" / "emb" / "READY"
    if ready.exists() and not args.force:
        print("cache/ already present, nothing to do (use --force to download again)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        for name in PARTS:
            path = Path(tmp) / name
            download(f"{BASE}/{name}", path)
            with zipfile.ZipFile(path) as zf:
                if zf.testzip() is not None:
                    sys.exit(f"{name} is corrupted, run the script again")
                zf.extractall(ROOT)
            path.unlink()
    if not ready.exists():
        sys.exit("download finished but cache/emb/READY is missing")
    print("cache/ ready")


if __name__ == "__main__":
    main()

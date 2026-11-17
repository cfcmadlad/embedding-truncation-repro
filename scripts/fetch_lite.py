from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import CACHE, LITE, _fetch_classification, _fetch_retrieval, _write
from src.encode import MODELS

DATA = {"NanoSciFact": _fetch_retrieval, "Banking77": _fetch_classification}
KIT_DIRS = ["src", "scripts", "demo", "tests", "docs", "results", "figures", "lite"]
KIT_FILES = ["README.md", "REPRO_NOTES.md", "RESULTS.md", "requirements-demo.txt", "requirements.txt", "run.bat", "run.sh", "pytest.ini"]
SKIP_PARTS = {"__pycache__", ".cache", ".pytest_cache"}


def fetch_model(key: str, force: bool) -> None:
    target = LITE / "models" / key
    if (target / "modules.json").exists() and not force:
        print(f"model      {key}: present")
        return
    from huggingface_hub import list_repo_files, snapshot_download

    repo = MODELS[key].hf_id
    files = list_repo_files(repo)
    allow = ["*.json", "*.txt", "*.model", "1_Pooling/*", "2_Normalize/*", "*.safetensors" if any(f.endswith(".safetensors") for f in files) else "*.bin"]
    snapshot_download(repo, local_dir=str(target), allow_patterns=allow)
    shutil.rmtree(target / ".cache", ignore_errors=True)
    print(f"model      {key}: {repo} saved to {target.relative_to(ROOT)}")


def fetch_data(name: str, force: bool) -> None:
    target = LITE / "data" / f"{name}.json"
    if target.exists() and not force:
        print(f"data       {name}: present")
        return
    source = CACHE / "text" / f"{name}.json"
    if source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    else:
        _write(target, DATA[name](name))
    print(f"data       {name}: {target.relative_to(ROOT)} ({target.stat().st_size / 1e6:.1f} MB)")


def build_zip() -> Path:
    out = ROOT / "dist" / "embedding-truncation-lite.zip"
    out.parent.mkdir(exist_ok=True)
    top = "embedding-truncation-lite"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in KIT_FILES:
            if (ROOT / name).exists():
                zf.write(ROOT / name, f"{top}/{name}")
        for folder in KIT_DIRS:
            for path in sorted((ROOT / folder).rglob("*")):
                if path.is_file() and not SKIP_PARTS & set(path.parts):
                    zf.write(path, f"{top}/{path.relative_to(ROOT).as_posix()}")
    print(f"zip        {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.0f} MB)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["minilm"], choices=list(MODELS))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--zip", action="store_true")
    args = ap.parse_args()
    for key in args.models:
        fetch_model(key, args.force)
    for name in DATA:
        fetch_data(name, args.force)
    if args.zip:
        build_zip()
    print("lite kit ready")


if __name__ == "__main__":
    main()

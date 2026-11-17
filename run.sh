#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1 TOKENIZERS_PARALLELISM=false

if [ ! -x ".venv/bin/python" ]; then
  found=""
  for v in ${TRUNBED_PY:-} 3.13 3.12 3.11 3.10; do
    if command -v "python$v" >/dev/null 2>&1; then
      "python$v" -m venv .venv && found="$v" && break
    fi
  done
  if [ -z "$found" ] && command -v python3 >/dev/null 2>&1 && python3 -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] < (3,14) else 1)"; then
    python3 -m venv .venv && found="python3"
  fi
  if [ -z "$found" ]; then
    echo "FAIL: no Python 3.10 to 3.13 found. Install Python 3.12 (macOS: brew install python@3.12, Linux: sudo apt install python3.12 python3.12-venv) and run again."
    exit 1
  fi
fi
PY=".venv/bin/python"

TRUST="--trusted-host pypi.org --trusted-host files.pythonhosted.org"
if [ ! -f ".venv/deps.ok" ]; then
  echo "Installing dependencies once. This needs internet and takes 5 to 8 minutes."
  "$PY" -m pip install --quiet --upgrade pip >/dev/null 2>&1 || "$PY" -m pip install --quiet --upgrade pip $TRUST >/dev/null 2>&1
  if [ "$(uname -s)" = "Linux" ]; then
    "$PY" -m pip install --quiet "torch>=2.3,<3" --index-url https://download.pytorch.org/whl/cpu >/dev/null 2>&1 || true
  fi
  if ! "$PY" -m pip install --quiet -r requirements-demo.txt; then
    echo "First attempt failed, probably antivirus HTTPS scanning. Retrying with certificate checks relaxed for pypi.org and files.pythonhosted.org only."
    if ! "$PY" -m pip install --quiet -r requirements-demo.txt $TRUST; then
      echo "FAIL: pip install failed. Check the internet connection and run ./run.sh again."
      exit 1
    fi
  fi
  echo ok > .venv/deps.ok
fi

mode="${1:-live}"
[ $# -gt 0 ] && shift
case "$mode" in
  live) exec "$PY" scripts/live_small.py "$@" ;;
  fast) exec "$PY" scripts/live_small.py --fast "$@" ;;
  replay) exec "$PY" scripts/live_small.py --replay "$@" ;;
  check) exec "$PY" scripts/check_setup.py ;;
  selfcheck) exec "$PY" scripts/self_check.py ;;
  test) exec "$PY" -m pytest -q ;;
  streamlit)
    if [ ! -f cache/emb/READY ]; then
      echo "FAIL: the Streamlit demo needs the full cache. Run: $PY scripts/fetch_cache.py"
      exit 1
    fi
    exec "$PY" -m streamlit run demo/app.py "$@" ;;
  *) echo "usage: ./run.sh [live | fast | replay | check | selfcheck | test | streamlit]"; exit 2 ;;
esac

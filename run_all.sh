#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUTF8=1 HF_HUB_OFFLINE=1
PY=${PYTHON:-python}
if [ ! -f cache/emb/READY ]; then
  $PY scripts/encode_all.py
  touch cache/emb/READY
fi
mkdir -p results figures
$PY scripts/e0_full_scores.py
$PY scripts/e1_truncation.py
$PY scripts/e2_spearman.py
$PY scripts/e3_pca.py
$PY scripts/e4_mrl.py
$PY scripts/e5_remove_vs_zero.py
$PY scripts/e6_outliers.py
$PY scripts/e7_attribution_retrieval.py
$PY scripts/e7_attribution_classification.py
$PY scripts/e8_degrading_removal.py
$PY scripts/e9_shared_degrading.py
$PY scripts/e10_anisotropy.py
$PY scripts/x1_leakage.py
$PY scripts/x2_bootstrap.py
$PY scripts/x3_theory.py
if [ "${RUN_S1:-0}" = "1" ]; then
  HF_HUB_OFFLINE=0 $PY scripts/s1_llm_truncation.py --prepare
  $PY scripts/s1_llm_truncation.py
fi
$PY scripts/demo_cli.py --model mpnet --dataset NanoSciFact --k 50 --method random --seed 0

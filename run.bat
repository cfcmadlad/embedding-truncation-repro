@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set CUDA_VISIBLE_DEVICES=-1
set TOKENIZERS_PARALLELISM=false

if not exist ".venv\Scripts\python.exe" (
  set FOUND=
  for %%V in (%TRUNBED_PY% 3.13 3.12 3.11 3.10) do (
    if not defined FOUND (
      py -%%V -c "import sys" >nul 2>nul && (
        py -%%V -m venv .venv && set FOUND=%%V
      )
    )
  )
  if not defined FOUND (
    python -c "import sys; assert (3,10) <= sys.version_info[:2] < (3,14)" >nul 2>nul && (
      python -m venv .venv && set FOUND=python
    )
  )
  if not defined FOUND (
    echo FAIL: no Python 3.10 to 3.13 found. Install Python 3.12 from python.org and tick "Add python.exe to PATH".
    exit /b 1
  )
)
set PY=.venv\Scripts\python.exe

if not exist ".venv\deps.ok" (
  echo Installing dependencies once. This needs internet and takes 5 to 8 minutes.
  "%PY%" -m pip install --quiet --upgrade pip >nul 2>nul
  if errorlevel 1 "%PY%" -m pip install --quiet --upgrade pip --trusted-host pypi.org --trusted-host files.pythonhosted.org >nul 2>nul
  "%PY%" -m pip install --quiet -r requirements-demo.txt
  if errorlevel 1 (
    echo First attempt failed, probably antivirus HTTPS scanning. Retrying with certificate checks relaxed for pypi.org and files.pythonhosted.org only.
    "%PY%" -m pip install --quiet -r requirements-demo.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org
    if errorlevel 1 (
      echo FAIL: pip install failed. Check the internet connection and run run.bat again.
      exit /b 1
    )
  )
  echo ok> ".venv\deps.ok"
)

set MODE=%~1
if "%MODE%"=="" set MODE=live
shift
set REST=
:collect
if "%~1"=="" goto run
set REST=%REST% %1
shift
goto collect

:run
if /i "%MODE%"=="live" "%PY%" scripts\live_small.py %REST% & exit /b !errorlevel!
if /i "%MODE%"=="fast" "%PY%" scripts\live_small.py --fast %REST% & exit /b !errorlevel!
if /i "%MODE%"=="replay" "%PY%" scripts\live_small.py --replay %REST% & exit /b !errorlevel!
if /i "%MODE%"=="check" "%PY%" scripts\check_setup.py & exit /b !errorlevel!
if /i "%MODE%"=="selfcheck" "%PY%" scripts\self_check.py & exit /b !errorlevel!
if /i "%MODE%"=="test" "%PY%" -m pytest -q & exit /b !errorlevel!
if /i "%MODE%"=="streamlit" (
  if not exist "cache\emb\READY" (
    echo FAIL: the Streamlit demo needs the full cache. Run: %PY% scripts\fetch_cache.py
    exit /b 1
  )
  "%PY%" -m streamlit run demo\app.py %REST%
  exit /b !errorlevel!
)
echo usage: run.bat [live ^| fast ^| replay ^| check ^| selfcheck ^| test ^| streamlit]
exit /b 2

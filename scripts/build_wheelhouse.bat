@echo off
setlocal enabledelayedexpansion
REM Run this at HOME (internet available). Rebuilt for Python 3.14 — the
REM facility PC's default "python" resolves to 3.14 and 3.10 could not be
REM made to take priority there across two visits.
REM
REM Same "py -3.14" discipline as before, for the same reason: conditional
REM dependencies (python_version < 'X' markers) get evaluated by whatever
REM Python is actually RUNNING pip, not by --python-version alone.

set PYVER=314
set PLATFORM=win_amd64
set OUTDIR=wheelhouse

echo Building offline wheelhouse for cp%PYVER% / %PLATFORM%
echo Output: %OUTDIR%\
echo.

if not exist %OUTDIR% mkdir %OUTDIR%

echo [1/3] Downloading torch + torchvision from PyTorch's CPU-only index...
py -3.14 -m pip download torch torchvision ^
    --index-url https://download.pytorch.org/whl/cpu ^
    -d %OUTDIR% ^
    --python-version %PYVER% ^
    --platform %PLATFORM% ^
    --only-binary=:all:

if errorlevel 1 (
    echo.
    echo FAILED downloading torch/torchvision CPU build for cp314.
    echo If this fails, torch's CPU index may not yet have a 3.14 build -
    echo report back the exact error before proceeding further.
    exit /b 1
)

echo.
echo [2/3] Downloading ultralytics itself, --no-deps...
py -3.14 -m pip download ultralytics --no-deps ^
    -d %OUTDIR% ^
    --python-version %PYVER% ^
    --platform %PLATFORM% ^
    --only-binary=:all:

if errorlevel 1 (
    echo.
    echo FAILED downloading ultralytics for cp314.
    exit /b 1
)

echo.
echo [3/3] Downloading everything else from requirements.txt (default PyPI)...
py -3.14 -m pip download -r requirements.txt ^
    -d %OUTDIR% ^
    --python-version %PYVER% ^
    --platform %PLATFORM% ^
    --only-binary=:all:

if errorlevel 1 (
    echo.
    echo FAILED - a package has no cp314/win_amd64 wheel yet.
    echo Check the error above for WHICH package - if it's pyarrow or
    echo streamlit, that's the known-risk item - report the exact
    echo package/version back before working around it.
    exit /b 1
)

echo.
dir /b %OUTDIR%\*.whl > %OUTDIR%\..\versions.txt
powershell -ExecutionPolicy Bypass -File scripts\compute_hashes.ps1

echo.
echo Done. Wheelhouse ready at %OUTDIR%\
dir /b %OUTDIR%
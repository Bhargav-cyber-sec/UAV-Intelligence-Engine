@echo off
setlocal enabledelayedexpansion

echo ==============================================================================
echo   UAV Intelligence Engine - Week 2 Test Suite Verification
echo ==============================================================================
echo.

set PYTHON_CMD=

if exist "venv\Scripts\python.exe" (
    set PYTHON_CMD=venv\Scripts\python.exe
    echo [*] Using local virtualenv: !PYTHON_CMD!
    goto :found_python
)

if exist "..\Phase1_Environment_Setup\venv\Scripts\python.exe" (
    set PYTHON_CMD=..\Phase1_Environment_Setup\venv\Scripts\python.exe
    echo [*] Using parent Phase1 virtualenv: !PYTHON_CMD!
    goto :found_python
)

where py >nul 2>&1
if %errorlevel% equ 0 (
    py -3.10 -c "import sys" >nul 2>&1
    if !errorlevel! equ 0 (
        set PYTHON_CMD=py -3.10
        echo [*] Using py -3.10 launcher
        goto :found_python
    )
    py -3.14 -c "import sys" >nul 2>&1
    if !errorlevel! equ 0 (
        set PYTHON_CMD=py -3.14
        echo [*] Using py -3.14 launcher
        goto :found_python
    )
)

where python >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=python
    echo [*] Using system python on PATH
    goto :found_python
)

echo [ERROR] No suitable Python interpreter found!
echo Please ensure Python 3.14 or the Phase 1 virtualenv is present.
exit /b 1

:found_python
echo [*] Python interpreter resolved.
echo [*] Running full test suite (28 automated tests)...
echo.

%PYTHON_CMD% -m pytest tests/ --basetemp=tests/tmp_pytest -v

if %errorlevel% equ 0 (
    echo.
    echo ==============================================================================
    echo   [SUCCESS] ALL TESTS PASSED! Week 2 verification complete.
    echo ==============================================================================
) else (
    echo.
    echo ==============================================================================
    echo   [FAILED] Some tests did not pass. Check output above for details.
    echo ==============================================================================
)

echo.
pause

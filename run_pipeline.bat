@echo off
setlocal enabledelayedexpansion

echo ==============================================================================
echo   UAV Intelligence Engine - Run Week 2 Pipeline
echo ==============================================================================
echo.

set PYTHON_CMD=

if exist "venv\Scripts\python.exe" (
    set PYTHON_CMD=venv\Scripts\python.exe
    goto :found_python
)

if exist "..\Phase1_Environment_Setup\venv\Scripts\python.exe" (
    set PYTHON_CMD=..\Phase1_Environment_Setup\venv\Scripts\python.exe
    goto :found_python
)

where py >nul 2>&1
if %errorlevel% equ 0 (
    py -3.14 -c "import sys" >nul 2>&1
    if !errorlevel! equ 0 (
        set PYTHON_CMD=py -3.14
        goto :found_python
    )
)

where python >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=python
    goto :found_python
)

echo [ERROR] No suitable Python interpreter found!
echo Please ensure Python 3.14 or the Phase 1 virtualenv is present.
exit /b 1

:found_python
set VIDEO_INPUT=%1
if "%VIDEO_INPUT%"=="" (
    if exist "data\sample.mp4" (
        set VIDEO_INPUT=data\sample.mp4
        echo [*] No video specified, defaulting to bundled test clip: data\sample.mp4
    ) else (
        echo [USAGE] run_pipeline.bat ^<path_to_video^> [output_directory]
        exit /b 1
    )
)

echo [*] Launching pipeline on: %VIDEO_INPUT%
echo.

%PYTHON_CMD% scripts\run_pipeline.py %*

echo.
pause

@echo off
setlocal enabledelayedexpansion
set LOG=install_log.txt
echo Install started: %date% %time% > %LOG%

echo [0/13] Verifying wheelhouse integrity (checking for transfer corruption)...
powershell -ExecutionPolicy Bypass -File scripts\verify_hashes.ps1 >> %LOG% 2>&1
if errorlevel 1 (echo Wheelhouse integrity check FAILED — re-transfer the package & goto :fail)

echo [1/13] Checking Python 3.14 specifically (via py launcher)...
py -3.14 --version >> %LOG% 2>&1
if errorlevel 1 (echo Python 3.14 not found via py launcher & goto :fail)

echo [2/13] Checking architecture...
py -3.14 -c "import platform; print(platform.machine())" >> %LOG% 2>&1

echo [3/13] Creating virtual environment...
py -3.14 -m venv venv >> %LOG% 2>&1
if errorlevel 1 goto :fail

echo [4/13] Installing from local wheelhouse (offline, no index)...
venv\Scripts\pip install --no-index --find-links=wheelhouse -r requirements.txt >> %LOG% 2>&1
if errorlevel 1 goto :fail

echo [4b/13] Installing torch/torchvision/ultralytics (not in requirements.txt
echo         by design - see build_wheelhouse.bat comments)...
venv\Scripts\pip install --no-index --find-links=wheelhouse torch torchvision >> %LOG% 2>&1
if errorlevel 1 goto :fail
venv\Scripts\pip install --no-index --find-links=wheelhouse ultralytics --no-deps >> %LOG% 2>&1
if errorlevel 1 goto :fail

echo [5/13] Verifying core imports...
venv\Scripts\python -c "import yaml, dotenv, click, torch, torchvision, ultralytics, cv2, faster_whisper, streamlit, reportlab" >> %LOG% 2>&1
if errorlevel 1 goto :fail

echo [6/13] Checking bundled FFmpeg/FFprobe (tools\ffmpeg\bin, not system PATH)...
if not exist tools\ffmpeg\bin\ffmpeg.exe (echo FFmpeg missing at tools\ffmpeg\bin\ffmpeg.exe - was it included in the transfer package? & goto :fail)
if not exist tools\ffmpeg\bin\ffprobe.exe (echo FFprobe missing at tools\ffmpeg\bin\ffprobe.exe - was it included in the transfer package? & goto :fail)
tools\ffmpeg\bin\ffmpeg.exe -version >> %LOG% 2>&1

echo [7/13] Checking YOLO weights present...
if not exist models\yolo\yolo11n.pt (echo YOLO weights missing at models\yolo\yolo11n.pt - was it included in the transfer package? & goto :fail)

echo [8/13] Checking Whisper model present... (skipped until Phase 6)
echo [9/13] Checking Ollama binary... (skipped until Phase 8)
echo [10/13] Checking local Qwen model pulled... (skipped until Phase 8)
echo [11/13] Checking Piper TTS voices... (skipped until Phase 9)
echo [12/13] Checking Streamlit launches... (skipped until Phase 9)

echo [13/13] Running smoke test (config loader)...
venv\Scripts\python -m src.config >> %LOG% 2>&1
if errorlevel 1 goto :fail

echo.
echo Install completed successfully: %date% %time% >> %LOG%
echo INSTALL OK — see %LOG% for details
goto :eof

:fail
echo.
echo INSTALL FAILED — see %LOG% for details
exit /b 1
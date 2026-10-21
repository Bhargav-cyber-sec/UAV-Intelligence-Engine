@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM  train_visdrone.bat — One-click VisDrone Transfer Learning launcher
REM  Run from inside: week 2\
REM ─────────────────────────────────────────────────────────────────────────────
echo.
echo ============================================================
echo   UAV Intelligence Engine — VisDrone Transfer Learning
echo ============================================================
echo.

REM Detect available Python 3.10
where py >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python launcher (py.exe) not found.
    echo         Install Python 3.10 from https://www.python.org/
    pause & exit /b 1
)

py -3.10 --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python 3.10 not found. Install from https://www.python.org/
    pause & exit /b 1
)

REM Check ultralytics installed
py -3.10 -c "import ultralytics" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [INFO] Installing ultralytics...
    py -3.10 -m pip install ultralytics
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to install ultralytics.
        pause & exit /b 1
    )
)

echo [INFO] Ultralytics OK.
echo.

REM ── GPU or CPU? ──────────────────────────────────────────────────────────────
py -3.10 -c "import torch; print(torch.cuda.is_available())" 2>nul | findstr "True" >nul
if %ERRORLEVEL% equ 0 (
    echo [INFO] GPU detected — using CUDA device 0.
    set DEVICE=0
    set BATCH=8
    set IMGSZ=1280
    set EPOCHS=100
) else (
    echo [INFO] No GPU — falling back to CPU.
    echo [INFO] CPU training is slow. Recommending 30 epochs / 640px for today.
    set DEVICE=cpu
    set BATCH=4
    set IMGSZ=640
    set EPOCHS=30
)

echo.
echo   Device   : %DEVICE%
echo   Epochs   : %EPOCHS%
echo   Img size : %IMGSZ%
echo   Batch    : %BATCH%
echo.
echo [INFO] Starting training... VisDrone dataset will be auto-downloaded if needed.
echo        (approx 1.7 GB download — keep internet connected during first run)
echo.

py -3.10 scripts/train_visdrone.py ^
    --model yolo11n.pt ^
    --epochs %EPOCHS% ^
    --imgsz %IMGSZ% ^
    --batch %BATCH% ^
    --device %DEVICE% ^
    --output yolo11_visdrone.pt ^
    --workers 0

if %ERRORLEVEL% equ 0 (
    echo.
    echo ============================================================
    echo   Training complete!
    echo   Model saved to: models\yolo\yolo11_visdrone.pt
    echo.
    echo   To activate the new model, edit config\paths.yaml:
    echo     yolo_weights: models/yolo/yolo11_visdrone.pt
    echo ============================================================
) else (
    echo.
    echo [ERROR] Training failed. Check the output above for details.
)

echo.
pause

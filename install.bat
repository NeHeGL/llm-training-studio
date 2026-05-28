@echo off
setlocal enabledelayedexpansion
title LLM Training Studio - Installer

echo.
echo  ============================================================
echo   LLM Training Studio Installer - NeHe Productions
echo  ============================================================
echo.

:: Run from the project root
cd /d "%~dp0"

:: -- Check Python ---------------------------------------------
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found. Please install Python 3.10+
    echo          from https://www.python.org/downloads/
    echo          Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo  [OK] %PYVER%

:: -- Create virtual environment --------------------------------
echo.
echo  [0/5] Setting up virtual environment...
if not exist "%~dp0.venv\Scripts\python.exe" (
    python -m venv "%~dp0.venv"
    if errorlevel 1 (
        echo  [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo  [OK] Virtual environment created at .venv\
) else (
    echo  [OK] Virtual environment already exists.
)

:: Use the venv Python from here on
set PYTHON="%~dp0.venv\Scripts\python.exe"
set PIP=%PYTHON% -m pip

:: -- Upgrade pip first ----------------------------------------
echo.
echo  [1/5] Upgrading pip...
%PIP% cache purge >nul 2>&1
%PIP% install --upgrade pip --quiet
if errorlevel 1 (
    echo  [WARN] pip upgrade failed - continuing anyway
)
echo  [OK] pip up to date

:: -- Detect CUDA version via nvidia-smi -----------------------
echo.
echo  [2/5] Detecting GPU / CUDA version...

set CUDA_VER=
set TORCH_INDEX=
set GPU_NAME=

nvidia-smi --query-gpu=name --format=csv,noheader >nul 2>&1
if errorlevel 1 (
    echo  [INFO] No NVIDIA GPU detected - will install CPU-only PyTorch.
    set TORCH_INDEX=https://download.pytorch.org/whl/cpu
    set GPU_TAG=cpu
    goto :install_torch
)

:: GPU detected - get name
for /f "tokens=*" %%g in ('nvidia-smi --query-gpu=name --format=csv,noheader 2^>nul') do (
    set _TMP=%%g
    echo !_TMP! | findstr /i "ERROR" >nul || (set GPU_NAME=%%g & goto :got_gpu)
)
:: Fallback: parse GPU name from nvidia-smi plain text
for /f "tokens=*" %%g in ('nvidia-smi 2^>nul ^| findstr /i "GeForce\|Quadro\|RTX\|GTX\|Tesla\|A100\|H100"') do (
    set GPU_NAME=%%g
    goto :got_gpu
)
set GPU_NAME=NVIDIA GPU (detected)
:got_gpu
echo  [OK] GPU: !GPU_NAME!

:: Get CUDA version from nvidia-smi (e.g. "CUDA Version: 12.8")
for /f "tokens=3" %%c in ('nvidia-smi 2^>nul ^| findstr /i "CUDA Version"') do (
    set CUDA_VER=%%c
    goto :got_cuda
)
:got_cuda

if "!CUDA_VER!"=="" (
    echo  [WARN] Could not read CUDA version - defaulting to cu121
    set TORCH_INDEX=https://download.pytorch.org/whl/cu121
    set GPU_TAG=cu121
    goto :install_torch
)

echo  [OK] CUDA Version: !CUDA_VER!

:: Parse major/minor version
for /f "tokens=1,2 delims=." %%m in ("!CUDA_VER!") do (
    set CUDA_MAJOR=%%m
    set CUDA_MINOR=%%n
)

:: Map CUDA version to PyTorch index URL
set GPU_TAG=cu128
set TORCH_INDEX=https://download.pytorch.org/whl/cu128

if !CUDA_MAJOR! LSS 12 (
    set GPU_TAG=cu118
    set TORCH_INDEX=https://download.pytorch.org/whl/cu118
    goto :install_torch
)
if !CUDA_MINOR! LSS 1 (
    set GPU_TAG=cu118
    set TORCH_INDEX=https://download.pytorch.org/whl/cu118
    goto :install_torch
)
if !CUDA_MINOR! LSS 4 (
    set GPU_TAG=cu121
    set TORCH_INDEX=https://download.pytorch.org/whl/cu121
    goto :install_torch
)
if !CUDA_MINOR! LSS 8 (
    set GPU_TAG=cu124
    set TORCH_INDEX=https://download.pytorch.org/whl/cu124
    goto :install_torch
)

:install_torch
echo.
echo  [3/5] Installing PyTorch (%GPU_TAG%)...
echo        Index URL: !TORCH_INDEX!
echo.
%PIP% install torch torchvision torchaudio --index-url !TORCH_INDEX!
if errorlevel 1 (
    echo.
    echo  [ERROR] PyTorch install failed.
    echo          Check your internet connection and try again.
    pause
    exit /b 1
)
echo.
echo  [OK] PyTorch installed

:: -- Install requirements.txt ---------------------------------
echo.
echo  [4/5] Installing Python dependencies (requirements.txt)...
echo.
%PIP% install -r requirements.txt
if errorlevel 1 (
    echo.
    echo  [ERROR] Some packages failed to install.
    echo          See errors above - you may need to run as Administrator
    echo          or check your internet connection.
    pause
    exit /b 1
)

:: -- Download llama-quantize (needed for q4_k_m GGUF export) --
echo.
echo  [5/5] Installing llama-quantize (needed for q4_k_m GGUF export)...
echo.

set LLAMA_DIR=%~dp0tools\llama
set LLAMA_EXE=%LLAMA_DIR%\llama-quantize.exe

if exist "!LLAMA_EXE!" (
    echo  [OK] llama-quantize already installed at !LLAMA_EXE!
    goto :llama_done
)

if not exist "!LLAMA_DIR!" mkdir "!LLAMA_DIR!"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\install_llama_quantize.ps1"
if errorlevel 1 (
    echo.
    echo  [WARN] Could not auto-install llama-quantize.
    echo         q4_k_m export will fall back to f16 GGUF until it is installed.
    echo         To install manually, download llama.cpp from:
    echo         https://github.com/ggerganov/llama.cpp/releases
    echo         and place llama-quantize.exe in: !LLAMA_DIR!\
    echo.
) else (
    echo  [OK] llama-quantize installed - q4_k_m GGUF export is ready
)

:llama_done
echo.
echo  ============================================================
echo   Installation complete!
echo  ============================================================
echo.
echo   Virtual environment: .venv\
echo   To start LLM Training Studio, run:  launch_web.bat
echo   Or desktop app:                     launch_app.bat
echo.
echo   GPU:      !GPU_NAME!
echo   CUDA:     !CUDA_VER!  (!GPU_TAG!)
echo   PyTorch:  installed from !TORCH_INDEX!
echo.
pause

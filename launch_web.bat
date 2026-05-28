@echo off
setlocal enabledelayedexpansion
title LLM Training Studio

:: ── Re-launch minimized if not already ───────────────────────
if not defined LLM_MINIMIZED (
    set LLM_MINIMIZED=1
    start /min "" "%~f0"
    exit /b
)

echo.
echo  ============================================================
echo   LLM Training Studio - NeHe Productions
echo  ============================================================
echo.

:: Run from the project root so relative paths in server.py work correctly
cd /d "%~dp0"

set PYTHONPYCACHEPREFIX=%~dp0__pycache__

:: ── Pick the right Python (prefer .venv if present) ──────────
if exist "%~dp0.venv\Scripts\python.exe" (
    set PYTHON="%~dp0.venv\Scripts\python.exe"
) else (
    set PYTHON=python
)

:: ── Quick dependency check ───────────────────────────────────
:: Just check that flask is importable — if not, tell the user to run install.bat
%PYTHON% -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo  [WARN] Flask is not installed. Running installer first...
    echo.
    call "%~dp0install.bat"
    if errorlevel 1 (
        echo  [ERROR] Installation failed. Fix the errors above and try again.
        pause
        exit /b 1
    )
)

:: ── Launch the server ────────────────────────────────────────
echo  [OK] Starting LLM Training Studio...
echo       URL: http://localhost:5001
echo.
echo  Press Ctrl+C to stop the server.
echo.

%PYTHON% train/server.py

:: If we get here, the server exited
echo.
echo  [INFO] Server stopped.
pause

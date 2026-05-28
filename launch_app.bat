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
echo   LLM Training Studio - PyQt6 Desktop App
echo  ============================================================
echo.

:: Run from the project root so relative paths in server.py work correctly
cd /d "%~dp0"

set PYTHONPYCACHEPREFIX=%~dp0__pycache__

:: ── Pick the right Python (prefer .venv if present) ──────────
if exist "%~dp0.venv\Scripts\python.exe" (
    set PYTHON="%~dp0.venv\Scripts\python.exe"
    set PYTHONW="%~dp0.venv\Scripts\pythonw.exe"
) else (
    set PYTHON=python
    set PYTHONW=pythonw
)

:: ── Quick dependency check ───────────────────────────────────
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

:: ── Check if PyQt6 is installed ───────────────────────────────
%PYTHON% -c "import PyQt6.QtWebEngineWidgets" >nul 2>&1
if errorlevel 1 (
    echo  [WARN] PyQt6-WebEngine not found. Installing now...
    echo.
    %PYTHON% -m pip install PyQt6 PyQt6-WebEngine
    if errorlevel 1 (
        echo.
        echo  [ERROR] Installation failed. See errors above.
        pause
        exit /b 1
    )
    echo.
    echo  [OK] Installation complete.
    echo.
) else (
    echo  [OK] PyQt6 already installed.
    echo.
)

:: ── Launch the Flask server in the background (this console) ─
echo  [OK] Starting LLM Training Studio...
echo       URL: http://localhost:5001
echo.
echo  Press Ctrl+C to stop the server.
echo.

:: Start the Qt desktop app in background (no console, pythonw)
:: It polls the server and shows the window once it's ready.
start "LLM Training Studio" %PYTHONW% desktop-view\launch_app.py

:: Run the server in the foreground — this console becomes the server log
%PYTHON% train\server.py --no-open

:: If we get here, the server exited
echo.
echo  [INFO] Server stopped.
pause

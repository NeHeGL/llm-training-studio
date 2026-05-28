"""
LLM Training Studio — PyQt6 Desktop App
Author: Jeff Molofee (aka NeHe) — 2026
=========================================
Wraps the existing Flask server (train/server.py) in a native Windows
desktop window using PyQt6 + QtWebEngine (Chromium-based rendering).

The server is launched separately by launch_app.bat as its own console
process (train/server.py --no-open).  This script connects to it.

Requirements:
    pip install PyQt6 PyQt6-WebEngine

Usage:
    python desktop-view/launch_app.py
    — or double-click launch_app.bat (auto-installs PyQt6 if missing)

The existing web UI (launch_web.bat / browser) is completely unaffected.
Both can be used interchangeably — they talk to the same server on port 5001.
"""

import os
import sys
import urllib.request

# ── Resolve project root so imports work regardless of CWD ───────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "train")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if TRAIN_DIR not in sys.path:
    sys.path.insert(0, TRAIN_DIR)

# ── Consolidate __pycache__ at project root (same as launch_app.bat) ─────────
# sys.pycache_prefix redirects bytecode cache for THIS process.
# PYTHONPYCACHEPREFIX env var passes the setting to child processes (pythonw, etc.)
if not sys.pycache_prefix:
    sys.pycache_prefix = os.path.join(ROOT, "__pycache__")
os.environ.setdefault("PYTHONPYCACHEPREFIX", os.path.join(ROOT, "__pycache__"))

PORT = 5001
URL  = f"http://localhost:{PORT}"


# ── Dependency check ──────────────────────────────────────────────────────────
def _check_deps():
    missing = []
    try:
        import PyQt6  # noqa: F401
    except ImportError:
        missing.append("PyQt6")
    try:
        import PyQt6.QtWebEngineWidgets  # noqa: F401
    except ImportError:
        missing.append("PyQt6-WebEngine")

    if missing:
        print("[app] Missing packages: " + ", ".join(missing))
        print("      Run:  pip install PyQt6 PyQt6-WebEngine")
        input("Press Enter to exit...")
        sys.exit(1)


# ── Server helpers ────────────────────────────────────────────────────────────
def _is_server_running():
    """Return True if something is already listening on PORT."""
    try:
        urllib.request.urlopen(URL, timeout=1)
        return True
    except Exception:
        return False


# ── Splash screen ─────────────────────────────────────────────────────────────
SPLASH_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    background:#0d1520; color:#e2e8f0;
    font-family:'Segoe UI',system-ui,sans-serif;
    display:flex; align-items:center; justify-content:center;
    height:100vh; user-select:none;
  }
  .box { display:flex; flex-direction:column; align-items:center; gap:18px; }
  .logo { font-size:3rem; }
  .title { font-size:1.3rem; font-weight:700; color:#7dd3fc; }
  .sub { font-size:0.8rem; color:#475569; }
  .spinner {
    width:34px; height:34px;
    border:3px solid #1e2a3a; border-top-color:#38bdf8;
    border-radius:50%;
    animation:spin 0.9s linear infinite;
  }
  @keyframes spin { to { transform:rotate(360deg); } }
</style>
</head>
<body>
  <div class="box">
    <div class="logo">&#9889;</div>
    <div class="title">LLM Training Studio</div>
    <div class="sub">NeHe Productions</div>
    <div class="spinner"></div>
    <div class="sub">Connecting to server&#8230;</div>
  </div>
</body>
</html>
"""


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    _check_deps()

    from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    from PyQt6.QtWebEngineCore import QWebEngineSettings
    from PyQt6.QtCore import QUrl, Qt, QTimer

    app = QApplication(sys.argv)
    app.setApplicationName("LLM Training Studio")
    app.setOrganizationName("NeHe Productions")

    # ── Splash screen ─────────────────────────────────────────────────────────
    class SplashView(QWebEngineView):
        pass

    splash_view = SplashView()
    splash_view.setFixedSize(480, 280)
    splash_view.setWindowFlags(
        Qt.WindowType.FramelessWindowHint |
        Qt.WindowType.WindowStaysOnTopHint
    )
    splash_view.setHtml(SPLASH_HTML)
    splash_view.show()

    # Center the splash
    screen = app.primaryScreen().geometry()
    splash_view.move(
        (screen.width()  - splash_view.width())  // 2,
        (screen.height() - splash_view.height()) // 2,
    )

    # ── Main window (hidden until server is ready) ────────────────────────────
    class StudioWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("LLM Training Studio")
            self.resize(1440, 900)
            self.setMinimumSize(900, 600)

            self.browser = QWebEngineView()
            settings = self.browser.settings()
            settings.setAttribute(QWebEngineSettings.WebAttribute.LocalStorageEnabled, True)
            settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
            settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptCanOpenWindows, True)

            self.setCentralWidget(self.browser)

        def load_studio(self):
            self.browser.setUrl(QUrl(URL))

    main_win = StudioWindow()

    # ── Poll until server is ready, then show main window ────────────────────
    def _on_ready():
        splash_view.close()
        screen = app.primaryScreen().geometry()
        main_win.move(
            (screen.width()  - main_win.width())  // 2,
            (screen.height() - main_win.height()) // 2,
        )
        main_win.load_studio()
        main_win.show()
        main_win.raise_()
        main_win.activateWindow()

    def _on_timeout():
        splash_view.close()
        QMessageBox.critical(
            None,
            "LLM Training Studio - Error",
            f"Could not connect to the server on port {PORT}.\n\n"
            "Make sure the 'LLM Studio Server' console window is running.\n"
            "If it crashed, check that console window for error details.\n\n"
            "Re-run launch_app.bat to restart both the server and this app."
        )
        app.quit()

    class ServerPoller:
        def __init__(self):
            self._attempts = 0
            self._max = 80          # 80 x 250 ms = 20 s timeout
            self._timer = QTimer()
            self._timer.setInterval(250)
            self._timer.timeout.connect(self._poll)

        def start(self):
            self._timer.start()

        def _poll(self):
            self._attempts += 1
            if _is_server_running():
                self._timer.stop()
                _on_ready()
            elif self._attempts >= self._max:
                self._timer.stop()
                _on_timeout()

    poller = ServerPoller()
    poller.start()

    print(f"[app] Waiting for server on {URL}...")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

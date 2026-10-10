"""PyInstaller entry point for the desktop app (PixelPipeline.exe).

PIXEL_PIPELINE_SMOKE_TEST=<seconds> opens the main window, quits after that long and
exits 0 (1 on error, traceback in smoke-test-error.txt). CI uses it to prove the frozen
app starts, since a windowed exe has no console to print to.
"""
import os
import sys
import traceback

from pixel_pipeline.ui.app import main


def smoke_test(seconds: float) -> int:
    """Open the window, visit every tab, make map tiles and a Godot demo (drawing real
    previews), then quit."""
    try:
        import tempfile
        from pathlib import Path

        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication

        from pixel_pipeline.ui import crash
        from pixel_pipeline.ui.app import MainWindow

        crash.install()
        app = QApplication(sys.argv[:1])
        win = MainWindow()
        win.show()
        for i in range(win.tabs.count()):
            win.tabs.setCurrentIndex(i)
            app.processEvents()
        for tab in (win.map_tab, win.demo_tab):
            tab.sync = True
        win.tabs.setCurrentWidget(win.map_tab)
        win.map_tab.make()
        app.processEvents()
        win.tabs.setCurrentWidget(win.demo_tab)
        win.demo_tab.make(Path(tempfile.mkdtemp()) / "Demo")
        app.processEvents()
        win.resize(900, 600)
        app.processEvents()
        QTimer.singleShot(int(seconds * 1000), app.quit)
        return app.exec()
    except Exception:  # noqa: BLE001 - report instead of PyInstaller's blocking error dialog
        with open("smoke-test-error.txt", "w", encoding="utf-8") as f:
            traceback.print_exc(file=f)
        return 1


if __name__ == "__main__":
    seconds = os.environ.get("PIXEL_PIPELINE_SMOKE_TEST")
    sys.exit(smoke_test(float(seconds)) if seconds else main())

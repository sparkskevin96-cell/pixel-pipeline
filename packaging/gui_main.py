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
    try:
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication

        from pixel_pipeline.ui.app import MainWindow

        app = QApplication(sys.argv[:1])
        win = MainWindow()
        win.show()
        QTimer.singleShot(int(seconds * 1000), app.quit)
        return app.exec()
    except Exception:  # noqa: BLE001 - report instead of PyInstaller's blocking error dialog
        with open("smoke-test-error.txt", "w", encoding="utf-8") as f:
            traceback.print_exc(file=f)
        return 1


if __name__ == "__main__":
    seconds = os.environ.get("PIXEL_PIPELINE_SMOKE_TEST")
    sys.exit(smoke_test(float(seconds)) if seconds else main())

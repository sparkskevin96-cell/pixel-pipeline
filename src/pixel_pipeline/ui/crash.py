"""Crash reports for the desktop app. A windowed .exe has no console, so without this a crash
just closes the window and leaves nothing to go on.

- Python errors (also inside button handlers) show a box with the error, copy the details to
  the clipboard (ready to paste to Claude) and append them to errors.log.
- Hard crashes (inside Qt or a native library) are written to faults.log by faulthandler, and
  the next start says the last run crashed and shows what was written.

Logs: %LOCALAPPDATA%\\PixelPipeline\\logs on Windows, ~/.pixel-pipeline/logs elsewhere.
"""
from __future__ import annotations

import atexit
import datetime
import faulthandler
import os
import platform
import sys
import threading
import traceback
from pathlib import Path

LOG_DIR = (Path(os.environ["LOCALAPPDATA"]) / "PixelPipeline" / "logs" if os.environ.get("LOCALAPPDATA")
           else Path.home() / ".pixel-pipeline" / "logs")
_fault_file = None
_previous: str | None = None          # what the last, crashed run left in faults.log


def _stamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def describe() -> str:
    frozen = "exe" if getattr(sys, "frozen", False) else "source"
    return (f"Pixel Pipeline ({frozen}), Python {platform.python_version()}, "
            f"{platform.system()} {platform.release()} {platform.version()}")


def report(text: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / "errors.log"
    with path.open("a", encoding="utf-8") as f:
        f.write(f"\n--- {_stamp()} {describe()}\n{text}\n")
    return path


def _show(text: str, path: Path):
    """A box with the error; the details go to the clipboard."""
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
        app = QApplication.instance()
        if app is None:
            return
        QApplication.clipboard().setText(f"{describe()}\n{text}")
        box = QMessageBox(QMessageBox.Warning, "Pixel Pipeline hit an error",
                          text.strip().splitlines()[-1][:300])
        box.setInformativeText("The details are copied: paste them to Claude. They are also saved in\n"
                               f"{path}")
        box.setDetailedText(text)
        box.exec()
    except Exception:  # noqa: BLE001 - never let the error box itself crash the app
        pass


def _hook(kind, value, tb):
    text = "".join(traceback.format_exception(kind, value, tb))
    path = report(text)
    sys.__stderr__ and sys.__stderr__.write(text)
    _show(text, path)


def install() -> None:
    """Call first thing in main()."""
    global _fault_file, _previous
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _previous = previous_crash()
    marker = LOG_DIR / "running"
    _fault_file = (LOG_DIR / "faults.log").open("a", encoding="utf-8")
    _fault_file.write(f"\n--- started {_stamp()} {describe()}\n")
    _fault_file.flush()
    faulthandler.enable(_fault_file)
    sys.excepthook = _hook
    threading.excepthook = lambda a: _hook(a.exc_type, a.exc_value, a.exc_traceback)
    marker.write_text(_stamp(), encoding="utf-8")
    atexit.register(lambda: marker.unlink(missing_ok=True))


def previous_crash() -> str | None:
    """If the last run ended without closing normally: the end of faults.log (else None)."""
    marker = LOG_DIR / "running"
    if not marker.exists():
        return None
    try:
        tail = (LOG_DIR / "faults.log").read_text(encoding="utf-8", errors="replace")[-6000:]
    except OSError:
        tail = ""
    runs = tail.split("\n--- started ")
    last = runs[-1] if len(runs) > 1 else tail
    return last.strip() or "(nothing was written before it closed)"


def show_previous_crash(parent=None) -> None:
    """Tell the person the last run crashed, with the details ready to paste."""
    text = _previous
    if text is None:
        return
    from PySide6.QtWidgets import QApplication, QMessageBox
    QApplication.clipboard().setText(f"{describe()}\nLast run crashed:\n{text}")
    box = QMessageBox(QMessageBox.Warning, "Pixel Pipeline closed unexpectedly last time",
                      "The last run crashed. The details are copied: paste them to Claude so it "
                      "can be fixed.", parent=parent)
    box.setDetailedText(text)
    box.exec()

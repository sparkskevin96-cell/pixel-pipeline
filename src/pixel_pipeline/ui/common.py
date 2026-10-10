"""Shared pieces of the desktop app's tool tabs: background jobs, pixel-art previews and the
small sprite options box (type, retro system, size, colours)."""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFormLayout, QLabel, QMessageBox, QSizePolicy, QSpinBox, QWidget,
)

from ..settings import SUBJECTS, Settings, settings_for
from ..systems import SYSTEMS


def checkerboard(rgba: np.ndarray, cell: int = 8) -> np.ndarray:
    """Composite RGBA over a grey checkerboard so transparency is visible."""
    h, w = rgba.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    bg = np.where(((yy // cell + xx // cell) % 2 == 0)[..., None], 205.0, 165.0)
    a = rgba[..., 3:4].astype(np.float32) / 255
    rgb = rgba[..., :3].astype(np.float32) * a + bg * (1 - a)
    return np.concatenate([np.rint(rgb).astype(np.uint8), np.full((h, w, 1), 255, np.uint8)], -1)


def to_pixmap(rgba: np.ndarray) -> QPixmap:
    rgba = np.ascontiguousarray(rgba)
    h, w = rgba.shape[:2]
    return QPixmap.fromImage(QImage(rgba.data, w, h, w * 4, QImage.Format_RGBA8888).copy())


def grid(images: list[np.ndarray], columns: int, gap: int = 2) -> np.ndarray:
    """Same-size RGBA images side by side, wrapping after `columns`, transparent gaps."""
    h = max(i.shape[0] for i in images)
    w = max(i.shape[1] for i in images)
    cols = max(1, min(columns, len(images)))
    rows = math.ceil(len(images) / cols)
    out = np.zeros((rows * h + (rows - 1) * gap, cols * w + (cols - 1) * gap, 4), np.uint8)
    for k, im in enumerate(images):
        r, c = divmod(k, cols)
        y, x = r * (h + gap) + h - im.shape[0], c * (w + gap) + (w - im.shape[1]) // 2
        out[y:y + im.shape[0], x:x + im.shape[1]] = im
    return out


class PixelView(QLabel):
    """Pixel art enlarged by a whole number to fill the space (crisp), or shrunk smoothly when
    it's bigger than the space (maps), on a checkerboard."""

    def __init__(self, placeholder: str = "", minimum: tuple[int, int] = (320, 280)):
        super().__init__(placeholder)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(*minimum)
        # Ignored: the picture never makes the label (and the window) grow, so a resize can't
        # feed back into a bigger picture and another resize.
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setStyleSheet("background:#1c2030;color:#9aa3b5;border-radius:6px;")
        self.setWordWrap(True)
        self.rgba: np.ndarray | None = None

    def show_rgba(self, rgba: np.ndarray | None):
        self.rgba = rgba
        if rgba is None:
            self.clear()
            return
        h, w = rgba.shape[:2]
        aw, ah = max(1, self.width() - 8), max(1, self.height() - 8)
        k = min(aw / w, ah / h)
        if k >= 1:
            k = max(1, min(16, int(k)))
            big = np.repeat(np.repeat(rgba, k, 0), k, 1)
            self.setPixmap(to_pixmap(checkerboard(big, cell=max(4, k * 2))))
        else:
            pix = to_pixmap(checkerboard(rgba, cell=8))
            self.setPixmap(pix.scaled(int(w * k), int(h * k), Qt.KeepAspectRatio,
                                      Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.rgba is not None:
            self.show_rgba(self.rgba)


class _Worker(QObject):
    progress = Signal(str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, job):
        super().__init__()
        self.job = job

    def run(self):
        try:
            self.done.emit(self.job(self.progress.emit))
        except Exception as e:  # noqa: BLE001 - shown to the user
            self.failed.emit(str(e) or type(e).__name__)


class ToolTab(QWidget):
    """A tab whose slow work (rendering, AI) runs off the UI thread."""
    title = "pixel-pipeline"

    def __init__(self, status=None):
        super().__init__()
        self.sync = False                 # tests: run jobs right here and let errors raise
        self._status = status or (lambda text: None)
        self.busy_widgets: list[QWidget] = []
        self._thread = self._worker = None
        self._on_done = None

    def say(self, text: str):
        self._status(text)

    def run(self, job, on_done, what: str = "Working…"):
        """job(say) -> result off the UI thread, then on_done(result) back on it."""
        if self._thread is not None and self._thread.isRunning():
            self.say("Still working on the last one…")
            return
        self.say(what)
        if self.sync:
            on_done(job(self.say))
            return
        self._busy(True)
        self._on_done = on_done
        thread, worker = QThread(self), _Worker(job)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self.say)
        worker.done.connect(self._job_done)
        worker.failed.connect(self._job_failed)
        for sig in (worker.done, worker.failed):
            sig.connect(thread.quit)
        thread.finished.connect(lambda: self._busy(False))
        thread.finished.connect(worker.deleteLater)
        self._thread, self._worker = thread, worker
        thread.start()

    @Slot(object)
    def _job_done(self, result):
        self._on_done(result)

    @Slot(str)
    def _job_failed(self, message: str):
        self.say("Stopped: " + message.splitlines()[0])
        QMessageBox.warning(self, self.title, message)

    def _busy(self, busy: bool):
        for w in self.busy_widgets:
            w.setEnabled(not busy)
        if busy:
            QApplication.setOverrideCursor(Qt.BusyCursor)
        else:
            QApplication.restoreOverrideCursor()

    def warn(self, message: str):
        if self.sync:
            raise RuntimeError(message)
        QMessageBox.warning(self, self.title, message)


def spin(lo: int, hi: int, value: int, special: str | None = None) -> QSpinBox:
    box = QSpinBox()
    box.setRange(lo, hi)
    box.setValue(value)
    if special:
        box.setSpecialValueText(special)
    return box


class SpriteOptions(QWidget):
    """Type, retro system, size and colours -> Settings (the rest from the type's preset)."""

    def __init__(self, subject: str = "person", width: int = 48, height: int = 48,
                 colors: int = 24, subjects=SUBJECTS):
        super().__init__()
        self.subject = QComboBox()
        self.subject.addItems(list(subjects))
        self.subject.setCurrentText(subject)
        self.system = QComboBox()
        self.system.addItem("Off (free colours)", "off")
        for key, sysm in SYSTEMS.items():
            self.system.addItem(sysm.name, key)
        self.width_ = spin(4, 512, width)
        self.height_ = spin(4, 512, height)
        self.colors = spin(2, 64, colors)
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow("Type", self.subject)
        form.addRow("Retro system", self.system)
        form.addRow("Width", self.width_)
        form.addRow("Height", self.height_)
        form.addRow("Colours", self.colors)
        self.system.currentIndexChanged.connect(self._system_changed)

    def _system_changed(self, *_):
        free = (self.system.currentData() or "off") == "off"
        for w in (self.width_, self.height_, self.colors):
            w.setEnabled(free)
        tip = "" if free else "Set by the retro system (readable look)"
        for w in (self.width_, self.height_, self.colors):
            w.setToolTip(tip)

    def settings(self, **extra) -> Settings:
        system = self.system.currentData() or "off"
        if system != "off":
            return settings_for(self.subject.currentText(), system=system, **extra)
        return settings_for(self.subject.currentText(), width=self.width_.value(),
                            height=self.height_.value(), colors=self.colors.value(), **extra)

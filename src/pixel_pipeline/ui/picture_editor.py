"""Clean up a cut-out picture before it becomes a 3D model: rub out what isn't the subject
(a leash, a hand, a bit of couch) or bring back what the cut-out lost (a pale leg).

PictureEditor(rgba, original_rgb) is a dialog; .result() is the edited RGBA.
"""
from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QRadioButton, QSlider, QVBoxLayout,
)

from .common import checkerboard, to_pixmap

VIEW = 720


def brush(alpha: np.ndarray, original: np.ndarray, x0: float, y0: float, x1: float, y1: float,
          radius: float, restore: bool = False) -> None:
    """Erase (alpha 0) or restore (back to `original` alpha) along a stroke, in place."""
    mask = np.zeros(alpha.shape, np.uint8)
    cv2.line(mask, (round(x0), round(y0)), (round(x1), round(y1)), 1,
             max(1, round(2 * radius)), cv2.LINE_8)
    cv2.circle(mask, (round(x1), round(y1)), max(1, round(radius)), 1, -1)
    hit = mask > 0
    alpha[hit] = original[hit] if restore else 0


class _Canvas(QLabel):
    def __init__(self, editor: "PictureEditor"):
        super().__init__()
        self.editor = editor
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self._last = None

    def _to_image(self, pos: QPoint):
        e = self.editor
        pix = self.pixmap()
        if pix is None or pix.isNull():
            return None
        ox = (self.width() - pix.width()) / 2
        oy = (self.height() - pix.height()) / 2
        return (pos.x() - ox) / e.zoom, (pos.y() - oy) / e.zoom

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self.editor.begin_stroke()
            self._last = self._to_image(ev.position().toPoint())
            if self._last:
                self.editor.stroke(*self._last, *self._last)

    def mouseMoveEvent(self, ev):
        if self._last is not None and ev.buttons() & Qt.LeftButton:
            p = self._to_image(ev.position().toPoint())
            if p:
                self.editor.stroke(*self._last, *p)
                self._last = p

    def mouseReleaseEvent(self, ev):
        self._last = None


class PictureEditor(QDialog):
    """Rub out or bring back parts of the cut-out. Ctrl+Z undoes a stroke."""


    def __init__(self, rgba: np.ndarray, restore_alpha: np.ndarray | None = None, parent=None,
                 title: str = "Clean up the picture"):
        """rgba: the cut-out with the photo's colours everywhere (alpha = the cut-out).
        restore_alpha: what "Bring back" returns to (default: the cut-out as it came)."""
        super().__init__(parent)
        self.setWindowTitle(title)
        self._full = rgba.copy()
        full_restore = (restore_alpha if restore_alpha is not None else rgba[..., 3]).copy()
        # edit around the subject (plus a margin to bring lost parts back), not the whole photo
        ys, xs = np.nonzero(rgba[..., 3] >= 128)
        H, W = rgba.shape[:2]
        if len(xs):
            m = int(0.15 * max(xs.max() - xs.min(), ys.max() - ys.min())) + 8
            x0, x1 = max(0, xs.min() - m), min(W, xs.max() + m + 1)
            y0, y1 = max(0, ys.min() - m), min(H, ys.max() + m + 1)
        else:
            x0, y0, x1, y1 = 0, 0, W, H
        self.rgba = self._full[y0:y1, x0:x1]               # a view: edits land in _full
        self.ghost = self.rgba[..., 3].copy()             # the cut-out as it came
        self.original = full_restore[y0:y1, x0:x1]
        self.undo: list[np.ndarray] = []
        h, w = self.rgba.shape[:2]
        self.zoom = min(VIEW / max(h, w), 4.0)
        self.canvas = _Canvas(self)
        self.canvas.setMinimumSize(round(w * self.zoom), round(h * self.zoom))
        self.erase = QRadioButton("Rub out")
        self.erase.setChecked(True)
        self.restore = QRadioButton("Bring back")
        self.size = QSlider(Qt.Horizontal)
        self.size.setRange(2, 120)
        self.size.setValue(20)
        tools = QHBoxLayout()
        for wdg in (self.erase, self.restore, QLabel("Brush"), self.size):
            tools.addWidget(wdg)
        tools.addStretch(1)
        note = QLabel("Rub out anything that isn't the subject (a leash, a hand, the floor). "
                      "Bring back parts the cut-out lost. Ctrl+Z undoes.")
        note.setWordWrap(True)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Build the model")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(note)
        lay.addLayout(tools)
        lay.addWidget(self.canvas, 1)
        lay.addWidget(buttons)
        self.redraw()

    def radius(self) -> float:
        return self.size.value() / self.zoom / 2

    def begin_stroke(self):
        self.undo.append(self.rgba[..., 3].copy())
        del self.undo[:-30]

    def stroke(self, x0, y0, x1, y1):
        alpha = self.rgba[..., 3].copy()
        brush(alpha, self.original, x0, y0, x1, y1, self.radius(), self.restore.isChecked())
        self.rgba[..., 3] = alpha
        self.redraw()

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Z and ev.modifiers() & Qt.ControlModifier and self.undo:
            self.rgba[..., 3] = self.undo.pop()
            self.redraw()
            return
        super().keyPressEvent(ev)

    def redraw(self):
        h, w = self.rgba.shape[:2]
        view = cv2.resize(self.rgba, (max(1, round(w * self.zoom)), max(1, round(h * self.zoom))),
                          interpolation=cv2.INTER_AREA if self.zoom < 1 else cv2.INTER_NEAREST)
        # what's rubbed out shows faintly, so you can see what you're doing
        ghost = cv2.resize(self.ghost, (view.shape[1], view.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
        shown = view.copy()
        gone = (view[..., 3] < 128) & (ghost >= 128)
        shown[gone, 3] = 60
        self.canvas.setPixmap(to_pixmap(checkerboard(shown, cell=10)))

    def result(self) -> np.ndarray:
        """The whole picture, edited."""
        return self._full

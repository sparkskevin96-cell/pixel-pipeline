"""The front drawing of a new drawn character, for the user's OK before the GPU draws the rest.

Shows the key drawing big (crisp), its idle bent from that drawing (playing), the picture it was
made from (small) and how long the rest takes. Buttons: Looks good (draw the other directions
paired to this drawing + bend everything: library.approve_preview), Redraw this side (a new
seed: library.redraw_preview), Change words… (new words / house-style words, optionally a new
model too), Not now (it stays in the library as a draft: right-click it later), Throw it away.
Earlier tries are small thumbnails: clicking one uses it again (library.pick_preview, no GPU).

The host page gives `run(job, on_done)` (its own thread runner) so the work shows in its status
bar like everything else; in tests the hosts run jobs synchronously.
"""
from __future__ import annotations

import threading

import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

from .common import PixelView, checkerboard, to_pixmap


def _load(path) -> np.ndarray | None:
    try:
        from PIL import Image
        return np.asarray(Image.open(path).convert("RGBA")).copy()
    except Exception:  # noqa: BLE001 - a missing file just shows nothing
        return None


def _crop(rgba: np.ndarray, pad: int = 2) -> np.ndarray:
    ys, xs = np.nonzero(rgba[..., 3] > 0)
    if not len(ys):
        return rgba
    y0, y1 = max(0, ys.min() - pad), min(rgba.shape[0], ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(rgba.shape[1], xs.max() + 1 + pad)
    return rgba[y0:y1, x0:x1]


def _crop_all(frames: list[np.ndarray], pad: int = 2) -> list[np.ndarray]:
    """Frames cut to the box all of them fit in (the animation doesn't jump)."""
    if not frames:
        return frames
    on = np.any([f[..., 3] > 0 for f in frames], 0)
    ys, xs = np.nonzero(on)
    if not len(ys):
        return frames
    y0, y1 = max(0, ys.min() - pad), min(on.shape[0], ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(on.shape[1], xs.max() + 1 + pad)
    return [f[y0:y1, x0:x1] for f in frames]


def _icon(rgba: np.ndarray | None, px: int = 64) -> QIcon:
    if rgba is None:
        return QIcon()
    rgba = _crop(rgba)
    k = max(1, px // max(rgba.shape[:2]))
    big = np.repeat(np.repeat(rgba, k, 0), k, 1)
    return QIcon(to_pixmap(checkerboard(big, cell=max(4, k * 2))).scaled(
        px, px, Qt.KeepAspectRatio, Qt.FastTransformation))


class WordsDialog(QDialog):
    """Change the character's words (and the project's house-style words) for a new front
    drawing; "New model too" makes a new picture and 3D shape from the words as well."""

    def __init__(self, words: str, style_words: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Change words")
        self.words = QLineEdit(words)
        self.words.setMinimumWidth(420)
        self.style_words = QLineEdit(style_words)
        self.style_words.setToolTip("The project's house style, added to everything you make in "
                                    "this project (changing it here changes it for all of them)")
        self.new_model = QCheckBox("New model too (a new picture and 3D shape from the words)")
        self.new_model.setToolTip("When the shape itself is the problem (wrong pose, a lump for a "
                                  "sword): ~2.5 min more GPU. Off: the same 3D model is drawn again "
                                  "with the new words.")
        form = QFormLayout(self)
        form.addRow("Words", self.words)
        form.addRow("House style (whole project)", self.style_words)
        form.addRow(self.new_model)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Redraw the front")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)


class _Failed:
    def __init__(self, error: Exception):
        self.error = error


class CharacterReview(QWidget):
    """Review panel for a draft character (library.in_preview)."""

    approved = Signal(object)        # the finished Character (every direction drawn)
    changed = Signal(object)         # the draft changed (redrawn / an older try picked)
    closed = Signal(object)          # kept as a draft (the Character) or thrown away (None)

    def __init__(self, run, say=None, parent=None):
        super().__init__(parent)
        self._run = run
        self._say = say or (lambda text: None)
        self.sync = False                # tests: no message boxes
        self.last_error = ""
        self.character = None
        self.info: dict | None = None
        self.frames: list[np.ndarray] = []
        self._i = 0
        self._cancel: threading.Event | None = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._next_frame)
        self._build()

    # ------------------------------------------------------------------ layout
    def _build(self):
        self.title = QLabel("")
        self.title.setStyleSheet("font-size:15px;font-weight:600;")
        self.title.setWordWrap(True)
        self.key_view = PixelView("The front drawing", minimum=(300, 300))
        self.idle_view = PixelView("Idle", minimum=(180, 180))
        self.picture_view = PixelView("Picture", minimum=(120, 120))
        self.estimate = QLabel("")
        self.estimate.setWordWrap(True)
        self.estimate.setStyleSheet("color:#c9d1e0;")
        self.words = QLabel("")
        self.words.setWordWrap(True)
        self.words.setStyleSheet("color:#9aa3b5;")

        key_box = QGroupBox("Front drawing (as the game will show it)")
        QVBoxLayout(key_box).addWidget(self.key_view)
        idle_box = QGroupBox("Idle, bent from it")
        il = QVBoxLayout(idle_box)
        il.addWidget(self.idle_view, 1)
        pic_box = QGroupBox("Made from")
        pl = QVBoxLayout(pic_box)
        pl.addWidget(self.picture_view, 1)
        pl.addWidget(self.words)
        side = QVBoxLayout()
        self.picture_view.setMaximumHeight(170)           # the picture is only a reminder
        side.addWidget(idle_box, 4)
        side.addWidget(pic_box, 1)
        row = QHBoxLayout()
        row.addWidget(key_box, 3)
        row.addLayout(side, 2)

        self.tries = QListWidget()
        self.tries.setViewMode(QListWidget.IconMode)
        self.tries.setIconSize(QSize(56, 56))
        self.tries.setGridSize(QSize(76, 84))
        self.tries.setMovement(QListWidget.Static)
        self.tries.setFixedHeight(96)
        self.tries.setStyleSheet("QListWidget{background:#1c2030;color:#c9d1e0;border-radius:6px;}")
        self.tries.itemClicked.connect(self._try_clicked)
        self.tries_box = QGroupBox("Earlier tries (click one to use it again)")
        QVBoxLayout(self.tries_box).addWidget(self.tries)

        self.approve_btn = QPushButton("Looks good — make all 16 directions")
        self.approve_btn.setStyleSheet("font-weight:600;font-size:14px;padding:6px 14px;"
                                       "background:#2f7d4a;color:white;border-radius:4px;")
        self.approve_btn.clicked.connect(lambda: self.approve())
        self.redraw_btn = QPushButton("Redraw this side")
        self.redraw_btn.setToolTip("A new seed for this drawing (the earlier ones stay below to "
                                   "pick from); ~30 s of GPU")
        self.redraw_btn.clicked.connect(lambda: self.redraw())
        self.words_btn = QPushButton("Change words…")
        self.words_btn.clicked.connect(self._words_clicked)
        self.later_btn = QPushButton("Not now (keep as a draft)")
        self.later_btn.setToolTip("It stays in your Character Library marked 'waiting for your OK'"
                                  ": right-click it -> Review the front drawing… to carry on")
        self.later_btn.clicked.connect(lambda: self.keep_draft())
        self.discard_btn = QPushButton("Throw it away")
        self.discard_btn.setFlat(True)
        self.discard_btn.setStyleSheet("color:#d07070;")
        self.discard_btn.clicked.connect(lambda: self.discard())
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setToolTip("Stop after the current drawing (what's drawn is kept: Looks good "
                                 "again carries on)")
        self.stop_btn.clicked.connect(self.stop)
        self.stop_btn.setVisible(False)
        buttons = QHBoxLayout()
        for b in (self.approve_btn, self.redraw_btn, self.words_btn, self.later_btn,
                  self.stop_btn):
            buttons.addWidget(b)
        buttons.addStretch(1)
        buttons.addWidget(self.discard_btn)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.title)
        lay.addLayout(row, 1)
        lay.addWidget(self.estimate)
        lay.addWidget(self.tries_box)
        lay.addLayout(buttons)
        self.action_buttons = [self.approve_btn, self.redraw_btn, self.words_btn, self.later_btn,
                               self.discard_btn, self.tries]

    # ------------------------------------------------------------------ showing
    def set_character(self, ch):
        """Show a draft (or refresh it after a change)."""
        from ..library import preview_info
        self.character = ch
        self.info = info = preview_info(ch) if ch is not None else None
        self._timer.stop()
        if info is None:
            self.title.setText("Nothing waiting for your OK.")
            self.key_view.show_rgba(None)
            self.idle_view.show_rgba(None)
            self.frames = []
            return
        d = info["direction"]
        side = "front" if d == "S" else "side"
        self.title.setText(f"{ch.name}: here's the {side} ({d}) — happy with it? "
                           "Nothing else is drawn until you say so.")
        key = _load(info["key"])
        self.key_view.show_rgba(_crop(key) if key is not None else None)   # big: no empty frame
        self.frames = _crop_all([f for f in (_load(p) for p in info["frames"]) if f is not None])
        self._i = 0
        self._next_frame()
        if len(self.frames) > 1:
            self._timer.start(max(40, info["frame_ms"]))
        pic = _load(info["picture"]) if info["picture"] is not None else None
        self.picture_view.show_rgba(pic)
        if pic is None:
            self.picture_view.setText("(no picture)")
        self.words.setText(f"“{info['words']}”" if info["words"] else "")
        from ..metrics import made_in
        so_far = made_in(ch.folder)
        self.estimate.setText(f"Looks good: {info['estimate']} (Stop keeps what's drawn)."
                              + (f"  So far: {so_far.replace('made in ', '')}." if so_far else ""))
        n = 16
        try:
            from ..library import _keys_meta
            n = int(((_keys_meta(ch) or {}).get("camera") or {}).get("directions") or 16)
        except (TypeError, ValueError):
            pass
        self.directions = n
        self.approve_btn.setText(f"Looks good — make all {n} directions")
        self.tries.clear()
        for a in info["attempts"]:
            item = QListWidgetItem(_icon(_load(a["key"])), f"#{a['n']}" + (" ✓" if a["current"]
                                                                           else ""))
            item.setData(Qt.UserRole, int(a["n"]))
            item.setToolTip(f"Try {a['n']}: seed {a['seed']}, “{a.get('words', '')}”"
                            + (" (showing)" if a["current"] else ""))
            self.tries.addItem(item)
        self.tries_box.setVisible(len(info["attempts"]) > 1)

    def _next_frame(self):
        if self.frames:
            self.idle_view.show_rgba(self.frames[self._i % len(self.frames)])
            self._i += 1

    def _busy(self, busy: bool, cancellable: bool = False):
        for w in self.action_buttons:
            w.setEnabled(not busy)
        self.stop_btn.setVisible(busy and cancellable)
        if not busy:
            self._cancel = None

    def _go(self, job, on_done, cancellable: bool = False, **progress):
        """job(say) on the host's runner; errors come back here (the buttons work again, the
        message is shown; Stop isn't an error)."""
        self._busy(True, cancellable)

        def safe(say):
            try:
                return job(say)
            except Exception as e:  # noqa: BLE001 - shown to the user
                return _Failed(e)

        def done(result):
            self._busy(False)
            if isinstance(result, _Failed):
                from ..drawnrig import Cancelled
                from ..progress import Stopped
                text = str(result.error) or type(result.error).__name__
                self.last_error = text
                self._say(text.splitlines()[0] if text.startswith("Stopped") else
                          "Stopped: " + text.splitlines()[0])
                if not self.sync and not isinstance(result.error, (Cancelled, Stopped)):
                    QMessageBox.warning(self, "Couldn't draw it", text)
                if self.character is not None:
                    self.set_character(self.character)
                return
            on_done(result)
        if progress:                     # the host's runner shows the bars (plan / title)
            self._run(safe, done, **progress)
        else:
            self._run(safe, done)

    def _new_cancel(self):
        self._cancel = threading.Event()
        return self._cancel.is_set

    def stop(self):
        if self._cancel is not None:
            self._cancel.set()
            self._say("Stopping after this drawing… (what's drawn is kept: Looks good carries on)")

    # ------------------------------------------------------------------ actions
    def approve(self):
        """Draw the other directions paired to this drawing, then bend every animation."""
        from ..library import approve_preview
        ch = self.character
        if ch is None:
            return
        cancel = self._new_cancel()
        self._timer.stop()

        def job(say):
            return approve_preview(ch, progress=say, cancel=cancel)
        from ..progress import character_plan
        n = int(getattr(self, "directions", 16) or 16)
        self._go(job, self._approved, cancellable=True,
                 plan=character_plan(model=False, drawn=True, approve=True, directions=n),
                 title=f"Drawing {ch.name}: every direction")

    def _approved(self, ch):
        self.character = None
        self.approved.emit(ch)

    def redraw(self, seed: int | None = None):
        """A new seed for this side only."""
        from ..library import redraw_preview
        ch = self.character
        if ch is None:
            return

        def job(say):
            return redraw_preview(ch, seed=seed, progress=say)
        from ..progress import character_plan
        self._go(job, self._changed, plan=character_plan(model=False, drawn=True, preview=True),
                 title=f"Redrawing {ch.name}'s front")

    def change_words(self, words: str, new_model: bool = False, style_words: str | None = None):
        """Redraw the front with new words (the same 3D model unless new_model). style_words:
        new house-style words for the whole project."""
        from ..library import redraw_preview
        ch = self.character
        if ch is None:
            return
        if style_words is not None:
            self._set_style_words(style_words)

        def job(say):
            return redraw_preview(ch, words=words, new_model=new_model, progress=say)
        from ..progress import character_plan
        self._go(job, self._changed,
                 plan=character_plan(model=new_model, drawn=True, preview=True),
                 title=f"Redrawing {ch.name}'s front")

    def _set_style_words(self, words: str):
        import dataclasses
        import os
        if os.environ.get("PIXEL_LIBRARY"):          # a library outside the projects: no style
            return
        from ..assets import current_project
        proj = current_project()
        if proj.style.style_prompt != words.strip():
            proj.save_style(dataclasses.replace(proj.style, style_prompt=words.strip()))

    def _words_clicked(self, _checked=False):
        if self.character is None:
            return
        style_words = ""
        try:
            from ..library import style
            style_words = style().style_prompt
        except Exception:  # noqa: BLE001 - no project: no house style words
            pass
        dlg = WordsDialog(self.character.prompt, style_words, self)
        try:
            ok = dlg.exec()
            words, new_style, new_model = (dlg.words.text().strip(), dlg.style_words.text(),
                                           dlg.new_model.isChecked())
        finally:
            dlg.setParent(None)
        if ok and words:
            self.change_words(words, new_model,
                              new_style if new_style.strip() != style_words.strip() else None)

    def pick(self, n: int):
        """Use try n again (no GPU)."""
        from ..library import pick_preview
        ch = self.character
        if ch is None:
            return
        self._changed(pick_preview(ch, n))

    def _try_clicked(self, item):
        n = item.data(Qt.UserRole)
        if self.info is not None and n != self.info.get("current"):
            self.pick(n)

    def _changed(self, ch):
        self.set_character(ch)
        self.changed.emit(ch)

    def keep_draft(self):
        """Leave it in the library as a draft (waiting for your OK)."""
        ch = self.character
        self._timer.stop()
        if ch is not None:
            self._say(f"{ch.name} is kept as a draft (waiting for your OK): right-click it in "
                      "the Character Library -> Review the front drawing… to carry on.")
        self.closed.emit(ch)

    def discard(self, ask: bool = True):
        """Remove the draft from the library (the 3D model and drawings go too)."""
        from ..library import remove
        ch = self.character
        if ch is None:
            return
        if ask and QMessageBox.question(
                self, "Throw it away?", f"Remove {ch.name} from the library (its model and "
                "drawings too)?") != QMessageBox.Yes:
            return
        remove(ch)
        self._timer.stop()
        self.character = None
        self._say(f"{ch.name} thrown away.")
        self.closed.emit(None)

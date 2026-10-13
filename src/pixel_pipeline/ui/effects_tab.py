"""The Effects tab: pick an effect (each one plays in the list), set its colours, size and
frames, watch it big, press Make -> an effect asset in the current project (effects.py)."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QPushButton, QVBoxLayout,
)

from .common import PixelView, ToolTab, spin, to_pixmap

ICON = 64


def _icon(rgba: np.ndarray, px: int = ICON) -> QIcon:
    """A frame enlarged crisp (whole number) on the dark preview background."""
    from PIL import Image
    im = Image.fromarray(rgba, "RGBA")
    k = px / max(im.size)
    k = int(k) if k >= 1 else k
    im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.NEAREST)
    canvas = Image.new("RGBA", (px, px), (28, 32, 46, 255))
    canvas.alpha_composite(im, ((px - im.width) // 2, (px - im.height) // 2))
    return QIcon(to_pixmap(np.asarray(canvas)))


def _list(icon: int, grid: tuple[int, int]) -> QListWidget:
    lst = QListWidget()
    lst.setViewMode(QListWidget.IconMode)
    lst.setIconSize(QSize(icon, icon))
    lst.setGridSize(QSize(*grid))
    lst.setResizeMode(QListWidget.Adjust)
    lst.setMovement(QListWidget.Static)
    lst.setWordWrap(True)
    return lst


class EffectsTab(ToolTab):
    """Procedural animated effects (no GPU) -> sheet + .json + .aseprite + Godot .tres."""
    title = "Effects"
    changed = Signal()            # an effect asset was made

    def __init__(self, status=None):
        super().__init__(status)
        from ..effects import EFFECT_KINDS, SPECS, THEMES
        self.fx = None                     # the effect shown big
        self.tick_n = 0
        self.kind_fx: dict = {}            # kind -> Effect for the list icons
        self.kinds = _list(ICON, (96, 100))
        for kind in EFFECT_KINDS:
            item = QListWidgetItem(SPECS[kind][5])
            item.setData(Qt.UserRole, kind)
            self.kinds.addItem(item)
        self.kinds.currentItemChanged.connect(self._kind_changed)

        self.theme = QComboBox()
        self.theme.addItem("Effect's own", None)
        for t in THEMES:
            self.theme.addItem(t, t)
        self.colours = QLineEdit()
        self.colours.setPlaceholderText("or your colours: #ff8800  or  #331100 #ff8800 #ffffcc")
        self.size_ = spin(0, 512, 0, "Project scale")
        self.frames = spin(0, 32, 0, "Effect's own")
        self.loop = QComboBox()
        for label, v in (("Effect's own", None), ("Loop", True), ("Once", False)):
            self.loop.addItem(label, v)
        self.seed = spin(0, 9999, 0)
        self.dither = QCheckBox("Dithered edges")
        self.dither.setChecked(True)
        self.name = QLineEdit()
        self.name.setPlaceholderText("Name (optional)")
        for w in (self.theme, self.loop):
            w.currentIndexChanged.connect(self.preview)
        for w in (self.size_, self.frames, self.seed):
            w.valueChanged.connect(self.preview)
        self.dither.toggled.connect(self.preview)
        self.colours.editingFinished.connect(self.preview)
        self.make_btn = QPushButton("Make effect")
        self.make_btn.setStyleSheet("font-weight:600;")
        self.make_btn.clicked.connect(self.make)
        self.busy_widgets = [self.make_btn]

        opts = QGroupBox("Effect")
        form = QFormLayout(opts)
        form.addRow("Colours", self.theme)
        form.addRow("", self.colours)
        form.addRow("Size (px)", self.size_)
        form.addRow("Frames", self.frames)
        form.addRow("Plays", self.loop)
        form.addRow("Variation", self.seed)
        form.addRow(self.dither)
        form.addRow(self.name)
        form.addRow(self.make_btn)
        self.info = QLabel()
        self.info.setWordWrap(True)
        form.addRow(self.info)

        self.view = PixelView("Pick an effect", (260, 220))
        self.saved = _list(64, (92, 100))
        self.saved.currentItemChanged.connect(self._saved_changed)

        left = QVBoxLayout()
        left.addWidget(QLabel("Effects"))
        left.addWidget(self.kinds, 1)
        mid = QVBoxLayout()
        mid.addWidget(self.view, 1)
        mid.addWidget(QLabel("Saved effects (this project)"))
        mid.addWidget(self.saved, 1)
        lay = QHBoxLayout(self)
        lay.addLayout(left, 2)
        lay.addLayout(mid, 3)
        lay.addWidget(opts, 2)

        self._render_icons()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(90)
        self.kinds.setCurrentRow(0)
        self.refresh()

    # -- settings
    def _style(self):
        from ..assets import current_project
        return current_project().style

    def current_kind(self) -> str:
        item = self.kinds.currentItem()
        return item.data(Qt.UserRole) if item else "slash"

    def select_kind(self, kind: str):
        for i in range(self.kinds.count()):
            if self.kinds.item(i).data(Qt.UserRole) == kind:
                self.kinds.setCurrentRow(i)
                return
        raise ValueError(kind)

    def options(self) -> dict:
        text = self.colours.text().strip()
        return {"theme": self.theme.currentData(), "colours": text or None,
                "size": self.size_.value() or None, "frames": self.frames.value() or None,
                "loop": self.loop.currentData(), "seed": self.seed.value(),
                "dither": 0.5 if self.dither.isChecked() else 0.0}

    # -- previews
    def _render_icons(self):
        from ..effects import render_effect
        st = self._style()
        for i in range(self.kinds.count()):
            kind = self.kinds.item(i).data(Qt.UserRole)
            self.kind_fx[kind] = render_effect(kind, style=st)
            self.kinds.item(i).setIcon(_icon(self.kind_fx[kind].frames[0]))

    def _kind_changed(self, *_):
        self.preview()

    def preview(self, *_):
        from ..effects import render_effect
        try:
            self.fx = render_effect(self.current_kind(), style=self._style(), **self.options())
        except ValueError as e:
            self.info.setText(str(e))
            return
        w, h = self.fx.size
        self.info.setText(f"{w}x{h}, {len(self.fx.frames)} frames x {self.fx.frame_ms} ms, "
                          f"{'loops' if self.fx.loop else 'plays once'}, {len(self.fx.palette)} colours")
        self.view.show_rgba(self.fx.frames[0])

    def tick(self):
        """Next frame of every animation (the timer calls this; tests may too)."""
        self.tick_n += 1
        for i in range(self.kinds.count()):
            fx = self.kind_fx.get(self.kinds.item(i).data(Qt.UserRole))
            if fx:
                frames = fx.frames + ([] if fx.loop else [fx.frames[-1]] * 3)
                self.kinds.item(i).setIcon(_icon(frames[self.tick_n % len(frames)]))
        if self.fx is not None:
            frames = self.fx.frames + ([] if self.fx.loop else [np.zeros_like(self.fx.frames[0])] * 4)
            self.view.show_rgba(frames[self.tick_n % len(frames)])

    # -- make + saved
    def make(self):
        from ..effects import make_effect
        kind, opts, name = self.current_kind(), self.options(), self.name.text().strip() or None
        self.run(lambda say: make_effect(kind, name, **opts), self._made, "Making the effect…")

    def _made(self, asset):
        self.last = asset
        self.say(f"Saved {asset.name} -> {asset.folder}")
        self.refresh()
        self.changed.emit()

    def refresh(self):
        """Saved effects of the current project, newest first."""
        from ..assets import list_assets
        self.saved.clear()
        for a in list_assets("effect"):
            item = QListWidgetItem(a.name)
            if a.thumb.exists():
                item.setIcon(QIcon(QPixmap(str(a.thumb))))
            item.setData(Qt.UserRole, a)
            item.setToolTip(str(a.folder))
            self.saved.addItem(item)

    def _saved_changed(self, item, _prev=None):
        if item is None:
            return
        a = item.data(Qt.UserRole)
        e = a.extra
        self.info.setText(f"{a.name}: {e.get('effect')} {e.get('size')}, {e.get('frames')} frames, "
                          f"{a.folder}")


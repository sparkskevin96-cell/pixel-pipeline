"""The Effects tab: pick an effect (each one plays in the list), set its colours, size and
frames, watch it big at its real frame timing on a dark or light ground with or without
glow, press Make -> an effect asset in the current project (effects.py / spellfx.py)."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QPushButton, QVBoxLayout,
)

from ..fxcore import composite
from .common import PixelView, ToolTab, spin, to_pixmap

ICON = 64
ICON_SIZE = 44                 # premium effects render their list icons this big (fast)
GROUNDS = {"dark": (24, 22, 36), "light": (226, 222, 210)}
ANGLES = (("Hit to the right", 0.0), ("Hit to the left", 3.1416), ("Hit upward", -1.5708),
          ("Hit downward", 1.5708), ("Up-right", -0.7854), ("Down-right", 0.7854))


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
    """Procedural animated effects (no GPU) -> sheet + glow + .json + .aseprite + Godot .tres."""
    title = "Effects"
    changed = Signal()            # an effect asset was made

    def __init__(self, status=None):
        super().__init__(status)
        from ..effects import EFFECT_KINDS, THEMES, effect_label
        from ..spellfx import SPELL_THEMES
        self.fx = None                     # the effect shown big
        self.tick_n = 0
        self.frame_i = 0                   # frame of the big preview
        self.kind_fx: dict = {}            # kind -> Effect for the list icons
        self.kinds = _list(ICON, (96, 100))
        for kind in EFFECT_KINDS:
            item = QListWidgetItem(effect_label(kind))
            item.setData(Qt.UserRole, kind)
            self.kinds.addItem(item)
        self.kinds.currentItemChanged.connect(self._kind_changed)

        self.theme = QComboBox()
        self.theme.addItem("Effect's own", None)
        for t in dict.fromkeys([*SPELL_THEMES, *THEMES]):
            self.theme.addItem("project palette" if t == "project" else t, t)
        self.colours = QLineEdit()
        self.colours.setPlaceholderText("or your colours: #ff8800  or  #331100 #ff8800 #ffffcc")
        self.size_name = QComboBox()
        for label, v in (("Project scale", None), ("Small (64)", "small"), ("Medium (96)", "medium"),
                         ("Large (128)", "large"), ("Huge (192)", "huge"), ("Custom", "custom")):
            self.size_name.addItem(label, v)
        self.size_ = spin(0, 512, 0, "Project scale")
        self.frames = spin(0, 48, 0, "Effect's own")
        self.loop = QComboBox()
        for label, v in (("Effect's own", None), ("Loop", True), ("Once", False)):
            self.loop.addItem(label, v)
        self.seed = spin(0, 9999, 0)
        self.hits = spin(1, 3, 3)              # sword combo
        self.angle = QComboBox()               # hit sparks
        for label, v in ANGLES:
            self.angle.addItem(label, v)
        self.outline = QComboBox()             # premium effects
        for label, v in (("Effect's own", None), ("None", "none"), ("Dark edge (light ground)", "dark"),
                         ("Bright rim", "rim")):
            self.outline.addItem(label, v)
        self.wind = QComboBox()                # smoke and dust only
        for label, v in (("Still air", 0.0), ("Breeze to the right", 0.5),
                         ("Wind to the right", 1.0), ("Breeze to the left", -0.5),
                         ("Wind to the left", -1.0)):
            self.wind.addItem(label, v)
        self.dither = QCheckBox("Dithered edges (classic)")
        self.dither.setChecked(True)
        self.ground = QComboBox()
        for label, v in (("Dark ground", "dark"), ("Light ground", "light")):
            self.ground.addItem(label, v)
        self.glow = QCheckBox("Glow (bloom preview)")
        self.glow.setChecked(True)
        self.paint = QCheckBox("Hand-painted pass (GPU, ComfyUI)")
        self.paint.setToolTip("SDXL img2img paints over the effect, then it is snapped back to its own "
                              "pixels and colours. Needs ComfyUI running.")
        self.name = QLineEdit()
        self.name.setPlaceholderText("Name (optional)")
        for w in (self.theme, self.loop, self.wind, self.size_name, self.angle, self.outline):
            w.currentIndexChanged.connect(self.preview)
        for w in (self.size_, self.frames, self.seed, self.hits):
            w.valueChanged.connect(self.preview)
        self.dither.toggled.connect(self.preview)
        self.colours.editingFinished.connect(self.preview)
        self.ground.currentIndexChanged.connect(self._show)
        self.glow.toggled.connect(self._show)
        self.make_btn = QPushButton("Make effect")
        self.make_btn.setStyleSheet("font-weight:600;")
        self.make_btn.clicked.connect(self.make)
        self.busy_widgets = [self.make_btn]

        opts = QGroupBox("Effect")
        form = QFormLayout(opts)
        form.addRow("Colours", self.theme)
        form.addRow("", self.colours)
        form.addRow("Size", self.size_name)
        form.addRow("Custom px", self.size_)
        form.addRow("Frames", self.frames)
        form.addRow("Plays", self.loop)
        form.addRow("Variation", self.seed)
        form.addRow("Hits", self.hits)
        form.addRow("Direction", self.angle)
        form.addRow("Edge", self.outline)
        form.addRow("Wind", self.wind)
        form.addRow(self.dither)
        form.addRow(self.paint)
        form.addRow(self.name)
        form.addRow(self.make_btn)
        self.info = QLabel()
        self.info.setWordWrap(True)
        form.addRow(self.info)

        self.view = PixelView("Pick an effect", (260, 220))
        bar = QHBoxLayout()
        bar.addWidget(self.ground)
        bar.addWidget(self.glow)
        bar.addStretch(1)
        self.saved = _list(64, (92, 100))
        self.saved.currentItemChanged.connect(self._saved_changed)

        left = QVBoxLayout()
        left.addWidget(QLabel("Effects"))
        left.addWidget(self.kinds, 1)
        mid = QVBoxLayout()
        mid.addWidget(self.view, 1)
        mid.addLayout(bar)
        mid.addWidget(QLabel("Saved effects (this project)"))
        mid.addWidget(self.saved, 1)
        lay = QHBoxLayout(self)
        lay.addLayout(left, 2)
        lay.addLayout(mid, 3)
        lay.addWidget(opts, 2)

        self._render_icons()
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._timed)
        self.kinds.setCurrentRow(0)
        self.refresh()
        self.timer.start(90)

    # -- settings
    def _style(self):
        from ..assets import current_project
        return current_project().style

    def current_kind(self) -> str:
        item = self.kinds.currentItem()
        return item.data(Qt.UserRole) if item else "fire_burst"

    def select_kind(self, kind: str):
        for i in range(self.kinds.count()):
            if self.kinds.item(i).data(Qt.UserRole) == kind:
                self.kinds.setCurrentRow(i)
                return
        raise ValueError(kind)

    def options(self) -> dict:
        from ..effects import SPELLS
        kind = self.current_kind()
        text = self.colours.text().strip()
        size = self.size_name.currentData()
        if size == "custom":
            size = self.size_.value() or None
        o = {"theme": self.theme.currentData(), "colours": text or None, "size": size,
             "frames": self.frames.value() or None, "loop": self.loop.currentData(),
             "seed": self.seed.value()}
        if kind in SPELLS:
            if kind == "sword_combo":
                o["hits"] = self.hits.value()
            if kind == "hit_spark":
                o["angle"] = self.angle.currentData()
            if self.outline.currentData():
                o["outline"] = self.outline.currentData()
        else:
            o["dither"] = 0.5 if self.dither.isChecked() else 0.0
            o["wind"] = self.wind.currentData() if self.wind.isEnabled() else 0.0
            if o["theme"] == "project":
                o["theme"] = None
        return o

    # -- previews
    def _render_icons(self):
        from ..effects import SPELLS, render_effect
        st = self._style()
        for i in range(self.kinds.count()):
            kind = self.kinds.item(i).data(Qt.UserRole)
            self.kind_fx[kind] = render_effect(kind, style=st, size=ICON_SIZE if kind in SPELLS else None)
            self.kinds.item(i).setIcon(_icon(self._icon_frame(self.kind_fx[kind])))

    @staticmethod
    def _icon_frame(fx):
        from ..effects import peak_frame
        return peak_frame(fx)

    def _kind_changed(self, *_):
        from ..effects import SPELLS, WINDY
        kind = self.current_kind()
        premium = kind in SPELLS
        self.wind.setEnabled(kind in WINDY)
        self.dither.setEnabled(not premium)
        self.hits.setEnabled(kind == "sword_combo")
        self.angle.setEnabled(kind == "hit_spark")
        self.outline.setEnabled(premium)
        self.preview()

    def preview(self, *_):
        from ..effects import render_effect
        try:
            self.fx = render_effect(self.current_kind(), style=self._style(), **self.options())
        except ValueError as e:
            self.info.setText(str(e))
            return
        w, h = self.fx.size
        durs = self.fx.frame_durations()
        timing = (f"{durs[0]} ms each" if len(set(durs)) == 1 else f"{min(durs)}-{max(durs)} ms, "
                  f"{sum(durs)} ms in all")
        anims = ", ".join(f"{t}{' (loop)' if lp else ''}" for t, _, _, lp in self.fx.tag_list())
        self.info.setText(f"{w}x{h}, {len(self.fx.frames)} frames, {timing}; {anims}; "
                          f"{len(self.fx.palette)} colours"
                          + (f", glow map, {self.fx.blend} blend" if self.fx.glow is not None else ""))
        self.frame_i = 0
        self._show()

    def _sequence(self):
        """(frames, glows, durations) of the big preview, with a pause after one-shots."""
        fx = self.fx
        frames, durs = list(fx.frames), fx.frame_durations()
        glows = list(fx.glow) if fx.glow is not None else [None] * len(frames)
        if not fx.loop:
            frames.append(np.zeros_like(fx.frames[0]))
            glows.append(None)
            durs = durs + [400]
        return frames, glows, durs

    def _show(self, *_):
        if self.fx is None:
            return
        frames, glows, _ = self._sequence()
        i = self.frame_i % len(frames)
        ground = GROUNDS[self.ground.currentData() or "dark"]
        self.view.show_rgba(composite(frames[i], ground, glows[i] if self.glow.isChecked() else None))

    def tick(self):
        """Next frame of every animation (the timer calls this; tests may too)."""
        self.tick_n += 1
        for i in range(self.kinds.count()):
            fx = self.kind_fx.get(self.kinds.item(i).data(Qt.UserRole))
            if fx:
                frames = fx.frames + ([] if fx.loop else [fx.frames[-1]] * 3)
                self.kinds.item(i).setIcon(_icon(frames[self.tick_n % len(frames)]))
        if self.fx is not None:
            self.frame_i += 1
            self._show()

    def _timed(self):
        self.tick()
        ms = 90
        if self.fx is not None:
            _, _, durs = self._sequence()
            ms = durs[self.frame_i % len(durs)]
        self.timer.start(max(20, int(ms)))

    # -- make + saved
    def make(self):
        from ..effects import make_effect
        kind, opts, name = self.current_kind(), self.options(), self.name.text().strip() or None
        paint = self.paint.isChecked()
        self.run(lambda say: make_effect(kind, name, paint=paint, **opts), self._made,
                 "Painting the effect on your GPU…" if paint else "Making the effect…")

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

"""World -> Weather & sky: weather / atmosphere effects, animated water and parallax backgrounds
(weather.py, water.py, backgrounds.py). Pick what, a time of day and how strong; the preview
plays live (weather over a background of the same time of day, water as a little lake, the
background scrolling); Make saves it to the project; Export… writes it for Godot or Unity."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QPushButton, QVBoxLayout,
)

from .common import PixelView, ToolTab, spin

WHAT = (("Weather & atmosphere", "weather"), ("Animated water", "water"), ("Parallax background", "background"))
TOD_LABELS = (("Sunset", "sunset"), ("Night", "night"), ("Dawn", "dawn"), ("Noon", "noon"))
WINDS = (("Breeze", None), ("Still", 0.0), ("Wind to the right", 0.8), ("Wind to the left", -0.8))
PREVIEW = (240, 135)


class WeatherTab(ToolTab):
    title = "Weather & sky"
    changed = Signal()

    def __init__(self, status=None, client=None):
        super().__init__(status)
        from ..backgrounds import SCENES
        from ..water import THEMES
        from ..weather import KINDS
        self.client = client
        self.items = {"weather": [(KINDS[k][0], k) for k in KINDS],
                      "water": [(t.title(), t) for t in THEMES],
                      "background": [(s.title(), s) for s in SCENES]}
        self.what = QComboBox()
        for label, v in WHAT:
            self.what.addItem(label, v)
        self.list = QListWidget()
        self.tod = QComboBox()
        for label, v in TOD_LABELS:
            self.tod.addItem(label, v)
        self.neutral = QCheckBox("Neutral colours (no time of day)")
        self.intensity = spin(30, 200, 100)
        self.intensity.setSuffix(" %")
        self.wind = QComboBox()
        for label, v in WINDS:
            self.wind.addItem(label, v)
        self.seed = spin(0, 9999, 0)
        self.words = QLineEdit()
        self.words.setPlaceholderText("Background from words (your GPU), e.g. misty bamboo valley")
        self.over = QComboBox()                       # what the weather preview plays over
        for label, v in (("Over a background", "bg"), ("Over dark ground", "dark")):
            self.over.addItem(label, v)
        self.name = QLineEdit()
        self.name.setPlaceholderText("Name (optional)")
        self.make_btn = QPushButton("Make && save")
        self.make_btn.setStyleSheet("font-weight:600;")
        self.make_btn.clicked.connect(self.make)
        self.engine = QComboBox()
        for label, v in (("Godot", "godot-2d"), ("Unity", "unity"), ("GameMaker", "gamemaker"), ("Generic", "generic")):
            self.engine.addItem(label, v)
        self.export_btn = QPushButton("Export…")
        self.export_btn.clicked.connect(self.export_dialog)
        self.busy_widgets = [self.make_btn, self.export_btn]
        self.info = QLabel()
        self.info.setWordWrap(True)
        opts = QGroupBox("Settings")
        form = QFormLayout(opts)
        form.addRow("Time of day", self.tod)
        form.addRow("", self.neutral)
        form.addRow("Strength", self.intensity)
        form.addRow("Wind", self.wind)
        form.addRow("Variation", self.seed)
        form.addRow("Preview", self.over)
        form.addRow(self.words)
        form.addRow(self.name)
        form.addRow(self.make_btn)
        row = QHBoxLayout()
        row.addWidget(self.engine)
        row.addWidget(self.export_btn)
        form.addRow(row)
        form.addRow(self.info)
        self.view = PixelView("Pick something to preview", (360, 220))
        self.saved = QListWidget()
        self.saved.setMaximumHeight(130)
        self.saved.currentItemChanged.connect(self._saved_changed)
        left = QVBoxLayout()
        left.addWidget(self.what)
        left.addWidget(self.list, 1)
        mid = QVBoxLayout()
        mid.addWidget(self.view, 1)
        mid.addWidget(QLabel("Saved weather, water and backgrounds (this project)"))
        mid.addWidget(self.saved)
        lay = QHBoxLayout(self)
        lay.addLayout(left, 2)
        lay.addLayout(mid, 4)
        lay.addWidget(opts, 2)
        self.current = None                 # Weather | WaterTiles | Background
        self.backdrop = {}                  # tod -> background composite for weather previews
        self.frame = 0
        self.last = None                    # last asset made / picked
        self.what.currentIndexChanged.connect(self._fill)
        self.list.currentItemChanged.connect(self.preview)
        for w in (self.tod, self.wind, self.over):
            w.currentIndexChanged.connect(self.preview)
        for w in (self.intensity, self.seed):
            w.valueChanged.connect(self.preview)
        self.neutral.toggled.connect(self.preview)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self._fill()
        self.refresh()
        self.timer.start(80)

    # -- settings
    def kind(self) -> str:
        return self.what.currentData()

    def item(self) -> str:
        it = self.list.currentItem()
        return it.data(Qt.UserRole) if it else self.items[self.kind()][0][1]

    def select(self, what: str, item: str):
        self.what.setCurrentIndex([v for _, v in WHAT].index(what))
        for i in range(self.list.count()):
            if self.list.item(i).data(Qt.UserRole) == item:
                self.list.setCurrentRow(i)
                return
        raise ValueError(item)

    def tod_value(self):
        if self.kind() == "weather" and self.neutral.isChecked():
            return None
        return self.tod.currentData()

    def options(self) -> dict:
        k = self.kind()
        if k == "weather":
            return {"tod": self.tod_value(), "intensity": self.intensity.value() / 100, "wind": self.wind.currentData(),
                    "seed": self.seed.value()}
        if k == "water":
            return {"tod": self.tod_value(), "seed": self.seed.value()}
        return {"tod": self.tod_value(), "seed": self.seed.value()}

    def _style(self):
        from ..assets import current_project
        return current_project().style

    def _fill(self, *_):
        self.list.blockSignals(True)
        self.list.clear()
        for label, v in self.items[self.kind()]:
            it = QListWidgetItem(label)
            it.setData(Qt.UserRole, v)
            self.list.addItem(it)
        self.list.blockSignals(False)
        k = self.kind()
        self.intensity.setEnabled(k == "weather")
        self.wind.setEnabled(k == "weather")
        self.neutral.setEnabled(k == "weather")
        self.over.setEnabled(k == "weather")
        self.words.setEnabled(k == "background")
        self.list.setCurrentRow(0)

    # -- preview
    def _backdrop(self, tod):
        tod = tod or "noon"
        if tod not in self.backdrop:
            from ..backgrounds import render_background
            bg = render_background("mountains", tod, width=PREVIEW[0], height=PREVIEW[1], seed=3)
            self.backdrop[tod] = bg.composite()
        return self.backdrop[tod]

    def preview(self, *_):
        k, o = self.kind(), self.options()
        try:
            if k == "weather":
                from ..weather import render_weather
                self.current = render_weather(self.item(), style=self._style(), **o)
                w = self.current
                self.info.setText(f"{w.meta['label']}: {w.size[0]}x{w.size[1]} tile, {len(w.fx.frames)} frames, "
                                  f"{w.meta['loop_ms']} ms loop{'' if w.fx.loop else ' (one strike)'}, "
                                  f"{len(w.fx.palette)} colours, {len(w.particles)} particle textures, "
                                  f"{len(w.emitters)} particle layers" + (", glow map" if w.fx.glow is not None else ""))
            elif k == "water":
                from ..water import render_water
                self.current = render_water(None, theme=self.item(), style=self._style(), **o)
                wt = self.current
                self.info.setText(f"{wt.theme} water: {wt.tile} px tiles, {wt.n} frames x {wt.ms} ms; "
                                  "open water, 47 shore tiles, side + iso tiles")
            else:
                from ..backgrounds import render_background
                self.current = render_background(self.item(), o["tod"], seed=o["seed"], style=self._style())
                bg = self.current
                self.info.setText(f"{bg.scene} at {bg.tod}: {bg.size[0]}x{bg.size[1]}, layers "
                                  + ", ".join(f"{L.name} x{L.scroll:g}" for L in bg.layers))
        except ValueError as e:
            self.info.setText(str(e))
            self.current = None
        self.frame = 0
        self._show()

    def frame_rgba(self, i: int) -> np.ndarray | None:
        """What the preview shows at tick i (tests read it)."""
        c, k = self.current, self.kind()
        if c is None:
            return None
        if k == "weather":
            from ..weather import preview_over
            if self.over.currentData() == "bg":
                back = self._backdrop(c.tod)
            else:
                back = np.zeros((PREVIEW[1], PREVIEW[0], 4), np.uint8)
                back[..., :3] = (24, 22, 36)
            n = len(c.fx.frames) + (0 if c.fx.loop else 10)
            out = preview_over(c, back, i % n if i % n < len(c.fx.frames) else len(c.fx.frames) - 1)
            return np.dstack([out, np.full(out.shape[:2], 255, np.uint8)])
        if k == "water":
            from ..water import example_map
            return example_map(c, i % c.n, 10, 6)
        return c.at_scroll(i * 2)

    def _show(self):
        img = self.frame_rgba(self.frame)
        if img is not None:
            self.view.show_rgba(img)

    def tick(self):
        if self.current is None:
            return
        self.frame += 1
        self._show()

    # -- make / export
    def make(self):
        k, item, o = self.kind(), self.item(), self.options()
        name = self.name.text().strip() or None
        words = self.words.text().strip()
        client = self.client
        if k == "weather":
            from ..weather import make_weather

            def job(say):
                return make_weather(item, name, **o)
        elif k == "water":
            from ..water import make_water

            def job(say):
                return make_water(name, theme=item, **o)
        else:
            from ..backgrounds import make_background

            def job(say):
                return make_background(item, o["tod"], name, seed=o["seed"], prompt=words or None, client=client)
        self.run(job, self._made, "Drawing the background on your GPU…" if (k == "background" and words)
                 else "Making it…")

    def _made(self, asset):
        self.last = asset
        self.say(f"Saved {asset.name} -> {asset.folder}")
        self.refresh()
        self.changed.emit()

    def export_last(self, folder, preset: str | None = None):
        from pathlib import Path
        from ..exports import export_asset
        if self.last is None:
            raise ValueError("Make or pick one first")
        preset = preset or self.engine.currentData()
        out = Path(folder) / self.last.folder.name
        files = export_asset(self.last, preset, out)
        self.say(f"{self.last.name} for {preset} -> {out} ({len(files)} files)")
        return files

    def export_dialog(self):
        if self.last is None:
            self.say("Make one (or pick a saved one) first.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Export to")
        if folder:
            self.export_last(folder)

    def refresh(self):
        from ..assets import list_assets
        self.saved.clear()
        for kind in ("weather", "water", "background"):
            for a in list_assets(kind):
                it = QListWidgetItem(f"{a.name}  [{kind}]")
                if a.thumb.exists():
                    it.setIcon(QIcon(QPixmap(str(a.thumb))))
                it.setData(Qt.UserRole, a)
                self.saved.addItem(it)

    def open_asset(self, a):
        """Show a saved weather / water / background (the Make page made it): its kind and look
        picked in the lists, its time of day set, selected in the saved list (Export uses it)."""
        self.refresh()
        key = {"weather": "weather", "water": "theme", "background": "scene"}.get(a.kind)
        tod = a.extra.get("tod")
        if tod:
            i = self.tod.findData(tod)
            if i >= 0:
                self.tod.setCurrentIndex(i)
        if a.kind == "weather":
            self.neutral.setChecked(not tod)
        try:
            self.select(a.kind, a.extra.get(key))
        except ValueError:                  # a background drawn from words: no list entry
            self.what.setCurrentIndex([v for _, v in WHAT].index(a.kind))
        for i in range(self.saved.count()):
            if Path(self.saved.item(i).data(Qt.UserRole).folder) == Path(a.folder):
                self.saved.setCurrentRow(i)
                break
        self.last = a
        self.info.setText(f"{a.name}: {a.folder}")

    def _saved_changed(self, item, _prev=None):
        if item is not None:
            self.last = item.data(Qt.UserRole)
            self.info.setText(f"{self.last.name}: {self.last.folder}")

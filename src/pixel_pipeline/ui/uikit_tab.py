"""The UI kit page: buttons, panels, bars, fonts and icons in the project's house style
(uikit.py).

Left: theme, scale, what to show (a mock game screen with every piece, or the sheet), icons
(drawn, or drawn by your GPU), the project's saved kits. Centre: the live preview. Buttons:
Save kit, Make icons (GPU), New kit, Export for Godot / Unity.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QPushButton, QVBoxLayout,
)

from .. import uikit as U
from .common import PixelView, ToolTab

SHOW = (("screen", "Game screen (every piece)"), ("sheet", "Sheet (all pieces)"))


class UIKitTab(ToolTab):
    """UI kit: theme + house style -> preview -> "ui" asset -> Godot Theme / Unity sprites.
    `changed` fires when a kit is saved."""
    title = "UI kit"
    changed = Signal()

    def __init__(self, status=None):
        super().__init__(status)
        self.client = None                 # generate.ComfyClient; None = the local ComfyUI
        self.asset = None                  # the kit being edited (None = not saved yet)
        self.kit: U.Kit | None = None
        self._filling = False
        self._build()
        self.refresh()
        self.preview()

    def _build(self):
        self.name = QLineEdit("UI kit")
        self.theme = QComboBox()
        self.theme.addItems(list(U.THEME_NAMES))
        self.scale = QComboBox()
        self.scale.addItem("Project default", 0)
        for s in U.SCALES:
            self.scale.addItem(f"{s}x", s)
        self.show_ = QComboBox()
        for key, text in SHOW:
            self.show_.addItem(text, key)
        for w in (self.theme, self.scale, self.show_):
            w.currentIndexChanged.connect(self.preview)
        self.icons_note = QLabel("Icons: drawn (no GPU)")
        self.icons_note.setWordWrap(True)
        self.saved = QListWidget()
        self.saved.setViewMode(QListWidget.IconMode)
        self.saved.setIconSize(QSize(64, 64))
        self.saved.setGridSize(QSize(92, 96))
        self.saved.setResizeMode(QListWidget.Adjust)
        self.saved.setMovement(QListWidget.Static)
        self.saved.currentItemChanged.connect(self._saved_picked)
        self.save_btn = QPushButton("Save kit")
        self.save_btn.setStyleSheet("font-weight:600;")
        self.save_btn.clicked.connect(self.save)
        self.gpu_btn = QPushButton("Make icons (GPU)")
        self.gpu_btn.setToolTip("Heart, coin, gem, key, sword, shield, potion and gear drawn by "
                                "your GPU in the house style (ComfyUI); the kit works without it")
        self.gpu_btn.clicked.connect(self.make_icons)
        self.drawn_btn = QPushButton("Use drawn icons")
        self.drawn_btn.clicked.connect(self.drawn_icons)
        self.new_btn = QPushButton("New kit")
        self.new_btn.clicked.connect(self.new_kit)
        self.export_godot = QPushButton("Export for Godot…")
        self.export_godot.clicked.connect(lambda: self.export("godot"))
        self.export_unity = QPushButton("Export for Unity…")
        self.export_unity.clicked.connect(lambda: self.export("unity"))
        self.busy_widgets = [self.save_btn, self.gpu_btn, self.drawn_btn, self.new_btn,
                             self.export_godot, self.export_unity]

        box = QGroupBox("UI kit")
        form = QFormLayout(box)
        form.addRow("Name", self.name)
        form.addRow("Theme", self.theme)
        form.addRow("Scale", self.scale)
        form.addRow("Show", self.show_)
        form.addRow(self.icons_note)
        note = QLabel("Outline, shading, detail and palette come from the project's house style "
                      "(Library -> Style…).")
        note.setWordWrap(True)
        form.addRow(note)
        for b in (self.save_btn, self.gpu_btn, self.drawn_btn, self.new_btn, self.export_godot,
                  self.export_unity):
            form.addRow(b)
        left = QVBoxLayout()
        left.addWidget(box)
        left.addWidget(QLabel("UI kits (this project)"))
        left.addWidget(self.saved, 1)
        self.view = PixelView("The UI kit", (480, 300))
        lay = QHBoxLayout(self)
        lay.addLayout(left, 2)
        lay.addWidget(self.view, 5)

    # ----------------------------------------------------------------- state
    def _icons(self) -> dict:
        return U._gpu_icons_of(self.asset) if self.asset is not None else {}

    def current_kit(self) -> "U.Kit":
        from ..assets import current_project
        scale = self.scale.currentData() or None
        return U.make_kit(self.theme.currentText(), current_project().style, scale, self._icons())

    def preview(self, *_):
        if self._filling:
            return
        self.kit = self.current_kit()
        img = U.mock_screen(self.kit) if self.show_.currentData() == "screen" else U.pack(self.kit)[0]
        self.view.show_rgba(img)
        gpu = self.asset is not None and self.asset.extra.get("icons") == "gpu"
        self.icons_note.setText("Icons: drawn by your GPU" if gpu else "Icons: drawn (no GPU)")

    def refresh(self):
        self.saved.blockSignals(True)
        self.saved.clear()
        for a in U.list_uis():
            item = QListWidgetItem(a.name)
            if a.thumb.exists():
                item.setIcon(QIcon(QPixmap(str(a.thumb))))
            item.setData(Qt.UserRole, a)
            self.saved.addItem(item)
            if self.asset is not None and a.folder == self.asset.folder:
                self.saved.setCurrentItem(item)
        self.saved.blockSignals(False)

    # ----------------------------------------------------------------- actions
    def save(self):
        """Write the kit (a new "ui" asset, or the one being edited)."""
        name = self.name.text().strip() or "UI kit"
        theme, scale = self.theme.currentText(), self.scale.currentData() or None
        if self.asset is None:
            self.asset = U.new_ui(name, theme, scale=scale)
        else:
            self.asset.name = name
            U.update_ui(self.asset, theme=theme, scale=scale or 0)      # 0 = project default
        self.refresh()
        self.preview()
        self.say(f"Saved {self.asset.name} -> {self.asset.folder}")
        self.changed.emit()
        return self.asset

    def make_icons(self):
        asset = self.asset or self.save()
        client = self.client
        self.run(lambda say: U.make_gpu_icons(asset, client=client, progress=say),
                 self._icons_made, "Drawing icons on your GPU…")

    def _icons_made(self, asset):
        self.asset = asset
        self.refresh()
        self.preview()
        self.say(f"{asset.name}: icons drawn by your GPU")
        self.changed.emit()

    def drawn_icons(self):
        if self.asset is None:
            return
        U.update_ui(self.asset, drawn_icons=True)
        self.preview()
        self.refresh()
        self.changed.emit()

    def _saved_picked(self, item, _prev=None):
        if item is not None:
            self.open_kit(item.data(Qt.UserRole))

    def open_kit(self, asset):
        self.asset = asset
        self._filling = True
        try:
            self.name.setText(asset.name)
            self.theme.setCurrentText(asset.extra.get("theme", "wood"))
            scale = asset.extra.get("scale")
            i = self.scale.findData(int(scale)) if scale else 0
            self.scale.setCurrentIndex(max(0, i))
        finally:
            self._filling = False
        self.preview()
        self.say(f"{asset.name} -> {asset.folder}")

    def new_kit(self):
        self.asset = None
        self.saved.clearSelection()
        self.name.setText("UI kit")
        self.preview()

    def export(self, engine: str, folder: str | None = None):
        """The kit being edited (saved first) -> a folder for Godot or Unity."""
        asset = self.asset or self.save()
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, f"Export the UI kit for {engine.title()}")
            if not folder:
                return None
        out = Path(folder) / (f"ui_{asset.folder.name}" if engine == "godot" else f"UIKit_{asset.folder.name}")
        written = U.export_kit(asset, out, engine)
        self.say(f"{asset.name} for {engine.title()} -> {out} ({len(written)} files)")
        return out

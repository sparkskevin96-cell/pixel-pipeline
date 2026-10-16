"""The Portraits tab: pick a character from the library, tick expressions, Make -> painted
(or pixel-art) dialogue portraits side by side, saved as a portrait asset (portraits.py).
Painting needs ComfyUI on your GPU (docs/LOCAL_AI.md)."""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QPushButton, QVBoxLayout, QWidget,
)

from .common import PixelView, ToolTab, grid, spin

EXTRA_EXPRESSIONS = ("determined", "smug", "hurt", "thinking")


def _list(icon: int, grid_: tuple[int, int]) -> QListWidget:
    lst = QListWidget()
    lst.setViewMode(QListWidget.IconMode)
    lst.setIconSize(QSize(icon, icon))
    lst.setGridSize(QSize(*grid_))
    lst.setResizeMode(QListWidget.Adjust)
    lst.setMovement(QListWidget.Static)
    lst.setWordWrap(True)
    return lst


class PortraitsTab(ToolTab):
    """Library character -> expressions as painted busts + pixel-art busts + a UI frame."""
    title = "Portraits"
    changed = Signal()            # a portrait asset was made

    def __init__(self, status=None):
        super().__init__(status)
        from ..portraits import EXPRESSIONS
        self.client = None                   # generate.ComfyClient; None = the local ComfyUI
        self.asset = None
        self.chars = _list(64, (92, 100))
        self.style = QComboBox()
        self.style.addItem("Painted (illustrated bust)", "painted")
        self.style.addItem("Pixel art (project palette)", "pixel")
        self.style.currentIndexChanged.connect(self._show_current)
        from ..assets import current_project
        self.size_ = spin(16, 256, current_project().style.scale(96))
        self.checks: dict[str, QCheckBox] = {}
        box = QWidget()
        g = QGridLayout(box)
        g.setContentsMargins(0, 0, 0, 0)
        for k, e in enumerate((*EXPRESSIONS, *EXTRA_EXPRESSIONS)):
            c = QCheckBox(e)
            c.setChecked(e in EXPRESSIONS)
            if e == "neutral":
                c.setEnabled(False)          # every other expression is made from it
                c.setToolTip("Always made: the other expressions are painted from it")
            self.checks[e] = c
            g.addWidget(c, k // 3, k % 3)
        self.make_btn = QPushButton("Make portraits")
        self.make_btn.setStyleSheet("font-weight:600;")
        self.make_btn.clicked.connect(self.make)
        self.refresh_btn = QPushButton("Refresh library")
        self.refresh_btn.clicked.connect(self.refresh)
        self.busy_widgets = [self.make_btn]
        opts = QGroupBox("Portraits")
        form = QFormLayout(opts)
        note = QLabel("Crops the character's head and shoulders from its picture, paints it on "
                      "your GPU (ComfyUI), then each expression from that one so the face stays "
                      "the same. Pixel-art versions and a 9-slice frame are saved too.")
        note.setWordWrap(True)
        form.addRow(note)
        form.addRow("Show", self.style)
        form.addRow("Pixel size", self.size_)
        form.addRow(QLabel("Expressions"))
        form.addRow(box)
        form.addRow(self.make_btn)
        form.addRow(self.refresh_btn)

        self.view = PixelView("Pick a character and press Make portraits", (360, 200))
        self.saved = _list(64, (110, 100))
        self.saved.currentItemChanged.connect(self._saved_changed)
        left = QVBoxLayout()
        left.addWidget(QLabel("Characters"))
        left.addWidget(self.chars, 1)
        mid = QVBoxLayout()
        mid.addWidget(self.view, 2)
        mid.addWidget(QLabel("Portraits (this project)"))
        mid.addWidget(self.saved, 1)
        lay = QHBoxLayout(self)
        lay.addLayout(left, 2)
        lay.addLayout(mid, 4)
        lay.addWidget(opts, 2)
        self.refresh()

    def refresh(self):
        """Characters from the library and portrait assets of the project."""
        from ..assets import list_assets
        from ..library import list_characters
        keep = self.current_character()
        self.chars.clear()
        for ch in list_characters():
            item = QListWidgetItem(ch.name)
            if ch.thumb.exists():
                item.setIcon(QIcon(QPixmap(str(ch.thumb))))
            item.setData(Qt.UserRole, ch)
            item.setToolTip("has a picture" if ch.picture.exists() else
                            "no picture.png: painted from its sprite (less faithful)")
            self.chars.addItem(item)
            if keep is not None and ch.folder == keep.folder:
                self.chars.setCurrentItem(item)
        if self.chars.currentItem() is None and self.chars.count():
            self.chars.setCurrentRow(0)
        self.saved.blockSignals(True)
        self.saved.clear()
        for a in list_assets("portrait"):
            item = QListWidgetItem(a.name)
            if a.thumb.exists():
                item.setIcon(QIcon(QPixmap(str(a.thumb))))
            item.setData(Qt.UserRole, a)
            item.setToolTip(", ".join(a.extra.get("expressions", [])))
            self.saved.addItem(item)
        self.saved.blockSignals(False)

    def current_character(self):
        item = self.chars.currentItem() if hasattr(self, "chars") else None
        return item.data(Qt.UserRole) if item else None

    def select_character(self, name: str):
        for i in range(self.chars.count()):
            if self.chars.item(i).data(Qt.UserRole).name == name:
                self.chars.setCurrentRow(i)
                return
        raise ValueError(name)

    def expressions(self) -> list[str]:
        return [e for e, c in self.checks.items() if c.isChecked()]

    def make(self):
        ch = self.current_character()
        if ch is None:
            self.warn("Add a character to the library first (1 Characters tab).")
            return
        from ..portraits import make_portraits
        exprs, style, size, client = self.expressions(), self.style.currentData(), \
            self.size_.value(), self.client
        from ..makeany import GPU
        from ..progress import Stage
        n = max(1, len(exprs))
        plan = [Stage("crop", "Head and shoulders", 0.03, r"cropping"),
                Stage("neutral", "Painting (neutral)", 0.5, r"\(neutral\)", pictures=1),
                Stage("expr", f"{n - 1} expressions", max(0.05, GPU["inpaint"] * (n - 1)),
                      r"painting .*\((?!neutral)|no face found", pictures=max(1, n - 1)),
                Stage("write", "Sheets", 0.05, r"writing the sheets")]
        self.run(lambda say: make_portraits(ch, exprs, style, size, client=client, progress=say),
                 self._made, f"Painting portraits of {ch.name}…", plan=plan,
                 title=f"Portraits of {ch.name}")

    def _made(self, asset):
        self.refresh()
        self.show_asset(asset)
        self.changed.emit()

    def show_asset(self, asset):
        """The asset's expressions side by side, in the chosen style."""
        from ..portraits import load_portraits
        self.asset = asset
        images = load_portraits(asset, self.style.currentData())
        if images:
            self.view.show_rgba(grid(list(images.values()), len(images), gap=4))
        self.say(f"{asset.name}: {', '.join(images)} -> {asset.folder}")

    def _show_current(self, *_):
        if self.asset is not None:
            self.show_asset(self.asset)

    def _saved_changed(self, item, _prev=None):
        if item is not None:
            self.show_asset(item.data(Qt.UserRole))

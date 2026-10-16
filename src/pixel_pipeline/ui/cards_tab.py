"""The Cards tab: card art + card frames for card games (cards.py).

Left: where the picture comes from (a library character, an item, a prop, or words), the art
style (painted / pixel) and an optional scene. Centre: the composed card, live as you type
(placeholder art from the source's own picture until you make art), the art variations to pick
from, and this project's saved cards. Right: the card's fields (name, cost, type, rules text,
attack / health or a power badge), rarity (frame metal), faction (colours), size; Make art (your
GPU, ComfyUI), Save card, New card, Export for Godot / Unity.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QPlainTextEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from .. import cards as C
from .common import PixelView, ToolTab

SOURCE_LABELS = {"character": "Library character", "item": "Item", "prop": "Prop",
                 "words": "Words (spells, anything)"}


def _icons(icon: int, grid: tuple[int, int]) -> QListWidget:
    lst = QListWidget()
    lst.setViewMode(QListWidget.IconMode)
    lst.setIconSize(QSize(icon, icon))
    lst.setGridSize(QSize(*grid))
    lst.setResizeMode(QListWidget.Adjust)
    lst.setMovement(QListWidget.Static)
    lst.setWordWrap(True)
    return lst


def _stat_spin(value=None) -> QSpinBox:
    box = QSpinBox()
    box.setRange(-1, 99)
    box.setSpecialValueText("none")          # -1 shows as "none" = no badge
    box.setValue(-1 if value is None else int(value))
    return box


class CardsTab(ToolTab):
    """Cards: source -> art (GPU, variations) + procedural frame + text -> card asset.
    `changed` fires when a card is saved or its art is made / picked."""
    title = "Cards"
    changed = Signal()

    def __init__(self, status=None):
        super().__init__(status)
        self.client = None                 # generate.ComfyClient; None = the local ComfyUI
        self.asset = None                  # the card being edited (None = not saved yet)
        self._pictures: dict = {}          # source folder -> cut-out picture (placeholder art)
        self._filling = False
        self._build()
        self.refresh()
        self.preview()

    # ----------------------------------------------------------------- layout
    def _build(self):
        # source
        self.source_kind = QComboBox()
        for k in C.SOURCE_KINDS:
            self.source_kind.addItem(SOURCE_LABELS[k], k)
        self.source_kind.currentIndexChanged.connect(self._kind_changed)
        self.sources = _icons(56, (84, 88))
        self.sources.currentItemChanged.connect(self._source_changed)
        self.words = QLineEdit()
        self.words.setPlaceholderText("e.g. a huge fireball exploding, flames and sparks")
        self.words.editingFinished.connect(self._words_changed)
        self.art_style = QComboBox()
        self.art_style.addItem("Painted (card illustration)", "painted")
        self.art_style.addItem("Pixel art (project palette)", "pixel")
        self.scene = QLineEdit()
        self.scene.setPlaceholderText("scene (optional), e.g. in a snowy mountain pass")
        src_box = QGroupBox("Picture from")
        sl = QVBoxLayout(src_box)
        sl.addWidget(self.source_kind)
        sl.addWidget(self.sources, 1)
        sl.addWidget(self.words)
        form = QFormLayout()
        form.addRow("Art style", self.art_style)
        form.addRow("Scene", self.scene)
        sl.addLayout(form)

        # card view, variations, saved cards
        self.view = PixelView("Pick a source and fill in the card", (300, 380))
        self.variations = _icons(96, (112, 82))
        self.variations.setMaximumHeight(100)
        self.variations.setFlow(QListWidget.LeftToRight)
        self.variations.currentRowChanged.connect(self._variation_picked)
        self.saved = _icons(64, (96, 104))
        self.saved.currentItemChanged.connect(self._saved_picked)

        # fields
        self.name = QLineEdit("New card")
        self.cost = _stat_spin(1)
        self.type_ = QLineEdit("Unit")
        self.text = QPlainTextEdit()
        self.text.setPlaceholderText("Rules text, e.g. Charge. Deal 1 damage to all enemies.")
        self.text.setMaximumHeight(90)
        self.attack = _stat_spin(2)
        self.health = _stat_spin(3)
        self.power = _stat_spin()
        self.rarity = QComboBox()
        self.rarity.addItems(list(C.RARITIES))
        self.faction = QComboBox()
        self.faction.addItems(list(C.FACTIONS))
        self.size_ = QComboBox()
        for k, (w, h) in C.SIZES.items():
            self.size_.addItem(f"{k} ({w}x{h})", k)
        for w in (self.name, self.type_):
            w.textChanged.connect(self.preview)
        self.text.textChanged.connect(self.preview)
        for w in (self.cost, self.attack, self.health, self.power):
            w.valueChanged.connect(self.preview)
        for w in (self.rarity, self.faction, self.size_):
            w.currentIndexChanged.connect(self.preview)
        fields = QGroupBox("Card")
        ff = QFormLayout(fields)
        ff.addRow("Name", self.name)
        ff.addRow("Cost", self.cost)
        ff.addRow("Type", self.type_)
        ff.addRow("Text", self.text)
        ff.addRow("Attack", self.attack)
        ff.addRow("Health", self.health)
        ff.addRow("Power", self.power)
        ff.addRow("Rarity", self.rarity)
        ff.addRow("Faction", self.faction)
        ff.addRow("Size", self.size_)
        note = QLabel("Attack/health = a unit (two badges). Leave them none for spells and "
                      "items; Power adds one badge at the bottom.")
        note.setWordWrap(True)
        ff.addRow(note)
        self.make_btn = QPushButton("Make art (GPU)")
        self.make_btn.setStyleSheet("font-weight:600;")
        self.make_btn.setToolTip("Three illustrations on your GPU (ComfyUI); pick one below the card")
        self.make_btn.clicked.connect(self.make_art)
        self.save_btn = QPushButton("Save card")
        self.save_btn.clicked.connect(self.save)
        self.new_btn = QPushButton("New card")
        self.new_btn.clicked.connect(self.new_card)
        self.export_godot = QPushButton("Export for Godot…")
        self.export_godot.clicked.connect(lambda: self.export("godot"))
        self.export_unity = QPushButton("Export for Unity…")
        self.export_unity.clicked.connect(lambda: self.export("unity"))
        self.busy_widgets = [self.make_btn, self.save_btn, self.new_btn]
        for b in (self.make_btn, self.save_btn, self.new_btn, self.export_godot, self.export_unity):
            ff.addRow(b)

        mid = QVBoxLayout()
        mid.addWidget(self.view, 3)
        mid.addWidget(QLabel("Art variations (click to use)"))
        mid.addWidget(self.variations)
        mid.addWidget(QLabel("Cards (this project)"))
        mid.addWidget(self.saved, 1)
        lay = QHBoxLayout(self)
        lay.addWidget(src_box, 2)
        lay.addLayout(mid, 4)
        lay.addWidget(fields, 2)
        self._kind_changed()

    # ----------------------------------------------------------------- sources
    def refresh(self):
        """Sources of the chosen kind and the project's saved cards."""
        self._kind_changed()
        self.saved.blockSignals(True)
        self.saved.clear()
        for a in C.list_cards():
            item = QListWidgetItem(a.name)
            if a.thumb.exists():
                item.setIcon(QIcon(QPixmap(str(a.thumb))))
            item.setData(Qt.UserRole, a)
            self.saved.addItem(item)
            if self.asset is not None and a.folder == self.asset.folder:
                self.saved.setCurrentItem(item)
        self.saved.blockSignals(False)

    def _kind_changed(self, *_):
        kind = self.source_kind.currentData()
        words = kind == "words"
        self.words.setVisible(words)
        self.sources.setVisible(not words)
        self.sources.blockSignals(True)
        self.sources.clear()
        if not words:
            for name, folder, thumb in self._source_items(kind):
                item = QListWidgetItem(name)
                if thumb and Path(thumb).exists():
                    item.setIcon(QIcon(QPixmap(str(thumb))))
                item.setData(Qt.UserRole, str(folder))
                self.sources.addItem(item)
        self.sources.blockSignals(False)

    @staticmethod
    def _source_items(kind: str):
        if kind == "character":
            from ..library import list_characters
            return [(c.name, c.folder, c.thumb) for c in list_characters()]
        from ..assets import list_assets
        return [(a.name, a.folder, a.thumb) for a in list_assets(kind)]

    def select_source(self, kind: str, name: str | None = None, words: str = ""):
        """Pick a source (tests / scripting): kind and the entry's name, or words."""
        self.source_kind.setCurrentIndex(C.SOURCE_KINDS.index(kind))
        if kind == "words":
            self.words.setText(words)
            self._words_changed()
            return
        for i in range(self.sources.count()):
            if self.sources.item(i).text() == name:
                self.sources.setCurrentRow(i)
                return
        raise ValueError(f"No {kind} named {name!r}")

    def current_source(self) -> "C.Source | None":
        kind = self.source_kind.currentData()
        try:
            if kind == "words":
                return C.source_from("words", words=self.words.text()) \
                    if self.words.text().strip() else None
            item = self.sources.currentItem()
            return C.source_from(kind, item.data(Qt.UserRole)) if item else None
        except ValueError:
            return None

    def _source_changed(self, item, _prev=None):
        if item is None or self._filling:
            return
        self._start_from_source()

    def _words_changed(self):
        if self.source_kind.currentData() == "words" and self.words.text().strip():
            self._start_from_source()

    def _start_from_source(self):
        """A new source starts a new (unsaved) card with fields to match."""
        src = self.current_source()
        if src is None:
            return
        self.asset = None
        self.variations.clear()
        info = C.default_info(src)
        keep = (self.rarity.currentText(), self.faction.currentText(), self.size_.currentData())
        info.rarity, info.faction, info.size = keep
        self.set_fields(info)
        self.say(f"New card from {src.name}: fill it in, then Make art (GPU) or Save card")

    # ----------------------------------------------------------------- fields
    def info(self) -> "C.CardInfo":
        def val(box):
            return None if box.value() < 0 else box.value()
        return C.CardInfo(self.name.text().strip() or "Card", val(self.cost),
                          self.type_.text().strip(), self.text.toPlainText().strip(),
                          val(self.attack), val(self.health), val(self.power),
                          self.rarity.currentText(), self.faction.currentText(),
                          self.size_.currentData())

    def set_fields(self, info: "C.CardInfo"):
        self._filling = True
        try:
            self.name.setText(info.name)
            self.type_.setText(info.type)
            self.text.setPlainText(info.text)
            for box, v in ((self.cost, info.cost), (self.attack, info.attack),
                           (self.health, info.health), (self.power, info.power)):
                box.setValue(-1 if v is None else int(v))
            self.rarity.setCurrentText(info.rarity)
            self.faction.setCurrentText(info.faction)
            self.size_.setCurrentIndex(list(C.SIZES).index(info.size))
        finally:
            self._filling = False
        self.preview()

    # ----------------------------------------------------------------- preview
    def _placeholder_picture(self, src):
        if src is None or src.kind == "words":
            return None
        if src.folder not in self._pictures:
            try:
                self._pictures[src.folder] = C.source_picture(src)
            except Exception:  # noqa: BLE001 - no picture: plain placeholder
                self._pictures[src.folder] = None
        return self._pictures[src.folder]

    def card_image(self) -> np.ndarray:
        """The card as the fields say right now (the saved art if any, else a placeholder)."""
        info = self.info()
        frame, lay = C.render_frame(info.rarity, info.faction, info.size, info.stats,
                                    C._palette())
        a = lay["slots"]["art"]
        art = None
        if self.asset is not None:
            p = self.asset.folder / "art.png"
            if p.exists():
                from PIL import Image
                art = np.asarray(Image.open(p).convert("RGBA"))
                if art.shape[:2] != (a["h"], a["w"]):
                    raw = self.asset.folder / f"art_raw_v{self.asset.extra.get('art_choice', 1)}.png"
                    if raw.exists():
                        art = np.asarray(Image.open(raw).convert("RGBA"))
        if art is None:
            src = self.current_source() if self.asset is None else \
                C.Source(**self.asset.extra.get("source", {}))
            art = C.placeholder_art(a["w"], a["h"], info.faction, self._placeholder_picture(src))
        card, _ = C.compose_card(info, art, frame, lay, C._palette())
        return card

    def preview(self, *_):
        if self._filling:
            return
        try:
            self.view.show_rgba(self.card_image())
        except ValueError as e:
            self.say(str(e))

    # ----------------------------------------------------------------- actions
    def save(self):
        """Write the card (new asset, or the one being edited)."""
        info = self.info()
        style, scene = self.art_style.currentData(), self.scene.text().strip() or None
        if self.asset is None:
            src = self.current_source() or C.Source("words", "", info.name, info.name)
            self.asset = C.new_card(info, src, art_style=style, scene=scene)
        else:
            C.update_card(self.asset, art_style=style, scene=scene,
                          **{k: getattr(info, k) for k in ("name", "cost", "type", "text",
                                                           "attack", "health", "power",
                                                           "rarity", "faction", "size")})
        self.refresh()
        self.preview()
        self.say(f"Saved {self.asset.name} -> {self.asset.folder}")
        self.changed.emit()
        return self.asset

    def make_art(self):
        if self.asset is None and self.current_source() is None:
            self.warn("Pick a character, item or prop, or write some words first.")
            return
        asset = self.save()
        style, client = self.art_style.currentData(), self.client
        from ..makeany import GPU
        self.run(lambda say: C.make_art(asset, style=style, client=client, progress=say),
                 self._art_made, f"Making art for {asset.name} on your GPU…",
                 estimate=GPU["card_art"] * C.VARIATIONS * 60 + 10, title=f"Card art for {asset.name}")

    def _art_made(self, asset):
        self.asset = asset
        self._show_variations()
        self.refresh()
        self.preview()
        self.say(f"{asset.name}: pick the art you like below the card")
        self.changed.emit()

    def _show_variations(self):
        self.variations.blockSignals(True)
        self.variations.clear()
        if self.asset is not None:
            for k, p in enumerate(C.art_variations(self.asset), 1):
                item = QListWidgetItem(f"{k}")
                item.setIcon(QIcon(QPixmap(str(p))))
                self.variations.addItem(item)
            choice = int(self.asset.extra.get("art_choice", 1))
            if 0 < choice <= self.variations.count():
                self.variations.setCurrentRow(choice - 1)
        self.variations.blockSignals(False)

    def pick_art(self, index: int):
        """Use art variation `index` (1-based)."""
        if self.asset is None:
            return
        C.choose_art(self.asset, index)
        self.preview()
        self.refresh()
        self.changed.emit()

    def _variation_picked(self, row: int):
        if row >= 0:
            self.pick_art(row + 1)

    def _saved_picked(self, item, _prev=None):
        if item is not None:
            self.open_card(item.data(Qt.UserRole))

    def open_card(self, asset):
        self.asset = asset
        src = asset.extra.get("source", {})
        self._filling = True
        try:
            if src.get("kind") in C.SOURCE_KINDS:
                self.source_kind.setCurrentIndex(C.SOURCE_KINDS.index(src["kind"]))
                if src["kind"] == "words":
                    self.words.setText(src.get("prompt", ""))
            self.art_style.setCurrentIndex(C.ART_STYLES.index(asset.extra.get("art_style", "painted")))
            self.scene.setText(asset.extra.get("scene") or "")
        finally:
            self._filling = False
        self.set_fields(C._info_of(asset))
        self._show_variations()
        self.say(f"{asset.name} -> {asset.folder}")

    def new_card(self):
        self.asset = None
        self.variations.clear()
        self.saved.clearSelection()
        self.set_fields(C.CardInfo(rarity=self.rarity.currentText(),
                                   faction=self.faction.currentText(),
                                   size=self.size_.currentData(), attack=2, health=3))

    def export(self, engine: str, folder: str | None = None):
        """Every card of the project -> a folder for Godot or Unity."""
        cards = C.list_cards()
        if not cards:
            self.warn("Save a card first.")
            return None
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, f"Export cards for {engine.title()}")
            if not folder:
                return None
        out = Path(folder) / ("cards" if engine == "godot" else "Cards")
        written = C.export_cards(cards, out, engine)
        self.say(f"{len(cards)} cards for {engine.title()} -> {out} ({len(written)} files)")
        return out

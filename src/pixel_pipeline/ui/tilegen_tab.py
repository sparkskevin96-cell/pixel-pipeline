"""World -> Tile variations: a terrain drawn by the GPU in the house style -> pick a texture ->
its tileset (paths, shores, cliffs, fences) previewed on a small map -> Save to the library."""
from __future__ import annotations

from PySide6.QtCore import QSize, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QGroupBox, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QPushButton,
)

from .. import assets
from .common import PixelView, ToolTab, checkerboard, spin, to_pixmap
from .tools import _layout

CAMERA_CHOICES = (("The project's", None), ("Top-down", "topdown"), ("Side-scroller", "side"),
                  ("Isometric", "iso"))
ROLE_NOTES = {
    "ground": "16 corner tiles blend it into the base (Godot: Terrains tab, Match Corners).",
    "water": "47 shore tiles with a foam line (Godot: Match Corners and Sides).",
    "cliff": "47 tiles: raised ground with a lit lip, plus rock wall tiles 1-2 tiles high below "
             "open south sides (maps place them by the drop).",
    "line": "16 fence/wall tiles by connected sides, with collision; paint them on a layer "
            "above the ground (Godot: Match Sides).",
}


class TilegenTab(ToolTab):
    """Terrain + words -> GPU texture variations -> a tileset in the house style."""
    title = "Tile variations"
    changed = Signal()

    def __init__(self, status=None, client=None):
        super().__init__(status)
        from ..tilegen import TERRAINS
        self.client = client                     # tests: the fake ComfyUI
        self.swatches: list = []
        self.tileset = None
        self.terrain = QComboBox()
        self.terrain.addItems(list(TERRAINS))
        self.terrain.setCurrentText("dirt")
        self.terrain.currentTextChanged.connect(self._terrain_changed)
        self.words = QLineEdit()
        self.words.setPlaceholderText("e.g. mossy forest path (empty = the terrain's own words)")
        self.base = QComboBox()
        self.base.addItems(["grass", "dirt", "sand", "snow", "stone"])
        self.camera = QComboBox()
        for label, key in CAMERA_CHOICES:
            self.camera.addItem(label, key)
        self.count = spin(1, 4, 3)
        self.variants = spin(1, 4, 3)
        self.seams = QCheckBox("Repaint the seams on the GPU (one more pass each)")
        self.overhang = QCheckBox("Cliffs: grass hangs over the edge, moss in the rock")
        self.overhang.setChecked(True)
        self.overhang.toggled.connect(self._rebuild)
        self.note = QLabel()
        self.note.setWordWrap(True)
        box = QGroupBox("Terrain")
        form = QFormLayout(box)
        form.addRow("Terrain", self.terrain)
        form.addRow("Words", self.words)
        form.addRow("Over (base)", self.base)
        form.addRow("Camera", self.camera)
        form.addRow("Pictures to pick from", self.count)
        form.addRow("Inside variations", self.variants)
        form.addRow(self.seams)
        form.addRow(self.overhang)
        form.addRow(self.note)
        self.picks = QListWidget()
        self.picks.setViewMode(QListWidget.IconMode)
        self.picks.setIconSize(QSize(64, 64))
        self.picks.setFixedHeight(96)
        self.picks.currentRowChanged.connect(self._picked)
        picks = QGroupBox("Variations (pick one)")
        QFormLayout(picks).addRow(self.picks)
        self.view = PixelView("Make (GPU) draws textures; pick one to see its tiles on a sample map")
        self.make_btn = QPushButton("Make (GPU)")
        self.make_btn.clicked.connect(self.make)
        self.quick_btn = QPushButton("Built-in look (no GPU)")
        self.quick_btn.clicked.connect(self.quick)
        self.save_btn = QPushButton("Save to library")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save)
        self.busy_widgets = [self.make_btn, self.quick_btn, self.save_btn]
        _layout(self, [box, picks], self.view, [self.make_btn, self.quick_btn, self.save_btn])
        self._terrain_changed(self.terrain.currentText())

    def _terrain_changed(self, name: str):
        from ..tilegen import role_of
        self.note.setText(ROLE_NOTES[role_of(name)])

    def _style(self):
        return assets.project_style()

    def _camera(self):
        from ..tilegen import camera_for
        return self.camera.currentData() or camera_for(self._style())

    def make(self):
        terrain, words = self.terrain.currentText(), self.words.text().strip()
        count, seams, client, camera = self.count.value(), self.seams.isChecked(), self.client, self._camera()
        style = self._style()

        def job(say):
            from ..tilegen import make_swatches
            return make_swatches(terrain, words, count=count, camera=camera, client=client,
                                 style=style, inpaint_seams=seams, say=say)
        self.run(job, self._made, "Drawing textures on the GPU…")

    def quick(self):
        from ..tilegen import builtin_swatch
        st = self._style()
        self._made([builtin_swatch(self.terrain.currentText(), st.tile, 0, st)])

    def _made(self, swatches):
        self.swatches = list(swatches)
        self.picks.blockSignals(True)
        self.picks.clear()
        for i, sw in enumerate(self.swatches):
            big = sw.texture.repeat(4, 0).repeat(4, 1)
            item = QListWidgetItem(QIcon(to_pixmap(checkerboard(big))), f"{i + 1}")
            self.picks.addItem(item)
        self.picks.blockSignals(False)
        self.picks.setCurrentRow(0)
        self._picked(0)
        off = [str(i + 1) for i, sw in enumerate(self.swatches) if getattr(sw, "colour_miss", 0) > 0]
        if off:
            self.say(f"Texture(s) {', '.join(off)} still don't look like {self.swatches[0].terrain} "
                     "after the retries (wrong colours); try other words or Make again.")

    def _rebuild(self, _on=None):
        self._picked(self.picks.currentRow())

    def _picked(self, row: int):
        if not 0 <= row < len(self.swatches):
            return
        from ..tilegen import build_tileset, preview
        sw = self.swatches[row]
        try:
            self.tileset = build_tileset(sw.terrain, sw.texture, picture=sw.picture,
                                         base=self.base.currentText(), camera=self._camera(),
                                         variants=self.variants.value(), seed=sw.seed,
                                         overhang=self.overhang.isChecked())
        except ValueError as e:
            self.tileset = None
            self.save_btn.setEnabled(False)
            self.warn(str(e))
            return
        self.view.show_rgba(preview(self.tileset))
        self.save_btn.setEnabled(True)
        self.say(f"{sw.terrain}: {len(self.tileset.tiles)} tiles ({self.tileset.kind}). Save puts it in "
                 "the library with a Godot TileSet, Tiled .tsx and a sheet for Unity.")

    def save(self):
        if self.tileset is None:
            return None
        from ..tilegen import save_tileset
        row = self.picks.currentRow()
        sw = self.swatches[row] if 0 <= row < len(self.swatches) else None
        name = (self.words.text().strip() or self.tileset.terrain).capitalize() + " tiles"
        a = save_tileset(self.tileset, name, swatch=sw, words=self.words.text().strip())
        self.say(f"Saved {a.name} ({a.folder}). The Maps page and its exports now draw with it.")
        self.changed.emit()
        return a

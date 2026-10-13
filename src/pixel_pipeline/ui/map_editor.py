"""Map editor for the World tab: a library of map assets, new maps from templates, a top-down
paint canvas (height, terrain, props, team starts, fill, undo/redo) and a live isometric
preview. Maps are saved as "map" assets in the current project (maplib). The canvas draws with
the project's generated tilesets (tilegen.MapLook: shores, paths, cliffs, fences) when there
are any; Tiles picks one per terrain (newest by default, or Built-in) and the map remembers it."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QIcon, QImage, QKeySequence, QPainter, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QInputDialog,
    QLabel, QLineEdit, QListView, QListWidget, QListWidgetItem, QPushButton, QScrollArea,
    QSizePolicy, QSplitter, QToolButton, QVBoxLayout, QWidget,
)

from .. import assets, maplib
from ..diorama import TERRAINS
from .common import PixelView, ToolTab, spin, to_pixmap

CELL = 16
TOOLS = (("raise", "Raise"), ("lower", "Lower"), ("terrain", "Paint"), ("props", "Props"),
         ("spawns", "Starts"), ("fill", "Fill"))
TOOL_HINTS = {
    "raise": "Click/drag: raise one level (half a tile). Right-click: lower.",
    "lower": "Click/drag: lower one level. Right-click: raise.",
    "terrain": "Click/drag: paint the terrain. Right-click: pick the terrain under the cursor.",
    "props": "Click: place the prop. Right-click: remove it.",
    "spawns": "Click: a start cell for the team. Right-click: remove it.",
    "fill": "Click: fill the connected area of one terrain and height.",
}
EXPORT_LABELS = (("godot-3d", "Godot HD-2D board (3D viewer project)"),
                 ("godot-tilemap", "Godot TileMap (top-down)"),
                 ("godot-iso", "Godot TileMap (isometric)"),
                 ("tiled", "Tiled map (.tmj)"),
                 ("json", "JSON + PNG layers"))


class MapCanvas(QWidget):
    """The top-down grid: draws the map picture at a whole-number zoom, reports cells."""
    pressed = Signal(int, int, bool)        # col, row, right button
    dragged = Signal(int, int, bool)
    released = Signal()
    hovered = Signal(int, int)

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.zoom = 2
        self.cell = CELL                     # px per map cell in the picture (a tileset's size)
        self.image: QImage | None = None
        self._down = None
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def show_rgba(self, rgba: np.ndarray) -> None:
        rgba = np.ascontiguousarray(rgba)
        h, w = rgba.shape[:2]
        self.image = QImage(rgba.data, w, h, w * 4, QImage.Format_RGBA8888).copy()
        self.setFixedSize(w * self.zoom, h * self.zoom)
        self.update()

    def set_zoom(self, zoom: int) -> None:
        self.zoom = max(1, min(6, zoom))
        if self.image is not None:
            self.setFixedSize(self.image.width() * self.zoom, self.image.height() * self.zoom)
        self.update()

    def cell_at(self, pos) -> tuple[int, int]:
        k = self.cell * self.zoom
        return int(pos.x() // k), int(pos.y() // k)

    def paintEvent(self, event):
        if self.image is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform, False)
        p.drawImage(self.rect(), self.image)
        p.end()

    def mousePressEvent(self, e):
        right = e.button() == Qt.RightButton
        self._down = right
        self.pressed.emit(*self.cell_at(e.position()), right)

    def mouseMoveEvent(self, e):
        c, r = self.cell_at(e.position())
        self.hovered.emit(c, r)
        if self._down is not None:
            self.dragged.emit(c, r, self._down)

    def mouseReleaseEvent(self, e):
        self._down = None
        self.released.emit()


def _swatch(rgba: np.ndarray, size: int = 24) -> QIcon:
    big = np.repeat(np.repeat(rgba, max(1, size // rgba.shape[0]), 0), max(1, size // rgba.shape[1]), 1)
    return QIcon(to_pixmap(big))


class MapEditor(ToolTab):
    """Library of map assets + editor. Signals: map_saved(Asset), map_changed()."""
    title = "Maps"
    map_saved = Signal(object)
    map_changed = Signal()

    def __init__(self, status=None, project: assets.Project | None = None):
        super().__init__(status)
        self.project = project
        self.asset: assets.Asset | None = None
        self.look = None                         # tilegen.MapLook or None (built-in textures)
        self.tile_combos: dict = {}
        self.doc = maplib.new_map("flat")
        self.edit = maplib.MapEdit(self.doc)
        self.tool = "raise"
        self.rotation = 0
        self.dirty = False
        self._stroke: set = set()
        self._iso_timer = QTimer(self)
        self._iso_timer.setSingleShot(True)
        self._iso_timer.setInterval(150)
        self._iso_timer.timeout.connect(self.render_preview_now)

        # ---- left: library + new from template
        self.library = QListWidget()
        self.library.setViewMode(QListView.IconMode)
        self.library.setIconSize(QSize(120, 90))
        self.library.setResizeMode(QListView.Adjust)
        self.library.setMovement(QListView.Static)
        self.library.setSpacing(4)
        self.library.setMinimumWidth(150)
        self.library.itemClicked.connect(self._library_clicked)
        self.template = QComboBox()
        for t in maplib.TEMPLATES:
            self.template.addItem(t.title(), t)
        self.template.currentIndexChanged.connect(self._template_changed)
        self.cols = spin(8, 64, 16)
        self.rows = spin(8, 64, 12)
        self.seed = spin(0, 99999, 0)
        self.new_btn = QPushButton("New map")
        self.new_btn.clicked.connect(self._new_clicked)
        newbox = QGroupBox("New from template")
        fl = QFormLayout(newbox)
        fl.addRow("Template", self.template)
        fl.addRow("Width", self.cols)
        fl.addRow("Depth", self.rows)
        fl.addRow("Seed", self.seed)
        fl.addRow(self.new_btn)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(QLabel("Maps in this project"))
        ll.addWidget(self.library, 1)
        ll.addWidget(newbox)

        # ---- centre: tools + canvas
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)
        tools_row = QHBoxLayout()
        self.tool_buttons = {}
        for key, label in TOOLS:
            b = QToolButton()
            b.setText(label)
            b.setCheckable(True)
            b.setToolButtonStyle(Qt.ToolButtonTextOnly)
            b.clicked.connect(lambda _=False, k=key: self.set_tool(k))
            self.tool_group.addButton(b)
            self.tool_buttons[key] = b
            tools_row.addWidget(b)
        self.brush = spin(1, 3, 1)
        tools_row.addWidget(QLabel("Brush"))
        tools_row.addWidget(self.brush)
        self.undo_btn = QToolButton()
        self.undo_btn.setText("Undo")
        self.undo_btn.clicked.connect(self.undo)
        self.redo_btn = QToolButton()
        self.redo_btn.setText("Redo")
        self.redo_btn.clicked.connect(self.redo)
        zin, zout = QToolButton(), QToolButton()
        zin.setText("+")
        zout.setText("−")
        zin.clicked.connect(lambda: self.set_zoom(self.canvas.zoom + 1))
        zout.clicked.connect(lambda: self.set_zoom(self.canvas.zoom - 1))
        for w in (self.undo_btn, self.redo_btn, QLabel("Zoom"), zout, zin):
            tools_row.addWidget(w)
        tools_row.addStretch(1)

        self.palette = QListWidget()             # terrains / props / teams, by tool
        self.palette.setViewMode(QListView.ListMode)
        self.palette.setIconSize(QSize(24, 24))
        self.palette.setMaximumWidth(140)
        self.canvas = MapCanvas()
        self.canvas.pressed.connect(self._canvas_pressed)
        self.canvas.dragged.connect(self._canvas_dragged)
        self.canvas.released.connect(self._canvas_released)
        self.canvas.hovered.connect(self._canvas_hovered)
        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setAlignment(Qt.AlignCenter)
        scroll.setStyleSheet("background:#1c2030;")
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.cell_info = QLabel(" ")
        mid = QWidget()
        ml = QVBoxLayout(mid)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.addLayout(tools_row)
        row = QHBoxLayout()
        row.addWidget(self.palette)
        row.addWidget(scroll, 1)
        ml.addLayout(row, 1)
        ml.addWidget(self.hint)
        ml.addWidget(self.cell_info)

        # ---- right: iso preview
        self.iso_view = PixelView("Isometric preview", (260, 220))
        self.rotate_btn = QPushButton("Rotate 90°")
        self.rotate_btn.clicked.connect(self.rotate_preview)
        self.problems = QLabel()
        self.problems.setWordWrap(True)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.tiles_box = QGroupBox("Tiles")
        self.tiles_form = QFormLayout(self.tiles_box)
        rl.addWidget(self.iso_view, 1)
        rl.addWidget(self.rotate_btn)
        rl.addWidget(self.problems)
        rl.addWidget(self.tiles_box)

        split = QSplitter()
        split.addWidget(left)
        split.addWidget(mid)
        split.addWidget(right)
        split.setStretchFactor(1, 3)
        split.setStretchFactor(2, 2)

        # ---- bottom: name, save, export
        self.name = QLineEdit("New map")
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.save)
        self.export_btn = QPushButton("Export…")
        self.export_btn.clicked.connect(self.export_dialog)
        bottom = QHBoxLayout()
        bottom.addWidget(QLabel("Name"))
        bottom.addWidget(self.name, 1)
        bottom.addWidget(self.save_btn)
        bottom.addWidget(self.export_btn)
        self.busy_widgets = [self.save_btn, self.export_btn, self.new_btn]

        lay = QVBoxLayout(self)
        lay.addWidget(split, 1)
        lay.addLayout(bottom)

        for seq, fn in ((QKeySequence.Undo, self.undo), (QKeySequence.Redo, self.redo),
                        (QKeySequence("Ctrl+Y"), self.redo)):
            sc = QShortcut(seq, self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)

        self._template_changed()
        self.refresh_tiles()
        self.set_tool("raise")
        self.refresh_library()
        self._redraw()

    # ------------------------------------------------------------------ library

    def _project(self) -> assets.Project:
        return self.project or assets.current_project()

    def refresh_library(self) -> None:
        self.library.clear()
        for a in maplib.list_maps(self._project()):
            item = QListWidgetItem(a.name)
            if a.thumb.exists():
                item.setIcon(QIcon(QPixmap(str(a.thumb))))
            item.setData(Qt.UserRole, str(a.folder))
            item.setToolTip(f"{a.name}  ({a.extra.get('template', '')}, "
                            f"{'x'.join(str(v) for v in a.extra.get('size', []))})")
            self.library.addItem(item)

    def _library_clicked(self, item: QListWidgetItem) -> None:
        folder = Path(item.data(Qt.UserRole))
        for a in maplib.list_maps(self._project()):
            if a.folder == folder:
                self.open_asset(a)
                return

    def open_asset(self, asset: assets.Asset) -> None:
        self.asset = asset
        self._set_doc(maplib.load_map(asset))
        self.name.setText(asset.name)
        self.dirty = False
        self.say(f"Opened {asset.name}.")

    def _template_changed(self, *_):
        t = self.template.currentData()
        c, r = maplib.DEFAULT_SIZES.get(t, (16, 12))
        self.cols.setValue(c)
        self.rows.setValue(r)

    def _new_clicked(self):
        self.new_from_template(self.template.currentData(), self.cols.value(), self.rows.value(),
                               self.seed.value())

    def new_from_template(self, template: str, cols: int | None = None, rows: int | None = None,
                          seed: int = 0) -> maplib.MapDoc:
        self.asset = None
        self._set_doc(maplib.new_map(template, cols, rows, seed))
        self.name.setText(template.title() if seed == 0 else f"{template.title()} {seed}")
        self.dirty = True
        self.say(f"New {template} map, {self.doc.cols} x {self.doc.rows}. Paint, then Save.")
        return self.doc

    def _set_doc(self, doc: maplib.MapDoc) -> None:
        self.doc = doc
        self.edit = maplib.MapEdit(doc)
        self.refresh_tiles()
        self._changed()

    # ------------------------------------------------------------------ generated tiles

    def refresh_tiles(self) -> None:
        """The Tiles choices: one row per terrain / cliff / fence / wall the project has saved
        top-down tilesets for (newest first, then Built-in)."""
        from ..tilegen import look_options
        while self.tiles_form.rowCount():
            self.tiles_form.removeRow(0)
        self.tile_combos = {}
        try:
            opts = look_options(self._project())
        except Exception:  # noqa: BLE001 - no project yet
            opts = {}
        for key, items in sorted(opts.items()):
            cb = QComboBox()
            for i, a in enumerate(items):
                cb.addItem(a.name + (" (newest)" if i == 0 else ""), a.folder.name)
            cb.addItem("Built-in", "")
            want = self.doc.tiles.get(key)
            cb.setCurrentIndex(max(0, cb.findData(want)) if want is not None else 0)
            cb.currentIndexChanged.connect(lambda _=0, k=key: self._tiles_changed(k))
            self.tiles_form.addRow(key.title(), cb)
            self.tile_combos[key] = cb
        if not opts:
            note = QLabel("Built-in textures. Tilesets saved in World -> Tile variations draw here.")
            note.setWordWrap(True)
            self.tiles_form.addRow(note)
        self._update_look()

    def choose_tiles(self, key: str, folder_name: str) -> None:
        """Draw `key` (water, dirt, cliff, fence...) with this saved set ("" = built-in)."""
        cb = self.tile_combos[key]
        i = cb.findData(folder_name)
        if i < 0:
            raise ValueError(f"No tileset {folder_name!r} for {key}")
        cb.setCurrentIndex(i)

    def _tiles_changed(self, key: str) -> None:
        cb = self.tile_combos[key]
        if cb.currentIndex() == 0:
            self.doc.tiles.pop(key, None)            # follow the newest set
        else:
            self.doc.tiles[key] = cb.currentData()
        self._update_look()
        self.set_tool(self.tool)
        self._changed()

    def _update_look(self) -> None:
        self.look = maplib.map_look(self.doc, self._project())
        self.canvas.cell = self.look.tile if self.look is not None else CELL
        self.canvas.set_zoom(self.canvas.zoom)

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh_tiles()                          # sets saved since, another project
        self._redraw()

    # ------------------------------------------------------------------ tools

    def set_tool(self, tool: str) -> None:
        self.tool = tool
        self.tool_buttons[tool].setChecked(True)
        self.hint.setText(TOOL_HINTS[tool])
        keep = self.palette.currentItem().data(Qt.UserRole) if self.palette.currentItem() else None
        self.palette.clear()
        if tool in ("terrain", "fill"):
            tex = maplib.terrain_textures(16)
            for t in TERRAINS:
                sw = self.look.plain(t)[0][0] if self.look is not None else tex[t][0]
                it = QListWidgetItem(_swatch(sw), t.title())
                it.setData(Qt.UserRole, t)
                self.palette.addItem(it)
        elif tool == "props":
            for k in maplib.PROP_KINDS:
                it = QListWidgetItem(_swatch(maplib.prop_icon(k, 16)), k.title())
                it.setData(Qt.UserRole, k)
                self.palette.addItem(it)
        elif tool == "spawns":
            for team, label in ((0, "Blue team"), (1, "Red team")):
                sw = np.zeros((16, 16, 4), np.uint8)
                sw[...] = (*maplib.TEAM_RGB[team], 255)
                it = QListWidgetItem(_swatch(sw), label)
                it.setData(Qt.UserRole, team)
                self.palette.addItem(it)
        self.palette.setVisible(self.palette.count() > 0)
        rows = [self.palette.item(i).data(Qt.UserRole) for i in range(self.palette.count())]
        self.palette.setCurrentRow(rows.index(keep) if keep in rows else 0)

    def choice(self):
        it = self.palette.currentItem()
        return it.data(Qt.UserRole) if it else None

    def select(self, value) -> None:
        """Pick a terrain / prop / team in the palette (tests and right-click pick)."""
        for i in range(self.palette.count()):
            if self.palette.item(i).data(Qt.UserRole) == value:
                self.palette.setCurrentRow(i)
                return
        raise ValueError(f"{value!r} isn't in the {self.tool} palette")

    def apply(self, c: int, r: int, right: bool = False) -> None:
        """The current tool on one cell (no undo step: see press())."""
        e, b = self.edit, self.brush.value()
        if not (0 <= c < self.doc.cols and 0 <= r < self.doc.rows):
            return
        if self.tool in ("raise", "lower"):
            up = (self.tool == "raise") != right
            e.raise_cells(c, r, 1 if up else -1, b)
        elif self.tool == "terrain":
            if right:
                self.select(self.doc.diorama.terrain[r, c])
                return
            e.paint(c, r, self.choice() or "grass", b)
        elif self.tool == "props":
            if right:
                e.erase_prop(c, r)
            else:
                e.put_prop(c, r, self.choice() or "tree")
        elif self.tool == "spawns":
            if right:
                e.clear_spawn(c, r)
            else:
                e.set_spawn(c, r, int(self.choice() or 0))
        elif self.tool == "fill":
            e.fill(c, r, self.choice() or "grass")

    def press(self, c: int, r: int, right: bool = False) -> None:
        """Start a stroke: one undo step, then the tool on this cell."""
        if not (0 <= c < self.doc.cols and 0 <= r < self.doc.rows):
            return
        if not (self.tool == "terrain" and right):
            self.edit.begin()
        self._stroke = {(c, r)}
        self.apply(c, r, right)
        self._changed()

    def drag(self, c: int, r: int, right: bool = False) -> None:
        if self.tool == "fill" or (c, r) in self._stroke:
            return
        self._stroke.add((c, r))
        self.apply(c, r, right)
        self._changed()

    def release(self) -> None:
        self._stroke = set()

    def _canvas_pressed(self, c, r, right):
        self.press(c, r, right)

    def _canvas_dragged(self, c, r, right):
        self.drag(c, r, right)

    def _canvas_released(self):
        self.release()

    def _canvas_hovered(self, c, r):
        d = self.doc.diorama
        if 0 <= c < d.cols and 0 <= r < d.rows:
            prop = self.doc.prop_at(c, r)
            self.cell_info.setText(f"Cell {c},{r}   level {int(d.height[r, c])}   {d.terrain[r, c]}"
                                   + (f"   {prop}" if prop else ""))

    def undo(self) -> bool:
        ok = self.edit.undo()
        if ok:
            self._changed()
        return ok

    def redo(self) -> bool:
        ok = self.edit.redo()
        if ok:
            self._changed()
        return ok

    def set_zoom(self, zoom: int) -> None:
        self.canvas.set_zoom(zoom)

    # ------------------------------------------------------------------ views

    def _changed(self) -> None:
        self.dirty = True
        self._redraw()
        self.map_changed.emit()

    def _redraw(self) -> None:
        self.canvas.show_rgba(maplib.render_topdown(self.doc, self.canvas.cell, grid=True, look=self.look))
        self.undo_btn.setEnabled(bool(self.edit.undo_stack))
        self.redo_btn.setEnabled(bool(self.edit.redo_stack))
        if self.sync:
            self.render_preview_now()
        else:
            self._iso_timer.start()

    def render_preview_now(self) -> np.ndarray:
        img = maplib.render_iso(self.doc, self.rotation, 16, background=(28, 32, 48, 255))
        self.iso_view.show_rgba(img)
        probs = maplib.validate(self.doc)
        self.problems.setText("Ready for a game." if not probs else "Check: " + "; ".join(probs[:3]))
        return img

    def rotate_preview(self) -> None:
        self.rotation = (self.rotation + 1) % 4
        self.render_preview_now()

    # ------------------------------------------------------------------ save / export

    def save(self) -> assets.Asset:
        name = self.name.text().strip() or "map"
        self.asset = maplib.save_map(self.doc, name, project=self._project(), asset=self.asset)
        self.dirty = False
        self.refresh_library()
        self.say(f"Saved {name} to {self.asset.folder}")
        self.map_saved.emit(self.asset)
        return self.asset

    def export_to(self, target: str, folder) -> list[Path]:
        """Save if needed, then export (on a worker thread unless sync)."""
        if self.asset is None or self.dirty:
            self.save()
        asset, out = self.asset, Path(folder)
        result: list = []

        def done(files):
            result.extend(files)
            self.say(f"Exported {len(files)} files to {out}")
        self.run(lambda say: maplib.export_map(asset, target, out), done, f"Exporting {target}…")
        return result

    def export_dialog(self) -> None:
        labels = [label for _, label in EXPORT_LABELS]
        label, ok = QInputDialog.getItem(self, "Export map", "Export as", labels, 0, False)
        if not ok:
            return
        target = dict((lbl, key) for key, lbl in EXPORT_LABELS)[label]
        folder = QFileDialog.getExistingDirectory(self, "Export into folder")
        if folder:
            self.export_to(target, folder)

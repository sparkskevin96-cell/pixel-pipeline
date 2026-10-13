"""The Library page: every asset of the current project (characters, items, props, maps,
portraits, effects) in one browsable grid, with export presets, re-skins and the creation
queue.

    tab = AssetsTab(status=window.statusBar().showMessage)
    tab.library_changed.connect(...)     # something was added, re-skinned, deleted or re-tagged
    tab.project_changed.connect(...)     # (project name) the current project was switched
"""
from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, QSize, Qt, QThread, QUrl, Signal, Slot
from PySide6.QtGui import QColor, QDesktopServices, QIcon
from PySide6.QtWidgets import (
    QButtonGroup, QColorDialog, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListView,
    QListWidget, QListWidgetItem, QMessageBox, QPushButton, QRadioButton, QSlider, QSpinBox,
    QSplitter, QVBoxLayout, QWidget,
)

from .. import assets as A
from .. import exports, jobqueue, reskin
from .common import PixelView, ToolTab, checkerboard, to_pixmap

KIND_CHIPS = (("All", None), ("Characters", "character"), ("Items", "item"), ("Props", "prop"),
              ("Maps", "map"), ("Portraits", "portrait"), ("Effects", "effect"))
OUTLINES = (("Shaded outline", "selout"), ("Solid outline", "solid"), ("No outline", "none"))
THUMB = 96


def asset_picture(asset) -> np.ndarray | None:
    """The asset's picture for lists: thumb.png, icon.png, the sheet's first frame, or any png."""
    folder = Path(asset.folder)
    for name in ("thumb.png", "icon.png"):
        if (folder / name).exists():
            try:
                return np.asarray(Image.open(folder / name).convert("RGBA"))
            except OSError:
                pass
    sheet = folder / "sheet.png"
    if sheet.exists():
        try:
            img = np.asarray(Image.open(sheet).convert("RGBA"))
        except OSError:
            return None
        try:
            meta = json.loads(sheet.with_suffix(".json").read_text(encoding="utf-8"))
            f = exports._frames(meta)[0]
            return img[f.y:f.y + f.h, f.x:f.x + f.w]
        except (OSError, ValueError, KeyError, IndexError):
            return img
    pngs = sorted(folder.glob("*.png")) if folder.exists() else []
    if pngs:
        try:
            return np.asarray(Image.open(pngs[0]).convert("RGBA"))
        except OSError:
            return None
    return None


def _icon(rgba: np.ndarray | None, px: int = THUMB) -> QIcon:
    if rgba is None:
        rgba = np.zeros((px, px, 4), np.uint8)
        rgba[..., :3], rgba[..., 3] = 60, 255
    h, w = rgba.shape[:2]
    k = max(1, px // max(h, w))
    big = np.repeat(np.repeat(rgba, k, 0), k, 1)
    pix = to_pixmap(checkerboard(big, cell=max(4, k * 2)))
    if max(pix.width(), pix.height()) > px:
        pix = pix.scaled(px, px, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return QIcon(pix)


def _words(text: str) -> list[str]:
    return [t.strip() for t in text.split(",") if t.strip()]


# --------------------------------------------------------------------------- dialogs

class StyleDialog(QDialog):
    """A project's name + look (assets.Style) in plain words."""

    def __init__(self, parent=None, style: A.Style | None = None, name: str = "",
                 title: str = "New project"):
        super().__init__(parent)
        self.setWindowTitle(title)
        from ..palettes import BUILTIN
        st = style or A.Style()
        self.name = QLineEdit(name)
        self.name.setPlaceholderText("e.g. Harbour Tactics")
        self.palette = QComboBox()
        self.palette.addItem("Pick colours automatically", "auto")
        for p in BUILTIN:
            self.palette.addItem(p, p)
        if self.palette.findData(st.palette) < 0:
            self.palette.addItem(st.palette, st.palette)
        self.palette.setCurrentIndex(self.palette.findData(st.palette))
        self.colours = self._spin(2, 64, st.colours, " colours")
        self.outline = QComboBox()
        for label, key in OUTLINES:
            self.outline.addItem(label, key)
        self.outline.setCurrentIndex(max(0, self.outline.findData(st.outline)))
        self.height_ = self._spin(16, 256, st.person_height, " px tall")
        self.tilt = self._spin(0, 90, round(st.elevation), "°")
        self.tilt.setToolTip("0 = straight from the side, 30 = HD-2D / isometric, 90 = from above")
        self.directions = QComboBox()
        for n in (4, 8, 16):
            self.directions.addItem(f"{n} directions", n)
        self.directions.setCurrentIndex(max(0, self.directions.findData(st.directions)))
        self.light = self._spin(0, 100, round(st.light * 100), " %")
        self.tile = self._spin(4, 256, st.tile, " px")
        self.ppu = QDoubleSpinBox()
        self.ppu.setRange(4, 512)
        self.ppu.setValue(st.ppu)
        self.ppu.setSuffix(" px per tile")
        self.icon_ = self._spin(8, 256, st.icon, " px")
        form = QFormLayout(self)
        form.addRow("Project name", self.name)
        form.addRow("Colours", self.palette)
        form.addRow("Colours per sprite", self.colours)
        form.addRow("Outline", self.outline)
        form.addRow("A standing person is", self.height_)
        form.addRow("Camera tilt", self.tilt)
        form.addRow("Facing", self.directions)
        form.addRow("Shading baked in", self.light)
        form.addRow("Map tiles", self.tile)
        form.addRow("Sprites in 3D scenes", self.ppu)
        form.addRow("Inventory icons", self.icon_)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    @staticmethod
    def _spin(lo, hi, value, suffix=""):
        s = QSpinBox()
        s.setRange(lo, hi)
        s.setValue(int(value))
        s.setSuffix(suffix)
        return s

    def style(self) -> A.Style:
        return A.Style(palette=self.palette.currentData(), colours=self.colours.value(),
                       outline=self.outline.currentData(), person_height=self.height_.value(),
                       elevation=float(self.tilt.value()), directions=self.directions.currentData(),
                       light=self.light.value() / 100, tile=self.tile.value(),
                       ppu=float(self.ppu.value()), icon=self.icon_.value())


class ExportDialog(QDialog):
    """Preset + folder."""

    def __init__(self, parent=None, what: str = "", folder: str = ""):
        super().__init__(parent)
        self.setWindowTitle(f"Export {what}".strip())
        self.preset = QComboBox()
        for key, label in exports.PRESETS.items():
            self.preset.addItem(label, key)
        self.folder = QLineEdit(folder)
        browse = QPushButton("Choose…")
        browse.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addWidget(self.folder, 1)
        row.addWidget(browse)
        form = QFormLayout(self)
        form.addRow("For", self.preset)
        form.addRow("Into folder", row)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Export into", self.folder.text())
        if d:
            self.folder.setText(d)

    def values(self) -> tuple[str, str]:
        return self.preset.currentData(), self.folder.text().strip()


class ReskinDialog(QDialog):
    """New colours for a character, previewed before saving."""

    def __init__(self, character, parent=None):
        super().__init__(parent)
        self.character = character
        self.setWindowTitle(f"Re-skin {character.name}")
        self.custom = None
        self.by_colour = QRadioButton("Change the main colour to")
        self.by_hue = QRadioButton("Turn every colour")
        self.by_colour.setChecked(True)
        self.colour = QComboBox()
        for name in ("blue", "red", "green", "purple", "yellow", "orange", "pink", "cyan",
                     "teal", "brown", "gold", "silver", "white", "black"):
            self.colour.addItem(name.title(), name)
        self.pick = QPushButton("Pick…")
        self.pick.clicked.connect(self._pick)
        self.hue = QSlider(Qt.Horizontal)
        self.hue.setRange(-180, 180)
        self.saturation = QSlider(Qt.Horizontal)
        self.saturation.setRange(0, 200)
        self.saturation.setValue(100)
        self.lightness = QSlider(Qt.Horizontal)
        self.lightness.setRange(50, 150)
        self.lightness.setValue(100)
        self.name = QLineEdit(f"{character.name} Blue")
        self.view = PixelView("", (200, 200))
        self.error = QLabel("")
        self.error.setStyleSheet("color:#e07070;")
        crow = QHBoxLayout()
        crow.addWidget(self.colour, 1)
        crow.addWidget(self.pick)
        form = QFormLayout()
        form.addRow(self.by_colour, crow)
        form.addRow(self.by_hue, self.hue)
        form.addRow("Colour strength", self.saturation)
        form.addRow("Brightness", self.lightness)
        form.addRow("New name", self.name)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay = QHBoxLayout(self)
        lay.addWidget(self.view, 1)
        side = QVBoxLayout()
        side.addLayout(form)
        side.addWidget(self.error)
        side.addStretch(1)
        side.addWidget(buttons)
        lay.addLayout(side)
        for sig in (self.by_colour.toggled, self.colour.currentIndexChanged, self.hue.valueChanged,
                    self.saturation.valueChanged, self.lightness.valueChanged):
            sig.connect(self.update_preview)
        self.colour.currentIndexChanged.connect(self._name_from_colour)
        self.update_preview()

    def _pick(self):
        c = QColorDialog.getColor(QColor("#2f5fd8"), self, "Main colour")
        if c.isValid():
            self.custom = c.name()
            self.colour.addItem(self.custom, self.custom)
            self.colour.setCurrentIndex(self.colour.count() - 1)

    @Slot()
    def _name_from_colour(self, *_):
        self.name.setText(f"{self.character.name} {self.colour.currentText()}")

    def options(self) -> dict:
        opts = {"saturation": self.saturation.value() / 100, "lightness": self.lightness.value() / 100}
        if self.by_colour.isChecked():
            opts["target_colour"] = self.colour.currentData()
        else:
            opts["hue"] = float(self.hue.value())
        return opts

    @Slot()
    def update_preview(self, *_):
        try:
            self.view.show_rgba(reskin.preview(self.character, **self.options()))
            self.error.setText("")
        except (ValueError, OSError) as e:
            self.error.setText(str(e))


# --------------------------------------------------------------------------- queue thread

class _QueueWorker(QObject):
    update = Signal(object)
    finished = Signal()
    failed = Signal(str)

    def __init__(self, project, queue, workers, pause_s, stop):
        super().__init__()
        self.args = (project, queue, workers, pause_s, stop)

    @Slot()
    def run(self):
        project, queue, workers, pause_s, stop = self.args
        try:
            jobqueue.run_queue(project, workers, pause_s, stop, on_update=self.update.emit,
                               queue=queue)
        except Exception as e:  # noqa: BLE001 - shown to the user
            self.failed.emit(str(e) or type(e).__name__)
        self.finished.emit()


# --------------------------------------------------------------------------- the page

class AssetsTab(ToolTab):
    """Library browser for every asset kind of the current project."""
    title = "Library"
    library_changed = Signal()
    project_changed = Signal(str)

    def __init__(self, status=None):
        super().__init__(status)
        self.assets: list = []
        self.selected = None
        self.kind: str | None = None
        self.worker_fns: dict | None = None        # None = jobqueue.default_workers()
        self._qthread = self._qworker = None
        self._stop = threading.Event()
        self.project = A.current_project()
        self.queue = jobqueue.Queue(self.project)

        # top: project
        self.project_box = QComboBox()
        self.project_box.currentIndexChanged.connect(self._project_picked)
        new_btn = QPushButton("New project…")
        new_btn.clicked.connect(self._new_project_clicked)
        style_btn = QPushButton("Project look…")
        style_btn.clicked.connect(self._style_clicked)
        top = QHBoxLayout()
        top.addWidget(QLabel("Project"))
        top.addWidget(self.project_box, 1)
        top.addWidget(new_btn)
        top.addWidget(style_btn)

        # filters
        self.chips = QButtonGroup(self)
        self.chips.setExclusive(True)
        chips = QHBoxLayout()
        for i, (label, kind) in enumerate(KIND_CHIPS):
            b = QPushButton(label)
            b.setCheckable(True)
            b.setChecked(kind is None)
            b.setStyleSheet("QPushButton{border-radius:10px;padding:3px 10px;}"
                            "QPushButton:checked{background:#3b5bdb;color:white;}")
            self.chips.addButton(b, i)
            chips.addWidget(b)
        chips.addStretch(1)
        self.chips.idClicked.connect(self._chip_clicked)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search names, tags, prompts…")
        self.search.textChanged.connect(self.refresh)
        self.tag_box = QComboBox()
        self.collection_box = QComboBox()
        for box in (self.tag_box, self.collection_box):
            box.currentIndexChanged.connect(self.refresh)
        filters = QHBoxLayout()
        filters.addWidget(self.search, 2)
        filters.addWidget(QLabel("Tag"))
        filters.addWidget(self.tag_box, 1)
        filters.addWidget(QLabel("Collection"))
        filters.addWidget(self.collection_box, 1)

        # grid
        self.grid = QListWidget()
        self.grid.setViewMode(QListView.IconMode)
        self.grid.setIconSize(QSize(THUMB, THUMB))
        self.grid.setResizeMode(QListView.Adjust)
        self.grid.setMovement(QListView.Static)
        self.grid.setSpacing(6)
        self.grid.setWordWrap(True)
        self.grid.currentRowChanged.connect(self._row_changed)
        self.empty = QLabel("Nothing here yet. Make characters in the Characters tab, or queue "
                            "some below.")
        self.empty.setWordWrap(True)
        self.empty.setStyleSheet("color:#9aa3b5;")

        # details
        self.preview = PixelView("Pick something", (220, 220))
        self.name_label = QLabel("")
        self.name_label.setStyleSheet("font-size:15px;font-weight:600;")
        self.kind_label = QLabel("")
        self.tags = QLineEdit()
        self.tags.setPlaceholderText("tags, separated, by commas")
        self.collections = QLineEdit()
        self.collections.setPlaceholderText("collections, e.g. starter kit")
        save_tags = QPushButton("Save tags")
        save_tags.clicked.connect(self.save_tags)
        self.tags.returnPressed.connect(self.save_tags)
        self.collections.returnPressed.connect(self.save_tags)
        self.export_btn = QPushButton("Export…")
        self.export_btn.clicked.connect(self._export_clicked)
        self.export_coll_btn = QPushButton("Export collection…")
        self.export_coll_btn.clicked.connect(self._export_collection_clicked)
        self.reskin_btn = QPushButton("Re-skin…")
        self.reskin_btn.clicked.connect(self._reskin_clicked)
        self.open_btn = QPushButton("Open folder")
        self.open_btn.clicked.connect(self.open_folder)
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.setStyleSheet("color:#e07070;")
        self.delete_btn.clicked.connect(self._delete_clicked)
        form = QFormLayout()
        form.addRow("Tags", self.tags)
        form.addRow("Collections", self.collections)
        form.addRow("", save_tags)
        btns = QHBoxLayout()
        for b in (self.export_btn, self.reskin_btn, self.open_btn, self.delete_btn):
            btns.addWidget(b)
        details = QWidget()
        dl = QVBoxLayout(details)
        dl.addWidget(self.preview, 1)
        dl.addWidget(self.name_label)
        dl.addWidget(self.kind_label)
        dl.addLayout(form)
        dl.addLayout(btns)
        dl.addWidget(self.export_coll_btn)
        self.busy_widgets = [self.export_btn, self.export_coll_btn, self.reskin_btn, self.delete_btn]

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addLayout(chips)
        ll.addLayout(filters)
        ll.addWidget(self.grid, 1)
        ll.addWidget(self.empty)
        split = QSplitter()
        split.addWidget(left)
        split.addWidget(details)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)

        # queue
        # (a checkable QGroupBox would disable its contents when folded: a toggle button instead)
        self.queue_box = QGroupBox()
        self.queue_toggle = QPushButton("▸ Creation queue (one at a time, with a rest between)")
        self.queue_toggle.setCheckable(True)
        self.queue_toggle.setFlat(True)
        self.queue_toggle.setStyleSheet("text-align:left;font-weight:600;")
        self.queue_toggle.toggled.connect(self.show_queue)
        self.queue_inner = QWidget()
        self.queue_inner.setVisible(False)
        self.job_kind = QComboBox()
        for k in jobqueue.KINDS:
            self.job_kind.addItem(k.title(), k)
        self.job_name = QLineEdit()
        self.job_name.setPlaceholderText("Name")
        self.job_prompt = QLineEdit()
        self.job_prompt.setPlaceholderText("What it looks like, e.g. a knight in a red cape")
        self.job_variations = QSpinBox()
        self.job_variations.setRange(1, 20)
        self.job_variations.setPrefix("x")
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self._add_job_clicked)
        self.pause = QSpinBox()
        self.pause.setRange(0, 600)
        self.pause.setValue(20)
        self.pause.setSuffix(" s rest")
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start_queue)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_queue)
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self._remove_job_clicked)
        retry_btn = QPushButton("Retry")
        retry_btn.clicked.connect(self._retry_job_clicked)
        clear_btn = QPushButton("Clear finished")
        clear_btn.clicked.connect(self._clear_jobs_clicked)
        self.job_list = QListWidget()
        self.job_list.setMaximumHeight(140)
        add_row = QHBoxLayout()
        for w, s in ((self.job_kind, 0), (self.job_name, 1), (self.job_prompt, 3),
                     (self.job_variations, 0), (add_btn, 0)):
            add_row.addWidget(w, s)
        run_row = QHBoxLayout()
        for w in (self.start_btn, self.stop_btn, self.pause, retry_btn, remove_btn, clear_btn):
            run_row.addWidget(w)
        run_row.addStretch(1)
        ql = QVBoxLayout(self.queue_inner)
        ql.setContentsMargins(0, 0, 0, 0)
        ql.addLayout(add_row)
        ql.addWidget(self.job_list)
        ql.addLayout(run_row)
        qb = QVBoxLayout(self.queue_box)
        qb.addWidget(self.queue_toggle)
        qb.addWidget(self.queue_inner)

        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(split, 1)
        lay.addWidget(self.queue_box)

        self._fill_projects()
        self.refresh()
        self.refresh_queue()

    # ---------------------------------------------------------------- projects

    def _fill_projects(self):
        self.project_box.blockSignals(True)
        self.project_box.clear()
        for p in A.list_projects():
            self.project_box.addItem(p.name)
        self.project_box.setCurrentText(self.project.name)
        self.project_box.blockSignals(False)

    @Slot(int)
    def _project_picked(self, _i):
        name = self.project_box.currentText()
        if name and name != self.project.name:
            self.switch_project(name)

    def switch_project(self, name: str):
        if self.queue_running():
            self.say("Stop the queue before switching projects.")
            self._fill_projects()
            return
        self.project = A.set_current_project(name)
        self.queue = jobqueue.Queue(self.project)
        self._fill_projects()
        self.selected = None
        self.refresh()
        self.refresh_queue()
        self.project_changed.emit(self.project.name)
        self.library_changed.emit()
        self.say(f"Project: {self.project.name}")

    def new_project(self, name: str, style: A.Style | None = None):
        name = name.strip()
        if not name:
            raise ValueError("Give the project a name.")
        A.create_project(name, style or A.Style())
        self.switch_project(name)
        return self.project

    @Slot()
    def _new_project_clicked(self):
        dlg = StyleDialog(self, A.Style(), "")
        ok = dlg.exec() == QDialog.Accepted
        name, style = dlg.name.text(), dlg.style()
        dlg.setParent(None)
        if ok:
            try:
                self.new_project(name, style)
            except ValueError as e:
                self.warn(str(e))

    @Slot()
    def _style_clicked(self):
        dlg = StyleDialog(self, self.project.style, self.project.name, "Project look")
        dlg.name.setEnabled(False)
        ok = dlg.exec() == QDialog.Accepted
        style = dlg.style()
        dlg.setParent(None)
        if ok:
            self.project.save_style(style)
            self.say("Saved the project's look. New assets use it; existing ones keep theirs.")

    # ---------------------------------------------------------------- browsing

    @Slot(int)
    def _chip_clicked(self, i: int):
        self.set_kind(KIND_CHIPS[i][1])

    def set_kind(self, kind: str | None):
        self.kind = kind
        for i, (_, k) in enumerate(KIND_CHIPS):
            if k == kind:
                self.chips.button(i).setChecked(True)
        self.refresh()

    def _fill_combo(self, box: QComboBox, everything: str, values: list[str]):
        cur = box.currentData()
        box.blockSignals(True)
        box.clear()
        box.addItem(everything, None)
        for v in values:
            box.addItem(v, v)
        i = box.findData(cur)
        box.setCurrentIndex(i if i >= 0 else 0)
        box.blockSignals(False)

    @Slot()
    def refresh(self, *_):
        """Re-read the project's assets with the current filters."""
        self._fill_combo(self.tag_box, "Any tag", A.all_tags(self.project))
        self._fill_combo(self.collection_box, "Any collection", A.all_collections(self.project))
        keep = Path(self.selected.folder) if self.selected is not None else None
        self.assets = A.list_assets(self.kind, self.search.text(), self.tag_box.currentData(),
                                    self.collection_box.currentData(), project=self.project)
        self.grid.blockSignals(True)
        self.grid.clear()
        row = -1
        for i, a in enumerate(self.assets):
            item = QListWidgetItem(_icon(asset_picture(a)), a.name)
            item.setToolTip(f"{a.name} ({a.kind})" + (f"\nTags: {', '.join(a.tags)}" if a.tags else ""))
            item.setSizeHint(QSize(THUMB + 24, THUMB + 36))
            self.grid.addItem(item)
            if keep is not None and Path(a.folder) == keep:
                row = i
        self.grid.blockSignals(False)
        self.empty.setVisible(not self.assets)
        if row >= 0:
            self.grid.setCurrentRow(row)
            self._show(self.assets[row])
        else:
            self._show(None)

    @Slot(int)
    def _row_changed(self, row: int):
        self._show(self.assets[row] if 0 <= row < len(self.assets) else None)

    def select(self, name_or_asset):
        """Select an asset by name (or the Asset)."""
        for i, a in enumerate(self.assets):
            if a is name_or_asset or a.name == name_or_asset or \
                    (hasattr(name_or_asset, "folder") and Path(a.folder) == Path(name_or_asset.folder)):
                self.grid.setCurrentRow(i)
                self._show(a)
                return a
        raise KeyError(f"No asset {name_or_asset!r} in the list")

    def _show(self, asset):
        self.selected = asset
        has = asset is not None
        for b in (self.export_btn, self.open_btn, self.delete_btn, self.tags, self.collections):
            b.setEnabled(has)
        self.reskin_btn.setEnabled(has and asset.kind == "character"
                                   and (Path(asset.folder) / "sheet.png").exists())
        self.export_coll_btn.setEnabled(bool(A.all_collections(self.project)))
        if not has:
            self.preview.show_rgba(None)
            self.preview.setText("Pick something")
            self.name_label.setText("")
            self.kind_label.setText("")
            self.tags.setText("")
            self.collections.setText("")
            return
        self.preview.show_rgba(asset_picture(asset))
        self.name_label.setText(asset.name)
        bits = [asset.kind]
        if asset.extra.get("body"):
            bits.append(f"{asset.extra['body']}, {asset.extra.get('size', '')}".strip(", "))
        if asset.extra.get("reskin_of"):
            bits.append(f"re-skin of {asset.extra['reskin_of'].get('name', '')}")
        self.kind_label.setText(" · ".join(bits))
        self.tags.setText(", ".join(asset.tags))
        self.collections.setText(", ".join(asset.collections))

    @Slot()
    def save_tags(self):
        if self.selected is None:
            return
        A.set_tags(self.selected, _words(self.tags.text()), _words(self.collections.text()))
        self.say(f"Saved tags for {self.selected.name}.")
        self.refresh()
        self.library_changed.emit()

    # ---------------------------------------------------------------- actions

    def _need(self):
        if self.selected is None:
            raise ValueError("Pick an asset first.")
        return self.selected

    def export_selected(self, preset: str, folder) -> list[Path]:
        a = self._need()
        files = exports.export_asset(a, preset, folder)
        self.say(f"Exported {a.name} for {exports.PRESETS[preset]}: {len(files)} files in {folder}")
        return files

    def export_collection_to(self, collection: str, preset: str, folder) -> list[Path]:
        files = exports.export_collection(collection, preset, folder, project=self.project)
        self.say(f"Exported “{collection}” ({len(files)} files) to {folder}")
        return files

    @Slot()
    def _export_clicked(self):
        a = self.selected
        if a is None:
            return
        dlg = ExportDialog(self, a.name, str(Path.home() / "Pixel Pipeline" / "Exports" / A.slug(a.name)))
        ok = dlg.exec() == QDialog.Accepted
        preset, folder = dlg.values()
        dlg.setParent(None)
        if ok and folder:
            self.run(lambda say: exports.export_asset(a, preset, folder), self._exported,
                     f"Exporting {a.name}…")

    @Slot()
    def _export_collection_clicked(self):
        colls = A.all_collections(self.project)
        if not colls:
            self.warn("No collections yet: type one in an asset's Collections box.")
            return
        dlg = ExportDialog(self, "collection")
        pick = QComboBox()
        pick.addItems(colls)
        if self.selected is not None and self.selected.collections:
            pick.setCurrentText(self.selected.collections[0])
        dlg.layout().insertRow(0, "Collection", pick)
        dlg.folder.setText(str(Path.home() / "Pixel Pipeline" / "Exports" / A.slug(pick.currentText())))
        ok = dlg.exec() == QDialog.Accepted
        preset, folder = dlg.values()
        name = pick.currentText()
        dlg.setParent(None)
        if ok and folder:
            project = self.project
            self.run(lambda say: exports.export_collection(name, preset, folder, project=project),
                     self._exported, f"Exporting {name}…")

    def _exported(self, files):
        if files:
            self.say(f"Exported {len(files)} files to {Path(files[0]).parent}")

    def reskin_selected(self, name: str, **opts):
        a = self._need()
        if a.kind != "character":
            raise ValueError("Re-skins are for characters.")
        ch = reskin.reskin(a.folder, name, **opts)
        self.say(f"Made {ch.name}.")
        self.refresh()
        try:
            self.select(ch.name)
        except KeyError:
            pass
        self.library_changed.emit()
        return ch

    @Slot()
    def _reskin_clicked(self):
        a = self.selected
        if a is None or a.kind != "character":
            return
        dlg = ReskinDialog(reskin._character(a.folder), self)
        ok = dlg.exec() == QDialog.Accepted
        name, opts = dlg.name.text().strip(), dlg.options()
        dlg.setParent(None)
        if ok and name:
            try:
                self.reskin_selected(name, **opts)
            except ValueError as e:
                self.warn(str(e))

    @Slot()
    def open_folder(self):
        if self.selected is not None and not self.sync:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.selected.folder)))

    def delete_selected(self, confirm: bool = True) -> bool:
        a = self._need()
        if confirm and not self.sync:
            answer = QMessageBox.question(self, "Delete", f"Delete {a.name} ({a.kind}) and all its "
                                          "files? This can't be undone.")
            if answer != QMessageBox.Yes:
                return False
        shutil.rmtree(a.folder, ignore_errors=True)
        self.say(f"Deleted {a.name}.")
        self.selected = None
        self.refresh()
        self.library_changed.emit()
        return True

    @Slot()
    def _delete_clicked(self):
        if self.selected is not None:
            self.delete_selected(confirm=True)

    # ---------------------------------------------------------------- queue

    def add_job(self, kind: str, name: str, prompt: str = "", variations: int = 1,
                options: dict | None = None):
        job = self.queue.add(kind, name, prompt, options, variations)
        self.refresh_queue()
        return job

    @Slot()
    def _add_job_clicked(self):
        try:
            self.add_job(self.job_kind.currentData(), self.job_name.text(), self.job_prompt.text(),
                         self.job_variations.value())
        except ValueError as e:
            self.warn(str(e))
            return
        self.job_name.clear()
        self.job_prompt.clear()

    def _selected_job(self):
        item = self.job_list.currentItem()
        return item.data(Qt.UserRole) if item is not None else None

    @Slot()
    def _remove_job_clicked(self):
        jid = self._selected_job()
        if jid:
            self.queue.remove(jid)
            self.refresh_queue()

    @Slot()
    def _retry_job_clicked(self):
        jid = self._selected_job()
        if jid:
            self.queue.retry(jid)
            self.refresh_queue()

    @Slot()
    def _clear_jobs_clicked(self):
        self.queue.clear_finished()
        self.refresh_queue()

    @Slot(bool)
    def show_queue(self, shown: bool = True):
        """Unfold (or fold) the queue panel."""
        self.queue_inner.setVisible(shown)
        self.queue_toggle.blockSignals(True)
        self.queue_toggle.setChecked(shown)
        self.queue_toggle.blockSignals(False)
        self.queue_toggle.setText(("▾ " if shown else "▸ ") + self.queue_toggle.text()[2:])

    def refresh_queue(self):
        cur = self._selected_job()
        self.job_list.clear()
        colours = {"done": "#6fcf7f", "failed": "#e07070", "running": "#7fb2ff"}
        for j in self.queue.jobs:
            item = QListWidgetItem(j.summary())
            item.setData(Qt.UserRole, j.id)
            if j.status in colours:
                item.setForeground(QColor(colours[j.status]))
            self.job_list.addItem(item)
            if j.id == cur:
                self.job_list.setCurrentItem(item)
        running = self.queue_running()
        self.start_btn.setEnabled(not running and bool(self.queue.pending()))
        self.stop_btn.setEnabled(running)

    def queue_running(self) -> bool:
        return self._qthread is not None and self._qthread.isRunning()

    @Slot()
    def start_queue(self):
        """Work through the queue (on a QThread; right here when sync)."""
        if self.queue_running() or not self.queue.pending():
            return
        self._stop.clear()
        workers = self.worker_fns
        if self.sync:
            jobqueue.run_queue(self.project, workers, self.pause.value(), self._stop,
                               on_update=self._job_updated, queue=self.queue)
            self._queue_finished()
            return
        thread, worker = QThread(self), _QueueWorker(self.project, self.queue, workers,
                                                     self.pause.value(), self._stop)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.update.connect(self._job_updated)
        worker.failed.connect(self._queue_failed)
        worker.finished.connect(thread.quit)
        thread.finished.connect(self._queue_finished)
        thread.finished.connect(worker.deleteLater)
        self._qthread, self._qworker = thread, worker
        thread.start()
        self.say("Queue started.")
        self.refresh_queue()

    @Slot()
    def stop_queue(self):
        self._stop.set()
        self.say("Stopping after the current one…")

    @Slot(object)
    def _job_updated(self, job):
        self.refresh_queue()
        if job.results and job.status in ("running", "done"):
            self.refresh()
            self.library_changed.emit()

    @Slot(str)
    def _queue_failed(self, message: str):
        self.say("Queue stopped: " + message.splitlines()[0])

    @Slot()
    def _queue_finished(self):
        self._qthread = self._qworker = None
        self.refresh_queue()
        self.refresh()
        left = len(self.queue.pending())
        self.say("Queue finished." if not left else f"Queue stopped; {left} still to do.")

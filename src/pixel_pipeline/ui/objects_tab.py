"""Items and Props tabs: make inventory items (icon + world sprite) or props and buildings
(3D model -> sprites at the project's scale), see them, keep them in the project.

Left: this kind's assets in the current project (objects.py / assets.py), click one to see it.
Top: describe it (Make, your GPU), or start from a photo or a 3D model. Centre: items show the
icon variations to pick from, the world sprite turning through its directions and the model
spinning; props show the picture, the model and the sprite. Bottom: name, category / size
(items) or height next to a person (props), directions, the item's look (flat from the icon or
3D) or the props' hand-pixelled finish, Make; Redo sprites re-renders with new settings from the
saved model or picture (no GPU).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QIcon, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton, QSplitter, QToolButton,
    QVBoxLayout, QWidget,
)

from .. import objects as O
from .common import PixelView, ToolTab, checkerboard, to_pixmap

IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
MODEL_FILTER = "3D models (*.glb *.gltf *.obj *.ply *.stl)"
WORDS = {"item": ("item", "items", "e.g. an iron sword with a leather grip"),
         "prop": ("prop", "props and buildings", "e.g. a small stone cottage with a thatched roof")}


def _rgba(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGBA"))


def previews(asset, elevation: float = 30.0) -> dict:
    """What the tab shows for an asset, read from its folder (and the model spun once)."""
    f = asset.folder
    out = {"asset": asset, "icons": [_rgba(p) for p in O.icon_paths(asset)],
           "world": O.load_world(asset), "spin": [], "picture": None}
    for name in ("model.front.png", "picture_v1.png"):
        if (f / name).exists():
            out["picture"] = _rgba(f / name)
            break
    model = f / "model.glb"
    if model.exists():
        from ..turnaround import render_model, union_crop
        out["spin"] = union_crop(render_model(model, 16, elevation, size=256, points=400_000))
    return out


class ObjectsTab(ToolTab):
    """Items (kind "item") or props and buildings (kind "prop"). `changed` fires whenever an
    asset of this kind is made or changed (icon picked, sprites redone)."""

    changed = Signal()

    def __init__(self, kind: str = "item", status=None):
        super().__init__(status)
        if kind not in WORDS:
            raise ValueError("kind must be 'item' or 'prop'")
        self.kind = kind
        self.title = "Items" if kind == "item" else "Props"
        self.client = None                  # ComfyClient; None = the default address
        self.asset = None                   # the asset on show
        self.shown: dict = {}
        self._world_i = self._spin_i = 0
        self._setting = False
        self.var_buttons: list[QToolButton] = []
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._build()
        self.refresh_library()

    @property
    def hint(self) -> str:
        one, many, _ = WORDS[self.kind]
        return (f"Describe an {one} and press Make (your GPU), or start from a photo or a 3D "
                f"model. Click one of your {many} on the left to see it.") if self.kind == "item" \
            else (f"Describe a prop or building and press Make (your GPU), or start from a photo "
                  f"or a 3D model. Click one of your {many} on the left to see it.")

    # ---- layout ---------------------------------------------------------------
    def _build(self):
        one, many, example = WORDS[self.kind]
        self.prompt = QLineEdit()
        self.prompt.setPlaceholderText(example)
        self.prompt.returnPressed.connect(self.make)
        self.photo_btn = QPushButton("From a photo…")
        self.photo_btn.setToolTip("A photo or drawing of the whole thing on a plain background")
        self.photo_btn.clicked.connect(lambda: self.from_picture())
        self.model_btn = QPushButton("From a 3D model…")
        self.model_btn.setToolTip("A .glb/.gltf/.obj from Blender, Tripo, Meshy… (no GPU needed)")
        self.model_btn.clicked.connect(lambda: self.from_model())
        top = QHBoxLayout()
        top.addWidget(self.prompt, 1)
        top.addWidget(self.photo_btn)
        top.addWidget(self.model_btn)

        # previews
        self.icon_view = PixelView("Icon", (200, 200))
        self.world_view = PixelView("In the world (every direction)", (200, 200))
        self.model_view = PixelView("3D model (spins)", (200, 200))
        self.picture_view = PixelView("Picture", (200, 200))
        self.var_group = QButtonGroup(self)
        self.var_group.setExclusive(True)
        self.var_group.idClicked.connect(self.pick_icon)
        self.var_row = QHBoxLayout()
        self.var_row.addStretch(1)
        panels = QHBoxLayout()
        if self.kind == "item":
            icon_box = QGroupBox("Icon (pick one)")
            il = QVBoxLayout(icon_box)
            il.addWidget(self.icon_view, 1)
            il.addLayout(self.var_row)
            panels.addWidget(icon_box, 1)
            boxes = (("In the world", self.world_view), ("3D model", self.model_view))
        else:
            boxes = (("Picture", self.picture_view), ("3D model", self.model_view),
                     ("In the game (every direction)", self.world_view))
        for title, view in boxes:
            box = QGroupBox(title)
            QVBoxLayout(box).addWidget(view)
            panels.addWidget(box, 1)

        # bottom row
        self.name = QLineEdit()
        self.name.setMinimumWidth(160)
        self.name.setPlaceholderText("Name, e.g. Iron Sword" if self.kind == "item"
                                     else "Name, e.g. Stone Cottage")
        self.directions = QComboBox()
        self.directions.addItem("Auto", None)
        for n in O.DIRECTION_CHOICES:
            self.directions.addItem("1 (flat)" if n == 1 and self.kind == "item" else str(n), n)
        self.directions.setToolTip("How many facings the sprite gets. Auto: the project's "
                                   "directions for items; props: 4 for buildings, 1 for trees, "
                                   "barrels and other things that look the same all round")
        keep = QHBoxLayout()
        keep.addWidget(QLabel("Name"))
        keep.addWidget(self.name, 1)
        if self.kind == "item":
            self.category = QComboBox()
            self.category.addItems(list(O.CATEGORIES))
            self.size_ = QComboBox()
            self.size_.addItems(list(O.ITEM_SIZES))
            self.size_.setCurrentText("medium")
            self.size_.setToolTip("How big it looks lying in the world, next to a person")
            self.world = QCheckBox("World sprite")
            self.world.setChecked(True)
            self.world.setToolTip("Also a sprite of the item lying in the world")
            self.look = QComboBox()
            for text, data in (("Auto", "auto"), ("Flat (from icon)", "flat"),
                               ("3D model", "3d")):
                self.look.addItem(text, data)
            self.look.setToolTip("How the world sprite is drawn. Flat (from icon): the icon's "
                                 "picture laid on the ground, thin parts thickened so they read "
                                 "(no 3D, no GPU). 3D model: rendered from a 3D model (GPU to "
                                 "make one). Auto: flat for weapons and keys, 3D for potions, "
                                 "food and armour")
            for w in (QLabel("Category"), self.category, QLabel("Size"), self.size_,
                      QLabel("Directions"), self.directions, self.world, QLabel("Look"),
                      self.look):
                keep.addWidget(w)
        else:
            self.height_ = QDoubleSpinBox()
            self.height_.setRange(0.1, 12.0)
            self.height_.setSingleStep(0.1)
            self.height_.setDecimals(1)
            self.height_.setValue(1.0)
            self.height_.setSuffix(" people")
            self.height_.setToolTip("How tall it stands next to a person (a barrel 0.6, a tree "
                                    "2.5, a house 3)")
            self.height_.valueChanged.connect(self._height_edited)
            self.like = QComboBox()
            self.like.addItem("Guess from the words", None)
            for names, h, d in O.PROP_PRESETS:
                self.like.addItem(f"like a {names[0]} ({h:g})", (h, d))
            self.like.addItem("Your own height", "own")
            self.like.currentIndexChanged.connect(self._like_changed)
            self.hand = QCheckBox("Hand-pixelled")
            self.hand.setChecked(True)
            self.hand.setToolTip("Finish the sprites the way a pixel artist would: a few colours "
                                 "per material with warm lights and cool shadows, clean light "
                                 "and shadow sides, specks cleaned up, a dark outline per "
                                 "material. Off: the plain render")
            for w in (QLabel("Height"), self.height_, self.like, QLabel("Directions"),
                      self.directions, self.hand):
                keep.addWidget(w)
        self.make_btn = QPushButton("Make")
        self.make_btn.setStyleSheet("font-weight:600;")
        self.make_btn.setDefault(True)
        self.make_btn.clicked.connect(self.make)
        self.redo_btn = QPushButton("Redo sprites")
        self.redo_btn.setToolTip("Render again with these settings from the saved 3D model "
                                 "(no GPU)")
        self.redo_btn.clicked.connect(self.redo)
        self.redo_btn.setEnabled(False)
        keep.addWidget(self.make_btn)
        keep.addWidget(self.redo_btn)
        self.turn_btn = None
        if self.kind == "prop":
            self.turn_btn = QPushButton("Turn 90°")
            self.turn_btn.setToolTip("Make another side the front (S) and render again (no GPU)")
            self.turn_btn.clicked.connect(lambda: self.redo(turn=90))
            self.turn_btn.setEnabled(False)
            keep.addWidget(self.turn_btn)

        # library
        self.library = QListWidget()
        self.library.setViewMode(QListWidget.IconMode)
        self.library.setIconSize(QSize(64, 64))
        self.library.setGridSize(QSize(92, 96))
        self.library.setResizeMode(QListWidget.Adjust)
        self.library.setMovement(QListWidget.Static)
        self.library.setMinimumWidth(200)
        self.library.itemClicked.connect(self._item_clicked)
        self.lib_title = QLabel()
        self.lib_title.setStyleSheet("font-weight:600;")
        new_btn = QPushButton("New")
        new_btn.clicked.connect(self.new_object)
        folder_btn = QPushButton("Show folder")
        folder_btn.clicked.connect(self.show_folder)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(self.lib_title)
        ll.addWidget(self.library, 1)
        lb = QHBoxLayout()
        lb.addWidget(new_btn)
        lb.addWidget(folder_btn)
        ll.addLayout(lb)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addLayout(top)
        rl.addLayout(panels, 1)
        rl.addLayout(keep)
        split = QSplitter()
        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setSizes([220, 1000])
        outer = QVBoxLayout(self)
        outer.addWidget(split)
        self.busy_widgets = [self.make_btn, self.redo_btn, self.photo_btn, self.model_btn]
        if self.turn_btn is not None:
            self.busy_widgets.append(self.turn_btn)

    # ---- prop height presets --------------------------------------------------
    def _like_changed(self, _i):
        data = self.like.currentData()
        if isinstance(data, tuple):
            self._setting = True
            self.height_.setValue(data[0])
            self.directions.setCurrentIndex(self.directions.findData(data[1]))
            self._setting = False

    def _height_edited(self, _v):
        if not self._setting:
            self.like.blockSignals(True)
            self.like.setCurrentIndex(self.like.count() - 1)      # "Your own height"
            self.like.blockSignals(False)

    # ---- library --------------------------------------------------------------
    def project(self):
        from ..assets import current_project
        return current_project()

    def refresh_library(self, select=None):
        from ..assets import list_assets
        self.library.clear()
        items = list_assets(self.kind, project=self.project())
        for a in items:
            it = QListWidgetItem(a.name)
            if a.thumb.exists():
                it.setIcon(QIcon(QPixmap(str(a.thumb))))
            it.setData(Qt.UserRole, str(a.folder))
            tip = a.extra.get("category") or (f"{a.extra.get('height_people', '?')} people tall")
            it.setToolTip(f"{a.name}: {tip}. {a.extra.get('prompt', '')}")
            self.library.addItem(it)
            if select is not None and a.folder == select.folder:
                self.library.setCurrentItem(it)
        self.lib_title.setText(f"Your {WORDS[self.kind][1]} ({len(items)})")

    def show_folder(self):
        folder = self.asset.folder if self.asset else self.project().kind_folder(self.kind)
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def new_object(self):
        self.asset = None
        self.shown = {}
        self._timer.stop()
        self.name.clear()
        self.prompt.clear()
        for v in (self.icon_view, self.world_view, self.model_view, self.picture_view):
            v.show_rgba(None)
        self._set_variations([])
        self.library.clearSelection()
        self.redo_btn.setEnabled(False)
        if self.turn_btn is not None:
            self.turn_btn.setEnabled(False)
        self.say(self.hint)

    def _item_clicked(self, item):
        self.open_asset(O.open_asset(item.data(Qt.UserRole)))

    def open_asset(self, asset):
        """Show an asset of this kind from its folder."""
        self.asset = asset
        self.name.setText(asset.name)
        self.prompt.setText(asset.extra.get("prompt", ""))
        elev = self.project().style.elevation
        self.run(lambda say: previews(asset, elev), self._show, f"Opening {asset.name}…")

    # ---- making ---------------------------------------------------------------
    def _directions(self):
        return self.directions.currentData()

    def _job(self, prompt=None, picture=None, model=None):
        name = self.name.text().strip() or (prompt or "").strip()[:30] or \
            Path(str(picture or model or self.kind)).stem.replace("_", " ").title()
        project, client, directions = self.project(), self.client, self._directions()
        elev = project.style.elevation
        if self.kind == "item":
            opts = dict(category=self.category.currentText(), size=self.size_.currentText(),
                        world=self.world.isChecked(), directions=directions,
                        style=self.look.currentData())
            maker = O.make_item
        else:
            own = self.like.currentData() is not None
            opts = dict(height_people=self.height_.value() if own else None,
                        directions=directions, hand=self.hand.isChecked())
            maker = O.make_prop

        def job(say):
            asset = maker(name, prompt, picture, model, project=project, progress=say,
                          client=client, **opts)
            return previews(asset, elev)
        return job

    def make(self):
        prompt = self.prompt.text().strip()
        if not prompt:
            self.say("Type what you want first.")
            return
        self.run(self._job(prompt=prompt), self._made, f"Making “{prompt}”…")

    def from_picture(self, path=None):
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, "A photo or drawing", "", IMAGE_FILTER)
        if path:
            self.run(self._job(prompt=self.prompt.text().strip() or None, picture=str(path)),
                     self._made, f"Making from {Path(path).name}…")

    def from_model(self, path=None):
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, "3D model", "", MODEL_FILTER)
        if path:
            self.run(self._job(prompt=self.prompt.text().strip() or None, model=str(path)),
                     self._made, f"Making from {Path(path).name}…")

    def redo(self, turn: float = 0.0):
        """Render the sprites again with the bottom row's settings (no GPU); props: turn =
        degrees more to turn the model (another side becomes the front)."""
        asset = self.asset
        if asset is None:
            return
        project, directions = self.project(), self._directions()
        if self.kind == "item":
            opts = dict(size=self.size_.currentText(), directions=directions,
                        style=self.look.currentData())
        else:
            opts = dict(height_people=self.height_.value(), directions=directions,
                        hand=self.hand.isChecked())
            if turn:
                opts["yaw"] = float(asset.extra.get("yaw") or 0.0) + turn
        elev = project.style.elevation

        def job(say):
            O.rerender(asset, project=project, progress=say, **opts)
            if asset.kind == "item" and (asset.folder / "picture_v1.png").exists():
                O.redo_icons(asset, project)          # the project's style may have changed
            return previews(asset, elev)
        self.run(job, self._made, f"Redoing {asset.name}…")

    def _made(self, payload):
        self._show(payload)
        self.refresh_library(select=payload["asset"])
        self.changed.emit()

    def pick_icon(self, index: int):
        """Make icon variation `index` (1-based) the item's icon.png."""
        if self.asset is None or self.kind != "item":
            return
        asset, project = self.asset, self.project()
        O.choose_icon(asset, index, project)
        for i, b in enumerate(self.var_buttons, 1):
            b.setChecked(i == index)
        self.shown["icons"] = [_rgba(p) for p in O.icon_paths(asset)]
        self.icon_view.show_rgba(_rgba(asset.folder / "icon.png"))
        if asset.extra.get("world_from") in ("picture", "icon"):
            self.shown["world"] = O.load_world(asset)
            self._world_i = 0
            self._tick()
        self.refresh_library(select=asset)
        self.say(f"{asset.name}: icon {index} is the icon now.")
        self.changed.emit()

    # ---- showing --------------------------------------------------------------
    def _set_variations(self, icons: list[np.ndarray], choice: int = 1):
        for b in self.var_buttons:
            self.var_group.removeButton(b)
            self.var_row.removeWidget(b)
            b.setParent(None)
        self.var_buttons = []
        for i, rgba in enumerate(icons, 1):
            b = QToolButton()
            b.setCheckable(True)
            k = max(1, 48 // max(rgba.shape[:2]))
            big = np.repeat(np.repeat(rgba, k, 0), k, 1)
            b.setIcon(QIcon(to_pixmap(checkerboard(big, cell=max(4, k * 2)))))
            b.setIconSize(QSize(big.shape[1], big.shape[0]))
            b.setToolTip(f"Use icon {i}")
            b.setChecked(i == choice)
            self.var_group.addButton(b, i)
            self.var_row.insertWidget(self.var_row.count() - 1, b)
            self.var_buttons.append(b)

    def _show(self, payload):
        self.shown = payload
        asset = self.asset = payload["asset"]
        ex = asset.extra
        self.name.setText(asset.name)
        if self.kind == "item":
            choice = int(ex.get("icon_choice", 1))
            self._set_variations(payload["icons"], choice)
            icon = asset.folder / "icon.png"
            self.icon_view.show_rgba(_rgba(icon) if icon.exists() else None)
            if ex.get("category") in O.CATEGORIES:
                self.category.setCurrentText(ex["category"])
            if ex.get("size") in O.ITEM_SIZES:
                self.size_.setCurrentText(ex["size"])
            j = self.look.findData(ex.get("world_style", "auto"))
            self.look.setCurrentIndex(max(0, j))
        else:
            self._setting = True
            self.height_.setValue(float(ex.get("height_people", 1.0)))
            self._setting = False
            self.hand.setChecked(bool(ex.get("hand_pixel", True)))
            self.picture_view.show_rgba(payload["picture"])
        i = self.directions.findData(ex.get("directions"))
        if i >= 0 and ex.get("directions"):
            self.directions.setCurrentIndex(i)
        for view, frames, empty in ((self.world_view, payload["world"], "No world sprite"),
                                    (self.model_view, payload["spin"], "No 3D model")):
            if not frames:
                view.show_rgba(None)
                view.setText(empty)
        self._world_i = self._spin_i = 0
        self._tick()
        if payload["world"] or payload["spin"]:
            self._timer.start(120)
        self.redo_btn.setEnabled((asset.folder / "model.glb").exists()
                                 or (self.kind == "item" and bool(payload["icons"])))
        if self.turn_btn is not None:
            self.turn_btn.setEnabled((asset.folder / "model.glb").exists())
        what = f"{len(payload['world'])} direction(s)" if payload["world"] else "no world sprite"
        seeds = ex.get("seeds") or []
        seed = f" Seeds {', '.join(map(str, seeds))}." if seeds else ""
        self.say(f"{asset.name}: {what}.{seed} Files: {asset.folder}")

    def _tick(self):
        """One step of the spinning previews (world sprite through its directions, model)."""
        world, spin = self.shown.get("world") or [], self.shown.get("spin") or []
        if world:
            self.world_view.show_rgba(world[self._world_i % len(world)])
            self._world_i += 1
        if spin:
            self.model_view.show_rgba(spin[self._spin_i % len(spin)])
            self._spin_i += 1

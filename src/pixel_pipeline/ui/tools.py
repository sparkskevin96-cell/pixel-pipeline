"""The desktop app's tool tabs: Prompt, Directions, Teams, Portrait, Map tiles, Godot demo.

Each tab is a form on the left, a big pixel preview on the right, and Make / Save buttons;
slow work runs off the UI thread (ToolTab.run)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QTimer, QUrl
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget, QPushButton,
    QRadioButton, QSplitter, QVBoxLayout, QWidget,
)

from ..settings import settings_for
from ..turnaround import DIRECTION_COUNTS
from .common import PixelView, SpriteOptions, ToolTab, grid, spin

SOURCE_FILTER = ("3D models, photos or a video (*.glb *.gltf *.obj *.ply *.stl *.png *.jpg *.jpeg "
                 "*.webp *.heic *.mp4 *.mov *.m4v *.webm *.mkv);;All files (*)")
IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.webp *.heic *.bmp)"
MODEL_FILTER = "3D models (*.glb *.gltf *.obj *.ply *.stl)"
BUILTIN_TERRAINS = ["grass", "water", "sand", "road", "dirt", "stone", "snow"]


def _row(*widgets) -> QWidget:
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    for x in widgets:
        lay.addWidget(x)
    return w


def _layout(tab: QWidget, left: list[QWidget], right: QWidget, buttons: list[QWidget]):
    """Form column on the left, preview on the right, buttons under the form."""
    column = QWidget()
    lay = QVBoxLayout(column)
    for w in left:
        lay.addWidget(w)
    lay.addStretch(1)
    for b in buttons:
        lay.addWidget(b)
    column.setMinimumWidth(320)
    column.setMaximumWidth(420)
    split = QSplitter()
    split.addWidget(column)
    split.addWidget(right)
    split.setStretchFactor(1, 1)
    outer = QVBoxLayout(tab)
    outer.addWidget(split)


def open_folder(path: Path):
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


class _Player:
    """Plays an AnimatedSet in a PixelView: every direction at once, each action twice."""

    def __init__(self, view: PixelView, columns: int = 8):
        self.view, self.columns = view, columns
        self.anim = None
        self.i = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.step)

    def play(self, anim):
        self.anim, self.i = anim, 0
        self.step()
        self.timer.start(anim.actions[0].frame_ms)

    def stop(self):
        self.timer.stop()
        self.anim = None

    def step(self):
        a = self.anim
        if a is None:
            return
        per = 2 * a.frames
        k = self.i % (per * len(a.actions))
        act, f = divmod(k, per)
        self.view.show_rgba(grid([d[f % a.frames].rgba for d in a.sprites[act]], self.columns))
        self.view.setToolTip(f"{a.actions[act].name}: clockwise from facing right")
        self.timer.setInterval(a.actions[act].frame_ms)
        self.i = k + 1


# --------------------------------------------------------------------------- directions

class DirectionsTab(ToolTab):
    """3D model / photos / video -> 4, 8 or 16 facing directions, animated for rigged models."""
    title = "Directions"

    def __init__(self, status=None):
        super().__init__(status)
        self.sources: list[Path] = []
        self.colours_from: Path | None = None
        self.sprites = None          # list[Sprite] (still) ...
        self.anim = None             # ... or AnimatedSet
        self.clip_names: list[str] = []
        self.setAcceptDrops(True)

        self.files = QListWidget()
        self.files.setMaximumHeight(110)
        add = QPushButton("Add model, photos or video…")
        add.clicked.connect(self.add_dialog)
        clear = QPushButton("Clear")
        clear.clicked.connect(self.clear_sources)
        src = QGroupBox("Source")
        sl = QVBoxLayout(src)
        sl.addWidget(QLabel("A 3D model (.glb) is rendered from every side. Photos: one per "
                            "direction, starting in front and walking to your right. Or a video "
                            "of that walk."))
        sl.itemAt(0).widget().setWordWrap(True)
        sl.addWidget(self.files)
        sl.addWidget(_row(add, clear))

        self.options = SpriteOptions("person", 48, 48, 24)
        self.directions = QComboBox()
        self.directions.addItems([str(n) for n in reversed(DIRECTION_COUNTS)])
        self.elevation = QDoubleSpinBox()
        self.elevation.setRange(0, 89)
        self.elevation.setValue(30)
        self.elevation.setSuffix("°")
        self.elevation.setToolTip("0 = side view, 30 = top-down / isometric")
        self.yaw = QDoubleSpinBox()
        self.yaw.setRange(-180, 180)
        self.yaw.setSingleStep(90)
        self.yaw.setSuffix("°")
        self.yaw.setToolTip("Turn the model if S doesn't face the viewer")
        self.light = QDoubleSpinBox()
        self.light.setRange(0, 1)
        self.light.setSingleStep(0.05)
        self.light.setValue(0.35)
        self.animate = QCheckBox("Play its animations")
        self.animate.setEnabled(False)
        self.autorig = QCheckBox("Add a skeleton + animations (humanoid)")
        self.autorig.setEnabled(False)
        self.autorig.setToolTip("For a standing character without animations (e.g. from Prompt to "
                                "Model): finds the joints and adds idle, walk, run, attack, hurt "
                                "and death. Best in an A- or T-pose.")
        self.rig = None
        self.frames = spin(1, 32, 8)
        self.animations = QLineEdit()
        self.animations.setPlaceholderText("all (or e.g. idle,walk)")
        self.colours_btn = QPushButton("Colours from…")
        self.colours_btn.setToolTip("The original coloured model, when rigging it (Mixamo) "
                                    "dropped its colours")
        self.colours_btn.clicked.connect(self.pick_colours_from)
        self.reverse = QCheckBox("I walked to the left")
        self.mirror = QCheckBox("Only half way round (mirror the rest)")
        look = QGroupBox("Sprites")
        fl = QFormLayout(look)
        fl.addRow(self.options)
        fl.addRow("Directions", self.directions)
        three = QGroupBox("3D model")
        tl = QFormLayout(three)
        tl.addRow("Camera tilt", self.elevation)
        tl.addRow("Turn", self.yaw)
        tl.addRow("Shading", self.light)
        tl.addRow(self.animate)
        tl.addRow(self.autorig)
        tl.addRow("Frames each", self.frames)
        tl.addRow("Animations", self.animations)
        tl.addRow(self.colours_btn)
        photos = QGroupBox("Photos / video")
        pl = QVBoxLayout(photos)
        pl.addWidget(self.reverse)
        pl.addWidget(self.mirror)

        self.view = PixelView("Add a 3D model, photos or a video, then Make sprites")
        self.player = _Player(self.view)
        self.make_btn = QPushButton("Make sprites")
        self.make_btn.clicked.connect(self.make)
        self.skel_btn = QPushButton("Show the skeleton")
        self.skel_btn.setEnabled(False)
        self.skel_btn.clicked.connect(self.show_skeleton)
        self.save_btn = QPushButton("Save sheet + Aseprite + Godot…")
        self.save_btn.clicked.connect(self.save_dialog)
        self.save_btn.setEnabled(False)
        self.busy_widgets = [self.make_btn, self.save_btn]
        _layout(self, [src, look, three, photos], self.view, [self.make_btn, self.skel_btn, self.save_btn])

    # ---- sources
    def add_dialog(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "3D model, photos or a video", "", SOURCE_FILTER)
        if paths:
            self.add_sources([Path(p) for p in paths])

    def add_sources(self, paths: list[Path]):
        for p in paths:
            if p not in self.sources:
                self.sources.append(p)
                self.files.addItem(p.name)
        self._check_animations()

    def clear_sources(self):
        self.sources = []
        self.files.clear()
        self._check_animations()

    def pick_colours_from(self):
        path, _ = QFileDialog.getOpenFileName(self, "Original coloured model", "", MODEL_FILTER)
        self.colours_from = Path(path) if path else None
        self.colours_btn.setText(f"Colours from: {self.colours_from.name}" if self.colours_from
                                 else "Colours from…")

    def _check_animations(self):
        from .model_studio import animation_names
        rigged = [p for p in self.sources if p.suffix.lower() in (".glb", ".gltf")]
        self.clip_names = animation_names(rigged[0]) if rigged else []
        if rigged and len(self.sources) > 1:
            try:
                from ..anim3d import load_rig
                self.clip_names = [c.name for c in load_rig(rigged).clips]
            except (ValueError, KeyError, OSError):
                pass
        self.animate.setEnabled(bool(self.clip_names))
        self.animate.setChecked(bool(self.clip_names))
        one_model = len(self.sources) == 1 and self.sources[0].suffix.lower() in (
            ".glb", ".gltf", ".obj", ".ply", ".stl")
        self.autorig.setEnabled(one_model and not self.clip_names)
        if not self.autorig.isEnabled():
            self.autorig.setChecked(False)
        self.animate.setText(f"Play its animations ({', '.join(self.clip_names)})"
                             if self.clip_names else "Play its animations")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        self.add_sources([Path(u.toLocalFile()) for u in event.mimeData().urls()])

    # ---- make / save
    def make(self):
        if not self.sources:
            self.say("Add a 3D model, photos or a video first.")
            return
        n = int(self.directions.currentText())
        s = self.options.settings()
        sources = list(self.sources)
        animate = self.animate.isChecked() and self.animate.isEnabled()
        autorig = self.autorig.isChecked() and self.autorig.isEnabled()
        names = [x.strip() for x in self.animations.text().split(",") if x.strip()] or None
        el, yaw, light = self.elevation.value(), self.yaw.value(), self.light.value()
        frames, colours_from = self.frames.value(), self.colours_from
        reverse, mirror = self.reverse.isChecked(), self.mirror.isChecked()

        def job(say):
            if autorig:
                import tempfile

                from ..anim3d import animated_turnaround
                from ..autorig import ACTIONS, auto_rig
                say("Finding the joints and adding animations…")
                wanted = [x for x in (names or ACTIONS) if x in ACTIONS] or list(ACTIONS)
                tmp = Path(tempfile.mkdtemp(prefix="pixel-pipeline-rig-"))
                rig = auto_rig(sources[0], tmp / (sources[0].stem + ".rigged.glb"), actions=wanted, yaw=yaw)
                say(f"Playing {len(wanted)} animations from {n} directions…")
                made = animated_turnaround([rig.path], s, n, frames, None, elevation=el, light=light,
                                           colours_from=colours_from)
                return (*made, rig)
            if animate:
                from ..anim3d import animated_turnaround
                say(f"Playing the animations from {n} directions…")
                return animated_turnaround(sources, s, n, frames, names, elevation=el, yaw=yaw,
                                           light=light, colours_from=colours_from)
            from ..turnaround import turnaround_from
            say(f"Making {n} directions…")
            return turnaround_from(sources, s, n, elevation=el, yaw=yaw, light=light,
                                   reverse=reverse, mirror=mirror)
        self.run(job, self._made, "Working…")

    def _made(self, result):
        made, what = result[:2]
        self.rig = result[2] if len(result) > 2 else None
        self.skel_btn.setEnabled(self.rig is not None)
        self.player.stop()
        if isinstance(made, list):
            self.sprites, self.anim = made, None
            self.view.show_rgba(grid([sp.rgba for sp in made], 4))
            first = made[0]
            count = f"{len(made)} directions"
        else:
            self.sprites, self.anim = None, made
            self.player.columns = 4
            self.player.play(made)
            first = made.sprites[0][0][0]
            count = (f"{len(made.actions)} animations x {len(made.directions)} directions x "
                     f"{made.frames} frames")
        w, h = first.size
        self.save_btn.setEnabled(True)
        self.say(f"{what} -> {count}, {w}x{h}, {len(first.palette)} colours. Save when it looks right.")

    def show_skeleton(self):
        if self.rig is not None:
            self.player.stop()
            self.view.show_rgba(self.rig.preview)
            notes = "; ".join(self.rig.skeleton.notes)
            self.say("The joints found (green = left side, red = right)." + (f" Note: {notes}" if notes else "")
                     + " Make sprites again to play the animations.")

    def save_dialog(self):
        stem = self.sources[0].stem if self.sources else "sprite"
        default = str((self.sources[0].parent if self.sources else Path.cwd()) / f"{stem}_dirs.png")
        path, _ = QFileDialog.getSaveFileName(self, "Save sprite sheet", default, "Sprite sheet (*.png)")
        if path:
            self.save(Path(path))

    def save(self, sheet: Path) -> list[Path]:
        sheet.parent.mkdir(parents=True, exist_ok=True)
        try:
            if self.anim is not None:
                import shutil

                from ..anim3d import write_animated, write_animated_preview
                written = write_animated(self.anim, sheet, aseprite=True, godot=True)
                if self.rig is not None:     # the rigged model too, for Godot 3D / Blender
                    glb = sheet.with_name(sheet.stem + ".rigged.glb")
                    shutil.copyfile(self.rig.path, glb)
                    written.append(glb)
                written.append(write_animated_preview(self.anim, sheet.with_name(sheet.stem + "_anim.gif")))
            else:
                from ..turnaround import write_preview, write_turnaround
                written = write_turnaround(self.sprites, sheet, aseprite=True, godot=True)
                written.append(write_preview(self.sprites, sheet.with_name(sheet.stem + "_spin.gif")))
        except Exception as e:  # noqa: BLE001
            self.warn(f"Save failed: {e}")
            return []
        self.say("Saved " + ", ".join(p.name for p in written))
        return written


# --------------------------------------------------------------------------- teams

class TeamsTab(ToolTab):
    """A sprite or sheet -> player / enemy / ally / other / spent colours + Godot shader."""
    title = "Team colours"

    def __init__(self, status=None, current_sprite=None):
        super().__init__(status)
        self.current_sprite = current_sprite or (lambda: None)
        self.rgba: np.ndarray | None = None
        self.path: Path | None = None
        self.colour: str | None = None
        self.result = None
        open_btn = QPushButton("Open sprite or sheet…")
        open_btn.clicked.connect(self.open_dialog)
        use_btn = QPushButton("Use the Sprite tab's sprite")
        use_btn.clicked.connect(self.use_current)
        self.colour_btn = QPushButton("Team colour: automatic")
        self.colour_btn.setToolTip("The colour family that changes per team. Automatic = the main "
                                   "saturated colour (armour, tunic), not gold trim or skin.")
        self.colour_btn.clicked.connect(self.pick_colour)
        auto_btn = QPushButton("Automatic")
        auto_btn.clicked.connect(lambda: self.set_colour(None))
        box = QGroupBox("Sprite")
        bl = QVBoxLayout(box)
        note = QLabel("Fire Emblem style: one sheet, recoloured blue / red / green / purple, and grey "
                      "when a unit has moved. Godot gets a palette-swap shader, so one sheet serves "
                      "every side.")
        note.setWordWrap(True)
        bl.addWidget(note)
        bl.addWidget(open_btn)
        bl.addWidget(use_btn)
        bl.addWidget(_row(self.colour_btn, auto_btn))
        self.view = PixelView("Open a sprite or a sheet")
        self.save_btn = QPushButton("Save team PNGs + Godot shader…")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_dialog)
        self.busy_widgets = [self.save_btn]
        _layout(self, [box], self.view, [self.save_btn])

    def open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Sprite or sheet", "", "PNG (*.png)")
        if path:
            self.open_path(Path(path))

    def open_path(self, path: Path):
        from ..export import load_sprite
        self.set_sprite(load_sprite(path), path)

    def use_current(self):
        sprite = self.current_sprite()
        if sprite is None:
            self.say("Make a sprite in the Sprite tab first.")
            return
        self.set_sprite(sprite.rgba, None)

    def set_sprite(self, rgba: np.ndarray, path: Path | None):
        self.rgba, self.path = rgba, path
        self.refresh()

    def pick_colour(self):
        c = QColorDialog.getColor(QColor(self.colour or "#2f5fd8"), self, "Team colour")
        if c.isValid():
            self.set_colour(c.name())

    def set_colour(self, colour: str | None):
        self.colour = colour
        self.colour_btn.setText(f"Team colour: {colour}" if colour else "Team colour: automatic")
        self.refresh()

    def refresh(self):
        if self.rgba is None:
            return
        from ..teams import team_variants
        variants, colours, pals = team_variants(self.rgba, self.colour)
        self.result = (variants, colours, pals)
        corner = [im[:192, :192] for im in [self.rgba, *variants.values()]]   # sheets: a corner
        self.view.show_rgba(grid(corner, 3, gap=6))
        changed = int((pals["player"] != colours).any(1).sum())
        self.save_btn.setEnabled(True)
        self.say(f"As drawn, player, enemy, ally, other, spent: {changed} of {len(colours)} colours "
                 "change per team." + ("" if changed else " Pick the team colour by hand."))

    def save_dialog(self):
        out = QFileDialog.getExistingDirectory(self, "Folder for the team files",
                                               str(self.path.parent if self.path else Path.cwd()))
        if out:
            self.save(Path(out))

    def save(self, out: Path) -> list[Path]:
        from PIL import Image

        from ..teams import write_godot_teams
        variants, colours, pals = self.result
        stem = self.path.stem if self.path else "sprite"
        out.mkdir(parents=True, exist_ok=True)
        written = []
        for team, img in variants.items():
            p = out / f"{stem}_{team}.png"
            Image.fromarray(img).save(p)
            written.append(p)
        written += write_godot_teams(colours, pals, out, stem)
        self.say("Saved " + ", ".join(p.name for p in written))
        return written


# --------------------------------------------------------------------------- portrait

class PortraitTab(ToolTab):
    """Photo / picture / prompt -> head-and-shoulders dialogue portrait."""
    title = "Portrait"

    def __init__(self, status=None):
        super().__init__(status)
        self.picture = None
        self.path: Path | None = None
        self.sprite = None
        open_btn = QPushButton("Open photo or picture…")
        open_btn.clicked.connect(self.open_dialog)
        prompt_btn = QPushButton("From a prompt (your GPU)…")
        prompt_btn.clicked.connect(self.prompt_dialog)
        self.width_ = spin(16, 512, 96)
        self.height_ = spin(16, 512, 96)
        self.colors = spin(2, 64, 32)
        self.flip = QCheckBox("Face the other way")
        self.scale = spin(1, 16, 4)
        box = QGroupBox("Portrait")
        fl = QFormLayout(box)
        fl.addRow(QLabel("Finds the face (or the top of the subject), crops head and shoulders, "
                         "cuts out the background."))
        fl.itemAt(0).widget().setWordWrap(True)
        fl.addRow(open_btn)
        fl.addRow(prompt_btn)
        fl.addRow("Width", self.width_)
        fl.addRow("Height", self.height_)
        fl.addRow("Colours", self.colors)
        fl.addRow(self.flip)
        fl.addRow("Also save enlarged x", self.scale)
        for w in (self.width_, self.height_, self.colors):
            w.editingFinished.connect(self.make)
        self.flip.toggled.connect(self.make)
        self.view = PixelView("Open a photo or a picture")
        self.save_btn = QPushButton("Save portrait…")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_dialog)
        self.busy_widgets = [self.save_btn, open_btn, prompt_btn]
        _layout(self, [box], self.view, [self.save_btn])

    def open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Photo or picture", "", IMAGE_FILTER)
        if path:
            self.open_path(Path(path))

    def open_path(self, path: Path):
        self.picture, self.path = path, path
        self.make()

    def prompt_dialog(self):
        text, ok = QInputDialog.getText(self, "Portrait from a prompt",
                                        "Who? (needs ComfyUI running, see docs/LOCAL_AI.md)")
        if not ok or not text.strip():
            return

        def job(say):
            from ..generate import generate_images
            from ..model3d import MODEL_NEGATIVE
            from ..teams import PORTRAIT_PROMPT
            say("Making the picture on your GPU…")
            (img, seed), = generate_images(PORTRAIT_PROMPT.format(prompt=text), MODEL_NEGATIVE,
                                           aspect=1.0, lora="none")
            return img
        slug = "-".join("".join(c if c.isalnum() else " " for c in text.lower()).split())[:40]

        def done(img):
            self.picture, self.path = img, Path.cwd() / f"{slug or 'portrait'}.png"
            self.make()
        self.run(job, done, "Generating…")

    def make(self, *_):
        if self.picture is None:
            return
        from ..teams import make_portrait
        s = settings_for("person", width=self.width_.value(), height=self.height_.value(),
                         colors=self.colors.value(), outline_style="selout")
        picture, flip = self.picture, self.flip.isChecked()
        self.run(lambda say: make_portrait(picture, s, flip), self._made, "Making the portrait…")

    def _made(self, sprite):
        self.sprite = sprite
        self.view.show_rgba(sprite.rgba)
        self.save_btn.setEnabled(True)
        w, h = sprite.size
        self.say(f"Portrait {w}x{h}, {len(sprite.palette)} colours")

    def save_dialog(self):
        default = str((self.path.parent if self.path else Path.cwd())
                      / f"{self.path.stem if self.path else 'portrait'}_portrait.png")
        path, _ = QFileDialog.getSaveFileName(self, "Save portrait", default, "PNG (*.png)")
        if path:
            self.save(Path(path))

    def save(self, out: Path) -> list[Path]:
        if self.path is not None and not isinstance(self.picture, Path):
            self.picture.save(out.with_suffix(".picture.png"))
        self.sprite.save(out, 1)
        written = [out]
        if self.scale.value() > 1:
            big = out.with_name(f"{out.stem}@{self.scale.value()}x.png")
            self.sprite.save(big, self.scale.value())
            written.append(big)
        self.say("Saved " + ", ".join(p.name for p in written))
        return written


# --------------------------------------------------------------------------- map tiles

class _TerrainRow(QWidget):
    def __init__(self, name: str, enabled: bool = True):
        super().__init__()
        self.on = QCheckBox()
        self.on.setChecked(enabled)
        self.name = QComboBox()
        self.name.setEditable(True)
        self.name.addItems(BUILTIN_TERRAINS)
        self.name.setCurrentText(name)
        self.source = QLineEdit()
        self.source.setPlaceholderText("built-in, a photo, or prompt: lava rocks")
        photo = QPushButton("Photo…")
        photo.clicked.connect(self.pick_photo)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        for w, stretch in ((self.on, 0), (self.name, 1), (self.source, 2), (photo, 0)):
            lay.addWidget(w, stretch)

    def pick_photo(self):
        path, _ = QFileDialog.getOpenFileName(self, "Texture photo", "", IMAGE_FILTER)
        if path:
            self.source.setText(path)

    def spec(self) -> str | None:
        if not self.on.isChecked() or not self.name.currentText().strip():
            return None
        src = self.source.text().strip()
        if src.lower().startswith("prompt:") or src.lower().startswith("prompt "):
            src = "prompt:" + src[7:].strip()
        return self.name.currentText().strip() + (f"={src}" if src else "")


MAP_STYLES = (("Top-down (tactics, RPG)", "topdown"), ("Side-scroller (platformer)", "sidescroller"),
              ("Isometric", "isometric"))
MAP_NOTES = {
    "topdown": ("Terrains (first = base)", "Each other terrain blends into the base with 15 corner "
                "tiles (shores, edges)."),
    "sidescroller": ("Ground, then its top", "The first terrain is the ground, the second grows on "
                     "every open top (dirt + grass, stone + snow). 16 edge tiles with collision and "
                     "one-way platforms."),
    "isometric": ("Terrains", "Each terrain becomes a flat floor tile and a block."),
}


class MapTab(ToolTab):
    """Terrains -> top-down, side-scroller or isometric tiles + a Godot TileSet."""
    title = "Map tiles"

    def __init__(self, status=None):
        super().__init__(status)
        self.ts = None
        self.built_style = "topdown"
        self.style = QComboBox()
        for label, key in MAP_STYLES:
            self.style.addItem(label, key)
        self.style.currentIndexChanged.connect(self._style_changed)
        self.rows = [_TerrainRow("grass"), _TerrainRow("water"), _TerrainRow("sand"),
                     _TerrainRow("road"), _TerrainRow("dirt", False)]
        self.box = QGroupBox("Terrains (first = base)")
        bl = QVBoxLayout(self.box)
        self.note = QLabel()
        self.note.setWordWrap(True)
        bl.addWidget(_row(QLabel("Style"), self.style))
        bl.addWidget(self.note)
        for r in self.rows:
            bl.addWidget(r)
        box = self.box
        self.tile = spin(8, 128, 32)
        self.variants = spin(1, 15, 4)
        self.seed = spin(0, 9999, 0)
        opts = QGroupBox("Tiles")
        fl = QFormLayout(opts)
        fl.addRow("Tile size (px)", self.tile)
        fl.addRow("Grass variations", self.variants)
        fl.addRow("Pattern seed", self.seed)
        self.view = PixelView("Make tiles to see an example map")
        self.make_btn = QPushButton("Make tiles")
        self.make_btn.clicked.connect(self.make)
        self.save_btn = QPushButton("Save tiles + Godot TileSet…")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_dialog)
        self.busy_widgets = [self.make_btn, self.save_btn]
        _layout(self, [box, opts], self.view, [self.make_btn, self.save_btn])
        self._style_changed()

    def _style_changed(self, *_):
        from ..tilesets import DEFAULT_TERRAINS
        style = self.style.currentData()
        title, text = MAP_NOTES[style]
        self.box.setTitle(title)
        self.note.setText(text + " Leave a source empty for the built-in look, pick a texture photo, "
                          "or type prompt: ... to make it on your GPU.")
        names = DEFAULT_TERRAINS[style]
        for i, r in enumerate(self.rows):
            r.on.setChecked(i < len(names))
            if i < len(names):
                r.name.setCurrentText(names[i])
                r.source.clear()
        self.variants.setEnabled(style == "topdown")

    def specs(self) -> list[str]:
        return [s for s in (r.spec() for r in self.rows) if s]

    def make(self):
        specs = self.specs()
        if not specs:
            self.say("Tick at least one terrain.")
            return
        tile, variants, seed = self.tile.value(), self.variants.value(), self.seed.value()
        style = self.style.currentData()

        def job(say):
            from ..maptiles import terrain
            from ..tilesets import build_style
            client = None
            if any("=prompt:" in s for s in specs):
                from ..generate import ComfyClient
                client = ComfyClient()
            built = []
            for spec in specs:
                name, _, source = spec.partition("=")
                say(f"Making {name}…")
                built.append(terrain(name, source or None, tile, seed=seed, client=client))
            return style, build_style(style, built, variants, seed)
        self.run(job, self._made, "Making tiles…")

    def _made(self, result):
        from ..tilesets import style_preview
        self.built_style, self.ts = result
        self.view.show_rgba(style_preview(self.built_style, self.ts, self.seed.value()))
        self.save_btn.setEnabled(True)
        names = ", ".join(self.ts.names) if isinstance(self.ts.names, (list, tuple)) else self.ts.names
        self.say(f"{self.built_style}: {names}. Preview: an example map.")

    def save_dialog(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save tile atlas", str(Path.cwd() / "terrain.png"),
                                              "PNG (*.png)")
        if path:
            self.save(Path(path))

    def save(self, out: Path) -> list[Path]:
        from ..tilesets import write_style
        written = write_style(self.built_style, self.ts, out, True, True, self.seed.value())
        how = {"topdown": "paint in the Terrains tab", "sidescroller": "paint the ground in the "
               "Terrains tab (collision included)", "isometric": "floors from source 0, blocks from "
               "source 1"}[self.built_style]
        self.say("Saved " + ", ".join(p.name for p in written) + f". In Godot: TileMapLayer, set the "
                 f"TileSet, {how}.")
        return written


# --------------------------------------------------------------------------- Godot demo

class DemoTab(ToolTab):
    """Everything together: a Godot project with a map and units that walk where you click."""
    title = "Godot demo"

    def __init__(self, status=None):
        super().__init__(status)
        self.unit_files: list[Path] = []
        self.out: Path | None = None
        self.knight = QRadioButton("Built-in knight (idle + walk)")
        self.knight.setChecked(True)
        self.mine = QRadioButton("My unit:")
        pick = QPushButton("Choose model, photos or video…")
        pick.clicked.connect(self.pick_unit)
        self.unit_label = QLabel("")
        self.unit_label.setWordWrap(True)
        unit = QGroupBox("Unit")
        ul = QVBoxLayout(unit)
        note = QLabel("Writes a Godot 4 project (640x360, x3 = 1080p): a map from the tiles and four "
                      "units in team colours. Open it, press F5, click your unit, then a tile.")
        note.setWordWrap(True)
        ul.addWidget(note)
        ul.addWidget(self.knight)
        ul.addWidget(_row(self.mine, pick))
        ul.addWidget(self.unit_label)
        self.options = SpriteOptions("person", 40, 40, 20)
        self.directions = QComboBox()
        self.directions.addItems(["8", "16", "4"])
        self.frames = spin(1, 16, 6)
        self.terrains = QLineEdit("grass,water,sand,road")
        self.tile = spin(16, 64, 32)
        opts = QGroupBox("Project")
        fl = QFormLayout(opts)
        fl.addRow(self.options)
        fl.addRow("Directions", self.directions)
        fl.addRow("Frames per animation", self.frames)
        fl.addRow("Terrains", self.terrains)
        fl.addRow("Tile size", self.tile)
        self.view = PixelView("The start screen shows here")
        self.make_btn = QPushButton("Make Godot project…")
        self.make_btn.clicked.connect(self.make_dialog)
        self.open_btn = QPushButton("Open the project folder")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(lambda: self.out and open_folder(self.out))
        self.busy_widgets = [self.make_btn]
        _layout(self, [unit, opts], self.view, [self.make_btn, self.open_btn])

    def pick_unit(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Unit: model(s), photos or a video", "", SOURCE_FILTER)
        if paths:
            self.unit_files = [Path(p) for p in paths]
            self.unit_label.setText(", ".join(p.name for p in self.unit_files))
            self.mine.setChecked(True)

    def make_dialog(self):
        parent = QFileDialog.getExistingDirectory(self, "Where to put the project", str(Path.cwd()))
        if parent:
            self.make(_new_folder(Path(parent), "PixelPipelineDemo"))

    def make(self, out: Path):
        from ..demo import make_demo, parse_terrains
        unit = self.unit_files if self.mine.isChecked() and self.unit_files else None
        names, sources = parse_terrains(t for t in self.terrains.text().split(",") if t.strip())
        s = self.options.settings()
        n, frames, tile = int(self.directions.currentText()), self.frames.value(), self.tile.value()

        def job(say):
            return make_demo(out, unit, s, directions=n, frames=frames, terrains=names,
                             terrain_sources=sources, tile=tile, progress=say)
        self.out = out
        self.run(job, self._made, "Making the Godot project…")

    def _made(self, result):
        written, preview = result
        self.view.show_rgba(preview)
        self.open_btn.setEnabled(True)
        self.say(f"Godot project in {self.out} ({len(written)} files). In Godot: Import, pick "
                 "project.godot, press F5.")


def _new_folder(parent: Path, name: str) -> Path:
    """parent/name, or name-2, -3 ... if that exists and isn't an earlier demo."""
    k, out = 1, parent / name
    while out.exists() and any(out.iterdir()) and not (out / "project.godot").exists():
        k += 1
        out = parent / f"{name}-{k}"
    return out


# --------------------------------------------------------------------------- prompt (online AIs)

class KeysDialog(QDialog):
    """API keys for the online services, stored for this Windows user only."""

    def __init__(self, parent=None):
        from ..online import PROVIDERS, load_keys
        super().__init__(parent)
        self.setWindowTitle("API keys")
        keys = load_keys()
        form = QFormLayout(self)
        self.edits = {}
        for prov in ("pixellab", "retrodiffusion"):
            info = PROVIDERS[prov]
            edit = QLineEdit(keys.get(prov, ""))
            edit.setEchoMode(QLineEdit.Password)
            edit.setPlaceholderText("paste your API key")
            link = QLabel(f'<a href="{info["signup"]}">get a key</a>')
            link.setOpenExternalLinks(True)
            form.addRow(info["name"], _row(edit, link))
            self.edits[prov] = edit
        self.status = QLabel("Keys are saved in your user folder (.pixel-pipeline/keys.json) and "
                             "only sent to their own service.")
        self.status.setWordWrap(True)
        form.addRow(self.status)
        check = QPushButton("Check balances")
        check.clicked.connect(self.check)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        form.addRow(check)
        form.addRow(buttons)

    def keys(self) -> dict:
        return {p: e.text().strip() for p, e in self.edits.items() if e.text().strip()}

    def check(self):
        from ..online import OnlineError, pixellab_balance, retrodiffusion_balance
        lines = []
        for prov, fn in (("pixellab", pixellab_balance), ("retrodiffusion", retrodiffusion_balance)):
            key = self.keys().get(prov)
            if not key:
                continue
            try:
                lines.append(f"{prov}: {fn(key)}")
            except OnlineError as e:
                lines.append(f"{prov}: {e}")
        self.status.setText("\n".join(lines) or "Paste a key first.")

    def save(self):
        from ..online import save_keys
        save_keys(self.keys())
        self.accept()


class PromptTab(ToolTab):
    """Type what you want; PixelLab, Retro Diffusion and your GPU each make it, side by side.
    Only the prompt text is sent, never your pictures."""
    title = "Prompt"

    def __init__(self, status=None, send_to_sprite=None):
        from ..online import PIXELLAB_DIRECTIONS, PIXELLAB_MODELS, PIXELLAB_VIEWS, RD_STYLES
        super().__init__(status)
        self.send_to_sprite = send_to_sprite
        self.cands = []
        self.prompt = QLineEdit()
        self.prompt.setPlaceholderText("e.g. a knight with a red cape and a big sword")
        self.prompt.returnPressed.connect(self.make)
        self.options = SpriteOptions("person", 48, 64, 24)
        self.seed = spin(0, 2**31 - 1, 0, "random")
        what = QGroupBox("What")
        wl = QFormLayout(what)
        note = QLabel("Each service draws it new from your words. Nothing from your PC is sent "
                      "except the text.")
        note.setWordWrap(True)
        wl.addRow(note)
        wl.addRow(self.prompt)
        wl.addRow(self.options)
        wl.addRow("Seed", self.seed)

        self.use_pl = QCheckBox("PixelLab")
        self.use_pl.setChecked(True)
        self.pl_model = QComboBox()
        self.pl_model.addItems(list(PIXELLAB_MODELS))
        self.pl_model.setToolTip("pixen = fast, pixflux / bitforge = classic, pro = best (slower)")
        self.pl_view = QComboBox()
        self.pl_view.addItems(["any view", *PIXELLAB_VIEWS])
        self.pl_dir = QComboBox()
        self.pl_dir.addItems(["any direction", *PIXELLAB_DIRECTIONS])
        self.use_rd = QCheckBox("Retro Diffusion")
        self.use_rd.setChecked(True)
        self.rd_style = QComboBox()
        self.rd_style.setEditable(True)
        self.rd_style.addItems(list(RD_STYLES))
        self.rd_style.setToolTip("rd_fast $0.03, rd_plus $0.06, rd_pro $0.18 per image. "
                                 "fast/plus draw at 64 px or more (small sprites are made 2x)")
        self.use_gpu = QCheckBox("Your GPU (ComfyUI, free)")
        keys_btn = QPushButton("Keys…")
        keys_btn.clicked.connect(lambda: KeysDialog(self).exec())
        who = QGroupBox("Who makes it")
        gl = QFormLayout(who)
        gl.addRow(self.use_pl, self.pl_model)
        gl.addRow("   PixelLab camera", self.pl_view)
        gl.addRow("   PixelLab facing", self.pl_dir)
        gl.addRow(self.use_rd, self.rd_style)
        gl.addRow(self.use_gpu)
        gl.addRow(keys_btn)

        self.view = PixelView("Results show here: what each service drew, and the sprite on "
                              "your settings")
        self.make_btn = QPushButton("Make")
        self.make_btn.clicked.connect(self.make)
        self.pick = QComboBox()
        self.send_btn = QPushButton("Open in the Sprite tab")
        self.send_btn.clicked.connect(self.send)
        self.save_btn = QPushButton("Save all…")
        self.save_btn.clicked.connect(self.save_dialog)
        for b in (self.send_btn, self.save_btn):
            b.setEnabled(False)
        self.busy_widgets = [self.make_btn, self.save_btn, self.send_btn]
        _layout(self, [what, who], self.view,
                [self.make_btn, _row(self.pick, self.send_btn), self.save_btn])

    def providers(self) -> list[str]:
        return [p for p, box in (("pixellab", self.use_pl), ("retrodiffusion", self.use_rd),
                                 ("gpu", self.use_gpu)) if box.isChecked()]

    def make(self):
        from ..online import prompt_candidates
        text, providers = self.prompt.text().strip(), self.providers()
        if not text or not providers:
            self.say("Type what you want and tick at least one service.")
            return
        s = self.options.settings()
        kw = dict(seed=self.seed.value() or None, pixellab_model=self.pl_model.currentText(),
                  rd_style=self.rd_style.currentText().strip() or "rd_fast__default",
                  view=None if self.pl_view.currentIndex() == 0 else self.pl_view.currentText(),
                  direction=None if self.pl_dir.currentIndex() == 0 else self.pl_dir.currentText())
        self.run(lambda say: prompt_candidates(text, s, providers, progress=say, **kw),
                 self._made, "Asking…")

    def _made(self, cands):
        from ..online import PROVIDERS, compare_sheet
        self.cands = cands
        self.view.show_rgba(np.asarray(compare_sheet(cands, scale=1).convert("RGBA")))
        self.pick.clear()
        good = [c for c in cands if c.sprite is not None]
        for c in good:
            self.pick.addItem(PROVIDERS[c.provider]["name"], c.provider)
        for b in (self.send_btn, self.save_btn):
            b.setEnabled(bool(good))
        bad = [f"{PROVIDERS[c.provider]['name']}: {c.error.splitlines()[0]}" for c in cands if c.error]
        self.say(f"{len(good)} of {len(cands)} made it. " + " | ".join(bad))

    def chosen(self):
        prov = self.pick.currentData()
        return next((c for c in self.cands if c.provider == prov), None)

    def send(self):
        c = self.chosen()
        if c is not None and self.send_to_sprite:
            self.send_to_sprite(c)

    def save_dialog(self):
        words = "".join(ch if ch.isalnum() else " " for ch in self.prompt.text().lower()).split()
        default = str(Path.cwd() / f"{'-'.join(words)[:40] or 'sprite'}.png")
        path, _ = QFileDialog.getSaveFileName(self, "Save results", default, "PNG (*.png)")
        if path:
            self.save(Path(path))

    def save(self, out: Path) -> list[Path]:
        from ..online import write_candidates
        written = write_candidates(self.cands, out)
        self.say("Saved " + ", ".join(p.name for p in written))
        return written

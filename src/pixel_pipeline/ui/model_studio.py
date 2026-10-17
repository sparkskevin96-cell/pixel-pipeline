"""Characters: make a character from words, a photo or a 3D model, watch it play, keep it.

Left: your Character Library (library.py). Click one to see it play. Right, top to bottom:
1. make it (words -> GPU picture -> 3D shape; or a photo; or a 3D model), 2. look at it
(picture, spinning model, sprites), 3. name it and press Add to library: it gets a skeleton,
every animation from 16 directions at the one library scale, a normal map and a thumbnail,
ready for the Game tab. More settings shows the raw numbers (directions, size, colours, tilt).

Prompt -> front picture -> 3D shape (Hunyuan3D in ComfyUI on this PC) -> painted from the
picture -> spinning preview + 4/8/16-direction sprites -> Save all (.glb, picture, sprite
sheet, .json, .aseprite, Godot SpriteFrames + facing script, spinning GIF).
Tick Animate to play the model's animations; a model without any (a Prompt to Model
character, say) first gets a skeleton and idle/walk/run/attack/hurt/death (autorig.py).

Run with:  pixel-pipeline-model   (or File -> Prompt to 3D model in pixel-pipeline-ui)
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtCore import QSize
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPushButton, QSpinBox,
    QSplitter, QStackedWidget, QVBoxLayout, QWidget,
)

from ..settings import settings_for
from ..turnaround import (direction_names, render_model, turnaround_sprites, union_crop,
                          write_preview, write_turnaround)
from ..progress import Stage, character_plan, model_plan, wrap
from .common import checkerboard, to_pixmap

IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
MODEL_FILTER = "3D models (*.glb *.gltf *.obj *.ply *.stl)"
PANEL = 300
VIEW_CHOICES = {"auto": None, "front": "front", "front+back": "front+back", "all four": "all"}


class _Worker(QObject):
    progress = Signal(str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, job):
        super().__init__()
        self.job = job

    def run(self):
        try:
            self.done.emit(self.job(self.progress.emit))
        except Exception as e:  # noqa: BLE001 - shown to the user
            self.failed.emit(str(e) or type(e).__name__)


def _label(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setAlignment(Qt.AlignCenter)
    lab.setMinimumSize(PANEL, PANEL)
    lab.setStyleSheet("background:#1c2030;color:#9aa3b5;border-radius:8px;")
    lab.setWordWrap(True)
    return lab


def _fit(rgba: np.ndarray, size: int = PANEL) -> np.ndarray:
    """Enlarge with nearest neighbour (pixel art stays crisp) or shrink smoothly to fit."""
    import cv2
    h, w = rgba.shape[:2]
    k = size / max(h, w)
    if k >= 2:
        k = int(k)
        return np.repeat(np.repeat(rgba, k, 0), k, 1)
    interp = cv2.INTER_NEAREST if k >= 1 else cv2.INTER_AREA
    return cv2.resize(rgba, (max(1, round(w * k)), max(1, round(h * k))), interpolation=interp)


class ModelStudio(QMainWindow):
    """The Characters tab: make a character, see it play, keep it in the library."""

    library_changed = Signal()

    def __init__(self, client=None):
        super().__init__()
        self.setWindowTitle("Prompt to Model")
        self.resize(1080, 640)
        self.client = client              # ComfyClient; None = default address
        self.sync = False                 # tests: run jobs on the calling thread
        self.last_error = ""
        self.work = Path(tempfile.mkdtemp(prefix="pixel-pipeline-model-"))
        self.result = None                # ModelResult
        self.sprites = None
        self.anim = None                  # AnimatedSet when Animate is ticked
        self.anim_names: list[str] = []
        self.rigged: Path | None = None   # skeleton + animations added by Animate
        self.back_path: Path | None = None  # picture of the far side
        self.character = None             # the library Character on show, if any
        self._anim_i = 0
        self.spin: list[np.ndarray] = []
        self._spin_i = 0
        self._thread = None
        self._spin_timer = QTimer(self)
        self._spin_timer.timeout.connect(self._next_spin)
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._next_anim_frame)
        self._build()
        self.statusBar().showMessage(self.HINT)
        self.refresh_library()

    HINT = ("Describe a character and press Make from words (your GPU), or start from a photo or "
            "a 3D model. Then name it and press Add to library. Click a character on the left "
            "to see it play.")

    # ---- layout ---------------------------------------------------------------
    def _build(self):
        self.prompt = QLineEdit()
        self.prompt.setPlaceholderText("e.g. a knight with a red cape and a big sword")
        self.prompt.returnPressed.connect(self.make)
        self.subject = QComboBox()
        self.subject.addItems(["person", "animal", "object"])
        self.subject.setToolTip("person: two legs (idle walk run attack hurt death). animal: four "
                                "legs (idle walk run sit happy lie down death). object: no skeleton.")
        self.seed = QSpinBox()
        self.seed.setRange(0, 2**31 - 1)
        self.seed.setSpecialValueText("random")
        self.make_btn = QPushButton("Make from words")
        self.make_btn.setDefault(True)
        self.make_btn.clicked.connect(self.make)
        pic_btn = QPushButton("From a photo…")
        pic_btn.setToolTip("A photo or drawing of the whole subject; you can rub out a leash or a "
                           "hand before the 3D shape is built")
        pic_btn.clicked.connect(self.from_picture)
        model_btn = QPushButton("From a 3D model…")
        model_btn.setToolTip("A .glb/.gltf/.obj from Blender, Tripo, Meshy… (no GPU needed)")
        model_btn.clicked.connect(self.open_model)
        self.views = QComboBox()
        self.views.addItems(list(VIEW_CHOICES))
        self.views.setToolTip("Which sides your GPU draws to paint the model: auto = front + back "
                              "from words, front only for your own photo. The back and sides are "
                              "drawn from the 3D shape, so they match it.")
        top = QHBoxLayout()
        top.addWidget(self.prompt, 1)
        for w in (QLabel("Kind"), self.subject, QLabel("Views"), self.views, self.make_btn,
                  pic_btn, model_btn):
            top.addWidget(w)
        self.seed.setToolTip("The same words + seed make the same picture again")

        self.back_btn = QPushButton("Back picture…")
        self.back_btn.setToolTip("A picture of the other side, for the parts the front can't see")
        self.back_btn.clicked.connect(self.pick_back)
        self.mirror = QCheckBox("No back picture: mirror the front")
        self.mirror.setToolTip("Paint the far side from the picture mirrored: good for animals and "
                               "people photographed from the side, marked about the same on both sides")
        self.only_colours = QLineEdit()
        self.only_colours.setPlaceholderText("Only these colours, e.g. black, white  "
                                             "(empty = from the picture)")
        self.only_colours.setToolTip("Names (black, white, brown, tan, cream, grey, pink, …) or hex "
                                     "like #1c1c20. The picture and model are snapped to them, so a "
                                     "white dog in shade stays white and sun glare on black fur goes.")
        self.edit_btn = QPushButton("Clean up picture…")
        self.edit_btn.setToolTip("Rub out a leash or hand, bring back a lost leg, then build again")
        self.edit_btn.clicked.connect(self._edit_clicked)
        self.repaint_btn = QPushButton("Repaint")
        self.repaint_btn.setToolTip("Paint the same shape again with the colour / back settings "
                                    "(no GPU)")
        self.repaint_btn.clicked.connect(self.repaint)
        for b in (self.edit_btn, self.repaint_btn):
            b.setEnabled(False)
        paint = QHBoxLayout()
        for w in (self.back_btn, self.mirror, self.only_colours, self.edit_btn, self.repaint_btn):
            paint.addWidget(w, 1 if w is self.only_colours else 0)
        self.paint_box = QGroupBox("Photo options")
        self.paint_box.setLayout(paint)

        self.picture_view = _label("Front picture")
        self.model_view = _label("3D model (spins)")
        self.sprite_view = _label("Direction sprites")
        panels_w = QWidget()
        panels = QHBoxLayout(panels_w)
        panels.setContentsMargins(0, 0, 0, 0)
        for title, view in (("Picture", self.picture_view), ("3D model", self.model_view),
                            ("In the game (every direction)", self.sprite_view)):
            box = QGroupBox(title)
            lay = QVBoxLayout(box)
            lay.addWidget(view)
            panels.addWidget(box, 1)
        # a drawn character's front drawing waiting for the OK (library.in_preview)
        from .character_review import CharacterReview
        self.review = CharacterReview(self._run, self.statusBar().showMessage)
        self.review.approved.connect(self._review_approved)
        self.review.changed.connect(self._review_changed)
        self.review.closed.connect(self._review_closed)
        self.panels_page = panels_w
        self.stack = QStackedWidget()
        self.stack.addWidget(panels_w)
        self.stack.addWidget(self.review)

        self.directions = QComboBox()
        self.directions.addItems(["16", "8", "4"])
        self.width_ = QSpinBox()
        self.width_.setRange(8, 512)
        self.width_.setValue(48)
        self.height_ = QSpinBox()
        self.height_.setRange(8, 512)
        self.height_.setValue(48)
        self.colors = QSpinBox()
        self.colors.setRange(2, 64)
        self.colors.setValue(24)
        self.elevation = QDoubleSpinBox()
        self.elevation.setRange(0, 89)
        self.elevation.setValue(30)
        self.elevation.setSuffix("°")
        self.animate = QCheckBox("Animate")
        self.animate.setEnabled(False)
        self.animate.setToolTip("Play the model's animations from every direction. A model "
                                "without any gets a skeleton: people idle, walk, run, attack, "
                                "hurt, death; animals (Subject: animal, four legs) idle, walk, "
                                "run, sit, happy, lie down, death. Or open a rigged .glb (Mixamo) "
                                "to play its own.")
        self.frames = QSpinBox()
        self.frames.setRange(1, 32)
        self.frames.setValue(8)
        self.frames.setToolTip("Frames per animation")
        self.swing = QCheckBox("Capes/hair/tails swing")
        self.swing.setToolTip("Find a cape, long hair or a tail hanging off the body, give it spring "
                              "bones and bake its swing into every animation (it trails when "
                              "running, settles when standing). Used when a skeleton is added.")
        self.drawn = QCheckBox("Drawn sprites")
        self.drawn.setToolTip("Add to library: the GPU draws the character as pixel art from every "
                              "direction over the model, and the skeleton bends those drawings "
                              "for each frame (crisper than rendering the 3D model; ~6-10 min "
                              "GPU for 16 directions, Stop keeps what's drawn). Off: the 3D "
                              "model is rendered (no GPU). Right-click a drawn character in "
                              "the library to redraw one direction.")
        from ..drawnrig import available
        self.drawn.setChecked(available())
        self.front_first = QCheckBox("Front first, wait for my OK")
        self.front_first.setChecked(True)
        self.front_first.setToolTip("Drawn sprites: draw only the front (~1 min of GPU) and show it "
                                    "with a short idle; the other directions are drawn when you "
                                    "say it looks good (or redraw it first). Off: all at once.")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setToolTip("Stop drawing after the current picture (the drawings so far are "
                                 "kept: Add to library again carries on)")
        self.stop_btn.clicked.connect(self.stop)
        self.stop_btn.setVisible(False)
        self._cancel = None               # threading.Event of the running job
        self.redo_btn = QPushButton("Update sprites")
        self.redo_btn.clicked.connect(self.update_sprites)
        self.save_btn = QPushButton("Save all…")
        self.save_btn.clicked.connect(self.save_all)
        for b in (self.redo_btn, self.save_btn):
            b.setEnabled(False)
        self.name = QLineEdit()
        self.name.setPlaceholderText("Name, e.g. Geela")
        self.size_ = QComboBox()
        self.size_.setToolTip("People: small / normal / large. Animals: sized next to a person "
                              "(small = a dog, medium = a wolf, large = a bear)")
        self.subject.currentTextChanged.connect(self._kind_changed)
        self.add_btn = QPushButton("Add to library")
        self.add_btn.setToolTip("Skeleton + every animation from 16 directions + normal map, saved "
                                "in your Character Library at the one game scale")
        self.add_btn.setStyleSheet("font-weight:600;")
        self.add_btn.clicked.connect(self.add_to_library)
        self.add_btn.setEnabled(False)
        self.more_btn = QPushButton("More settings ▸")
        self.more_btn.setCheckable(True)
        self.more_btn.toggled.connect(self._toggle_more)
        self.more_anim = QComboBox()
        self.more_anim.setToolTip("Another animation for the character on show (people: cast, "
                                  "shoot, block, thrust, jump, wave, kneel, victory; animals: "
                                  "bark, howl, jump, eat), or a real motion-captured move (Mocap: "
                                  "walk, sword slash, dance…; replaces a move of the same name). "
                                  "Its other animations stay as they are.")
        self.add_anim_btn = QPushButton("Add animation")
        self.add_anim_btn.clicked.connect(self._add_anim_clicked)
        self.add_anim_btn.setEnabled(False)
        keep = QHBoxLayout()
        for w in (QLabel("Name"), self.name, QLabel("Size"), self.size_, self.add_btn,
                  self.stop_btn, self.more_anim, self.add_anim_btn):
            keep.addWidget(w, 1 if w is self.name else 0)
        keep.addStretch(1)
        keep.addWidget(self.more_btn)
        self._kind_changed(self.subject.currentText())

        form = QFormLayout()
        row = QHBoxLayout()
        for label, w in (("Seed", self.seed), ("Directions", self.directions), ("Width", self.width_),
                         ("Height", self.height_), ("Colours", self.colors),
                         ("Camera tilt", self.elevation)):
            row.addWidget(QLabel(label))
            row.addWidget(w)
        row.addWidget(self.animate)
        row.addWidget(self.swing)
        row.addWidget(self.drawn)
        row.addWidget(self.front_first)
        row.addWidget(self.frames)
        row.addStretch(1)
        row.addWidget(self.redo_btn)
        row.addWidget(self.save_btn)
        form.addRow(row)
        self.more = QWidget()
        self.more.setLayout(form)
        self.more.setVisible(False)

        # the library
        self.library = QListWidget()
        self.library.setViewMode(QListWidget.IconMode)
        self.library.setIconSize(QSize(72, 72))
        self.library.setGridSize(QSize(96, 104))
        self.library.setResizeMode(QListWidget.Adjust)
        self.library.setMovement(QListWidget.Static)
        self.library.setMinimumWidth(220)
        self.library.itemClicked.connect(lambda item: self.open_character(item.data(Qt.UserRole)))
        self.library.setContextMenuPolicy(Qt.CustomContextMenu)
        self.library.customContextMenuRequested.connect(self._library_menu)
        self.lib_title = QLabel("Your characters")
        self.lib_title.setStyleSheet("font-weight:600;")
        new_btn = QPushButton("New character")
        new_btn.clicked.connect(self.new_character)
        folder_btn = QPushButton("Show folder")
        folder_btn.clicked.connect(self.show_library_folder)
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
        lay = QVBoxLayout(right)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(top)
        lay.addWidget(self.paint_box)
        lay.addWidget(self.stack, 1)
        from .common import ProgressPanel
        self.progress_panel = ProgressPanel()     # bars + about how long is left + Stop
        lay.addWidget(self.progress_panel)
        lay.addLayout(keep)
        lay.addWidget(self.more)
        split = QSplitter()
        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setSizes([240, 1000])
        self.setCentralWidget(split)

    # ---- library --------------------------------------------------------------
    def _kind_changed(self, kind: str):
        from ..library import DEFAULT_SIZE, SIZES
        self.size_.clear()
        sizes = SIZES.get(kind)
        if sizes:
            self.size_.addItems(list(sizes))
            self.size_.setCurrentText(DEFAULT_SIZE[kind])
        self.size_.setEnabled(bool(sizes))
        if hasattr(self, "add_btn"):
            self._busy(False) if self._thread is None else None

    def _toggle_more(self, on: bool):
        self.more.setVisible(on)
        self.more_btn.setText("More settings ▾" if on else "More settings ▸")

    def refresh_library(self, select=None):
        from ..library import in_preview, list_characters
        self.library.clear()
        chars = list_characters()
        waiting = 0
        for ch in chars:
            draft = in_preview(ch)
            waiting += draft
            item = QListWidgetItem(ch.name + ("\nwaiting for OK" if draft else ""))
            if ch.thumb.exists():
                pix = QPixmap(str(ch.thumb))
                item.setIcon(QIcon(_badged(pix) if draft else pix))
            item.setData(Qt.UserRole, ch)
            item.setToolTip(
                (f"{ch.name}: a draft — only its front is drawn, waiting for your OK. Click it "
                 "to review (or right-click).") if draft else
                (f"{ch.name}: {ch.kind}, {ch.size}. Class {ch.stats_class}: "
                 "HP %d  ATK %d  DEF %d  MOV %d  JMP %d  RNG %d" % ch.stats())
                + self._quality_note(ch).replace("; ", chr(10)))
            if draft:
                item.setForeground(QColor("#e0a020"))
            self.library.addItem(item)
            if select is not None and ch.folder == select.folder:
                self.library.setCurrentItem(item)
        self.lib_title.setText(f"Your characters ({len(chars)})"
                               + (f" — {waiting} waiting for your OK" if waiting else ""))

    def show_library_folder(self):
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        from ..library import root
        root().mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(root())))

    def new_character(self):
        self.character = None
        self.result = None
        self.sprites = None
        self.anim = None
        self._anim_timer.stop()
        self._spin_timer.stop()
        self.spin = []
        self.name.clear()
        self.prompt.clear()
        for view, text in ((self.picture_view, "Picture"), (self.model_view, "3D model (spins)"),
                           (self.sprite_view, "Every direction, animated")):
            view.clear()
            view.setText(text)
        self.library.clearSelection()
        self.review.set_character(None)
        self.stack.setCurrentWidget(self.panels_page)
        self._busy(False)
        self.statusBar().showMessage(self.HINT)

    def open_character(self, ch):
        """Show a library character: its picture, spinning model, and every animation playing."""
        from ..library import in_preview, load_animated
        from ..model3d import ModelResult
        if in_preview(ch):                       # a draft: its front drawing, for the OK
            self.show_review(ch)
            return
        self.stack.setCurrentWidget(self.panels_page)
        self.character = ch
        self.name.setText(ch.name)
        self.subject.setCurrentText(ch.kind)
        if self.size_.findText(ch.size) >= 0:
            self.size_.setCurrentText(ch.size)

        def job(say):
            say(f"Opening {ch.name}…")
            model = ch.model if ch.model.exists() else ch.rigged
            views = render_model(model, 16, self.elevation.value(), size=320) if model.exists() else []
            spin = union_crop(views) if views else []
            anim = load_animated(ch)
            sprites = [d[0] for d in anim.sprites[0]]
            pic = ch.picture if ch.picture.exists() else None
            res = ModelResult(model, pic, None, None)
            return res, spin, sprites, anim, [a.name for a in anim.actions], ch.rigged
        self._run(job, self._show_character)

    def _show_character(self, payload):
        self._show(payload)
        ch = self.character
        self._fill_more_anim()
        self.statusBar().showMessage(
            f"{ch.name}: {ch.kind}, {ch.size}, playing {len(self.anim_names)} animations from every "
            f"direction. It's ready for the Game tab.{self._quality_note(ch)} Files: {ch.folder}")

    @staticmethod
    def _quality_note(ch) -> str:
        """' 16/16 directions passed; made in 14 min (11 on the GPU).' (or '')"""
        from ..library import quality_of
        from ..metrics import made_in
        q = quality_of(ch).get("text") or ""
        t = made_in(ch.folder)
        bits = [b for b in (q, t) if b]
        return (" " + "; ".join(bits) + ".") if bits else ""

    def _fill_more_anim(self):
        from ..library import _known_motions
        self.more_anim.clear()
        ch = self.character
        if ch is not None:
            have = set(self.anim_names)
            self.more_anim.addItems([m for m in _known_motions(ch.kind) if m not in have])
            if ch.kind != "animal":
                self.more_anim.addItems([MOCAP + m for m in _mocap_moves()])
        self._busy(False)

    def _add_anim_clicked(self, _checked=False):
        self.add_animation()

    def add_animation(self, name: str | None = None):
        """Append an animation to the library character on show and re-render its sheet."""
        from ..library import add_character_animations
        ch = self.character
        name = name or self.more_anim.currentText()
        if ch is None or not name:
            return

        def job(say):
            if name.startswith(MOCAP):                   # motion capture (mocap.py)
                from ..library import add_mocap
                return add_mocap(ch, [name[len(MOCAP):]], progress=say)
            return add_character_animations(ch, [name], progress=say)
        self._run(job, self.open_character, f"Adding {name} to {ch.name}…", estimate=90,
                  title=f"Adding {name}")

    def add_to_library(self):
        """Rig + animate + render the current model at the library scale, then keep it."""
        from ..library import add_character
        if self.result is None:
            return
        name = self.name.text().strip() or (self.prompt.text().strip()[:30] or "Character")
        prompt = self.prompt.text().strip()
        kind = self.subject.currentText()
        size = self.size_.currentText()
        model = self.result.model
        swing = self.swing.isChecked()
        drawn = self.drawn.isChecked() and kind in ("person", "animal")
        preview = drawn and self.front_first.isChecked()     # the front first, then the OK
        cancel = self._new_cancel() if drawn else None
        picture = self.result.picture if self.result.picture and \
            Path(self.result.picture).suffix.lower() == ".png" else None

        def job(say):
            # springs go on the rig before anything is drawn or rendered (one render)
            secs = getattr(self, "_model_seconds", None)
            return add_character(name, model, kind=kind, size=size, picture=picture, progress=say,
                                 prompt=prompt, drawn=drawn, cancel=cancel, preview=preview,
                                 springs=swing, timing={"model": secs} if secs else None)

        def done(ch):
            self.refresh_library(select=ch)
            self.library_changed.emit()
            self.open_character(ch)
            self.show_notice(ch)
        n = self._library_directions()
        plan = character_plan(model=False, drawn=drawn, preview=preview, directions=n)
        self._run(job, done, f"Adding {name} to the library…", plan=plan,
                  title=f"Adding {name}" + (" (front first)" if preview else ""))

    # ---- the front drawing, for the OK ------------------------------------------
    def show_review(self, ch):
        """A draft character: its front drawing + idle, Looks good / Redraw / Change words."""
        self.character = ch
        self.name.setText(ch.name)
        self.subject.setCurrentText(ch.kind)
        if self.size_.findText(ch.size) >= 0:
            self.size_.setCurrentText(ch.size)
        self.prompt.setText(ch.prompt)
        self._anim_timer.stop()
        self._spin_timer.stop()
        self.review.sync = self.sync
        self.review.set_character(ch)
        self.stack.setCurrentWidget(self.review)
        self._busy(False)
        info = self.review.info or {}
        self.statusBar().showMessage(f"{ch.name}: the front is drawn — does it look right? The "
                                     f"rest: {info.get('estimate', 'a few minutes of GPU')}.")

    def _review_approved(self, ch):
        self.refresh_library(select=ch)
        self.library_changed.emit()
        self.open_character(ch)

    def _review_changed(self, ch):
        self.refresh_library(select=ch)
        self.library_changed.emit()

    def _review_closed(self, ch):
        self.refresh_library(select=ch)
        self.library_changed.emit()
        self.new_character()

    def show_notice(self, ch):
        """Something the user must know about an add (the GPU failed: 3D render instead, the
        drawings kept): a message box that doesn't block, and the status bar."""
        text = getattr(ch, "notice", "")
        if not text:
            return
        from PySide6.QtWidgets import QMessageBox
        self.statusBar().showMessage(text.splitlines()[0])
        box = QMessageBox(QMessageBox.Warning, "Drawn sprites stopped", text, QMessageBox.Ok, self)
        box.setModal(False)
        box.show()
        self._notice_box = box

    # ---- drawn characters: stop, redraw ----------------------------------------
    def _new_cancel(self):
        import threading
        self._cancel = threading.Event()
        return self._cancel.is_set

    def stop(self):
        """Stop the drawing after the picture in progress (what's drawn is kept)."""
        if self._cancel is not None:
            self._cancel.set()
            self.statusBar().showMessage("Stopping after this picture… (the drawings so far are "
                                         "kept: do it again to carry on)")

    def _library_menu(self, pos):
        from PySide6.QtWidgets import QInputDialog, QMenu
        from ..library import is_drawn
        item = self.library.itemAt(pos)
        if item is None:
            return
        ch = item.data(Qt.UserRole)
        from ..library import drawing_unfinished, in_preview
        if in_preview(ch):
            menu = QMenu(self)
            review = menu.addAction("Review the front drawing…")
            finish = menu.addAction("Finish the drawn sprites (it looks good: every direction)")
            act = menu.exec(self.library.mapToGlobal(pos))
            if act is review:
                self.show_review(ch)
            elif act is finish:
                self.show_review(ch)
                self.review.approve()
            return
        if drawing_unfinished(ch):
            menu = QMenu(self)
            finish = menu.addAction("Finish the drawn sprites (carry on where the GPU stopped)")
            if menu.exec(self.library.mapToGlobal(pos)) is finish:
                self.finish_drawing(ch)
            return
        menu = QMenu(self)
        check = menu.addAction("Check and fix quality (every direction)…")
        if not is_drawn(ch):
            if menu.exec(self.library.mapToGlobal(pos)) is check:
                self.check_quality(ch)
            return
        one = menu.addAction("Redraw one direction…")
        every = menu.addAction("Draw it again (every direction)")
        act = menu.exec(self.library.mapToGlobal(pos))
        if act is check:
            self.check_quality(ch)
        elif act is one:
            names = direction_names(_drawn_directions(ch))
            d, ok = QInputDialog.getItem(self, "Redraw a direction", f"{ch.name}: which direction "
                                         "looks wrong? (S = facing you, E = facing right)",
                                         names, 0, False)
            if ok and d:
                self.redraw_direction(ch, [d])
        elif act is every:
            self.redraw_direction(ch, None)

    def check_quality(self, ch):
        """Library right-click -> Check and fix quality: the automatic per-direction check on an
        existing character (library.check_and_fix): CPU re-bend at the project's size / camera,
        the failing directions drawn again on the GPU (after saying how many GPU minutes)."""
        from PySide6.QtWidgets import QMessageBox
        from ..library import check_and_fix, fix_estimate, is_drawn
        est = fix_estimate(ch) if is_drawn(ch) else {"text": f"Checks every direction of {ch.name} "
                                                    "(3D-rendered: nothing is redrawn).", "gpu_min": 0}
        if not self.sync and is_drawn(ch):
            ok = QMessageBox.question(self, "Check and fix quality", est["text"] + "\n\nGo ahead?")
            if ok != QMessageBox.Yes:
                return
        cancel = self._new_cancel()
        self.character = ch

        def job(say):
            q = check_and_fix(ch, progress=say, cancel=cancel)
            from ..dirquality import summary_text
            say(f"{ch.name}: {q.get('text') or summary_text(q)}")
            return ch
        self._run(job, self.open_character, f"Checking {ch.name}'s directions…",
                  title=f"Checking {ch.name}")

    def finish_drawing(self, ch):
        """Carry on drawing a character whose drawing stopped (GPU crash): the missing
        directions only, then its sheets are bent from the drawings."""
        from ..library import finish_drawing
        cancel = self._new_cancel()
        self.character = ch

        def job(say):
            return finish_drawing(ch, progress=say, cancel=cancel)
        self._run(job, self.open_character, f"Finishing {ch.name}'s drawings…",
                  plan=character_plan(model=False, drawn=True, approve=True,
                                      directions=_drawn_directions(ch)),
                  title=f"Finishing {ch.name}")

    def redraw_direction(self, ch, directions=None):
        """Draw a drawn library character's direction(s) again with new seeds (None = all) and
        bend the new drawings for every animation; the other directions stay as they are."""
        from ..library import redraw
        cancel = self._new_cancel()
        self.character = ch

        def job(say):
            return redraw(ch, directions, progress=say, cancel=cancel)
        n = len(directions) if directions else _drawn_directions(ch)
        self._run(job, self.open_character, f"Redrawing {ch.name}…",
                  plan=character_plan(model=False, drawn=True, approve=True, directions=n + 1)
                  if directions else character_plan(model=False, drawn=True, directions=n),
                  title=f"Redrawing {ch.name}")

    # ---- jobs -----------------------------------------------------------------
    def _run(self, job, on_done, what: str = "", *, plan=None, estimate=None, title: str = ""):
        """Run job(progress) off the UI thread; on_done(result) back on it. progress is a
        progress.Tracker (text like before, (step, text), (done, total)); the progress panel
        under the pictures shows the bars, about how long is left and Stop. plan: the job's
        progress.Stage list; estimate: seconds."""
        from .common import JobRunner
        if what:
            self.statusBar().showMessage(what)
        runner = JobRunner(self, self.progress_panel, self.statusBar().showMessage, self._failed,
                           self._stopped)
        self._runner = runner
        title = title or (what.rstrip("…. ") if what else "") or "Working"
        if self.sync:
            self._busy(True)
            try:
                runner.start(job, on_done, sync=True, plan=plan, estimate=estimate, title=title,
                             on_stop=self.stop)
            except Exception as e:  # noqa: BLE001
                self._failed(str(e))
            finally:
                self._busy(False)
            return
        self._busy(True)
        runner.busy = None
        runner.start(job, on_done, plan=plan, estimate=estimate, title=title, on_stop=self.stop)
        runner.thread.finished.connect(lambda: self._busy(False))
        self._thread, self._worker = runner.thread, runner.worker

    def _stopped(self, message: str):
        first = (message or "Stopped").splitlines()[0]
        self._failed(first if first.startswith("Stopped") else "Stopped: " + first)

    def _library_directions(self) -> int:
        try:
            from ..library import style
            return int(style().directions or 16)
        except Exception:  # noqa: BLE001 - no project: the library default
            return 16

    def _busy(self, busy: bool):
        if hasattr(self, "stop_btn"):
            self.stop_btn.setVisible(False)          # the progress panel's Stop does it
            if not busy:
                self._cancel = None
        self.make_btn.setEnabled(not busy)
        self.redo_btn.setEnabled(not busy and self.result is not None)
        self.save_btn.setEnabled(not busy and self.sprites is not None)
        if hasattr(self, "add_btn"):
            can_rig = self.subject.currentText() in ("person", "animal")
            self.add_btn.setEnabled(not busy and self.result is not None and can_rig
                                    and self.character is None)
            self.add_btn.setText("In your library" if self.character is not None else "Add to library")
        if hasattr(self, "add_anim_btn"):
            self.add_anim_btn.setEnabled(not busy and self.character is not None
                                         and self.more_anim.count() > 0
                                         and getattr(self, "stack", None) is not None
                                         and self.stack.currentWidget() is not self.review)
        has_pic = self.result is not None and self.result.picture is not None \
            and Path(self.result.picture).suffix.lower() == ".png"
        self.edit_btn.setEnabled(not busy and has_pic)
        self.repaint_btn.setEnabled(not busy and has_pic and self._shape() is not None)
        if busy:
            QApplication.setOverrideCursor(Qt.BusyCursor)
        else:
            QApplication.restoreOverrideCursor()

    def _failed(self, message: str):
        self.statusBar().showMessage("Stopped: " + message.splitlines()[0])
        self.last_error = message
        if not self.sync and not message.startswith("Stopped"):   # Stop isn't an error;
            # tests (sync) can't click a message box away
            QMessageBox.warning(self, "Couldn't make the model", message)

    def _model_job(self, prompt=None, picture=None, shape=None, reuse=None):
        from ..model3d import prompt_to_model
        seed = self.seed.value() or None
        subject = self.subject.currentText()
        name = "-".join("".join(c if c.isalnum() else " " for c in (prompt or "model").lower()).split())[:40]
        out = self.work / f"{name or 'model'}.glb"
        paint = self._paint_options()
        for view, path in (reuse or {}).items():      # repaint: the back/sides drawn before
            if view in ("left", "right") or (view == "back" and not paint["back"]
                                             and not paint["mirror_back"]):
                paint[view] = str(path)

        def job(say):
            import time as _time
            t_model = _time.time()
            res = prompt_to_model(prompt, out, subject=subject, picture=picture, shape=shape,
                                  client=self.client, seed=seed,
                                  progress=wrap(say), **paint)
            self._model_seconds = _time.time() - t_model
            say("Rendering the model from every side…")
            return res, *self._sprite_job_body(res.model, say)
        return job

    def _model_plan(self, picture: bool = False) -> list:
        """Stages of words/picture -> model -> sprites (for the progress bars)."""
        plan = model_plan(views=self.views.currentText() != "front")
        if picture:
            plan = [s for s in plan if s.key != "picture"]
        n = int(self.directions.currentText())
        return plan + [Stage("sprites", f"Sprites from {n} directions", 0.3 + 0.5 * self.animate.isChecked(),
                             r"rendering the model|playing \d+ animations|skeleton")]

    def _sprite_job_body(self, model: Path, say=lambda text: None):
        n = int(self.directions.currentText())
        s = settings_for(self.subject.currentText(), width=self.width_.value(),
                         height=self.height_.value(), colors=self.colors.value())
        views = render_model(model, n, self.elevation.value(), size=320)
        spin = union_crop(views)
        sprites = turnaround_sprites(union_crop(views, s.margin, s.alpha_threshold), s)
        names = animation_names(model)
        anim, rigged = None, None
        if self.animate.isChecked():
            from ..anim3d import animated_turnaround
            target = model
            if not names:
                say("Adding a skeleton and animations…")
                out = self.work / (Path(model).stem + ".rigged.glb")
                if self.subject.currentText() == "animal":     # four legs: walk, run, sit…
                    from ..quadruped import auto_rig_quadruped
                    rig = auto_rig_quadruped(model, out)
                else:
                    from ..autorig import auto_rig
                    rig = auto_rig(model, out)
                if self.swing.isChecked():
                    from ..springs import add_springs
                    say("Swinging the capes, hair and tails…")
                    try:
                        add_springs(rig.path)
                    except ValueError as e:              # nothing hangs off this one
                        say(str(e))
                target, rigged, names = rig.path, rig.path, rig.clips
            say(f"Playing {len(names)} animations from every side…")
            anim, _ = animated_turnaround([target], s, n, self.frames.value(),
                                          elevation=self.elevation.value())
        return spin, sprites, anim, names, rigged

    def _paint_options(self) -> dict:
        from ..model3d import parse_colours
        text = self.only_colours.text().strip()
        colours = parse_colours(text) if text else None      # a typo shows before any GPU work
        return {"back": str(self.back_path) if self.back_path else None,
                "mirror_back": self.mirror.isChecked(), "colours": colours,
                "views": VIEW_CHOICES[self.views.currentText()]}

    def _shape(self) -> Path | None:
        """The uncoloured shape of the current model (to repaint without the GPU)."""
        res = self.result
        if res is None:
            return None
        for p in (res.raw, Path(res.model).with_suffix(".shape.glb")):
            if p and Path(p).exists():
                return Path(p)
        return None

    def pick_back(self):
        path, _ = QFileDialog.getOpenFileName(self, "Back picture (the other side)", "",
                                              IMAGE_FILTER)
        self.back_path = Path(path) if path else None
        self.back_btn.setText(f"Back: {self.back_path.name}" if self.back_path else "Back picture…")

    def _checked_options(self) -> bool:
        try:
            self._paint_options()
            return True
        except ValueError as e:
            self._failed(str(e))
            return False

    def repaint(self):
        shape = self._shape()
        if shape is None or not self._checked_options():
            return
        picture, seed = self.result.picture, self.result.seed
        drawn = {k: v for k, v in (getattr(self.result, "views", None) or {}).items()
                 if v and Path(v).exists()}
        job = self._model_job(prompt=self.prompt.text().strip() or None, picture=picture,
                              shape=shape, reuse=drawn)

        def keep_shape(say):
            res, *rest = job(say)
            res.raw, res.seed = shape, seed
            res.views = {**drawn, **res.views}
            return (res, *rest)
        self._run(keep_shape, self._show)

    def _edit_clicked(self, _checked=False):
        self.edit_picture()

    def edit_picture(self, rgba=None, restore_alpha=None) -> bool:
        """Clean up the picture, then build the model from it. Returns True if it builds."""
        from ..pipeline import load_image
        from .picture_editor import PictureEditor
        if rgba is None:
            if self.result is None or not self.result.picture:
                return False
            rgba = load_image(self.result.picture)
            restore_alpha = np.full(rgba.shape[:2], 255, np.uint8)
        dlg = PictureEditor(rgba, restore_alpha, self)
        try:
            ok = dlg.exec()
            edited = dlg.result().copy()
        finally:
            dlg.setParent(None)               # back to Python: freed once, when dlg goes
        if not ok:
            self.statusBar().showMessage("Picture not used.")
            return False
        return self.build_from(edited)

    def build_from(self, rgba: np.ndarray) -> bool:
        from PIL import Image
        if not self._checked_options():
            return False
        path = self.work / "picture.edited.png"
        Image.fromarray(rgba).save(path)
        self._run(self._model_job(prompt=self.prompt.text().strip() or None, picture=path),
                  self._show, "Building the model from the picture…",
                  plan=self._model_plan(picture=True), title="Model from a picture")
        return True

    def make(self):
        self.character = None             # something new: not (yet) in the library
        prompt = self.prompt.text().strip()
        if not prompt:
            self.statusBar().showMessage("Type what you want first.")
            return
        self._run(self._model_job(prompt=prompt), self._show, f"Making “{prompt}”…",
                  plan=self._model_plan(), title=f"Making “{prompt[:40]}”")

    def from_picture(self, path: str | None = None, edit: bool = True):
        """A photo or drawing -> cut out -> clean up (rub out a leash…) -> model."""
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, "Front picture", "", IMAGE_FILTER)
        if not path or not self._checked_options():
            return
        self.character = None
        if not edit:
            self._run(self._model_job(prompt=self.prompt.text().strip() or None, picture=path),
                      self._show, "Building the model from the picture…",
                      plan=self._model_plan(picture=True), title="Model from a picture")
            return
        from ..model3d import cut_out

        def job(say):
            say("Cutting out the subject…")
            return cut_out(path)
        self._run(job, lambda rgba: self.edit_picture(rgba, np.full(rgba.shape[:2], 255, np.uint8)))

    def open_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "3D model", "", MODEL_FILTER)
        if path:
            self.load_model(Path(path))

    def load_model(self, path: Path):
        """Skip the GPU: use a model made elsewhere (Tripo, Meshy, Blender, ...)."""
        self.character = None             # something new: not (yet) in the library
        from ..model3d import ModelResult

        def job(say):
            say("Rendering the model from every side…")
            return ModelResult(path, path, None, None), *self._sprite_job_body(path, say)
        self._run(job, self._show)

    def update_sprites(self):
        if self.result is None:
            return
        model = self.result.model
        self._run(lambda say: (self.result, *self._sprite_job_body(model, say)), self._show)

    # ---- results --------------------------------------------------------------
    def _show(self, payload):
        self.result, self.spin, self.sprites, self.anim, self.anim_names, self.rigged = payload
        res = self.result
        self.animate.setEnabled(True)
        self.animate.setText(f"Animate ({len(self.anim_names)})" if self.anim_names
                             else "Animate (adds a skeleton)")
        if res.picture and Path(res.picture).suffix.lower() == ".png":
            from ..pipeline import load_image
            self.picture_view.setPixmap(to_pixmap(checkerboard(_fit(load_image(res.picture)))))
        self._spin_i = 0
        self._next_spin()
        self._spin_timer.start(110)
        self._show_sprites()
        self._busy(False)
        seed = f" Seed {res.seed}." if res.seed is not None else ""
        if not self.anim:
            seed += (" It has animations: tick Animate and press Update sprites." if self.anim_names
                     else " Tick Animate and press Update sprites to make it walk, run, attack…")
        self.statusBar().showMessage(f"Done: {len(self.sprites)} directions.{seed} "
                                     "Change the sprite settings and press Update sprites, "
                                     "or Save all.")

    def _next_spin(self):
        if not self.spin:
            return
        frame = self.spin[self._spin_i % len(self.spin)]
        self.model_view.setPixmap(to_pixmap(checkerboard(_fit(frame))))
        self._spin_i += 1

    def _show_sprites(self, sprites=None):
        sprites = sprites or self.sprites
        n = len(sprites)
        cols = 4 if n > 4 else n
        h, w = sprites[0].rgba.shape[:2]
        sheet = np.zeros((h * ((n + cols - 1) // cols), w * cols, 4), np.uint8)
        for i, s in enumerate(sprites):
            r, c = divmod(i, cols)
            sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = s.rgba
        self.sprite_view.setPixmap(to_pixmap(checkerboard(_fit(sheet), cell=6)))
        self.sprite_view.setToolTip("Clockwise from facing right: " + " ".join(direction_names(n)))
        if sprites is self.sprites:          # a new result: restart the animation
            self._anim_i = 0
            if self.anim is not None:
                self._anim_timer.start(self.anim.actions[0].frame_ms)
            else:
                self._anim_timer.stop()

    def _next_anim_frame(self):
        """Every direction playing one action; each action twice, then the next."""
        a = self.anim
        if a is None:
            return
        per = 2 * a.frames
        k = self._anim_i % (per * len(a.actions))
        act, f = divmod(k, per)
        self._show_sprites([d[f % a.frames] for d in a.sprites[act]])
        self._anim_i = k + 1
        self._anim_timer.setInterval(a.actions[act].frame_ms)
        self.sprite_view.setToolTip(f"{a.actions[act].name}, clockwise from facing right")

    def save_all(self, target: str | None = None):
        if self.sprites is None:
            return
        if target is None:
            default = str(Path.cwd() / Path(self.result.model).name)
            target, _ = QFileDialog.getSaveFileName(self, "Save model and sprites", default,
                                                    "3D model (*.glb)")
            if not target:
                return
        out = Path(target).with_suffix(".glb")
        try:
            written = []
            out.parent.mkdir(parents=True, exist_ok=True)
            if Path(self.result.model).resolve() != out.resolve():
                shutil.copyfile(self.result.model, out)
            written.append(out)
            if self.result.picture and Path(self.result.picture).suffix.lower() == ".png" \
                    and Path(self.result.picture) != Path(self.result.model):
                pic = out.with_suffix(".front.png")
                shutil.copyfile(self.result.picture, pic)
                written.append(pic)
            sheet = out.with_name(out.stem + "_dirs.png")
            if self.rigged is not None:
                rigged = out.with_suffix(".rigged.glb")
                shutil.copyfile(self.rigged, rigged)
                written.append(rigged)
            if self.anim is not None:
                from ..anim3d import write_animated, write_animated_preview
                written += write_animated(self.anim, sheet, aseprite=True, godot=True)
                written.append(write_animated_preview(self.anim,
                                                      sheet.with_name(sheet.stem + "_anim.gif")))
            else:
                from ..model3d import export_for_rigging
                written += write_turnaround(self.sprites, sheet, aseprite=True, godot=True)
                written.append(write_preview(self.sprites, sheet.with_name(sheet.stem + "_spin.gif")))
                written.append(export_for_rigging(out, out.with_suffix(".obj")))   # for Mixamo
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Save failed", str(e))
            return []
        self.statusBar().showMessage("Saved " + ", ".join(p.name for p in written))
        return written


def _badged(pix: QPixmap) -> QPixmap:
    """A thumbnail with a "waiting for your OK" badge (an amber ? in the corner)."""
    from PySide6.QtGui import QFont, QPainter
    out = QPixmap(pix)
    p = QPainter(out)
    try:
        r = max(10, out.width() // 4)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QColor("#e0a020"))
        p.setPen(QColor("#1c2030"))
        p.drawEllipse(out.width() - r - 2, 2, r, r)
        f = QFont()
        f.setBold(True)
        f.setPixelSize(int(r * 0.8))
        p.setFont(f)
        p.drawText(out.width() - r - 2, 2, r, r, Qt.AlignCenter, "?")
    finally:
        p.end()
    return out


def _drawn_directions(ch) -> int:
    import json
    try:
        cam = json.loads((ch.folder / "keys" / "keys.json").read_text(encoding="utf-8"))["camera"]
        return int(cam["directions"])
    except (OSError, ValueError, KeyError):
        return 16


def animation_names(model: Path) -> list[str]:
    """Animation names in a .glb/.gltf (empty for other formats or unreadable files)."""
    if Path(model).suffix.lower() not in (".glb", ".gltf"):
        return []
    try:
        from ..anim3d import load_rig
        return [c.name for c in load_rig([model]).clips]
    except (ValueError, KeyError, OSError):
        return []


def main(argv: list[str] | None = None) -> int:
    app = QApplication.instance() or QApplication(sys.argv if argv is None else argv)
    win = ModelStudio()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())


MOCAP = "Mocap: "


def _mocap_moves() -> list[str]:
    """Motion-capture library names whose files are downloaded (mocap_library.json)."""
    try:
        from ..mocap import library_index, mocap_root
        root = mocap_root()
        return [n for n, e in library_index()["clips"].items() if (root / e["file"]).exists()]
    except Exception:  # noqa: BLE001 - no index, no mocap
        return []

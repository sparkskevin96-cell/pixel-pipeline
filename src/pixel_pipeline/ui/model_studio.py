"""Prompt to Model: type what you want, get a coloured 3D model and its direction sprites.

Prompt -> front picture -> 3D shape (Hunyuan3D in ComfyUI on this PC) -> painted from the
picture -> spinning preview + 4/8/16-direction sprites -> Save all (.glb, picture, sprite
sheet, .json, .aseprite, Godot SpriteFrames + facing script, spinning GIF).
A rigged, animated model (Open a 3D model…) can also be played: tick Animate.

Run with:  pixel-pipeline-model   (or File -> Prompt to 3D model in pixel-pipeline-ui)
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from ..settings import settings_for
from ..turnaround import (direction_names, render_model, turnaround_sprites, union_crop,
                          write_preview, write_turnaround)
from .common import checkerboard, to_pixmap

IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
MODEL_FILTER = "3D models (*.glb *.gltf *.obj *.ply *.stl)"
PANEL = 300


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
    """One window: prompt in, model + sprites out."""

    def __init__(self, client=None):
        super().__init__()
        self.setWindowTitle("Prompt to Model")
        self.resize(1080, 640)
        self.client = client              # ComfyClient; None = default address
        self.sync = False                 # tests: run jobs on the calling thread
        self.work = Path(tempfile.mkdtemp(prefix="pixel-pipeline-model-"))
        self.result = None                # ModelResult
        self.sprites = None
        self.anim = None                  # AnimatedSet when Animate is ticked
        self.anim_names: list[str] = []
        self._anim_i = 0
        self.spin: list[np.ndarray] = []
        self._spin_i = 0
        self._thread = None
        self._spin_timer = QTimer(self)
        self._spin_timer.timeout.connect(self._next_spin)
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._next_anim_frame)
        self._build()
        self.statusBar().showMessage("Type what you want and press Make model. Needs ComfyUI "
                                     "running with Hunyuan3D (see docs/LOCAL_AI.md).")

    # ---- layout ---------------------------------------------------------------
    def _build(self):
        self.prompt = QLineEdit()
        self.prompt.setPlaceholderText("e.g. a knight with a red cape and a big sword")
        self.prompt.returnPressed.connect(self.make)
        self.subject = QComboBox()
        self.subject.addItems(["person", "animal", "object"])
        self.seed = QSpinBox()
        self.seed.setRange(0, 2**31 - 1)
        self.seed.setSpecialValueText("random")
        self.make_btn = QPushButton("Make model")
        self.make_btn.setDefault(True)
        self.make_btn.clicked.connect(self.make)
        pic_btn = QPushButton("From a picture…")
        pic_btn.clicked.connect(self.from_picture)
        model_btn = QPushButton("Open a 3D model…")
        model_btn.clicked.connect(self.open_model)
        top = QHBoxLayout()
        top.addWidget(self.prompt, 1)
        for w in (QLabel("Subject"), self.subject, QLabel("Seed"), self.seed, self.make_btn,
                  pic_btn, model_btn):
            top.addWidget(w)

        self.picture_view = _label("Front picture")
        self.model_view = _label("3D model (spins)")
        self.sprite_view = _label("Direction sprites")
        panels = QHBoxLayout()
        for title, view in (("1. Picture", self.picture_view), ("2. Model", self.model_view),
                            ("3. Sprites", self.sprite_view)):
            box = QGroupBox(title)
            lay = QVBoxLayout(box)
            lay.addWidget(view)
            panels.addWidget(box, 1)

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
        self.animate.setToolTip("Open a rigged, animated .glb/.gltf (e.g. from Mixamo) to play "
                                "its animations from every direction")
        self.frames = QSpinBox()
        self.frames.setRange(1, 32)
        self.frames.setValue(8)
        self.frames.setToolTip("Frames per animation")
        self.redo_btn = QPushButton("Update sprites")
        self.redo_btn.clicked.connect(self.update_sprites)
        self.save_btn = QPushButton("Save all…")
        self.save_btn.clicked.connect(self.save_all)
        for b in (self.redo_btn, self.save_btn):
            b.setEnabled(False)
        form = QFormLayout()
        row = QHBoxLayout()
        for label, w in (("Directions", self.directions), ("Width", self.width_),
                         ("Height", self.height_), ("Colours", self.colors),
                         ("Camera tilt", self.elevation)):
            row.addWidget(QLabel(label))
            row.addWidget(w)
        row.addWidget(self.animate)
        row.addWidget(self.frames)
        row.addStretch(1)
        row.addWidget(self.redo_btn)
        row.addWidget(self.save_btn)
        form.addRow(row)

        root = QWidget()
        lay = QVBoxLayout(root)
        lay.addLayout(top)
        lay.addLayout(panels, 1)
        lay.addLayout(form)
        self.setCentralWidget(root)

    # ---- jobs -----------------------------------------------------------------
    def _run(self, job, on_done):
        """Run job(progress) off the UI thread; on_done(result) back on it."""
        self._busy(True)
        if self.sync:
            try:
                on_done(job(self.statusBar().showMessage))
            except Exception as e:  # noqa: BLE001
                self._failed(str(e))
            finally:
                self._busy(False)
            return
        thread, worker = QThread(self), _Worker(job)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self.statusBar().showMessage)
        worker.done.connect(on_done)
        worker.failed.connect(self._failed)
        for sig in (worker.done, worker.failed):
            sig.connect(thread.quit)
        thread.finished.connect(lambda: self._busy(False))
        thread.finished.connect(worker.deleteLater)
        self._thread, self._worker = thread, worker
        thread.start()

    def _busy(self, busy: bool):
        self.make_btn.setEnabled(not busy)
        self.redo_btn.setEnabled(not busy and self.result is not None)
        self.save_btn.setEnabled(not busy and self.sprites is not None)
        if busy:
            QApplication.setOverrideCursor(Qt.BusyCursor)
        else:
            QApplication.restoreOverrideCursor()

    def _failed(self, message: str):
        self.statusBar().showMessage("Stopped: " + message.splitlines()[0])
        QMessageBox.warning(self, "Couldn't make the model", message)

    def _model_job(self, prompt=None, picture=None, shape=None):
        from ..model3d import prompt_to_model
        seed = self.seed.value() or None
        subject = self.subject.currentText()
        name = "-".join("".join(c if c.isalnum() else " " for c in (prompt or "model").lower()).split())[:40]
        out = self.work / f"{name or 'model'}.glb"

        def job(say):
            res = prompt_to_model(prompt, out, subject=subject, picture=picture, shape=shape,
                                  client=self.client, seed=seed,
                                  progress=lambda step, text: say(text))
            say("Rendering the model from every side…")
            return res, *self._sprite_job_body(res.model, say)
        return job

    def _sprite_job_body(self, model: Path, say=lambda text: None):
        n = int(self.directions.currentText())
        s = settings_for(self.subject.currentText(), width=self.width_.value(),
                         height=self.height_.value(), colors=self.colors.value())
        views = render_model(model, n, self.elevation.value(), size=320)
        spin = union_crop(views)
        sprites = turnaround_sprites(union_crop(views, s.margin, s.alpha_threshold), s)
        names = animation_names(model)
        anim = None
        if names and self.animate.isChecked():
            from ..anim3d import animated_turnaround
            say(f"Playing {len(names)} animations from every side…")
            anim, _ = animated_turnaround([model], s, n, self.frames.value(),
                                          elevation=self.elevation.value())
        return spin, sprites, anim, names

    def make(self):
        prompt = self.prompt.text().strip()
        if not prompt:
            self.statusBar().showMessage("Type what you want first.")
            return
        self._run(self._model_job(prompt=prompt), self._show)

    def from_picture(self):
        path, _ = QFileDialog.getOpenFileName(self, "Front picture", "", IMAGE_FILTER)
        if path:
            self._run(self._model_job(prompt=self.prompt.text().strip() or None, picture=path),
                      self._show)

    def open_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "3D model", "", MODEL_FILTER)
        if path:
            self.load_model(Path(path))

    def load_model(self, path: Path):
        """Skip the GPU: use a model made elsewhere (Tripo, Meshy, Blender, ...)."""
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
        self.result, self.spin, self.sprites, self.anim, self.anim_names = payload
        res = self.result
        self.animate.setEnabled(bool(self.anim_names))
        if self.anim_names:
            self.animate.setText(f"Animate ({len(self.anim_names)})")
            self.animate.setToolTip("Animations: " + ", ".join(self.anim_names))
        if res.picture and Path(res.picture).suffix.lower() == ".png":
            from ..pipeline import load_image
            self.picture_view.setPixmap(to_pixmap(checkerboard(_fit(load_image(res.picture)))))
        self._spin_i = 0
        self._next_spin()
        self._spin_timer.start(110)
        self._show_sprites()
        self._busy(False)
        seed = f" Seed {res.seed}." if res.seed is not None else ""
        if self.anim_names and not self.anim:
            seed += " It has animations: tick Animate and press Update sprites."
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

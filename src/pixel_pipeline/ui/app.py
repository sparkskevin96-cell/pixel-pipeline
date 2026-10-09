"""Desktop app: every tool in one window, one tab each.

Sprite: drop in a photo, tweak settings, watch the sprite update live (plus video and idle
animations). Directions, Teams, Portrait, Map tiles, Godot demo and Prompt to Model are in
ui/tools.py and ui/model_studio.py.

Run with:  pixel-pipeline-ui [photo]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGroupBox, QInputDialog, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QScrollArea, QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from ..animate import IDLE_MS, IDLE_STYLES, SUBJECT_IDLE, idle_frames
from ..aseprite import write_aseprite
from ..cutout import CutoutUnavailable, remove_background
from ..export import write_animation
from ..palettes import BUILTIN
from ..pipeline import finish, load_image, prepare
from ..pixelize import DOWNSCALE_MODES
from ..quantize import DITHER_MODES
from ..settings import ANCHORS, FITS, OUTLINE_STYLES, SUBJECTS, Settings, settings_for
from ..systems import SYSTEMS
from ..video import ALIGN_MODES, video_to_sprites
from .common import checkerboard, to_pixmap

IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.heic *.heif *.webp *.bmp *.tif *.tiff *.gif)"
VIDEO_FILTER = "Videos (*.mp4 *.mov *.m4v *.avi *.webm *.mkv)"
CUTOUT_MODELS = ["isnet-general-use", "u2net_human_seg", "birefnet-general-lite", "birefnet-general",
                 "birefnet-portrait", "isnet-anime", "u2netp"]
LOAD_FILE, LOSPEC = "Load file…", "Lospec…"
ORIGINAL_MAX = 420
ICON = Path(__file__).with_name("icon.png")


def _spin(lo, hi, step=1, decimals=None):
    box = QDoubleSpinBox() if decimals is not None else QSpinBox()
    box.setRange(lo, hi)
    box.setSingleStep(step)
    if decimals is not None:
        box.setDecimals(decimals)
    return box


def _combo(items, editable=False):
    box = QComboBox()
    box.addItems(list(items))
    box.setEditable(editable)
    return box


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pixel Pipeline")
        self.resize(1280, 820)
        self.setAcceptDrops(True)
        self.source: np.ndarray | None = None
        self.source_path: Path | None = None
        self.alpha_cache: dict[str, np.ndarray] = {}
        self.sprite = None
        self.anim: dict | None = None  # {"frames", "ms", "palette", "tag", "path"} from a video
        self._anim_i = 0
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._next_anim_frame)
        self._loading = False
        self._palette_prev = "auto"

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self.render)

        self._build_controls()
        self._build_views()
        self._build_menu()
        self.apply_preset("object")
        self.statusBar().showMessage("Drop a photo onto the window, or File → Open (Ctrl+O)")

    # ---- layout ---------------------------------------------------------------
    def _build_controls(self):
        self.subject = _combo(SUBJECTS)
        self.system = QComboBox()
        self.system.addItem("Off (free colours)", "off")
        for key, sysm in SYSTEMS.items():
            self.system.addItem(sysm.name, key)
        self.look = QComboBox()
        self.look.addItem("Readable (console colours, recognizable size)", "readable")
        self.look.addItem("Authentic (one tiny sprite of that era)", "authentic")
        self.look.setEnabled(False)
        self.width_ = _spin(1, 1024)
        self.height_ = _spin(0, 1024)
        self.height_.setSpecialValueText("auto")
        self.fit = _combo(FITS)
        self.anchor = _combo(ANCHORS)
        self.remove_bg = QCheckBox("Remove background")
        self.model = _combo(CUTOUT_MODELS, editable=True)
        self.crop = QCheckBox("Crop to subject")
        self.patch = _spin(2, 32)
        self.detail = _spin(0, 6)
        self.downscale = _combo(DOWNSCALE_MODES)
        self.brightness = _spin(-1, 1, 0.05, 2)
        self.contrast = _spin(0, 3, 0.05, 2)
        self.saturation = _spin(0, 3, 0.05, 2)
        self.tileable = QCheckBox("Tileable (seamless)")
        self.palette = _combo(["auto", *BUILTIN, LOAD_FILE, LOSPEC])
        self.colors = _spin(2, 64)
        self.dither = _combo(DITHER_MODES)
        self.dither_strength = _spin(0, 2, 0.1, 2)
        self.cleanup = QCheckBox("Clean up stray pixels")
        self.outline = QCheckBox("1px outline")
        self.outline_style = _combo(OUTLINE_STYLES)
        self.outline_color = _combo(["auto", "#000000"], editable=True)
        self.zoom = _spin(1, 32)
        self.zoom.setValue(8)
        self.export_scale = _spin(1, 32)

        groups = [
            ("Subject", [("Type", self.subject), ("Retro system", self.system), ("Look", self.look),
                         ("", self.remove_bg), ("Model", self.model),
                         ("", self.crop)]),
            ("Size", [("Width", self.width_), ("Height", self.height_), ("Fit", self.fit),
                      ("Anchor", self.anchor)]),
            ("Pixels", [("Sample patch", self.patch), ("Detail", self.detail),
                        ("Downscale", self.downscale), ("Brightness", self.brightness),
                        ("Contrast", self.contrast), ("Saturation", self.saturation),
                        ("", self.tileable)]),
            ("Colour", [("Palette", self.palette), ("Colours (auto)", self.colors),
                        ("Dither", self.dither), ("Dither strength", self.dither_strength)]),
            ("Finish", [("", self.cleanup), ("", self.outline), ("Outline style", self.outline_style),
                        ("Outline colour", self.outline_color)]),
            ("View / export", [("Preview zoom", self.zoom), ("PNG export scale", self.export_scale)]),
        ]
        panel = QWidget()
        column = QVBoxLayout(panel)
        for title, rows in groups:
            box = QGroupBox(title)
            form = QFormLayout(box)
            for label, widget in rows:
                form.addRow(label, widget)
            column.addWidget(box)
        column.addStretch(1)
        self.controls = QScrollArea()
        self.controls.setWidget(panel)
        self.controls.setWidgetResizable(True)
        self.controls.setMinimumWidth(300)

        for w in (self.width_, self.height_, self.patch, self.detail, self.brightness,
                  self.contrast, self.saturation, self.colors, self.dither_strength):
            w.valueChanged.connect(self.schedule)
        for w in (self.fit, self.anchor, self.model, self.downscale, self.dither, self.outline_style,
                  self.outline_color):
            w.currentTextChanged.connect(self.schedule)
        for w in (self.remove_bg, self.crop, self.tileable, self.cleanup, self.outline):
            w.toggled.connect(self.schedule)
        self.zoom.valueChanged.connect(self.show_sprite)
        self.subject.currentTextChanged.connect(self.apply_preset)
        self.system.currentIndexChanged.connect(lambda *_: self.apply_preset(self.subject.currentText()))
        self.look.currentIndexChanged.connect(lambda *_: self.apply_preset(self.subject.currentText()))
        self.palette.currentTextChanged.connect(self._palette_changed)

    def _build_views(self):
        self.original_view = QLabel("Drop a photo here")
        self.original_view.setAlignment(Qt.AlignCenter)
        self.original_view.setMinimumWidth(260)
        self.sprite_view = QLabel()
        self.sprite_view.setAlignment(Qt.AlignCenter)
        sprite_scroll = QScrollArea()
        sprite_scroll.setWidget(self.sprite_view)
        sprite_scroll.setWidgetResizable(True)
        split = QSplitter()
        split.addWidget(self.controls)
        split.addWidget(self.original_view)
        split.addWidget(sprite_scroll)
        split.setSizes([320, 420, 500])

        from .model_studio import ModelStudio
        from .tools import DemoTab, DirectionsTab, MapTab, PortraitTab, PromptTab, TeamsTab
        say = self.statusBar().showMessage
        self.prompt_tab = PromptTab(say, send_to_sprite=self.use_candidate)
        self.directions_tab = DirectionsTab(say)
        self.teams_tab = TeamsTab(say, current_sprite=lambda: self.sprite)
        self.portrait_tab = PortraitTab(say)
        self.map_tab = MapTab(say)
        self.demo_tab = DemoTab(say)
        self.studio = ModelStudio()
        self.studio.setWindowFlags(Qt.Widget)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        for widget, title, tip in (
            (split, "Sprite", "Photo, picture or video -> sprite, idle and walk animations"),
            (self.prompt_tab, "Prompt", "Words -> sprite from PixelLab, Retro Diffusion and your "
                                        "GPU, side by side"),
            (self.directions_tab, "Directions", "3D model, photos or video -> 4/8/16 facing "
                                                "directions, animated for rigged models"),
            (self.teams_tab, "Teams", "Player / enemy / ally colours + Godot palette shader"),
            (self.portrait_tab, "Portrait", "Head-and-shoulders dialogue portrait"),
            (self.map_tab, "Map tiles", "Terrain tiles with transitions + Godot TileSet"),
            (self.demo_tab, "Godot demo", "A Godot project using it all"),
            (self.studio, "Prompt to Model", "Prompt -> 3D model on your GPU -> direction sprites"),
        ):
            self.tabs.addTab(widget, title)
            self.tabs.setTabToolTip(self.tabs.count() - 1, tip)
        self.setCentralWidget(self.tabs)

    def _build_menu(self):
        menu = self.menuBar().addMenu("&File")
        for text, shortcut, slot in (
            ("&Open photo…", QKeySequence.Open, self.open_dialog),
            ("Export &PNG…", "Ctrl+E", self.export_png),
            ("Export .&aseprite…", "Ctrl+Shift+E", self.export_aseprite),
            ("Export &idle animation…", "Ctrl+I", self.export_idle),
            (None, None, None),
            ("Open &video…", "Ctrl+Shift+O", self.open_video),
            ("&Generate from prompt (local GPU)…", "Ctrl+G", self.generate_dialog),
            ("Prompt to 3D &model + directions (local GPU)", "Ctrl+M", self.open_model_studio),
            ("Export video a&nimation…", "Ctrl+Shift+A", self.export_video_animation),
            (None, None, None),
            ("&Quit", QKeySequence.Quit, self.close),
        ):
            if text is None:
                menu.addSeparator()
                continue
            action = QAction(text, self)
            action.setShortcut(shortcut)
            action.triggered.connect(slot)
            menu.addAction(action)

    # ---- settings <-> controls -----------------------------------------------
    def apply_preset(self, subject: str):
        system = self.system.currentData() or "off"
        s = settings_for(subject, system=system, look=self.look.currentData())
        self.palette.setEnabled(system == "off")
        self.look.setEnabled(system != "off")
        self._loading = True
        try:
            self.subject.blockSignals(True)
            self.subject.setCurrentText(subject)
            self.subject.blockSignals(False)
            self.width_.setValue(s.width)
            self.height_.setValue(s.height or 0)
            self.fit.setCurrentText(s.fit)
            self.anchor.setCurrentText(s.anchor)
            self.remove_bg.setChecked(s.remove_background)
            self.model.setCurrentText(s.cutout_model)
            self.crop.setChecked(s.crop_to_subject)
            self.patch.setValue(s.patch)
            self.detail.setValue(s.detail)
            self.downscale.setCurrentText(s.downscale)
            self.brightness.setValue(s.brightness)
            self.contrast.setValue(s.contrast)
            self.saturation.setValue(s.saturation)
            self.tileable.setChecked(s.tileable)
            self.colors.setValue(s.colors)
            self.dither.setCurrentText(s.dither)
            self.dither_strength.setValue(s.dither_strength)
            self.cleanup.setChecked(s.cleanup)
            self.outline.setChecked(s.outline)
            self.outline_style.setCurrentText(s.outline_style)
            self.outline_color.setCurrentText(s.outline_color)
        finally:
            self._loading = False
        self.schedule()

    def current_settings(self) -> Settings:
        return Settings(
            system=self.system.currentData() or "off", look=self.look.currentData() or "readable",
            subject=self.subject.currentText(),
            width=self.width_.value(), height=self.height_.value() or None,
            fit=self.fit.currentText(), anchor=self.anchor.currentText(),
            remove_background=self.remove_bg.isChecked(), cutout_model=self.model.currentText(),
            crop_to_subject=self.crop.isChecked(), patch=self.patch.value(),
            detail=self.detail.value(), downscale=self.downscale.currentText(),
            brightness=self.brightness.value(), contrast=self.contrast.value(),
            saturation=self.saturation.value(), tileable=self.tileable.isChecked(),
            palette=self.palette.currentText(), colors=self.colors.value(),
            dither=self.dither.currentText(), dither_strength=self.dither_strength.value(),
            cleanup=self.cleanup.isChecked(), outline=self.outline.isChecked(),
            outline_style=self.outline_style.currentText(),
            outline_color=self.outline_color.currentText(),
        )

    def _palette_changed(self, text: str):
        if self._loading:
            return
        choice = None
        if text == LOAD_FILE:
            path, _ = QFileDialog.getOpenFileName(
                self, "Palette file", "", "Palettes (*.gpl *.hex *.txt *.png *.gif)")
            choice = path or None
        elif text == LOSPEC:
            slug, ok = QInputDialog.getText(
                self, "Lospec palette", "Palette slug from lospec.com/palette-list (e.g. resurrect-64):")
            choice = f"lospec:{slug.strip().lower()}" if ok and slug.strip() else None
        else:
            self._palette_prev = text
            self.schedule()
            return
        self._loading = True
        try:
            if choice:
                if self.palette.findText(choice) < 0:
                    self.palette.insertItem(self.palette.count() - 2, choice)
                self.palette.setCurrentText(choice)
                self._palette_prev = choice
            else:
                self.palette.setCurrentText(self._palette_prev)
        finally:
            self._loading = False
        self.schedule()

    # ---- rendering -------------------------------------------------------------
    def schedule(self, *_):
        if not self._loading and self.source is not None:
            self._timer.start()

    def _isolated(self, s: Settings) -> np.ndarray:
        if not s.remove_background or (self.source[..., 3] < 255).any():
            return self.source
        alpha = self.alpha_cache.get(s.cutout_model)
        if alpha is None:
            self.statusBar().showMessage(
                f"Removing background with {s.cutout_model}… (first use downloads the model)")
            QApplication.setOverrideCursor(Qt.WaitCursor)
            QApplication.processEvents()
            try:
                alpha = remove_background(np.ascontiguousarray(self.source[..., :3]), s.cutout_model)
            finally:
                QApplication.restoreOverrideCursor()
            self.alpha_cache[s.cutout_model] = alpha
        out = self.source.copy()
        out[..., 3] = alpha
        return out

    def render(self):
        if self.source is None:
            return
        self._stop_anim()
        s = self.current_settings()
        try:
            rgba = self._isolated(s)
            self.sprite = finish(prepare(rgba, s), s)
        except CutoutUnavailable as e:
            self._loading = True
            self.remove_bg.setChecked(False)
            self._loading = False
            QMessageBox.warning(self, "Background removal not installed", str(e))
            self.schedule()
            return
        except Exception as e:  # noqa: BLE001 - show any pipeline error instead of crashing the UI
            self.statusBar().showMessage(f"Error: {e}")
            return
        self._show_original(rgba)
        self.show_sprite()
        w, h = self.sprite.size
        self.statusBar().showMessage(f"{w}×{h} px · {len(self.sprite.palette)} colours")

    def _show_original(self, rgba: np.ndarray):
        h, w = rgba.shape[:2]
        scale = min(1.0, ORIGINAL_MAX / max(h, w))
        pix = to_pixmap(checkerboard(rgba, cell=max(4, int(12 / scale))))
        self.original_view.setPixmap(pix.scaled(int(w * scale), int(h * scale),
                                                Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def show_sprite(self, *_):
        if self.anim is not None:
            self._show_rgba(self.anim["frames"][self._anim_i])
        elif self.sprite is not None:
            self._show_rgba(self.sprite.rgba)

    def _show_rgba(self, rgba: np.ndarray):
        z = self.zoom.value()
        big = np.repeat(np.repeat(rgba, z, 0), z, 1)
        self.sprite_view.setPixmap(to_pixmap(checkerboard(big, cell=max(4, z))))

    # ---- video -----------------------------------------------------------------
    def _stop_anim(self):
        self._anim_timer.stop()
        self.anim = None

    def _next_anim_frame(self):
        if self.anim is None:
            return
        self._anim_i = (self._anim_i + 1) % len(self.anim["frames"])
        self.show_sprite()

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open video", "", VIDEO_FILTER)
        if not path:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Video to animation")
        form = QFormLayout(dlg)
        frames, start, end = _spin(2, 64), _spin(0, 3600, 0.1, 2), _spin(0, 3600, 0.1, 2)
        frames.setValue(8)
        end.setSpecialValueText("end of clip")
        align, tag = _combo(ALIGN_MODES), QLineEdit("walk")
        form.addRow("Frames", frames)
        form.addRow("Start (s)", start)
        form.addRow("End (s)", end)
        form.addRow("Align", align)
        form.addRow("Animation name", tag)
        form.addRow(QLabel("Uses the current settings on the left. 'track' animates in place."))
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        if dlg.exec() != QDialog.Accepted:
            return
        self.load_video(Path(path), frames.value(), start.value() or None, end.value() or None,
                        align.currentText(), tag.text().strip() or "walk")

    def load_video(self, path: Path, frames: int = 8, start=None, end=None, align="track",
                   tag="walk"):
        self.statusBar().showMessage(f"Converting {path.name}… (background removal runs per frame)")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            sprites, ms = video_to_sprites(path, self.current_settings(), frames, start, end,
                                           align=align)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Video conversion failed", str(e))
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.anim = {"frames": [s.rgba for s in sprites], "ms": ms, "palette": sprites[0].palette,
                     "tag": tag, "path": path}
        self._anim_i = 0
        self._anim_timer.start(ms)
        self.show_sprite()
        w, h = sprites[0].size
        self.statusBar().showMessage(
            f"{path.name}: {len(sprites)} frames {w}×{h} @ {ms} ms · File → Export video animation")

    def export_video_animation(self):
        if self.anim is None:
            QMessageBox.information(self, "No animation", "Open a video first (File → Open video).")
            return
        stem = self.anim["path"].stem
        default = str(self.anim["path"].parent / f"{stem}.{self.anim['tag']}.png")
        path, _ = QFileDialog.getSaveFileName(self, "Export animation", default, "Sprite sheet (*.png)")
        if not path:
            return
        try:
            written = write_animation(self.anim["frames"], Path(path), self.anim["ms"],
                                      self.anim["tag"], aseprite=True, godot=True,
                                      palette=self.anim["palette"])
            self.statusBar().showMessage("Saved " + ", ".join(p.name for p in written))
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Export failed", str(e))

    # ---- files -----------------------------------------------------------------
    def open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open photo", "", IMAGE_FILTER)
        if path:
            self.open_path(Path(path))

    def open_path(self, path: Path):
        try:
            rgba = load_image(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Couldn't open image", f"{path}\n\n{e}")
            return
        self.set_source(rgba, path)

    def generate_dialog(self):
        """Prompt -> ComfyUI on this PC -> the raw image becomes the source, so every setting
        on the left still applies live."""
        prompt, ok = QInputDialog.getText(self, "Generate from prompt",
                                          "What should your GPU make? (needs ComfyUI running, see docs/LOCAL_AI.md)")
        if not ok or not prompt.strip():
            return
        from ..generate import ComfyUIError, generate_sprite
        s = self.current_settings()
        self.statusBar().showMessage(f"Generating “{prompt}” on your GPU…")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            (_, raw, seed), = generate_sprite(prompt, s, subject=s.subject,
                                              system=None if s.system == "off" else s.system)
        except (ComfyUIError, ValueError) as e:
            QMessageBox.warning(self, "Generation failed", str(e))
            return
        finally:
            QApplication.restoreOverrideCursor()
        slug = "-".join("".join(c if c.isalnum() else " " for c in prompt.lower()).split())[:40]
        self.set_source(load_image(raw), Path.cwd() / f"{slug or 'generated'}.png")
        self.statusBar().showMessage(f"Generated “{prompt}” (seed {seed}). Tweak anything on the left.")

    def open_model_studio(self):
        self.tabs.setCurrentWidget(self.studio)

    def use_candidate(self, cand):
        """A Prompt tab result -> Sprite tab source, so every setting there applies."""
        slug = "-".join("".join(c if c.isalnum() else " " for c in self.prompt_tab.prompt.text().lower())
                        .split())[:40]
        self.remove_bg.setChecked(False)          # services already give a transparent background
        self.set_source(load_image(cand.raw), Path.cwd() / f"{slug or 'sprite'}-{cand.provider}.png")
        self.tabs.setCurrentIndex(0)

    def set_source(self, rgba, path: Path):
        self.source = rgba
        self.source_path = path
        self.alpha_cache = {}
        self.setWindowTitle(f"pixel-pipeline — {path.name}")
        self._show_original(self.source)
        self.render()

    def _save_path(self, suffix: str, filt: str) -> str:
        stem = self.source_path.stem if self.source_path else "sprite"
        default = str((self.source_path.parent if self.source_path else Path.cwd()) / f"{stem}.sprite{suffix}")
        path, _ = QFileDialog.getSaveFileName(self, "Export", default, filt)
        return path

    def export_png(self):
        if self.sprite is None:
            return
        path = self._save_path(".png", "PNG (*.png)")
        if path:
            try:
                self.sprite.save(path, self.export_scale.value())
                self.statusBar().showMessage(f"Saved {path}")
            except Exception as e:  # noqa: BLE001
                QMessageBox.warning(self, "Export failed", str(e))

    def export_aseprite(self):
        if self.sprite is None:
            return
        path = self._save_path(".aseprite", "Aseprite (*.aseprite *.ase)")
        if path:
            try:
                write_aseprite(path, [self.sprite.rgba], self.sprite.palette)
                self.statusBar().showMessage(f"Saved {path}")
            except Exception as e:  # noqa: BLE001
                QMessageBox.warning(self, "Export failed", str(e))

    def export_idle(self):
        """Idle loop as sprite sheet + .aseprite + Godot SpriteFrames."""
        if self.sprite is None:
            return
        styles = list(IDLE_STYLES)
        default = styles.index(SUBJECT_IDLE.get(self.subject.currentText(), "breathe"))
        style, ok = QInputDialog.getItem(self, "Idle animation", "Style:", styles, default, False)
        if not ok:
            return
        path = self._save_path(f".{style}.png", "Sprite sheet (*.png)")
        if not path:
            return
        try:
            frames = idle_frames(self.sprite.rgba, style)
            written = write_animation(frames, Path(path), IDLE_MS[style], "idle", aseprite=True, godot=True,
                                      palette=self.sprite.palette)
            self.statusBar().showMessage("Saved " + ", ".join(p.name for p in written))
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Export failed", str(e))

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            self.open_path(Path(urls[0].toLocalFile()))


def main(argv: list[str] | None = None) -> int:
    app = QApplication(sys.argv if argv is None else argv)
    app.setApplicationName("Pixel Pipeline")
    if ICON.exists():
        app.setWindowIcon(QIcon(str(ICON)))
    args = app.arguments()[1:]
    win = MainWindow()
    win.show()
    if "--model" in args:  # straight into Prompt to Model
        win.open_model_studio()
    elif args:
        win.open_path(Path(args[0]))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

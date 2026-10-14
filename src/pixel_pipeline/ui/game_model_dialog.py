"""Library -> "Export 3D model…": build a game-ready 3D model (gamemodel.py) for a character,
item or prop and export it for Godot and/or Unity, with a turning preview of the result.

    dlg = GameModelDialog(asset, parent)
    if dlg.exec() == QDialog.Accepted:
        opts, engines, folder = dlg.values()
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QVBoxLayout,
)

from .. import gamemodel
from .common import PixelView

LOOK_CHOICES = (("Smooth (1024 texture, normal map)", "smooth"),
                ("Pixel-crisp low poly (small texture, nearest, palette)", "pixel"))
DETAIL_CHOICES = (("Light", 0.5), ("Normal", 1.0), ("Detailed", 1.6))
ENGINE_CHOICES = (("Godot + Unity", ("godot", "unity")), ("Godot", ("godot",)),
                  ("Unity", ("unity",)), ("Just build it (model_game folder)", ()))


def can_make(asset) -> bool:
    """A library character / item / prop with a 3D model."""
    if asset is None or getattr(asset, "kind", "") not in ("character", "item", "prop"):
        return False
    folder = Path(asset.folder)
    return (folder / "model.glb").exists() or (folder / "rigged.glb").exists()


def turntable_frames(asset) -> list[np.ndarray]:
    """The 8 preview frames of <asset>/model_game/preview.png (empty if not built)."""
    p = Path(asset.folder) / gamemodel.FOLDER / "preview.png"
    if not p.exists():
        return []
    strip = np.asarray(Image.open(p).convert("RGBA"))
    h = strip.shape[0]
    return [strip[:, i:i + h] for i in range(0, strip.shape[1] - h + 1, h)]


def info_text(asset) -> str:
    p = Path(asset.folder) / gamemodel.FOLDER / "info.json"
    if not p.exists():
        return "Not built yet: Export builds it first (about 1-3 minutes, no GPU)."
    d = json.loads(p.read_text(encoding="utf-8"))
    lods = ", ".join(f"LOD{i} {l['tris']:,}" for i, l in enumerate(d["lods"]))
    extra = f"; {d['bones']} bones, {len(d['clips'])} clips" if d.get("skinned") else ""
    return (f"Built ({d['look']}): {lods} triangles, {d['lods'][0]['texture']} px texture"
            f"{extra}. From {d['source_tris']:,} triangles.")


class GameModelDialog(QDialog):
    def __init__(self, asset, parent=None, folder: str = ""):
        super().__init__(parent)
        self.asset = asset
        self.setWindowTitle(f"3D model: {asset.name}")
        self.view = PixelView("No preview yet", (256, 256))
        self.frames = turntable_frames(asset)
        self.frame = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        if self.frames:
            self.view.show_rgba(self.frames[0])
            self.timer.start(160)
        self.info = QLabel(info_text(asset))
        self.info.setWordWrap(True)
        self.look = QComboBox()
        for label, key in LOOK_CHOICES:
            self.look.addItem(label, key)
        try:
            from ..assets import current_project
            want = gamemodel.default_look(current_project().style)
            self.look.setCurrentIndex([k for _, k in LOOK_CHOICES].index(want))
        except Exception:  # noqa: BLE001 - no project: smooth
            pass
        self.detail = QComboBox()
        for label, k in DETAIL_CHOICES:
            self.detail.addItem(label, k)
        self.detail.setCurrentIndex(1)
        self.engine = QComboBox()
        for label, k in ENGINE_CHOICES:
            self.engine.addItem(label, k)
        self.rebuild = QCheckBox("Build again (else reuse the last build)")
        self.rebuild.setChecked(not self.frames)
        self.folder = QLineEdit(folder)
        browse = QPushButton("Choose…")
        browse.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addWidget(self.folder, 1)
        row.addWidget(browse)
        form = QFormLayout()
        form.addRow("Look", self.look)
        form.addRow("Detail", self.detail)
        form.addRow("For", self.engine)
        form.addRow("Into folder", row)
        form.addRow("", self.rebuild)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Build + export")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(self.view, 1)
        lay.addWidget(self.info)
        lay.addLayout(form)
        lay.addWidget(buttons)

    def _tick(self):
        if self.frames:
            self.frame = (self.frame + 1) % len(self.frames)
            self.view.show_rgba(self.frames[self.frame])

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Export into", self.folder.text())
        if d:
            self.folder.setText(d)

    def values(self) -> tuple[dict, tuple, str, bool]:
        """(build options for gamemodel.game_model_for, engines, folder, rebuild)."""
        self.timer.stop()
        kind_budget = 12_000 if self.asset.kind == "character" else (
            2_500 if self.asset.kind == "item" else 4_000)
        opts = {"look": self.look.currentData(),
                "budget": int(kind_budget * self.detail.currentData())}
        return opts, self.engine.currentData(), self.folder.text().strip(), self.rebuild.isChecked()


def build_and_export(asset, opts: dict, engines, folder, rebuild: bool = True, say=None) -> list[Path]:
    """The job behind the dialog: (re)build model_game, then export for each engine into
    <folder>/<engine>/."""
    from .. import exports
    files: list[Path] = []
    if rebuild or not gamemodel.has_game_model(asset):
        g = gamemodel.game_model_for(asset, progress=say, **opts)
        files += g.files
    for e in engines:
        out = Path(folder) / e if len(engines) > 1 else Path(folder)
        files += exports.export_asset(asset, f"{e}-3d-model", out)
    return files

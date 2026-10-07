import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
try:
    from PySide6.QtWidgets import QApplication
except ImportError as e:  # missing system GL libs on headless machines
    pytest.skip(f"Qt unavailable: {e}", allow_module_level=True)

from pixel_pipeline.ui.app import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_preset_and_render(app, blob_png):
    win = MainWindow()
    assert win.subject.currentText() == "object"
    win.apply_preset("person")
    assert win.subject.currentText() == "person"
    assert (win.width_.value(), win.height_.value()) == (32, 48)
    win.open_path(blob_png)
    assert win.sprite is not None and win.sprite.size == (32, 48)
    win.palette.setCurrentText("gameboy")
    win.render()
    assert len(win.sprite.palette) == 4


def test_video_import_animates_and_exports(app, tmp_path):
    from test_animation import make_video
    clip = tmp_path / "clip.mp4"
    make_video(clip)
    win = MainWindow()
    win.apply_preset("object")
    win.remove_bg.setChecked(False)
    win.load_video(clip, frames=4)
    assert win.anim is not None and len(win.anim["frames"]) == 4
    win._next_anim_frame()
    assert win._anim_i == 1
    from pixel_pipeline.export import write_animation
    written = write_animation(win.anim["frames"], tmp_path / "walk.png", win.anim["ms"], "walk",
                              aseprite=True, godot=True, palette=win.anim["palette"])
    assert all(p.exists() for p in written)

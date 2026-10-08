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


def test_model_studio_makes_saves_and_updates(app, tmp_path, monkeypatch):
    trimesh = pytest.importorskip("trimesh")
    import numpy as np
    from PIL import Image

    import pixel_pipeline.model3d as model3d
    from pixel_pipeline.ui.model_studio import ModelStudio

    def fake_prompt_to_model(prompt, out, *, progress=None, seed=None, **kw):
        body = trimesh.creation.box(extents=[0.6, 1.2, 0.4])
        body.visual.vertex_colors = np.tile(np.array([200, 40, 40, 255], np.uint8), (8, 1))
        body.export(out)
        pic = out.with_suffix(".front.png")
        Image.new("RGBA", (64, 64), (200, 40, 40, 255)).save(pic)
        progress("picture", "Making a front picture")
        return model3d.ModelResult(out, pic, None, 42)

    monkeypatch.setattr(model3d, "prompt_to_model", fake_prompt_to_model)
    win = ModelStudio()
    win.sync = True
    win.prompt.setText("a red box knight")
    win.directions.setCurrentText("8")
    win.make()
    assert len(win.sprites) == 8 and len(win.spin) == 8 and win.result.seed == 42
    assert not win.model_view.pixmap().isNull() and not win.sprite_view.pixmap().isNull()
    assert win.save_btn.isEnabled()
    written = win.save_all(str(tmp_path / "knight.glb"))
    names = {p.name for p in written}
    assert {"knight.glb", "knight.front.png", "knight_dirs.png", "knight_dirs.tres",
            "knight_dirs_facing.gd", "knight_dirs_spin.gif"} <= names
    win.directions.setCurrentText("4")
    win.update_sprites()
    assert len(win.sprites) == 4
    assert not win.animate.isEnabled()                      # a still model: nothing to play


def test_model_studio_plays_a_rigged_model(app, tmp_path):
    from test_anim3d import write_robot
    from pixel_pipeline.ui.model_studio import ModelStudio
    robot = write_robot(tmp_path / "robot.glb")
    win = ModelStudio()
    win.sync = True
    win.directions.setCurrentText("4")
    win.width_.setValue(24)
    win.height_.setValue(24)
    win.load_model(robot)
    assert win.animate.isEnabled() and win.anim is None and win.anim_names == ["wave", "slide", "fall"]
    win.animate.setChecked(True)
    win.frames.setValue(3)
    win.update_sprites()
    assert win.anim is not None and win.anim.frames == 3 and len(win.anim.actions) == 3
    for _ in range(4):
        win._next_anim_frame()
    assert win._anim_i == 4 and "wave" in win.sprite_view.toolTip()
    names = {p.name for p in win.save_all(str(tmp_path / "out" / "robot.glb"))}
    assert {"robot_dirs.tres", "robot_dirs_facing.gd", "robot_dirs_anim.gif"} <= names

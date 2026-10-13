"""Characters tab: the Views choice and adding an animation to a library character."""
import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")


@pytest.fixture
def app():
    from conftest import qt_app
    return qt_app()


def test_views_choice_reaches_prompt_to_model(app, tmp_path, monkeypatch):
    from PIL import Image

    import pixel_pipeline.model3d as model3d
    from pixel_pipeline.ui.model_studio import ModelStudio
    calls = []

    def fake(prompt, out, *, progress=None, seed=None, **kw):
        calls.append(kw)
        trimesh.creation.box(extents=[0.6, 1.2, 0.4]).export(out.with_suffix(".shape.glb"))
        body = trimesh.creation.box(extents=[0.6, 1.2, 0.4])
        body.visual.vertex_colors = np.tile(np.array([200, 40, 40, 255], np.uint8), (8, 1))
        body.export(out)
        pic = out.with_suffix(".front.png")
        Image.new("RGBA", (64, 64), (200, 40, 40, 255)).save(pic)
        back = out.with_suffix(".back.png")
        Image.new("RGBA", (64, 64), (40, 200, 40, 255)).save(back)
        return model3d.ModelResult(out, pic, out.with_suffix(".shape.glb"), 1,
                                   {} if kw.get("shape") else {"back": back})

    monkeypatch.setattr(model3d, "prompt_to_model", fake)
    win = ModelStudio()
    win.sync = True
    win.directions.setCurrentText("4")
    win.prompt.setText("a knight")
    assert win.views.currentText() == "auto"
    win.make()
    assert calls[-1]["views"] is None
    win.views.setCurrentText("all four")
    win.make()
    assert calls[-1]["views"] == "all"
    win.repaint()                                  # repaint keeps the back drawn before
    assert calls[-1]["back"].endswith(".back.png") and calls[-1]["shape"] is not None
    win.close()


def test_add_animation_to_a_library_character(app, tmp_path, monkeypatch):
    from test_autorig import humanoid

    from pixel_pipeline import library
    from pixel_pipeline.assets import Style
    from pixel_pipeline.autorig import auto_rig
    from pixel_pipeline.ui.model_studio import ModelStudio
    small = Style(directions=4, person_height=24)
    monkeypatch.setattr(library, "style", lambda: small)
    ch = library.Character("Mage", "person", "normal", "mage", 1.0, library.root() / "mage")
    ch.folder.mkdir(parents=True)
    auto_rig(humanoid(tmp_path / "h.glb"), ch.rigged, actions=("idle",))
    library.render_sheets(ch.rigged, ch.sheet, "person", "normal", frames=2)
    library.make_thumb(ch.sheet, ch.thumb)
    ch.save()
    win = ModelStudio()
    win.sync = True
    win.open_character(ch)
    items = [win.more_anim.itemText(i) for i in range(win.more_anim.count())]
    assert "cast" in items and "idle" not in items and win.add_anim_btn.isEnabled()
    win.more_anim.setCurrentText("cast")
    win.add_animation()
    assert win.last_error == ""
    assert win.anim_names == ["idle", "cast"]
    assert "cast" not in [win.more_anim.itemText(i) for i in range(win.more_anim.count())]
    win.close()

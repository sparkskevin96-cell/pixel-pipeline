"""Procedural effects: every kind renders, loops wrap, files load, palettes are respected."""
import json

import numpy as np
import pytest
from PIL import Image

from pixel_pipeline import effects
from pixel_pipeline.assets import Style, create_project, list_assets
from pixel_pipeline.palettes import BUILTIN, hex_to_rgb


@pytest.mark.parametrize("kind", effects.EFFECT_KINDS)
def test_every_kind_renders(kind):
    fx = effects.render_effect(kind)
    (w, h), n = effects.SPECS[kind][0], effects.SPECS[kind][1]
    assert fx.size == (w, h) and len(fx.frames) == n
    alpha = np.stack([f[..., 3] for f in fx.frames])
    assert np.isin(alpha, (0, 255)).all()                         # binary alpha
    assert all(f[..., 3].any() for f in fx.frames[:-1])          # something on (nearly) every frame
    cols = np.unique(np.concatenate([f[f[..., 3] > 0][:, :3] for f in fx.frames]), axis=0)
    assert len(cols) <= 12                                        # limited palette
    assert not np.array_equal(fx.frames[0], fx.frames[len(fx.frames) // 2])   # it moves
    again = effects.render_effect(kind)
    assert all(np.array_equal(a, b) for a, b in zip(fx.frames, again.frames))  # deterministic


@pytest.mark.parametrize("kind", ["fire", "magic_circle", "sparkle", "arrow_trail"])
def test_loops_wrap(kind):
    """Frame n (one past the end) equals frame 0: render 2n frames, compare n with 0."""
    n = effects.SPECS[kind][1]
    a = effects.render_effect(kind, frames=n)
    b = effects.render_effect(kind, frames=2 * n)
    assert a.loop and np.array_equal(b.frames[0], a.frames[0])
    assert np.array_equal(b.frames[2 * n - 2], a.frames[n - 1])  # same motion at half speed


def test_theme_colours_size_frames():
    fire = effects.render_effect("fire", theme="ice")
    ice = set(map(tuple, hex_to_rgb(effects.THEMES["ice"]).tolist()))
    used = {tuple(c) for f in fire.frames for c in f[f[..., 3] > 0][:, :3].tolist()}
    assert used <= ice
    one = effects.render_effect("hit", colours=["#ff00aa"], size=48, frames=8)
    assert one.size == (48, 48) and len(one.frames) == 8 and len(one.ramp) == 6
    two = effects.render_effect("hit", colours="ffffff 330000")
    assert two.ramp[0].tolist() == [51, 0, 0]                    # sorted dark -> bright
    st = Style(person_height=32)
    assert effects.render_effect("explosion", style=st).size == (32, 32)
    assert effects.effect_size("fire", (10, 20)) == (10, 20)
    with pytest.raises(ValueError):
        effects.render_effect("nope")
    with pytest.raises(ValueError):
        effects.render_effect("fire", theme="lava")


def test_project_palette_snaps():
    pico = set(map(tuple, hex_to_rgb(BUILTIN["pico-8"]).tolist()))
    for kind in ("explosion", "heal"):
        fx = effects.render_effect(kind, style=Style(palette="pico-8"))
        used = {tuple(c) for f in fx.frames for c in f[f[..., 3] > 0][:, :3].tolist()}
        assert used <= pico


def test_make_effect_files(tmp_path):
    from test_aseprite import read_ase
    proj = create_project("FX", Style(person_height=48))
    asset = effects.make_effect("explosion", project=proj, theme="poison")
    assert asset.kind == "effect" and asset.name == "Explosion (poison)"
    assert asset.extra["effect"] == "explosion" and asset.extra["size"] == [48, 48]
    assert {"effect", "explosion", "poison"} <= set(asset.tags)
    f = asset.folder
    sheet = Image.open(f / "sheet.png")
    assert sheet.size == (48 * 10, 48)
    meta = json.loads((f / "sheet.json").read_text())
    assert len(meta["frames"]) == 10 and meta["meta"]["frameTags"][0]["name"] == "explosion"
    assert meta["meta"]["loop"] is False
    tres = (f / "sheet.tres").read_text()
    assert '"loop": false' in tres and 'name": &"explosion"' in tres and "sheet.png" in tres
    ase = read_ase(f / "sheet.aseprite")
    assert len(ase["frames"]) == 10 and ase["depth"] == 8           # indexed
    assert ase["tags"][0][:3] == ("explosion", 0, 9)
    gif = Image.open(f / "preview.gif")
    assert gif.n_frames == 11                                    # + a pause for one-shots
    assert Image.open(f / "thumb.png").size == (96, 96)
    assert [a.name for a in list_assets("effect", project=proj)] == ["Explosion (poison)"]
    loop = effects.make_effect("fire", "Torch", project=proj)
    assert loop.folder.name == "torch" and '"loop": true' in (loop.folder / "sheet.tres").read_text()


def test_effects_tab(tmp_path):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from conftest import qt_app
    qt_app()
    from pixel_pipeline.ui.effects_tab import EffectsTab
    tab = EffectsTab()
    tab.sync = True
    tab.timer.stop()
    assert tab.kinds.count() == len(effects.EFFECT_KINDS)
    tab.select_kind("fire")
    first = tab.view.rgba.copy()
    tab.tick()
    assert not np.array_equal(first, tab.view.rgba)            # the big preview plays
    tab.theme.setCurrentIndex(tab.theme.findData("poison"))
    tab.frames.setValue(6)
    assert len(tab.fx.frames) == 6
    got = []
    tab.changed.connect(lambda: got.append(1))
    tab.name.setText("Green flame")
    tab.make()
    assert got == [1] and tab.saved.count() == 1
    assert tab.last.extra["theme"] == "poison" and tab.last.extra["frames"] == 6
    tab.saved.setCurrentRow(0)
    assert "Green flame" in tab.info.text()
    tab.colours.setText("#zzz")
    tab.preview()                                             # bad colours -> message, no crash

"""Procedural effects: every kind renders, loops wrap, files load, palettes are respected."""
import json

import numpy as np
import pytest
from PIL import Image

from pixel_pipeline import effects
from pixel_pipeline.assets import Style, create_project, list_assets
from pixel_pipeline.palettes import BUILTIN, hex_to_rgb


@pytest.mark.parametrize("kind", effects.CLASSIC_KINDS)
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


@pytest.mark.parametrize("kind", ["heal", "level_up"])
def test_optional_loops_wrap(kind):
    n = effects.SPECS[kind][1]
    a = effects.render_effect(kind, frames=n, loop=True)
    b = effects.render_effect(kind, frames=2 * n, loop=True)
    assert np.array_equal(b.frames[0], a.frames[0])
    assert np.array_equal(b.frames[2 * n - 2], a.frames[n - 1])


def _drift_x(fx):
    """How far the visible pixels' middle moves sideways from the 2nd frame to the 2nd-last."""
    a, b = fx.frames[1], fx.frames[-2]
    return float(np.nonzero(b[..., 3])[1].mean() - np.nonzero(a[..., 3])[1].mean())


@pytest.mark.parametrize("kind", ["smoke", "dust"])
def test_wind_pushes_smoke_and_dust(kind):
    still = effects.render_effect(kind)
    right = effects.render_effect(kind, wind=1.0)
    left = effects.render_effect(kind, wind=-1.0)
    if kind == "smoke":                     # it drifts downwind as it rises
        assert _drift_x(left) < -1 and _drift_x(right) > 1 and abs(_drift_x(still)) < 3
    else:                                   # the kicked-up dust leans downwind
        mid = lambda fx: np.mean([np.nonzero(f[..., 3])[1].mean() for f in fx.frames])  # noqa: E731
        assert mid(left) < mid(still) - 1 < mid(right) - 2
    # other kinds ignore it
    assert all(np.array_equal(a, b) for a, b in zip(effects.render_effect("hit").frames,
                                                       effects.render_effect("hit", wind=1).frames))


def test_smoke_breaks_up():
    """Smoke thins out: the late frames are smaller than the peak and have holes or pieces."""
    import cv2
    fx = effects.render_effect("smoke")
    area = [int((f[..., 3] > 0).sum()) for f in fx.frames]
    peak = int(np.argmax(area))
    assert 0 < peak < len(area) - 2 and area[-1] < area[peak] * 0.5
    pieces = [cv2.connectedComponents((f[..., 3] > 0).astype(np.uint8), connectivity=4)[0] - 1
              for f in fx.frames[peak:]]
    assert max(pieces) >= 3                                       # wisps, not one shrinking ball
    shades = {tuple(c) for c in fx.frames[1][fx.frames[1][..., 3] > 0][:, :3].tolist()}
    late = {tuple(c) for c in fx.frames[-2][fx.frames[-2][..., 3] > 0][:, :3].tolist()}
    lum = lambda cs: np.mean([sum(c) for c in cs])               # noqa: E731
    assert lum(late) < lum(shades)                                # fades darker as it goes


def test_dark_theme_rim():
    """The dark theme's outer dark pixels get a light rim, so it shows on dark ground."""
    rim = tuple(hex_to_rgb([effects.RIMS["dark"]])[0].tolist())
    fx = effects.render_effect("slash", theme="dark")
    used = {tuple(c) for f in fx.frames for c in f[f[..., 3] > 0][:, :3].tolist()}
    assert rim in used and fx.rim is not None
    plain = effects.render_effect("slash", theme="dark", rim=False)
    assert rim not in {tuple(c) for f in plain.frames for c in f[f[..., 3] > 0][:, :3].tolist()}
    darkest = tuple(hex_to_rgb([effects.THEMES["dark"][0]])[0].tolist())
    for f in fx.frames:                                           # no near-black on the outside
        m = f[..., 3] > 0
        edge = effects._outer_edge(m)
        assert darkest not in {tuple(c) for c in f[edge][:, :3].tolist()}
    fire = effects.render_effect("fire")                          # other themes: no rim
    assert fire.rim is None


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
    smoke = effects.make_effect("smoke", project=proj, wind=-0.5, theme="dark")
    assert smoke.extra["wind"] == -0.5 and smoke.extra["rim"]


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
    assert not tab.wind.isEnabled()
    tab.select_kind("smoke")
    assert tab.wind.isEnabled()
    tab.wind.setCurrentIndex(tab.wind.findData(1.0))
    assert tab.options()["wind"] == 1.0
    tab.colours.setText("#zzz")
    tab.preview()                                             # bad colours -> message, no crash


def test_effects_tab_premium(tmp_path):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from conftest import qt_app
    qt_app()
    from pixel_pipeline.ui.effects_tab import EffectsTab
    tab = EffectsTab()
    tab.sync = True
    tab.timer.stop()
    assert tab.current_kind() == effects.SPELL_KINDS[0]
    tab.select_kind("sword_combo")
    assert tab.hits.isEnabled() and not tab.angle.isEnabled() and not tab.dither.isEnabled()
    tab.size_name.setCurrentIndex(tab.size_name.findData("small"))
    tab.hits.setValue(2)
    assert max(tab.fx.size) == 64 and len(tab.fx.tag_list()) == 2
    assert tab.options()["hits"] == 2 and "dither" not in tab.options()
    dark = tab.view.rgba.copy()
    tab.ground.setCurrentIndex(tab.ground.findData("light"))
    assert not np.array_equal(dark, tab.view.rgba)                # light ground
    tab.glow.setChecked(False)
    tab.frame_i = 3
    tab._show()
    plain = tab.view.rgba.copy()
    tab.glow.setChecked(True)
    assert not np.array_equal(plain, tab.view.rgba)               # bloom on
    tab.frame_i = 0
    tab._timed()                                                  # real timing: next frame's ms
    assert tab.timer.interval() == tab.fx.frame_durations()[1]
    tab.timer.stop()
    tab.select_kind("hit_spark")
    tab.angle.setCurrentIndex(1)
    assert abs(tab.options()["angle"] - 3.1416) < 1e-3
    tab.make()
    assert tab.last.extra["effect"] == "hit_spark" and (tab.last.folder / "sheet_glow.png").exists()

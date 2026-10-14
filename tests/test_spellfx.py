"""Premium effects (spellfx.py + fxcore.py): every kind renders crisp, many frames with real
timing, glow maps, loops wrap, files + exports carry the timing, Godot loads them."""
import json
import math
import os
import shutil
import subprocess

import cv2
import numpy as np
import pytest
from PIL import Image

from pixel_pipeline import effects, exports, fxcore, spellfx
from pixel_pipeline.assets import Style, create_project
from pixel_pipeline.palettes import BUILTIN, hex_to_rgb

GODOT = os.environ.get("GODOT_BIN") or shutil.which("godot")


def _cols(fx):
    return {tuple(c) for f in fx.frames for c in f[f[..., 3] > 0][:, :3].tolist()}


@pytest.mark.parametrize("kind", spellfx.SPELL_KINDS)
def test_every_spell_renders(kind):
    fx = effects.render_effect(kind, size="small")
    n = spellfx.SPELLS[kind][2]
    assert len(fx.frames) == n and 12 <= n <= 32
    assert max(fx.size) == 64
    alpha = np.stack([f[..., 3] for f in fx.frames])
    assert np.isin(alpha, (0, 255)).all()                         # binary alpha, no AA
    assert sum(f[..., 3].any() for f in fx.frames) >= n - 1
    assert len(_cols(fx)) >= 10                                   # rich ramps, not 4 colours
    assert len(fx.durations) == n and all(d % 10 == 0 and d >= 20 for d in fx.durations)
    if not fx.loop:
        assert len(set(fx.durations)) > 1                         # anticipation/impact/fade timing
    # glow: same layout, only where the effect is
    assert len(fx.glow) == n and fx.glow[0].shape == fx.frames[0].shape
    for f, g in zip(fx.frames, fx.glow):
        assert not ((g[..., 3] > 0) & (f[..., 3] == 0)).any()
    assert any(g[..., 3].any() for g in fx.glow)
    # crisp: hardly any lone pixels (single-pixel noise)
    lone = total = 0
    for f in fx.frames:
        m = (f[..., 3] > 0).astype(np.uint8)
        k, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        lone += int((stats[1:, cv2.CC_STAT_AREA] == 1).sum())
        total += int(m.sum())
    assert lone <= max(12, total * 0.02)
    again = effects.render_effect(kind, size="small")
    assert all(np.array_equal(a, b) for a, b in zip(fx.frames, again.frames))   # deterministic


def test_sizes_themes_palette():
    assert spellfx.spell_size("fireball", "huge") == (192, 120)
    assert spellfx.spell_size("fire_burst", None, Style(person_height=32)) == (48, 48)
    assert spellfx.spell_size("fire_burst", (50, 40)) == (50, 40)
    with pytest.raises(ValueError):
        spellfx.spell_size("fire_burst", "giant")
    for theme in spellfx.SPELL_THEMES[:-1]:
        fx = effects.render_effect("hit_spark", theme=theme, size=48)
        ramps = spellfx.spell_ramps(theme)
        allowed = {tuple(c) for r in ramps.values() for c in r.tolist()}
        assert _cols(fx) <= allowed and fx.theme == theme
    # hue-shifted fire ramp: deep purple dark end, yellow-white top
    fire = spellfx.spell_ramps("fire")["main"].astype(int)
    assert fire[0][2] > fire[0][1] and fire[-1].min() > 200
    pico = set(map(tuple, hex_to_rgb(BUILTIN["pico-8"]).tolist()))
    fx = effects.render_effect("hit_spark", theme="project", style=Style(palette="pico-8"), size=48)
    assert _cols(fx) <= pico
    mine = effects.render_effect("hit_spark", colours="#ff00aa", size=48)
    assert mine.theme is None and len(_cols(mine)) >= 4
    with pytest.raises(ValueError):
        effects.render_effect("fire_burst", theme="lava")
    with pytest.raises(ValueError):
        effects.render_effect("fire_burst", frames=3)


def test_frames_rescale_keeps_time():
    a = effects.render_effect("hit_spark", size=48)
    b = effects.render_effect("hit_spark", size=48, frames=20)
    assert len(b.frames) == 20
    assert abs(sum(a.durations) - sum(b.durations)) <= 20 * 10


def _wraps(kind, **opts):
    """Frame n (one loop later) is frame 0: draw the beat at p = 1 too."""
    _, _, n0, theme, *_, maker = spellfx.SPELLS[kind]
    w, h = spellfx.spell_size(kind, 48)
    ramps = spellfx.spell_ramps(theme)

    def frames(p_shift):
        ctx = fxcore.Ctx(w, h, np.random.default_rng(0), dict(opts))
        beats, draw, loops = maker(ctx, None)
        loop_tag = next(t for t, lp in loops.items() if lp)
        b = next(b for b in beats if b.tag == loop_tag)
        b.p += p_shift
        cv = ctx.canvas()
        draw(cv, b)
        return fxcore.finish(cv, ramps)[0]
    return np.array_equal(frames(0.0), frames(1.0))


@pytest.mark.parametrize("kind,opts", [("poison_cloud", {}), ("buff_aura", {}), ("fireball", {}),
                                       ("holy_light", {"_loop": True})])
def test_loops_wrap(kind, opts):
    assert _wraps(kind, **opts)


def test_tags_hits_angle():
    fb = effects.render_effect("fireball", size="small")
    assert [(t, lp) for t, _, _, lp in fb.tags] == [("fly", True), ("impact", False)]
    assert fb.tags[0][1] == 0 and fb.tags[1][2] == len(fb.frames) - 1
    one = effects.render_effect("sword_combo", size=48, hits=1)
    three = effects.render_effect("sword_combo", size=48)
    assert [t[0] for t in one.tags] == ["swing1"] and len(three.tags) == 3
    with pytest.raises(ValueError):
        effects.render_effect("sword_combo", hits=4)
    hl = effects.render_effect("holy_light", size=48, loop=True)
    assert hl.loop and len(set(hl.durations)) == 1

    def spray_x(fx):
        f = fx.frames[6]
        return np.nonzero(f[..., 3])[1].mean() - f.shape[1] / 2
    assert spray_x(effects.render_effect("hit_spark", size=64)) > 3
    assert spray_x(effects.render_effect("hit_spark", size=64, angle=math.pi)) < -3


def test_make_effect_files_and_exports(tmp_path):
    from test_aseprite import read_ase
    proj = create_project("FX2", Style(person_height=64))
    asset = effects.make_effect("fireball", project=proj, theme="ice", size="small")
    assert asset.name == "Fireball + impact (ice)" and asset.extra["blend"] == "add"
    f = asset.folder
    for name in ("sheet.png", "sheet_glow.png", "sheet.json", "sheet.aseprite", "sheet.tres",
                 "sheet_glow.tres", "sheet_add.tres", "sheet_demo.tscn", "sheet_demo.gd", "preview.gif",
                 "thumb.png"):
        assert (f / name).exists(), name
    assert Image.open(f / "sheet.png").size == Image.open(f / "sheet_glow.png").size
    meta = json.loads((f / "sheet.json").read_text())
    fx = effects.render_effect("fireball", theme="ice", size="small")
    assert [fr["duration"] for fr in meta["frames"]] == fx.durations
    assert meta["meta"]["loops"] == {"fly": True, "impact": False} and meta["meta"]["blend"] == "add"
    assert [t["name"] for t in meta["meta"]["frameTags"]] == ["fly", "impact"]
    tres = (f / "sheet.tres").read_text()
    assert 'name": &"fly"' in tres and 'name": &"impact"' in tres and '"duration": 2.25' in tres
    assert "blend_mode = 1" in (f / "sheet_add.tres").read_text()
    ase = read_ase(f / "sheet.aseprite")
    assert [fr["duration"] for fr in ase["frames"]] == fx.durations
    assert Image.open(f / "preview.gif").n_frames == len(fx.frames)      # loops: no pause frame
    # unity: one clip per animation, its own loop flag and frame times
    plans = {p["name"]: p for p in exports.unity_clip_plan(exports.source(asset))}
    assert plans["fly"]["loop"] and not plans["impact"]["loop"]
    assert plans["impact"]["length"] == pytest.approx(sum(fx.durations[8:]) / 1000)
    out = exports.export_asset(asset, "unity", tmp_path / "unity")
    names = {p.name for p in out}
    assert {"fireball_impact_ice.png", "fireball_impact_ice.png.meta", "impact.anim"} <= names
    g2 = exports.export_asset(asset, "godot-2d", tmp_path / "godot")
    assert any(p.name.endswith("_glow.png") for p in g2)
    assert '"loop": false' in (tmp_path / "godot" / "fireball_impact_ice.tres").read_text()


@pytest.mark.skipif(not GODOT, reason="set GODOT_BIN to load the effect in Godot")
def test_godot_loads_effect(tmp_path):
    proj = create_project("FX3", Style())
    asset = effects.make_effect("sword_combo", project=proj, size="small", hits=2)
    game = tmp_path / "game"
    shutil.copytree(asset.folder, game / "fx")
    (game / "project.godot").write_text('config_version=5\n\n[application]\n\nconfig/name="fx"\n')
    subprocess.run([GODOT, "--headless", "--path", str(game), "--import"], capture_output=True, timeout=300)
    (game / "check.gd").write_text(
        'extends SceneTree\nvar demo\nvar n := 0\n'
        'func _initialize() -> void:\n'
        '\tvar sf: SpriteFrames = load("res://fx/sheet.tres")\n'
        '\tfor a in sf.get_animation_names():\n'
        '\t\tvar t := 0.0\n'
        '\t\tfor i in sf.get_frame_count(a):\n'
        '\t\t\tt += sf.get_frame_duration(a, i) / sf.get_animation_speed(a)\n'
        '\t\tprint("ANIM ", a, " ", sf.get_frame_count(a), " ", sf.get_animation_loop(a), " ", int(round(t * 1000)))\n'
        '\tvar m = load("res://fx/sheet_add.tres")\n'
        '\tprint("BLEND ", m.blend_mode)\n'
        '\tdemo = load("res://fx/sheet_demo.tscn").instantiate()\n'
        '\troot.add_child(demo)\n'
        'func _process(_d: float) -> bool:\n'
        '\tn += 1\n'
        '\tif n == 20:\n'
        '\t\tprint("PLAYING ", demo.get_node("Fx").is_playing(), " ", demo.get_node("Fx").animation)\n'
        '\t\treturn true\n'
        '\treturn false\n')
    run = subprocess.run([GODOT, "--headless", "--path", str(game), "-s", "check.gd"],
                         capture_output=True, text=True, timeout=180)
    log = run.stdout + run.stderr
    fx = effects.render_effect("sword_combo", size="small", hits=2)
    for name, a, b, _ in fx.tags:
        assert f"ANIM {name} {b - a + 1} false {sum(fx.durations[a:b + 1])}" in log, log[-2000:]
    assert "BLEND 1" in log and "PLAYING true swing1" in log, log[-2000:]
    assert "SCRIPT ERROR" not in log and "Parse Error" not in log, log[-2000:]


def test_paint_effect_with_fake_comfy(comfy):
    from pixel_pipeline import fxpaint
    fake, client = comfy
    fx = effects.render_effect("hit_spark", size=48)
    st = Style(style_prompt="moody watercolour")
    painted = fxpaint.paint_effect(fx, client=client, style=st)
    assert len(fake.prompts) == 1                                  # the whole sheet in one call
    wf = json.dumps(fake.prompts[0])
    assert "spell effect" in wf and "moody watercolour" in wf
    assert painted.durations == fx.durations and painted.glow is fx.glow
    pal = {tuple(c) for c in fx.palette.tolist()}
    for a, b in zip(fx.frames, painted.frames):
        assert np.array_equal(a[..., 3], b[..., 3])                # same silhouettes
        assert {tuple(c) for c in b[b[..., 3] > 0][:, :3].tolist()} <= pal


from test_generate import comfy, white_key  # noqa: E402,F401  (pytest fixtures)


def test_previews(tmp_path):
    fxs = [effects.render_effect(k, size=40) for k in ("hit_spark", "poison_cloud")]
    spellfx.contact_sheet(fxs, tmp_path / "c.png")
    spellfx.preview_gif(fxs, tmp_path / "c.gif")
    assert Image.open(tmp_path / "c.png").size[1] > 4 * 30
    g = Image.open(tmp_path / "c.gif")
    assert g.n_frames >= 10

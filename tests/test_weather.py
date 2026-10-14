"""Weather, water and parallax backgrounds (weather.py, water.py, backgrounds.py): loops that tile,
time-of-day colours, files for Godot / Unity, the HD-2D test bed, the Weather & sky page."""
import json
import os
import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image
from test_generate import comfy, white_key  # noqa: F401  (pytest fixtures)

from pixel_pipeline import assets, backgrounds as B, water as WA, weather as W

GODOT = os.environ.get("GODOT_BIN") or shutil.which("godot")


def _wrap_jump(img: np.ndarray, axis: int, q: float | None = None) -> tuple[float, float]:
    """(mean change across the wrap seam, mean change between interior neighbours) on `axis`;
    q: the interior's q-th percentile line instead of its mean (sparse pictures: rain, snow)."""
    a = img.astype(np.float32)
    seam = np.abs(np.take(a, 0, axis) - np.take(a, -1, axis)).mean()
    d = np.abs(np.diff(a, axis=axis))
    lines = d.mean(axis=1 - axis if d.ndim == 2 else tuple(i for i in range(d.ndim) if i != axis))
    inner = float(np.percentile(lines, q)) if q is not None else float(d.mean())
    return seam, inner


# --------------------------------------------------------------------------- weather

@pytest.mark.parametrize("kind", W.WEATHER_KINDS)
def test_every_weather_renders(kind):
    w = W.render_weather(kind, scale=1.0)
    n = W.KINDS[kind][2]
    assert len(w.fx.frames) == n and len(w.fx.durations) == n
    assert all(f.shape == w.fx.frames[0].shape for f in w.fx.frames)
    alpha = np.stack([f[..., 3] for f in w.fx.frames])
    assert (alpha > 0).any(axis=(1, 2)).sum() >= n - 2              # something on (nearly) every frame
    if kind in ("fog", "god_rays", "cloud_shadows", "sandstorm"):
        assert len(np.unique(alpha)) >= 4                            # banded see-through steps
    if kind == "lightning":
        assert not w.fx.loop and len(set(w.fx.durations)) > 1 and len(w.meta["flash"]) == n
    else:
        assert w.fx.loop and w.meta["wrap"] in ("xy", "x")
        assert not np.array_equal(w.fx.frames[0], w.fx.frames[n // 2])   # it moves
    if kind in ("embers", "fireflies", "god_rays", "lightning"):
        assert w.fx.glow is not None and any(g[..., 3].any() for g in w.fx.glow)
        for f, g in zip(w.fx.frames, w.fx.glow):
            assert not ((g[..., 3] > 0) & (f[..., 3] == 0)).any()     # glow only where it is
    if kind != "lightning":
        assert w.particles and w.emitters or kind == "cloud_shadows"
        for e in w.emitters:
            assert e.texture in w.particles and len(w.particles[e.texture]) == e.frames


def test_overlays_tile_seamlessly():
    """Drawn things that cross an edge come back on the other side: the wrap seam is no
    harder than any two neighbouring columns / rows."""
    for kind in ("rain", "snow", "fog", "leaves", "cloud_shadows"):
        w = W.render_weather(kind, scale=1.0)
        for axis in (0, 1):
            jumps = [_wrap_jump(f[..., 3], axis) for f in w.fx.frames]       # averaged over the loop
            seam, inner = np.mean([j[0] for j in jumps]), np.mean([j[1] for j in jumps])
            assert seam <= inner * 1.6 + 2, (kind, axis, seam, inner)


def test_time_of_day_and_palette():
    noon = W.render_weather("rain", tod="noon")
    night = W.render_weather("rain", tod="night")
    lum = lambda w: np.mean([f[f[..., 3] > 0][:, :3].mean() for f in w.fx.frames if f[..., 3].any()])  # noqa: E731
    assert lum(night) < lum(noon) - 10
    pal = np.array([[10, 10, 30], [60, 80, 120], [200, 220, 255]], np.uint8)
    w = W.render_weather("snow", palette=pal)
    cols = {tuple(c) for f in w.fx.frames for c in f[f[..., 3] > 0][:, :3].tolist()}
    assert cols <= {tuple(c) for c in pal.tolist()}
    with pytest.raises(ValueError, match="Unknown weather"):
        W.render_weather("hail")
    with pytest.raises(ValueError, match="time of day"):
        W.render_weather("rain", tod="midnight")
    side = W.render_weather("rain", style=assets.Style(tile_style="side"))
    assert not any(e.area == "ground" for e in side.emitters)      # no splashes on a side view


def test_make_weather_files(tmp_path):
    proj = assets.create_project("Wet", type="hd2d")
    a = W.make_weather("rain", project=proj, tod="night")
    assert a.kind == "weather" and a.folder.parent.name == "weather"
    f = a.folder
    for rel in ("sheet.png", "sheet.json", "sheet.tres", "sheet.aseprite", "preview.gif", "thumb.png",
                "particles/drop_near.png", "particles/splash.png", "weather.json", "sheet_overlay.tscn",
                "sheet_particles2d.tscn", "sheet_particles3d.tscn", "weather_overlay.gdshader",
                "weather_overlay_add.gdshader", "weather_fx.gd"):
        assert (f / rel).exists(), rel
    info = json.loads((f / "weather.json").read_text())
    assert info["loop"] and len(info["emitters"]) == 5
    near = next(e for e in info["emitters"] if e["name"] == "near")
    assert near["units"]["velocity"][1] < 0                          # y up for Unity / 3D
    meta = json.loads((f / "sheet.json").read_text())["meta"]
    assert meta["loop"] is True and meta["frameTags"][0]["name"] == "rain"
    strip = Image.open(f / "particles" / "splash.png")
    assert strip.size == (10 * 4, 7)                                 # 4 frames side by side
    assert a in assets.list_assets("weather", project=proj)


def test_exports_carry_the_extras(tmp_path):
    proj = assets.create_project("Ex", type="topdown-rpg")
    from pixel_pipeline.exports import export_asset
    a = W.make_weather("embers", project=proj)
    g = export_asset(a, "godot-2d", tmp_path / "g")
    names = {p.name for p in g}
    assert {"sheet_overlay.tscn", "sheet_particles2d.tscn", "weather_fx.gd", "ember.png"} <= names
    u = export_asset(a, "unity", tmp_path / "u")
    assert any(p.name == "ember.png.meta" for p in u) and any(p.name.endswith("_weather.json.meta") for p in u)
    wa = WA.make_water(project=proj, frames=4)
    u = export_asset(wa, "unity", tmp_path / "uw")
    assert any(p.name.endswith("shore_tileset.png.meta") for p in u)
    g = export_asset(wa, "godot-2d", tmp_path / "gw")
    assert any(p.name == "shore_tileset.tres" for p in g)
    bg = B.make_background("forest", "dawn", project=proj, width=160, height=90)
    u = export_asset(bg, "unity", tmp_path / "ub")
    assert {"PixelParallax.cs", "parallax.json"} <= {p.name for p in u}
    assert sum(p.suffix == ".meta" for p in u) == 6
    g = export_asset(bg, "godot-2d", tmp_path / "gb")
    assert any(p.suffix == ".tscn" for p in g) and not any(p.name == "sheet.png" for p in g)


@pytest.mark.skipif(not GODOT, reason="set GODOT_BIN to load the weather scenes in Godot")
def test_godot_loads_every_weather_scene(tmp_path):
    game = tmp_path / "game"
    game.mkdir()
    (game / "project.godot").write_text('config_version=5\n\n[application]\n\nconfig/name="w"\n')
    for k in W.WEATHER_KINDS:
        W.write_weather(W.render_weather(k), game / k)
    subprocess.run([GODOT, "--headless", "--path", str(game), "--import"], capture_output=True, timeout=300)
    lines = ["extends SceneTree", "var n := 0", "func _initialize() -> void:"]
    for k in W.WEATHER_KINDS:
        for s in ("overlay", "particles2d", "particles3d"):
            lines += [f'\tvar s_{k}_{s} = load("res://{k}/sheet_{s}.tscn")',
                      f'\tif s_{k}_{s} == null:', f'\t\tprint("FAIL {k} {s}")', "\telse:",
                      f'\t\tvar i_{k}_{s} = s_{k}_{s}.instantiate()', f'\t\troot.add_child(i_{k}_{s})',
                      f'\t\tprint("OK {k} {s} ", i_{k}_{s}.get_child_count())']
    lines += ['\tvar p: GPUParticles2D = load("res://rain/sheet_particles2d.tscn").instantiate().get_node("Near")',
              '\tprint("ALIGN ", p.process_material.get_particle_flag(ParticleProcessMaterial.PARTICLE_FLAG_ALIGN_Y_TO_VELOCITY), " ", p.process_material.gravity)',
              "func _process(_d: float) -> bool:", "\tn += 1", "\treturn n > 20"]
    (game / "check.gd").write_text("\n".join(lines) + "\n")
    run = subprocess.run([GODOT, "--headless", "--path", str(game), "-s", "check.gd"], capture_output=True,
                         text=True, timeout=300)
    log = run.stdout + run.stderr
    assert "SCRIPT ERROR" not in log and "Parse Error" not in log and "FAIL" not in log, log[-3000:]
    assert log.count("OK ") == 3 * len(W.WEATHER_KINDS), log[-3000:]
    assert "ALIGN true (0.0, 0.0, 0.0)" in log


# --------------------------------------------------------------------------- water

def test_water_loops_and_tiles():
    wt = WA.render_water(16, frames=6, seed=3)
    assert wt.n == 6 and wt.frames[0].shape == (16, 16, 4) and (wt.frames[0][..., 3] == 255).all()
    for f in wt.frames:
        for axis in (0, 1):
            seam, inner = _wrap_jump(f[..., :3], axis)
            assert seam <= inner * 2.5 + 4
    rng = np.random.default_rng(3)
    F = WA._fields(16, rng)
    end, _ = WA._open_frame(16, 1.0, F, wt.ramp)                       # one loop on = frame 0
    first, _ = WA._open_frame(16, 0.0, F, wt.ramp)
    assert np.array_equal(end, first)
    assert len(wt.shore) == 48 and all(len(v) == 6 for v in wt.shore.values())
    assert wt.side_surface[0][0, 0, 3] == 0 and wt.side_body[0][..., 3].min() > 0   # air above the crest
    assert wt.iso[0].shape == (8, 16, 4)
    lava = WA.render_water(16, theme="lava", frames=4)
    assert any(g[..., 3].any() for g in lava.glow)


def test_water_shores_match_their_neighbours():
    wt = WA.render_water(16, frames=4)
    img = WA.example_map(wt, 2, 12, 8)
    t = 16
    # every pair of neighbouring cells meets without a jump bigger than the texture's own
    jumps, inner = [], np.abs(np.diff(img[..., :3].astype(int), axis=1)).mean()
    for x in range(1, 12):
        a, b = img[:, x * t - 1, :3].astype(int), img[:, x * t, :3].astype(int)
        jumps.append(np.abs(a - b).mean())
    assert np.mean(jumps) <= inner * 2.5 + 4


def test_make_water_matches_a_tilegen_set():
    from pixel_pipeline import tilegen as tg
    proj = assets.create_project("Lake", type="topdown-rpg")
    sw = tg.builtin_swatch("water", 16, 0, proj.style)
    ts = tg.save_tileset(tg.build_tileset("water", sw.texture, base="sand", style=proj.style, project=proj),
                         "Lagoon", project=proj)
    a = WA.make_water(project=proj, frames=4)
    assert a.extra["from_tileset"] == ts.folder.name and a.extra["tile"] == 16 and a.extra["base"] == "sand"
    for rel in ("shore_tileset.png", "shore_tileset.tres", "shore_tileset.tsx", "water.png", "side_body.png",
                "iso_water.png", "example.gif", "sheet.png", "water.json"):
        assert (a.folder / rel).exists(), rel
    atlas = Image.open(a.folder / "shore_tileset.png")
    assert atlas.size == (8 * 4 * 16, 6 * 16)


@pytest.mark.skipif(not GODOT, reason="set GODOT_BIN to paint the animated water in Godot")
def test_godot_paints_animated_shores(tmp_path):
    game = tmp_path / "game"
    WA.write_water(WA.render_water(16, frames=6), game / "water")
    (game / "project.godot").write_text('config_version=5\n\n[application]\n\nconfig/name="w"\n')
    subprocess.run([GODOT, "--headless", "--path", str(game), "--import"], capture_output=True, timeout=300)
    (game / "check.gd").write_text(
        'extends SceneTree\nfunc _initialize() -> void:\n'
        '\tvar ts: TileSet = load("res://water/shore_tileset.tres")\n'
        '\tvar src: TileSetAtlasSource = ts.get_source(0)\n'
        '\tprint("TILES ", src.get_tiles_count(), " FRAMES ", src.get_tile_animation_frames_count(Vector2i(6, 0)))\n'
        '\tvar layer := TileMapLayer.new()\n\tlayer.tile_set = ts\n\troot.add_child(layer)\n'
        '\tvar ground: Array[Vector2i] = []\n\tvar lake: Array[Vector2i] = []\n'
        '\tfor y in 8:\n\t\tfor x in 10:\n\t\t\tground.append(Vector2i(x, y))\n'
        '\t\t\tif x >= 2 and x <= 7 and y >= 2 and y <= 5:\n\t\t\t\tlake.append(Vector2i(x, y))\n'
        '\tlayer.set_cells_terrain_connect(ground, 0, 0)\n\tlayer.set_cells_terrain_connect(lake, 0, 1)\n'
        '\tprint("MID ", layer.get_cell_atlas_coords(Vector2i(4, 3)), " CORNER ", layer.get_cell_atlas_coords(Vector2i(2, 2)))\n'
        '\tquit()\n')
    run = subprocess.run([GODOT, "--headless", "--path", str(game), "-s", "check.gd"], capture_output=True,
                         text=True, timeout=200)
    log = run.stdout + run.stderr
    assert "TILES 48 FRAMES 6" in log, log[-2000:]
    lay = {(c, r): m for m, c, r in WA.shore_layout(WA.render_water(16, frames=6))}
    assert f"MID ({7 * 6}, 5)" in log and lay[(7, 5)] == 255, log[-2000:]
    corner = log.split("CORNER (")[1].split(")")[0]
    cx, cy = (int(v) for v in corner.split(", "))
    assert cx % 6 == 0 and lay[(cx // 6, cy)] not in (-1, 255)        # an outer-corner shore tile


# --------------------------------------------------------------------------- backgrounds

@pytest.mark.parametrize("scene", list(B.SCENES))
def test_backgrounds_layer_and_wrap(scene):
    bg = B.render_background(scene, "sunset", width=240, height=135)
    assert [L.name for L in bg.layers] == list(B.LAYER_NAMES)
    assert (bg.layers[0].rgba[..., 3] == 255).all()                   # the sky is opaque
    scrolls = [L.scroll for L in bg.layers]
    assert scrolls == sorted(scrolls)                                  # far scrolls slowest
    for L in bg.layers[2:]:
        a = L.rgba[..., 3] > 0
        top = np.argmax(a, axis=0)
        steps = np.abs(np.diff(top.astype(int)))
        assert abs(int(top[0]) - int(top[-1])) <= max(6, int(steps.max())), (scene, L.name)   # it wraps
    comp = bg.composite()
    assert (comp[..., 3] == 255).all()
    assert len(np.unique(comp[..., :3].reshape(-1, 3), axis=0)) >= 24  # rich, not 4 colours


def test_times_of_day_differ():
    m = {t: B.render_background("mountains", t, width=160, height=90).composite()[..., :3].mean((0, 1)) for t in B.TODS}
    assert m["night"].sum() < m["dawn"].sum() < m["noon"].sum()
    assert m["sunset"][0] > m["sunset"][2]                              # warm


def test_background_files_and_godot_scene(tmp_path):
    bg = B.render_background("coast", "night", width=160, height=90)
    files = B.write_background(bg, tmp_path / "bg")
    tscn = (tmp_path / "bg" / "background_parallax.tscn").read_text()
    assert tscn.count('type="Parallax2D"') == 5 and "autoscroll = Vector2(-4, 0)" in tscn
    info = json.loads((tmp_path / "bg" / "parallax.json").read_text())
    assert [L["scroll"] for L in info["layers"]] == [L.scroll for L in bg.layers]
    assert (tmp_path / "bg" / "unity" / "PixelParallax.cs").exists() and (tmp_path / "bg" / "preview.gif").exists()
    assert len(files) > 10
    if GODOT:
        game = tmp_path / "game"
        shutil.copytree(tmp_path / "bg", game / "bg")
        (game / "project.godot").write_text('config_version=5\n\n[application]\n\nconfig/name="b"\n')
        subprocess.run([GODOT, "--headless", "--path", str(game), "--import"], capture_output=True, timeout=300)
        (game / "c.gd").write_text('extends SceneTree\nfunc _initialize() -> void:\n'
                                   '\tvar s = load("res://bg/background_parallax.tscn").instantiate()\n'
                                   '\troot.add_child(s)\n\tprint("LAYERS ", s.get_child_count(), " AUTO ", s.get_child(1).autoscroll)\n\tquit()\n')
        run = subprocess.run([GODOT, "--headless", "--path", str(game), "-s", "c.gd"], capture_output=True,
                             text=True, timeout=200)
        assert "LAYERS 5 AUTO (-4.0, 0.0)" in run.stdout + run.stderr, (run.stdout + run.stderr)[-2000:]


def test_background_from_words_with_the_fake_comfyui(comfy, monkeypatch):
    fake, client = comfy
    monkeypatch.setattr(B, "depth_available", lambda: False)         # never download the model
    st = assets.Style(style_prompt="moody watercolour pixels")
    bg = B.background_from_prompt("misty bamboo valley", "dawn", width=160, height=90, client=client, seed=4,
                                  style=st, style_lora="")
    text = json.dumps(fake.prompts[0])
    assert "misty bamboo valley" in text and "dawn" in text and "moody watercolour pixels" in text
    assert len(bg.layers) == 3 and bg.layers[0].rgba.shape == (90, 160, 4) and bg.meta["depth"] == "guess"
    assert (bg.layers[0].rgba[..., 3] == 255).all()


# --------------------------------------------------------------------------- HD-2D test bed

def _hd2d(tmp_path, weather, light="night"):
    from pixel_pipeline.diorama import Diorama, Unit
    from pixel_pipeline.hd2d import make_hd2d
    from test_glow import fake_sheet
    d = Diorama(6, 5, np.ones((5, 6), int), np.full((5, 6), "cobble", object), [("torch", 1, 1)],
                [Unit("knight", 2, 2, 0)], name="wet")
    d.terrain[4, :] = "water"
    out = tmp_path / "p"
    files = make_hd2d(out, {"knight": fake_sheet(tmp_path / "k.png")}, d, iso=False, lighting=light,
                      weather=weather, water="pixel", tiles=False)
    return out, files


def test_hd2d_weather_files_and_light(tmp_path):
    from pixel_pipeline.hd2d import LIGHTING
    out, _ = _hd2d(tmp_path, "storm")
    info = json.loads((out / "weather" / "weather.json").read_text())
    assert info["kinds"] == ["rain", "lightning"] and info["wet"] > 0
    for k in info["kinds"]:
        assert (out / "weather" / k / "sheet_particles3d.tscn").exists()
    light = json.loads((out / "lighting.json").read_text())
    assert light["weather"] == ["rain", "lightning"]
    for name, L in light["presets"].items():
        assert L["sun_energy"] < LIGHTING[name]["sun_energy"] and L["fog_density"] >= 0.012
    assert "WATER_SHADER_PIXEL" not in (out / "shaders" / "water.gdshader").read_text()
    assert "hint_screen_texture" in (out / "shaders" / "water.gdshader").read_text()
    assert "_build_weather()" in (out / "board.gd").read_text()
    (tmp_path / "x").mkdir()
    with pytest.raises(ValueError, match="Unknown weather"):
        _hd2d(tmp_path / "x", "hail")


@pytest.mark.skipif(not GODOT, reason="set GODOT_BIN to run the weather on the board in Godot")
def test_hd2d_weather_runs_in_godot(tmp_path):
    out, _ = _hd2d(tmp_path, ["rain", "fog", "cloud_shadows"])
    subprocess.run([GODOT, "--headless", "--path", str(out), "--import"], capture_output=True, timeout=300)
    (out / "w.gd").write_text(
        "extends SceneTree\nvar f := 0\nvar b\nfunc _initialize() -> void:\n\tchange_scene_to_file(\"res://real3d.tscn\")\n"
        "func _process(_d: float) -> bool:\n\tf += 1\n\tif f == 10:\n\t\tb = root.get_child(root.get_child_count() - 1)\n"
        "\t\tprint(\"WEATHER \", b.weather.scenes.size(), \" WET \", b.weather.wet_done, \" FOG \", snappedf(b.env.fog_density, 0.001))\n"
        "\t\tb.apply_lighting(\"noon\")\n\t\treturn true\n\treturn false\n")
    run = subprocess.run([GODOT, "--headless", "--path", str(out), "-s", "w.gd"], capture_output=True, text=True,
                         timeout=200)
    log = run.stdout + run.stderr
    assert "SCRIPT ERROR" not in log and "Parse Error" not in log, log[-3000:]
    assert "WEATHER 3 WET true FOG 0.032" in log, log[-3000:]
    for scene in ("res://fake3d.tscn", "res://real3d.tscn"):
        run = subprocess.run([GODOT, "--headless", "--path", str(out), scene, "--quit-after", "30"],
                             capture_output=True, text=True, timeout=200)
        assert "SCRIPT ERROR" not in run.stdout + run.stderr, (run.stdout + run.stderr)[-2000:]


# --------------------------------------------------------------------------- the page

def test_weather_page(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from conftest import qt_app
    qt_app()
    from pixel_pipeline.ui.weather_tab import WeatherTab
    assets.set_current_project(assets.create_project("Sky", type="side-scroller").name)
    tab = WeatherTab()
    tab.sync = True
    got = []
    tab.changed.connect(lambda: got.append(1))
    tab.select("weather", "snow")
    a, b = tab.frame_rgba(0), tab.frame_rgba(5)
    assert a.shape == (135, 240, 4) and not np.array_equal(a, b)
    tab.tod.setCurrentIndex(tab.tod.findData("noon"))
    assert tab.current.tod == "noon"
    tab.make()
    assert got and tab.last.kind == "weather" and tab.last.extra["weather"] == "snow"
    files = tab.export_last(tmp_path / "out", "godot-2d")
    assert any(p.name == "sheet_overlay.tscn" for p in files)
    tab.select("water", "lagoon")
    assert tab.frame_rgba(1).shape[1] == 10 * tab.current.tile
    tab.make()
    assert tab.last.kind == "water"
    tab.select("background", "desert")
    x0, x1 = tab.frame_rgba(0), tab.frame_rgba(20)
    assert x0.shape == (270, 480, 4) and not np.array_equal(x0, x1)    # it scrolls
    tab.make()
    assert tab.last.kind == "background"
    assert tab.saved.count() == 3
    tab.timer.stop()

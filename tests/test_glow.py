"""Lighting + glow: emissive masks (glow.py), their trip through exports, and the HD-2D board's
time-of-day presets, light props, spell lights and effects (loaded in Godot when GODOT_BIN is set)."""
import json
import os
import re
import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image

from pixel_pipeline import exports, glow
from pixel_pipeline.diorama import Diorama, Unit
from pixel_pipeline.hd2d import LIGHTING, add_effect, add_unit, make_hd2d

from test_exports import make_character
from test_hd2d import fake_sheet

GODOT = os.environ.get("GODOT_BIN") or shutil.which("godot")


def lantern_figure():
    """A dark figure holding an orange flame (white core) with one red eye."""
    a = np.zeros((32, 24, 4), np.uint8)
    a[6:30, 6:18] = (60, 50, 70, 255)
    a[2:7, 15:19] = (255, 150, 30, 255)
    a[3:5, 16:18] = (255, 240, 190, 255)
    a[9, 9] = (255, 40, 40, 255)
    return a


def test_auto_glow_finds_lights_not_cloth():
    g = glow.auto_glow(lantern_figure())
    assert g.shape == (32, 24, 4)
    assert (g[2:7, 15:19, 3] > 0).all()                 # the flame and its core
    assert g[3, 16, 3] == 255 and g[9, 9, 3] > 0         # core full strength, the eye glows
    assert not g[20, 8, 3]                               # the dark body does not
    cape = np.zeros((32, 24, 4), np.uint8)
    cape[6:30, 6:18] = (70, 60, 60, 255)
    cape[10:28, 6:18] = (240, 210, 40, 255)              # a big bright yellow cape
    assert not glow.auto_glow(cape)[..., 3].any()
    assert not glow.auto_glow(np.zeros((8, 8, 4), np.uint8))[..., 3].any()
    fx = glow.auto_glow(cape, effect=True)                  # effects: everything glows
    assert (fx[..., 3] > 0).sum() == (cape[..., 3] > 0).sum()


def test_mask_strength_reads_any_convention():
    white = np.zeros((1, 3, 4), np.uint8)
    white[0, 0] = (255, 255, 255, 255)
    white[0, 1] = (255, 255, 255, 128)
    white[0, 2] = (255, 120, 0, 255)                     # a coloured mask: brightest channel
    assert np.allclose(glow.mask_strength(white), [1.0, 128 / 255, 1.0], atol=1e-3)
    grey = np.array([[0, 255]], np.uint8)
    assert np.allclose(glow.mask_strength(grey), [0, 1])


def test_write_glow_sheet_per_frame(tmp_path):
    sheet = np.concatenate([lantern_figure(), np.zeros((32, 24, 4), np.uint8)], 0)
    png = tmp_path / "torchbearer.png"
    Image.fromarray(sheet).save(png)
    png.with_suffix(".json").write_text(json.dumps({"frames": [
        {"filename": "idle_S 0", "frame": {"x": 0, "y": 0, "w": 24, "h": 32}},
        {"filename": "idle_S 1", "frame": {"x": 0, "y": 32, "w": 24, "h": 32}}]}))
    out = glow.write_glow_sheet(png)
    assert out == tmp_path / "torchbearer_glow.png"
    g = np.asarray(Image.open(out))
    assert g.shape == (64, 24, 4) and g[:32, ..., 3].any() and not g[32:, ..., 3].any()
    Image.fromarray(np.zeros((64, 24, 4), np.uint8)).save(out)  # an existing (hand-made) map is kept
    assert glow.write_glow_sheet(png) == out and not np.asarray(Image.open(out))[..., 3].any()
    c = glow.write_glow_sheet(png, force=True)
    r, gg, b = glow.glow_colour(png, c)
    assert r == 1.0 and r > gg > b                       # an orange light
    dark = tmp_path / "dark.png"
    Image.fromarray(np.full((8, 8, 4), (40, 40, 50, 255), np.uint8)).save(dark)
    assert glow.write_glow_sheet(dark) is None and not (tmp_path / "dark_glow.png").exists()


def _glowing_knight(tmp_path):
    ch = make_character(tmp_path)
    sheet = np.asarray(Image.open(ch.sheet)).copy()
    sheet[0:2, 0:2] = (120, 220, 255, 255)               # a gem on every frame row's corner
    Image.fromarray(np.where(sheet[..., 3:4] > 0, 255, 0).astype(np.uint8).repeat(4, -1)).save(
        ch.folder / "sheet_glow.png")
    return ch


def test_exports_carry_the_glow(tmp_path):
    ch = _glowing_knight(tmp_path)
    src = exports.source(ch)
    assert src.glow == ch.folder / "sheet_glow.png"
    files = {p.name for p in exports.export_asset(ch, "godot-2d", tmp_path / "g2")}
    assert "knight_glow.png" in files
    files = exports.export_asset(ch, "godot-hd2d", tmp_path / "game")
    unit = tmp_path / "game" / "units" / "knight"
    assert {"knight_glow.png", "knight_material.tres"} <= {p.name for p in files}
    tres = (unit / "knight_material.tres").read_text()
    assert 'path="res://units/knight/knight_glow.png"' in tres and "has_glow = true" in tres
    assert "has_normals = true" in tres and (tmp_path / "game" / "shaders" / "sprite.gdshader").exists()
    out = tmp_path / "unity"
    exports.export_asset(ch, "unity", out, ppu=16)
    meta = (out / "knight.png.meta").read_text()
    gmeta = (out / "knight_glow.png.meta").read_text()
    gguid = re.search(r"^guid: (\w+)", gmeta, re.M).group(1)
    assert f"guid: {gguid}, type: 3}}\n      name: _EmissionMap" in meta
    assert "name: _NormalMap" in meta and "_EmissionMap" in (out / "knight_README.txt").read_text()


def test_unity_export_without_glow_has_no_emission(tmp_path):
    ch = make_character(tmp_path)                        # grey and red: nothing glows
    out = tmp_path / "unity"
    exports.export_asset(ch, "unity", out)
    assert not (out / "knight_glow.png").exists()
    assert "_EmissionMap" not in (out / "knight.png.meta").read_text()


def test_project_lighting_effects_and_glow(tmp_path):
    torch = tmp_path / "torchbearer.png"
    fake_sheet(torch)
    sheet = np.asarray(Image.open(torch)).copy()
    for y in range(0, sheet.shape[0], 24):               # a flame above every frame's hand
        sheet[y + 1:y + 4, 10:13] = (255, 150, 30, 255)
        sheet[y + 2, 11] = (255, 240, 200, 255)
    Image.fromarray(sheet).save(torch)
    fx = tmp_path / "fx" / "sheet.png"
    fx.parent.mkdir()
    fake_sheet(fx, anims=("magic",), dirs=("S",))
    d = Diorama(6, 5, np.ones((5, 6), int), np.full((5, 6), "cobble", object),
                [("brazier", 1, 1), ("lantern", 4, 1), ("crystal", 1, 3), ("torch", 4, 3)],
                [Unit("knight", 2, 2, 0), Unit("dog", 3, 2, 1)], name="lit")
    out = tmp_path / "p"
    make_hd2d(out, {"knight": torch, "dog": fake_sheet(tmp_path / "dog.png")}, d, iso=False,
              lighting="night", effects={"spark": fx}, place_effects=[("spark", 2, 4)])
    light = json.loads((out / "lighting.json").read_text())
    assert light["default"] == "night" and set(light["presets"]) == set(LIGHTING) == {"dawn", "noon", "sunset", "night"}
    assert light["presets"]["night"]["torches"] > light["presets"]["noon"]["torches"]
    assert (out / "units/knight/knight_glow.png").exists() and not (out / "units/dog/dog_glow.png").exists()
    info = json.loads((out / "effects/spark/spark.effect.json").read_text())
    assert len(info["frames"]) == 2 and (out / "effects/spark/spark_glow.png").exists()
    assert json.loads((out / "map/map.json").read_text())["effects"] == [{"name": "spark", "col": 2, "row": 4}]
    for rel in ("light_flicker.gd", "shaders/ember.gdshader", "shaders/blob.gdshader"):
        assert (out / rel).exists(), rel
    with pytest.raises(ValueError, match="Unknown lighting"):
        make_hd2d(tmp_path / "q", {}, Diorama(2, 2, np.ones((2, 2), int), np.full((2, 2), "grass", object), [], []),
                  iso=False, lighting="midnight")
    folder = add_unit(tmp_path / "p2", "knight", torch, glow=False)
    assert not (folder / "knight_glow.png").exists()
    assert add_effect(tmp_path / "p2", "spark", fx).name == "spark"


LIGHT_GD = """extends SceneTree
var f := 0
var b
func _initialize() -> void:
	change_scene_to_file("res://real3d.tscn")
func _process(_d: float) -> bool:
	f += 1
	if f == 10:
		b = root.get_child(root.get_child_count() - 1)
		print("LIGHT ", b.light_name, " FIRES ", b.fire_lights.size(), " FLAMES ", b.flames.size(), " FX ", b.live_effects.size())
		b.apply_lighting("noon")
		print("NOON ", snappedf(b.fire_lights[0].level, 0.01), " SUN ", snappedf(b.sun.light_energy, 0.01))
		b.spell_light(Vector3(2, 1, 2), Color(0.7, 0.4, 1.0))
		b.play_effect("spark", Vector3(3, 0.5, 2))
		print("BLOCKED ", b.blocked.has(Vector2i(1, 1)), " ", b.blocked.has(Vector2i(4, 1)))
	if f == 120:
		print("AFTER FX ", b.live_effects.size())
		return true
	return false
"""


@pytest.mark.skipif(not GODOT, reason="set GODOT_BIN to light the board in Godot")
def test_lights_in_godot(tmp_path):
    fx = tmp_path / "fx" / "sheet.png"
    fx.parent.mkdir()
    fake_sheet(fx, anims=("magic",), dirs=("S",))
    d = Diorama(6, 5, np.ones((5, 6), int), np.full((5, 6), "cobble", object),
                [("brazier", 1, 1), ("lantern", 4, 1), ("crystal", 1, 3), ("torch", 4, 3)],
                [Unit("knight", 2, 2, 0), Unit("dog", 3, 2, 1)], name="lit")
    out = tmp_path / "p"
    make_hd2d(out, {"knight": fake_sheet(tmp_path / "k.png"), "dog": fake_sheet(tmp_path / "dog.png")}, d,
              iso=False, lighting="night", effects={"spark": fx}, place_effects=[("spark", 2, 4)])
    (out / "light.gd").write_text(LIGHT_GD)
    run = subprocess.run([GODOT, "--headless", "--fixed-fps", "60", "--path", str(out), "-s", "light.gd"],
                         capture_output=True, text=True, timeout=180)
    log = run.stdout + run.stderr
    assert "ERROR" not in log, log[-2000:]
    assert "LIGHT night FIRES 4 FLAMES 2 FX 1" in log, log[-2000:]
    assert f"NOON {LIGHTING['noon']['torches']} SUN {LIGHTING['noon']['sun_energy']}" in log, log[-2000:]
    assert "BLOCKED true false" in log and "AFTER FX 1" in log      # the one-shot effect ended
    for scene, light in (("res://fake3d.tscn", "dawn"), ("res://real3d.tscn", "sunset")):
        run = subprocess.run([GODOT, "--headless", "--path", str(out), scene, "--quit-after", "30", "--",
                              f"--light={light}", "--bands=0"], capture_output=True, text=True, timeout=180)
        assert "ERROR" not in run.stdout + run.stderr, (run.stdout + run.stderr)[-2000:]


@pytest.mark.skipif(not GODOT, reason="set GODOT_BIN to load the exported material in Godot")
def test_exported_material_loads_in_godot(tmp_path):
    ch = _glowing_knight(tmp_path)
    game = tmp_path / "game"
    (game / "project.godot").parent.mkdir(parents=True, exist_ok=True)
    (game / "project.godot").write_text('config_version=5\n\n[application]\n\nconfig/name="m"\n')
    exports.export_asset(ch, "godot-hd2d", game)
    subprocess.run([GODOT, "--headless", "--path", str(game), "--import"], capture_output=True, timeout=300)
    (game / "check.gd").write_text('extends SceneTree\nfunc _initialize() -> void:\n'
                                   '\tvar m = load("res://units/knight/knight_material.tres")\n'
                                   '\tprint("GLOW ", m.get_shader_parameter("has_glow"), " ", m.get_shader_parameter("glow") != null)\n'
                                   '\tquit()\n')
    run = subprocess.run([GODOT, "--headless", "--path", str(game), "-s", "check.gd"],
                         capture_output=True, text=True, timeout=180)
    log = run.stdout + run.stderr
    assert "GLOW true true" in log, log[-2000:]

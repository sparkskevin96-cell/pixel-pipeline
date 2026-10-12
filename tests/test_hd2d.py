import json
import os
import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image

from pixel_pipeline.diorama import Diorama, harbour, path_cost, textures
from pixel_pipeline.hd2d import add_unit, make_hd2d

GODOT = os.environ.get("GODOT_BIN") or shutil.which("godot")


def fake_sheet(path, anims=("idle", "walk"), dirs=("E", "S", "W", "N"), w=16, h=24):
    """A tiny sheet + json like turnaround --animate writes: a block figure, feet at row 20."""
    frames, n = [], len(anims) * len(dirs) * 2
    sheet = np.zeros((h * n, w, 4), np.uint8)
    i = 0
    for a in anims:
        for d in dirs:
            for k in range(2):
                sheet[i * h + 4:i * h + 20, 4:12] = (200, 60 + 40 * k, 60, 255)
                frames.append({"filename": f"{a}_{d} {k}", "duration": 150,
                               "frame": {"x": 0, "y": i * h, "w": w, "h": h}})
                i += 1
    Image.fromarray(sheet).save(path)
    meta = {"frames": frames, "meta": {"image": path.name, "size": {"w": w, "h": h * n}}}
    path.with_suffix(".json").write_text(json.dumps(meta))
    return path


def test_harbour_map_and_movement(tmp_path):
    d = harbour()
    assert d.height.shape == (d.rows, d.cols) and {"planks", "cobble", "stone", "water"} <= set(d.terrain.ravel())
    for u in d.units:
        assert d.walkable(u.col, u.row)
    knight = d.units[0]
    reach = path_cost(d, (knight.col, knight.row), move=4, jump=2)
    assert all(d.terrain[r, c] != "water" for c, r in reach)
    assert max(reach.values()) <= 4 and len(reach) > 5
    back = Diorama.load(d.save(tmp_path / "map.json"))
    assert (back.height == d.height).all() and back.units == d.units


def test_textures_are_seamless_tiles():
    for name, (top, side) in textures(harbour()).items():
        assert top.shape == (32, 32, 4) and side.shape == (32, 32, 4), name


def test_project_files(tmp_path):
    dog = fake_sheet(tmp_path / "dog.png")
    knight = fake_sheet(tmp_path / "knight_anim.png")
    out = tmp_path / "proj"
    make_hd2d(out, {"dog": dog, "knight": knight}, iso=False)
    for rel in ("project.godot", "menu.tscn", "real3d.tscn", "fake3d.tscn", "board.gd",
                "shaders/sprite.gdshader", "map/map.json", "map/tex/planks_top.png",
                "units/dog/dog.png", "units/dog/dog.unit.json", "units/knight/knight.json"):
        assert (out / rel).exists(), rel
    info = json.loads((out / "units/dog/dog.unit.json").read_text())
    assert info["foot"] == 20 and info["anims"] == ["idle", "walk"]
    with pytest.raises(ValueError, match="no sprite sheet"):
        make_hd2d(tmp_path / "p2", {"dog": dog}, iso=False)       # the map also has a knight


def test_normal_map_is_copied(tmp_path):
    dog = fake_sheet(tmp_path / "dog.png")
    Image.new("RGBA", (16, 24 * 16), (128, 128, 255, 255)).save(tmp_path / "dog_n.png")
    folder = add_unit(tmp_path / "proj", "dog", dog)
    assert (folder / "dog_n.png").exists()


@pytest.mark.skipif(not GODOT, reason="set GODOT_BIN to load the scenes in Godot")
def test_scenes_load_in_godot(tmp_path):
    out = tmp_path / "proj"
    make_hd2d(out, {"dog": fake_sheet(tmp_path / "dog.png"),
                    "knight": fake_sheet(tmp_path / "knight.png")}, iso=False)
    for scene in ("res://real3d.tscn", "res://fake3d.tscn", "res://menu.tscn"):
        run = subprocess.run([GODOT, "--headless", "--path", str(out), scene, "--quit-after", "30"],
                             capture_output=True, text=True, timeout=180)
        log = run.stdout + run.stderr
        assert "SCRIPT ERROR" not in log and "Parse Error" not in log, log[-2000:]
        assert "ERROR" not in log, log[-2000:]


def test_cli_hd2d(tmp_path, capsys):
    from pixel_pipeline.cli import main
    dog = fake_sheet(tmp_path / "dog.png")
    knight = fake_sheet(tmp_path / "knight.png")
    assert main(["hd2d", "-o", str(tmp_path / "p"), "--unit", f"dog={dog}", "--unit", f"knight={knight}"]) == 0
    assert (tmp_path / "p" / "real3d.tscn").exists() and (tmp_path / "p" / "iso" / "iso.tscn").exists()
    assert "HD-2D project" in capsys.readouterr().out


FIGHT_GD = """extends SceneTree
var f := 0
var b
func _initialize() -> void:
	change_scene_to_file("res://real3d.tscn")
func _process(_d: float) -> bool:
	f += 1
	if f == 10:
		b = root.get_child(root.get_child_count() - 1)
		var k = b.units[0]
		b._click(Vector2i(k["col"], k["row"]))
		print("PHASE ", b.phase)
		b._click(Vector2i(3, 2))
	if f == 80:
		print("PHASE ", b.phase, " TARGETS ", b.attack_cells.size())
		var dog = b.units[1]
		print("DAMAGE ", b._damage(b.units[0], dog))
		b._click(Vector2i(dog["col"], dog["row"]))
	if f == 160 and b.units.size() > 1:
		print("HP ", b.units[1]["hp"], " PHASE ", b.phase, " TEAM ", b.active_team)
	if f == 400:
		print("END ", b.units.size(), " ", b.banner.text)
		return true
	return false
"""


@pytest.mark.skipif(not GODOT, reason="set GODOT_BIN to play a fight in Godot")
@pytest.mark.parametrize("dog_hp", [30, 10])
def test_fight_damage_turns_and_victory(tmp_path, dog_hp):
    from pixel_pipeline.diorama import Unit, damage
    h = np.ones((5, 6), int)
    t = np.full((5, 6), "cobble", object)
    knight = Unit("knight", 2, 2, 0, "E", attack=20)
    dog = Unit("dog", 4, 2, 1, "W", hp=dog_hp)
    d = Diorama(6, 5, h, t, [], [knight, dog], name="arena")
    out = tmp_path / "p"
    make_hd2d(out, {"dog": fake_sheet(tmp_path / "dog.png", anims=("idle", "walk", "death")),
                    "knight": fake_sheet(tmp_path / "knight.png", anims=("idle", "walk", "attack"))}, d, iso=False)
    (out / "fight.gd").write_text(FIGHT_GD)
    run = subprocess.run([GODOT, "--headless", "--fixed-fps", "60", "--path", str(out), "-s", "fight.gd"],
                         capture_output=True, text=True, timeout=180)
    log = run.stdout + run.stderr
    assert "SCRIPT ERROR" not in log, log[-2000:]
    hit = damage(knight, dog)
    assert "PHASE move" in log and "PHASE act TARGETS 1" in log and f"DAMAGE {hit}" in log
    if dog_hp > hit:
        assert f"HP {dog_hp - hit} PHASE select TEAM 1" in log   # knight done -> red's turn
        assert "END 2 " in log
    else:
        assert "END 1 Blue wins!" in log                          # death, then victory

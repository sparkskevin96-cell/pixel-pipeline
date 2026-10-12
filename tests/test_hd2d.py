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

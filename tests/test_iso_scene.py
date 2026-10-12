"""The isometric 2D scene: files, map.json round trip, block geometry, and (with GODOT_BIN) the
scene running in Godot: range = diorama.path_cost, a move walks there and passes the turn."""
import json
import os
import shutil
import subprocess

import numpy as np
from PIL import Image
import pytest

from pixel_pipeline.cli import main
from pixel_pipeline.diorama import Diorama, harbour, path_cost
from pixel_pipeline.godot import write_sprite_frames
from pixel_pipeline.iso_scene import DIRECTIONS, iso_column, unit_metrics, write_iso_scene

GODOT = os.environ.get("GODOT_BIN") or shutil.which("godot")


def fake_sheet(folder, name="knight", tres=True):
    """16 x 24 frames, idle (1) + walk (2) in 16 directions, feet on row 21."""
    folder.mkdir(parents=True, exist_ok=True)
    frame = np.zeros((24, 16, 4), np.uint8)
    frame[4:22, 5:11] = (200, 60, 60, 255)
    names = [f"{a}_{d} {i}" for a, n in (("idle", 1), ("walk", 2)) for d in DIRECTIONS for i in range(n)]
    Image.fromarray(np.concatenate([frame] * len(names), 1)).save(folder / f"{name}.png")
    rects = [(i * 16, 0, 16, 24) for i in range(len(names))]
    (folder / f"{name}.json").write_text(json.dumps({"frames": [
        {"filename": nm, "frame": {"x": x, "y": y, "w": w, "h": h}, "duration": 100}
        for nm, (x, y, w, h) in zip(names, rects)]}))
    if tres:
        anims = {}
        for i, nm in enumerate(names):
            anims.setdefault(nm.split(" ")[0], {"frames": [], "duration_ms": 100})["frames"].append(i)
        write_sprite_frames(folder / f"{name}.tres", folder / f"{name}.png", rects, anims)
    return folder / f"{name}.png"


def test_column_has_no_gaps_and_grows_with_height():
    top = np.full((32, 32, 4), 255, np.uint8)
    for levels in (0, 1, 5):
        col = iso_column(top, top, 32, levels, 8)
        assert col.shape == (16 + levels * 8, 32, 4)
        a = col[..., 3] > 0
        # solid hexagon: every column of pixels is one unbroken run
        for x in range(32):
            ys = np.nonzero(a[:, x])[0]
            if levels == 0 and x in (0, 31):
                continue                        # the flat diamond's tips (neighbours fill them)
            assert len(ys) and ys.max() - ys.min() + 1 == len(ys)


def test_unit_metrics_find_the_feet(tmp_path):
    png = fake_sheet(tmp_path, tres=False)
    m = unit_metrics(png, png.with_suffix(".json"))
    assert m["foot"] == [8, 21] and m["body"] == [-3, -17, 3, 1] and m["size"] == [16, 24]


def test_write_iso_scene_files_and_map_round_trip(tmp_path):
    sheet = fake_sheet(tmp_path / "sheets")
    d = harbour()
    out = tmp_path / "game"
    files = write_iso_scene(d, out, {"knight": sheet})
    for rel in ("iso/iso.tscn", "iso/iso.gd", "iso/iso.json", "iso/blocks.png", "iso/water.png",
                "iso/torch.png", "iso/units/knight/knight.png", "iso/units/knight/knight.tres",
                "map/map.json", "project.godot"):
        assert out / rel in files and (out / rel).exists(), rel
    back = Diorama.load(out / "map" / "map.json")
    assert back.to_dict() == d.to_dict()
    conf = json.loads((out / "iso" / "iso.json").read_text())
    assert conf["units"]["knight"]["tres"] == "res://iso/units/knight/knight.tres"
    assert "_pawn" in conf["units"] and conf["level_px"] == 8
    blocks = np.array(Image.open(out / "iso" / "blocks.png"))
    x, y, w, h = conf["blocks_rects"]["stone"]["6"]
    assert h == 16 + (6 + 2) * 8 and blocks[y:y + h, x:x + w, 3].any()
    assert 'run/main_scene="res://iso/iso.tscn"' in (out / "project.godot").read_text()


def test_existing_project_and_map_are_left_alone(tmp_path):
    out = tmp_path / "game"
    out.mkdir()
    (out / "project.godot").write_text("menu project")
    small = Diorama(3, 2, np.ones((2, 3), int), np.full((2, 3), "grass", object), name="tiny")
    small.save(out / "map" / "map.json")
    files = write_iso_scene(harbour(), out)
    assert (out / "project.godot").read_text() == "menu project"
    assert out / "map" / "map.json" not in files
    assert Diorama.load(out / "map" / "map.json").name == "tiny"


def test_cli_iso_demo(tmp_path, capsys):
    sheet = fake_sheet(tmp_path / "s", "dog", tres=False)
    assert main(["iso-demo", "-o", str(tmp_path / "g"), "--unit", f"dog={sheet}"]) == 0
    assert "iso.tscn" in capsys.readouterr().out
    assert (tmp_path / "g" / "iso" / "units" / "dog" / "dog.json").exists()


CHECK = """extends SceneTree

var iso
var n := 0
var mover
var start_team := 0

func _initialize() -> void:
	iso = load("res://iso/iso.tscn").instantiate()
	root.add_child(iso)

func _process(_d: float) -> bool:
	n += 1
	if n == 3:
		var out := {}
		for u in iso.units:
			var r: Dictionary = iso.reachable(u["cell"], true)
			var cells := []
			for c in r:
				cells.append([c.x, c.y, r[c]])
			out["%d,%d" % [u["cell"].x, u["cell"].y]] = cells
		print("RANGES " + JSON.stringify(out))
		print("UNITS %d %s" % [iso.units.size(), iso.units[0]["sprite"].animation])
		start_team = iso.active_team
		mover = iso.units[0]
		var body: Rect2 = mover["body"]
		iso.click(mover["node"].position + body.get_center())       # click the sprite
		print("SELECTED %s" % str(iso.selected == mover))
		var far: Vector2i = mover["cell"]
		for c in iso.reach:
			if iso.reach[c] > iso.reach.get(far, 0):
				far = c
		print("TARGET %d %d" % [far.x, far.y])
		iso.click(iso.top(far) + Vector2(0, 2))                    # click the cell's top
		print("WALK %s" % mover["sprite"].animation)
	elif n > 3 and not iso.busy:
		print("ARRIVED %d %d %s %s" % [mover["cell"].x, mover["cell"].y, mover["sprite"].animation,
			str(iso.active_team != start_team)])
		return true
	return n > 3000
"""


@pytest.mark.skipif(not GODOT, reason="set GODOT_BIN to run the iso scene in Godot")
def test_iso_scene_runs_in_godot(tmp_path):
    out = tmp_path / "game"
    d = harbour()
    write_iso_scene(d, out, {"knight": fake_sheet(tmp_path / "s"),
                             "dog": fake_sheet(tmp_path / "s2", "dog", tres=False)})
    (out / "check.gd").write_text(CHECK)
    run = [GODOT, "--headless", "--path", str(out)]
    subprocess.run(run + ["--import"], capture_output=True, timeout=300)
    res = subprocess.run(run + ["-s", "check.gd"], capture_output=True, text=True, timeout=300)
    log = res.stdout + res.stderr
    assert "SCRIPT ERROR" not in log and "Parse Error" not in log, log
    lines = {x.split(" ", 1)[0]: x.split(" ", 1)[1] for x in res.stdout.splitlines() if " " in x}
    ranges = json.loads(lines["RANGES"])
    for u in d.units:
        want = path_cost(d, (u.col, u.row))
        got = {(c, r): s for c, r, s in ranges[f"{u.col},{u.row}"]}
        assert got == want, u
    count, anim = lines["UNITS"].split()
    assert count == "4" and anim.startswith("idle_")
    assert lines["SELECTED"] == "true" and lines["WALK"].startswith("walk_")
    tx, ty = lines["TARGET"].split()
    ax, ay, anim, passed = lines["ARRIVED"].split()
    assert (ax, ay) == (tx, ty) and anim.startswith("idle_") and passed == "true"

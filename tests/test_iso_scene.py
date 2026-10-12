"""The isometric 2D scene: files, map.json round trip, block geometry, overlays, and (with
GODOT_BIN) the scene running in Godot: range = diorama.path_cost with each unit's own move/jump,
a move walks there and the unit is done; a scripted 1v1 fight checks damage = diorama.damage,
hurt/death fallbacks, done-greying, the automatic turn pass and the victory banner."""
import json
import os
import shutil
import subprocess

import numpy as np
from PIL import Image
import pytest

from pixel_pipeline.cli import main
from pixel_pipeline.diorama import Diorama, Unit, damage, harbour, path_cost
from pixel_pipeline.godot import write_sprite_frames
from pixel_pipeline.iso_scene import (DIRECTIONS, attack_sprite, iso_column, range_sprite,
                                      unit_metrics, write_iso_scene)

GODOT = os.environ.get("GODOT_BIN") or shutil.which("godot")


def fake_sheet(folder, name="knight", tres=True, size=(16, 24), anims=(("idle", 1), ("walk", 2))):
    """Frames `size` (w, h) with a block body standing on row h-3, the given (anim, frame count)
    in 16 directions (default idle 1 + walk 2, feet on row 21 of 16 x 24)."""
    folder.mkdir(parents=True, exist_ok=True)
    fw, fh = size
    frame = np.zeros((fh, fw, 4), np.uint8)
    frame[4:fh - 2, fw // 2 - 3:fw // 2 + 3] = (200, 60, 60, 255)
    names = [f"{a}_{d} {i}" for a, n in anims for d in DIRECTIONS for i in range(n)]
    Image.fromarray(np.concatenate([frame] * len(names), 1)).save(folder / f"{name}.png")
    rects = [(i * fw, 0, fw, fh) for i in range(len(names))]
    (folder / f"{name}.json").write_text(json.dumps({"frames": [
        {"filename": nm, "frame": {"x": x, "y": y, "w": w, "h": h}, "duration": 100}
        for nm, (x, y, w, h) in zip(names, rects)]}))
    if tres:
        anims_ = {}
        for i, nm in enumerate(names):
            anims_.setdefault(nm.split(" ")[0], {"frames": [], "duration_ms": 100})["frames"].append(i)
        write_sprite_frames(folder / f"{name}.tres", folder / f"{name}.png", rects, anims_)
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
    # small animal sheets (40x40 dog, 48x40 wolf, 56x48 bear), some with walk frames only
    for size, anims in (((40, 40), (("idle", 1),)), ((48, 40), (("walk", 2),)), ((56, 48), (("idle", 2),))):
        png = fake_sheet(tmp_path / f"s{size[0]}", "a", False, size, anims)
        m = unit_metrics(png, png.with_suffix(".json"))
        assert m["size"] == list(size) and m["foot"] == [size[0] // 2, size[1] - 3]
        assert m["body"][1] == -(size[1] - 7) and m["body"][3] == 1


def test_overlays_are_saturated_purple_and_red():
    import colorsys
    for spr, hue in ((range_sprite(32), 270), (attack_sprite(32), 0)):
        inside = spr[..., 3] > 0
        assert spr[inside, 3].min() >= 180                      # mostly opaque fill
        h, s_, _ = colorsys.rgb_to_hsv(*(spr[8, 16, :3] / 255))  # the centre: fill
        assert s_ > 0.75 and min(abs(h * 360 - hue), 360 - abs(h * 360 - hue)) < 25
        edge = spr[8, 0:3][inside[8, 0:3]][0, :3].astype(int)       # left tip = outline, lighter
        assert edge.sum() > spr[8, 16, :3].astype(int).sum()


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
		print("ARRIVED %d %d %s %s %s %.2f %s" % [mover["cell"].x, mover["cell"].y, mover["sprite"].animation,
			str(iso.active_team == start_team), str(mover["done"]), mover["sprite"].modulate.r, iso.phase])
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
        want = path_cost(d, (u.col, u.row), u.move, u.jump)
        got = {(c, r): s for c, r, s in ranges[f"{u.col},{u.row}"]}
        assert got == want, u
    count, anim = lines["UNITS"].split()
    assert count == "4" and anim.startswith("idle_")
    assert lines["SELECTED"] == "true" and lines["WALK"].startswith("walk_")
    tx, ty = lines["TARGET"].split()
    ax, ay, anim, same_team, done, grey, phase = lines["ARRIVED"].split()
    # no enemy in reach on arrival: the unit is done (greyed); its team mate still has to act
    assert (ax, ay) == (tx, ty) and anim.startswith("idle_")
    assert done == "true" and grey == "0.55" and same_team == "true" and phase == "select"


FIGHT = """extends SceneTree
# blue knight walks next to the red dog, attacks; red dog stays and bites back; blue hits again
# and the dog dies -> removed, blue wins.
var iso
var n := 0
var stage := 0
var knight
var dog

func _initialize() -> void:
	iso = load("res://iso/iso.tscn").instantiate()
	root.add_child(iso)
	iso.hit.connect(func(a, t, d):
		print("HIT %s %d %d %d %d %d %s" % [a["name"], a["cell"].x, a["cell"].y, t["cell"].x, d, t["hp"], t["sprite"].animation]))
	iso.died.connect(func(u): print("DIED %s %d" % [u["name"], iso.units.size()]))
	iso.turn_started.connect(func(t): print("TURN %d %.2f %.2f" % [t, knight["sprite"].modulate.r, dog["sprite"].modulate.r]))
	iso.game_over.connect(func(w): print("OVER %d" % w))

func body(u) -> Vector2:
	return u["node"].position + u["body"].get_center()

func _process(_d: float) -> bool:
	n += 1
	if n > 4000:
		return true                       # (also ends the run if a step below errors)
	if n < 3 or iso.busy:
		return false
	if stage == 0:
		knight = iso.units[0]
		dog = iso.units[1]
		print("STATS %d %d %d %d %d %d %d" % [knight["hp"], knight["atk"], knight["def"], knight["move"], knight["jump"], knight["reach"], dog["hp"]])
		iso.click(body(knight))
		iso.click(iso.top(Vector2i(2, 1)) + Vector2(0, 2))
		stage = 1
	elif stage == 1 and iso.phase == "act":
		print("RED %s" % str(iso.attack_cells.keys()))
		iso.hover(body(dog))
		print("PREVIEW " + iso.hud.text.split("\\n")[2])
		print("HUD " + iso.hud.text.split("\\n")[1])
		iso.click(body(dog))
		stage = 2
	elif stage == 2 and iso.active_team == 1 and iso.phase == "select":
		iso.click(body(dog))
		iso.click(body(dog))              # again: stay and act
		iso.click(body(knight))
		stage = 3
	elif stage == 3 and iso.active_team == 0 and iso.phase == "select":
		iso.click(body(knight))
		iso.click(body(knight))
		iso.click(body(dog))
		stage = 4
	elif stage == 4 and iso.phase == "over":
		iso.click(body(knight))
		print("END %d %s %s %s" % [iso.units.size(), str(iso.unit_on(Vector2i(3, 1)) == null),
			str(iso.selected == null), iso.banner.get_child(3).text])
		return true
	return n > 4000
"""


@pytest.mark.skipif(not GODOT, reason="set GODOT_BIN to run the iso scene in Godot")
def test_iso_fight_in_godot(tmp_path):
    # 1v1 on a strip: cols 0-2 at level 2, cols 3-4 at level 1 (the knight hits from above)
    h = np.array([[2, 2, 2, 1, 1]] * 3)
    knight = Unit("knight", 0, 1, 0, "E", attack=20)
    dog = Unit("dog", 3, 1, 1, "W", hp=40)
    d = Diorama(5, 3, h, np.full((3, 5), "grass", object), units=[knight, dog], name="duel")
    out = tmp_path / "game"
    # the knight's sheet has attack/hurt/death; the 40x40 dog has only idle + walk (fallbacks)
    write_iso_scene(d, out, {
        "knight": fake_sheet(tmp_path / "k", anims=(("idle", 1), ("walk", 2), ("attack", 3),
                                                     ("hurt", 2), ("death", 2))),
        "dog": fake_sheet(tmp_path / "d", "dog", tres=False, size=(40, 40))})
    (out / "fight.gd").write_text(FIGHT)
    run = [GODOT, "--headless", "--path", str(out)]
    subprocess.run(run + ["--import"], capture_output=True, timeout=300)
    res = subprocess.run(run + ["-s", "fight.gd"], capture_output=True, text=True, timeout=120)
    log = res.stdout + res.stderr
    assert "SCRIPT ERROR" not in log and "Parse Error" not in log, log
    out_lines = [x for x in res.stdout.splitlines() if " " in x]
    first = {x.split(" ", 1)[0]: x.split(" ", 1)[1] for x in reversed(out_lines)}
    assert first["STATS"].split() == ["34", "20", "8", "4", "2", "1", "40"], log
    assert first["RED"] == "[(3, 1)]"
    up = damage(knight, dog, 2, 1)                 # from level 2 down onto level 1: +25%
    back = damage(dog, knight, 1, 2)
    assert up == 23 and back == 5
    assert first["PREVIEW"] == f"Hit Dog for {up}"
    assert first["HUD"] == "Knight  HP 34/34  ATK 20 DEF 8 MOV 4"
    hits = [x.split()[1:] for x in out_lines if x.startswith("HIT ")]
    assert [(h_[0], h_[1], h_[2], h_[4], h_[5]) for h_ in hits] == [
        ("knight", "2", "1", str(up), str(40 - up)),
        ("dog", "3", "1", str(back), str(34 - back)),
        ("knight", "2", "1", str(up), "0")], log
    assert hits[1][6].startswith("hurt_")          # the knight has a hurt clip; the dog not
    assert hits[0][6].startswith("idle_")
    turns = [x.split()[1:] for x in out_lines if x.startswith("TURN ")]
    # red's turn: the knight that acted is greyed; blue's next turn: un-greyed, the dog grey
    assert turns[:2] == [["1", "0.55", "1.00"], ["0", "1.00", "0.55"]], log
    assert first["DIED"] == "dog 1" and first["OVER"] == "0"
    assert first["END"] == "1 true true Blue wins!"

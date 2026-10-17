"""The game runtime (motion.py + godot_runtime/pp_motion.gd + the HD-2D board): planted-feet gait,
stride measured into sheets / unit.json / manifest, per-clip frame counts, and in headless Godot:
animation speed tracking movement, height following over a step, gradual turning, eased
starts/stops, 2D HP bars, the feet-depth sprite shader."""
import json
import os
import re
import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image

from pixel_pipeline import motion
from pixel_pipeline.diorama import Diorama, Unit
from pixel_pipeline.hd2d import make_hd2d

from test_hd2d import fake_sheet

GODOT = os.environ.get("GODOT_BIN") or shutil.which("godot")


@pytest.fixture(scope="module")
def rigged(tmp_path_factory):
    pytest.importorskip("trimesh")
    from test_autorig import humanoid
    from pixel_pipeline.autorig import auto_rig
    d = tmp_path_factory.mktemp("rig")
    return auto_rig(humanoid(d / "h.glb"), d / "h.rigged.glb", actions=["idle", "walk", "run"]).path


def test_planted_feet_gait(rigged):
    from pixel_pipeline.anim3d import load_rig
    rig = load_rig([rigged])
    clips = {c.name: c for c in rig.clips}
    walk, run = motion.rig_stride(rig, clips["walk"]), motion.rig_stride(rig, clips["run"])
    assert 0.3 < walk["heights"] < 1.2 and run["heights"] > walk["heights"]
    # a planted ankle stays on the ground (same height) and moves back at one speed
    foot = rig.names.index("foot_l")
    ts = clips["walk"].start + clips["walk"].duration * np.arange(48) / 48
    P = np.array([rig.worlds(clips["walk"], t)[foot][:3, 3] for t in ts])
    low = P[:, 1] <= P[:, 1].min() + 0.005
    assert low.sum() >= 20                                 # ~58% of the cycle on the ground
    v = np.diff(P[low][:, 2])
    assert np.all(v < 0) and np.ptp(v) < 0.25 * abs(v.mean())   # steady backward slide


def test_per_clip_frames_and_stride_in_the_sheet(rigged, tmp_path):
    from pixel_pipeline import settings_for
    from pixel_pipeline.anim3d import animated_turnaround, write_animated
    s = settings_for("person", width=24, height=32, colors=8)
    anim, what = animated_turnaround([rigged], s, 4, {"*": 2, "walk": 6, "run": 4}, points=40_000, normals=True)
    assert anim.counts() == [2, 6, 4] and "walk 6" in what
    png = tmp_path / "sheet.png"
    write_animated(anim, png, normals=True, godot=True)
    meta = json.loads(png.with_suffix(".json").read_text())
    rows = motion.rows_of(meta)
    assert len(rows["walk_E"]) == 6 and len(rows["idle_S"]) == 2 and len(rows["run_N"]) == 4
    assert Image.open(png).size == Image.open(tmp_path / "sheet_n.png").size
    assert motion.frame_counts_of(png.with_suffix(".json")) == {"idle": 2, "walk": 6, "run": 4}
    info = motion.measure_sheet(png, rigged, None, 30.0)
    assert info["stride"]["walk"]["source"] == "rig" and info["stride"]["walk"]["frames"] == 6
    assert 0.3 < info["stride"]["walk"]["heights"] < 1.6 and info["body_px"] > 10
    motion.annotate_sheet(png, info)
    assert motion.sheet_stride(png.with_suffix(".json"))[0]["run"]["px"] > info["stride"]["walk"]["px"]


def strided_sheet(path, px=21.0):
    """A fake sheet whose json says its walk covers `px` sheet pixels a cycle (as if measured)."""
    fake_sheet(path, anims=("idle", "walk", "attack"), dirs=("E", "SE", "S", "SW", "W", "NW", "N", "NE"))
    js = path.with_suffix(".json")
    meta = json.loads(js.read_text())
    meta["meta"]["stride"] = {"walk": {"px": px, "cycle_ms": 300, "frames": 2, "source": "rig"}}
    meta["meta"]["body_px"] = 16
    js.write_text(json.dumps(meta))
    return path


def test_unit_json_has_stride_in_tiles(tmp_path):
    d = Diorama(4, 3, np.full((3, 4), 2), np.full((3, 4), "grass", object), [], [Unit("knight", 1, 1)],
                sea=False)
    make_hd2d(tmp_path / "p", {"knight": strided_sheet(tmp_path / "k.png")}, d, iso=False, tiles=False)
    info = json.loads((tmp_path / "p/units/knight/knight.unit.json").read_text())
    assert info["stride"]["walk"]["units"] == pytest.approx(21.0 / 42.0, abs=1e-3)
    assert (tmp_path / "p/pp_runtime/pp_motion.gd").exists()
    assert json.loads((tmp_path / "p/pp_runtime/motion.json").read_text())["turn_rate"] > 0
    assert "feet_depth" in (tmp_path / "p/shaders/sprite.gdshader").read_text()
    mj = json.loads((tmp_path / "p/map/map.json").read_text())
    assert "tree" in mj["blocking"] and "bush" in mj["blocking"]


WALK_GD = """extends SceneTree
var f := 0
var b
var u
var rows := []
var done := false
var last_yaw := 0.0
var max_turn := 0.0
var cycles := 0.0
var last_phase := 0.0
func _initialize() -> void:
	change_scene_to_file("res://real3d.tscn")
func _process(d: float) -> bool:
	f += 1
	if f == 10:
		b = root.get_child(root.get_child_count() - 1)
		u = b.units[0]
		last_yaw = u["yaw"]
		b.face_to(u, PI)                                  # turn around: through the facings
	if f == 40:
		b.scripted = true
		b._walk(u, [Vector2i(1, 2), Vector2i(2, 2), Vector2i(3, 2), Vector2i(4, 2), Vector2i(4, 1)])
		b.action_done.connect(func(): done = true)
		last_phase = u["phase"]
	if f > 10 and f < 40:
		max_turn = maxf(max_turn, absf(wrapf(u["yaw"] - last_yaw, -PI, PI)))
		last_yaw = u["yaw"]
	if f > 40 and not done:
		var p: Vector3 = u["node"].position
		var g: float = b.ground_at(p.x, p.z)
		rows.append([p.x, p.y, p.z, g, u["speed"], u["anim"]])
		if u["anim"] == "walk":
			cycles += fposmod(u["phase"] - last_phase, 1.0)
		last_phase = u["phase"]
	if done or f > 900:
		var sink := 0.0
		var dist := 0.0
		for i in range(1, rows.size()):
			sink = maxf(sink, rows[i][3] - rows[i][1])
			dist += Vector2(rows[i][0] - rows[i - 1][0], rows[i][2] - rows[i - 1][2]).length()
		var speeds := rows.map(func(r): return r[4])
		print("WALK " + JSON.stringify({"done": done, "frames": rows.size(), "sink": sink, "dist": dist,
			"cycles": cycles, "first_speed": speeds[0] if speeds.size() else -1.0,
			"peak": speeds.max() if speeds.size() else -1.0, "last_speed": speeds[speeds.size() - 1] if speeds.size() else -1.0,
			"max_turn_step": max_turn, "top": b.top_speed(u), "y_end": u["node"].position.y, "anim": u["anim"],
			"bar_2d": u["bar"] is TextureRect, "cell": [u["col"], u["row"]]}))
		return true
	return false
"""


@pytest.mark.skipif(not GODOT, reason="set GODOT_BIN to run the runtime in Godot")
def test_board_walk_tracks_stride_height_and_turning(tmp_path):
    h = np.full((4, 6), 2, int)
    h[:, 3:] = 3                                          # a step up between col 2 and col 3
    h[1, 4] = 4                                           # and another onto the last cell
    t = np.full((4, 6), "grass", object)
    d = Diorama(6, 4, h, t, [], [Unit("knight", 1, 2, 0, "E")], sea=False)
    out = tmp_path / "p"
    make_hd2d(out, {"knight": strided_sheet(tmp_path / "k.png")}, d, iso=False, tiles=False)
    (out / "walk.gd").write_text(WALK_GD)
    run = subprocess.run([GODOT, "--headless", "--fixed-fps", "60", "--path", str(out), "-s", "walk.gd"],
                         capture_output=True, text=True, timeout=240)
    log = run.stdout + run.stderr
    assert "SCRIPT ERROR" not in log and "Parse Error" not in log, log[-3000:]
    r = json.loads(re.search(r"WALK (\{.*\})", log).group(1))
    assert r["done"] and r["cell"] == [4, 1] and r["bar_2d"]
    stride = 21.0 / 42.0                                  # tiles per walk cycle
    assert r["peak"] <= r["top"] + 1e-3 and r["top"] == pytest.approx(stride * 1.9, rel=0.01)
    # feet planted: cycles played = distance / stride (within the settle at the end)
    assert r["cycles"] == pytest.approx(r["dist"] / stride, rel=0.08)
    assert r["first_speed"] < 0.5 * r["peak"] and r["last_speed"] < 0.9 * r["peak"]   # eased in and out
    assert r["sink"] < 0.02                               # never under the ground over the steps
    assert r["y_end"] == pytest.approx(4 * 0.5, abs=0.01)
    assert r["max_turn_step"] < 0.3                       # a turn shows the facings in between

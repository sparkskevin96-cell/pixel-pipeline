"""The Godot demo project: built-in rigged knight, map, team materials, click-to-move."""
import json
import os
import shutil
import subprocess

import numpy as np
import pytest

from pixel_pipeline import settings_for
from pixel_pipeline.anim3d import load_rig
from pixel_pipeline.cli import main
from pixel_pipeline.demo import BLOCKING, make_demo, mannequin, parse_terrains

GODOT = os.environ.get("GODOT_BIN") or shutil.which("godot")


def small():
    return settings_for("person", width=24, height=24, colors=12)


def test_mannequin_is_rigged_with_idle_and_walk_loops(tmp_path):
    rig = load_rig([mannequin(tmp_path / "k.glb")])
    assert [(c.name, c.loop) for c in rig.clips] == [("idle", True), ("walk", True)]
    rest = np.concatenate(rig.vertices(rig.worlds()))
    assert 1.8 < np.ptp(rest[:, 1]) < 1.9 and rest[:, 1].min() == pytest.approx(0, abs=1e-6)
    walk = rig.clips[1]
    feet = [np.concatenate(rig.vertices(rig.worlds(walk, t)))[16:24, 2].mean() for t in (0, 0.4)]
    assert abs(feet[0] - feet[1]) > 0.2                      # the left boot swings


def test_make_demo_writes_a_project(tmp_path):
    out = tmp_path / "demo"
    written, preview = make_demo(out, settings=small(), directions=4, frames=2, progress=lambda t: None)
    names = {p.relative_to(out).as_posix() for p in written}
    assert {"project.godot", "main.tscn", "main.gd", "unit.gd", "map/map.json", "map/terrain.tres",
            "units/unit.png", "units/unit.tres", "units/unit_facing.gd", "units/unit_player.tres",
            "units/unit_enemy.tres", "units/team_colours.gdshader"} <= names
    project = (out / "project.godot").read_text()
    assert "viewport_width=640" in project and 'scale_mode="integer"' in project
    assert "default_texture_filter=0" in project
    data = json.loads((out / "map" / "map.json").read_text())
    assert (data["cols"], data["rows"]) == (20, 12) and len(data["cells"]) == 240
    blocked = {tuple(b) for b in data["blocked"]}
    assert {(x, 11) for x in range(20)} <= blocked                    # half-visible last row
    assert "water" in BLOCKING and len(blocked) > 20 + 5                # the lake blocks too
    assert [u["team"] for u in data["units"]] == ["player", "player", "enemy", "enemy"]
    for u in data["units"]:
        assert tuple(u["cell"]) not in blocked
    assert data["unit"]["idle"] == "idle" and data["unit"]["walk"] == "walk"
    assert preview.shape == (360, 640, 4) and preview[..., 3].min() == 255
    assert "walk_SE" not in (out / "units" / "unit.tres").read_text()   # 4 directions only
    with pytest.raises(ValueError, match="isn't empty"):
        (tmp_path / "other").mkdir()
        (tmp_path / "other" / "x.txt").write_text("x")
        make_demo(tmp_path / "other", settings=small(), progress=lambda t: None)


def test_still_unit_and_terrain_specs(tmp_path):
    from test_anim3d import write_robot
    robot = write_robot(tmp_path / "robot.glb", anims=False)
    names, sources = parse_terrains(["grass", "lava=prompt:hot rocks"])
    assert names == ("grass", "lava") and sources == {"lava": "prompt:hot rocks"}
    pytest.importorskip("trimesh")
    written, _ = make_demo(tmp_path / "d", [robot], small(), directions=4, terrains=("grass", "dirt"),
                           progress=lambda t: None)
    data = json.loads((tmp_path / "d" / "map" / "map.json").read_text())
    assert data["unit"]["idle"] == "" and data["terrains"] == ["grass", "dirt"]
    assert '&"E", &"S", &"W", &"N"' in (tmp_path / "d" / "units" / "unit_facing.gd").read_text()


CHECK = """extends SceneTree

func _init():
\tvar main = load("res://main.tscn").instantiate()
\troot.add_child(main)
\tawait process_frame
\tvar hero = main.units.get_child(0)
\tvar enemy = main.units.get_child(2)
\tprint("A ", main.units.get_child_count(), " ", hero.sprite.animation, " ", enemy.sprite.animation,
\t\t" ", enemy.sprite.material.get_shader_parameter("team"))
\tmain.click(hero.cell)
\tvar start = hero.cell
\tvar target = start
\tfor x in range(start.x + 5, start.x, -1):
\t\tfor y in [start.y - 2, start.y + 2, start.y]:
\t\t\tif target == start and main.path_to(hero, Vector2i(x, y)).size() > 0:
\t\t\t\ttarget = Vector2i(x, y)
\tmain.click(target)
\tprint("B ", hero.moving(), " ", String(hero.sprite.animation).begins_with("walk_"))
\tfor i in 2000:
\t\tif not hero.moving():
\t\t\tbreak
\t\thero._process(0.05)
\tprint("C ", hero.cell == target, " ", hero.spent, " ", String(hero.sprite.animation).begins_with("idle_"))
\tmain.click(start)
\tprint("D ", main.selected == null)
\tmain.new_turn()
\tprint("E ", hero.spent)
\tquit()
"""


@pytest.mark.skipif(not GODOT, reason="set GODOT_BIN to run the demo in Godot")
def test_demo_runs_in_godot(tmp_path):
    out = tmp_path / "demo"
    make_demo(out, settings=small(), directions=8, frames=2, progress=lambda t: None)
    (out / "check.gd").write_text(CHECK)
    run = [GODOT, "--headless", "--path", str(out)]
    subprocess.run(run + ["--import"], capture_output=True, timeout=300)
    res = subprocess.run(run + ["-s", "check.gd"], capture_output=True, text=True, timeout=300)
    lines = {x[0]: x[2:] for x in res.stdout.splitlines() if x[:2] in ("A ", "B ", "C ", "D ", "E ")}
    assert lines.get("A") == "4 idle_E idle_W 2", res.stdout + res.stderr
    assert lines["B"] == "true true" and lines["C"] == "true true true"
    assert lines["D"] == "true" and lines["E"] == "false"
    assert "SCRIPT ERROR" not in res.stderr, res.stderr


def test_cli_demo(tmp_path, capsys):
    out = tmp_path / "demo"
    assert main(["demo", "-o", str(out), "--directions", "4", "--frames", "2", "-W", "24", "-H", "24",
                 "--preview"]) == 0
    text = capsys.readouterr().out
    assert "Open" in text and (out / "project.godot").exists() and (tmp_path / "demo_preview.png").exists()


def test_mcp_godot_demo(tmp_path):
    import asyncio
    pytest.importorskip("mcp")
    from pixel_pipeline.mcp_server import server
    result = asyncio.run(server.call_tool("godot_demo", {
        "out_dir": str(tmp_path / "d"), "directions": 4, "frames": 2, "width": 24, "height": 24}))
    content = result[0] if isinstance(result, tuple) else getattr(result, "content", result)
    assert "built-in knight" in content[0].text and getattr(content[1], "type", "") == "image"

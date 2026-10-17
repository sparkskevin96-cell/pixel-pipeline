"""Autoplay for the HD-2D board (hd2d_autoplay): script checks, files, and the timeline running to
the end in headless Godot (tactics turns with moves/attacks/casts/kills, action mode with free
movement, swings, knockback, bushes, pickups)."""
import json
import os
import re
import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image

from pixel_pipeline.diorama import Diorama, Unit
from pixel_pipeline.hd2d import add_effect, make_hd2d
from pixel_pipeline.hd2d_autoplay import unit_ids, validate

from test_hd2d import fake_sheet

GODOT = os.environ.get("GODOT_BIN") or shutil.which("godot")


def fake_effect(path, tags=(("fly", 2), ("impact", 3))):
    """A tiny effect sheet (effects.write_effect layout) with frame tags."""
    w = h = 16
    n = sum(k for _, k in tags)
    sheet = np.zeros((h, w * n, 4), np.uint8)
    frames, meta_tags, i = [], [], 0
    for name, k in tags:
        meta_tags.append({"name": name, "from": i, "to": i + k - 1, "direction": "forward"})
        for j in range(k):
            sheet[4:12, i * w + 4:i * w + 12] = (255, 160 + 20 * j, 60, 255)
            frames.append({"filename": f"{name} {j}", "duration": 60, "frame": {"x": i * w, "y": 0, "w": w, "h": h}})
            i += 1
    Image.fromarray(sheet).save(path)
    path.with_suffix(".json").write_text(json.dumps(
        {"frames": frames, "meta": {"size": {"w": w * n, "h": h}, "frameTags": meta_tags}}))
    return path


def arena(units, cols=8, rows=6, props=()):
    h = np.full((rows, cols), 2, int)
    h[:, cols - 2:] = 3                                    # a step up on the right
    t = np.full((rows, cols), "grass", object)
    t[0, :] = "cobble"
    return Diorama(cols, rows, h, t, list(props), units, name="arena", sea=False)


TACTICS = {
    "mode": "tactics", "seed": 3, "camera": {"yaw": 45, "pitch": 34, "dist": 20, "follow": "knight"},
    "names": {"knight": "Sir Test"},
    "steps": [
        {"do": "banner", "text": "PLAYER PHASE", "time": 0.3},
        {"do": "select", "unit": "knight", "time": 0.2},
        {"do": "move", "unit": "knight", "to": [3, 2]},
        {"do": "attack", "unit": "knight", "target": "dog", "damage": 5, "effect": "fx"},
        {"do": "camera", "yaw": -45, "time": 0.5},
        {"do": "cast", "unit": "mage", "target": "dog", "effect": "fx", "tag": "impact",
         "projectile": "fx", "damage": 50, "windup": 0.1, "after": 0.2},
        {"do": "cast", "unit": "mage", "target": "knight", "heal": 3, "windup": 0.1, "after": 0.1},
        {"do": "end_turn", "time": 0.2},
        {"do": "move", "unit": "dog#2", "to": [5, 3]},
        {"do": "attack", "unit": "dog#2", "target": "knight", "damage": 4},
        {"do": "end_turn", "time": 0.2},
        {"do": "banner", "text": "VICTORY", "time": 0.2},
    ]}


def test_validate_and_ids():
    assert unit_ids(["knight", "dog", "dog", "dog"]) == ["knight", "dog", "dog#2", "dog#3"]
    assert validate(TACTICS, ["knight", "mage", "dog", "dog#2"]) == []
    errs = validate({"mode": "racing", "steps": [{"do": "fly"}, {"do": "move", "unit": "x"},
                                                {"do": "goto", "unit": "knight"}]}, ["knight"])
    assert any("mode" in e for e in errs) and any("fly" in e for e in errs)
    assert any("no unit 'x'" in e for e in errs) and any("needs 'to'" in e for e in errs)
    assert any("needs 'path'" in e for e in errs)


def _tactics_project(tmp_path):
    units = [Unit("knight", 1, 2, 0, "E", hp=30), Unit("mage", 0, 3, 0, "E", hp=20),
             Unit("dog", 4, 2, 1, "W", hp=12), Unit("dog", 6, 3, 1, "W", hp=12)]
    out = tmp_path / "p"
    sheets = {"knight": fake_sheet(tmp_path / "knight.png", anims=("idle", "walk", "attack", "hurt")),
              "mage": fake_sheet(tmp_path / "mage.png", anims=("idle", "walk", "cast")),
              "dog": fake_sheet(tmp_path / "dog.png", anims=("idle", "walk", "death"))}
    make_hd2d(out, sheets, arena(units), iso=False, tiles=False, autoplay=TACTICS,
              effects={"fx": fake_effect(tmp_path / "fx.png")})
    return out


def test_project_files_and_bad_script(tmp_path):
    out = _tactics_project(tmp_path)
    assert json.loads((out / "autoplay.json").read_text())["mode"] == "tactics"
    gd = (out / "autoplay.gd").read_text()
    assert "func _goto" in gd and "func _banner" in gd
    assert "_start_autoplay" in (out / "board.gd").read_text()
    fx = json.loads((out / "effects" / "fx" / "fx.effect.json").read_text())
    assert fx["tags"] == {"fly": [0, 1], "impact": [2, 4]}
    with pytest.raises(ValueError, match="no unit 'ghost'"):
        make_hd2d(tmp_path / "q", {"knight": tmp_path / "knight.png"}, arena([Unit("knight", 1, 1)]),
                  iso=False, tiles=False, autoplay={"steps": [{"do": "select", "unit": "ghost"}]})


def test_unit_ppu(tmp_path):
    out = tmp_path / "p"
    make_hd2d(out, {"knight": fake_sheet(tmp_path / "knight.png")}, arena([Unit("knight", 1, 1)]),
              iso=False, tiles=False, unit_ppu={"knight": 63.0})
    assert json.loads((out / "units" / "knight" / "knight.unit.json").read_text())["ppu"] == 63.0


def _run(out, scene="res://real3d.tscn"):
    run = subprocess.run([GODOT, "--headless", "--fixed-fps", "30", "--quit-after", "3000", "--path", str(out), scene,
                          "--", "--autoplay", "--autoplay-quit"],
                         capture_output=True, text=True, timeout=240)
    log = run.stdout + run.stderr
    assert "SCRIPT ERROR" not in log and "Parse Error" not in log, log[-3000:]
    m = re.search(r"AUTOPLAY UNITS (\{.*?\})", log, re.S)
    assert m and "AUTOPLAY DONE" in log, log[-3000:]
    return log, json.loads(m.group(1))


@pytest.mark.skipif(not GODOT, reason="set GODOT_BIN to run the autoplay in Godot")
def test_tactics_autoplay_runs_to_the_end(tmp_path):
    out = _tactics_project(tmp_path)
    log, hp = _run(out)
    n = len(TACTICS["steps"])
    assert f"AUTOPLAY DONE steps={n}" in log
    # knight 30 (healed +3, capped) - 4 (bitten) ; first dog: 5 + 50 -> dead ; dog#2 untouched
    assert hp == {"knight": 26, "mage": 20, "dog": 0, "dog#2": 12}
    log2, hp2 = _run(out)                                   # deterministic: the same again
    assert hp2 == hp
    t1 = re.search(r"AUTOPLAY DONE steps=\d+ time=([\d.]+)", log).group(1)
    t2 = re.search(r"AUTOPLAY DONE steps=\d+ time=([\d.]+)", log2).group(1)
    assert t1 == t2


ACTION = {
    "mode": "action", "hero": "knight", "coins": 7, "poof": "fx", "lean": 0.8,
    "camera": {"yaw": 0, "pitch": 64, "dist": 16, "follow": "knight", "dof": 0.03},
    "steps": [
        {"do": "goto", "unit": "knight", "path": [[2, 2], [2.5, 3]], "speed": 3, "follow": True},
        {"do": "swing", "unit": "knight", "dir": "E", "cut": [[3, 3]], "drop": {"id": "c1", "kind": "coin"}},
        {"do": "pickup", "id": "c1"},
        {"do": "goto", "unit": "dog", "path": [[4, 2]], "speed": 5, "async": True},
        {"do": "wait", "time": 0.3},
        {"do": "hurt", "unit": "knight", "by": "dog", "damage": 3},
        {"do": "swing", "unit": "knight", "toward": "dog", "hits": ["dog"], "damage": 20, "knock": 0.8,
         "drops": {"dog": {"id": "h1", "kind": "heart"}}},
        {"do": "wait", "time": 1.5},
        {"do": "pickup", "id": "h1"},
        {"do": "cast", "unit": "knight", "at": [6, 1], "projectile": "arrow", "effect": "fx", "windup": 0.1},
        {"do": "spawn", "id": "g1", "kind": "gem", "at": [1, 1]},
        {"do": "fade", "to": 1.0, "time": 0.3},
    ]}


@pytest.mark.skipif(not GODOT, reason="set GODOT_BIN to run the autoplay in Godot")
def test_action_autoplay_runs_to_the_end(tmp_path):
    units = [Unit("knight", 1, 2, 0, "E", hp=24), Unit("dog", 6, 2, 1, "W", hp=8)]
    out = tmp_path / "p"
    sheets = {"knight": fake_sheet(tmp_path / "knight.png", anims=("idle", "walk", "attack", "hurt"),
                                   dirs=("E", "SE", "S", "SW", "W", "NW", "N", "NE")),
              "dog": fake_sheet(tmp_path / "dog.png", anims=("idle", "walk", "run", "death"))}
    make_hd2d(out, sheets, arena(units, props=[("bush", 3, 3), ("pot", 5, 5)]), iso=False, tiles=False,
              autoplay=ACTION, effects={"fx": fake_effect(tmp_path / "fx.png")})
    log, hp = _run(out)
    assert f"AUTOPLAY DONE steps={len(ACTION['steps'])}" in log
    assert hp == {"knight": 24, "dog": 0}               # 24 - 3 bitten + 4 heart (capped at 24)
    assert "AUTOPLAY COINS 8 PICKUPS 1" in log          # the bush's coin; the gem still lies there

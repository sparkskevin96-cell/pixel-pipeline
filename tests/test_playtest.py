"""playtest.py + godot_runtime: maps, card rules, runtime install helpers (no Godot, no GPU).
The games themselves are checked headless by playtest.smoke_test / --pp-selftest when GODOT_BIN is set."""
import os
from pathlib import Path

import pytest

from pixel_pipeline import godot_runtime, playtest


def test_runtime_files_present():
    for n in ("pp_debug.gd", "pp_tactics.gd", "action_play.gd", "side_game.gd", "side_player.gd",
              "side_enemy.gd", "pickup_2d.gd", "card_game.gd", "pp_manifest.gd"):
        assert n in godot_runtime.FILES
        assert godot_runtime.source(n).strip()


def test_install_autoloads_and_settings(tmp_path):
    pg = tmp_path / "project.godot"
    pg.write_text('config_version=5\n\n[application]\n\nconfig/name="x"\n', encoding="utf-8")
    files = godot_runtime.install(tmp_path, ["pp_debug.gd"])
    assert files[0].exists() and files[0].parent.name == "pp_runtime"
    godot_runtime.add_autoloads(pg, {"PPDebug": "res://pp_runtime/pp_debug.gd"})
    godot_runtime.add_autoloads(pg, {"PPDebug": "res://pp_runtime/pp_debug.gd"})       # kept once
    godot_runtime.set_setting(pg, "application", "run/main_scene", '"res://main.tscn"')
    godot_runtime.set_setting(pg, "application", "run/main_scene", '"res://other.tscn"')
    godot_runtime.set_setting(pg, "display", "window/size/window_width_override", "1280")
    text = pg.read_text(encoding="utf-8")
    assert text.count('PPDebug="*res://pp_runtime/pp_debug.gd"') == 1
    assert text.count("run/main_scene=") == 1 and 'run/main_scene="res://other.tscn"' in text
    assert "[display]" in text and "window_width_override=1280" in text


@pytest.mark.parametrize("make", [playtest.tactics_map, playtest.action_map])
def test_board_maps(make):
    d = make()
    assert d.height.shape == (d.rows, d.cols) and d.terrain.shape == (d.rows, d.cols)
    teams = {u.team for u in d.units}
    assert teams == {0, 1}
    for u in d.units:                                     # units stand on dry land
        assert d.terrain[u.row, u.col] != "water"
    for _, c, r in d.props:
        assert 0 <= c < d.cols and 0 <= r < d.rows


def test_tactics_map_has_stairs_water_torches():
    d = playtest.tactics_map()
    assert (d.terrain == "water").any() and d.height.max() - d.height.min() >= 4
    kinds = {k for k, _, _ in d.props}
    assert {"torch", "brazier"} <= kinds
    # the plateau (level 4) is reachable one level at a time from level 2
    assert 3 in set(d.height.ravel())


def test_platformer_level():
    lv = playtest.platformer_level()
    solid = {tuple(c) for c in lv["solid"]}
    sx, sy = lv["start"]
    assert (sx, sy) not in solid and (sx, sy + 1) in solid       # starts standing on ground
    fx, fy = lv["flag"]
    assert (fx, fy + 1) in solid
    for e in lv["enemies"]:
        assert (e["x"], e["y"]) not in solid


def test_card_rules_and_text():
    for cid, d in playtest.CARD_DEFS.items():
        name, cost, typ, rarity, faction, art, ops, target, fx, anim = d
        assert typ in ("Attack", "Skill", "Power") and target in ("enemy", "self")
        assert fx in playtest.FX_SPECS or fx in ("slash", "spark")
        text = playtest.card_text(ops)
        assert text.endswith(".") and len(text) < 60, cid
    assert playtest.card_text([["dmg", 3, 3]]) == "Deal 3 damage 3 times."
    assert set(playtest.STARTER) <= set(playtest.CARD_DEFS)


def test_launchers_and_readme(tmp_path):
    game = tmp_path / "cards"
    game.mkdir()
    bats = playtest.write_launchers(game, "cards", godot="C:/x/Godot_v4_console.exe", also_in=tmp_path)
    names = sorted(p.name for p in bats)
    assert names == ["Edit_cards.bat", "Edit_cards.bat", "Play_cards.bat", "Play_cards.bat"]
    play = (game / "Play_cards.bat").read_text(encoding="utf-8")
    assert "GODOT_BIN" in play and '--path "%~dp0."' in play
    assert '--path "%~dp0cards\\."' in (tmp_path / "Play_cards.bat").read_text(encoding="utf-8")
    assert " -e " in (game / "Edit_cards.bat").read_text(encoding="utf-8")
    r = playtest.write_readme(game, "cards", "about")
    assert "F12" in r.read_text(encoding="utf-8")


def test_staging_project_keeps_out_of_user_projects(tmp_path):
    proj = playtest.staging_project(tmp_path, "side", "side-scroller", directions=2, elevation=30.0)
    assert Path(proj.folder).parent == tmp_path / "projects"
    st = proj.style
    assert st.directions == 2 and st.elevation == 30.0 and st.project_type == "side-scroller"

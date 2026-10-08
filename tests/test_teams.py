import os
import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image

import pixel_pipeline.teams as teams
from pixel_pipeline import settings_for
from pixel_pipeline.cli import main
from pixel_pipeline.teams import (_lch, make_portrait, portrait_crop, team_family, team_variants,
                                  write_godot_teams)

BLUE, BLUE_DARK, GOLD, SKIN = (60, 90, 200), (30, 45, 120), (240, 200, 60), (240, 200, 160)


def unit():
    img = np.zeros((20, 12, 4), np.uint8)
    img[0:5, 3:9] = (*SKIN, 255)
    img[5:14, 2:10] = (*BLUE, 255)
    img[12:14, 2:10] = (*BLUE_DARK, 255)
    img[9, 2:10] = (*GOLD, 255)
    img[14:20, 3:9] = (40, 40, 40, 255)
    return img


@pytest.fixture(autouse=True)
def no_face_model(monkeypatch):
    """Tests never download the face detector: portraits use the subject's top."""
    monkeypatch.setattr(teams, "_face_model_path", None)


def hue(rgb):
    return _lch(np.array([rgb], np.uint8))[0, 2]


def test_team_family_is_the_blue_armour_not_the_gold_trim():
    img = unit()
    cols, counts = teams.sprite_colours(img)
    fam = {tuple(c) for c in cols[team_family(cols, counts)].tolist()}
    assert fam == {BLUE, BLUE_DARK}
    fam_gold = {tuple(c) for c in cols[team_family(cols, counts, "#f0c83c")].tolist()}
    assert fam_gold == {GOLD}


def test_variants_move_only_the_team_colours_and_keep_shading():
    img = unit()
    variants, cols, pals = team_variants(img)
    assert set(variants) == {"player", "enemy", "ally", "other", "spent"}
    enemy = variants["enemy"]
    armour, dark = enemy[7, 5, :3], enemy[13, 5, :3]
    assert abs((hue(armour) - hue((210, 58, 50)) + 180) % 360 - 180) < 25      # red now
    assert _lch(np.array([dark]))[0, 0] < _lch(np.array([armour]))[0, 0]     # still darker
    assert tuple(enemy[2, 5, :3]) == SKIN and tuple(enemy[9, 5, :3]) == GOLD   # untouched
    spent = variants["spent"][7, 5, :3].astype(int)
    assert np.ptp(spent) < 20                                                  # grey
    assert (variants["ally"][..., 3] == img[..., 3]).all()


def test_godot_team_files(tmp_path):
    _, cols, pals = team_variants(unit())
    written = write_godot_teams(cols, pals, tmp_path, "unit")
    strip = np.array(Image.open(tmp_path / "unit_teams.png"))
    assert strip.shape == (6, len(cols), 3)                     # drawn + 4 teams + spent
    assert "texelFetch(palette, ivec2(i, team), 0)" in (tmp_path / "team_colours.gdshader").read_text()
    enemy = (tmp_path / "unit_enemy.tres").read_text()
    assert "shader_parameter/team = 2" in enemy and f"shader_parameter/colors = {len(cols)}" in enemy
    assert len(written) == 2 + 6


GODOT = os.environ.get("GODOT_BIN") or shutil.which("godot")
XVFB = shutil.which("xvfb-run")


@pytest.mark.skipif(not (GODOT and XVFB), reason="needs GODOT_BIN and xvfb-run to render")
def test_shader_swaps_colours_in_godot(tmp_path):
    img = unit()
    Image.fromarray(img).save(tmp_path / "unit.png")
    _, cols, pals = team_variants(img)
    write_godot_teams(cols, pals, tmp_path, "unit")
    (tmp_path / "project.godot").write_text(
        'config_version=5\n[application]\nconfig/name="t"\n[rendering]\n'
        'renderer/rendering_method="gl_compatibility"\n')
    (tmp_path / "shot.gd").write_text(
        "extends SceneTree\nfunc _init():\n"
        "\tvar s = Sprite2D.new()\n\ts.texture = load(\"res://unit.png\")\n\ts.centered = false\n"
        "\ts.material = load(\"res://unit_enemy.tres\")\n\troot.add_child(s)\n"
        "\tawait process_frame\n\tawait process_frame\n\tawait RenderingServer.frame_post_draw\n"
        "\tvar img = root.get_viewport().get_texture().get_image()\n"
        "\tprint(\"PX \", img.get_pixel(5, 7).to_html(false), \" \", img.get_pixel(5, 2).to_html(false))\n"
        "\tquit()\n")
    run = [XVFB, "-a", GODOT, "--rendering-driver", "opengl3", "--path", str(tmp_path)]
    subprocess.run(run + ["--import"], capture_output=True, timeout=180)
    out = subprocess.run(run + ["-s", "shot.gd"], capture_output=True, text=True, timeout=180).stdout
    want = pals["enemy"][[tuple(c) for c in cols.tolist()].index(BLUE)]
    line = next(x for x in out.splitlines() if x.startswith("PX "))
    armour, skin = line.split()[1:]
    got = np.array([int(armour[i:i + 2], 16) for i in (0, 2, 4)])
    assert np.abs(got - want).max() <= 2
    assert skin == "%02x%02x%02x" % SKIN


def test_portrait_crop_takes_the_top_of_a_full_body():
    fig = np.zeros((200, 60, 4), np.uint8)
    fig[10:200, 10:50, 3] = 255
    x0, y0, x1, y1 = portrait_crop(fig)
    assert y0 <= 10 and 60 <= y1 - y0 <= 100 and abs((x0 + x1) / 2 - 30) <= 2


def test_make_portrait_and_cli(tmp_path, capsys):
    fig = np.zeros((240, 100, 4), np.uint8)
    fig[10:60, 30:70] = (240, 190, 150, 255)      # head
    fig[60:240, 15:85] = (200, 40, 40, 255)       # body
    Image.fromarray(fig).save(tmp_path / "hero.png")
    sprite = make_portrait(tmp_path / "hero.png", settings_for("person", width=48, height=48))
    assert sprite.size == (48, 48)
    skin = (np.abs(sprite.rgba[..., :3].astype(int) - (240, 190, 150)).sum(-1) < 60).sum()
    assert skin > 48 * 48 * 0.15                                                  # mostly face
    assert main(["portrait", str(tmp_path / "hero.png"), "-o", str(tmp_path / "p.png"),
                 "-W", "64", "-H", "64", "--flip", "--scale", "4"]) == 0
    assert Image.open(tmp_path / "p@4x.png").size == (256, 256)
    Image.fromarray(unit()).save(tmp_path / "unit.png")
    assert main(["teams", str(tmp_path / "unit.png"), "-o", str(tmp_path / "t"), "--godot"]) == 0
    assert "2 of 5 colours are team colours" in capsys.readouterr().out
    assert (tmp_path / "t" / "unit_enemy.png").exists() and (tmp_path / "t" / "unit_spent.tres").exists()


def test_mcp_team_colours(tmp_path):
    import asyncio
    pytest.importorskip("mcp")
    from pixel_pipeline.mcp_server import server
    Image.fromarray(unit()).save(tmp_path / "unit.png")
    result = asyncio.run(server.call_tool("team_colours", {"sprite_path": str(tmp_path / "unit.png")}))
    content = result[0] if isinstance(result, tuple) else getattr(result, "content", result)
    assert content[0].text.startswith("2 of 5 colours") and getattr(content[1], "type", "") == "image"
    assert (tmp_path / "unit_teams.png").exists()

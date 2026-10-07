import numpy as np
import pytest

from pixel_pipeline import convert, settings_for
from pixel_pipeline.cli import main
from pixel_pipeline.palettes import BUILTIN, hex_to_rgb
from pixel_pipeline.systems import NES_HEX, RGB333_MD, RGB555, SYSTEMS


def colours(sprite):
    px = sprite.rgba.reshape(-1, 4)
    return np.unique(px[px[:, 3] == 255][:, :3], axis=0)


def within(cols, allowed):
    return (cols[:, None, :] == allowed[None]).all(-1).any(1).all()


@pytest.mark.parametrize("system", list(SYSTEMS))
def test_every_system_respects_its_colour_limit(blob, system):
    s = settings_for("person", system=system)
    sprite = convert(blob, s)
    assert (sprite.size[0], sprite.size[1]) == SYSTEMS[system].sizes["person"]
    assert len(colours(sprite)) <= SYSTEMS[system].sprite_colors


def test_nes_uses_hardware_palette_and_three_colours(blob):
    sprite = convert(blob, settings_for("animal", system="nes"))
    cols = colours(sprite)
    assert len(cols) <= 3 and within(cols, hex_to_rgb(NES_HEX))


def test_gameboy_uses_the_four_greens(blob):
    assert within(colours(convert(blob, settings_for("object", system="gameboy"))),
                  hex_to_rgb(BUILTIN["gameboy"]))


@pytest.mark.parametrize("system,levels", [("snes", RGB555), ("gba", RGB555), ("genesis", RGB333_MD)])
def test_colour_depth_systems_snap_every_channel(blob, system, levels):
    cols = colours(convert(blob, settings_for("animal", system=system)))
    assert np.isin(cols, levels).all()


def test_explicit_settings_still_win_over_system(blob):
    s = settings_for("person", system="snes", width=40, colors=8)
    assert (s.width, s.colors, s.system, s.outline_style) == (40, 8, "snes", "selout")
    assert len(colours(convert(blob, s))) <= 8


def test_background_gets_system_resolution(gradient):
    s = settings_for("background", system="gameboy")
    assert (s.width, s.height, s.colors) == (160, 144, 4)


def test_cli_system_flag(tmp_path, blob_png, capsys):
    out = tmp_path / "nes.png"
    assert main(["convert", str(blob_png), "-s", "animal", "--system", "nes", "-o", str(out)]) == 0
    assert "(16x16, 3 colours)" in capsys.readouterr().out


def test_drawn_rows_become_a_sprite_and_snap_to_a_system():
    from pixel_pipeline.drawing import sprite_from_rows
    rows = ["..00..", ".0110.", "012210", ".0110.", "..00.."]
    pal = ["#101820", "#c83030", "#f0d080"]
    free = sprite_from_rows(rows, pal)
    assert free.size == (6, 5) and free.rgba[0, 0, 3] == 0
    assert colours(free).tolist() == sorted([[16, 24, 32], [200, 48, 48], [240, 208, 128]])
    gb = sprite_from_rows(rows, pal, system="gameboy")
    assert within(colours(gb), hex_to_rgb(BUILTIN["gameboy"]))
    with pytest.raises(ValueError, match="palette has 3"):
        sprite_from_rows(["05"], pal)

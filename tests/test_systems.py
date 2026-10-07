import numpy as np
import pytest

from pixel_pipeline import convert, settings_for
from pixel_pipeline.cli import main
from pixel_pipeline.palettes import hex_to_rgb
from pixel_pipeline.systems import DMG_HEX, LOOKS, NES_HEX, READABLE_SIZES, RGB222, RGB333_MD, RGB555, SYSTEMS


def colours(sprite):
    px = sprite.rgba.reshape(-1, 4)
    return np.unique(px[px[:, 3] == 255][:, :3], axis=0)


def within(cols, allowed):
    return (cols[:, None, :] == allowed[None]).all(-1).any(1).all()


@pytest.mark.parametrize("look", LOOKS)
@pytest.mark.parametrize("system", list(SYSTEMS))
def test_every_system_respects_its_colour_limit(blob, system, look):
    s = settings_for("person", system=system, look=look)
    sprite = convert(blob, s)
    assert (sprite.size[0], sprite.size[1]) == SYSTEMS[system].size("person", look)
    assert len(colours(sprite)) <= SYSTEMS[system].max_colors("person", look)


def test_readable_is_the_default_and_bigger_than_authentic():
    readable, authentic = settings_for("person", system="nes"), settings_for("person", system="nes",
                                                                            look="authentic")
    assert readable.look == "readable" and (readable.width, readable.height) == READABLE_SIZES["person"]
    assert (authentic.width, authentic.height, authentic.colors) == (16, 24, 3)
    assert readable.colors == 12
    with pytest.raises(ValueError, match="Unknown look"):
        settings_for("person", system="nes", look="fancy")


@pytest.mark.parametrize("look,limit", [("authentic", 3), ("readable", 12)])
def test_nes_uses_hardware_palette(blob, look, limit):
    sprite = convert(blob, settings_for("animal", system="nes", look=look))
    cols = colours(sprite)
    assert len(cols) <= limit and within(cols, hex_to_rgb(NES_HEX))


def test_gameboy_uses_the_four_greens(blob):
    assert within(colours(convert(blob, settings_for("object", system="gameboy"))), hex_to_rgb(DMG_HEX))


def test_gameboy_maps_brightness_to_all_four_shades(gradient):
    # A colourful gradient: nearest-green matching would collapse it; brightness mapping
    # spreads it over every shade, dark to light.
    s = settings_for("object", system="gameboy", remove_background=False, crop_to_subject=False)
    cols = colours(convert(gradient, s))
    assert len(cols) == 4 and within(cols, hex_to_rgb(DMG_HEX))


def test_master_system_colours_are_its_64(blob):
    cols = colours(convert(blob, settings_for("animal", system="sms")))
    assert np.isin(cols, RGB222).all()


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
    assert main(["convert", str(blob_png), "-s", "animal", "--system", "nes", "--look", "authentic",
                 "-o", str(out)]) == 0
    assert "(16x16, 3 colours)" in capsys.readouterr().out
    assert main(["convert", str(blob_png), "-s", "animal", "--system", "nes", "-o", str(out)]) == 0
    assert "(48x48," in capsys.readouterr().out


def test_drawn_rows_become_a_sprite_and_snap_to_a_system():
    from pixel_pipeline.drawing import sprite_from_rows
    rows = ["..00..", ".0110.", "012210", ".0110.", "..00.."]
    pal = ["#101820", "#c83030", "#f0d080"]
    free = sprite_from_rows(rows, pal)
    assert free.size == (6, 5) and free.rgba[0, 0, 3] == 0
    assert colours(free).tolist() == sorted([[16, 24, 32], [200, 48, 48], [240, 208, 128]])
    gb = sprite_from_rows(rows, pal, system="gameboy")
    assert within(colours(gb), hex_to_rgb(DMG_HEX))
    with pytest.raises(ValueError, match="palette has 3"):
        sprite_from_rows(["05"], pal)

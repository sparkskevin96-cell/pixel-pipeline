import numpy as np
import pytest
from PIL import Image

from pixel_pipeline.color import luminance
from pixel_pipeline.palettes import BUILTIN, auto_palette, load_palette


def test_builtin_sizes():
    sizes = {name: len(load_palette(name)) for name in BUILTIN}
    assert sizes == {"pico-8": 16, "sweetie-16": 16, "endesga-32": 32, "gameboy": 4, "1bit": 2}


def test_auto_is_none():
    assert load_palette("auto") is None


def test_gpl_file(tmp_path):
    p = tmp_path / "x.gpl"
    p.write_text("GIMP Palette\nName: test\nColumns: 2\n#\n255   0   0\tRed\n  0 128 255 Blue\n")
    assert load_palette(str(p)).tolist() == [[255, 0, 0], [0, 128, 255]]


def test_hex_file(tmp_path):
    p = tmp_path / "x.hex"
    p.write_text("ff0000\n#00ff00\n0000FF\n")
    assert load_palette(str(p)).tolist() == [[255, 0, 0], [0, 255, 0], [0, 0, 255]]


def test_image_palette_keeps_order_and_skips_transparent(tmp_path):
    arr = np.array([[[9, 9, 9, 255], [200, 0, 0, 255], [9, 9, 9, 255], [1, 2, 3, 0]]], np.uint8)
    p = tmp_path / "pal.png"
    Image.fromarray(arr, "RGBA").save(p)
    assert load_palette(str(p)).tolist() == [[9, 9, 9], [200, 0, 0]]


def test_unknown_palette_and_bad_slug():
    with pytest.raises(ValueError, match="Unknown palette"):
        load_palette("not-a-palette")
    with pytest.raises(ValueError, match="slug"):
        load_palette("lospec:../etc/passwd")


def test_auto_palette_size_and_order():
    rng = np.random.default_rng(1)
    pixels = rng.integers(0, 256, (500, 3), dtype=np.uint8)
    pal = auto_palette(pixels, 8)
    assert len(pal) == 8
    lum = luminance(pal)
    assert (np.diff(lum) >= 0).all()
    few = np.array([[1, 2, 3], [4, 5, 6]] * 10, np.uint8)
    assert len(auto_palette(few, 8)) == 2

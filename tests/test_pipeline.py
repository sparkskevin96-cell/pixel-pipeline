import sys

import numpy as np
import pytest

from pixel_pipeline import convert, convert_batch, settings_for
from pixel_pipeline.color import luminance
from pixel_pipeline.cutout import CutoutUnavailable
from pixel_pipeline.palettes import BUILTIN, hex_to_rgb


def opaque_colors(sprite):
    px = sprite.rgba.reshape(-1, 4)
    return np.unique(px[px[:, 3] == 255][:, :3], axis=0)


def in_palette(colors, palette):
    return (colors[:, None, :] == palette[None]).all(-1).any(1).all()


def test_size_palette_and_transparency(blob):
    s = settings_for("object", width=24, height=20, colors=8)
    sprite = convert(blob, s)
    assert sprite.size == (24, 20)
    assert sprite.rgba[0, 0, 3] == 0
    assert set(np.unique(sprite.rgba[..., 3])) <= {0, 255}
    assert len(sprite.palette) <= 8
    assert in_palette(opaque_colors(sprite), sprite.palette)
    assert ((sprite.indices >= 0) == (sprite.rgba[..., 3] == 255)).all()


def test_outline_is_darkest_colour_on_silhouette_edge(blob):
    sprite = convert(blob, settings_for("object", width=32, height=32))
    mask = sprite.rgba[..., 3] == 255
    padded = np.pad(mask, 1)
    edge = mask & ~(padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:])
    darkest = sprite.palette[np.argmin(luminance(sprite.palette))]
    assert edge.any()
    assert (sprite.rgba[edge][:, :3] == darkest).all()


def test_custom_outline_colour_added_to_palette(blob):
    sprite = convert(blob, settings_for("object", palette="gameboy", outline_color="#ff00ff"))
    assert [255, 0, 255] in sprite.palette.tolist()
    assert len(sprite.palette) == 5


def test_height_auto_keeps_aspect(gradient):
    s = settings_for("background", width=40, height=0, fit="contain")  # 0 = keep aspect
    assert convert(gradient, s).size == (40, 20)


def test_builtin_palette_is_respected(blob):
    sprite = convert(blob, settings_for("animal", palette="pico-8", dither="floyd"))
    assert in_palette(opaque_colors(sprite), hex_to_rgb(BUILTIN["pico-8"]))


def test_bottom_anchor_puts_subject_on_floor(blob):
    sprite = convert(blob, settings_for("person", width=32, height=48))
    rows = np.nonzero((sprite.rgba[..., 3] == 255).any(1))[0]
    assert rows.max() >= 45      # sits on the floor (a rounded tip may round away)
    assert rows.min() > 10       # empty headroom above a wide subject


def test_batch_shares_one_palette(blob, gradient):
    s = settings_for("object", remove_background=False, colors=6)
    a, b = convert_batch([blob, gradient], s)
    assert np.array_equal(a.palette, b.palette)
    a2, b2 = convert_batch([blob, gradient], s, shared_palette=False)
    assert not np.array_equal(a2.palette, b2.palette)


def _seam(rgba):
    return np.abs(rgba[:, 0, :3].astype(int) - rgba[:, -1, :3].astype(int)).mean()


def test_tileable_removes_the_wrap_seam(gradient):
    base = dict(width=48, height=24, dither="none", colors=16)
    plain = convert(gradient, settings_for("background", **base))
    tiled = convert(gradient, settings_for("background", tileable=True, **base))
    assert _seam(tiled.rgba) < _seam(plain.rgba) / 3


def test_missing_rembg_raises_helpful_error(gradient, monkeypatch):
    monkeypatch.setitem(sys.modules, "rembg", None)
    with pytest.raises(CutoutUnavailable, match="pixel-pipeline\\[cutout\\]"):
        convert(gradient, settings_for("animal"))


def test_unknown_setting_rejected():
    with pytest.raises(TypeError):
        settings_for("object", widht=10)

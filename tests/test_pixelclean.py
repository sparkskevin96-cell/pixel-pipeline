"""Pixel clean-up of rendered sprites (pixelclean.py)."""
import cv2
import numpy as np
import pytest

from pixel_pipeline.finishing import outline_ring, selout_colors
from pixel_pipeline.pipeline import Sprite, sprite_from_indices
from pixel_pipeline.pixelclean import _N4, _pixel_perfect, boundary, clean_sprite, clean_sprites, snap_palette

PAL = np.array([[20, 20, 30], [200, 60, 60], [205, 66, 64], [250, 250, 250], [90, 30, 30],
                [60, 20, 20]], np.uint8)


def disc_sprite(r=10, size=32, outline="selout", thick=False, spur=False):
    yy, xx = np.mgrid[:size, :size]
    body = (yy - size / 2) ** 2 + (xx - size / 2) ** 2 <= r * r
    if spur:
        body[size // 2 - r - 1, size // 2] = True          # a 1 px nub on the top edge
    idx = np.where(body, 1, -1).astype(np.int16)
    # thick: an 8-connected ring (square dilation), which doubles every staircase corner
    ring = (cv2.dilate(body.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~body
            if thick else outline_ring(body))
    if outline == "selout":
        idx[ring] = selout_colors(idx, ring, PAL)[ring]
    else:
        idx[ring] = 0
    sp = sprite_from_indices(idx, PAL)
    n = np.zeros_like(sp.rgba)
    n[sp.rgba[..., 3] > 0] = (128, 128, 255, 255)
    sp.normal = n
    return sp


def needless(mask, body):
    """Outline pixels no body pixel touches side-on (doubled corners, stubs)."""
    near = np.zeros_like(body)
    for dy, dx in _N4:
        near |= np.roll(np.roll(body, dy, 0), dx, 1)
    return int((mask & ~body & ~near).sum())


def test_outline_becomes_pixel_perfect_and_stays_closed():
    plain = disc_sprite()
    assert needless(plain.rgba[..., 3] > 0, plain.indices == 1) == 0   # finish()'s ring already is
    out = clean_sprite(plain, "selout")
    assert not ((out.rgba[..., 3] > 0) & (plain.rgba[..., 3] == 0)).any()  # only takes away
    assert ((out.normal[..., 3] > 0) == (out.rgba[..., 3] > 0)).all()
    # an 8-connected ring doubles every staircase corner: those go, the ring stays closed
    sp = disc_sprite(thick=True)
    body = sp.indices == 1
    idx = sp.indices.astype(np.int32).copy()
    assert needless(idx >= 0, body) > 0
    removed = _pixel_perfect(idx, body)
    mask = idx >= 0
    assert removed.any() and needless(mask, body) == 0
    for dy, dx in _N4:                       # no body pixel touches see-through side-on
        assert not (body & ~np.roll(np.roll(mask, -dy, 0), -dx, 1)).any()


def test_nub_on_a_flat_edge_is_flattened():
    sp = disc_sprite(r=10, size=32, spur=True)
    top = 16 - 10 - 1
    assert sp.rgba[top - 1, 16, 3] == 255                      # the nub's own outline
    out = clean_sprite(sp, "selout")
    assert out.rgba[top - 1, 16, 3] == 0 and out.rgba[top, 16, 3] == 255
    assert out.indices[top, 16] != 1                           # the nub is outline now


def test_specks_go_and_islands_merge_but_details_stay():
    sp = disc_sprite(r=9, size=40)
    idx = sp.indices.copy()
    idx[3, 3] = 1                                 # a lone pixel far off the body
    idx[20, 20] = 2                               # a near-identical shade: banding noise
    idx[17, 22] = 3                               # a bright eye glint: stays
    sp = Sprite(sprite_from_indices(idx, PAL).rgba, idx, PAL, sp.normal)
    out = clean_sprite(sp, "selout")
    assert out.indices[3, 3] == -1 and out.rgba[3, 3, 3] == 0
    assert out.indices[20, 20] == 1
    assert out.indices[17, 22] == 3


def test_shared_palette_snapped_the_same_for_every_frame():
    frames = [disc_sprite(r) for r in (8, 9, 10)]
    target = np.array([[0, 0, 0], [255, 0, 0], [255, 255, 255]], np.uint8)
    pal, lut = snap_palette(PAL, target)
    assert len(pal) <= 3 and lut[1] == lut[2]
    out = clean_sprites(frames, "selout", palette=target)
    for o in out:
        used = {tuple(c) for c in o.rgba[o.rgba[..., 3] > 0, :3].tolist()}
        assert used <= {tuple(c) for c in target.tolist()}


def test_solid_and_none_outlines():
    out = clean_sprite(disc_sprite(outline="solid"), "solid")
    ring = boundary(out.rgba[..., 3] > 0)
    assert (out.indices[ring] == 0).all()                    # still the outline colour
    plain = disc_sprite(outline="solid")
    idx = plain.indices.copy()
    idx[16, 16] = -1                                         # a pinhole
    sp = Sprite(sprite_from_indices(idx, PAL).rgba, idx, PAL, plain.normal)
    sp.normal[16, 16] = 0
    out = clean_sprite(sp, "none")
    assert out.indices[16, 16] >= 0 and out.normal[16, 16, 3] == 255


@pytest.mark.parametrize("dither", ["none", "bayer4"])
def test_animated_turnaround_clean(tmp_path, dither):
    trimesh = pytest.importorskip("trimesh")
    from test_autorig import humanoid
    from pixel_pipeline import settings_for
    from pixel_pipeline.anim3d import animated_turnaround
    from pixel_pipeline.autorig import auto_rig
    rig = auto_rig(humanoid(tmp_path / "h.glb"), tmp_path / "h.rigged.glb", actions=("idle",))
    s = settings_for("person", width=32, height=40, colors=12, dither=dither)
    s.outline_style = "selout"
    raw, _ = animated_turnaround([rig.path], s, 4, 2, points=60_000, size=160, normals=True)
    clean, _ = animated_turnaround([rig.path], s, 4, 2, points=60_000, size=160, normals=True,
                                   clean=True)
    for a, b in zip(raw.flat(), clean.flat()):
        assert b.rgba.shape == a.rgba.shape
        assert ((b.normal[..., 3] > 0) == (b.rgba[..., 3] > 0)).all()
    assert trimesh

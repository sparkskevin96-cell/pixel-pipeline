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


# --------------------------------------------------------------------------- hand-pixelled

from pixel_pipeline.color import rgb_to_lab  # noqa: E402
from pixel_pipeline.pixelclean import hand_pixel, material_groups  # noqa: E402


def two_materials(size=40, seed=0, normals=True, pal=None):
    """A noisy render-like sprite: left half grey stone, right half tan thatch, each with a
    lightness gradient + noise, quantised to a shared 24-colour palette; normals face left
    on the left half, right on the right half."""
    from pixel_pipeline.palettes import auto_palette
    from pixel_pipeline.quantize import quantize
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    body = (np.abs(yy - size / 2) < size * 0.4) & (np.abs(xx - size / 2) < size * 0.4)
    t = (yy / size)[..., None]
    stone = np.array([150, 152, 160]) * (1.1 - 0.6 * t)
    thatch = np.array([200, 160, 90]) * (1.1 - 0.6 * t)
    rgb = np.where((xx < size / 2)[..., None], stone, thatch) + rng.normal(0, 4, (size, size, 3))
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    pal = auto_palette(rgb[body], 24) if pal is None else pal
    idx = quantize(rgb, body, pal, "none", 0.0)
    ring = outline_ring(body)
    idx[ring] = selout_colors(idx, ring, pal)[ring]
    sp = sprite_from_indices(idx, pal)
    if normals:
        n = np.zeros_like(sp.rgba)
        mask = sp.rgba[..., 3] > 0
        n[mask & (xx < size / 2)] = (40, 128, 220, 255)     # facing left (lit side)
        n[mask & (xx >= size / 2)] = (216, 128, 220, 255)   # facing right
        sp.normal = n
    return sp


def test_material_groups_split_hues_not_lightness():
    lab = rgb_to_lab(np.array([[200, 160, 90], [110, 85, 45], [60, 45, 25],     # tan ramp
                               [160, 162, 170], [80, 82, 90]], np.float32))     # grey ramp
    g = material_groups(lab, np.array([50, 50, 20, 50, 50]))
    assert g[0] == g[1] == g[2] and g[3] == g[4] and g[0] != g[3]
    assert (material_groups(lab, np.array([50, 0, 0, 50, 0]))[[1, 2, 4]] == -1).all()


def test_hand_pixel_ramps_outline_and_frames_alike():
    first = two_materials()
    sprites = [first, two_materials(seed=1, pal=first.palette)]       # one palette, as rendered
    out = hand_pixel(sprites, "selout")
    assert len(out) == 2 and out[0].palette is out[1].palette             # one palette
    for o, s in zip(out, sprites):
        assert ((o.rgba[..., 3] > 0) == (o.indices >= 0)).all()
        assert ((o.rgba[..., 3] > 0) <= (s.rgba[..., 3] > 0)).all()       # never grows
        assert ((o.normal[..., 3] > 0) == (o.rgba[..., 3] > 0)).all()
        used = np.unique(o.indices[o.indices >= 0])
        assert len(used) < len(np.unique(s.indices[s.indices >= 0]))      # fewer colours
        assert len(used) <= 3 * (5 + 1) + 1                   # a few ramps (+ outline shades)
        # each half keeps its own hue: stone stays bluish grey, thatch warm
        lab = rgb_to_lab(o.rgba[..., :3].astype(np.float32))
        body = (o.rgba[..., 3] > 0) & ~boundary(o.rgba[..., 3] > 0)
        h = o.rgba.shape[1] // 2
        assert lab[:, :h][body[:, :h]][:, 2].mean() < 6 < lab[:, h:][body[:, h:]][:, 2].mean()
        # the outline is darker than the body it wraps
        ring = (o.rgba[..., 3] > 0) & ~body
        assert lab[ring][:, 0].mean() < lab[body][:, 0].mean() - 10
    # relight from the normals: within a material, the part turned to the light comes out
    # lighter, and the step between it and the part turned away grows
    turned = two_materials()
    n, mask = turned.normal, turned.rgba[..., 3] > 0
    top = np.zeros_like(mask)
    top[: mask.shape[0] // 2] = True
    n[mask & top] = (40, 128, 220, 255)                 # top half faces the light (left)
    n[mask & ~top] = (216, 128, 220, 255)               # bottom half faces away
    flat = hand_pixel([two_materials(normals=False)], "selout", smooth=0)[0]
    lit = hand_pixel([turned], "selout", relight=40, smooth=0)[0]
    lf = rgb_to_lab(flat.rgba[..., :3].astype(np.float32))[..., 0]
    ll = rgb_to_lab(lit.rgba[..., :3].astype(np.float32))[..., 0]
    q = (lit.rgba[..., 3] > 0) & ~boundary(lit.rgba[..., 3] > 0) & (flat.rgba[..., 3] > 0)
    gap_lit = ll[q & top].mean() - ll[q & ~top].mean()
    gap_flat = lf[q & top].mean() - lf[q & ~top].mean()
    assert gap_lit > gap_flat + 6 and ll[q & top].mean() > lf[q & top].mean()


def test_hand_pixel_hue_shift_and_palette():
    o = hand_pixel([two_materials()], "selout", relight=0)[0]
    lab = rgb_to_lab(o.palette.astype(np.float32))
    used = np.unique(o.indices[o.indices >= 0])
    warm = [i for i in used if lab[i, 2] > 10]                              # the thatch ramp
    light, dark = max(warm, key=lambda i: lab[i, 0]), min(warm, key=lambda i: lab[i, 0])
    hue = lambda i: np.degrees(np.arctan2(lab[i, 2], lab[i, 1])) % 360      # noqa: E731
    assert hue(light) > hue(dark)                     # lights toward yellow, shadows toward red/violet
    pico = hand_pixel([two_materials()], "selout", palette="pico-8")[0]
    from pixel_pipeline.palettes import load_palette
    allowed = {tuple(c) for c in load_palette("pico-8")}
    assert {tuple(c) for c in pico.rgba[pico.rgba[..., 3] > 0][:, :3]} <= allowed
    # dither is off by default; on, pixels half way between two steps alternate
    d = hand_pixel([two_materials()], "selout", relight=0, dither=True)[0]
    assert d.rgba.shape == o.rgba.shape and not np.array_equal(d.rgba, o.rgba)


def test_clean_sprite_default_unchanged_by_reshade_option():
    sp = disc_sprite(spur=True)
    a, b = clean_sprite(sp, "selout"), clean_sprite(sp, "selout", reshade=True)
    assert np.array_equal(a.rgba, b.rgba)
    c = clean_sprite(sp, "selout", reshade=False)
    assert ((c.rgba[..., 3] > 0) == (a.rgba[..., 3] > 0)).all()

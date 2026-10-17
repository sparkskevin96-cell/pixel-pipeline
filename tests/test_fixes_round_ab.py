"""Fixes round (2026-10-17): relit back views, 3D model checks, real animal sizes, action camera,
8-way facings, tile details at 64 px, leg proportions, action balance."""
from __future__ import annotations

import io

import numpy as np
import pytest

from pixel_pipeline import dirquality as dq
from pixel_pipeline.assets import Style, pixel_grid, style_for_type


# --------------------------------------------------------------------------- camera (4)

def test_action_view_shows_about_eleven_tiles_whole_scale():
    g = pixel_grid(Style(tile=64, project_type="topdown-rpg"))
    assert g["scale"] == 3 and g["base"] == [640, 360]          # 10 x 5.6 tiles (was 15 x 8.4)
    g = pixel_grid(style_for_type("topdown-rpg", "detailed"))    # 32 px tiles
    assert g["scale"] == 5 and g["base"][0] / 32 == 12
    assert pixel_grid(style_for_type("hd2d", "detailed"))["scale"] == 2   # HD-2D unchanged
    assert pixel_grid(style_for_type("side-scroller"))["scale"] == 6
    assert pixel_grid(Style(tile=64, project_type="topdown-rpg", pixel_scale=2))["scale"] == 2


# --------------------------------------------------------------------------- back relight (1)

def _view(H=40, W=30):
    from pixel_pipeline.drawnrig import RestView
    mask = np.zeros((H, W), bool)
    mask[5:35, 8:22] = True
    yy, xx = np.mgrid[0:H, 0:W]
    nx = (xx - 15) / 8.0
    normal = np.dstack([np.clip(nx, -1, 1), np.zeros((H, W)), np.sqrt(np.clip(1 - nx ** 2, 0, 1))])
    return RestView(mask, np.zeros((H, W, 3)), normal, np.zeros((H, W, 1)), np.zeros((H, W)),
                    np.zeros(0, int), np.zeros(0, bool))


def test_relight_opens_up_a_black_back_view_keeping_its_hue():
    from pixel_pipeline.color import rgb_to_lab
    from pixel_pipeline.drawnrig import Camera, relight_strength, relight_view
    v = _view()
    rgb = np.zeros(v.mask.shape + (3,), np.uint8)
    rgb[v.mask] = (40, 12, 14)                                  # a near-black red cloak
    rgb[5:35, 8:12] = (12, 12, 14)                              # and a black half
    cam = Camera(270.0, 30.0, 10.0, 15.0, 30.0, 30, 40)
    assert relight_strength(90.0, 90.0) == 0.0 and relight_strength(270.0, 90.0) == 1.0
    assert 0.7 < relight_strength(0.0, 90.0) < 0.9                       # a side view
    out, st = relight_view(rgb, v.mask, v, cam, ref_L=35.0, strength=1.0, ref_chroma=40.0)
    L0 = rgb_to_lab(rgb[v.mask].astype(np.float32))[:, 0]
    L1 = rgb_to_lab(out[v.mask].astype(np.float32))[:, 0]
    assert np.median(L1) > np.median(L0) + 5                             # lifted
    assert (L1 < 18).mean() < (L0 < 18).mean()                          # less near-black mass
    lab = rgb_to_lab(out[15:25, 14:20].reshape(-1, 3).astype(np.float32))
    assert (lab[:, 1] > 8).all()                                         # still red
    assert np.ptp(L1) >= np.ptp(L0)                                      # form shading added
    same, _ = relight_view(rgb, v.mask, v, cam, 35.0, 0.0)
    assert (same == rgb).all()                                           # front views untouched


def test_dirquality_judges_side_views_with_the_turned_away_ones_and_dark_materials_are_not_mud():
    assert dq.side_of("W", "S") == "back" and dq.side_of("SW", "S") == "front"
    assert dq.side_of("NE", "S") == "back" and dq.side_of("SE") == "front"
    purple = np.zeros((30, 20, 4), np.uint8)
    purple[2:28, 3:17] = (40, 20, 60, 255)                      # a deep purple robe: L < 18, coloured
    assert dq.blotch(purple, 0.0) == 0.0
    black = purple.copy()
    black[2:28, 3:17, :3] = (14, 14, 16)
    assert dq.blotch(black, 0.0) > 0.5
    a = np.zeros((10, 10, 4), np.uint8)
    a[..., 3] = 255
    a[..., :3] = (120, 30, 30)
    b = a.copy()
    b[..., :3] = (170, 60, 60)                                  # a lighter shade of the same red
    assert dq.foreign(b, a) == 0.0


# --------------------------------------------------------------------------- model checks (2)

def _figure_with_flyaways():
    import trimesh
    body = trimesh.creation.box((0.3, 1.6, 0.25))
    body.apply_translation((0, 0.8, 0))
    head = trimesh.creation.icosphere(2, 0.14)
    head.apply_translation((0, 1.75, 0))
    floater = trimesh.creation.icosphere(1, 0.05)
    floater.apply_translation((0.7, 1.4, 0.4))                 # a bow piece floating off the body
    # a thin ribbon arc flying out over the shoulder (a torus segment, ~1 cm thick)
    arc = trimesh.creation.torus(0.5, 0.012)                    # a vertical ring (axis z)
    arc.apply_translation((0.35, 1.6, 0.0))
    spear = trimesh.creation.cylinder(0.015, 1.4)
    spear.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, (0, 0, 1)))
    spear.apply_translation((0.0, 0.6, 0.2))                    # a straight held spear: kept
    return trimesh.util.concatenate([body, head, floater, arc, spear])


def test_shape_check_drops_floaters_and_ribbon_arcs_keeps_a_spear():
    import trimesh
    from pixel_pipeline import modelcheck
    m = _figure_with_flyaways()
    glb = m.export(file_type="glb")
    fixed, rep = modelcheck.fix_shape(glb)
    assert rep["floaters"] >= 1
    assert any(not r["kept"] for r in rep["ribbons"])           # the arc goes
    assert any(r["kept"] for r in rep["ribbons"])               # the spear stays
    out = trimesh.load(io.BytesIO(fixed), file_type="glb", force="mesh")
    hi = out.vertices[out.vertices[:, 1] > 1.75]
    assert hi[:, 0].max() < 0.3                                  # the arc out over the shoulder is gone
    assert out.bounds[1][2] > 0.8 and out.bounds[0][2] < -0.4    # the spear (along z) still reaches out
    assert 0 < rep["trimmed"] < 0.6
    clean, rep2 = modelcheck.fix_shape(trimesh.creation.box((0.3, 1.6, 0.25)).export(file_type="glb"))
    assert rep2["floaters"] == 0 and rep2["trimmed"] == 0.0 and modelcheck.score(rep2) == 0.0
    _, rep3 = modelcheck.fix_shape(glb, ribbons=False)          # the words asked for ribbons
    assert rep3["trimmed"] < rep["trimmed"]


def test_picture_check_flags_a_flying_ribbon_not_a_plain_figure():
    from pixel_pipeline import modelcheck
    a = np.zeros((256, 256, 4), np.uint8)
    a[40:250, 100:156, 3] = 255                                 # a standing figure
    plain = modelcheck.check_picture(a)
    assert plain["bad"] is None
    yy, xx = np.mgrid[0:256, 0:256]
    r = np.hypot(yy - 60, xx - 200)
    a[(np.abs(r - 50) < 2) & (yy < 90), 3] = 255                # a thin arc flying off the shoulder
    assert modelcheck.check_picture(a)["bad"]


def test_picture_prompt_keeps_people_compact_unless_asked():
    from pixel_pipeline.model3d import flyaway_negative, model_prompt
    pos, neg = model_prompt("an elf archer in green", "person", style=False)
    assert "compact silhouette" in pos and "flowing ribbons" in neg and "floating objects" in neg
    _, neg2 = model_prompt("a dancer with flowing ribbons", "person", style=False)
    assert "flowing ribbons" not in neg2
    assert "windswept cape" not in flyaway_negative("a hero in a windswept cape")


# --------------------------------------------------------------------------- animal sizes (3)

def test_animal_heights_against_a_person():
    from pixel_pipeline.library import animal_height
    assert animal_height("a grey wolf") == 0.5
    assert animal_height("Big Bear") > animal_height("wolf") > animal_height("a fox") > animal_height("cat")
    assert animal_height("a horse") > 1.0
    assert animal_height("wolf", "large") > animal_height("wolf")
    assert animal_height("some creature") == 0.45


def test_px_scale_prefers_the_sheet_meta():
    from pixel_pipeline.motion import px_scale
    assert px_scale(100, "animal", "medium", {"px_scale": 2.0}) == 2.0
    assert px_scale(68, "animal", "medium") == 2.0


def test_remap_drawing_places_a_drawing_at_another_scale():
    from PIL import Image
    from pixel_pipeline.drawnrig import Camera, remap_drawing
    old = Camera(0.0, 30.0, 20.0, 50.0, 40.0, 100, 50)
    new = Camera(0.0, 30.0, 30.0, 60.0, 70.0, 120, 80)
    pic = Image.new("RGB", (320, 320), "white")
    pic.paste((200, 0, 0), (150, 150, 170, 170))                 # a dot at frame px (x0 + 15..17)
    canvas, (x0, y0, size), kp = remap_drawing(pic, (40, 20, 32), old, new)
    assert (size * kp) % 8 == 0 and canvas.size == (size * kp, size * kp)
    a = np.asarray(canvas)
    ys, xs = np.nonzero((a[..., 0] > 150) & (a[..., 1] < 80))
    # the dot's centre in new frame px = new.ox + (old px - old.ox) * 1.5
    cx = x0 + xs.mean() / kp
    want = new.ox + (40 + 160 / 10 - old.ox) * 1.5
    assert abs(cx - want) < 0.6


# --------------------------------------------------------------------------- tiles (6)

def test_64px_tiles_draw_details_at_the_tile_scale():
    from pixel_pipeline import tilepaint as tp
    assert tp.detail_scale(32) == 1 and tp.detail_scale(64) == 2 and tp.detail_scale(16) == 1
    a = np.random.default_rng(0).integers(0, 3, (16, 16))
    b = tp.scale2x(a)
    assert b.shape == (32, 32) and set(np.unique(b)) <= set(np.unique(a))   # no new values
    big = tp.drawn_tile("grass", 64, 3)
    small = tp.drawn_tile("grass", 32, 3)
    assert big.shape == (64, 64, 4)
    # seamless: wrapping columns / rows match as well as inside ones
    edge = np.abs(big[:, 0, :3].astype(int) - big[:, -1, :3]).mean()
    inner = np.abs(big[:, 31, :3].astype(int) - big[:, 32, :3]).mean()
    assert edge < inner * 2 + 8

    def blobs(t):                 # mean size of single-colour pieces = detail size
        import cv2
        key = (t[..., 0].astype(int) << 16) | (t[..., 1].astype(int) << 8) | t[..., 2]
        sizes = []
        for v in np.unique(key):
            n, _, st, _ = cv2.connectedComponentsWithStats((key == v).astype(np.uint8), connectivity=4)
            sizes += list(st[1:, cv2.CC_STAT_AREA])
        sizes = np.array(sizes)
        return np.mean(sizes[(sizes >= 2) & (sizes < 200)])
    assert blobs(big) >= 1.5 * blobs(small) or blobs(big) >= 3


# --------------------------------------------------------------------------- strides (7)

def test_hips_follow_the_proportions_when_a_robe_hides_the_crotch():
    from pixel_pipeline import autorig as ar
    assert ar.heads_for("a dwarf warrior") == 4.0 and ar.heads_for("knight") is None
    assert ar.heads_for("", "chibi") == 2.5
    rng = np.random.default_rng(0)
    # a robed figure: a full-width robe to the floor, a narrower torso, a head
    H = 1.8
    pts = []
    for (x0, x1, y0, y1) in ((-0.25, 0.25, 0.0, 1.0), (-0.18, 0.18, 1.0, 1.5), (-0.1, 0.1, 1.5, 1.8),
                             (-0.55, -0.18, 1.35, 1.45), (0.18, 0.55, 1.35, 1.45)):
        n = 6000
        pts.append(np.c_[rng.uniform(x0, x1, n), rng.uniform(y0, y1, n), rng.uniform(-0.1, 0.1, n)])
    pts = np.vstack(pts)
    sk, _, _ = ar.find_skeleton(pts)
    assert sk.joints["hips"][1] >= 0.42 * sk.height              # not at the robe's hem
    mv = ar._Moves(sk)
    assert mv.gait_info(False)["stride"] / sk.height > 0.5        # a person's walk, not a shuffle
    sk4, _, _ = ar.find_skeleton(pts, heads=4.0)
    assert sk4.joints["hips"][1] < sk.joints["hips"][1]           # a dwarf's are lower

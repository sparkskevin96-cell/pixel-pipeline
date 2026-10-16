"""Tiles v3 (tilepaint.py): shapes drawn at the tile's scale, SDXL paint folded seamless, stones
re-drawn as pixel art, soft ground drawn in the swatch's colours; make_swatches' default methods
via a fake GPU (echo of the init canvas) and the fake ComfyUI."""
import json

import numpy as np
import pytest
from PIL import Image
from test_generate import comfy, white_key  # noqa: F401  (pytest fixtures)

from pixel_pipeline import assets, tilegen as tg, tilepaint as tp


def _wrap_jump(tex: np.ndarray) -> tuple[float, float]:
    a = np.asarray(tex)[..., :3].astype(int)
    seam = (np.abs(a[:, 0] - a[:, -1]).mean() + np.abs(a[0] - a[-1]).mean()) / 2
    inner = (np.abs(np.diff(a, axis=1)).mean() + np.abs(np.diff(a, axis=0)).mean()) / 2
    return seam, inner


def _painted(canvas: np.ndarray, seed: int = 0) -> np.ndarray:
    """A fake SDXL repaint of a periodic canvas: the same layout, each copy with its own noise
    and a soft shift, like four paintings of one tile."""
    rng = np.random.default_rng(seed)
    out = canvas.astype(np.float32) + rng.normal(0, 6, canvas.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def test_fold_wraps_exactly_without_ghosts():
    tile = tp.structure("road", 32, 3)
    canvas = tp.init_canvas(tile)
    assert canvas.shape == (1024, 1024, 3)
    assert (canvas[:512, :512] == canvas[512:, 512:]).all()          # periodic: 2 x 2 copies
    pic = tp.fold(_painted(canvas))
    seam, inner = _wrap_jump(pic)
    assert pic.shape == (512, 512, 3) and seam <= inner * 1.5 + 1
    # the copies are one layout, so the fold is that layout (no half-offset ghost)
    ref = canvas[:512, :512].astype(int)
    assert np.abs(pic.astype(int) - ref).mean() < 6
    four = tp.fold_canvas(_painted(tp.init_canvas(tile, 256, 4)), 4)
    assert four.shape == (256, 256, 3) and _wrap_jump(four)[0] <= _wrap_jump(four)[1] * 1.5 + 1


def test_area_downscale_has_no_direction():
    import cv2
    rng = np.random.default_rng(0)
    noise = cv2.GaussianBlur(rng.random((512, 512)).astype(np.float32), (0, 0), 6)
    noise = ((noise - noise.min()) / np.ptp(noise) * 200 + 30).astype(np.uint8)
    rgb = np.dstack([noise, (noise * 0.8).astype(np.uint8), (noise * 0.5).astype(np.uint8)])
    tile = tp.finish_tile(rgb, 32, "dirt")
    skew_x, skew_y, xy = tg._grain(tile)
    assert abs(xy) < 0.15 and abs(skew_x) < 0.2 and abs(skew_y) < 0.2
    assert len(np.unique(tile[..., :3].reshape(-1, 3), axis=0)) <= tp.RAMP_N["dirt"]
    big = np.asarray(Image.fromarray(np.tile(tile[..., :3], (4, 4, 1))).resize((512, 512), Image.NEAREST))
    assert tg.pattern_stats(big)["rows"] < 0.15                      # no streaks


@pytest.mark.parametrize("t", [16, 32, 48])
def test_drawn_terrains_wrap_and_stay_clean(t):
    for terrain in ("grass", "dirt", "sand", "snow", "stone", "fence", "wall", "water"):
        tex = tp.drawn_tile(terrain, t, seed=2)
        assert tex.shape == (t, t, 4) and (tex[..., 3] == 255).all()
        if terrain in ("grass", "dirt", "sand", "snow", "water"):   # slabs / boards / bricks end on the tile edge by design
            seam, inner = _wrap_jump(tex)
            assert seam <= inner * 2.2 + 3, terrain
        if terrain in tp.SOFT:
            assert len(np.unique(tex[..., :3].reshape(-1, 3), axis=0)) <= tp.SOFT[terrain][2]
    from pixel_pipeline.color import rgb_to_lab
    snow = tp.drawn_tile("snow", t, 1)
    L = rgb_to_lab(snow[..., :3].reshape(-1, 3).astype(np.float32))[:, 0]
    assert L.min() > 55                                              # no dark ground in snow
    # colours follow the swatch (red desert sand), shapes stay the drawn ones
    red = np.zeros((64, 64, 3), np.uint8)
    red[:] = (190, 90, 60)
    a, b = tp.drawn_tile("sand", t, 4), tp.drawn_tile("sand", t, 4, picture=red)
    assert b[..., 0].mean() - b[..., 2].mean() > a[..., 0].mean() - a[..., 2].mean()
    la, lb = (rgb_to_lab(x[..., :3].reshape(-1, 3).astype(np.float32))[:, 0] for x in (a, b))
    assert np.corrcoef(la, lb)[0, 1] > 0.8


def test_drawn_variants_share_the_ramp_and_differ_in_layout():
    tex = tp.drawn_tile("grass", 32, 5)
    vs = tp.drawn_variants("grass", tex, 5, 3, tp.drawn_ramp("grass"))
    cols = {tuple(c) for c in tex[..., :3].reshape(-1, 3).tolist()}
    assert len(vs) == 3
    for v in vs:
        assert {tuple(c) for c in v[..., :3].reshape(-1, 3).tolist()} <= cols
        assert (v[..., :3] != tex[..., :3]).any(-1).mean() > 0.15        # mostly plain lawn: details move
    assert tp.drawn_variants("water", tp.drawn_tile("water", 32, 1), 1) == []


def _cobble_painting(seed: int = 0, size: int = 512) -> np.ndarray:
    """A periodic painting of rounded grey stones in dark joints, lit from the upper left."""
    import cv2
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), 45, np.uint8)
    pts = tp.scatter(size, 30, rng, size / 6.5)
    for y, x in pts:
        r = int(rng.integers(size // 14, size // 10))
        g = int(rng.integers(110, 170))
        for dy in (-size, 0, size):
            for dx in (-size, 0, size):
                cv2.circle(img, (x + dx, y + dy), r, (g, g - 5, g - 12), -1, cv2.LINE_AA)
                cv2.circle(img, (x + dx - r // 3, y + dy - r // 3), r // 3, (g + 40, g + 35, g + 25), -1,
                           cv2.LINE_AA)
    return img


def test_stones_are_redrawn_as_clean_pixel_shapes():
    pic = _cobble_painting(1)
    tile = tp.stone_tile(pic, 32, "road")
    ids, joint = tp.segment_stones(pic, 32, "road")
    assert tile.shape == (32, 32, 4) and 0.08 < joint.mean() < 0.6
    n = len(np.unique(ids[~joint]))
    assert 8 <= n <= 80                                              # stones, not noise or one blob
    seam, inner = _wrap_jump(tile)
    assert seam <= inner * 2 + 3
    st = tg.tile_stats(tile)
    assert st["blob"] >= tg.STRUCTURE_MIN["road"] and tg.tile_miss("road", tile) == 0
    # a lit rim: the stone pixels just below / right of a joint are lighter than the ones above it
    from pixel_pipeline.color import rgb_to_lab
    L = rgb_to_lab(tile[..., :3].reshape(-1, 3).astype(np.float32))[:, 0].reshape(32, 32)
    lit = ~joint & (np.roll(joint, 1, 0) | np.roll(joint, 1, 1))
    dark = ~joint & (np.roll(joint, -1, 0) | np.roll(joint, -1, 1)) & ~lit
    assert L[lit].mean() > L[dark].mean() + 3 and L[joint].mean() < L[~joint].mean() - 10
    small = tp.stone_tile(pic, 16, "road")
    assert small.shape == (16, 16, 4) and _wrap_jump(small)[0] <= _wrap_jump(small)[1] * 2 + 3


def _echo_gpu(calls):
    """generate_images stand-in: img2img returns a 'painting' of its init (cobbles drawn over the
    canvas layout for stony terrains), txt2img a flat swatch in the terrain's colour."""
    def fake(prompt, negative="", *, seed=None, count=1, init_image=None, denoise=None, **kw):
        calls.append({"prompt": prompt, "seed": seed, "init": init_image is not None,
                      "denoise": denoise, **{k: kw[k] for k in ("style_lora", "style_lora_strength") if k in kw}})
        if init_image is not None:
            a = np.asarray(init_image)[..., :3]
            return [(Image.fromarray(_painted(a, seed or 0)), seed)]
        col = (60, 140, 50) if "grass" in prompt else (190, 160, 110)
        return [(Image.new("RGB", (256, 256), col), (seed or 0) + i) for i in range(count)]
    return fake


def test_make_swatches_paints_stones_and_draws_soft_ground(monkeypatch):
    calls = []
    monkeypatch.setattr("pixel_pipeline.generate.generate_images", _echo_gpu(calls))
    st = assets.Style(tile=32)
    st.style_lora = "house.safetensors"
    road, = tg.make_swatches("road", count=1, seed=7, style=st)
    assert road.method == "paint" and len(road.sources) == 4 and road.variants_raw is not None
    assert [c["init"] for c in calls] == [True, True]                # the paint + the variants pass
    assert calls[0]["denoise"] == tp.DENOISE["road"]
    assert calls[0]["style_lora"] == "house.safetensors"
    assert calls[0]["style_lora_strength"] == tp.TILE_STYLE_STRENGTH
    assert road.picture.shape == (512, 512, 3) and road.raw.shape == (1024, 1024, 3)
    assert _wrap_jump(road.texture)[0] <= _wrap_jump(road.texture)[1] * 2 + 3
    calls.clear()
    grass, = tg.make_swatches("grass", count=1, seed=7, style=st)
    assert grass.method == "drawn" and len(calls) == 1 and not calls[0]["init"]
    assert len(grass.sources) == 3 and grass.texture[..., 1].mean() > grass.texture[..., 0].mean()
    calls.clear()
    old, = tg.make_swatches("grass", count=1, seed=7, style=st, method="swatch", retries=0)
    assert old.method == "swatch"
    side = assets.style_for_type("side-scroller")
    assert tg.tile_method("grass", tg.camera_for(side)) == "swatch"
    assert tg.tile_method("cliff") == "paint" and tg.tile_method("fence") == "drawn"
    with pytest.raises(ValueError):
        tg.tile_method("grass", method="nope")


def test_a_painting_that_misses_falls_back_to_the_drawn_layout(monkeypatch):
    calls = []

    def green(prompt, negative="", *, seed=None, count=1, init_image=None, **kw):
        calls.append(seed)
        return [(Image.new("RGB", (1024, 1024), (40, 160, 40)), seed)]   # a lawn, not a road
    monkeypatch.setattr("pixel_pipeline.generate.generate_images", green)
    said = []
    sw, = tg.make_swatches("road", count=1, seed=3, style=assets.Style(tile=16), retries=1,
                           say=said.append)
    assert len(calls) == 2 and sw.method == "neo" and len(sw.sources) == 3
    assert any("drawn layout" in s for s in said)
    assert sw.texture[..., 1].mean() < sw.texture[..., 0].mean() + 25        # grey stones


def test_sets_vary_by_layout_and_keep_edges(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("pixel_pipeline.generate.generate_images", _echo_gpu(calls))
    st = assets.Style(tile=32)
    for terrain in ("road", "dirt"):
        sw, = tg.make_swatches(terrain, count=1, seed=11, style=st)
        gs = tg.build_tileset(terrain, sw.texture, picture=sw.picture, seed=sw.seed, style=st,
                              sources=sw.sources)
        xy, w = gs.choices()[(1, 1, 1, 1)]
        tiles = [gs.atlas[r * 32:(r + 1) * 32, c * 32:(c + 1) * 32, :3].astype(int) for c, r in xy]
        assert len(tiles) >= 4
        base = tiles[0]
        for v in tiles[1:]:                       # the 1 px border band is the base's
            assert (v[0] == base[0]).all() and (v[:, 0] == base[:, 0]).all()
            assert (v[-1] == base[-1]).all() and (v[:, -1] == base[:, -1]).all()
        inner = [t[4:-4, 4:-4] for t in tiles]
        assert np.mean([(a != inner[0]).any(-1).mean() for a in inner[1:]]) > 0.25
        # fewer visible repeats than one tile copied
        filled = tg.fill_area([np.dstack([t.astype(np.uint8), np.full((32, 32), 255, np.uint8)])
                               for t in tiles], w, 12)
        one = np.tile(np.dstack([base.astype(np.uint8), np.full((32, 32), 255, np.uint8)]), (12, 12, 1))
        assert tg.repeat_score(filled, 32) < tg.repeat_score(one, 32) - 0.2


def test_saved_v3_sets_rebuild_on_the_cpu(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("pixel_pipeline.generate.generate_images", _echo_gpu(calls))
    proj = assets.create_project("Tiles v3", type="topdown-rpg")
    st = proj.style
    for terrain in ("road", "sand"):
        sw, = tg.make_swatches(terrain, count=1, seed=4, style=st)
        gs = tg.build_tileset(terrain, sw.texture, picture=sw.picture, seed=sw.seed, style=st,
                              project=proj, sources=sw.sources)
        a = tg.save_tileset(gs, f"{terrain} v3", project=proj, swatch=sw)
        assert a.extra["method"] == sw.method
        again = tg.rebuild_tileset(a)
        assert (again.texture == sw.texture).all()
        assert again.atlas.shape == gs.atlas.shape
    assert (tg.Path(a.folder) / "swatch.png").exists()


def test_no_gpu_builtins_are_drawn():
    for terrain in ("grass", "dirt", "sand", "snow", "water", "stone"):
        sw = tg.builtin_swatch(terrain, 16)
        assert sw.method == "drawn" and sw.texture.shape == (16, 16, 4)
    sw = tg.builtin_swatch("road", 16)
    assert sw.method == "neo" and len(sw.sources) == 3
    old = tg.builtin_swatch("grass", 16, even=False)
    assert old.method == "swatch"


def test_fake_comfyui_gets_an_img2img_paint(comfy):
    fake, client = comfy
    st = assets.Style(tile=16)
    tg.make_swatches("road", count=1, client=client, seed=2, style=st, style_lora="", retries=0)
    wf = fake.prompts[0]
    k = next(n for n in wf.values() if n["class_type"] == "KSampler")["inputs"]
    assert abs(k["denoise"] - tp.DENOISE["road"]) < 1e-6
    assert any(n["class_type"] == "VAEEncode" for n in wf.values())
    text = json.dumps(wf)
    assert "cobble" in text and "top-down" in text

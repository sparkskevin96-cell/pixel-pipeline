"""Tile weak spots, round 2 (2026-10-16): words shape drawn ground (tilewords), stone_tile splits
merged / odd stones, cliff + board walls as boulder faces, bright tops capped on the HD-2D board
(terrain shader), makeany's GPU estimate for painted tiles (2 pictures per swatch). Fake GPU and
fake ComfyUI only."""
import json

import numpy as np
import pytest
from PIL import Image
from test_generate import comfy, white_key  # noqa: F401  (pytest fixtures)

from pixel_pipeline import assets, makeany, neo_terrain as nt, tilegen as tg, tilepaint as tp
from pixel_pipeline.color import rgb_to_lab
from pixel_pipeline.tilewords import PLAIN, details


# --------------------------------------------------------------------------- words -> shapes

@pytest.mark.parametrize("terrain, words, want", [
    ("dirt", "mossy dirt path", {"moss": 1.0}),
    ("grass", "flower meadow grass", {"flowers": 1.0}),
    ("grass", "tall grass", {"tall": 1.0}),
    ("grass", "autumn leaves", {"leaves": 1.0}),
    ("dirt", "cracked dry earth", {"cracks": 1.0}),
    ("dirt", "muddy road with puddles", {"puddles": 1.0}),
    ("dirt", "rocky pebbly dirt", {"big_pebbles": True}),
    ("sand", "rippled dune sand", {"ripples": 2.0, "dunes": True}),
    ("snow", "fresh snow with footprints", {"fresh": True, "footprints": True}),
    ("stone", "mossy cracked stone floor", {"moss": 1.0, "cracks": 1.0}),
])
def test_words_become_shape_details(terrain, words, want):
    d = details(terrain, words)
    for k, v in want.items():
        assert getattr(d, k) == v, (k, d)
    assert d.words


def test_words_a_terrain_cannot_show_are_ignored():
    assert details("snow", "mossy snow").plain()           # no moss on snow
    assert details("grass", "").plain() and details("grass", "bright green") == PLAIN
    assert details("water", "mossy").plain()                # water is not drawn by these painters
    assert details("grass", "short grass").tufts < 1.0
    assert not details("grass", "dryad").cracks             # whole words only


def _share(tex, cond):
    return float(cond(tex[..., :3].astype(int)).mean())


def test_drawn_tiles_take_the_words_shapes_and_still_wrap():
    for terrain in ("grass", "dirt", "sand", "snow", "stone"):
        plain = tp.drawn_tile(terrain, 32, 3)
        assert (tp.drawn_tile(terrain, 32, 3, words="") == plain).all()        # plain = as before
        assert (tp.drawn_tile(terrain, 32, 3, words="bright") == plain).all()
    grass = tp.drawn_tile("grass", 32, 3)
    meadow = tp.drawn_tile("grass", 32, 3, words="flower meadow")
    ramp = {tuple(c) for c in grass[..., :3].reshape(-1, 3)}
    new = [tuple(c) for c in meadow[..., :3].reshape(-1, 3) if tuple(c) not in ramp]
    assert len(new) >= 12                                                    # petals + eyes
    dirt = tp.drawn_tile("dirt", 32, 3)
    dark = lambda a: rgb_to_lab(a.reshape(-1, 3).astype(np.float32))[:, 0] < 22
    cracked = tp.drawn_tile("dirt", 32, 3, words="cracked dry earth")
    assert dark(cracked[..., :3]).mean() > dark(dirt[..., :3]).mean() + 0.05
    puddles = tp.drawn_tile("dirt", 32, 3, words="muddy puddles")
    blue = lambda a: (a[..., 2] > a[..., 0] + 30)
    assert _share(puddles, blue) > 0.02 and _share(dirt, blue) == 0
    mossy = tp.drawn_tile("dirt", 32, 3, words="mossy")
    green = lambda a: (a[..., 1] > a[..., 0] + 15) & (a[..., 1] > a[..., 2])
    assert _share(mossy, green) > 0.03 and _share(dirt, green) < 0.005
    leaves = tp.drawn_tile("grass", 32, 3, words="autumn leaves")
    orange = lambda a: (a[..., 0] > a[..., 1] + 40)
    assert _share(leaves, orange) > 0.02
    # index-level details: longer ripples, a trail of prints, taller blades
    assert (tp.sand_idx(32, 3, details("sand", "rippled dune")) == 3).sum() > (tp.sand_idx(32, 3) == 3).sum()
    assert (tp.snow_idx(32, 3, details("snow", "footprints")) == 0).sum() > (tp.snow_idx(32, 3) == 0).sum() + 20
    assert (tp.grass_idx(32, 3, details("grass", "tall grass")) == 5).sum() > (tp.grass_idx(32, 3) == 5).sum()
    for words in ("flower meadow", "muddy puddles", "cracked", "tall grass", "autumn leaves"):
        tex = tp.drawn_tile("grass" if "grass" in words or "flower" in words or "leaves" in words else "dirt",
                            32, 5, words=words)
        big = np.tile(tex, (2, 2, 1))[..., :3].astype(int)               # the wrap is a real seam
        seam = np.abs(big[:, 31] - big[:, 32]).mean()
        inner = np.abs(np.diff(big[:, 8:24], axis=1)).mean()
        assert seam <= inner * 2.5 + 4, words


def test_words_go_through_swatches_saving_and_rebuilds(monkeypatch):
    def fake(prompt, negative="", *, seed=None, count=1, init_image=None, **kw):
        return [(Image.new("RGB", (256, 256), (60, 140, 50)), (seed or 0) + i) for i in range(count)]
    monkeypatch.setattr("pixel_pipeline.generate.generate_images", fake)
    proj = assets.create_project("Round 2 words", type="topdown-rpg")
    st = proj.style
    sw, = tg.make_swatches("grass", "flower meadow grass", count=1, seed=4, style=st)
    assert sw.method == "drawn" and sw.words == "flower meadow grass"
    plain, = tg.make_swatches("grass", "", count=1, seed=4, style=st)
    assert (sw.texture != plain.texture).any(-1).mean() > 0.03
    gs = tg.build_tileset("grass", sw.texture, picture=sw.picture, seed=sw.seed, style=st,
                          project=proj, sources=sw.sources)
    a = tg.save_tileset(gs, "meadow", project=proj, swatch=sw)
    assert a.extra["words"] == "flower meadow grass"
    again = tg.rebuild_tileset(a)
    assert (again.texture == sw.texture).all()
    built = tg.builtin_swatch("dirt", 32, 1, words="cracked dry earth")
    assert built.words and (built.texture != tg.builtin_swatch("dirt", 32, 1).texture).any()


def test_fake_comfyui_swatch_prompt_keeps_the_words(comfy):
    fake, client = comfy
    sw, = tg.make_swatches("dirt", "mossy dirt path", count=1, client=client, seed=2,
                           style=assets.Style(tile=16), style_lora="", retries=0)
    assert "mossy" in json.dumps(fake.prompts[0]) and sw.words == "mossy dirt path"
    assert sw.texture.shape == (16, 16, 4)


# --------------------------------------------------------------------------- stones

def test_odd_stones_split_along_a_new_joint():
    t = 32
    stones = np.zeros((t, t), int)
    stones[4:10, 4:26] = 1                     # an L: a long bar ...
    stones[10:24, 4:10] = 1                    # ... and its leg
    stones[14:22, 14:22] = 2                   # a fine square stone
    gap = stones == 0
    n, solid, aspect = tp.stone_shape(*np.nonzero(stones == 1), t)
    assert solid < tp.SOLIDITY_MIN
    out, gap2 = tp.split_stones(stones, gap, t, "road")
    assert out.max() >= 3                      # the L became two stones, the square stayed
    sq = out[18, 18]
    assert sq > 0 and (out == sq).sum() == 64
    for v in range(1, out.max() + 1):
        ys, xs = np.nonzero(out == v)
        _, s, asp = tp.stone_shape(ys, xs, t)
        assert s >= 0.7 and asp <= 3.0
    # different stones never touch: a joint between every pair
    right, down = np.roll(out, -1, 1), np.roll(out, -1, 0)
    assert not ((out > 0) & (right > 0) & (out != right)).any()
    assert not ((out > 0) & (down > 0) & (out != down)).any()


def test_stone_shape_handles_stones_across_the_tile_edge():
    t = 32
    m = np.zeros((t, t), bool)
    m[10:16, 28:] = True
    m[10:16, :4] = True                        # one 8 x 6 stone cut by the edge
    n, solid, aspect = tp.stone_shape(*np.nonzero(m), t)
    assert n == 48 and solid > 0.95 and aspect < 1.6


def test_stone_tile_has_no_merged_monsters():
    rng = np.random.default_rng(0)
    canvas = tp.init_canvas(tp.structure("road", 32, 1), 256, 2, 0.5)
    pic = tp.fold(np.clip(canvas + rng.normal(0, 6, canvas.shape), 0, 255).astype(np.uint8))
    tile = tp.stone_tile(pic, 32, "road")
    assert tile.shape == (32, 32, 4)
    L = rgb_to_lab(tile[..., :3].reshape(-1, 3).astype(np.float32))[:, 0].reshape(32, 32)
    from scipy import ndimage as ndi
    lab, n = ndi.label(L > np.sort(np.unique(L))[1] + 0.1)
    sizes = np.bincount(lab.ravel())[1:]
    assert n >= 8 and sizes.max() <= tp.STONE_AREA["road"][1] * 32 * 32 * 1.6


# --------------------------------------------------------------------------- cliff faces

def test_boulder_faces_wrap_keep_seams_and_feed_cliff_sets():
    ramp = nt.RAMPS["rock"]
    face, height = tp.boulder_face(96, 80, ramp, 3, seams=(16, 48), unit=32)
    assert face.shape == (80, 96, 4) and height.shape == (80, 96)
    L = rgb_to_lab(face[..., :3].reshape(-1, 3).astype(np.float32))[:, 0].reshape(80, 96)
    joint = L <= rgb_to_lab(ramp[1:2].astype(np.float32))[0, 0] + 1.0       # the 2 joint colours
    for seam in (16, 48):                      # the row above a seam is joint all the way across
        assert joint[seam - 1].mean() > 0.9
    a = face[..., :3].astype(int)
    assert np.abs(a[:, 0] - a[:, -1]).mean() <= np.abs(np.diff(a, axis=1)).mean() * 2.5 + 4
    old, _ = nt.rock_face(96, 80, ramp, 3, seams=(16, 48), unit=32)
    assert (old != face).any()
    # cliff sets cut their strip from it (the blob + face tile layout is unchanged)
    st = assets.Style(tile=32)
    rock = tp.neo_tile("cliff", 32, 0)
    gs = tg.build_tileset("cliff", rock, seed=1, style=st)
    faces = [d for d in gs.tiles if "face" in d]
    assert len(faces) == len(tg.FACE_PARTS) * 4 * tg.FACE_PHASES
    tints = tp.stone_tints(rock)
    assert tints.shape == (3, 2)


# --------------------------------------------------------------------------- HD-2D board

def test_bright_tops_are_capped_and_spread_for_the_board():
    snow = tp.drawn_tile("snow", 32, 0)
    capped = nt.board_albedo(snow, "snow")
    L0 = rgb_to_lab(snow[..., :3].reshape(-1, 3).astype(np.float32))[:, 0]
    L1 = rgb_to_lab(capped[..., :3].reshape(-1, 3).astype(np.float32))[:, 0]
    assert np.percentile(L1, 98) <= nt.ALBEDO_CAP["snow"] + 1.0
    assert L1.std() > L0.std() * 1.3                         # the drawn steps spread apart
    lab1 = rgb_to_lab(capped[..., :3].reshape(-1, 3).astype(np.float32))
    dark = L1 < np.percentile(L1, 20)
    assert lab1[dark, 2].mean() < lab1[~dark, 2].mean()      # shadows lean blue
    grass = tp.drawn_tile("grass", 32, 0)
    assert (nt.board_albedo(grass, "grass") == grass).all()  # under the cap: untouched


def test_board_writes_the_terrain_shader_and_relief(tmp_path):
    from pixel_pipeline import hd2d
    from pixel_pipeline.diorama import harbour
    info = nt.write_board_textures(["grass", "snow", "sand"], tmp_path / "tex", 16)
    assert info["terrains"]["snow"]["relief"] == nt.BOARD_RELIEF["snow"]
    assert "terrain.gdshader" in hd2d.BOARD_GD and "terrain_bands" in hd2d.BOARD_GD
    assert "floor(d * bands + 0.5)" in hd2d.TERRAIN_SHADER
    assert hd2d.LIGHTING["noon"]["white"] > 1.0
    p = hd2d.write_lighting(tmp_path, "noon")
    assert json.loads(p.read_text())["terrain_bands"] == hd2d.TERRAIN_BANDS
    assert "dof_blur_near_distance = cam_dist - 7.0" in hd2d.BOARD_GD


# --------------------------------------------------------------------------- GPU estimate

def test_painted_tilesets_count_two_pictures_per_swatch():
    road = makeany.route("cobblestone road tiles")
    grass = makeany.route("grass tiles")
    assert road.kind == grass.kind == "tileset"
    n = int(road.params.get("count", 3))
    assert road.gpu_minutes() == pytest.approx(2 * n * makeany.GPU["paint_picture"])
    assert grass.gpu_minutes() == pytest.approx(int(grass.params.get("count", 3)) * makeany.GPU["tile"])
    assert makeany.tile_pictures(road) == (2 * n, makeany.GPU["paint_picture"])
    stages, _ = makeany.progress_plan(road)
    assert sum(s.minutes for s in stages) == pytest.approx(road.gpu_minutes() + 0.05 * n)
    side = assets.style_for_type("side-scroller")                 # side cameras keep 1 picture
    assert makeany.tile_pictures(road, side)[0] == n

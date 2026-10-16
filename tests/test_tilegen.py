"""Tile auto-variations (tilegen.py): GPU swatches via the fake ComfyUI, seamless textures,
corner / blob / line sets, Godot + Tiled + Unity files, the map export and the UI page."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from test_generate import comfy, white_key  # noqa: F401  (pytest fixtures)

from pixel_pipeline import assets, tilegen as tg

GODOT = os.environ.get("GODOT_BIN") or shutil.which("godot")


def _wrap_jump(tex: np.ndarray) -> tuple[float, float]:
    """(mean colour jump across the wrap seams, mean jump between interior neighbours)."""
    a = tex[..., :3].astype(int)
    seam = (np.abs(a[:, 0] - a[:, -1]).mean() + np.abs(a[0] - a[-1]).mean()) / 2
    inner = (np.abs(np.diff(a, axis=1)).mean() + np.abs(np.diff(a, axis=0)).mean()) / 2
    return seam, inner


def test_seamless_picture_wraps():
    yy, xx = np.mgrid[0:300, 0:300]
    rgb = np.stack([xx * 0.8, yy * 0.8, (xx + yy) * 0.4], -1).astype(np.uint8)   # no wrap at all
    out = tg.seamless(rgb, size=128)
    seam, inner = _wrap_jump(out)
    assert seam <= inner * 2 + 1
    tex = tg.pixel_texture(out, 16, assets.Style())
    assert tex.shape == (16, 16, 4) and (tex[..., 3] == 255).all()


def test_gpu_swatches_carry_the_house_style_and_wrap(comfy):
    fake, client = comfy
    proj = assets.create_project("Tiles", type="topdown-rpg")
    st = proj.style
    st.style_prompt = "moonlit watercolour pixels"
    proj.save_style(st)
    assets.set_current_project("Tiles")
    sw = tg.make_swatches("dirt", "mossy forest path", count=2, client=client, seed=5,
                          style=proj.style, style_lora="", retries=0)
    assert len(sw) == 2 and [s.seed for s in sw] == [5, 6]
    text = json.dumps(fake.prompts[0])
    assert "mossy forest path" in text and "moonlit watercolour pixels" in text and "top-down" in text
    for s in sw:
        assert s.texture.shape == (16, 16, 4)                     # the project's tile size
        seam, inner = _wrap_jump(s.texture)
        assert seam <= inner * 2 + 2
    # seam repaint on the GPU: an inpaint pass (mask) over the offset copy
    n = len(fake.prompts)
    tg.make_swatches("water", count=1, client=client, seed=1, style=proj.style, style_lora="",
                     inpaint_seams=True, retries=0, method="swatch")
    assert len(fake.prompts) == n + 2
    side = assets.style_for_type("side-scroller")
    pos, _ = tg.swatch_prompt("cliff", "", tg.camera_for(side), side)
    assert "side view" in pos and "cliff" in pos


def test_path_prompts_and_the_colour_check(comfy, monkeypatch):
    pos, neg = tg.swatch_prompt("dirt", "forest path")
    assert pos.index("packed dirt path") < pos.index("forest path")      # the material leads
    assert "leaves" in neg and "grass" in neg and ", forest," not in neg   # asked for: kept out of neg
    _, neg2 = tg.swatch_prompt("dirt", "mossy forest path")
    assert "moss," not in neg2
    brown = np.zeros((64, 64, 3), np.uint8)
    brown[:] = (125, 84, 52)
    green = brown.copy()
    green[:] = (40, 120, 40)
    assert tg.colour_miss("dirt", brown) == 0 and tg.colour_miss("dirt", green) > 0.5
    assert tg.colour_miss("water", green) > 0 and tg.colour_miss("water", np.full((8, 8, 3), (40, 90, 200), np.uint8)) == 0
    # a swatch that misses is drawn again (new seed, colour word), up to `retries` times
    fake, client = comfy
    calls = []

    def fake_generate(prompt, negative="", *, seed=None, count=1, **kw):
        from PIL import Image as I
        calls.append((prompt, seed))
        col = (125, 84, 52) if "(brown" in prompt and len(calls) >= 3 else (40, 120, 40)
        return [(I.new("RGB", (64, 64), col), (seed or 0) + i) for i in range(count)]
    monkeypatch.setattr("pixel_pipeline.generate.generate_images", fake_generate)
    sw, = tg.make_swatches("dirt", "forest path", count=1, seed=7, style=assets.Style(tile=16), retries=2)
    assert len(calls) == 3 and sw.colour_miss == 0 and sw.seed == 7 + 2000
    calls.clear()
    sw, = tg.make_swatches("dirt", count=1, seed=7, style=assets.Style(tile=16), retries=0)
    assert len(calls) == 1 and sw.colour_miss > 0


def test_palette_lock_snaps_every_tile():
    st = assets.Style(palette="pico-8", tile=16)
    sw = tg.builtin_swatch("water", 16, style=st)
    ts = tg.build_tileset("water", sw.texture, style=st, base=tg.builtin_swatch("grass", 16, style=st).texture)
    pal = {tuple(c) for c in tg.project_palette(st).tolist()}
    on = ts.atlas[..., 3] > 0
    assert {tuple(c) for c in ts.atlas[on][:, :3].tolist()} <= pal


def test_variants_keep_the_edges():
    tex = tg.builtin_swatch("road", 16).texture
    vs = tg.interior_variants(tex, 4, seed=2)
    assert len(vs) == 4
    for v in vs[1:]:
        for edge in (np.s_[0], np.s_[-1], np.s_[:, 0], np.s_[:, -1]):
            assert (v[edge] == tex[edge]).all()
        assert (v != tex).any()


def test_blob_tiles_agree_across_every_shared_edge():
    t = 16
    m, jx, jy = tg._blob_geometry(t, "water", 0)
    assert len(tg.BLOB_MASKS) == 47 and tg.BLOB_MASKS[-1] == 255
    rng = np.random.default_rng(3)
    on = rng.random((12, 12)) < 0.55
    region = np.zeros((12 * t, 12 * t), bool)
    for (y, x), v in np.ndenumerate(on):
        if v:
            region[y * t:(y + 1) * t, x * t:(x + 1) * t] = tg.blob_region(
                t, tg.neighbour_mask(on, y, x), m, jx, jy)
    for (y, x), v in np.ndenumerate(on[:, :-1]):                 # left | right
        if v and on[y, x + 1]:
            a = region[y * t:(y + 1) * t, (x + 1) * t - 1]
            b = region[y * t:(y + 1) * t, (x + 1) * t]
            assert (a == b).all(), (y, x)
    for (y, x), v in np.ndenumerate(on[:-1]):                    # top / bottom
        if v and on[y + 1, x]:
            a = region[(y + 1) * t - 1, x * t:(x + 1) * t]
            b = region[(y + 1) * t, x * t:(x + 1) * t]
            assert (a == b).all(), (y, x)


def test_cliff_faces_are_real_rock_walls(tmp_path):
    t = 16
    rock = tg.builtin_swatch("cliff", t).texture
    grass = tg.builtin_swatch("grass", t).texture
    gs = tg.build_tileset("cliff", rock, base=grass)
    ch = gs.choices()
    assert len(ch[255][0]) > 1 and all(m in ch for m in tg.BLOB_MASKS)          # the 47 + variants
    faces = [d for d in gs.tiles if d.get("face")]
    assert len(faces) == len(tg.FACE_PARTS) * 4 * tg.FACE_PHASES
    def tile(xy):
        c, r = xy
        return gs.atlas[r * t:(r + 1) * t, c * t:(c + 1) * t, :3].astype(int)
    # an open-south tile's face is ~40% of the tile and the wall tile below carries it a whole
    # tile further: their seam is a joint over a lit block top (strata run on)
    edge = tile(ch[255 & ~(tg.S | tg.SE | tg.SW)][0][0])
    mid = tile(tg.face_choice(ch, "mid", 3, 0))
    base = tile(tg.face_choice(ch, "base", 3, 0))
    bare = tg.build_tileset("cliff", rock, base=grass, overhang=False)          # no grass drips over it:
    c0, r0 = bare.choices()[255 & ~(tg.S | tg.SE | tg.SW)][0][0]                 # rock to the bottom
    edge0 = bare.atlas[r0 * t:(r0 + 1) * t, c0 * t:(c0 + 1) * t, :3].astype(int)
    assert (np.abs(edge0 - grass[..., :3]).sum(-1) > 30)[-3:, 3:-3].mean() > 0.75
    assert (np.abs(edge - grass[..., :3]).sum(-1) > 30)[-3:, 3:-3].mean() > 0.5   # drips + moss: some
    assert (np.abs(mid - grass[..., :3]).sum(-1) > 30).mean() > 0.8                # a full tile of rock
    assert edge[-1].sum(-1).mean() < mid[0].sum(-1).mean()                      # joint, then lit top
    assert (np.abs(base - grass[..., :3]).sum(-1) < 30)[-2:].mean() > 0.5        # ground at its foot
    # neighbouring phases are neighbouring pieces of one strip: no seam between wall tiles
    for p in range(tg.FACE_PHASES):
        a = tile(ch[("face", "mid", 3)][0][p])
        b = tile(ch[("face", "mid", 3)][0][(p + 1) % tg.FACE_PHASES])
        assert np.abs(a[:, -1] - b[:, 0]).mean() < np.abs(a[:, :-1] - a[:, 1:]).mean() * 3 + 10
    plain = tg.build_tileset("cliff", rock, base=grass, overhang=False)
    assert not (plain.atlas == gs.atlas).all() and plain.meta["overhang"] is False
    # the example map shows the walls; Godot / Tiled: face tiles are plain tiles, no terrain
    used = {tuple(c) for c in gs.example.reshape(-1, 2).tolist()}
    assert used & {(d["col"], d["row"]) for d in faces}
    tg.write_tileset(gs, tmp_path / "c")
    tres = (tmp_path / "c" / "tileset.tres").read_text(encoding="utf-8")
    d = faces[0]
    assert f"{d['col']}:{d['row']}/0 = 0" in tres and f"{d['col']}:{d['row']}/0/terrain" not in tres
    layout = json.loads((tmp_path / "c" / "tileset.json").read_text(encoding="utf-8"))
    assert tg.tile_choices(layout["tiles"])[("face", "foot", 0)][0] == ch[("face", "foot", 0)][0]


def test_line_set_connects_and_has_collision():
    tex = tg.builtin_swatch("fence", 16).texture
    ts = tg.build_tileset("fence", tex, base=tg.builtin_swatch("grass", 16).texture)
    assert ts.kind == "line" and ts.atlas.shape == (5 * 16, 4 * 16, 4)
    tile = {d["mask"]: ts.atlas[d["row"] * 16:(d["row"] + 1) * 16, d["col"] * 16:(d["col"] + 1) * 16]
            for d in ts.tiles if "mask" in d}
    ew = tile[tg.LE | tg.LW]
    assert (ew[:, 0, 3] > 0).any() and (ew[:, -1, 3] > 0).any()
    assert ((ew[:, 0, 3] > 0) == (ew[:, -1, 3] > 0)).all()       # rails line up with the neighbour
    assert not (tile[0][:, 0, 3] > 0).any()                      # lone post: nothing at the edges
    assert all(d["boxes"] for d in ts.tiles if "mask" in d)


def test_save_writes_godot_tiled_unity_and_lists(tmp_path):
    proj = assets.create_project("Shore", type="topdown-rpg")
    gs = tg.build_tileset("water", tg.builtin_swatch("water", 16).texture, project=proj)
    a = tg.save_tileset(gs, "Lagoon", project=proj)
    assert "tileset" in assets.KINDS
    assert [x.name for x in assets.list_assets("tileset", project=proj)] == ["Lagoon"]
    f = a.folder
    for name in ("tileset.png", "tileset.tres", "tileset.tsx", "tileset.json", "example.tmj",
                 "texture.png", "example.png", "sheet.png", "sheet.json", "thumb.png"):
        assert (f / name).exists(), name
    tres = (f / "tileset.tres").read_text(encoding="utf-8")
    assert "terrain_set_0/mode = 0" in tres and "terrains_peering_bit/bottom_right_corner = 1" in tres
    tsx = (f / "tileset.tsx").read_text(encoding="utf-8")
    assert 'type="mixed"' in tsx and tsx.count("<wangtile") == len(gs.tiles)
    from pixel_pipeline import exports                            # not a unit: a clear message
    with pytest.raises(ValueError, match="is a tileset"):         # (was KeyError: 0)
        exports.export_asset(a, "godot-hd2d", tmp_path / "hd2d")
    tmj = json.loads((f / "example.tmj").read_text(encoding="utf-8"))
    assert tmj["tilesets"][0]["source"] == "tileset.tsx" and max(tmj["layers"][0]["data"]) > 1
    from pixel_pipeline.exports import export_asset
    files = export_asset(a, "unity", tmp_path / "unity")
    meta = next(p for p in files if p.name.endswith(".png.meta")).read_text(encoding="utf-8")
    assert meta.count("spriteID") >= 47
    assert not (tmp_path / "unity" / "clips").exists()           # tiles, not animations


def test_side_and_iso_cameras(tmp_path):
    grass = tg.builtin_swatch("grass", 16).texture
    side = tg.build_tileset("dirt", tg.builtin_swatch("dirt", 16).texture, camera="side", base=grass)
    assert side.kind == "side" and side.atlas.shape[0] == 6 * 16             # + variants row
    files = tg.write_tileset(side, tmp_path / "side")
    tres = (tmp_path / "side" / "tileset.tres").read_text(encoding="utf-8")
    assert "terrain_set_0/mode = 2" in tres and "0:5/0/probability" in tres
    iso = tg.build_tileset("stone", tg.builtin_swatch("stone", 16).texture, camera="iso", base=grass)
    assert iso.kind == "iso" and iso.extra_atlas is not None
    assert (tmp_path / "iso" / "tileset_blocks.png") in tg.write_tileset(iso, tmp_path / "iso")
    assert files and tg.preview(iso).shape[2] == 4
    with pytest.raises(ValueError):
        tg.build_tileset("fence", grass, camera="iso", base=grass)


def test_map_export_paints_with_the_generated_texture(tmp_path):
    from pixel_pipeline import maplib
    proj = assets.create_project("Painted", type="topdown-rpg")
    assets.set_current_project("Painted")
    tex = np.zeros((16, 16, 4), np.uint8)
    tex[..., :3] = (250, 10, 200)                                  # unmistakable
    tex[::2, ::2, :3] = (240, 20, 190)
    tex[..., 3] = 255
    gs = tg.build_tileset("water", tex, project=proj, base=tg.builtin_swatch("grass", 16).texture)
    tg.save_tileset(gs, "Pink water", project=proj)
    assert set(tg.project_textures(proj, 16)) == {"water"}
    doc = maplib.new_map("harbour", seed=1)
    asset = maplib.save_map(doc, "Harbour", project=proj)
    maplib.export_map(asset, "godot-tilemap", tmp_path / "out")
    atlas = np.asarray(Image.open(tmp_path / "out" / "terrain.png").convert("RGB"))
    assert ((atlas == (250, 10, 200)).all(-1)).sum() > 50
    assert (tmp_path / "out" / "tilesets" / "pink-water" / "tileset.tres").exists()


def test_tilegen_tab(comfy, tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from conftest import qt_app
    qt_app()
    from pixel_pipeline.ui.tilegen_tab import TilegenTab
    fake, client = comfy
    assets.create_project("UI tiles", type="topdown-rpg")
    assets.set_current_project("UI tiles")
    tab = TilegenTab(client=client)
    tab.sync = True
    tab.terrain.setCurrentText("water")
    tab.words.setText("swamp water")
    tab.count.setValue(2)
    tab.make()
    assert tab.picks.count() == 2 and tab.view.rgba is not None and tab.save_btn.isEnabled()
    tab.picks.setCurrentRow(1)
    assert tab.tileset.kind == "blob"
    a = tab.save()
    assert a.kind == "tileset" and (a.folder / "tileset.tres").exists() and a.extra["words"] == "swamp water"
    tab.terrain.setCurrentText("fence")
    tab.quick()
    assert tab.tileset.kind == "line"


CHECK = """extends SceneTree
func paint(path, size, cells, terrain_set, terrain, base):
\tvar layer = TileMapLayer.new()
\tlayer.tile_set = load(path)
\troot.add_child(layer)
\tif base >= 0:
\t\tvar all: Array[Vector2i] = []
\t\tfor y in size:
\t\t\tfor x in size:
\t\t\t\tall.append(Vector2i(x, y))
\t\tlayer.set_cells_terrain_connect(all, terrain_set, base)
\tlayer.set_cells_terrain_connect(cells, terrain_set, terrain)
\tvar rows = []
\tfor y in size:
\t\tvar r = []
\t\tfor x in size:
\t\t\tvar c = layer.get_cell_atlas_coords(Vector2i(x, y))
\t\t\tr.append("%d/%d" % [c.x, c.y])
\t\trows.append(",".join(r))
\treturn ";".join(rows)
func _init():
\tvar lake: Array[Vector2i] = []
\tfor y in range(2, 6):
\t\tfor x in range(2, 6):
\t\t\tlake.append(Vector2i(x, y))
\tprint("BLOB ", paint("res://water/tileset.tres", 8, lake, 0, 1, 0))
\tvar fence: Array[Vector2i] = [Vector2i(1, 1), Vector2i(2, 1), Vector2i(3, 1), Vector2i(1, 2), Vector2i(1, 3)]
\tprint("LINE ", paint("res://fence/tileset.tres", 5, fence, 0, 0, -1))
\tvar src: TileSetAtlasSource = load("res://fence/tileset.tres").get_source(0)
\tprint("COLL ", src.get_tile_data(Vector2i(2, 2), 0).get_collision_polygons_count(0))
\tvar path: Array[Vector2i] = [Vector2i(2, 2), Vector2i(3, 2), Vector2i(2, 3), Vector2i(3, 3)]
\tprint("CORNER ", paint("res://dirt/tileset.tres", 6, path, 0, 1, 0))
\tquit()
"""


def _grid(out: str, tag: str):
    line = next(x for x in out.splitlines() if x.startswith(tag + " "))
    return [[tuple(map(int, c.split("/"))) for c in row.split(",")] for row in line[len(tag) + 1:].split(";")]


@pytest.mark.skipif(not GODOT, reason="set GODOT_BIN to load the TileSets in Godot")
def test_godot_paints_generated_sets(tmp_path):
    grass = tg.builtin_swatch("grass", 16).texture
    sets = {"water": tg.build_tileset("water", tg.builtin_swatch("water", 16).texture, base=grass),
            "fence": tg.build_tileset("fence", tg.builtin_swatch("fence", 16).texture, base=grass),
            "dirt": tg.build_tileset("dirt", tg.builtin_swatch("dirt", 16).texture, base=grass)}
    for name, gs in sets.items():
        tg.write_tileset(gs, tmp_path / name)
    (tmp_path / "project.godot").write_text('config_version=5\n[application]\nconfig/name="t"\n')
    (tmp_path / "check.gd").write_text(CHECK)
    run = [GODOT, "--headless", "--path", str(tmp_path)]
    subprocess.run(run + ["--import"], capture_output=True, timeout=180)
    res = subprocess.run(run + ["-s", "check.gd"], capture_output=True, text=True, timeout=180)
    out = res.stdout
    assert "BLOB " in out, res.stdout + res.stderr
    blob = _grid(out, "BLOB")
    look = sets["water"].lookup()
    pos = {xy: m for m, xys in look.items() for xy in xys}
    assert all(c != (-1, -1) for row in blob for c in row)
    assert pos[blob[3][3]] == 255 and pos[blob[2][2]] == tg.canonical(tg.E | tg.SE | tg.S)
    assert pos[blob[2][5]] == tg.canonical(tg.W | tg.SW | tg.S)
    assert pos[blob[2][4]] == tg.canonical(tg.E | tg.SE | tg.S | tg.SW | tg.W)
    assert pos[blob[0][0]] == -1 and pos[blob[7][7]] == -1           # grass up to the map edge
    line = _grid(out, "LINE")
    lpos = {(d["col"], d["row"]): d["mask"] for d in sets["fence"].tiles if "mask" in d}
    assert lpos[line[1][1]] == tg.LE | tg.LS and lpos[line[1][2]] == tg.LE | tg.LW
    assert lpos[line[1][3]] == tg.LW and lpos[line[3][1]] == tg.LN
    assert int(next(x for x in out.splitlines() if x.startswith("COLL ")).split()[1]) >= 2
    corner = _grid(out, "CORNER")
    assert any(c[1] == 1 and c[0] != 14 for row in corner for c in row)   # path edge tiles


# --------------------------------------------------------------------------- fewer visible repeats

GPU_SETS = Path(os.environ.get("PIXEL_GPU_TILESETS", r"C:/Users/spark/Pictures/PixelPipeline/"
                               r"tilegen_gpu_home/Projects/Tilegen GPU/tilesets"))


def _repeats(gs: tg.GenTileset) -> dict:
    """Lattice autocorrelation + blurred repeat contrast of 32x32 tiles of the terrain's full
    tiles and of its plain base, each placed by the set's own weights (as Godot would)."""
    full, plain = (255, -1) if gs.kind == "blob" else ((1, 1, 1, 1), (0, 0, 0, 0))
    out = {}
    t = gs.tile
    for name, key in (("terrain", full), ("base", plain)):
        xys, w = gs.choices()[key]
        area = tg.fill_area([gs.atlas[r * t:(r + 1) * t, c * t:(c + 1) * t] for c, r in xys], w, 32)
        assert area.shape == (32 * t, 32 * t, 4)
        out[name] = (tg.repeat_score(area, t), tg.repeat_contrast(area, t))
    return out


def _old_and_new(terrain: str, raw: np.ndarray, st: assets.Style):
    """The 2026-10-13 way (no evening out, 1-4 shifted interiors at 0.35) and today's."""
    pic = tg.seamless(raw)
    old = tg.build_tileset(terrain, tg.pixel_texture(pic, st.tile, st), picture=pic, style=st,
                           base=tg.builtin_swatch("grass", st.tile, style=st, even=False).texture, mix=False)
    pic = tg.even_out(pic)
    new = tg.build_tileset(terrain, tg.pixel_texture(pic, st.tile, st), picture=pic, style=st,
                           base=tg.builtin_swatch("grass", st.tile, style=st).texture)
    return old, new


def test_metric_tells_a_copied_tile_from_noise():
    rng = np.random.default_rng(0)
    tile = np.dstack([rng.integers(0, 255, (16, 16, 3)), np.full((16, 16), 255)]).astype(np.uint8)
    copied = tg.fill_area([tile], [1], 16)
    noise = np.dstack([rng.integers(0, 255, (256, 256, 3)), np.full((256, 256), 255)]).astype(np.uint8)
    assert tg.repeat_score(copied, 16) > 0.99 and abs(tg.repeat_score(noise, 16)) < 0.05
    blotch = tile.copy()
    blotch[:8, :, :3] = 255                                    # half the tile bright: obvious period
    assert tg.repeat_contrast(tg.fill_area([blotch], [1], 16), 16) > 5 * tg.repeat_contrast(noise, 16)


def test_variety_keeps_edges_and_spreads_the_weight():
    for terrain, t in (("grass", 16), ("stone", 16), ("water", 32)):
        tex = tg.builtin_swatch(terrain, t).texture
        vs, w = tg.variety(tex, 8, seed=3)
        assert len(vs) == len(w) >= 6 and w[0] == 1 and min(w) >= 0.5
        for v in vs[1:]:
            for edge in (np.s_[0], np.s_[-1], np.s_[:, 0], np.s_[:, -1]):
                assert (v[edge] == tex[edge]).all()
            assert (v != tex).any()
    lit = tg.builtin_swatch("stone", 16).texture.copy()        # stones lit from the left
    lit[:, ::4, :3] = 250
    lit[:, 1::4, :3] = 30
    assert "flip_h" not in tg.symmetries(lit) and "turn_180" not in tg.symmetries(lit)


def test_builtin_base_has_no_blotch():
    old = tg.builtin_swatch("grass", 16, even=False).texture
    new = tg.builtin_swatch("grass", 16).texture
    assert tg.repeat_contrast(tg.fill_area([new], [1]), 16) < 0.75 * tg.repeat_contrast(tg.fill_area([old], [1]), 16)


def test_fewer_repeats_on_the_fake_gpu_swatch(comfy):
    fake, client = comfy
    st = assets.Style(tile=16)
    from pixel_pipeline.generate import generate_images
    (img, _), = generate_images("water", "", client=client, seed=4, aspect=1.0, style_lora="")
    old, new = _old_and_new("water", np.asarray(img.convert("RGB")), st)
    o, n = _repeats(old), _repeats(new)
    print(f"fake swatch, 32x32 tiles  autocorrelation terrain {o['terrain'][0]:.3f} -> "
          f"{n['terrain'][0]:.3f}, base {o['base'][0]:.3f} -> {n['base'][0]:.3f}; blurred contrast "
          f"base {o['base'][1]:.2f} -> {n['base'][1]:.2f}")
    assert n["terrain"][0] < o["terrain"][0] - 0.2 and n["base"][0] < o["base"][0] - 0.15
    assert n["base"][1] < 0.75 * o["base"][1]
    sw, = tg.make_swatches("water", count=1, client=client, seed=4, style=st, style_lora="")
    gs = tg.build_tileset("water", sw.texture, picture=sw.picture, style=st)
    assert len(gs.choices()[255][0]) >= 6                       # make_swatches evens out by default


@pytest.mark.skipif(not GPU_SETS.exists(), reason="no real-GPU tilesets on this machine")
@pytest.mark.parametrize("terrain", ["water", "dirt", "cliff"])
def test_fewer_repeats_on_real_gpu_swatches(terrain):
    raw = GPU_SETS / f"{terrain}-tiles" / "swatch.png"
    if not raw.exists():
        pytest.skip(f"no {raw}")
    old, new = _old_and_new(terrain, np.asarray(Image.open(raw).convert("RGB")), assets.Style(tile=16))
    o, n = _repeats(old), _repeats(new)
    print(f"{terrain} GPU swatch  autocorrelation terrain {o['terrain'][0]:.3f} -> {n['terrain'][0]:.3f}, "
          f"base {o['base'][0]:.3f} -> {n['base'][0]:.3f}; blurred contrast terrain "
          f"{o['terrain'][1]:.2f} -> {n['terrain'][1]:.2f}, base {o['base'][1]:.2f} -> {n['base'][1]:.2f}")
    assert n["terrain"][0] < o["terrain"][0] - 0.1 and n["base"][0] < o["base"][0] - 0.15
    assert n["base"][1] < 0.75 * o["base"][1] and n["terrain"][1] < max(1.0, o["terrain"][1])


def test_weights_reach_godot_tiled_and_the_example(tmp_path):
    gs = tg.build_tileset("dirt", tg.builtin_swatch("dirt", 16).texture,
                          base=tg.builtin_swatch("grass", 16).texture)
    files = tg.write_tileset(gs, tmp_path / "d")
    tres = (tmp_path / "d" / "tileset.tres").read_text(encoding="utf-8")
    assert "/probability = 0.5" in tres and "/probability = 0.35" not in tres
    tsx = (tmp_path / "d" / "tileset.tsx").read_text(encoding="utf-8")
    assert 'probability="0.5"' in tsx and files
    used = {tuple(c) for c in gs.example.reshape(-1, 2).tolist()}
    assert len(used & set(gs.choices()[(0, 0, 0, 0)][0])) >= 4      # many base variants in use
    cliff = tg.build_tileset("cliff", tg.builtin_swatch("cliff", 16).texture,
                             base=tg.builtin_swatch("grass", 16).texture)
    t = 16
    fulls = [cliff.atlas[r * t:(r + 1) * t, c * t:(c + 1) * t] for c, r in cliff.choices()[255][0]]
    assert len({f.tobytes() for f in fulls}) == len(fulls) > 1        # plateau variants differ


# --------------------------------------------------------------------------- maps drawn with the sets

@pytest.fixture
def look_project():
    proj = assets.create_project("Looks", type="topdown-rpg")
    assets.set_current_project("Looks")
    grass = tg.builtin_swatch("grass", 16).texture
    for terrain in ("water", "dirt", "cliff", "fence"):
        gs = tg.build_tileset(terrain, tg.builtin_swatch(terrain, 16).texture, project=proj, base=grass)
        tg.save_tileset(gs, f"{terrain.title()} one", project=proj)
    return proj


def _look_map():
    from pixel_pipeline import maplib
    doc = maplib.new_map("flat", 12, 10, seed=3)
    d = doc.diorama
    d.terrain[2:5, 2:6] = "water"
    d.terrain[7, 1:9] = "dirt"
    d.height[1:4, 8:] = 5                                         # a plateau: 3 levels up
    d.props = [("fence", c, 8) for c in range(2, 7)] + [("tree", 10, 8)]
    doc.spawns = [(0, 0, 9), (1, 11, 0)]
    return doc


def test_map_look_layers_follow_the_autotile_rules(look_project):
    from pixel_pipeline import maplib
    doc = _look_map()
    look = maplib.map_look(doc, look_project)
    assert set(look.sets) == {"water", "dirt", "cliff", "fence"} and look.props == {"fence"}
    layers = {ly.key: ly for ly in look.layers(doc.diorama, doc.seed)}
    assert list(layers) == ["ground", "dirt", "water", "cliff", "cliff_face", "fence"]
    has = {k: {(int(y), int(x)) for y, x in np.argwhere(ly.cells[..., 0] >= 0)} for k, ly in layers.items()}
    water = {(r, c) for r in range(2, 5) for c in range(2, 6)}
    assert has["water"] == water - {(3, 3), (3, 4)}               # shores; the middle is plain water
    g = layers["ground"].cells
    assert g[2, 2][1] == look.terrains.index("grass") and g[3, 3][1] == look.terrains.index("water")
    assert {(6, 3), (8, 3), (6, 0), (8, 9)} <= has["dirt"] and not has["dirt"] & {(7, 3), (4, 3), (0, 0)}
    assert {(3, 8), (3, 11), (2, 8)} <= has["cliff"] and not has["cliff"] & {(4, 9), (2, 10), (0, 0)}
    # the rock wall stands on the low cells below the plateau's south edge (a 3-level drop: 1 tile)
    assert has["cliff_face"] == {(4, 8), (4, 9), (4, 10), (4, 11)}
    face = layers["cliff_face"].cells
    choices = look.sets["cliff"].choices
    assert tuple(face[4, 8]) == tg.face_choice(choices, "foot", 2, 8)            # its west end
    assert tuple(face[4, 9]) == tg.face_choice(choices, "foot", 3, 9) != tuple(face[4, 10])
    fence = layers["fence"].cells
    assert tuple(fence[8, 2]) == (tg.LE % 4, tg.LE // 4) and tuple(fence[8, 4]) == ((tg.LE | tg.LW) % 4, 2)
    img = look.render(doc.diorama, doc.seed)
    assert img.shape == (160, 192, 4) and (img[..., 3] == 255).all()
    plain = maplib.render_topdown(doc, 16)
    tiled = maplib.render_topdown(doc, 16, look=look)
    assert tiled.shape == plain.shape and (tiled != plain).any()
    # the same cells every time; repainting one cell keeps the others' variants
    d2 = doc.copy()
    d2.diorama.terrain[0, 11] = "sand"
    g2 = look.layers(d2.diorama, doc.seed)[0].cells
    same = np.ones(g.shape[:2], bool)
    same[0, 11] = False
    assert (g2[same] == g[same]).all()


def test_map_look_picks_a_set_per_terrain(look_project):
    from pixel_pipeline import maplib
    tex = np.zeros((16, 16, 4), np.uint8)
    tex[..., :3] = (250, 10, 200)
    tex[..., 3] = 255
    gs = tg.build_tileset("water", tex, project=look_project, base=tg.builtin_swatch("grass", 16).texture)
    tg.save_tileset(gs, "Pink water", project=look_project)
    opts = tg.look_options(look_project)
    assert [a.name for a in opts["water"]] == ["Pink water", "Water one"]       # newest first
    doc = _look_map()
    assert maplib.map_look(doc, look_project).sets["water"].name == "pink-water"
    doc.tiles["water"] = "water-one"
    assert maplib.map_look(doc, look_project).sets["water"].name == "water-one"
    doc.tiles = {"water": "", "dirt": "", "cliff": "", "fence": ""}
    assert maplib.map_look(doc, look_project) is None                          # all built-in
    back = maplib.MapDoc.from_dict(json.loads(json.dumps(doc.to_dict())))
    assert back.tiles == doc.tiles and back.copy().tiles == doc.tiles


def _compose_tiled(folder):
    """The visible tile layers of an exported .tmj, composited from its .tsj images."""
    tmj = json.loads((folder / "map.tmj").read_text(encoding="utf-8"))
    sets = []
    for ref in tmj["tilesets"]:
        tsj = json.loads((folder / ref["source"]).read_text(encoding="utf-8"))
        sets.append((ref["firstgid"], tsj, np.asarray(Image.open(folder / tsj["image"]).convert("RGBA"))))
    t, w, h = tmj["tilewidth"], tmj["width"], tmj["height"]
    img = Image.new("RGBA", (w * t, h * t))
    for layer in tmj["layers"]:
        if layer["type"] != "tilelayer" or not layer.get("visible", True) or layer["name"] == "height":
            continue
        for i, gid in enumerate(layer["data"]):
            if not gid:
                continue
            first, tsj, atlas = max((s for s in sets if s[0] <= gid), key=lambda s: s[0])
            c, r = divmod(gid - first, tsj["columns"])[::-1]
            tile = Image.fromarray(np.ascontiguousarray(atlas[r * t:(r + 1) * t, c * t:(c + 1) * t]))
            img.alpha_composite(tile, ((i % w) * t, (i // w) * t))
    return np.asarray(img).astype(int)


def test_exports_match_the_editor_look(look_project, tmp_path):
    from pixel_pipeline import maplib
    doc = _look_map()
    asset = maplib.save_map(doc, "Looks", project=look_project)
    doc = maplib.load_map(asset)
    look = maplib.map_look(doc, look_project)
    want = look.render(doc.diorama, doc.seed).astype(int)
    preview = np.asarray(Image.open(asset.folder / "preview.png"))
    assert (preview == maplib.render_topdown(doc, 16, look=look)).all()
    maplib.export_map(asset, "tiled", tmp_path / "tiled")
    assert np.abs(_compose_tiled(tmp_path / "tiled") - want).max() <= 1
    tmj = json.loads((tmp_path / "tiled" / "map.tmj").read_text(encoding="utf-8"))
    assert [ly["name"] for ly in tmj["layers"]][:3] == ["terrain", "ground", "dirt"]
    maplib.export_map(asset, "json", tmp_path / "json")
    colour = np.asarray(Image.open(tmp_path / "json" / "colour.png"))
    assert (colour == maplib.render_topdown(doc, 16, look=look)).all()
    out = tmp_path / "tm"
    maplib.export_map(asset, "godot-tilemap", out)
    scene = (out / "map.tscn").read_text(encoding="utf-8")
    for node in ("Terrain", "Dirt", "Water", "Cliff", "Cliff_Face", "Fence", "Height", "Props"):
        assert f'[node name="{node}" type="TileMapLayer"' in scene, node
    assert 'path="res://ground.tres"' in scene and "probability" in (out / "ground.tres").read_text()
    ov = (out / "tilesets" / "water-one" / "overlay.tres").read_text(encoding="utf-8")
    assert 'path="overlay.png"' in ov and "terrains_peering_bit" in ov
    props = np.asarray(Image.open(out / "props.png"))
    k = maplib.PROP_KINDS.index("fence")
    assert not props[:, k * 16:(k + 1) * 16, 3].any()            # the Fence layer draws fences
    maplib.export_map(asset, "godot-3d", tmp_path / "g3")
    top = np.asarray(Image.open(tmp_path / "g3" / "map" / "tex" / "water_top.png").convert("RGBA"))
    assert (top == look.plain("water")[0][0]).all()


COMPOSE = """extends SceneTree
func _init() -> void:
\tvar s: Node = load("res://map.tscn").instantiate()
\tvar t: int = 16
\tvar img := Image.create(s.get_meta("cols") * t, s.get_meta("rows") * t, false, Image.FORMAT_RGBA8)
\tfor n in s.get_children():
\t\tif not (n is TileMapLayer) or n.name in ["Height", "Props"]:
\t\t\tcontinue
\t\tvar src: TileSetAtlasSource = n.tile_set.get_source(0)
\t\tvar atlas: Image = src.texture.get_image()
\t\tatlas.convert(Image.FORMAT_RGBA8)
\t\tfor c in n.get_used_cells():
\t\t\timg.blend_rect(atlas, src.get_tile_texture_region(n.get_cell_atlas_coords(c)), c * t)
\timg.save_png("res://compose.png")
\tprint("COMPOSED")
\ts.free()
\tquit()
"""


@pytest.mark.skipif(not GODOT, reason="set GODOT_BIN to load the TileMap export in Godot")
def test_godot_draws_the_map_like_the_editor(look_project, tmp_path):
    from pixel_pipeline import maplib
    asset = maplib.save_map(_look_map(), "Looks", project=look_project)
    doc = maplib.load_map(asset)
    out = tmp_path / "tm"
    maplib.export_map(asset, "godot-tilemap", out)
    (out / "compose.gd").write_text(COMPOSE)
    run = [GODOT, "--headless", "--path", str(out)]
    subprocess.run(run + ["--import"], capture_output=True, timeout=300)
    res = subprocess.run(run + ["-s", "compose.gd"], capture_output=True, text=True, timeout=300)
    assert "COMPOSED" in res.stdout, res.stdout + res.stderr
    got = np.asarray(Image.open(out / "compose.png").convert("RGBA")).astype(int)
    want = maplib.map_look(doc, look_project).render(doc.diorama, doc.seed).astype(int)
    assert np.abs(got - want)[..., :3].max() <= 1


def test_map_editor_paints_with_the_sets(look_project):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from conftest import qt_app
    qt_app()
    from pixel_pipeline import maplib
    from pixel_pipeline.ui.map_editor import MapEditor
    ed = MapEditor(project=look_project)
    ed.sync = True
    try:
        assert set(ed.tile_combos) == {"water", "dirt", "cliff", "fence"}
        ed._set_doc(_look_map())
        assert ed.look is not None and ed.canvas.cell == 16
        ed.set_zoom(2)
        assert ed.canvas.width() == 12 * 16 * 2
        img = ed.canvas.image
        want = maplib.render_topdown(ed.doc, 16, grid=True, look=ed.look)
        assert (img.width(), img.height()) == (want.shape[1], want.shape[0])
        ed.set_tool("terrain")
        ed.select("water")
        ed.press(9, 7)
        ed.release()
        assert ed.doc.diorama.terrain[7, 9] == "water"
        ed.choose_tiles("water", "")                               # built-in water
        assert ed.doc.tiles == {"water": ""} and "water" not in ed.look.sets
        ed.choose_tiles("fence", "fence-one")
        assert "fence" not in ed.doc.tiles                         # newest = the default
        ed.name.setText("Painted")
        asset = ed.save()
        assert maplib.load_map(asset).tiles == {"water": ""}
        ed.open_asset(asset)
        assert ed.tile_combos["water"].currentData() == ""
    finally:
        ed.setParent(None)

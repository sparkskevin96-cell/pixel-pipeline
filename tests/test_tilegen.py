"""Tile auto-variations (tilegen.py): GPU swatches via the fake ComfyUI, seamless textures,
corner / blob / line sets, Godot + Tiled + Unity files, the map export and the UI page."""
import json
import os
import shutil
import subprocess

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
                          style=proj.style, style_lora="")
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
                     inpaint_seams=True)
    assert len(fake.prompts) == n + 2
    side = assets.style_for_type("side-scroller")
    pos, _ = tg.swatch_prompt("cliff", "", tg.camera_for(side), side)
    assert "side view" in pos and "cliff" in pos


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

import json
import os
import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image

from pixel_pipeline import assets, maplib
from pixel_pipeline.diorama import Diorama, TERRAINS, path_cost
from pixel_pipeline.hd2d import BOARD_GD

GODOT = os.environ.get("GODOT_BIN") or shutil.which("godot")


@pytest.mark.parametrize("template", maplib.TEMPLATES)
def test_templates_are_playable(template):
    doc = maplib.new_map(template, seed=2)
    d = doc.diorama
    assert (d.cols, d.rows) == maplib.DEFAULT_SIZES[template]
    assert d.height.shape == (d.rows, d.cols) and d.terrain.shape == (d.rows, d.cols)
    assert set(d.terrain.ravel()) <= set(TERRAINS)
    assert maplib.validate(doc) == []
    assert sum(t == 0 for t, _, _ in doc.spawns) >= 2 and sum(t == 1 for t, _, _ in doc.spawns) >= 2
    assert all(d.terrain[r, c] != "water" for _, c, r in d.props)
    assert d.units == []


def test_template_details():
    assert "snow" in maplib.new_map("snowfield").diorama.terrain
    forest = maplib.new_map("forest", seed=1)
    assert any(k == "tree" for k, _, _ in forest.diorama.props)
    assert forest.diorama.height.max() <= 4 and not forest.diorama.sea
    castle = maplib.new_map("castle")
    assert castle.diorama.height.max() == 8
    assert {"battlement", "well"} <= {k for k, _, _ in castle.diorama.props}
    village = maplib.new_map("village")
    assert {"house", "well", "fence"} <= {k for k, _, _ in village.diorama.props}
    flat = maplib.new_map("flat", 30, 9)
    assert (flat.cols, flat.rows) == (30, 9) and (flat.diorama.height == 2).all()
    # seeded: same seed same map, other seed other map
    a, b, c = (maplib.new_map("forest", seed=s) for s in (5, 5, 6))
    assert (a.diorama.height == b.diorama.height).all() and a.diorama.props == b.diorama.props
    assert a.diorama.props != c.diorama.props
    with pytest.raises(ValueError):
        maplib.new_map("moon")


def test_odd_sizes_still_valid():
    for t in maplib.TEMPLATES:
        doc = maplib.new_map(t, 9, 30, seed=3)
        assert maplib.validate(doc) == [], t


def test_save_load_roundtrip():
    doc = maplib.new_map("village", seed=4)
    asset = maplib.save_map(doc, "Riverside", tags=["town", "summer"])
    assert asset.kind == "map" and asset.extra["template"] == "village"
    assert asset.extra["size"] == [doc.cols, doc.rows]
    for name in ("map.json", "preview.png", "thumb.png", "iso_preview.png", "asset.json"):
        assert (asset.folder / name).exists(), name
    tile = assets.current_project().style.tile
    with Image.open(asset.folder / "tex" / "grass_top.png") as top:
        assert top.size == (tile, tile)
    found = maplib.list_maps()
    assert [a.name for a in found] == ["Riverside"] and found[0].tags == ["summer", "town"]
    back = maplib.load_map(found[0])
    assert (back.diorama.height == doc.diorama.height).all()
    assert (back.diorama.terrain == doc.diorama.terrain).all()
    assert back.diorama.props == doc.diorama.props and back.spawns == doc.spawns
    assert back.diorama.name == "Riverside" and back.diorama.sea is False
    data = json.loads((asset.folder / "map.json").read_text())
    assert data["spawns"][0].keys() == {"team", "col", "row"}
    # the board can still read it as a plain Diorama
    assert Diorama.load(asset.folder / "map.json").cols == doc.cols
    # save over the same asset
    maplib.MapEdit(back).raise_cells(0, 0, 3)
    again = maplib.save_map(back, "Riverside 2", asset=found[0])
    assert again.folder == asset.folder and len(maplib.list_maps()) == 1
    assert maplib.load_map(again).diorama.height[0, 0] == doc.diorama.height[0, 0] + 3


def test_load_diorama_puts_units_on_starts():
    doc = maplib.new_map("castle")
    asset = maplib.save_map(doc, "Keep")
    d = maplib.load_diorama(asset, blue=["knight", "archer", "dog", "mage", "cleric", "rogue"],
                            red=["wolf", "bear"])
    assert len(d.units) == 8
    starts = {(c, r) for _, c, r in doc.spawns}
    assert {(u.col, u.row) for u in d.units if u.team == 1} <= starts
    assert len({(u.col, u.row) for u in d.units}) == 8
    assert all(d.walkable(u.col, u.row) for u in d.units)
    assert maplib.load_diorama(asset).units == []


def test_edit_operations_and_undo():
    doc = maplib.new_map("flat", 10, 8)
    e = maplib.MapEdit(doc)
    e.begin()
    e.raise_cells(4, 4, 1, brush=2)
    assert doc.diorama.height[3:6, 3:6].tolist() == [[3] * 3] * 3
    assert doc.diorama.height[0, 0] == 2
    e.begin()
    e.lower_cells(4, 4, 5)
    assert doc.diorama.height[4, 4] == maplib.MIN_LEVEL
    e.begin()
    e.paint(1, 1, "stone", brush=1)
    assert doc.diorama.terrain[1, 1] == "stone"
    e.begin()
    e.put_prop(2, 2, "tree")
    assert doc.prop_at(2, 2) == "tree" and not doc.diorama.walkable(2, 2)
    e.begin()
    e.set_spawn(2, 2, 1)
    assert (1, 2, 2) in doc.spawns and doc.prop_at(2, 2) is None
    e.begin()
    n = e.fill(9, 0, "sand")
    assert n > 50 and doc.diorama.terrain[7, 9] == "sand" and doc.diorama.terrain[1, 1] == "stone"
    # undo everything back, then redo one
    for _ in range(6):
        assert e.undo()
    assert not e.undo()
    assert (doc.diorama.height == 2).all() and (doc.diorama.terrain == "grass").all()
    assert e.redo()
    assert doc.diorama.height[4, 4] == 3
    e.begin()
    e.paint(0, 0, "water")
    assert not e.redo_stack
    e.begin()
    e.put_prop(0, 0, "rock")                       # nothing on water
    assert doc.prop_at(0, 0) is None


def test_renders():
    doc = maplib.new_map("castle")
    top = maplib.render_topdown(doc, 8)
    assert top.shape == (doc.rows * 8, doc.cols * 8, 4)
    a = maplib.render_iso(doc, 0, 16)
    b = maplib.render_iso(doc, 1, 16)
    assert a.shape[2] == 4 and (a[..., 3] > 0).mean() > 0.3
    assert a.shape != b.shape or not (a == b).all()
    r = maplib.rotate_doc(doc, 1)
    assert (r.cols, r.rows) == (doc.rows, doc.cols)
    assert maplib.rotate_doc(doc, 4).diorama.props == doc.diorama.props
    # unknown props are drawn as a box, not an error
    doc.diorama.props.append(("statue", 1, doc.rows - 1))
    maplib.render_iso(doc, 0, 16)
    maplib.render_topdown(doc, 16)


def test_board_hooks():
    # hd2d's board: snow material, inland water, and props3d.gd / box fallback for new kinds
    assert '"snow"]' in BOARD_GD and 'map.get("sea", true)' in BOARD_GD
    assert "res://map/props3d.gd" in BOARD_GD and "_:" in BOARD_GD
    assert all(k in maplib.PROPS3D_GD for k in ("tree", "rock", "house", "wall", "well", "fence"))


@pytest.fixture
def forest_asset():
    return maplib.save_map(maplib.new_map("forest", seed=1), "Old Forest")


def test_export_json(forest_asset, tmp_path):
    files = maplib.export_map(forest_asset, "json", tmp_path / "j")
    names = {p.name for p in files}
    assert {"map.json", "height.png", "terrain.png", "props.json", "legend.json"} <= names
    doc = maplib.load_map(forest_asset)
    legend = json.loads((tmp_path / "j" / "legend.json").read_text())
    h = np.asarray(Image.open(tmp_path / "j" / "height.png"))
    assert h.shape == (doc.rows, doc.cols) and h.dtype in (np.uint16, np.int32)
    levels = h.astype(int) // legend["height_scale"] + legend["height_offset"]
    assert (levels == doc.diorama.height).all()
    t = np.asarray(Image.open(tmp_path / "j" / "terrain.png"))
    assert legend["terrains"][t[0, 0]] == doc.diorama.terrain[0, 0]


def test_export_tiled(forest_asset, tmp_path):
    maplib.export_map(forest_asset, "tiled", tmp_path / "t")
    doc = maplib.load_map(forest_asset)
    tmj = json.loads((tmp_path / "t" / "map.tmj").read_text())
    assert (tmj["width"], tmj["height"]) == (doc.cols, doc.rows)
    layers = {l["name"]: l for l in tmj["layers"]}
    assert len(layers["terrain"]["data"]) == doc.cols * doc.rows
    assert len(layers["height"]["data"]) == doc.cols * doc.rows
    hs = next(t for t in tmj["tilesets"] if t["source"] == "heights.tsj")["firstgid"]
    assert layers["height"]["data"][5] - hs == doc.diorama.height[0, 5]
    assert len(layers["props"]["objects"]) == len(doc.diorama.props)
    assert len(layers["spawns"]["objects"]) == len(doc.spawns)
    for ts in tmj["tilesets"]:
        tsj = json.loads((tmp_path / "t" / ts["source"]).read_text())
        assert (tmp_path / "t" / tsj["image"]).exists()


def test_export_godot_files(forest_asset, tmp_path):
    f3 = maplib.export_map(forest_asset, "godot-3d", tmp_path / "g3")
    root = tmp_path / "g3"
    for rel in ("project.godot", "view.tscn", "view.gd", "board.gd", "map/map.json",
                "map/props3d.gd", "map/tex/grass_top.png"):
        assert (root / rel).exists(), rel
    assert 'run/main_scene="res://view.tscn"' in (root / "project.godot").read_text()
    assert json.loads((root / "map" / "map.json").read_text())["units"] == []
    assert all(p.exists() for p in f3)
    maplib.export_map(forest_asset, "godot-tilemap", tmp_path / "tm")
    for rel in ("map.tscn", "terrain.tres", "terrain.png", "heights.tres", "props.tres"):
        assert (tmp_path / "tm" / rel).exists(), rel
    maplib.export_map(forest_asset, "godot-iso", tmp_path / "iso")
    assert (tmp_path / "iso" / "iso_map.tscn").exists() and (tmp_path / "iso" / "iso_columns.tres").exists()
    with pytest.raises(ValueError):
        maplib.export_map(forest_asset, "unity", tmp_path / "x")


def _godot(args, timeout=180):
    run = subprocess.run([GODOT, "--headless", *args], capture_output=True, text=True, timeout=timeout)
    return run.stdout + run.stderr


def _clean(log):
    bad = [l for l in log.splitlines() if ("ERROR" in l or "SCRIPT ERROR" in l or "Parse Error" in l)
           and "leaked" not in l and "still in use" not in l]
    return bad


@pytest.mark.skipif(not GODOT, reason="set GODOT_BIN to load the exports in Godot")
def test_godot_loads_3d_export(tmp_path):
    asset = maplib.save_map(maplib.new_map("castle"), "Keep")
    out = tmp_path / "g3"
    maplib.export_map(asset, "godot-3d", out)
    log = _godot(["--path", str(out), "res://view.tscn", "--quit-after", "20"])
    assert not _clean(log), log[-2000:]


@pytest.mark.skipif(not GODOT, reason="set GODOT_BIN to load the exports in Godot")
def test_godot_loads_tilemap_export(tmp_path):
    asset = maplib.save_map(maplib.new_map("village"), "Town")
    doc = maplib.load_map(asset)
    out = tmp_path / "tm"
    maplib.export_map(asset, "godot-tilemap", out)
    (out / "check.gd").write_text(
        "extends SceneTree\nfunc _init() -> void:\n"
        "\tvar s: Node = load(\"res://map.tscn\").instantiate()\n"
        "\tvar t: TileMapLayer = s.get_node(\"Terrain\")\n"
        "\tvar h: TileMapLayer = s.get_node(\"Height\")\n"
        "\tvar p: TileMapLayer = s.get_node(\"Props\")\n"
        "\tprint(\"CELLS %d %d %d\" % [t.get_used_cells().size(), h.get_used_cells().size(), p.get_used_cells().size()])\n"
        "\tprint(\"LEVEL %d\" % h.get_cell_tile_data(Vector2i(0, 0)).get_custom_data(\"level\"))\n"
        "\tprint(\"STARTS %d\" % s.get_node(\"Spawns\").get_child_count())\n"
        "\ts.free()\n\tquit()\n")
    _godot(["--path", str(out), "--import"])
    log = _godot(["--path", str(out), "-s", "check.gd"])
    assert not _clean(log), log[-2000:]
    n = doc.cols * doc.rows
    assert f"CELLS {n} {n} {len(doc.diorama.props)}" in log, log[-1500:]
    assert f"LEVEL {doc.diorama.height[0, 0]}" in log
    assert f"STARTS {len(doc.spawns)}" in log
    iso = tmp_path / "iso"
    maplib.export_map(asset, "godot-iso", iso)
    _godot(["--path", str(iso), "--import"])
    log = _godot(["--path", str(iso), "res://iso_map.tscn", "--quit-after", "10"])
    assert not _clean(log), log[-2000:]


# --------------------------------------------------------------------------- editor UI

@pytest.fixture
def editor():
    from conftest import qt_app
    qt_app()
    from pixel_pipeline.ui.map_editor import MapEditor
    ed = MapEditor()
    ed.sync = True
    yield ed
    ed.setParent(None)


def test_editor_paint_undo_save_export(editor, tmp_path):
    ed = editor
    ed.new_from_template("flat", 10, 8)
    ed.set_tool("raise")
    ed.press(3, 3)
    ed.drag(4, 3)
    ed.drag(4, 3)                      # same cell twice in a stroke: once
    ed.release()
    assert ed.doc.diorama.height[3, 3] == 3 and ed.doc.diorama.height[3, 4] == 3
    ed.press(3, 3, right=True)         # right-click lowers
    ed.release()
    assert ed.doc.diorama.height[3, 3] == 2
    ed.set_tool("terrain")
    ed.select("snow")
    ed.brush.setValue(2)
    ed.press(6, 6)
    ed.release()
    assert ed.doc.diorama.terrain[5:8, 5:8].tolist() == [["snow"] * 3] * 3
    ed.press(0, 0, right=True)         # right-click picks the terrain
    assert ed.choice() == "grass"
    ed.set_tool("props")
    ed.select("house")
    ed.press(1, 1)
    ed.release()
    assert ed.doc.prop_at(1, 1) == "house"
    ed.set_tool("spawns")
    ed.select(1)
    ed.press(8, 0)
    ed.release()
    assert (1, 8, 0) in ed.doc.spawns
    ed.set_tool("fill")
    ed.select("sand")
    ed.press(0, 7)
    ed.release()
    assert ed.doc.diorama.terrain[0, 0] == "sand"
    assert ed.undo() and ed.doc.diorama.terrain[0, 0] == "grass"
    assert ed.redo() and ed.doc.diorama.terrain[0, 0] == "sand"
    img = ed.render_preview_now()
    ed.rotate_preview()
    assert ed.rotation == 1 and img.shape[2] == 4
    saved = []
    ed.map_saved.connect(saved.append)
    ed.name.setText("Test Field")
    asset = ed.save()
    assert saved == [asset] and ed.library.count() == 1 and not ed.dirty
    back = maplib.load_map(asset)
    assert back.prop_at(1, 1) == "house" and back.diorama.terrain[6, 6] == "snow"
    files = ed.export_to("tiled", tmp_path / "out")
    assert (tmp_path / "out" / "map.tmj").exists() and files


def test_editor_templates_and_library(editor):
    ed = editor
    for t in ("castle", "forest"):
        ed.new_from_template(t, seed=1)
        ed.save()
    assert ed.library.count() == 2
    first = maplib.list_maps()[-1]
    ed.open_asset(first)
    assert ed.doc.template == first.extra["template"] and ed.name.text() == first.name
    ed.set_zoom(4)
    assert ed.canvas.width() == ed.doc.cols * 16 * 4

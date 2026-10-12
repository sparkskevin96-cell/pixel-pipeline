"""Side-scroller and isometric tiles."""
import json
import os
import shutil
import subprocess

import numpy as np
import pytest

from pixel_pipeline.cli import main
from pixel_pipeline.maptiles import terrain
from pixel_pipeline.tilesets import (BOTTOM, LEFT, RIGHT, TOP, build_iso_set, build_side_set,
                                     build_style, example_level, render_level, side_masks,
                                     style_preview, write_iso, write_side)

GODOT = os.environ.get("GODOT_BIN") or shutil.which("godot")


@pytest.fixture(scope="module")
def side():
    return build_side_set(terrain("dirt", size=16), terrain("grass", size=16))


def tile_of(ss, mask):
    c, r = ss.masks[mask]
    t = ss.tile
    return ss.atlas[r * t:(r + 1) * t, c * t:(c + 1) * t]


def test_side_tiles_cap_only_open_tops(side):
    alone = tile_of(side, 0)
    inner = tile_of(side, TOP | RIGHT | BOTTOM | LEFT)
    assert (alone[: side.lip, :, 3] == 0).mean() > 0.6               # air above an open top
    assert (inner[..., 3] == 255).all()                              # buried ground: solid dirt
    grass = alone[side.lip + 1, 8, :3].astype(int)
    assert grass[1] > grass[0] and grass[1] > grass[2]               # green cap
    assert inner[8, 8, 1] <= inner[8, 8, 0] + 10                     # no cap inside
    # caps line up across neighbours: same cap rows at the shared edge
    left, right = tile_of(side, RIGHT | BOTTOM), tile_of(side, LEFT | BOTTOM)
    assert np.array_equal((left[:, -1, 1] > left[:, -1, 0]), (right[:, 0, 1] > right[:, 0, 0]))


def test_side_masks_and_level(side):
    solid = np.array([[0, 1, 1], [1, 1, 1]], bool)
    m = side_masks(solid)
    assert m[0, 0] == -1 and m[0, 1] == RIGHT | BOTTOM and m[1, 1] == TOP | LEFT | RIGHT
    level, plats = example_level()
    img = render_level(side, level, plats)
    assert img.shape == (11 * 16, 20 * 16, 4) and len(plats) == 4


def test_iso_floor_and_block_shapes():
    iso = build_iso_set([terrain("grass", size=32), terrain("stone", size=32)], 32)
    assert iso.floors.shape == (16, 64, 4) and iso.blocks.shape == (32, 64, 4)
    f = iso.floors[:, :32, 3] > 0
    assert f[8, 16] and not f[0, 0] and not f[15, 31] and f[8].sum() > 28   # a diamond
    b = iso.blocks[:, :32]
    assert (b[24, 16, 3] > 0) and b[24, 4, :3].sum() > b[24, 28, :3].sum()  # lit left side
    with pytest.raises(ValueError, match="multiple of 4"):
        build_iso_set([terrain("grass", size=30)], 30)


def test_writers_and_cli(side, tmp_path, capsys):
    names = [p.name for p in write_side(side, tmp_path / "side.png")]
    assert names == ["side.png", "side.tres", "side_example.png", "side_example.json"]
    tres = (tmp_path / "side.tres").read_text()
    assert "terrain_set_0/mode = 2" in tres and "one_way = true" in tres and tres.count("/terrain = 0") == 16
    iso = build_iso_set([terrain("grass", size=32)], 32)
    names = [p.name for p in write_iso(iso, tmp_path / "iso.png")]
    assert names == ["iso.png", "iso_blocks.png", "iso.tres", "iso_example.png", "iso_example.json"]
    assert "tile_shape = 1" in (tmp_path / "iso.tres").read_text()
    assert main(["maptiles", "--style", "sidescroller", "-t", "stone", "-t", "snow", "--tile", "16",
                 "-o", str(tmp_path / "s2.png")]) == 0
    assert "sidescroller: 2 terrains (stone, snow)" in capsys.readouterr().out
    assert main(["maptiles", "--style", "isometric", "-o", str(tmp_path / "i2.png")]) == 0
    data = json.loads((tmp_path / "i2_example.json").read_text())
    assert data["terrains"] == ["grass", "water", "stone", "sand"]
    for style in ("topdown", "sidescroller", "isometric"):
        built = build_style(style, [terrain("grass", size=16), terrain("water", size=16)])
        assert style_preview(style, built).shape[2] == 4


def test_map_tab_styles(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from conftest import qt_app
    qt_app()
    from pixel_pipeline.ui.tools import MapTab
    tab = MapTab()
    tab.sync = True
    tab.tile.setValue(16)
    tab.style.setCurrentIndex(1)
    assert tab.specs() == ["dirt", "grass"] and not tab.variants.isEnabled()
    tab.make()
    assert tab.built_style == "sidescroller"
    assert (tmp_path / "p.tres") in tab.save(tmp_path / "p.png")
    tab.style.setCurrentIndex(2)
    tab.make()
    assert (tmp_path / "q_blocks.png") in tab.save(tmp_path / "q.png")


CHECK = """extends SceneTree
func _init():
\tvar side = TileMapLayer.new()
\tside.tile_set = load("res://side.tres")
\troot.add_child(side)
\tvar cells: Array[Vector2i] = []
\tfor x in range(3):
\t\tfor y in range(2):
\t\t\tcells.append(Vector2i(x, y))
\tside.set_cells_terrain_connect(cells, 0, 0)
\tprint("S ", side.get_cell_atlas_coords(Vector2i(0, 0)), side.get_cell_atlas_coords(Vector2i(1, 0)),
\t\tside.get_cell_atlas_coords(Vector2i(2, 1)), side.get_cell_atlas_coords(Vector2i(1, 1)))
\tvar src: TileSetAtlasSource = side.tile_set.get_source(0)
\tprint("P ", src.get_tile_data(Vector2i(2, 1), 0).get_collision_polygons_count(0), " ",
\t\tsrc.get_tile_data(Vector2i(1, 4), 0).is_collision_polygon_one_way(0, 0))
\tvar iso = TileMapLayer.new()
\tiso.tile_set = load("res://iso.tres")
\tprint("I ", iso.map_to_local(Vector2i(1, 0)), iso.map_to_local(Vector2i(0, 1)))
\tside.free()
\tiso.free()
\tquit()
"""


@pytest.mark.skipif(not GODOT, reason="set GODOT_BIN to load the TileSets in Godot")
def test_godot_paints_side_terrain_and_iso_grid(side, tmp_path):
    write_side(side, tmp_path / "side.png", example=False)
    write_iso(build_iso_set([terrain("grass", size=32)], 32), tmp_path / "iso.png", example=False)
    (tmp_path / "project.godot").write_text('config_version=5\n[application]\nconfig/name="t"\n')
    (tmp_path / "check.gd").write_text(CHECK)
    run = [GODOT, "--headless", "--path", str(tmp_path)]
    subprocess.run(run + ["--import"], capture_output=True, timeout=300)
    out = subprocess.run(run + ["-s", "check.gd"], capture_output=True, text=True, timeout=300).stdout
    lines = {x[0]: x[2:] for x in out.splitlines() if x[:2] in ("S ", "P ", "I ")}
    # (0,0) open top+left -> mask 6 at (2,1); (1,0) -> 14 at (2,3); (2,1) -> 9 at (1,2); (1,1) -> 11 at (3,2)
    assert lines.get("S") == "(2, 1)(2, 3)(1, 2)(3, 2)", out
    assert lines["P"] == "1 true" and lines["I"] == "(32.0, 16.0)(0.0, 16.0)"

import json
import os
import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image

from pixel_pipeline.cli import main
from pixel_pipeline.maptiles import (build_terrain_set, builtin_texture, cells_from_vertices, corner_mask,
                                     example_vertex_grid, render_map, terrain, write_terrain)


def test_builtin_textures_are_seamless_and_repeatable():
    a = builtin_texture("grass", 32, seed=1)
    assert np.array_equal(a, builtin_texture("grass", 32, seed=1))
    tiled = np.tile(a[..., :3].astype(int), (2, 2, 1))
    seam = np.abs(np.diff(tiled, axis=1))[:, 31].mean()        # across the repeat
    inside = np.abs(np.diff(tiled, axis=1))[:, :31].mean()
    assert seam < inside * 1.6


def test_corner_masks_agree_across_shared_edges():
    jitter = np.random.default_rng(0).random((16, 16))
    left = corner_mask(16, (0, 1, 0, 1), jitter)           # overlay on its right side
    right = corner_mask(16, (1, 1, 1, 1), jitter)
    assert left[:, -1].all() and right.all() and not left[:, 0].any()
    assert not corner_mask(16, (0, 0, 0, 0), jitter).any()


def test_terrain_set_layout_and_map():
    ts = build_terrain_set([terrain(n, size=16) for n in ("grass", "water", "road")], variants=3)
    assert ts.atlas.shape == (3 * 16, 15 * 16, 4)
    table = ts.lookup()
    assert len(table[(0, 0, 0, 0)]) == 3 and (14, 1) in table[(1, 1, 1, 1)]
    vg = example_vertex_grid(12, 8, 3, seed=2)
    for y in range(8):                        # never two different non-base terrains in a cell
        for x in range(12):
            c = {int(v) for v in vg[y:y + 2, x:x + 2].ravel()} - {0}
            assert len(c) <= 1
    cells = cells_from_vertices(vg, ts)
    img = render_map(cells, ts)
    assert img.shape == (8 * 16, 12 * 16, 4) and (img[..., 3] == 255).all()


def test_write_terrain_and_cli(tmp_path, capsys):
    ts = build_terrain_set([terrain("grass", size=16), terrain("water", size=16)])
    written = write_terrain(ts, tmp_path / "t.png", preview=(6, 4))
    assert [p.name for p in written] == ["t.png", "t.tres", "t_example.png", "t_example.json"]
    tres = (tmp_path / "t.tres").read_text()
    assert "terrain_set_0/mode = 1" in tres and 'terrain_set_0/terrain_1/name = "water"' in tres
    assert "14:1/0/terrains_peering_bit/bottom_right_corner = 1" in tres
    assert "0:0/15 = 15" in tres                       # grass next to empty map edges
    assert json.loads((tmp_path / "t_example.json").read_text())["cols"] == 6
    tex = tmp_path / "rock.png"
    Image.fromarray((np.random.default_rng(0).random((64, 64, 3)) * 255).astype(np.uint8)).save(tex)
    assert main(["maptiles", "-o", str(tmp_path / "m" / "map.png"), "-t", "grass",
                 "-t", f"rock={tex}", "--tile", "16", "--example", "8x5"]) == 0
    assert "2 terrains (grass, rock), 16px tiles" in capsys.readouterr().out


GODOT = os.environ.get("GODOT_BIN") or shutil.which("godot")


@pytest.mark.skipif(not GODOT, reason="set GODOT_BIN to a Godot 4 binary to run")
def test_godot_terrain_painter_uses_the_tiles(tmp_path):
    ts = build_terrain_set([terrain("grass", size=16), terrain("water", size=16)])
    write_terrain(ts, tmp_path / "terrain.png", preview=None)
    (tmp_path / "project.godot").write_text('config_version=5\n[application]\nconfig/name="t"\n')
    (tmp_path / "check.gd").write_text(
        "extends SceneTree\nfunc _init():\n"
        "\tvar layer = TileMapLayer.new()\n\tlayer.tile_set = load(\"res://terrain.tres\")\n"
        "\troot.add_child(layer)\n"
        "\tvar all: Array[Vector2i] = []\n\tfor y in 8:\n\t\tfor x in 8:\n\t\t\tall.append(Vector2i(x, y))\n"
        "\tlayer.set_cells_terrain_connect(all, 0, 0)\n"
        "\tvar lake: Array[Vector2i] = [Vector2i(3, 3), Vector2i(4, 3), Vector2i(3, 4), Vector2i(4, 4)]\n"
        "\tlayer.set_cells_terrain_connect(lake, 0, 1)\n"
        "\tvar rows = []\n\tfor y in 8:\n\t\tvar r = []\n"
        "\t\tfor x in 8:\n\t\t\tvar c = layer.get_cell_atlas_coords(Vector2i(x, y))\n"
        "\t\t\tr.append(\"%d/%d\" % [c.x, c.y])\n\t\trows.append(\",\".join(r))\n"
        "\tprint(\"MAP \", \";\".join(rows))\n\tquit()\n")
    subprocess.run([GODOT, "--headless", "--path", str(tmp_path), "--import"], capture_output=True, timeout=120)
    out = subprocess.run([GODOT, "--headless", "--path", str(tmp_path), "-s", "check.gd"],
                         capture_output=True, text=True, timeout=120).stdout
    line = next(x for x in out.splitlines() if x.startswith("MAP "))
    cells = [[tuple(map(int, c.split("/"))) for c in row.split(",")] for row in line[4:].split(";")]
    assert all(c != (-1, -1) for row in cells for c in row)       # every cell, edges included
    assert cells[0][0][1] == 0                                     # grass at the map edge
    water = [c for row in cells for c in row if c[1] == 1]
    assert water and any(c[0] != 14 for c in water)                # lake has shore tiles

import json

import numpy as np
from PIL import Image

from pixel_pipeline.cli import main
from pixel_pipeline.export import pack_sheet, slice_tiles, write_tileset


def test_pack_sheet_grid():
    frames = [np.full((4, 5, 4), i, np.uint8) for i in range(5)]
    sheet, rects = pack_sheet(frames, columns=2)
    assert sheet.shape == (12, 10, 4)
    assert rects[3] == (5, 4, 5, 4)
    assert (sheet[4:8, 5:10] == 3).all()


def test_tiles_dedupe_and_empty(tmp_path):
    a = np.zeros((8, 8, 4), np.uint8)
    a[..., 3] = 255
    b = a.copy()
    b[..., 0] = 200
    img = np.zeros((16, 16, 4), np.uint8)  # bottom-right tile stays empty
    img[:8, :8], img[:8, 8:], img[8:, :8] = a, b, a
    tiles, tile_map = slice_tiles(img, 8)
    assert len(tiles) == 2
    assert tile_map.tolist() == [[0, 1], [0, -1]]
    tmj = json.loads(write_tileset(img, 8, tmp_path / "t.png").read_text())
    assert tmj["layers"][0]["data"] == [1, 2, 1, 0]
    assert tmj["tilesets"][0]["tilecount"] == 2


def test_cli_convert_sheet_tileset(tmp_path, blob_png, capsys):
    out = tmp_path / "s.png"
    assert main(["convert", str(blob_png), "-s", "animal", "-W", "20", "-H", "16",
                 "--palette", "sweetie-16", "--scale", "3", "-o", str(out), "--aseprite"]) == 0
    assert Image.open(out).size == (60, 48)
    assert out.with_suffix(".aseprite").exists()

    frames_dir = tmp_path / "frames"
    assert main(["convert", str(blob_png), str(blob_png), "-o", str(frames_dir)]) == 0
    frames = sorted(frames_dir.glob("*.png"))
    assert len(frames) == 1  # same stem twice -> same output name

    sheet = tmp_path / "sheet.png"
    assert main(["sheet", str(out), str(out), "-o", str(sheet), "--tag", "idle", "--aseprite"]) == 0
    meta = json.loads(sheet.with_suffix(".json").read_text())
    assert meta["meta"]["frameTags"][0]["name"] == "idle" and len(meta["frames"]) == 2
    assert sheet.with_suffix(".aseprite").exists()

    assert main(["tileset", str(out), "--tile", "12", "-o", str(tmp_path / "tiles.png")]) == 0
    assert (tmp_path / "tiles.tmj").exists()

    assert main(["palettes"]) == 0
    assert "pico-8" in capsys.readouterr().out


def test_cli_bad_palette_exits_2(blob_png, tmp_path, capsys):
    assert main(["convert", str(blob_png), "--palette", "nope", "-o", str(tmp_path / "x.png")]) == 2
    assert "Unknown palette" in capsys.readouterr().err

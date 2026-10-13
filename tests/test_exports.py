import json
import re

import numpy as np
import pytest
from PIL import Image

from pixel_pipeline import exports, library
from pixel_pipeline.anim3d import Action, AnimatedSet, write_animated
from pixel_pipeline.pipeline import Sprite
from pixel_pipeline.turnaround import direction_names

W, H, FRAMES = 12, 16, 3
RED = [200, 30, 30]
DARK_RED = [120, 20, 20]
GREY = [128, 128, 140]


def _sprite(a: int, d: int, f: int) -> Sprite:
    rgba = np.zeros((H, W, 4), np.uint8)
    rgba[2:15, 3:9, :3] = GREY                      # body
    rgba[4:13, 2:5, :3] = RED                       # cape
    rgba[10:13, 2:5, :3] = DARK_RED                 # cape shade
    rgba[0:2, 4 + (f + d) % 3, :3] = [230, 190, 150]  # moving head bit
    rgba[2:15, 2:9, 3] = 255
    rgba[0:2, 4 + (f + d) % 3, 3] = 255
    rgba[13 + a % 2, 3:9, :3] = [40, 40, 40]        # feet differ per action
    nrm = np.zeros_like(rgba)
    nrm[..., 0], nrm[..., 1], nrm[..., 2] = 128, 128, 255
    nrm[..., 3] = rgba[..., 3]
    return Sprite(rgba, np.where(rgba[..., 3] > 0, 0, -1), np.zeros((1, 3), np.uint8), nrm)


def make_character(tmp_path, name="Knight"):
    """A small library character: idle + walk x 4 directions x 3 frames, with normal map,
    .aseprite and Godot SpriteFrames."""
    dirs = direction_names(4)
    actions = [Action("idle", True, 150), Action("walk", True, 100)]
    sprites = [[[_sprite(a, d, f) for f in range(FRAMES)] for d in range(len(dirs))]
               for a in range(len(actions))]
    src = tmp_path / "src" / "hero.png"
    src.parent.mkdir(parents=True, exist_ok=True)
    write_animated(AnimatedSet(actions, dirs, sprites), src, aseprite=True, godot=True, normals=True)
    return library.import_sheet(name, src)


@pytest.fixture
def knight(tmp_path):
    return make_character(tmp_path)


ROWS = 2 * 4


def test_source_reads_frames_and_tags(knight):
    src = exports.source(knight)
    assert len(src.frames) == ROWS * FRAMES and src.normals is not None
    assert [t[0] for t in src.tags][:2] == ["idle_E", "idle_S"]
    assert src.groups()[0][0] == "idle_E" and len(src.groups()) == ROWS
    assert exports.source(knight.folder).name == "Knight"


def test_godot_2d(knight, tmp_path):
    out = tmp_path / "godot"
    files = exports.export_asset(knight, "godot-2d", out)
    names = {p.name for p in files}
    assert {"knight.png", "knight.tres", "knight_facing.gd", "knight_n.png"} <= names
    tres = (out / "knight.tres").read_text()
    assert 'path="knight.png"' in tres and tres.count("AtlasTexture_") >= ROWS * FRAMES
    assert '&"walk_N"' in tres
    assert "knight.tres" in (out / "knight_facing.gd").read_text()
    exports.export_asset(knight, "godot-2d", out, res_path="res://units/knight")
    assert 'path="res://units/knight/knight.png"' in (out / "knight.tres").read_text()


def test_godot_hd2d(knight, tmp_path):
    files = exports.export_asset(knight, "godot-hd2d", tmp_path / "game", ppu=40)
    unit = tmp_path / "game" / "units" / "knight"
    assert {p.name for p in files} >= {"knight.png", "knight.json", "knight_n.png", "knight.unit.json"}
    info = json.loads((unit / "knight.unit.json").read_text())
    assert info["frame_w"] == W and info["frame_h"] == H and info["ppu"] == 40
    assert info["anims"] == ["idle", "walk"]


def test_unity_meta(knight, tmp_path):
    out = tmp_path / "unity"
    exports.export_asset(knight, "unity", out, ppu=16)
    meta = (out / "knight.png.meta").read_text()
    sheet_h = Image.open(out / "knight.png").size[1]
    assert re.search(r"^guid: [0-9a-f]{32}$", meta, re.M)
    assert "spriteMode: 2" in meta and "filterMode: 0" in meta and "textureType: 8" in meta
    assert "enableMipMap: 0" in meta and "spritePixelsToUnits: 16" in meta
    assert re.findall(r"textureCompression: (\d+)", meta) == ["0", "0"]
    rects = re.findall(r"rect:\n\s+serializedVersion: 2\n\s+x: (\d+)\n\s+y: (\d+)\n"
                       r"\s+width: (\d+)\n\s+height: (\d+)", meta)
    assert len(rects) == ROWS * FRAMES
    names = re.findall(r"^      name: (\S+)$", meta, re.M)
    assert names[0] == "idle_E_0" and len(set(names)) == len(names)
    # first sheet row (y 0 in the PNG) sits at the TOP in Unity: y = sheet height - frame height
    x0, y0, w0, h0 = map(int, rects[0])
    assert (x0, w0, h0) == (0, W, H) and y0 == sheet_h - H
    assert int(rects[-1][1]) == 0                          # last row at the bottom
    assert "pivot: {x: 0.5, y: 0.0}" in meta and "alignment: 7" in meta
    nmeta = (out / "knight_n.png.meta").read_text()
    assert "textureType: 1" in nmeta and "sRGBTexture: 0" in nmeta
    assert re.search(r"^guid: (\w+)", nmeta, re.M).group(1) != re.search(r"^guid: (\w+)", meta, re.M).group(1)
    assert (out / "knight_README.txt").exists()
    # same asset again -> same GUID (Unity keeps references)
    exports.export_asset(knight, "unity", out, ppu=16)
    assert (out / "knight.png.meta").read_text() == meta


def test_gamemaker_strips(knight, tmp_path):
    out = tmp_path / "gm"
    files = exports.export_asset(knight, "gamemaker", out)
    strips = sorted(p for p in files if p.name.endswith(f"_strip{FRAMES}.png") and "_normal_" not in p.name)
    assert len(strips) == ROWS
    assert (out / f"knight_walk_S_strip{FRAMES}.png").exists()
    assert (out / f"knight_walk_S_normal_strip{FRAMES}.png").exists()
    for p in strips:
        assert Image.open(p).size == (W * FRAMES, H)
    summary = json.loads((out / "knight_gamemaker.json").read_text())
    assert len(summary["sprites"]) == ROWS and summary["sprites"][0]["origin"][0] == W // 2
    # the strip's frames are the sheet's frames, in order
    sheet = np.asarray(Image.open(knight.sheet).convert("RGBA"))
    strip = np.asarray(Image.open(out / f"knight_idle_E_strip{FRAMES}.png").convert("RGBA"))
    assert (strip[:, W:2 * W] == sheet[0:H, W:2 * W]).all()


def test_aseprite_copy(knight, tmp_path):
    from test_aseprite import read_ase
    files = exports.export_asset(knight, "aseprite", tmp_path / "ase")
    ase = read_ase(files[0])
    assert files[0].name == "knight.aseprite" and len(ase["frames"]) == ROWS * FRAMES
    assert [t[0] for t in ase["tags"]][:2] == ["idle_E", "idle_S"]


def test_aseprite_written_when_missing(knight, tmp_path):
    from test_aseprite import read_ase
    (knight.folder / "sheet.aseprite").unlink()
    files = exports.export_asset(knight, "aseprite", tmp_path / "ase")
    ase = read_ase(files[0])
    assert len(ase["frames"]) == ROWS * FRAMES and len(ase["tags"]) == ROWS
    assert ase["frames"][0]["duration"] == 150


def test_generic(knight, tmp_path):
    out = tmp_path / "gen"
    exports.export_asset(knight, "generic", out)
    meta = json.loads((out / "knight.json").read_text())
    assert isinstance(meta["frames"], dict) and len(meta["frames"]) == ROWS * FRAMES
    assert meta["frames"]["walk_N 2"]["frame"]["w"] == W and meta["meta"]["image"] == "knight.png"
    assert len(meta["meta"]["frameTags"]) == ROWS
    assert (out / "knight_n.png").exists()
    singles = list((out / "frames").glob("*.png"))
    assert len(singles) == ROWS * FRAMES and Image.open(singles[0]).size == (W, H)


def test_icon_only_item(tmp_path, monkeypatch):
    from pixel_pipeline import assets as A
    monkeypatch.delenv("PIXEL_LIBRARY", raising=False)
    item = A.new_asset("item", "Iron Sword")
    icon = np.zeros((32, 32, 4), np.uint8)
    icon[4:28, 14:18] = [180, 180, 190, 255]
    Image.fromarray(icon, "RGBA").save(item.folder / "icon.png")
    for preset in ("generic", "unity", "gamemaker", "godot-2d", "aseprite"):
        files = exports.export_asset(item, preset, tmp_path / preset)
        assert files, preset
    assert (tmp_path / "gamemaker" / "iron_sword.png").exists()
    assert "spriteMode: 1" in (tmp_path / "unity" / "iron_sword.png.meta").read_text()
    with pytest.raises(ValueError):
        exports.export_asset(item, "godot-hd2d", tmp_path / "hd")


def test_collection(knight, tmp_path):
    from pixel_pipeline import assets as A
    other = make_character(tmp_path / "b", "Archer")
    for ch in (knight, other):
        a = next(a for a in A.list_assets("character") if a.folder == ch.folder)
        A.set_tags(a, collections=["starter"])
    out = tmp_path / "coll"
    exports.export_collection("starter", "generic", out)
    index = json.loads((out / "index.json").read_text())
    assert index["collection"] == "starter" and len(index["assets"]) == 2
    assert {a["folder"] for a in index["assets"]} == {"knight", "archer"}
    assert (out / "knight" / "knight.png").exists() and (out / "archer" / "archer.json").exists()
    exports.export_collection([knight, other], "godot-hd2d", tmp_path / "proj")
    assert (tmp_path / "proj" / "units" / "archer" / "archer.unit.json").exists()
    with pytest.raises(ValueError):
        exports.export_collection("nothing here", "generic", out)
    with pytest.raises(ValueError):
        exports.export_asset(knight, "flash", out)


def test_item_with_turnaround_sheet_and_icon(tmp_path, monkeypatch):
    """objects.py items: icon.png + a sheet whose frames are named by direction only."""
    from pixel_pipeline import assets as A
    from pixel_pipeline.export import write_sheet
    monkeypatch.delenv("PIXEL_LIBRARY", raising=False)
    item = A.new_asset("item", "Axe")
    frames = [_sprite(0, d, 0).rgba for d in range(4)]
    write_sheet(frames, item.folder / "sheet.png", 4, 100, None, direction_names(4))
    Image.fromarray(frames[1], "RGBA").save(item.folder / "icon.png")
    src = exports.source(item)
    assert [k for k, _ in src.groups()] == direction_names(4)
    files = {p.name for p in exports.export_asset(item, "gamemaker", tmp_path / "gm")}
    assert {"axe_E.png", "axe_S.png", "axe_icon.png"} <= files
    files = {p.name for p in exports.export_asset(item, "unity", tmp_path / "u")}
    assert {"axe.png.meta", "axe_icon.png", "axe_icon.png.meta"} <= files
    exports.export_asset(item, "godot-2d", tmp_path / "g")
    assert '&"SE"' not in (tmp_path / "g" / "axe.tres").read_text()
    assert '&"S"' in (tmp_path / "g" / "axe.tres").read_text()

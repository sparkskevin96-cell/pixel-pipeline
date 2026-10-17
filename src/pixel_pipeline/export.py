"""Sprite sheets (Aseprite/TexturePacker JSON) and tilesets (Tiled .tmj)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from . import __version__
from .aseprite import write_aseprite
from .godot import write_sprite_frames


def load_rgba(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGBA")).copy()


def pixel_scale(rgba: np.ndarray) -> int:
    """Integer upscale factor of nearest-neighbour-enlarged pixel art (1 if none)."""
    h, w = rgba.shape[:2]
    for k in range(math.gcd(h, w), 1, -1):
        if h % k or w % k:
            continue
        small = rgba[::k, ::k]
        if np.array_equal(np.repeat(np.repeat(small, k, 0), k, 1), rgba):
            return k
    return 1


def load_sprite(path) -> np.ndarray:
    """Load pixel art at true 1x size (undoes an upscaled export) with binary alpha."""
    rgba = load_rgba(path)
    k = pixel_scale(rgba)
    if k > 1:
        rgba = np.ascontiguousarray(rgba[::k, ::k])
    rgba[..., 3] = np.where(rgba[..., 3] >= 128, 255, 0)
    return rgba


def pack_sheet(frames: list[np.ndarray], columns: int | None = None, padding: int = 0):
    """Grid-pack RGBA frames. Returns (sheet RGBA, [(x, y, w, h), ...])."""
    if not frames:
        raise ValueError("No frames to pack")
    cols = columns or math.ceil(math.sqrt(len(frames)))
    rows = math.ceil(len(frames) / cols)
    cw = max(f.shape[1] for f in frames)
    ch = max(f.shape[0] for f in frames)
    sheet = np.zeros((rows * ch + (rows - 1) * padding, cols * cw + (cols - 1) * padding, 4), np.uint8)
    rects = []
    for i, f in enumerate(frames):
        x, y = (i % cols) * (cw + padding), (i // cols) * (ch + padding)
        sheet[y:y + f.shape[0], x:x + f.shape[1]] = f
        rects.append((x, y, f.shape[1], f.shape[0]))
    return sheet, rects


MAX_SHEET = 8192        # px: sheets wrap into column blocks beyond this height (pack_rows)


def pack_rows(rows: list[list[np.ndarray]], padding: int = 0):
    """Pack frames ROW BY ROW: every row (e.g. one animation facing one way) starts on a new line
    of the grid; columns = the longest row, shorter rows leave the rest of their line empty.
    With rows all the same length this is exactly pack_sheet(frames, that length).
    Returns (sheet RGBA, [(x, y, w, h), ...] in row order)."""
    frames = [f for r in rows for f in r]
    if not frames:
        raise ValueError("No frames to pack")
    cols = max(len(r) for r in rows)
    cw = max(f.shape[1] for f in frames)
    ch = max(f.shape[0] for f in frames)
    lines = [r for r in rows if r]
    # taller than MAX_SHEET (2x characters: 144 rows of 128 px = 18432, over GPUs' 16384 and
    # Unity's 8192): the lines continue in a second (third...) block of columns to the right
    per = max(1, (MAX_SHEET + padding) // (ch + padding))
    blocks = math.ceil(len(lines) / per)
    per = math.ceil(len(lines) / blocks)
    bw = cols * cw + (cols - 1) * padding
    sheet = np.zeros((per * ch + (per - 1) * padding, blocks * bw + (blocks - 1) * padding, 4), np.uint8)
    rects = []
    for j, r in enumerate(lines):
        bx, by = (j // per) * (bw + padding), (j % per) * (ch + padding)
        for i, f in enumerate(r):
            x, y = bx + i * (cw + padding), by
            sheet[y:y + f.shape[0], x:x + f.shape[1]] = f
            rects.append((x, y, f.shape[1], f.shape[0]))
    if blocks > 1:                       # the sheet ends where its frames do (= sheet_json's size)
        sheet = sheet[:max(y + h for _, y, _, h in rects), :max(x + w for x, _, w, _ in rects)]
    return sheet, rects


def sheet_json(rects, image_name: str, durations: list[int], tag: str | None = None,
               names: list[str] | None = None) -> dict:
    """Aseprite 'json-array' format -- also read by Godot/Unity/Phaser atlas importers."""
    names = names or [f"frame {i}" for i in range(len(rects))]
    h = max(y + rh for _, y, _, rh in rects)
    w = max(x + rw for x, _, rw, _ in rects)
    return {
        "frames": [{
            "filename": name,
            "frame": {"x": x, "y": y, "w": rw, "h": rh},
            "rotated": False,
            "trimmed": False,
            "spriteSourceSize": {"x": 0, "y": 0, "w": rw, "h": rh},
            "sourceSize": {"w": rw, "h": rh},
            "duration": d,
        } for (x, y, rw, rh), d, name in zip(rects, durations, names)],
        "meta": {
            "app": "pixel-pipeline",
            "version": __version__,
            "image": image_name,
            "format": "RGBA8888",
            "size": {"w": w, "h": h},
            "scale": "1",
            "frameTags": [{"name": tag, "from": 0, "to": len(rects) - 1, "direction": "forward"}]
            if tag else [],
        },
    }


def write_sheet(frames: list[np.ndarray], out_png, columns: int | None = None,
                duration: int = 100, tag: str | None = None, names: list[str] | None = None) -> Path:
    """Write sheet PNG + matching .json; returns the JSON path."""
    out_png = Path(out_png)
    sheet, rects = pack_sheet(frames, columns)
    Image.fromarray(sheet, "RGBA").save(out_png)
    meta = sheet_json(rects, out_png.name, [duration] * len(frames), tag, names)
    out_json = out_png.with_suffix(".json")
    out_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_json


def slice_tiles(rgba: np.ndarray, tile: int):
    """Cut an image into tile x tile pieces, dropping duplicates and empty tiles.

    Returns (unique tiles, map) where map[row, col] is a tile index or -1 for empty.
    """
    rows, cols = rgba.shape[0] // tile, rgba.shape[1] // tile
    if rows == 0 or cols == 0:
        raise ValueError(f"Image {rgba.shape[1]}x{rgba.shape[0]} is smaller than one {tile}px tile")
    tiles: list[np.ndarray] = []
    seen: dict[bytes, int] = {}
    tile_map = np.full((rows, cols), -1, np.int32)
    for r in range(rows):
        for c in range(cols):
            t = rgba[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile]
            if not t[..., 3].any():
                continue
            key = t.tobytes()
            if key not in seen:
                seen[key] = len(tiles)
                tiles.append(t)
            tile_map[r, c] = seen[key]
    return tiles, tile_map


def tiled_map(tile_map: np.ndarray, tile: int, tileset_image: str, tileset_size: tuple[int, int],
              tile_count: int, columns: int) -> dict:
    """Minimal Tiled JSON map (.tmj); Godot, LDtk and most engines can import it."""
    rows, cols = tile_map.shape
    return {
        "type": "map", "version": "1.10", "orientation": "orthogonal", "renderorder": "right-down",
        "width": cols, "height": rows, "tilewidth": tile, "tileheight": tile, "infinite": False,
        "nextlayerid": 2, "nextobjectid": 1,
        "layers": [{
            "id": 1, "name": "Tiles", "type": "tilelayer", "visible": True, "opacity": 1,
            "x": 0, "y": 0, "width": cols, "height": rows,
            "data": (tile_map.reshape(-1) + 1).tolist(),  # Tiled gids: 0 = empty, firstgid = 1
        }],
        "tilesets": [{
            "firstgid": 1, "name": Path(tileset_image).stem, "image": tileset_image,
            "imagewidth": tileset_size[0], "imageheight": tileset_size[1],
            "tilewidth": tile, "tileheight": tile, "tilecount": tile_count,
            "columns": columns, "margin": 0, "spacing": 0,
        }],
    }


def write_tileset(rgba: np.ndarray, tile: int, out_png, columns: int = 8) -> Path:
    """Write tileset PNG + Tiled map (.tmj) rebuilding the original image; returns the .tmj path."""
    out_png = Path(out_png)
    tiles, tile_map = slice_tiles(rgba, tile)
    cols = max(1, min(columns, len(tiles)))
    sheet, _ = pack_sheet(tiles, cols)
    Image.fromarray(sheet, "RGBA").save(out_png)
    tmj = tiled_map(tile_map, tile, out_png.name, (sheet.shape[1], sheet.shape[0]), len(tiles), cols)
    out_tmj = out_png.with_suffix(".tmj")
    out_tmj.write_text(json.dumps(tmj, indent=2), encoding="utf-8")
    return out_tmj


def write_animation(frames: list[np.ndarray], out: Path, duration: int, tag: str | None = None,
                    columns: int | None = None, aseprite: bool = False, godot: bool = False,
                    palette: np.ndarray | None = None, names: list[str] | None = None) -> list[Path]:
    """Sheet PNG + JSON, and optionally .aseprite and Godot .tres. Returns written paths."""
    out = Path(out)
    columns = columns or len(frames)
    written = [out, write_sheet(frames, out, columns, duration, tag, names)]
    if aseprite:
        if len({f.shape for f in frames}) != 1:
            raise ValueError(".aseprite animation needs all frames the same size")
        tags = [(tag, 0, len(frames) - 1)] if tag else []
        ase = out.with_suffix(".aseprite")
        write_aseprite(ase, frames, palette, durations=duration, tags=tags)
        written.append(ase)
    if godot:
        _, rects = pack_sheet(frames, columns)
        written.append(write_sprite_frames(out.with_suffix(".tres"), out.name, rects,
                                           name=tag or "default", duration_ms=duration))
    return written

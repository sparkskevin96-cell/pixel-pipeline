"""Map tiles: seamless terrain tiles with corner transitions, and a Godot TileSet whose
terrains paint them for you.

The first terrain is the base (usually grass). Every other terrain gets 15 tiles that blend
it into the base from any combination of corners (shores, sand and road edges) plus its full
tile, so a map never shows a hard square edge. In Godot: select the TileMapLayer, open the
Terrains tab and paint; the right edge tiles are picked automatically ("Match Corners").
Terrains only blend into the base, so keep a strip of base between two other terrains
(water next to sand needs grass between them).

Each terrain comes from a built-in pattern (instant, no GPU), a photo of the texture, or a
prompt made on your GPU (ComfyUI). All tiles are seamless: they repeat without seams.
"""
from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .palettes import hex_to_rgb
from .quantize import bayer_matrix

# name: (ramp dark -> light, edge colour where it meets the base, style)
BUILTIN = {
    "grass": (["24451f", "2f5d28", "3f7a33", "57963c", "79b34c"], "24451f", "grass"),
    "water": (["173563", "1f4c8a", "2a66ad", "3f84c8", "9fd4f2"], "cfeaf7", "water"),
    "sand": (["a8874e", "c4a468", "d8bd80", "e8d29c", "f6e8c0"], "a8874e", "specks"),
    "road": (["5c3d24", "7a5232", "966942", "b08454", "c9a171"], "4c321e", "pebbles"),
    "dirt": (["4e3622", "6a4a2e", "86603c", "a07650", "b88e64"], "3e2a1a", "specks"),
    "stone": (["4a4a52", "62626c", "7c7c86", "9696a0", "b4b4bc"], "3a3a42", "flagstones"),
    "snow": (["9fb0c8", "bccbdc", "d6e2ee", "eaf1f8", "ffffff"], "8a9cb6", "specks"),
}


def _periodic_noise(size: int, cells: int, rng) -> np.ndarray:
    """Smooth value noise in [0, 1] that wraps around (seamless), `cells` bumps per side."""
    grid = rng.random((cells, cells))
    t = (np.arange(size) + 0.5) * cells / size
    i0 = np.floor(t).astype(int) % cells
    i1 = (i0 + 1) % cells
    f = t - np.floor(t)
    f = f * f * (3 - 2 * f)
    rows = grid[i0][:, i0] * (1 - f)[None, :] + grid[i0][:, i1] * f[None, :]
    rows2 = grid[i1][:, i0] * (1 - f)[None, :] + grid[i1][:, i1] * f[None, :]
    return rows * (1 - f)[:, None] + rows2 * f[:, None]


def low_band(field: np.ndarray, cycles: float = 1.5) -> np.ndarray:
    """The large-scale part of a periodic field (its frequencies up to `cycles` per side; the
    mean left out)."""
    h, w = field.shape
    k2 = (np.fft.fftfreq(h) * h)[:, None] ** 2 + (np.fft.fftfreq(w) * w)[None, :] ** 2
    low = (k2 <= cycles ** 2).astype(np.float64)
    low[0, 0] = 0
    return np.real(np.fft.ifft2(np.fft.fft2(field) * low))


def even_level(level: np.ndarray, cycles: float = 1.5) -> np.ndarray:
    """A periodic field with its lowest frequencies (up to ~`cycles` per side) taken out, same
    mean and spread: no tile-sized blotch that shows the period when the tile repeats."""
    out = level - low_band(level, cycles)
    sd = out.std()
    return (out - out.mean()) * (level.std() / sd if sd > 1e-9 else 0) + level.mean()


def builtin_texture(name: str, size: int = 32, seed: int = 0, even: bool = False) -> np.ndarray:
    """A seamless pixel-art texture tile (RGBA) from a built-in terrain. even: no large-scale
    brightness blotch (tilegen's sets; the map renderers keep the old look)."""
    if name not in BUILTIN:
        raise ValueError(f"Unknown built-in terrain '{name}' (use {', '.join(BUILTIN)})")
    ramp_hex, _, style = BUILTIN[name]
    ramp = hex_to_rgb(ramp_hex)
    rng = np.random.default_rng(zlib.crc32(f"{name}/{size}/{seed}".encode()))
    big = _periodic_noise(size, max(2, size // 16), rng)
    fine = _periodic_noise(size, max(4, size // 4), rng)
    level = 0.65 * big + 0.35 * fine
    if even:
        level = even_level(level)
    dither = np.tile(bayer_matrix(4), (size // 4 + 1, size // 4 + 1))[:size, :size]
    level = np.clip(level + 0.18 * dither, 0, 0.999)
    idx = (1 + level * 3).astype(int)                       # mid tones: ramp[1..3]
    if style == "water":
        waves = _periodic_noise(size, max(2, size // 8), rng)
        streak = (np.sin((np.arange(size)[:, None] / size) * np.pi * 2 * max(2, size // 8)
                         + waves * 6) > 0.92) & (fine > 0.45)
        idx[streak] = 4
    elif style == "grass":
        blades = rng.random((size, size)) < 0.06
        idx[blades & (level > 0.45)] = 4
        idx[(rng.random((size, size)) < 0.04) & (level < 0.5)] = 0
    elif style in ("specks", "pebbles"):
        dots = rng.random((size, size))
        idx[dots < 0.035] = 0
        idx[dots > 0.975] = 4
        if style == "pebbles":
            peb = np.argwhere(dots > 0.992)
            for y, x in peb:
                idx[y, (x + 1) % size] = 4
                idx[(y + 1) % size, x] = 0
    elif style == "flagstones":
        stones = max(2, size // 12)
        h = size // stones
        y = np.arange(size)[:, None]
        x = np.arange(size)[None, :]
        row = y // h
        shift = (row % 2) * (h // 2)
        col = ((x + shift) % size) // h
        shade = rng.random((stones + 1, stones + 1))
        idx = np.clip((1 + shade[row % stones, col % stones] * 2.6).astype(int) + (fine > 0.7), 1, 4)
        idx[(y % h == 0) | ((x + shift) % h == 0)] = 0
    out = np.zeros((size, size, 4), np.uint8)
    out[..., :3] = ramp[np.clip(idx, 0, len(ramp) - 1)]
    out[..., 3] = 255
    return out


def texture_from_image(image, size: int = 32, colors: int = 10) -> np.ndarray:
    """A photo / generated picture of a texture -> seamless pixel-art tile (RGBA)."""
    from .pipeline import convert
    from .settings import Settings
    s = Settings(subject="background", width=size, height=size, fit="cover", anchor="center",
                 remove_background=False, crop_to_subject=False, tileable=True, patch=4, detail=0,
                 colors=colors, dither="none", cleanup=False, outline=False)
    return convert(image, s).rgba


TEXTURE_PROMPT = ("seamless tileable texture of {prompt}, seen from directly above, flat even "
                  "lighting, no perspective, no shadows, fills the whole image, game map tile, "
                  "pixel art")


def texture_from_prompt(prompt: str, size: int = 32, *, client=None, seed: int | None = None,
                        colors: int = 10) -> np.ndarray:
    from .generate import generate_images
    (img, _), = generate_images(TEXTURE_PROMPT.format(prompt=prompt),
                                "perspective, horizon, objects, characters, text, border, frame",
                                client=client, seed=seed, aspect=1.0)
    return texture_from_image(img, size, colors)


@dataclass
class Terrain:
    name: str
    texture: np.ndarray          # size x size RGBA, seamless
    edge: tuple[int, int, int]   # colour where it meets the base


def terrain(name: str, source: str | None = None, size: int = 32, *, seed: int = 0, client=None,
            colors: int = 10) -> Terrain:
    """source: None/'builtin' (needs a built-in name), a picture path, or 'prompt:<text>'."""
    if source in (None, "", "builtin"):
        tex = builtin_texture(name, size, seed)
        edge = tuple(int(v) for v in hex_to_rgb([BUILTIN[name][1]])[0])
        return Terrain(name, tex, edge)
    if source.startswith("prompt:"):
        tex = texture_from_prompt(source[7:].strip() or name, size, client=client, seed=seed,
                                  colors=colors)
    else:
        if not Path(source).exists():
            raise FileNotFoundError(f"Texture picture not found: {source}")
        tex = texture_from_image(source, size, colors)
    cols = tex[..., :3].reshape(-1, 3).astype(int)
    lum = cols.sum(1)
    dark = cols[lum <= np.percentile(lum, 10)].mean(0)
    edge = tuple(int(v) for v in np.clip(dark * 0.85, 0, 255))
    return Terrain(name, tex, edge)


# --------------------------------------------------------------------------- transitions

CORNERS = ("top_left_corner", "top_right_corner", "bottom_left_corner", "bottom_right_corner")


def corner_mask(size: int, corners: tuple[int, int, int, int], jitter: np.ndarray,
                amount: float = 0.2) -> np.ndarray:
    """True where the overlay terrain is, for a tile whose corners (tl, tr, bl, br) are
    1 = overlay / 0 = base. Neighbouring tiles agree along their shared edge."""
    t = (np.arange(size) + 0.5) / size
    s = t * t * (3 - 2 * t)
    sx, sy = s[None, :], s[:, None]
    tl, tr, bl, br = corners
    f = tl * (1 - sx) * (1 - sy) + tr * sx * (1 - sy) + bl * (1 - sx) * sy + br * sx * sy
    return (f + amount * (jitter - 0.5)) > 0.5


def blend_tile(base: Terrain, over: Terrain, corners, jitter) -> np.ndarray:
    size = base.texture.shape[0]
    mask = corner_mask(size, corners, jitter)
    out = base.texture.copy()
    out[mask] = over.texture[mask]
    if mask.any() and not mask.all():
        pad = np.pad(mask, 1, mode="edge")
        touch_base = ~pad[:-2, 1:-1] | ~pad[2:, 1:-1] | ~pad[1:-1, :-2] | ~pad[1:-1, 2:]
        touch_over = pad[:-2, 1:-1] | pad[2:, 1:-1] | pad[1:-1, :-2] | pad[1:-1, 2:]
        out[mask & touch_base, :3] = over.edge
        shade = ~mask & touch_over
        out[shade, :3] = (out[shade, :3].astype(np.float32) * 0.78).astype(np.uint8)
    return out


def _decorate(texture: np.ndarray, rng, count: int | None = None) -> np.ndarray:
    """A variant with the same edges (so variants sit next to each other without seams):
    a few small light/dark details away from the border."""
    out = texture.copy()
    size = texture.shape[0]
    cols = texture[..., :3].reshape(-1, 3)
    lum = cols.astype(int).sum(1)
    light, dark = cols[lum.argmax()], cols[lum.argmin()]
    accents = [light, dark]
    mean = cols.mean(0)
    if mean[1] > mean[0] and mean[1] > mean[2]:          # grass: a few flowers too
        accents += [np.array([236, 214, 96]), np.array([240, 240, 232])]
    for _ in range(count or max(2, size // 10)):
        y, x = rng.integers(3, size - 4, 2)
        c = accents[rng.integers(len(accents))]
        for dy, dx in ((0, 0), (0, 1), (1, 0), (-1, 0), (0, -1))[: rng.integers(1, 6)]:
            out[y + dy, x + dx, :3] = c
    return out


@dataclass
class TerrainSet:
    atlas: np.ndarray            # RGBA
    tile: int
    names: list[str]
    colours: list[tuple[int, int, int]]
    tiles: list[dict]            # {"col", "row", "terrain", "corners": (tl, tr, bl, br) terrain ids}

    def lookup(self) -> dict:
        """corners (terrain ids) -> [(col, row), ...] (several for base variants)."""
        out: dict = {}
        for t in self.tiles:
            out.setdefault(tuple(t["corners"]), []).append((t["col"], t["row"]))
        return out


def build_terrain_set(terrains: list[Terrain], variants: int = 4, seed: int = 0) -> TerrainSet:
    """Atlas: row 0 = base variants, then a row per other terrain: its 15 corner
    combinations (bit 1 tl, 2 tr, 4 bl, 8 br; the last one is the full tile)."""
    if len(terrains) < 1:
        raise ValueError("Give at least one terrain.")
    size = terrains[0].texture.shape[0]
    rng = np.random.default_rng(seed)
    jitter = _periodic_noise(size, max(2, size // 8), rng)
    cols = max(15, variants)
    atlas = np.zeros((size * len(terrains), size * cols, 4), np.uint8)
    tiles = []
    base = terrains[0]
    for v in range(variants):
        tile = base.texture if v == 0 else _decorate(base.texture, rng)
        atlas[:size, v * size:(v + 1) * size] = tile
        tiles.append({"col": v, "row": 0, "terrain": 0, "corners": (0, 0, 0, 0)})
    for k, over in enumerate(terrains[1:], start=1):
        for m in range(1, 16):
            bits = ((m >> 0) & 1, (m >> 1) & 1, (m >> 2) & 1, (m >> 3) & 1)
            tile = blend_tile(base, over, bits, jitter)
            atlas[k * size:(k + 1) * size, (m - 1) * size:m * size] = tile
            ids = tuple(k if b else 0 for b in bits)
            tiles.append({"col": m - 1, "row": k, "terrain": k if sum(bits) >= 2 else 0,
                          "corners": ids})
    colours = [tuple(int(c) for c in t.texture[..., :3].reshape(-1, 3).mean(0)) for t in terrains]
    return TerrainSet(atlas, size, [t.name for t in terrains], colours, tiles)


def write_godot_terrain_tileset(ts: TerrainSet, out_tres, atlas_png) -> Path:
    """TileSet .tres: one atlas source, one terrain set (Match Corners) with every terrain,
    each tile tagged with its corner terrains so Godot's terrain painter picks it."""
    out_tres = Path(out_tres)
    lines = ['[gd_resource type="TileSet" load_steps=3 format=3]', "",
             f'[ext_resource type="Texture2D" path="{Path(atlas_png).name}" id="1_atlas"]', "",
             '[sub_resource type="TileSetAtlasSource" id="TileSetAtlasSource_terrain"]',
             'texture = ExtResource("1_atlas")',
             f"texture_region_size = Vector2i({ts.tile}, {ts.tile})"]
    for t in ts.tiles:
        key = f"{t['col']}:{t['row']}/0"
        lines += [f"{key} = 0", f"{key}/terrain_set = 0", f"{key}/terrain = {t['terrain']}"]
        prob = t.get("probability", 0.35 if t["row"] == 0 and t["col"] > 0 else None)
        if prob is not None and prob != 1:
            lines.append(f"{key}/probability = {prob:g}")
        for name, tid in zip(CORNERS, t["corners"]):
            lines.append(f"{key}/terrains_peering_bit/{name} = {tid}")
    # Godot counts empty cells as their own terrain: grass at the map's edge needs tiles whose
    # outer corners are "empty". Alternatives of the first base tile cover every combination.
    lines.append("0:0/next_alternative_id = 16")
    for alt, m in enumerate(range(15), start=1):
        key = f"0:0/{alt}"
        lines += [f"{key} = {alt}", f"{key}/terrain_set = 0", f"{key}/terrain = 0"]
        for bit, name in enumerate(CORNERS):
            if (m >> bit) & 1:                      # set bits = grass, the rest stay empty (-1)
                lines.append(f"{key}/terrains_peering_bit/{name} = 0")
    lines += ["", "[resource]", f"tile_size = Vector2i({ts.tile}, {ts.tile})",
              "terrain_set_0/mode = 1"]
    for i, (name, (r, g, b)) in enumerate(zip(ts.names, ts.colours)):
        lines += [f'terrain_set_0/terrain_{i}/name = "{name}"',
                  f"terrain_set_0/terrain_{i}/color = Color({r / 255:.3f}, {g / 255:.3f}, {b / 255:.3f}, 1)"]
    lines += ['sources/0 = SubResource("TileSetAtlasSource_terrain")', ""]
    out_tres.write_text("\n".join(lines), encoding="utf-8")
    return out_tres


# --------------------------------------------------------------------------- example maps

def example_vertex_grid(cols: int, rows: int, n_terrains: int, seed: int = 0) -> np.ndarray:
    """(rows+1, cols+1) terrain id per tile corner: a lake, a road and patches, with base
    between different terrains so every cell has a tile."""
    rng = np.random.default_rng(seed)
    h, w = rows + 1, cols + 1
    grid = np.zeros((h, w), int)
    yy, xx = np.mgrid[0:h, 0:w]
    if n_terrains > 1:   # lake
        cx, cy = w * 0.7, h * 0.35
        d = ((xx - cx) / (w * 0.22)) ** 2 + ((yy - cy) / (h * 0.28)) ** 2
        grid[d + 0.25 * rng.random((h, w)) < 1.0] = 1
    if n_terrains > 2:   # patch
        cx, cy = w * 0.2, h * 0.7
        d = ((xx - cx) / (w * 0.15)) ** 2 + ((yy - cy) / (h * 0.25)) ** 2
        grid[d + 0.3 * rng.random((h, w)) < 1.0] = 2
    if n_terrains > 3:   # road across the middle, two vertices wide
        path = (h // 2 + np.round(1.5 * np.sin(np.arange(w) / max(1, w) * 6.28))).astype(int)
        for x in range(w):
            grid[np.clip([path[x], path[x] + 1], 0, h - 1), x] = 3
    for k in range(4, n_terrains):
        cx, cy = rng.random() * w, rng.random() * h
        d = ((xx - cx) / 2.5) ** 2 + ((yy - cy) / 2.5) ** 2
        grid[d < 1] = k
    # different non-base terrains may not share a cell: push base between them
    out = grid.copy()
    for y in range(h):
        for x in range(w):
            if grid[y, x]:
                win = grid[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
                if ((win != 0) & (win != grid[y, x])).any() and grid[y, x] > win[win != 0].min():
                    out[y, x] = 0
    return out


def cells_from_vertices(vgrid: np.ndarray, ts: TerrainSet, seed: int = 0) -> np.ndarray:
    """(rows, cols, 2) atlas coords per cell for a corner grid."""
    rng = np.random.default_rng(seed)
    table = ts.lookup()
    rows, cols = vgrid.shape[0] - 1, vgrid.shape[1] - 1
    out = np.zeros((rows, cols, 2), int)
    for y in range(rows):
        for x in range(cols):
            key = (int(vgrid[y, x]), int(vgrid[y, x + 1]), int(vgrid[y + 1, x]), int(vgrid[y + 1, x + 1]))
            choices = table.get(key)
            if choices is None:
                choices = table[(0, 0, 0, 0)]
            out[y, x] = choices[rng.integers(len(choices))] if key == (0, 0, 0, 0) else choices[0]
    return out


def render_map(cells: np.ndarray, ts: TerrainSet) -> np.ndarray:
    t = ts.tile
    rows, cols = cells.shape[:2]
    img = np.zeros((rows * t, cols * t, 4), np.uint8)
    for y in range(rows):
        for x in range(cols):
            c, r = cells[y, x]
            img[y * t:(y + 1) * t, x * t:(x + 1) * t] = ts.atlas[r * t:(r + 1) * t, c * t:(c + 1) * t]
    return img


def write_terrain(ts: TerrainSet, out_png, godot: bool = True, preview: tuple[int, int] | None = (20, 12),
                  seed: int = 0) -> list[Path]:
    """Atlas PNG (+ TileSet .tres) (+ an example map PNG and its cells as JSON)."""
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(ts.atlas, "RGBA").save(out_png)
    written = [out_png]
    if godot:
        written.append(write_godot_terrain_tileset(ts, out_png.with_suffix(".tres"), out_png))
    if preview:
        vg = example_vertex_grid(preview[0], preview[1], len(ts.names), seed)
        cells = cells_from_vertices(vg, ts, seed)
        p = out_png.with_name(out_png.stem + "_example.png")
        Image.fromarray(render_map(cells, ts), "RGBA").save(p)
        j = out_png.with_name(out_png.stem + "_example.json")
        j.write_text(json.dumps({"tile": ts.tile, "terrains": ts.names, "cols": preview[0],
                                 "rows": preview[1], "cells": cells.tolist()}), encoding="utf-8")
        written += [p, j]
    return written

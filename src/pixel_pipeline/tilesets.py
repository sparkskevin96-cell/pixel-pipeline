"""Side-scroller and isometric tiles (map tiles in maptiles.py are top-down).

Side-scroller: one ground terrain (dirt, stone ...) with a top terrain (grass, snow ...) on
every exposed top. 16 tiles cover every combination of open sides, so ground can take any
shape; Godot's terrain painter (Match Sides) picks the right one, and each tile already has
its collision shape. Four one-way platform pieces come with it.

Isometric: for every terrain a flat floor diamond (2:1) and a block (a cube: the terrain on
top, darker sides), in a Godot TileSet with the isometric tile shape.

All terrains come from the same sources as map tiles: built-in, a texture photo, or a prompt.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .maptiles import Terrain, _periodic_noise

TOP, RIGHT, BOTTOM, LEFT = 1, 2, 4, 8            # bits: that neighbour is ground too
SIDES = ((TOP, "top_side"), (RIGHT, "right_side"), (BOTTOM, "bottom_side"), (LEFT, "left_side"))


def _shade(rgb: np.ndarray, k: float) -> np.ndarray:
    return np.clip(rgb.astype(np.float32) * k, 0, 255).astype(np.uint8)


def _darkest(tex: np.ndarray) -> np.ndarray:
    cols = tex[..., :3].reshape(-1, 3)
    return cols[cols.astype(int).sum(1).argmin()]


# --------------------------------------------------------------------------- side-scroller

@dataclass
class SideSet:
    atlas: np.ndarray
    tile: int
    names: tuple[str, str]                 # (ground, top)
    lip: int                               # transparent rows above an exposed top
    masks: dict[int, tuple[int, int]] = field(default_factory=dict)   # mask -> (col, row)
    platforms: dict[str, tuple[int, int]] = field(default_factory=dict)


def side_tile(ground: Terrain, top: Terrain, mask: int, cap_depth: np.ndarray, blades: np.ndarray,
              lip: int) -> np.ndarray:
    """One ground tile; mask bits = which neighbours are ground (the other sides are open)."""
    t = ground.texture.shape[0]
    out = ground.texture.copy()
    edge = np.array(ground.edge, np.uint8)
    top_tex, top_dark = top.texture, _darkest(top.texture)
    y = np.arange(t)[:, None]
    x = np.arange(t)[None, :]
    if not mask & LEFT:
        out[:, :1, :3] = edge
        out[:, 1:3, :3] = _shade(out[:, 1:3, :3], 0.8)
    if not mask & RIGHT:
        out[:, -1:, :3] = edge
        out[:, -3:-1, :3] = _shade(out[:, -3:-1, :3], 0.8)
    if not mask & BOTTOM:
        out[-1:, :, :3] = edge
        out[-3:-1, :, :3] = _shade(out[-3:-1, :, :3], 0.8)
    if not mask & TOP:
        depth = lip + cap_depth[None, :]
        cap = (y >= lip) & (y < depth)
        out[cap] = top_tex[cap]
        out[(y == depth) & (y < t)] = np.append(top_dark, 255)          # shadow under the grass
        # grass hanging down an open side
        for open_side, cols in ((not mask & LEFT, slice(0, 2)), (not mask & RIGHT, slice(t - 2, t))):
            if open_side:
                hang = (y >= lip) & (y < depth + 2)
                region = np.zeros((t, t), bool)
                region[:, cols] = True
                out[hang & region] = top_tex[hang & region]
        out[:lip, :, 3] = 0                                              # air above the ground
        tips = (y < lip) & (y >= lip - blades[None, :])
        out[tips] = top_tex[tips]
        out[tips, 3] = 255
        r = max(1, t // 10)                                              # rounded open corners
        for open_side, cx in ((not mask & LEFT, 0), (not mask & RIGHT, t - 1)):
            if open_side:
                d = (np.abs(x - cx) - r) ** 2 + (y - lip - r) ** 2
                corner = (np.abs(x - cx) < r) & (y < lip + r) & (y >= lip) & (d > r * r)
                out[corner, 3] = 0
    return out


def platform_tile(ground: Terrain, top: Terrain, kind: str, cap_depth: np.ndarray, lip: int) -> np.ndarray:
    """A thin one-way platform piece: 'left', 'middle', 'right' or 'single'."""
    t = ground.texture.shape[0]
    out = np.zeros((t, t, 4), np.uint8)
    body = lip + max(cap_depth.max() + 2, t // 2 - lip)
    y = np.arange(t)[:, None]
    x = np.arange(t)[None, :]
    solid = np.broadcast_to((y >= lip) & (y < body), (t, t)).copy()
    if kind in ("left", "single"):
        solid &= ~((x < 2) & (y > body - 3))
    if kind in ("right", "single"):
        solid &= ~((x > t - 3) & (y > body - 3))
    out[solid] = ground.texture[solid]
    cap = solid & (y < lip + cap_depth[None, :])
    out[cap] = top.texture[cap]
    edge = np.append(np.array(ground.edge, np.uint8), 255)
    out[(y == body - 1) & solid] = edge
    if kind in ("left", "single"):
        out[solid & (x == 0)] = edge
    if kind in ("right", "single"):
        out[solid & (x == t - 1)] = edge
    return out


def build_side_set(ground: Terrain, top: Terrain, seed: int = 0) -> SideSet:
    """16 ground tiles (every set of open sides) in a 4x4 block, then a row of platforms."""
    t = ground.texture.shape[0]
    if top.texture.shape[0] != t:
        raise ValueError("Both terrains need the same tile size")
    rng = np.random.default_rng(seed)
    wave = _periodic_noise(t, max(2, t // 8), rng)[0]                  # wraps: caps line up
    cap_depth = np.round(t * 0.2 + wave * t * 0.12).astype(int)
    lip = max(2, t // 8)
    blades = np.where(rng.random(t) < 0.3, rng.integers(1, lip + 1, t), 0)
    atlas = np.zeros((5 * t, 4 * t, 4), np.uint8)
    masks = {}
    for m in range(16):
        c, r = m % 4, m // 4
        atlas[r * t:(r + 1) * t, c * t:(c + 1) * t] = side_tile(ground, top, m, cap_depth, blades, lip)
        masks[m] = (c, r)
    platforms = {}
    for c, kind in enumerate(("left", "middle", "right", "single")):
        atlas[4 * t:5 * t, c * t:(c + 1) * t] = platform_tile(ground, top, kind, cap_depth, lip)
        platforms[kind] = (c, 4)
    return SideSet(atlas, t, (ground.name, top.name), lip, masks, platforms)


def side_masks(solid: np.ndarray) -> np.ndarray:
    """Per cell of a ground grid (True = ground): the neighbour mask (outside = open)."""
    s = np.pad(solid, 1)
    m = (s[:-2, 1:-1] * TOP) + (s[1:-1, 2:] * RIGHT) + (s[2:, 1:-1] * BOTTOM) + (s[1:-1, :-2] * LEFT)
    return np.where(solid, m, -1)


def example_level(cols: int = 20, rows: int = 11, seed: int = 0) -> tuple[np.ndarray, list]:
    """A small level: rolling ground with a pit, a step and two floating platforms.
    Returns (solid grid, [(col, row, kind)] platforms)."""
    rng = np.random.default_rng(seed)
    solid = np.zeros((rows, cols), bool)
    height = np.full(cols, 3)
    height[cols // 2:] = 4
    height[3:5] = 2
    pit = slice(int(cols * 0.65), int(cols * 0.65) + 2)
    height[pit] = 0
    for x in range(cols):
        solid[rows - height[x]:, x] = True
    solid[rows - 6, 2:4] = True                                            # a floating block
    plats = []
    y = rows - 6
    start = int(cols * 0.6) + int(rng.integers(0, 2))
    plats += [(start, y, "left"), (start + 1, y, "middle"), (start + 2, y, "right")]
    plats += [(cols - 3, rows - 8, "single")]
    return solid, plats


def render_level(ss: SideSet, solid: np.ndarray, plats: list) -> np.ndarray:
    t = ss.tile
    rows, cols = solid.shape
    img = np.zeros((rows * t, cols * t, 4), np.uint8)
    masks = side_masks(solid)
    for yy in range(rows):
        for xx in range(cols):
            if masks[yy, xx] >= 0:
                c, r = ss.masks[int(masks[yy, xx])]
                img[yy * t:(yy + 1) * t, xx * t:(xx + 1) * t] = ss.atlas[r * t:(r + 1) * t, c * t:(c + 1) * t]
    for xx, yy, kind in plats:
        c, r = ss.platforms[kind]
        img[yy * t:(yy + 1) * t, xx * t:(xx + 1) * t] = ss.atlas[r * t:(r + 1) * t, c * t:(c + 1) * t]
    return img


def write_godot_side_tileset(ss: SideSet, out_tres, atlas_png) -> Path:
    """TileSet with a Match Sides terrain (paint ground with the Terrains tab) and collision:
    ground tiles solid from the top of the grass down, platforms one-way."""
    t, h = ss.tile, ss.tile / 2
    top = -h + ss.lip

    def box(y0, y1):
        return f"PackedVector2Array({-h:g}, {y0:g}, {h:g}, {y0:g}, {h:g}, {y1:g}, {-h:g}, {y1:g})"
    lines = ['[gd_resource type="TileSet" load_steps=3 format=3]', "",
             f'[ext_resource type="Texture2D" path="{Path(atlas_png).name}" id="1_atlas"]', "",
             '[sub_resource type="TileSetAtlasSource" id="TileSetAtlasSource_side"]',
             'texture = ExtResource("1_atlas")', f"texture_region_size = Vector2i({t}, {t})"]
    for m, (c, r) in ss.masks.items():
        key = f"{c}:{r}/0"
        lines += [f"{key} = 0", f"{key}/terrain_set = 0", f"{key}/terrain = 0",
                  f"{key}/physics_layer_0/polygon_0/points = {box(-h if m & TOP else top, h)}"]
        lines += [f"{key}/terrains_peering_bit/{name} = 0" for bit, name in SIDES if m & bit]
    plat_bottom = top + max(4, t // 4)
    for kind, (c, r) in ss.platforms.items():
        key = f"{c}:{r}/0"
        lines += [f"{key} = 0", f"{key}/physics_layer_0/polygon_0/points = {box(top, plat_bottom)}",
                  f"{key}/physics_layer_0/polygon_0/one_way = true"]
    r_, g_, b_ = (int(v) for v in ss.atlas[ss.tile // 2, ss.tile // 2, :3])
    lines += ["", "[resource]", f"tile_size = Vector2i({t}, {t})",
              "physics_layer_0/collision_layer = 1", "terrain_set_0/mode = 2",
              f'terrain_set_0/terrain_0/name = "{ss.names[0]}"',
              f"terrain_set_0/terrain_0/color = Color({r_ / 255:.3f}, {g_ / 255:.3f}, {b_ / 255:.3f}, 1)",
              'sources/0 = SubResource("TileSetAtlasSource_side")', ""]
    out_tres = Path(out_tres)
    out_tres.write_text("\n".join(lines), encoding="utf-8")
    return out_tres


def write_side(ss: SideSet, out_png, godot: bool = True, example: bool = True, seed: int = 0) -> list[Path]:
    """Atlas PNG (+ TileSet) (+ an example level PNG and JSON)."""
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(ss.atlas, "RGBA").save(out_png)
    written = [out_png]
    if godot:
        written.append(write_godot_side_tileset(ss, out_png.with_suffix(".tres"), out_png))
    if example:
        solid, plats = example_level(seed=seed)
        p = out_png.with_name(out_png.stem + "_example.png")
        Image.fromarray(render_level(ss, solid, plats), "RGBA").save(p)
        j = out_png.with_name(out_png.stem + "_example.json")
        j.write_text(json.dumps({"tile": ss.tile, "ground": [[int(x), int(y)] for y, x in np.argwhere(solid)],
                                 "platforms": [[x, y, k, *ss.platforms[k]] for x, y, k in plats]}),
                     encoding="utf-8")
        written += [p, j]
    return written


# --------------------------------------------------------------------------- isometric

@dataclass
class IsoSet:
    floors: np.ndarray                     # row of W x H diamonds
    blocks: np.ndarray                     # row of W x (H + depth) cubes
    width: int
    names: list[str]

    @property
    def height(self) -> int:
        return self.width // 2


def _diamond_uv(w: int):
    """For a W x W/2 diamond: texture (u, v) in [0, 1) per pixel and the inside mask."""
    h = w // 2
    px = np.arange(w)[None, :] + 0.5
    py = np.arange(h)[:, None] + 0.5
    a = (px - w / 2) / (w / 2)
    b = py / (h / 2)
    u, v = (a + b) / 2, (b - a) / 2
    inside = (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
    return u, v, inside


def iso_floor(tex: np.ndarray, w: int) -> np.ndarray:
    t = tex.shape[0]
    u, v, inside = _diamond_uv(w)
    out = np.zeros((w // 2, w, 4), np.uint8)
    iu = np.clip((u * t).astype(int), 0, t - 1)
    iv = np.clip((v * t).astype(int), 0, t - 1)
    out[inside] = tex[iv[inside], iu[inside]]
    out[~inside] = 0
    return out


def iso_block(top_tex: np.ndarray, side_tex: np.ndarray, w: int, depth: int | None = None) -> np.ndarray:
    """A cube: the floor diamond on top, the left side x0.72 and the right side x0.55."""
    h = w // 2
    d = depth or h
    t = side_tex.shape[0]
    out = np.zeros((h + d, w, 4), np.uint8)
    x = np.arange(w)[None, :]
    y = np.arange(h + d)[:, None]
    left_top = h / 2 + x / 2                     # upper edge of the left face (x < w/2)
    right_top = h - (x - w / 2) / 2              # upper edge of the right face (x >= w/2)
    for is_left, top_edge, k in ((True, left_top, 0.72), (False, right_top, 0.55)):
        side = (x < w / 2) if is_left else (x >= w / 2)
        face = side & (y >= np.floor(top_edge)) & (y < np.floor(top_edge) + d)
        fx = (x if is_left else x - w // 2) * t // (w // 2)
        fy = np.clip(((y - np.floor(top_edge)) * t / d).astype(int), 0, t - 1)
        tex = side_tex[fy, np.broadcast_to(np.clip(fx, 0, t - 1), fy.shape)]
        out[face] = tex[face]
        out[face, :3] = _shade(out[face, :3], k)
    floor = iso_floor(top_tex, w)
    on = floor[..., 3] > 0
    out[:h][on] = floor[on]
    edge = np.append(_shade(_darkest(side_tex)[None], 0.7)[0], 255)
    out[h:, w // 2 - 1:w // 2 + 1][out[h:, w // 2 - 1:w // 2 + 1, 3] > 0] = edge  # front corner line
    return out


def build_iso_set(terrains: list[Terrain], width: int = 32) -> IsoSet:
    if width % 4:
        raise ValueError("Isometric tile width must be a multiple of 4 (e.g. 32 or 64)")
    floors = np.concatenate([iso_floor(t.texture, width) for t in terrains], 1)
    blocks = np.concatenate([iso_block(t.texture, _side_texture(t), width) for t in terrains], 1)
    return IsoSet(floors, blocks, width, [t.name for t in terrains])


def _side_texture(t: Terrain) -> np.ndarray:
    """Sides of a grass block show dirt; other terrains show themselves."""
    if t.name in ("grass", "snow"):
        from .maptiles import builtin_texture
        return builtin_texture("dirt", t.texture.shape[0])
    return t.texture


def example_iso(cols: int = 10, rows: int = 10, n: int = 3, seed: int = 0):
    """(floor terrain per cell, block height per cell) for a small island."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:rows, 0:cols]
    d = np.hypot(xx - cols / 2 + 0.5, yy - rows / 2 + 0.5)
    floor = np.where(d < cols * 0.42, 0, min(1, n - 1))
    if n > 2:
        floor[(xx + yy) % 7 == 3] = 2
    blocks = np.zeros((rows, cols), int)
    for _ in range(4):
        x, y = rng.integers(2, cols - 2), rng.integers(2, rows - 2)
        blocks[y, x] = 1
    return floor, blocks


def render_iso(iso: IsoSet, floor: np.ndarray, blocks: np.ndarray) -> np.ndarray:
    w, h = iso.width, iso.height
    rows, cols = floor.shape
    pad = h
    img = Image.new("RGBA", ((cols + rows) * w // 2, (cols + rows) * h // 2 + h + pad), (0, 0, 0, 0))
    for s in range(rows + cols - 1):            # back to front
        for y in range(rows):
            x = s - y
            if not 0 <= x < cols:
                continue
            sx = (x - y) * w // 2 + (rows - 1) * w // 2
            sy = (x + y) * h // 2 + pad
            k = int(floor[y, x])
            tile = iso.floors[:, k * w:(k + 1) * w]
            img.alpha_composite(Image.fromarray(np.ascontiguousarray(tile)), (sx, sy))
            if blocks[y, x]:
                b = iso.blocks[:, k * w:(k + 1) * w]
                img.alpha_composite(Image.fromarray(np.ascontiguousarray(b)), (sx, sy - h))
    return np.asarray(img)


def write_godot_iso_tileset(iso: IsoSet, out_tres, floors_png, blocks_png) -> Path:
    """Isometric TileSet (diamond-down layout): source 0 = floors, source 1 = blocks (taller
    than a cell, so their texture origin lifts them onto it)."""
    w, h = iso.width, iso.height
    bh = iso.blocks.shape[0]
    lines = ['[gd_resource type="TileSet" load_steps=5 format=3]', "",
             f'[ext_resource type="Texture2D" path="{Path(floors_png).name}" id="1_floors"]',
             f'[ext_resource type="Texture2D" path="{Path(blocks_png).name}" id="2_blocks"]', "",
             '[sub_resource type="TileSetAtlasSource" id="TileSetAtlasSource_floors"]',
             'texture = ExtResource("1_floors")', f"texture_region_size = Vector2i({w}, {h})"]
    lines += [f"{i}:0/0 = 0" for i in range(len(iso.names))]
    lines += ["", '[sub_resource type="TileSetAtlasSource" id="TileSetAtlasSource_blocks"]',
              'texture = ExtResource("2_blocks")', f"texture_region_size = Vector2i({w}, {bh})"]
    for i in range(len(iso.names)):
        lines += [f"{i}:0/0 = 0", f"{i}:0/0/texture_origin = Vector2i(0, {(bh - h) // 2})"]
    lines += ["", "[resource]", "tile_shape = 1", "tile_layout = 5", f"tile_size = Vector2i({w}, {h})",
              'sources/0 = SubResource("TileSetAtlasSource_floors")',
              'sources/1 = SubResource("TileSetAtlasSource_blocks")', ""]
    out_tres = Path(out_tres)
    out_tres.write_text("\n".join(lines), encoding="utf-8")
    return out_tres


def write_iso(iso: IsoSet, out_png, godot: bool = True, example: bool = True, seed: int = 0) -> list[Path]:
    """<name>.png floors, <name>_blocks.png, TileSet .tres, example render + JSON."""
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    blocks_png = out_png.with_name(out_png.stem + "_blocks.png")
    Image.fromarray(iso.floors, "RGBA").save(out_png)
    Image.fromarray(iso.blocks, "RGBA").save(blocks_png)
    written = [out_png, blocks_png]
    if godot:
        written.append(write_godot_iso_tileset(iso, out_png.with_suffix(".tres"), out_png, blocks_png))
    if example:
        floor, blocks = example_iso(n=len(iso.names), seed=seed)
        p = out_png.with_name(out_png.stem + "_example.png")
        Image.fromarray(render_iso(iso, floor, blocks), "RGBA").save(p)
        j = out_png.with_name(out_png.stem + "_example.json")
        j.write_text(json.dumps({"width": iso.width, "terrains": iso.names, "floor": floor.tolist(),
                                 "blocks": blocks.tolist()}), encoding="utf-8")
        written += [p, j]
    return written


# --------------------------------------------------------------------------- one entry point

STYLES = ("topdown", "sidescroller", "isometric")
DEFAULT_TERRAINS = {"topdown": ["grass", "water", "sand", "road"], "sidescroller": ["dirt", "grass"],
                    "isometric": ["grass", "water", "stone", "sand"]}


def build_style(style: str, terrains: list[Terrain], variants: int = 4, seed: int = 0):
    """topdown -> TerrainSet, sidescroller -> SideSet (ground, then its top), isometric -> IsoSet."""
    if style == "topdown":
        from .maptiles import build_terrain_set
        return build_terrain_set(terrains, variants, seed)
    if style == "sidescroller":
        if not terrains:
            raise ValueError("Give the ground terrain, then the one on top (e.g. dirt, grass)")
        return build_side_set(terrains[0], terrains[1] if len(terrains) > 1 else terrains[0], seed)
    if style == "isometric":
        return build_iso_set(terrains, terrains[0].texture.shape[0])
    raise ValueError(f"style must be one of {', '.join(STYLES)}")


def write_style(style: str, built, out_png, godot: bool = True, example: bool = True,
                seed: int = 0) -> list[Path]:
    if style == "topdown":
        from .maptiles import write_terrain
        return write_terrain(built, out_png, godot, (20, 11) if example else None, seed)
    if style == "sidescroller":
        return write_side(built, out_png, godot, example, seed)
    return write_iso(built, out_png, godot, example, seed)


def style_preview(style: str, built, seed: int = 0) -> np.ndarray:
    """An example map in this style (RGBA)."""
    if style == "topdown":
        from .maptiles import cells_from_vertices, example_vertex_grid, render_map
        return render_map(cells_from_vertices(example_vertex_grid(20, 11, len(built.names), seed), built,
                                              seed), built)
    if style == "sidescroller":
        return render_level(built, *example_level(seed=seed))
    return render_iso(built, *example_iso(n=len(built.names), seed=seed))

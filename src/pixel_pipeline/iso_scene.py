"""The isometric 2D look of the tactics demo (Tactics Ogre / Disgaea style) for a Diorama.

    write_iso_scene(harbour(), "game", {"knight": "knight.png", "dog": "dog.png"})

writes `game/iso/` (block atlas, water frames, prop sprites, unit sheets, iso.json, iso.tscn,
iso.gd) and `game/map/map.json` if it is missing; `game/project.godot` only when the folder has
none (then iso.tscn is the main scene). The scene builds the board at runtime from map.json, so
the same map drives the 3D and fake-3D looks.

Geometry: tile W wide, diamonds W x W/2, one height level = W/4 px (half a block). Cell (c, r)
at level h has its top-diamond centre at ((c - r) * W/2, (c + r) * W/4 - h * W/4). Every cell is
ONE pre-rendered column sprite (top diamond + sides from its height down to BASE levels below
the water), so stacked heights have no seams. Painter's order = z_index: (c + r) * 8 + layer
(block 0, water 1, range 2, shadow 3, cursor 4, prop/unit 5): back diagonals first; anything
standing on a cell is drawn after that cell's column and before every cell in front of it.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from .diorama import BLOCKING_PROPS, LEVEL, TERRAINS, WALKABLE, Diorama, textures
from .tilesets import _darkest, _shade, iso_floor

BASE = -2                  # columns reach this many levels below level 0 (the diorama's skirt)
WATER_FRAMES = 4
TEAM_COLOURS = ["#5a9bff", "#ff5a4d", "#5ee06a", "#ffd34d"]
DIRECTIONS = ("E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW",
              "W", "WNW", "NW", "NNW", "N", "NNE", "NE", "ENE")


# --------------------------------------------------------------------------- blocks

def _rim(floor: np.ndarray) -> np.ndarray:
    """Light upper edges, dark lower edges on a top diamond: the bevel that makes cells readable."""
    out = floor.copy()
    inside = out[..., 3] > 0
    up = np.zeros_like(inside)
    up[1:] = inside[:-1]
    down = np.zeros_like(inside)
    down[:-1] = inside[1:]
    low = inside & ~down
    high = inside & ~up
    out[low, :3] = _shade(out[low, :3], 0.78)
    out[high, :3] = _shade(out[high, :3], 1.18)
    return out


def iso_column(top: np.ndarray, side: np.ndarray, w: int, levels: int, level_px: int,
               top_level: int = 0) -> np.ndarray:
    """A column W wide: the top diamond, then `levels` levels of side (left x0.72, right x0.55)
    with the side texture at 1:1 scale with the top (no stretching however tall), a faint line
    at every level step (so heights can be counted) and a dark front corner."""
    h = w // 2
    d = max(0, levels) * level_px
    out = np.zeros((h + d, w, 4), np.uint8)
    t = side.shape[0]
    x = np.arange(w)[None, :]
    y = np.arange(h + d)[:, None]
    for is_left, top_edge, k in ((True, h / 2 + x / 2, 0.72), (False, h - (x - w / 2) / 2, 0.55)):
        on_side = (x < w / 2) if is_left else (x >= w / 2)
        y0 = np.floor(top_edge)
        face = on_side & (y >= y0) & (y < y0 + d)
        rel = (y - y0).astype(int)
        fx = np.clip(((x if is_left else x - w // 2) * t) // (w // 2), 0, t - 1)
        fy = ((rel - top_level * level_px) * t * 2 // w) % t
        tex = side[fy, np.broadcast_to(fx, fy.shape)]
        out[face] = tex[face]
        out[face, :3] = _shade(out[face, :3], k)
        step = face & (rel % level_px == 0) & (rel > 0)
        out[step, :3] = _shade(out[step, :3], 0.84)
    if d:
        edge = np.append(_shade(_darkest(side)[None], 0.7)[0], 255)
        mid = out[h:, w // 2 - 1:w // 2 + 1]
        mid[mid[..., 3] > 0] = edge                                   # front corner line
    floor = _rim(iso_floor(top, w))
    on = floor[..., 3] > 0
    out[:h][on] = floor[on]
    return out


def _water_textures(size: int, seed: int):
    from .maptiles import builtin_texture
    water = builtin_texture("water", size, seed)
    water[..., :3] = np.clip(water[..., :3].astype(np.float32) * (0.8, 0.86, 0.9), 0, 255).astype(np.uint8)
    bed = water.copy()
    bed[..., :3] = _shade(bed[..., :3], 0.55)
    wall = water.copy()
    g = np.linspace(0.62, 0.38, size)[:, None, None]
    wall[..., :3] = np.clip(wall[..., :3].astype(np.float32) * g, 0, 255).astype(np.uint8)
    return water, bed, wall


def build_blocks(d: Diorama, tile: int = 32, seed: int = 0, max_level: int | None = None):
    """Atlas of every terrain x every height: (RGBA atlas, {terrain: {height: [x, y, w, h]}})."""
    lp = tile // 4
    lo = min(0, int(d.height.min()))
    hi = max(int(d.height.max()), max_level if max_level is not None else 8)
    tex = textures(None, tile, seed)
    _, bed, wall = _water_textures(tile, seed)
    tex["water"] = (bed, wall)
    names = list(TERRAINS)
    slot_h = tile // 2 + (hi - BASE) * lp
    atlas = np.zeros((slot_h * len(names), tile * (hi - lo + 1), 4), np.uint8)
    rects: dict = {}
    for ti, name in enumerate(names):
        top, side = tex[name]
        rects[name] = {}
        for k, level in enumerate(range(lo, hi + 1)):
            col = iso_column(top, side, tile, level - BASE, lp, level)
            x, y = k * tile, ti * slot_h
            atlas[y:y + col.shape[0], x:x + tile] = col
            rects[name][str(level)] = [x, y, tile, col.shape[0]]
    return atlas, rects


def water_frames(tile: int = 32, seed: int = 0, frames: int = WATER_FRAMES) -> np.ndarray:
    """Water surface diamonds, the seamless texture drifting diagonally: a strip of frames."""
    water, _, _ = _water_textures(tile, seed)
    rng = np.random.default_rng(seed + 99)
    out = []
    for f in range(frames):
        t = np.roll(water, (f * tile // frames, f * tile // (2 * frames)), (0, 1)).copy()
        glints = rng.random((tile, tile)) < 0.005
        t[glints, :3] = (200, 226, 240)
        out.append(iso_floor(t, tile))
    return np.concatenate(out, 1)


# --------------------------------------------------------------------------- props + overlays

def _ellipse(w: int, h: int, cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w]
    return ((x + 0.5 - cx) / rx) ** 2 + ((y + 0.5 - cy) / ry) ** 2 <= 1


def _wood(n: int, seed: int, base=(140, 96, 58)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((n, n, 4), np.uint8)
    img[..., 3] = 255
    tone = rng.uniform(0.85, 1.1, (n, 1))
    img[..., :3] = np.clip(np.array(base)[None, None] * tone[..., None], 0, 255)
    img[::4, :, :3] = _shade(img[::4, :, :3], 0.7)
    return img


def crate_sprite(w: int) -> np.ndarray:
    """A small wooden crate: a cube with a dark frame on every face."""
    from .tilesets import iso_block
    n = 16
    t = _wood(n, 3)
    t[:2] = t[-2:] = (78, 50, 30, 255)
    t[:, :2] = t[:, -2:] = (78, 50, 30, 255)
    for i in range(n):                                   # diagonal brace
        t[i, max(0, min(n - 1, i))] = (96, 62, 36, 255)
    top = t.copy()
    top[..., :3] = _shade(top[..., :3], 1.15)
    return iso_block(top, t, w)


def barrel_sprite(w: int) -> np.ndarray:
    hgt = int(w * 1.45)
    out = np.zeros((hgt, w, 4), np.uint8)
    ry = max(2, w // 4)
    body = np.zeros((hgt, w), bool)
    body[ry:hgt - ry] = True
    body |= _ellipse(w, hgt, w / 2, hgt - ry, w / 2, ry)
    ramp = np.array([[92, 58, 34], [128, 84, 48], [158, 108, 64], [124, 80, 46], [86, 54, 32], [64, 40, 26]])
    xs = np.clip(((np.arange(w) + 0.5) / w * len(ramp)).astype(int), 0, len(ramp) - 1)
    out[body, :3] = ramp[xs][None].repeat(hgt, 0)[body]
    out[body, 3] = 255
    for hy in (ry + 2, hgt - ry - 3):                    # iron hoops
        rows = slice(hy, hy + 2)
        m = body[rows]
        out[rows][m] = (58, 56, 60, 255)
    lid = _ellipse(w, hgt, w / 2, ry, w / 2, ry)
    out[lid] = (170, 122, 76, 255)
    inner = _ellipse(w, hgt, w / 2, ry, w / 2 - 1.5, ry - 1)
    out[lid & ~inner] = (96, 62, 36, 255)
    return out


def post_sprite(w: int, hgt: int) -> np.ndarray:
    out = np.zeros((hgt, w, 4), np.uint8)
    ramp = [(150, 110, 70), (112, 78, 48), (80, 54, 34), (62, 42, 28)]
    for x in range(w):
        out[:, x] = (*ramp[min(len(ramp) - 1, x * len(ramp) // w)], 255)
    out[0] = (176, 136, 92, 255)
    out[hgt // 3, :, :3] = _shade(out[hgt // 3, :, :3], 0.75)
    return out


def torch_frames(tile: int, frames: int = 4) -> np.ndarray:
    """Torch on a pole with a flickering flame: a strip of `frames` frames."""
    w = max(10, tile // 3)
    hgt = int(tile * 0.95)
    pole = post_sprite(3, hgt - 10)
    rng = np.random.default_rng(5)
    out = []
    for f in range(frames):
        img = np.zeros((hgt, w, 4), np.uint8)
        x0 = w // 2 - 1
        img[10:, x0:x0 + 3] = pole
        img[9:11, x0 - 1:x0 + 4] = (70, 66, 70, 255)         # iron cup
        flame_h = 7 + (f % 2)
        for y in range(flame_h):
            k = y / flame_h                                   # 0 at the tip
            half = max(0.5, k * 2.6 + rng.uniform(-0.4, 0.4))
            cx = w / 2 + 0.5 * np.sin(f * 1.7 + y) * (1 - k)
            for x in range(w):
                dx = abs(x + 0.5 - cx)
                if dx <= half:
                    col = (255, 244, 170) if dx < half * 0.45 and k > 0.35 else \
                          (255, 170, 40) if dx < half * 0.8 else (214, 82, 30)
                    img[9 - flame_h + y, x] = (*col, 255)
        out.append(img)
    return np.concatenate(out, 1)


def glow_sprite(w: int) -> np.ndarray:
    h = w // 2
    out = np.zeros((h, w, 4), np.uint8)
    for rx, a in ((1.0, 14), (0.66, 14), (0.36, 18)):
        m = _ellipse(w, h, w / 2, h / 2, w / 2 * rx, h / 2 * rx)
        out[m] = (255, 150, 60, 0)
        out[m, 3] = np.minimum(255, out[m, 3].astype(int) + a)
    return out


def shadow_sprite(w: int) -> np.ndarray:
    h = max(4, w // 3)
    out = np.zeros((h, w, 4), np.uint8)
    out[_ellipse(w, h, w / 2, h / 2, w / 2, h / 2)] = (0, 0, 0, 70)
    out[_ellipse(w, h, w / 2, h / 2, w / 2 - 2, h / 2 - 1)] = (0, 0, 0, 100)
    return out


def _diamond_mask(w: int) -> np.ndarray:
    return iso_floor(np.full((8, 8, 4), 255, np.uint8), w)[..., 3] > 0


def _outline(mask: np.ndarray) -> np.ndarray:
    m = np.pad(mask, 1)
    inner = m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
    return mask & ~inner


def range_sprite(w: int) -> np.ndarray:
    """Purple movement-range diamond with a lighter edge."""
    mask = _diamond_mask(w)
    out = np.zeros((w // 2, w, 4), np.uint8)
    out[mask] = (150, 84, 255, 140)
    out[_outline(mask)] = (200, 160, 255, 230)
    return out


def cursor_sprite(w: int) -> np.ndarray:
    """White diamond outline (modulated per use: hover cursor, team ring under a unit)."""
    mask = _diamond_mask(w)
    out = np.zeros((w // 2, w, 4), np.uint8)
    out[_outline(mask)] = (255, 255, 255, 255)
    return out


def pawn_sheet(folder: Path, name: str = "pawn") -> tuple[Path, Path]:
    """A plain stand-in unit (for map units without a sheet): idle + walk in all 16 directions."""
    fw, fh = 24, 32
    base = np.zeros((fh, fw, 4), np.uint8)
    base[_ellipse(fw, fh, fw / 2, 9, 4.5, 4.5)] = (226, 196, 160, 255)            # head
    body = _ellipse(fw, fh, fw / 2, 21, 6, 8) & (np.mgrid[0:fh, 0:fw][0] < 28)
    base[body] = (150, 150, 168, 255)
    base[28:30, 8:11] = base[28:30, 13:16] = (60, 60, 70, 255)                    # feet
    frames, rects, names = [], [], []
    for anim, n in (("idle", 1), ("walk", 2)):
        for dname in DIRECTIONS:
            for i in range(n):
                f = np.roll(base, -i, 0) if anim == "walk" else base
                rects.append((len(frames) * fw, 0, fw, fh))
                frames.append(f)
                names.append(f"{anim}_{dname} {i}")
    folder.mkdir(parents=True, exist_ok=True)
    png = folder / f"{name}.png"
    Image.fromarray(np.concatenate(frames, 1)).save(png)
    data = {"frames": [{"filename": nm, "frame": {"x": x, "y": y, "w": w, "h": h}, "duration": 250}
                       for nm, (x, y, w, h) in zip(names, rects)],
            "meta": {"image": png.name, "size": {"w": len(frames) * fw, "h": fh}}}
    js = folder / f"{name}.json"
    js.write_text(json.dumps(data), encoding="utf-8")
    return png, js


# --------------------------------------------------------------------------- unit sheets

def _sheet_frames(data: dict) -> list[tuple[str, dict]]:
    fr = data.get("frames", [])
    if isinstance(fr, dict):
        return [(k, v["frame"]) for k, v in fr.items()]
    return [(f["filename"], f["frame"]) for f in fr]


def unit_metrics(png, js) -> dict:
    """Where a sheet's feet are (the ground point in a frame) and the body box around them,
    from the idle frames (or all frames): {"foot": [x, y], "body": [x0, y0, x1, y1], "size"}."""
    img = np.array(Image.open(png).convert("RGBA"))
    frames = _sheet_frames(json.loads(Path(js).read_text(encoding="utf-8")))
    use = [f for n, f in frames if n.startswith("idle_")] or [f for _, f in frames]
    boxes = []
    for f in use:
        a = img[f["y"]:f["y"] + f["h"], f["x"]:f["x"] + f["w"], 3] > 0
        ys, xs = np.nonzero(a)
        if len(ys):
            boxes.append((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    fw, fh = (use[0]["w"], use[0]["h"]) if use else (32, 32)
    if not boxes:
        return {"foot": [fw // 2, fh - 1], "body": [-8, -fh, 8, 0], "size": [fw, fh]}
    b = np.median(np.array(boxes), 0)
    foot = [fw // 2, int(round(b[3])) - 1]
    body = [int(b[0]) - foot[0], int(b[1]) - foot[1], int(b[2]) - foot[0], int(b[3]) - foot[1]]
    return {"foot": foot, "body": body, "size": [fw, fh]}


def _copy_unit(name: str, sheet, folder: Path) -> tuple[dict, list[Path]]:
    sheet = Path(sheet)
    if not sheet.exists():
        raise FileNotFoundError(f"unit sheet for '{name}' not found: {sheet}")
    js = sheet.with_suffix(".json")
    if not js.exists():
        raise FileNotFoundError(f"unit sheet '{sheet}' needs its frame list {js.name} next to it")
    dst = folder / name
    if dst.exists():
        shutil.rmtree(dst)                       # no stale sheets from an earlier unit
    dst.mkdir(parents=True)
    written = []
    for src in (sheet, js, sheet.with_suffix(".tres")):
        if src.exists():
            written.append(Path(shutil.copyfile(src, dst / src.name)))
    res = f"res://iso/units/{name}/"
    tres = sheet.with_suffix(".tres")
    entry = {"png": res + sheet.name, "json": res + js.name,
             "tres": res + tres.name if tres.exists() else "", **unit_metrics(sheet, js)}
    return entry, written


# --------------------------------------------------------------------------- Godot files

PROJECT = """config_version=5

[application]

config/name="{name}"
run/main_scene="res://iso/iso.tscn"
config/features=PackedStringArray("4.3", "GL Compatibility")

[display]

window/size/viewport_width=640
window/size/viewport_height=360
window/size/window_width_override=1280
window/size/window_height_override=720
window/stretch/mode="viewport"
window/stretch/scale_mode="integer"

[rendering]

renderer/rendering_method="gl_compatibility"
renderer/rendering_method.mobile="gl_compatibility"
textures/canvas_textures/default_texture_filter=0
2d/snap/snap_2d_transforms_to_pixel=true
2d/snap/snap_2d_vertices_to_pixel=true
"""

TSCN = """[gd_scene load_steps=2 format=3]

[ext_resource type="Script" path="res://iso/iso.gd" id="1_iso"]

[node name="Iso" type="Node2D"]
texture_filter = 1
script = ExtResource("1_iso")
"""

WATER_SHADER = """shader_type canvas_item;
// Water surface: the frames drift (iso.gd swaps them); here a few pixels glint, picked per
// world pixel and changing in steps (no smooth blending: stays pixel art).
varying vec2 world;

void vertex() {
\tworld = (MODEL_MATRIX * vec4(VERTEX, 0.0, 1.0)).xy;
}

float hash(vec2 p) {
\treturn fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

void fragment() {
\tvec4 c = texture(TEXTURE, UV);
\tfloat t = floor(TIME * 3.0);
\tfloat h = hash(floor(world) + vec2(t * 7.0, t * 3.0));
\tif (c.a > 0.0 && h > 0.997) {
\t\tc.rgb = mix(c.rgb, vec3(0.86, 0.95, 1.0), 0.75);
\t}
\tCOLOR = c;
}
"""

ISO_GD = r'''extends Node2D
## Isometric tactics view (written by pixel-pipeline iso_scene.py; regenerate rather than edit).
## Board from res://map/map.json, look from res://iso/iso.json.
## Click a unit of the active team -> purple range; click a purple cell -> it walks there and
## the turn passes. Tab/Space = end turn, Esc/right click = deselect, WASD/arrows = pan,
## mouse wheel = zoom (integer steps).
##
## Painter's order: every sprite's z_index = (col + row) * K + layer, so back diagonals are
## drawn first and whatever stands on a cell is drawn after its column but before the cells
## in front of it (which may hide it, e.g. a unit behind a wall).

const MAP_PATH := "res://map/map.json"
const CONF_PATH := "res://iso/iso.json"
const DIRS: Array[String] = ["E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW", "N", "NNE", "NE", "ENE"]
const WALKABLE: Array[String] = ["grass", "stone", "cobble", "planks", "dirt", "sand"]
const BLOCKING: Array[String] = ["crate", "barrel", "post"]
const STEPS: Array[Vector2i] = [Vector2i(1, 0), Vector2i(-1, 0), Vector2i(0, 1), Vector2i(0, -1)]
const TEAM_NAMES: Array[String] = ["Blue", "Red", "Green", "Gold"]
const K := 8
const Z_BLOCK := 0
const Z_WATER := 1
const Z_RANGE := 2
const Z_SHADOW := 3
const Z_CURSOR := 4
const Z_THING := 5
const STEP_TIME := 0.28
const PAN_SPEED := 220.0

var conf: Dictionary = {}
var cols := 0
var rows := 0
var heights: Array = []
var terrain: Array = []
var blocked := {}
var tile := 32
var hw := 16
var hh := 8
var lv := 8
var base := -2
var water_px := 2
var move_range := 4
var jump := 2

var units: Array = []          # Dictionaries: node sprite ring name team cell from path t facing action body
var teams: Array = []
var active_team := 0
var selected: Variant = null   # a unit Dictionary or null
var reach := {}                # Vector2i -> steps
var came_from := {}
var busy := false

var cam: Camera2D
var zoom_level := 2
var hud: Label
var cursor: Sprite2D
var range_layer: Node2D
var range_tex: Texture2D
var ring_tex: Texture2D
var water_tex: AtlasTexture
var water_frames := 4
var water_clock := 0.0
var glows: Array = []
var pick: Array = []           # [[Vector2i, PackedVector2Array]] front to back
var clock := 0.0


func _ready() -> void:
	# pixel-perfect 640x360 even inside a 1080p project (the HD-2D menu); the menu resets it
	var win := get_tree().root
	win.content_scale_size = Vector2i(640, 360)
	win.content_scale_mode = Window.CONTENT_SCALE_MODE_VIEWPORT
	win.content_scale_stretch = Window.CONTENT_SCALE_STRETCH_INTEGER
	conf = _json(CONF_PATH)
	var m: Dictionary = _json(MAP_PATH)
	cols = int(m["cols"])
	rows = int(m["rows"])
	heights = m["height"]
	terrain = m["terrain"]
	tile = int(conf["tile"])
	hw = tile / 2
	hh = tile / 4
	lv = int(conf["level_px"])
	base = int(conf["base"])
	water_px = roundi(float(m.get("water_level", 0.15)) / float(m.get("level", 0.5)) * lv)
	move_range = int(conf.get("move", 4))
	jump = int(conf.get("jump", 2))
	for p in m.get("props", []):
		if String(p["kind"]) in BLOCKING:
			blocked[Vector2i(int(p["col"]), int(p["row"]))] = true
	range_tex = load(conf["range"])
	ring_tex = load(conf["cursor"])
	_build_map()
	_build_props(m.get("props", []))
	_build_units(m.get("units", []))
	_build_ui()
	if not teams.is_empty():
		active_team = int(teams[0])
	_update_hud()


func _json(path: String) -> Dictionary:
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		push_error("missing " + path)
		return {}
	var data = JSON.parse_string(f.get_as_text())
	return data if data is Dictionary else {}


# ------------------------------------------------------------------ board

func inside(c: Vector2i) -> bool:
	return c.x >= 0 and c.y >= 0 and c.x < cols and c.y < rows


func hgt(c: Vector2i) -> int:
	return int(heights[c.y][c.x])


func ter(c: Vector2i) -> String:
	return String(terrain[c.y][c.x])


func walkable(c: Vector2i) -> bool:
	return inside(c) and ter(c) in WALKABLE and not blocked.has(c)


func depth(c: Vector2i) -> int:
	return (c.x + c.y) * K + K


## Screen point of the centre of a cell's top diamond at level h.
func top_at(c: Vector2i, h: float) -> Vector2:
	return Vector2((c.x - c.y) * hw, (c.x + c.y) * hh - h * lv)


func top(c: Vector2i) -> Vector2:
	return top_at(c, hgt(c))


func _build_map() -> void:
	var atlas: Texture2D = load(conf["blocks"])
	var blocks: Dictionary = conf["blocks_rects"]
	water_frames = int(conf["water_frames"])
	water_tex = AtlasTexture.new()
	water_tex.atlas = load(conf["water"])
	water_tex.region = Rect2(0, 0, tile, tile / 2)
	var water_mat := ShaderMaterial.new()
	water_mat.shader = load(conf["water_shader"])
	var order: Array = []
	for r in rows:
		for c in cols:
			var cell := Vector2i(c, r)
			var tname := ter(cell)
			var rects: Dictionary = blocks.get(tname, blocks["stone"])
			var h := clampi(hgt(cell), int(conf["min_level"]), int(conf["max_level"]))
			var rect: Array = rects[str(h)]
			var s := Sprite2D.new()
			s.texture = atlas
			s.centered = false
			s.region_enabled = true
			s.region_rect = Rect2(rect[0], rect[1], rect[2], rect[3])
			s.position = top(cell) - Vector2(hw, hh)
			s.z_index = depth(cell) + Z_BLOCK
			add_child(s)
			if tname == "water":
				var w := Sprite2D.new()
				w.texture = water_tex
				w.material = water_mat
				w.centered = false
				w.position = top(cell) - Vector2(hw, hh + water_px)
				w.z_index = depth(cell) + Z_WATER
				add_child(w)
			var p := top(cell)
			var bottom := float((hgt(cell) - base) * lv)
			order.append([c + r, cell, PackedVector2Array([
				p + Vector2(0, -hh), p + Vector2(hw, 0), p + Vector2(hw, bottom),
				p + Vector2(0, hh + bottom), p + Vector2(-hw, bottom), p + Vector2(-hw, 0)])])
	order.sort_custom(func(a, b): return a[0] > b[0])
	for o in order:
		pick.append([o[1], o[2]])
	range_layer = Node2D.new()
	add_child(range_layer)
	cursor = Sprite2D.new()
	cursor.texture = ring_tex
	cursor.centered = false
	cursor.visible = false
	cursor.modulate = Color(1, 1, 1, 0.9)
	add_child(cursor)


func _build_props(props: Array) -> void:
	var defs: Dictionary = conf["props"]
	for p in props:
		var kind := String(p["kind"])
		if not defs.has(kind):
			continue
		var d: Dictionary = defs[kind]
		var cell := Vector2i(int(p["col"]), int(p["row"]))
		if not inside(cell):
			continue
		var node := Node2D.new()
		node.position = top(cell)
		node.z_index = depth(cell) + Z_THING
		add_child(node)
		var sh := Sprite2D.new()
		sh.texture = load(conf["shadow"])
		sh.z_index = Z_SHADOW - Z_THING
		sh.scale = Vector2(0.7, 0.7) if kind != "crate" else Vector2(1, 1)
		node.add_child(sh)
		var tex: Texture2D = load(d["png"])
		var n := int(d.get("frames", 1))
		var anchor := Vector2(d["anchor"][0], d["anchor"][1])
		if n > 1:
			var sf := SpriteFrames.new()
			var fw := tex.get_width() / n
			for i in n:
				var at := AtlasTexture.new()
				at.atlas = tex
				at.region = Rect2(i * fw, 0, fw, tex.get_height())
				sf.add_frame("default", at)
			sf.set_animation_speed("default", 8.0)
			var a := AnimatedSprite2D.new()
			a.sprite_frames = sf
			a.centered = false
			a.offset = -anchor
			a.play("default")
			a.frame = (cell.x * 7 + cell.y * 3) % n
			node.add_child(a)
		else:
			var s := Sprite2D.new()
			s.texture = tex
			s.centered = false
			s.offset = -anchor
			node.add_child(s)
		if d.has("glow"):
			var g := Sprite2D.new()
			g.texture = load(d["glow"])
			var mat := CanvasItemMaterial.new()
			mat.blend_mode = CanvasItemMaterial.BLEND_MODE_ADD
			g.material = mat
			g.z_as_relative = false
			g.z_index = 4000
			g.position = top(cell)
			add_child(g)
			glows.append([g, float(cell.x * 1.3 + cell.y * 2.1)])


func _build_units(list: Array) -> void:
	var defs: Dictionary = conf["units"]
	var shadow: Texture2D = load(conf["shadow"])
	for u in list:
		var uname := String(u["name"])
		var d: Dictionary = defs.get(uname, defs["_pawn"])
		var cell := Vector2i(int(u["col"]), int(u["row"]))
		var team := int(u.get("team", 0))
		var node := Node2D.new()
		add_child(node)
		var sh := Sprite2D.new()
		sh.texture = shadow
		sh.z_index = Z_SHADOW - Z_THING
		node.add_child(sh)
		var ring := Sprite2D.new()
		ring.texture = ring_tex
		ring.modulate = Color(conf["teams"][team % conf["teams"].size()])
		ring.modulate.a = 0.75
		ring.scale = Vector2(0.75, 0.75)
		ring.z_index = Z_CURSOR - Z_THING
		node.add_child(ring)
		var spr := AnimatedSprite2D.new()
		spr.sprite_frames = _frames(d)
		spr.centered = false
		spr.offset = -Vector2(d["foot"][0], d["foot"][1])
		node.add_child(spr)
		var b: Array = d["body"]
		var unit := {"node": node, "sprite": spr, "ring": ring, "name": uname, "team": team,
			"cell": cell, "from": cell, "path": [], "t": 0.0, "facing": 4, "action": "idle",
			"body": Rect2(b[0], b[1], b[2] - b[0], b[3] - b[1])}
		units.append(unit)
		if not teams.has(team):
			teams.append(team)
		var fi := DIRS.find(String(u.get("facing", "S")))
		var a := float(maxi(fi, 0)) * TAU / DIRS.size()
		face(unit, Vector2(cos(a), sin(a)))
		_place(unit, top(cell), depth(cell))
	teams.sort()


func _frames(d: Dictionary) -> SpriteFrames:
	if String(d.get("tres", "")) != "" and ResourceLoader.exists(d["tres"]):
		var res = load(d["tres"])
		if res is SpriteFrames:
			return res
	var sheet: Texture2D = load(d["png"])
	var data: Dictionary = _json(d["json"])
	var sf := SpriteFrames.new()
	var raw = data.get("frames", [])
	var items: Array = []
	if raw is Dictionary:
		for k in raw:
			items.append([String(k), raw[k]])
	else:
		for f in raw:
			items.append([String(f["filename"]), f])
	for it in items:
		var anim: String = it[0].rsplit(" ", true, 1)[0]
		if not sf.has_animation(anim):
			sf.add_animation(anim)
			sf.set_animation_speed(anim, 1000.0 / maxf(1.0, float(it[1].get("duration", 100))))
		var r: Dictionary = it[1]["frame"]
		var at := AtlasTexture.new()
		at.atlas = sheet
		at.region = Rect2(r["x"], r["y"], r["w"], r["h"])
		sf.add_frame(anim, at)
	return sf


# ------------------------------------------------------------------ units

## Face a direction on the ground: v in grid terms (x = +col, y = +row). Sprites are named by
## the direction on a flat ground seen from the front, and the iso grid axes run diagonally on
## screen, so +col faces SE, +row SW, -col NW, -row NE (8 of the 16 directions are used).
func face(u: Dictionary, v: Vector2) -> void:
	var g := Vector2(v.x - v.y, v.x + v.y)
	if not g.is_zero_approx():
		u["facing"] = wrapi(roundi(g.angle() / TAU * 8.0), 0, 8) * 2
	_show(u)


func act(u: Dictionary, action: String) -> void:
	u["action"] = action
	_show(u)


func _show(u: Dictionary) -> void:
	var spr: AnimatedSprite2D = u["sprite"]
	var anim := "%s_%s" % [u["action"], DIRS[u["facing"]]]
	if not spr.sprite_frames.has_animation(anim):
		anim = "idle_" + DIRS[u["facing"]]
		if not spr.sprite_frames.has_animation(anim):
			return
	if anim == String(spr.animation) and spr.is_playing():
		return
	var same := String(spr.animation).begins_with(String(u["action"]) + "_")
	var keep := spr.frame
	var progress := spr.frame_progress
	spr.play(anim)
	if same:
		spr.set_frame_and_progress(keep, progress)


func _place(u: Dictionary, at: Vector2, z: int) -> void:
	var node: Node2D = u["node"]
	node.position = at.round()
	node.z_index = z + Z_THING


func unit_on(c: Vector2i):
	for u in units:
		if u["cell"] == c:
			return u
	return null


## Cells reachable from start (diorama.path_cost: `move` steps, climb <= jump levels, drop
## <= 2 * jump), not through other units unless ignore_units. Fills came_from for paths.
func reachable(start: Vector2i, ignore_units := false) -> Dictionary:
	var best := {start: 0}
	came_from = {}
	var occupied := {}
	if not ignore_units:
		for u in units:
			if u["cell"] != start:
				occupied[u["cell"]] = true
	var frontier: Array[Vector2i] = [start]
	while not frontier.is_empty():
		var nxt: Array[Vector2i] = []
		for c in frontier:
			for d in STEPS:
				var n: Vector2i = c + d
				if not walkable(n) or occupied.has(n):
					continue
				var dh := hgt(n) - hgt(c)
				if dh > jump or -dh > 2 * jump:
					continue
				var cost: int = int(best[c]) + 1
				if cost <= move_range and cost < int(best.get(n, 99)):
					best[n] = cost
					came_from[n] = c
					nxt.append(n)
		frontier = nxt
	return best


func select(u) -> void:
	_clear_range()
	selected = u
	if u == null:
		_update_hud()
		return
	reach = reachable(u["cell"])
	for c in reach:
		if c == u["cell"]:
			continue
		var s := Sprite2D.new()
		s.texture = range_tex
		s.centered = false
		s.position = top(c) - Vector2(hw, hh)
		s.z_index = depth(c) + Z_RANGE
		range_layer.add_child(s)
	_update_hud()


func _clear_range() -> void:
	for ch in range_layer.get_children():
		ch.queue_free()
	reach = {}


## Walk the selected unit to a highlighted cell; the turn passes when it arrives.
func move_selected(target: Vector2i) -> bool:
	if selected == null or busy or not reach.has(target) or target == selected["cell"]:
		return false
	var path: Array = []
	var c := target
	while c != selected["cell"]:
		path.push_front(c)
		c = came_from[c]
	var u: Dictionary = selected
	_clear_range()
	u["path"] = path
	u["from"] = u["cell"]
	u["t"] = 0.0
	face(u, Vector2(path[0] - u["cell"]))
	act(u, "walk")
	busy = true
	_update_hud()
	return true


func end_turn() -> void:
	if busy:
		return
	select(null)
	if teams.is_empty():
		return
	active_team = int(teams[(teams.find(active_team) + 1) % teams.size()])
	_update_hud()


func _step(u: Dictionary, delta: float) -> void:
	u["t"] = float(u["t"]) + delta / STEP_TIME
	while float(u["t"]) >= 1.0 and not u["path"].is_empty():
		u["t"] = float(u["t"]) - 1.0
		u["cell"] = u["path"].pop_front()
		u["from"] = u["cell"]
		if not u["path"].is_empty():
			face(u, Vector2(u["path"][0] - u["cell"]))
	if u["path"].is_empty():
		u["t"] = 0.0
		_place(u, top(u["cell"]), depth(u["cell"]))
		act(u, "idle")
		busy = false
		selected = null
		end_turn()
		return
	var a: Vector2i = u["from"]
	var b: Vector2i = u["path"][0]
	var t: float = u["t"]
	var ha := hgt(a)
	var hb := hgt(b)
	var flat := top_at(a, 0).lerp(top_at(b, 0), t)
	var lift := lerpf(ha, hb, t) * lv
	if ha != hb:                                   # a hop over the step
		lift += sin(PI * t) * (absi(hb - ha) * lv * 0.5 + 5.0)
	var z := maxi(depth(a), depth(b))
	if hb > ha and depth(b) > depth(a) and t < 0.5:
		z = depth(a)                               # still behind the wall it climbs
	_place(u, flat - Vector2(0, lift), z)


# ------------------------------------------------------------------ input + view

func cell_at(p: Vector2) -> Vector2i:
	for e in pick:
		if Geometry2D.is_point_in_polygon(p, e[1]):
			return e[0]
	return Vector2i(-1, -1)


func unit_at(p: Vector2):
	var best: Variant = null
	for u in units:
		var node: Node2D = u["node"]
		var r: Rect2 = u["body"]
		if Rect2(node.position + r.position, r.size).has_point(p):
			if best == null or node.z_index > best["node"].z_index:
				best = u
	return best


func click(p: Vector2) -> void:
	if busy:
		return
	var u: Variant = unit_at(p)
	if u != null and u["team"] == active_team and u != selected:
		select(u)
		return
	var c := cell_at(p)
	if selected != null and reach.has(c) and c != selected["cell"]:
		move_selected(c)
		return
	var on: Variant = null
	if inside(c):
		on = unit_on(c)
	if on != null and on["team"] == active_team and on != selected:
		select(on)
	else:
		select(null)


func _unhandled_input(e: InputEvent) -> void:
	if e is InputEventMouseButton and e.pressed:
		match e.button_index:
			MOUSE_BUTTON_WHEEL_UP:
				set_zoom(zoom_level + 1)
			MOUSE_BUTTON_WHEEL_DOWN:
				set_zoom(zoom_level - 1)
			MOUSE_BUTTON_LEFT:
				click(get_global_mouse_position())
			MOUSE_BUTTON_RIGHT:
				select(null)
	elif e is InputEventMouseMotion:
		hover(get_global_mouse_position())
	elif e is InputEventKey and e.pressed and not e.echo:
		if e.keycode == KEY_TAB or e.keycode == KEY_SPACE:
			end_turn()
		elif e.keycode == KEY_ESCAPE:
			if selected == null and ResourceLoader.exists("res://menu.tscn"):
				get_tree().change_scene_to_file("res://menu.tscn")
			else:
				select(null)


func hover(p: Vector2) -> void:
	var c := cell_at(p)
	cursor.visible = inside(c) and ter(c) != "water"
	if cursor.visible:
		cursor.position = top(c) - Vector2(hw, hh)
		cursor.z_index = depth(c) + Z_CURSOR


func set_zoom(z: int) -> void:
	zoom_level = clampi(z, 1, 4)
	cam.zoom = Vector2(zoom_level, zoom_level)


func centre_on(c: Vector2i) -> void:
	cam.position = (top(c) - Vector2(0, 16)).round()


func _build_ui() -> void:
	cam = Camera2D.new()
	add_child(cam)
	cam.make_current()
	var box := Rect2(pick[0][1][0], Vector2.ZERO)
	for e in pick:
		for q in e[1]:
			box = box.expand(q)
	cam.position = box.get_center().round()
	set_zoom(int(conf.get("zoom", 1)))
	var layer := CanvasLayer.new()
	add_child(layer)
	var panel := ColorRect.new()
	panel.color = Color(0.05, 0.05, 0.1, 0.72)
	panel.position = Vector2(4, 4)
	panel.size = Vector2(200, 30)
	layer.add_child(panel)
	hud = Label.new()
	hud.position = Vector2(9, 6)
	hud.add_theme_font_size_override("font_size", 10)
	hud.add_theme_constant_override("line_spacing", -2)
	layer.add_child(hud)
	var strip := ColorRect.new()
	strip.color = Color(0.05, 0.05, 0.1, 0.6)
	strip.position = Vector2(0, 342)
	strip.size = Vector2(640, 18)
	layer.add_child(strip)
	var help := Label.new()
	help.text = "Click unit > purple cell   Tab/Space end turn   WASD pan   wheel zoom"
	help.position = Vector2(6, 344)
	help.add_theme_font_size_override("font_size", 8)
	help.modulate = Color(1, 1, 1, 0.85)
	layer.add_child(help)


func _update_hud() -> void:
	if hud == null:
		return
	var team_name: String = TEAM_NAMES[active_team % TEAM_NAMES.size()]
	var line2 := "Pick a unit"
	if selected != null:
		line2 = "Selected: %s" % String(selected["name"]).capitalize()
	if busy:
		line2 = "Moving..."
	hud.text = "%s team's turn\n%s" % [team_name, line2]
	hud.modulate = Color(conf["teams"][active_team % conf["teams"].size()]).lerp(Color.WHITE, 0.45)


func _process(delta: float) -> void:
	clock += delta
	var v := Vector2.ZERO
	if Input.is_physical_key_pressed(KEY_A) or Input.is_physical_key_pressed(KEY_LEFT):
		v.x -= 1
	if Input.is_physical_key_pressed(KEY_D) or Input.is_physical_key_pressed(KEY_RIGHT):
		v.x += 1
	if Input.is_physical_key_pressed(KEY_W) or Input.is_physical_key_pressed(KEY_UP):
		v.y -= 1
	if Input.is_physical_key_pressed(KEY_S) or Input.is_physical_key_pressed(KEY_DOWN):
		v.y += 1
	if v != Vector2.ZERO:
		cam.position = (cam.position + v * PAN_SPEED * delta / zoom_level).round()
	water_clock += delta
	if water_clock >= 0.3:
		water_clock = 0.0
		var f := int(water_tex.region.position.x / tile + 1) % water_frames
		water_tex.region = Rect2(f * tile, 0, tile, tile / 2)
	range_layer.modulate.a = 0.75 + 0.25 * sin(clock * 5.0)
	for g in glows:
		var s: Sprite2D = g[0]
		s.modulate.a = 0.75 + 0.25 * sin(clock * 9.0 + g[1]) * sin(clock * 5.3 + g[1] * 2.0)
	if selected != null and not busy:
		var ring: Sprite2D = selected["ring"]
		ring.modulate.a = 0.5 + 0.5 * absf(sin(clock * 4.0))
	for u in units:
		if not u["path"].is_empty():
			_step(u, delta)
'''


def write_iso_scene(diorama: Diorama, project_dir, units: dict | None = None, tile: int = 32, *,
                    seed: int = 0, move: int = 4, jump: int = 2, zoom: int = 1,
                    overwrite_map: bool = False) -> list[Path]:
    """Write the isometric scene into `project_dir/iso/` (see the module doc).
    units: {map unit name: sprite sheet .png} (with its .json frame list and, if there, the
    SpriteFrames .tres beside it); map units without a sheet get a plain stand-in.
    Returns the written files."""
    if tile % 8:
        raise ValueError("tile must be a multiple of 8 (e.g. 32)")
    root = Path(project_dir)
    iso = root / "iso"
    iso.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def save(img: np.ndarray, name: str) -> str:
        p = iso / name
        Image.fromarray(img).save(p)
        written.append(p)
        return f"res://iso/{name}"

    map_json = root / "map" / "map.json"
    if overwrite_map or not map_json.exists():
        written.append(diorama.save(map_json))
    board = Diorama.load(map_json)

    atlas, rects = build_blocks(board, tile, seed)
    lp = tile // 4
    levels = [int(k) for k in rects["stone"]]
    conf: dict = {
        "tile": tile, "level_px": lp, "base": BASE, "level": LEVEL,
        "min_level": min(levels), "max_level": max(levels),
        "move": move, "jump": jump, "zoom": zoom, "teams": TEAM_COLOURS,
        "walkable": sorted(WALKABLE), "blocking": sorted(BLOCKING_PROPS),
        "blocks": save(atlas, "blocks.png"), "blocks_rects": rects,
        "water": save(water_frames(tile, seed), "water.png"), "water_frames": WATER_FRAMES,
        "range": save(range_sprite(tile), "range.png"),
        "cursor": save(cursor_sprite(tile), "cursor.png"),
        "shadow": save(shadow_sprite(int(tile * 0.6) // 2 * 2), "shadow.png"),
    }
    shader = iso / "water.gdshader"
    shader.write_text(WATER_SHADER, encoding="utf-8")
    written.append(shader)
    conf["water_shader"] = "res://iso/water.gdshader"

    cw = max(8, int(tile * 0.6) // 4 * 4)
    crate = crate_sprite(cw)
    barrel = barrel_sprite(max(8, int(tile * 0.4) // 2 * 2))
    post = post_sprite(4, int(tile * 0.6))
    torch = torch_frames(tile)
    tw = torch.shape[1] // 4
    conf["props"] = {
        # anchor = the pixel that sits on the cell centre (bottom middle; the crate's front
        # corner sits half a diamond below its centre)
        "crate": {"png": save(crate, "crate.png"), "anchor": [cw // 2, crate.shape[0] - cw // 4]},
        "barrel": {"png": save(barrel, "barrel.png"),
                   "anchor": [barrel.shape[1] // 2, barrel.shape[0] - 2]},
        "post": {"png": save(post, "post.png"), "anchor": [2, post.shape[0] - 1]},
        "torch": {"png": save(torch, "torch.png"), "frames": 4, "anchor": [tw // 2, torch.shape[0] - 1],
                  "glow": save(glow_sprite(tile * 2), "glow.png")},
    }

    udir = iso / "units"
    defs = {}
    for name, sheet in (units or {}).items():
        entry, files = _copy_unit(name, sheet, udir)
        defs[name] = entry
        written += files
    png, js = pawn_sheet(udir / "_pawn", "pawn")
    written += [png, js]
    defs["_pawn"] = {"png": "res://iso/units/_pawn/pawn.png", "json": "res://iso/units/_pawn/pawn.json",
                     "tres": "", **unit_metrics(png, js)}
    conf["units"] = defs

    for name, text in (("iso.json", json.dumps(conf, indent=1)), ("iso.gd", ISO_GD), ("iso.tscn", TSCN)):
        p = iso / name
        p.write_text(text, encoding="utf-8")
        written.append(p)
    project = root / "project.godot"
    if not project.exists():
        project.write_text(PROJECT.format(name=f"{diorama.name} (isometric)"), encoding="utf-8")
        written.append(project)
    return written

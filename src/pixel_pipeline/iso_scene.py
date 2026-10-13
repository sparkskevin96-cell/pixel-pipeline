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

Play follows the 3D looks' rules: per-unit stats from map.json, purple move range (own move /
jump), then a red attack range (enemies within `reach`), diorama.damage() for hits, done units
greyed, automatic turn pass, last team standing wins (see the header of ISO_GD).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from .diorama import (BLOCKING_PROPS, CLASSES, DEFAULT_STATS, LEVEL, TERRAINS, WALKABLE, Diorama,
                      textures)
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


def range_sprite(w: int, fill=(150, 50, 255, 200), edge=(226, 196, 255, 255)) -> np.ndarray:
    """Movement-range diamond: a saturated, mostly opaque fill (stays clearly purple over green
    grass, grey stone and brown planks alike) inside a lighter 1px outline."""
    mask = _diamond_mask(w)
    out = np.zeros((w // 2, w, 4), np.uint8)
    out[mask] = fill
    out[_outline(mask)] = edge
    return out


def attack_sprite(w: int) -> np.ndarray:
    """The red attack-range diamond (same style as the purple move range)."""
    return range_sprite(w, (240, 36, 36, 200), (255, 190, 176, 255))


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
##
## Play (the same rules as the 3D looks): click an own unit that has not acted -> purple move
## range (its own move/jump). Click a purple cell to walk there, or the unit itself to stay.
## Then red cells = enemies within `reach` (Manhattan distance): click one to attack
## (diorama.damage: attack - defence/2, x5/4 from higher ground, min 1); Space/Tab/Esc/right
## click = wait. Done units grey out until their team's next turn; when the whole team is done
## the turn passes (Space/Tab with nothing selected ends it early). Last team standing wins.
## WASD/arrows = pan, mouse wheel = zoom (integer steps), Esc with nothing selected = menu.
##
## Painter's order: every sprite's z_index = (col + row) * K + layer, so back diagonals are
## drawn first and whatever stands on a cell is drawn after its column but before the cells
## in front of it (which may hide it, e.g. a unit behind a wall). HP bars and damage numbers
## are drawn above everything.

signal hit(attacker: Dictionary, target: Dictionary, amount: int)
signal died(unit: Dictionary)
signal turn_started(team: int)
signal game_over(winner: int)

const MAP_PATH := "res://map/map.json"
const CONF_PATH := "res://iso/iso.json"
const DIRS: Array[String] = ["E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW", "N", "NNE", "NE", "ENE"]
const WALKABLE: Array[String] = ["grass", "stone", "cobble", "planks", "dirt", "sand", "snow"]
const BLOCKING: Array[String] = ["crate", "barrel", "post", "tree", "pine", "rock", "house", "wall", "well", "fence"]
const STEPS: Array[Vector2i] = [Vector2i(1, 0), Vector2i(-1, 0), Vector2i(0, 1), Vector2i(0, -1)]
const TEAM_NAMES: Array[String] = ["Blue", "Red", "Green", "Gold"]
const ONCE: Array[String] = ["attack", "hurt", "death", "lie_down"]
const STATS: Array[String] = ["hp", "attack", "defence", "move", "jump", "reach"]
const K := 8
const Z_BLOCK := 0
const Z_WATER := 1
const Z_RANGE := 2
const Z_SHADOW := 3
const Z_CURSOR := 4
const Z_THING := 5
const Z_TOP := 3500            # hp bars, damage numbers: above the whole board
const STEP_TIME := 0.28
const PAN_SPEED := 220.0
const BAR_W := 16
const DONE_GREY := Color(0.55, 0.55, 0.55)

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

var units: Array = []          # Dictionaries: node sprite ring bar fill name team cell from path t facing action body hp max_hp atk def move jump reach done
var teams: Array = []
var active_team := 0
var phase := "select"          # select | move | walk | act | busy | over
var busy: bool:
	get:
		return phase == "walk" or phase == "busy"
var selected: Variant = null   # a unit Dictionary or null
var reach := {}                # Vector2i -> steps (the purple move range)
var attack_cells := {}         # Vector2i -> enemy unit (the red attack range)
var came_from := {}
var hovered: Variant = null
var preview := ""
var winner := -1

var cam: Camera2D
var zoom_level := 2
var hud: Label
var hud_panel: ColorRect
var banner: CanvasLayer
var cursor: Sprite2D
var range_layer: Node2D
var attack_layer: Node2D
var range_tex: Texture2D
var attack_tex: Texture2D
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
	for p in m.get("props", []):
		if String(p["kind"]) in BLOCKING:
			blocked[Vector2i(int(p["col"]), int(p["row"]))] = true
	range_tex = load(conf["range"])
	attack_tex = load(conf["attack"])
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
	attack_layer = Node2D.new()
	add_child(attack_layer)
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


## A unit's stat: map.json's value, else the class table by name (diorama.CLASSES), else default.
func _stat(u: Dictionary, i: int) -> int:
	var v := int(u.get(STATS[i], 0))
	if v > 0:
		return v
	var classes: Dictionary = conf.get("classes", {})
	var row: Array = classes.get(String(u["name"]), conf.get("default_stats", [28, 10, 5, 4, 2, 1]))
	return int(row[i])


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
		ring.modulate = team_colour(team)
		ring.modulate.a = 0.75
		ring.scale = Vector2(0.75, 0.75)
		ring.z_index = Z_CURSOR - Z_THING
		node.add_child(ring)
		var spr := AnimatedSprite2D.new()      # native sheet pixels: never scaled
		spr.sprite_frames = _frames(d)
		spr.centered = false
		spr.offset = -Vector2(d["foot"][0], d["foot"][1])
		node.add_child(spr)
		var b: Array = d["body"]
		var hp := _stat(u, 0)
		var unit := {"node": node, "sprite": spr, "ring": ring, "name": uname, "team": team,
			"cell": cell, "from": cell, "path": [], "t": 0.0, "facing": 4, "action": "idle",
			"body": Rect2(b[0], b[1], b[2] - b[0], b[3] - b[1]),
			"hp": hp, "max_hp": hp, "atk": _stat(u, 1), "def": _stat(u, 2), "move": _stat(u, 3),
			"jump": _stat(u, 4), "reach": _stat(u, 5), "done": false}
		_make_bar(unit)
		units.append(unit)
		if not teams.has(team):
			teams.append(team)
		var fi := DIRS.find(String(u.get("facing", "S")))
		var a := float(maxi(fi, 0)) * TAU / DIRS.size()
		face(unit, Vector2(cos(a), sin(a)))
		_place(unit, top(cell), depth(cell))
	teams.sort()


func team_colour(team: int) -> Color:
	return Color(conf["teams"][team % conf["teams"].size()])


func _frames(d: Dictionary) -> SpriteFrames:
	var sf: SpriteFrames = null
	if String(d.get("tres", "")) != "" and ResourceLoader.exists(d["tres"]):
		var res = load(d["tres"])
		if res is SpriteFrames:
			sf = res
	if sf == null:
		var sheet: Texture2D = load(d["png"])
		var data: Dictionary = _json(d["json"])
		sf = SpriteFrames.new()
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
	for anim in sf.get_animation_names():          # one-shots stop on their last frame
		if String(anim).rsplit("_", true, 1)[0] in ONCE:
			sf.set_animation_loop(anim, false)
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
	_show(u, action in ONCE)


func has_anim(u: Dictionary, action: String) -> bool:
	var sf: SpriteFrames = u["sprite"].sprite_frames
	return sf.has_animation("%s_%s" % [action, DIRS[u["facing"]]])


## Seconds one pass of an action's animation takes in the unit's current direction (0 if none).
func anim_length(u: Dictionary, action: String) -> float:
	var sf: SpriteFrames = u["sprite"].sprite_frames
	var anim := "%s_%s" % [action, DIRS[u["facing"]]]
	if not sf.has_animation(anim):
		return 0.0
	var total := 0.0
	for i in sf.get_frame_count(anim):
		total += sf.get_frame_duration(anim, i)
	return total / maxf(0.1, sf.get_animation_speed(anim))


func _show(u: Dictionary, restart := false) -> void:
	var spr: AnimatedSprite2D = u["sprite"]
	var anim := "%s_%s" % [u["action"], DIRS[u["facing"]]]
	if not spr.sprite_frames.has_animation(anim):
		anim = "idle_" + DIRS[u["facing"]]
		if not spr.sprite_frames.has_animation(anim):
			return
	if restart:
		spr.play(anim)
		spr.set_frame_and_progress(0, 0.0)
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


func unit_on(c: Vector2i) -> Variant:
	for u in units:
		if u["cell"] == c:
			return u
	return null


## A small pixel HP bar above the unit: team-coloured frame, green -> yellow -> red fill.
func _make_bar(u: Dictionary) -> void:
	var b: Rect2 = u["body"]
	var bar := Node2D.new()
	bar.z_as_relative = false
	bar.z_index = Z_TOP
	bar.position = Vector2(-BAR_W / 2 - 1, b.position.y - 6)
	var parts: Array = []
	for spec in [[Vector2.ZERO, Vector2(BAR_W + 2, 4), team_colour(u["team"]).darkened(0.1)],
			[Vector2(1, 1), Vector2(BAR_W, 2), Color(0.1, 0.07, 0.12)],
			[Vector2(1, 1), Vector2(BAR_W, 2), Color.WHITE]]:
		var r := ColorRect.new()
		r.mouse_filter = Control.MOUSE_FILTER_IGNORE
		r.position = spec[0]
		r.size = spec[1]
		r.color = spec[2]
		bar.add_child(r)
		parts.append(r)
	u["node"].add_child(bar)
	u["bar"] = bar
	u["fill"] = parts[2]
	_update_bar(u, false)


func _update_bar(u: Dictionary, animate := true) -> void:
	var f: ColorRect = u["fill"]
	var ratio := clampf(float(u["hp"]) / maxf(1.0, float(u["max_hp"])), 0.0, 1.0)
	var w := float(ceili(ratio * BAR_W)) if u["hp"] > 0 else 0.0
	f.color = Color(0.36, 0.86, 0.32) if ratio > 0.5 else (Color(0.96, 0.8, 0.2) if ratio > 0.25 else Color(0.92, 0.24, 0.2))
	if animate:
		create_tween().tween_property(f, "size:x", w, 0.3)
	else:
		f.size.x = w


## Cells reachable from start (diorama.path_cost: `move` steps, climb <= jump levels, drop
## <= 2 * jump), not through other units unless ignore_units. move/jump default to the stats of
## the unit standing on start. Fills came_from for paths.
func reachable(start: Vector2i, ignore_units := false, move_range := -1, jump := -1) -> Dictionary:
	var here: Variant = unit_on(start)
	if move_range < 0:
		move_range = int(here["move"]) if here != null else 4
	if jump < 0:
		jump = int(here["jump"]) if here != null else 2
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


## diorama.damage(): attack - defence/2, x5/4 when the attacker stands higher, at least 1.
@warning_ignore("integer_division")
func damage(a: Dictionary, t: Dictionary) -> int:
	var dmg: int = int(a["atk"]) - int(t["def"]) / 2
	if hgt(a["cell"]) > hgt(t["cell"]):
		dmg = dmg * 5 / 4
	return maxi(1, dmg)


func _overlay(layer: Node2D, tex: Texture2D, c: Vector2i) -> void:
	var s := Sprite2D.new()
	s.texture = tex
	s.centered = false
	s.position = top(c) - Vector2(hw, hh)
	s.z_index = depth(c) + Z_RANGE
	layer.add_child(s)


## Select a unit of the active team that has not acted: its purple move range (phase move).
func select(u) -> void:
	_clear_range()
	if selected != null and not is_same(selected, u):
		var old_ring: Sprite2D = selected["ring"]
		old_ring.modulate.a = 0.75
	selected = u
	if u == null:
		phase = "select"
		_update_hud()
		return
	phase = "move"
	reach = reachable(u["cell"])
	for c in reach:
		if c != u["cell"]:
			_overlay(range_layer, range_tex, c)
	_update_hud()


func _clear_range() -> void:
	for t in attack_cells.values():
		var ts: AnimatedSprite2D = t["sprite"]
		ts.self_modulate = Color.WHITE
	for layer in [range_layer, attack_layer]:
		for ch in layer.get_children():
			ch.queue_free()
	reach = {}
	attack_cells = {}


## Walk the selected unit to a highlighted cell; its action phase starts when it arrives.
func move_selected(target: Vector2i) -> bool:
	if selected == null or phase != "move" or not reach.has(target) or target == selected["cell"]:
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
	phase = "walk"
	_update_hud()
	return true


## Action phase: red cells on the enemies within reach; none -> the unit is done.
func enter_act(u: Dictionary) -> void:
	_clear_range()
	selected = u
	for e in units:
		if e["team"] != u["team"] and e["hp"] > 0:
			var dist: int = absi(e["cell"].x - u["cell"].x) + absi(e["cell"].y - u["cell"].y)
			if dist <= int(u["reach"]):
				attack_cells[e["cell"]] = e
	if attack_cells.is_empty():
		finish(u)
		return
	phase = "act"
	for c in attack_cells:
		_overlay(attack_layer, attack_tex, c)
		var edge := Sprite2D.new()                 # outline over the target's feet: a tall
		edge.texture = ring_tex                    # sprite would hide the red cell under it
		edge.centered = false
		edge.modulate = Color(1.0, 0.32, 0.26)
		edge.position = top(c) - Vector2(hw, hh)
		edge.z_index = depth(c) + Z_THING + 1
		attack_layer.add_child(edge)
	_update_hud()


## The unit has acted: grey until its team's next turn; the turn passes once the team is done.
func finish(u: Dictionary) -> void:
	u["done"] = true
	u["sprite"].modulate = DONE_GREY
	var ring: Sprite2D = u["ring"]
	ring.modulate.a = 0.4
	_clear_range()
	selected = null
	phase = "select"
	preview = ""
	var all_done := true
	for e in units:
		if e["team"] == active_team and not e["done"]:
			all_done = false
	if all_done:
		end_turn()
	_update_hud()


func end_turn() -> void:
	if busy or phase == "over":
		return
	select(null)
	teams = _living_teams()
	if teams.is_empty():
		return
	var i := teams.find(active_team)
	active_team = int(teams[(i + 1) % teams.size()])
	for u in units:
		if u["team"] == active_team:
			u["done"] = false
			u["sprite"].modulate = Color.WHITE
			var ring: Sprite2D = u["ring"]
			ring.modulate.a = 0.75
	preview = ""
	_update_hud()
	turn_started.emit(active_team)


func _living_teams() -> Array:
	var out: Array = []
	for u in units:
		if not out.has(u["team"]):
			out.append(u["team"])
	out.sort()
	return out


## a attacks t: face each other, attack anim (or a lunge), the hit lands part-way through:
## hp drops, damage number, hurt anim (or a flash + shake); at 0 hp the target dies.
func attack(a: Dictionary, t: Dictionary) -> void:
	if phase != "act" or not attack_cells.has(t["cell"]):
		return
	phase = "busy"
	_clear_range()
	preview = ""
	var dmg := damage(a, t)
	face(a, Vector2(t["cell"] - a["cell"]))
	face(t, Vector2(a["cell"] - t["cell"]))
	var hurt_len := clampf(anim_length(t, "hurt"), 0.35, 1.2) if has_anim(t, "hurt") else 0.4
	var spr: AnimatedSprite2D = a["sprite"]
	var tw := create_tween()
	if has_anim(a, "attack"):
		act(a, "attack")
		var l := clampf(anim_length(a, "attack"), 0.3, 1.6)
		tw.tween_interval(l * 0.55)
		tw.tween_callback(_land.bind(a, t, dmg))
		tw.tween_interval(maxf(l * 0.45, hurt_len))
	else:                                          # no attack clip: lunge at the target
		var toward: Vector2 = (top(t["cell"]) - top(a["cell"])).normalized() * 6.0
		tw.tween_property(spr, "position", toward.round(), 0.1)
		tw.tween_callback(_land.bind(a, t, dmg))
		tw.tween_property(spr, "position", Vector2.ZERO, 0.14)
		tw.tween_interval(maxf(0.1, hurt_len - 0.14))
	tw.tween_interval(0.15)
	tw.tween_callback(_after_hit.bind(a, t))
	_update_hud()


func _land(a: Dictionary, t: Dictionary, dmg: int) -> void:
	t["hp"] = maxi(0, int(t["hp"]) - dmg)
	_update_bar(t)
	_number(t, dmg)
	var ts: AnimatedSprite2D = t["sprite"]
	var fx := create_tween()
	fx.tween_property(ts, "self_modulate", Color(1.0, 0.35, 0.3), 0.04)
	fx.tween_property(ts, "self_modulate", Color.WHITE, 0.08)
	fx.tween_property(ts, "self_modulate", Color(1.0, 0.35, 0.3), 0.06)
	fx.tween_property(ts, "self_modulate", Color.WHITE, 0.12)
	if t["hp"] > 0 and has_anim(t, "hurt"):
		act(t, "hurt")
	elif t["hp"] > 0:                              # no hurt clip: shake
		var sh := create_tween()
		for x in [2, -2, 2, -1, 0]:
			sh.tween_property(ts, "position:x", float(x), 0.05)
	hit.emit(a, t, dmg)


func _after_hit(a: Dictionary, t: Dictionary) -> void:
	act(a, "idle")
	if t["hp"] > 0:
		act(t, "idle")
		finish(a)
	else:
		_kill(t, a)


func _kill(t: Dictionary, a: Dictionary) -> void:
	var tw := create_tween()
	var clip := "death" if has_anim(t, "death") else ("lie_down" if has_anim(t, "lie_down") else "")
	if clip != "":
		act(t, clip)
		tw.tween_interval(clampf(anim_length(t, clip), 0.3, 1.8) + 0.3)
	tw.tween_property(t["node"], "modulate:a", 0.0, 0.45)
	tw.tween_callback(func():
		units.erase(t)                             # frees its cell
		t["node"].queue_free()
		died.emit(t)
		if not _check_victory():
			finish(a))


func _check_victory() -> bool:
	var alive := _living_teams()
	if alive.size() > 1:
		return false
	winner = int(alive[0]) if not alive.is_empty() else -1
	teams = alive
	phase = "over"
	selected = null
	_clear_range()
	cursor.visible = false
	_show_banner()
	_update_hud()
	game_over.emit(winner)
	return true


func _number(t: Dictionary, dmg: int) -> void:
	var l := Label.new()
	l.text = str(dmg)
	l.mouse_filter = Control.MOUSE_FILTER_IGNORE
	l.add_theme_font_size_override("font_size", 12)
	l.add_theme_color_override("font_color", Color(1.0, 0.93, 0.45))
	l.add_theme_color_override("font_outline_color", Color(0.25, 0.05, 0.05))
	l.add_theme_constant_override("outline_size", 4)
	l.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	l.size = Vector2(40, 16)
	l.z_as_relative = false
	l.z_index = Z_TOP + 1
	var b: Rect2 = t["body"]
	var at: Vector2 = t["node"].position + Vector2(-20, b.position.y - 22)
	l.position = at.round()
	add_child(l)
	var tw := create_tween()
	tw.tween_property(l, "position:y", at.y - 10, 0.35).set_ease(Tween.EASE_OUT).set_trans(Tween.TRANS_QUAD)
	tw.tween_interval(0.45)
	tw.tween_property(l, "modulate:a", 0.0, 0.3)
	tw.tween_callback(l.queue_free)


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
		enter_act(u)
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


func unit_at(p: Vector2) -> Variant:
	var best: Variant = null
	for u in units:
		var node: Node2D = u["node"]
		var r: Rect2 = u["body"]
		if Rect2(node.position + r.position, r.size).has_point(p):
			if best == null or node.z_index > best["node"].z_index:
				best = u
	return best


func _ready_unit(u) -> bool:
	return u != null and u["team"] == active_team and not u["done"]


func click(p: Vector2) -> void:
	if busy or phase == "over":
		return
	var c := cell_at(p)
	var u: Variant = unit_at(p)
	if phase == "act":
		var target: Variant = null
		if u != null and attack_cells.has(u["cell"]) and is_same(attack_cells[u["cell"]], u):
			target = u
		elif attack_cells.has(c):
			target = attack_cells[c]
		if target != null:
			attack(selected, target)
		return
	if phase == "move" and selected != null:
		if (u != null and is_same(u, selected)) or c == selected["cell"]:
			enter_act(selected)                    # stay put and act
			return
		if _ready_unit(u):
			select(u)
			return
		if reach.has(c) and unit_on(c) == null:
			move_selected(c)
			return
	var on: Variant = u
	if not _ready_unit(on) and inside(c):
		on = unit_on(c)
	select(on if _ready_unit(on) else null)


## Space/Tab: the selected unit waits (done); with nothing selected, end the turn.
func wait_or_end_turn() -> void:
	if busy or phase == "over":
		return
	if selected != null:
		finish(selected)
	else:
		end_turn()


## Esc / right click: cancel the move range, or end the action phase (no undo of a move).
func cancel() -> void:
	if busy or phase == "over":
		return
	if phase == "act":
		finish(selected)
	else:
		select(null)


func _unhandled_input(e: InputEvent) -> void:
	if phase == "over":
		if e is InputEventKey and e.pressed and e.keycode == KEY_ESCAPE and ResourceLoader.exists("res://menu.tscn"):
			get_tree().change_scene_to_file("res://menu.tscn")
		return
	if e is InputEventMouseButton and e.pressed:
		match e.button_index:
			MOUSE_BUTTON_WHEEL_UP:
				set_zoom(zoom_level + 1)
			MOUSE_BUTTON_WHEEL_DOWN:
				set_zoom(zoom_level - 1)
			MOUSE_BUTTON_LEFT:
				click(get_global_mouse_position())
				hover(get_global_mouse_position())
			MOUSE_BUTTON_RIGHT:
				cancel()
	elif e is InputEventMouseMotion:
		hover(get_global_mouse_position())
	elif e is InputEventKey and e.pressed and not e.echo:
		if e.keycode == KEY_TAB or e.keycode == KEY_SPACE:
			wait_or_end_turn()
		elif e.keycode == KEY_ESCAPE:
			if phase == "select" and ResourceLoader.exists("res://menu.tscn"):
				get_tree().change_scene_to_file("res://menu.tscn")
			else:
				cancel()


func hover(p: Vector2) -> void:
	var c := cell_at(p)
	cursor.visible = inside(c) and ter(c) != "water" and phase != "over"
	if cursor.visible:
		cursor.position = top(c) - Vector2(hw, hh)
		cursor.z_index = depth(c) + Z_CURSOR
	hovered = unit_at(p)
	if hovered == null and inside(c):
		hovered = unit_on(c)
	preview = ""
	if phase == "act" and selected != null:
		var t: Variant = null
		if hovered != null and attack_cells.has(hovered["cell"]):
			t = hovered
		elif attack_cells.has(c):
			t = attack_cells[c]
		if t != null:
			var dmg := damage(selected, t)
			preview = "Hit %s for %d%s" % [String(t["name"]).capitalize(), dmg, "  KO!" if dmg >= int(t["hp"]) else ""]
	_update_hud()


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
	hud_panel = ColorRect.new()
	hud_panel.color = Color(0.05, 0.05, 0.1, 0.72)
	hud_panel.position = Vector2(4, 4)
	hud_panel.size = Vector2(236, 42)
	hud_panel.mouse_filter = Control.MOUSE_FILTER_IGNORE
	layer.add_child(hud_panel)
	hud = Label.new()
	hud.position = Vector2(9, 6)
	hud.add_theme_font_size_override("font_size", 10)
	hud.add_theme_constant_override("line_spacing", -2)
	layer.add_child(hud)
	var strip := ColorRect.new()
	strip.color = Color(0.05, 0.05, 0.1, 0.6)
	strip.position = Vector2(0, 342)
	strip.size = Vector2(640, 18)
	strip.mouse_filter = Control.MOUSE_FILTER_IGNORE
	layer.add_child(strip)
	var help := Label.new()
	help.text = "Unit > purple cell (unit = stay) > red enemy    Space wait / end turn    Esc cancel    WASD pan    wheel zoom"
	help.position = Vector2(6, 344)
	help.add_theme_font_size_override("font_size", 8)
	help.modulate = Color(1, 1, 1, 0.85)
	layer.add_child(help)


func _show_banner() -> void:
	banner = CanvasLayer.new()
	banner.layer = 5
	add_child(banner)
	var band := ColorRect.new()
	band.color = Color(0.03, 0.03, 0.07, 0.82)
	band.position = Vector2(0, 148)
	band.size = Vector2(640, 56)
	band.mouse_filter = Control.MOUSE_FILTER_IGNORE
	banner.add_child(band)
	var col := team_colour(winner) if winner >= 0 else Color.WHITE
	for edge in [148, 202]:
		var line := ColorRect.new()
		line.color = col
		line.position = Vector2(0, edge)
		line.size = Vector2(640, 2)
		banner.add_child(line)
	var l := Label.new()
	l.text = "%s wins!" % TEAM_NAMES[winner % TEAM_NAMES.size()] if winner >= 0 else "Draw"
	l.add_theme_font_size_override("font_size", 28)
	l.add_theme_color_override("font_color", col.lerp(Color.WHITE, 0.3))
	l.add_theme_color_override("font_outline_color", Color(0, 0, 0))
	l.add_theme_constant_override("outline_size", 6)
	l.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	l.position = Vector2(0, 150)
	l.size = Vector2(640, 40)
	banner.add_child(l)
	if ResourceLoader.exists("res://menu.tscn"):
		var sub := Label.new()
		sub.text = "Esc: menu"
		sub.add_theme_font_size_override("font_size", 8)
		sub.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
		sub.position = Vector2(0, 189)
		sub.size = Vector2(640, 12)
		banner.add_child(sub)


func _alive(u) -> bool:
	for e in units:
		if is_same(e, u):
			return true
	return false


func unit_line(u: Dictionary) -> String:
	return "%s  HP %d/%d  ATK %d DEF %d MOV %d" % [String(u["name"]).capitalize(), u["hp"], u["max_hp"],
		u["atk"], u["def"], u["move"]]


func _update_hud() -> void:
	if hud == null:
		return
	var team_name: String = TEAM_NAMES[active_team % TEAM_NAMES.size()]
	var shown: Variant = selected if selected != null else hovered
	if shown != null and not _alive(shown):
		shown = null
	var line2 := unit_line(shown) if shown != null else ("Pick a unit" if phase != "over" else "")
	var line3 := ""
	match phase:
		"select":
			line3 = "Click a unit  (Space: end turn)"
		"move":
			line3 = "Purple: move   unit: stay   Space: wait"
		"walk":
			line3 = "Moving..."
		"act":
			line3 = preview if preview != "" else "Red: attack   Space: wait"
		"busy":
			line3 = "..."
		"over":
			line3 = "%s wins!" % TEAM_NAMES[winner % TEAM_NAMES.size()] if winner >= 0 else "Draw"
	var line1 := "%s team's turn" % team_name if phase != "over" else "Battle over"
	hud.text = "%s\n%s\n%s" % [line1, line2, line3]
	hud.modulate = team_colour(active_team).lerp(Color.WHITE, 0.45)


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
	var pulse := 0.92 + 0.08 * sin(clock * 4.0)    # gentle: stays clearly coloured
	range_layer.modulate.a = pulse
	attack_layer.modulate.a = pulse
	if phase == "act":                             # targets glow red
		var k := 0.5 + 0.5 * sin(clock * 6.0)
		for t in attack_cells.values():
			var ts: AnimatedSprite2D = t["sprite"]
			ts.self_modulate = Color(1.0, 0.62 + 0.2 * k, 0.6 + 0.2 * k)
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
                    seed: int = 0, zoom: int = 1, overwrite_map: bool = False) -> list[Path]:
    """Write the isometric scene into `project_dir/iso/` (see the module doc).
    units: {map unit name: sprite sheet .png} (with its .json frame list and, if there, the
    SpriteFrames .tres beside it); map units without a sheet get a plain stand-in. Unit stats
    (hp attack defence move jump reach) come from map.json, else diorama.CLASSES by name.
    Sprites are drawn at their native sheet pixel size (no per-unit scaling).
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
        "zoom": zoom, "teams": TEAM_COLOURS,
        "classes": {k: list(v) for k, v in CLASSES.items()}, "default_stats": list(DEFAULT_STATS),
        "walkable": sorted(WALKABLE), "blocking": sorted(BLOCKING_PROPS),
        "blocks": save(atlas, "blocks.png"), "blocks_rects": rects,
        "water": save(water_frames(tile, seed), "water.png"), "water_frames": WATER_FRAMES,
        "range": save(range_sprite(tile), "range.png"),
        "attack": save(attack_sprite(tile), "attack.png"),
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
    # props from the map library (trees, rocks, houses, walls, wells, fences…)
    from .maplib import prop_sprite
    snowy = "snow" in set(diorama.terrain.ravel())
    for kind in sorted({k for k, _, _ in diorama.props} - set(conf["props"])):
        img, anchor = prop_sprite(kind, tile, snow=snowy)
        conf["props"][kind] = {"png": save(img, f"prop_{kind}.png"), "anchor": [int(anchor[0]), int(anchor[1])]}

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

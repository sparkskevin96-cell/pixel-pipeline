"""A ready-to-open Godot 4 project that shows the outputs working together.

A 640x360 game (x3 = 1080p, integer scaling, nearest filtering) with a terrain map from
map tiles, and units in 8 or 16 directions with team colours: click your unit, click a
tile, and it walks there (A* around water), facing where it goes and playing its walk;
after moving it greys out (spent) until Space starts the next turn.

The unit is your 3D model (rigged + animated: walk/idle; still: just turns), photos or a
video around a subject, or, with nothing given, a built-in mannequin with idle and walk
animations, so the demo works with no files at all.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from .anim3d import GlbWriter, animated_turnaround, load_rig, write_animated
from .export import pack_sheet
from .maptiles import (build_terrain_set, cells_from_vertices, example_vertex_grid, render_map,
                       terrain, write_godot_terrain_tileset)
from .settings import Settings
from .teams import recolour, sprite_colours, team_family, team_palettes, write_godot_teams
from .turnaround import direction_names, is_model, turnaround_from, write_turnaround

VIEW_W, VIEW_H = 640, 360           # x3 = 1920x1080
BLOCKING = {"water", "lava", "wall", "deep_water"}

# --------------------------------------------------------------------------- mannequin

SKIN, HAIR, TUNIC, TROUSERS, BOOTS = "#e8b48c", "#4a3020", "#2f5fd8", "#3a3a48", "#2a1e18"
STEEL, SHIELD, EYES = "#c8ccd4", "#c89a3c", "#20202a"


def _linear(hex_colour: str) -> list[float]:
    c = np.array([int(hex_colour.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)])
    return [*np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).tolist(), 1.0]


def _box(lo, hi):
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    v = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], float)
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return v, f


def _quat_x(deg: float) -> list[float]:
    h = np.radians(deg) / 2
    return [float(np.sin(h)), 0.0, 0.0, float(np.cos(h))]


def mannequin(path) -> Path:
    """A rigged knight (glTF, y up, facing +z, 1.8 units tall) with 'idle' and 'walk' loops:
    blue tunic and cape (the team colour), sword in the right hand, shield on the left."""
    joints = ["hips", "leg_l", "leg_r", "arm_l", "arm_r"]   # +x is the knight's left
    parts = [  # (lo, hi, colour, joint)
        ((0.03, 0.12, -0.08), (0.17, 0.92, 0.08), TROUSERS, 1),
        ((-0.17, 0.12, -0.08), (-0.03, 0.92, 0.08), TROUSERS, 2),
        ((0.02, 0.0, -0.09), (0.18, 0.14, 0.13), BOOTS, 1),
        ((-0.18, 0.0, -0.09), (-0.02, 0.14, 0.13), BOOTS, 2),
        ((-0.22, 0.88, -0.13), (0.22, 1.45, 0.13), TUNIC, 0),
        ((-0.21, 0.55, -0.17), (0.21, 1.42, -0.13), TUNIC, 0),          # cape
        ((-0.17, 1.43, -0.16), (0.17, 1.8, 0.16), SKIN, 0),             # head (big: reads small)
        ((-0.18, 1.66, -0.17), (0.18, 1.86, 0.08), HAIR, 0),            # hair on top
        ((-0.18, 1.43, -0.17), (0.18, 1.68, -0.08), HAIR, 0),           # and at the back
        ((0.04, 1.6, 0.16), (0.1, 1.66, 0.165), EYES, 0),
        ((-0.1, 1.6, 0.16), (-0.04, 1.66, 0.165), EYES, 0),
        ((0.22, 0.95, -0.07), (0.33, 1.42, 0.07), TUNIC, 3),
        ((-0.33, 0.95, -0.07), (-0.22, 1.42, 0.07), TUNIC, 4),
        ((0.33, 0.96, -0.18), (0.38, 1.38, 0.18), SHIELD, 3),
        ((-0.31, 0.9, -0.05), (-0.25, 1.0, 0.2), BOOTS, 4),             # hilt
        ((-0.3, 0.92, 0.2), (-0.26, 0.98, 0.75), STEEL, 4),             # blade, pointing ahead
    ]
    pos, faces, cols, jnt = [], [], [], []
    for lo, hi, colour, j in parts:
        v, f = _box(lo, hi)
        faces.append(f + sum(len(p) for p in pos))
        pos.append(v)
        cols.append(np.tile(_linear(colour), (8, 1)))
        jnt.append(np.full(8, j))
    pos, faces, cols, jnt = (np.vstack(pos), np.vstack(faces), np.vstack(cols),
                             np.concatenate(jnt))
    origin = {"hips": (0, 0.95, 0), "leg_l": (0.1, 0.92, 0), "leg_r": (-0.1, 0.92, 0),
              "arm_l": (0.27, 1.42, 0), "arm_r": (-0.27, 1.42, 0)}
    g = GlbWriter()
    nodes = [{"name": "Armature", "children": [1]},
             {"name": "hips", "translation": [0, 0.95, 0], "children": [2, 3, 4, 5]}]
    for name in joints[1:]:
        o = np.subtract(origin[name], origin["hips"])
        nodes.append({"name": name, "translation": o.tolist()})
    ibm = np.tile(np.eye(4, dtype=np.float32), (len(joints), 1, 1))
    for i, name in enumerate(joints):
        ibm[i, :3, 3] = -np.asarray(origin[name])
    j4 = np.zeros((len(pos), 4), np.uint8)
    j4[:, 0] = jnt
    w4 = np.zeros((len(pos), 4), np.uint8)
    w4[:, 0] = 255
    g.doc["skins"] = [{"joints": [1, 2, 3, 4, 5], "inverseBindMatrices": g.accessor(
        ibm.transpose(0, 2, 1).reshape(-1, 16), "MAT4")}]
    g.doc["meshes"] = [{"primitives": [{"attributes": {
        "POSITION": g.accessor(pos, "VEC3"), "COLOR_0": g.accessor(cols, "VEC4"),
        "JOINTS_0": g.accessor(j4, "VEC4", 5121),
        "WEIGHTS_0": g.accessor(w4, "VEC4", 5121, normalized=True)},
        "indices": g.accessor(faces.reshape(-1), "SCALAR", 5123)}]}]
    nodes.append({"name": "knight", "mesh": 0, "skin": 0})
    g.doc["nodes"] = nodes
    g.doc["scenes"] = [{"nodes": [0, 6]}]

    def channels(times, tracks):
        t = g.accessor(np.asarray(times, float), "SCALAR")
        samplers, chans = [], []
        for node, path, values in tracks:
            kind = "VEC4" if path == "rotation" else "VEC3"
            samplers.append({"input": t, "output": g.accessor(np.asarray(values, float), kind)})
            chans.append({"sampler": len(samplers) - 1, "target": {"node": node, "path": path}})
        return {"samplers": samplers, "channels": chans}
    walk_t = [0, 0.2, 0.4, 0.6, 0.8]
    swing = [25, 0, -25, 0, 25]
    walk = channels(walk_t, [
        (1, "translation", [[0, y, 0] for y in (0.93, 0.96, 0.93, 0.96, 0.93)]),
        (2, "rotation", [_quat_x(a) for a in swing]),
        (3, "rotation", [_quat_x(-a) for a in swing]),
        (4, "rotation", [_quat_x(-0.8 * a) for a in swing]),
        (5, "rotation", [_quat_x(0.8 * a) for a in swing])])
    idle = channels([0, 0.6, 1.2], [
        (1, "translation", [[0, 0.95, 0], [0, 0.935, 0], [0, 0.95, 0]]),
        (4, "rotation", [_quat_x(0), _quat_x(4), _quat_x(0)]),
        (5, "rotation", [_quat_x(0), _quat_x(-4), _quat_x(0)])])
    g.doc["animations"] = [{"name": "idle", **idle}, {"name": "walk", **walk}]
    return g.save(path)


# --------------------------------------------------------------------------- Godot files

PROJECT = """config_version=5

[application]

config/name="{name}"
run/main_scene="res://main.tscn"
config/features=PackedStringArray("4.3", "GL Compatibility")

[display]

window/size/viewport_width={w}
window/size/viewport_height={h}
window/size/window_width_override={ww}
window/size/window_height_override={wh}
window/stretch/mode="viewport"
window/stretch/scale_mode="integer"

[rendering]

renderer/rendering_method="gl_compatibility"
renderer/rendering_method.mobile="gl_compatibility"
textures/canvas_textures/default_texture_filter=0
2d/snap/snap_2d_transforms_to_pixel=true
"""

MAIN_TSCN = """[gd_scene load_steps=3 format=3]

[ext_resource type="Script" path="res://main.gd" id="1_main"]
[ext_resource type="TileSet" path="res://map/terrain.tres" id="2_tiles"]

[node name="Main" type="Node2D"]
script = ExtResource("1_main")

[node name="Ground" type="TileMapLayer" parent="."]
tile_set = ExtResource("2_tiles")

[node name="Overlay" type="Node2D" parent="."]

[node name="Units" type="Node2D" parent="."]
y_sort_enabled = true

[node name="Hint" type="Label" parent="."]
offset_left = 4.0
offset_top = 2.0
theme_override_colors/font_outline_color = Color(0, 0, 0, 1)
theme_override_constants/outline_size = 3
theme_override_font_sizes/font_size = 10
text = "Click your unit, then a tile. Space: next turn."
"""

MAIN_GD = """extends Node2D
## pixel-pipeline demo: click your unit, then a tile; it walks there around water and greys
## out (spent). Space starts the next turn. Units, tiles and colours were made by
## pixel-pipeline; swap in your own by re-running `pixel-pipeline demo` with --unit.

const MAP := "res://map/map.json"

@onready var ground: TileMapLayer = $Ground
@onready var overlay: Node2D = $Overlay
@onready var units: Node2D = $Units
var astar := AStarGrid2D.new()
var tile := 32
var selected: DemoUnit = null
var hover := Vector2i(-1, -1)


func _ready() -> void:
\tvar data: Dictionary = JSON.parse_string(FileAccess.get_file_as_string(MAP))
\ttile = int(data.tile)
\tfor c in data.cells:
\t\tground.set_cell(Vector2i(int(c[0]), int(c[1])), 0, Vector2i(int(c[2]), int(c[3])))
\tastar.region = Rect2i(0, 0, int(data.cols), int(data.rows))
\tastar.cell_size = Vector2(tile, tile)
\tastar.diagonal_mode = AStarGrid2D.DIAGONAL_MODE_ONLY_IF_NO_OBSTACLES
\tastar.update()
\tfor b in data.blocked:
\t\tastar.set_point_solid(Vector2i(int(b[0]), int(b[1])))
\tvar frames: SpriteFrames = load(data.unit.frames)
\tvar facing: Script = load(data.unit.script)
\tfor u in data.units:
\t\tvar unit := DemoUnit.new()
\t\tunit.name = u.name
\t\tunit.team = u.team
\t\tunits.add_child(unit)
\t\tunit.setup(frames, facing, load(data.unit.materials[u.team]),
\t\t\t\tVector2i(int(u.cell[0]), int(u.cell[1])), tile, data.unit.idle, data.unit.walk)
\t\tunit.face(Vector2(u.facing[0], u.facing[1]))
\t\tunit.arrived.connect(_on_arrived)
\toverlay.draw.connect(_draw_overlay)


func _process(_delta: float) -> void:
\tif selected and selected.moving():
\t\toverlay.queue_redraw()


func _unhandled_input(event: InputEvent) -> void:
\tif event is InputEventMouseMotion:
\t\thover = ground.local_to_map(get_local_mouse_position())
\t\toverlay.queue_redraw()
\telif event is InputEventMouseButton and event.pressed and event.button_index == MOUSE_BUTTON_LEFT:
\t\tclick(ground.local_to_map(get_local_mouse_position()))
\telif event.is_action_pressed("ui_accept"):
\t\tnew_turn()


func unit_at(cell: Vector2i) -> DemoUnit:
\tfor u in units.get_children():
\t\tif u.cell == cell:
\t\t\treturn u
\treturn null


func path_to(unit: DemoUnit, target: Vector2i) -> Array[Vector2i]:
\tif not astar.is_in_boundsv(target) or astar.is_point_solid(target) or unit_at(target):
\t\treturn []
\tfor u in units.get_children():
\t\tif u != unit:
\t\t\tastar.set_point_solid(u.cell, true)
\tvar cells := astar.get_id_path(unit.cell, target)
\tfor u in units.get_children():
\t\tif u != unit:
\t\t\tastar.set_point_solid(u.cell, false)
\tif cells.size() > 0:
\t\tcells.remove_at(0)
\treturn cells


func click(target: Vector2i) -> void:
\tvar unit := unit_at(target)
\tif unit and unit.team == "player" and not unit.spent:
\t\tselected = unit
\telif selected and not selected.moving():
\t\tvar cells := path_to(selected, target)
\t\tif not cells.is_empty():
\t\t\tselected.walk(cells)
\toverlay.queue_redraw()


func new_turn() -> void:
\tfor u in units.get_children():
\t\tu.set_spent(false)


func _on_arrived(unit: DemoUnit) -> void:
\tunit.set_spent(true)
\tif unit == selected:
\t\tselected = null
\toverlay.queue_redraw()


func _draw_overlay() -> void:
\tif selected:
\t\tvar box := Rect2((selected.position - Vector2(tile, tile) / 2).round(), Vector2(tile, tile))
\t\toverlay.draw_rect(box, Color(1, 0.85, 0.2), false, 1.0)
\t\tif not selected.moving():
\t\t\tfor c in path_to(selected, hover):
\t\t\t\toverlay.draw_rect(Rect2(Vector2(c) * tile + Vector2(tile, tile) * 0.4, Vector2(tile, tile) * 0.2),
\t\t\t\t\t\tColor(1, 1, 1, 0.7))
\tif astar.is_in_boundsv(hover):
\t\toverlay.draw_rect(Rect2(Vector2(hover) * tile, Vector2(tile, tile)), Color(1, 1, 1, 0.6), false, 1.0)
"""

UNIT_GD = """class_name DemoUnit
extends Node2D
## A map unit: walks a path of cells facing where it goes (8/16-direction sprite with the
## facing script made by pixel-pipeline) and shows its team colours via the palette shader.

signal arrived(unit: DemoUnit)

@export var speed := 80.0
var cell := Vector2i.ZERO
var team := "player"
var spent := false
var tile := 32
var path: Array[Vector2i] = []
var idle_action := &""
var walk_action := &""
var sprite  # AnimatedSprite2D running the facing script (untyped: face() comes from that script)


func setup(frames: SpriteFrames, facing: Script, team_material: Material, at: Vector2i,
\t\ttile_size: int, idle: String, walk: String) -> void:
\ttile = tile_size
\tidle_action = StringName(idle)
\twalk_action = StringName(walk)
\tsprite = AnimatedSprite2D.new()
\tsprite.sprite_frames = frames
\tsprite.set_script(facing)
\tsprite.material = team_material.duplicate()
\tvar size := frames.get_frame_texture(frames.get_animation_names()[0], 0).get_size()
\tsprite.offset = Vector2(0, roundi(-size.y / 2 + tile * 0.3))   # feet just below the middle
\tadd_child(sprite)
\tcell = at
\tposition = centre(at)
\t_act(idle_action)


func centre(c: Vector2i) -> Vector2:
\treturn Vector2(c) * tile + Vector2(tile, tile) / 2


func face(direction: Vector2) -> void:
\tsprite.face(direction)


func moving() -> bool:
\treturn not path.is_empty()


func walk(cells: Array[Vector2i]) -> void:
\tpath = cells
\t_act(walk_action)


func set_spent(value: bool) -> void:
\tspent = value
\tsprite.material.set_shader_parameter("spent", 1.0 if value else 0.0)


func _act(action: StringName) -> void:
\tif action != &"" and sprite.has_method("act"):
\t\tsprite.act(action)


func _process(delta: float) -> void:
\tif path.is_empty():
\t\treturn
\tvar target := centre(path[0])
\tvar before := position
\tposition = position.move_toward(target, speed * delta)
\tsprite.face(position - before)
\tif position.is_equal_approx(target):
\t\tcell = path.pop_front()
\t\tif path.is_empty():
\t\t\t_act(idle_action)
\t\t\tarrived.emit(self)
"""


def _pick(names: list[str], *wanted: str) -> str:
    for w in wanted:
        hit = next((n for n in names if w in n), None)
        if hit:
            return hit
    return names[0] if names else ""


def _free_cells(blocked: set, cols: int, rows: int, start: tuple[int, int], count: int,
                taken: set) -> list[tuple[int, int]]:
    """`count` free cells nearest to `start`, in a little column."""
    order = sorted(((x, y) for x in range(cols) for y in range(1, rows)),
                   key=lambda c: (abs(c[0] - start[0]) + 2 * abs(c[1] - start[1]), c))
    out = []
    for c in order:
        if c not in blocked and c not in taken and all(abs(c[1] - o[1]) >= 2 or c[0] != o[0] for o in out):
            out.append(c)
            taken.add(c)
            if len(out) == count:
                break
    return out


def parse_terrains(specs) -> tuple[tuple[str, ...], dict]:
    """['grass', 'water=C:/lake.jpg', 'lava=prompt:glowing lava'] -> (names, {name: source})."""
    names, sources = [], {}
    for spec in specs:
        name, _, source = spec.partition("=")
        names.append(name.strip())
        if source.strip():
            sources[name.strip()] = source.strip()
    return tuple(names), sources


def make_demo(out_dir, unit: list | None = None, settings: Settings | None = None, *,
              animate: bool | None = None, animations: list[str] | None = None,
              colours_from=None, directions: int = 8, frames: int = 6, elevation: float = 30.0,
              yaw: float = 0.0, terrains: tuple[str, ...] = ("grass", "water", "sand", "road"),
              terrain_sources: dict | None = None, tile: int = 32, seed: int = 0,
              client=None, name: str = "Pixel Pipeline Demo",
              progress=print) -> tuple[list[Path], np.ndarray]:
    """Write a Godot 4 project to out_dir. Returns (files written, a preview of the start
    screen as RGBA, drawn here the way Godot will show it)."""
    from .settings import settings_for
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()) and not (out / "project.godot").exists():
        raise ValueError(f"{out} isn't empty and isn't a demo project; pick a new folder")
    (out / "units").mkdir(parents=True, exist_ok=True)
    (out / "map").mkdir(parents=True, exist_ok=True)
    s = settings or settings_for("person", width=40, height=40, colors=20)
    written: list[Path] = []

    # ---- the unit
    tmp = None
    if not unit:
        tmp = Path(tempfile.mkdtemp(prefix="pixel-pipeline-demo-"))
        unit = [mannequin(tmp / "knight.glb")]
        animate = True
    unit = [Path(u) for u in unit]
    if animate is None:
        animate = (len(unit) >= 1 and is_model(unit[0]) and unit[0].suffix.lower() in (".glb", ".gltf")
                   and bool(load_rig(unit).clips))
    sheet = out / "units" / "unit.png"
    progress("Making the unit's sprites…")
    if animate:
        anim, what = animated_turnaround(unit, s, directions, frames, animations, elevation=elevation,
                                         yaw=yaw, colours_from=colours_from)
        written += write_animated(anim, sheet, godot=True)
        rgba_frames = [sp.rgba for sp in anim.flat()]
        actions = [a.name for a in anim.actions]
        facing = {d: anim.sprites[0][i][0].rgba for i, d in enumerate(anim.directions)}
    else:
        sprites, what = turnaround_from(unit, s, directions, elevation=elevation, yaw=yaw)
        written += write_turnaround(sprites, sheet, godot=True)
        rgba_frames = [sp.rgba for sp in sprites]
        actions = []
        facing = {d: sp.rgba for d, sp in zip(direction_names(directions), sprites)}
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
        what = "built-in knight: " + what.split(": ", 1)[-1]
    big, _ = pack_sheet(rgba_frames)
    colours, counts = sprite_colours(big)
    pals = team_palettes(colours, team_family(colours, counts))
    written += write_godot_teams(colours, pals, out / "units", "unit")

    # ---- the map
    progress("Making the map tiles…")
    sources = terrain_sources or {}
    ts = build_terrain_set([terrain(n, sources.get(n), tile, seed=seed, client=client)
                            for n in terrains], seed=seed)
    atlas = out / "map" / "terrain.png"
    Image.fromarray(ts.atlas, "RGBA").save(atlas)
    written += [atlas, write_godot_terrain_tileset(ts, out / "map" / "terrain.tres", atlas)]
    cols, rows = -(-VIEW_W // tile), -(-VIEW_H // tile)       # cover the screen
    vg = example_vertex_grid(cols, rows, len(ts.names), seed)
    cells = cells_from_vertices(vg, ts, seed)
    stop = [i for i, n in enumerate(ts.names) if n in BLOCKING]
    wet = np.isin(vg, stop).astype(int)
    corners = wet[:-1, :-1] + wet[:-1, 1:] + wet[1:, :-1] + wet[1:, 1:]
    blocked = {(int(x), int(y)) for y, x in zip(*np.nonzero(corners >= 2))}
    blocked |= {(x, y) for x in range(cols) for y in range(rows)        # cut off by the screen edge
                if (x + 1) * tile > VIEW_W or (y + 1) * tile > VIEW_H}
    taken: set = set()
    team_cells = [("player", _free_cells(blocked, cols, rows, (2, rows // 2), 2, taken), (1, 0)),
                  ("enemy", _free_cells(blocked, cols, rows, (cols - 3, rows // 2), 2, taken), (-1, 0))]
    units = [{"name": f"{team.title()}{i + 1}", "team": team, "cell": list(c), "facing": list(f)}
             for team, cs, f in team_cells for i, c in enumerate(cs)]
    data = {
        "tile": tile, "cols": cols, "rows": rows, "terrains": ts.names,
        "cells": [[x, y, int(cells[y, x, 0]), int(cells[y, x, 1])] for y in range(rows) for x in range(cols)],
        "blocked": sorted([list(b) for b in blocked]),
        "units": units,
        "unit": {"frames": "res://units/unit.tres", "script": "res://units/unit_facing.gd",
                 "materials": {t: f"res://units/unit_{t}.tres" for t in ("player", "enemy", "ally", "other")},
                 "idle": _pick(actions, "idle", "stand", "breath"),
                 "walk": _pick(actions, "walk", "run", "move")},
    }
    files = {
        out / "map" / "map.json": json.dumps(data),
        out / "project.godot": PROJECT.format(name=name, w=VIEW_W, h=VIEW_H, ww=VIEW_W * 3, wh=VIEW_H * 3),
        out / "main.tscn": MAIN_TSCN,
        out / "main.gd": MAIN_GD,
        out / "unit.gd": UNIT_GD,
        out / "README.txt": (f"{name}: open this folder in Godot 4.3+ (Import > project.godot) and "
                             "press F5.\nClick your unit (left), then a tile: it walks there and "
                             "greys out. Space: next turn.\nUnit: " + what + "\n"),
    }
    for path, text in files.items():
        path.write_text(text, encoding="utf-8")
        written.append(path)
    progress(f"Unit: {what}")
    return written, _preview(cells, ts, units, facing, colours, pals, tile)


def _preview(cells, ts, units, facing, colours, pals, tile) -> np.ndarray:
    """The start screen: the map with every unit in its team colours, as main.gd sets it up."""
    screen = Image.fromarray(render_map(cells, ts)[:VIEW_H, :VIEW_W], "RGBA")
    for u in sorted(units, key=lambda u: u["cell"][1]):
        rgba = facing["E" if u["facing"][0] > 0 else "W"]
        sprite = Image.fromarray(recolour(rgba, colours, pals[u["team"]]), "RGBA")
        h, w = rgba.shape[:2]
        cx, cy = (u["cell"][0] + 0.5) * tile, (u["cell"][1] + 0.5) * tile
        oy = round(-h / 2 + tile * 0.3)
        screen.alpha_composite(sprite, (int(cx - w // 2), int(cy + oy - h // 2)))
    return np.asarray(screen)

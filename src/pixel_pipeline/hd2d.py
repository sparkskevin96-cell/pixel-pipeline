"""HD-2D tactics demo: a Godot 4 project with one battle map in three looks.

    real 3D   pixel-art sprites standing in a lit 3D diorama (Triangle Strategy / Octopath):
              perspective telephoto camera that turns in 45° steps, sprites that face the
              camera and pick their direction sprite from it, lit through normal maps,
              tilt-shift blur, bloom, sunset light, torches, sparkling water
    fake 3D   the same scene through an orthographic camera rendered at 640x360 and scaled up
              with nearest neighbour: reads as hand-drawn isometric pixel art, still turns
    iso 2D    classic isometric tiles stacked by height (Tactics Ogre / Disgaea), iso_scene.py

    make_hd2d(out_dir, units={"dog": "dog.png", "knight": "knight.png"})

Units are sprite sheets from `turnaround --animate` (sheet .png + .json, optional _n.png
normal map). The board is diorama.harbour() unless you pass one. Gameplay: click one of the
active team's units, its movement range lights up, click a lit cell to walk there, Space ends
the turn. Q/E turn the camera, WASD pans, the wheel zooms, Esc goes back to the menu.

Lighting: time-of-day presets (LIGHTING: dawn, noon, sunset, night with moonlight; L cycles them),
sprites lit by every light through their normal maps in a few toon bands (B toggles smooth), rim
light, contact + projected shadows; <sheet>_glow.png pixels are emissive, so bloom picks out only
fire, crystals, lanterns and magic. Torch/brazier/lantern/crystal props flicker and throw embers,
spells flash light, effects (add_effect) glow and light their cell.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from .diorama import Diorama, harbour, write_textures

PROJECT = """config_version=5

[application]

config/name="{name}"
run/main_scene="res://menu.tscn"
config/features=PackedStringArray("4.6", "Forward Plus")

[display]

window/size/viewport_width=1920
window/size/viewport_height=1080
window/size/mode=3
window/stretch/mode="canvas_items"
window/stretch/scale_mode="integer"

[rendering]

textures/canvas_textures/default_texture_filter=0
anti_aliasing/quality/msaa_3d=0
"""

MENU_TSCN = """[gd_scene load_steps=2 format=3]

[ext_resource type="Script" path="res://menu.gd" id="1"]

[node name="Menu" type="Control"]
layout_mode = 3
anchors_preset = 15
anchor_right = 1.0
anchor_bottom = 1.0
script = ExtResource("1")
"""

MENU_GD = """extends Control
# Pick a look. Each scene goes back here with Esc.

const LOOKS = [
	["Real 3D  (HD-2D: lit diorama, tilt-shift)", "res://real3d.tscn"],
	["Fake 3D  (3D through a 640x360 pixel camera)", "res://fake3d.tscn"],
	["Pixel-perfect 3D  (one pixel grid: tiles and sprites 1 texel = 1 pixel)", "res://pixel3d.tscn"],
	["Isometric 2D  (stacked tiles, Tactics Ogre style)", "res://iso/iso.tscn"],
]

func _ready() -> void:
	var win := get_tree().root                   # the iso look switches to 640x360 pixels
	win.content_scale_size = Vector2i(1920, 1080)
	win.content_scale_mode = Window.CONTENT_SCALE_MODE_CANVAS_ITEMS
	win.content_scale_stretch = Window.CONTENT_SCALE_STRETCH_INTEGER   # never a non-integer scale
	var bg := ColorRect.new()
	bg.color = Color(0.06, 0.05, 0.09)
	bg.set_anchors_preset(Control.PRESET_FULL_RECT)
	add_child(bg)
	var box := VBoxContainer.new()
	box.set_anchors_preset(Control.PRESET_CENTER)
	box.add_theme_constant_override("separation", 18)
	add_child(box)
	var title := Label.new()
	title.text = "%s" % ProjectSettings.get_setting("application/config/name")
	title.add_theme_font_size_override("font_size", 56)
	title.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	box.add_child(title)
	for look in LOOKS:
		if not ResourceLoader.exists(look[1]) and not FileAccess.file_exists(look[1]):
			continue
		var b := Button.new()
		b.text = look[0]
		b.add_theme_font_size_override("font_size", 30)
		b.custom_minimum_size = Vector2(900, 70)
		b.pressed.connect(func(): get_tree().change_scene_to_file(look[1]))
		box.add_child(b)
	var hint := Label.new()
	hint.text = "Click a unit, click a lit cell to move. Space: end turn. Q/E: turn camera. WASD: pan. Wheel: zoom. L: time of day. Esc: menu."
	hint.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	box.add_child(hint)
	box.position = (get_viewport_rect().size - box.get_combined_minimum_size()) / 2.0
"""

REAL_TSCN = """[gd_scene load_steps=2 format=3]

[ext_resource type="Script" path="res://board.gd" id="1"]

[node name="Real3D" type="Node3D"]
script = ExtResource("1")
mode = "real"
"""

FAKE_TSCN = """[gd_scene load_steps=2 format=3]

[ext_resource type="Script" path="res://board.gd" id="1"]

[node name="Fake3D" type="Control"]
layout_mode = 3
anchors_preset = 15
anchor_right = 1.0
anchor_bottom = 1.0

[node name="Pixels" type="SubViewportContainer" parent="."]
texture_filter = 1
layout_mode = 1
anchors_preset = 15
anchor_right = 1.0
anchor_bottom = 1.0
stretch = true
stretch_shrink = 3

[node name="View" type="SubViewport" parent="Pixels"]
handle_input_locally = false
size = Vector2i(640, 360)
render_target_update_mode = 4

[node name="Board" type="Node3D" parent="Pixels/View"]
script = ExtResource("1")
mode = "fake"
"""

# Pixel-perfect 3D (board mode "pixel"): the board in a SubViewport of screen / scale pixels
# (res://pixel.json), shown `scale` times bigger with nearest filtering. pixel_tscn(scale).
PIXEL_TSCN = """[gd_scene load_steps=2 format=3]

[ext_resource type="Script" path="res://board.gd" id="1"]

[node name="Pixel3D" type="Control"]
layout_mode = 3
anchors_preset = 15
anchor_right = 1.0
anchor_bottom = 1.0

[node name="Pixels" type="SubViewportContainer" parent="."]
texture_filter = 1
layout_mode = 1
anchors_preset = 15
anchor_right = 1.0
anchor_bottom = 1.0
stretch = true
stretch_shrink = {scale}

[node name="View" type="SubViewport" parent="Pixels"]
handle_input_locally = false
size = Vector2i({w}, {h})
render_target_update_mode = 4

[node name="Board" type="Node3D" parent="Pixels/View"]
script = ExtResource("1")
mode = "pixel"
"""


def pixel_tscn(scale: int, screen=(1920, 1080)) -> str:
    return PIXEL_TSCN.format(scale=int(scale), w=screen[0] // int(scale), h=screen[1] // int(scale))


def write_pixel_look(out, tile: int, scale: int, pitch: float) -> list[Path]:
    """res://pixel.json + res://pixel3d.tscn: the pixel-perfect look of the board (one pixel
    grid: `tile` texels per tile on screen at 1 texel = 1 pixel, `scale` whole screen pixels per
    pixel, the camera at `pitch` degrees)."""
    out = Path(out)
    pj = out / "pixel.json"
    pj.write_text(json.dumps({"tile": int(tile), "scale": int(scale), "pitch": float(pitch),
                              "base": [1920 // int(scale), 1080 // int(scale)]}), encoding="utf-8")
    ts = out / "pixel3d.tscn"
    ts.write_text(pixel_tscn(scale), encoding="utf-8")
    return [pj, ts]


# The board's sprite shader is the game runtime's (godot_runtime/pp_sprite3d.gdshader): camera-facing
# quads lit through the normal maps in toon bands, depth taken at the FEET (no clipping into the
# ground, walls, stairs or each other) and pixel-snapped (no shimmer while they walk).
from .godot_runtime import source as _runtime_source  # noqa: E402

SPRITE_SHADER = _runtime_source("pp_sprite3d.gdshader")

TERRAIN_SHADER = """shader_type spatial;
// The board's ground (cell tops and walls): the pixel texture + its per-texel normal map, lit
// like the sprites - each light's diffuse snapped to `bands` steps, so the light falls in whole
// texel bands (pixel-art shading) instead of smearing, and the drawn shapes of bright materials
// (snow, sand) survive noon. Lower bands lean to `shade_tint` (hue-shifted, cool shadows).
// Bright tops are capped below ~0.85 albedo in Python (neo_terrain.board_albedo).
render_mode depth_draw_opaque;

uniform sampler2D albedo_tex : source_color, filter_nearest_mipmap;
uniform sampler2D normal_tex : hint_normal, filter_nearest_mipmap;
uniform bool has_normals = false;
uniform float normal_depth = 1.0;
uniform vec3 tint = vec3(1.0);
uniform bool use_vcol = false;
uniform float roughness = 0.95;
uniform float specular = 0.0;                    // wet ground (weather) gets a little
uniform float bands = 5.0;                       // diffuse steps per light, 0 = smooth
uniform float wrap = 0.1;
uniform vec3 shade_tint = vec3(0.9, 0.93, 1.06);

void fragment() {
	vec3 c = texture(albedo_tex, UV).rgb * tint;
	if (use_vcol) {
		c *= COLOR.rgb;
	}
	ALBEDO = c;
	ROUGHNESS = roughness;
	SPECULAR = specular;
	if (has_normals) {
		NORMAL_MAP = texture(normal_tex, UV).rgb;
		NORMAL_MAP_DEPTH = normal_depth;
	}
}

void light() {
	float d = clamp((dot(NORMAL, LIGHT) + wrap) / (1.0 + wrap), 0.0, 1.0);
	if (bands > 0.5) {
		d = floor(d * bands + 0.5) / bands;      // whole steps per texel (normals are nearest)
	}
	float lit = d * ATTENUATION;
	DIFFUSE_LIGHT += LIGHT_COLOR * lit * mix(shade_tint, vec3(1.0), lit) / PI;
	if (specular > 0.0) {
		vec3 h = normalize(LIGHT + VIEW);
		float sp = pow(clamp(dot(NORMAL, h), 0.0, 1.0), 40.0) * specular * ATTENUATION;
		SPECULAR_LIGHT += LIGHT_COLOR * (sp > 0.25 ? 0.5 : 0.0) / PI;
	}
}
"""

SHADOW_BLOB_SHADER = """shader_type spatial;
// A soft dark ellipse under a unit (HD-2D contact shadow), on top of the terrain.
render_mode unshaded, cull_disabled, depth_draw_never, shadows_disabled;
uniform float strength = 0.4;
void fragment() {
	float d = length(UV * 2.0 - 1.0);
	float a = smoothstep(1.0, 0.25, d);
	ALBEDO = vec3(0.02, 0.01, 0.04);
	ALPHA = a * strength;
}
"""

WATER_SHADER = """shader_type spatial;
// Deep blue water with moving sparkles: light catches the ripples (glow turns them to bloom).
render_mode cull_disabled;

uniform sampler2D ripples : filter_linear, repeat_enable;
uniform vec3 deep : source_color = vec3(0.04, 0.10, 0.30);
uniform vec3 shallow : source_color = vec3(0.10, 0.28, 0.62);
uniform float sparkle = 2.5;
uniform float spec = 0.8;          // sun highlight strength (lighting.json "water_spec")
uniform float glare = 0.6;         // the highlight never goes brighter than this (no white-out at noon)

void vertex() {
	VERTEX.y += sin(TIME * 1.3 + VERTEX.x * 0.9) * 0.02 + cos(TIME * 1.1 + VERTEX.z * 0.7) * 0.02;
}

void light() {
	// diffuse + a CAPPED, banded sun highlight: a high sun mirrored into a steep camera used to
	// blow the whole pond out to white (the action map at noon)
	DIFFUSE_LIGHT += LIGHT_COLOR * clamp(dot(NORMAL, LIGHT), 0.0, 1.0) * ATTENUATION / PI;
	vec3 h = normalize(LIGHT + VIEW);
	float sp = pow(clamp(dot(NORMAL, h), 0.0, 1.0), 90.0) * spec * ATTENUATION;
	sp = min(floor(sp * 4.0) / 4.0, glare);
	SPECULAR_LIGHT += LIGHT_COLOR * sp;
}

void fragment() {
	vec2 w = (INV_VIEW_MATRIX * vec4(VERTEX, 1.0)).xz;
	float a = texture(ripples, w * 0.11 + vec2(TIME * 0.015, TIME * 0.01)).r;
	float b = texture(ripples, w * 0.17 - vec2(TIME * 0.012, -TIME * 0.02)).r;
	float r = a * b;
	ALBEDO = mix(deep, shallow, smoothstep(0.1, 0.5, r));
	// glints only on the rare highest ripples, flickering, so the water twinkles
	float g = texture(ripples, w * 0.9 + vec2(TIME * 0.05, -TIME * 0.04)).r;
	float s = smoothstep(0.78, 0.84, g) * smoothstep(0.35, 0.55, r);
	EMISSION = vec3(0.85, 0.92, 1.0) * s * sparkle;
	ROUGHNESS = 0.08;
	METALLIC = 0.0;
	SPECULAR = 0.8;
	ALPHA = 0.88;
}
"""

WATER_SHADER_PIXEL = """shader_type spatial;
// Pixel-art water (make_hd2d(water="pixel"), water.py's look in 3D): ripples in whole water
// pixels stepping at 8 fps, colour bands, crest lines, glints that bloom, foam where the water
// meets the shore (from the depth buffer), and a rippling reflection of what stands above it
// (sprites, torches: taken from the screen, so bright things still bloom in the water).
render_mode cull_disabled;

uniform sampler2D ripples : filter_linear, repeat_enable;
uniform sampler2D screen_tex : hint_screen_texture, filter_linear_mipmap;
uniform sampler2D depth_tex : hint_depth_texture, filter_nearest;
uniform vec3 deep : source_color = vec3(0.04, 0.10, 0.30);
uniform vec3 shallow : source_color = vec3(0.10, 0.28, 0.62);
uniform float sparkle = 2.5;
uniform float pixel = 0.03125;
uniform float bands = 5.0;
uniform float reflect = 0.4;
uniform float reflect_offset = 0.045;
uniform float foam_width = 0.16;

void vertex() {
	VERTEX.y += sin(TIME * 1.3 + VERTEX.x * 0.9) * 0.015 + cos(TIME * 1.1 + VERTEX.z * 0.7) * 0.015;
}

void fragment() {
	vec2 w = (INV_VIEW_MATRIX * vec4(VERTEX, 1.0)).xz;
	vec2 q = floor(w / pixel) * pixel + pixel * 0.5;
	float t = floor(TIME * 8.0) / 8.0;
	float a = texture(ripples, q * 0.07 + vec2(t * 0.012, t * 0.008)).r;
	float b = texture(ripples, q * 0.11 - vec2(t * 0.01, -t * 0.016)).r;
	float r = a * b;
	float lv = floor(smoothstep(0.08, 0.5, r) * bands) / bands;
	vec3 col = mix(deep, shallow, lv * 0.6);
	float c = texture(ripples, vec2(q.x * 0.35, q.y * 1.4) + vec2(t * 0.03, 0.0)).r;
	float crest = step(0.74, c) * step(0.3, r);
	col = mix(col, shallow * 1.35 + vec3(0.04), crest * 0.45);
	float d = texture(depth_tex, SCREEN_UV).r;
	vec4 v = INV_PROJECTION_MATRIX * vec4(SCREEN_UV * 2.0 - 1.0, d, 1.0);
	float thick = -v.z / v.w + VERTEX.z;
	float foam = 1.0 - smoothstep(0.0, foam_width, thick);
	float fq = step(0.55, foam) + step(0.2, foam) * step(0.5, fract((q.x + q.y) * 6.0 + t * 0.5));
	vec2 ruv = SCREEN_UV + vec2((a - 0.5) * 0.012, -reflect_offset + (b - 0.5) * 0.01);
	vec3 refl = textureLod(screen_tex, clamp(ruv, vec2(0.001), vec2(0.999)), 0.0).rgb;
	float fres = reflect * (0.6 + 0.4 * pow(1.0 - abs(dot(NORMAL, VIEW)), 2.0));
	col = mix(col, refl, fres * 0.7);
	col = mix(col, mix(shallow, vec3(0.85, 0.92, 1.0), 0.6), clamp(fq, 0.0, 1.0) * 0.55);
	ALBEDO = col;
	float bright = max(max(refl.r, refl.g), refl.b);
	EMISSION = refl * smoothstep(0.75, 1.2, bright) * reflect * 1.5;
	float g = texture(ripples, q * 0.9 + vec2(t * 0.05, -t * 0.04)).r;
	EMISSION += vec3(0.85, 0.92, 1.0) * step(0.8, g) * smoothstep(0.35, 0.55, r) * sparkle;
	ROUGHNESS = 0.06;
	METALLIC = 0.0;
	SPECULAR = 0.8;
	ALPHA = 0.9;
}
"""
WATER_LOOKS = ("classic", "pixel")

FLAME_SHADER = """shader_type spatial;
// A pixel-art flame on a camera-facing quad: a coarse pixel grid, three hard colour bands,
// HDR bright so only the fire blooms.
render_mode unshaded, cull_disabled, blend_add, shadows_disabled;
uniform float seed = 0.0;
uniform vec2 grid = vec2(10.0, 14.0);
uniform vec3 hot : source_color = vec3(1.0, 0.82, 0.42);
uniform vec3 mid : source_color = vec3(1.0, 0.42, 0.06);
uniform vec3 cool : source_color = vec3(0.6, 0.08, 0.02);
uniform float energy = 1.7;
uniform float level = 1.0;            // time of day: flames look brighter at night
void vertex() {
	MODELVIEW_MATRIX = VIEW_MATRIX * mat4(INV_VIEW_MATRIX[0], INV_VIEW_MATRIX[1], INV_VIEW_MATRIX[2], MODEL_MATRIX[3]);
}
float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
void fragment() {
	vec2 g = (floor(UV * grid) + 0.5) / grid;            // snap to the flame's pixel grid
	vec2 p = g * 2.0 - 1.0;                               // y down: -1 top, 1 bottom
	float t = floor(TIME * 10.0 + seed * 3.0);            // flicker in steps, like sprite frames
	float sway = sin(TIME * 5.0 + seed * 2.0 + p.y * 3.0) * 0.12 * (1.0 - p.y) * 0.5;
	float w = mix(0.2, 0.95, smoothstep(-1.0, 0.6, p.y));
	float body = 1.0 - abs(p.x - sway) / max(w, 0.05);
	body -= (1.0 - p.y) * 0.22 * hash(floor(UV * grid) + t);
	body *= smoothstep(1.0, 0.7, p.y);
	if (body <= 0.05) { discard; }
	vec3 col = body > 0.72 ? hot * 1.4 : (body > 0.36 ? mid : cool);
	ALBEDO = col * energy * mix(0.8, 1.1, clamp(level, 0.0, 1.5));
	ALPHA = 1.0;
}
"""

EMBER_SHADER = """shader_type spatial;
// Embers and sparks: tiny square pixels, HDR bright, facing the camera.
render_mode unshaded, cull_disabled, blend_add, shadows_disabled;
uniform float energy = 2.4;
void vertex() {
	MODELVIEW_MATRIX = VIEW_MATRIX * mat4(INV_VIEW_MATRIX[0], INV_VIEW_MATRIX[1], INV_VIEW_MATRIX[2], MODEL_MATRIX[3]);
}
void fragment() {
	ALBEDO = COLOR.rgb * energy;
	ALPHA = COLOR.a;
}
"""

LIGHT_GD = """extends OmniLight3D
# The light of a flame, lantern, crystal or spell: energy (and a little position) wobble like fire.
# `level` comes from the time of day (torches burn brighter at night, are nearly off at noon).

@export var base_energy := 2.0
@export var flicker := 0.18          # 0 = steady
@export var pulse := 0.0             # slow breathing (crystals, magic)
@export var seed := 0.0
var level := 1.0
var _home := Vector3.ZERO
var _noise := FastNoiseLite.new()

func _ready() -> void:
	_home = position
	_noise.seed = int(seed * 1013.0)
	_noise.frequency = 1.0

func _process(_delta: float) -> void:
	var t := Time.get_ticks_msec() / 1000.0
	var f := _noise.get_noise_1d(t * 7.0 + seed * 11.0) * flicker * 1.6 + sin(t * 23.0 + seed * 3.1) * flicker * 0.35
	var p := sin(t * 1.6 + seed) * pulse
	light_energy = max(0.0, base_energy * level * (1.0 + f + p))
	visible = level > 0.02
	if flicker > 0.0:
		position = _home + Vector3(_noise.get_noise_2d(t * 5.0, seed * 7.0),
			_noise.get_noise_2d(seed * 7.0, t * 5.0) * 0.5, _noise.get_noise_2d(t * 5.0 + 40.0, seed)) * 0.05
"""

BOARD_GD = r'''extends Node3D
# The battle map in 3D, built at runtime from res://map/map.json, and the units on it.
# mode "real": perspective telephoto + tilt-shift + bloom.  "fake": orthographic, rendered
# small by the parent SubViewport and scaled up: 3D that reads as pixel art.
# mode "pixel" (pixel-perfect, 2026-10-17): like fake, but ONE pixel grid: res://pixel.json
# {tile, scale, pitch} - the SubViewport is the screen / scale (a whole number), the ortho camera
# shows one world tile as `tile` pixels, and the board is stretched (z by 1/sin(pitch), y by
# 1/cos(pitch)) so cell tops AND walls are drawn 1 texel = 1 pixel (an oblique 3/4 view, like
# 16-bit top-down games); sprites are cards of 1 texel = 1 pixel (the sprite shader's
# whole_texels). The camera keeps yaw 0 and moves in whole pixels. Gameplay coordinates are the
# board's own (local) ones in every mode.

@export var mode := "real"
# The game runtime (godot_runtime/pp_motion.gd, res://pp_runtime/): eased movement, turning through
# the 16 facings, stride-matched walk/run (feet planted), settle on stop, hops on steps, hit-stop,
# shake, critically damped camera. Sprites: res://shaders/sprite.gdshader = pp_sprite3d.gdshader
# (depth at the feet: no clipping into walls, stairs, props or each other; pixel-snapped).
const M := preload("res://pp_runtime/pp_motion.gd")

const DIRS := ["E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW", "N", "NNE", "NE", "ENE"]
const MOVE := 4
const JUMP := 2

var map := {}
var level := 0.5
var cols := 0
var rows := 0
var heights := []
var terrain := []
var units := []          # [{node, mesh, mat, sheet, col, row, team, yaw, anim, t}]
var sheets := {}         # name -> {tex, ntex, size, frames: {"walk_E": [{rect, ms}]}, foot}
var active_team := 0
var selected: Variant = null
var reach := {}          # Vector2i -> [steps, parent]
var overlay: Node3D
var cursor: MeshInstance3D
var moving := false
var pivot: Node3D
var cam: Camera3D
var cam_yaw := 45.0
var cam_yaw_target := 45.0
var cam_dist := 24.0
var hud: Label
var blocked := {}
# lighting (res://lighting.json: time-of-day presets; L cycles them, B toggles the toon bands)
var env: Environment
var sun: DirectionalLight3D
var fill: DirectionalLight3D
var lighting := {}       # preset name -> settings
var light_name := "sunset"
var bands := 4.0
var sprite_bands := 3.0  # the sprites' light steps (lighting.json "sprite_bands"; calmer than the ground)
var terrain_bands := 5.0 # the ground's light steps (lighting.json "terrain_bands"; B toggles both)
var terrain_mats := []   # the ground's ShaderMaterials (terrain.gdshader)
var flames := []         # flame materials (brighter at night)
var fire_lights := []    # light_flicker.gd lights of torches, braziers, lanterns, crystals
var blobs := []          # contact shadow materials
var water_mat: ShaderMaterial
var effects := {}        # name -> effect sheet (res://effects/<name>/)
var live_effects := []   # playing effects: {node, mat, e, t, loop, light, energy}
var weather: Node = null  # res://weather/board_weather.gd (make_hd2d(weather=...))
# autoplay (res://autoplay.gd, `--autoplay[=res://autoplay.json]`): a script drives the board
var scripted := false    # true: no player input, no automatic turn passing; action_done after each act
signal action_done
var cam_pitch := -1.0    # degrees; -1 = the look's default (the sheets' drawn elevation, else real 38, fake 30)
var pixel := {}          # res://pixel.json (mode "pixel"): tile px, scale, pitch
var tile_px := 32.0      # pixel mode: screen pixels per world tile (= the tile textures' texels)
var view_pitch := 38.0   # the angle the ground is seen at (pixel mode: 90 - tops are drawn 1:1)
var sheet_elev := 0.0    # the elevation the unit sheets were drawn at (unit.json, 0 = unknown)
var cam_turn_rate := 8.0 # how fast the camera eases to cam_yaw_target
var prop_nodes := {}     # Vector2i -> the nodes a prop on that cell built (autoplay cuts bushes)
var number_lift := 1.6   # damage numbers start this high over a unit
var conf := {}           # motion tunables (res://pp_runtime/motion.json over pp_motion DEFAULTS)
var freeze := 0.0        # hit-stop: sprites, movers and effects hold while > 0
var trauma := 0.0        # camera shake (0..1, squared)
var cam_target = null    # Vector3 the camera follows (critically damped), null = stays
var cam_vel := Vector3.ZERO
var cam_smooth := 0.28
var bars_layer: CanvasLayer   # HP bars + damage numbers drawn in 2D over the 3D view (no DOF blur)
var floaters := []       # [{node, pos, t, life, rise}] 2D labels that follow a 3D point
var obstacles := {}      # Vector2i -> {"r": radius (0 = the whole cell), "kind"} (props that block)
var apron := 0           # cells of scenery drawn around the map (map.json "apron")
var move_speed := 2.2    # tiles/s a tactics unit wants to walk a path (capped by its stride)
var clock := 0.0
# what blocks the FEET (collision radius at the ground, tiles; 0 = the whole cell): trunks, not
# canopies, so the gaps between props are as wide as they look
const BLOCKING := {"crate": 0.38, "barrel": 0.26, "post": 0.12, "tree": 0.2, "rock": 0.3, "house": 0.0,
	"wall": 0.0, "well": 0.4, "fence": 0.0, "battlement": 0.0, "brazier": 0.24, "crystal": 0.26,
	"bush": 0.24, "pot": 0.2, "statue": 0.26, "cabin": 0.0}
const LIGHT_ORDER := ["dawn", "noon", "sunset", "night"]
const SPELL := {"mage": Color(0.72, 0.42, 1.0), "cleric": Color(1.0, 0.84, 0.45), "physician": Color(0.5, 1.0, 0.62)}

func _ready() -> void:
	conf = M.settings()
	cam_smooth = float(conf.get("cam_smooth", 0.28))
	map = JSON.parse_string(FileAccess.get_file_as_string("res://map/map.json"))
	level = map.get("level", 0.5)
	apron = int(map.get("apron", 0))
	cols = int(map["cols"])
	rows = int(map["rows"])
	heights = map["height"]
	terrain = map["terrain"]
	if mode == "pixel":
		_pixel_setup()
	_environment()
	_build_terrain()
	_build_water()
	_build_props()
	_build_units()
	_build_camera()
	overlay = Node3D.new()
	add_child(overlay)
	_build_cursor()
	_build_effects()
	_build_weather()
	_hud()
	apply_lighting(light_name)
	_start_autoplay()

func _pixel_setup() -> void:
	if FileAccess.file_exists("res://pixel.json"):
		pixel = JSON.parse_string(FileAccess.get_file_as_string("res://pixel.json"))
	tile_px = float(pixel.get("tile", 32))
	if cam_pitch < 0.0:
		cam_pitch = float(pixel.get("pitch", 45.0))
	var pr := deg_to_rad(clampf(cam_pitch, 10.0, 80.0))
	# oblique: a tile's depth and a level's height both land on whole texels on screen
	scale = Vector3(1.0, 1.0 / cos(pr), 1.0 / sin(pr))
	view_pitch = 90.0

func g(p: Vector3) -> Vector3:
	# board (gameplay) coordinates -> the world the camera sees (the pixel look stretches the board)
	return to_global(p)

func _start_autoplay() -> void:
	# --autoplay=res://x.json (or --autoplay = res://autoplay.json), or autoplay.json with "auto": true
	var path := ""
	for a in OS.get_cmdline_user_args():
		if a == "--autoplay":
			path = "res://autoplay.json"
		elif a.begins_with("--autoplay="):
			path = a.substr(11)
	if path == "" and FileAccess.file_exists("res://autoplay.json"):
		var j = JSON.parse_string(FileAccess.get_file_as_string("res://autoplay.json"))
		if j is Dictionary and bool(j.get("auto", false)):
			path = "res://autoplay.json"
	if path == "" or not FileAccess.file_exists("res://autoplay.gd"):
		return
	var ap: Node = load("res://autoplay.gd").new()
	ap.name = "Autoplay"
	add_child(ap)
	ap.start(self, path)

# ------------------------------------------------------------------ helpers

func _tex(path: String) -> Texture2D:
	if not FileAccess.file_exists(path):
		return null
	var img := Image.load_from_file(ProjectSettings.globalize_path(path))
	if img == null:
		return null
	img.generate_mipmaps()
	return ImageTexture.create_from_image(img)

func top_y(c: int, r: int) -> float:
	return float(heights[r][c]) * level

func cell_pos(c: int, r: int) -> Vector3:
	return Vector3(c + 0.5, top_y(c, r), r + 0.5)

func walkable(c: int, r: int) -> bool:
	if c < 0 or r < 0 or c >= cols or r >= rows:
		return false
	if terrain[r][c] == "water" or blocked.has(Vector2i(c, r)):
		return false
	return true

# ------------------------------------------------------------------ world

func _environment() -> void:
	if FileAccess.file_exists("res://lighting.json"):
		var j = JSON.parse_string(FileAccess.get_file_as_string("res://lighting.json"))
		if j is Dictionary:
			lighting = j.get("presets", {})
			light_name = j.get("default", light_name)
			bands = float(j.get("bands", bands))
			sprite_bands = float(j.get("sprite_bands", sprite_bands))
			terrain_bands = float(j.get("terrain_bands", terrain_bands))
	for a in OS.get_cmdline_user_args():            # --light=night --bands=0 (screenshots, tests)
		if a.begins_with("--light="):
			light_name = a.substr(8)
		elif a.begins_with("--bands="):
			bands = float(a.substr(8))
	var we := WorldEnvironment.new()
	env = Environment.new()
	env.background_mode = Environment.BG_COLOR
	env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.tonemap_mode = Environment.TONE_MAPPER_ACES
	# bloom only on HDR-bright pixels: emissive sprite pixels (_glow.png), flames, embers, glints
	env.glow_enabled = true
	env.glow_bloom = 0.0
	env.glow_hdr_threshold = 1.05
	env.glow_hdr_scale = 2.0
	env.glow_strength = 1.0
	env.glow_blend_mode = Environment.GLOW_BLEND_MODE_SCREEN
	var levels := [0.0, 0.55, 1.0, 0.75, 0.4, 0.15, 0.0]
	for i in levels.size():
		env.set_glow_level(i, levels[i])
	env.ssao_enabled = mode == "real"
	env.ssao_intensity = 1.6
	env.fog_enabled = true
	env.adjustment_enabled = true
	we.environment = env
	add_child(we)
	sun = DirectionalLight3D.new()
	sun.shadow_enabled = true
	sun.shadow_blur = 1.5
	sun.directional_shadow_max_distance = 80.0
	add_child(sun)
	fill = DirectionalLight3D.new()                 # cool bounce from the other side
	add_child(fill)
	if lighting.is_empty():                         # an old project: the sunset it always had
		lighting = {"sunset": {"sun": [1.0, 0.72, 0.45], "sun_energy": 1.25, "sun_angle": [-38, -60],
			"fill": [0.45, 0.55, 1.0], "fill_energy": 0.35, "fill_angle": [-55, 120],
			"ambient": [0.45, 0.38, 0.62], "ambient_energy": 0.55, "background": [0.03, 0.02, 0.05]}}

func _c(v, fallback := Color.WHITE) -> Color:
	return Color(v[0], v[1], v[2]) if v is Array and v.size() >= 3 else fallback

func apply_lighting(name: String) -> void:
	# a time-of-day preset: sun (or moon), fill, ambient, fog, torches, rim light, water
	if not lighting.has(name):
		name = "sunset" if lighting.has("sunset") else str(lighting.keys()[0])
	light_name = name
	var L: Dictionary = lighting[name]
	env.background_color = _c(L.get("background"), Color(0.03, 0.02, 0.05))
	env.ambient_light_color = _c(L.get("ambient"), Color(0.45, 0.38, 0.62))
	env.ambient_light_energy = float(L.get("ambient_energy", 0.55))
	env.tonemap_exposure = float(L.get("exposure", 1.05))
	env.tonemap_white = float(L.get("white", 1.0))
	env.glow_intensity = float(L.get("glow", 0.8)) * (1.0 if mode == "real" else 0.75)
	env.adjustment_saturation = float(L.get("saturation", 1.15))
	env.adjustment_contrast = float(L.get("contrast", 1.06))
	env.fog_light_color = _c(L.get("fog"), Color(0.5, 0.45, 0.6))
	env.fog_density = float(L.get("fog_density", 0.004))
	env.fog_sky_affect = 0.0
	var vol := float(L.get("volumetric", 0.0))
	env.volumetric_fog_enabled = mode == "real" and vol > 0.0
	env.volumetric_fog_density = vol
	env.volumetric_fog_albedo = _c(L.get("fog"), Color(0.5, 0.5, 0.6))
	env.volumetric_fog_length = 40.0
	sun.light_color = _c(L.get("sun"))
	sun.light_energy = float(L.get("sun_energy", 1.2))
	var sa: Array = L.get("sun_angle", [-38, -60])
	sun.rotation_degrees = Vector3(sa[0], sa[1], 0)
	sun.light_volumetric_fog_energy = 0.15
	fill.light_color = _c(L.get("fill"), Color(0.45, 0.55, 1.0))
	fill.light_energy = float(L.get("fill_energy", 0.35))
	var fa: Array = L.get("fill_angle", [-55, 120])
	fill.rotation_degrees = Vector3(fa[0], fa[1], 0)
	var torch := float(L.get("torches", 1.0))
	for l in fire_lights:
		l.level = torch
	for f in flames:
		f.set_shader_parameter("level", torch)
	for b in blobs:
		b.set_shader_parameter("strength", float(L.get("blob", 0.4)))
	for m in terrain_mats:
		m.set_shader_parameter("bands", terrain_bands if bands > 0.5 else 0.0)
	for u in units:
		_sprite_lighting(u["mat"])
	for e in live_effects:
		_sprite_lighting(e["mat"])
	if water_mat != null:
		water_mat.set_shader_parameter("deep", _c(L.get("water_deep"), Color(0.04, 0.10, 0.30)))
		water_mat.set_shader_parameter("shallow", _c(L.get("water_shallow"), Color(0.10, 0.28, 0.62)))
		water_mat.set_shader_parameter("sparkle", float(L.get("sparkle", 2.5)))
		water_mat.set_shader_parameter("spec", float(L.get("water_spec", 0.8)))
		water_mat.set_shader_parameter("glare", float(L.get("glare", 0.6)))
	if weather != null:
		weather.apply(L)
	if hud != null:
		_update_hud()

func _sprite_lighting(mat: ShaderMaterial) -> void:
	var L: Dictionary = lighting.get(light_name, {})
	mat.set_shader_parameter("bands", sprite_bands if bands > 0.5 else 0.0)
	mat.set_shader_parameter("band_soft", float(L.get("band_soft", 0.18)))
	mat.set_shader_parameter("rim", float(L.get("rim", 0.5)))
	mat.set_shader_parameter("wrap", float(L.get("wrap", 0.35)))
	var amb := _c(L.get("ambient"), Color(0.45, 0.38, 0.62)) * float(L.get("ambient_energy", 0.55))
	mat.set_shader_parameter("lift", Vector3(amb.r, amb.g, amb.b) * float(L.get("lift", 0.3)))

var tex_info := {}       # res://map/tex/terrain.json: variants, weights, lips per terrain

func _material(albedo: Texture2D, normal: Texture2D, rough: float, vcol: bool, relief := 1.0) -> ShaderMaterial:
	# the ground: terrain.gdshader (banded per-texel light, see its header)
	var m := ShaderMaterial.new()
	m.shader = load("res://shaders/terrain.gdshader")
	m.set_shader_parameter("albedo_tex", albedo)
	m.set_shader_parameter("roughness", rough)
	if normal != null:
		m.set_shader_parameter("normal_tex", normal)
		m.set_shader_parameter("has_normals", true)
		m.set_shader_parameter("normal_depth", relief)
	m.set_shader_parameter("use_vcol", vcol)
	m.set_shader_parameter("bands", terrain_bands)
	terrain_mats.append(m)
	return m

func _lip_material(albedo: Texture2D) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.albedo_texture = albedo
	m.texture_filter = BaseMaterial3D.TEXTURE_FILTER_NEAREST_WITH_MIPMAPS
	m.roughness = 0.95
	m.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA_SCISSOR
	m.alpha_scissor_threshold = 0.3
	return m

func _materials() -> Dictionary:
	# [top, side, lip] per terrain; tops = the variants side by side (_tops.png) when
	# terrain.json lists them, else the single _top.png repeated per cell (older projects)
	if FileAccess.file_exists("res://map/tex/terrain.json"):
		var j = JSON.parse_string(FileAccess.get_file_as_string("res://map/tex/terrain.json"))
		if j is Dictionary:
			tex_info = j.get("terrains", {})
	var mats := {}
	for name in ["grass", "stone", "cobble", "planks", "dirt", "sand", "water", "snow"]:
		var info: Dictionary = tex_info.get(name, {})
		var top: Texture2D = null
		var top_n: Texture2D = null
		if info.has("variants"):
			top = _tex("res://map/tex/%s_tops.png" % name)
			top_n = _tex("res://map/tex/%s_tops_n.png" % name)
		if top == null:
			info = {}
			top = _tex("res://map/tex/%s_top.png" % name)
		if top == null:
			continue
		tex_info[name] = info
		var m := _material(top, top_n, 0.95, false, float(info.get("relief", 1.0)))
		var s := _material(_tex("res://map/tex/%s_side.png" % name), _tex("res://map/tex/%s_side_n.png" % name), 1.0, true)
		if not info.has("variants"):
			s.set_shader_parameter("tint", Vector3(0.86, 0.84, 0.9))
		var lip: StandardMaterial3D = null
		if info.has("lip"):
			lip = _lip_material(_tex("res://map/tex/" + str(info["lip"])))
		mats[name] = [m, s, lip]
	return mats

func _quad(st: SurfaceTool, a: Vector3, b: Vector3, c: Vector3, d: Vector3, n: Vector3, uv: Array, col: Array = []) -> void:
	var verts := [[a, uv[0], 0], [b, uv[1], 1], [c, uv[2], 2], [a, uv[0], 0], [c, uv[2], 2], [d, uv[3], 3]]
	for idx in verts:
		st.set_normal(n)
		st.set_uv(idx[1])
		if not col.is_empty():
			st.set_color(col[idx[2]])
		st.add_vertex(idx[0])

func _variant(name: String, c: int, r: int) -> int:
	# which top variant a cell shows: a hash of the cell, by weight (same every run)
	var info: Dictionary = tex_info.get(name, {})
	var w: Array = info.get("weights", [])
	if w.size() <= 1:
		return 0
	var u := fposmod(sin(float(c) * 12.9898 + float(r) * 78.233) * 43758.5453, 1.0)
	var total := 0.0
	for x in w:
		total += float(x)
	var acc := 0.0
	for i in w.size():
		acc += float(w[i]) / total
		if u < acc:
			return i
	return w.size() - 1

func _cell_src(c: int, r: int) -> Vector2i:
	# apron cells (outside the map) copy the nearest edge cell: the land goes on past the edge
	return Vector2i(clampi(c, 0, cols - 1), clampi(r, 0, rows - 1))

func _build_terrain() -> void:
	var mats := _materials()
	var tops := {}
	var sides := {}
	var lips := {}
	var used := {}
	var body := StaticBody3D.new()
	add_child(body)
	for r in range(-apron, rows + apron):
		for c in range(-apron, cols + apron):
			var src := _cell_src(c, r)
			var outside := src != Vector2i(c, r)
			var t: String = terrain[src.y][src.x]
			if t == "water":
				continue
			if outside and t == "planks":
				t = "grass" if tex_info.has("grass") or mats.has("grass") else t
			var y := top_y(src.x, src.y)
			if not tops.has(t):
				tops[t] = SurfaceTool.new()
				tops[t].begin(Mesh.PRIMITIVE_TRIANGLES)
				sides[t] = SurfaceTool.new()
				sides[t].begin(Mesh.PRIMITIVE_TRIANGLES)
				lips[t] = SurfaceTool.new()
				lips[t].begin(Mesh.PRIMITIVE_TRIANGLES)
			var info: Dictionary = tex_info.get(t, {})
			var uv_top := [Vector2(c, r), Vector2(c + 1, r), Vector2(c + 1, r + 1), Vector2(c, r + 1)]
			if info.has("variants"):
				var nv := float(info["variants"])
				var v := float(_variant(t, c, r))
				uv_top = [Vector2(v / nv, 0), Vector2((v + 1.0) / nv, 0), Vector2((v + 1.0) / nv, 1), Vector2(v / nv, 1)]
			_quad(tops[t], Vector3(c, y, r), Vector3(c + 1, y, r), Vector3(c + 1, y, r + 1), Vector3(c, y, r + 1),
				Vector3.UP, uv_top)
			# walls down to each lower neighbour (or below the water for edges)
			for nb in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
				var nc: int = c + nb[0]
				var nr: int = r + nb[1]
				var ny := -1.0
				var nsrc := _cell_src(nc, nr)
				var inside := nc >= -apron and nr >= -apron and nc < cols + apron and nr < rows + apron
				if inside and terrain[nsrc.y][nsrc.x] != "water":
					ny = top_y(nsrc.x, nsrc.y)
				if ny >= y:
					continue
				var n := Vector3(nb[0], 0, nb[1])
				var p0 := Vector3(c + 1, 0, r)
				var p1 := Vector3(c, 0, r)
				if nb[0] == 1:
					p0 = Vector3(c + 1, 0, r + 1)
					p1 = Vector3(c + 1, 0, r)
				elif nb[0] == -1:
					p0 = Vector3(c, 0, r)
					p1 = Vector3(c, 0, r + 1)
				elif nb[1] == 1:
					p0 = Vector3(c, 0, r + 1)
					p1 = Vector3(c + 1, 0, r + 1)
				used[t] = true
				var u0 := 0.0
				var u1 := 1.0
				# darker toward the foot of the wall (occlusion); its lit top stays full
				var foot := clampf(1.0 - 0.16 * (y - ny), 0.6, 0.92)
				var top_c := Color(1, 1, 1)
				var foot_c := Color(foot, foot * 0.97, foot * 1.02)
				_quad(sides[t], p0 + Vector3(0, y, 0), p1 + Vector3(0, y, 0), p1 + Vector3(0, ny, 0), p0 + Vector3(0, ny, 0),
					n, [Vector2(u0, -y), Vector2(u1, -y), Vector2(u1, -ny), Vector2(u0, -ny)], [top_c, top_c, foot_c, foot_c])
				if info.has("lip"):
					# the overhang strip: the top's ground hanging over the wall's edge
					var full := float(info.get("lip_height", 0.5))
					var lh: float = minf(full, y - ny)
					var off := n * 0.004
					_quad(lips[t], p0 + Vector3(0, y, 0) + off, p1 + Vector3(0, y, 0) + off, p1 + Vector3(0, y - lh, 0) + off,
						p0 + Vector3(0, y - lh, 0) + off, n, [Vector2(u0, 0), Vector2(u1, 0), Vector2(u1, lh / full), Vector2(u0, lh / full)])
			if outside:
				continue                               # scenery only: no picking, no walking
			var shape := CollisionShape3D.new()
			var box := BoxShape3D.new()
			box.size = Vector3(1, y + 1.0, 1)
			shape.shape = box
			shape.position = Vector3(c + 0.5, (y - 1.0) / 2.0, r + 0.5)
			shape.set_meta("cell", Vector2i(c, r))
			body.add_child(shape)
	body.set_meta("terrain", true)
	for t in tops:
		for pair in [[tops[t], 0], [sides[t], 1], [lips[t], 2]]:
			if pair[1] >= 1 and not used.has(t):
				continue
			if pair[1] == 2 and (not mats.has(t) or mats[t][2] == null):
				continue
			var st: SurfaceTool = pair[0]
			st.generate_tangents()
			var mi := MeshInstance3D.new()
			mi.mesh = st.commit()
			mi.material_override = mats[t][pair[1]] if mats.has(t) else null
			if pair[1] == 2:
				mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
			add_child(mi)
	# posts under the pier, down into the water
	for r in rows:
		for c in cols:
			if terrain[r][c] == "planks" and (c + r) % 2 == 0:
				var post := MeshInstance3D.new()
				var bm := BoxMesh.new(); bm.size = Vector3(0.14, top_y(c, r) + 1.0, 0.14)
				post.mesh = bm
				post.position = Vector3(c + 0.15, (top_y(c, r) - 1.0) / 2.0, r + 0.15)
				post.material_override = mats["planks"][1] if mats.has("planks") else null
				add_child(post)

func _build_water() -> void:
	var water := MeshInstance3D.new()
	var pm := PlaneMesh.new()
	pm.size = Vector2(cols + 40, rows + 40) if map.get("sea", true) else Vector2(cols + 2 * apron, rows + 2 * apron)
	pm.subdivide_width = 60
	pm.subdivide_depth = 60
	water.mesh = pm
	water.position = Vector3(cols / 2.0, float(map.get("water_level", 0.15)), rows / 2.0)
	var sm := ShaderMaterial.new()
	sm.shader = load("res://shaders/water.gdshader")
	var noise := FastNoiseLite.new()
	noise.frequency = 0.08
	noise.fractal_octaves = 3
	var nt := NoiseTexture2D.new()
	nt.noise = noise
	nt.seamless = true
	nt.width = 256; nt.height = 256
	sm.set_shader_parameter("ripples", nt)
	water.material_override = sm
	water_mat = sm
	add_child(water)

func _build_props() -> void:
	var wood := StandardMaterial3D.new()
	wood.albedo_texture = _tex("res://map/tex/planks_top.png")
	wood.texture_filter = BaseMaterial3D.TEXTURE_FILTER_NEAREST
	var i := 0
	var ext = null                        # optional: res://map/props3d.gd builds more prop kinds
	if FileAccess.file_exists("res://map/props3d.gd"):
		ext = load("res://map/props3d.gd").new()
	for p in map.get("props", []):
		var c := int(p["col"]); var r := int(p["row"])
		var before := get_child_count()
		var pk := str(p["kind"])
		if BLOCKING.has(pk) or pk in map.get("blocking", []):
			blocked[Vector2i(c, r)] = true
			obstacles[Vector2i(c, r)] = {"r": _prop_radius(pk), "kind": pk}
		_build_prop(p, c, r, i, ext, wood)
		i += 1
		var built := []
		for k in range(before, get_child_count()):
			built.append(get_child(k))
		prop_nodes[Vector2i(c, r)] = prop_nodes.get(Vector2i(c, r), []) + built

func _prop_radius(kind: String) -> float:
	# map.json "radii" (diorama.BLOCKING_RADII, the same numbers the Python gap check uses) wins
	var over: Dictionary = map.get("radii", {})
	return float(over.get(kind, BLOCKING.get(kind, 0.0)))

func _build_prop(p: Dictionary, c: int, r: int, i: int, ext, wood: Material) -> void:
		var base := cell_pos(c, r)
		var kind: String = p["kind"]
		if ext != null and ext.build(self, kind, c, r, base, i):
			return
		if kind in ["bush", "pot", "cave"]:
			_small_prop(kind, base, c, r)
			return
		if BLOCKING.has(kind) or kind in map.get("blocking", []):
			blocked[Vector2i(c, r)] = true            # a tree, house, well... (diorama.BLOCKING_PROPS)
			obstacles[Vector2i(c, r)] = {"r": _prop_radius(kind), "kind": kind}
		if LIGHT_PROPS.has(kind):
			_light_prop(kind, base, i, wood)
			return
		var mi := MeshInstance3D.new()
		match kind:
			"crate":
				var bm := BoxMesh.new(); bm.size = Vector3(0.7, 0.7, 0.7); mi.mesh = bm
				mi.position = base + Vector3(0, 0.35, 0); mi.rotation_degrees.y = 12.0 * (i % 3)
			"barrel":
				var cm := CylinderMesh.new(); cm.top_radius = 0.3; cm.bottom_radius = 0.3; cm.height = 0.8; mi.mesh = cm
				mi.position = base + Vector3(0, 0.4, 0)
			"post":
				var pb := BoxMesh.new(); pb.size = Vector3(0.18, 1.0, 0.18); mi.mesh = pb
				mi.position = base + Vector3(0, 0.5, 0)
			_:                                # unknown kind: a plain box so it still shows
				var ub := BoxMesh.new(); ub.size = Vector3(0.6, 0.6, 0.6); mi.mesh = ub
				mi.position = base + Vector3(0, 0.3, 0)
		mi.material_override = wood
		add_child(mi)

func _small_prop(kind: String, base: Vector3, c: int, r: int) -> void:
	# a bush (cuttable leaves), a clay pot (low things a hero can cut or lift), or a cave mouth
	# in the wall on the cell's north side
	if kind == "cave":
		var hole := _std(Color(0.02, 0.015, 0.03), Color.BLACK, 0.0, 0.0, 1.0)
		var bm := BoxMesh.new(); bm.size = Vector3(0.78, 0.82, 0.06)
		_mesh(bm, hole, base + Vector3(0, 0.41, -0.49))
		var arch := CylinderMesh.new(); arch.top_radius = 0.39; arch.bottom_radius = 0.39; arch.height = 0.06
		arch.radial_segments = 16
		_mesh(arch, hole, base + Vector3(0, 0.82, -0.49), Vector3(PI / 2.0, 0, 0))
		var rock := _std(Color(0.42, 0.4, 0.44), Color.BLACK, 0.0, 0.0, 0.95)
		for x in [-0.46, 0.46]:
			var post := BoxMesh.new(); post.size = Vector3(0.16, 1.0, 0.16)
			_mesh(post, rock, base + Vector3(x, 0.5, -0.44))
		var lintel := BoxMesh.new(); lintel.size = Vector3(1.08, 0.16, 0.18)
		_mesh(lintel, rock, base + Vector3(0, 1.08, -0.44))
		return
	blocked[Vector2i(c, r)] = true
	var k := fposmod(sin(float(c) * 3.7 + float(r) * 9.1) * 437.5, 1.0)
	if kind == "bush":
		var greens := [Color(0.18, 0.42, 0.16), Color(0.24, 0.5, 0.2), Color(0.3, 0.58, 0.22)]
		var offs := [Vector3(0, 0.22, 0), Vector3(0.16, 0.17, 0.1), Vector3(-0.15, 0.16, 0.08), Vector3(0.02, 0.15, -0.16)]
		for j in offs.size():
			var sm := SphereMesh.new()
			sm.radius = 0.24 if j == 0 else 0.2
			sm.height = sm.radius * 1.6
			sm.radial_segments = 8; sm.rings = 4
			_mesh(sm, _std(greens[(j + int(k * 3.0)) % 3], Color.BLACK, 0.0, 0.0, 0.95), base + offs[j])
		var dot := SphereMesh.new(); dot.radius = 0.04; dot.height = 0.08; dot.radial_segments = 6; dot.rings = 3
		for j in 3:                                    # a few berries / flowers
			var a := TAU * (float(j) / 3.0 + k)
			_mesh(dot, _std(Color(0.95, 0.85, 0.5) if j % 2 == 0 else Color(0.9, 0.35, 0.4)),
				base + Vector3(cos(a) * 0.2, 0.32, sin(a) * 0.2))
	else:
		var clay := _std(Color(0.62, 0.36, 0.22), Color.BLACK, 0.0, 0.0, 0.7)
		var body := SphereMesh.new(); body.radius = 0.22; body.height = 0.38; body.radial_segments = 10; body.rings = 5
		_mesh(body, clay, base + Vector3(0, 0.19, 0))
		var neck := CylinderMesh.new(); neck.top_radius = 0.12; neck.bottom_radius = 0.1; neck.height = 0.1
		_mesh(neck, _std(Color(0.5, 0.28, 0.17)), base + Vector3(0, 0.4, 0))

# ------------------------------------------------------------------ light sources
# Torches, braziers, lanterns and crystals light the board: a flickering OmniLight3D
# (light_flicker.gd) + a pixel flame + embers. Spells and effects flash light briefly.

const LIGHT_PROPS := {
	# kind: [light colour, energy, range, flicker, pulse, flame size (0 = none), embers]
	"torch": [Color(1.0, 0.55, 0.24), 2.2, 3.8, 0.18, 0.0, 0.4, 7],
	"brazier": [Color(1.0, 0.5, 0.2), 2.6, 6.0, 0.2, 0.0, 0.72, 14],
	"lantern": [Color(1.0, 0.72, 0.42), 1.5, 3.8, 0.06, 0.0, 0.0, 0],
	"crystal": [Color(0.35, 0.78, 1.0), 1.8, 4.5, 0.0, 0.2, 0.0, 6],
}

func _std(col: Color, emit := Color.BLACK, emit_energy := 0.0, metal := 0.0, rough := 0.9) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.albedo_color = col
	m.metallic = metal
	m.roughness = rough
	if emit_energy > 0.0:
		m.emission_enabled = true
		m.emission = emit
		m.emission_energy_multiplier = emit_energy
	return m

func _mesh(mesh: Mesh, mat: Material, pos: Vector3, rot := Vector3.ZERO) -> MeshInstance3D:
	var mi := MeshInstance3D.new()
	mi.mesh = mesh
	mi.material_override = mat
	mi.position = pos
	mi.rotation = rot
	add_child(mi)
	return mi

func _flame(pos: Vector3, size: float, seed: float) -> void:
	var flame := MeshInstance3D.new()
	var q := QuadMesh.new(); q.size = Vector2(size * 0.75, size)
	flame.mesh = q
	flame.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	var fm := ShaderMaterial.new(); fm.shader = load("res://shaders/flame.gdshader")
	fm.set_shader_parameter("seed", seed)
	fm.set_shader_parameter("grid", Vector2(round(size * 22.0), round(size * 30.0)))
	flame.material_override = fm
	flame.position = pos
	add_child(flame)
	flames.append(fm)

func _fire_light(pos: Vector3, spec: Array, seed: float) -> OmniLight3D:
	var l: OmniLight3D = load("res://light_flicker.gd").new()
	l.light_color = spec[0]
	l.base_energy = spec[1]
	l.omni_range = spec[2]
	l.omni_attenuation = 2.0
	l.flicker = spec[3]
	l.pulse = spec[4]
	l.seed = seed
	l.light_volumetric_fog_energy = 2.5
	l.light_specular = 0.12
	l.position = pos
	add_child(l)
	fire_lights.append(l)
	return l

func _embers(pos: Vector3, amount: int, colour: Color, spread: float, rise: float, one_shot := false) -> CPUParticles3D:
	var p := CPUParticles3D.new()
	p.amount = amount
	p.lifetime = 1.8 if not one_shot else 0.7
	p.lifetime_randomness = 0.5
	p.one_shot = one_shot
	p.explosiveness = 1.0 if one_shot else 0.0
	p.preprocess = 0.0 if one_shot else 2.0
	p.emission_shape = CPUParticles3D.EMISSION_SHAPE_SPHERE
	p.emission_sphere_radius = spread
	p.direction = Vector3.UP
	p.spread = 25.0 if not one_shot else 180.0
	p.initial_velocity_min = rise * 0.5
	p.initial_velocity_max = rise
	p.gravity = Vector3(0.12, 0.15, 0.04) if not one_shot else Vector3(0, -2.5, 0)
	p.scale_amount_min = 0.6
	p.scale_amount_max = 1.0
	var q := QuadMesh.new(); q.size = Vector2(0.036, 0.036)
	var m := ShaderMaterial.new(); m.shader = load("res://shaders/ember.gdshader")
	q.material = m
	p.mesh = q
	var g := Gradient.new()
	g.offsets = PackedFloat32Array([0.0, 0.55, 1.0])
	g.colors = PackedColorArray([colour.lightened(0.25), colour, Color(colour.r * 0.5, colour.g * 0.3, colour.b * 0.3, 0.0)])
	p.color_ramp = g
	p.position = pos
	p.emitting = true
	add_child(p)
	return p

func _light_prop(kind: String, base: Vector3, i: int, wood: Material) -> void:
	var spec: Array = LIGHT_PROPS[kind]
	var iron := _std(Color(0.16, 0.14, 0.15), Color.BLACK, 0.0, 0.6, 0.45)
	var top := base
	match kind:
		"torch":
			var tb := BoxMesh.new(); tb.size = Vector3(0.1, 1.3, 0.1)
			_mesh(tb, wood, base + Vector3(0.38, 0.65, 0.38))
			top = base + Vector3(0.38, 1.42, 0.38)
		"brazier":
			var leg := CylinderMesh.new(); leg.top_radius = 0.07; leg.bottom_radius = 0.17; leg.height = 0.45
			_mesh(leg, iron, base + Vector3(0, 0.225, 0))
			var bowl := CylinderMesh.new(); bowl.top_radius = 0.34; bowl.bottom_radius = 0.18; bowl.height = 0.2
			_mesh(bowl, iron, base + Vector3(0, 0.55, 0))
			var coals := CylinderMesh.new(); coals.top_radius = 0.29; coals.bottom_radius = 0.29; coals.height = 0.02
			_mesh(coals, _std(Color(0.3, 0.08, 0.02), Color(1.0, 0.32, 0.06), 1.1), base + Vector3(0, 0.655, 0))
			top = base + Vector3(0, 0.86, 0)
		"lantern":
			var post := BoxMesh.new(); post.size = Vector3(0.08, 1.25, 0.08)
			_mesh(post, wood, base + Vector3(-0.36, 0.625, 0.36))
			var arm := BoxMesh.new(); arm.size = Vector3(0.3, 0.05, 0.05)
			_mesh(arm, wood, base + Vector3(-0.23, 1.2, 0.36))
			var glass := BoxMesh.new(); glass.size = Vector3(0.15, 0.2, 0.15)
			top = base + Vector3(-0.12, 1.02, 0.36)
			_mesh(glass, _std(Color(1.0, 0.8, 0.5), Color(1.0, 0.66, 0.3), 3.0), top)
			var cap := PrismMesh.new(); cap.size = Vector3(0.2, 0.08, 0.2)
			_mesh(cap, iron, top + Vector3(0, 0.14, 0))
		"crystal":
			var rock := SphereMesh.new(); rock.radius = 0.3; rock.height = 0.3; rock.radial_segments = 7; rock.rings = 3
			_mesh(rock, _std(Color(0.32, 0.3, 0.36)), base + Vector3(0, 0.05, 0))
			var glass := _std(Color(0.25, 0.6, 0.85), Color(0.3, 0.75, 1.0), 1.8, 0.0, 0.15)
			for k in 3:
				var h: float = [0.75, 0.48, 0.4][k]
				var off: Vector3 = [Vector3(0, 0, 0), Vector3(0.17, 0, 0.08), Vector3(-0.14, 0, -0.1)][k]
				var tilt: Vector3 = [Vector3(0, 0, 0.05), Vector3(0.25, 0.4, -0.35), Vector3(-0.3, 1.2, 0.3)][k]
				var up := CylinderMesh.new(); up.top_radius = 0.0; up.bottom_radius = h * 0.2; up.height = h * 0.7
				up.radial_segments = 6; up.rings = 1
				var mi := _mesh(up, glass, base + off + Vector3(0, 0.15 + h * 0.35, 0), tilt)
				mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
			top = base + Vector3(0, 0.7, 0)
	if spec[5] > 0.0:
		_flame(top, spec[5], float(i))
	_fire_light(top + Vector3(0, 0.35 if kind == "brazier" else 0.1, 0), spec, float(i))
	if spec[6] > 0:
		var rise := 0.6 if kind != "crystal" else 0.15
		_embers(top + Vector3(0, 0.1, 0), spec[6], spec[0], 0.08 if kind != "crystal" else 0.35, rise)

func spell_light(pos: Vector3, colour: Color, energy := 4.0, time := 0.6, reach := 4.5, sparks := 14) -> OmniLight3D:
	# a brief light (a spell, a hit): flares up, then fades; sparks fly out
	var l := OmniLight3D.new()
	l.light_color = colour
	l.omni_range = reach
	l.omni_attenuation = 1.4
	l.light_energy = 0.0
	l.light_specular = 0.1
	l.light_cull_mask = 0xFFFFF & ~2                 # not the effect sprites (layer 2)
	l.position = pos
	add_child(l)
	var tw := create_tween()
	tw.tween_property(l, "light_energy", energy, time * 0.15)
	tw.tween_property(l, "light_energy", 0.0, time * 0.85)
	tw.tween_callback(l.queue_free)
	if sparks > 0:
		var p := _embers(pos, sparks, colour, 0.12, 2.2, true)
		get_tree().create_timer(1.5).timeout.connect(p.queue_free)
	return l

# ------------------------------------------------------------------ effects
# res://effects/<name>/<name>.png (+ .json, _glow.png, .effect.json) from hd2d.add_effect. Placed
# ones (map.json "effects": [{name, col, row}]) loop with their own light; casters play one on hit.

func _load_effect(name: String) -> Variant:
	if effects.has(name):
		return effects[name]
	var base := "res://effects/%s/%s" % [name, name]
	if not FileAccess.file_exists(base + ".effect.json"):
		return null
	var info = JSON.parse_string(FileAccess.get_file_as_string(base + ".effect.json"))
	var tex := _tex(base + ".png")
	var size := Vector2(tex.get_width(), tex.get_height())
	var rects := []
	for f in info["frames"]:
		rects.append({"rect": Rect2(f[0] / size.x, f[1] / size.y, f[2] / size.x, f[3] / size.y), "ms": float(f[4])})
	var e := {"tex": tex, "gtex": _tex(base + "_glow.png"), "frames": rects, "w": float(info["frame_w"]),
		"h": float(info["frame_h"]), "ppu": float(info.get("ppu", 42.0)), "loop": bool(info.get("loop", false)),
		"light": _c(info.get("light"), Color(1.0, 0.8, 0.5)), "ground": bool(info.get("ground", false)),
		"tags": info.get("tags", {})}
	effects[name] = e
	return e

func _build_weather() -> void:
	# weather assets over the board (weather_hd2d.py writes res://weather/ when asked)
	if not FileAccess.file_exists("res://weather/board_weather.gd"):
		return
	weather = load("res://weather/board_weather.gd").new()
	add_child(weather)
	weather.setup(self)

func _build_effects() -> void:
	if DirAccess.dir_exists_absolute("res://effects"):
		for name in DirAccess.get_directories_at("res://effects"):
			_load_effect(name)
	for p in map.get("effects", []):
		var c := int(p["col"]); var r := int(p["row"])
		play_effect(str(p["name"]), cell_pos(c, r), true)

func play_effect(name: String, pos: Vector3, loop := false, energy := 1.6, tag := "", scale := 1.0, flip := false) -> Dictionary:
	# tag: play only that animation of the sheet (e.g. a fireball's "fly" loop, then "impact")
	var e = _load_effect(name)
	if e == null:
		return {}
	var frames: Array = e["frames"]
	if tag != "" and e["tags"].has(tag):
		var span: Array = e["tags"][tag]
		frames = frames.slice(int(span[0]), int(span[1]) + 1)
	var mi := MeshInstance3D.new()
	var q := QuadMesh.new()
	q.size = Vector2(e["w"] / e["ppu"], e["h"] / e["ppu"]) * scale
	if e["ground"]:
		q.orientation = PlaneMesh.FACE_Y
		q.size = Vector2(q.size.x, q.size.y * 1.6)     # drawn foreshortened: flat again
	else:
		q.center_offset = Vector3(0, q.size.y / 2.0, 0)
	mi.mesh = q
	mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	var mat := ShaderMaterial.new()
	mat.shader = load("res://shaders/sprite.gdshader")
	mat.set_shader_parameter("sheet", e["tex"])
	if e["gtex"] != null:
		mat.set_shader_parameter("glow", e["gtex"])
		mat.set_shader_parameter("has_glow", true)
	mat.set_shader_parameter("glow_energy", 1.3 if loop else 2.2)
	mat.set_shader_parameter("face_camera_fully", mode != "real")
	mat.set_shader_parameter("sprite_ppu", e["ppu"] / maxf(0.0001, scale))
	mat.set_shader_parameter("flat_on_ground", e["ground"])
	mat.set_shader_parameter("flip_x", flip)
	_sprite_lighting(mat)
	mi.material_override = mat
	mi.position = pos + Vector3(0, 0.02, 0)
	mi.layers = 2                                     # effects: lit by the scene, not by spell lights
	add_child(mi)
	var l := OmniLight3D.new()
	l.light_color = e["light"]
	l.omni_range = 4.0
	l.omni_attenuation = 1.4
	l.light_energy = 0.0
	l.light_volumetric_fog_energy = 2.0
	l.light_specular = 0.0
	l.position = pos + Vector3(0, 0.5 if e["ground"] else max(0.4, q.size.y * 0.5), 0)
	l.light_cull_mask = 0xFFFFF & ~2                 # lights the board, not the effects (no self-lit white-out)
	add_child(l)
	var fx := {"node": mi, "mat": mat, "e": e, "t": 0.0, "loop": loop, "light": l, "energy": energy, "frames": frames}
	live_effects.append(fx)
	return fx

func stop_effect(fx: Dictionary) -> void:
	if fx.is_empty() or not live_effects.has(fx):
		return
	fx["node"].queue_free()
	fx["light"].queue_free()
	live_effects.erase(fx)

func _animate_effects(delta: float) -> void:
	for fx in live_effects.duplicate():
		var e: Dictionary = fx["e"]
		var frames: Array = fx.get("frames", e["frames"])
		fx["t"] += delta
		var total := 0.0
		for f in frames: total += f["ms"]
		var ms: float = fx["t"] * 1000.0
		if not fx["loop"] and ms >= total:
			fx["node"].queue_free()
			fx["light"].queue_free()
			live_effects.erase(fx)
			continue
		var k := fmod(ms, total)
		var pick = frames[0]
		for f in frames:
			if k < f["ms"]:
				pick = f
				break
			k -= f["ms"]
		var r: Rect2 = pick["rect"]
		fx["mat"].set_shader_parameter("frame", Vector4(r.position.x, r.position.y, r.size.x, r.size.y))
		var env_k := 0.8 + 0.2 * sin(fx["t"] * 5.0) if fx["loop"] else sin(PI * clamp(ms / total, 0.0, 1.0))
		fx["light"].light_energy = fx["energy"] * env_k

# ------------------------------------------------------------------ units

func _load_sheet(name: String) -> Dictionary:
	if sheets.has(name):
		return sheets[name]
	var base := "res://units/%s/%s" % [name, name]
	var meta = JSON.parse_string(FileAccess.get_file_as_string(base + ".json"))
	var info = JSON.parse_string(FileAccess.get_file_as_string(base + ".unit.json"))
	var tex := _tex(base + ".png")
	var size := Vector2(meta["meta"]["size"]["w"], meta["meta"]["size"]["h"])
	var frames := {}
	for f in meta["frames"]:
		var key: String = f["filename"].split(" ")[0]
		if not frames.has(key):
			frames[key] = []
		var fr = f["frame"]
		frames[key].append({"rect": Rect2(fr["x"] / size.x, fr["y"] / size.y, fr["w"] / size.x, fr["h"] / size.y),
			"ms": float(f.get("duration", 120))})
	var s := {"tex": tex, "ntex": _tex(base + "_n.png") if FileAccess.file_exists(base + "_n.png") else null,
		"gtex": _tex(base + "_glow.png"),
		"frames": frames, "w": float(info["frame_w"]), "h": float(info["frame_h"]),
		"foot": float(info["foot"]), "ppu": float(info["ppu"]), "anims": info["anims"],
		"stride": info.get("stride", {}), "body": float(info.get("body_px", info["frame_h"] * 0.7)) / float(info["ppu"]),
		"weak": info.get("weak", []), "elevation": float(info.get("elevation", 0.0))}
	if s["elevation"] > 1.0 and sheet_elev <= 1.0:
		sheet_elev = s["elevation"]
	sheets[name] = s
	return s

func _build_units() -> void:
	var shader := load("res://shaders/sprite.gdshader")
	for u in map.get("units", []):
		var s := _load_sheet(u["name"])
		var node := Node3D.new()
		var mesh := MeshInstance3D.new()
		var q := QuadMesh.new()
		q.size = Vector2(s["w"] / s["ppu"], s["h"] / s["ppu"])
		# the quad's bottom edge sits at the sprite's feet
		q.center_offset = Vector3(0, q.size.y / 2.0 - (s["h"] - s["foot"]) / s["ppu"], 0)
		mesh.mesh = q
		var mat := ShaderMaterial.new()
		mat.shader = shader
		mat.set_shader_parameter("sheet", s["tex"])
		if s["ntex"] != null:
			mat.set_shader_parameter("normals", s["ntex"])
			mat.set_shader_parameter("has_normals", true)
		if s["gtex"] != null:
			mat.set_shader_parameter("glow", s["gtex"])
			mat.set_shader_parameter("has_glow", true)
		mat.set_shader_parameter("face_camera_fully", mode != "real")
		mat.set_shader_parameter("sprite_ppu", s["ppu"])
		mat.set_shader_parameter("tint", Color(1, 1, 1) if int(u["team"]) == 0 else Color(1.0, 0.82, 0.82))
		mesh.material_override = mat
		mesh.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		node.add_child(mesh)
		# its shadow comes from a twin drawn only into shadow maps (pushed back along the light's
		# rays: the sprite's silhouette lands on the terrain, but the sprite doesn't shade itself)
		var twin := MeshInstance3D.new()
		twin.mesh = q
		twin.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_SHADOWS_ONLY
		var tmat: ShaderMaterial = mat.duplicate()
		tmat.set_shader_parameter("shadow_caster", true)
		twin.material_override = tmat
		node.add_child(twin)
		# a soft contact shadow on the ground, like HD-2D
		var blob := MeshInstance3D.new()
		var bq := QuadMesh.new(); bq.size = Vector2(0.85, 0.55); bq.orientation = PlaneMesh.FACE_Y
		blob.mesh = bq
		var bm := ShaderMaterial.new()
		bm.shader = load("res://shaders/blob.gdshader")
		blob.material_override = bm
		blob.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		blob.position.y = 0.025
		node.add_child(blob)
		blobs.append(bm)
		add_child(node)
		var c := int(u["col"]); var r := int(u["row"])
		node.position = cell_pos(c, r)
		var hp := int(u.get("hp", 28))
		# the HP bar is drawn in 2D over the view (a 3D bar got blurred by the depth of field into
		# a soft blob above each unit - the "cyan blobs" over the blue team: blue frame + green fill)
		var bar := TextureRect.new()
		bar.texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
		bar.mouse_filter = Control.MOUSE_FILTER_IGNORE
		bar.scale = Vector2.ONE * (3.0 if mode == "real" else 1.0)
		bar.set_meta("lift", q.size.y + q.center_offset.y - q.size.y / 2.0 + 0.12)
		_bars().add_child(bar)
		var unit := {"node": node, "mesh": mesh, "mat": mat, "shadow_mat": tmat, "sheet": s, "col": c, "row": r,
			"team": int(u["team"]), "yaw": _dir_yaw(u.get("facing", "S")), "anim": "idle", "t": randf() * 2.0,
			"name": u["name"], "id": _unit_id(str(u.get("id", u["name"]))), "hp": hp, "max_hp": hp, "atk": int(u.get("attack", 10)),
			"def": int(u.get("defence", 5)), "move": int(u.get("move", MOVE)), "jump": int(u.get("jump", JUMP)),
			"reach": int(u.get("reach", 1)), "done": false, "bar": bar, "hold": false,
			"tint": Color(1, 1, 1) if int(u["team"]) == 0 else Color(1.0, 0.82, 0.82),
			"phase": randf(), "rate": 0.0, "speed": 0.0, "vel": Vector3.ZERO, "vy": 0.0, "mover": null,
			"settle": 0.0, "settle_to": 0.0, "loco": "walk", "frozen": 0.0, "offset": Vector3.ZERO}
		unit["yaw_target"] = unit["yaw"]
		mat.set_shader_parameter("depth_bias", float(conf.get("depth_bias", 0.35)))
		units.append(unit)
		_update_bar(unit)
		blocked[Vector2i(c, r)] = true

func _bars() -> CanvasLayer:
	if bars_layer == null:
		bars_layer = CanvasLayer.new()
		bars_layer.layer = 2
		add_child(bars_layer)
	return bars_layer

func _unit_id(want: String) -> String:
	# a unique id per unit: its name, the 2nd of the same name "wolf#2", ...
	var n := 1
	var id := want
	while units.any(func(x): return x["id"] == id):
		n += 1
		id = "%s#%d" % [want, n]
	return id

func _dir_yaw(name: String) -> float:
	# map directions: N = toward row 0 (-z), E = +x, S = +z
	var i := DIRS.find(name)
	return deg_to_rad(i * 22.5) if i >= 0 else PI / 2.0

func _dir_cont(world_yaw: float, e: float) -> float:
	# the world direction, seen from this camera, as a continuous index into the 16 sprite
	# directions. The sheets were drawn at `e`; the ground is seen at `view_pitch`: the drawing
	# whose ON-SCREEN direction matches (tan k = tan a * sin(view) / sin(drawn)), so a steeper
	# camera than the drawings doesn't make walkers drift off their facing.
	var d := Vector3(cos(world_yaw), 0, sin(world_yaw))
	var right := cam.global_transform.basis.x
	var fwd := -cam.global_transform.basis.z
	fwd.y = 0
	fwd = fwd.normalized()
	var a := atan2(-d.dot(fwd), d.dot(right))      # screen angle, y down (toward camera = S)
	if e > 1.0 and absf(view_pitch - e) > 2.0:
		a = atan2(sin(a) * sin(deg_to_rad(view_pitch)) / sin(deg_to_rad(e)), cos(a))
	return fposmod(a, TAU) / (TAU / 16.0)

var _dir_tables := {}

func _eight_table(e: float) -> Array:
	# the 8 world facings of an 8-way (or 4-way) game -> 8 DIFFERENT drawings, each the nearest
	# free one to its on-screen angle (most certain first). Plain rounding showed some drawings
	# twice and never used others once the camera is steeper than the drawings.
	var b := cam.global_transform.basis.x
	var key := "%.1f|%.1f|%.2f|%.2f" % [e, view_pitch, b.x, b.z]
	if _dir_tables.has(key):
		return _dir_tables[key]
	var t := []
	for i in 8:
		t.append(_dir_cont(i * PI / 4.0, e))
	var order := range(8)
	order.sort_custom(func(x, y): return absf(t[x] - round(t[x])) < absf(t[y] - round(t[y])))
	var taken := {}
	var out := []
	out.resize(8)
	for i in order:
		var best := -1
		var best_d := 99.0
		for j in 16:
			if taken.has(j):
				continue
			var dd := absf(fposmod(t[i] - j + 8.0, 16.0) - 8.0)
			if dd < best_d:
				best_d = dd
				best = j
		taken[best] = true
		out[i] = best
	_dir_tables[key] = out
	return out

func _sprite_dir(world_yaw: float, elev := -1.0) -> String:
	var e := elev if elev > 1.0 else sheet_elev      # this sheet's drawn angle (else the board's)
	var w8 := world_yaw / (PI / 4.0)
	if absf(w8 - round(w8)) < 0.02:                  # one of the 8 game facings: no duplicates
		return DIRS[_eight_table(e)[posmod(int(round(w8)), 8)]]
	return DIRS[int(round(_dir_cont(world_yaw, e))) % 16]

const LOCO := ["walk", "run", "sneak"]

func face_to(u: Dictionary, yaw: float, snap := false) -> void:
	# turn a unit toward a world yaw: through the facings in between (conf.turn_rate), or at once
	u["yaw_target"] = yaw
	if snap:
		u["yaw"] = yaw

func strides(u: Dictionary) -> Dictionary:
	return u["sheet"].get("stride", {})

func has_clip(u: Dictionary, clip: String) -> bool:
	return clip in u["sheet"]["anims"]

func measured(u: Dictionary) -> bool:
	for v in strides(u).values():
		if v is Dictionary and str(v.get("source", "")) in ["rig", "frames"]:
			return true
	return false

func top_speed(u: Dictionary) -> float:
	# the fastest this unit moves with its feet planted (its run, else walk, at the top cadence);
	# a sheet with no measured stride isn't held back (its feet can't be matched anyway)
	if not measured(u):
		return 99.0
	return M.max_speed(strides(u), has_clip(u, "run"), u["sheet"]["body"])

func clip_length(u: Dictionary, anim: String) -> float:
	var s: Dictionary = u["sheet"]
	var key := "%s_%s" % [anim, _sprite_dir(u["yaw"])]
	if not s["frames"].has(key):
		for k in s["frames"]:
			if str(k).begins_with(anim + "_"):
				key = k
				break
	if not s["frames"].has(key):
		return 0.0
	var total := 0.0
	for f in s["frames"][key]: total += f["ms"]
	return total / 1000.0

func set_locomotion(u: Dictionary, speed: float) -> void:
	# a moving unit: walk or run by speed, played at speed / stride cycles per second
	u["speed"] = speed
	if speed < 0.02:
		return
	var lc := M.locomotion(speed, strides(u), has_clip(u, "run"), u["sheet"]["body"], str(u["loco"]))
	var clip: String = lc["clip"] if has_clip(u, lc["clip"]) else "walk"
	u["loco"] = clip
	u["rate"] = lc["rate"]
	if u["anim"] != clip:
		var was_loco: bool = u["anim"] in LOCO
		var keep: float = u["phase"]                 # walk <-> run keeps the step's phase
		_play(u, clip)
		if was_loco:
			u["phase"] = keep
	u["settle"] = 0.0

func _animate_units(delta: float) -> void:
	var turn_rate := float(conf.get("turn_rate", 900.0))
	for u in units:
		var s: Dictionary = u["sheet"]
		var d := delta
		if freeze > 0.0 or u.get("frozen", 0.0) > 0.0:
			d = 0.0                                    # hit-stop: the frame holds
			u["frozen"] = maxf(0.0, u.get("frozen", 0.0) - delta)
		u["t"] += d * float(u.get("anim_speed", 1.0))
		if u.has("yaw_target"):
			u["yaw"] = M.turn(u["yaw"], u["yaw_target"], turn_rate, d)
		var anim: String = u["anim"] if u["anim"] in s["anims"] else "idle"
		var settling: bool = u["settle"] > 0.0 and anim == "idle"
		if settling:
			anim = u["loco"] if u["loco"] in s["anims"] else anim
		var key := _frame_key(s, anim, u["yaw"], u.has("yaw_target") and absf(angle_difference(u["yaw"], u["yaw_target"])) > 0.05)
		if key == "":
			continue
		var fr: Array = s["frames"][key]
		var pick = fr[0]
		if anim in LOCO:
			# stride-matched: the cycle advances by distance travelled (rate = speed / stride), so
			# the planted foot stays put; a unit walking with no speed set plays the clip's own rate
			var rate: float = u["rate"] if u["rate"] > 0.0 else 1000.0 / maxf(1.0, _total(fr))
			if settling:                                # finish the step: to the next contact pose
				u["settle"] -= d
				var to: float = u["settle_to"]
				var left: float = to - u["phase"]
				u["phase"] = minf(to, u["phase"] + maxf(rate * 0.7, left / maxf(u["settle"], 0.016)) * d)
				if u["phase"] >= to - 0.0001 or u["settle"] <= 0.0:
					u["settle"] = 0.0
					u["phase"] = fposmod(to, 1.0)
					u["t"] = 0.0
			else:
				u["phase"] = fposmod(u["phase"] + rate * d, 1.0)
			pick = fr[M.frame_at(u["phase"], fr.size())]
		else:
			var total := _total(fr)
			var t := fmod(u["t"] * 1000.0, total)
			if u["hold"]:
				t = min(u["t"] * 1000.0, total - 1.0)
			for f in fr:
				if t < f["ms"]:
					pick = f
					break
				t -= f["ms"]
		var r: Rect2 = pick["rect"]
		u["mat"].set_shader_parameter("frame", Vector4(r.position.x, r.position.y, r.size.x, r.size.y))
		u["shadow_mat"].set_shader_parameter("frame", Vector4(r.position.x, r.position.y, r.size.x, r.size.y))
		u["mat"].set_shader_parameter("flash", 0.18 * (0.5 + 0.5 * sin(Time.get_ticks_msec() / 150.0)) if u == selected else u.get("hit", 0.0))
		u["mat"].set_shader_parameter("tint", u["tint"] * (0.5 if u["done"] else 1.0))

func _frame_key(s: Dictionary, anim: String, yaw: float, turning := false) -> String:
	# the sheet row for this facing; a sheet with fewer directions (8, 4) uses its nearest one.
	# While turning, a direction the quality check left weak (unit.json "weak") is passed over
	# for its neighbour: the in-between facings flash by, so they must not be the broken ones.
	var name := _sprite_dir(yaw, float(s.get("elevation", -1.0)))
	var key := "%s_%s" % [anim, name]
	if turning and name in s.get("weak", []):
		var j := DIRS.find(name)
		for step in [1, -1]:
			var k3 := "%s_%s" % [anim, DIRS[(j + step + 16) % 16]]
			if s["frames"].has(k3) and not (DIRS[(j + step + 16) % 16] in s.get("weak", [])):
				return k3
	if s["frames"].has(key):
		return key
	var i := DIRS.find(name)
	for step in [1, -1, 2, -2, 3, -3, 4, -4]:
		var k2 := "%s_%s" % [anim, DIRS[(i + step + 16) % 16]]
		if s["frames"].has(k2):
			return k2
	return ""

func _total(fr: Array) -> float:
	var total := 0.0
	for f in fr: total += f["ms"]
	return total

# ------------------------------------------------------------------ camera

func _build_camera() -> void:
	pivot = Node3D.new()
	pivot.position = Vector3(cols / 2.0, 1.5, rows / 2.0)
	for u in units:                      # start on the first unit of the team that plays first
		if u["team"] == active_team:
			pivot.position = (u["node"].position + Vector3(cols / 2.0, 1.5, rows / 2.0)) / 2.0
			break
	add_child(pivot)
	cam = Camera3D.new()
	pivot.add_child(cam)
	cam.top_level = true                          # placed in world space (the board may be stretched)
	if mode == "fake":
		cam.projection = Camera3D.PROJECTION_ORTHOGONAL
		cam.size = 11.0
		cam_dist = 40.0
	elif mode == "pixel":
		cam.projection = Camera3D.PROJECTION_ORTHOGONAL
		cam.size = get_viewport().get_visible_rect().size.y / tile_px   # one tile = tile_px pixels
		cam_dist = 40.0
	else:
		cam.fov = 22.0                                # telephoto: the diorama look
		var attr := CameraAttributesPractical.new()
		attr.dof_blur_far_enabled = true
		attr.dof_blur_near_enabled = true
		attr.dof_blur_amount = 0.12
		cam.attributes = attr
	_place_camera()

func default_pitch() -> float:
	# the camera looks at the sprites from the angle they were drawn at (unit.json elevation);
	# old sheets without it: the look's own default
	if sheet_elev > 1.0:
		return clampf(sheet_elev, 20.0, 70.0)
	return 38.0 if mode == "real" else (45.0 if mode == "pixel" else 30.0)

func _place_camera() -> void:
	var pitch := deg_to_rad(cam_pitch if cam_pitch > 0.0 else default_pitch())
	if mode == "pixel":
		cam_yaw = 0.0                             # the oblique stretch is along the board's axes
		cam_yaw_target = 0.0
	else:
		view_pitch = rad_to_deg(pitch)
	var yaw := deg_to_rad(cam_yaw)
	var at := g(pivot.position)
	cam.global_position = at + Vector3(sin(yaw) * cos(pitch), sin(pitch), cos(yaw) * cos(pitch)) * cam_dist
	cam.look_at(at, Vector3.UP)
	if trauma > 0.0:                                  # shake: offsets on the lens, not the rig
		var o := M.shake_offset(trauma, clock, 0.32 if mode == "real" else 0.5)
		cam.h_offset = o.x
		cam.v_offset = o.y
	elif cam.h_offset != 0.0 or cam.v_offset != 0.0:
		cam.h_offset = 0.0
		cam.v_offset = 0.0
	if mode == "real" and cam.attributes:
		# tilt-shift: sharp in a band around the focus (wide enough that the near and mid ground
		# keep their pixel detail), blurred toward the far edge and the very front
		cam.attributes.dof_blur_far_distance = cam_dist + 5.0
		cam.attributes.dof_blur_far_transition = 8.0
		cam.attributes.dof_blur_near_distance = cam_dist - 7.0
		cam.attributes.dof_blur_near_transition = 4.0

func add_trauma(k: float) -> void:
	trauma = clampf(trauma + k, 0.0, 1.0)

func hit_stop(t := -1.0) -> void:
	freeze = maxf(freeze, float(conf.get("hitstop", 0.075)) if t < 0.0 else t)

func follow_point(p) -> void:
	cam_target = p

func _clamp_view(p: Vector3) -> Vector3:
	# the camera looks inside the map (with the apron around it the screen edge is never empty)
	var m := 1.0 if apron > 0 else 0.0
	return Vector3(clampf(p.x, -m, cols + m), p.y, clampf(p.z, -m, rows + m))

func _process(delta: float) -> void:
	clock += delta
	if freeze > 0.0:
		freeze = maxf(0.0, freeze - delta)
	trauma = maxf(0.0, trauma - float(conf.get("shake_decay", 2.6)) * delta)
	_move_units(delta)
	if cam_target != null:
		var sd := M.smooth_damp(pivot.position, _clamp_view(cam_target), cam_vel, cam_smooth, delta)
		pivot.position = sd[0]
		cam_vel = sd[1]
	var pan := Vector3.ZERO
	if Input.is_key_pressed(KEY_W) or Input.is_key_pressed(KEY_UP): pan.z -= 1
	if Input.is_key_pressed(KEY_S) or Input.is_key_pressed(KEY_DOWN): pan.z += 1
	if Input.is_key_pressed(KEY_A) or Input.is_key_pressed(KEY_LEFT): pan.x -= 1
	if Input.is_key_pressed(KEY_D) or Input.is_key_pressed(KEY_RIGHT): pan.x += 1
	if scripted:
		pan = Vector3.ZERO
	if pan != Vector3.ZERO:
		cam_target = null                             # the player pans: stop following
		pivot.position += pan.normalized().rotated(Vector3.UP, deg_to_rad(cam_yaw)) * delta * 8.0
	cam_yaw = lerp(cam_yaw, cam_yaw_target, min(1.0, delta * cam_turn_rate))
	_place_camera()
	if mode == "fake" or mode == "pixel":
		_snap_camera()
	_animate_units(delta)
	_animate_effects(0.0 if freeze > 0.0 else delta)
	_place_overlay()
	if overlay.get_child_count() > 0:
		for o in overlay.get_children():
			o.material_override.albedo_color.a = 0.42 + 0.12 * sin(Time.get_ticks_msec() / 220.0)

func _snap_camera() -> void:
	# move the camera in whole screen pixels so the pixel grid doesn't swim while panning
	var vp := get_viewport().get_visible_rect().size
	var unit := cam.size / vp.y
	var b := cam.global_transform.basis
	var p := cam.global_position
	var x := snappedf(p.dot(b.x), unit)
	var y := snappedf(p.dot(b.y), unit)
	var z := p.dot(b.z)
	cam.global_position = b.x * x + b.y * y + b.z * z

# ------------------------------------------------------------------ play
# select an own unit -> purple move range -> click a cell (or the unit) -> red cells: enemies
# within reach -> click one to attack, Space to wait. Done units grey out; when the whole team
# is done the turn passes. Last team standing wins.

var phase := "select"      # select, move, act, busy, over
var attack_cells := {}     # Vector2i -> target unit
var banner: Label

func _build_cursor() -> void:
	cursor = MeshInstance3D.new()
	var q := QuadMesh.new(); q.size = Vector2(0.96, 0.96); q.orientation = PlaneMesh.FACE_Y
	cursor.mesh = q
	var m := StandardMaterial3D.new()
	m.albedo_color = Color(1.0, 1.0, 1.0, 0.35)
	m.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	m.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	m.emission_enabled = true
	m.emission = Color(0.6, 0.6, 0.7)
	cursor.material_override = m
	cursor.visible = false
	add_child(cursor)

func _hud() -> void:
	var layer := CanvasLayer.new()
	add_child(layer)
	hud = Label.new()
	hud.position = Vector2(16, 12)
	hud.add_theme_font_size_override("font_size", 20 if mode == "real" else 10)
	hud.add_theme_color_override("font_outline_color", Color.BLACK)
	hud.add_theme_constant_override("outline_size", 6)
	layer.add_child(hud)
	banner = Label.new()
	banner.set_anchors_preset(Control.PRESET_FULL_RECT)
	banner.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	banner.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	banner.add_theme_font_size_override("font_size", 96 if mode == "real" else 40)
	banner.add_theme_color_override("font_outline_color", Color.BLACK)
	banner.add_theme_constant_override("outline_size", 14)
	banner.visible = false
	layer.add_child(banner)
	_update_hud()

func _team_name(t: int) -> String:
	return "Blue" if t == 0 else "Red"

func _update_hud(extra: String = "") -> void:
	var sel := ""
	if selected != null:
		sel = "   %s  HP %d/%d  ATK %d  DEF %d  MOV %d" % [selected["name"].capitalize(), selected["hp"],
			selected["max_hp"], selected["atk"], selected["def"], selected["move"]]
	var help: String = {"select": "Click one of your units.", "move": "Click a purple cell to move (or the unit to stay).",
		"act": "Click a red enemy to attack, Space to wait.", "busy": "", "over": "Esc: menu"}[phase]
	hud.text = "%s team's turn%s\n%s %s\nSpace: end turn   Q/E: turn camera   Wheel: zoom   L: light (%s)   B: pixel bands   Esc: menu" % [
		_team_name(active_team), sel, help, extra, light_name]

func _place_overlay() -> void:
	# HP bars and damage numbers follow their 3D points in whole pixels
	if bars_layer != null:
		for b in bars_layer.get_children():
			if b.has_meta("lift"):
				b.visible = false
	for u in units:
		var bar: Control = u["bar"]
		if not is_instance_valid(bar):
			continue
		var p: Vector3 = u["node"].position + Vector3(0, float(bar.get_meta("lift", 1.2)), 0)
		var vis: bool = u.get("bar_on", true) and not cam.is_position_behind(g(p)) and u["node"].visible
		bar.visible = vis and bar.texture != null
		if bar.visible:
			var sp := cam.unproject_position(g(p))
			var w: float = bar.texture.get_width() * bar.scale.x
			var h: float = bar.texture.get_height() * bar.scale.y
			bar.position = (sp - Vector2(w / 2.0, h)).round()
	for f in floaters.duplicate():
		var l: Label = f["node"]
		f["t"] += get_process_delta_time()
		if f["t"] >= f["life"] or not is_instance_valid(l):
			if is_instance_valid(l):
				l.queue_free()
			floaters.erase(f)
			continue
		var k: float = f["t"] / f["life"]
		var p3: Vector3 = f["pos"] + Vector3(0, f["rise"] * (1.0 - pow(1.0 - k, 2.0)), 0)
		if cam.is_position_behind(g(p3)):
			l.visible = false
			continue
		l.visible = true
		l.position = (cam.unproject_position(g(p3)) - l.size / 2.0).round()
		l.modulate.a = 1.0 if k < 0.45 else clampf(1.0 - (k - 0.45) / 0.55, 0.0, 1.0)

func float_text(at: Vector3, text: String, colour := Color(1.0, 0.95, 0.4), life := 0.8, size := -1) -> Label:
	var l := Label.new()
	l.text = text
	l.add_theme_font_size_override("font_size", size if size > 0 else (44 if mode == "real" else 16))
	l.add_theme_color_override("font_color", colour)
	l.add_theme_color_override("font_outline_color", Color(0.05, 0.02, 0.05))
	l.add_theme_constant_override("outline_size", 12 if mode == "real" else 4)
	l.mouse_filter = Control.MOUSE_FILTER_IGNORE
	_bars().add_child(l)
	l.reset_size()
	floaters.append({"node": l, "pos": at, "t": 0.0, "life": life, "rise": 0.9})
	l.position = (cam.unproject_position(g(at)) - l.size / 2.0).round()
	return l

func _update_bar(u: Dictionary) -> void:
	var w := 22
	var img := Image.create(w + 2, 4, false, Image.FORMAT_RGBA8)
	img.fill(Color(0.05, 0.05, 0.08))
	var frame := Color(0.35, 0.55, 1.0) if u["team"] == 0 else Color(1.0, 0.35, 0.3)
	for x in w + 2:
		img.set_pixel(x, 0, frame)
		img.set_pixel(x, 3, frame)
	img.set_pixel(0, 1, frame); img.set_pixel(0, 2, frame)
	img.set_pixel(w + 1, 1, frame); img.set_pixel(w + 1, 2, frame)
	var k := float(u["hp"]) / float(u["max_hp"])
	var fill := Color(0.3, 0.9, 0.35) if k > 0.5 else (Color(0.95, 0.85, 0.25) if k > 0.25 else Color(0.95, 0.3, 0.25))
	for x in int(round(k * w)):
		img.set_pixel(1 + x, 1, fill)
		img.set_pixel(1 + x, 2, fill.darkened(0.25))
	u["bar"].texture = ImageTexture.create_from_image(img)

func _cell_under_mouse(pos: Vector2) -> Variant:
	var from := cam.project_ray_origin(pos)
	var to := from + cam.project_ray_normal(pos) * 200.0
	var hit := get_world_3d().direct_space_state.intersect_ray(PhysicsRayQueryParameters3D.create(from, to))
	if hit.is_empty():
		return null
	var p: Vector3 = to_local(hit["position"] - hit["normal"] * 0.01)
	var c := int(floor(p.x)); var r := int(floor(p.z))
	if c < 0 or r < 0 or c >= cols or r >= rows:
		return null
	return Vector2i(c, r)

func _unit_at(cell: Vector2i) -> Variant:
	for u in units:
		if u["col"] == cell.x and u["row"] == cell.y:
			return u
	return null

func _unhandled_input(event: InputEvent) -> void:
	if scripted:
		return
	if event is InputEventKey and event.pressed and not event.echo:
		match event.keycode:
			KEY_ESCAPE:
				if phase == "act":
					_finish(selected)
				elif selected != null and phase == "move":
					_deselect()
				else:
					get_tree().change_scene_to_file("res://menu.tscn")
			KEY_Q: cam_yaw_target -= 45.0
			KEY_E: cam_yaw_target += 45.0
			KEY_L:
				var names: Array = LIGHT_ORDER.filter(func(n): return lighting.has(n))
				if names.is_empty():
					names = lighting.keys()
				apply_lighting(names[(names.find(light_name) + 1) % names.size()])
			KEY_B:
				bands = 0.0 if bands > 0.5 else 4.0
				apply_lighting(light_name)
			KEY_SPACE, KEY_TAB:
				if phase == "act":
					_finish(selected)
				elif phase in ["select", "move"]:
					_end_turn()
	elif event is InputEventMouseButton and event.pressed:
		if event.button_index == MOUSE_BUTTON_WHEEL_UP:
			_zoom(0.9)
		elif event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			_zoom(1.1)
		elif event.button_index == MOUSE_BUTTON_RIGHT and phase == "act":
			_finish(selected)
		elif event.button_index == MOUSE_BUTTON_LEFT:
			var cell = _cell_under_mouse(event.position)
			if cell != null:
				_click(cell)
	elif event is InputEventMouseMotion:
		var cell = _cell_under_mouse(event.position)
		cursor.visible = cell != null
		if cell != null:
			cursor.position = cell_pos(cell.x, cell.y) + Vector3(0, 0.03, 0)
			if phase == "act" and attack_cells.has(cell):
				var t: Dictionary = attack_cells[cell]
				_update_hud("  Hit %s for %d" % [t["name"].capitalize(), _damage(selected, t)])

func _zoom(k: float) -> void:
	if mode == "fake":
		cam.size = clamp(cam.size * k, 6.0, 30.0)
	else:
		cam_dist = clamp(cam_dist * k, 12.0, 70.0)

func _click(cell: Vector2i) -> void:
	if phase in ["busy", "over"]:
		return
	var u: Variant = _unit_at(cell)
	if phase == "act":
		if attack_cells.has(cell):
			_attack(selected, attack_cells[cell])
		return
	if u != null and u["team"] == active_team and not u["done"]:
		if phase == "move" and u == selected:
			_clear_range()
			_enter_act(u)                          # stay put and act
			return
		selected = u
		phase = "move"
		_show_range()
	elif phase == "move" and reach.has(cell) and cell != Vector2i(selected["col"], selected["row"]):
		_walk(selected, _path_to(cell))
	else:
		_deselect()
	_update_hud()

func _deselect() -> void:
	selected = null
	phase = "select"
	_clear_range()
	_update_hud()

func _overlay_cell(cell: Vector2i, colour: Color, glow: Color) -> void:
	var m := MeshInstance3D.new()
	var q := QuadMesh.new(); q.size = Vector2(0.92, 0.92); q.orientation = PlaneMesh.FACE_Y
	m.mesh = q
	var mat := StandardMaterial3D.new()
	mat.albedo_color = colour
	mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	mat.emission_enabled = true
	mat.emission = glow
	mat.emission_energy_multiplier = 0.3        # tinted, not blown out to white by the bloom
	m.material_override = mat
	m.position = cell_pos(cell.x, cell.y) + Vector3(0, 0.02, 0)
	overlay.add_child(m)

func _show_range() -> void:
	_clear_range()
	var start := Vector2i(selected["col"], selected["row"])
	var move: int = selected["move"]
	var jump: int = selected["jump"]
	reach = {start: [0, start]}
	var frontier := [start]
	while frontier.size() > 0:
		var nxt := []
		for p in frontier:
			for d in [Vector2i(1, 0), Vector2i(-1, 0), Vector2i(0, 1), Vector2i(0, -1)]:
				var n: Vector2i = p + d
				if not walkable(n.x, n.y):
					continue
				var dh: int = int(heights[n.y][n.x]) - int(heights[p.y][p.x])
				if dh > jump or -dh > 2 * jump:
					continue
				var cost: int = reach[p][0] + 1
				if cost <= move and (not reach.has(n) or cost < reach[n][0]):
					reach[n] = [cost, p]
					nxt.append(n)
		frontier = nxt
	for cell in reach:
		_overlay_cell(cell, Color(0.75, 0.25, 1.0, 0.45), Color(0.55, 0.1, 0.9))

func _clear_range() -> void:
	for o in overlay.get_children():
		o.queue_free()
	reach = {}
	attack_cells = {}

func _path_to(cell: Vector2i) -> Array:
	var path := [cell]
	while reach[path[0]][1] != path[0]:
		path.push_front(reach[path[0]][1])
	return path

func _walk(u: Dictionary, path: Array) -> void:
	# along the grid path: eased start and stop, corners rounded, turning through the facings,
	# hops on level changes, walk/run by speed at the stride's rate (pp_motion)
	phase = "busy"
	_clear_range()
	blocked.erase(Vector2i(u["col"], u["row"]))
	var pts := []
	for c in path:
		pts.append(Vector2(c.x + 0.5, c.y + 0.5))
	move_along(u, pts, minf(move_speed, top_speed(u)), func():
		var last: Vector2i = path[path.size() - 1]
		u["col"] = last.x; u["row"] = last.y
		blocked[last] = true
		if scripted:
			action_done.emit()
		else:
			_enter_act(u))

func ground_at(x: float, z: float, radius := 0.0) -> float:
	# the ground under a point: the highest map cell its foot circle touches (water: its level)
	var best := -INF
	var pts := [Vector2(x, z)]
	if radius > 0.0:
		pts += [Vector2(x - radius, z), Vector2(x + radius, z), Vector2(x, z - radius), Vector2(x, z + radius)]
	for q in pts:
		var c := clampi(int(floor(q.x)), 0, cols - 1)
		var r := clampi(int(floor(q.y)), 0, rows - 1)
		if terrain[r][c] == "water":
			continue
		best = maxf(best, top_y(c, r))
	if best == -INF:
		best = float(map.get("water_level", 0.15))
	return best

func _round_path(pts: Array, radius: float) -> Array:
	# corners cut by a quadratic curve `radius` before and after each turn (sampled)
	if pts.size() < 3 or radius <= 0.0:
		return pts
	var out := [pts[0]]
	for i in range(1, pts.size() - 1):
		var a: Vector2 = pts[i - 1]
		var b: Vector2 = pts[i]
		var c: Vector2 = pts[i + 1]
		var din := (b - a).normalized()
		var dout := (c - b).normalized()
		if din.dot(dout) > 0.995:
			out.append(b)
			continue
		var r := minf(radius, minf(a.distance_to(b), b.distance_to(c)) * 0.5)
		var p0 := b - din * r
		var p2 := b + dout * r
		for k in 7:
			var t := float(k) / 6.0
			out.append(p0.lerp(b, t).lerp(b.lerp(p2, t), t))
	out.append(pts[pts.size() - 1])
	return out

func move_along(u: Dictionary, pts: Array, speed: float, done: Callable = Callable(), round_corners := true) -> void:
	# move a unit through ground points (Vector2 x, z) at up to `speed` tiles/s; done() at the end.
	# Its height follows the ground (hops over steps); the facing turns toward the way it goes.
	var p0: Vector3 = u["node"].position
	var line := [Vector2(p0.x, p0.z)]
	for q in pts:
		var v: Vector2 = q if q is Vector2 else Vector2(q.x, q.z)
		if v.distance_to(line[line.size() - 1]) > 0.01:
			line.append(v)
	if round_corners:
		line = _round_path(line, float(conf.get("corner", 0.32)))
	var lens := [0.0]
	for i in range(1, line.size()):
		lens.append(lens[i - 1] + line[i].distance_to(line[i - 1]))
	var total: float = lens[lens.size() - 1]
	if total < 0.01:
		if done.is_valid():
			done.call()
		return
	# level changes along the way (where the feet cross a cell border onto other ground)
	var steps := []
	var n := int(ceil(total / 0.04))
	var prev_y := ground_at(line[0].x, line[0].y)
	for i in range(1, n + 1):
		var s := total * float(i) / float(n)
		var q := _line_at(line, lens, s)
		var y := ground_at(q.x, q.y)
		if absf(y - prev_y) > 0.01:
			steps.append([s, prev_y, y])
		prev_y = y
	u["mover"] = {"line": line, "lens": lens, "total": total, "s": 0.0, "want": speed, "steps": steps,
		"done": done, "y0": ground_at(line[0].x, line[0].y)}
	u["vy"] = 0.0
	if u["speed"] < 0.05:
		u["speed"] = 0.0

func stop_moving(u: Dictionary) -> void:
	u["mover"] = null
	u["speed"] = 0.0
	if u["anim"] in LOCO:
		_play(u, "idle")

func _line_at(line: Array, lens: Array, s: float) -> Vector2:
	for i in range(1, line.size()):
		if s <= lens[i] or i == line.size() - 1:
			var seg: float = lens[i] - lens[i - 1]
			var k := 0.0 if seg < 0.0001 else clampf((s - lens[i - 1]) / seg, 0.0, 1.0)
			return line[i - 1].lerp(line[i], k)
	return line[line.size() - 1]

func _path_y(m: Dictionary, s: float, q: Vector2) -> float:
	var hop := float(conf.get("hop", 0.14))
	var w := 0.32                                     # the hop starts/ends this far from the border
	for st in m["steps"]:
		var d: float = s - float(st[0])
		if absf(d) < w:
			return M.hop_y((d + w) / (2.0 * w), float(st[1]), float(st[2]), hop)
	return ground_at(q.x, q.y)

func _move_units(delta: float) -> void:
	if freeze > 0.0:
		return
	for u in units:
		var m = u["mover"]
		if m == null:
			continue
		var remaining: float = m["total"] - m["s"]
		var sp := M.path_speed(u["speed"], m["want"], remaining, float(conf.get("accel", 14.0)),
			float(conf.get("decel", 18.0)), delta)
		sp = maxf(sp, 0.35 if remaining > 0.02 else 0.0)  # never stalls short of the end
		m["s"] = minf(m["total"], m["s"] + sp * delta)
		var q := _line_at(m["line"], m["lens"], m["s"])
		var ahead := _line_at(m["line"], m["lens"], minf(m["total"], m["s"] + 0.12))
		if ahead.distance_to(q) > 0.01:
			face_to(u, atan2(ahead.y - q.y, ahead.x - q.x))
		u["node"].position = Vector3(q.x, _path_y(m, m["s"], q), q.y) + u.get("offset", Vector3.ZERO)
		set_locomotion(u, sp)
		if m["s"] >= m["total"] - 0.0001:
			u["mover"] = null
			u["speed"] = 0.0
			_play(u, "idle")
			var cb: Callable = m["done"]
			if cb.is_valid():
				cb.call()

func _play(u: Dictionary, anim: String, once := false) -> void:
	if anim == "idle" and u["anim"] in LOCO and u["anim"] in u["sheet"]["anims"] and not once:
		# stopping: the walk finishes its step onto the next contact pose, then idles
		u["settle"] = float(conf.get("settle", 0.16))
		u["settle_to"] = M.settle_target(u["phase"])
		if u["settle_to"] < u["phase"]:
			u["settle_to"] += 1.0
	elif anim != "idle":
		u["settle"] = 0.0
	if anim in LOCO and u["anim"] not in LOCO:
		u["phase"] = 0.0
	u["anim"] = anim
	u["t"] = 0.0
	u["hold"] = once
	u["anim_speed"] = 1.0

func _enter_act(u: Dictionary) -> void:
	attack_cells = {}
	for e in units:
		if e["team"] != u["team"] and e["hp"] > 0:
			var dist: int = abs(e["col"] - u["col"]) + abs(e["row"] - u["row"])
			if dist <= u["reach"]:
				attack_cells[Vector2i(e["col"], e["row"])] = e
	if attack_cells.is_empty():
		_finish(u)
		return
	phase = "act"
	for cell in attack_cells:
		_overlay_cell(cell, Color(1.0, 0.18, 0.15, 0.5), Color(0.9, 0.1, 0.05))
	_update_hud()

func _damage(a: Dictionary, t: Dictionary) -> int:
	# the same rule as diorama.damage(): attack - defence/2, +25% from higher ground, min 1
	var dmg: int = a["atk"] - int(t["def"] / 2)
	if int(heights[a["row"]][a["col"]]) > int(heights[t["row"]][t["col"]]):
		dmg = int(dmg * 5 / 4)
	return max(1, dmg)

func _attack(a: Dictionary, t: Dictionary, dmg := -1, fx := "", fx_colour := Color(0, 0, 0, 0)) -> void:
	# dmg / fx: autoplay overrides (the blow's damage; an effect played on the target as it lands)
	phase = "busy"
	_clear_range()
	if dmg < 0:
		dmg = _damage(a, t)
	face_to(a, atan2(float(t["row"] - a["row"]), float(t["col"] - a["col"])))
	face_to(t, atan2(float(a["row"] - t["row"]), float(a["col"] - t["col"])))
	var tw := create_tween()
	var magic: Color = SPELL.get(a["name"], Color(0, 0, 0, 0))
	var clip := "cast" if magic.a > 0.0 and "cast" in a["sheet"]["anims"] else "attack"
	var home: Vector3 = a["node"].position
	var toward: Vector3 = t["node"].position - home
	toward.y = 0.0
	toward = toward.normalized() if toward.length() > 0.001 else Vector3(0, 0, 1)
	# anticipation: a short wind-up lean away from the target (and the turn finishes)
	tw.tween_property(a["node"], "position", home - toward * 0.07, float(conf.get("anticipation", 0.12))) \
		.set_trans(Tween.TRANS_SINE).set_ease(Tween.EASE_OUT)
	if magic.a > 0.0:                             # casters: a charge light in their hands
		spell_light(home + Vector3(0, 0.9, 0), magic, 2.5, 0.6, 3.0, 0)
	if clip in a["sheet"]["anims"]:
		tw.tween_callback(func(): _play(a, clip, true))
		# the blow lands on the clip's strike frame (the cast pose for spells), not on a timer
		var hit_at := M.hit_time(clip, clip_length(a, clip))
		tw.tween_property(a["node"], "position", home + toward * 0.1, maxf(0.05, hit_at)) \
			.set_trans(Tween.TRANS_QUAD).set_ease(Tween.EASE_IN)
	else:                                         # no attack clip: lunge at the target
		tw.tween_property(a["node"], "position", home + toward * 0.35, 0.12).set_trans(Tween.TRANS_QUAD).set_ease(Tween.EASE_IN)
	tw.tween_callback(func():
		hit_stop()                                 # both freeze a moment: the hit lands
		a["frozen"] = float(conf.get("hitstop", 0.075))
		add_trauma(float(conf.get("shake", 0.06)) * (2.2 if magic.a > 0.0 else 3.0))
		_knockback(t, toward)
		t["hp"] = max(0, t["hp"] - dmg)
		var at: Vector3 = t["node"].position + Vector3(0, 0.7, 0)
		if fx != "":
			play_effect(fx, t["node"].position + Vector3(0, 0.1, 0), false, 2.5)
			spell_light(at, fx_colour if fx_colour.a > 0.0 else Color(1.0, 0.85, 0.6), 2.4, 0.35, 3.0, 10)
		elif magic.a > 0.0:
			spell_light(at, magic, 5.0, 0.8, 5.0, 18)
			var cfx := _effect_for(a)
			if cfx != "":
				play_effect(cfx, t["node"].position, false, 3.0)
		else:
			spell_light(at, Color(1.0, 0.78, 0.45), 1.6, 0.25, 2.5, 8)
		_update_bar(t)
		_number(t, dmg)
		t["hit"] = 0.6
		if "hurt" in t["sheet"]["anims"] and t["hp"] > 0:
			_play(t, "hurt", true))
	tw.tween_interval(float(conf.get("hitstop", 0.075)))
	tw.tween_method(func(v): t["hit"] = v, 0.6, 0.0, 0.3)
	# follow-through: the attacker eases back home while its clip plays out
	tw.parallel().tween_property(a["node"], "position", home, 0.3).set_trans(Tween.TRANS_SINE).set_ease(Tween.EASE_IN_OUT)
	tw.tween_interval(float(conf.get("follow_through", 0.18)))
	tw.tween_callback(func():
		if t["hp"] > 0:
			_play(t, "idle")
			_play(a, "idle")
			_finish(a)
		else:
			_kill(t, a))

func _knockback(t: Dictionary, dir: Vector3) -> void:
	# pushed back a little by the blow, then it steps back onto its cell
	if not is_instance_valid(t["node"]):
		return
	var k := float(conf.get("knockback", 0.22))
	var tw := create_tween()
	tw.tween_method(func(v):
		if is_instance_valid(t["node"]):
			t["offset"] = dir * v
			var base := cell_pos(int(t["col"]), int(t["row"]))
			t["node"].position = base + t["offset"], 0.0, k, 0.1).set_trans(Tween.TRANS_QUAD).set_ease(Tween.EASE_OUT).set_delay(float(conf.get("hitstop", 0.075)))
	tw.tween_method(func(v):
		if is_instance_valid(t["node"]):
			t["offset"] = dir * v
			var base := cell_pos(int(t["col"]), int(t["row"]))
			t["node"].position = base + t["offset"], k, 0.0, 0.28).set_trans(Tween.TRANS_SINE).set_ease(Tween.EASE_IN_OUT)

func _effect_for(a: Dictionary) -> String:
	if effects.has(a["name"]):
		return a["name"]
	for n in effects:
		if not effects[n]["loop"]:
			return n
	return ""

func _kill(t: Dictionary, a: Dictionary) -> void:
	var anims: Array = t["sheet"]["anims"]
	var tw := create_tween()
	if "death" in anims or "lie_down" in anims:
		_play(t, "death" if "death" in anims else "lie_down", true)
		tw.tween_interval(1.3)
	tw.tween_method(func(v): t["mat"].set_shader_parameter("tint", t["tint"] * v), 1.0, 0.0, 0.4)
	tw.tween_callback(func():
		blocked.erase(Vector2i(t["col"], t["row"]))
		units.erase(t)
		if is_instance_valid(t["bar"]):
			t["bar"].queue_free()
		t["node"].queue_free()
		_play(a, "idle")
		_finish(a))

func _number(t: Dictionary, dmg: int, text := "", colour := Color(1.0, 0.95, 0.4)) -> void:
	# a damage number in 2D over the view (crisp: no depth-of-field blur), rising from the unit
	float_text(t["node"].position + Vector3(0, number_lift, 0), text if text != "" else str(dmg), colour)

func _finish(u: Dictionary) -> void:
	_clear_range()
	if u != null:
		u["done"] = true
	selected = null
	phase = "select"
	if scripted:                       # the script passes turns and calls the winner
		action_done.emit()
		return
	for team in [0, 1]:
		var alive := units.filter(func(x): return x["team"] == team)
		if alive.is_empty():
			phase = "over"
			banner.text = "%s wins!" % _team_name(1 - team)
			banner.visible = true
			_update_hud()
			return
	var waiting := units.filter(func(x): return x["team"] == active_team and not x["done"])
	if waiting.is_empty():
		_end_turn()
	else:
		_update_hud()

func _end_turn() -> void:
	if phase == "over":
		return
	selected = null
	_clear_range()
	active_team = 1 - active_team
	for u in units:
		u["done"] = false
	phase = "select"
	_update_hud()
'''


# Time-of-day presets (res://lighting.json; L cycles them in the game). Angles: [elevation
# (negative = down), azimuth] in degrees. torches = how bright fires burn (night 1.5, noon 0.2);
# rim = sprite rim / backlight; blob = contact shadow; volumetric = fog density for light shafts
# and halos around torches (real 3D only).
LIGHTING = {
    "dawn": {
        "sun": [1.0, 0.72, 0.6], "sun_energy": 1.05, "sun_angle": [-14, 75],
        "fill": [0.5, 0.58, 1.0], "fill_energy": 0.3, "fill_angle": [-50, -110],
        "ambient": [0.58, 0.54, 0.78], "ambient_energy": 0.5, "background": [0.2, 0.17, 0.28],
        "fog": [0.62, 0.5, 0.62], "fog_density": 0.002, "volumetric": 0.0,
        "torches": 0.6, "rim": 0.7, "blob": 0.35, "exposure": 1.05, "glow": 0.75,
        "saturation": 1.1, "contrast": 1.04,
        "water_deep": [0.08, 0.12, 0.3], "water_shallow": [0.32, 0.36, 0.62], "sparkle": 2.2},
    "noon": {
        "sun": [1.0, 0.96, 0.88], "sun_energy": 1.6, "sun_angle": [-62, -35],
        "fill": [0.55, 0.68, 1.0], "fill_energy": 0.3, "fill_angle": [-60, 145],
        "ambient": [0.62, 0.7, 0.85], "ambient_energy": 0.6, "background": [0.45, 0.62, 0.85],
        "fog": [0.6, 0.72, 0.9], "fog_density": 0.0015, "volumetric": 0.0,
        "torches": 0.2, "rim": 0.25, "blob": 0.5, "exposure": 0.93, "white": 1.6, "glow": 0.55,
        "saturation": 1.12, "contrast": 1.05,
        "water_deep": [0.03, 0.16, 0.36], "water_shallow": [0.1, 0.4, 0.7], "sparkle": 3.0,
        "water_spec": 0.35, "glare": 0.35},
    "sunset": {
        "sun": [1.0, 0.6, 0.32], "sun_energy": 1.3, "sun_angle": [-20, -60],
        "fill": [0.42, 0.5, 1.0], "fill_energy": 0.35, "fill_angle": [-55, 120],
        "ambient": [0.5, 0.38, 0.62], "ambient_energy": 0.5, "background": [0.1, 0.05, 0.12],
        "fog": [0.5, 0.26, 0.3], "fog_density": 0.002, "volumetric": 0.0,
        "torches": 1.0, "rim": 0.8, "blob": 0.4, "exposure": 1.05, "glow": 0.85,
        "saturation": 1.15, "contrast": 1.06,
        "water_deep": [0.06, 0.07, 0.24], "water_shallow": [0.36, 0.26, 0.5], "sparkle": 2.6},
    "night": {
        "sun": [0.55, 0.68, 1.0], "sun_energy": 0.42, "sun_angle": [-48, 130],
        "fill": [0.3, 0.32, 0.7], "fill_energy": 0.12, "fill_angle": [-60, -50],
        "ambient": [0.24, 0.28, 0.52], "ambient_energy": 0.55, "background": [0.01, 0.015, 0.04],
        "fog": [0.12, 0.15, 0.3], "fog_density": 0.004, "volumetric": 0.012,
        "torches": 1.25, "rim": 0.9, "lift": 0.45, "blob": 0.45, "exposure": 1.1, "glow": 1.0,
        "saturation": 1.08, "contrast": 1.08,
        "water_deep": [0.01, 0.03, 0.1], "water_shallow": [0.05, 0.1, 0.26], "sparkle": 1.8},
}
LIGHTING_NAMES = tuple(LIGHTING)
TERRAIN_BANDS = 5.0      # the ground's diffuse steps per light (terrain.gdshader)


def write_lighting(project, default: str = "sunset", bands: float = 4.0) -> Path:
    """res://lighting.json: every preset, the one the game starts with, and the toon bands
    (0 = smooth sprite lighting)."""
    if default not in LIGHTING:
        raise ValueError(f"Unknown lighting '{default}' (use {', '.join(LIGHTING)})")
    path = Path(project) / "lighting.json"
    path.write_text(json.dumps({"default": default, "bands": bands, "terrain_bands": TERRAIN_BANDS,
                                "presets": LIGHTING}, indent=1),
                    encoding="utf-8")
    return path


def _foot_row(sheet: np.ndarray, frames: list) -> int:
    """Lowest opaque row (within a frame) of the standing pose: where the feet are."""
    lows = []
    for f in frames:
        r = f["frame"]
        a = sheet[r["y"]:r["y"] + r["h"], r["x"]:r["x"] + r["w"], 3]
        ys = np.nonzero(a.max(1) >= 128)[0]
        if len(ys):
            lows.append(ys.max())
    return int(np.median(lows)) + 1 if lows else frames[0]["frame"]["h"]


def add_unit(project: Path, name: str, sheet_png, ppu: float = 42.0, glow: bool = True) -> Path:
    """Copy a unit's sheet (+ .json, + _n.png and _glow.png if there) into res://units/<name>/
    and write <name>.unit.json (frame size, feet row, pixels per tile, animations). No glow map
    next to the sheet: glow.auto_glow makes one in the unit folder when anything glows (fire,
    gems, eyes); glow=False skips that."""
    sheet_png = Path(sheet_png)
    folder = project / "units" / name
    folder.mkdir(parents=True, exist_ok=True)
    meta = json.loads(sheet_png.with_suffix(".json").read_text(encoding="utf-8"))
    shutil.copyfile(sheet_png, folder / f"{name}.png")
    (folder / f"{name}.json").write_text(json.dumps(meta), encoding="utf-8")
    normals = sheet_png.with_name(sheet_png.stem + "_n.png")
    if normals.exists():
        shutil.copyfile(normals, folder / f"{name}_n.png")
        if not (meta.get("meta") or {}).get("normals_smooth"):   # older sheets: calm the light
            from .normals import smooth_sheet
            smooth_sheet(folder / f"{name}.png", folder / f"{name}_n.png")
    from .glow import glow_sheet_path, write_glow_sheet
    own = glow_sheet_path(sheet_png)
    dst = folder / f"{name}_glow.png"
    if own.exists():
        shutil.copyfile(own, dst)
    else:
        dst.unlink(missing_ok=True)
        if glow:                                          # frames come from the copied .json
            write_glow_sheet(folder / f"{name}.png", out=dst, force=True)
    for extra in (".tres", "_facing.gd"):                 # the iso 2D look uses SpriteFrames
        src = sheet_png.with_name(sheet_png.stem + extra)
        if src.exists():
            text = src.read_text(encoding="utf-8").replace(sheet_png.name, f"{name}.png")
            text = text.replace(f'path="res://{sheet_png.name}"', f'path="res://units/{name}/{name}.png"')
            (folder / f"{name}{extra}").write_text(text, encoding="utf-8")
    frames = meta["frames"]
    w, h = frames[0]["frame"]["w"], frames[0]["frame"]["h"]
    sheet = np.asarray(Image.open(sheet_png).convert("RGBA"))
    idle = [f for f in frames if f["filename"].startswith("idle_")] or frames
    anims = sorted({f["filename"].split(" ")[0].rsplit("_", 1)[0] for f in frames})
    info = {"frame_w": w, "frame_h": h, "foot": _foot_row(sheet, idle), "ppu": ppu, "anims": anims}
    info.update(unit_stride(sheet_png, meta, ppu))
    info.update(sheet_view(sheet_png, meta))
    (folder / f"{name}.unit.json").write_text(json.dumps(info), encoding="utf-8")
    return folder


def sheet_view(sheet_png, meta: dict | None = None) -> dict:
    """{elevation the sheet was drawn at, weak: directions its quality check left failing}
    from the sheet json meta / the character's keys/keys.json (empty when unknown)."""
    out = {}
    m = (meta or {}).get("meta") or {}
    if m.get("elevation") is not None:
        out["elevation"] = float(m["elevation"])
    keys = Path(sheet_png).parent / "keys" / "keys.json"
    if keys.exists():
        try:
            kj = json.loads(keys.read_text(encoding="utf-8"))
        except ValueError:
            kj = {}
        cam = kj.get("camera") or {}
        if "elevation" not in out and cam.get("elevation") is not None:
            out["elevation"] = float(cam["elevation"])
        weak = (kj.get("quality") or {}).get("failed") or []
        if weak:
            out["weak"] = list(weak)
    return out


def unit_stride(sheet_png, meta: dict, ppu: float) -> dict:
    """{"stride": {clip: {units, px, cycle_ms, frames}}, "body_px"} for unit.json: the sheet's
    measured stride (motion.py; sheet json meta.stride) in tiles at this ppu, or one measured
    from the frames now (else a default from the body height). board.gd plays walk/run at
    speed / stride cycles per second, so the feet stay planted."""
    from .motion import measure_sheet
    m = meta.get("meta", {}) or {}
    stride, body = dict(m.get("stride") or {}), m.get("body_px")
    if not stride:
        try:
            got = measure_sheet(sheet_png)
            stride, body = got["stride"], got["body_px"]
        except Exception:  # noqa: BLE001 - odd sheets: the runtime's default stride
            stride = {}
    out = {k: {"units": round(float(v["px"]) / ppu, 4), "px": v["px"], "cycle_ms": v.get("cycle_ms"),
               "frames": v.get("frames"), "source": v.get("source", "")}
           for k, v in stride.items() if v.get("px")}
    return {"stride": out, "body_px": body}


def add_effect(project, name: str, sheet_png, ppu: float = 42.0, loop: bool | None = None,
               ground: bool | None = None) -> Path:
    """Copy an effect sheet (effects.py: sheet.png + .json, maybe sheet_glow.png) into
    res://effects/<name>/ with <name>.effect.json (frames, loop, the colour of its light). No
    glow map: the whole effect glows by brightness. Casters play it on hit; map.json "effects"
    [{name, col, row}] place looping ones (a magic circle, a fire) that light their cell."""
    from .glow import glow_colour, glow_sheet_path, write_glow_sheet
    sheet_png = Path(sheet_png)
    folder = Path(project) / "effects" / name
    folder.mkdir(parents=True, exist_ok=True)
    meta = json.loads(sheet_png.with_suffix(".json").read_text(encoding="utf-8"))
    png = folder / f"{name}.png"
    shutil.copyfile(sheet_png, png)
    (folder / f"{name}.json").write_text(json.dumps(meta), encoding="utf-8")
    glow = folder / f"{name}_glow.png"
    if glow_sheet_path(sheet_png).exists():
        shutil.copyfile(glow_sheet_path(sheet_png), glow)
    else:
        write_glow_sheet(png, effect=True, force=True)
    raw = meta["frames"]
    items = list(raw.values()) if isinstance(raw, dict) else raw
    frames = [[f["frame"]["x"], f["frame"]["y"], f["frame"]["w"], f["frame"]["h"], f.get("duration", 80)]
              for f in items]
    kind = ""
    tags = meta.get("meta", {}).get("frameTags", [])
    if tags:
        kind = tags[0].get("name", "")
    if loop is None or ground is None:
        try:
            from .effects import SPECS
            spec = SPECS.get(kind)
        except ImportError:
            spec = None
        loop = bool(spec[3]) if loop is None and spec else bool(loop)
        ground = kind in ("magic_circle", "dust", "heal") if ground is None else ground
    info = {"frame_w": frames[0][2], "frame_h": frames[0][3], "frames": frames, "ppu": ppu,
            "loop": loop, "ground": ground, "light": glow_colour(png, glow) or [1.0, 0.8, 0.5],
            "tags": {t["name"]: [int(t["from"]), int(t["to"])] for t in tags if "name" in t}}
    (folder / f"{name}.effect.json").write_text(json.dumps(info), encoding="utf-8")
    return folder


def board_look(tiles, tile_px: int):
    """make_hd2d's tiles= -> a tilegen.MapLook or None (see make_hd2d)."""
    if tiles is False:
        return None
    if tiles is not None and hasattr(tiles, "sets"):
        return tiles
    try:
        from .tilegen import MapLook
        return MapLook.for_project(tiles, tile=tile_px)
    except Exception:  # noqa: BLE001 - no project / unreadable sets: the procedural look
        return None


def make_hd2d(out_dir, units: dict, diorama: Diorama | None = None, *, name: str = "HD-2D Tactics",
              iso: bool = True, tile_px: int = 32, ppu: float = 42.0, lighting: str = "sunset",
              bands: float = 4.0, effects: dict | None = None, place_effects=(),
              tiles=None, weather=None, water: str = "classic", autoplay=None, ui_kit=None,
              unit_ppu: dict | None = None, apron: int = 0, iso_scale: dict | None = None,
              pixel: dict | None = None) -> list[Path]:
    """Write the Godot project: menu, real 3D and fake 3D scenes (and the iso 2D look when
    iso_scene is available). units: {name: sheet.png from turnaround --animate}. lighting: the
    time of day it starts at (LIGHTING: dawn, noon, sunset, night; L cycles them), bands: toon
    steps for sprite lighting (0 = smooth). effects: {name: effect sheet.png};
    place_effects: [(name, col, row)] looping on the map. tiles: what textures the terrain -
    None = the current project's saved tilesets at tile_px when it has any (tilegen.MapLook:
    tops from each terrain's own set with a variant per cell, walls in the cliff set's rock,
    stone tiers in the wall set's, planks from the fence set), a Project / MapLook = those,
    False = the procedural neo 16-bit look. weather: rain | storm | snow | fog | god_rays | leaves |
    ... (weather.py kinds, a saved weather asset, or a list): the effect over the board + matching
    light (weather_hd2d.install; needs `godot --import`). water: classic | pixel (WATER_SHADER_PIXEL:
    banded pixel ripples, shore foam, reflections). autoplay: a timeline (dict or .json path,
    hd2d_autoplay) written to res://autoplay.json; the game plays it with `--autoplay` (or on its
    own when it has "auto": true). ui_kit: a UI kit folder (uikit.py) whose pieces the autoplay HUD
    uses (res://ui/). unit_ppu: {unit name: pixels per tile} for sheets drawn at another scale (a
    144x96 sheet next to 96x64 ones: 1.5 x ppu). apron: cells of scenery drawn around the map
    (the land goes on past its edge, so a camera near the edge never shows the void; not
    walkable). iso_scale: {unit: k} the iso look rescales those sheets by (default: the smallest
    unit_ppu / each one's, so all share one world scale and nothing is enlarged). pixel:
    {scale, pitch} for the pixel-perfect look (res://pixel3d.tscn, board mode "pixel"; tile =
    tile_px); None = assets.pixel_grid of the current project (always written: the menu offers
    it). Every look draws sprites at whole texels (one texel = k x k screen pixels).

    Every project gets the game runtime in res://pp_runtime/ (pp_motion.gd + motion.json): board.gd
    moves, turns, animates (stride-matched feet), hops, hit-stops and follows with it."""
    if water not in WATER_LOOKS:
        raise ValueError(f"Unknown water look '{water}' (use {', '.join(WATER_LOOKS)})")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d = diorama or harbour()
    known = set(units)
    for u in d.units:
        if u.name not in known:
            raise ValueError(f"The map has a unit '{u.name}' but no sprite sheet for it "
                             f"(given: {', '.join(sorted(known)) or 'none'}).")
    written = [d.save(out / "map" / "map.json")]
    from .diorama import BLOCKING_PROPS
    mj = json.loads(written[0].read_text(encoding="utf-8"))
    if place_effects:
        mj["effects"] = [{"name": n, "col": int(c), "row": int(r)} for n, c, r in place_effects]
    mj["blocking"] = sorted(BLOCKING_PROPS | {"bush", "pot"})   # board.gd: these props block
    from .diorama import BLOCKING_RADII
    mj["radii"] = dict(BLOCKING_RADII)                           # ... with these feet radii
    if apron:
        mj["apron"] = int(apron)
    written[0].write_text(json.dumps(mj), encoding="utf-8")
    written.append(write_lighting(out, lighting, bands))
    for ename, sheet in (effects or {}).items():
        written.append(add_effect(out, ename, sheet, ppu))
    write_textures(d, out / "map" / "tex", tile_px, look=board_look(tiles, tile_px))
    for uname, sheet in units.items():
        written.append(add_unit(out, uname, sheet, (unit_ppu or {}).get(uname, ppu)))
    from .assets import pixel_grid
    grid = pixel_grid(tile=tile_px)
    px = {"scale": grid["scale"], "pitch": grid["pitch"], **(pixel or {})}
    written += write_pixel_look(out, tile_px, int(px["scale"]), float(px["pitch"]))
    (out / "shaders").mkdir(exist_ok=True)
    files = {
        "project.godot": PROJECT.format(name=name),
        "menu.tscn": MENU_TSCN, "menu.gd": MENU_GD,
        "real3d.tscn": REAL_TSCN, "fake3d.tscn": FAKE_TSCN, "board.gd": BOARD_GD,
        "shaders/sprite.gdshader": SPRITE_SHADER,
        "shaders/water.gdshader": WATER_SHADER_PIXEL if water == "pixel" else WATER_SHADER,
        "shaders/flame.gdshader": FLAME_SHADER, "shaders/ember.gdshader": EMBER_SHADER,
        "shaders/blob.gdshader": SHADOW_BLOB_SHADER, "light_flicker.gd": LIGHT_GD,
        "shaders/terrain.gdshader": TERRAIN_SHADER,
    }
    for rel, text in files.items():
        p = out / rel
        p.write_text(text, encoding="utf-8")
        written.append(p)
    from .godot_runtime import install as install_runtime
    from .motion import write_motion_json
    written += install_runtime(out, ["pp_motion.gd"])          # res://pp_runtime/
    written.append(write_motion_json(out / "pp_runtime"))
    from .hd2d_autoplay import install as install_autoplay
    if autoplay is not None:
        from .hd2d_autoplay import validate, unit_ids
        script = autoplay if isinstance(autoplay, dict) else json.loads(Path(autoplay).read_text(encoding="utf-8"))
        errs = validate(script, unit_ids([u.name for u in d.units]))
        if errs:
            raise ValueError("Autoplay script: " + "; ".join(errs[:6]))
        autoplay = script
    written += install_autoplay(out, autoplay, ui_kit)
    if weather:
        from .weather_hd2d import install
        written += install(out, weather, tod=lighting, ppu=ppu)
    if iso:
        try:
            from .iso_scene import write_iso_scene
        except ImportError:
            write_iso_scene = None
        if write_iso_scene is not None:
            sheets = {n: out / "units" / n / f"{n}.png" for n in units}
            own = {n: float((unit_ppu or {}).get(n, ppu)) for n in units}
            base = min(own.values()) if own else ppu              # one world scale: only shrink
            scale = iso_scale or {n: base / own[n] for n in units}
            written += write_iso_scene(d, out, sheets, tile=tile_px, unit_scale=scale)
    return written

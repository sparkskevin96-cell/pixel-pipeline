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
window/stretch/mode="canvas_items"

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
	["Isometric 2D  (stacked tiles, Tactics Ogre style)", "res://iso/iso.tscn"],
]

func _ready() -> void:
	var win := get_tree().root                   # the iso look switches to 640x360 pixels
	win.content_scale_size = Vector2i(1920, 1080)
	win.content_scale_mode = Window.CONTENT_SCALE_MODE_CANVAS_ITEMS
	win.content_scale_stretch = Window.CONTENT_SCALE_STRETCH_FRACTIONAL
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
	hint.text = "Click a unit, click a lit cell to move. Space: end turn. Q/E: turn camera. WASD: pan. Wheel: zoom. Esc: menu."
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

SPRITE_SHADER = """shader_type spatial;
// A frame of a sprite sheet on a quad that stands up and faces the camera (turning about
// the vertical axis only, like HD-2D), lit by the scene through the sheet's normal map.
render_mode cull_disabled, depth_prepass_alpha;

uniform sampler2D sheet : source_color, filter_nearest;
uniform sampler2D normals : hint_normal, filter_nearest;
uniform bool has_normals = false;
uniform vec4 frame = vec4(0.0, 0.0, 1.0, 1.0);   // uv x, y, w, h
uniform bool face_camera_fully = false;           // fake 3D: no foreshortening
uniform vec3 tint = vec3(1.0);
uniform float flash = 0.0;                        // selection glow

void vertex() {
	vec3 up = face_camera_fully ? INV_VIEW_MATRIX[1].xyz : vec3(0.0, 1.0, 0.0);
	vec3 right = normalize(INV_VIEW_MATRIX[0].xyz);
	vec3 back = normalize(cross(right, up));
	if (face_camera_fully) { back = normalize(INV_VIEW_MATRIX[2].xyz); }
	float sx = length(MODEL_MATRIX[0].xyz);
	float sy = length(MODEL_MATRIX[1].xyz);
	MODELVIEW_MATRIX = VIEW_MATRIX * mat4(vec4(right * sx, 0.0), vec4(up * sy, 0.0),
										  vec4(back, 0.0), MODEL_MATRIX[3]);
	MODELVIEW_NORMAL_MATRIX = mat3(MODELVIEW_MATRIX);
}

void fragment() {
	vec2 uv = frame.xy + UV * frame.zw;
	vec4 c = texture(sheet, uv);
	if (c.a < 0.5) { discard; }
	ALBEDO = c.rgb * tint + vec3(flash);
	ROUGHNESS = 0.85;
	SPECULAR = 0.15;
	if (has_normals) {
		NORMAL_MAP = texture(normals, uv).rgb;
		NORMAL_MAP_DEPTH = 1.0;
	}
}
"""

WATER_SHADER = """shader_type spatial;
// Deep blue water with moving sparkles: light catches the ripples (glow turns them to bloom).
render_mode cull_disabled;

uniform sampler2D ripples : filter_linear, repeat_enable;
uniform vec3 deep : source_color = vec3(0.04, 0.10, 0.30);
uniform vec3 shallow : source_color = vec3(0.10, 0.28, 0.62);
uniform float sparkle = 2.5;

void vertex() {
	VERTEX.y += sin(TIME * 1.3 + VERTEX.x * 0.9) * 0.02 + cos(TIME * 1.1 + VERTEX.z * 0.7) * 0.02;
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

FLAME_SHADER = """shader_type spatial;
render_mode unshaded, cull_disabled, blend_add;
uniform float seed = 0.0;
void vertex() {
	MODELVIEW_MATRIX = VIEW_MATRIX * mat4(INV_VIEW_MATRIX[0], INV_VIEW_MATRIX[1], INV_VIEW_MATRIX[2], MODEL_MATRIX[3]);
}
void fragment() {
	vec2 p = UV * 2.0 - 1.0;
	float flick = 0.85 + 0.15 * sin(TIME * 17.0 + seed * 7.0);
	float d = length(vec2(p.x * 1.6, (p.y + 0.2) * 1.0));
	float core = smoothstep(1.0, 0.0, d) * flick;
	ALBEDO = mix(vec3(1.0, 0.25, 0.02), vec3(1.0, 0.85, 0.4), core) * core * 4.0;
	ALPHA = core;
}
"""

BOARD_GD = r'''extends Node3D
# The battle map in 3D, built at runtime from res://map/map.json, and the units on it.
# mode "real": perspective telephoto + tilt-shift + bloom.  "fake": orthographic, rendered
# small by the parent SubViewport and scaled up: 3D that reads as pixel art.

@export var mode := "real"

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

func _ready() -> void:
	map = JSON.parse_string(FileAccess.get_file_as_string("res://map/map.json"))
	level = map.get("level", 0.5)
	cols = int(map["cols"])
	rows = int(map["rows"])
	heights = map["height"]
	terrain = map["terrain"]
	_environment()
	_build_terrain()
	_build_water()
	_build_props()
	_build_units()
	_build_camera()
	overlay = Node3D.new()
	add_child(overlay)
	_build_cursor()
	_hud()

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
	var we := WorldEnvironment.new()
	var env := Environment.new()
	env.background_mode = Environment.BG_COLOR
	env.background_color = Color(0.03, 0.02, 0.05)
	env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.ambient_light_color = Color(0.45, 0.38, 0.62)
	env.ambient_light_energy = 0.55
	env.tonemap_mode = Environment.TONE_MAPPER_ACES
	env.tonemap_exposure = 1.05
	env.glow_enabled = true
	env.glow_intensity = 0.9 if mode == "real" else 0.6
	env.glow_bloom = 0.08
	env.glow_hdr_threshold = 0.9
	env.glow_blend_mode = Environment.GLOW_BLEND_MODE_SOFTLIGHT
	env.ssao_enabled = mode == "real"
	env.ssao_intensity = 1.6
	env.adjustment_enabled = true
	env.adjustment_saturation = 1.18
	env.adjustment_contrast = 1.06
	we.environment = env
	add_child(we)
	var sun := DirectionalLight3D.new()
	sun.light_color = Color(1.0, 0.72, 0.45)       # low warm sunset
	sun.light_energy = 1.25
	sun.shadow_enabled = true
	sun.directional_shadow_max_distance = 80.0
	sun.rotation_degrees = Vector3(-38, -60, 0)
	add_child(sun)
	var fill := DirectionalLight3D.new()            # cool bounce from the other side
	fill.light_color = Color(0.45, 0.55, 1.0)
	fill.light_energy = 0.35
	fill.rotation_degrees = Vector3(-55, 120, 0)
	add_child(fill)

func _materials() -> Dictionary:
	var mats := {}
	for name in ["grass", "stone", "cobble", "planks", "dirt", "sand", "water", "snow"]:
		var top := _tex("res://map/tex/%s_top.png" % name)
		if top == null:
			continue
		var m := StandardMaterial3D.new()
		m.albedo_texture = top
		m.texture_filter = BaseMaterial3D.TEXTURE_FILTER_NEAREST_WITH_MIPMAPS
		m.roughness = 0.95
		var s := StandardMaterial3D.new()
		s.albedo_texture = _tex("res://map/tex/%s_side.png" % name)
		s.texture_filter = BaseMaterial3D.TEXTURE_FILTER_NEAREST_WITH_MIPMAPS
		s.roughness = 1.0
		s.albedo_color = Color(0.86, 0.84, 0.9)
		mats[name] = [m, s]
	return mats

func _quad(st: SurfaceTool, a: Vector3, b: Vector3, c: Vector3, d: Vector3, n: Vector3, uv: Array) -> void:
	for idx in [[a, uv[0]], [b, uv[1]], [c, uv[2]], [a, uv[0]], [c, uv[2]], [d, uv[3]]]:
		st.set_normal(n)
		st.set_uv(idx[1])
		st.add_vertex(idx[0])

func _build_terrain() -> void:
	var mats := _materials()
	var tops := {}
	var sides := {}
	var used := {}
	var body := StaticBody3D.new()
	add_child(body)
	for r in rows:
		for c in cols:
			var t: String = terrain[r][c]
			if t == "water":
				continue
			var y := top_y(c, r)
			if not tops.has(t):
				tops[t] = SurfaceTool.new()
				tops[t].begin(Mesh.PRIMITIVE_TRIANGLES)
				sides[t] = SurfaceTool.new()
				sides[t].begin(Mesh.PRIMITIVE_TRIANGLES)
			_quad(tops[t], Vector3(c, y, r), Vector3(c + 1, y, r), Vector3(c + 1, y, r + 1), Vector3(c, y, r + 1),
				Vector3.UP, [Vector2(c, r), Vector2(c + 1, r), Vector2(c + 1, r + 1), Vector2(c, r + 1)])
			# walls down to each lower neighbour (or below the water for edges)
			for nb in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
				var nc: int = c + nb[0]
				var nr: int = r + nb[1]
				var ny := -1.0
				if nc >= 0 and nr >= 0 and nc < cols and nr < rows and terrain[nr][nc] != "water":
					ny = top_y(nc, nr)
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
				_quad(sides[t], p0 + Vector3(0, y, 0), p1 + Vector3(0, y, 0), p1 + Vector3(0, ny, 0), p0 + Vector3(0, ny, 0),
					n, [Vector2(u0, -y), Vector2(u1, -y), Vector2(u1, -ny), Vector2(u0, -ny)])
			var shape := CollisionShape3D.new()
			var box := BoxShape3D.new()
			box.size = Vector3(1, y + 1.0, 1)
			shape.shape = box
			shape.position = Vector3(c + 0.5, (y - 1.0) / 2.0, r + 0.5)
			shape.set_meta("cell", Vector2i(c, r))
			body.add_child(shape)
	body.set_meta("terrain", true)
	for t in tops:
		for pair in [[tops[t], 0], [sides[t], 1]]:
			if pair[1] == 1 and not used.has(t):
				continue
			var st: SurfaceTool = pair[0]
			st.generate_tangents()
			var mi := MeshInstance3D.new()
			mi.mesh = st.commit()
			mi.material_override = mats[t][pair[1]] if mats.has(t) else null
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
	pm.size = Vector2(cols + 40, rows + 40) if map.get("sea", true) else Vector2(cols, rows)
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
		var base := cell_pos(c, r)
		var kind: String = p["kind"]
		if ext != null and ext.build(self, kind, c, r, base, i):
			i += 1
			continue
		if kind in ["crate", "barrel", "post"]:
			blocked[Vector2i(c, r)] = true
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
			"torch":
				var tb := BoxMesh.new(); tb.size = Vector3(0.1, 1.3, 0.1); mi.mesh = tb
				mi.position = base + Vector3(0.38, 0.65, 0.38)
				var flame := MeshInstance3D.new()
				var q := QuadMesh.new(); q.size = Vector2(0.35, 0.5); flame.mesh = q
				var fm := ShaderMaterial.new(); fm.shader = load("res://shaders/flame.gdshader")
				fm.set_shader_parameter("seed", float(i)); flame.material_override = fm
				flame.position = mi.position + Vector3(0, 0.8, 0)
				add_child(flame)
				var light := OmniLight3D.new()
				light.light_color = Color(1.0, 0.55, 0.2)
				light.light_energy = 2.2
				light.omni_range = 4.5
				light.omni_attenuation = 1.6
				light.position = flame.position
				light.set_meta("flicker", float(i))
				add_child(light)
			_:                                # unknown kind: a plain box so it still shows
				var ub := BoxMesh.new(); ub.size = Vector3(0.6, 0.6, 0.6); mi.mesh = ub
				mi.position = base + Vector3(0, 0.3, 0)
		mi.material_override = wood
		add_child(mi)
		i += 1

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
		"frames": frames, "w": float(info["frame_w"]), "h": float(info["frame_h"]),
		"foot": float(info["foot"]), "ppu": float(info["ppu"]), "anims": info["anims"]}
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
		mat.set_shader_parameter("face_camera_fully", mode == "fake")
		mat.set_shader_parameter("tint", Color(1, 1, 1) if int(u["team"]) == 0 else Color(1.0, 0.82, 0.82))
		mesh.material_override = mat
		mesh.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_ON
		node.add_child(mesh)
		# a soft blob shadow on the ground, like HD-2D
		var blob := MeshInstance3D.new()
		var bq := QuadMesh.new(); bq.size = Vector2(0.8, 0.5); bq.orientation = PlaneMesh.FACE_Y
		blob.mesh = bq
		var bm := StandardMaterial3D.new()
		bm.albedo_color = Color(0, 0, 0, 0.35); bm.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
		bm.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
		blob.material_override = bm
		blob.position.y = 0.02
		node.add_child(blob)
		add_child(node)
		var c := int(u["col"]); var r := int(u["row"])
		node.position = cell_pos(c, r)
		var hp := int(u.get("hp", 28))
		var bar := Sprite3D.new()
		bar.billboard = BaseMaterial3D.BILLBOARD_ENABLED
		bar.no_depth_test = true
		bar.shaded = false
		bar.texture_filter = BaseMaterial3D.TEXTURE_FILTER_NEAREST
		bar.pixel_size = 0.035 if mode == "real" else 0.03
		bar.render_priority = 10
		bar.position = Vector3(0, q.size.y + q.center_offset.y - q.size.y / 2.0 + 0.15, 0)
		node.add_child(bar)
		var unit := {"node": node, "mesh": mesh, "mat": mat, "sheet": s, "col": c, "row": r,
			"team": int(u["team"]), "yaw": _dir_yaw(u.get("facing", "S")), "anim": "idle", "t": randf() * 2.0,
			"name": u["name"], "hp": hp, "max_hp": hp, "atk": int(u.get("attack", 10)),
			"def": int(u.get("defence", 5)), "move": int(u.get("move", MOVE)), "jump": int(u.get("jump", JUMP)),
			"reach": int(u.get("reach", 1)), "done": false, "bar": bar, "hold": false,
			"tint": Color(1, 1, 1) if int(u["team"]) == 0 else Color(1.0, 0.82, 0.82)}
		units.append(unit)
		_update_bar(unit)
		blocked[Vector2i(c, r)] = true

func _dir_yaw(name: String) -> float:
	# map directions: N = toward row 0 (-z), E = +x, S = +z
	var i := DIRS.find(name)
	return deg_to_rad(i * 22.5) if i >= 0 else PI / 2.0

func _sprite_dir(world_yaw: float) -> String:
	# the world direction, seen from this camera, as one of the 16 sprite directions
	var d := Vector3(cos(world_yaw), 0, sin(world_yaw))
	var right := cam.global_transform.basis.x
	var fwd := -cam.global_transform.basis.z
	fwd.y = 0
	fwd = fwd.normalized()
	var a := atan2(-d.dot(fwd), d.dot(right))      # screen angle, y down (toward camera = S)
	var idx := int(round(fposmod(a, TAU) / (TAU / 16.0))) % 16
	return DIRS[idx]

func _animate_units(delta: float) -> void:
	for u in units:
		var s: Dictionary = u["sheet"]
		u["t"] += delta
		var anim: String = u["anim"] if u["anim"] in s["anims"] else "idle"
		var key := "%s_%s" % [anim, _sprite_dir(u["yaw"])]
		if not s["frames"].has(key):
			continue
		var fr: Array = s["frames"][key]
		var total := 0.0
		for f in fr: total += f["ms"]
		var t := fmod(u["t"] * 1000.0, total)
		if u["hold"]:
			t = min(u["t"] * 1000.0, total - 1.0)
		var pick = fr[0]
		for f in fr:
			if t < f["ms"]:
				pick = f
				break
			t -= f["ms"]
		var r: Rect2 = pick["rect"]
		u["mat"].set_shader_parameter("frame", Vector4(r.position.x, r.position.y, r.size.x, r.size.y))
		u["mat"].set_shader_parameter("flash", 0.18 * (0.5 + 0.5 * sin(Time.get_ticks_msec() / 150.0)) if u == selected else u.get("hit", 0.0))
		u["mat"].set_shader_parameter("tint", u["tint"] * (0.5 if u["done"] else 1.0))

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
	if mode == "fake":
		cam.projection = Camera3D.PROJECTION_ORTHOGONAL
		cam.size = 11.0
		cam_dist = 40.0
	else:
		cam.fov = 22.0                                # telephoto: the diorama look
		var attr := CameraAttributesPractical.new()
		attr.dof_blur_far_enabled = true
		attr.dof_blur_near_enabled = true
		attr.dof_blur_amount = 0.16
		cam.attributes = attr
	_place_camera()

func _place_camera() -> void:
	var pitch := deg_to_rad(38.0 if mode == "real" else 30.0)
	var yaw := deg_to_rad(cam_yaw)
	cam.position = Vector3(sin(yaw) * cos(pitch), sin(pitch), cos(yaw) * cos(pitch)) * cam_dist
	cam.look_at(pivot.global_position, Vector3.UP)
	if mode == "real" and cam.attributes:
		# tilt-shift: sharp in a band around the focus, blurred near and far
		cam.attributes.dof_blur_far_distance = cam_dist + 2.0
		cam.attributes.dof_blur_far_transition = 6.0
		cam.attributes.dof_blur_near_distance = cam_dist - 3.0
		cam.attributes.dof_blur_near_transition = 3.5

func _process(delta: float) -> void:
	var pan := Vector3.ZERO
	if Input.is_key_pressed(KEY_W) or Input.is_key_pressed(KEY_UP): pan.z -= 1
	if Input.is_key_pressed(KEY_S) or Input.is_key_pressed(KEY_DOWN): pan.z += 1
	if Input.is_key_pressed(KEY_A) or Input.is_key_pressed(KEY_LEFT): pan.x -= 1
	if Input.is_key_pressed(KEY_D) or Input.is_key_pressed(KEY_RIGHT): pan.x += 1
	if pan != Vector3.ZERO:
		pivot.position += pan.normalized().rotated(Vector3.UP, deg_to_rad(cam_yaw)) * delta * 8.0
	cam_yaw = lerp(cam_yaw, cam_yaw_target, min(1.0, delta * 8.0))
	_place_camera()
	if mode == "fake":
		_snap_camera()
	_animate_units(delta)
	for l in get_children():
		if l is OmniLight3D and l.has_meta("flicker"):
			var k: float = l.get_meta("flicker")
			l.light_energy = 2.0 + 0.35 * sin(Time.get_ticks_msec() / 90.0 + k * 3.1) + 0.2 * sin(Time.get_ticks_msec() / 37.0 + k)
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
	hud.text = "%s team's turn%s\n%s %s\nSpace: end turn   Q/E: turn camera   Wheel: zoom   Esc: menu" % [
		_team_name(active_team), sel, help, extra]

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
	var p: Vector3 = hit["position"] - hit["normal"] * 0.01
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
	phase = "busy"
	_clear_range()
	blocked.erase(Vector2i(u["col"], u["row"]))
	_play(u, "walk")
	var tw := create_tween()
	for i in range(1, path.size()):
		var a: Vector2i = path[i - 1]
		var b: Vector2i = path[i]
		var to := cell_pos(b.x, b.y)
		var hop := absf(top_y(b.x, b.y) - top_y(a.x, a.y)) > 0.01
		tw.tween_callback(func(): u["yaw"] = atan2(float(b.y - a.y), float(b.x - a.x)))
		if hop:
			var mid := (cell_pos(a.x, a.y) + to) / 2.0 + Vector3(0, 0.6, 0)
			tw.tween_property(u["node"], "position", mid, 0.13)
			tw.tween_property(u["node"], "position", to, 0.13)
		else:
			tw.tween_property(u["node"], "position", to, 0.26)
	tw.tween_callback(func():
		var last: Vector2i = path[path.size() - 1]
		u["col"] = last.x; u["row"] = last.y
		blocked[last] = true
		_play(u, "idle")
		_enter_act(u))

func _play(u: Dictionary, anim: String, once := false) -> void:
	u["anim"] = anim
	u["t"] = 0.0
	u["hold"] = once

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

func _attack(a: Dictionary, t: Dictionary) -> void:
	phase = "busy"
	_clear_range()
	var dmg := _damage(a, t)
	a["yaw"] = atan2(float(t["row"] - a["row"]), float(t["col"] - a["col"]))
	t["yaw"] = atan2(float(a["row"] - t["row"]), float(a["col"] - t["col"]))
	var tw := create_tween()
	if "attack" in a["sheet"]["anims"]:
		_play(a, "attack", true)
		tw.tween_interval(0.45)
	else:                                         # no attack clip: lunge at the target
		var home: Vector3 = a["node"].position
		var toward: Vector3 = (t["node"].position - home).normalized() * 0.35
		tw.tween_property(a["node"], "position", home + toward, 0.12)
		tw.tween_property(a["node"], "position", home, 0.16)
	tw.tween_callback(func():
		t["hp"] = max(0, t["hp"] - dmg)
		_update_bar(t)
		_number(t, dmg)
		t["hit"] = 0.6
		if "hurt" in t["sheet"]["anims"] and t["hp"] > 0:
			_play(t, "hurt", true))
	tw.tween_method(func(v): t["hit"] = v, 0.6, 0.0, 0.35)
	tw.tween_interval(0.25)
	tw.tween_callback(func():
		if t["hp"] > 0:
			_play(t, "idle")
			_play(a, "idle")
			_finish(a)
		else:
			_kill(t, a))

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
		t["node"].queue_free()
		_play(a, "idle")
		_finish(a))

func _number(t: Dictionary, dmg: int) -> void:
	var l := Label3D.new()
	l.text = str(dmg)
	l.font_size = 64 if mode == "real" else 40
	l.pixel_size = 0.008 if mode == "real" else 0.012
	l.outline_size = 14
	l.modulate = Color(1.0, 0.95, 0.4)
	l.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	l.no_depth_test = true
	l.texture_filter = BaseMaterial3D.TEXTURE_FILTER_NEAREST if mode == "fake" else BaseMaterial3D.TEXTURE_FILTER_LINEAR
	l.position = t["node"].position + Vector3(0, 1.6, 0)
	add_child(l)
	var tw := create_tween()
	tw.set_parallel(true)
	tw.tween_property(l, "position", l.position + Vector3(0, 0.9, 0), 0.8)
	tw.tween_property(l, "modulate:a", 0.0, 0.8).set_delay(0.35)
	tw.chain().tween_callback(l.queue_free)

func _finish(u: Dictionary) -> void:
	_clear_range()
	if u != null:
		u["done"] = true
	selected = null
	phase = "select"
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


def add_unit(project: Path, name: str, sheet_png, ppu: float = 42.0) -> Path:
    """Copy a unit's sheet (+ .json, + _n.png if there) into res://units/<name>/ and write
    <name>.unit.json (frame size, feet row, pixels per tile, animations)."""
    sheet_png = Path(sheet_png)
    folder = project / "units" / name
    folder.mkdir(parents=True, exist_ok=True)
    meta = json.loads(sheet_png.with_suffix(".json").read_text(encoding="utf-8"))
    shutil.copyfile(sheet_png, folder / f"{name}.png")
    (folder / f"{name}.json").write_text(json.dumps(meta), encoding="utf-8")
    normals = sheet_png.with_name(sheet_png.stem + "_n.png")
    if normals.exists():
        shutil.copyfile(normals, folder / f"{name}_n.png")
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
    (folder / f"{name}.unit.json").write_text(json.dumps(info), encoding="utf-8")
    return folder


def make_hd2d(out_dir, units: dict, diorama: Diorama | None = None, *, name: str = "HD-2D Tactics",
              iso: bool = True, tile_px: int = 32, ppu: float = 42.0) -> list[Path]:
    """Write the Godot project: menu, real 3D and fake 3D scenes (and the iso 2D look when
    iso_scene is available). units: {name: sheet.png from turnaround --animate}."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d = diorama or harbour()
    known = set(units)
    for u in d.units:
        if u.name not in known:
            raise ValueError(f"The map has a unit '{u.name}' but no sprite sheet for it "
                             f"(given: {', '.join(sorted(known)) or 'none'}).")
    written = [d.save(out / "map" / "map.json")]
    write_textures(d, out / "map" / "tex", tile_px)
    for uname, sheet in units.items():
        written.append(add_unit(out, uname, sheet, ppu))
    (out / "shaders").mkdir(exist_ok=True)
    files = {
        "project.godot": PROJECT.format(name=name),
        "menu.tscn": MENU_TSCN, "menu.gd": MENU_GD,
        "real3d.tscn": REAL_TSCN, "fake3d.tscn": FAKE_TSCN, "board.gd": BOARD_GD,
        "shaders/sprite.gdshader": SPRITE_SHADER, "shaders/water.gdshader": WATER_SHADER,
        "shaders/flame.gdshader": FLAME_SHADER,
    }
    for rel, text in files.items():
        p = out / rel
        p.write_text(text, encoding="utf-8")
        written.append(p)
    if iso:
        try:
            from .iso_scene import write_iso_scene
        except ImportError:
            write_iso_scene = None
        if write_iso_scene is not None:
            sheets = {n: out / "units" / n / f"{n}.png" for n in units}
            written += write_iso_scene(d, out, sheets, tile=tile_px)
    return written

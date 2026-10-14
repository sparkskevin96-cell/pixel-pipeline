"""Godot 4 scenes for weather assets (weather.py): files sit next to sheet.png in the asset folder,
so copying the folder anywhere into a Godot project works (relative paths).

    <name>_overlay.tscn      CanvasLayer: the looping tile drawn over the whole screen at whole-pixel
                             scale (weather_overlay.gdshader), following the camera; lightning =
                             flash + bolt; for 2D games
    <name>_particles2d.tscn  Node2D with a GPUParticles2D per depth layer covering a 480x270 view
                             (put it under your Camera2D)
    <name>_particles3d.tscn  Node3D with a GPUParticles3D per layer (billboard quads at the
                             project's pixels per tile) over a 16 x 12 m area: 3D / HD-2D boards;
                             cloud shadows = a drifting Decal
    weather_overlay.gdshader / weather_overlay_add.gdshader, weather_fx.gd (drives them)
"""
from __future__ import annotations

import math
from pathlib import Path

OVERLAY_SHADER = """shader_type canvas_item;
render_mode unshaded{blend};
// pixel-pipeline weather overlay: a looping sprite-sheet tile repeated over the screen in whole
// art pixels (pixel_scale screen px each), scrolling with `scroll` (art px/s) and the camera.
uniform sampler2D sheet : filter_nearest, repeat_disable;
uniform sampler2D glow_sheet : filter_nearest, repeat_disable;
uniform float glow_strength = 0.0;
uniform vec2 frame_size = vec2(128.0, 128.0);
uniform int frames = 1;
uniform int columns = 1;
uniform float fps = 25.0;
uniform float pixel_scale = 4.0;
uniform vec2 scroll = vec2(0.0);
uniform vec2 offset = vec2(0.0);
uniform int wrap_mode = 0;   // 0 = both ways, 1 = across only (a band at the top)
uniform float opacity = 1.0;
uniform vec4 tint : source_color = vec4(1.0);

void fragment() {
	vec2 screen_px = floor(SCREEN_UV / SCREEN_PIXEL_SIZE / pixel_scale);
	vec2 px = screen_px + floor(offset + scroll * TIME);
	if (wrap_mode == 1 && screen_px.y >= frame_size.y) {
		discard;
	}
	vec2 cell = mod(px, frame_size);
	if (wrap_mode == 1) {
		cell.y = screen_px.y;
	}
	int f = int(mod(floor(TIME * fps), float(frames)));
	vec2 origin = vec2(float(f % columns), float(f / columns)) * frame_size;
	vec2 uv = (origin + cell + 0.5) / vec2(textureSize(sheet, 0));
	vec4 c = texture(sheet, uv) * tint;
	vec4 g = texture(glow_sheet, uv);
	c.rgb += g.rgb * g.a * glow_strength;
	c.a *= opacity;
	COLOR = c;
}
"""

FX_GD = """extends Node
## pixel-pipeline weather: keeps the overlay at whole-pixel scale and following the camera,
## throws lightning (bolt + screen / sky flash) and drifts cloud-shadow decals.
@export var mode := "overlay"            # overlay | lightning2d | lightning3d | clouds3d
@export var base_height := 270           # your game's pixel-art height (480x270 = 4x at 1080p)
@export var follow_camera := true
@export var parallax := 1.0
@export var flash: PackedFloat32Array    # flash per bolt frame (lightning)
@export var flash_ms: PackedInt32Array   # ms per bolt frame
@export var interval := Vector2(3.0, 8.0)  # seconds between strikes
@export var drift := Vector2(0.6, 0.2)   # clouds3d: metres per second
@export var period := 8.0                # clouds3d: the shadow tile's size in metres
@export var flash_energy := 3.0
var strikes := 0
var _mat: ShaderMaterial
var _t := 0.0
var _next := 0.6
var _s := -1.0                           # seconds since the strike started, -1 = none
var _origin := Vector3.ZERO

func _ready() -> void:
	var o := get_node_or_null("Overlay")
	if o != null and o.material is ShaderMaterial:
		_mat = o.material
	var c := get_node_or_null("Clouds")
	if c != null:
		_origin = c.position
	_set_flash(0.0)

func _process(delta: float) -> void:
	_t += delta
	if _mat != null:
		var vp := get_viewport()
		var k := maxf(1.0, floorf(vp.get_visible_rect().size.y / float(base_height)))
		_mat.set_shader_parameter("pixel_scale", k)
		if follow_camera:
			_mat.set_shader_parameter("offset", -vp.get_canvas_transform().origin / k * parallax)
	if mode.begins_with("lightning"):
		_lightning(delta)
	elif mode == "clouds3d":
		var c := get_node_or_null("Clouds") as Node3D
		if c != null:
			c.position = _origin + Vector3(fposmod(drift.x * _t, period), 0.0, fposmod(drift.y * _t, period))

func strike() -> void:
	_s = 0.0
	strikes += 1
	var bolt := get_node_or_null("Bolt")
	if bolt is AnimatedSprite2D:
		var vp := get_viewport().get_visible_rect().size
		bolt.position.x = randf_range(0.2, 0.8) * vp.x
		bolt.visible = true
		bolt.frame = 0
		bolt.play()
	elif bolt is Sprite3D:
		bolt.position.x = randf_range(-5.0, 5.0)
		bolt.position.z = randf_range(-3.0, 3.0)
		bolt.visible = true

func _lightning(delta: float) -> void:
	if _s < 0.0:
		_next -= delta
		if _next <= 0.0:
			_next = randf_range(interval.x, interval.y)
			strike()
		return
	_s += delta
	var acc := 0.0
	var i := -1
	for k in flash_ms.size():
		acc += flash_ms[k] / 1000.0
		if _s < acc:
			i = k
			break
	var bolt := get_node_or_null("Bolt")
	if i < 0:
		_s = -1.0
		_set_flash(0.0)
		if bolt != null:
			bolt.visible = false
		return
	_set_flash(flash[i] if i < flash.size() else 0.0)
	if bolt is Sprite3D:
		bolt.frame = mini(i, bolt.hframes - 1)

func _set_flash(v: float) -> void:
	var f := get_node_or_null("Flash")
	if f is ColorRect:
		f.color.a = v * 0.6
	elif f is Light3D:
		f.light_energy = v * flash_energy
		f.visible = v > 0.01
"""


def _f(v) -> str:
    return f"{float(v):.4g}"


def _gradient(gid: str, e) -> list[str]:
    fi, fo = e.fade
    fi, fo = max(0.0, min(0.45, fi)), max(0.0, min(0.45, fo))
    a = e.alpha
    offs = [0.0, max(fi, 0.001), min(1 - fo, 0.999), 1.0]
    alphas = [0.0 if fi > 0 else a, a, a, 0.0 if fo > 0 else a]
    cols = ", ".join(f"1, 1, 1, {_f(x)}" for x in alphas)
    return [f'[sub_resource type="Gradient" id="{gid}"]',
            f"offsets = PackedFloat32Array({', '.join(_f(o) for o in offs)})",
            f"colors = PackedColorArray({cols})", "",
            f'[sub_resource type="GradientTexture1D" id="{gid}_tex"]',
            f'gradient = SubResource("{gid}")', ""]


def _speed(e) -> float:
    return math.hypot(*e.velocity)


def _anim_lines(e, prefix="") -> list[str]:
    if e.frames <= 1:
        return []
    if e.anim == "life":
        return [f"{prefix}anim_speed_min = 1.0", f"{prefix}anim_speed_max = 1.0"]
    cycles = e.anim_fps * e.lifetime / e.frames
    return [f"{prefix}anim_speed_min = {_f(cycles * 0.8)}", f"{prefix}anim_speed_max = {_f(cycles * 1.2)}",
            f"{prefix}anim_offset_max = 1.0"]


def _process_lines(pid, e, dirv, vel_scale, ext, centre_note, dims3=False, grav=(0.0, 0.0, 0.0)) -> list[str]:
    sp = _speed(e) * vel_scale
    lines = [f'[sub_resource type="ParticleProcessMaterial" id="{pid}"]']
    if not dims3:
        lines.append("particle_flag_disable_z = true")
    if e.align:
        lines.append("particle_flag_align_y = true")
    lines += ["emission_shape = 3", f"emission_box_extents = Vector3({_f(ext[0])}, {_f(ext[1])}, {_f(ext[2])})",
              f"direction = Vector3({_f(dirv[0])}, {_f(dirv[1])}, {_f(dirv[2])})",
              f"spread = {_f(e.spread)}",
              f"initial_velocity_min = {_f(sp * (1 - e.speed_var))}",
              f"initial_velocity_max = {_f(sp * (1 + e.speed_var))}",
              f"gravity = Vector3({_f(grav[0])}, {_f(grav[1])}, {_f(grav[2])})",
              f"scale_min = {_f(e.scale[0])}", f"scale_max = {_f(e.scale[1])}",
              f'color_ramp = SubResource("{pid}_fade_tex")']
    if e.spin:
        lines += ["angle_min = -180.0", "angle_max = 180.0",
                  f"angular_velocity_min = {_f(-e.spin)}", f"angular_velocity_max = {_f(e.spin)}"]
    if e.sway > 0:
        lines += ["turbulence_enabled = true", f"turbulence_noise_strength = {_f(1 + 4 * e.sway)}",
                  f"turbulence_noise_scale = {_f(3.0 if not dims3 else 6.0)}",
                  f"turbulence_influence_min = {_f(0.02 + 0.08 * e.sway)}",
                  f"turbulence_influence_max = {_f(0.05 + 0.15 * e.sway)}"]
    lines += _anim_lines(e)
    del centre_note
    return lines + [""]


def _area_2d(e, sw, sh):
    """(centre, extents, lifetime) of a 2D emitter covering a sw x sh view."""
    vx, vy = e.velocity
    life = e.lifetime
    if e.area == "sky":
        life = max(life, (sh + 40) / max(abs(vy), 1e-3))
        drift = vx * life
        return (sw / 2 - drift / 2, -12), (sw / 2 + abs(drift) / 2 + 16, 6), life
    if e.area == "below":
        life = max(life, (sh + 40) / max(abs(vy), 1e-3)) if abs(vy) > 1 else life
        return (sw / 2, sh + 12), (sw / 2 + 16, 6), life
    if e.area == "side":
        life = max(life, (sw + 40) / max(abs(vx), 1e-3))
        return (-12 if vx >= 0 else sw + 12, sh / 2), (6, sh / 2 + 16), life
    return (sw / 2, sh / 2), (sw / 2 + 8, sh / 2 + 8), life


def _area_3d(e, size, ppu):
    """(centre, extents, lifetime) in metres for a 3D emitter over a size (x, y, z) box."""
    X, Y, Z = size
    vx, vy = e.velocity[0] / ppu, e.velocity[1] / ppu
    life = e.lifetime
    if e.area == "sky":
        life = max(life, (Y + 1) / max(abs(vy), 1e-3))
        drift = vx * life
        return (-drift / 2, Y, 0), (X / 2 + abs(drift) / 2, 0.2, Z / 2), life
    if e.area == "below":
        return (0, 0.1, 0), (X / 2, 0.1, Z / 2), max(life, Y / max(abs(vy), 1e-3) * 0.5)
    if e.area == "ground":
        return (0, 0.05, 0), (X / 2, 0.02, Z / 2), life
    if e.area == "side":
        life = max(life, (X + 2) / max(abs(vx), 1e-3))
        return (-X / 2 - 1 if vx >= 0 else X / 2 + 1, Y * 0.3, 0), (0.3, Y * 0.3, Z / 2), life
    return (0, Y * 0.35, 0), (X / 2, Y * 0.35, Z / 2), life


def _ext(lines, path, rid, typ="Texture2D"):
    lines.append(f'[ext_resource type="{typ}" path="{path}" id="{rid}"]')


def write_overlay_scene(w, folder: Path, name: str) -> list[Path]:
    fx = w.fx
    n = len(fx.frames)
    fw, fh = fx.size
    lines = ["[gd_scene format=3]", ""]
    written = []
    root = name.capitalize() + "Weather"
    if w.kind == "lightning":
        _ext(lines, f"{name}.tres", "1_bolt", "SpriteFrames")
        _ext(lines, "weather_fx.gd", "2_gd", "Script")
        flash = ", ".join(_f(v) for v in w.meta["flash"])
        ms = ", ".join(str(int(v)) for v in fx.frame_durations())
        lines += ["", f'[node name="{root}" type="CanvasLayer"]', "layer = 5", "",
                  '[node name="Fx" type="Node" parent="."]', 'script = ExtResource("2_gd")',
                  'mode = "lightning2d"', f"flash = PackedFloat32Array({flash})",
                  f"flash_ms = PackedInt32Array({ms})", "",
                  '[node name="Flash" type="ColorRect" parent="Fx"]', "anchors_preset = 15",
                  "anchor_right = 1.0", "anchor_bottom = 1.0", "mouse_filter = 2",
                  "color = Color(0.82, 0.88, 1, 0)", "",
                  '[node name="Bolt" type="AnimatedSprite2D" parent="Fx"]', "visible = false",
                  "texture_filter = 1", "scale = Vector2(3, 3)", f"position = Vector2(480, {fh * 3 // 2})",
                  'sprite_frames = ExtResource("1_bolt")', 'animation = &"strike"', ""]
    else:
        add = fx.blend == "add"
        shader = "weather_overlay_add.gdshader" if add else "weather_overlay.gdshader"
        _ext(lines, f"{name}.png", "1_sheet")
        _ext(lines, shader, "2_shader", "Shader")
        _ext(lines, "weather_fx.gd", "3_gd", "Script")
        glow = fx.glow is not None
        if glow:
            _ext(lines, f"{name}_glow.png", "4_glow")
        ms = fx.frame_durations()[0]
        scroll = w.meta.get("scroll", [0, 0])
        lines += ["", '[sub_resource type="ShaderMaterial" id="mat"]', 'shader = ExtResource("2_shader")',
                  'shader_parameter/sheet = ExtResource("1_sheet")',
                  *(['shader_parameter/glow_sheet = ExtResource("4_glow")', "shader_parameter/glow_strength = 0.6"]
                    if glow else []),
                  f"shader_parameter/frame_size = Vector2({fw}, {fh})", f"shader_parameter/frames = {n}",
                  f"shader_parameter/columns = {n}", f"shader_parameter/fps = {_f(1000 / ms)}",
                  "shader_parameter/pixel_scale = 4.0",
                  f"shader_parameter/scroll = Vector2({_f(scroll[0])}, {_f(scroll[1])})",
                  f"shader_parameter/wrap_mode = {1 if w.meta.get('wrap') == 'x' else 0}",
                  "shader_parameter/opacity = 1.0", "",
                  f'[node name="{root}" type="CanvasLayer"]', "layer = 5", "",
                  '[node name="Fx" type="Node" parent="."]', 'script = ExtResource("3_gd")',
                  'mode = "overlay"', "", '[node name="Overlay" type="ColorRect" parent="Fx"]',
                  'material = SubResource("mat")', "anchors_preset = 15", "anchor_right = 1.0",
                  "anchor_bottom = 1.0", "mouse_filter = 2", ""]
    p = folder / f"{name}_overlay.tscn"
    p.write_text("\n".join(lines), encoding="utf-8")
    written.append(p)
    return written


def write_particles2d(w, folder: Path, name: str) -> Path:
    sw, sh = w.meta.get("screen", [480, 270])
    lines = ["[gd_scene format=3]", ""]
    subs, nodes = [], []
    root = name.capitalize() + "Particles2D"
    if w.kind in ("lightning", "cloud_shadows"):
        p = folder / f"{name}_particles2d.tscn"          # these are screen effects in 2D
        text = (folder / f"{name}_overlay.tscn").read_text(encoding="utf-8")
        p.write_text(text.replace('type="CanvasLayer"]\nlayer = 5', 'type="CanvasLayer"]\nlayer = 5\nfollow_viewport_enabled = true'),
                     encoding="utf-8")
        return p
    tex_ids = {}
    for i, e in enumerate(w.emitters):
        if e.texture not in tex_ids:
            tex_ids[e.texture] = f"{len(tex_ids) + 1}_{e.texture}"
            _ext(lines, f"particles/{e.texture}.png", tex_ids[e.texture])
    lines.append("")
    for i, e in enumerate(w.emitters):
        pid, cid = f"pm_{e.name}", f"cm_{e.name}"
        centre, ext, life = _area_2d(e, sw, sh)
        sp = _speed(e) or 1.0
        dirv = (e.velocity[0] / sp, e.velocity[1] / sp, 0.0) if _speed(e) else (0.0, -1.0, 0.0)
        subs += _gradient(f"{pid}_fade", e)
        subs += _process_lines(pid, e, dirv, 1.0, (ext[0], ext[1], 1.0), None, grav=(e.gravity[0], e.gravity[1], 0))
        subs += [f'[sub_resource type="CanvasItemMaterial" id="{cid}"]',
                 *(["blend_mode = 1"] if e.blend == "add" else [])]
        if e.frames > 1:
            subs += ["particles_animation = true", f"particles_anim_h_frames = {e.frames}",
                     "particles_anim_v_frames = 1", f"particles_anim_loop = {'true' if e.anim == 'loop' else 'false'}"]
        subs.append("")
        amount = max(1, int(e.amount))
        nodes += ["", f'[node name="{e.name.capitalize()}" type="GPUParticles2D" parent="."]',
                  f"z_index = {int(round(e.depth * 10)) - 5}",
                  f'material = SubResource("{cid}")', f"position = Vector2({_f(centre[0])}, {_f(centre[1])})",
                  f"amount = {amount}", f'texture = ExtResource("{tex_ids[e.texture]}")',
                  f"lifetime = {_f(life)}", f"preprocess = {_f(life)}",
                  f"visibility_rect = Rect2({_f(-centre[0] - 64)}, {_f(-centre[1] - 64)}, {sw + 128}, {sh + 128})",
                  f'process_material = SubResource("{pid}")']
    lines += subs
    lines += [f'[node name="{root}" type="Node2D"]', "texture_filter = 1"] + nodes + [""]
    p = folder / f"{name}_particles2d.tscn"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def write_particles3d(w, folder: Path, name: str, ppu: float = 42.0, size=(16.0, 8.0, 12.0)) -> Path:
    lines = ["[gd_scene format=3]", ""]
    root = name.capitalize() + "Particles3D"
    p = folder / f"{name}_particles3d.tscn"
    if w.kind == "cloud_shadows":
        tw, th = w.size
        period = tw / ppu * 2.5                      # a 192 px tile covers ~11 m
        _ext(lines, "particles/cloud_shadow_2x2.png", "1_tex")
        _ext(lines, "weather_fx.gd", "2_gd", "Script")
        sc = w.meta.get("scroll", [10, 0])
        lines += ["", f'[node name="{root}" type="Node3D"]', 'script = ExtResource("2_gd")', 'mode = "clouds3d"',
                  f"period = {_f(period)}", f"drift = Vector2({_f(sc[0] / ppu * 2.5)}, {_f(sc[1] / ppu * 2.5)})", "",
                  '[node name="Clouds" type="Decal" parent="."]',
                  f"position = Vector3({_f(-period / 2)}, 0, {_f(-period / 2)})",
                  f"size = Vector3({_f(period * 2)}, 20, {_f(period * 2)})",
                  'texture_albedo = ExtResource("1_tex")', "albedo_mix = 1.0", "upper_fade = 0.05",
                  "lower_fade = 0.05", "cull_mask = 1", ""]
        p.write_text("\n".join(lines), encoding="utf-8")
        return p
    if w.kind == "lightning":
        n = len(w.fx.frames)
        _ext(lines, f"{name}.png", "1_sheet")
        _ext(lines, "weather_fx.gd", "2_gd", "Script")
        flash = ", ".join(_f(v) for v in w.meta["flash"])
        ms = ", ".join(str(int(v)) for v in w.fx.frame_durations())
        fw, fh = w.size
        lines += ["", f'[node name="{root}" type="Node3D"]', 'script = ExtResource("2_gd")', 'mode = "lightning3d"',
                  f"flash = PackedFloat32Array({flash})", f"flash_ms = PackedInt32Array({ms})", "",
                  '[node name="Bolt" type="Sprite3D" parent="."]', "visible = false",
                  f"position = Vector3(0, {_f(fh / ppu * 2.5 / 2)}, 0)", f"pixel_size = {_f(2.5 / ppu)}",
                  "billboard = 2", "shaded = false", "texture_filter = 0", 'texture = ExtResource("1_sheet")',
                  f"hframes = {n}", "", '[node name="Flash" type="DirectionalLight3D" parent="."]',
                  "visible = false", "transform = Transform3D(1, 0, 0, 0, 0.5, 0.866, 0, -0.866, 0.5, 0, 10, 0)",
                  "light_color = Color(0.75, 0.82, 1, 1)", "light_energy = 0.0", ""]
        p.write_text("\n".join(lines), encoding="utf-8")
        return p
    tex_ids = {}
    for e in w.emitters:
        if e.texture not in tex_ids:
            tex_ids[e.texture] = f"{len(tex_ids) + 1}_{e.texture}"
            _ext(lines, f"particles/{e.texture}.png", tex_ids[e.texture])
            if e.texture in w.particle_glow:
                _ext(lines, f"particles/{e.texture}_glow.png", tex_ids[e.texture] + "_g")
    lines.append("")
    subs, nodes = [], []
    for e in w.emitters:
        pid, mid, qid = f"pm_{e.name}", f"mat_{e.name}", f"quad_{e.name}"
        centre, ext, life = _area_3d(e, size, ppu)
        sp = _speed(e)
        dirv = (e.velocity[0] / sp, -e.velocity[1] / sp, 0.0) if sp else (0.0, 1.0, 0.0)
        subs += _gradient(f"{pid}_fade", e)
        subs += _process_lines(pid, e, dirv, 1.0 / ppu, ext, None, dims3=True,
                               grav=(e.gravity[0] / ppu, -e.gravity[1] / ppu, 0.0))
        pw, ph = w.particles[e.texture][0].shape[1], w.particles[e.texture][0].shape[0]
        subs += [f'[sub_resource type="StandardMaterial3D" id="{mid}"]', "transparency = 1",
                 "shading_mode = 0", "vertex_color_use_as_albedo = true",
                 f'albedo_texture = ExtResource("{tex_ids[e.texture]}")', "texture_filter = 0",
                 f"billboard_mode = {2 if e.align else 3}", "billboard_keep_scale = true",
                 *(["blend_mode = 1"] if e.blend == "add" else [])]
        if e.frames > 1:
            subs += [f"particles_anim_h_frames = {e.frames}", "particles_anim_v_frames = 1",
                     f"particles_anim_loop = {'true' if e.anim == 'loop' else 'false'}"]
        if e.texture in w.particle_glow:
            subs += ["emission_enabled = true", f'emission_texture = ExtResource("{tex_ids[e.texture]}_g")',
                     "emission_energy_multiplier = 2.5"]
        subs += ["", f'[sub_resource type="QuadMesh" id="{qid}"]', f'material = SubResource("{mid}")',
                 f"size = Vector2({_f(pw / ppu * (0.7 if e.align else 1.5))}, {_f(ph / ppu * 1.5)})", ""]
        X, Y, Z = size
        nodes += ["", f'[node name="{e.name.capitalize()}" type="GPUParticles3D" parent="."]',
                  f"transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, {_f(centre[0])}, {_f(centre[1])}, {_f(centre[2])})",
                  f"amount = {max(1, int(e.amount * 2))}", f"lifetime = {_f(life)}", f"preprocess = {_f(life)}",
                  f"visibility_aabb = AABB({_f(-X)}, {_f(-Y - 2)}, {_f(-Z)}, {_f(2 * X)}, {_f(2 * Y + 4)}, {_f(2 * Z)})",
                  f'process_material = SubResource("{pid}")', f'draw_pass_1 = SubResource("{qid}")']
    lines += subs
    lines += [f'[node name="{root}" type="Node3D"]'] + nodes + [""]
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def write_godot_weather(w, folder, name: str = "sheet", ppu: float = 42.0) -> list[Path]:
    """The shared shaders + script and the three scenes, next to sheet.png."""
    import numpy as np
    from PIL import Image
    folder = Path(folder)
    out = []
    for fname, blend in (("weather_overlay.gdshader", ""), ("weather_overlay_add.gdshader", ", blend_add")):
        p = folder / fname
        p.write_text(OVERLAY_SHADER.replace("{blend}", blend), encoding="utf-8")
        out.append(p)
    gd = folder / "weather_fx.gd"
    gd.write_text(FX_GD, encoding="utf-8")
    out.append(gd)
    if w.kind == "cloud_shadows":                    # the decal shows its texture once: 2x2 copies
        base = w.particles["cloud_shadow"][0]
        p = folder / "particles" / "cloud_shadow_2x2.png"
        p.parent.mkdir(exist_ok=True)
        Image.fromarray(np.tile(base, (2, 2, 1)), "RGBA").save(p)
        out.append(p)
    out += write_overlay_scene(w, folder, name)
    out.append(write_particles2d(w, folder, name))
    out.append(write_particles3d(w, folder, name, ppu))
    return out

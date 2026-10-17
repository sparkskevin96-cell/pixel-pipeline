"""Autoplay for the HD-2D board: a deterministic timeline that plays the game by itself.

    make_hd2d(out, units, d, autoplay={"mode": "tactics", "steps": [...]})
    godot --path out res://real3d.tscn -- --autoplay              # watch it
    godot --path out --write-movie f.avi --fixed-fps 30 res://real3d.tscn -- --autoplay --autoplay-quit

make_hd2d writes the script to res://autoplay.json and res://autoplay.gd (always, it's small);
board.gd starts it when the game runs with `--autoplay[=res://other.json]`, or on its own when the
json has "auto": true. While it runs the board takes no input (`scripted`), turns pass only when
the script says so and every act ends with board.action_done.

Two modes:
  tactics  the HD-2D battle: select -> purple move range -> walk the grid path -> attack / cast
           (red target cell, attack/hurt/death clips, damage numbers, HP bars), turn banners,
           a unit panel from the project's UI kit.
  action   real time, no grid: units walk along waypoints (8-way facing), swing (slash effect,
           knockback, hurt flash), get bitten, die (death clip + poof), cut bushes, drop and pick
           up hearts/coins/gems; sprites lean toward a steep camera; a top-left HUD (hearts, coin
           counter, item box) drawn with the UI kit pieces.

Script keys: mode, seed, light (a preset name), presets ({name: lighting dict} added to
lighting.json's), bands, camera ({yaw, pitch, dist, follow, at, rate, orbit, dof}), hero (action:
whose hearts the HUD shows), names ({unit id: shown name}), lean (action sprites, 0-1), hud_scale,
tail (seconds kept after the last step with --autoplay-quit), steps.

Points are in CELL coordinates: [c, r] = the centre of that cell; fractions move inside it (the
hero walks [3.5, 6] -> world x 4.0, z 6.5). Units are named by id: the sheet name, the 2nd unit
of the same name "<name>#2" (board.gd `_unit_id`).

Steps ({"do": ...}; "when": seconds waits until then; "async": true runs it alongside the next ones):
  wait {time}                      camera {yaw?, pitch?, dist?, time, follow?, at?, rate?, orbit?, dof?, wait?}
  banner {text, colour?, time?}    fade {to: 0-1, time}            light {preset}
  select {unit, time?}             move {unit, to: [c, r]}         attack {unit, target, damage?, effect?, shake?}
  cast {unit, target | at, effect?, tag?, projectile?, projectile_tag?, charge?, colour?, damage? | heal?,
        windup?, after?, shake?, scale?, arc?, speed?}
  end_turn {text?}  done {unit} (waits: greyed)  face {unit, dir | toward}  anim {unit, anim, once?}
  goto {unit, path: [[c, r], ...], speed?, anim?, follow?}          (action)
  swing {unit, dir? | toward?, hits: [ids], damage?, knock?, effect?, cut: [[c, r]], drops: {id: pickup}}
  hurt {unit, by?, damage, knock?}  spawn {id, kind: heart|coin|gem, at}  pickup {id, unit?, value?}
  cut {cells: [[c, r]], drop?}     effect {name, at, tag?, scale?, loop?}  kill {unit}   quit
Every step that names a missing unit is skipped with a warning (the run still finishes).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

MODES = ("tactics", "action")
STEPS = ("wait", "camera", "banner", "fade", "light", "select", "move", "attack", "cast", "end_turn",
         "face", "anim", "done", "goto", "swing", "hurt", "spawn", "pickup", "cut", "effect", "kill", "quit")
UNIT_STEPS = {"select", "move", "attack", "cast", "face", "anim", "done", "goto", "swing", "hurt", "kill"}


def validate(script: dict, unit_ids=None) -> list[str]:
    """Problems with an autoplay script (empty = fine): unknown mode/steps, missing fields,
    units the map doesn't have (unit_ids: the board's ids, see unit_ids())."""
    errs = []
    if script.get("mode", "tactics") not in MODES:
        errs.append(f"mode must be one of {', '.join(MODES)}")
    ids = set(unit_ids) if unit_ids is not None else None
    for i, s in enumerate(script.get("steps", [])):
        what = s.get("do")
        if what not in STEPS:
            errs.append(f"step {i}: unknown 'do': {what!r}")
            continue
        if what in UNIT_STEPS and "unit" not in s:
            errs.append(f"step {i} ({what}): needs 'unit'")
        if what == "move" and "to" not in s:
            errs.append(f"step {i} (move): needs 'to'")
        if what == "goto" and not s.get("path"):
            errs.append(f"step {i} (goto): needs 'path'")
        if what in ("attack",) and "target" not in s:
            errs.append(f"step {i} (attack): needs 'target'")
        if what == "cast" and "target" not in s and "at" not in s:
            errs.append(f"step {i} (cast): needs 'target' or 'at'")
        if ids is not None:
            for key in ("unit", "target", "by"):
                if key in s and s[key] not in ids:
                    errs.append(f"step {i} ({what}): no unit '{s[key]}' (have {', '.join(sorted(ids))})")
            for h in s.get("hits", []):
                if h not in ids:
                    errs.append(f"step {i} ({what}): no unit '{h}'")
    return errs


def unit_ids(names) -> list[str]:
    """The ids board.gd gives units named `names` (in map order): name, name#2, name#3..."""
    seen, out = {}, []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n}#{seen[n]}")
    return out


def install(project, script, ui_kit=None) -> list[Path]:
    """res://autoplay.json (+ res://autoplay.gd) and, given a UI kit folder (uikit.py: kit.json,
    pieces/, fonts/), its pieces in res://ui/ with res://ui/ui.json (nine-slice margins, colours)."""
    project = Path(project)
    if isinstance(script, (str, Path)):
        script = json.loads(Path(script).read_text(encoding="utf-8"))
    written = []
    if script is not None:
        p = project / "autoplay.json"
        p.write_text(json.dumps(script, indent=1), encoding="utf-8")
        written.append(p)
    gd = project / "autoplay.gd"
    gd.write_text(AUTOPLAY_GD, encoding="utf-8")
    written.append(gd)
    if ui_kit is not None:
        written += install_ui(project, ui_kit)
    return written


def install_ui(project, kit_folder) -> list[Path]:
    kit_folder = Path(kit_folder)
    kit = json.loads((kit_folder / "kit.json").read_text(encoding="utf-8"))
    dst = Path(project) / "ui"
    dst.mkdir(parents=True, exist_ok=True)
    out, pieces = [], {}
    for piece in kit.get("pieces", []):
        src = kit_folder / piece["file"]
        if src.exists():
            shutil.copyfile(src, dst / src.name)
            out.append(dst / src.name)
            pieces[piece["name"]] = {"file": src.name, "w": piece["w"], "h": piece["h"],
                                     "margins": piece.get("margins", {})}
    font = None
    for f in sorted((kit_folder / "fonts").glob("*_1x.fnt")) if (kit_folder / "fonts").exists() else []:
        for extra in (f, f.with_suffix(".png")):
            if extra.exists():
                shutil.copyfile(extra, dst / extra.name)
                out.append(dst / extra.name)
        font = f.name
    info = {"pieces": pieces, "colours": kit.get("colours", {}), "font": font, "theme": kit.get("theme")}
    j = dst / "ui.json"
    j.write_text(json.dumps(info, indent=1), encoding="utf-8")
    return out + [j]


AUTOPLAY_GD = r'''extends Node
# Autoplay: a deterministic timeline that plays the HD-2D board by itself (trailers, tests,
# demos). Written by pixel_pipeline.hd2d_autoplay (its docstring lists every step).
# Started by board.gd with --autoplay; --autoplay-quit quits when the timeline is over,
# --autoplay-log prints each step.

var board: Node3D
var data := {}
var mode := "tactics"
var clock := 0.0
var finished := false
var quit_when_done := false
var log_steps := false
var tail := 1.0
var all := {}                  # id -> unit dict (kept after a unit dies)
var follow = null              # a unit, [units], a Vector3 or null: where the camera looks
var follow_rate := 2.2
var follow_offset := Vector3.ZERO
var orbit := 0.0               # camera turn, degrees per second
var shake := 0.0
var rng := RandomNumberGenerator.new()
var ui: CanvasLayer
var ui_info := {}
var ui_font: Font = null
var hud_scale := 4.0
var fade_rect: ColorRect
var panel: Control = null
var panel_parts := {}
var turn_plate: Control = null
var turn := 1
var hero = null
var hearts_box: HBoxContainer = null
var coin_label: Label = null
var coins := 0
var coins_shown := 0.0
var pickups := {}
var tex_cache := {}
var steps_done := 0
const M := preload("res://pp_runtime/pp_motion.gd")
const TEAM_COLOURS := [Color(0.22, 0.42, 0.95), Color(0.85, 0.2, 0.18)]

func start(b: Node3D, path: String) -> void:
	board = b
	var j = JSON.parse_string(FileAccess.get_file_as_string(path))
	if not (j is Dictionary):
		push_error("autoplay: can't read " + path)
		return
	data = j
	mode = str(data.get("mode", "tactics"))
	rng.seed = int(data.get("seed", 7))
	seed(int(data.get("seed", 7)))
	for a in OS.get_cmdline_user_args():
		if a == "--autoplay-quit":
			quit_when_done = true
		elif a == "--autoplay-log":
			log_steps = true
	tail = float(data.get("tail", 1.0))
	hud_scale = float(data.get("hud_scale", 4.0 if mode == "action" else 3.0))
	board.scripted = true
	board.hud.visible = false
	board.cursor.visible = false
	for u in board.units:
		all[u["id"]] = u
		u["t"] = rng.randf() * 2.0                   # idle phases: varied, the same every run
	var presets: Dictionary = data.get("presets", {})
	for k in presets:
		board.lighting[k] = presets[k]
	if data.has("bands"):
		board.bands = float(data["bands"])
	if data.has("light") or data.has("bands"):
		board.apply_lighting(str(data.get("light", board.light_name)))
	_setup_ui()
	if mode == "action":
		_setup_action()
	else:
		_setup_tactics()
	_camera_now(data.get("camera", {}))
	_run()

# ------------------------------------------------------------------ timeline

func _run() -> void:
	await get_tree().process_frame
	for s in data.get("steps", []):
		if not (s is Dictionary):
			continue
		if s.has("when"):                          # wait for that time on the clock
			while clock < float(s["when"]):
				await get_tree().process_frame
		if log_steps:
			print("autoplay %6.2f  %-9s camera dist %.1f pitch %.1f yaw %.1f" % [clock, str(s.get("do", "")), board.cam_dist, board.cam_pitch, board.cam_yaw])
		if bool(s.get("async", false)):
			_do(s)
		else:
			await _do(s)
		steps_done += 1
	finished = true
	var hp := {}
	for id in all:
		hp[id] = int(all[id]["hp"]) if board.units.has(all[id]) else 0
	print("AUTOPLAY UNITS " + JSON.stringify(hp, "", true))
	print("AUTOPLAY COINS %d PICKUPS %d" % [coins, pickups.size()])
	print("AUTOPLAY DONE steps=%d time=%.2f" % [steps_done, clock])
	if quit_when_done:
		await _sleep(tail)
		get_tree().quit()

func _sleep(t: float) -> void:
	if t > 0.0:
		await get_tree().create_timer(t, false).timeout

func _do(s: Dictionary) -> void:
	var what := str(s.get("do", ""))
	for key in ["unit", "target", "by"]:
		if s.has(key) and _u(s[key]) == null and not (what == "cast" and key == "target"):
			push_warning("autoplay: no unit '%s' (step %s)" % [str(s[key]), what])
			return
	match what:
		"wait": await _sleep(float(s.get("time", 1.0)))
		"camera": await _camera(s)
		"banner": await _banner(s)
		"fade": await _fade(s)
		"light": board.apply_lighting(str(s.get("preset", board.light_name)))
		"select": await _select(s)
		"move": await _move(s)
		"attack": await _attack(s)
		"cast": await _cast(s)
		"end_turn": await _end_turn(s)
		"face": _face(_u(s["unit"]), s)
		"anim":
			var u = _u(s["unit"])
			board._play(u, str(s.get("anim", "idle")), bool(s.get("once", false)))
			await _sleep(float(s.get("time", 0.0)))
		"done":
			board._finish(_u(s["unit"]))
			_hide_panel()
			await _sleep(float(s.get("time", 0.3)))
		"goto": await _goto(s)
		"swing": await _swing(s)
		"hurt": await _hurt(s)
		"spawn": _spawn(s)
		"pickup": await _pickup(s)
		"cut": _cut(s.get("cells", []), s.get("drop", {}))
		"effect":
			board.play_effect(str(s["name"]), _point(s.get("at", [0, 0])) + Vector3(0, float(s.get("lift", 0.0)), 0),
				bool(s.get("loop", false)), float(s.get("energy", 2.0)), str(s.get("tag", "")), float(s.get("scale", 1.0)))
			await _sleep(float(s.get("time", 0.0)))
		"kill":
			await _die(_u(s["unit"]), s.get("drop", {}))
		"quit": get_tree().quit()
		_: push_warning("autoplay: unknown step '%s'" % what)

# ------------------------------------------------------------------ helpers

func _u(id) -> Variant:
	if id is Dictionary:
		return id
	var key := str(id)
	if all.has(key):
		return all[key]
	return null

func _alive(u) -> bool:
	return u != null and is_instance_valid(u["node"]) and board.units.has(u)

func _pos(u) -> Vector3:
	return u["node"].position

func _point(v) -> Vector3:
	# cell coordinates (c, r = that cell's centre; fractions inside it) -> world, on the ground
	if v is Vector3:
		return v
	if v is String or v is Dictionary:
		var u = _u(v)
		return _pos(u) if u != null else Vector3.ZERO
	var x := float(v[0]) + 0.5
	var z := float(v[1]) + 0.5
	return Vector3(x, _ground(x, z), z)

func _ground(x: float, z: float) -> float:
	var c := clampi(int(floor(x)), 0, board.cols - 1)
	var r := clampi(int(floor(z)), 0, board.rows - 1)
	return board.top_y(c, r)

func _col(v, fallback := Color(1, 1, 1)) -> Color:
	if v is Array and v.size() >= 3:
		return Color(float(v[0]), float(v[1]), float(v[2]))
	if v is String and v != "":
		return Color.from_string(v, fallback)
	return fallback

func _snap8(a: float) -> float:
	return round(a / (PI / 4.0)) * (PI / 4.0)

func _yaw_to(from: Vector3, to: Vector3) -> float:
	var a := atan2(to.z - from.z, to.x - from.x)
	return _snap8(a) if mode == "action" else a

func _face(u, s: Dictionary) -> void:
	# turns through the facings in between (board.face_to; "snap": true turns at once)
	var snap := bool(s.get("snap", false))
	if s.has("dir"):
		board.face_to(u, board._dir_yaw(str(s["dir"])), snap)
	elif s.has("toward"):
		board.face_to(u, _yaw_to(_pos(u), _point(s["toward"])), snap)

func _cell_of(u) -> Vector2i:
	return Vector2i(int(u["col"]), int(u["row"]))

func _sync_cell(u) -> void:
	var p: Vector3 = _pos(u)
	board.blocked.erase(_cell_of(u))
	u["col"] = clampi(int(floor(p.x)), 0, board.cols - 1)
	u["row"] = clampi(int(floor(p.z)), 0, board.rows - 1)

func _process(delta: float) -> void:
	clock += delta
	var target = _follow_point()
	# the board's critically damped follow (no lag jumps, no overshoot); rate -> smoothing time
	board.cam_smooth = 1.0 / maxf(follow_rate, 0.1)
	board.cam_target = target + follow_offset if target != null else null
	if orbit != 0.0:
		board.cam_yaw += orbit * delta
		board.cam_yaw_target = board.cam_yaw
	if shake > 0.001:                              # a script's shake: trauma on the board's camera
		board.add_trauma(sqrt(shake / 0.32))
		shake = 0.0
	if mode == "action":
		var c: Dictionary = board.conf
		for u in board.units:                      # free movement: feet follow the ground
			if u["mover"] != null:
				continue
			var p: Vector3 = u["node"].position
			var gy: float = board.ground_at(p.x, p.z, float(c.get("foot_radius", 0.22)))
			var f := M.ground_follow(p.y, float(u["vy"]), gy, float(c.get("rise", 0.07)), float(c.get("gravity", 22.0)), delta)
			p.y = f[0]
			u["vy"] = f[1]
			u["node"].position = p
		for id in pickups:
			var pk: Dictionary = pickups[id]
			if pk["bob"]:
				pk["node"].position.y = pk["y"] + 0.07 + 0.06 * sin(clock * 4.0 + pk["phase"])
		if coin_label != null and int(coins_shown) != coins:
			coins_shown = move_toward(coins_shown, float(coins), delta * 18.0)
			coin_label.text = "%03d" % int(coins_shown)

func _follow_point() -> Variant:
	if follow is Dictionary:
		if is_instance_valid(follow["node"]):
			return follow["node"].position + Vector3(0, 0.5, 0)
		return null
	if follow is Array:
		var sum := Vector3.ZERO
		var n := 0
		for u in follow:
			if u is Dictionary and is_instance_valid(u["node"]):
				sum += u["node"].position
				n += 1
		return sum / float(n) + Vector3(0, 0.5, 0) if n > 0 else null
	if follow is Vector3:
		return follow
	return null

func _target(v) -> Variant:
	# a camera follow target from the script: a unit id, [ids], [c, r], "none"
	if v is String:
		if v == "none":
			return null
		return _u(v)
	if v is Array and v.size() > 0 and v[0] is String:
		var list := []
		for id in v:
			if _u(id) != null:
				list.append(_u(id))
		return list
	if v is Array:
		return _point(v) + Vector3(0, 0.5, 0)
	return null

# ------------------------------------------------------------------ camera

func _camera_now(c: Dictionary) -> void:
	if c.has("yaw"):
		board.cam_yaw = float(c["yaw"])
		board.cam_yaw_target = board.cam_yaw
	if c.has("pitch"):
		board.cam_pitch = float(c["pitch"])
	if c.has("dist"):
		board.cam_dist = float(c["dist"])
	if c.has("rate"):
		follow_rate = float(c["rate"])
	if c.has("orbit"):
		orbit = float(c["orbit"])
	if c.has("turn_rate"):
		board.cam_turn_rate = float(c["turn_rate"])
	if c.has("dof") and board.cam.attributes is CameraAttributesPractical:
		board.cam.attributes.dof_blur_amount = float(c["dof"])
	if c.has("follow"):
		follow = _target(c["follow"])
	elif c.has("at"):
		follow = _point(c["at"]) + Vector3(0, 0.5, 0)
	var p = _follow_point()
	if p != null:
		board.pivot.position = p + follow_offset
		board.cam_target = board.pivot.position
		board.cam_vel = Vector3.ZERO
	board._place_camera()

func _camera(s: Dictionary) -> void:
	var time := float(s.get("time", 1.5))
	if s.has("follow"):
		follow = _target(s["follow"])
	elif s.has("at"):
		follow = _point(s["at"]) + Vector3(0, 0.5, 0)
	if s.has("rate"):
		follow_rate = float(s["rate"])
	if s.has("orbit"):
		orbit = float(s["orbit"])
	if s.has("offset"):
		var o: Array = s["offset"]
		follow_offset = Vector3(float(o[0]), float(o[1]), float(o[2]))
	if s.has("shake"):
		shake = float(s["shake"])
	var tw := create_tween().set_parallel(true).set_trans(Tween.TRANS_SINE).set_ease(Tween.EASE_IN_OUT)
	var any := false
	if s.has("yaw"):
		tw.tween_property(board, "cam_yaw", float(s["yaw"]), time)
		tw.tween_property(board, "cam_yaw_target", float(s["yaw"]), time)
		any = true
	if s.has("pitch"):
		if board.cam_pitch < 0.0:
			board.cam_pitch = board.default_pitch()
		tw.tween_property(board, "cam_pitch", float(s["pitch"]), time)
		any = true
	if s.has("dist"):
		tw.tween_property(board, "cam_dist", float(s["dist"]), time)
		any = true
	if s.has("dof") and board.cam.attributes is CameraAttributesPractical:
		tw.tween_property(board.cam.attributes, "dof_blur_amount", float(s["dof"]), time)
		any = true
	if not any:
		tw.kill()
		return
	if bool(s.get("wait", false)):
		await tw.finished

# ------------------------------------------------------------------ UI (the project's UI kit)

func _setup_ui() -> void:
	ui = CanvasLayer.new()
	ui.layer = 5
	add_child(ui)
	if FileAccess.file_exists("res://ui/ui.json"):
		var j = JSON.parse_string(FileAccess.get_file_as_string("res://ui/ui.json"))
		if j is Dictionary:
			ui_info = j
	var fname := str(ui_info.get("font", ""))
	if fname != "" and FileAccess.file_exists("res://ui/" + fname):
		var f := FontFile.new()
		if f.load_bitmap_font("res://ui/" + fname) == OK:
			f.fixed_size_scale_mode = TextServer.FIXED_SIZE_SCALE_INTEGER_ONLY   # crisp at any size
			ui_font = f
	fade_rect = ColorRect.new()
	fade_rect.color = Color(0, 0, 0, float(data.get("fade_in", 0.0)))
	fade_rect.set_anchors_preset(Control.PRESET_FULL_RECT)
	fade_rect.mouse_filter = Control.MOUSE_FILTER_IGNORE
	var top := CanvasLayer.new()
	top.layer = 9
	add_child(top)
	top.add_child(fade_rect)

func _ui_tex(name: String) -> Texture2D:
	if tex_cache.has(name):
		return tex_cache[name]
	var t: Texture2D = null
	var path := "res://ui/%s.png" % name
	if FileAccess.file_exists(path):
		var img := Image.load_from_file(ProjectSettings.globalize_path(path))
		if img != null:
			t = ImageTexture.create_from_image(img)
	tex_cache[name] = t
	return t

func _nine(name: String, size: Vector2) -> Control:
	# a nine-slice kit piece at kit pixels (the HUD root is scaled to whole pixels)
	var t := _ui_tex(name)
	if t == null:
		var r := ColorRect.new()
		r.color = Color(0.1, 0.08, 0.1, 0.8)
		r.size = size
		return r
	var n := NinePatchRect.new()
	n.texture = t
	var m: Dictionary = ui_info.get("pieces", {}).get(name, {}).get("margins", {})
	n.patch_margin_left = int(m.get("left", 4))
	n.patch_margin_top = int(m.get("top", 4))
	n.patch_margin_right = int(m.get("right", 4))
	n.patch_margin_bottom = int(m.get("bottom", 4))
	n.size = size
	n.texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
	return n

func _icon(name: String, pos: Vector2) -> TextureRect:
	var r := TextureRect.new()
	r.texture = _ui_tex(name)
	r.position = pos
	r.texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
	return r

func _label(text: String, pos: Vector2, colour := Color(1.0, 0.96, 0.86), size := 10) -> Label:
	var l := Label.new()
	l.text = text
	l.position = pos
	if ui_font != null:
		l.add_theme_font_override("font", ui_font)
		l.add_theme_font_size_override("font_size", size)
	else:
		l.add_theme_font_size_override("font_size", size)
	l.add_theme_color_override("font_color", colour)
	l.add_theme_color_override("font_shadow_color", Color(0.1, 0.05, 0.03, 0.9))
	l.add_theme_constant_override("shadow_offset_x", 1)
	l.add_theme_constant_override("shadow_offset_y", 1)
	l.texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
	return l

func _hud_root() -> Control:
	var root := Control.new()
	root.scale = Vector2(hud_scale, hud_scale)
	root.texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
	root.mouse_filter = Control.MOUSE_FILTER_IGNORE
	ui.add_child(root)
	return root

# ------------------------------------------------------------------ tactics

func _setup_tactics() -> void:
	var root := _hud_root()
	var W := 1920.0 / hud_scale
	var H := 1080.0 / hud_scale
	panel = Control.new()
	panel.position = Vector2(8, H - 52)
	root.add_child(panel)
	var box := _nine("panel", Vector2(132, 44))
	panel.add_child(box)
	var face := TextureRect.new()
	face.position = Vector2(6, 5)
	face.size = Vector2(34, 34)
	face.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	face.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	face.texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
	panel.add_child(face)
	var name_l := _label("", Vector2(44, 6), Color(1.0, 0.95, 0.8))
	panel.add_child(name_l)
	var hp_l := _label("", Vector2(44, 17), Color(0.95, 0.9, 0.8))
	panel.add_child(hp_l)
	var frame := _nine("bar_frame", Vector2(80, 9))
	frame.position = Vector2(44, 29)
	panel.add_child(frame)
	var fill := _nine("bar_health", Vector2(80, 9))
	fill.position = Vector2(44, 29)
	panel.add_child(fill)
	panel_parts = {"face": face, "name": name_l, "hp": hp_l, "fill": fill, "box": box}
	panel.modulate.a = 0.0
	turn_plate = Control.new()
	turn_plate.position = Vector2(8, 8)
	root.add_child(turn_plate)
	var plate := _nine("nameplate", Vector2(118, 18))
	turn_plate.add_child(plate)
	var tl := _label("", Vector2(6, 4))
	turn_plate.add_child(tl)
	panel_parts["turn"] = tl
	_update_turn()

func _update_turn() -> void:
	if panel_parts.has("turn"):
		panel_parts["turn"].text = "TURN %d   %s PHASE" % [turn, "PLAYER" if board.active_team == 0 else "ENEMY"]
		panel_parts["turn"].add_theme_color_override("font_color", Color(0.75, 0.85, 1.0) if board.active_team == 0 else Color(1.0, 0.7, 0.62))

func _portrait(u) -> Texture2D:
	var s: Dictionary = u["sheet"]
	var key := "idle_S" if s["frames"].has("idle_S") else str(s["frames"].keys()[0])
	var r: Rect2 = s["frames"][key][0]["rect"]
	var tex: Texture2D = s["tex"]
	var size := Vector2(tex.get_width(), tex.get_height())
	var px := Rect2(r.position * size, r.size * size)
	var img := tex.get_image()
	if img == null:
		return null
	var sub := img.get_region(Rect2i(px))
	var used := sub.get_used_rect()
	var a := AtlasTexture.new()
	a.atlas = tex
	var top := Rect2(px.position + Vector2(used.position), Vector2(used.size))
	top.size.y = min(top.size.y, max(top.size.x * 1.1, top.size.y * 0.6))   # head and shoulders
	a.region = top
	return a

func _show_panel(u) -> void:
	if panel == null or u == null:
		return
	if not u.has("portrait"):
		u["portrait"] = _portrait(u)
	panel_parts["face"].texture = u["portrait"]
	var names: Dictionary = data.get("names", {})
	panel_parts["name"].text = str(names.get(u["id"], str(u["name"]).capitalize()))
	panel_parts["name"].add_theme_color_override("font_color", Color(0.72, 0.84, 1.0) if u["team"] == 0 else Color(1.0, 0.68, 0.6))
	_update_panel(u)
	var tw := create_tween()
	tw.tween_property(panel, "modulate:a", 1.0, 0.2)

func _update_panel(u) -> void:
	if panel == null or u == null:
		return
	panel_parts["hp"].text = "HP %d/%d" % [int(u["hp"]), int(u["max_hp"])]
	var k := clampf(float(u["hp"]) / float(u["max_hp"]), 0.0, 1.0)
	panel_parts["fill"].size.x = max(6.0, 80.0 * k)
	panel_parts["fill"].visible = k > 0.0

func _hide_panel() -> void:
	if panel != null:
		create_tween().tween_property(panel, "modulate:a", 0.0, 0.25)

func _cursor(cell: Vector2i) -> void:
	board.cursor.visible = true
	board.cursor.position = board.cell_pos(cell.x, cell.y) + Vector3(0, 0.03, 0)

func _select(s: Dictionary) -> void:
	var u = _u(s["unit"])
	board.selected = u
	board.phase = "move"
	board._show_range()
	_cursor(_cell_of(u))
	_show_panel(u)
	if bool(s.get("follow", true)):
		follow = u
	await _sleep(float(s.get("time", 0.9)))

func _bfs(u, goal: Vector2i) -> Array:
	# a path past the unit's move range (scripts may walk further than one turn)
	var start := _cell_of(u)
	var prev := {start: start}
	var frontier := [start]
	while frontier.size() > 0 and not prev.has(goal):
		var nxt := []
		for p in frontier:
			for d in [Vector2i(1, 0), Vector2i(-1, 0), Vector2i(0, 1), Vector2i(0, -1)]:
				var n: Vector2i = p + d
				if prev.has(n) or not board.walkable(n.x, n.y):
					continue
				var dh: int = int(board.heights[n.y][n.x]) - int(board.heights[p.y][p.x])
				if dh > int(u["jump"]) or -dh > 2 * int(u["jump"]):
					continue
				prev[n] = p
				nxt.append(n)
		frontier = nxt
	if not prev.has(goal):
		return [start, goal]
	var path := [goal]
	while path[0] != start:
		path.push_front(prev[path[0]])
	return path

func _move(s: Dictionary) -> void:
	var u = _u(s["unit"])
	var to: Array = s["to"]
	var cell := Vector2i(int(to[0]), int(to[1]))
	if board.selected != u or board.reach.is_empty():
		board.selected = u
		board._show_range()
	var path: Array = board._path_to(cell) if board.reach.has(cell) else _bfs(u, cell)
	_cursor(cell)
	await _sleep(float(s.get("aim", 0.45)))
	board.cursor.visible = false
	follow = u
	if path.size() < 2:
		board._clear_range()
		return
	board._walk(u, path)
	await board.action_done

func _attack(s: Dictionary) -> void:
	var a = _u(s["unit"])
	var t = _u(s["target"])
	if not _alive(a) or not _alive(t):
		return
	board.selected = a
	board._clear_range()
	board._overlay_cell(_cell_of(t), Color(1.0, 0.18, 0.15, 0.5), Color(0.9, 0.1, 0.05))
	board.cursor.visible = false                 # the red cell alone marks the target
	_show_panel(t)
	follow = [a, t]
	await _sleep(float(s.get("aim", 0.55)))
	board.selected = null                        # no selection pulse while it strikes
	var dmg := int(s.get("damage", -1))
	if dmg < 0:
		dmg = board._damage(a, t)
	# when board._attack's blow lands: the wind-up, then the clip's strike frame
	var anims: Array = a["sheet"]["anims"]
	var clip := "cast" if board.SPELL.has(str(a["name"])) and "cast" in anims else "attack"
	var hit_at := float(board.conf.get("anticipation", 0.12)) + (M.hit_time(clip, board.clip_length(a, clip)) if clip in anims else 0.12)
	if s.has("shake"):
		get_tree().create_timer(hit_at, false).timeout.connect(func(): shake = float(s["shake"]))
	get_tree().create_timer(hit_at + 0.05, false).timeout.connect(func(): _update_panel(t))
	board._attack(a, t, dmg, str(s.get("effect", "")), _col(s.get("colour"), Color(0, 0, 0, 0)))
	await board.action_done
	_hide_panel()
	await _sleep(float(s.get("after", 0.25)))

func _end_turn(s: Dictionary) -> void:
	board._end_turn()
	if board.active_team == 0:
		turn += 1
	_update_turn()
	if bool(s.get("banner", true)):
		var text := str(s.get("text", "PLAYER PHASE" if board.active_team == 0 else "ENEMY PHASE"))
		await _banner({"text": text, "colour": s.get("colour", null), "time": s.get("time", 1.3)})

# ------------------------------------------------------------------ banners and fades

func _banner(s: Dictionary) -> void:
	var col := _col(s.get("colour"), TEAM_COLOURS[board.active_team])
	var hold := float(s.get("time", 1.3))
	var root := Control.new()
	root.set_anchors_preset(Control.PRESET_FULL_RECT)
	root.mouse_filter = Control.MOUSE_FILTER_IGNORE
	ui.add_child(root)
	var band := ColorRect.new()
	band.color = Color(col.r * 0.35, col.g * 0.35, col.b * 0.35, 0.78)
	band.size = Vector2(1920, 150)
	band.position = Vector2(0, 465)
	band.pivot_offset = Vector2(960, 75)
	band.scale = Vector2(1, 0)
	root.add_child(band)
	for y in [0.0, 146.0]:
		var line := ColorRect.new()
		line.color = Color(col.r * 0.6 + 0.4, col.g * 0.6 + 0.4, col.b * 0.6 + 0.4, 0.95)
		line.size = Vector2(1920, 4)
		line.position = Vector2(0, y)
		band.add_child(line)
	var l := Label.new()
	l.text = str(s.get("text", ""))
	l.size = Vector2(1920, 150)
	l.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	l.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	if ui_font != null:
		l.add_theme_font_override("font", ui_font)
		l.add_theme_font_size_override("font_size", 80)
	else:
		l.add_theme_font_size_override("font_size", 84)
	l.add_theme_color_override("font_color", Color(1.0, 0.97, 0.88))
	l.add_theme_color_override("font_outline_color", Color(col.r * 0.25, col.g * 0.25, col.b * 0.25))
	l.add_theme_constant_override("outline_size", 16)
	l.position = Vector2(-260, 465)
	l.modulate.a = 0.0
	root.add_child(l)
	var tw := create_tween()
	tw.tween_property(band, "scale:y", 1.0, 0.18).set_trans(Tween.TRANS_CUBIC).set_ease(Tween.EASE_OUT)
	tw.parallel().tween_property(l, "position:x", 0.0, 0.35).set_trans(Tween.TRANS_BACK).set_ease(Tween.EASE_OUT)
	tw.parallel().tween_property(l, "modulate:a", 1.0, 0.25)
	tw.tween_interval(hold)
	tw.tween_property(l, "position:x", 260.0, 0.3).set_trans(Tween.TRANS_CUBIC).set_ease(Tween.EASE_IN)
	tw.parallel().tween_property(l, "modulate:a", 0.0, 0.3)
	tw.parallel().tween_property(band, "scale:y", 0.0, 0.25).set_delay(0.08)
	tw.tween_callback(root.queue_free)
	if bool(s.get("wait", true)):
		await tw.finished

func _fade(s: Dictionary) -> void:
	var tw := create_tween()
	tw.tween_property(fade_rect, "color:a", float(s.get("to", 0.0)), float(s.get("time", 0.6)))
	if bool(s.get("wait", true)):
		await tw.finished

# ------------------------------------------------------------------ spells, shots, blows (both modes)

func _hit(t, dmg: int, from: Vector3, knock := 0.0, text := "", colour := Color(1.0, 0.95, 0.4)) -> void:
	# damage (or heal: dmg < 0) with a number, flash, hurt clip and knockback
	t["hp"] = clampi(int(t["hp"]) - dmg, 0, int(t["max_hp"]))
	board._update_bar(t)
	board._number(t, abs(dmg), text, colour)
	if t == hero:
		_update_hearts()
	if dmg <= 0:
		return
	board.hit_stop()                               # the blow lands: a short freeze
	var tw := create_tween()
	tw.tween_method(func(v): t["hit"] = v, 0.75, 0.0, 0.4)
	if "hurt" in t["sheet"]["anims"] and t["hp"] > 0:
		board._play(t, "hurt", true)
		get_tree().create_timer(0.45, false).timeout.connect(func():
			if t["anim"] == "hurt":
				board._play(t, "idle"))
	if knock > 0.0 and _alive(t):
		var d: Vector3 = _pos(t) - from
		d.y = 0.0
		if d.length() < 0.01:
			d = Vector3(0, 0, 1)
		var goal: Vector3 = _pos(t) + d.normalized() * knock
		var start: Vector3 = _pos(t)
		goal = _clamp_free(start, goal)
		var kt := create_tween()
		kt.tween_interval(float(board.conf.get("hitstop", 0.075)))
		kt.tween_method(func(k):
			if is_instance_valid(t["node"]):
				var q := start.lerp(goal, k)
				t["node"].position.x = q.x
				t["node"].position.z = q.z, 0.0, 1.0, 0.18).set_ease(Tween.EASE_OUT).set_trans(Tween.TRANS_QUAD)
		kt.tween_callback(func():
			if _alive(t):
				_sync_cell(t))

func _clamp_free(from: Vector3, goal: Vector3) -> Vector3:
	# knockback stops before water, walls more than a step up and the map edge
	var steps := 6
	var last := from
	for i in range(1, steps + 1):
		var p := from.lerp(goal, float(i) / float(steps))
		var c := int(floor(p.x))
		var r := int(floor(p.z))
		if c < 0 or r < 0 or c >= board.cols or r >= board.rows or board.terrain[r][c] == "water":
			break
		if board.top_y(c, r) - _ground(from.x, from.z) > 0.6:
			break
		last = p
	return last

func _projectile(s: Dictionary, from: Vector3, to: Vector3, col: Color) -> void:
	var kind := str(s.get("projectile", ""))
	var speed := float(s.get("speed", 9.0))
	var dur: float = max(0.18, from.distance_to(to) / speed)
	var arc := float(s.get("arc", 0.5 if kind == "arrow" else 0.15))
	var node: Node3D = null
	var fx := {}
	var light: OmniLight3D = null
	var lift := 0.0
	if kind == "arrow":
		node = Node3D.new()
		var shaft := MeshInstance3D.new()
		var bm := BoxMesh.new(); bm.size = Vector3(0.035, 0.035, 0.62)
		shaft.mesh = bm
		var wood := StandardMaterial3D.new(); wood.albedo_color = Color(0.55, 0.4, 0.25)
		shaft.material_override = wood
		node.add_child(shaft)
		var tip := MeshInstance3D.new()
		var tm := BoxMesh.new(); tm.size = Vector3(0.07, 0.07, 0.12)
		tip.mesh = tm
		var glow := StandardMaterial3D.new(); glow.albedo_color = col; glow.emission_enabled = true
		glow.emission = col; glow.emission_energy_multiplier = 4.0
		tip.material_override = glow
		tip.position = Vector3(0, 0, -0.34)
		node.add_child(tip)
		board.add_child(node)
		light = OmniLight3D.new(); light.light_color = col; light.light_energy = 1.2; light.omni_range = 2.0
		board.add_child(light)
	else:
		var a2: Vector2 = board.cam.unproject_position(board.g(from))
		var b2: Vector2 = board.cam.unproject_position(board.g(to))
		fx = board.play_effect(kind, from, true, float(s.get("projectile_light", 0.9)), str(s.get("projectile_tag", "fly")),
			float(s.get("projectile_scale", 0.85)), b2.x < a2.x)
		if fx.is_empty():
			return
		fx["mat"].set_shader_parameter("glow_energy", float(s.get("projectile_glow", 0.8)))
		node = fx["node"]
		light = fx["light"]
		lift = fx["node"].mesh.size.y * 0.5
	var tw := create_tween()
	tw.tween_method(func(k):
		var p := from.lerp(to, k) + Vector3(0, sin(PI * k) * arc, 0)
		if kind == "arrow":
			var ahead := from.lerp(to, min(1.0, k + 0.02)) + Vector3(0, sin(PI * min(1.0, k + 0.02)) * arc, 0)
			node.position = p
			if ahead.distance_to(p) > 0.0001:
				node.look_at(ahead, Vector3.UP)
		else:
			node.position = p - Vector3(0, lift, 0)
		light.position = p, 0.0, 1.0, dur)
	await tw.finished
	if fx.is_empty():
		node.queue_free()
		light.queue_free()
	else:
		board.stop_effect(fx)

func _cast(s: Dictionary) -> void:
	var a = _u(s["unit"])
	if not _alive(a):
		return
	var t = _u(s["target"]) if s.has("target") else null
	if t != null and not _alive(t):
		t = null
	var to: Vector3 = _pos(t) if t != null else _point(s.get("at", [0, 0]))
	var heal := int(s.get("heal", 0))
	var col := _col(s.get("colour"), Color(0.72, 0.45, 1.0))
	board.face_to(a, _yaw_to(_pos(a), to))
	if mode == "tactics":
		board.selected = a
		board._clear_range()
		var cell := Vector2i(int(floor(to.x)), int(floor(to.z)))
		if heal > 0:
			board._overlay_cell(cell, Color(0.3, 1.0, 0.45, 0.45), Color(0.15, 0.8, 0.3))
		else:
			board._overlay_cell(cell, Color(1.0, 0.18, 0.15, 0.5), Color(0.9, 0.1, 0.05))
		board.cursor.visible = false
		if t != null:
			_show_panel(t)
		follow = [a, t] if t != null else a
		await _sleep(float(s.get("aim", 0.55)))
		board.selected = null
		board._clear_range()
	var anims: Array = a["sheet"]["anims"]
	var anim := str(s.get("anim", "cast" if "cast" in anims else "attack"))
	if anim in anims:
		board._play(a, anim, true)
	board.spell_light(_pos(a) + Vector3(0, 0.9, 0), col, float(s.get("charge_light", 1.6)), 0.9, 3.2, 0)
	if s.has("charge"):
		board.play_effect(str(s["charge"]), _pos(a), false, 1.6, str(s.get("charge_tag", "")), float(s.get("charge_scale", 1.0)))
	# the spell leaves on the cast pose: the clip's strike frame (unless the script says when)
	var windup := float(s.get("windup", M.hit_time(anim, board.clip_length(a, anim)) if anim in anims else 0.5))
	await _sleep(windup)
	if s.has("projectile"):
		await _projectile(s, _pos(a) + Vector3(0, 0.55, 0) + Vector3(cos(a["yaw"]), 0, sin(a["yaw"])) * 0.3,
			to + Vector3(0, 0.45, 0), col)
	if s.has("effect"):
		var hit: Dictionary = board.play_effect(str(s["effect"]), to + Vector3(0, float(s.get("lift", 0.0)), 0), false,
			float(s.get("energy", 2.0)), str(s.get("tag", "")), float(s.get("scale", 1.0)))
		if not hit.is_empty() and s.has("glow"):          # how hard its bright pixels bloom
			hit["mat"].set_shader_parameter("glow_energy", float(s["glow"]))
	board.spell_light(to + Vector3(0, 0.8, 0), col, float(s.get("light", 4.0)), 0.9, 5.0, 18)
	if s.has("shake"):
		shake = float(s["shake"])
	if t != null:
		if heal > 0:
			_hit(t, -heal, to, 0.0, "+%d" % heal, Color(0.45, 1.0, 0.55))
		else:
			var dmg := int(s.get("damage", -1))
			if dmg < 0:
				dmg = board._damage(a, t)
			_hit(t, dmg, _pos(a), float(s.get("knock", 0.0)))
		_update_panel(t)
	await _sleep(float(s.get("after", 0.65)))
	board._play(a, "idle")
	if mode == "tactics":
		_hide_panel()
		if t != null and t["hp"] <= 0:
			board._kill(t, a)
			await board.action_done
		else:
			board._finish(a)
	elif t != null and t["hp"] <= 0:
		await _die(t, s.get("drop", {}))

# ------------------------------------------------------------------ action mode

func _setup_action() -> void:
	board.cam_turn_rate = 3.0
	board.number_lift = float(data.get("number_lift", 1.0))
	var lean := float(data.get("lean", 0.8))
	for u in board.units:
		u["bar_on"] = false
		u["mat"].set_shader_parameter("lean", lean)
	board.cursor.visible = false
	hero = _u(data.get("hero", ""))
	if hero == null and board.units.size() > 0:
		hero = board.units[0]
	coins = int(data.get("coins", 0))
	coins_shown = float(coins)
	var root := _hud_root()
	# item box (left), coins, hearts (ALttP's layout, the kit's pieces)
	var slot := _nine("slot", Vector2(22, 22))
	slot.position = Vector2(8, 8)
	root.add_child(slot)
	var item := _icon("icon_" + str(data.get("item", "sword")), Vector2(11, 11))
	root.add_child(item)
	root.add_child(_icon("icon_coin", Vector2(36, 7)))
	coin_label = _label("%03d" % coins, Vector2(52, 10))
	root.add_child(coin_label)
	var life := _label("- LIFE -", Vector2(1920.0 / hud_scale - 104, 2), Color(1.0, 0.85, 0.5))
	root.add_child(life)
	hearts_box = HBoxContainer.new()
	hearts_box.add_theme_constant_override("separation", -3)
	hearts_box.position = Vector2(1920.0 / hud_scale - 8 - 13 * 7, 12)
	root.add_child(hearts_box)
	_update_hearts()

func _update_hearts() -> void:
	if hearts_box == null or hero == null:
		return
	for c in hearts_box.get_children():
		c.queue_free()
	var per := 4
	var n := int(ceil(float(hero["max_hp"]) / per))
	for i in n:
		var v := int(hero["hp"]) - i * per
		var name := "icon_heart" if v >= per else ("icon_heart_half" if v > 0 else "icon_heart_empty")
		var r := TextureRect.new()
		r.texture = _ui_tex(name)
		r.texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
		hearts_box.add_child(r)

func _goto(s: Dictionary) -> void:
	var u = _u(s["unit"])
	if not _alive(u):
		return
	# the board's mover: eased start/stop, rounded corners, turning through the facings, walk or
	# run by speed at the stride's rate (feet planted; a speed the stride can't reach is capped)
	var speed := minf(float(s.get("speed", 2.6)), board.top_speed(u))
	var anim := str(s.get("anim", "walk"))
	if bool(s.get("follow", false)):
		follow = u
	board.blocked.erase(_cell_of(u))
	var pts := []
	for p in s.get("path", []):
		var w := _point(p)
		pts.append(Vector2(w.x, w.z))
	var arrived := [false]
	board.move_along(u, pts, speed, func(): arrived[0] = true)
	while _alive(u) and not arrived[0]:
		await get_tree().process_frame
	if not _alive(u):
		return
	if s.has("face"):
		_face(u, {"dir": s["face"]} if s["face"] is String and not all.has(s["face"]) else {"toward": s["face"]})
	if u["anim"] == anim:
		board._play(u, "idle")
	_sync_cell(u)

func _swing(s: Dictionary) -> void:
	var u = _u(s["unit"])
	if not _alive(u):
		return
	if s.has("dir") or s.has("toward"):
		_face(u, s)
	board._play(u, "attack", true)
	var windup := float(s.get("windup", 0.16))
	var L: float = board.clip_length(u, "attack")
	if L > 0.0:                                    # the strike frame lands on the hit
		u["anim_speed"] = clampf(M.hit_time("attack", L) / maxf(windup, 0.05), 1.0, 4.0)
	await _sleep(windup)
	var fwd := Vector3(cos(u["yaw"]), 0, sin(u["yaw"]))
	var at: Vector3 = _pos(u) + fwd * float(s.get("reach", 0.55))
	var right: Vector3 = board.cam.global_transform.basis.x
	board.play_effect(str(s.get("effect", "slash")), at + Vector3(0, 0.05, 0), false, float(s.get("energy", 0.7)), "",
		float(s.get("scale", 1.0)), fwd.dot(right) < -0.2)
	if s.has("shake"):
		shake = float(s["shake"])
	var drops: Dictionary = s.get("drops", {})
	for id in s.get("hits", []):
		var t = _u(id)
		if not _alive(t):
			continue
		if s.has("spark"):
			board.play_effect(str(s["spark"]), _pos(t) + Vector3(0, 0.15, 0), false, 2.0, "", float(s.get("spark_scale", 0.8)))
		board.spell_light(_pos(t) + Vector3(0, 0.6, 0), Color(1.0, 0.92, 0.75), float(s.get("hit_light", 1.0)), 0.25, 2.5, 8)
		_hit(t, int(s.get("damage", 4)), _pos(u), float(s.get("knock", 0.7)))
		if t["hp"] <= 0:
			_die(t, drops.get(id, {}))
	if s.has("cut"):
		_cut(s["cut"], s.get("drop", {}))
	await _sleep(float(s.get("time", 0.34)))
	if u["anim"] == "attack":
		board._play(u, "idle")

func _hurt(s: Dictionary) -> void:
	var t = _u(s["unit"])
	var a = _u(s["by"]) if s.has("by") else null
	if not _alive(t):
		return
	var from: Vector3 = _pos(t) + Vector3(0, 0, -0.1)
	if a != null and _alive(a):
		board.face_to(a, _yaw_to(_pos(a), _pos(t)))
		from = _pos(a)
		if "attack" in a["sheet"]["anims"]:
			board._play(a, "attack", true)
			await _sleep(0.3)
		else:                                      # a lunge (animals have no attack clip)
			var home: Vector3 = _pos(a)
			var toward: Vector3 = (_pos(t) - home)
			toward.y = 0
			toward = toward.normalized() * 0.4
			var lt := create_tween()
			lt.tween_property(a["node"], "position", home + toward, 0.1)
			lt.tween_property(a["node"], "position", home, 0.18)
			await _sleep(0.1)
	board.spell_light(_pos(t) + Vector3(0, 0.6, 0), Color(1.0, 0.45, 0.35), 0.9, 0.3, 2.5, 6)
	_hit(t, int(s.get("damage", 2)), from, float(s.get("knock", 0.55)), "", Color(1.0, 0.5, 0.45))
	if t == hero:
		_blink(t, 0.9)
	await _sleep(float(s.get("time", 0.3)))
	if a != null and _alive(a) and a["anim"] == "attack":
		board._play(a, "idle")

func _blink(u, time: float) -> void:
	var tw := create_tween()
	tw.tween_method(func(k):
		if is_instance_valid(u["mesh"]):
			u["mesh"].visible = int(k * time * 16.0) % 2 == 0 or k >= 1.0, 0.0, 1.0, time)

func _die(t, drop) -> void:
	if t == null or not _alive(t):
		return
	t["dying"] = true
	var anims: Array = t["sheet"]["anims"]
	if "death" in anims or "lie_down" in anims:
		board._play(t, "death" if "death" in anims else "lie_down", true)
		await _sleep(0.75)
	if not is_instance_valid(t["node"]):
		return
	var at: Vector3 = _pos(t)
	board.play_effect(str(data.get("poof", "poof")), at, false, 1.5, "", float(data.get("poof_scale", 1.0)))
	board.spell_light(at + Vector3(0, 0.5, 0), Color(1.0, 0.95, 0.85), 1.5, 0.4, 2.5, 10)
	var tw := create_tween()
	tw.tween_method(func(v): t["mat"].set_shader_parameter("tint", t["tint"] * v), 1.0, 0.0, 0.3)
	await tw.finished
	board.blocked.erase(_cell_of(t))
	board.units.erase(t)
	if is_instance_valid(t["node"]):
		t["node"].queue_free()
	if drop is Dictionary and drop.has("id"):
		var d: Dictionary = drop.duplicate()
		d["at"] = [at.x - 0.5, at.z - 0.5]
		_spawn(d)

func _spawn(s: Dictionary) -> void:
	var kind := str(s.get("kind", "coin"))
	var sp := Sprite3D.new()
	sp.texture = _ui_tex("icon_" + kind)
	sp.pixel_size = float(s.get("size", 0.028))
	sp.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	sp.shaded = false
	sp.texture_filter = BaseMaterial3D.TEXTURE_FILTER_NEAREST
	sp.alpha_cut = SpriteBase3D.ALPHA_CUT_DISCARD
	sp.offset = Vector2(0, 8)
	sp.modulate = Color(1.25, 1.25, 1.25)
	var at := _point(s.get("at", [0, 0]))
	sp.position = at
	sp.scale = Vector3(0.3, 0.3, 0.3)
	board.add_child(sp)
	var cols := {"heart": Color(1.0, 0.35, 0.4), "coin": Color(1.0, 0.82, 0.3), "gem": Color(0.35, 0.75, 1.0)}
	var l := OmniLight3D.new()
	l.light_color = cols.get(kind, Color(1, 1, 1))
	l.light_energy = 0.9
	l.omni_range = 1.6
	l.position = at + Vector3(0, 0.4, 0)
	board.add_child(l)
	var pk := {"node": sp, "light": l, "y": at.y, "kind": kind, "phase": float(pickups.size()) * 1.7, "bob": false,
		"value": int(s.get("value", 5 if kind == "gem" else 1))}
	pickups[str(s.get("id", "pickup%d" % pickups.size()))] = pk
	var tw := create_tween()
	tw.set_parallel(true)
	tw.tween_property(sp, "scale", Vector3.ONE, 0.25).set_trans(Tween.TRANS_BACK).set_ease(Tween.EASE_OUT)
	tw.tween_method(func(k): sp.position.y = at.y + sin(PI * k) * 0.7, 0.0, 1.0, 0.4)
	tw.chain().tween_callback(func(): pk["bob"] = true)

func _pickup(s: Dictionary) -> void:
	var id := str(s.get("id", ""))
	if not pickups.has(id):
		push_warning("autoplay: no pickup '%s'" % id)
		return
	var pk: Dictionary = pickups[id]
	pickups.erase(id)
	var sp: Sprite3D = pk["node"]
	var at: Vector3 = sp.position
	var u = _u(s.get("unit", "")) if s.has("unit") else hero
	if s.has("sparkle") or data.has("sparkle"):
		board.play_effect(str(s.get("sparkle", data.get("sparkle", ""))), at, false, 1.8, "", float(s.get("sparkle_scale", 0.7)))
	board.spell_light(at + Vector3(0, 0.4, 0), pk["light"].light_color, 1.2, 0.5, 2.5, 10)
	match pk["kind"]:
		"heart":
			if u != null:
				_hit(u, -4, at, 0.0, "+", Color(1.0, 0.5, 0.55))
		_:
			coins += int(s.get("value", pk["value"]))
	var tw := create_tween()
	tw.set_parallel(true)
	tw.tween_property(sp, "position:y", at.y + 0.9, 0.35)
	tw.tween_property(sp, "modulate:a", 0.0, 0.35)
	tw.chain().tween_callback(func():
		sp.queue_free()
		pk["light"].queue_free())
	await _sleep(float(s.get("time", 0.0)))

func _cut(cells: Array, drop) -> void:
	for cell in cells:
		var k := Vector2i(int(cell[0]), int(cell[1]))
		var at := _point(cell)
		for n in board.prop_nodes.get(k, []):
			if is_instance_valid(n):
				n.queue_free()
		board.prop_nodes.erase(k)
		board.blocked.erase(k)
		board.obstacles.erase(k)
		board._embers(at + Vector3(0, 0.25, 0), 22, Color(0.36, 0.62, 0.22), 0.25, 2.2, true)
		if data.has("leaves"):
			board.play_effect(str(data["leaves"]), at, false, 1.0, "", float(data.get("leaves_scale", 0.8)))
		if drop is Dictionary and drop.has("id"):
			var d: Dictionary = drop.duplicate()
			d["at"] = cell
			_spawn(d)
'''

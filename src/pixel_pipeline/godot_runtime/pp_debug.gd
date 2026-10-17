extends CanvasLayer
## Playtest helper (pixel-pipeline godot_runtime), an autoload named PPDebug.
##   F1   controls hint on/off            F3  debug overlay (FPS + game state + collision shapes)
##   F12  report: screenshot + state saved to res://reports/<time>.png/.txt
##   F5   restart the scene
## Games join by putting a node in the group "pp_debug" with
##   debug_state() -> Dictionary   (lines shown in the overlay and written into reports)
##   debug_draw(on: bool)          (optional: show their own shapes, e.g. 3D hit circles)
## res://playtest.json {"style", "title", "controls"} gives the hint text.
## Checks (command line, after --):  --pp-selftest  runs selftest() of the sources (a coroutine
## returning a Dictionary), prints "PPTEST {json}" and quits;  --pp-shot=<png> [--pp-shot-delay=s]
## calls shot_setup() of the sources, waits, saves a screenshot and quits.

var style := ""
var title := ""
var controls := ""
var hint_at := "bottom-left"     # playtest.json "hint_at": bottom-left | top-left | top-right
var on := false
var hint_on := true
var panel: PanelContainer
var label: Label
var hint: Label
var toast: Label
var toast_t := 0.0
var inputs: Array = []          # recent input events: "12.34 key W"
var history: Array = []         # [{t, state}] the last ~3 s, sampled 4x per second
var sample_t := 0.0
var clock := 0.0
var fps_min := 999.0
var fps_window: Array = []


func _ready() -> void:
	layer = 120
	process_mode = Node.PROCESS_MODE_ALWAYS
	if FileAccess.file_exists("res://playtest.json"):
		var j = JSON.parse_string(FileAccess.get_file_as_string("res://playtest.json"))
		if j is Dictionary:
			style = str(j.get("style", ""))
			title = str(j.get("title", ""))
			controls = str(j.get("controls", ""))
			hint_at = str(j.get("hint_at", hint_at))
	panel = PanelContainer.new()
	var sb := StyleBoxFlat.new()
	sb.bg_color = Color(0.03, 0.03, 0.06, 0.72)
	sb.set_content_margin_all(6)
	panel.add_theme_stylebox_override("panel", sb)
	panel.mouse_filter = Control.MOUSE_FILTER_IGNORE
	panel.visible = false
	add_child(panel)
	label = _make_label(Color(0.85, 1.0, 0.85))
	panel.add_child(label)
	hint = _make_label(Color(1.0, 0.97, 0.85))
	hint.text = controls + "\nF1 hide this   F3 debug   F12 report   F5 restart"
	add_child(hint)
	toast = _make_label(Color(1.0, 0.9, 0.4))
	toast.visible = false
	add_child(toast)
	get_viewport().size_changed.connect(_layout)
	_layout()
	var args := OS.get_cmdline_user_args()
	var shot := ""
	var delay := 3.0
	for a in args:
		if a.begins_with("--pp-shot="):
			shot = a.substr(10)
		elif a.begins_with("--pp-shot-delay="):
			delay = float(a.substr(16))
	if "--pp-selftest" in args:
		_selftest.call_deferred()
	elif shot != "":
		_shot.call_deferred(shot, delay)


func _selftest() -> void:
	await get_tree().create_timer(1.0).timeout
	var out := {}
	for n in _sources():
		if is_instance_valid(n) and n.has_method("selftest"):
			var r = await n.selftest()
			if r is Dictionary:
				out.merge(r, true)
	out["fps"] = Engine.get_frames_per_second()
	out["state"] = state()
	print("PPTEST " + JSON.stringify(out))
	get_tree().quit()


func _shot(path: String, delay: float) -> void:
	await get_tree().create_timer(0.5).timeout
	for n in _sources():
		if is_instance_valid(n) and n.has_method("shot_setup"):
			await n.shot_setup()
	await get_tree().create_timer(delay).timeout
	await RenderingServer.frame_post_draw
	var img := get_viewport().get_texture().get_image()
	img.save_png(path)
	print("PPSHOT " + path)
	get_tree().quit()


func _make_label(col: Color) -> Label:
	var l := Label.new()
	l.add_theme_color_override("font_color", col)
	l.add_theme_color_override("font_outline_color", Color(0, 0, 0, 0.95))
	l.add_theme_constant_override("outline_size", 4)
	l.mouse_filter = Control.MOUSE_FILTER_IGNORE
	return l


func _font_px() -> int:
	# readable on a 1080p window and inside a 640x360 pixel viewport alike
	var h := get_viewport().get_visible_rect().size.y
	return maxi(8, int(round(15.0 * h / 1080.0)))


func _layout() -> void:
	var vp := get_viewport().get_visible_rect().size
	var fs := _font_px()
	for l in [label, hint, toast]:
		l.add_theme_font_size_override("font_size", fs)
		l.add_theme_constant_override("outline_size", maxi(2, fs / 4))
	panel.position = Vector2(vp.x * 0.005, vp.y * 0.16)
	var lines := hint.text.count("\n") + 1
	match hint_at:
		"top-right":
			hint.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
			hint.size = Vector2(vp.x * 0.99, 0)
			hint.position = Vector2(0, vp.y * 0.01)
		"top-left":
			hint.position = Vector2(vp.x * 0.008, vp.y * 0.01)
		"under-bar":
			hint.position = Vector2(vp.x * 0.008, vp.y * 0.07)
		_:
			var hgt := maxf(hint.get_combined_minimum_size().y, fs * lines * 1.45)
			hint.position = Vector2(vp.x * 0.008, vp.y - hgt - vp.y * 0.01)
	toast.position = Vector2(vp.x * 0.35, vp.y * 0.06)


func _sources() -> Array:
	return get_tree().get_nodes_in_group("pp_debug")


func state() -> Dictionary:
	var out := {}
	for n in _sources():
		if is_instance_valid(n) and n.has_method("debug_state"):
			var d = n.debug_state()
			if d is Dictionary:
				out.merge(d, true)
	return out


func _format(d: Dictionary) -> String:
	var lines := []
	for k in d:
		var v = d[k]
		if v is float:
			v = "%.2f" % v
		elif v is Vector2 or v is Vector3:
			v = str(v.snapped(Vector3.ONE * 0.01) if v is Vector3 else v.snapped(Vector2.ONE * 0.01))
		lines.append("%s: %s" % [k, v])
	return "\n".join(lines)


func _process(delta: float) -> void:
	clock += delta
	var fps := Engine.get_frames_per_second()
	fps_window.append(1.0 / maxf(delta, 0.0001))
	if fps_window.size() > 120:
		fps_window.pop_front()
	if toast_t > 0.0:
		toast_t -= delta
		toast.visible = toast_t > 0.0
	sample_t -= delta
	if sample_t <= 0.0:
		sample_t = 0.25
		history.append({"t": snappedf(clock, 0.01), "state": state()})
		while history.size() > 12:
			history.pop_front()
	if on:
		var low := 9999.0
		for f in fps_window:
			low = minf(low, f)
		var head := "%s   FPS %d (low %d)   %s" % [title if title != "" else style, fps, int(low),
			"%.0f MB" % (OS.get_static_memory_usage() / 1048576.0)]
		label.text = head + "\n" + _format(state())


func _input(e: InputEvent) -> void:
	if e is InputEventKey and e.pressed and not e.echo:
		_log("key " + OS.get_keycode_string(e.keycode))
		match e.keycode:
			KEY_F1:
				hint_on = not hint_on
				hint.visible = hint_on
				get_viewport().set_input_as_handled()
			KEY_F3:
				toggle()
				get_viewport().set_input_as_handled()
			KEY_F12:
				get_viewport().set_input_as_handled()
				report()
			KEY_F5:
				get_viewport().set_input_as_handled()
				get_tree().reload_current_scene()
			KEY_F11:                      # fullscreen <-> window (pixel games keep a whole-number scale)
				get_viewport().set_input_as_handled()
				var w := get_window()
				w.mode = Window.MODE_WINDOWED if w.mode == Window.MODE_FULLSCREEN else Window.MODE_FULLSCREEN
	elif e is InputEventMouseButton and e.pressed:
		_log("mouse %d at %s" % [e.button_index, e.position.round()])


func _log(s: String) -> void:
	inputs.append("%.2f %s" % [clock, s])
	while inputs.size() > 40:
		inputs.pop_front()


func toggle() -> void:
	on = not on
	panel.visible = on
	_collision_shapes(on)
	for n in _sources():
		if is_instance_valid(n) and n.has_method("debug_draw"):
			n.debug_draw(on)


func _collision_shapes(show: bool) -> void:
	# 2D: Godot's own collision debug drawing, switched on at runtime
	get_tree().debug_collisions_hint = show
	var stack: Array = [get_tree().root]
	while not stack.is_empty():
		var n: Node = stack.pop_back()
		if n is CollisionShape2D or n is CollisionPolygon2D:
			n.queue_redraw()
		elif n is TileMapLayer:
			n.collision_visibility_mode = TileMapLayer.DEBUG_VISIBILITY_MODE_FORCE_SHOW if show else TileMapLayer.DEBUG_VISIBILITY_MODE_DEFAULT
		stack.append_array(n.get_children())


func report() -> void:
	var dir := ProjectSettings.globalize_path("res://reports")
	DirAccess.make_dir_recursive_absolute(dir)
	var stamp := Time.get_datetime_string_from_system(false, true).replace(":", "-").replace(" ", "_")
	toast.visible = false
	await RenderingServer.frame_post_draw
	var tex := get_viewport().get_texture()
	var png := "%s/%s_%s.png" % [dir, style if style != "" else "game", stamp]
	if tex != null:
		var img := tex.get_image()
		if img != null:
			img.save_png(png)
	var lines := ["Playtest report: %s" % (title if title != "" else style),
		"time: %s   game clock: %.2f s   FPS: %d" % [Time.get_datetime_string_from_system(false, true), clock, Engine.get_frames_per_second()],
		"screenshot: %s" % png.get_file(), "",
		"YOUR NOTE (type what looks wrong here):", "", "",
		"--- state now ---", _format(state()), "",
		"--- state over the last 3 seconds ---"]
	for h in history:
		lines.append("[t %.2f] %s" % [h["t"], JSON.stringify(h["state"])])
	lines += ["", "--- recent input ---"]
	lines += inputs
	var f := FileAccess.open(png.get_basename() + ".txt", FileAccess.WRITE)
	if f != null:
		f.store_string("\n".join(lines) + "\n")
		f.close()
	toast.text = "Report saved: reports/" + png.get_file().get_basename()
	toast.visible = true
	toast_t = 2.5
	print("PPDEBUG REPORT " + png)

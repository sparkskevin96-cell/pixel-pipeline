extends Node
## Playtest helper (pixel-pipeline godot_runtime), an autoload named PPTactics: a simple enemy
## AI for the tactics boards the app writes - the HD-2D board (hd2d.py board.gd: real / fake 3D
## looks) and the isometric 2D scene (iso_scene.py iso.gd). Team 0 is the player, every other
## team is played by the AI: each unit walks to the reachable cell nearest a player unit
## (preferring cells it can attack from, higher ground first), then hits the weakest player unit
## in reach. It also feeds PPDebug (F3) with the hovered/selected unit's cell, height, facing,
## animation and frame.

const AI_TEAM_DELAY := 0.6
const THINK := 0.35

var scene = null
var kind := ""                 # "board" | "iso" | ""
var running := false
var debug_on := false
var debug_node: Node = null
var hovered: Variant = null    # board: the unit under the mouse
var last_mouse := Vector2.ZERO
var redraw_t := 0.0


func _ready() -> void:
	add_to_group("pp_debug")
	process_mode = Node.PROCESS_MODE_ALWAYS


func _process(_delta: float) -> void:
	var cur := get_tree().current_scene
	if cur != scene:
		scene = cur
		running = false
		debug_node = null
		kind = ""
		if cur != null and "units" in cur and cur.has_method("_end_turn") and cur.has_method("_walk"):
			kind = "board"
		elif cur != null and cur.has_signal("turn_started") and cur.has_method("move_selected"):
			kind = "iso"
		if kind == "board" and cur.pivot != null and cur.cols * cur.rows <= 200:
			# a small map: start on its middle, far enough out to see all of it
			cur.pivot.position = Vector3(cur.cols / 2.0, 1.5, cur.rows / 2.0)
			cur.cam_dist = maxf(cur.cam_dist, 2.6 * maxf(cur.cols, cur.rows) + 2.0)
		if debug_on and kind != "":
			debug_draw.call_deferred(true)
	if debug_on and kind != "":
		redraw_t -= _delta
		if redraw_t <= 0.0:
			redraw_t = 0.4
			debug_draw(true)
	if kind == "" or not is_instance_valid(scene) or running:
		return
	if kind == "board":
		if int(scene.active_team) != 0 and scene.phase == "select" and not scene.scripted:
			_board_turn()
	elif kind == "iso":
		if int(scene.active_team) != 0 and scene.phase == "select":
			_iso_turn()


func _input(e: InputEvent) -> void:
	if e is InputEventMouseMotion:
		last_mouse = e.position
	# the player can't click for the AI while it plays (keys F1-F12 still reach PPDebug)
	if running and kind == "iso" and (e is InputEventMouseButton or (e is InputEventKey and e.keycode < KEY_F1)):
		if e is InputEventMouseButton and e.button_index in [MOUSE_BUTTON_WHEEL_UP, MOUSE_BUTTON_WHEEL_DOWN]:
			return
		get_viewport().set_input_as_handled()


func _wait(t: float) -> void:
	await get_tree().create_timer(t, false).timeout


func _manhattan(a: Vector2i, b: Vector2i) -> int:
	return absi(a.x - b.x) + absi(a.y - b.y)


# ------------------------------------------------------------------ HD-2D board (board.gd)

func _cell(u: Dictionary) -> Vector2i:
	return Vector2i(int(u["col"]), int(u["row"])) if kind == "board" else u["cell"]


func _pick_cell(u: Dictionary, options: Array, foes: Array, height_of: Callable) -> Vector2i:
	var best: Vector2i = _cell(u)
	var best_score := -INF
	for c in options:
		var near := 9999
		for f in foes:
			near = mini(near, _manhattan(c, _cell(f)))
		var score := -float(near)
		if near <= int(u["reach"]):
			score = 100.0 + float(height_of.call(c)) * 0.5 - (0.3 if near < int(u["reach"]) else 0.0)
		if c == _cell(u):
			score += 0.05                     # don't shuffle for nothing
		if score > best_score:
			best_score = score
			best = c
	return best


func _weakest(u: Dictionary, foes: Array) -> Variant:
	var pick: Variant = null
	for f in foes:
		if _manhattan(_cell(u), _cell(f)) <= int(u["reach"]) and int(f["hp"]) > 0:
			if pick == null or int(f["hp"]) < int(pick["hp"]):
				pick = f
	return pick


func _board_foes(team: int) -> Array:
	return scene.units.filter(func(x): return int(x["team"]) != team and int(x["hp"]) > 0)


func _board_over() -> bool:
	var teams := {}
	for x in scene.units:
		teams[int(x["team"])] = true
	return teams.size() < 2


func _banner(text: String, col: Color) -> void:
	if kind != "board" or scene.banner == null:
		return
	scene.banner.text = text
	scene.banner.modulate = col
	scene.banner.visible = true
	await _wait(0.9)
	if is_instance_valid(scene) and scene.phase != "over":
		scene.banner.visible = false
		scene.banner.modulate = Color.WHITE


func _board_turn() -> void:
	running = true
	var b = scene
	var team := int(b.active_team)
	b.scripted = true
	b.cursor.visible = false
	await _banner("ENEMY PHASE", Color(1.0, 0.55, 0.5))
	for u in b.units.filter(func(x): return int(x["team"]) == team):
		if not is_instance_valid(b) or _board_over():
			break
		if not b.units.has(u) or u["done"]:
			continue
		var foes := _board_foes(team)
		if foes.is_empty():
			break
		b.follow_point(u["node"].position + Vector3(0, 0.5, 0))     # the damped camera glides there
		b.selected = u
		b.phase = "move"
		b._show_range()
		b._update_hud()
		await _wait(THINK)
		var goal := _pick_cell(u, b.reach.keys(), foes, func(c): return int(b.heights[c.y][c.x]))
		if goal != Vector2i(int(u["col"]), int(u["row"])) and b.reach.has(goal):
			b._walk(u, b._path_to(goal))
			await b.action_done
		else:
			b._clear_range()
		if not is_instance_valid(b):
			return
		var target = _weakest(u, _board_foes(team))
		if target != null:
			b._attack(u, target)
			await b.action_done
		else:
			b._finish(u)                       # scripted: emits action_done right away
		await _wait(0.15)
	if not is_instance_valid(b):
		running = false
		return
	b.scripted = false
	b.selected = null
	b._clear_range()
	if _board_over():                          # the same banner board.gd shows for a winner
		b.phase = "over"
		var won: int = int(b.units[0]["team"]) if b.units.size() > 0 else team
		b.banner.text = "%s wins!" % b._team_name(won)
		b.banner.modulate = Color.WHITE
		b.banner.visible = true
		b._update_hud()
	else:
		b._end_turn()
		_banner("PLAYER PHASE", Color(0.6, 0.8, 1.0))
	running = false


# ------------------------------------------------------------------ isometric (iso.gd)

func _iso_turn() -> void:
	running = true
	var s = scene
	var team := int(s.active_team)
	await _wait(AI_TEAM_DELAY)
	for u in s.units.filter(func(x): return int(x["team"]) == team):
		if not is_instance_valid(s) or s.phase == "over":
			break
		if not s.units.has(u) or u["done"]:
			continue
		var foes: Array = s.units.filter(func(x): return int(x["team"]) != team and int(x["hp"]) > 0)
		if foes.is_empty():
			break
		s.centre_on(u["cell"])
		s.select(u)
		await _wait(THINK)
		var options: Array = s.reach.keys().filter(func(c): return s.unit_on(c) == null or c == u["cell"])
		var goal := _pick_cell(u, options, foes, func(c): return s.hgt(c))
		if goal != u["cell"] and s.move_selected(goal):
			while is_instance_valid(s) and s.phase == "walk":
				await get_tree().process_frame
		else:
			s.enter_act(u)
		if not is_instance_valid(s):
			return
		if s.phase == "act":
			var target = _weakest(u, foes)
			if target != null and s.attack_cells.has(target["cell"]):
				s.attack(u, target)
				while is_instance_valid(s) and s.phase == "busy":
					await get_tree().process_frame
			elif s.phase == "act":
				s.finish(u)
		await _wait(0.15)
		if is_instance_valid(s) and int(s.active_team) != team:
			break                              # finish() already passed the turn
	if is_instance_valid(s) and s.phase != "over" and int(s.active_team) == team:
		s.end_turn()
	running = false


# ------------------------------------------------------------------ PPDebug (F3 / F12)

func _board_frame(u: Dictionary) -> String:
	var s: Dictionary = u["sheet"]
	var anim: String = u["anim"] if u["anim"] in s["anims"] else "idle"
	var dir: String = scene._sprite_dir(u["yaw"])
	var key := "%s_%s" % [anim, dir]
	if not s["frames"].has(key):
		return "%s (missing)" % key
	var fr: Array = s["frames"][key]
	var total := 0.0
	for f in fr:
		total += f["ms"]
	var t := fmod(u["t"] * 1000.0, total)
	if u["hold"]:
		t = min(u["t"] * 1000.0, total - 1.0)
	var i := 0
	for f in fr:
		if t < f["ms"]:
			break
		t -= f["ms"]
		i += 1
	return "%s frame %d/%d" % [key, mini(i, fr.size() - 1) + 1, fr.size()]


func _board_hover() -> Variant:
	if scene.cam == null:
		return null
	var cell = scene._cell_under_mouse(last_mouse)
	return scene._unit_at(cell) if cell != null else null


func debug_state() -> Dictionary:
	if kind == "" or not is_instance_valid(scene):
		return {}
	var d := {"turn": "player (blue)" if int(scene.active_team) == 0 else "AI (team %d)" % int(scene.active_team),
		"phase": scene.phase, "ai playing": running}
	if kind == "board":
		d["camera"] = "yaw %.0f  pitch %.0f  dist %.1f  look %s" % [scene.cam_yaw, scene.cam_pitch if scene.cam_pitch > 0 else (38.0 if scene.mode == "real" else 30.0), scene.cam_dist, scene.mode]
		var cell = scene._cell_under_mouse(last_mouse)
		if cell != null:
			d["mouse cell"] = "%s  level %d  %s" % [cell, int(scene.heights[cell.y][cell.x]), scene.terrain[cell.y][cell.x]]
		for pair in [["selected", scene.selected], ["hovered", _board_hover()]]:
			var u = pair[1]
			if u == null:
				continue
			var p: Vector3 = u["node"].position
			d[pair[0]] = "%s team %d  hp %d/%d  cell (%d,%d) level %d  pos %s" % [u["id"], u["team"], u["hp"], u["max_hp"], u["col"], u["row"], int(scene.heights[u["row"]][u["col"]]), p.snapped(Vector3.ONE * 0.01)]
			d[pair[0] + " facing"] = "world %.0f deg -> sprite %s" % [rad_to_deg(u["yaw"]), scene._sprite_dir(u["yaw"])]
			d[pair[0] + " anim"] = _board_frame(u)
	else:
		for pair in [["selected", scene.selected], ["hovered", scene.hovered]]:
			var u = pair[1]
			if u == null:
				continue
			var spr: AnimatedSprite2D = u["sprite"]
			d[pair[0]] = "%s team %d  hp %d/%d  cell %s level %d  pos %s" % [u["name"], u["team"], u["hp"], u["max_hp"], u["cell"], s_hgt(u["cell"]), u["node"].position]
			d[pair[0] + " anim"] = "%s frame %d/%d" % [str(spr.animation), spr.frame + 1, spr.sprite_frames.get_frame_count(spr.animation) if spr.sprite_frames.has_animation(spr.animation) else 0]
		d["zoom"] = scene.zoom_level
	return d


func s_hgt(c: Vector2i) -> int:
	return int(scene.hgt(c)) if scene.inside(c) else 0


func debug_draw(show: bool) -> void:
	debug_on = show
	if is_instance_valid(debug_node):
		debug_node.queue_free()
	debug_node = null
	if not show or kind == "" or not is_instance_valid(scene):
		return
	if kind == "board":
		# blocked cells (props, units) in red, every unit's cell outlined in its team colour
		var root := Node3D.new()
		var red := StandardMaterial3D.new()
		red.albedo_color = Color(1.0, 0.1, 0.1, 0.35)
		red.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
		red.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
		red.no_depth_test = true
		for cell in scene.blocked:
			var m := MeshInstance3D.new()
			var q := QuadMesh.new(); q.size = Vector2(0.9, 0.9); q.orientation = PlaneMesh.FACE_Y
			m.mesh = q
			m.material_override = red
			m.position = scene.cell_pos(cell.x, cell.y) + Vector3(0, 0.04, 0)
			root.add_child(m)
		for u in scene.units:
			var ring := MeshInstance3D.new()
			var tm := TorusMesh.new(); tm.inner_radius = 0.36; tm.outer_radius = 0.42
			ring.mesh = tm
			var mat := StandardMaterial3D.new()
			mat.albedo_color = Color(0.3, 0.6, 1.0) if int(u["team"]) == 0 else Color(1.0, 0.4, 0.3)
			mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
			mat.no_depth_test = true
			ring.material_override = mat
			ring.position = scene.cell_pos(int(u["col"]), int(u["row"])) + Vector3(0, 0.05, 0)
			root.add_child(ring)
		scene.add_child(root)
		debug_node = root
	else:
		var root := Node2D.new()
		root.z_index = 4000
		for u in scene.units:
			var r := ReferenceRect.new()
			r.border_color = Color(0.3, 0.6, 1.0) if int(u["team"]) == 0 else Color(1.0, 0.4, 0.3)
			r.editor_only = false
			var body: Rect2 = u["body"]
			r.position = u["node"].position + body.position
			r.size = body.size
			root.add_child(r)
		for cell in scene.blocked:
			var dot := Polygon2D.new()
			var c: Vector2 = scene.top(cell)
			dot.polygon = PackedVector2Array([c + Vector2(-6, 0), c + Vector2(0, -3), c + Vector2(6, 0), c + Vector2(0, 3)])
			dot.color = Color(1, 0.1, 0.1, 0.6)
			root.add_child(dot)
		scene.add_child(root)
		debug_node = root


# ------------------------------------------------------------------ checks (PPDebug --pp-selftest / --pp-shot)

func _snapshot() -> Dictionary:
	var d := {}
	for u in scene.units:
		var key := "%s@%d" % [u.get("id", u["name"]), int(u["team"])]
		while d.has(key):
			key += "'"
		d[key] = [str(_cell(u)), int(u["hp"])]
	return d


func selftest() -> Dictionary:
	var t := 0.0
	while (kind == "" or not is_instance_valid(scene)) and t < 10.0:
		await get_tree().process_frame
		t += get_process_delta_time()
	if kind == "":
		return {"tactics": "no board found"}
	await _wait(0.5)
	var before := _snapshot()
	var out := {"kind": kind, "before": before}
	# the player's knight: select -> move one step toward the river -> act/finish (real clicks)
	var k = null
	for u in scene.units:
		if int(u["team"]) == 0:
			k = u
			break
	if k != null and kind == "board":
		scene._click(_cell(k))
		var goal := _cell(k) + Vector2i(1, 0)
		if scene.reach.has(goal):
			scene._click(goal)
			await _until(func(): return scene.phase != "busy", 8.0)
		if scene.phase == "act":
			scene._finish(scene.selected)
		out["player_moved"] = _cell(k) == goal
	elif k != null:
		scene.select(k)
		var goal2: Vector2i = _cell(k) + Vector2i(1, 0)
		if scene.move_selected(goal2):
			await _until(func(): return scene.phase != "walk", 8.0)
		if scene.phase == "act":
			scene.finish(k)
		out["player_moved"] = _cell(k) == goal2
	# pass the turn: the AI plays every red unit, then it's the player's turn again
	if int(scene.active_team) == 0 and scene.phase != "over":
		if kind == "board":
			scene._end_turn()
		else:
			scene.end_turn()
	await _until(func(): return running, 5.0)
	out["ai_started"] = running
	await _until(func(): return not running and (int(scene.active_team) == 0 or scene.phase == "over"), 60.0)
	out["after"] = _snapshot()
	out["turn_back_to_player"] = int(scene.active_team) == 0
	var moved := 0
	for key in before:
		if out["after"].has(key) and out["after"][key][0] != before[key][0] and not key.ends_with("@0"):
			moved += 1
	out["ai_units_moved"] = moved
	return {"tactics": out}


func _until(cond: Callable, limit: float) -> void:
	var t := 0.0
	while not cond.call() and t < limit:
		await get_tree().process_frame
		t += maxf(get_process_delta_time(), 0.001)


func shot_setup() -> void:
	await _until(func(): return kind != "", 10.0)
	if kind == "board":
		for u in scene.units:
			if int(u["team"]) == 0 and str(u["name"]) == "mage":
				scene._click(_cell(u))
				break
	elif kind == "iso":
		for u in scene.units:
			if int(u["team"]) == 0:
				scene.select(u)
				break

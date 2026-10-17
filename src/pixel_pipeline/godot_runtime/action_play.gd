extends "res://autoplay.gd"
## Playtest helper (pixel-pipeline godot_runtime): live Link-to-the-Past-style controls on the
## HD-2D board (hd2d.py board.gd). It extends the autoplay player (hd2d_autoplay.py) and reuses
## its HUD (UI kit hearts / coins / item box), swing / hit / knockback / death / drops / pickups /
## projectiles / banners; the timeline is replaced by input.
##   WASD / arrows  walk (8 directions)        Space / J  sword        K  shoot (res://action.json
##   "shot": fireball | arrow)                  R  restart              Esc  quit
## The scene root carries this script: it builds the board (res://real3d.tscn) as its child and
## plays res://action.json (the autoplay settings: hero, camera, light, coins, item, presets).
## Movement is the game runtime's (pp_motion.gd): velocity eased in and out, facing turned through
## the 16 sprite directions, walk/run played at speed / stride (feet planted), feet following the
## ground over steps; collision = a circle against cliffs, water, buildings (cell boxes) and props
## (trees, pots, bushes, crates: circles), pushed out so the hero slides along them.

const CONF := "res://action.json"
var speed := 3.1                # cells per second
var radius := 0.24              # the hero's collision half-size (a square on the ground)
var sword_reach := 0.95
var attack_lock := 0.0
var invuln := 0.0
var shoot_cd := 0.0
var foes := {}                  # unit id -> {"state", "cd", "stun", "hp", "home", "wander", "ranged"}
var over := false
var cleared := false
var props_at := {}              # Vector2i -> prop kind (bushes / pots can be cut)
var drop_n := 0
var moving := false
var dbg: Node3D = null
var dbg_on := false
var shot := "fireball"
var test_dir := Vector2.ZERO        # selftest: a held direction
var hero_vel := Vector3.ZERO        # tiles/s on the ground
var foe_vel := {}                   # unit id -> Vector3
var walked := 0.0                   # selftest: ground covered
var anim_cycles := 0.0              # selftest: walk/run cycles played while walking
var max_sink := 0.0                 # selftest: deepest the feet went under the ground


func _ready() -> void:
	add_to_group("pp_debug")
	var b: Node3D = load("res://real3d.tscn").instantiate()
	add_child(b)
	start(b, CONF)


func start(b: Node3D, path: String) -> void:
	super.start(b, path)
	shot = str(data.get("shot", "fireball"))
	speed = float(data.get("speed", speed))
	radius = float(board.conf.get("body_radius", radius))
	follow = hero
	follow_rate = float(data.get("follow_rate", 5.0))
	for p in board.map.get("props", []):
		props_at[Vector2i(int(p["col"]), int(p["row"]))] = str(p["kind"])
	for u in board.units:                       # free movement: units don't hold cells
		board.blocked.erase(Vector2i(int(u["col"]), int(u["row"])))
		if u != hero:
			foes[u["id"]] = {"state": "idle", "cd": 1.0, "stun": 0.0, "hp": int(u["hp"]),
				"home": _pos(u), "wander": Vector3.ZERO, "wt": randf() * 2.0,
				"ranged": str(u["name"]) in data.get("ranged", [])}
	if data.has("title"):
		_banner({"text": str(data["title"]), "time": 1.0, "wait": false})


func _run() -> void:
	pass                                        # no timeline: the player drives


# ------------------------------------------------------------------ ground + collision

func _level(c: int, r: int) -> int:
	return int(board.heights[r][c])


func _cell_ok(c: int, r: int, from_level: int) -> bool:
	if c < 0 or r < 0 or c >= board.cols or r >= board.rows:
		return false
	if board.terrain[r][c] == "water":
		return false
	var ob = board.obstacles.get(Vector2i(c, r), null)
	if ob != null and float(ob["r"]) <= 0.0:          # a house, a wall, a fence: the whole cell
		return false
	return absi(_level(c, r) - from_level) <= 1      # stairs are one level per step


func _level_at(p: Vector3) -> int:
	return _level(clampi(int(floor(p.x)), 0, board.cols - 1), clampi(int(floor(p.z)), 0, board.rows - 1))


func _resolve(p: Vector2, rad: float, from_level: int) -> Vector2:
	# push a circle out of solid cells (boxes) and round props (circles): what's left of a move
	# into a wall runs along it (sliding)
	for _i in 3:
		var c0 := int(floor(p.x))
		var r0 := int(floor(p.y))
		var pushed := false
		for dr in [-1, 0, 1]:
			for dc in [-1, 0, 1]:
				var c: int = c0 + dc
				var r: int = r0 + dr
				if not _cell_ok(c, r, from_level):
					var q := Vector2(clampf(p.x, c, c + 1), clampf(p.y, r, r + 1))
					var d := p - q
					var dist := d.length()
					if dist < rad:
						if dist > 0.00001:
							p += d / dist * (rad - dist)
						else:                       # the centre inside the box: out the short way
							var outs := [Vector2(c - rad - p.x, 0), Vector2(c + 1 + rad - p.x, 0),
								Vector2(0, r - rad - p.y), Vector2(0, r + 1 + rad - p.y)]
							var best: Vector2 = outs[0]
							for o in outs:
								if o.length() < best.length():
									best = o
							p += best
						pushed = true
				else:
					var ob = board.obstacles.get(Vector2i(c, r), null)
					if ob != null and float(ob["r"]) > 0.0:
						var centre := Vector2(c + 0.5, r + 0.5)
						var d2 := p - centre
						var dist2 := d2.length()
						var need := rad + float(ob["r"])
						if dist2 < need:
							p += (d2 / dist2 if dist2 > 0.00001 else Vector2(0, 1)) * (need - dist2)
							pushed = true
		if not pushed:
			break
	return p


func can_stand(p: Vector3, from: Vector3, rad: float = -1.0) -> bool:
	var rr := radius if rad < 0.0 else rad
	var q := _resolve(Vector2(p.x, p.z), rr, _level_at(from))
	return q.distance_to(Vector2(p.x, p.z)) < 0.002


func _slide(u: Dictionary, step: Vector3, rad: float = -1.0) -> bool:
	# move, then push out of whatever it ran into (walls slide it along instead of stopping it)
	var rr := radius if rad < 0.0 else rad
	var p: Vector3 = _pos(u)
	var to := _resolve(Vector2(p.x + step.x, p.z + step.z), rr, _level_at(p))
	if not _cell_ok(int(floor(to.x)), int(floor(to.y)), _level_at(p)):
		return false
	var moved := to.distance_to(Vector2(p.x, p.z)) > 0.0001
	if moved:
		u["node"].position.x = to.x
		u["node"].position.z = to.y
		_sync_cell(u)
		board.blocked.erase(_cell_of(u))
	return moved


# ------------------------------------------------------------------ input

func _input_dir() -> Vector2:
	var v := Vector2.ZERO
	if Input.is_physical_key_pressed(KEY_W) or Input.is_physical_key_pressed(KEY_UP): v.y -= 1
	if Input.is_physical_key_pressed(KEY_S) or Input.is_physical_key_pressed(KEY_DOWN): v.y += 1
	if Input.is_physical_key_pressed(KEY_A) or Input.is_physical_key_pressed(KEY_LEFT): v.x -= 1
	if Input.is_physical_key_pressed(KEY_D) or Input.is_physical_key_pressed(KEY_RIGHT): v.x += 1
	if test_dir != Vector2.ZERO:
		v = test_dir
	return v.normalized()


func _unhandled_input(e: InputEvent) -> void:
	if not (e is InputEventKey) or not e.pressed or e.echo:
		return
	match e.physical_keycode:
		KEY_SPACE, KEY_J:
			_attack_now()
		KEY_K:
			_shoot()
		KEY_R:
			get_tree().reload_current_scene()
		KEY_ESCAPE:
			get_tree().quit()


func _process(delta: float) -> void:
	super._process(delta)
	if board == null or hero == null:
		return
	attack_lock = maxf(0.0, attack_lock - delta)
	invuln = maxf(0.0, invuln - delta)
	shoot_cd = maxf(0.0, shoot_cd - delta)
	if over:
		return
	if not _alive(hero) or int(hero["hp"]) <= 0:
		_lose()
		return
	_move_hero(delta)
	_think(delta)
	_auto_pickup()
	if dbg_on:
		_debug_update()


func _move_hero(delta: float) -> void:
	var v := _input_dir()
	var busy: bool = attack_lock > 0.0 or hero["anim"] == "hurt"
	var wish := Vector3.ZERO
	if v != Vector2.ZERO and not busy:
		# screen directions -> world: the camera looks along -z at yaw 0 (screen down = +z)
		var yaw := deg_to_rad(board.cam_yaw)
		var right := Vector3(cos(yaw), 0, -sin(yaw))
		var down := Vector3(sin(yaw), 0, cos(yaw))
		wish = (right * v.x + down * v.y).normalized() * minf(speed, board.top_speed(hero))
		board.face_to(hero, _snap8(atan2(wish.z, wish.x)))     # turns through the facings
	var c: Dictionary = board.conf
	hero_vel = M.approach(hero_vel, wish, float(c.get("accel", 14.0)), float(c.get("decel", 18.0)) * (2.0 if busy else 1.0), delta)
	var before: Vector3 = _pos(hero)
	if hero_vel.length() > 0.001:
		_slide(hero, hero_vel * delta)
	var after: Vector3 = _pos(hero)
	var moved := Vector2(after.x - before.x, after.z - before.z).length()
	if moved < hero_vel.length() * delta * 0.3:          # stopped by a wall: lose the speed into it
		hero_vel *= 0.5
	var sp := moved / maxf(delta, 0.0001)
	max_sink = maxf(max_sink, board.ground_at(after.x, after.z) - after.y)   # feet under the ground?
	if sp > 0.05 and not busy:
		board.set_locomotion(hero, sp)                 # walk / run at speed / stride: feet planted
		walked += moved
		anim_cycles += float(hero["rate"]) * delta
		moving = true
	elif moving:
		moving = false
		if hero["anim"] in ["walk", "run"]:
			board._play(hero, "idle")                  # settles onto the next footfall


func _front() -> Vector3:
	return Vector3(cos(hero["yaw"]), 0, sin(hero["yaw"]))


func _attack_now() -> void:
	if over or attack_lock > 0.0 or not _alive(hero):
		return
	attack_lock = 0.38
	moving = false
	var fwd := _front()
	var hp: Vector3 = _pos(hero)
	var hits := []
	var drops := {}
	for id in foes:
		var t = _u(id)
		if not _alive(t) or t.get("dying", false):
			continue
		var d: Vector3 = _pos(t) - hp
		d.y = 0
		if d.length() < 0.45 or (d.length() < sword_reach + 0.3 and d.normalized().dot(fwd) > 0.35):
			hits.append(id)
			if int(t["hp"]) <= int(data.get("sword", 3)):
				drops[id] = _drop()
	var cut := []
	var drop := {}
	var at := hp + fwd * 0.75
	for c in [Vector2i(int(floor(at.x)), int(floor(at.z))), Vector2i(int(floor(hp.x + fwd.x * 0.4)), int(floor(hp.z + fwd.z * 0.4)))]:
		if props_at.get(c, "") in ["bush", "pot"] and not cut.has([c.x, c.y]):
			cut.append([c.x, c.y])
			props_at.erase(c)
			if randf() < 0.55:
				drop = _drop()
	var s := {"unit": hero["id"], "effect": "slash", "scale": 1.0, "windup": 0.1, "time": 0.22,
		"hits": hits, "damage": int(data.get("sword", 3)), "knock": 0.9, "spark": "spark", "hit_light": 0.6,
		"drops": drops}
	if hits.size() > 0:
		s["shake"] = 0.035
	if cut.size() > 0:
		s["cut"] = cut
		s["drop"] = drop
	for id in hits:
		foes[id]["stun"] = 0.5
	_swing(s)


func _drop() -> Dictionary:
	drop_n += 1
	var kind := "heart" if randf() < 0.3 else "coin"
	return {"id": "drop%d" % drop_n, "kind": kind, "value": 1}


func _shoot() -> void:
	if over or shoot_cd > 0.0 or attack_lock > 0.0 or not _alive(hero):
		return
	shoot_cd = 0.55
	attack_lock = 0.25
	moving = false
	var fwd := _front()
	var from: Vector3 = _pos(hero) + Vector3(0, 0.55, 0) + fwd * 0.3
	var start: Vector3 = _pos(hero)
	var to := from
	var target = null
	var p := start
	for i in range(1, 29):                      # up to 7 cells, a quarter cell per step
		var q := start + fwd * (0.25 * i)
		if not can_stand(q, p, 0.05):
			break
		p = q
		to = Vector3(q.x, from.y + (_ground(q.x, q.z) - _ground(start.x, start.z)), q.z)
		for id in foes:
			var t = _u(id)
			if _alive(t) and not t.get("dying", false):
				var d: Vector3 = _pos(t) - q
				d.y = 0
				if d.length() < 0.45:
					target = t
					break
		if target != null:
			break
	board._play(hero, "attack", true)
	var col := Color(1.0, 0.55, 0.2) if shot == "fireball" else Color(1.0, 0.9, 0.6)
	await _projectile({"projectile": shot, "projectile_tag": "fly", "speed": 9.0, "arc": 0.08 if shot == "fireball" else 0.25,
		"projectile_scale": 0.6}, from, to, col)
	if hero != null and hero["anim"] == "attack":
		board._play(hero, "idle")
	if shot == "fireball":
		board.play_effect("explosion", to - Vector3(0, 0.45, 0), false, 1.6, "", 0.45)
	board.spell_light(to, col, 2.2, 0.35, 3.0, 10)
	if target != null and _alive(target) and not target.get("dying", false):
		foes[target["id"]]["stun"] = 0.6
		_hit(target, int(data.get("shot_damage", 2)), _pos(hero), 0.6)
		if int(target["hp"]) <= 0:
			_die(target, _drop())


# ------------------------------------------------------------------ enemies

func _think(delta: float) -> void:
	var hp: Vector3 = _pos(hero)
	var alive := 0
	for id in foes:
		var t = _u(id)
		if not _alive(t) or t.get("dying", false):
			continue
		alive += 1
		var f: Dictionary = foes[id]
		f["cd"] = maxf(0.0, f["cd"] - delta)
		if int(t["hp"]) < int(f["hp"]):
			f["hp"] = int(t["hp"])
			f["stun"] = maxf(f["stun"], 0.45)
		if f["stun"] > 0.0:
			f["stun"] -= delta
			foe_vel[id] = Vector3.ZERO
			continue
		var d: Vector3 = hp - _pos(t)
		d.y = 0
		var dist := d.length()
		var sees := dist < float(data.get("aggro", 5.5)) and absf(_ground(hp.x, hp.z) - _ground(_pos(t).x, _pos(t).z)) < 1.2
		f["state"] = "chase" if sees or (f["state"] == "chase" and dist < 8.0) else "idle"
		var run_anim: String = "run" if "run" in t["sheet"]["anims"] else "walk"
		if f["state"] == "chase":
			if f["ranged"]:
				if dist < 6.0 and f["cd"] <= 0.0:
					f["cd"] = 2.2
					_enemy_shot(t)
				elif dist > 3.0:
					_walk_toward(t, d.normalized(), 1.8, "walk", delta)
				elif t["anim"] in ["walk", "run"]:
					_stop_foe(t)
				board.face_to(t, _snap8(atan2(d.z, d.x)))
			elif dist > 0.7:
				_walk_toward(t, d.normalized(), float(data.get("foe_speed", 2.5)), run_anim, delta)
			else:
				board.face_to(t, _snap8(atan2(d.z, d.x)))
				_stop_foe(t)
				if f["cd"] <= 0.0 and invuln <= 0.0:
					f["cd"] = 1.1
					invuln = 1.0
					_hurt({"unit": hero["id"], "by": id, "damage": int(data.get("bite", 2)), "knock": 0.6, "time": 0.2})
				elif t["anim"] in ["walk", "run"]:
					_stop_foe(t)
		else:
			f["wt"] -= delta
			if f["wt"] <= 0.0:
				f["wt"] = randf_range(1.2, 2.6)
				var a := randf() * TAU
				f["wander"] = Vector3(cos(a), 0, sin(a)) if randf() < 0.6 else Vector3.ZERO
				if (_pos(t) - f["home"]).length() > 2.0:
					f["wander"] = (f["home"] - _pos(t)).normalized()
			if f["wander"] != Vector3.ZERO:
				_walk_toward(t, f["wander"], 0.9, "walk", delta)
			elif t["anim"] in ["walk", "run"]:
				_stop_foe(t)
	if alive == 0 and not cleared and foes.size() > 0:
		cleared = true
		_banner({"text": "AREA CLEAR!", "time": 1.6, "wait": false, "colour": [1.0, 0.8, 0.3]})


func _walk_toward(t: Dictionary, dir: Vector3, spd: float, anim: String, delta: float) -> void:
	board.face_to(t, _snap8(atan2(dir.z, dir.x)))
	var c: Dictionary = board.conf
	var v: Vector3 = foe_vel.get(t["id"], Vector3.ZERO)
	v = M.approach(v, dir * minf(spd, board.top_speed(t)), float(c.get("accel", 14.0)), float(c.get("decel", 18.0)), delta)
	foe_vel[t["id"]] = v
	var before: Vector3 = _pos(t)
	if not _slide(t, v * delta, 0.2):
		foes[t["id"]]["wander"] = Vector3.ZERO
		foe_vel[t["id"]] = Vector3.ZERO
	var after: Vector3 = _pos(t)
	var sp := Vector2(after.x - before.x, after.z - before.z).length() / maxf(delta, 0.0001)
	if t["anim"] != "hurt" and t["anim"] != "attack":
		if sp > 0.05:
			board.set_locomotion(t, sp)
		elif t["anim"] in ["walk", "run"]:
			board._play(t, "idle")


func _stop_foe(t: Dictionary) -> void:
	foe_vel[t["id"]] = Vector3.ZERO
	if t["anim"] in ["walk", "run"]:
		board._play(t, "idle")


func _enemy_shot(t: Dictionary) -> void:
	var from: Vector3 = _pos(t) + Vector3(0, 0.55, 0)
	var aim: Vector3 = _pos(hero) + Vector3(0, 0.5, 0)
	if "attack" in t["sheet"]["anims"]:
		board._play(t, "attack", true)
	await _sleep(0.3)
	if not _alive(t) or hero == null:
		return
	await _projectile({"projectile": "arrow", "speed": 8.0, "arc": 0.35}, from, aim, Color(1.0, 0.5, 0.3))
	if _alive(t) and t["anim"] == "attack":
		board._play(t, "idle")
	if hero != null and _alive(hero) and invuln <= 0.0 and (_pos(hero) + Vector3(0, 0.5, 0)).distance_to(aim) < 0.6:
		invuln = 1.0
		_hit(hero, 1, from, 0.4, "", Color(1.0, 0.5, 0.45))
		_blink(hero, 0.9)


func _auto_pickup() -> void:
	var hp: Vector3 = _pos(hero)
	for id in pickups.keys():
		var pk: Dictionary = pickups[id]
		if not pk["bob"]:
			continue
		var p: Vector3 = pk["node"].position
		if Vector2(p.x - hp.x, p.z - hp.z).length() < 0.55:
			_pickup({"id": id, "sparkle": "sparkle", "sparkle_scale": 0.6})


func _lose() -> void:
	over = true
	if _alive(hero) and "death" in hero["sheet"]["anims"]:
		board._play(hero, "death", true)
	_banner({"text": "GAME OVER - R to retry", "time": 99.0, "wait": false, "colour": [0.9, 0.25, 0.2]})


# ------------------------------------------------------------------ PPDebug (F3 / F12)

func _frame_of(u: Dictionary) -> String:
	var s: Dictionary = u["sheet"]
	var anim: String = u["anim"] if u["anim"] in s["anims"] else "idle"
	var key := "%s_%s" % [anim, board._sprite_dir(u["yaw"])]
	if not s["frames"].has(key):
		return key + " (missing)"
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


func debug_state() -> Dictionary:
	if board == null or hero == null or not _alive(hero):
		return {"hero": "gone"}
	var p: Vector3 = _pos(hero)
	var c := _cell_of(hero)
	var d := {"hero": "%s hp %d/%d  coins %d" % [hero["id"], hero["hp"], hero["max_hp"], coins],
		"hero motion": "speed %.2f t/s  %s at %.2f cycles/s (stride %.2f t)  phase %.2f" % [hero_vel.length(), hero.get("loco", ""), float(hero["rate"]), M.stride_of(board.strides(hero), str(hero.get("loco", "walk")), hero["sheet"]["body"]), float(hero["phase"])],
		"hero pos": "x %.2f  z %.2f  y(feet) %.2f  ground %.2f" % [p.x, p.z, p.y, _ground(p.x, p.z)],
		"hero cell": "%s level %d %s" % [c, _level(c.x, c.y), board.terrain[c.y][c.x]],
		"hero facing": "world %.0f deg -> sprite %s" % [rad_to_deg(hero["yaw"]), board._sprite_dir(hero["yaw"])],
		"hero anim": _frame_of(hero),
		"locks": "attack %.2f  shoot %.2f  invulnerable %.2f" % [attack_lock, shoot_cd, invuln],
		"camera": "yaw %.0f pitch %.0f dist %.1f" % [board.cam_yaw, board.cam_pitch, board.cam_dist],
		"pickups on the ground": pickups.size()}
	var near = null
	var nd := 999.0
	for id in foes:
		var t = _u(id)
		if _alive(t):
			var dd: float = (_pos(t) - p).length()
			if dd < nd:
				nd = dd
				near = t
	if near != null:
		var f: Dictionary = foes[near["id"]]
		d["nearest foe"] = "%s %s hp %d dist %.2f stun %.2f" % [near["id"], f["state"], near["hp"], nd, maxf(0.0, f["stun"])]
		d["nearest foe anim"] = _frame_of(near)
	d["foes left"] = foes.keys().filter(func(id): return _alive(_u(id))).size()
	return d


func debug_draw(show: bool) -> void:
	dbg_on = show
	if is_instance_valid(dbg):
		dbg.queue_free()
	dbg = null
	if not show or board == null:
		return
	dbg = Node3D.new()
	board.add_child(dbg)
	var red := _dbg_mat(Color(1.0, 0.15, 0.1, 0.35))
	for cell in board.blocked:
		var m := MeshInstance3D.new()
		var q := QuadMesh.new(); q.size = Vector2(0.95, 0.95); q.orientation = PlaneMesh.FACE_Y
		m.mesh = q
		m.material_override = red
		m.position = board.cell_pos(cell.x, cell.y) + Vector3(0, 0.03, 0)
		dbg.add_child(m)
	_debug_update()


func _dbg_mat(col: Color) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.albedo_color = col
	m.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	m.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	m.no_depth_test = true
	return m


func _debug_update() -> void:
	if not is_instance_valid(dbg):
		return
	for n in dbg.get_children():
		if n.has_meta("live"):
			n.free()
	var list := [[hero, radius, Color(0.3, 1.0, 0.4, 0.6)]]
	for id in foes:
		var t = _u(id)
		if _alive(t):
			list.append([t, 0.2, Color(1.0, 0.5, 0.2, 0.6)])
	for e in list:
		var box := MeshInstance3D.new()
		var q := QuadMesh.new(); q.size = Vector2(e[1] * 2.0, e[1] * 2.0); q.orientation = PlaneMesh.FACE_Y
		box.mesh = q
		box.material_override = _dbg_mat(e[2])
		box.position = _pos(e[0]) + Vector3(0, 0.06, 0)
		box.set_meta("live", true)
		dbg.add_child(box)
	var arc := MeshInstance3D.new()
	var tm := TorusMesh.new(); tm.inner_radius = sword_reach - 0.03; tm.outer_radius = sword_reach
	arc.mesh = tm
	arc.material_override = _dbg_mat(Color(1.0, 1.0, 0.3, 0.5))
	arc.position = _pos(hero) + Vector3(0, 0.07, 0)
	arc.set_meta("live", true)
	dbg.add_child(arc)


# ------------------------------------------------------------------ checks (PPDebug --pp-selftest / --pp-shot)

func _hold(dir: Vector2, t: float) -> void:
	test_dir = dir
	await _sleep(t)
	test_dir = Vector2.ZERO


func selftest() -> Dictionary:
	await _sleep(0.5)
	var out := {}
	var p0: Vector3 = _pos(hero)
	await _hold(Vector2(1, 0), 0.8)                # walk east along the road
	out["walked_east"] = snappedf(_pos(hero).x - p0.x, 0.01)
	var p1: Vector3 = _pos(hero)
	await _hold(Vector2(0, 1), 1.2)                # south: the map edge / pots stop it
	out["walked_south"] = snappedf(_pos(hero).z - p1.z, 0.01)
	await _hold(Vector2(-1, -1), 0.6)              # diagonal
	out["facing_after_diagonal"] = board._sprite_dir(hero["yaw"])
	var bushes := props_at.values().count("bush") + props_at.values().count("pot")
	_attack_now()
	await _sleep(0.5)
	_shoot()
	await _sleep(1.2)
	# walk north toward the wolves and fight whatever comes
	var hits := 0
	for i in 8:
		await _hold(Vector2(0, -1), 0.35)
		_attack_now()
		await _sleep(0.45)
	out["hero_hp"] = int(hero["hp"])
	out["foes_alive"] = foes.keys().filter(func(id): return _alive(_u(id))).size()
	out["foes_chasing"] = foes.values().filter(func(f): return f["state"] == "chase").size()
	out["cuttable_left"] = props_at.values().count("bush") + props_at.values().count("pot")
	# footsteps: cycles played per tile walked = 1 / stride (the feet stayed planted)
	out["walked_tiles"] = snappedf(walked, 0.01)
	out["cycles_per_tile"] = snappedf(anim_cycles / maxf(walked, 0.001), 0.001)
	var st: Dictionary = board.strides(hero)
	var used: String = str(hero.get("loco", "walk"))
	out["stride_" + used] = snappedf(M.stride_of(st, used, hero["sheet"]["body"]), 0.001)
	out["max_sink"] = snappedf(max_sink, 0.001)
	out["obstacles"] = board.obstacles.size()
	out["cuttable_before"] = bushes
	out["coins"] = coins
	out["over"] = over
	return {"action": out}


func shot_setup() -> void:
	await _hold(Vector2(1, 0), 0.7)
	await _hold(Vector2(0, -1), 0.15)
	_attack_now()
	await _sleep(0.12)

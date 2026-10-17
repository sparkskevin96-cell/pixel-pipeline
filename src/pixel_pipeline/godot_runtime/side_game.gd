extends Node2D
## Side-scroller level (pixel-pipeline godot_runtime). Everything comes from the app's export
## (res://art/pixel_pipeline_manifest.json) by id, laid out by res://level.json (playtest.py
## platformer_level): the tileset's ground painted with Godot terrains (set 0, terrain 0) + its
## one-way platform tiles, the parallax background scene, the hero and enemy SpriteFrames, effect
## SpriteFrames, the UI kit's icons + font for the HUD.

const Manifest := preload("res://pp_runtime/pp_manifest.gd")
const Player := preload("res://pp_runtime/side_player.gd")
const Enemy := preload("res://pp_runtime/side_enemy.gd")
const Pickup := preload("res://pp_runtime/pickup_2d.gd")

var lv := {}
var man
var tile := 32
var tiles: TileMapLayer
var player: CharacterBody2D
var enemies: Array = []
var cam: Camera2D
var coins := 0
var hud: CanvasLayer
var hearts: HBoxContainer
var coin_label: Label
var banner: Label
var ui_dir := ""
var font: Font = null
var over := false
var won := false
var arrows: Array = []          # {node, vel, life}
var fx_frames := {}
var start_time := 0.0


func _ready() -> void:
	add_to_group("pp_debug")
	lv = JSON.parse_string(FileAccess.get_file_as_string("res://level.json"))
	man = Manifest.new("res://art")
	tile = int(lv.get("tile", 32))
	_background()
	_tiles()
	_pickups()
	_player()
	_enemies()
	_flag()
	_camera()
	_hud()
	start_time = Time.get_ticks_msec() / 1000.0


func _background() -> void:
	var e: Dictionary = man.entry(str(lv.get("background", "")))
	var p: String = man.res(e, "scene")
	if p != "" and ResourceLoader.exists(p):
		var bg: Node = load(p).instantiate()
		add_child(bg)


func _tiles() -> void:
	var e: Dictionary = man.entry(str(lv.get("tileset", "")))
	tiles = TileMapLayer.new()
	tiles.name = "Tiles"
	var ts_path: String = man.res(e, "tileset")
	if ts_path != "" and ResourceLoader.exists(ts_path):
		tiles.tile_set = load(ts_path)
	else:
		push_error("side_game: no tileset " + ts_path)
		return
	add_child(tiles)
	var cells: Array[Vector2i] = []
	for c in lv["solid"]:
		cells.append(Vector2i(int(c[0]), int(c[1])))
	tiles.set_cells_terrain_connect(cells, 0, 0, false)
	var src := tiles.tile_set.get_source_id(0)
	var at: Dictionary = lv.get("platform_atlas", {})
	for p in lv.get("platforms", []):
		var a = at.get(str(p[2]), [0, 4])
		tiles.set_cell(Vector2i(int(p[0]), int(p[1])), src, Vector2i(int(a[0]), int(a[1])))


func _cell_feet(c) -> Vector2:
	return Vector2(float(c[0]) * tile + tile / 2.0, float(c[1] + 1) * tile)


func _frames_of(id: String) -> Array:
	var e: Dictionary = man.entry(id)
	return [e, man.sprite_frames(e)]


func _player() -> void:
	var fe := _frames_of(str(lv.get("hero", "")))
	var e: Dictionary = fe[0]
	player = Player.new()
	player.name = "Player"
	var fs: Vector2 = man.frame_size(e)
	player.setup(self, fe[1], man.anchor(e), fs, fs.y * 0.82)
	for clip in ["walk", "run"]:                      # stride (px per cycle): feet planted
		var st = e.get("animations", {}).get(clip, {}).get("stride", {})
		if st is Dictionary and st.has("px"):
			player.strides[clip] = float(st["px"])
	player.position = _cell_feet(lv["start"])
	add_child(player)


func _enemies() -> void:
	var kinds: Dictionary = lv.get("enemy_kinds", {})
	var i := 0
	for spec in lv.get("enemies", []):
		var fe := _frames_of(str(kinds.get(str(spec["kind"]), "")))
		var e: Dictionary = fe[0]
		if fe[1] == null:
			continue
		var en := Enemy.new()
		i += 1
		en.name = "%s%d" % [spec["kind"], i]
		var fs: Vector2 = man.frame_size(e)
		en.setup(self, fe[1], man.anchor(e), fs, Vector2(clampf(fs.x * 0.5, 14, 60), clampf(fs.y * 0.6, 12, 48)))
		en.position = _cell_feet([spec["x"], spec["y"]])
		add_child(en)
		enemies.append(en)


func _icon(name: String) -> Texture2D:
	if ui_dir == "":
		var e: Dictionary = man.entry(str(lv.get("ui", "")))
		ui_dir = man.folder(e)
	return man.texture(ui_dir.path_join("pieces/%s.png" % name))


func _pickups() -> void:
	for c in lv.get("coins", []):
		_spawn_pickup("coin", Vector2(float(c[0]) * tile + tile / 2.0, float(c[1]) * tile + tile / 2.0))
	for c in lv.get("hearts", []):
		_spawn_pickup("heart", Vector2(float(c[0]) * tile + tile / 2.0, float(c[1]) * tile + tile / 2.0))


func _spawn_pickup(kind: String, at: Vector2) -> void:
	var p := Pickup.new()
	p.setup(kind, _icon("icon_" + kind), 1)
	p.position = at
	p.body_entered.connect(_on_pickup.bind(p))
	add_child(p)


func _on_pickup(body: Node, p) -> void:
	if body != player or p.taken or player.dead:
		return
	if p.kind == "heart":
		player.heal(2)
	else:
		coins += p.value
	play_fx("sparkle", p.global_position, 0.8)
	p.take()
	_update_hud()


func _flag() -> void:
	var f := Area2D.new()
	f.name = "Flag"
	f.collision_mask = 2
	f.position = _cell_feet(lv["flag"])
	var pole := Line2D.new()
	pole.points = PackedVector2Array([Vector2(0, 0), Vector2(0, -88)])
	pole.width = 3
	pole.default_color = Color(0.85, 0.85, 0.9)
	f.add_child(pole)
	var cloth := Polygon2D.new()
	cloth.polygon = PackedVector2Array([Vector2(2, -88), Vector2(30, -78), Vector2(2, -68)])
	cloth.color = Color(0.95, 0.3, 0.25)
	f.add_child(cloth)
	var c := CollisionShape2D.new()
	var r := RectangleShape2D.new()
	r.size = Vector2(16, 88)
	c.shape = r
	c.position = Vector2(0, -44)
	f.add_child(c)
	f.body_entered.connect(func(b):
		if b == player and not won and not over:
			won = true
			player.test_input = {"x": 0}
			_banner("LEVEL CLEAR!  %d coins  (R to play again)" % coins, Color(1.0, 0.85, 0.35)))
	add_child(f)


func _camera() -> void:
	cam = Camera2D.new()
	cam.position_smoothing_enabled = true
	cam.position_smoothing_speed = 7.0
	cam.limit_left = 0
	cam.limit_top = 0
	cam.limit_right = int(lv["cols"]) * tile
	cam.limit_bottom = int(lv["rows"]) * tile
	cam.offset = Vector2(0, -24)
	player.add_child(cam)
	cam.make_current()


func _hud() -> void:
	hud = CanvasLayer.new()
	hud.layer = 10
	add_child(hud)
	var e: Dictionary = man.entry(str(lv.get("ui", "")))
	var fnt: String = man.folder(e).path_join("fonts/pixelkit_1x.fnt")
	if ResourceLoader.exists(fnt):
		font = load(fnt)
	hearts = HBoxContainer.new()
	hearts.position = Vector2(8, 6)
	hearts.add_theme_constant_override("separation", 1)
	hud.add_child(hearts)
	var ci := TextureRect.new()
	ci.texture = _icon("icon_coin")
	ci.position = Vector2(560, 6)
	hud.add_child(ci)
	coin_label = _label("000", Vector2(580, 9))
	hud.add_child(coin_label)
	banner = _label("", Vector2(0, 150))
	banner.size = Vector2(640, 40)
	banner.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	banner.visible = false
	hud.add_child(banner)
	_update_hud()


func _label(text: String, pos: Vector2) -> Label:
	var l := Label.new()
	l.text = text
	l.position = pos
	if font != null:
		l.add_theme_font_override("font", font)
	l.add_theme_font_size_override("font_size", 10)
	l.add_theme_color_override("font_color", Color(1.0, 0.96, 0.86))
	l.add_theme_color_override("font_shadow_color", Color(0.1, 0.05, 0.03, 0.9))
	l.add_theme_constant_override("shadow_offset_x", 1)
	l.add_theme_constant_override("shadow_offset_y", 1)
	return l


func _update_hud() -> void:
	if hearts == null:
		return
	for c in hearts.get_children():
		c.queue_free()
	for i in int(ceil(player.max_hp / 2.0)):
		var v: int = player.hp - i * 2
		var r := TextureRect.new()
		r.texture = _icon("icon_heart" if v >= 2 else ("icon_heart_half" if v == 1 else "icon_heart_empty"))
		hearts.add_child(r)
	coin_label.text = "%03d" % coins


func _banner(text: String, col: Color) -> void:
	banner.text = text
	banner.add_theme_color_override("font_color", col)
	banner.visible = true


# ------------------------------------------------------------------ play

func _unhandled_input(e: InputEvent) -> void:
	if e is InputEventKey and e.pressed and not e.echo:
		match e.physical_keycode:
			KEY_R:
				get_tree().reload_current_scene()
			KEY_ESCAPE:
				get_tree().quit()


func _physics_process(delta: float) -> void:
	if player == null:
		return
	_contacts()
	_move_arrows(delta)
	if not over and player.global_position.y > float(lv["rows"]) * tile + 48.0:
		_fell()
	if not over and player.dead:
		over = true
		_banner("GAME OVER  -  R to retry", Color(1.0, 0.4, 0.35))


func _rect(b: CharacterBody2D, w: float, h: float) -> Rect2:
	return Rect2(b.global_position.x - w / 2.0, b.global_position.y - h, w, h)


func _contacts() -> void:
	if player.dead or won:
		return
	var ps: Vector2 = player.shape.shape.size
	var pr := _rect(player, ps.x, ps.y)
	for en in enemies:
		if not is_instance_valid(en) or en.dead:
			continue
		var er := _rect(en, en.size.x, en.size.y)
		if not pr.intersects(er):
			continue
		if player.velocity.y > 40.0 and pr.end.y < er.position.y + er.size.y * 0.55:
			en.take_hit(2, player.global_position.x)          # stomp
			player.bounce()
			play_fx("spark", Vector2(en.global_position.x, er.position.y), 0.9)
		elif player.invuln <= 0.0:
			player.take_hit(1, en.global_position.x)
			play_fx("spark", (pr.get_center() + er.get_center()) / 2.0, 0.7)
			_update_hud()


func player_shot(p) -> void:
	var a := Node2D.new()
	var line := Line2D.new()
	line.points = PackedVector2Array([Vector2(-9, 0), Vector2(9, 0)])
	line.width = 2
	line.default_color = Color(0.92, 0.84, 0.62)
	a.add_child(line)
	var tip := Polygon2D.new()
	tip.polygon = PackedVector2Array([Vector2(9, -2), Vector2(13, 0), Vector2(9, 2)])
	tip.color = Color(0.85, 0.9, 1.0)
	a.add_child(tip)
	a.scale.x = float(p.facing)
	a.position = p.global_position + Vector2(p.facing * 12.0, -p.body_h * 0.55)
	add_child(a)
	arrows.append({"node": a, "vel": Vector2(p.facing * 380.0, 0), "life": 1.2})


func _move_arrows(delta: float) -> void:
	var space := get_world_2d().direct_space_state
	for ar in arrows.duplicate():
		var n: Node2D = ar["node"]
		ar["life"] -= delta
		var step: Vector2 = ar["vel"] * delta
		var q := PhysicsRayQueryParameters2D.create(n.global_position, n.global_position + step * 1.5)
		q.collision_mask = 1
		var hit := space.intersect_ray(q)
		var done: bool = ar["life"] <= 0.0 or not hit.is_empty()
		for en in enemies:
			if is_instance_valid(en) and not en.dead and _rect(en, en.size.x, en.size.y).grow(3).has_point(n.global_position):
				en.take_hit(1, n.global_position.x - signf(step.x) * 10.0)
				play_fx("spark", n.global_position, 0.7)
				done = true
				break
		if done:
			arrows.erase(ar)
			n.queue_free()
		else:
			n.position += step


func enemy_died(en) -> void:
	await get_tree().create_timer(0.8, false).timeout
	if not is_instance_valid(en):
		return
	play_fx("poof", en.global_position + Vector2(0, -8), 1.0)
	_spawn_pickup("coin", en.global_position + Vector2(0, -14))
	en.queue_free()


func _fell() -> void:
	player.take_hit(1, player.global_position.x)
	_update_hud()
	if player.dead:
		return
	player.global_position = player.last_safe + Vector2(-float(player.facing) * 24.0, -4.0)
	player.velocity = Vector2.ZERO


func play_fx(kind: String, at: Vector2, scale_k := 1.0) -> void:
	var id := str(lv.get("effects", {}).get(kind, ""))
	if not fx_frames.has(kind):
		fx_frames[kind] = man.sprite_frames(man.entry(id)) if id != "" else null
	var sf: SpriteFrames = fx_frames[kind]
	if sf == null:
		return
	var s := AnimatedSprite2D.new()
	s.sprite_frames = sf
	s.position = at
	s.scale = Vector2(scale_k, scale_k)
	s.z_index = 5
	var names := sf.get_animation_names()
	if names.is_empty():
		return
	add_child(s)
	s.play(names[0])
	s.animation_finished.connect(s.queue_free)
	var wr: WeakRef = weakref(s)
	get_tree().create_timer(1.5, false).timeout.connect(func():
		var n = wr.get_ref()
		if n != null:
			n.queue_free())


# ------------------------------------------------------------------ PPDebug + checks

func debug_state() -> Dictionary:
	if player == null:
		return {}
	var d: Dictionary = player.debug_text()
	var c := tiles.local_to_map(player.global_position + Vector2(0, -2)) if tiles != null else Vector2i.ZERO
	d["player cell"] = c
	d["coins"] = coins
	d["camera"] = cam.get_screen_center_position().round() if cam != null else Vector2.ZERO
	var near = null
	var nd := 1e9
	for en in enemies:
		if is_instance_valid(en):
			var dd: float = en.global_position.distance_to(player.global_position)
			if dd < nd:
				nd = dd
				near = en
	if near != null:
		d["nearest enemy"] = near.debug_text()
	d["enemies left"] = enemies.filter(func(x): return is_instance_valid(x) and not x.dead).size()
	d["arrows"] = arrows.size()
	return d


func _wait(t: float) -> void:
	await get_tree().create_timer(t, false).timeout


func selftest() -> Dictionary:
	var out := {}
	await _wait(0.4)
	out["on_floor_at_start"] = player.is_on_floor()
	var x0: float = player.global_position.x
	player.test_input = {"x": 1}
	await _wait(1.0)
	out["ran_px"] = int(player.global_position.x - x0)
	player.jump_now()
	player.test_input = {"x": 1, "jump": true}
	var y0: float = player.global_position.y
	var top := y0
	for i in 40:
		await get_tree().physics_frame
		top = minf(top, player.global_position.y)
	out["jump_height_px"] = int(y0 - top)
	player.test_input = {"x": 1}
	await _wait(1.5)
	player.attack()
	await _wait(0.8)
	player.test_input = {"x": 1, "sprint": true}
	for i in 6:
		await _wait(0.5)
		if player.is_on_floor() and player.is_on_wall():
			player.jump_now()
			player.test_input = {"x": 1, "sprint": true, "jump": true}
		else:
			player.test_input = {"x": 1, "sprint": true}
	player.test_input = {}
	out["player_x"] = int(player.global_position.x)
	out["hp"] = player.hp
	out["coins"] = coins
	out["enemies_left"] = enemies.filter(func(x): return is_instance_valid(x) and not x.dead).size()
	out["tiles_painted"] = tiles.get_used_cells().size() if tiles != null else 0
	out["anim"] = str(player.sprite.animation)
	return {"platformer": out}


func shot_setup() -> void:
	player.test_input = {"x": 1}
	await _wait(1.6)
	player.jump_now()
	player.test_input = {"x": 1, "jump": true}
	await _wait(0.25)
	player.test_input = {}

extends CharacterBody2D
## Platformer hero (pixel-pipeline godot_runtime). Built by side_game.gd from a manifest
## character: an AnimatedSprite2D with its SpriteFrames (<action>_E / _W), feet on the body's
## bottom edge. Run / sprint, jump with coyote time, jump buffer and variable height, attack
## (shoots an arrow), stomp, hurt + knockback + blink, death.

signal died
signal hurt_by(amount: int)

const GRAVITY := 1150.0
const WALK := 110.0
const SPRINT := 175.0
const ACCEL := 1500.0
const FRICTION := 1800.0
const AIR_ACCEL := 1000.0
const JUMP := -395.0
const JUMP_CUT := 0.45
const MAX_FALL := 540.0
const COYOTE := 0.1
const BUFFER := 0.12

var game: Node = null
var sprite: AnimatedSprite2D
var shape: CollisionShape2D
var facing := 1                 # 1 = E, -1 = W
var action := "idle"
var hp := 6
var max_hp := 6
var coyote := 0.0
var buffer := 0.0
var invuln := 0.0
var attack_t := 0.0
var shot_done := true
var dead := false
var stun := 0.0
var body_h := 40.0
var anims := {}
var test_input := {}            # selftest: {"x": -1..1, "jump": bool, "sprint": bool}
var last_safe := Vector2.ZERO
var u := 1.0                    # tile / 32: speeds, jump and gravity scale with the pixel grid
var strides := {}               # {walk: px per cycle, run: ...} (manifest animations[clip].stride.px)
var stopping := 0.0             # settle: the walk finishes its step before idle


func setup(g: Node, frames: SpriteFrames, feet: Vector2, frame: Vector2, height_px: float) -> void:
	game = g
	u = float(g.tile) / 32.0 if "tile" in g else 1.0
	collision_layer = 2
	collision_mask = 1
	floor_snap_length = 6.0
	sprite = AnimatedSprite2D.new()
	sprite.sprite_frames = frames
	sprite.centered = false
	sprite.offset = -feet                 # feet at the body origin (its bottom centre)
	add_child(sprite)
	body_h = clampf(height_px * 0.78, 16.0, 120.0)
	shape = CollisionShape2D.new()
	var r := RectangleShape2D.new()
	r.size = Vector2(clampf(height_px * 0.26, 10.0, 30.0), body_h)
	shape.shape = r
	shape.position = Vector2(0, -body_h / 2.0)
	add_child(shape)
	for n in frames.get_animation_names():
		anims[str(n)] = true
	_play("idle")


func _has(a: String) -> bool:
	return anims.has("%s_E" % a)


func _play(a: String, restart := false) -> void:
	var want := a if _has(a) else ("walk" if a == "run" and _has("walk") else "idle")
	var name := "%s_%s" % [want, "E" if facing > 0 else "W"]
	if not anims.has(name):
		return
	action = a
	if restart or str(sprite.animation) != name:
		var keep := sprite.frame if str(sprite.animation).begins_with(want + "_") and not restart else 0
		sprite.play(name)
		sprite.frame = keep


func _key(keys: Array) -> bool:
	for k in keys:
		if Input.is_physical_key_pressed(k):
			return true
	return false


func _axis() -> float:
	if test_input.has("x"):
		return float(test_input["x"])
	return float(_key([KEY_D, KEY_RIGHT])) - float(_key([KEY_A, KEY_LEFT]))


func _jump_held() -> bool:
	return bool(test_input.get("jump", false)) or _key([KEY_SPACE, KEY_W, KEY_UP])


func _unhandled_input(e: InputEvent) -> void:
	if dead or not (e is InputEventKey) or not e.pressed or e.echo:
		return
	match e.physical_keycode:
		KEY_SPACE, KEY_W, KEY_UP:
			buffer = BUFFER
		KEY_J, KEY_K:
			attack()


func jump_now() -> void:
	buffer = BUFFER


func attack() -> void:
	if dead or attack_t > 0.0 or stun > 0.0:
		return
	attack_t = _anim_len("attack") if _has("attack") else 0.35
	attack_t = clampf(attack_t, 0.3, 0.7)
	shot_done = false
	sprite.speed_scale = 1.0
	_play("attack", true)


func _anim_len(a: String) -> float:
	var name := "%s_E" % a
	if not anims.has(name):
		return 0.0
	var sf := sprite.sprite_frames
	var total := 0.0
	for i in sf.get_frame_count(name):
		total += sf.get_frame_duration(name, i)
	return total / maxf(0.1, sf.get_animation_speed(name))


func _physics_process(delta: float) -> void:
	if dead:
		velocity.y = minf(velocity.y + GRAVITY * u * delta, MAX_FALL * u)
		velocity.x = move_toward(velocity.x, 0, FRICTION * delta)
		move_and_slide()
		return
	invuln = maxf(0.0, invuln - delta)
	stun = maxf(0.0, stun - delta)
	sprite.visible = invuln <= 0.0 or int(invuln * 14.0) % 2 == 0
	var x := 0.0 if stun > 0.0 else _axis()
	var sprint := bool(test_input.get("sprint", false)) or _key([KEY_SHIFT])
	var top := (SPRINT if sprint else WALK) * u
	if is_on_floor():
		coyote = COYOTE
		last_safe = global_position
	else:
		coyote = maxf(0.0, coyote - delta)
	buffer = maxf(0.0, buffer - delta)
	if x != 0.0:
		facing = 1 if x > 0.0 else -1
		velocity.x = move_toward(velocity.x, x * top, (ACCEL if is_on_floor() else AIR_ACCEL) * u * delta)
	elif stun <= 0.0:
		velocity.x = move_toward(velocity.x, 0.0, (FRICTION if is_on_floor() else AIR_ACCEL * 0.5) * u * delta)
	velocity.y = minf(velocity.y + GRAVITY * u * delta, MAX_FALL * u)
	if buffer > 0.0 and coyote > 0.0:
		velocity.y = JUMP * u
		buffer = 0.0
		coyote = 0.0
	if velocity.y < 0.0 and not _jump_held():
		velocity.y = maxf(velocity.y, JUMP * u * JUMP_CUT)          # let go early: a short hop
	move_and_slide()
	# shooting: the arrow leaves halfway through the attack clip
	if attack_t > 0.0:
		var full := clampf(_anim_len("attack"), 0.3, 0.7) if _has("attack") else 0.35
		attack_t -= delta
		if not shot_done and attack_t <= full * 0.5:
			shot_done = true
			if game != null and game.has_method("player_shot"):
				game.player_shot(self)
	_animate(x)


func _rate_for(clip: String, speed_px: float) -> float:
	# speed_scale that plays the clip at speed / stride cycles per second (feet planted), kept
	# inside a sane cadence (a short-legged rig can't keep up with a fast platformer: it then
	# plays at its top cadence and the feet slide a little instead of the legs blurring)
	var name := "%s_%s" % [clip, "E" if facing > 0 else "W"]
	var total := 0.0
	var sf := sprite.sprite_frames
	if not sf.has_animation(name):
		return 1.0
	for i in sf.get_frame_count(name):
		total += sf.get_frame_duration(name, i)
	total /= maxf(0.1, sf.get_animation_speed(name))
	var st := float(strides.get(clip, 0.0))
	if st <= 0.5 or total <= 0.0:
		return 1.0
	var cad := speed_px / st                      # cycles per second wanted
	var top := 2.9 if clip == "run" else 1.9
	return clampf(cad, 0.5, top) * total


func _animate(x: float) -> void:
	if attack_t > 0.0 or (stun > 0.0 and action == "hurt"):
		return
	if not is_on_floor():
		sprite.speed_scale = 1.0
		_play("jump" if _has("jump") else "idle")
		if action == "jump" and _has("jump"):
			# the jump clip: rise on its first half, fall on the second
			var n := sprite.sprite_frames.get_frame_count(sprite.animation)
			sprite.pause()
			sprite.frame = clampi(int(remap(velocity.y, JUMP * u, MAX_FALL * u * 0.6, 0.0, n - 1)), 0, n - 1)
		return
	var sp := absf(velocity.x)
	var run_from := float(strides.get("walk", 0.0)) * 1.9 * 0.92 if strides.has("walk") else SPRINT * u * 0.8
	if sp > maxf(run_from, 40.0) and _has("run"):
		_play("run")
		sprite.speed_scale = _rate_for("run", sp)
		stopping = 0.0
	elif sp > 8.0:
		_play("walk")
		sprite.speed_scale = _rate_for("walk", sp)
		stopping = 0.12
	elif stopping > 0.0 and action == "walk":
		stopping -= get_physics_process_delta_time()     # settle onto the next footfall
		var n := sprite.sprite_frames.get_frame_count(sprite.animation)
		if sprite.frame == 0 or sprite.frame == n / 2 or stopping <= 0.0:
			stopping = 0.0
			sprite.speed_scale = 1.0
			_play("idle")
	else:
		sprite.speed_scale = 1.0
		_play("idle")
	if not sprite.is_playing():
		sprite.play()


func take_hit(amount: int, from_x: float) -> void:
	if dead or invuln > 0.0:
		return
	hp = maxi(0, hp - amount)
	invuln = 1.1
	stun = 0.3
	velocity = Vector2(signf(global_position.x - from_x) * 170.0, -220.0)
	sprite.speed_scale = 1.0
	if _has("hurt"):
		_play("hurt", true)
	hurt_by.emit(amount)
	if hp <= 0:
		dead = true
		_play("death", true)
		died.emit()


func heal(n: int) -> void:
	hp = mini(max_hp, hp + n)


func bounce() -> void:
	velocity.y = JUMP * u * 0.7
	coyote = 0.0


func feet_y() -> float:
	return global_position.y


func debug_text() -> Dictionary:
	var sf := sprite.sprite_frames
	var a := str(sprite.animation)
	return {"player pos": global_position.round(), "player velocity": velocity.round(),
		"on floor": is_on_floor(), "facing": "E" if facing > 0 else "W",
		"player anim": "%s frame %d/%d%s" % [a, sprite.frame + 1, sf.get_frame_count(a) if sf.has_animation(a) else 0, "" if sprite.is_playing() else " (held)"],
		"jump": "coyote %.2f  buffer %.2f" % [coyote, buffer],
		"player": "hp %d/%d  invulnerable %.2f  stun %.2f  attack %.2f" % [hp, max_hp, invuln, stun, maxf(attack_t, 0.0)]}

extends CharacterBody2D
## Side-view walker enemy (pixel-pipeline godot_runtime): patrols, turns at walls and ledges,
## runs at the player when it sees them on about the same height. Built by side_game.gd from a
## manifest character (its E / W rows).

const GRAVITY := 1150.0

var game: Node = null
var sprite: AnimatedSprite2D
var shape: CollisionShape2D
var facing := -1
var hp := 2
var state := "patrol"
var u := 1.0                    # tile / 32 (see side_player.gd)
var patrol_speed := 38.0
var chase_speed := 105.0
var sight := 150.0
var hit_flash := 0.0
var stun := 0.0
var dead := false
var anims := {}
var size := Vector2(30, 20)


func setup(g: Node, frames: SpriteFrames, feet: Vector2, frame: Vector2, body: Vector2) -> void:
	game = g
	u = float(g.tile) / 32.0 if "tile" in g else 1.0
	patrol_speed *= u
	chase_speed *= u
	collision_layer = 4
	collision_mask = 1
	floor_snap_length = 6.0
	sprite = AnimatedSprite2D.new()
	sprite.sprite_frames = frames
	sprite.centered = false
	sprite.offset = -feet
	add_child(sprite)
	size = body
	shape = CollisionShape2D.new()
	var r := RectangleShape2D.new()
	r.size = body
	shape.shape = r
	shape.position = Vector2(0, -body.y / 2.0)
	add_child(shape)
	for n in frames.get_animation_names():
		anims[str(n)] = true
	_play("walk")


func _play(a: String, restart := false) -> void:
	var name := "%s_%s" % [a, "E" if facing > 0 else "W"]
	if not anims.has(name):
		name = "idle_%s" % ("E" if facing > 0 else "W")
		if not anims.has(name):
			return
	if restart or str(sprite.animation) != name:
		sprite.play(name)


func _ground_ahead() -> bool:
	var space := get_world_2d().direct_space_state
	var from := global_position + Vector2(facing * (size.x * 0.5 + 4.0), -4.0)
	var q := PhysicsRayQueryParameters2D.create(from, from + Vector2(0, 24))
	q.collision_mask = 1
	return not space.intersect_ray(q).is_empty()


func _physics_process(delta: float) -> void:
	velocity.y = minf(velocity.y + GRAVITY * u * delta, 540.0 * u)
	if dead:
		velocity.x = move_toward(velocity.x, 0, 600 * delta)
		move_and_slide()
		return
	hit_flash = maxf(0.0, hit_flash - delta)
	sprite.modulate = Color(1, 0.45, 0.45) if hit_flash > 0.0 else Color.WHITE
	stun = maxf(0.0, stun - delta)
	var p = game.player if game != null else null
	var speed := patrol_speed
	if p != null and is_instance_valid(p) and not p.dead:
		var d: Vector2 = p.global_position - global_position
		if absf(d.x) < sight and absf(d.y) < 48.0:
			state = "chase"
			facing = 1 if d.x > 0 else -1
		elif state == "chase" and absf(d.x) > sight * 1.6:
			state = "patrol"
	if state == "chase":
		speed = chase_speed
	if stun > 0.0:
		velocity.x = move_toward(velocity.x, 0, 500 * delta)
	else:
		if is_on_floor() and (is_on_wall() or not _ground_ahead()):
			if state == "patrol" or not _ground_ahead():
				facing = -facing
				if state == "chase":
					state = "patrol"
		velocity.x = facing * speed
	move_and_slide()
	if stun <= 0.0:
		_play("run" if state == "chase" and anims.has("run_E") else "walk")


func take_hit(amount: int, from_x: float) -> void:
	if dead:
		return
	hp -= amount
	hit_flash = 0.25
	stun = 0.35
	velocity = Vector2(signf(global_position.x - from_x) * 160.0, -160.0) * u
	if hp <= 0:
		dead = true
		collision_layer = 0
		_play("death", true)
		if game != null and game.has_method("enemy_died"):
			game.enemy_died(self)


func debug_text() -> String:
	return "%s hp %d %s pos %s vel %s anim %s %d" % [name, hp, state, global_position.round(), velocity.round(), sprite.animation, sprite.frame + 1]

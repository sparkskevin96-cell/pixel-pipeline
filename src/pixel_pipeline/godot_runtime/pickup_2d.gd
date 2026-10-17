extends Area2D
## A bobbing pickup (pixel-pipeline godot_runtime): coin / heart / gem from the UI kit's icons.
## side_game.gd connects body_entered; `kind` and `value` say what it gives.

var kind := "coin"
var value := 1
var base_y := 0.0
var t := 0.0
var taken := false


func setup(k: String, tex: Texture2D, v := 1) -> void:
	kind = k
	value = v
	collision_layer = 8
	collision_mask = 2
	monitoring = true
	var s := Sprite2D.new()
	s.texture = tex
	s.name = "Sprite"
	add_child(s)
	var c := CollisionShape2D.new()
	var r := CircleShape2D.new()
	r.radius = 9.0
	c.shape = r
	add_child(c)
	t = randf() * TAU


func _ready() -> void:
	base_y = position.y


func _process(delta: float) -> void:
	if taken:
		return
	t += delta
	position.y = base_y + round(sin(t * 3.0) * 2.0)


func take() -> void:
	if taken:
		return
	taken = true
	set_deferred("monitoring", false)
	var tw := create_tween()
	tw.set_parallel(true)
	tw.tween_property(self, "position:y", position.y - 18.0, 0.3)
	tw.tween_property(self, "modulate:a", 0.0, 0.3)
	tw.chain().tween_callback(queue_free)

extends Control
## A small deck-builder (pixel-pipeline godot_runtime), Slay-the-Spire style, over the app's
## export: cards are the exported Card.tscn scenes (cards.py frames, art, pixel fonts), enemies
## and the hero are library characters (SpriteFrames), hits play the exported spell effects, the
## UI uses the UI kit (theme.tres, pieces, icons, font). res://deck.json (playtest.py) lists the
## cards with their rules, the starting deck, the reward pool, the characters and effects.
##   Map: fight -> fight | rest | shop -> elite | rest -> boss.   Fight: 3 energy, draw 5,
##   click a card then the enemy (or drag it onto the enemy; skills: drag up / click twice),
##   End turn (E). After a win: pick 1 of 3 cards (or skip). V: view your deck.

const Manifest := preload("res://pp_runtime/pp_manifest.gd")
const HAND_Y := 544.0
const ENEMIES := {
	"wolf": {"name": "Grey Wolf", "hp": 26, "char": "wolf", "scale": 4.0, "gold": 15,
		"pattern": [["attack", 6, 1], ["attack", 3, 2], ["buff", 2, 0]]},
	"bandit": {"name": "Bandit", "hp": 38, "char": "bandit", "scale": 4.0, "gold": 20,
		"pattern": [["attack", 9, 1], ["block_attack", 6, 5], ["attack", 12, 1]]},
	"shadow": {"name": "Shadow Knight", "hp": 56, "char": "shadow", "scale": 4.0, "gold": 40, "elite": true,
		"pattern": [["block", 12, 0], ["attack", 14, 1], ["attack", 6, 2], ["buff", 3, 0]]},
	"warlock": {"name": "Blood Warlock", "hp": 78, "char": "warlock", "scale": 4.0, "gold": 0, "boss": true,
		"pattern": [["attack", 7, 2], ["poison", 4, 0], ["charge", 0, 0], ["attack", 24, 1]]},
}
const MAP_ROWS := [["fight:wolf"], ["fight:bandit", "rest", "shop"], ["elite:shadow", "rest"], ["boss:warlock"]]

var defs := {}                 # deck.json
var portrait_nodes := []       # the hero's painted portrait (deck.json "portrait")
var man
var ui: Control                # 640x360 coordinates, scaled 2x (the UI kit at whole pixels)
var cards_layer: Control       # 1280x720 coordinates (cards at 1x)
var overlay: Control           # screens over everything (map, reward, deck, rest, shop, end)
var font: Font = null
var rng := RandomNumberGenerator.new()

# run state
var deck: Array = []           # card ids
var hp := 60
var max_hp := 60
var gold := 50
var floor_i := -1
var path_taken: Array = []
var screen := "map"            # map | fight | reward | rest | shop | over | won
# fight state
var draw_pile: Array = []
var discard: Array = []
var exhausted: Array = []
var hand: Array = []           # [{"id", "view"}]
var energy := 3
var max_energy := 3
var block := 0
var strength := 0
var poison := 0
var enemy := {}                # {"key", "name", "hp", "max_hp", "block", "str", "poison", "burn", "turn", "intent"}
var busy := false
var turn := 0
var hovered = null             # a hand entry
var selected = null
var dragging = null
var drag_from := Vector2.ZERO
var press_at := Vector2.ZERO
var log_lines: Array = []
var deck_view: Control = null
# nodes
var hero: AnimatedSprite2D
var foe: AnimatedSprite2D
var foe_root: Node2D
var hero_root: Node2D
var hp_label: Label
var gold_label: Label
var floor_label: Label
var energy_label: Label
var draw_label: Label
var discard_label: Label
var hero_bar: Label
var foe_bar: Label
var intent_box: HBoxContainer
var end_button: Button
var message: Label
var message_t := 0.0


func _ready() -> void:
	add_to_group("pp_debug")
	rng.seed = int(Time.get_unix_time_from_system())
	defs = JSON.parse_string(FileAccess.get_file_as_string("res://deck.json"))
	man = Manifest.new("res://art")
	var bg := ColorRect.new()
	bg.color = Color(0.09, 0.07, 0.11)
	bg.set_anchors_preset(Control.PRESET_FULL_RECT)
	bg.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(bg)
	ui = Control.new()
	ui.scale = Vector2(2, 2)
	ui.size = Vector2(640, 360)
	ui.mouse_filter = Control.MOUSE_FILTER_IGNORE
	ui.texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
	var e: Dictionary = man.entry(str(defs.get("ui", "")))
	var th: String = man.res(e, "theme")
	if th != "" and ResourceLoader.exists(th):
		ui.theme = load(th)
	var fnt: String = man.folder(e).path_join("fonts/pixelkit_1x.fnt")
	if ResourceLoader.exists(fnt):
		font = load(fnt)
	add_child(ui)
	cards_layer = Control.new()
	cards_layer.size = Vector2(1280, 720)
	cards_layer.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(cards_layer)
	overlay = Control.new()
	overlay.scale = Vector2(2, 2)
	overlay.size = Vector2(640, 360)
	overlay.mouse_filter = Control.MOUSE_FILTER_IGNORE
	overlay.theme = ui.theme
	add_child(overlay)
	_build_fight_ui()
	deck = defs.get("starter", []).duplicate()
	_show_map()


# ------------------------------------------------------------------ small helpers

func _icon(name: String) -> Texture2D:
	var e: Dictionary = man.entry(str(defs.get("ui", "")))
	return man.texture(man.folder(e).path_join("pieces/%s.png" % name))


func _label(text: String, pos: Vector2, parent: Control, col := Color(1.0, 0.96, 0.86)) -> Label:
	var l := Label.new()
	l.text = text
	l.position = pos
	if font != null:
		l.add_theme_font_override("font", font)
	l.add_theme_font_size_override("font_size", 10)
	l.add_theme_color_override("font_color", col)
	l.add_theme_color_override("font_shadow_color", Color(0.08, 0.04, 0.03, 0.95))
	l.add_theme_constant_override("shadow_offset_x", 1)
	l.add_theme_constant_override("shadow_offset_y", 1)
	l.mouse_filter = Control.MOUSE_FILTER_IGNORE
	parent.add_child(l)
	return l


func _panel(rect: Rect2, parent: Control, variation := "WindowPanel") -> Panel:
	var p := Panel.new()
	p.theme_type_variation = variation
	p.position = rect.position
	p.size = rect.size
	parent.add_child(p)
	return p


func _button(text: String, rect: Rect2, parent: Control, cb: Callable) -> Button:
	var b := Button.new()
	b.text = text
	b.position = rect.position
	b.size = rect.size
	if font != null:
		b.add_theme_font_override("font", font)
		b.add_theme_font_size_override("font_size", 10)
	b.pressed.connect(cb)
	parent.add_child(b)
	return b


func _say(text: String, t := 1.6) -> void:
	message.text = text
	message.visible = true
	message_t = t
	_log(text)


func _log(text: String) -> void:
	log_lines.append(text)
	while log_lines.size() > 8:
		log_lines.pop_front()


func _clear(c: Control) -> void:
	for n in c.get_children():
		n.queue_free()


func _card_def(id: String) -> Dictionary:
	return defs["cards"].get(id, {})


func _card_view(id: String) -> Control:
	# the exported Card.tscn of that card (it loads its own card.json) inside a holder Control
	var d := _card_def(id)
	var e: Dictionary = man.entry(str(d.get("asset", "")))
	var holder := Control.new()
	holder.size = Vector2(126, 176)
	holder.pivot_offset = Vector2(63, 176)
	holder.mouse_filter = Control.MOUSE_FILTER_PASS
	holder.set_meta("id", id)
	var sp: String = man.res(e, "scene")
	if sp != "" and ResourceLoader.exists(sp):
		var c: Control = load(sp).instantiate()
		c.mouse_filter = Control.MOUSE_FILTER_IGNORE
		c.texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
		holder.add_child(c)
		holder.size = c.custom_minimum_size if c.custom_minimum_size != Vector2.ZERO else holder.size
	else:
		var r := ColorRect.new()
		r.color = Color(0.3, 0.2, 0.2)
		r.size = holder.size
		holder.add_child(r)
		_label(str(d.get("name", id)), Vector2(6, 6), holder)
	return holder


func _char_sprite(role: String, scale_k: float, facing: String) -> AnimatedSprite2D:
	# scale_k = the size (inside the 2x ui) of a library-scale sheet; sheets drawn bigger
	# (manifest px_scale, e.g. 1.5 for a 144x96 knight) are shown smaller, at a WHOLE screen
	# scale so their pixels stay crisp and every character shares one world scale
	var e: Dictionary = man.entry(str(defs["chars"].get(role, "")))
	var s := AnimatedSprite2D.new()
	s.sprite_frames = man.sprite_frames(e)
	s.centered = false
	s.offset = -man.anchor(e)
	var total := maxf(1.0, round(scale_k * 2.0 / float(e.get("px_scale", 1.0))))
	scale_k = total / 2.0
	s.scale = Vector2(scale_k, scale_k)
	s.set_meta("facing", facing)
	_anim(s, "idle")
	return s


func _anim(s: AnimatedSprite2D, action: String, then_idle := false) -> void:
	if s == null or s.sprite_frames == null:
		return
	var f: String = s.get_meta("facing", "E")
	for name in ["%s_%s" % [action, f], "%s_%s" % [action, f.substr(f.length() - 1)], "idle_%s" % f]:
		if s.sprite_frames.has_animation(name):
			s.play(name)
			break
	if then_idle:
		var dur := 0.0
		var a := str(s.animation)
		for i in s.sprite_frames.get_frame_count(a):
			dur += s.sprite_frames.get_frame_duration(a, i) / maxf(0.1, s.sprite_frames.get_animation_speed(a))
		var wr: WeakRef = weakref(s)
		get_tree().create_timer(clampf(dur, 0.2, 1.4), false).timeout.connect(func():
			var n = wr.get_ref()
			if n != null:
				_anim(n, "idle"))


func _fx(name: String, at_ui: Vector2, scale_k := 2.0) -> void:
	var id := str(defs.get("fx", {}).get(name, ""))
	var sf: SpriteFrames = man.sprite_frames(man.entry(id)) if id != "" else null
	if sf == null or sf.get_animation_names().is_empty():
		return
	var s := AnimatedSprite2D.new()
	s.sprite_frames = sf
	s.position = at_ui
	s.scale = Vector2(scale_k, scale_k) / 2.0          # inside the 2x ui
	s.z_index = 20
	ui.add_child(s)
	var names := sf.get_animation_names()
	var anim: String = names[names.size() - 1] if "impact" in names else names[0]
	if sf.has_animation("impact"):
		anim = "impact"
	sf.set_animation_loop(anim, false)
	s.play(anim)
	s.animation_finished.connect(s.queue_free)


func _number(text: String, at_ui: Vector2, col: Color) -> void:
	var l := _label(text, at_ui - Vector2(12, 0), ui, col)
	l.z_index = 30
	l.add_theme_font_size_override("font_size", 20 if font == null else 10)
	l.scale = Vector2(2, 2)
	var tw := create_tween()
	tw.set_parallel(true)
	tw.tween_property(l, "position:y", at_ui.y - 34, 0.8).set_trans(Tween.TRANS_QUAD).set_ease(Tween.EASE_OUT)
	tw.tween_property(l, "modulate:a", 0.0, 0.45).set_delay(0.45)
	tw.chain().tween_callback(l.queue_free)


func _shake(n: Node2D, amount := 4.0) -> void:
	var home := n.position
	var tw := create_tween()
	for x in [amount, -amount, amount * 0.6, -amount * 0.4, 0.0]:
		tw.tween_property(n, "position:x", home.x + x, 0.04)


# ------------------------------------------------------------------ fight layout

func _build_fight_ui() -> void:
	_panel(Rect2(0, 0, 640, 22), ui, "InsetPanel")
	var hi := TextureRect.new()
	hi.texture = _icon("icon_heart")
	hi.position = Vector2(6, 3)
	ui.add_child(hi)
	hp_label = _label("", Vector2(24, 6), ui)
	var gi := TextureRect.new()
	gi.texture = _icon("icon_coin")
	gi.position = Vector2(110, 3)
	ui.add_child(gi)
	gold_label = _label("", Vector2(128, 6), ui)
	floor_label = _label("", Vector2(200, 6), ui)
	_button("Deck (V)", Rect2(560, 2, 76, 18), ui, _toggle_deck_view)
	hero_root = Node2D.new()
	hero_root.position = Vector2(150, 222)
	ui.add_child(hero_root)
	hero = _char_sprite("hero", 1.5, "E")
	hero_root.add_child(hero)
	var pp := str(defs.get("portrait", ""))
	if pp != "" and FileAccess.file_exists(pp):            # the painted picture, not the tiny sprite
		var frame := _panel(Rect2(8, 24, 52, 52), ui, "Slot")
		frame.mouse_filter = Control.MOUSE_FILTER_IGNORE
		var img := Image.load_from_file(ProjectSettings.globalize_path(pp))
		if img != null:
			var face := TextureRect.new()
			face.texture = ImageTexture.create_from_image(img)
			face.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
			face.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
			face.texture_filter = CanvasItem.TEXTURE_FILTER_LINEAR
			face.position = Vector2(11, 27)
			face.size = Vector2(46, 46)
			face.mouse_filter = Control.MOUSE_FILTER_IGNORE
			ui.add_child(face)
			portrait_nodes = [frame, face]
	hero_bar = _label("", Vector2(110, 228), ui)
	foe_root = Node2D.new()
	foe_root.position = Vector2(470, 222)
	ui.add_child(foe_root)
	foe_bar = _label("", Vector2(420, 228), ui)
	intent_box = HBoxContainer.new()
	intent_box.position = Vector2(440, 70)
	intent_box.add_theme_constant_override("separation", 2)
	ui.add_child(intent_box)
	var orb := _panel(Rect2(14, 262, 44, 30), ui, "Slot")
	orb.mouse_filter = Control.MOUSE_FILTER_IGNORE
	energy_label = _label("", Vector2(20, 271), ui, Color(1.0, 0.85, 0.35))
	draw_label = _label("", Vector2(6, 340), ui)
	discard_label = _label("", Vector2(560, 340), ui)
	end_button = _button("End turn (E)", Rect2(548, 262, 84, 22), ui, _end_turn)
	message = _label("", Vector2(0, 120), ui, Color(1.0, 0.9, 0.5))
	message.size = Vector2(640, 20)
	message.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	message.visible = false
	_set_fight_visible(false)


func _set_fight_visible(v: bool) -> void:
	for n in [hero_root, foe_root, hero_bar, foe_bar, intent_box, energy_label, draw_label, discard_label, end_button] + portrait_nodes:
		n.visible = v
	for n in ui.get_children():
		if n is Panel and n.theme_type_variation == "Slot":
			n.visible = v
	cards_layer.visible = v


func _update_ui() -> void:
	hp_label.text = "%d/%d" % [hp, max_hp]
	gold_label.text = "%d" % gold
	floor_label.text = "Floor %d/%d   %s" % [maxi(floor_i + 1, 0), MAP_ROWS.size(), " > ".join(path_taken)]
	if screen != "fight":
		return
	energy_label.text = "%d/%d" % [energy, max_energy]
	draw_label.text = "Draw %d" % draw_pile.size()
	discard_label.text = "Discard %d" % discard.size()
	var st := ""
	if block > 0: st += "  Block %d" % block
	if strength != 0: st += "  Str %d" % strength
	if poison > 0: st += "  Poison %d" % poison
	hero_bar.text = "HP %d/%d%s" % [hp, max_hp, st]
	var es := ""
	if enemy.get("block", 0) > 0: es += "  Block %d" % enemy["block"]
	if enemy.get("str", 0) != 0: es += "  Str %d" % enemy["str"]
	if enemy.get("poison", 0) > 0: es += "  Poison %d" % enemy["poison"]
	if enemy.get("burn", 0) > 0: es += "  Burn %d" % enemy["burn"]
	foe_bar.text = "%s  HP %d/%d%s" % [enemy.get("name", ""), enemy.get("hp", 0), enemy.get("max_hp", 0), es]
	_show_intent()
	for h in hand:
		var ok: bool = int(_card_def(h["id"]).get("cost", 0)) <= energy and not busy
		h["view"].modulate = Color.WHITE if ok else Color(0.62, 0.62, 0.68)


func _show_intent() -> void:
	_clear(intent_box)
	if enemy.is_empty() or enemy["hp"] <= 0:
		return
	var it: Array = enemy["intent"]
	var icon := "icon_sword"
	var text := ""
	match str(it[0]):
		"attack":
			text = str(int(it[1]) + int(enemy["str"])) + ("x%d" % int(it[2]) if int(it[2]) > 1 else "")
		"block_attack":
			icon = "icon_shield"
			text = "%d + %d" % [int(it[1]), int(it[2]) + int(enemy["str"])]
		"block":
			icon = "icon_shield"
			text = str(it[1])
		"buff":
			icon = "icon_gem"
			text = "+%d str" % int(it[1])
		"poison":
			icon = "icon_potion"
			text = "poison %d" % int(it[1])
		"charge":
			icon = "icon_gear"
			text = "charging..."
	var r := TextureRect.new()
	r.texture = _icon(icon)
	intent_box.add_child(r)
	var l := Label.new()
	l.text = text
	if font != null:
		l.add_theme_font_override("font", font)
		l.add_theme_font_size_override("font_size", 10)
	l.add_theme_color_override("font_color", Color(1.0, 0.8, 0.7))
	intent_box.add_child(l)


# ------------------------------------------------------------------ map + screens

func _show_map() -> void:
	screen = "map"
	_set_fight_visible(false)
	_clear(overlay)
	_panel(Rect2(120, 30, 400, 300), overlay)
	_label("Choose your path", Vector2(270, 40), overlay)
	var next := floor_i + 1
	for row in MAP_ROWS.size():
		var nodes: Array = MAP_ROWS[row]
		var y := 290.0 - row * 66.0
		for i in nodes.size():
			var kind: String = nodes[i]
			var x: float = 320.0 + (i - (nodes.size() - 1) / 2.0) * 110.0
			var label := _node_label(kind)
			var b := _button(label, Rect2(x - 48, y - 12, 96, 24), overlay, _enter_node.bind(row, kind))
			b.disabled = row != next
			if row < next:
				b.text = ("[x] " if path_taken.size() > row and path_taken[row] == label else "") + label
			if row > 0:
				for j in MAP_ROWS[row - 1].size():
					var px: float = 320.0 + (j - (MAP_ROWS[row - 1].size() - 1) / 2.0) * 110.0
					var ln := Line2D.new()
					ln.points = PackedVector2Array([Vector2(px, y + 54), Vector2(x, y + 12)])
					ln.width = 1
					ln.default_color = Color(0.8, 0.7, 0.5, 0.5)
					overlay.add_child(ln)
	_update_ui()


func _node_label(kind: String) -> String:
	var parts := kind.split(":")
	match parts[0]:
		"fight":
			return "Fight: " + str(ENEMIES[parts[1]]["name"])
		"elite":
			return "Elite: " + str(ENEMIES[parts[1]]["name"])
		"boss":
			return "Boss: " + str(ENEMIES[parts[1]]["name"])
		"rest":
			return "Rest"
		"shop":
			return "Shop"
	return kind


func _enter_node(row: int, kind: String) -> void:
	floor_i = row
	path_taken.append(_node_label(kind))
	_clear(overlay)
	var parts := kind.split(":")
	match parts[0]:
		"fight", "elite", "boss":
			_start_fight(parts[1])
		"rest":
			_show_rest()
		"shop":
			_show_shop()


func _after_node() -> void:
	if floor_i >= MAP_ROWS.size() - 1:
		_end_screen(true)
	else:
		_show_map()


func _show_rest() -> void:
	screen = "rest"
	_clear(overlay)
	_panel(Rect2(170, 90, 300, 160), overlay)
	_label("A quiet campfire.", Vector2(260, 104), overlay)
	var heal := int(round(max_hp * 0.3))
	_button("Rest: heal %d HP" % heal, Rect2(220, 140, 200, 24), overlay, func():
		hp = mini(max_hp, hp + heal)
		_log("Rested: +%d HP" % heal)
		_after_node())
	_button("Remove a card", Rect2(220, 176, 200, 24), overlay, func(): _pick_from_deck("Remove which card?", _remove_card, _after_node))
	_update_ui()


func _show_shop() -> void:
	screen = "shop"
	_clear(overlay)
	_panel(Rect2(40, 30, 560, 300), overlay)
	_label("Shop  -  you have %d gold" % gold, Vector2(60, 40), overlay)
	var pool: Array = defs.get("reward_pool", []).duplicate()
	pool.shuffle()
	for i in 3:
		var id: String = pool[i]
		var price := 35 + 15 * ["common", "rare", "epic"].find(str(_card_def(id).get("rarity", "common")))
		var v := _card_view(id)
		v.position = Vector2(120 + i * 400, 130)
		cards_layer.add_child(v)
		cards_layer.visible = true
		var b := _button("Buy: %d gold" % price, Rect2(70 + i * 200, 250, 110, 22), overlay, func(): pass)
		b.disabled = gold < price
		b.pressed.connect(func():
			if gold >= price:
				gold -= price
				deck.append(id)
				_log("Bought " + str(_card_def(id)["name"]))
				v.queue_free()
				b.disabled = true
				_update_ui())
	var rm := _button("Remove a card: 50 gold", Rect2(120, 290, 180, 22), overlay, func(): pass)
	rm.disabled = gold < 50
	rm.pressed.connect(func():
		if gold >= 50:
			_clear(cards_layer)
			_pick_from_deck("Remove which card? (50 gold)", func(id):
				gold -= 50
				_remove_card(id), _show_map_after_shop))
	_button("Leave", Rect2(400, 290, 120, 22), overlay, _show_map_after_shop)
	_update_ui()


func _show_map_after_shop() -> void:
	_clear(cards_layer)
	cards_layer.visible = false
	_after_node()


func _remove_card(id: String) -> void:
	deck.erase(id)
	_log("Removed " + str(_card_def(id).get("name", id)))


func _pick_from_deck(title: String, on_pick: Callable, then: Callable) -> void:
	_clear(overlay)
	_clear(cards_layer)
	cards_layer.visible = true
	_panel(Rect2(10, 26, 620, 330), overlay)
	_label(title, Vector2(24, 34), overlay)
	var ids := deck.duplicate()
	ids.sort()
	for i in ids.size():
		var v := _card_view(ids[i])
		v.scale = Vector2(0.8, 0.8)
		v.position = Vector2(40 + (i % 10) * 120, 90 + (i / 10) * 150)
		var id: String = ids[i]
		v.gui_input.connect(func(ev):
			if ev is InputEventMouseButton and ev.pressed and ev.button_index == MOUSE_BUTTON_LEFT:
				on_pick.call(id)
				_clear(cards_layer)
				cards_layer.visible = false
				then.call())
		cards_layer.add_child(v)
	_button("Cancel", Rect2(520, 330, 100, 20), overlay, func():
		_clear(cards_layer)
		cards_layer.visible = false
		then.call())


func _toggle_deck_view() -> void:
	if is_instance_valid(deck_view):
		deck_view.queue_free()
		deck_view = null
		return
	deck_view = Control.new()
	deck_view.size = Vector2(1280, 720)
	deck_view.mouse_filter = Control.MOUSE_FILTER_STOP
	var dim := ColorRect.new()
	dim.color = Color(0.03, 0.02, 0.05, 0.86)
	dim.size = Vector2(1280, 720)
	deck_view.add_child(dim)
	add_child(deck_view)
	var title := Label.new()
	title.text = "Your deck: %d cards   (draw %d / discard %d / exhausted %d)   V or click to close" % [deck.size(), draw_pile.size(), discard.size(), exhausted.size()]
	if font != null:
		title.add_theme_font_override("font", font)
		title.add_theme_font_size_override("font_size", 10)
	title.scale = Vector2(2, 2)
	title.position = Vector2(30, 14)
	deck_view.add_child(title)
	var ids := deck.duplicate()
	ids.sort()
	for i in ids.size():
		var v := _card_view(ids[i])
		v.scale = Vector2(0.9, 0.9)
		v.position = Vector2(30 + (i % 10) * 124, 60 + (i / 10) * 168)
		v.mouse_filter = Control.MOUSE_FILTER_IGNORE
		deck_view.add_child(v)
	deck_view.gui_input.connect(func(ev):
		if ev is InputEventMouseButton and ev.pressed:
			_toggle_deck_view())


func _end_screen(won: bool) -> void:
	screen = "won" if won else "over"
	_set_fight_visible(false)
	_clear(cards_layer)
	_clear(overlay)
	_panel(Rect2(160, 100, 320, 150), overlay)
	_label("VICTORY!" if won else "DEFEAT", Vector2(285, 120), overlay, Color(1.0, 0.85, 0.35) if won else Color(1.0, 0.4, 0.35))
	_label("Floor %d   deck %d cards   gold %d" % [floor_i + 1, deck.size(), gold], Vector2(220, 150), overlay)
	_button("New run (R)", Rect2(250, 200, 140, 24), overlay, func(): get_tree().reload_current_scene())


# ------------------------------------------------------------------ fight flow

func _start_fight(key: String) -> void:
	screen = "fight"
	var e: Dictionary = ENEMIES[key]
	enemy = {"key": key, "name": e["name"], "hp": e["hp"], "max_hp": e["hp"], "block": 0, "str": 0, "poison": 0,
		"burn": 0, "turn": 0, "intent": e["pattern"][0]}
	if is_instance_valid(foe):
		foe.queue_free()
	foe = _char_sprite(str(e["char"]), float(e["scale"]) / 2.0, "W")
	foe_root.add_child(foe)
	var fh: float = man.anchor(man.entry(str(defs["chars"].get(str(e["char"]), "")))).y * foe.scale.y
	intent_box.position = foe_root.position + Vector2(-24, -fh - 16)
	foe_root.modulate = Color.WHITE
	draw_pile = deck.duplicate()
	draw_pile.shuffle()
	discard.clear()
	exhausted.clear()
	block = 0
	strength = 0
	poison = 0
	turn = 0
	_set_fight_visible(true)
	_clear(cards_layer)
	hand.clear()
	_say(str(e["name"]) + (" (boss)" if e.get("boss", false) else (" (elite)" if e.get("elite", false) else "")) + " appears!")
	_begin_turn()


func _begin_turn() -> void:
	turn += 1
	energy = max_energy
	block = 0
	if poison > 0:
		_hurt_player(poison, true)
		poison -= 1
		if hp <= 0:
			return
	_draw_cards(5)
	busy = false
	_update_ui()


func _draw_cards(n: int) -> void:
	for i in n:
		if hand.size() >= 10:
			break
		if draw_pile.is_empty():
			if discard.is_empty():
				break
			draw_pile = discard.duplicate()
			discard.clear()
			draw_pile.shuffle()
			_say("Shuffled the discard pile into the draw pile", 1.0)
		var id: String = draw_pile.pop_back()
		var v := _card_view(id)
		v.position = Vector2(20, 640)            # flies in from the draw pile
		v.scale = Vector2(0.3, 0.3)
		cards_layer.add_child(v)
		var h := {"id": id, "view": v}
		hand.append(h)
		v.mouse_entered.connect(_hover.bind(h, true))
		v.mouse_exited.connect(_hover.bind(h, false))
		v.gui_input.connect(_card_input.bind(h))
	_layout_hand()


func _hand_pos(i: int, n: int) -> Vector2:
	var spacing := minf(118.0, 760.0 / maxf(1.0, n))
	var x := 640.0 - (n - 1) * spacing / 2.0 + i * spacing - 63.0
	var y := HAND_Y + absf(i - (n - 1) / 2.0) * 6.0
	return Vector2(x, y)


func _layout_hand() -> void:
	for i in hand.size():
		var h: Dictionary = hand[i]
		var v: Control = h["view"]
		if h == dragging:
			continue
		var lift := 0.0
		var sc := 1.0
		if h == hovered and dragging == null:
			sc = 2.0
			lift = 0.0
		if h == selected:
			lift = -40.0
			sc = 1.0 if h != hovered else 2.0
		v.z_index = 50 if (h == hovered or h == selected) else i
		v.rotation = 0.0 if sc > 1.0 or h == selected else deg_to_rad((i - (hand.size() - 1) / 2.0) * 2.5)
		var tw := v.create_tween().set_parallel(true).set_trans(Tween.TRANS_QUAD).set_ease(Tween.EASE_OUT)
		var p := _hand_pos(i, hand.size()) + Vector2(0, lift)
		if sc > 1.0:
			p.y = 720.0 - 176.0
		tw.tween_property(v, "position", p, 0.14)
		tw.tween_property(v, "scale", Vector2(sc, sc), 0.12)


func _hover(h: Dictionary, on: bool) -> void:
	if screen != "fight":
		return
	if on:
		hovered = h
	elif hovered == h:
		hovered = null
	_layout_hand()


func _card_input(ev: InputEvent, h: Dictionary) -> void:
	if busy or screen != "fight":
		return
	if ev is InputEventMouseButton and ev.button_index == MOUSE_BUTTON_LEFT:
		if ev.pressed:
			press_at = get_global_mouse_position()
			dragging = null
			drag_from = h["view"].position
			set_meta("press_card", h)
		accept_event()


func _input(ev: InputEvent) -> void:
	if ev is InputEventKey and ev.pressed and not ev.echo:
		match ev.physical_keycode:
			KEY_E:
				if screen == "fight":
					_end_turn()
			KEY_V:
				_toggle_deck_view()
			KEY_R:
				if screen in ["won", "over"]:
					get_tree().reload_current_scene()
			KEY_ESCAPE:
				if selected != null:
					selected = null
					_layout_hand()
				elif is_instance_valid(deck_view):
					_toggle_deck_view()
				else:
					get_tree().quit()
	if screen != "fight" or busy:
		return
	if ev is InputEventMouseMotion:
		var h = get_meta("press_card", null)
		if h != null and dragging == null and get_global_mouse_position().distance_to(press_at) > 8.0:
			dragging = h
			selected = null
			hovered = null
			h["view"].z_index = 60
			h["view"].rotation = 0.0
			h["view"].scale = Vector2.ONE
		if dragging != null:
			dragging["view"].position = get_global_mouse_position() - Vector2(63, 88)
	elif ev is InputEventMouseButton and ev.button_index == MOUSE_BUTTON_LEFT and not ev.pressed:
		var h = get_meta("press_card", null)
		set_meta("press_card", null)
		var at := get_global_mouse_position()
		if dragging != null:
			var d: Dictionary = dragging
			dragging = null
			if not _try_play(d, at, true):
				_layout_hand()
		elif h != null:
			if selected == h and str(_card_def(h["id"]).get("target", "")) == "self":
				_try_play(h, Vector2(640, 200), false)
			else:
				selected = h
				_layout_hand()
				if not _can_afford(h):
					_say("Not enough energy", 1.0)
					selected = null
					_shake_card(h)
		elif selected != null:
			# a click away from the hand: the enemy (attacks) or anywhere up top (skills)
			if not _try_play(selected, at, false):
				selected = null
				_layout_hand()


func _can_afford(h: Dictionary) -> bool:
	return int(_card_def(h["id"]).get("cost", 0)) <= energy


func _shake_card(h: Dictionary) -> void:
	var v: Control = h["view"]
	var tw := v.create_tween()
	var x := v.position.x
	for k in [6.0, -6.0, 4.0, -2.0, 0.0]:
		tw.tween_property(v, "position:x", x + k, 0.04)


func _foe_rect() -> Rect2:
	return Rect2(foe_root.position * 2.0 - Vector2(110, 200), Vector2(220, 240))


func _try_play(h: Dictionary, at: Vector2, dragged: bool) -> bool:
	var d := _card_def(h["id"])
	var to_enemy := str(d.get("target", "enemy")) == "enemy"
	if to_enemy and not _foe_rect().has_point(at):
		if dragged:
			_say("Drop attacks on the enemy", 0.8)
		return false
	if not to_enemy and at.y > HAND_Y - 20.0:
		return false
	if not _can_afford(h):
		_say("Not enough energy", 1.0)
		_shake_card(h)
		return false
	_play(h)
	return true


func _play(h: Dictionary) -> void:
	var d := _card_def(h["id"])
	busy = true
	selected = null
	hovered = null
	hand.erase(h)
	energy -= int(d.get("cost", 0))
	var v: Control = h["view"]
	v.z_index = 70
	var to_enemy := str(d.get("target", "enemy")) == "enemy"
	var target_ui: Vector2 = foe_root.position + Vector2(0, -50) if to_enemy else hero_root.position + Vector2(0, -50)
	_log("Played " + str(d.get("name", h["id"])))
	# lift, fly toward the target, effect, then off to the discard pile
	var tw := create_tween().set_trans(Tween.TRANS_QUAD)
	tw.tween_property(v, "position:y", v.position.y - 60, 0.1).set_ease(Tween.EASE_OUT)
	tw.parallel().tween_property(v, "scale", Vector2(1.15, 1.15), 0.1)
	tw.parallel().tween_property(v, "rotation", 0.0, 0.1)
	tw.tween_property(v, "position", target_ui * 2.0 - Vector2(63, 140), 0.2).set_ease(Tween.EASE_IN)
	await tw.finished
	_anim(hero, str(d.get("anim", "idle")), true)
	await _apply(d)
	var exhaust := false
	for op in d.get("ops", []):
		if str(op[0]) == "exhaust":
			exhaust = true
	var tw2 := create_tween().set_parallel(true).set_trans(Tween.TRANS_QUAD).set_ease(Tween.EASE_IN)
	var dest := Vector2(1200, 660) if not exhaust else v.position + Vector2(0, -80)
	tw2.tween_property(v, "position", dest, 0.25)
	tw2.tween_property(v, "scale", Vector2(0.25, 0.25), 0.25)
	tw2.tween_property(v, "modulate:a", 0.0, 0.25)
	await tw2.finished
	v.queue_free()
	if exhaust:
		exhausted.append(h["id"])
	else:
		discard.append(h["id"])
	busy = false
	_layout_hand()
	_update_ui()
	if enemy["hp"] <= 0:
		_won_fight()


func _apply(d: Dictionary) -> void:
	var fx := str(d.get("fx", ""))
	for op in d.get("ops", []):
		var k := str(op[0])
		var val := int(op[1])
		var times := int(op[2]) if op.size() > 2 else 1
		match k:
			"dmg":
				for t in times:
					if enemy["hp"] <= 0:
						break
					_fx(fx, foe_root.position + Vector2(0, -36))
					_hit_enemy(val + strength)
					await get_tree().create_timer(0.22, false).timeout
			"block":
				block += val
				_fx("buff" if fx != "frost" else fx, hero_root.position + Vector2(0, -30))
				_number("+%d block" % val, hero_root.position + Vector2(0, -90), Color(0.6, 0.8, 1.0))
			"heal":
				var before := hp
				hp = mini(max_hp, hp + val)
				_fx("heal", hero_root.position + Vector2(0, -36))
				_number("+%d" % (hp - before), hero_root.position + Vector2(0, -90), Color(0.5, 1.0, 0.55))
			"draw":
				_fx(fx, hero_root.position + Vector2(0, -36))
				_draw_cards(val)
			"str":
				strength += val
				_fx("buff", hero_root.position + Vector2(0, -36))
				_number("+%d strength" % val, hero_root.position + Vector2(0, -90), Color(1.0, 0.7, 0.4))
			"poison":
				enemy["poison"] += val
				_fx("poison", foe_root.position + Vector2(0, -30))
				_number("poison %d" % val, foe_root.position + Vector2(0, -100), Color(0.6, 1.0, 0.4))
			"burn":
				enemy["burn"] += val
				_number("burn %d" % val, foe_root.position + Vector2(0, -110), Color(1.0, 0.55, 0.3))
			"energy":
				energy += val
				_fx("dark", hero_root.position + Vector2(0, -36))
				_number("+%d energy" % val, hero_root.position + Vector2(0, -100), Color(1.0, 0.85, 0.35))
			"lose_hp":
				_hurt_player(val, true)
		_update_ui()
	await get_tree().create_timer(0.18, false).timeout


func _hit_enemy(dmg: int) -> void:
	var left := dmg
	if enemy["block"] > 0:
		var b: int = mini(enemy["block"], left)
		enemy["block"] -= b
		left -= b
		if b > 0:
			_number("blocked %d" % b, foe_root.position + Vector2(10, -120), Color(0.6, 0.8, 1.0))
	enemy["hp"] = maxi(0, enemy["hp"] - left)
	if left > 0:
		_number(str(left), foe_root.position + Vector2(rng.randf_range(-14, 14), -100), Color(1.0, 0.95, 0.4))
		_shake(foe_root)
		foe.modulate = Color(1.0, 0.5, 0.5)
		var wr: WeakRef = weakref(foe)
		get_tree().create_timer(0.12, false).timeout.connect(func():
			var n = wr.get_ref()
			if n != null:
				n.modulate = Color.WHITE)
		if enemy["hp"] > 0:
			_anim(foe, "hurt", true)
	_update_ui()


func _hurt_player(dmg: int, direct := false) -> void:
	var left := dmg
	if not direct and block > 0:
		var b := mini(block, left)
		block -= b
		left -= b
		if b > 0:
			_number("blocked %d" % b, hero_root.position + Vector2(-10, -120), Color(0.6, 0.8, 1.0))
	hp = maxi(0, hp - left)
	if left > 0:
		_number(str(left), hero_root.position + Vector2(rng.randf_range(-14, 14), -100), Color(1.0, 0.45, 0.4))
		_shake(hero_root)
		_anim(hero, "hurt", true)
	_update_ui()
	if hp <= 0:
		busy = true
		_anim(hero, "death")
		await get_tree().create_timer(1.2, false).timeout
		_end_screen(false)


func _end_turn() -> void:
	if screen != "fight" or busy:
		return
	busy = true
	selected = null
	hovered = null
	for h in hand:                          # the hand goes to the discard pile
		var v: Control = h["view"]
		var tw := v.create_tween().set_parallel(true)
		tw.tween_property(v, "position", Vector2(1200, 660), 0.22)
		tw.tween_property(v, "scale", Vector2(0.25, 0.25), 0.22)
		tw.chain().tween_callback(v.queue_free)
		discard.append(h["id"])
	hand.clear()
	_update_ui()
	await get_tree().create_timer(0.35, false).timeout
	await _enemy_turn()
	if screen == "fight" and hp > 0 and enemy["hp"] > 0:
		_begin_turn()


func _enemy_turn() -> void:
	enemy["block"] = 0
	for st in ["poison", "burn"]:
		if enemy[st] > 0:
			_number("%s %d" % [st, enemy[st]], foe_root.position + Vector2(0, -130), Color(0.6, 1.0, 0.4) if st == "poison" else Color(1.0, 0.55, 0.3))
			_hit_enemy(enemy[st])
			enemy[st] -= 1
			await get_tree().create_timer(0.3, false).timeout
	if enemy["hp"] <= 0:
		_won_fight()
		return
	var it: Array = enemy["intent"]
	var e: Dictionary = ENEMIES[enemy["key"]]
	match str(it[0]):
		"attack":
			for t in int(it[2]):
				await _enemy_lunge()
				_fx("slash" if str(e["char"]) != "wolf" else "spark", hero_root.position + Vector2(0, -40))
				_hurt_player(int(it[1]) + int(enemy["str"]))
				if hp <= 0:
					return
				await get_tree().create_timer(0.25, false).timeout
		"block_attack":
			enemy["block"] += int(it[1])
			await _enemy_lunge()
			_hurt_player(int(it[2]) + int(enemy["str"]))
		"block":
			enemy["block"] += int(it[1])
			_fx("buff", foe_root.position + Vector2(0, -36))
		"buff":
			enemy["str"] += int(it[1])
			_fx("levelup", foe_root.position + Vector2(0, -36))
			_number("+%d strength" % int(it[1]), foe_root.position + Vector2(0, -110), Color(1.0, 0.6, 0.4))
		"poison":
			poison += int(it[1])
			_fx("poison", hero_root.position + Vector2(0, -30))
			_number("poisoned", hero_root.position + Vector2(0, -110), Color(0.6, 1.0, 0.4))
		"charge":
			_fx("dark", foe_root.position + Vector2(0, -40))
			_say(str(enemy["name"]) + " gathers dark power...", 1.2)
	if hp <= 0:
		return
	enemy["turn"] += 1
	var pat: Array = e["pattern"]
	enemy["intent"] = pat[int(enemy["turn"]) % pat.size()]
	await get_tree().create_timer(0.35, false).timeout
	_update_ui()


func _enemy_lunge() -> void:
	_anim(foe, "attack" if foe.sprite_frames.has_animation("attack_W") else "run", true)
	var home := foe_root.position
	var tw := create_tween()
	tw.tween_property(foe_root, "position:x", home.x - 40, 0.12).set_ease(Tween.EASE_OUT)
	tw.tween_property(foe_root, "position:x", home.x, 0.18)
	await get_tree().create_timer(0.12, false).timeout


func _won_fight() -> void:
	if screen != "fight":
		return
	screen = "reward"
	busy = true
	var e: Dictionary = ENEMIES[enemy["key"]]
	_anim(foe, "death")
	var tw := create_tween()
	tw.tween_interval(0.7)
	tw.tween_property(foe_root, "modulate:a", 0.0, 0.4)
	await tw.finished
	for h in hand:
		h["view"].queue_free()
	hand.clear()
	var g := int(e.get("gold", 15)) + rng.randi_range(0, 8)
	gold += g
	_log("Won against %s (+%d gold)" % [e["name"], g])
	if e.get("boss", false) or floor_i >= MAP_ROWS.size() - 1:
		_end_screen(true)
		return
	_show_reward(g)


func _show_reward(g: int) -> void:
	_set_fight_visible(false)
	cards_layer.visible = true
	_clear(cards_layer)
	_clear(overlay)
	_panel(Rect2(60, 40, 520, 290), overlay)
	_label("Victory! +%d gold.  Pick a card for your deck:" % g, Vector2(80, 52), overlay)
	var pool: Array = defs.get("reward_pool", []).duplicate()
	pool.shuffle()
	for i in 3:
		var id: String = pool[i]
		var v := _card_view(id)
		v.position = Vector2(260 + i * 300, 170)
		v.pivot_offset = Vector2(63, 88)
		cards_layer.add_child(v)
		v.mouse_entered.connect(func(): v.create_tween().tween_property(v, "scale", Vector2(1.5, 1.5), 0.1))
		v.mouse_exited.connect(func(): v.create_tween().tween_property(v, "scale", Vector2.ONE, 0.1))
		v.gui_input.connect(func(ev):
			if ev is InputEventMouseButton and ev.pressed and ev.button_index == MOUSE_BUTTON_LEFT:
				_take_reward(id))
	_button("Skip", Rect2(270, 300, 100, 22), overlay, _take_reward.bind(""))
	_update_ui()


func _take_reward(id: String) -> void:
	if screen != "reward":
		return
	if id != "":
		deck.append(id)
		_log("Took " + str(_card_def(id).get("name", id)))
	_clear(cards_layer)
	cards_layer.visible = false
	_after_node()


func _process(delta: float) -> void:
	if message_t > 0.0:
		message_t -= delta
		message.visible = message_t > 0.0


# ------------------------------------------------------------------ PPDebug + checks

func debug_state() -> Dictionary:
	var d := {"screen": screen, "floor": "%d / %d" % [floor_i + 1, MAP_ROWS.size()], "path": " > ".join(path_taken),
		"player": "hp %d/%d  block %d  str %d  poison %d  energy %d/%d  gold %d" % [hp, max_hp, block, strength, poison, energy, max_energy, gold],
		"piles": "deck %d  draw %d  hand %d  discard %d  exhausted %d" % [deck.size(), draw_pile.size(), hand.size(), discard.size(), exhausted.size()],
		"busy": busy}
	if hovered != null:
		var cd := _card_def(hovered["id"])
		d["hovered card"] = "%s (%s) cost %d %s  playable %s" % [cd.get("name", ""), hovered["id"], int(cd.get("cost", 0)), cd.get("type", ""), _can_afford(hovered)]
	if selected != null:
		d["selected card"] = selected["id"]
	if dragging != null:
		d["dragging"] = "%s at %s" % [dragging["id"], dragging["view"].position.round()]
	if not enemy.is_empty() and screen == "fight":
		d["enemy"] = "%s hp %d/%d block %d str %d poison %d burn %d intent %s" % [enemy["name"], enemy["hp"], enemy["max_hp"], enemy["block"], enemy["str"], enemy["poison"], enemy["burn"], str(enemy["intent"])]
		d["hero anim"] = "%s frame %d" % [hero.animation, hero.frame + 1]
		d["enemy anim"] = "%s frame %d" % [foe.animation, foe.frame + 1] if is_instance_valid(foe) else ""
	d["log"] = " | ".join(log_lines.slice(-4))
	return d


func _wait(t: float) -> void:
	await get_tree().create_timer(t, false).timeout


func _until(cond: Callable, limit: float) -> void:
	var t := 0.0
	while not cond.call() and t < limit:
		await get_tree().process_frame
		t += maxf(get_process_delta_time(), 0.001)


func selftest() -> Dictionary:
	var out := {"screens": []}
	await _wait(0.3)
	_toggle_deck_view()
	await _wait(0.2)
	out["deck_view_cards"] = deck_view.get_child_count() - 2 if is_instance_valid(deck_view) else -1
	_toggle_deck_view()
	var fights := 0
	for row in MAP_ROWS.size():
		if screen != "map":
			break
		_enter_node(row, MAP_ROWS[row][0])
		out["screens"].append(screen)
		if screen != "fight":
			continue
		fights += 1
		var turns := 0
		while screen == "fight" and turns < 14:
			turns += 1
			await _until(func(): return not busy, 5.0)
			var guard := 0
			while screen == "fight" and guard < 12:
				guard += 1
				var playable := hand.filter(func(h): return _can_afford(h))
				if playable.is_empty():
					break
				var h: Dictionary = playable[0]
				var to_enemy := str(_card_def(h["id"]).get("target", "")) == "enemy"
				_try_play(h, _foe_rect().get_center() if to_enemy else Vector2(640, 200), true)
				await _until(func(): return not busy, 5.0)
			if screen == "fight":
				_end_turn()
				await _until(func(): return not busy or screen != "fight", 10.0)
		out["fight%d" % fights] = {"turns": turns, "result": screen, "hp": hp}
		if screen == "reward":
			_take_reward(str(defs["reward_pool"][0]))
		if screen in ["over", "won"]:
			break
	out["final_screen"] = screen
	out["deck"] = deck.size()
	out["gold"] = gold
	return {"cards": out}


func shot_setup() -> void:
	_enter_node(0, MAP_ROWS[0][0])
	await _wait(0.8)
	if hand.size() > 2:
		hovered = hand[2]
		_layout_hand()

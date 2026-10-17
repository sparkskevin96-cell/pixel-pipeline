extends RefCounted
## Reads the pixel_pipeline_manifest.json that gameexport.export_to_game writes, so a game loads
## the app's assets by id ("character/elf-archer") or by name ("elf_archer").
##   var m = preload("res://pp_runtime/pp_manifest.gd").new("res://art")
##   var e: Dictionary = m.entry("character/elf-archer")
##   var frames: SpriteFrames = m.sprite_frames(e)

var root := "res://art"
var data := {}


func _init(res_root := "res://art") -> void:
	root = res_root
	var path := root.path_join("pixel_pipeline_manifest.json")
	if FileAccess.file_exists(path):
		var j = JSON.parse_string(FileAccess.get_file_as_string(path))
		if j is Dictionary:
			data = j
	if data.is_empty():
		push_error("pp_manifest: no manifest at " + path)


func entry(id_or_name: String) -> Dictionary:
	var assets: Dictionary = data.get("assets", {})
	if assets.has(id_or_name):
		return assets[id_or_name]
	var by_name: Dictionary = data.get("by_name", {})
	if by_name.has(id_or_name):
		return assets.get(by_name[id_or_name], {})
	return {}


func res(e: Dictionary, role: String) -> String:
	return str(e.get("res", {}).get(role, ""))


func folder(e: Dictionary) -> String:
	return root.path_join(str(e.get("folder", "")))


func sprite_frames(e: Dictionary) -> SpriteFrames:
	var p := res(e, "sprite_frames")
	if p != "" and ResourceLoader.exists(p):
		return load(p)
	return null


func anchor(e: Dictionary) -> Vector2:
	# the feet: bottom-centre of the standing pixels, in frame pixels
	var a: Dictionary = e.get("anchor", {})
	var f: Dictionary = e.get("frame", {"w": 32, "h": 32})
	return Vector2(float(a.get("x", f["w"] / 2.0)), float(a.get("y", f["h"])))


func frame_size(e: Dictionary) -> Vector2:
	var f: Dictionary = e.get("frame", {"w": 32, "h": 32})
	return Vector2(float(f["w"]), float(f["h"]))


func has_anim(e: Dictionary, action: String) -> bool:
	return e.get("animations", {}).has(action)


func texture(path: String) -> Texture2D:
	if ResourceLoader.exists(path):
		return load(path)
	if FileAccess.file_exists(path):
		var img := Image.load_from_file(ProjectSettings.globalize_path(path))
		if img != null:
			return ImageTexture.create_from_image(img)
	return null

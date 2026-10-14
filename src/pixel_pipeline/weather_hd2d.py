"""Weather in the HD-2D test bed (hd2d.make_hd2d(weather=...)): the weather assets' 3D particle
scenes over the board plus lighting that matches - wet, darker, glossier ground and grey light
for rain; cool bright light for snow; thick fog in its own colour; dusty orange haze for
sandstorms; light shafts (volumetric fog in the real-3D look) for god rays; lightning flashes.

    make_hd2d(out, units, weather="rain")          # or "storm" (rain + lightning), "snow", "fog",
                                                   # ["leaves", "god_rays"], a saved asset...
Files: res://weather/<kind>/ (write_weather output), res://weather/weather.json (what to load,
wet / snow amounts, volumetric sun), res://weather/board_weather.gd (board.gd adds it when the
file exists). lighting.json presets are adjusted per weather (tweak_lighting), so L still cycles
the time of day and every preset stays rainy. The scenes reference imported textures: run
`godot --import` once (the Game tab and the editor do)."""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

PRESETS = {
    "storm": ["rain", "lightning"],
    "blizzard": ["snow"],
}

BOARD_WEATHER_GD = r'''extends Node3D
# pixel-pipeline weather over the HD-2D board: loads res://weather/<kind>/sheet_particles3d.tscn
# for each kind in weather.json, fits them to the board, puts splashes / ripples on the cell
# tops, makes the ground wet (darker, glossy) or snowy, and keeps the sun's light shafts.
var board
var info := {}
var scenes := []
var wet_done := false

func setup(b) -> void:
	board = b
	info = JSON.parse_string(FileAccess.get_file_as_string("res://weather/weather.json"))
	position = Vector3(board.cols / 2.0, 0.0, board.rows / 2.0)
	for kind in info.get("kinds", []):
		var path := "res://weather/%s/sheet_particles3d.tscn" % kind
		if not ResourceLoader.exists(path):
			push_warning("weather: missing " + path)
			continue
		var s: Node3D = load(path).instantiate()
		add_child(s)
		scenes.append(s)
		_fit(s, str(kind))
	_ground()

func _fit(s: Node3D, kind: String) -> void:
	var sx: float = board.cols / 16.0 + 0.15
	var sz: float = board.rows / 12.0 + 0.15
	for p in s.get_children():
		if p is GPUParticles3D:
			var pm: ParticleProcessMaterial = p.process_material.duplicate()
			var e: Vector3 = pm.emission_box_extents
			pm.emission_box_extents = Vector3(e.x * sx, e.y, e.z * sz)
			if kind in ["fog", "sandstorm"] and p.position.y > 0.5 and p.position.y < 6.0:
				p.position.y = 1.0               # fog banks hug the board
				pm.emission_box_extents.y = 0.9
			p.process_material = pm
			p.amount = maxi(1, int(p.amount * sx * sz))
			var v: AABB = p.visibility_aabb
			p.visibility_aabb = AABB(v.position * Vector3(sx, 1, sz), v.size * Vector3(sx, 1, sz))
			if str(p.name) in ["Splash", "Ripple"]:
				_on_tops(p)

func _on_tops(p: GPUParticles3D) -> void:
	# emit from the cell tops (dry land and the water surface), not a flat box
	var pts := PackedVector3Array()
	var rng := RandomNumberGenerator.new()
	rng.seed = 7
	for i in 512:
		var c := rng.randi_range(0, board.cols - 1)
		var r := rng.randi_range(0, board.rows - 1)
		var y: float = board.top_y(c, r)
		if board.terrain[r][c] == "water":
			y = float(board.map.get("water_level", 0.15)) + 0.03
		pts.append(Vector3(c + rng.randf(), y + 0.02, r + rng.randf()) - position - p.position)
	var img := Image.create(pts.size(), 1, false, Image.FORMAT_RGBF)
	for i in pts.size():
		img.set_pixel(i, 0, Color(pts[i].x, pts[i].y, pts[i].z))
	var pm: ParticleProcessMaterial = p.process_material
	pm.emission_shape = ParticleProcessMaterial.EMISSION_SHAPE_POINTS
	pm.emission_point_texture = ImageTexture.create_from_image(img)
	pm.emission_point_count = pts.size()

func _ground() -> void:
	var wet := float(info.get("wet", 0.0))
	var snow := float(info.get("snow", 0.0))
	if wet <= 0.0 and snow <= 0.0:
		return
	var seen := {}
	for n in board.get_children():
		if n is MeshInstance3D and n.material_override is StandardMaterial3D:
			var m: StandardMaterial3D = n.material_override
			if seen.has(m):
				continue
			seen[m] = true
			if wet > 0.0:                     # rain: darker, glossy, a little bluer
				m.albedo_color = m.albedo_color * Color(1.0 - 0.3 * wet, 1.0 - 0.28 * wet, 1.0 - 0.22 * wet)
				m.roughness = lerpf(m.roughness, 0.25, wet)
				m.metallic_specular = lerpf(0.5, 0.85, wet)
			if snow > 0.0:                    # snow: frosted, cooler
				m.albedo_color = m.albedo_color.lerp(Color(0.92, 0.95, 1.05), snow)
	wet_done = true

func apply(L: Dictionary) -> void:
	# called by board.apply_lighting after each time-of-day change
	if board.sun != null:
		board.sun.light_volumetric_fog_energy = float(info.get("sun_fog", 0.15))
'''


def tweak_lighting(preset: dict, kinds: list) -> dict:
    """A time-of-day preset (hd2d.LIGHTING entry) adjusted for the weather."""
    L = copy.deepcopy(preset)

    def mix(key, target, k):
        v = L.get(key, target)
        L[key] = [round(a * (1 - k) + b * k, 3) for a, b in zip(v, target)]

    dark = sum(L.get("background", [0, 0, 0])) < 0.3          # night: keep it dark
    if "rain" in kinds:
        storm = "lightning" in kinds
        L["sun_energy"] = round(L["sun_energy"] * (0.4 if storm else 0.55), 3)
        L["fill_energy"] = round(L.get("fill_energy", 0.3) * 1.25, 3)
        L["ambient_energy"] = round(L.get("ambient_energy", 0.5) * 1.05, 3)
        mix("ambient", [0.42, 0.48, 0.62], 0.35)
        mix("fog", [0.2, 0.24, 0.32] if dark else [0.42, 0.48, 0.58], 0.55)
        L["fog_density"] = max(L.get("fog_density", 0.002), 0.012)
        L["saturation"] = round(L.get("saturation", 1.1) * 0.85, 3)
        L["contrast"] = round(L.get("contrast", 1.05) * 0.98, 3)
        L["exposure"] = round(L.get("exposure", 1.0) * (0.9 if storm else 0.95), 3)
        L["sparkle"] = round(L.get("sparkle", 2.5) * 0.5, 3)
        mix("background", [0.08, 0.09, 0.12] if dark else [0.3, 0.33, 0.4], 0.5)
    if "snow" in kinds:
        L["sun_energy"] = round(L["sun_energy"] * 0.85, 3)
        L["ambient_energy"] = round(L.get("ambient_energy", 0.5) * 1.15, 3)
        mix("ambient", [0.72, 0.78, 0.95], 0.3)
        mix("fog", [0.3, 0.34, 0.45] if dark else [0.8, 0.85, 0.95], 0.5)
        L["fog_density"] = max(L.get("fog_density", 0.002), 0.008)
        L["saturation"] = round(L.get("saturation", 1.1) * 0.9, 3)
    if "fog" in kinds:
        mix("fog", [0.2, 0.22, 0.3] if dark else [0.7, 0.72, 0.8], 0.6)
        L["fog_density"] = max(L.get("fog_density", 0.002), 0.032)
        L["volumetric"] = max(L.get("volumetric", 0.0), 0.01)
        L["sun_energy"] = round(L["sun_energy"] * 0.7, 3)
        L["contrast"] = round(L.get("contrast", 1.05) * 0.95, 3)
    if "sandstorm" in kinds:
        mix("fog", [0.78, 0.58, 0.38], 0.7)
        L["fog_density"] = max(L.get("fog_density", 0.002), 0.035)
        mix("sun", [1.0, 0.7, 0.4], 0.4)
        L["sun_energy"] = round(L["sun_energy"] * 0.6, 3)
        L["saturation"] = round(L.get("saturation", 1.1) * 0.85, 3)
        mix("ambient", [0.7, 0.55, 0.42], 0.3)
    if "god_rays" in kinds:
        L["volumetric"] = max(L.get("volumetric", 0.0), 0.006)
    return L


def _kinds(weather) -> list:
    """weather= -> [(kind, asset or None)]: a kind name, a preset (storm), an asset, or a list."""
    from .weather import KINDS
    items = weather if isinstance(weather, (list, tuple)) else [weather]
    out = []
    for w in items:
        if isinstance(w, str):
            for k in PRESETS.get(w, [w]):
                if k not in KINDS:
                    raise ValueError(f"Unknown weather '{k}' (use {', '.join(KINDS)} or {', '.join(PRESETS)})")
                out.append((k, None))
        else:                                  # a saved weather asset
            out.append((w.extra.get("weather"), w))
    return out


def install(project, weather, tod: str = "sunset", ppu: float = 42.0, seed: int = 0) -> list[Path]:
    """Write res://weather/ for make_hd2d and adjust its lighting.json (must exist)."""
    from .weather import render_weather, write_weather
    out = Path(project)
    wdir = out / "weather"
    wdir.mkdir(parents=True, exist_ok=True)
    kinds = _kinds(weather)
    written = []
    names = []
    for kind, asset in kinds:
        dst = wdir / kind
        if asset is not None:
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(asset.folder, dst)
            written.append(dst)
        else:
            w = render_weather(kind, tod=tod, seed=seed, ground=True)
            written += write_weather(w, dst, ppu=ppu)
        names.append(kind)
    info = {"kinds": names, "wet": 0.8 if "rain" in names else 0.0, "snow": 0.22 if "snow" in names else 0.0,
            "sun_fog": 1.6 if "god_rays" in names else (0.6 if "fog" in names else 0.15)}
    js = wdir / "weather.json"
    js.write_text(json.dumps(info, indent=1), encoding="utf-8")
    gd = wdir / "board_weather.gd"
    gd.write_text(BOARD_WEATHER_GD, encoding="utf-8")
    written += [js, gd]
    lj = out / "lighting.json"
    if lj.exists():
        data = json.loads(lj.read_text(encoding="utf-8"))
        data["presets"] = {k: tweak_lighting(v, names) for k, v in data["presets"].items()}
        data["weather"] = names
        lj.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return written

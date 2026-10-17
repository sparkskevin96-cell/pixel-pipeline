extends RefCounted
## pixel-pipeline GAME RUNTIME: motion that makes the app's sprites play well (Godot 4).
## Shipped with every export (res://<subdir>/pp_runtime/) and used by the HD-2D board, its autoplay
## and the playtest games. Static helpers, no state of their own:
##     const M := preload("res://pp_runtime/pp_motion.gd")
##     var conf := M.settings()                         # motion.json (pixel_pipeline.motion.MOTION)
##     vel = M.approach(vel, wish * speed, conf.accel, conf.decel, delta)   # ease in / out
##     yaw = M.turn(yaw, target_yaw, conf.turn_rate, delta)                 # through the 16 facings
##     var lc := M.locomotion(speed, strides, conf)    # {clip: walk|run, rate: cycles/s}
##     phase = fposmod(phase + lc.rate * delta, 1.0); frame = int(phase * frames)   # feet planted
##     var sd := M.smooth_damp(cam_pos, target, cam_vel, conf.cam_smooth, delta)    # [pos, vel]
## STRIDE (manifest animations[clip].stride.units, unit.json stride[clip].units, tiles per cycle):
## a clip played at speed / stride cycles per second keeps its planted foot still on the ground.
## The same algorithms are written out in docs/FOR_CLAUDE.md (for a Unity/C# runtime).

const DIRS16 := ["E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW", "N", "NNE", "NE", "ENE"]
const DEFAULTS := {
	"accel": 14.0, "decel": 18.0, "turn_rate": 900.0, "walk_speed": 2.6, "run_speed": 4.6,
	"run_blend": 0.72, "settle": 0.16, "corner": 0.32, "hop": 0.14, "step_time": 0.22,
	"rise": 0.07, "gravity": 22.0, "foot_radius": 0.22, "body_radius": 0.24, "hitstop": 0.075,
	"knockback": 0.22, "shake": 0.06, "shake_decay": 2.6, "anticipation": 0.12,
	"follow_through": 0.18, "cam_smooth": 0.28, "depth_bias": 0.35,
	"default_stride": {"walk": 1.05, "run": 1.9},
	# cadence limits (cycles per second) a clip may be played at; outside them it would look frantic
	# or like slow motion, so movers cap their speed at stride x cadence_max instead of skating
	"walk_cadence": [0.5, 1.9], "run_cadence": [0.8, 2.9],
}
const HIT_AT := {"attack": 0.5, "cast": 0.58, "shoot": 0.55, "thrust": 0.45, "bite": 0.5}   # impact share of a clip

static var _conf := {}


## The tunables: res://pp_runtime/motion.json (written with the export / the board) over DEFAULTS.
static func settings(path := "") -> Dictionary:
	if not _conf.is_empty() and path == "":
		return _conf
	var c := DEFAULTS.duplicate(true)
	for p in ([path] if path != "" else ["res://pp_runtime/motion.json", "res://art/pp_runtime/motion.json"]):
		if p != "" and FileAccess.file_exists(p):
			var j = JSON.parse_string(FileAccess.get_file_as_string(p))
			if j is Dictionary:
				c.merge(j, true)
			break
	_conf = c
	return c


# ------------------------------------------------------------------ movement

## Velocity eased toward `target` (any Vector2/Vector3): `accel` while speeding up or turning,
## `decel` while slowing down (units/s^2). No snapping to full speed, no instant stops.
static func approach(v, target, accel: float, decel: float, delta: float):
	var rate := accel if target.length() >= v.length() - 0.0001 else decel
	return v.move_toward(target, rate * delta)


## Speed along a path of `remaining` length: up to `want`, eased in from `speed` and braking
## so it comes to rest exactly at the end.
static func path_speed(speed: float, want: float, remaining: float, accel: float, decel: float, delta: float) -> float:
	var cap := sqrt(maxf(0.0, 2.0 * decel * remaining))
	var goal := minf(want, cap)
	if goal >= speed:
		return minf(goal, speed + accel * delta)
	return maxf(goal, speed - decel * delta * 1.5)


## A yaw (radians) turned toward `target` by at most rate_deg degrees per second, the short way.
## Sprites pick their facing from the yaw every frame, so a turn shows the facings in between.
static func turn(yaw: float, target: float, rate_deg: float, delta: float) -> float:
	var d := wrapf(target - yaw, -PI, PI)
	var step := deg_to_rad(rate_deg) * delta
	if absf(d) <= step:
		return target
	return yaw + signf(d) * step


## Critically damped spring (no overshoot, no lag jumps): [position, velocity] after delta.
## smooth_time ~ the time to catch up. Works for Vector2/Vector3/float.
static func smooth_damp(cur, target, vel, smooth_time: float, delta: float) -> Array:
	var t := maxf(0.0001, smooth_time)
	var omega := 2.0 / t
	var x := omega * delta
	var e := 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)
	var change = cur - target
	var temp = (vel + change * omega) * delta
	var nv = (vel - temp * omega) * e
	var out = target + (change + temp) * e
	return [out, nv]


# ------------------------------------------------------------------ locomotion (stride-matched)

## Tiles per cycle of `clip` from a stride dict ({clip: {units}} or {clip: units}); else the
## default share of the body height (body_tiles) or 1 tile.
static func stride_of(strides: Dictionary, clip: String, body_tiles := 1.0) -> float:
	var v = strides.get(clip, null)
	if v is Dictionary:
		v = v.get("units", null)
	if v != null and float(v) > 0.001:
		return float(v)
	var k: Dictionary = settings().get("default_stride", {})
	return float(k.get("run" if clip.begins_with("run") else "walk", 1.0)) * maxf(body_tiles, 0.2)


## The fastest a unit can go with its feet planted (run, else walk, at the top cadence).
static func max_speed(strides: Dictionary, has_run: bool, body_tiles := 1.0) -> float:
	var c := settings()
	if has_run:
		return stride_of(strides, "run", body_tiles) * float(c["run_cadence"][1])
	return stride_of(strides, "walk", body_tiles) * float(c["walk_cadence"][1])


## Which clip and how fast to play it for a ground speed (tiles/s): walk below the blend point,
## run above (when the sheet has one); rate = speed / stride cycles per second, so the planted
## foot moves backward exactly as fast as the ground goes by. {clip, rate, stride}
static func locomotion(speed: float, strides: Dictionary, has_run: bool, body_tiles := 1.0, current := "") -> Dictionary:
	var c := settings()
	var walk := stride_of(strides, "walk", body_tiles)
	var walk_top := walk * float(c["walk_cadence"][1])
	var clip := "walk"
	if has_run:
		var edge := walk_top * 0.92
		if current == "run":
			edge *= 0.85                      # hysteresis: no flicker at the blend speed
		if speed > edge:
			clip = "run"
	var st := stride_of(strides, clip, body_tiles)
	return {"clip": clip, "rate": speed / maxf(st, 0.001), "stride": st}


## Frame of a looping clip at a cycle phase (0..1): no frame is skipped or doubled by the clock.
static func frame_at(phase: float, frames: int) -> int:
	return clampi(int(floor(fposmod(phase, 1.0) * frames)), 0, maxi(frames - 1, 0))


## Where a stopping walk settles: the next contact pose (a foot down: phase 0 or 0.5).
static func settle_target(phase: float) -> float:
	return 0.5 if fposmod(phase, 1.0) < 0.5 else 1.0


# ------------------------------------------------------------------ height

## Height along a step from ground ya to yb over u in 0..1 (the feet cross the cell border at
## u = 0.5): smoothstep between them plus an arc that clears the higher edge by `hop`.
static func hop_y(u: float, ya: float, yb: float, hop: float) -> float:
	var k := clampf(u, 0.0, 1.0)
	var s := k * k * (3.0 - 2.0 * k)
	var arc := 4.0 * k * (1.0 - k) * (absf(yb - ya) * 0.5 + hop)
	return lerpf(ya, yb, s) + arc


## Free movement over steps: feet follow `ground` (the highest cell under the foot circle):
## climbing is quick (rise seconds, never under the surface), dropping off a ledge falls with
## gravity. [y, vertical speed]
static func ground_follow(y: float, vy: float, ground: float, rise: float, gravity: float, delta: float) -> Array:
	if ground >= y - 0.0001:
		var ny := move_toward(y, ground, maxf(ground - y, 0.02) / maxf(rise, 0.001) * delta * 1.6)
		return [maxf(ny, y), 0.0]
	vy -= gravity * delta
	y += vy * delta
	if y <= ground:
		return [ground, 0.0]
	return [y, vy]


# ------------------------------------------------------------------ feel

## Camera shake offset for a trauma value (0..1): trauma^2 x amplitude, smooth noise.
static func shake_offset(trauma: float, t: float, amplitude: float) -> Vector2:
	var k := trauma * trauma * amplitude
	return Vector2(sin(t * 47.0) + 0.5 * sin(t * 113.0 + 1.3), cos(t * 53.0 + 0.7) + 0.5 * sin(t * 97.0 + 2.1)) * k / 1.5


## Seconds into an attack/cast clip where the blow lands (HIT_AT share of its length).
static func hit_time(clip: String, length: float) -> float:
	return length * float(HIT_AT.get(clip, 0.5))


## The direction name (of 16) a world yaw shows as, seen from a camera basis.
static func sprite_dir(world_yaw: float, cam_basis: Basis) -> String:
	var d := Vector3(cos(world_yaw), 0, sin(world_yaw))
	var right := cam_basis.x
	var fwd := -cam_basis.z
	fwd.y = 0
	fwd = fwd.normalized()
	var a := atan2(-d.dot(fwd), d.dot(right))
	return DIRS16[int(round(fposmod(a, TAU) / (TAU / 16.0))) % 16]

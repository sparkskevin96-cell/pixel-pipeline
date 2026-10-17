shader_type spatial;
// pixel-pipeline game runtime: a sprite-sheet frame on a quad that stands up and faces the camera
// (turning about the vertical axis only, like HD-2D), lit by every scene light through the sheet's
// normal map. Used by the HD-2D board (res://shaders/sprite.gdshader is this file).
//
// DEPTH AT THE FEET (the HD-2D fix for clipping): the quad is drawn where it stands, but every
// pixel gets the depth of the sprite's FOOT point (the node origin), moved `depth_bias` toward the
// camera. The sprite is then a flat card at its feet as far as the depth buffer knows: ground,
// walls, stairs and props BEHIND the feet never cut into it (a tall knight next to a cliff, a
// sword over a crate), things IN FRONT of the feet (a tree, a wall edge, a unit nearer the camera)
// still hide it, and two sprites never cut into each other - the nearer feet win, every frame.
// PIXEL SNAP: the whole quad moves in whole screen pixels (its feet snapped), so its texels keep
// the same pixel pattern while it walks - no shimmer from sub-pixel positions.
// Pixel-stable lighting: each light's diffuse term is snapped to `bands` steps (a toon ramp), so
// a torch passing by moves whole bands over the pixel art instead of smearing it. Rim light on
// the silhouette and on edges turned toward a light (strongest when the light is behind).
// <sheet>_glow.png (alpha x max(rgb)) makes those pixels EMISSIVE: only they bloom.
render_mode cull_disabled, depth_prepass_alpha;

uniform sampler2D sheet : source_color, filter_nearest;
uniform sampler2D normals : hint_normal, filter_nearest;
uniform sampler2D glow : hint_default_black, filter_nearest;
uniform bool has_normals = false;
uniform bool has_glow = false;
uniform float glow_energy = 3.5;
uniform vec4 frame = vec4(0.0, 0.0, 1.0, 1.0);   // uv x, y, w, h
uniform bool face_camera_fully = false;           // fake 3D: no foreshortening
uniform bool shadow_caster = false;               // the shadow-only twin (see vertex)
uniform bool flat_on_ground = false;             // ground effects: the quad lies as placed
uniform bool flip_x = false;                      // mirrored (a projectile flying left)
uniform float lean = 0.0;                         // 0 = stand up (HD-2D), 1 = face a steep camera fully
uniform bool feet_depth = true;                   // depth of the feet (see the header)
uniform float depth_bias = 0.35;                  // world units toward the camera from the feet
uniform bool pixel_snap = true;                   // move in whole screen pixels
uniform vec3 tint = vec3(1.0);
uniform float flash = 0.0;                        // selection glow / hit flash
uniform float bands = 4.0;                        // toon ramp steps per light, 0 = smooth
uniform float wrap = 0.35;                        // half-lambert wrap: the dark side stays readable
uniform float rim = 0.5;                          // rim / backlight strength
uniform vec3 lift = vec3(0.0);                    // extra sky light on sprites (stay readable at night)

varying float edge;

void vertex() {
	if (!flat_on_ground) {
		vec3 up = face_camera_fully ? INV_VIEW_MATRIX[1].xyz : vec3(0.0, 1.0, 0.0);
		if (!face_camera_fully && lean > 0.0 && !shadow_caster) {
			up = normalize(mix(vec3(0.0, 1.0, 0.0), INV_VIEW_MATRIX[1].xyz, lean));
		}
		vec3 right = normalize(INV_VIEW_MATRIX[0].xyz);
		vec3 back = normalize(cross(right, up));
		if (face_camera_fully) { back = normalize(INV_VIEW_MATRIX[2].xyz); }
		float sx = length(MODEL_MATRIX[0].xyz);
		float sy = length(MODEL_MATRIX[1].xyz);
		MODELVIEW_MATRIX = VIEW_MATRIX * mat4(vec4(right * sx, 0.0), vec4(up * sy, 0.0),
											  vec4(back, 0.0), MODEL_MATRIX[3]);
		if (shadow_caster) {
			// rendered only into shadow maps, where the "camera" is the light: pushed away from the
			// light along its rays, so the ground shadow is the same but the visible sprite (in front)
			// no longer shadows itself
			MODELVIEW_MATRIX[3].z -= 0.45;
		}
		MODELVIEW_NORMAL_MATRIX = mat3(MODELVIEW_MATRIX);
		if (!shadow_caster) {
			vec4 clip = PROJECTION_MATRIX * (MODELVIEW_MATRIX * vec4(VERTEX, 1.0));
			vec4 foot = VIEW_MATRIX * vec4(MODEL_MATRIX[3].xyz, 1.0);
			vec4 fc = PROJECTION_MATRIX * foot;
			if (pixel_snap && fc.w > 0.0001) {
				vec2 px = (fc.xy / fc.w * 0.5 + 0.5) * VIEWPORT_SIZE;
				vec2 d = (floor(px) + 0.5 - px) / VIEWPORT_SIZE * 2.0;
				clip.xy += d * clip.w;
			}
			if (feet_depth) {
				foot.z += depth_bias;                     // toward the camera (it looks down -z)
				vec4 fb = PROJECTION_MATRIX * foot;
				clip.z = fb.z / fb.w * clip.w;            // the feet's depth on every pixel
			}
			POSITION = clip;
		}
	}
}

void fragment() {
	vec2 uv = frame.xy + vec2(flip_x ? 1.0 - UV.x : UV.x, UV.y) * frame.zw;
	vec4 c = texture(sheet, uv);
	if (c.a < 0.5) { discard; }
	ALBEDO = c.rgb * tint + vec3(flash);
	ROUGHNESS = 0.85;
	SPECULAR = 0.0;
	if (has_normals) {
		NORMAL_MAP = texture(normals, uv).rgb;
		NORMAL_MAP_DEPTH = 1.0;
	}
	// silhouette pixels (a transparent neighbour inside the frame): the rim light's home
	vec2 px = 1.0 / vec2(textureSize(sheet, 0));
	vec2 lo = frame.xy;
	vec2 hi = frame.xy + frame.zw;
	float e = 0.0;
	for (int k = 0; k < 4; k++) {
		vec2 d = k == 0 ? vec2(px.x, 0.0) : k == 1 ? vec2(-px.x, 0.0) : k == 2 ? vec2(0.0, px.y) : vec2(0.0, -px.y);
		vec2 q = uv + d;
		bool inside = all(greaterThanEqual(q, lo)) && all(lessThan(q, hi));
		e = max(e, (!inside || texture(sheet, q).a < 0.5) ? 1.0 : 0.0);
	}
	edge = e;
	EMISSION = ALBEDO * lift;
	if (has_glow) {
		vec4 g = texture(glow, uv);
		float m = g.a * max(g.r, max(g.g, g.b));
		EMISSION += c.rgb * tint * m * glow_energy;
	}
}

void light() {
	float d = clamp((dot(NORMAL, LIGHT) + wrap) / (1.0 + wrap), 0.0, 1.0) * ATTENUATION;
	if (bands > 0.5) {
		d = ceil(d * bands - 0.3) / bands;           // rounds up: dim lights keep a band
	}
	DIFFUSE_LIGHT += LIGHT_COLOR * d / PI;
	// rim: edges whose normal (in screen space) points toward the light
	vec2 n2 = NORMAL.xy;
	vec2 l2 = LIGHT.xy;
	float side = 0.0;
	if (length(n2) > 0.001 && length(l2) > 0.001) {
		side = clamp(dot(normalize(n2), normalize(l2)), 0.0, 1.0);
	}
	float back = clamp(-dot(LIGHT, VIEW), 0.0, 1.0);
	float outer = max(edge, smoothstep(0.45, 0.85, 1.0 - clamp(dot(NORMAL, VIEW), 0.0, 1.0)));
	float r = rim * outer * side * (0.3 + 0.7 * back) * ATTENUATION;
	r = r > 0.12 ? (bands > 0.5 ? ceil(r * 2.0) / 2.0 : r) : 0.0;  // crisp, not a smear
	SPECULAR_LIGHT += LIGHT_COLOR * r * 0.35 / PI;
}

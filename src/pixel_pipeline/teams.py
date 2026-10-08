"""Team colours (Fire Emblem style): one sprite, recoloured for every side.

The "team" colours are the sprite's dominant saturated hue (or the colour you name): every
palette colour of that hue family moves to the team's hue with its lightness kept, so the
shading survives. Teams: player blue, enemy red, ally green, other purple, and "spent" (a
unit that already moved, greyed out).

For Godot there's no need for a sheet per team: write_godot_teams() writes a palette strip
(row 0 = the sprite's colours, one row per team), a palette-swap shader and a material per
team. Put the material on the AnimatedSprite2D and set `team` (0 = as drawn, 1 = first team
...) and `spent` (0-1 grey).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .color import lab_to_rgb, rgb_to_lab
from .palettes import hex_to_rgb

TEAMS = {"player": "2f5fd8", "enemy": "d23a32", "ally": "2fa04a", "other": "a64fd0"}


def _lch(rgb: np.ndarray) -> np.ndarray:
    lab = rgb_to_lab(np.asarray(rgb, np.uint8).reshape(-1, 3)).astype(np.float64)
    return np.stack([lab[:, 0], np.hypot(lab[:, 1], lab[:, 2]),
                     np.degrees(np.arctan2(lab[:, 2], lab[:, 1])) % 360], 1)


def _rgb(lch: np.ndarray) -> np.ndarray:
    h = np.radians(lch[:, 2])
    lab = np.stack([lch[:, 0], lch[:, 1] * np.cos(h), lch[:, 1] * np.sin(h)], 1)
    return np.clip(np.rint(lab_to_rgb(lab.astype(np.float32))), 0, 255).astype(np.uint8)


def _hue_gap(a, b):
    return np.abs((np.asarray(a) - b + 180) % 360 - 180)


def sprite_colours(rgba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Opaque colours of a sprite (or sheet) and how many pixels use each."""
    px = rgba.reshape(-1, 4)
    px = px[px[:, 3] >= 128][:, :3]
    if not len(px):
        return np.zeros((0, 3), np.uint8), np.zeros(0, int)
    cols, counts = np.unique(px, axis=0, return_counts=True)
    return cols, counts


def team_family(colours: np.ndarray, counts: np.ndarray, colour: str | None = None,
                width: float = 28.0) -> np.ndarray:
    """Mask of the colours that count as the team colour: the given colour's hue family,
    or the saturated hue covering the most pixels."""
    lch = _lch(colours)
    if colour:
        ref = _lch(hex_to_rgb([colour.lstrip("#")]))[0]
        centre, min_c = ref[2], max(12.0, 0.55 * ref[1])   # its shades, not paler look-alikes
    else:
        sat = lch[:, 1] >= 25
        if not sat.any():
            return np.zeros(len(colours), bool)
        # gold, skin, brown and orange (hue ~45-110) are usually trim, not the team colour
        prior = np.where((lch[sat, 2] >= 45) & (lch[sat, 2] <= 110), 0.5, 1.0)
        bins = np.zeros(36)
        np.add.at(bins, (lch[sat, 2] // 10).astype(int) % 36, counts[sat] * lch[sat, 1] * prior)
        smooth = bins + np.roll(bins, 1) + np.roll(bins, -1)
        centre = (np.argmax(smooth) * 10 + 5) % 360
        fam = sat & (_hue_gap(lch[:, 2], centre) <= width)
        weight = counts[fam] * lch[fam, 1]
        rad = np.radians(lch[fam, 2])
        centre = float(np.degrees(np.arctan2((weight * np.sin(rad)).sum(),
                                             (weight * np.cos(rad)).sum())) % 360)
        min_c = max(12.0, 0.35 * lch[fam, 1].max())
    return (lch[:, 1] >= min_c) & (_hue_gap(lch[:, 2], centre) <= width)


def team_palettes(colours: np.ndarray, family: np.ndarray, teams=TEAMS, spent: bool = True) -> dict:
    """{name: colours recoloured for that team} (row-aligned with `colours`)."""
    lch = _lch(colours)
    out = {}
    if family.any():
        src_h = float(np.degrees(np.arctan2(np.sin(np.radians(lch[family, 2])).mean(),
                                            np.cos(np.radians(lch[family, 2])).mean())) % 360)
        src_c = float(np.median(lch[family, 1]))
    for name, hexcode in teams.items():
        new = colours.copy()
        if family.any():
            ref = _lch(hex_to_rgb([hexcode]))[0]
            moved = lch[family].copy()
            moved[:, 2] = (ref[2] + (moved[:, 2] - src_h)) % 360          # keep hue variation
            moved[:, 1] = moved[:, 1] * (ref[1] / max(src_c, 1e-6)) ** 0.5  # ease toward its chroma
            new[family] = _rgb(moved)
        out[name] = new
    if spent:
        grey = lch.copy()
        grey[:, 1] *= 0.12
        out["spent"] = _rgb(grey)
    return out


def recolour(rgba: np.ndarray, colours: np.ndarray, new: np.ndarray) -> np.ndarray:
    """Swap every pixel of colours[i] for new[i] (alpha kept)."""
    out = rgba.copy()
    flat = out.reshape(-1, 4)

    def key(c):
        c = c.astype(np.int64)
        return (c[..., 0] << 16) | (c[..., 1] << 8) | c[..., 2]
    keys = key(colours)
    order = np.argsort(keys)
    k = key(flat[:, :3])
    pos = np.clip(np.searchsorted(keys[order], k), 0, len(keys) - 1)
    hit = (keys[order][pos] == k) & (flat[:, 3] > 0)
    flat[hit, :3] = new[order[pos[hit]]]
    return out


def team_variants(rgba: np.ndarray, colour: str | None = None, teams=TEAMS,
                  spent: bool = True) -> tuple[dict, np.ndarray, dict]:
    """-> ({team: recoloured RGBA}, the sprite's colours, {team: palette})."""
    colours, counts = sprite_colours(rgba)
    family = team_family(colours, counts, colour)
    pals = team_palettes(colours, family, teams, spent)
    return {name: recolour(rgba, colours, pal) for name, pal in pals.items()}, colours, pals


SHADER = """shader_type canvas_item;
// Team colours: swaps each colour of row 0 of `palette` for the same column of row `team`.
// Made by pixel-pipeline; palette rows: %ROWS%.
uniform sampler2D palette : filter_nearest;
uniform int colors = 1;
uniform int team = 0;
uniform float spent : hint_range(0.0, 1.0) = 0.0;

void fragment() {
	vec4 c = texture(TEXTURE, UV);
	vec3 rgb = c.rgb;
	for (int i = 0; i < colors; i++) {
		vec3 src = texelFetch(palette, ivec2(i, 0), 0).rgb;
		if (distance(c.rgb, src) < 0.006) {
			rgb = texelFetch(palette, ivec2(i, team), 0).rgb;
			break;
		}
	}
	float grey = dot(rgb, vec3(0.299, 0.587, 0.114));
	COLOR = vec4(mix(rgb, vec3(grey * 0.9), spent), c.a);
}
"""


def write_godot_teams(colours: np.ndarray, palettes: dict, out_dir, name: str) -> list[Path]:
    """Palette strip PNG, team-colour shader and one ShaderMaterial .tres per team."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [colours] + [palettes[t] for t in palettes]
    strip = np.stack(rows).astype(np.uint8)
    strip_png = out_dir / f"{name}_teams.png"
    Image.fromarray(strip, "RGB").save(strip_png)
    shader = out_dir / "team_colours.gdshader"
    shader.write_text(SHADER.replace("%ROWS%", "0 as drawn, " + ", ".join(
        f"{i + 1} {t}" for i, t in enumerate(palettes))), encoding="utf-8")
    written = [strip_png, shader]
    for i, team in enumerate(["drawn", *palettes]):
        tres = out_dir / f"{name}_{team}.tres"
        tres.write_text(f"""[gd_resource type="ShaderMaterial" load_steps=3 format=3]

[ext_resource type="Shader" path="{shader.name}" id="1_shader"]
[ext_resource type="Texture2D" path="{strip_png.name}" id="2_palette"]

[resource]
shader = ExtResource("1_shader")
shader_parameter/palette = ExtResource("2_palette")
shader_parameter/colors = {len(colours)}
shader_parameter/team = {i}
shader_parameter/spent = 0.0
""", encoding="utf-8")
        written.append(tres)
    return written


# --------------------------------------------------------------------------- portraits

PORTRAIT_PROMPT = ("head and shoulders portrait of {prompt}, three-quarter view facing left, "
                   "looking at the viewer, centered, plain white background, soft even lighting, "
                   "clean shapes, game dialogue portrait, character concept art")


FACE_MODEL_URL = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
                  "face_detection_yunet/face_detection_yunet_2023mar.onnx")
FACE_MODEL_ENV = "PIXEL_PIPELINE_FACE_MODEL"
_face_model_path: Path | None | bool = False   # False = not looked for yet


def _face_model() -> Path | None:
    """YuNet face detector (MIT, 230 KB): $PIXEL_PIPELINE_FACE_MODEL, the cache, or a
    one-time download. None if it can't be had (portraits then use the subject's top)."""
    global _face_model_path
    if _face_model_path is not False:
        return _face_model_path
    import os
    import urllib.request
    env = os.environ.get(FACE_MODEL_ENV)
    path = Path(env) if env else Path.home() / ".cache" / "pixel-pipeline" / "models" / \
        "face_detection_yunet_2023mar.onnx"
    if not path.exists() and not env:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(FACE_MODEL_URL, timeout=30) as r:
                data = r.read()
            if not data.startswith(b"\x08"):   # an LFS pointer page instead of the model
                raise ValueError("not an ONNX file")
            path.write_bytes(data)
        except Exception:  # noqa: BLE001 - optional helper
            path = None
    _face_model_path = path if path is not None and path.exists() else None
    return _face_model_path


def face_box(rgba: np.ndarray) -> tuple[int, int, int, int] | None:
    """Most confident face (x, y, w, h), or None (no face, or no detector)."""
    import cv2
    model = _face_model()
    if model is None:
        return None
    h, w = rgba.shape[:2]
    k = min(1.0, 640 / max(h, w))
    small = cv2.resize(np.ascontiguousarray(rgba[..., 2::-1]), (max(1, round(w * k)), max(1, round(h * k))))
    try:
        det = cv2.FaceDetectorYN.create(str(model), "", (small.shape[1], small.shape[0]), 0.6)
        _, faces = det.detect(small)
    except (cv2.error, AttributeError):
        return None
    if faces is None or not len(faces):
        return None
    x, y, fw, fh = faces[np.argmax(faces[:, -1]), :4] / k
    return int(x), int(y), int(fw), int(fh)


def portrait_crop(rgba: np.ndarray, aspect: float = 1.0) -> tuple[int, int, int, int]:
    """Head-and-shoulders box (x0, y0, x1, y1): around the detected face, else the top of
    the subject (drawings and models where no face is found)."""
    h, w = rgba.shape[:2]
    face = face_box(rgba)
    if face is not None:
        fx, fy, fw, fh = face
        bw = 2.6 * fw
        bh = bw / aspect
        cx, top = fx + fw / 2, fy - 0.55 * fh
    else:
        ys, xs = np.nonzero(rgba[..., 3] >= 128)
        if not len(xs):
            return 0, 0, w, h
        sub_h = ys.max() - ys.min() + 1
        bh = 0.42 * sub_h if sub_h > 1.6 * (xs.max() - xs.min() + 1) else 0.7 * sub_h
        bw = bh * aspect
        top = ys.min() - 0.03 * sub_h
        row = rgba[int(ys.min() + 0.12 * sub_h)]
        on = np.nonzero(row[:, 3] >= 128)[0]
        cx = on.mean() if len(on) else (xs.min() + xs.max()) / 2
    x0, y0 = int(round(cx - bw / 2)), int(round(top))
    return x0, y0, int(round(x0 + bw)), int(round(y0 + bh))


def make_portrait(picture, settings, flip: bool = False):
    """Photo / drawing / front picture -> pixel-art portrait sprite (head and shoulders)."""
    from .pipeline import finish, isolate, load_image, prepare
    from .video import _crop_pad
    rgba = isolate(load_image(picture), settings)
    aspect = settings.width / (settings.height or settings.width)
    crop = _crop_pad(rgba, *portrait_crop(rgba, aspect))
    if flip:
        crop = np.ascontiguousarray(crop[:, ::-1])
    s = type(settings)(**{**settings.to_dict(), "crop_to_subject": False, "fit": "cover",
                          "anchor": "center"})
    return finish(prepare(crop, s), s)

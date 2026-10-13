"""Dialogue portraits for library characters: painted head-and-shoulders busts (Triangle
Strategy-like: a painted fantasy illustration) in several expressions, plus pixel-art versions
in the project's palette, and a 9-slice frame for game UI.

    make_portraits(character)  ->  Asset (kind "portrait") in <project>/portraits/<slug>/
        portrait_<expr>.png        painted bust, background cut out (painted_size square)
        portrait_<expr>_pixel.png  pixel-art bust (size x size, project palette)
        raw_<expr>.png             the image as the GPU made it (white background)
        sheet.png/.json            every expression in a row (the chosen style), json-array
        sheet_painted.*, sheet_pixel.*   both styles
        frame.png/.json/.tres      9-slice frame (portrait_frame), preview.png = framed row
        thumb.png

How the painted ones keep the character's identity: the bust is cropped from the character's
picture.png (YuNet face box, teams.portrait_crop), then SDXL img2img (no pixel LoRA) redraws it
as a painting at a moderate denoise. Every expression is then made FROM that neutral painting by
inpainting only the face (face_mask: YuNet eyes + mouth landmarks on the painting -> a feathered
ellipse over brows, eyes, cheeks, mouth and chin) at a high denoise with strong expression words
and "neutral expression" in the negative, same seed (seed = hash of the character's folder).
The result is composited back with the mask, so hair, clothes and background outside it are
pixel-identical to the neutral; whole-picture img2img at 0.58 (face_only=False, the old way)
barely changed faces and drifted identity when pushed. Characters without a picture fall back
to their front idle sprite, enlarged (needs a higher denoise, so less faithful).
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

EXPRESSIONS = ("neutral", "happy", "angry", "sad", "surprised")
ALL_EXPRESSIONS = ("neutral", "happy", "angry", "sad", "surprised", "determined", "smug", "hurt",
                   "thinking")
# (words:1.3) = ComfyUI prompt weighting: the expression has to win over the painting under it
EXPRESSION_WORDS = {
    "neutral": "calm neutral expression, closed mouth",
    "happy": "(big happy smile:1.4), joyful laughing face, open-mouth smile showing teeth, "
             "smiling eyes, raised cheeks",
    "angry": "(angry:1.3), (furrowed eyebrows:1.3), glaring eyes, scowl, gritted teeth",
    "sad": "(sad:1.5), (crying:1.3), (worried eyebrows:1.3), (frowning:1.3), teary eyes, pout",
    "surprised": "(shocked surprised face:1.5), (very wide open eyes:1.3), raised eyebrows, "
                 "(mouth wide open:1.4)",
    "determined": "(determined:1.4), (glaring:1.3), (v-shaped eyebrows:1.4), (furrowed "
                  "brow:1.2), lips pressed together",
    "smug": "(smug smirk:1.4), lopsided grin, half-lidded eyes, one eyebrow raised",
    "hurt": "(wincing in pain:1.5), (one eye closed:1.5), (clenched teeth:1.4), (frowning:1.2), "
            "sweat",
    "thinking": "(thinking:1.4), (eyes looking up:1.4), (pursed lips:1.3), (one eyebrow "
                "raised:1.3), pensive",
}
NEUTRAL_NEGATIVE = ("(neutral expression:1.3), expressionless, blank stare, emotionless, face "
                    "paint, tattoo, facial markings, scar, white streak, scratch, sparkles, "
                    "white dots")
EXPRESSION_NEGATIVE = {      # the opposite looks too, so e.g. "sad" can't come out smiling
    "happy": "frown, sad, angry",
    "angry": "smile, happy, calm",
    "sad": "smile, happy, grin, blood",
    "surprised": "closed mouth, narrowed eyes",
    "determined": "smile, open mouth",
    "smug": "frown, sad, open mouth",
    "hurt": "smile, grin, happy, calm, blood, tears",
    "thinking": "smile, open mouth, looking at the viewer",
}
PAINTED_PROMPT = ("{who}, head and shoulders portrait, {expression}, painted fantasy game "
                  "portrait, anime-inspired painterly illustration, detailed face, clean "
                  "lineart, soft cel shading, looking at the viewer, centered bust, plain white "
                  "background")
PAINTED_NEGATIVE = ("photo, photorealistic, 3d render, pixel art, sprite, blurry, lowres, jpeg "
                    "artifacts, text, watermark, signature, frame, border, multiple people, "
                    "cropped head, out of frame, deformed face, extra eyes, bad anatomy, "
                    "busy background, scenery")
# expressions repaint only the face: the expression goes first, words are about the face
# (the character's own words drowned the expression and "green armor" leaked into eyes and
# tears; without any, angry turned her into a man): only who the person is (person_words)
# stays, and match_colours pulls the repainted face and each eye back to the neutral's colours
EXPRESSION_PROMPT = ("close-up face of {person}, {expression}, painted fantasy game portrait, "
                     "anime-inspired painterly illustration, detailed expressive face, clean "
                     "lineart, soft cel shading")
NEUTRAL_DENOISE = 0.5        # picture -> painting
EXPRESSION_DENOISE = 0.9     # face-only inpaint of the neutral painting -> each expression
EXPRESSION_DENOISE_BY = {"sad": 0.95, "hurt": 0.95, "thinking": 0.95}   # eye-led ones need more
WHOLE_DENOISE = 0.58         # face_only=False: whole-picture img2img (0.4 barely changed faces)
SPRITE_DENOISE = 0.65        # enlarged sprite -> painting (no picture.png)
STYLES = ("painted", "pixel")
GEN_SIZE = 1024
PIXEL_ALPHA = 200            # cut-out alpha a pixel bust pixel needs


# --------------------------------------------------------------------------- character input

def _character(character):
    """A library Character, its folder, or its name -> Character."""
    from . import library
    if isinstance(character, library.Character):
        return character
    p = Path(str(character))
    if (p / "character.json").exists():
        d = json.loads((p / "character.json").read_text(encoding="utf-8"))
        return library.Character(d.get("name", p.name), d.get("kind", "person"),
                                 d.get("size", "normal"), d.get("role", ""),
                                 d.get("created", 0.0), p)
    ch = library.get(str(character))
    if ch is None:
        raise ValueError(f"No character '{character}' in the library")
    return ch


def character_prompt(ch) -> str:
    """The words that made the character (character.json / asset.json "prompt"), else its name."""
    for meta in ("character.json", "asset.json"):
        p = ch.folder / meta
        if p.exists():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except ValueError:
                continue
            text = d.get("prompt") or d.get("extra", {}).get("prompt")
            if text:
                return str(text)
    return ch.name


def _front_sprite(ch) -> np.ndarray | None:
    """The idle frame facing the viewer from the character's sheet."""
    js = ch.sheet.with_suffix(".json")
    if not ch.sheet.exists() or not js.exists():
        return None
    meta = json.loads(js.read_text(encoding="utf-8"))
    frames = {f["filename"]: f["frame"] for f in meta["frames"]}
    key = next((k for k in ("idle_S 0", "idle_SE 0", "idle_SSE 0") if k in frames),
               meta["frames"][0]["filename"])
    r = frames[key]
    im = np.asarray(Image.open(ch.sheet).convert("RGBA"))
    return im[r["y"]:r["y"] + r["h"], r["x"]:r["x"] + r["w"]].copy()


def bust_crop(rgba: np.ndarray) -> np.ndarray:
    """Head-and-shoulders square from a cut-out picture (face box, else the subject's top)."""
    from .teams import portrait_crop
    from .video import _crop_pad
    return _crop_pad(rgba, *portrait_crop(rgba, 1.0))


def _on_white(rgba: np.ndarray, size: int, smooth: bool = True) -> Image.Image:
    a = rgba[..., 3:4].astype(np.float32) / 255
    rgb = rgba[..., :3].astype(np.float32) * a + 255 * (1 - a)
    img = Image.fromarray(np.rint(rgb).astype(np.uint8), "RGB")
    if not smooth:     # pixel art: whole-number nearest first, then a gentle resize
        k = max(1, size // max(img.size))
        img = img.resize((img.width * k, img.height * k), Image.NEAREST)
    img.thumbnail((size, size), Image.LANCZOS)
    if max(img.size) < size:
        k = size / max(img.size)
        img = img.resize((round(img.width * k), round(img.height * k)), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


def init_bust(ch, settings=None) -> tuple[Image.Image, float, str]:
    """The picture the painting starts from: (RGB GEN_SIZE square on white, denoise, source)."""
    from .pipeline import isolate, load_image
    from .settings import settings_for
    s = settings or settings_for("person")
    if ch.picture.exists():
        rgba = isolate(load_image(ch.picture), s)
        return _on_white(bust_crop(rgba), GEN_SIZE), NEUTRAL_DENOISE, "picture"
    sprite = _front_sprite(ch)
    if sprite is None:
        raise ValueError(f"'{ch.name}' has no picture.png or sheet to make a portrait from")
    return _on_white(bust_crop(sprite), GEN_SIZE, smooth=False), SPRITE_DENOISE, "sprite"


def _white_key(rgb: np.ndarray, tol: float = 14.0) -> np.ndarray:
    """True = surely subject: not near-white (Lab distance to white >= tol). White parts
    (shirts, gaps between curls) are left to the cut-out model."""
    from .color import rgb_to_lab
    h, w = rgb.shape[:2]
    lab = rgb_to_lab(rgb.reshape(-1, 3)).reshape(h, w, 3)
    return np.linalg.norm(lab - np.array([100.0, 0.0, 0.0]), axis=-1) >= tol


def _cut_out(img: Image.Image) -> np.ndarray:
    """Generated bust on white -> RGBA with the background removed."""
    from .pipeline import isolate, load_image
    from .settings import settings_for
    import cv2
    rgba = load_image(img)
    rgba[..., 3] = 255
    out = isolate(rgba, settings_for("person"))
    # the paintings are on plain white: anything clearly not white is the bust, whatever the
    # model thinks (it made a knight's soft red hood half see-through)
    key = _white_key(rgba[..., :3])
    inner = cv2.erode(key.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0   # keep soft edges
    out[..., 3] = np.where(inner, 255, out[..., 3])
    # keep the bust; drop the specks the cut-out leaves around hair and edges
    solid = (out[..., 3] >= 128).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
    if n > 2:
        biggest = stats[1:, cv2.CC_STAT_AREA].max()
        small = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] < 0.02 * biggest]
        out[np.isin(lab, small), 3] = 0
    return out


# --------------------------------------------------------------------------- face mask

def face_landmarks(rgb: np.ndarray, threshold: float = 0.5) -> dict | None:
    """YuNet on an RGB/RGBA picture: {"box": (x, y, w, h), "eyes": [(x, y), (x, y)],
    "nose": (x, y), "mouth": [(x, y), (x, y)], "score"} of the most confident face, else None."""
    import cv2
    from .teams import _face_model
    model = _face_model()
    if model is None:
        return None
    h, w = rgb.shape[:2]
    k = min(1.0, 640 / max(h, w))
    small = cv2.resize(np.ascontiguousarray(rgb[..., 2::-1]),
                       (max(1, round(w * k)), max(1, round(h * k))))
    try:
        det = cv2.FaceDetectorYN.create(str(model), "", (small.shape[1], small.shape[0]),
                                        threshold)
        _, faces = det.detect(small)
    except (cv2.error, AttributeError):
        return None
    if faces is None or not len(faces):
        return None
    f = faces[np.argmax(faces[:, -1])] / np.r_[np.full(14, k), 1.0]
    pts = f[4:14].reshape(5, 2)
    return {"box": tuple(float(v) for v in f[:4]), "eyes": [tuple(pts[0]), tuple(pts[1])],
            "nose": tuple(pts[2]), "mouth": [tuple(pts[3]), tuple(pts[4])],
            "score": float(f[14])}


def _subject_top_face(rgb: np.ndarray) -> tuple[float, float, float, float, float]:
    """No face found (helmets, masks, odd paintings): guess it from the bust's own top on
    white. -> (cx, cy, rx, ry, angle)."""
    h, w = rgb.shape[:2]
    solid = (rgb[..., :3].astype(np.int16) < 235).any(-1)
    ys = np.nonzero(solid.any(1))[0]
    top = int(ys.min()) if len(ys) else int(0.1 * h)
    row = np.nonzero(solid[min(h - 1, top + int(0.18 * h))])[0]
    cx = float(np.median(row)) if len(row) else w / 2
    return cx, top + 0.3 * h, 0.15 * w, 0.17 * h, 0.0


def face_ellipse(rgb: np.ndarray, face="detect"):
    """(cx, cy, rx, ry, angle in radians) of the face region to repaint: brows to chin,
    cheek to cheek, turned with the eye line. From YuNet landmarks, else the face box, else
    the bust's top."""
    if isinstance(face, str):
        face = face_landmarks(rgb)
    if face is None:
        return _subject_top_face(rgb)
    (ax, ay), (bx, by) = face["eyes"]
    (mx1, my1), (mx2, my2) = face["mouth"]
    eye = np.array([(ax + bx) / 2, (ay + by) / 2])
    mouth = np.array([(mx1 + mx2) / 2, (my1 + my2) / 2])
    d = float(np.hypot(bx - ax, by - ay))
    fx, fy, fw, fh = face["box"]
    down = mouth - eye
    length = float(np.hypot(*down))
    if d < 0.15 * fw or length < 0.15 * fh:     # landmarks unusable: the box alone
        return fx + fw / 2, fy + 0.52 * fh, 0.46 * fw, 0.52 * fh, 0.0
    down = down / length
    angle = float(np.arctan2(-down[0], down[1]))        # 0 = upright
    top = -0.62 * d                                     # brows (along the face's own axis)
    bottom = length + 0.62 * d                          # chin
    c = eye + down * (top + bottom) / 2
    return float(c[0]), float(c[1]), 0.92 * d, (bottom - top) / 2, angle  # 1.0 d met the hair


def face_mask(rgb: np.ndarray, face="detect", *, core: float = 0.78,
              edge: float = 1.12) -> np.ndarray:
    """Soft float32 (H, W) mask over the face (1 inside, smooth falloff from `core` to `edge`
    of the face ellipse, 0 outside): what an expression may repaint."""
    h, w = rgb.shape[:2]
    cx, cy, rx, ry, ang = face_ellipse(rgb, face)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    ca, sa = np.cos(ang), np.sin(ang)
    u, v = dx * ca + dy * sa, -dx * sa + dy * ca
    r = np.sqrt((u / max(rx, 1)) ** 2 + (v / max(ry, 1)) ** 2)
    t = np.clip((edge - r) / (edge - core), 0, 1)
    return (t * t * (3 - 2 * t)).astype(np.float32)


def match_colours(painted: Image.Image, reference: Image.Image, mask: np.ndarray) -> Image.Image:
    """painted's colours inside the mask moved to reference's (a/b mean + spread in Lab,
    weighted by the mask): a repainted face keeps the neutral's skin tone and palette."""
    from .color import lab_to_rgb, rgb_to_lab
    w = mask.astype(np.float64).reshape(-1)
    if w.sum() < 1:
        return painted
    p = rgb_to_lab(np.asarray(painted.convert("RGB"), np.float32).reshape(-1, 3)).astype(np.float64)
    r = rgb_to_lab(np.asarray(reference.convert("RGB"), np.float32).reshape(-1, 3)).astype(np.float64)

    def stats(x):
        mu = (x * w[:, None]).sum(0) / w.sum()
        sd = np.sqrt(((x - mu) ** 2 * w[:, None]).sum(0) / w.sum()) + 1e-6
        return mu, sd

    pm, ps = stats(p)
    rm, rs = stats(r)
    out = p.copy()           # colour only (a, b): lightness changes are the expression (teeth)
    out[:, 1:] = (p[:, 1:] - pm[1:]) * np.clip(rs[1:] / ps[1:], 0.7, 1.4) + rm[1:]
    rgb = np.clip(lab_to_rgb(out.astype(np.float32)), 0, 255).reshape(painted.height, painted.width, 3)
    return Image.fromarray(np.rint(rgb).astype(np.uint8), "RGB")


_WOMAN = ("woman", "girl", "female", "lady", "queen", "princess", "witch", "priestess",
          "sorceress", "maiden", "heroine", "empress", "duchess", "nun", "mother")
_MAN = ("man", "boy", "male", "king", "prince", "lord", "monk", "priest", "father", "wizard",
        "sorcerer", "emperor", "duke")


def person_words(who: str) -> str:
    """'a woman' / 'a man' / 'a person' from the character's words (swordswoman -> woman):
    the identity a face-only repaint must keep, without costume words."""
    import re
    words = re.findall(r"[a-z]+", who.lower())
    if any(w.endswith(_WOMAN) for w in words):
        old = any(w in ("old", "elderly", "aged") for w in words)
        return "an old woman" if old else ("a girl" if "girl" in words else "a young woman"
                                          if "young" in words else "a woman")
    if any(w.endswith(_MAN) and not w.endswith(_WOMAN) for w in words):
        old = any(w in ("old", "elderly", "aged") for w in words)
        return "an old man" if old else ("a boy" if "boy" in words else "a young man"
                                        if "young" in words else "a man")
    return "a person"


def eye_discs(face: dict, size, offset=(0, 0)) -> list[np.ndarray]:
    """Soft round masks (one per eye, in a crop of `size` at `offset`) around the irises."""
    (ax, ay), (bx, by) = face["eyes"]
    d = float(np.hypot(bx - ax, by - ay))
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    out = []
    for x, y in face["eyes"]:
        cx, cy = x - offset[0], y - offset[1] + 0.06 * d     # the points sit on the upper lid
        r = np.hypot(xx - cx, yy - cy) / (0.26 * d)
        out.append(np.clip(1.5 - r, 0, 1).astype(np.float32) ** 1.5)
    return out


def face_crop_box(mask: np.ndarray, margin: float = 0.15) -> tuple[int, int, int, int]:
    """Square (x0, y0, x1, y1) around the mask's non-zero part plus `margin` each side, kept
    inside the picture: the face is repainted from this crop enlarged to GEN_SIZE (an SDXL
    face that fills the frame takes an expression far better than a small one)."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask > 0)
    if not len(xs):
        return 0, 0, w, h
    side = max(xs.max() - xs.min(), ys.max() - ys.min()) + 1
    side = min(int(round(side * (1 + 2 * margin))), w, h)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    x0 = int(np.clip(round(cx - side / 2), 0, w - side))
    y0 = int(np.clip(round(cy - side / 2), 0, h - side))
    return x0, y0, x0 + side, y0 + side


# --------------------------------------------------------------------------- frame

FRAME_COLOURS = ["1a1420", "3a2414", "7a4a1c", "c08a30", "f0d070"]   # outline, dark -> light


def frame_image(margin: int = 8, colours=None, palette: np.ndarray | None = None) -> np.ndarray:
    """A 9-slice frame texture (3*margin square, transparent middle): dark outline, bevelled
    metal band (light top-left, dark bottom-right), dark inner line, studs in the corners."""
    from .effects import snap_ramp
    from .palettes import hex_to_rgb
    m = max(4, int(margin))
    cols = hex_to_rgb([c.lstrip("#") for c in (colours or FRAME_COLOURS)])
    if palette is not None:
        snapped = [snap_ramp(c[None], palette)[0] for c in cols]
        cols = np.array(snapped, np.uint8)
    outline, dark, mid, light, shine = cols[0], cols[1], cols[2], cols[3], cols[-1]
    n = 3 * m
    out = np.zeros((n, n, 4), np.uint8)
    yy, xx = np.mgrid[0:n, 0:n]
    ring = np.minimum(np.minimum(xx, yy), np.minimum(n - 1 - xx, n - 1 - yy))  # 0 at the edge
    band = max(2, m // 2)

    def put(mask, c):
        out[mask, :3] = c
        out[mask, 3] = 255

    put(ring == 0, outline)
    metal = (ring >= 1) & (ring <= band)
    put(metal, mid)
    put(metal & (ring == 1) & ((yy == 1) | (xx == 1)), light)          # lit top + left
    put(metal & (ring == 1) & ((yy == n - 2) | (xx == n - 2)), dark)   # shaded bottom + right
    put(metal & (ring == band) & ((yy == band) | (xx == band)), dark)  # bevel: inner edge
    put(metal & (ring == band) & ((yy == n - 1 - band) | (xx == n - 1 - band)), light)
    put(ring == band + 1, outline)
    for cx in (m // 2, n - 1 - m // 2):                       # studs
        for cy in (m // 2, n - 1 - m // 2):
            d = np.abs(xx - cx) + np.abs(yy - cy)
            put(d <= 1, light)
            put((xx == cx) & (yy == cy), shine)
    return out


def nine_slice(frame: np.ndarray, w: int, h: int, margin: int) -> np.ndarray:
    """Stretch a 9-slice texture to w x h (edges repeat their middle row/column)."""
    m = margin
    n = frame.shape[0]
    xs = np.concatenate([np.arange(m), np.full(max(0, w - 2 * m), m + (n - 2 * m) // 2),
                         np.arange(n - m, n)])[:w]
    ys = np.concatenate([np.arange(m), np.full(max(0, h - 2 * m), m + (n - 2 * m) // 2),
                         np.arange(n - m, n)])[:h]
    return frame[ys][:, xs]


def portrait_frame(out_dir, margin: int | None = None, *, colours=None, project=None,
                   style=None, name: str = "frame") -> list[Path]:
    """frame.png (9-slice texture in the project palette) + frame.json (margins) +
    frame.tres (Godot StyleBoxTexture with the margins; also usable by NinePatchRect)."""
    from .effects import project_palette
    if style is None:
        from .assets import current_project
        style = (project or current_project()).style
    m = margin or max(4, round(8 * style.person_height / 64))
    img = frame_image(m, colours, project_palette(style))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    Image.fromarray(img, "RGBA").save(png)
    js = out_dir / f"{name}.json"
    js.write_text(json.dumps({"image": png.name, "size": img.shape[0], "margin": m,
                              "patch_margin": {"left": m, "top": m, "right": m, "bottom": m}},
                             indent=1), encoding="utf-8")
    tres = out_dir / f"{name}.tres"
    tres.write_text("\n".join([
        '[gd_resource type="StyleBoxTexture" load_steps=2 format=3]', "",
        f'[ext_resource type="Texture2D" path="{png.name}" id="1_frame"]', "",
        "[resource]", 'texture = ExtResource("1_frame")',
        *[f"texture_margin_{side} = {float(m)}" for side in ("left", "top", "right", "bottom")],
        "axis_stretch_horizontal = 1", "axis_stretch_vertical = 1", ""]), encoding="utf-8")
    return [png, js, tres]


def framed(bust: np.ndarray, frame: np.ndarray, margin: int, background=(34, 30, 44)) -> np.ndarray:
    """A bust inside the frame on a flat background (for previews / quick UI use)."""
    h, w = bust.shape[:2]
    pad = max(2, margin // 2) + 2
    W, H = w + 2 * pad, h + 2 * pad
    out = np.zeros((H, W, 4), np.uint8)
    out[..., :3] = background
    out[..., 3] = 255
    a = bust[..., 3:4].astype(np.float32) / 255
    region = out[pad:pad + h, pad:pad + w]
    region[..., :3] = np.rint(bust[..., :3] * a + region[..., :3] * (1 - a)).astype(np.uint8)
    f = nine_slice(frame, W, H, margin)
    on = f[..., 3] > 0
    out[on] = f[on]
    return out


# --------------------------------------------------------------------------- portraits

def _row(images: list[np.ndarray], gap: int = 0) -> tuple[np.ndarray, list]:
    h = max(i.shape[0] for i in images)
    w = max(i.shape[1] for i in images)
    out = np.zeros((h, len(images) * w + gap * (len(images) - 1), 4), np.uint8)
    rects = []
    for k, im in enumerate(images):
        x = k * (w + gap)
        out[:im.shape[0], x:x + im.shape[1]] = im
        rects.append((x, 0, w, h))
    return out, rects


def _write_sheet(images, names, out_png: Path, extra: dict) -> list[Path]:
    from .export import sheet_json
    sheet, rects = _row(images)
    Image.fromarray(sheet, "RGBA").save(out_png)
    meta = sheet_json(rects, out_png.name, [0] * len(images), None, list(names))
    meta["meta"].update(extra)
    js = out_png.with_suffix(".json")
    js.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return [out_png, js]


def _thumb(rgba: np.ndarray, out: Path, px: int = 96, pixel: bool = True) -> Path:
    im = Image.fromarray(rgba, "RGBA")
    box = im.getbbox()
    if box:
        im = im.crop(box)
    if pixel and max(im.size) <= px:
        k = max(1, px // max(im.size))
        im = im.resize((im.width * k, im.height * k), Image.NEAREST)
    else:
        im.thumbnail((px, px), Image.LANCZOS)
    canvas = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    canvas.alpha_composite(im, ((px - im.width) // 2, px - im.height))
    canvas.save(out)
    return out


def pixel_bust(painted: np.ndarray, size: int, style, palette: np.ndarray | None = None):
    """Painted bust (RGBA) -> pixel-art Sprite, size x size, in the project's palette/colours."""
    from .pipeline import finish, prepare
    from .settings import settings_for
    s = settings_for("person", width=size, height=size, colors=style.colours,
                     palette=style.palette, fit="contain", anchor="bottom",
                     crop_to_subject=True, remove_background=False, margin=0.0,
                     outline=style.outline != "none",
                     outline_style=style.outline if style.outline in ("selout", "solid") else "selout")
    return finish(prepare(painted, s), s, palette)


def _clean_bust(sprite, style):
    """Specks and 1-2 px colour islands off a pixel bust (pixelclean, as for sheets)."""
    from .pixelclean import clean_sprite
    outline = style.outline if style.outline in ("selout", "solid", "none") else "selout"
    return clean_sprite(sprite, outline, speck=6, islands=2)


def _clear_portrait(replace, info: dict):
    """An existing portrait asset emptied of its old images, with new info."""
    import shutil
    from .assets import Asset
    folder = Path(getattr(replace, "folder", replace))
    meta = folder / "asset.json"
    if not meta.exists():
        raise ValueError(f"No portrait asset in {folder}")
    d = json.loads(meta.read_text(encoding="utf-8"))
    if d.get("kind") != "portrait":
        raise ValueError(f"{folder} is a {d.get('kind')} asset, not a portrait")
    for p in folder.iterdir():
        if p.name != "asset.json":
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    a = Asset("portrait", d.get("name", folder.name), folder, tags=d.get("tags", []),
              collections=d.get("collections", []), created=d.get("created", 0.0),
              extra={**d.get("extra", {}), **info})
    a.save()
    return a


def make_portraits(character, expressions=EXPRESSIONS, style: str = "painted",
                   size: int | None = None, *, client=None, progress=None, project=None,
                   painted_size: int = 512, prompt: str | None = None, seed: int | None = None,
                   face_only: bool = True, expression_denoise: float | None = None,
                   replace=None, **gen_opts):
    """Portraits of a library character (Character, folder or name) in each expression.
    style: "painted" or "pixel" (which one the sheet/thumbnail show; both are always saved).
    face_only: expressions inpaint just the face of the neutral painting (default); False =
    the whole picture img2img'd at WHOLE_DENOISE, which is also what happens when YuNet finds
    no face on the neutral painting (asset.extra face_found False: helmets, visors, masks).
    expression_denoise overrides the denoise.
    replace: an existing portrait Asset (or its folder) to write over instead of a new one.
    size: pixel bust size (default 96 px for a 64 px person, scaled with the project).
    client: a generate.ComfyClient (default: the local ComfyUI). gen_opts go to
    generate_images (lora is always "none": the painted look). Returns the Asset."""
    from .assets import current_project, new_asset, set_tags
    from .generate import composite_masked, generate_images
    if style not in STYLES:
        raise ValueError(f"Unknown portrait style '{style}' (use {', '.join(STYLES)})")
    expressions = [e.strip() for e in expressions if e and e.strip()]
    if not expressions:
        raise ValueError("Pick at least one expression")
    if "neutral" in expressions:
        expressions.remove("neutral")
    expressions = ["neutral", *expressions]
    say = progress or (lambda t: None)
    ch = _character(character)
    proj = project or current_project()
    st = proj.style
    px = int(size or st.scale(96))
    who = prompt or character_prompt(ch)
    seed = int(seed if seed is not None else zlib.crc32(ch.folder.name.encode()) & 0x7FFFFFFF)
    gen_opts = {k: v for k, v in gen_opts.items() if k != "lora"}

    say("Cropping head and shoulders…")
    init, denoise, source = init_bust(ch)

    def paint(image, expr, dn, mask=None):
        words = EXPRESSION_WORDS.get(expr, f"({expr} expression:1.3)")
        negative = PAINTED_NEGATIVE
        if expr != "neutral":
            negative = ", ".join(x for x in (NEUTRAL_NEGATIVE, EXPRESSION_NEGATIVE.get(expr),
                                             PAINTED_NEGATIVE) if x)
        template = PAINTED_PROMPT if expr == "neutral" or mask is None else EXPRESSION_PROMPT
        text = template.format(who=who, expression=words, person=person)
        if expr == "neutral":     # house style on the bust; expressions repaint the face only
            from .assets import with_style
            text, negative = with_style(text, negative, st, "portrait")
        if mask is not None:          # the face alone, enlarged to the full generation size
            box = face_crop_box(mask)
            face = image.crop(box)
            m = mask[box[1]:box[3], box[0]:box[2]]
            big = GEN_SIZE      # resized here: prepare_init_image only enlarges by whole numbers
            big_m = np.asarray(Image.fromarray(m).resize((big, big), Image.BILINEAR), np.float32)
            (img, _), = generate_images(text, negative, client=client, seed=seed,
                                        init_image=face.resize((big, big), Image.LANCZOS),
                                        denoise=dn, aspect=1.0, lora="none", mask=big_m,
                                        **gen_opts)
            img = match_colours(img.convert("RGB").resize(face.size, Image.LANCZOS), face, m)
            if landmarks is not None:     # each eye keeps its colour (irises drifted blue)
                for disc in eye_discs(landmarks, face.size, box[:2]):
                    img = composite_masked(img, match_colours(img, face, disc), disc)
            out = image.copy()
            out.paste(composite_masked(face, img, m), box[:2])
            return out
        (img, _), = generate_images(text, negative, client=client, seed=seed, init_image=image,
                                    denoise=dn, aspect=1.0, lora="none", **gen_opts)
        return img.convert("RGB")

    raws, landmarks, person = {}, None, person_words(who)
    say(f"Painting {ch.name} (neutral) on your GPU…")
    raws["neutral"] = paint(init, "neutral", denoise)
    mask = landmarks = None
    face_found = None
    if face_only and len(expressions) > 1:
        landmarks = face_landmarks(np.asarray(raws["neutral"]))
        face_found = landmarks is not None
        if face_found:
            mask = face_mask(np.asarray(raws["neutral"]), landmarks)
        else:   # helmet, visor, hood, mask: inventing a face there gave random strangers
            face_only = False
            say(f"No face found on {ch.name} (helmet or mask?): expressions repaint the whole "
                "picture lightly instead, so they differ little")
    dn = expression_denoise or (EXPRESSION_DENOISE if face_only else WHOLE_DENOISE)
    for e in expressions[1:]:
        say(f"Painting {ch.name} ({e})…")
        edn = dn if expression_denoise or not face_only else EXPRESSION_DENOISE_BY.get(e, dn)
        raws[e] = paint(raws["neutral"], e, edn, mask)

    info = dict(character=ch.name, character_folder=str(ch.folder), expressions=expressions,
                style=style, size=px, painted_size=painted_size, seed=seed, prompt=who,
                source=source, face_only=bool(face_only), face_found=face_found,
                expression_denoise=dn)
    if replace is not None:
        asset = _clear_portrait(replace, info)
    else:
        asset = new_asset("portrait", f"{ch.name} portraits", proj, **info)
    folder = asset.folder
    if mask is not None:
        Image.fromarray(np.rint(mask * 255).astype(np.uint8), "L").save(folder / "face_mask.png")
    for e in expressions:
        raws[e].save(folder / f"raw_{e}.png")
    say("Writing the sheets…")
    _finish(asset, raws, st)
    say(f"Portraits of {ch.name}: {', '.join(expressions)}")
    return asset


def _finish(asset, raws: dict, st) -> None:
    """raw images -> cut-out painted busts, pixel busts (one palette), sheets, frame, preview,
    thumbnail; tags and the file list."""
    from .assets import set_tags
    from .effects import project_palette
    folder = asset.folder
    expressions = asset.extra["expressions"]
    style = asset.extra.get("style", "painted")
    painted_size = int(asset.extra.get("painted_size", 512))
    px = int(asset.extra.get("size") or st.scale(96))
    name = asset.extra.get("character", asset.name)
    painted, pixels = {}, {}
    palette = None
    for e in expressions:
        cut = _cut_out(raws[e].resize((painted_size, painted_size), Image.LANCZOS))
        painted[e] = cut
        Image.fromarray(cut, "RGBA").save(folder / f"portrait_{e}.png")
        solid = cut.copy()    # pixel art: only well-covered pixels (faded wisps became flecks)
        solid[..., 3] = np.where(cut[..., 3] >= PIXEL_ALPHA, 255, 0)
        sprite = _clean_bust(pixel_bust(solid, px, st, palette), st)
        palette = sprite.palette if palette is None else palette   # one palette for all
        pixels[e] = sprite.rgba
        sprite.save(folder / f"portrait_{e}_pixel.png")

    common = {"character": name, "expressions": expressions}
    _write_sheet([painted[e] for e in expressions], expressions, folder / "sheet_painted.png",
                 {**common, "style": "painted"})
    _write_sheet([pixels[e] for e in expressions], expressions, folder / "sheet_pixel.png",
                 {**common, "style": "pixel"})
    main = pixels if style == "pixel" else painted
    _write_sheet([main[e] for e in expressions], expressions, folder / "sheet.png",
                 {**common, "style": style})
    m = max(4, round(8 * st.person_height / 64))
    portrait_frame(folder, m, style=st)
    frame = frame_image(m, palette=project_palette(st))
    preview, _ = _row([framed(pixels[e], frame, m) for e in expressions], gap=4)
    Image.fromarray(preview, "RGBA").save(folder / "preview.png")
    _thumb(main["neutral"], folder / "thumb.png", pixel=style == "pixel")
    asset.extra["files"] = sorted(p.name for p in folder.iterdir())
    set_tags(asset, ["portrait", style, name])


def refinish_portraits(asset, project=None):
    """Re-make the cut-outs, pixel busts and sheets of a portrait asset from its raw_*.png
    (no GPU): after a cut-out or palette change, or a new project style."""
    from .assets import current_project
    st = (project or current_project()).style
    raws = {e: Image.open(asset.folder / f"raw_{e}.png").convert("RGB")
            for e in asset.extra["expressions"]}
    _finish(asset, raws, st)
    return asset


def load_portraits(asset, style: str | None = None) -> dict[str, np.ndarray]:
    """{expression: RGBA} of a portrait asset (its own style unless one is given)."""
    style = style or asset.extra.get("style", "painted")
    suffix = "_pixel" if style == "pixel" else ""
    out = {}
    for e in asset.extra.get("expressions", []):
        p = asset.folder / f"portrait_{e}{suffix}.png"
        if p.exists():
            out[e] = np.asarray(Image.open(p).convert("RGBA")).copy()
    return out

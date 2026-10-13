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
as a painting at a moderate denoise; every expression is then made FROM that neutral painting
at a lower denoise with the same seed (seed = hash of the character's folder), so the face,
hair and clothes stay put while the expression changes. Characters without a picture fall back
to their front idle sprite, enlarged (needs a higher denoise, so less faithful).
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

EXPRESSIONS = ("neutral", "happy", "angry", "sad", "surprised")
EXPRESSION_WORDS = {
    "neutral": "calm neutral expression, closed mouth",
    "happy": "happy expression, warm smile",
    "angry": "angry expression, furrowed brows, gritted teeth, scowl",
    "sad": "sad expression, downcast eyes, slight frown",
    "surprised": "surprised expression, wide eyes, open mouth, raised eyebrows",
    "determined": "determined expression, focused eyes, set jaw",
    "smug": "smug expression, smirk, half-lidded eyes",
    "hurt": "pained expression, wincing, one eye shut",
    "thinking": "thoughtful expression, eyes looking aside",
}
PAINTED_PROMPT = ("{who}, head and shoulders portrait, {expression}, painted fantasy game "
                  "portrait, anime-inspired painterly illustration, detailed face, clean "
                  "lineart, soft cel shading, looking at the viewer, centered bust, plain white "
                  "background")
PAINTED_NEGATIVE = ("photo, photorealistic, 3d render, pixel art, sprite, blurry, lowres, jpeg "
                    "artifacts, text, watermark, signature, frame, border, multiple people, "
                    "cropped head, out of frame, deformed face, extra eyes, bad anatomy, "
                    "busy background, scenery")
NEUTRAL_DENOISE = 0.5        # picture -> painting
EXPRESSION_DENOISE = 0.58    # neutral painting -> each expression (0.4 barely changed faces)
SPRITE_DENOISE = 0.65        # enlarged sprite -> painting (no picture.png)
STYLES = ("painted", "pixel")
GEN_SIZE = 1024


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


def _cut_out(img: Image.Image) -> np.ndarray:
    """Generated bust on white -> RGBA with the background removed."""
    from .pipeline import isolate, load_image
    from .settings import settings_for
    import cv2
    rgba = load_image(img)
    rgba[..., 3] = 255
    out = isolate(rgba, settings_for("person"))
    # keep the bust; drop the specks the cut-out leaves around hair and edges
    solid = (out[..., 3] >= 128).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
    if n > 2:
        biggest = stats[1:, cv2.CC_STAT_AREA].max()
        small = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] < 0.02 * biggest]
        out[np.isin(lab, small), 3] = 0
    return out


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


def make_portraits(character, expressions=EXPRESSIONS, style: str = "painted",
                   size: int | None = None, *, client=None, progress=None, project=None,
                   painted_size: int = 512, prompt: str | None = None, seed: int | None = None,
                   **gen_opts):
    """Portraits of a library character (Character, folder or name) in each expression.
    style: "painted" or "pixel" (which one the sheet/thumbnail show; both are always saved).
    size: pixel bust size (default 96 px for a 64 px person, scaled with the project).
    client: a generate.ComfyClient (default: the local ComfyUI). gen_opts go to
    generate_images (lora is always "none": the painted look). Returns the Asset."""
    from .assets import current_project, new_asset, set_tags
    from .generate import generate_images
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

    def paint(image, expr, dn):
        words = EXPRESSION_WORDS.get(expr, f"{expr} expression")
        (img, _), = generate_images(PAINTED_PROMPT.format(who=who, expression=words),
                                    PAINTED_NEGATIVE, client=client, seed=seed, init_image=image,
                                    denoise=dn, aspect=1.0, lora="none", **gen_opts)
        return img.convert("RGB")

    raws = {}
    say(f"Painting {ch.name} (neutral) on your GPU…")
    raws["neutral"] = paint(init, "neutral", denoise)
    for e in expressions[1:]:
        say(f"Painting {ch.name} ({e})…")
        raws[e] = paint(raws["neutral"], e, EXPRESSION_DENOISE)

    asset = new_asset("portrait", f"{ch.name} portraits", proj, character=ch.name,
                      character_folder=str(ch.folder), expressions=expressions, style=style,
                      size=px, painted_size=painted_size, seed=seed, prompt=who, source=source)
    folder = asset.folder
    painted, pixels = {}, {}
    palette = None
    for e in expressions:
        raws[e].save(folder / f"raw_{e}.png")
        cut = _cut_out(raws[e].resize((painted_size, painted_size), Image.LANCZOS))
        painted[e] = cut
        Image.fromarray(cut, "RGBA").save(folder / f"portrait_{e}.png")
        sprite = pixel_bust(cut, px, st, palette)
        palette = sprite.palette if palette is None else palette   # one palette for all
        pixels[e] = sprite.rgba
        sprite.save(folder / f"portrait_{e}_pixel.png")

    say("Writing the sheets…")
    common = {"character": ch.name, "expressions": expressions}
    _write_sheet([painted[e] for e in expressions], expressions, folder / "sheet_painted.png",
                 {**common, "style": "painted"})
    _write_sheet([pixels[e] for e in expressions], expressions, folder / "sheet_pixel.png",
                 {**common, "style": "pixel"})
    main = pixels if style == "pixel" else painted
    _write_sheet([main[e] for e in expressions], expressions, folder / "sheet.png",
                 {**common, "style": style})
    from .effects import project_palette
    m = max(4, round(8 * st.person_height / 64))
    portrait_frame(folder, m, style=st)
    frame = frame_image(m, palette=project_palette(st))
    preview, _ = _row([framed(pixels[e], frame, m) for e in expressions], gap=4)
    Image.fromarray(preview, "RGBA").save(folder / "preview.png")
    _thumb(main["neutral"], folder / "thumb.png", pixel=style == "pixel")
    asset.extra["files"] = sorted(p.name for p in folder.iterdir())
    set_tags(asset, ["portrait", style, ch.name])
    say(f"Portraits of {ch.name}: {', '.join(expressions)}")
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

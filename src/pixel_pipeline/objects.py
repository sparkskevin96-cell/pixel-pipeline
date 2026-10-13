"""Items and props: the things in a game world that aren't characters, kept in the project's
asset store (assets.py) so every game made with the project can reuse them.

ITEMS (kind "item": swords, potions, keys…), from words (your GPU), a photo or a 3D model:
    icon.png         the inventory icon: Style.icon px square, transparent, outlined and
                     coloured the project's way (+ icon_2x.png / icon_4x.png crisp previews)
    icon_v1..3.png   every variation (from words: one per seed); icon.png is the chosen one
    picture_v*.png   what each icon was converted from (to redo them in a new style, no GPU)
    sheet.*          optional world sprite: the item lying in the world, from every direction
                     (or 1 for flat items), with normal map, .aseprite and Godot SpriteFrames
    model.glb        the coloured 3D model the world sprite was rendered from (if any)
PROPS (kind "prop": trees, barrels, carts, statues, houses, walls, fences):
    model.glb        the coloured 3D model (use it directly in 3D / HD-2D scenes)
    sheet.*          rendered at the project's scale next to a person (height_people), at
                     Style.elevation, 1/4/8/16 directions, normal map, .aseprite, Godot .tres
    thumb.png        a small picture for lists

Words -> SDXL: icons with the Pixel Art XL LoRA (an icon-shaped prompt), models with a clean
picture and no LoRA -> Hunyuan3D (model3d.prompt_to_model, subject "object"). Every picture
goes through the converter (pipeline.py) with the project's Style, so it lands on its grid and
palette. Changing the size, height or directions later re-renders from model.glb: no GPU.
"""
from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from .assets import Asset, Project, Style, current_project, new_asset

CATEGORIES = ("weapon", "armour", "potion", "food", "key", "misc")
# an item's longest side in the world, in pixels for the default 64 px person (Style.scale)
ITEM_SIZES = {"tiny": 12, "small": 16, "medium": 24, "large": 36}
DIRECTION_CHOICES = (1, 4, 8, 16)
ICON_COLOURS = 16              # at most this many colours in an icon (32 px can't show more)
VARIATIONS = 3

CATEGORY_HINTS = {
    "weapon": "single weapon standing upright, pointing straight up, vertical",
    "armour": "single piece of armour, front view",
    "potion": "single glass potion bottle, front view",
    "food": "single food item",
    "key": "single key, upright",
    "misc": "single object",
}
ICON_PROMPT = ("pixel art game item icon of {prompt}, {hint}, centered, whole object in frame, "
               "plain white background, flat colors, bold dark outline, simple readable shapes, "
               "inventory icon, game asset")
ICON_NEGATIVE = ("3d render, realistic, photo, blurry, smooth gradient, noise, jpeg artifacts, "
                 "text, watermark, signature, logo, multiple objects, sprite sheet, hand, person, "
                 "character, table, floor, scenery, cast shadow, frame, border, user interface, "
                 "cropped, cut off, out of frame")
SMALL_ICON_HINT = "very simple, low detail, chunky pixels, bold silhouette"
# the clean picture a prop's 3D model is built from (no pixel LoRA). SDXL likes product
# sheets (five barrels, a lid, a stand), which image-to-3D fuses into one lump: insist on one.
PROP_PROMPT = ("a single {prompt} standing on its own, one object only, isolated, whole object in "
               "frame, three-quarter front view, centered, plain white background, soft even "
               "lighting, no shadows, clean shapes, game prop concept art")
PROP_NEGATIVE = ("multiple objects, two, pair, group, set, collection, variations, several views, "
                 "product sheet, character sheet, stack, pile, cropped, cut off, out of frame, "
                 "partial, crowd, text, watermark, busy background, scenery, landscape, floor, "
                 "dramatic lighting, harsh shadows, motion blur, blurry, pixel art, sprite sheet")
NO_DIAGONAL = ("potion", "armour")   # long thin items lie diagonally in the icon, except these

# (words, height in people, directions): first match on the name/prompt wins, so buildings come
# before "stone". Buildings and things with a front face 4 directions; round things 1.
PROP_PRESETS = (
    (("house", "cottage", "hut", "cabin", "shop", "inn", "tavern", "barn", "church", "chapel",
      "temple", "tower", "castle", "mill", "forge", "smithy", "building", "home", "shed"), 3.0, 4),
    (("tree", "oak", "pine", "palm", "willow", "birch"), 2.5, 1),
    (("tent",), 1.6, 4),
    (("statue", "monument", "shrine"), 1.5, 4),
    (("wall",), 1.5, 4),
    (("lamp", "lantern", "torch", "signpost", "sign", "post"), 1.4, 1),
    (("cart", "wagon", "carriage", "boat"), 1.0, 4),
    (("well", "fountain"), 1.0, 1),
    (("fence", "gate"), 0.6, 4),
    (("barrel", "keg"), 0.6, 1),
    (("bush", "shrub", "hedge"), 0.6, 1),
    (("crate", "box"), 0.5, 1),
    (("table", "bench", "chair", "stool", "bed"), 0.5, 4),
    (("rock", "boulder", "stone", "pot", "vase", "urn"), 0.5, 1),
    (("chest", "anvil"), 0.4, 4),
)
SHEET_FILES = (".png", ".json", "_n.png", ".aseprite", ".tres", "_facing.gd")


def _say(progress):
    return progress or (lambda text: None)


def guess_prop(text: str) -> tuple[float, int]:
    """(height in people, directions) for a prop described by `text` (default 1.0, 1)."""
    words = set(re.findall(r"[a-z]+", (text or "").lower()))
    words |= {w[:-1] for w in words if w.endswith("s")}
    for names, height, dirs in PROP_PRESETS:
        if words & set(names):
            return height, dirs
    return 1.0, 1


def _directions(n: int | None, default: int) -> int:
    n = int(n or default)
    if n not in DIRECTION_CHOICES:
        raise ValueError(f"directions must be one of {DIRECTION_CHOICES}, not {n}")
    return n


def open_asset(folder) -> Asset:
    """An item/prop back from its folder (asset.json)."""
    folder = Path(folder)
    d = json.loads((folder / "asset.json").read_text(encoding="utf-8"))
    return Asset(d["kind"], d.get("name", folder.name), folder, d.get("tags", []),
                 d.get("collections", []), d.get("created", 0.0), d.get("extra", {}))


def _project(project: Project | None) -> Project:
    return project or current_project()


# --------------------------------------------------------------------------- settings

def style_settings(st: Style, width: int, height: int, colours: int | None = None,
                   anchor: str = "center"):
    """Converter settings that follow the project's look (outline, palette, colours)."""
    from .settings import settings_for
    s = settings_for("object", width=width, height=height, colors=colours or st.colours,
                     anchor=anchor)
    if st.outline == "none":
        s.outline = False
    elif st.outline in ("selout", "solid"):
        s.outline_style = st.outline
    if st.palette and st.palette != "auto":
        s.palette = st.palette
    return s


def icon_settings(st: Style):
    return style_settings(st, st.icon, st.icon, min(st.colours, ICON_COLOURS), "center")


def icon_prompt(prompt: str, category: str = "misc", icon_px: int = 32) -> tuple[str, str]:
    text = re.sub(r"^\s*(an?\s+)?pixel[- ]?art\s+(icon\s+|sprite\s+)?(of\s+)?", "", prompt or "",
                  flags=re.IGNORECASE).strip().rstrip(".,; ")
    if not text:
        raise ValueError("Empty prompt: describe the item, e.g. 'an iron sword'.")
    positive = ICON_PROMPT.format(prompt=text, hint=CATEGORY_HINTS.get(category, "single object"))
    if icon_px <= 24:
        positive += ", " + SMALL_ICON_HINT
    return positive, ICON_NEGATIVE


# --------------------------------------------------------------------------- small pictures

def _crisp(rgba: np.ndarray, k: int) -> Image.Image:
    im = Image.fromarray(rgba, "RGBA")
    return im.resize((im.width * k, im.height * k), Image.NEAREST)


def make_thumb(rgba: np.ndarray, out: Path, px: int = 96) -> Path:
    """The sprite cropped, enlarged crisp by a whole number to fit px, centred on the floor."""
    im = Image.fromarray(rgba, "RGBA")
    box = im.getbbox()
    if box:
        im = im.crop(box)
    k = (px - 6) / max(im.size)
    k = max(1, int(k)) if k >= 1 else k
    im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.NEAREST)
    canvas = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    canvas.alpha_composite(im, ((px - im.width) // 2, max(0, (px - im.height) // 2)))
    canvas.save(out)
    return out


def icon_paths(asset: Asset) -> list[Path]:
    """Every icon variation, in order (icon_v1.png, icon_v2.png…)."""
    found = sorted(asset.folder.glob("icon_v*.png"),
                   key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))
    return found


def _use_icon(asset: Asset, index: int) -> Path:
    src = asset.folder / f"icon_v{index}.png"
    if not src.exists():
        raise ValueError(f"{asset.name} has no icon variation {index}")
    rgba = np.asarray(Image.open(src).convert("RGBA"))
    Image.fromarray(rgba, "RGBA").save(asset.folder / "icon.png")
    for k in (2, 4):
        _crisp(rgba, k).save(asset.folder / f"icon_{k}x.png")
    make_thumb(rgba, asset.thumb)
    asset.extra["icon_choice"] = index
    return asset.folder / "icon.png"


# --------------------------------------------------------------------------- icons

def main_subject(rgba: np.ndarray, reach: float = 0.02) -> np.ndarray:
    """Only the main thing in a cut-out picture: the biggest piece and the pieces touching its
    box (a fence's posts, a tree's loose leaves); other things it drew beside it (more barrels,
    sparkles, a stand off to the side) become transparent."""
    import cv2
    solid = (rgba[..., 3] >= 128).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
    if n <= 2:
        return rgba
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = stats[big, :4]
    m = reach * max(w, h)
    x0, y0, x1, y1 = x - m, y - m, x + w + m, y + h + m
    keep = np.zeros(n, bool)
    for i in range(1, n):
        bx, by, bw, bh = stats[i, :4]
        keep[i] = bx < x1 and bx + bw > x0 and by < y1 and by + bh > y0
    if keep[1:].all():
        return rgba
    # soft edge pixels (alpha < 128) go with the nearest solid piece
    dist, near = cv2.distanceTransformWithLabels(1 - solid, cv2.DIST_L2, 3,
                                                 labelType=cv2.DIST_LABEL_PIXEL)
    sy, sx = np.nonzero(solid)
    lut = np.zeros(int(near.max()) + 1, np.int32)
    lut[near[sy, sx]] = labels[sy, sx]
    owner = np.where(solid > 0, labels, lut[near])
    out = rgba.copy()
    out[..., 3] = np.where(keep[owner], rgba[..., 3], 0)
    return out


def elongation(alpha: np.ndarray) -> tuple[float, float]:
    """(long / short axis ratio, angle of the long axis in degrees, 0 = right, 90 = up)."""
    ys, xs = np.nonzero(alpha >= 128)
    if len(xs) < 3:
        return 1.0, 0.0
    pts = np.stack([xs - xs.mean(), -(ys - ys.mean())], 1).astype(np.float64)
    vals, vecs = np.linalg.eigh(np.cov(pts.T))
    ratio = math.sqrt(max(vals[1], 1e-9) / max(vals[0], 1e-9))
    vx, vy = vecs[:, 1]
    return ratio, math.degrees(math.atan2(vy, vx)) % 180.0


def head_is_forward(alpha: np.ndarray, angle: float, bins: int = 48) -> bool:
    """Along the long axis (pointing at `angle`): is the business end (sword blade, axe or
    hammer head, spear point, key bow) forward? The widest cross-section (a crossguard, an
    axe head) splits the item; the side that is wider on average is the head, the narrow
    one the grip or shaft."""
    ys, xs = np.nonzero(alpha >= 128)
    a = math.radians(angle)
    px, py = xs - xs.mean(), -(ys - ys.mean())
    t = px * math.cos(a) + py * math.sin(a)
    u = -px * math.sin(a) + py * math.cos(a)
    k = np.clip(((t - t.min()) / max(np.ptp(t), 1e-9) * bins).astype(int), 0, bins - 1)
    lo = np.full(bins, np.inf)
    hi = np.full(bins, -np.inf)
    np.minimum.at(lo, k, u)
    np.maximum.at(hi, k, u)
    width = np.where(np.isfinite(lo), hi - lo, 0.0)
    m = int(np.argmax(width))
    back, front = width[:max(0, m - 2)], width[m + 3:]
    if not len(back) or not len(front):
        return True                    # widest at an end: nothing to compare, keep as drawn
    return float(front.mean()) >= float(back.mean())


def diagonal(rgba: np.ndarray, min_ratio: float = 2.2) -> np.ndarray:
    """A long thin subject (sword, staff, key) turned to lie from bottom-left to top-right,
    as RPG inventory icons do, so it fills the square, with its head (blade, axe head…) at
    the top right (head_is_forward). Anything else is returned as it is."""
    import cv2
    ratio, angle = elongation(rgba[..., 3])
    if ratio < min_ratio:
        return rgba
    if not head_is_forward(rgba[..., 3], angle):
        angle += 180                                       # the head goes to the top right
    turn = (45.0 - angle + 180) % 360 - 180                # counter-clockwise, degrees
    if abs(turn) < 12:
        return rgba
    ys, xs = np.nonzero(rgba[..., 3] >= 8)
    crop = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape[:2]
    d = int(math.hypot(h, w)) + 4
    pad = np.zeros((d, d, 4), np.uint8)
    pad[(d - h) // 2:(d - h) // 2 + h, (d - w) // 2:(d - w) // 2 + w] = crop
    a = pad[..., 3:4].astype(np.float32) / 255           # premultiplied: no dark fringes
    pre = np.concatenate([pad[..., :3] * a, a * 255], -1)
    m = cv2.getRotationMatrix2D((d / 2, d / 2), turn, 1.0)
    rot = cv2.warpAffine(pre, m, (d, d), flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0, 0))
    alpha = np.clip(rot[..., 3:4], 0, 255)
    rgb = rot[..., :3] / np.maximum(alpha / 255, 1e-6)
    return np.clip(np.concatenate([rgb, alpha], -1), 0, 255).astype(np.uint8)


def redo_icons(asset: Asset, project: Project | None = None) -> list[Path]:
    """Convert every picture_v*.png into an icon with the project's current style (no GPU),
    keeping the chosen variation."""
    from .pipeline import convert, isolate, load_image
    st = _project(project).style
    s = icon_settings(st)
    turn = asset.extra.get("category", "misc") not in NO_DIAGONAL
    pictures = sorted(asset.folder.glob("picture_v*.png"),
                      key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))
    if not pictures:
        raise ValueError(f"{asset.name} has no pictures to make icons from")
    for old in icon_paths(asset):
        old.unlink()
    out = []
    for i, pic in enumerate(pictures, 1):
        rgba = main_subject(isolate(load_image(pic), s))
        if turn:
            rgba = diagonal(rgba)
        sprite = convert(rgba, s)
        path = asset.folder / f"icon_v{i}.png"
        Image.fromarray(sprite.rgba, "RGBA").save(path)
        out.append(path)
    choice = int(asset.extra.get("icon_choice", 1))
    _use_icon(asset, choice if 1 <= choice <= len(out) else 1)
    asset.extra["icon"] = st.icon
    asset.save()
    return out


def choose_icon(asset: Asset, index: int, project: Project | None = None) -> Path:
    """Make variation `index` (1-based) the item's icon.png (and its flat world sprite, if the
    world sprite was drawn from the picture rather than a 3D model)."""
    path = _use_icon(asset, int(index))
    asset.save()
    if asset.extra.get("world") and asset.extra.get("world_from") == "picture":
        render_item(asset, project=project)
    return path


def _model_picture(model: Path, st: Style) -> np.ndarray:
    """A three-quarter view of a model to make an icon from (transparent background)."""
    from .turnaround import render_model, union_crop
    views = render_model(model, 8, min(st.elevation, 30.0), size=384, light=max(st.light, 0.25))
    return union_crop([views[1]])[0]                       # SE: a three-quarter view


# --------------------------------------------------------------------------- world sprites

def _clear_sheet(folder: Path) -> None:
    for ext in SHEET_FILES:
        p = folder / f"sheet{ext}"
        if p.exists():
            p.unlink()


def _write_single(sprite, out: Path) -> list[Path]:
    """One frame: sheet + json + normal map + .aseprite + Godot SpriteFrames ('default')."""
    from .aseprite import write_aseprite
    from .export import pack_sheet, write_sheet
    from .godot import write_sprite_frames
    from .normals import write_normal_sheet
    out_json = write_sheet([sprite.rgba], out, 1, 100, None, ["default"])
    written = [out, out_json]
    if sprite.normal is not None:
        written.append(write_normal_sheet([sprite], out, 1))
    ase = out.with_suffix(".aseprite")
    write_aseprite(ase, [sprite.rgba], sprite.palette, durations=100)
    _, rects = pack_sheet([sprite.rgba], 1)
    tres = write_sprite_frames(out.with_suffix(".tres"), out.name, rects, name="default",
                               loop=False)
    return written + [ase, tres]


def write_world(sprites, out: Path) -> list[Path]:
    """Sheet files for 1, 4, 8 or 16 directions (old sheet files are removed first)."""
    from .turnaround import write_turnaround
    _clear_sheet(out.parent)
    if len(sprites) == 1:
        return _write_single(sprites[0], out)
    return write_turnaround(sprites, out, aseprite=True, godot=True, normals=True)


def lay_flat(pts: np.ndarray, nrm: np.ndarray, min_ratio: float = 1.8):
    """A long thing (sword, key, bone) lying on the ground: longest side along x, thinnest
    up; anything else is left standing. Points back on y = 0 around the y axis."""
    ext = np.ptp(pts, 0)
    order = np.argsort(ext)[::-1]                          # longest, middle, thinnest
    if ext[order[0]] < min_ratio * ext[order[1]]:
        return pts, nrm
    perm = np.zeros((3, 3))
    for new, old in zip((0, 2, 1), order):                 # longest -> x, middle -> z, thin -> y
        perm[new, old] = 1
    if np.linalg.det(perm) < 0:
        perm[2] *= -1
    pts, nrm = pts @ perm.T, nrm @ perm.T
    lo, hi = pts.min(0), pts.max(0)
    return pts - np.array([(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2]), nrm


def square_turn(pts: np.ndarray, step: float = 1.0) -> float:
    """Degrees to turn a model about its up axis (-45..45) so its footprint's box is smallest:
    a house's walls then run along the grid, and 4 directions show front, side, back, side."""
    xz = pts[:: max(1, len(pts) // 20000)][:, [0, 2]]

    def area(deg):
        a = math.radians(deg)
        x = xz[:, 0] * math.cos(a) + xz[:, 1] * math.sin(a)
        z = -xz[:, 0] * math.sin(a) + xz[:, 1] * math.cos(a)
        return np.ptp(x) * np.ptp(z)
    turns = np.arange(-45.0, 45.0, step)
    areas = np.array([area(t) for t in turns])
    best = float(turns[int(np.argmin(areas))])
    return 0.0 if areas.min() > 0.97 * area(0.0) else best   # already square enough


def turn_about_up(pts: np.ndarray, nrm: np.ndarray, deg: float):
    """Points and normals turned `deg` about y, re-centred on the y axis."""
    a = math.radians(deg)
    rot = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    pts, nrm = pts @ rot.T, nrm @ rot.T
    lo, hi = pts.min(0), pts.max(0)
    return pts - np.array([(lo[0] + hi[0]) / 2, 0, (lo[2] + hi[2]) / 2]), nrm


def render_world(model, st: Style, target_px: float, directions: int, measure: str = "height",
                 flat: bool = False, square: bool = False, yaw: float | None = None):
    """A model -> sprites at an exact scale: its height (measure "height") or its longest
    side ("longest") is target_px sprite pixels. 1 direction = facing the viewer (S).
    flat: a long thin model is laid down on the ground (lay_flat). square: turned so its
    footprint lines up with the grid (square_turn); yaw: turn by exactly this instead."""
    from .turnaround import load_model_points, render_points, turnaround_sprites, union_crop
    size = int(np.clip(target_px * 8, 256, 768))
    pts, cols, nrm = load_model_points(model, 8 * size * size)
    if flat:
        pts, nrm = lay_flat(pts, nrm)
    if yaw is None and square:
        yaw = square_turn(pts)
    if yaw:
        pts, nrm = turn_about_up(pts, nrm, yaw)
    e = math.radians(st.elevation)
    ce, se = math.cos(e), math.sin(e)
    # render_points' framing: render pixels per model unit
    radius = float(np.sqrt(pts[:, 0] ** 2 + pts[:, 2] ** 2).max())
    top = float(pts[:, 1].max()) * ce + radius * se
    bottom = float(pts[:, 1].min()) * ce - radius * se
    scale = size * 0.94 / max(2 * radius, top - bottom, 1e-9)
    units = float(np.ptp(pts[:, 1])) if measure == "height" else float(np.ptp(pts, 0).max())
    k = target_px / max(units, 1e-9) / scale               # sprite pixels per render pixel
    n = 4 if directions == 1 else directions
    views, nviews = render_points([(pts, nrm)], cols, n, st.elevation, size, light=st.light,
                                  normals=True)
    views, nviews = [v[0] for v in views], [v[0] for v in nviews]
    if directions == 1:
        views, nviews = [views[1]], [nviews[1]]           # S: the side the picture showed
    s = style_settings(st, 8, 8, st.colours, anchor="bottom")
    views, nviews = union_crop(views, s.margin, s.alpha_threshold, nviews)
    ch, cw = views[0].shape[:2]
    inset = 1 if s.outline else 0
    s.width = max(4, round(cw * k)) + 2 * inset
    s.height = max(4, round(ch * k)) + 2 * inset
    return turnaround_sprites(views, s, nviews)


def flat_world(picture, st: Style, target_px: float):
    """A flat item in the world without a 3D model: the picture at world size, with a flat
    (facing the camera) normal map."""
    from .normals import FLAT
    from .pipeline import convert
    s = style_settings(st, 8, 8, st.colours, anchor="bottom")
    inset = 1 if s.outline else 0
    s.width = s.height = max(4, round(target_px)) + 2 * inset
    sprite = convert(picture, s)
    nrm = np.zeros_like(sprite.rgba)
    nrm[..., :3] = FLAT
    nrm[..., 3] = sprite.rgba[..., 3]
    sprite.normal = nrm
    return [sprite]


def _front_frame(sprites) -> np.ndarray:
    """The frame facing the viewer (S), for thumbnails."""
    from .turnaround import direction_names
    if len(sprites) == 1:
        return sprites[0].rgba
    return sprites[direction_names(len(sprites)).index("S")].rgba


def load_world(asset: Asset) -> list[np.ndarray]:
    """The world sprite's frames back from sheet.png (direction order)."""
    sheet = asset.folder / "sheet.png"
    if not sheet.exists():
        return []
    meta = json.loads(sheet.with_suffix(".json").read_text(encoding="utf-8"))
    img = np.asarray(Image.open(sheet).convert("RGBA"))
    out = []
    for f in meta["frames"]:
        r = f["frame"]
        out.append(img[r["y"]:r["y"] + r["h"], r["x"]:r["x"] + r["w"]].copy())
    return out


# --------------------------------------------------------------------------- 3D

def _build_model(folder: Path, prompt, picture, client, seed, say):
    """Words or a picture -> model.glb (+ model.front.png, model.shape.glb, and from words
    model.back.png) on the GPU. Words: one clean prop picture (PROP_PROMPT); either way only
    the main subject of the cut-out goes to Hunyuan3D."""
    from .model3d import cut_out, prompt_to_model
    views = None
    if picture is None:
        from .generate import generate_images
        text = re.sub(r"^\s*(an?|the|one)\s+", "", prompt.strip(), flags=re.IGNORECASE)
        say(f"Making a picture of “{prompt}” on your GPU…")
        (picture, seed), = generate_images(PROP_PROMPT.format(prompt=text.rstrip(".")),
                                           PROP_NEGATIVE, client=client, seed=seed, aspect=1.0,
                                           lora="none")
        views = "front+back"                    # from words: the GPU draws the back too
    say("Cutting out the subject…")
    rgba = main_subject(cut_out(picture))
    say("Building a 3D model on your GPU (Hunyuan3D)…")
    res = prompt_to_model(prompt, folder / "model.glb", subject="object", picture=rgba,
                          client=client, seed=seed, views=views,
                          progress=lambda step, text: say(text))
    return seed if res.seed is None else res.seed


# --------------------------------------------------------------------------- items

def render_item(asset: Asset, size: str | None = None, directions: int | None = None,
                project: Project | None = None, progress=None) -> list[Path]:
    """(Re)render an item's world sprite without the GPU: from model.glb, or for 1 direction
    from the chosen picture. size / directions change the item's settings."""
    say = _say(progress)
    st = _project(project).style
    ex = asset.extra
    size = size or ex.get("size", "medium")
    if size not in ITEM_SIZES:
        raise ValueError(f"size must be one of {', '.join(ITEM_SIZES)}")
    directions = _directions(directions or ex.get("directions"), st.directions)
    target = st.scale(ITEM_SIZES[size])
    model = asset.folder / "model.glb"
    say(f"Rendering {asset.name} in the world from {directions} direction(s)…")
    if model.exists():
        sprites = render_world(model, st, target, directions, measure="longest",
                               flat=ex.get("category") != "potion")
        ex["world_from"] = "model"
    elif directions == 1:
        choice = int(ex.get("icon_choice", 1))
        pic = asset.folder / f"picture_v{choice}.png"
        if not pic.exists():
            raise ValueError(f"{asset.name} has no picture to draw a world sprite from")
        sprites = flat_world(pic, st, target)
        ex["world_from"] = "picture"
    else:
        raise ValueError(f"{asset.name} has no 3D model: make it again with a world sprite "
                         "(GPU), give it a 3D model, or use 1 direction (flat).")
    written = write_world(sprites, asset.folder / "sheet.png")
    ex.update(world=True, size=size, directions=directions, elevation=st.elevation,
              world_px=list(sprites[0].rgba.shape[1::-1]))
    asset.save()
    return written


def item_model(asset: Asset, client=None, project: Project | None = None, progress=None,
               render: bool = True) -> Path:
    """(Re)build an item's model.glb on the GPU from its chosen icon picture, so the world
    sprite is the same sword as the icon, then render the world sprite (render=True)."""
    choice = int(asset.extra.get("icon_choice", 1))
    pic = asset.folder / f"picture_v{choice}.png"
    if not pic.exists():
        raise ValueError(f"{asset.name} has no picture {choice} to build a model from")
    for old in ("model.glb", "model.front.png", "model.shape.glb"):
        (asset.folder / old).unlink(missing_ok=True)
    seeds = asset.extra.get("seeds") or []
    seed = seeds[choice - 1] if len(seeds) >= choice else asset.extra.get("seed")
    _build_model(asset.folder, None, pic, client, seed, _say(progress))
    asset.extra.update(model_from=choice)
    asset.save()
    if render:
        render_item(asset, project=project, progress=progress)
    return asset.folder / "model.glb"


def make_item(name: str, prompt: str | None = None, picture=None, model=None, *,
              category: str = "misc", world: bool = True, project: Project | None = None,
              progress=None, client=None, size: str = "medium", directions: int | None = None,
              seed: int | None = None, variations: int = VARIATIONS) -> Asset:
    """An inventory item: icon (variations from words, one from a photo or model) and,
    with world=True, a world sprite from a 3D model (made on the GPU from the words or the
    photo unless `model` is given; directions=1 = flat, drawn from the picture, no 3D)."""
    from .generate import generate_images
    from .pipeline import load_image
    if not (prompt or picture is not None or model is not None):
        raise ValueError("Give a prompt, a picture or a 3D model.")
    if category not in CATEGORIES:
        raise ValueError(f"category must be one of {', '.join(CATEGORIES)}")
    if size not in ITEM_SIZES:
        raise ValueError(f"size must be one of {', '.join(ITEM_SIZES)}")
    say = _say(progress)
    proj = _project(project)
    st = proj.style
    directions = _directions(directions, st.directions)
    source = "picture" if picture is not None else ("words" if prompt else "model")
    asset = new_asset("item", name, project=proj, prompt=prompt or "", category=category,
                      size=size, source=source)
    f = asset.folder
    try:
        seeds: list[int] = []
        if model is not None:
            shutil.copyfile(model, f / "model.glb")
        if picture is not None:
            Image.fromarray(load_image(picture), "RGBA").save(f / "picture_v1.png")
        elif prompt:
            positive, negative = icon_prompt(prompt, category, st.icon)
            say(f"Drawing {variations} icons of “{prompt}” on your GPU…")
            raw = generate_images(positive, negative, client=client, seed=seed,
                                  count=max(1, variations), aspect=1.0)
            for i, (img, s) in enumerate(raw, 1):
                img.convert("RGB").save(f / f"picture_v{i}.png")
                seeds.append(int(s))
        else:
            say("Drawing the icon from the model…")
            Image.fromarray(_model_picture(f / "model.glb", st), "RGBA").save(f / "picture_v1.png")
        asset.extra.update(seeds=seeds, seed=seeds[0] if seeds else seed, icon_choice=1)
        say("Converting the icons to the project's style…")
        redo_icons(asset, proj)
        asset.extra["world"] = False
        if world:
            if not (f / "model.glb").exists() and directions > 1:
                item_model(asset, client=client, project=proj, progress=say, render=False)
            render_item(asset, size, directions, proj, say)
        asset.save()
    except BaseException:
        shutil.rmtree(f, ignore_errors=True)
        raise
    say(f"{name}: done. Files: {f}")
    return asset


# --------------------------------------------------------------------------- props

def render_prop(asset: Asset, height_people: float | None = None, directions: int | None = None,
                project: Project | None = None, progress=None,
                yaw: float | None = None) -> list[Path]:
    """(Re)render a prop from its model.glb at the project's scale (no GPU): height_people =
    how tall it stands next to a person (barrel 0.6, tree 2.5, house 3). yaw: degrees to turn
    the model about its up axis (default: squared up to the grid, square_turn, the smallest
    turn; add 90 to make another side the front)."""
    say = _say(progress)
    st = _project(project).style
    ex = asset.extra
    model = asset.folder / "model.glb"
    if not model.exists():
        raise ValueError(f"{asset.name} has no 3D model (model.glb)")
    hp = float(height_people or ex.get("height_people") or 1.0)
    if hp <= 0:
        raise ValueError("height_people must be more than 0")
    directions = _directions(directions or ex.get("directions"), st.directions)
    say(f"Rendering {asset.name} ({hp:g} people tall) from {directions} direction(s)…")
    yaw = ex.get("yaw") if yaw is None else float(yaw) % 360
    if yaw is None:                       # buildings, carts, fences: walls along the grid
        from .turnaround import load_model_points
        yaw = square_turn(load_model_points(model, 60_000)[0]) if directions > 1 else 0.0
    sprites = render_world(model, st, hp * st.person_height, directions, measure="height",
                           yaw=yaw)
    ex["yaw"] = yaw
    written = write_world(sprites, asset.folder / "sheet.png")
    make_thumb(_front_frame(sprites), asset.thumb)
    ex.update(height_people=hp, directions=directions, elevation=st.elevation,
              person_height=st.person_height, frame=list(sprites[0].rgba.shape[1::-1]))
    asset.save()
    return written


def make_prop(name: str, prompt: str | None = None, picture=None, model=None, *,
              height_people: float | None = None, directions: int | None = None,
              project: Project | None = None, progress=None, client=None,
              seed: int | None = None) -> Asset:
    """A prop or building: 3D model (from words or a photo on the GPU, or your model file) ->
    sprites at the project's scale. height_people / directions default to a guess from the
    name and words (house 3.0 / 4 directions, tree 2.5 / 1, barrel 0.6 / 1; else 1.0 / 1)."""
    if not (prompt or picture is not None or model is not None):
        raise ValueError("Give a prompt, a picture or a 3D model.")
    say = _say(progress)
    proj = _project(project)
    g_height, g_dirs = guess_prop(f"{name} {prompt or ''}")
    hp = float(height_people or g_height)
    dirs = _directions(directions or g_dirs, g_dirs)
    source = "model" if model is not None else ("picture" if picture is not None else "words")
    asset = new_asset("prop", name, project=proj, prompt=prompt or "", source=source,
                      height_people=hp, directions=dirs)
    f = asset.folder
    try:
        if model is not None:
            shutil.copyfile(model, f / "model.glb")
        else:
            asset.extra["seed"] = _build_model(f, prompt, picture, client, seed, say)
        render_prop(asset, hp, dirs, proj, say)
    except BaseException:
        shutil.rmtree(f, ignore_errors=True)
        raise
    say(f"{name}: done. Files: {f}")
    return asset


def repaint(asset: Asset, project: Project | None = None, progress=None) -> list[Path]:
    """Colour model.glb again from model.shape.glb and the saved pictures (model.front.png,
    model.back/left/right.png), then redo the sprites. No GPU."""
    from .model3d import paint_from_picture
    from .pipeline import load_image
    f = asset.folder
    shape, front = f / "model.shape.glb", f / "model.front.png"
    if not (shape.exists() and front.exists()):
        raise ValueError(f"{asset.name} has no saved shape and picture to repaint from")
    _say(progress)(f"Painting {asset.name} again…")
    others = {k: load_image(f / f"model.{k}.png") if (f / f"model.{k}.png").exists() else None
              for k in ("back", "left", "right")}
    mesh, _ = paint_from_picture(shape, load_image(front), **others)
    mesh.export(f / "model.glb")
    return rerender(asset, project, progress)


def rerender(asset: Asset, project: Project | None = None, progress=None, **settings) -> list[Path]:
    """Redo an item's or prop's sprites with new settings, no GPU (items: size, directions;
    props: height_people, directions)."""
    if asset.kind == "prop":
        return render_prop(asset, project=project, progress=progress, **settings)
    if asset.kind == "item":
        return render_item(asset, project=project, progress=progress, **settings)
    raise ValueError(f"Not an item or prop: {asset.kind}")

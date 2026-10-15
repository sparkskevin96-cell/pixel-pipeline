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

from .assets import Asset, Project, Style, current_project, new_asset, project_style, with_style
from .subjectcheck import SHEET_NEGATIVE, shape_for, single_options

CATEGORIES = ("weapon", "armour", "potion", "food", "key", "misc")
# an item's longest side in the world, in pixels for the default 64 px person (Style.scale)
ITEM_SIZES = {"tiny": 12, "small": 16, "medium": 24, "large": 36}
DIRECTION_CHOICES = (1, 2, 4, 8, 16)      # 2 = E, W (side-scrollers)
ICON_COLOURS = 16              # at most this many colours in an icon (32 px can't show more)
VARIATIONS = 3

CATEGORY_HINTS = {
    "weapon": "single weapon standing upright, pointing straight up, vertical",
    "armour": "just that one piece of armour, front view",
    "potion": "single glass potion bottle, front view",
    "food": "single food item",
    "key": "single key, upright",
    "misc": "single object",
}
# subject first; no "inventory icon" / "game asset" (they mean a page of icons to SDXL). The
# layout comes from subjectcheck's start blob + check (QA 2026-10-15: 61% sheets before).
ICON_PROMPT = ("{prompt}, pixel art item icon of one single {thing}, {hint}, centered, whole "
               "object in frame, isolated on a plain white background, flat colors, bold dark "
               "outline, simple readable shapes")
ICON_NEGATIVE = ("3d render, realistic, photo, blurry, smooth gradient, noise, jpeg artifacts, "
                 "text, watermark, signature, logo, multiple objects, sprite sheet, hand, person, "
                 "character, table, floor, scenery, cast shadow, frame, border, user interface, "
                 "cropped, cut off, out of frame, " + SHEET_NEGATIVE)
SMALL_ICON_HINT = "very simple, low detail, chunky pixels, bold silhouette"
# the clean picture a prop's 3D model is built from (no pixel LoRA). SDXL likes product
# sheets (five barrels, a lid, a stand), which image-to-3D fuses into one lump: insist on one.
PROP_PROMPT = ("a single {prompt} standing on its own, one object only, isolated, whole object in "
               "frame, three-quarter front view, centered, plain white background, soft even "
               "lighting, no shadows, clean shapes, game prop illustration")
PROP_NEGATIVE = ("multiple objects, two, pair, group, set, collection, variations, several views, "
                 "product sheet, character sheet, stack, pile, cropped, cut off, out of frame, "
                 "partial, crowd, text, watermark, busy background, scenery, landscape, floor, "
                 "dramatic lighting, harsh shadows, motion blur, blurry, pixel art, " + SHEET_NEGATIVE)
NO_DIAGONAL = ("potion", "armour")   # long thin items lie diagonally in the icon, except these
# world sprites of items: flat (drawn from the icon picture) or 3D (from model.glb)
WORLD_STYLES = ("auto", "flat", "3d")
FLAT_CATEGORIES = ("weapon", "key")
SOLID_CATEGORIES = ("potion", "food", "armour")
# how much of a model's baked side light equalise_faces removes for props seen from several
# sides (a house: the picture saw its door side at a steep angle). A one-direction prop shows
# the side the picture saw, and round things (a potion, a barrel) only get blotchy from it.
HAND_EQUALISE = 0.85

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
SHEET_FILES = (".png", ".json", "_n.png", "_glow.png", ".aseprite", ".tres", "_facing.gd")


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


def icon_prompt(prompt: str, category: str = "misc", icon_px: int = 32,
                style: Style | None = None) -> tuple[str, str]:
    """(positive, negative) for an item icon; the house style (style, default: the current
    project's) is appended (assets.style_words "icon")."""
    text = re.sub(r"^\s*(an?\s+)?pixel[- ]?art\s+(icon\s+|sprite\s+)?(of\s+)?", "", prompt or "",
                  flags=re.IGNORECASE).strip().rstrip(".,; ")
    if not text:
        raise ValueError("Empty prompt: describe the item, e.g. 'an iron sword'.")
    thing = re.sub(r"^(an?|the|one)\s+", "", text, flags=re.IGNORECASE)
    positive = ICON_PROMPT.format(prompt=text, thing=thing,
                                  hint=CATEGORY_HINTS.get(category, "single object"))
    if icon_px <= 24:
        positive += ", " + SMALL_ICON_HINT
    return with_style(positive, ICON_NEGATIVE, style or project_style(), "icon")   # house style


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
    ratio, angle = elongation(rgba[..., 3])
    if ratio < min_ratio:
        return rgba
    if not head_is_forward(rgba[..., 3], angle):
        angle += 180                                       # the head goes to the top right
    turn = (45.0 - angle + 180) % 360 - 180                # counter-clockwise, degrees
    if abs(turn) < 12:
        return rgba
    return _rotate_rgba(rgba, turn)


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
    world sprite was drawn from the picture or icon rather than a 3D model)."""
    path = _use_icon(asset, int(index))
    asset.save()
    if asset.extra.get("world") and asset.extra.get("world_from") in ("picture", "icon"):
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
        from .glow import write_glow_sheet                # lanterns, crystals, braziers glow
        g = write_glow_sheet(out, force=True)
        if g is not None:
            written.append(g)
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


def equalise_faces(pts: np.ndarray, cols: np.ndarray, nrm: np.ndarray,
                   strength: float = 0.85, sectors: int = 8) -> np.ndarray:
    """Point colours with the light the source picture baked in taken out per face direction.
    A model painted from a picture is darker and greyer on the sides the picture saw at a
    steep angle (or not at all), so a cottage's door wall came out duller than its gable.
    Points are grouped by which way they face (walls / roof slopes / tops, in `sectors`
    compass sectors); each group's lightness is quantile-matched to all the groups of its
    kind together, and its saturation scaled to theirs (`strength` 0..1). The renderer's
    and hand_pixel's light from the normals then shades every side the same clean way."""
    from .color import lab_to_rgb, rgb_to_lab
    if strength <= 0 or len(pts) < 100:
        return cols
    n = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    centre = np.array([0.0, float(np.median(pts[:, 1])), 0.0])
    n = np.where(((pts - centre) * n).sum(1, keepdims=True) < 0, -n, n)   # face outward
    ny = n[:, 1]
    kind = np.select([ny > 0.85, ny > 0.35, ny > -0.35], [0, 1, 2], 3)    # top slope wall under
    az = np.arctan2(n[:, 0], n[:, 2])
    sector = np.clip(((az + np.pi) / (2 * np.pi) * sectors).astype(int), 0, sectors - 1)
    lab = rgb_to_lab(cols[:, :3].astype(np.float32).reshape(-1, 1, 3)).reshape(-1, 3)
    out = lab.copy()
    for k in (1, 2):                     # roof slopes and walls; tops face one way already
        ink = kind == k
        if ink.sum() < 200:
            continue
        ref = np.sort(lab[ink, 0])
        ref_c = float(np.hypot(lab[ink, 1], lab[ink, 2]).mean())
        for s in range(sectors):
            q = np.nonzero(ink & (sector == s))[0]
            if len(q) < max(50, 0.02 * ink.sum()):
                continue
            L = lab[q, 0]
            rank = np.argsort(np.argsort(L)) / max(len(q) - 1, 1)
            target = np.interp(rank, np.linspace(0, 1, len(ref)), ref)
            out[q, 0] = L + strength * (target - L)
            c = float(np.hypot(lab[q, 1], lab[q, 2]).mean())
            gain = float(np.clip(ref_c / max(c, 1e-3), 0.7, 1.6)) ** strength
            out[q, 1:] = lab[q, 1:] * gain
    rgb = lab_to_rgb(out.reshape(-1, 1, 3)).reshape(-1, 3)
    res = cols.copy()
    res[:, :3] = rgb.astype(cols.dtype)
    return res


def render_world(model, st: Style, target_px: float, directions: int, measure: str = "height",
                 flat: bool = False, square: bool = False, yaw: float | None = None,
                 equalise: float = 0.0, hand: bool | dict = False):
    """A model -> sprites at an exact scale: its height (measure "height") or its longest
    side ("longest") is target_px sprite pixels. 1 direction = facing the viewer (S).
    flat: a long thin model is laid down on the ground (lay_flat). square: turned so its
    footprint lines up with the grid (square_turn); yaw: turn by exactly this instead.
    equalise: take the picture's baked light out of the model's sides (equalise_faces).
    hand: the hand-pixelled finish (pixelclean.hand_pixel; a dict = its options)."""
    from .turnaround import load_model_points, render_points, turnaround_sprites, union_crop
    size = int(np.clip(target_px * 8, 256, 768))
    pts, cols, nrm = load_model_points(model, 8 * size * size)
    if equalise:
        cols = equalise_faces(pts, cols, nrm, equalise)
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
    sprites = turnaround_sprites(views, s, nviews)
    if hand:
        from .pixelclean import hand_pixel, outline_style
        opts = dict(hand) if isinstance(hand, dict) else {}
        if s.palette and s.palette != "auto":
            opts.setdefault("palette", s.palette)          # stay on the project's palette
        opts.setdefault("steps", st.ramp_steps())          # house style shading (soft = 5)
        sprites = hand_pixel(sprites, outline_style(s), **opts)
    return sprites


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


# --------------------------------------------------------------------------- flat from icon

def _rotate_rgba(rgba: np.ndarray, deg: float) -> np.ndarray:
    """Turned `deg` counter-clockwise (premultiplied cubic: no dark fringes) on a square
    canvas big enough for any turn."""
    import cv2
    ys, xs = np.nonzero(rgba[..., 3] >= 8)
    if not len(xs):
        return rgba
    crop = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape[:2]
    d = int(math.hypot(h, w)) + 4
    pad = np.zeros((d, d, 4), np.uint8)
    pad[(d - h) // 2:(d - h) // 2 + h, (d - w) // 2:(d - w) // 2 + w] = crop
    a = pad[..., 3:4].astype(np.float32) / 255
    pre = np.concatenate([pad[..., :3] * a, a * 255], -1)
    m = cv2.getRotationMatrix2D((d / 2, d / 2), deg, 1.0)
    rot = cv2.warpAffine(pre, m, (d, d), flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0, 0))
    alpha = np.clip(rot[..., 3:4], 0, 255)
    rgb = rot[..., :3] / np.maximum(alpha / 255, 1e-6)
    return np.clip(np.concatenate([rgb, alpha], -1), 0, 255).astype(np.uint8)


def lying_pose(rgba: np.ndarray) -> np.ndarray:
    """A cut-out item picture turned so its long axis runs left to right, head (blade, axe
    head, key bit) on the right, cropped to the item."""
    ratio, angle = elongation(rgba[..., 3])
    if not head_is_forward(rgba[..., 3], angle):
        angle += 180
    out = _rotate_rgba(rgba, -angle)
    ys, xs = np.nonzero(out[..., 3] >= 8)
    return out[ys.min():ys.max() + 1, xs.min():xs.max() + 1] if len(xs) else out


def thinnest(alpha: np.ndarray, part: float = 0.3) -> float:
    """How thick (rows) the thin parts of a left-to-right item are: the `part` quantile of
    the solid height of its columns, the ends left out (a sword: its blade, not the tip)."""
    solid = alpha >= 128
    cols = solid.sum(0)
    cols = cols[cols > 0]
    if len(cols) < 5:
        return float(cols.max()) if len(cols) else 0.0
    cut = max(1, len(cols) // 10)
    return float(np.quantile(cols[cut:-cut], part))


def thicken(rgba: np.ndarray, rows: float, point: bool = True) -> np.ndarray:
    """A left-to-right item with its thin parts (thinnest) `rows` thick: every column is
    stretched about its middle to gain the same number of rows, so a blade reaches `rows`
    and a crossguard or axe head stays that much bigger than it (the shape keeps its
    hierarchy, the way a sprite artist exaggerates small parts)."""
    import cv2
    h, w = rgba.shape[:2]
    solid = rgba[..., 3] >= 128
    has = solid.any(0)
    extra = rows - thinnest(rgba[..., 3])
    if not has.any() or extra <= 0:
        return rgba
    top = np.where(has, solid.argmax(0), 0)
    bottom = np.where(has, h - 1 - solid[::-1].argmax(0), 0)
    run = np.where(has, bottom - top + 1, 0).astype(np.float32)
    centre = np.where(has, (top + bottom) / 2.0, h / 2.0).astype(np.float32)
    xs = np.nonzero(has)[0]
    taper = (np.clip((xs.max() - np.arange(w)) / max(0.08 * w, 1), 0.3, 1.0) if point
             else np.ones(w))                     # point: the head end (a blade tip) tapers
    scale = np.where(has, (run + extra * taper) / np.maximum(run, 1), 1.0).astype(np.float32)
    k = max(3, (w // 40) | 1)                       # soft steps where the thickness changes
    scale = cv2.blur(cv2.erode(scale[None], np.ones((1, k), np.uint8))[0][None], (k, 1))[0]
    centre = cv2.blur(centre[None], (k, 1))[0]
    grow = int(math.ceil(float((run * (scale - 1)).max()) / 2)) + 2
    H = h + 2 * grow
    yo = np.arange(H, dtype=np.float32)[:, None] - grow
    map_y = (centre[None] + (yo - centre[None]) / scale[None]).astype(np.float32)
    map_x = np.broadcast_to(np.arange(w, dtype=np.float32)[None], (H, w)).astype(np.float32)
    a = rgba[..., 3:4].astype(np.float32) / 255
    pre = np.concatenate([rgba[..., :3] * a, a * 255], -1).astype(np.float32)
    out = cv2.remap(pre, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0, 0))
    alpha = np.clip(out[..., 3:4], 0, 255)
    rgb = out[..., :3] / np.maximum(alpha / 255, 1e-6)
    return np.clip(np.concatenate([rgb, alpha], -1), 0, 255).astype(np.uint8)


def _lit_on_top(rgba: np.ndarray) -> bool:
    """Is the upper half (above each column's middle) of a left-to-right item the lighter?"""
    from .color import luminance
    solid = rgba[..., 3] >= 128
    if not solid.any():
        return True
    h = rgba.shape[0]
    has = solid.any(0)
    top = solid.argmax(0)
    bottom = h - 1 - solid[::-1].argmax(0)
    mid = (top + bottom) / 2.0
    lum = luminance(rgba[..., :3])
    yy = np.arange(h)[:, None]
    up = solid & has[None] & (yy < mid[None])
    down = solid & has[None] & (yy > mid[None])
    if not up.any() or not down.any():
        return True
    return float(lum[up].mean()) >= float(lum[down].mean())


def icon_world(picture: np.ndarray, st: Style, target_px: float, directions: int,
               min_px: int | None = None, colours: int | None = None, tilt: float = 0.5):
    """A thin item (sword, key, wand, arrow) lying on the ground, drawn from its icon picture
    rather than a 3D model ("flat (from icon)"): the picture's own icon shading and outline
    survive, which a few-pixel-thick 3D render loses. The item is laid along the facing
    direction on the ground, its thin parts thickened (thicken) so they are at least `min_px`
    body pixels across plus the outline (default 2, 1 under 20 px long), and flipped so its lit edge always
    faces the light (top left). Foreshortening: the ground's depth is squashed by
    sin(elevation) for a real camera; tilt 0..1 tips the item up toward the viewer by that
    share (0.5 default: sprite artists draw items on the floor less squashed than they'd be,
    or a crossguard is a sliver). picture: the cut-out RGBA picture the icon was made from.
    Longest side = target_px. 1 direction = lying diagonally like the icon (head up right)."""
    from .normals import encode
    from .turnaround import turnaround_sprites, union_crop
    base = lying_pose(picture)
    length = base.shape[1]
    k = target_px / max(length, 1)                     # sprite pixels per picture pixel
    e = math.radians(st.elevation)
    ce, se = math.cos(e), math.sin(e)
    se = se + (1 - se) * float(np.clip(tilt, 0, 1))  # how squashed the ground depth looks
    s = style_settings(st, 8, 8, colours or min(st.colours, ICON_COLOURS), anchor="bottom")
    ring = 2 if s.outline else 0
    small = target_px < 20                            # a 16 px sword: 1 px of blade, or a stub
    if min_px is None:
        min_px = 1 if small else 2
    # parts thin along the item too (a crossguard, a key's teeth) about 2 px wide
    base = np.ascontiguousarray(thicken(np.ascontiguousarray(base.transpose(1, 0, 2)),
                                        (1.5 if small else 2.2) / k, point=False).transpose(1, 0, 2))
    k = target_px / max(base.shape[1], 1)
    lit = _lit_on_top(base)
    if directions == 1:
        angles = [-45.0]                               # NE: like the icon
    else:
        angles = [i * 360.0 / directions for i in range(directions)]
    views = []
    for a in angles:
        r = math.radians(a)
        # how much the item's width shrinks lying this way: its across-the-ground direction
        # (-sin a, cos a) is seen as (-sin a, cos a * sin(elevation)) on screen
        across = math.hypot(math.sin(r), math.cos(r) * se)
        need = (min_px + ring) / max(k * across, 1e-6)
        img = thicken(base, need)
        # the lit edge toward the light (top left): the picture's top edge lands on screen
        # direction (sin a, -cos a * se) (y down); flip the item over when that's away
        up = np.array([math.sin(r), -math.cos(r) * se])
        if (float(up @ np.array([-0.6, -0.8])) >= 0) != lit:
            img = img[::-1].copy()
        views.append(_foreshorten(img, se, a))
    size = max(max(v.shape[:2]) for v in views)
    canvas = []
    for v in views:
        c = np.zeros((size, size, 4), np.uint8)
        y0, x0 = (size - v.shape[0]) // 2, (size - v.shape[1]) // 2
        c[y0:y0 + v.shape[0], x0:x0 + v.shape[1]] = v
        canvas.append(c)
    flat = np.zeros(3, np.float32)
    flat[:] = (0.0, ce, math.sin(e))                          # lying flat: the ground's normal
    nviews = []
    for c in canvas:
        n = np.zeros_like(c)
        n[..., :3] = encode(flat)
        n[..., 3] = c[..., 3]
        nviews.append(n)
    crops, ncrops = union_crop(canvas, s.margin, s.alpha_threshold, nviews)
    ch, cw = crops[0].shape[:2]
    inset = 1 if s.outline else 0
    s.width = max(4, round(cw * k)) + 2 * inset
    s.height = max(4, round(ch * k)) + 2 * inset
    return turnaround_sprites(crops, s, ncrops)


def _foreshorten(img: np.ndarray, se: float, ground_deg: float) -> np.ndarray:
    """A left-to-right item picture as it looks lying on the ground facing ground_deg:
    the exact affine of the ground plane seen from the camera (x stays, depth x sin(elev))."""
    import cv2
    ys, xs = np.nonzero(img[..., 3] >= 8)
    crop = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape[:2]
    r = math.radians(ground_deg)
    # item axes on the ground (x along, y across) -> screen (y down)
    M = np.array([[math.cos(r), -math.sin(r)], [math.sin(r) * se, math.cos(r) * se]])
    corners = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]) @ M.T
    size = int(np.ceil(np.abs(corners).max() * 2)) + 4
    A = np.zeros((2, 3))
    A[:, :2] = M
    A[:, 2] = np.array([size / 2, size / 2]) - M @ np.array([w / 2, h / 2])
    a = crop[..., 3:4].astype(np.float32) / 255
    pre = np.concatenate([crop[..., :3] * a, a * 255], -1).astype(np.float32)
    out = cv2.warpAffine(pre, A, (size, size), flags=cv2.INTER_CUBIC,
                         borderValue=(0, 0, 0, 0))
    alpha = np.clip(out[..., 3:4], 0, 255)
    rgb = out[..., :3] / np.maximum(alpha / 255, 1e-6)
    return np.clip(np.concatenate([rgb, alpha], -1), 0, 255).astype(np.uint8)


def _front_frame(sprites) -> np.ndarray:
    """The frame facing the viewer (S), for thumbnails."""
    from .turnaround import direction_names
    if len(sprites) == 1:
        return sprites[0].rgba
    names = direction_names(len(sprites))
    return sprites[names.index("S") if "S" in names else 0].rgba          # 2 directions: E


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

def _build_model(folder: Path, prompt, picture, client, seed, say, st: Style | None = None):
    """Words or a picture -> model.glb (+ model.front.png, model.shape.glb, and from words
    model.back.png) on the GPU. Words: one clean prop picture (PROP_PROMPT); either way only
    the main subject of the cut-out goes to Hunyuan3D."""
    from .model3d import cut_out, prompt_to_model
    views = None
    if picture is None:
        from .subjectcheck import generate_single
        text = re.sub(r"^\s*(an?|the|one)\s+", "", prompt.strip(), flags=re.IGNORECASE)
        say(f"Making a picture of “{prompt}” on your GPU…")
        positive, negative = with_style(PROP_PROMPT.format(prompt=text.rstrip(".")),
                                        PROP_NEGATIVE, st or project_style(), "model-object")
        best, = generate_single(positive, negative, client=client, seed=seed, aspect=1.0,
                                lora="none", say=say, init=shape_for(text),
                                **single_options("model-object"))
        picture, seed = best.image, best.seed
        views = "front+back"                    # from words: the GPU draws the back too
    say("Cutting out the subject…")
    rgba = main_subject(cut_out(picture))
    say("Building a 3D model on your GPU (Hunyuan3D)…")
    res = prompt_to_model(prompt, folder / "model.glb", subject="object", picture=rgba,
                          client=client, seed=seed, views=views, style=st,
                          progress=lambda step, text: say(text))
    return seed if res.seed is None else res.seed


# --------------------------------------------------------------------------- items

def world_style(asset: Asset, want: str | None = None) -> str:
    """How an item's world sprite is drawn: "flat" (from the icon picture, icon_world) or
    "3d" (from model.glb). want / extra["world_style"]: "auto" (default), "flat" or "3d".
    Auto: an item made from your model file 3D; else weapons and keys flat (thin: a 3D
    render of them is a few pixels of outline), potions, food and armour 3D (chunky: their
    volume shows); misc by the icon's shape (long and thin = flat)."""
    want = want or asset.extra.get("world_style", "auto")
    if want not in WORLD_STYLES:
        raise ValueError(f"world style must be one of {', '.join(WORLD_STYLES)}")
    if want != "auto":
        return want
    if asset.extra.get("source") == "model":
        return "3d"                                   # you gave it a model: use it
    cat = asset.extra.get("category", "misc")
    if cat in FLAT_CATEGORIES:
        return "flat"
    if cat in SOLID_CATEGORIES:
        return "3d"
    icon = asset.folder / "icon.png"
    if icon.exists():
        ratio, _ = elongation(np.asarray(Image.open(icon).convert("RGBA"))[..., 3])
        if ratio >= 2.2:
            return "flat"
    return "3d"


def _icon_picture(asset: Asset, st: Style) -> np.ndarray | None:
    """The chosen icon's picture, cut out (main subject only), or None."""
    from .pipeline import isolate, load_image
    pic = asset.folder / f"picture_v{int(asset.extra.get('icon_choice', 1))}.png"
    if not pic.exists():
        return None
    return main_subject(isolate(load_image(pic), icon_settings(st)))


def render_item(asset: Asset, size: str | None = None, directions: int | None = None,
                project: Project | None = None, progress=None,
                style: str | None = None) -> list[Path]:
    """(Re)render an item's world sprite without the GPU. size / directions / style ("auto",
    "flat", "3d": world_style) change the item's settings. Flat = drawn from the chosen icon
    picture (icon_world), 3D = from model.glb; a chunky item with no model and 1 direction
    uses the picture as it is (flat_world)."""
    say = _say(progress)
    st = _project(project).style
    ex = asset.extra
    size = size or ex.get("size", "medium")
    if size not in ITEM_SIZES:
        raise ValueError(f"size must be one of {', '.join(ITEM_SIZES)}")
    directions = _directions(directions or ex.get("directions"), st.directions)
    if style is not None:
        world_style(asset, style)                     # checks it
        ex["world_style"] = style
    how = world_style(asset)
    target = st.scale(ITEM_SIZES[size])
    model = asset.folder / "model.glb"
    say(f"Rendering {asset.name} in the world from {directions} direction(s)…")
    picture = _icon_picture(asset, st) if how == "flat" else None
    if picture is not None:
        from .pixelclean import clean_sprites, outline_style
        sprites = icon_world(picture, st, target, directions)
        sprites = clean_sprites(sprites, outline_style(style_settings(st, 8, 8)), islands=1)
        ex["world_from"] = "icon"
    elif model.exists():
        sprites = render_world(model, st, target, directions, measure="longest",
                               flat=ex.get("category") != "potion",
                               hand=bool(ex.get("hand_pixel", True)))
        ex["world_from"] = "model"
    elif directions == 1:
        choice = int(ex.get("icon_choice", 1))
        pic = asset.folder / f"picture_v{choice}.png"
        if not pic.exists():
            raise ValueError(f"{asset.name} has no picture to draw a world sprite from")
        sprites = flat_world(pic, st, target)
        ex["world_from"] = "picture"
    else:
        raise ValueError(f"{asset.name} has no 3D model: make one (GPU), give it a 3D model, "
                         "draw it flat from the icon, or use 1 direction.")
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
    _build_model(asset.folder, None, pic, client, seed, _say(progress), _project(project).style)
    asset.extra.update(model_from=choice)
    asset.save()
    if render:
        render_item(asset, project=project, progress=progress)
    return asset.folder / "model.glb"


def make_item(name: str, prompt: str | None = None, picture=None, model=None, *,
              category: str = "misc", world: bool = True, project: Project | None = None,
              progress=None, client=None, size: str = "medium", directions: int | None = None,
              seed: int | None = None, variations: int = VARIATIONS,
              style: str = "auto") -> Asset:
    """An inventory item: icon (variations from words, one from a photo or model) and,
    with world=True, a world sprite. style (world_style): "flat" draws it from the chosen
    icon picture (no 3D, no GPU), "3d" renders a 3D model (made on the GPU from the words or
    the photo unless `model` is given), "auto" picks by category (weapons and keys flat,
    potions, food and armour 3D; a given model is used)."""
    from .pipeline import load_image
    from .subjectcheck import generate_single
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
    if style not in WORLD_STYLES:
        raise ValueError(f"style must be one of {', '.join(WORLD_STYLES)}")
    asset = new_asset("item", name, project=proj, prompt=prompt or "", category=category,
                      size=size, source=source, world_style=style)
    f = asset.folder
    try:
        seeds: list[int] = []
        if model is not None:
            shutil.copyfile(model, f / "model.glb")
        if picture is not None:
            Image.fromarray(load_image(picture), "RGBA").save(f / "picture_v1.png")
        elif prompt:
            positive, negative = icon_prompt(prompt, category, st.icon, st)
            say(f"Drawing {variations} icons of “{prompt}” on your GPU…")
            raw = generate_single(positive, negative, client=client, seed=seed,
                                  count=max(1, variations), aspect=1.0,
                                  say=say, init=shape_for(f"{name} {prompt}"),
                                  init_words=f"{name} {prompt}",
                                  **single_options("icon"))                   # best first
            for i, pick in enumerate(raw, 1):
                pick.image.convert("RGB").save(f / f"picture_v{i}.png")
                seeds.append(pick.seed)
            asset.extra["checks"] = [pick.verdict.summary() for pick in raw]
        else:
            say("Drawing the icon from the model…")
            Image.fromarray(_model_picture(f / "model.glb", st), "RGBA").save(f / "picture_v1.png")
        asset.extra.update(seeds=seeds, seed=seeds[0] if seeds else seed, icon_choice=1)
        say("Converting the icons to the project's style…")
        redo_icons(asset, proj)
        asset.extra["world"] = False
        if world:
            if (not (f / "model.glb").exists() and directions > 1
                    and world_style(asset) == "3d"):
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
                yaw: float | None = None, hand: bool | None = None) -> list[Path]:
    """(Re)render a prop from its model.glb at the project's scale (no GPU): height_people =
    how tall it stands next to a person (barrel 0.6, tree 2.5, house 3). yaw: degrees to turn
    the model about its up axis (default: squared up to the grid, square_turn, the smallest
    turn; add 90 to make another side the front). hand: the hand-pixelled finish (default
    on, kept in extra["hand_pixel"]): material ramps and clean-up (pixelclean.hand_pixel),
    and with 4+ directions every side lit alike (equalise_faces); off = the converter look."""
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
    hand = bool(ex.get("hand_pixel", True)) if hand is None else bool(hand)
    sprites = render_world(model, st, hp * st.person_height, directions, measure="height",
                           yaw=yaw, hand=hand,
                           equalise=HAND_EQUALISE if hand and directions > 1 else 0.0)
    ex["yaw"] = yaw
    ex["hand_pixel"] = hand
    written = write_world(sprites, asset.folder / "sheet.png")
    make_thumb(_front_frame(sprites), asset.thumb)
    ex.update(height_people=hp, directions=directions, elevation=st.elevation,
              person_height=st.person_height, frame=list(sprites[0].rgba.shape[1::-1]))
    asset.save()
    return written


def make_prop(name: str, prompt: str | None = None, picture=None, model=None, *,
              height_people: float | None = None, directions: int | None = None,
              project: Project | None = None, progress=None, client=None,
              seed: int | None = None, hand: bool = True, look: str = "auto") -> Asset:
    """A prop or building. look "drawn" (the default from words): drawn as pixel art by the GPU
    (draw_prop; 1 view, or 4 sides with directions=4). look "3d" (the default from a photo or
    a model file): 3D model (from words or a photo on the GPU, or your model file) ->
    sprites at the project's scale. height_people / directions default to a guess from the
    name and words (house 3.0 / 4 directions, tree 2.5 / 1, barrel 0.6 / 1; else 1.0 / 1).
    hand: the hand-pixelled finish (render_prop)."""
    if not (prompt or picture is not None or model is not None):
        raise ValueError("Give a prompt, a picture or a 3D model.")
    if look == "drawn" or (look == "auto" and prompt and picture is None and model is None):
        return draw_prop(name, prompt, height_people=height_people, directions=directions or 1,
                         project=project, progress=progress, client=client, seed=seed)
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
            asset.extra["seed"] = _build_model(f, prompt, picture, client, seed, say, proj.style)
        render_prop(asset, hp, dirs, proj, say, hand=hand)
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
    """Redo an item's or prop's sprites with new settings, no GPU (items: size, directions,
    style; props: height_people, directions, yaw, hand)."""
    if asset.kind == "prop":
        return render_prop(asset, project=project, progress=progress, **settings)
    if asset.kind == "item":
        return render_item(asset, project=project, progress=progress, **settings)
    raise ValueError(f"Not an item or prop: {asset.kind}")


# --------------------------------------------------------------------------- drawn props
# Buildings and most props in pixel-art games are drawn, not rendered: the GPU draws the prop
# directly as pixel art (SDXL + Pixel Art XL) at the project's camera angle, on a plain
# background, then the converter snaps it to the exact size and palette. A melty image-to-3D
# mesh posterised into colour bands (the first stone cottage) is what this replaces.

DRAWN_PROMPT = ("{prompt}, pixel art {view}, one single {what} on its own, isolated on a plain "
                "white background, centered, whole {what} in frame, clean simple shapes, crisp "
                "straight edges, readable silhouette, flat shading with clear light and shadow "
                "sides, limited palette, no ground, no scenery")
DRAWN_NEGATIVE = ("photo, realistic, 3d render, blurry, smooth gradient, noise, jpeg artifacts, "
                  "scenery, landscape, sky, grass field, trees in background, text, watermark, "
                  "multiple buildings, several views, cropped, out of frame, melted, distorted, "
                  "front view, flat facade, elevation drawing, straight-on, symmetrical front, "
                  + SHEET_NEGATIVE)
SIDE_WORDS = {"S": "front", "E": "right side", "N": "back", "W": "left side"}


def view_words(st: Style) -> str:
    """The camera, in words the image model knows."""
    e = st.elevation
    if e < 10:
        return "side view"
    if e < 40:
        return ("isometric three-quarter view from slightly above, seen from a corner, two sides "
                "visible, angled 45 degrees")
    return "top-down three-quarter view from above, seen from a corner, roof and two sides visible"


def drawn_prompt(prompt: str, st: Style, what: str | None = None) -> tuple[str, str]:
    """(positive, negative) for a drawn prop / building at the project's camera, house style
    appended (kind "prop")."""
    what = what or _what(prompt)
    return with_style(DRAWN_PROMPT.format(view=view_words(st), prompt=prompt, what=what),
                      DRAWN_NEGATIVE, st, "prop")


def _what(text: str) -> str:
    t = text.lower()
    return "building" if any(w in t for w in ("house", "cottage", "hut", "shop", "inn", "tower",
                                              "castle", "church", "barn", "mill", "tavern", "temple",
                                              "building", "cabin", "manor", "fort")) else "object"


def drop_ground(rgba: np.ndarray, band: float = 0.12) -> np.ndarray:
    """Remove the flat ground slab / drop shadow drawings like to put under a prop (asked for
    "no ground", the model still draws one): one flat colourless colour filling most of the
    bottom band, cut where it joins the bottom (textured stone walls are never one flat colour)."""
    import cv2
    a = rgba[..., 3] >= 128
    ys = np.nonzero(a)[0]
    if not len(ys):
        return rgba
    y0 = int(ys.max() - band * (ys.max() - ys.min()))
    rgb = rgba[..., :3].astype(np.int16)
    bottom = rgb[y0:][a[y0:]]
    if not len(bottom):
        return rgba
    grey = bottom[(bottom.max(1) - bottom.min(1)) < 12]
    if len(grey) < 0.35 * len(bottom):
        return rgba
    q = (grey // 6).astype(np.int32)
    keys, counts = np.unique(q[:, 0] * 10000 + q[:, 1] * 100 + q[:, 2], return_counts=True)
    k = keys[counts.argmax()]
    mode = np.array([k // 10000, (k // 100) % 100, k % 100]) * 6 + 3
    flat = (a & (np.abs(rgb - mode).max(-1) <= 8)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(flat, connectivity=8)
    out = rgba.copy()
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] - 1 >= y0:
            out[lab == i, 3] = 0
    return main_subject(out)


def drawn_sprite(picture, st: Style, target_h: float):
    """A drawn picture (white background) -> the prop sprite: cut out, the main piece only,
    exactly target_h px tall in the project's palette and outline, specks cleaned."""
    from .pipeline import convert, isolate, load_image
    from .pixelclean import clean_sprites
    s0 = style_settings(st, 64, 64)
    rgba = drop_ground(main_subject(isolate(load_image(picture), s0)))
    ys, xs = np.nonzero(rgba[..., 3] >= 128)
    if not len(xs):
        raise ValueError("The drawing came out empty after cutting out the background.")
    rgba = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h = max(8, round(target_h))
    w = max(4, round(h * rgba.shape[1] / rgba.shape[0]))
    s = style_settings(st, w, h, anchor="bottom")
    s.remove_background = False
    from .normals import from_drawing
    sprite = convert(rgba, s)
    sprite = clean_sprites([sprite], outline=s.outline_style if s.outline else None)[0]
    sprite.normal = from_drawing(sprite.rgba)               # lights still shade a drawn prop
    return sprite


def draw_prop(name: str, prompt: str, *, height_people: float | None = None, directions: int = 1,
              variations: int = 3, project: Project | None = None, progress=None, client=None,
              seed: int | None = None, asset: Asset | None = None) -> Asset:
    """A prop or building drawn as pixel art on the GPU (the default for props from words).

    directions 1: `variations` drawings to pick from (drawn_v1.png… -> sprite_v1.png…; the first
    is used, choose_drawn() picks another). directions 4: the S/E/N/W sides are drawn over the
    prop's 3D model seen from that side (model.glb, built from the first drawing if missing),
    so the sides agree while each is drawn clean (img2img with the pixel-art LoRA)."""
    from .generate import generate_images
    from .subjectcheck import generate_single
    say = _say(progress)
    proj = _project(project)
    st = proj.style
    g_height, _ = guess_prop(f"{name} {prompt}")
    hp = float(height_people or g_height)
    if asset is None:
        asset = new_asset("prop", name, project=proj, prompt=prompt, source="drawn",
                          height_people=hp, directions=directions, look="drawn")
    f = asset.folder
    what = _what(f"{name} {prompt}")
    text, drawn_negative = drawn_prompt(prompt, st, what)
    target = hp * st.person_height
    if directions == 1 or not (f / "model.glb").exists() and directions > 1:
        say(f"Drawing {name} on your GPU ({variations} versions)…")
        picks = generate_single(text, drawn_negative, client=client, seed=seed, count=variations,
                                aspect=1.0, say=say, init=shape_for(f"{name} {prompt}"),
                                init_words=f"{name} {prompt}",
                                **single_options("prop"))                     # best first
        pics = [(p.image, p.seed) for p in picks]
        asset.extra["checks"] = [p.verdict.summary() for p in picks]
        sprites = []
        for i, (img, used) in enumerate(pics, 1):
            img.save(f / f"drawn_v{i}.png")
            sp = drawn_sprite(img, st, target)
            Image.fromarray(sp.rgba).save(f / f"sprite_v{i}.png")
            sprites.append(sp)
        asset.extra["seeds"] = [used for _, used in pics]
        if directions == 1 and what == "building" and st.elevation >= 10:
            # image models draw buildings front-on; the project looks from a corner: draw the
            # corner view over the building's 3D shape seen from there (like the 4 sides)
            if not (f / "model.glb").exists():
                say("Building its 3D shape from the drawing, to draw it from the corner…")
                from .model3d import prompt_to_model
                prompt_to_model(None, f / "model.glb", picture=f / "drawn_v1.png", client=client,
                                subject="object", views="front", progress=lambda s_, t: say(t))
            from .turnaround import render_model, union_crop
            view = union_crop([render_model(f / "model.glb", 8, st.elevation, size=768,
                                            light=0.5)[1]])[0]                  # SE corner
            init = Image.new("RGB", (view.shape[1] + 80, view.shape[0] + 80), "white")
            init.paste(Image.fromarray(view), (40, 40), Image.fromarray(view))
            say(f"Drawing {name} from the corner…")
            (img, _), = generate_images(text, drawn_negative, client=client,
                                        seed=asset.extra["seeds"][0], count=1, init_image=init,
                                        denoise=0.7)
            img.save(f / "drawn_corner.png")
            corner = drawn_sprite(img, st, target)
            Image.fromarray(corner.rgba).save(f / "sprite_corner.png")
            sprites.insert(0, corner)
        if directions == 1:
            write_world([sprites[0]], f / "sheet.png")
            make_thumb(sprites[0].rgba, f / "thumb.png")
            asset.extra.update(height_people=hp, directions=1, look="drawn", chosen=1)
            asset.save()
            say(f"{name}: drawn. Files: {f}")
            return asset
        if not (f / "model.glb").exists():
            say("Building its 3D shape from the drawing, to draw the other sides over…")
            from .model3d import prompt_to_model
            prompt_to_model(None, f / "model.glb", picture=f / "drawn_v1.png", client=client,
                            subject="object", views="front", progress=lambda s, t: say(t))
    # 4 sides drawn over the model seen from each side
    from .turnaround import direction_names, render_model, union_crop
    sides = 2 if directions == 2 else 4                   # 2: E, W (side-scrollers)
    names = direction_names(sides)
    views = union_crop(render_model(f / "model.glb", sides, st.elevation, size=768, light=0.5))
    base_seed = (asset.extra.get("seeds") or [seed or 1234])[0]
    sprites = []
    for d, view in zip(names, views):
        init = Image.new("RGB", (view.shape[1] + 80, view.shape[0] + 80), "white")
        init.paste(Image.fromarray(view), (40, 40), Image.fromarray(view))
        say(f"Drawing the {SIDE_WORDS.get(d, d)} of {name}…")
        side = text.replace(", seen from a corner, two sides visible, angled 45 degrees", "")
        side = side.replace(f"pixel art {view_words(st)}",
                            f"pixel art {view_words(st)}, {SIDE_WORDS.get(d, d)} of the {what}")
        (img, _), = generate_images(side, drawn_negative, client=client, seed=base_seed, count=1,
                                    init_image=init, denoise=0.7)
        img.save(f / f"drawn_{d}.png")
        sprites.append(drawn_sprite(img, st, target))
    write_world(same_size(sprites), f / "sheet.png")
    make_thumb(sprites[0].rgba, f / "thumb.png")
    asset.extra.update(height_people=hp, directions=sides, look="drawn")
    asset.save()
    say(f"{name}: drawn from {sides} sides. Files: {f}")
    return asset


def same_size(sprites):
    """Pad drawn frames to one size, feet on one line, centred (sheets need equal frames)."""
    h = max(sp.rgba.shape[0] for sp in sprites)
    w = max(sp.rgba.shape[1] for sp in sprites)
    out = []
    for sp in sprites:
        sh, sw = sp.rgba.shape[:2]
        y, x = h - sh, (w - sw) // 2

        def pad(a, fill):
            big = np.full((h, w) + a.shape[2:], fill, a.dtype)
            big[y:y + sh, x:x + sw] = a
            return big
        sp.rgba = pad(sp.rgba, 0)
        sp.indices = pad(sp.indices, -1)
        if sp.normal is not None:
            sp.normal = pad(sp.normal, 0)
        out.append(sp)
    return out


def choose_drawn(asset: Asset, index: int, project: Project | None = None) -> Path:
    """Use another of the drawn versions (1-based) for a 1-direction drawn prop."""
    st = _project(project).style
    f = asset.folder
    pic = f / f"drawn_v{index}.png"
    if not pic.exists():
        raise ValueError(f"{asset.name} has no drawing number {index}")
    sp = drawn_sprite(pic, st, float(asset.extra.get("height_people", 1.0)) * st.person_height)
    write_world([sp], f / "sheet.png")
    make_thumb(sp.rgba, f / "thumb.png")
    asset.extra["chosen"] = index
    asset.save()
    return f / "sheet.png"

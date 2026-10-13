"""UI kit: procedural pixel-art game UI in the project's house style (no GPU needed).

    <project>/ui/<slug>/            (asset kind "ui")
        asset.json        theme, scale, icons ("drawn" | "gpu") in extra
        kit.json          every piece: sheet rect, file, 9-slice margins (left/top/right/bottom,
                          null = a plain sprite), group; text colours; font; the demo layout
        sheet.png         every piece packed (at the kit scale)
        pieces/<name>.png each piece alone (what the Godot / Unity exports use)
        fonts/pixelkit_<s>x.fnt + .png   the pixel font, ASCII 32-126, BMFont text format
        preview.png       a mock game screen made of the pieces
        icons_gpu/        icons drawn by the GPU (optional; they replace the drawn ones)
        thumb.png

Pieces are drawn at 1x and enlarged by a whole number (`scale`, default from Style.tile:
16-32 px tiles 1x, 48-64 2x...). THEMES give the materials (frame, fill, button face, accent)
and the corner/ornament style; the Style gives the outline (selout = each material's darkest,
solid = one near-black, none = no dark edge), the shading (soft = bevel + second step + shine,
cel = hard light/dark, flat = none), the detail (chunky = no ornaments, fine = extra trim) and
the palette (a fixed / locked one snaps every colour). Nine-slice pieces keep corners inside
their margins and edges uniform along the stretch, so engines stretching them (Godot
StyleBoxTexture, Unity Sliced) look like the preview.

The asset kind "ui" is in assets.KINDS (folder <project>/ui); register_kind is a no-op guard.
"""
from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from . import assets
from .cards import (GLYPHS, FONT_CAP, FONT_ROWS, LINE_HEIGHT, SPACE_W, PixelFont, _blit, _dilate,
                    _erode, _rgb, _shift, circle_mask, heart_mask, poly_mask, rrect_mask,
                    shield_mask, snap_image)

KIND = "ui"


def register_kind() -> None:
    """Make sure "ui" is an asset kind with the folder "ui" (assets.py lists it already)."""
    if KIND not in assets.KINDS:
        assets.KINDS = (*assets.KINDS, KIND)
    assets.FOLDERS[KIND] = "ui"


register_kind()

# --------------------------------------------------------------------------- themes

# ramps dark -> light: outline, dark, mid-dark, base, light, shine
THEMES = {
    "wood": dict(frame=["1c0f08", "4a2914", "6b3f1f", "8f5a2d", "b47a40", "dca468"],
                 fill=["120a06", "2e1c10", "3e2716", "4e321c", "644226", "7c5432"],
                 face=["1c0f08", "5a3418", "7c4c24", "a06a34", "c48c4c", "ecc088"],
                 accent=["1a1004", "5a3c0c", "94680c", "c8961c", "ecc444", "fff0a0"],
                 ink="f4e4c4", ink_shadow="1c0f08", label="fff4dc", label_shadow="3a1e0c",
                 corner="round", ornament="grain"),
    "stone": dict(frame=["101014", "2e2e36", "4a4a54", "66666f", "8a8a94", "b4b4bc"],
                  fill=["0c0c10", "1a1a20", "24242c", "2e2e38", "3a3a46", "4a4a58"],
                  face=["101014", "3a3a44", "545460", "70707c", "9696a2", "c4c4cc"],
                  accent=["0a1a10", "1a4a2c", "2a7a44", "3eaa5c", "6cd884", "c4ffd0"],
                  ink="e8e8f0", ink_shadow="101014", label="ffffff", label_shadow="24242c",
                  corner="square", ornament="blocks"),
    "parchment": dict(frame=["2a160a", "5c3418", "844c22", "a8682e", "c88c48", "e8b46c"],
                      fill=["5c4a30", "b09a70", "cdb88c", "e2d0a4", "efe2bc", "fbf4dc"],
                      face=["2a0c08", "6a1a12", "902818", "b43c20", "d8643c", "f8a070"],
                      accent=["2a160a", "6a4010", "9a6418", "c8902c", "e8bc54", "fff0a8"],
                      ink="2a1a0e", ink_shadow="", label="fbf0dc", label_shadow="4a120c",
                      corner="scroll", ornament="curl"),
    "metal": dict(frame=["0e1218", "2a323e", "465262", "647284", "8e9cb0", "d4dce8"],
                  fill=["080a10", "141a24", "1c2430", "242e3c", "2e3a4a", "3c4a5c"],
                  face=["0e1218", "34404e", "4e5c6e", "6e7e92", "98a8bc", "dce6f2"],
                  accent=["1a0e04", "5a2c08", "9a5010", "d07a1c", "f4a840", "ffe0a0"],
                  ink="e4ecf4", ink_shadow="0e1218", label="f4f8ff", label_shadow="1c2430",
                  corner="square", ornament="rivet"),
    "crystal": dict(frame=["100826", "2a1660", "44289a", "6440cc", "9070ec", "dcd0ff"],
                    fill=["080414", "120a28", "1a1038", "22164a", "2c1e5c", "3a2872"],
                    face=["100826", "2e1a6a", "48309e", "6a4ed0", "9a84f0", "e8e0ff"],
                    accent=["04161e", "0c3a50", "147088", "24a8c4", "5ce0f0", "d0fcff"],
                    ink="ece4ff", ink_shadow="100826", label="ffffff", label_shadow="2a1660",
                    corner="chamfer", ornament="gem"),
    "sci-fi": dict(frame=["020a10", "0a2430", "10485a", "187888", "2cb8c8", "a0f4ff"],
                   fill=["02060a", "060e14", "0a161e", "0e1e28", "142834", "1c3444"],
                   face=["020a10", "0c2a38", "124256", "1a6074", "2a90a4", "8ae8f8"],
                   accent=["1a0408", "5a0c1c", "9a1830", "d42c48", "f4607c", "ffc4d0"],
                   ink="c8f8ff", ink_shadow="020a10", label="e8ffff", label_shadow="0a2430",
                   corner="chamfer", ornament="glow"),
}
THEME_NAMES = tuple(THEMES)
BARS = {"health": ["2a0606", "6a1010", "a81c1c", "dc3a2c", "f6806a", "ffd8cc"],
        "mana": ["060c2e", "10227a", "1c3cba", "2c64e8", "64a0ff", "d0e4ff"],
        "xp": ["06200a", "10461a", "1c7a2c", "2cae40", "68dc6c", "d0ffc8"]}
GOLD = ["2a1604", "6a3e08", "a8680c", "dc9c1c", "f8d04c", "fff8c0"]
STEEL = ["16161c", "3a3e48", "5e6470", "8a909c", "bcc2cc", "f0f4f8"]
GEM = ["0a1838", "1a3c8c", "2c66cc", "4c94f0", "90c8ff", "ffffff"]
GLASS = ["1a2430", "3e5868", "6a8ea0", "9cc4d4", "cfeef8", "ffffff"]
LEATHER = ["1c0f08", "4a2914", "6b3f1f", "8f5a2d", "b47a40", "dca468"]
OUTLINE_SOLID = "0c0a10"
ICON = 16                      # icon size at 1x
ICONS = ("heart", "heart_half", "heart_empty", "coin", "gem", "key", "sword", "shield",
         "potion", "gear", "close", "check", "arrow_left", "arrow_right", "arrow_up",
         "arrow_down")
SCALES = (1, 2, 3, 4)
SCREEN = (480, 270)            # the mock game screen at 1x (1080p / 4)
SEGMENTS = 8

# --------------------------------------------------------------------------- font

# glyphs the card font lacks, so the kit font covers all of ASCII 32-126
_EXTRA_SRC = {
    "$": "..#.. .#### #.#.. .###. ..#.# ####. ..#..",
    "@": ".###. #...# #.### #.#.# #.### #.... .###.",
    "\\": "#.. #.. .#. .#. .#. ..# ..#",
    "^": ".#. #.# ... ... ... ... ...",
    "`": "#. .# .. .. .. .. ..",
    "{": "..# .#. .#. #.. .#. .#. ..#",
    "|": "# # # # # # #",
    "}": "#.. .#. .#. ..# .#. .#. #..",
    "~": "..... ..... .#... #.#.# ...#. ..... .....",
}


def _extra_glyphs() -> dict[str, np.ndarray]:
    out = {}
    for ch, src in _EXTRA_SRC.items():
        rows = src.split()
        g = np.zeros((FONT_ROWS, max(len(r) for r in rows)), bool)
        for y, r in enumerate(rows):
            g[y, :len(r)] = [c == "#" for c in r]
        out[ch] = g
    return out


KIT_GLYPHS = {**GLYPHS, **_extra_glyphs()}
FONT = PixelFont(KIT_GLYPHS)
FONT_FACE = "PixelKit"


def font_name(scale: int) -> str:
    return f"pixelkit_{scale}x"


def font_atlas(scale: int = 1) -> tuple[np.ndarray, dict]:
    """White glyphs (ASCII 32-126) in one RGBA atlas + {char: (x, y, w, h, advance)}."""
    chars = [chr(c) for c in range(32, 127)]
    per_row = 16
    cell_h = FONT_ROWS * scale
    cell_w = max(FONT.glyph(c).shape[1] for c in chars) * scale + 1
    rows = math.ceil(len(chars) / per_row)
    atlas = np.zeros((rows * (cell_h + 1), per_row * cell_w, 4), np.uint8)
    metrics = {}
    for i, c in enumerate(chars):
        r, k = divmod(i, per_row)
        x, y = k * cell_w, r * (cell_h + 1)
        if c == " ":
            m = np.zeros((cell_h, (SPACE_W - 1) * scale), bool)
        else:
            m = FONT.mask(c, scale)
        _blit(atlas, m, x, y, "ffffff")
        metrics[c] = (x, y, m.shape[1], m.shape[0], (FONT.glyph(c).shape[1] + 1) * scale)
    return atlas, metrics


def write_font(out_dir, scale: int = 1) -> list[Path]:
    """fonts/<pixelkit_Nx>.fnt (BMFont text: Godot imports it as a FontFile, Unity as a
    TextAsset the kit's editor script turns into a Font) + its .png atlas."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    atlas, metrics = font_atlas(scale)
    name = font_name(scale)
    png = out_dir / f"{name}.png"
    Image.fromarray(atlas, "RGBA").save(png)
    lines = [f'info face="{FONT_FACE}" size={LINE_HEIGHT * scale} bold=0 italic=0 charset="" '
             f'unicode=1 stretchH=100 smooth=0 aa=1 padding=0,0,0,0 spacing=1,1 outline=0',
             f"common lineHeight={LINE_HEIGHT * scale} base={FONT_CAP * scale} "
             f"scaleW={atlas.shape[1]} scaleH={atlas.shape[0]} pages=1 packed=0",
             f'page id=0 file="{png.name}"', f"chars count={len(metrics)}"]
    for c, (x, y, w, h, adv) in metrics.items():
        lines.append(f"char id={ord(c)} x={x} y={y} width={w} height={h} xoffset=0 yoffset=0 "
                     f"xadvance={adv} page=0 chnl=15")
    fnt = out_dir / f"{name}.fnt"
    fnt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [fnt, png]


# --------------------------------------------------------------------------- look

@dataclass
class Look:
    """The bits of the house style the kit follows."""
    outline: str = "selout"        # selout | solid | none
    shading: str = "soft"          # soft | cel | flat
    detail: str = "normal"         # chunky | normal | fine
    palette: np.ndarray | None = field(default=None, repr=False)


def look_of(style=None) -> Look:
    from .effects import project_palette
    st = style if style is not None else assets.project_style()
    return Look(getattr(st, "outline", "selout") or "selout", getattr(st, "shading", "soft") or "soft",
                getattr(st, "detail", "normal") or "normal", project_palette(st))


def default_scale(style=None) -> int:
    """Whole-number UI scale for a project: 16-32 px tiles 1x, 48-64 2x, bigger 3x+."""
    st = style if style is not None else assets.project_style()
    return max(1, min(4, round(int(getattr(st, "tile", 32) or 32) / 40 + 0.3)))


def _ramp(hexes) -> list[np.ndarray]:
    return [_rgb(h) for h in hexes]


def _lighter(ramp: list) -> list:
    """The ramp one step brighter (hover)."""
    return [ramp[0], ramp[2], ramp[3], ramp[4], ramp[5], ramp[5]]


def _greyed(ramp: list) -> list:
    """Disabled: the ramp's lightness in low-contrast grey."""
    out = []
    for c in ramp:
        v = float(np.dot(c.astype(np.float32), (0.3, 0.59, 0.11)))
        v = 40 + v * 0.45
        out.append(np.array([v, v, v + 6], np.float32).clip(0, 255).astype(np.uint8))
    out[0] = ramp[0]
    return out


def _put(img: np.ndarray, mask: np.ndarray, colour) -> None:
    img[mask, :3] = colour if isinstance(colour, np.ndarray) else _rgb(colour)
    img[mask, 3] = 255


def _edge_colour(ramp, look: Look):
    return {"solid": _rgb(OUTLINE_SOLID), "none": ramp[1]}.get(look.outline, ramp[0])


def paint_box(img: np.ndarray, mask: np.ndarray, ramp, look: Look, sunken: bool = False,
              edge: bool = True) -> None:
    """Fill a mask as a pixel-art box: 1px edge (the look's outline), base ramp[3], then the
    look's shading lit from the top-left (sunken: from the bottom-right)."""
    m = mask.astype(bool)
    if not m.any():
        return
    inner = _erode(m, 1) if edge else m
    if edge:
        _put(img, m & ~inner, _edge_colour(ramp, look))
    _put(img, inner, ramp[2] if sunken else ramp[3])
    if look.shading == "flat":
        return
    ul = inner & ~_shift(inner, 1, 1)
    dr = inner & ~_shift(inner, -1, -1)
    hi, lo = (ramp[1], ramp[4]) if sunken else (ramp[4], ramp[1])
    _put(img, dr & ~ul, lo)
    _put(img, ul & ~dr, hi)
    _put(img, ul & dr, ramp[2])
    if look.shading == "soft":
        in2 = inner & ~ul & ~dr
        if sunken:
            _put(img, in2 & ~_shift(in2, 1, 1) & ~(in2 & ~_shift(in2, -1, -1)), ramp[1])
        else:
            _put(img, in2 & ~_shift(in2, -1, -1) & ~(in2 & ~_shift(in2, 1, 1)), ramp[2])
            corner = inner & ~_shift(inner, 1, 0) & ~_shift(inner, 0, 1)
            _put(img, corner, ramp[5])


def _outer(H: int, W: int, corner: str, x0=0, y0=0, x1=None, y1=None) -> np.ndarray:
    """A box mask with the theme's corner shape."""
    x1 = W if x1 is None else x1
    y1 = H if y1 is None else y1
    if corner == "chamfer":
        c = min(3, (x1 - x0) // 4, (y1 - y0) // 4)
        return poly_mask(H, W, [(x0 + c, y0), (x1 - c, y0), (x1, y0 + c), (x1, y1 - c),
                                (x1 - c, y1), (x0 + c, y1), (x0, y1 - c), (x0, y0 + c)]) \
            & rrect_mask(H, W, x0, y0, x1, y1)
    r = {"round": 3, "scroll": 2, "square": 1}.get(corner, 1)
    r = min(r, (x1 - x0) / 3, (y1 - y0) / 3)
    return rrect_mask(H, W, x0, y0, x1, y1, r)


# --------------------------------------------------------------------------- pieces

@dataclass
class Piece:
    rgba: np.ndarray
    margins: dict | None = None    # 9-slice {left, top, right, bottom} at 1x; None = a sprite
    group: str = "misc"
    content: dict | None = None    # engine content margins where they differ (pressed button)


def _m(left, top=None, right=None, bottom=None) -> dict:
    top = left if top is None else top
    return {"left": left, "top": top, "right": left if right is None else right,
            "bottom": top if bottom is None else bottom}


BORDER = 4                      # frame thickness at 1x


def _corner_pts(N: int, H: int, k: float):
    return ((k, k), (N - 1 - k, k), (k, H - 1 - k), (N - 1 - k, H - 1 - k))


def _ornaments(img: np.ndarray, th: dict, look: Look, b: int, kind: str = "panel") -> None:
    """Corner / edge decoration by theme; chunky detail = none."""
    if look.detail == "chunky":
        return
    H, W = img.shape[:2]
    frame, accent = _ramp(th["frame"]), _ramp(th["accent"])
    orn = th["ornament"]
    if orn == "grain":                  # a plank seam along each band + corner nails
        mid = b // 2
        band = np.zeros((H, W), bool)
        band[mid, b:W - b] = True
        band[H - 1 - mid, b:W - b] = True
        band[b:H - b, mid] = True
        band[b:H - b, W - 1 - mid] = True
        _put(img, band & (img[..., 3] > 0), frame[2])
        for x, y in _corner_pts(W, H, mid):
            _put(img, circle_mask(H, W, x + 0.5, y + 0.5, 0.6), accent[4])
    elif orn == "rivet":
        for x, y in _corner_pts(W, H, b / 2 - 0.5):
            m = circle_mask(H, W, x + 0.5, y + 0.5, 1.1)
            _put(img, m, frame[1])
            _put(img, m & ~_shift(m, 1, 1), frame[5])
    elif orn == "blocks":               # corner stones
        s = b + 1
        for x0, y0 in ((0, 0), (W - s, 0), (0, H - s), (W - s, H - s)):
            m = rrect_mask(H, W, x0, y0, x0 + s, y0 + s, 0)
            paint_box(img, m, _ramp(th["face"]), look)
    elif orn == "gem":
        r = b / 2 + 0.3
        for x, y in _corner_pts(W, H, b / 2 - 0.5):
            cx, cy = x + 0.5, y + 0.5
            m = poly_mask(H, W, [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)])
            paint_box(img, m | circle_mask(H, W, cx, cy, 0.8), accent, look)
    elif orn == "glow":                 # neon ticks beside the chamfers
        for x, y, dx, dy in ((b, 1, 1, 0), (W - 1 - b, 1, -1, 0), (b, H - 2, 1, 0),
                             (W - 1 - b, H - 2, -1, 0)):
            for k in range(2):
                img[y, x + dx * k, :3] = accent[4]
    elif orn == "curl":                 # rolled scroll ends at the corners
        for x, y in _corner_pts(W, H, 1.5):
            m = circle_mask(H, W, x + 0.5, y + 0.5, 2.1)
            paint_box(img, m, frame, look)
            _put(img, circle_mask(H, W, x + 0.5, y + 0.5, 0.6), frame[1])


def _trim(img: np.ndarray, th: dict, inset: int, colour) -> None:
    H, W = img.shape[:2]
    ring = rrect_mask(H, W, inset, inset, W - inset, H - inset) & \
        ~rrect_mask(H, W, inset + 1, inset + 1, W - inset - 1, H - inset - 1)
    _put(img, ring, colour)


def panel(th: dict, look: Look, fill: str = "fill", b: int = BORDER, centre: int = 4,
          trim: bool | None = None) -> Piece:
    """Frame band + sunken fill; margins b+3 so corners and the fill's two bevel steps stay
    out of the stretched middle."""
    m = b + 3
    N = 2 * m + centre
    img = np.zeros((N, N, 4), np.uint8)
    paint_box(img, _outer(N, N, th["corner"]), _ramp(th["frame"]), look)
    paint_box(img, rrect_mask(N, N, b, b, N - b, N - b, 1 if th["corner"] == "round" else 0),
              _ramp(th[fill]), look, sunken=True)
    if look.shading != "flat" and th["ornament"] == "glow":
        _trim(img, th, b, _ramp(th["frame"])[4])
    if trim or (trim is None and look.detail == "fine"):
        _trim(img, th, b + 1, _ramp(th["accent"])[2])
    _ornaments(img, th, look, b)
    return Piece(img, _m(m), "panel")


def window(th: dict, look: Look, b: int = BORDER, centre: int = 4) -> Piece:
    """A panel with a title bar across the top (the title goes in the bar)."""
    bar = FONT_CAP + 5
    top = b + bar + 3
    m = b + 3
    W, H = 2 * m + centre, top + centre + m
    img = np.zeros((H, W, 4), np.uint8)
    paint_box(img, _outer(H, W, th["corner"]), _ramp(th["frame"]), look)
    paint_box(img, rrect_mask(H, W, b - 1, b - 1, W - b + 1, b + bar), _ramp(th["face"]), look)
    paint_box(img, rrect_mask(H, W, b, b + bar, W - b, H - b), _ramp(th["fill"]), look, sunken=True)
    if look.detail == "fine":
        ring = rrect_mask(H, W, b + 1, b + bar + 1, W - b - 1, H - b - 1) & \
            ~rrect_mask(H, W, b + 2, b + bar + 2, W - b - 2, H - b - 2)
        _put(img, ring, _ramp(th["accent"])[2])
    _ornaments(img, th, look, b)
    return Piece(img, _m(m, top, m, m), "panel")


def button(th: dict, look: Look, state: str = "normal") -> Piece:
    """normal / hover / pressed / disabled; 1px drop below unless pressed."""
    W, H = 12, 16
    face = _ramp(th["face"])
    if state == "hover":
        face = _lighter(face)
    elif state == "disabled":
        face = _greyed(face)
    img = np.zeros((H, W, 4), np.uint8)
    corner = th["corner"] if th["corner"] != "scroll" else "round"
    if state == "pressed":
        paint_box(img, _outer(H, W, corner, 0, 1, W, H), face, look, sunken=True)
        content = _m(4, 5, 4, 3)
    else:
        lip = _outer(H, W, corner, 0, 1, W, H)
        _put(img, lip, _edge_colour(face, look) if look.outline != "none" else face[0])
        paint_box(img, _outer(H, W, corner, 0, 0, W, H - 1), face, look)
        content = None
    return Piece(img, _m(4, 4, 4, 4), "button", content)


def tab(th: dict, look: Look, active: bool) -> Piece:
    """A tab, open at the bottom (it sits on a panel's top edge)."""
    W, H = 12, 13
    img = np.zeros((H + 4, W, 4), np.uint8)
    ramp = _ramp(th["face"]) if active else _ramp(th["frame"])[:1] + _ramp(th["frame"])[:-1]
    y0 = 0 if active else 2
    paint_box(img, _outer(H + 4, W, th["corner"] if th["corner"] != "scroll" else "round",
                          0, y0, W, H + 4), ramp, look)
    return Piece(np.ascontiguousarray(img[:H]), _m(4, 4, 4, 2), "tab")


def inset(th: dict, look: Look, n: int = 10) -> Piece:
    """A sunken field (text input, tooltip)."""
    img = np.zeros((n, n, 4), np.uint8)
    paint_box(img, rrect_mask(n, n, 0, 0, n, n, 1), _ramp(th["fill"]), look, sunken=True)
    return Piece(img, _m(3), "panel")


def _icon_canvas(n: int):
    return np.zeros((n, n, 4), np.uint8)


def checkbox(th: dict, look: Look, checked: bool) -> Piece:
    n = 11
    img = _icon_canvas(n)
    paint_box(img, rrect_mask(n, n, 0, 0, n, n, 1), _ramp(th["frame"]), look)
    paint_box(img, rrect_mask(n, n, 2, 2, n - 2, n - 2), _ramp(th["fill"]), look, sunken=True)
    if checked:
        tick = _line(n, n, 3.0, 5.6, 4.9, 7.5, 1.3) | _line(n, n, 4.9, 7.5, 8.2, 3.2, 1.3)
        _shade(img, tick, _ramp(th["accent"]), look)
    return Piece(img, None, "control")


def slider(th: dict, look: Look) -> dict[str, Piece]:
    out = {}
    W, H = 12, 7
    img = np.zeros((H, W, 4), np.uint8)
    paint_box(img, rrect_mask(H, W, 0, 0, W, H, 1), _ramp(th["fill"]), look, sunken=True)
    out["slider_track"] = Piece(img, _m(3), "control")
    img = np.zeros((H, W, 4), np.uint8)
    paint_box(img, rrect_mask(H, W, 0, 0, W, H, 1), _ramp(th["accent"]), look)
    out["slider_fill"] = Piece(img, _m(3), "control")
    for name, ramp in (("slider_knob", _ramp(th["face"])), ("slider_knob_hover", _lighter(_ramp(th["face"])))):
        kw, kh = 7, 11
        img = np.zeros((kh, kw, 4), np.uint8)
        paint_box(img, _outer(kh, kw, "round" if th["corner"] != "chamfer" else "chamfer"), ramp, look)
        img[3:kh - 3, kw // 2, :3] = _ramp(th["accent"])[3]
        out[name] = Piece(img, None, "control")
    return out


def scrollbar(th: dict, look: Look) -> dict[str, Piece]:
    W, H = 9, 12
    img = np.zeros((H, W, 4), np.uint8)
    paint_box(img, rrect_mask(H, W, 0, 0, W, H, 1), _ramp(th["fill"]), look, sunken=True)
    out = {"scroll_track": Piece(img, _m(3), "control")}
    for name, ramp in (("scroll_thumb", _ramp(th["face"])), ("scroll_thumb_hover", _lighter(_ramp(th["face"])))):
        img = np.zeros((H, W, 4), np.uint8)
        paint_box(img, rrect_mask(H, W, 1, 0, W - 1, H, 1), ramp, look)
        out[name] = Piece(img, _m(3), "control")
    return out


BAR_H = 9


def _bar_rows(look: Look) -> list[int]:
    return {"soft": [4, 3, 3, 2, 1], "cel": [4, 3, 3, 1, 1], "flat": [3, 3, 3, 3, 2]}[look.shading]


def bar_frame(th: dict, look: Look, W: int = 16, segments: int = 0) -> Piece:
    img = np.zeros((BAR_H, W, 4), np.uint8)
    frame = _ramp(th["frame"])
    paint_box(img, rrect_mask(BAR_H, W, 0, 0, W, BAR_H, 1), frame, look)
    _put(img, rrect_mask(BAR_H, W, 2, 2, W - 2, BAR_H - 2), _ramp(th["fill"])[0])
    if segments:
        for x in _dividers(W, segments):
            img[2:BAR_H - 2, x, :3] = frame[1]
    return Piece(img, None if segments else _m(3), "bar")


def _dividers(W: int, segments: int) -> list[int]:
    inner = W - 4
    return [2 + round(inner * k / segments) for k in range(1, segments)]


def bar_fill(kind: str, look: Look, W: int = 16, segments: int = 0) -> Piece:
    """The fill, padded like the frame (Godot draws the fill over the frame's whole rect)."""
    ramp = _ramp(BARS[kind])
    img = np.zeros((BAR_H, W, 4), np.uint8)
    for k, idx in enumerate(_bar_rows(look)):
        img[2 + k, 2:W - 2, :3] = ramp[idx]
        img[2 + k, 2:W - 2, 3] = 255
    if look.shading == "soft":
        img[3, 3:min(W - 3, 6), :3] = ramp[5]
    if segments:
        for x in _dividers(W, segments):
            img[2:BAR_H - 2, x] = 0
    return Piece(img, None if segments else _m(3), "bar")


def slot(th: dict, look: Look, selected: bool = False) -> Piece:
    n = ICON + 4
    img = np.zeros((n, n, 4), np.uint8)
    paint_box(img, rrect_mask(n, n, 0, 0, n, n, 1), _ramp(th["fill"]), look, sunken=True)
    if selected:
        acc = _ramp(th["accent"])
        ring = rrect_mask(n, n, 0, 0, n, n, 1) & ~rrect_mask(n, n, 2, 2, n - 2, n - 2)
        _put(img, ring, acc[4])
        _put(img, rrect_mask(n, n, 0, 0, n, n, 1) & ~rrect_mask(n, n, 1, 1, n - 1, n - 1, 1), acc[1])
    return Piece(img, _m(4), "slot")


def nameplate(th: dict, look: Look) -> Piece:
    W, H = 12, FONT_CAP + 7
    img = np.zeros((H, W, 4), np.uint8)
    paint_box(img, _outer(H, W, th["corner"] if th["corner"] != "scroll" else "round"),
              _ramp(th["face"]), look)
    return Piece(img, _m(4, 4, 4, 3), "panel")


def next_arrow(th: dict, look: Look) -> Piece:
    W, H = 9, 7
    img = np.zeros((H, W, 4), np.uint8)
    m = poly_mask(H, W, [(0.5, 0.5), (8.5, 0.5), (4.5, 5.8)])
    _shade(img, m, _ramp(th["accent"]), look, outline=True)
    return Piece(img, None, "panel")


# --------------------------------------------------------------------------- icons + cursors

def _line(H: int, W: int, x0, y0, x1, y1, width: float) -> np.ndarray:
    """Pixels whose centres lie within width/2 of the segment."""
    yy, xx = np.mgrid[0:H, 0:W] + 0.5
    dx, dy = x1 - x0, y1 - y0
    t = np.clip(((xx - x0) * dx + (yy - y0) * dy) / max(dx * dx + dy * dy, 1e-9), 0, 1)
    return (xx - x0 - t * dx) ** 2 + (yy - y0 - t * dy) ** 2 <= (width / 2) ** 2


def _cross(m: np.ndarray) -> np.ndarray:
    return m | _shift(m, 1, 0) | _shift(m, -1, 0) | _shift(m, 0, 1) | _shift(m, 0, -1)


def _shade(img: np.ndarray, mask: np.ndarray, ramp, look: Look, outline: bool = False) -> None:
    """Thin-shape shading: base, top-left light, bottom-right dark (no edge); outline=True
    adds the look's outline OUTSIDE the shape."""
    m = mask.astype(bool)
    if outline and look.outline != "none":
        _put(img, _cross(m) & ~m, _edge_colour(ramp, look))
    _put(img, m, ramp[3])
    if look.shading == "flat":
        return
    lit = m & (~_shift(m, 1, 0) | ~_shift(m, 0, 1))
    dark = m & (~_shift(m, -1, 0) | ~_shift(m, 0, -1))
    _put(img, dark & ~lit, ramp[2] if look.shading == "soft" else ramp[1])
    _put(img, lit & ~dark, ramp[4])
    if look.shading == "soft":
        ys, xs = np.nonzero(lit & ~dark)
        if len(ys):
            k = np.argmin(ys + xs)
            img[ys[k], xs[k], :3] = ramp[5]
            img[ys[k], xs[k], 3] = 255


def _icon(parts, look: Look, n: int = ICON) -> np.ndarray:
    """parts = [(mask, ramp)], drawn in order, then one outline around them all coloured by
    the part each outline pixel touches (selout) or the look's colour."""
    img = _icon_canvas(n)
    union = np.zeros((n, n), bool)
    for m, ramp in parts:
        _shade(img, m, ramp, look)
        union |= m
    if look.outline != "none":
        ring = _cross(union) & ~union
        for m, ramp in parts:
            _put(img, ring & _cross(m), _edge_colour(ramp, look))
    return img


def draw_icon(name: str, th: dict, look: Look) -> np.ndarray:
    """One 16x16 icon (ICONS)."""
    n = ICON
    H = W = n
    red, gold, steel = _ramp(BARS["health"]), _ramp(GOLD), _ramp(STEEL)
    accent, face = _ramp(th["accent"]), _ramp(th["face"])
    if name.startswith("heart"):
        m = heart_mask(H, W, 8, 8.2, 6.6)
        if name == "heart":
            return _icon([(m, red)], look)
        empty = _ramp(["1a0a0e", "2e1418", "3e1c22", "4a2228", "5a2a30", "6a343a"])
        if name == "heart_empty":
            return _icon([(m, empty)], look)
        left = m & (np.mgrid[0:H, 0:W][1] < 8)
        return _icon([(m & ~left, empty), (left, red)], look)
    if name == "coin":
        m = circle_mask(H, W, 8, 8, 6.4)
        img = _icon([(m, gold)], look)
        inner = circle_mask(H, W, 8, 8, 4.2) & ~circle_mask(H, W, 8, 8, 3.2)
        _put(img, inner & _shift(inner, 1, 1), gold[2])
        _put(img, _line(H, W, 8, 5.2, 8, 10.8, 1.2), gold[5] if look.shading != "flat" else gold[4])
        return img
    if name == "gem":
        top = poly_mask(H, W, [(4.5, 3), (11.5, 3), (14.5, 6.5), (1.5, 6.5)])
        bot = poly_mask(H, W, [(1.5, 6.5), (14.5, 6.5), (8, 14.5)])
        gem = _ramp(GEM)
        img = _icon([(bot, gem), (top, [gem[0], gem[2], gem[3], gem[4], gem[5], gem[5]])], look)
        _put(img, _line(H, W, 5, 6.6, 8, 13.5, 0.9) & bot, gem[2])
        return img
    if name == "key":
        bow = circle_mask(H, W, 5, 5, 3.6) & ~circle_mask(H, W, 5, 5, 1.4)
        shaft = _line(H, W, 6.5, 6.5, 13.5, 13.5, 2.1)
        teeth = _line(H, W, 11.2, 11.2, 13.4, 9.0, 1.9) | _line(H, W, 9.2, 9.2, 10.8, 7.6, 1.9)
        return _icon([(shaft | teeth, gold), (bow, gold)], look)
    if name == "sword":
        blade = _line(H, W, 6.0, 10.0, 13.6, 2.4, 2.2) | _line(H, W, 13.0, 3.0, 14.4, 1.6, 1.2)
        guard = _line(H, W, 2.8, 8.6, 7.4, 13.2, 2.0)
        grip = _line(H, W, 4.4, 11.6, 2.4, 13.6, 1.9)
        pommel = circle_mask(H, W, 2.0, 14.0, 1.4)
        return _icon([(blade, steel), (grip, _ramp(LEATHER)), (guard, gold), (pommel, gold)], look)
    if name == "shield":
        m = shield_mask(H, W, 8, 7.6, 6.4)
        rim = m & ~_erode(m, 1)
        img = _icon([(m, face)], look)
        _put(img, rim & m, steel[3] if look.outline == "none" else steel[2])
        _put(img, rim & m & ~_shift(m, 1, 0), steel[4])
        _put(img, _line(H, W, 8, 3.2, 8, 12.5, 1.2) | _line(H, W, 4, 6.8, 12, 6.8, 1.2), accent[3])
        return img
    if name == "potion":
        body = circle_mask(H, W, 8, 10.2, 5.0)
        neck = rrect_mask(H, W, 6, 3, 10, 7)
        cork = rrect_mask(H, W, 5, 1, 11, 4, 0.6)
        liquid = body & (np.mgrid[0:H, 0:W][0] >= 8)
        glass = _ramp(GLASS)
        img = _icon([(body | neck, glass), (liquid, red), (cork, _ramp(LEATHER))], look)
        _put(img, _line(H, W, 5.2, 8.6, 5.2, 11.0, 1.0) & body, glass[5])
        return img
    if name == "gear":
        yy, xx = np.mgrid[0:H, 0:W] + 0.5
        ang = np.arctan2(yy - 8, xx - 8)
        rad = np.hypot(yy - 8, xx - 8)
        teeth = (np.cos(ang * 8) > 0.15) & (rad <= 7.2)
        m = ((rad <= 5.4) | teeth) & (rad > 2.0)
        return _icon([(m, steel)], look)
    if name in ("close", "check"):
        ramp = red if name == "close" else _ramp(BARS["xp"])
        if name == "close":
            m = _line(H, W, 3.5, 3.5, 12.5, 12.5, 2.6) | _line(H, W, 12.5, 3.5, 3.5, 12.5, 2.6)
        else:
            m = _line(H, W, 2.8, 8.4, 6.2, 11.8, 2.6) | _line(H, W, 6.2, 11.8, 13.4, 4.0, 2.6)
        return _icon([(m, ramp)], look)
    if name.startswith("arrow_"):
        pts = [(4.5, 2.5), (12.5, 8), (4.5, 13.5)]
        img = _icon([(poly_mask(H, W, pts), accent)], look)
        turn = {"arrow_right": 0, "arrow_down": 3, "arrow_left": 2, "arrow_up": 1}[name]
        return np.ascontiguousarray(np.rot90(img, turn)) if turn else img
    raise ValueError(f"Unknown icon '{name}' (use {', '.join(ICONS)})")


_CURSOR = (
    "o...........",
    "oo..........",
    "o4o.........",
    "o44o........",
    "o443o.......",
    "o4433o......",
    "o44333o.....",
    "o443333o....",
    "o4433333o...",
    "o443333ooo..",
    "o43o33o.....",
    "o3o.o33o....",
    "oo..o32o....",
    "o....o2o....",
    ".....oo.....",
)
_HAND = (
    "....oo......",
    "...o44o.....",
    "...o43o.....",
    "...o43o.....",
    "...o43ooo...",
    "...o43o43oo.",
    ".oo.o3o3o43o",
    "o44oo3o3o33o",
    "o433o333333o",
    ".o43333333o.",
    "..o3333332o.",
    "..o3333322o.",
    "...o33222o..",
    "...o22222o..",
    "...oooooo...",
)


def cursor(rows, look: Look, ramp=None) -> np.ndarray:
    ramp = ramp or _ramp(["0c0a10", "5a5a64", "a0a0aa", "dcdce4", "ffffff", "ffffff"])
    H, W = len(rows), len(rows[0])
    img = np.zeros((H, W, 4), np.uint8)
    edge = _rgb(OUTLINE_SOLID) if look.outline != "none" else ramp[1]
    for y, r in enumerate(rows):
        for x, c in enumerate(r):
            if c == ".":
                continue
            col = edge if c == "o" else ramp[int(c)]
            if look.shading == "flat" and c != "o":
                col = ramp[3]
            img[y, x, :3] = col
            img[y, x, 3] = 255
    return img


# --------------------------------------------------------------------------- the kit

@dataclass
class Kit:
    theme: str
    scale: int
    look: Look
    pieces: dict[str, Piece]
    colours: dict[str, str]

    def image(self, name: str) -> np.ndarray:
        """A piece at the kit scale."""
        return enlarge(self.pieces[name].rgba, self.scale)

    def margins(self, name: str) -> dict | None:
        m = self.pieces[name].margins
        return None if m is None else {k: v * self.scale for k, v in m.items()}

    def content(self, name: str) -> dict | None:
        m = self.pieces[name].content
        return None if m is None else {k: v * self.scale for k, v in m.items()}


def enlarge(rgba: np.ndarray, k: int) -> np.ndarray:
    return rgba if k == 1 else np.repeat(np.repeat(rgba, k, 0), k, 1)


def _colours(th: dict, palette) -> dict:
    face = _ramp(th["face"])
    cols = {"ink": th["ink"], "ink_shadow": th["ink_shadow"], "label": th["label"],
            "label_shadow": th["label_shadow"], "label_hover": th["label"],
            "label_disabled": "%02x%02x%02x" % tuple(int(v) for v in _greyed(face)[4]),
            "title": th["label"], "accent": th["accent"][4]}
    if palette is not None and len(palette):
        keys = [k for k, v in cols.items() if v]
        arr = np.array([[_rgb(cols[k]) for k in keys]], np.uint8)
        rgba = np.concatenate([arr, np.full(arr.shape[:2] + (1,), 255, np.uint8)], -1)
        snapped = snap_image(rgba, palette)[0, :, :3]
        cols.update({k: "%02x%02x%02x" % tuple(int(c) for c in v) for k, v in zip(keys, snapped)})
    return cols


def make_kit(theme: str = "wood", style=None, scale: int | None = None,
             icons: dict | None = None) -> Kit:
    """Every piece for a theme in the house style. icons: {name: RGBA} to use instead of the
    drawn ones (GPU icons); scale None = default_scale(style)."""
    if theme not in THEMES:
        raise ValueError(f"Unknown theme '{theme}' (use {', '.join(THEMES)})")
    look = look_of(style)
    scale = int(scale or default_scale(style))
    if scale not in SCALES:
        raise ValueError(f"scale must be one of {', '.join(map(str, SCALES))}")
    th = THEMES[theme]
    p: dict[str, Piece] = {}
    p["panel"] = panel(th, look)
    p["window"] = window(th, look)
    p["dialogue"] = panel(th, look, b=BORDER + 1, trim=look.detail != "chunky")
    p["nameplate"] = nameplate(th, look)
    p["next_arrow"] = next_arrow(th, look)
    p["inset"] = inset(th, look)
    for s in ("normal", "hover", "pressed", "disabled"):
        p[f"button_{s}"] = button(th, look, s)
    p["tab_active"] = tab(th, look, True)
    p["tab_inactive"] = tab(th, look, False)
    p["checkbox_off"] = checkbox(th, look, False)
    p["checkbox_on"] = checkbox(th, look, True)
    p.update(slider(th, look))
    p.update(scrollbar(th, look))
    p["bar_frame"] = bar_frame(th, look)
    seg_w = 4 + 8 * SEGMENTS
    p["bar_seg_frame"] = bar_frame(th, look, seg_w, SEGMENTS)
    for kind in BARS:
        p[f"bar_{kind}"] = bar_fill(kind, look)
        p[f"bar_seg_{kind}"] = bar_fill(kind, look, seg_w, SEGMENTS)
    p["slot"] = slot(th, look)
    p["slot_selected"] = slot(th, look, True)
    p["cursor"] = Piece(cursor(_CURSOR, look), None, "cursor")
    p["cursor_hand"] = Piece(cursor(_HAND, look), None, "cursor")
    for name in ICONS:
        img = icons.get(name) if icons else None
        if img is None:
            img = draw_icon(name, th, look)
        p[f"icon_{name}"] = Piece(np.ascontiguousarray(img), None, "icon")
    p["inventory"] = Piece(_inventory(p, 4, 3), None, "panel")
    for piece in p.values():
        piece.rgba = snap_image(piece.rgba, look.palette)
    return Kit(theme, scale, look, p, _colours(th, look.palette))


def _inventory(p: dict, cols: int, rows: int) -> np.ndarray:
    """A ready-made inventory window: cols x rows slots under a title bar."""
    w = p["window"]
    gap = 1
    sl = p["slot"].rgba
    inner_w = cols * sl.shape[1] + (cols - 1) * gap
    inner_h = rows * sl.shape[0] + (rows - 1) * gap
    m = w.margins
    W = m["left"] + m["right"] + inner_w - 2
    H = m["top"] + m["bottom"] + inner_h - 2
    img = nine(w.rgba, m, W, H)
    for r in range(rows):
        for c in range(cols):
            x = m["left"] - 1 + c * (sl.shape[1] + gap)
            y = m["top"] - 1 + r * (sl.shape[0] + gap)
            img[y:y + sl.shape[0], x:x + sl.shape[1]] = _over(img[y:y + sl.shape[0], x:x + sl.shape[1]], sl)
    return img


def _over(dst: np.ndarray, src: np.ndarray) -> np.ndarray:
    out = dst.copy()
    on = src[..., 3] > 0
    out[on] = src[on]
    return out


def _axis(n: int, a: int, b: int, total: int) -> np.ndarray:
    """Source index per destination pixel along one axis of a 9-slice (nearest stretch)."""
    mid, dmid = n - a - b, total - a - b
    if dmid < 0:                                   # smaller than its corners: squeeze them
        k = total / max(1, a + b)
        return np.concatenate([np.arange(round(a * k)), n - round(b * k) + np.arange(round(b * k))]
                              )[:total].astype(int)
    centre = a + (np.arange(dmid) * mid // max(1, dmid)) if mid > 0 else np.full(dmid, a, int)
    return np.concatenate([np.arange(a), centre, np.arange(n - b, n)]).astype(int)


def nine(rgba: np.ndarray, margins: dict, w: int, h: int) -> np.ndarray:
    """A 9-slice piece stretched to w x h the way the engines do it (corners kept)."""
    H, W = rgba.shape[:2]
    ys = _axis(H, margins["top"], margins["bottom"], h)
    xs = _axis(W, margins["left"], margins["right"], w)
    return np.ascontiguousarray(rgba[np.ix_(ys, xs)])


# --------------------------------------------------------------------------- demo screen

# (kind, piece, x, y, w, h, extra) at 1x on the 480x270 mock screen. Shared by the preview,
# the Unity demo builder (kit.json "demo") and the Godot demo scene's positions.
def demo_layout() -> list[dict]:
    L = []

    def add(kind, piece, x, y, w=0, h=0, **extra):
        L.append({"kind": kind, "piece": piece, "x": x, "y": y, "w": w, "h": h, **extra})

    # HUD
    for i, name in enumerate(("heart", "heart", "heart", "heart_half", "heart_empty")):
        add("sprite", f"icon_{name}", 8 + i * 17, 8)
    add("text", "", 8, 31, text="HP", colour="label", shadow="label_shadow")
    add("bar", "bar_health", 24, 30, 80, BAR_H, value=0.7)
    add("text", "", 8, 43, text="MP", colour="label", shadow="label_shadow")
    add("bar", "bar_mana", 24, 42, 80, BAR_H, value=0.45)
    add("text", "", 8, 55, text="XP", colour="label", shadow="label_shadow")
    add("segbar", "bar_seg_xp", 24, 54, 4 + 8 * SEGMENTS, BAR_H, value=5 / 8)
    add("sprite", "icon_coin", 330, 8)
    add("text", "", 349, 13, text="1250", colour="label", shadow="label_shadow")
    add("sprite", "icon_gem", 394, 8)
    add("text", "", 413, 13, text="12", colour="label", shadow="label_shadow")
    # menu window
    add("nine", "window", 8, 72, 132, 186)
    add("text", "", 18, 79, text="Options", colour="title", shadow="label_shadow")
    for i, (state, label) in enumerate((("normal", "Resume"), ("hover", "Items"),
                                        ("pressed", "Save"), ("disabled", "Quit"))):
        y = 100 + i * 20
        add("button", f"button_{state}", 20, y, 108, 16, text=label,
            colour="label_disabled" if state == "disabled" else "label", shadow="label_shadow",
            pressed=state == "pressed")
    add("sprite", "checkbox_on", 20, 184)
    add("text", "", 35, 186, text="Music", colour="ink", shadow="ink_shadow")
    add("sprite", "checkbox_off", 76, 184)
    add("text", "", 91, 186, text="Sound", colour="ink", shadow="ink_shadow")
    add("text", "", 20, 204, text="Volume", colour="ink", shadow="ink_shadow")
    add("slider", "slider_track", 20, 218, 108, 7, value=0.6)
    add("nine", "inset", 20, 232, 108, 16)
    add("text", "", 25, 236, text="Name: Ayla_", colour="ink", shadow="")
    # inventory
    add("sprite", "tab_inactive", 0, 0)        # placeholder, replaced below (keeps order simple)
    L.pop()
    add("tab", "tab_active", 300, 34, 44, 13, text="Items", colour="label", shadow="label_shadow")
    add("tab", "tab_inactive", 345, 34, 40, 13, text="Gear", colour="ink", shadow="ink_shadow")
    add("tab", "tab_inactive", 386, 34, 44, 13, text="Magic", colour="ink", shadow="ink_shadow")
    add("nine", "panel", 296, 46, 176, 118)
    icons = ("sword", "shield", "potion", "key", "gem", "coin", "gear", "potion", "heart",
             "sword", "", "")
    for i in range(18):
        r, c = divmod(i, 6)
        x, y = 304 + c * 25, 54 + r * 25
        add("nine", "slot_selected" if i == 2 else "slot", x, y, 20, 20)
        if i < len(icons) and icons[i]:
            add("sprite", f"icon_{icons[i]}", x + 2, y + 2)
    add("text", "", 304, 134, text="Healing potion", colour="ink", shadow="ink_shadow")
    add("text", "", 304, 145, text="Restores 20 HP.", colour="ink", shadow="ink_shadow")
    add("scroll", "scroll_track", 456, 54, 9, 100, value=0.0)
    add("sprite", "icon_close", 454, 30)
    # dialogue
    add("nine", "dialogue", 148, 198, 324, 64)
    add("nine", "nameplate", 160, 189, 46, FONT_CAP + 7)
    add("text", "", 166, 193, text="Elder", colour="label", shadow="label_shadow")
    add("text", "", 162, 212, text="The bridge to the north fell in the storm.",
        colour="ink", shadow="ink_shadow")
    add("text", "", 162, 224, text="Take the old mine road, and mind the dark.",
        colour="ink", shadow="ink_shadow")
    add("sprite", "next_arrow", 454, 246)
    add("sprite", "cursor_hand", 88, 124)
    return L


def _background(w: int, h: int) -> np.ndarray:
    """A dusky little world behind the UI: sky bands, hills, a ground strip."""
    img = np.zeros((h, w, 4), np.uint8)
    img[..., 3] = 255
    sky = [(34, 32, 64), (52, 44, 84), (78, 58, 100), (118, 76, 104), (168, 104, 96)]
    band = max(1, int(h * 0.62) // len(sky))
    for i, c in enumerate(sky):
        img[i * band:(i + 1) * band, :, :3] = c
    img[len(sky) * band:, :, :3] = sky[-1]
    xx = np.arange(w)
    for amp, base, col in ((14, 0.55, (58, 52, 78)), (10, 0.66, (40, 50, 58)),
                           (6, 0.76, (34, 64, 44))):
        top = (h * base + amp * np.sin(xx / w * 9.0 + base * 7) + amp * 0.5 * np.sin(xx / 13.0)).astype(int)
        for x in range(w):
            img[top[x]:, x, :3] = col
    g = int(h * 0.84)
    img[g:, :, :3] = (52, 40, 30)
    img[g:g + 2, :, :3] = (70, 110, 52)
    return img


def mock_screen(kit: Kit, size: tuple[int, int] = SCREEN) -> np.ndarray:
    """Every piece on a mock game screen (at the kit scale)."""
    w, h = size
    img = _background(w, h)
    col = kit.colours
    P = kit.pieces

    def paste(src, x, y):
        sh, sw = src.shape[:2]
        x1, y1 = min(w, x + sw), min(h, y + sh)
        if x1 <= x or y1 <= y:
            return
        img[y:y1, x:x1] = _over(img[y:y1, x:x1], src[:y1 - y, :x1 - x])

    def text(t, x, y, c, sh=None):
        if not t:
            return
        FONT.draw(img, t, x, y, col.get(c, c), shadow=col.get(sh) or None if sh else None)

    for it in demo_layout():
        k, name, x, y, iw, ih = it["kind"], it["piece"], it["x"], it["y"], it["w"], it["h"]
        piece = P.get(name)
        if k == "text":
            text(it["text"], x, y, it["colour"], it.get("shadow"))
        elif k == "sprite":
            paste(piece.rgba, x, y)
        elif k in ("nine", "tab"):
            paste(nine(piece.rgba, piece.margins, iw, ih), x, y)
            if k == "tab":
                tw = FONT.width(it["text"])
                text(it["text"], x + (iw - tw) // 2, y + 4, it["colour"], it.get("shadow"))
        elif k == "button":
            paste(nine(piece.rgba, piece.margins, iw, ih), x, y)
            tw = FONT.width(it["text"])
            text(it["text"], x + (iw - tw) // 2, y + 4 + (1 if it.get("pressed") else 0),
                 it["colour"], it.get("shadow"))
        elif k == "bar":
            paste(nine(P["bar_frame"].rgba, P["bar_frame"].margins, iw, ih), x, y)
            fw = max(6, round(iw * it["value"]))
            paste(nine(piece.rgba, piece.margins, fw, ih), x, y)
        elif k == "segbar":
            paste(P["bar_seg_frame"].rgba, x, y)
            fill = piece.rgba.copy()
            fill[:, round(2 + (iw - 4) * it["value"]):] = 0
            paste(fill, x, y)
        elif k == "slider":
            paste(nine(piece.rgba, piece.margins, iw, ih), x, y)
            fw = max(6, round(iw * it["value"]))
            paste(nine(P["slider_fill"].rgba, P["slider_fill"].margins, fw, ih), x, y)
            knob = P["slider_knob"].rgba
            paste(knob, x + fw - knob.shape[1] // 2 - 1, y + (ih - knob.shape[0]) // 2)
        elif k == "scroll":
            paste(nine(piece.rgba, piece.margins, iw, ih), x, y)
            th = P["scroll_thumb"]
            paste(nine(th.rgba, th.margins, iw, ih // 3), x, y + round((ih - ih // 3) * it["value"]))
    return enlarge(img, kit.scale)


# --------------------------------------------------------------------------- sheet + files

def pack(kit: Kit, width: int = 256, gap: int = 2) -> tuple[np.ndarray, dict]:
    """Every piece in one sheet (shelf packing by group, 1x) -> (RGBA at kit scale,
    {name: (x, y, w, h)} at kit scale)."""
    x = y = shelf = 0
    rects = {}
    last_group = None
    for name, p in kit.pieces.items():
        h, w = p.rgba.shape[:2]
        if (x + w > width and x > 0) or (last_group is not None and p.group != last_group and x > 0
                                         and p.group in ("icon", "panel", "bar")):
            x, y, shelf = 0, y + shelf + gap, 0
        rects[name] = (x, y, w, h)
        x += w + gap
        shelf = max(shelf, h)
        last_group = p.group
    H = y + shelf
    W = max(r[0] + r[2] for r in rects.values())
    sheet = np.zeros((H, W, 4), np.uint8)
    for name, (x, y, w, h) in rects.items():
        sheet[y:y + h, x:x + w] = kit.pieces[name].rgba
    s = kit.scale
    return enlarge(sheet, s), {k: tuple(v * s for v in r) for k, r in rects.items()}


def kit_data(kit: Kit, rects: dict) -> dict:
    """kit.json: pieces (a list, JsonUtility-friendly), colours, font, demo layout."""
    s = kit.scale
    pieces = []
    for name, p in kit.pieces.items():
        x, y, w, h = rects[name]
        d = {"name": name, "file": f"pieces/{name}.png", "group": p.group,
             "x": x, "y": y, "w": w, "h": h, "nine": p.margins is not None,
             "margins": kit.margins(name) or {"left": 0, "top": 0, "right": 0, "bottom": 0}}
        if p.content:
            d["content"] = kit.content(name)
        pieces.append(d)
    demo = [{**it, "x": it["x"] * s, "y": it["y"] * s, "w": it["w"] * s, "h": it["h"] * s}
            for it in demo_layout()]
    return {"theme": kit.theme, "scale": s, "sheet": "sheet.png", "screen": [SCREEN[0] * s, SCREEN[1] * s],
            "look": {"outline": kit.look.outline, "shading": kit.look.shading,
                     "detail": kit.look.detail},
            "colours": kit.colours, "pieces": pieces,
            "font": {"file": f"fonts/{font_name(s)}.fnt", "size": LINE_HEIGHT * s,
                     "line_height": LINE_HEIGHT * s, "cap": FONT_CAP * s,
                     "big": f"fonts/{font_name(2 * s)}.fnt"},
            "demo": demo}


def write_kit(kit: Kit, out_dir) -> list[Path]:
    """sheet.png + kit.json + pieces/*.png + fonts + preview.png into a folder."""
    out = Path(out_dir)
    (out / "pieces").mkdir(parents=True, exist_ok=True)
    for old in (out / "pieces").glob("*.png"):
        old.unlink()
    sheet, rects = pack(kit)
    written = [out / "sheet.png", out / "kit.json", out / "preview.png"]
    Image.fromarray(sheet, "RGBA").save(written[0])
    written[1].write_text(json.dumps(kit_data(kit, rects), indent=1), encoding="utf-8")
    Image.fromarray(mock_screen(kit), "RGBA").save(written[2])
    for name in kit.pieces:
        p = out / "pieces" / f"{name}.png"
        Image.fromarray(kit.image(name), "RGBA").save(p)
        written.append(p)
    for s in sorted({kit.scale, 2 * kit.scale}):
        written += write_font(out / "fonts", s)
    return written


# --------------------------------------------------------------------------- GPU icons

ICON_WORDS = {"heart": ("a red heart, life icon", "misc"), "coin": ("a gold coin", "misc"),
              "gem": ("a blue gemstone", "misc"), "key": ("a golden key", "key"),
              "sword": ("a steel sword", "weapon"), "shield": ("a round wooden shield", "armour"),
              "potion": ("a red health potion", "potion"), "gear": ("a metal cog gear", "misc")}
GPU_ICONS = tuple(ICON_WORDS)


def gpu_icons(names=GPU_ICONS, style=None, client=None, seed: int | None = None,
              progress=None) -> dict[str, np.ndarray]:
    """Icons drawn by SDXL + Pixel Art XL (house style words added like item icons), each
    converted to 16x16 in the project's colours. Needs ComfyUI; the kit never does."""
    from .generate import generate_images
    from .objects import icon_prompt, main_subject, style_settings
    from .pipeline import convert, isolate, load_image
    st = style if style is not None else assets.project_style()
    s = style_settings(st, ICON, ICON, min(int(st.colours), 12), "center")
    say = progress or (lambda text: None)
    out = {}
    for i, name in enumerate(names):
        if name not in ICON_WORDS:
            raise ValueError(f"No GPU words for icon '{name}' (use {', '.join(ICON_WORDS)})")
        words, category = ICON_WORDS[name]
        say(f"Drawing the {name} icon on your GPU ({i + 1}/{len(names)})…")
        positive, negative = icon_prompt(words, category, ICON, st)
        (img, _s), = generate_images(positive, negative, client=client, count=1, aspect=1.0,
                                     seed=None if seed is None else seed + i)
        rgba = main_subject(isolate(load_image(img.convert("RGB")), s))
        sprite = convert(rgba, s)
        canvas = np.zeros((ICON, ICON, 4), np.uint8)
        r = sprite.rgba[:ICON, :ICON]
        y0, x0 = (ICON - r.shape[0]) // 2, (ICON - r.shape[1]) // 2
        canvas[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
        out[name] = canvas
    return out


# --------------------------------------------------------------------------- assets

def _gpu_icons_of(asset) -> dict:
    folder = asset.folder / "icons_gpu"
    if asset.extra.get("icons") != "gpu" or not folder.exists():
        return {}
    return {p.stem: np.asarray(Image.open(p).convert("RGBA")) for p in folder.glob("*.png")}


def kit_of(asset, project=None) -> Kit:
    """The kit an asset describes, drawn now with the project's style."""
    proj = project or assets.current_project()
    return make_kit(asset.extra.get("theme", "wood"), proj.style, asset.extra.get("scale"),
                    _gpu_icons_of(asset))


def render_ui(asset, project=None) -> list[Path]:
    kit = kit_of(asset, project)
    files = write_kit(kit, asset.folder)
    thumb = asset.folder / "thumb.png"
    _thumb(kit, thumb)
    asset.extra["scale"] = kit.scale
    asset.save()
    return files + [thumb]


def _thumb(kit: Kit, out: Path, px: int = 96) -> Path:
    """A window with a button and three icons, crisp."""
    P = kit.pieces
    img = nine(P["window"].rgba, P["window"].margins, 46, 46)
    for i, n in enumerate(("heart", "coin", "sword")):
        ic = P[f"icon_{n}"].rgba
        y, x = 18, 4 + i * 13
        img[y:y + 13, x:x + 13] = _over(img[y:y + 13, x:x + 13], ic[1:14, 1:14])
    b = nine(P["button_normal"].rgba, P["button_normal"].margins, 34, 10)
    img[33:43, 6:40] = _over(img[33:43, 6:40], b)
    k = max(1, px // 48)
    canvas = np.zeros((px, px, 4), np.uint8)
    big = enlarge(img, k)
    o = (px - big.shape[0]) // 2
    canvas[o:o + big.shape[0], o:o + big.shape[1]] = big
    Image.fromarray(canvas, "RGBA").save(out)
    return out


def new_ui(name: str = "UI kit", theme: str = "wood", *, scale: int | None = None,
           project=None, icons: dict | None = None) -> "assets.Asset":
    """A "ui" asset: the kit written into <project>/ui/<slug>/. icons = GPU icons to keep."""
    register_kind()
    if theme not in THEMES:
        raise ValueError(f"Unknown theme '{theme}' (use {', '.join(THEMES)})")
    proj = project or assets.current_project()
    asset = assets.new_asset(KIND, name, proj, theme=theme, scale=scale,
                             icons="gpu" if icons else "drawn")
    if icons:
        _save_icons(asset, icons)
    render_ui(asset, proj)
    return asset


def _save_icons(asset, icons: dict) -> None:
    folder = asset.folder / "icons_gpu"
    folder.mkdir(parents=True, exist_ok=True)
    for k, img in icons.items():
        Image.fromarray(np.ascontiguousarray(img), "RGBA").save(folder / f"{k}.png")
    asset.extra["icons"] = "gpu"


def update_ui(asset, project=None, *, theme: str | None = None, scale: int | None = None,
              icons: dict | None = None, drawn_icons: bool = False):
    """Change the theme / scale / icons and redraw (drawn_icons=True drops GPU icons)."""
    if theme is not None:
        if theme not in THEMES:
            raise ValueError(f"Unknown theme '{theme}' (use {', '.join(THEMES)})")
        asset.extra["theme"] = theme
    if scale is not None:
        asset.extra["scale"] = int(scale)
    if drawn_icons:
        shutil.rmtree(asset.folder / "icons_gpu", ignore_errors=True)
        asset.extra["icons"] = "drawn"
    if icons:
        _save_icons(asset, icons)
    render_ui(asset, project)
    return asset


def make_gpu_icons(asset, client=None, project=None, progress=None, seed: int | None = None):
    """Draw the GPU icons for a kit asset and redraw it with them."""
    proj = project or assets.current_project()
    icons = gpu_icons(style=proj.style, client=client, seed=seed, progress=progress)
    return update_ui(asset, proj, icons=icons)


def open_ui(folder) -> "assets.Asset":
    folder = Path(getattr(folder, "folder", folder))
    d = json.loads((folder / "asset.json").read_text(encoding="utf-8"))
    if d.get("kind") != KIND:
        raise ValueError(f"{folder} is a {d.get('kind')} asset, not a UI kit")
    return assets.Asset(KIND, d.get("name", folder.name), folder, d.get("tags", []),
                        d.get("collections", []), d.get("created", 0.0), d.get("extra", {}))


def list_uis(project=None) -> list:
    register_kind()
    return assets.list_assets(KIND, project=project)


# --------------------------------------------------------------------------- engine export

def export_kit(kit_or_asset, out_dir, engine: str = "godot") -> list[Path]:
    """A kit (Kit, ui asset or its folder) -> a folder to drop into a game project.
    godot: theme.tres (StyleBoxTexture 9-slices for Panel, Button, ProgressBar, TabContainer,
           CheckBox, HSlider, ScrollBar, LineEdit… + the pixel FontFile + type variations
           WindowPanel, DialogueBox, Nameplate, InsetPanel, Slot, HealthBar, ManaBar, XpBar),
           UIKitDemo.tscn + uikit_demo.gd, pieces/, fonts/, kit.json, README.md. Paths are
           relative, so the folder can go anywhere under res://.
    unity: pieces/*.png with .meta (point filter, no compression, 9-slice spriteBorder), fonts
           (.fnt + atlas), Editor/PixelPipelineUIKit.cs (builds a Unity Font from the .fnt and
           a demo Canvas from kit.json), kit.json, README.md."""
    if engine not in ("godot", "unity"):
        raise ValueError("engine must be godot or unity")
    if isinstance(kit_or_asset, Kit):
        kit = kit_or_asset
    else:
        asset = kit_or_asset if hasattr(kit_or_asset, "folder") else open_ui(kit_or_asset)
        kit = kit_of(asset)
    out = Path(out_dir)
    written = write_kit(kit, out)
    (out / "preview.png").unlink(missing_ok=True)
    written = [p for p in written if p.name != "preview.png"]
    if engine == "godot":
        written += _export_godot(kit, out)
    else:
        written += _export_unity(kit, out)
    return written


def _gd_color(hexs: str, a: float = 1.0) -> str:
    c = _rgb(hexs)
    return f"Color({c[0] / 255:.4g}, {c[1] / 255:.4g}, {c[2] / 255:.4g}, {a:g})"


# Godot theme: (type, stylebox name) -> piece, and type variations
GODOT_STYLES = {
    ("Panel", "panel"): "panel", ("PanelContainer", "panel"): "panel",
    ("PopupPanel", "panel"): "panel", ("PopupMenu", "panel"): "panel",
    ("PopupMenu", "hover"): "button_hover", ("TooltipPanel", "panel"): "inset",
    ("Button", "normal"): "button_normal", ("Button", "hover"): "button_hover",
    ("Button", "pressed"): "button_pressed", ("Button", "disabled"): "button_disabled",
    ("Button", "hover_pressed"): "button_pressed",
    ("OptionButton", "normal"): "button_normal", ("OptionButton", "hover"): "button_hover",
    ("OptionButton", "pressed"): "button_pressed", ("OptionButton", "disabled"): "button_disabled",
    ("LineEdit", "normal"): "inset", ("LineEdit", "read_only"): "inset",
    ("TextEdit", "normal"): "inset",
    ("ProgressBar", "background"): "bar_frame", ("ProgressBar", "fill"): "bar_health",
    ("TabContainer", "panel"): "panel", ("TabContainer", "tab_selected"): "tab_active",
    ("TabContainer", "tab_unselected"): "tab_inactive", ("TabContainer", "tab_hovered"): "tab_inactive",
    ("TabContainer", "tab_disabled"): "tab_inactive",
    ("TabBar", "tab_selected"): "tab_active", ("TabBar", "tab_unselected"): "tab_inactive",
    ("TabBar", "tab_hovered"): "tab_inactive", ("TabBar", "tab_disabled"): "tab_inactive",
    ("HSlider", "slider"): "slider_track", ("HSlider", "grabber_area"): "slider_fill",
    ("HSlider", "grabber_area_highlight"): "slider_fill",
    ("VSlider", "slider"): "slider_track", ("VSlider", "grabber_area"): "slider_fill",
    ("VSlider", "grabber_area_highlight"): "slider_fill",
    ("VScrollBar", "scroll"): "scroll_track", ("VScrollBar", "scroll_focus"): "scroll_track",
    ("VScrollBar", "grabber"): "scroll_thumb", ("VScrollBar", "grabber_highlight"): "scroll_thumb_hover",
    ("VScrollBar", "grabber_pressed"): "scroll_thumb_hover",
    ("HScrollBar", "scroll"): "scroll_track", ("HScrollBar", "scroll_focus"): "scroll_track",
    ("HScrollBar", "grabber"): "scroll_thumb", ("HScrollBar", "grabber_highlight"): "scroll_thumb_hover",
    ("HScrollBar", "grabber_pressed"): "scroll_thumb_hover",
    ("WindowPanel", "panel"): "window", ("DialogueBox", "panel"): "dialogue",
    ("Nameplate", "panel"): "nameplate", ("InsetPanel", "panel"): "inset",
    ("Slot", "panel"): "slot", ("SlotSelected", "panel"): "slot_selected",
    ("HealthBar", "fill"): "bar_health", ("ManaBar", "fill"): "bar_mana", ("XpBar", "fill"): "bar_xp",
}
GODOT_VARIATIONS = {"WindowPanel": "Panel", "DialogueBox": "PanelContainer",
                    "Nameplate": "PanelContainer", "InsetPanel": "PanelContainer", "Slot": "Panel",
                    "SlotSelected": "Panel", "HealthBar": "ProgressBar", "ManaBar": "ProgressBar",
                    "XpBar": "ProgressBar"}
GODOT_EMPTY = [("Button", "focus"), ("OptionButton", "focus"), ("LineEdit", "focus"),
               ("TabContainer", "tab_focus"), ("TabBar", "tab_focus"), ("HSlider", "grabber_area_focus"),
               *[("CheckBox", s) for s in ("normal", "hover", "pressed", "hover_pressed", "disabled", "focus")],
               ("CheckButton", "focus")]
GODOT_ICONS = {("CheckBox", "checked"): "checkbox_on", ("CheckBox", "unchecked"): "checkbox_off",
               ("CheckBox", "checked_disabled"): "checkbox_on",
               ("CheckBox", "unchecked_disabled"): "checkbox_off",
               ("HSlider", "grabber"): "slider_knob", ("HSlider", "grabber_highlight"): "slider_knob_hover",
               ("HSlider", "grabber_disabled"): "slider_knob",
               ("VSlider", "grabber"): "slider_knob", ("VSlider", "grabber_highlight"): "slider_knob_hover"}


def godot_theme(kit: Kit) -> str:
    """theme.tres text (relative paths: pieces/, fonts/ next to it)."""
    s = kit.scale
    col = kit.colours
    used = sorted({*GODOT_STYLES.values(), *GODOT_ICONS.values()})
    ext = [f'[ext_resource type="Texture2D" path="pieces/{n}.png" id="t_{n}"]' for n in used]
    ext.append(f'[ext_resource type="FontFile" path="fonts/{font_name(s)}.fnt" id="f_main"]')
    subs = []
    for n in sorted(set(GODOT_STYLES.values())):
        m = kit.margins(n)
        lines = [f'[sub_resource type="StyleBoxTexture" id="sb_{n}"]', f'texture = ExtResource("t_{n}")']
        lines += [f"texture_margin_{k} = {float(m[k])}" for k in ("left", "top", "right", "bottom")]
        c = kit.content(n)
        if c:
            lines += [f"content_margin_{k} = {float(c[k])}" for k in ("left", "top", "right", "bottom")]
        subs.append("\n".join(lines))
    subs.append('[sub_resource type="StyleBoxEmpty" id="sb_empty"]')
    props = [f'default_font = ExtResource("f_main")', f"default_font_size = {LINE_HEIGHT * s}"]
    for var, base in GODOT_VARIATIONS.items():
        props.append(f'{var}/base_type = &"{base}"')
    for (typ, name), piece in sorted(GODOT_STYLES.items()):
        props.append(f'{typ}/styles/{name} = SubResource("sb_{piece}")')
    for typ, name in GODOT_EMPTY:
        props.append(f'{typ}/styles/{name} = SubResource("sb_empty")')
    for (typ, name), piece in sorted(GODOT_ICONS.items()):
        props.append(f'{typ}/icons/{name} = ExtResource("t_{piece}")')
    label, ink = _gd_color(col["label"]), _gd_color(col["ink"])
    for typ in ("Button", "OptionButton"):
        props += [f"{typ}/colors/font_color = {label}", f"{typ}/colors/font_hover_color = {label}",
                  f"{typ}/colors/font_pressed_color = {label}",
                  f"{typ}/colors/font_hover_pressed_color = {label}",
                  f"{typ}/colors/font_focus_color = {label}",
                  f"{typ}/colors/font_disabled_color = {_gd_color(col['label_disabled'])}"]
    props += [f"CheckBox/colors/font_color = {ink}", f"CheckBox/colors/font_hover_color = {ink}",
              f"CheckBox/colors/font_pressed_color = {ink}", f"CheckBox/colors/font_focus_color = {ink}",
              f"CheckBox/colors/font_hover_pressed_color = {ink}",
              f"CheckBox/constants/h_separation = {4 * s}",
              f"Label/colors/font_color = {ink}", f"LineEdit/colors/font_color = {ink}",
              f"ProgressBar/colors/font_color = {label}",
              f"TabContainer/colors/font_selected_color = {label}",
              f"TabContainer/colors/font_unselected_color = {ink}",
              f"TabContainer/colors/font_hovered_color = {label}",
              f"TabBar/colors/font_selected_color = {label}", f"TabBar/colors/font_unselected_color = {ink}",
              f"TooltipLabel/colors/font_color = {ink}",
              f"PopupMenu/colors/font_color = {ink}", f"PopupMenu/colors/font_hover_color = {label}"]
    if col.get("ink_shadow"):
        props += [f"Label/colors/font_shadow_color = {_gd_color(col['ink_shadow'])}",
                  f"Label/constants/shadow_offset_x = {s}", f"Label/constants/shadow_offset_y = {s}",
                  "Label/constants/shadow_outline_size = 0"]
    head = f"[gd_resource type=\"Theme\" load_steps={len(ext) + len(subs) + 1} format=3]"
    return "\n\n".join([head, "\n".join(ext), *subs, "[resource]\n" + "\n".join(props)]) + "\n"


def _gd_str(s: str) -> str:
    return str(s).replace("\\", "\\\\").replace('"', '\\"')


def godot_demo(kit: Kit) -> str:
    """UIKitDemo.tscn: real Controls using theme.tres, laid out like the preview."""
    s = kit.scale
    ext = [("Theme", "theme.tres", "1_theme"), ("Script", "uikit_demo.gd", "2_script")]
    tex: dict[str, str] = {}

    def texture(piece):
        if piece not in tex:
            tex[piece] = f"t{len(tex) + 3}_{piece}"
            ext.append(("Texture2D", f"pieces/{piece}.png", tex[piece]))
        return tex[piece]

    nodes: list[str] = []

    def node(name, typ, parent=".", rect=None, props=()):
        lines = [f'[node name="{name}" type="{typ}" parent="{parent}"]']
        if rect:
            x, y, w, h = (v * s for v in rect)
            lines += [f"offset_left = {x}.0", f"offset_top = {y}.0", f"offset_right = {x + w}.0",
                      f"offset_bottom = {y + h}.0"]
        lines += list(props)
        nodes.append("\n".join(lines))

    W, H = SCREEN
    lay = demo_layout()
    node("Back", "ColorRect", rect=(0, 0, W, H), props=["color = Color(0.13, 0.12, 0.25, 1)",
                                                      "mouse_filter = 2"])
    n = 0
    for it in lay:
        k, piece = it["kind"], it["piece"]
        n += 1
        if k == "sprite" and piece.startswith(("icon_", "next_arrow")) and it["y"] < 60 \
                or k == "sprite" and piece in ("icon_close", "next_arrow"):
            node(f"Icon{n}", "TextureRect", rect=(it["x"], it["y"], 16, 16),
                 props=[f'texture = ExtResource("{texture(piece)}")', "mouse_filter = 2"])
    bars = {"bar_health": "HealthBar", "bar_mana": "ManaBar"}
    for it in lay:
        if it["kind"] == "bar":
            node(bars[it["piece"]].replace("Bar", ""), "ProgressBar",
                 rect=(it["x"], it["y"], it["w"], it["h"]),
                 props=[f'theme_type_variation = &"{bars[it["piece"]]}"',
                        f"value = {round(it['value'] * 100)}.0", "show_percentage = false"])
        elif it["kind"] == "segbar":
            node("Xp", "TextureProgressBar", rect=(it["x"], it["y"], it["w"], it["h"]),
                 props=[f"max_value = {SEGMENTS}.0", f"value = {round(it['value'] * SEGMENTS)}.0",
                        f'texture_under = ExtResource("{texture("bar_seg_frame")}")',
                        f'texture_progress = ExtResource("{texture("bar_seg_xp")}")'])
        elif it["kind"] == "text" and it["y"] < 60:
            node(f"Hud{it['text']}", "Label", rect=(it["x"], it["y"] - 1, 40, LINE_HEIGHT + 1),
                 props=[f'text = "{_gd_str(it["text"])}"'])
    # options window: a WindowPanel with a VBoxContainer of controls
    node("Options", "Panel", rect=(8, 72, 132, 186), props=['theme_type_variation = &"WindowPanel"'])
    node("Title", "Label", "Options", rect=(10, 6, 100, LINE_HEIGHT), props=['text = "Options"'])
    node("List", "VBoxContainer", "Options", rect=(12, 28, 108, 150),
         props=[f"theme_override_constants/separation = {4 * s}"])
    for label, extra in (("Resume", []), ("Items", []), ("Save", ["toggle_mode = true", "button_pressed = true"]),
                         ("Quit", ["disabled = true"])):
        node(label, "Button", "Options/List", props=[f'text = "{label}"',
                                                    f"custom_minimum_size = Vector2(0, {16 * s})", *extra])
    node("Music", "CheckBox", "Options/List", props=['text = "Music"', "button_pressed = true"])
    node("Volume", "HSlider", "Options/List", props=["value = 60.0",
                                                    f"custom_minimum_size = Vector2(0, {11 * s})"])
    node("Name", "LineEdit", "Options/List", props=['text = "Ayla"',
                                                   f"custom_minimum_size = Vector2(0, {16 * s})"])
    # inventory: tabs of slot grids + a scrollbar
    node("Inventory", "TabContainer", rect=(296, 34, 176, 130))
    icons = [it["piece"] for it in lay if it["kind"] == "sprite" and it["x"] >= 300 and it["y"] >= 54
             and it["piece"].startswith("icon_")]
    for t, name in enumerate(("Items", "Gear", "Magic")):
        node(name, "MarginContainer", "Inventory",
             props=[f"theme_override_constants/margin_{side} = {6 * s}"
                    for side in ("left", "top", "right", "bottom")])
        node("Grid", "GridContainer", f"Inventory/{name}",
             props=["columns = 6", f"theme_override_constants/h_separation = {5 * s}",
                    f"theme_override_constants/v_separation = {5 * s}"])
        for i in range(12 if t == 0 else 6):
            var = "SlotSelected" if (t == 0 and i == 2) else "Slot"
            node(f"Slot{i}", "Panel", f"Inventory/{name}/Grid",
                 props=[f'theme_type_variation = &"{var}"',
                        f"custom_minimum_size = Vector2({20 * s}, {20 * s})"])
            if t == 0 and i < len(icons):
                node("Icon", "TextureRect", f"Inventory/{name}/Grid/Slot{i}",
                     rect=(2, 2, 16, 16), props=[f'texture = ExtResource("{texture(icons[i])}")',
                                                 "mouse_filter = 2"])
    node("Scroll", "VScrollBar", rect=(456, 80, 9, 76), props=["page = 30.0"])
    # dialogue
    node("Dialogue", "PanelContainer", rect=(148, 198, 324, 64),
         props=['theme_type_variation = &"DialogueBox"'])
    node("Text", "Label", "Dialogue", props=[
        'text = "The bridge to the north fell in the storm. Take the old mine road, and mind the dark."',
        "autowrap_mode = 3"])
    node("Nameplate", "PanelContainer", rect=(160, 189, 46, FONT_CAP + 7),
         props=['theme_type_variation = &"Nameplate"'])
    node("Speaker", "Label", "Nameplate", props=['text = "Elder"'])
    node("Next", "TextureRect", rect=(454, 246, 9, 7),
         props=[f'texture = ExtResource("{texture("next_arrow")}")', "mouse_filter = 2"])
    texture("cursor")
    head = [f"[gd_scene load_steps={len(ext) + 1} format=3]", ""]
    head += [f'[ext_resource type="{t}" path="{p}" id="{i}"]' for t, p, i in ext]
    root = ['[node name="UIKitDemo" type="Control"]', "texture_filter = 1",
            f"offset_right = {W * s}.0", f"offset_bottom = {H * s}.0",
            'theme = ExtResource("1_theme")', 'script = ExtResource("2_script")']
    return "\n".join(head) + "\n\n" + "\n".join(root) + "\n\n" + "\n\n".join(nodes) + "\n"


GODOT_DEMO_GD = '''extends Control
## The UI kit demo (made by pixel-pipeline): the theme on real Controls. The pixel cursor is
## set here; move the Volume slider to see the bars follow.

const CURSOR := preload("pieces/cursor.png")

func _ready() -> void:
	Input.set_custom_mouse_cursor(CURSOR)
	$Options/List/Volume.value_changed.connect(_on_volume)

func _on_volume(v: float) -> void:
	$Health.value = v
	$Mana.value = 100.0 - v
'''

GODOT_README = """# UI kit for Godot 4 (made by pixel-pipeline)

Copy this folder anywhere into your project (paths inside are relative). Godot imports the
PNGs and the BMFont `.fnt` pixel fonts.

- `theme.tres`: a Theme. Set it on your root Control (or Project Settings > GUI > Theme >
  Custom) and Panel, PanelContainer, Button, OptionButton, CheckBox, LineEdit, ProgressBar,
  TabContainer / TabBar, HSlider / VSlider, H/VScrollBar, PopupMenu and tooltips use the kit.
  Type variations (Inspector > Theme Type Variation): `WindowPanel` (title bar on top),
  `DialogueBox`, `Nameplate`, `InsetPanel`, `Slot`, `SlotSelected`, `HealthBar`, `ManaBar`,
  `XpBar`. The default font is the pixel font at @SIZE@ px (@SCALE@x); `fonts/` also has it at
  twice that for titles.
- `UIKitDemo.tscn`: every piece on one screen (HUD hearts and bars, an options window, an
  inventory, a dialogue box). Segmented bars use TextureProgressBar with `bar_seg_frame.png`
  under `bar_seg_<kind>.png`.
- `pieces/`: every piece as its own PNG; `kit.json` has the 9-slice margins of each.
- Keep pixels crisp: Project Settings > Rendering > Textures > Default Texture Filter = Nearest
  (the demo sets Nearest on its root), and Display > Window > Stretch > Scale Mode = integer.
"""


def _export_godot(kit: Kit, out: Path) -> list[Path]:
    files = {"theme.tres": godot_theme(kit), "UIKitDemo.tscn": godot_demo(kit),
             "uikit_demo.gd": GODOT_DEMO_GD,
             "README.md": GODOT_README.replace("@SIZE@", str(LINE_HEIGHT * kit.scale))
             .replace("@SCALE@", str(kit.scale))}
    written = []
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8")
        written.append(out / name)
    return written


def _export_unity(kit: Kit, out: Path) -> list[Path]:
    from .exports import _unity_write, unity_guid, unity_meta
    written = []
    tag = f"uikit/{kit.theme}/{kit.scale}/"
    for png in sorted(out.rglob("*.png")):
        rel = png.relative_to(out).as_posix()
        size = Image.open(png).size
        meta = unity_meta(unity_guid(tag + rel), size, None, ppu=100, pivot=(0.5, 0.5))
        name = png.stem
        m = kit.margins(name) if png.parent.name == "pieces" and name in kit.pieces else None
        if m:                                       # Unity border = left, bottom, right, top
            meta = meta.replace("spriteBorder: {x: 0, y: 0, z: 0, w: 0}",
                                f"spriteBorder: {{x: {m['left']}, y: {m['bottom']}, "
                                f"z: {m['right']}, w: {m['top']}}}")
        p = png.with_name(png.name + ".meta")
        _unity_write(p, meta)
        written.append(p)
    (out / "Editor").mkdir(exist_ok=True)
    cs = out / "Editor" / "PixelPipelineUIKit.cs"
    _unity_write(cs, UNITY_CS)
    readme = out / "README.md"
    readme.write_text(UNITY_README, encoding="utf-8")
    return written + [cs, readme]


UNITY_CS = r'''// UI kit editor tools (made by pixel-pipeline). Needs com.unity.ugui (every standard template).
//   Tools > Pixel Pipeline > Build pixel fonts: every BMFont .fnt in the project -> a Unity
//     Font (.fontsettings + .mat next to it); runs by itself when a .fnt is imported.
//   Tools > Pixel Pipeline > Build UI kit demo: a Canvas with every piece laid out from the
//     selected kit.json (or the first one found), saved as UIKitDemo.unity beside it.
// If you import several kits, keep ONE copy of this file (the class names would clash).
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.UI;

public class PixelPipelineFontPostprocessor : AssetPostprocessor {
    static void OnPostprocessAllAssets(string[] imported, string[] deleted, string[] moved, string[] movedFrom) {
        foreach (var p in imported) {
            if (!p.EndsWith(".fnt", StringComparison.OrdinalIgnoreCase)) continue;
            var path = p;
            EditorApplication.delayCall += () => PixelPipelineUIKit.BuildFont(path);
        }
    }
}

public static class PixelPipelineUIKit {
    [Serializable] public class Margins { public int left, top, right, bottom; }
    [Serializable] public class PieceData { public string name, file, group; public int x, y, w, h; public bool nine; public Margins margins; }
    [Serializable] public class Colours { public string ink, ink_shadow, label, label_shadow, label_disabled, title, accent; }
    [Serializable] public class FontData { public string file, big; public int size, line_height, cap; }
    [Serializable] public class DemoItem { public string kind, piece, text, colour, shadow; public int x, y, w, h; public float value; public bool pressed; }
    [Serializable] public class KitData {
        public string theme; public int scale; public int[] screen; public Colours colours;
        public PieceData[] pieces; public FontData font; public DemoItem[] demo;
    }

    [MenuItem("Tools/Pixel Pipeline/Build pixel fonts")]
    public static void BuildAllFonts() {
        foreach (var guid in AssetDatabase.FindAssets("t:TextAsset")) {
            var p = AssetDatabase.GUIDToAssetPath(guid);
            if (p.EndsWith(".fnt", StringComparison.OrdinalIgnoreCase)) BuildFont(p);
        }
    }

    static Dictionary<string, string> Fields(string line) {
        var d = new Dictionary<string, string>();
        int i = line.IndexOf(' ');
        while (i >= 0 && i < line.Length) {
            while (i < line.Length && line[i] == ' ') i++;
            int eq = line.IndexOf('=', i);
            if (eq < 0) break;
            string key = line.Substring(i, eq - i);
            int j = eq + 1;
            string val;
            if (j < line.Length && line[j] == '"') { int end = line.IndexOf('"', j + 1); val = line.Substring(j + 1, end - j - 1); i = end + 1; }
            else { int end = line.IndexOf(' ', j); if (end < 0) end = line.Length; val = line.Substring(j, end - j); i = end; }
            d[key] = val;
        }
        return d;
    }

    static int I(Dictionary<string, string> d, string k) { return int.Parse(d[k], CultureInfo.InvariantCulture); }

    /// A BMFont text file -> a Unity Font asset (+ material) next to it.
    public static Font BuildFont(string fntPath) {
        if (!File.Exists(fntPath)) return null;
        var dir = Path.GetDirectoryName(fntPath).Replace('\\', '/');
        int lineHeight = 10, baseLine = 7, W = 1, H = 1, size = 10;
        string page = null;
        var chars = new List<Dictionary<string, string>>();
        foreach (var raw in File.ReadAllLines(fntPath)) {
            var line = raw.Trim();
            if (line.StartsWith("info ")) size = I(Fields(line), "size");
            else if (line.StartsWith("common ")) { var f = Fields(line); lineHeight = I(f, "lineHeight"); baseLine = I(f, "base"); W = I(f, "scaleW"); H = I(f, "scaleH"); }
            else if (line.StartsWith("page ")) page = Fields(line)["file"];
            else if (line.StartsWith("char ")) chars.Add(Fields(line));
        }
        if (page == null) return null;
        var tex = AssetDatabase.LoadAssetAtPath<Texture2D>(dir + "/" + page);
        if (tex == null) return null;
        string stem = dir + "/" + Path.GetFileNameWithoutExtension(fntPath);
        var mat = AssetDatabase.LoadAssetAtPath<Material>(stem + ".mat");
        if (mat == null) { mat = new Material(Shader.Find("GUI/Text Shader")); AssetDatabase.CreateAsset(mat, stem + ".mat"); }
        mat.mainTexture = tex;
        var font = AssetDatabase.LoadAssetAtPath<Font>(stem + ".fontsettings");
        if (font == null) { font = new Font(Path.GetFileNameWithoutExtension(fntPath)); AssetDatabase.CreateAsset(font, stem + ".fontsettings"); }
        font.material = mat;
        var infos = new List<CharacterInfo>();
        foreach (var c in chars) {
            int x = I(c, "x"), y = I(c, "y"), w = I(c, "width"), h = I(c, "height");
            int xo = I(c, "xoffset"), yo = I(c, "yoffset"), adv = I(c, "xadvance");
            float u0 = (float)x / W, u1 = (float)(x + w) / W, v1 = 1f - (float)y / H, v0 = 1f - (float)(y + h) / H;
            var ci = new CharacterInfo {
                index = I(c, "id"), advance = adv,
                uvBottomLeft = new Vector2(u0, v0), uvBottomRight = new Vector2(u1, v0),
                uvTopLeft = new Vector2(u0, v1), uvTopRight = new Vector2(u1, v1),
                minX = xo, maxX = xo + w, maxY = baseLine - yo, minY = baseLine - yo - h,
            };
            infos.Add(ci);
        }
        font.characterInfo = infos.ToArray();
        var so = new SerializedObject(font);
        so.FindProperty("m_LineSpacing").floatValue = lineHeight;
        so.FindProperty("m_FontSize").floatValue = size;
        var asc = so.FindProperty("m_Ascent");
        if (asc != null) asc.floatValue = baseLine;
        so.ApplyModifiedPropertiesWithoutUndo();
        EditorUtility.SetDirty(font);
        EditorUtility.SetDirty(mat);
        AssetDatabase.SaveAssets();
        return font;
    }

    static string FindKit() {
        var sel = Selection.activeObject != null ? AssetDatabase.GetAssetPath(Selection.activeObject) : "";
        if (sel.EndsWith("kit.json")) return sel;
        foreach (var guid in AssetDatabase.FindAssets("kit t:TextAsset")) {
            var p = AssetDatabase.GUIDToAssetPath(guid);
            if (Path.GetFileName(p) == "kit.json") return p;
        }
        return null;
    }

    [MenuItem("Tools/Pixel Pipeline/Build UI kit demo")]
    public static void BuildDemoMenu() {
        var kit = FindKit();
        if (kit == null) { Debug.LogWarning("No kit.json found (select one)"); return; }
        var root = BuildDemo(kit);
        var scene = Path.GetDirectoryName(kit).Replace('\\', '/') + "/UIKitDemo.unity";
        EditorSceneManager.SaveScene(root.scene, scene);
    }

    public static Color Hex(string hex) {
        Color c;
        return !string.IsNullOrEmpty(hex) && ColorUtility.TryParseHtmlString("#" + hex, out c) ? c : Color.white;
    }

    static string Pick(Colours c, string key) {
        switch (key) {
            case "ink": return c.ink; case "ink_shadow": return c.ink_shadow; case "label": return c.label;
            case "label_shadow": return c.label_shadow; case "label_disabled": return c.label_disabled;
            case "title": return c.title; case "accent": return c.accent; default: return "";
        }
    }

    /// A Canvas in the open scene with every demo item of kit.json (sprites from pieces/).
    public static GameObject BuildDemo(string kitJsonPath) {
        var dir = Path.GetDirectoryName(kitJsonPath).Replace('\\', '/');
        var kit = JsonUtility.FromJson<KitData>(File.ReadAllText(kitJsonPath));
        var font = BuildFont(dir + "/" + kit.font.file);
        var pieces = new Dictionary<string, PieceData>();
        foreach (var p in kit.pieces) pieces[p.name] = p;
        var canvasGo = new GameObject("UIKitDemo", typeof(RectTransform), typeof(Canvas), typeof(CanvasScaler), typeof(GraphicRaycaster));
        var canvas = canvasGo.GetComponent<Canvas>();
        canvas.renderMode = RenderMode.ScreenSpaceOverlay;
        canvas.pixelPerfect = true;
        var scaler = canvasGo.GetComponent<CanvasScaler>();
        scaler.uiScaleMode = CanvasScaler.ScaleMode.ConstantPixelSize;
        scaler.referencePixelsPerUnit = 100;
        var screen = new GameObject("Screen", typeof(RectTransform)).GetComponent<RectTransform>();
        screen.SetParent(canvasGo.transform, false);
        screen.anchorMin = screen.anchorMax = new Vector2(0.5f, 0.5f);
        screen.pivot = new Vector2(0.5f, 0.5f);
        screen.sizeDelta = new Vector2(kit.screen[0], kit.screen[1]);
        var back = Place(screen, "Back", 0, 0, kit.screen[0], kit.screen[1]).gameObject.AddComponent<Image>();
        back.color = new Color(0.13f, 0.12f, 0.25f, 1);
        int n = 0;
        foreach (var it in kit.demo) {
            n++;
            if (it.kind == "text") { Label(screen, it.text, it.x, it.y, kit, font, Pick(kit.colours, it.colour), Pick(kit.colours, it.shadow)); continue; }
            PieceData pd;
            if (!pieces.TryGetValue(it.piece, out pd)) continue;
            var sprite = AssetDatabase.LoadAssetAtPath<Sprite>(dir + "/" + pd.file);
            int w = it.w > 0 ? it.w : pd.w, h = it.h > 0 ? it.h : pd.h;
            if (it.kind == "bar" || it.kind == "slider" || it.kind == "segbar" || it.kind == "scroll") {
                string under = it.kind == "bar" ? "bar_frame" : it.kind == "segbar" ? "bar_seg_frame" : null;
                if (under != null) AddImage(screen, it.kind + n + "_frame", Sprite(dir, pieces[under]), it.x, it.y, w, h, pieces[under].nine);
                else AddImage(screen, it.kind + n, sprite, it.x, it.y, w, h, pd.nine);
                string fillName = it.kind == "slider" ? "slider_fill" : it.kind == "scroll" ? "scroll_thumb" : it.piece;
                var fp = pieces[fillName];
                if (it.kind == "scroll") AddImage(screen, it.kind + n + "_thumb", Sprite(dir, fp), it.x, it.y, w, h / 3, true);
                else if (it.kind == "segbar") {
                    var img = AddImage(screen, it.kind + n + "_fill", Sprite(dir, fp), it.x, it.y, w, h, false);
                    img.type = Image.Type.Filled; img.fillMethod = Image.FillMethod.Horizontal; img.fillAmount = it.value;
                } else AddImage(screen, it.kind + n + "_fill", Sprite(dir, fp), it.x, it.y, Mathf.Max(6 * kit.scale, Mathf.RoundToInt(w * it.value)), h, fp.nine);
                continue;
            }
            AddImage(screen, it.piece + "_" + n, sprite, it.x, it.y, w, h, pd.nine && (it.kind == "nine" || it.kind == "button" || it.kind == "tab"));
            if (!string.IsNullOrEmpty(it.text)) {
                var t = Label(screen, it.text, it.x, it.y + (it.pressed ? kit.scale : 0), kit, font, Pick(kit.colours, it.colour), Pick(kit.colours, it.shadow));
                var rt = (RectTransform)t.transform;
                rt.sizeDelta = new Vector2(w, h);
                t.alignment = TextAnchor.MiddleCenter;
            }
        }
        return canvasGo;
    }

    static Sprite Sprite(string dir, PieceData p) { return AssetDatabase.LoadAssetAtPath<Sprite>(dir + "/" + p.file); }

    static RectTransform Place(RectTransform parent, string name, int x, int y, int w, int h) {
        var rt = new GameObject(name, typeof(RectTransform)).GetComponent<RectTransform>();
        rt.SetParent(parent, false);
        rt.anchorMin = rt.anchorMax = new Vector2(0, 1);      // kit.json: top-left origin, y down
        rt.pivot = new Vector2(0, 1);
        rt.anchoredPosition = new Vector2(x, -y);
        rt.sizeDelta = new Vector2(w, h);
        return rt;
    }

    static Image AddImage(RectTransform parent, string name, Sprite sprite, int x, int y, int w, int h, bool sliced) {
        var img = Place(parent, name, x, y, w, h).gameObject.AddComponent<Image>();
        img.sprite = sprite;
        img.type = sliced ? Image.Type.Sliced : Image.Type.Simple;
        img.raycastTarget = false;
        return img;
    }

    static Text Label(RectTransform parent, string text, int x, int y, KitData kit, Font font, string colour, string shadow) {
        var rt = Place(parent, "Text " + text, x, y - kit.scale, 400 * kit.scale, kit.font.line_height + kit.scale);
        var t = rt.gameObject.AddComponent<Text>();
        t.font = font != null ? font : Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        t.fontSize = kit.font.size;
        t.text = text;
        t.color = Hex(colour);
        t.alignment = TextAnchor.UpperLeft;
        t.horizontalOverflow = HorizontalWrapMode.Overflow;
        t.verticalOverflow = VerticalWrapMode.Overflow;
        t.raycastTarget = false;
        if (!string.IsNullOrEmpty(shadow)) {
            var sh = rt.gameObject.AddComponent<Shadow>();
            sh.effectColor = Hex(shadow);
            sh.effectDistance = new Vector2(kit.scale, -kit.scale);
        }
        return t;
    }
}
'''

UNITY_README = """# UI kit for Unity (made by pixel-pipeline)

Needs the Unity UI package (com.unity.ugui; Unity 6 verified). Copy this folder (with the
.meta files) anywhere under `Assets/`.

- `pieces/`: every piece as a Sprite (Point filter, no compression). The 9-slice pieces
  (panel, window, dialogue, buttons, tabs, slots, bars, slider, scrollbar…) have their sprite
  borders set: use them in an Image with Image Type **Sliced**. Buttons: assign
  `button_normal` as the Image and `button_hover` / `button_pressed` / `button_disabled` in the
  Button's Sprite Swap transition.
- `fonts/`: the pixel font as BMFont (`.fnt` + atlas). `Editor/PixelPipelineUIKit.cs` turns
  each `.fnt` into a Unity Font when it is imported (or Tools > Pixel Pipeline > Build pixel
  fonts); use it in a legacy Text at font size = the .fnt size (or whole multiples).
- Tools > Pixel Pipeline > Build UI kit demo: a Canvas with every piece laid out like the
  preview (HUD, options window, inventory, dialogue), saved as `UIKitDemo.unity`.
- `kit.json`: sheet rects, 9-slice margins (left/top/right/bottom, top-left origin), text
  colours and the demo layout. Set the Canvas Scaler to whole-number scales so pixels stay
  square.
"""

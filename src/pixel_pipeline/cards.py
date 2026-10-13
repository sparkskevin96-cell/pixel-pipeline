"""Cards for card games: card ART (painted or pixel illustrations from a library character, an
item, a prop or plain words), procedural card FRAMES in the project palette, composed cards with
a built-in pixel font, and Godot / Unity exports.

    <project>/cards/<slug>/          (asset kind "card")
        asset.json        every card field + where the art came from (extra)
        card.json         the card for games: fields, art/frame file names, layout, font scales
        card.png          art + frame + text (card_2x.png / card_4x.png crisp previews)
        art.png           the chosen art, exactly the frame's art window
        art_v1..3.png     each variation at the art window (art_raw_v*.png = as the GPU made it)
        frame.png         the frame alone (transparent art window: put the art underneath)
        layout.json       slot rects: art, name, cost, type, text, attack, health, power, rarity
        ui/               9-slice pieces in the same style (border, panel, banner) + .json + .tres
        thumb.png

Frames: 63x88 card ratio drawn natively (no resampling, crisp) at SIZES small 126x176, medium
189x264, large 252x352. RARITIES pick the metal (common iron, rare silver, epic amethyst,
legendary gold), FACTIONS the banners, gems, parchment and ink. Units (attack + health) get two
stat badges at the bottom corners, spells/items none or a power badge. Every frame colour is
snapped to the project's palette when it has a fixed one.

Art: "painted" = SDXL without the pixel LoRA (img2img over the subject's picture placed on a
scene-coloured backdrop, or txt2img from words), "pixel" = the same with Pixel Art XL, then the
converter at the art window's size in the project's colours (to_pixel_art does it without the
GPU from any picture). Three variations by default (seeds seed, seed+1, seed+2).

The asset kind "card" is registered into assets.KINDS/FOLDERS at import (register_kind).
"""
from __future__ import annotations

import json
import math
import re
import shutil
import zlib
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
from PIL import Image

from . import assets

KIND = "card"


def register_kind() -> None:
    """Add "card" to the asset kinds (additive; assets.py owns the list)."""
    if KIND not in assets.KINDS:
        assets.KINDS = (*assets.KINDS, KIND)
    assets.FOLDERS.setdefault(KIND, "cards")


register_kind()

# --------------------------------------------------------------------------- choices

SIZES = {"small": (126, 176), "medium": (189, 264), "large": (252, 352)}
BASE_W = 126
RARITIES = ("common", "rare", "epic", "legendary")
FACTIONS = ("neutral", "fire", "nature", "water", "shadow", "holy")
ART_STYLES = ("painted", "pixel")
SOURCE_KINDS = ("character", "item", "prop", "words")
VARIATIONS = 3

# ramps dark -> light: outline, dark, mid-dark, base, light, shine
METALS = {
    "common": ["16161c", "34363e", "50545e", "6e737e", "9aa0aa", "d0d4dc"],
    "rare": ["121a2a", "34445e", "56708e", "7f9cba", "b4cce2", "f0f8ff"],
    "epic": ["1a0e24", "40245a", "603a84", "8658aa", "b48ad4", "eedcff"],
    "legendary": ["2a1606", "5c3410", "8a5a1a", "b88a2c", "e6c04e", "fff4b0"],
}
RARITY_GEMS = {
    "common": ["1c1c1c", "5a5a5a", "8c8c8c", "b8b8b8", "e0e0e0", "ffffff"],
    "rare": ["0a1838", "1a3c8c", "2c66cc", "4c94f0", "90c8ff", "ffffff"],
    "epic": ["200830", "4c1470", "7a28b0", "a850e0", "d494ff", "ffffff"],
    "legendary": ["301000", "782c00", "c05400", "f08a10", "ffc050", "ffffff"],
}
ATTACK_RAMP = ["2a1a04", "6a4410", "a8701c", "dca22c", "f6d264", "fff6c0"]
HEALTH_RAMP = ["2a0606", "6a1010", "a81c1c", "dc3a2c", "f6806a", "ffd8cc"]
NUMBER_COLOURS = ("ffffff", "16100c")          # stat/cost digits: fill, outline

FACTION_COLOURS = {
    "neutral": dict(banner=["2a1e18", "4e3626", "74533a", "9c7652", "c8a47a", "f4e6c8"],
                    parchment=["16100c", "a89270", "d2bf98", "e6d6b0", "f4eacc", "fffaea"],
                    ink="2a1e14",
                    gem=["0c1838", "1e3c80", "3466c0", "5aa0e8", "a0d4ff", "ffffff"],
                    backdrop=["0e0c10", "1c1820", "2c2630", "3c3440", "4e4450", "686070"],
                    sky=((70, 64, 100), (196, 150, 110))),
    "fire": dict(banner=["2a0c0a", "5a1610", "8c2614", "b83a1a", "e8823a", "ffe0a8"],
                 parchment=["1e0c08", "b08a66", "d8b48c", "ecd0a8", "f8e2c2", "fff4e4"],
                 ink="3a140c",
                 gem=["300806", "701410", "b82818", "e85a24", "ffa050", "fff0c0"],
                 backdrop=["140808", "241010", "361818", "4a2020", "5e2c28", "7a3c34"],
                 sky=((90, 30, 30), (232, 124, 52))),
    "nature": dict(banner=["0e1e10", "1c3a1e", "2e5a2c", "40763a", "7cb05a", "e0f4c0"],
                   parchment=["10160c", "a2a274", "c8c896", "dcdcac", "eeeecc", "fafae8"],
                   ink="1a2a14",
                   gem=["06200c", "10441c", "1c7a30", "34b048", "7ce07a", "e0ffd0"],
                   backdrop=["0a100a", "121c12", "1c2a1a", "283824", "34482e", "4a603e"],
                   sky=((40, 70, 50), (156, 184, 112))),
    "water": dict(banner=["0a1628", "142c4c", "1e4878", "2a62a0", "5aa0d0", "d4f0ff"],
                  parchment=["0c1420", "94a8ba", "bccad8", "d0dce6", "e6eef6", "f8fcff"],
                  ink="0e1e30",
                  gem=["041a2c", "0c3a5c", "14649a", "2496d0", "6cd4f4", "e0fcff"],
                  backdrop=["080c14", "101826", "182436", "22324a", "2e425e", "405678"],
                  sky=((30, 50, 90), (112, 172, 204))),
    "shadow": dict(banner=["120a1a", "24142e", "3a2048", "50306a", "7c4c94", "e4d0f4"],
                   parchment=["120c16", "948aa0", "b8aec2", "c8c0cc", "dcd6e2", "f2eef6"],
                   ink="1c1024",
                   gem=["14061e", "32104a", "58207c", "8c3cb4", "c078e4", "f4d8ff"],
                   backdrop=["08060a", "120c16", "1c1422", "281c30", "34263e", "463452"],
                   sky=((30, 20, 45), (112, 72, 142))),
    "holy": dict(banner=["2e2414", "5a4826", "86703a", "a88c48", "e2c878", "fff8dc"],
                 parchment=["1e1810", "b4a888", "dcd2b4", "ece2c4", "f8f0dc", "fffcf0"],
                 ink="2e2410",
                 gem=["2a2008", "5c4a12", "9c8020", "d8b434", "f8e480", "ffffe8"],
                 backdrop=["14120c", "221e14", "322c1e", "443c2a", "584e38", "70664a"],
                 sky=((124, 112, 84), (250, 232, 172))),
}

# --------------------------------------------------------------------------- pixel font

# 5x7 proportional pixel font: caps 7 rows, lowercase x-height 5, descenders 2 more rows.
_GLYPHS_SRC = {
    "A": ".###. #...# #...# ##### #...# #...# #...#",
    "B": "####. #...# #...# ####. #...# #...# ####.",
    "C": ".###. #...# #.... #.... #.... #...# .###.",
    "D": "####. #...# #...# #...# #...# #...# ####.",
    "E": "##### #.... #.... ####. #.... #.... #####",
    "F": "##### #.... #.... ####. #.... #.... #....",
    "G": ".###. #...# #.... #.### #...# #...# .####",
    "H": "#...# #...# #...# ##### #...# #...# #...#",
    "I": "### .#. .#. .#. .#. .#. ###",
    "J": "..### ...#. ...#. ...#. #..#. #..#. .##..",
    "K": "#...# #..#. #.#.. ##... #.#.. #..#. #...#",
    "L": "#.... #.... #.... #.... #.... #.... #####",
    "M": "#...# ##.## #.#.# #.#.# #...# #...# #...#",
    "N": "#...# ##..# #.#.# #..## #...# #...# #...#",
    "O": ".###. #...# #...# #...# #...# #...# .###.",
    "P": "####. #...# #...# ####. #.... #.... #....",
    "Q": ".###. #...# #...# #...# #.#.# #..#. .##.#",
    "R": "####. #...# #...# ####. #.#.. #..#. #...#",
    "S": ".###. #...# #.... .###. ....# #...# .###.",
    "T": "##### ..#.. ..#.. ..#.. ..#.. ..#.. ..#..",
    "U": "#...# #...# #...# #...# #...# #...# .###.",
    "V": "#...# #...# #...# #...# #...# .#.#. ..#..",
    "W": "#...# #...# #...# #.#.# #.#.# ##.## #...#",
    "X": "#...# #...# .#.#. ..#.. .#.#. #...# #...#",
    "Y": "#...# #...# .#.#. ..#.. ..#.. ..#.. ..#..",
    "Z": "##### ....# ...#. ..#.. .#... #.... #####",
    "a": ".... .... .##. ...# .### #..# .###",
    "b": "#... #... ###. #..# #..# #..# ###.",
    "c": ".... .... .### #... #... #... .###",
    "d": "...# ...# .### #..# #..# #..# .###",
    "e": ".... .... .##. #..# #### #... .###",
    "f": ".## #.. ### #.. #.. #.. #..",
    "g": ".... .... .### #..# #..# #..# .### ...# ###.",
    "h": "#... #... ###. #..# #..# #..# #..#",
    "i": "# . # # # # #",
    "j": ".# .. .# .# .# .# .# .# #.",
    "k": "#... #... #..# #.#. ##.. #.#. #..#",
    "l": "#. #. #. #. #. #. .#",
    "m": "..... ..... ####. #.#.# #.#.# #.#.# #.#.#",
    "n": ".... .... ###. #..# #..# #..# #..#",
    "o": ".... .... .##. #..# #..# #..# .##.",
    "p": ".... .... ###. #..# #..# #..# ###. #... #...",
    "q": ".... .... .### #..# #..# #..# .### ...# ...#",
    "r": "... ... #.# ##. #.. #.. #..",
    "s": ".... .... .### #... .##. ...# ###.",
    "t": ".#. .#. ### .#. .#. .#. ..#",
    "u": ".... .... #..# #..# #..# #..# .###",
    "v": "..... ..... #...# #...# .#.#. .#.#. ..#..",
    "w": "..... ..... #...# #...# #.#.# #.#.# .#.#.",
    "x": ".... .... #..# #..# .##. #..# #..#",
    "y": ".... .... #..# #..# #..# #..# .### ...# ###.",
    "z": ".... .... #### ...# .##. #... ####",
    "0": ".###. #...# #...# #...# #...# #...# .###.",
    "1": ".#. ##. .#. .#. .#. .#. ###",
    "2": ".###. #...# ....# ...#. ..#.. .#... #####",
    "3": ".###. #...# ....# ..##. ....# #...# .###.",
    "4": "...#. ..##. .#.#. #..#. ##### ...#. ...#.",
    "5": "##### #.... ####. ....# ....# #...# .###.",
    "6": ".###. #.... #.... ####. #...# #...# .###.",
    "7": "##### ....# ...#. ..#.. .#... .#... .#...",
    "8": ".###. #...# #...# .###. #...# #...# .###.",
    "9": ".###. #...# #...# .#### ....# ....# .###.",
    ".": ". . . . . . #",
    ",": ".. .. .. .. .. .. .# #.",
    "!": "# # # # # . #",
    "?": ".###. #...# ....# ...#. ..#.. ..... ..#..",
    ":": ". . # . . . #",
    ";": ".. .. .# .. .. .. .# #.",
    "-": "... ... ... ### ... ... ...",
    "+": "... ... .#. ### .#. ... ...",
    "=": "... ... ### ... ### ... ...",
    "/": "..# ..# .#. .#. .#. #.. #..",
    "(": ".# #. #. #. #. #. .#",
    ")": "#. .# .# .# .# .# #.",
    "'": "# # . . . . .",
    "\"": "#.# #.# ... ... ... ... ...",
    "%": "##..# ##..# ...#. ..#.. .#... #..## #..##",
    "*": "..... #.#.# .###. ##### .###. #.#.# .....",
    "#": ".#.#. ##### .#.#. .#.#. .#.#. ##### .#.#.",
    "&": ".##.. #..#. .##.. .#... #.#.# #..#. .##.#",
    "[": "## #. #. #. #. #. ##",
    "]": "## .# .# .# .# .# ##",
    "<": "...# ..#. .#.. #... .#.. ..#. ...#",
    ">": "#... .#.. ..#. ...# ..#. .#.. #...",
    "_": "..... ..... ..... ..... ..... ..... #####",
}
NAME_BOLD = False        # doubled-column bold smears w and m at 1x; names use a shadow instead
FONT_ROWS = 9            # cap height 7 + descender 2
FONT_CAP = 7
LINE_HEIGHT = 10
SPACE_W = 3


def _glyphs() -> dict[str, np.ndarray]:
    out = {}
    for ch, src in _GLYPHS_SRC.items():
        rows = src.split()
        w = max(len(r) for r in rows)
        g = np.zeros((FONT_ROWS, w), bool)
        for y, r in enumerate(rows):
            for x, c in enumerate(r):
                g[y, x] = c == "#"
        out[ch] = g
    out[" "] = np.zeros((FONT_ROWS, SPACE_W - 1), bool)
    return out


GLYPHS = _glyphs()
_ASCII = str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
                        "\u2013": "-", "\u2014": "-", "\u2026": "..."})


class PixelFont:
    """The built-in bitmap font drawn at whole-number scales; bold = each glyph doubled one
    pixel to the right. Unknown characters draw as '?'."""

    def __init__(self, glyphs: dict[str, np.ndarray] | None = None):
        self.glyphs = glyphs or GLYPHS

    def glyph(self, ch: str, bold: bool = False) -> np.ndarray:
        g = self.glyphs.get(ch)
        if g is None:
            g = self.glyphs["?"]
        if bold and g.any():
            b = np.zeros((g.shape[0], g.shape[1] + 1), bool)
            b[:, :-1] |= g
            b[:, 1:] |= g
            g = b
        return g

    def width(self, text: str, scale: int = 1, bold: bool = False, spacing: int = 1) -> int:
        text = str(text).translate(_ASCII)
        if not text:
            return 0
        w = sum(self.glyph(c, bold).shape[1] + spacing for c in text) - spacing
        return w * scale

    def mask(self, text: str, scale: int = 1, bold: bool = False, spacing: int = 1) -> np.ndarray:
        """(FONT_ROWS*scale, width) bool picture of one line."""
        text = str(text).translate(_ASCII)
        w = max(1, self.width(text, 1, bold, spacing))
        out = np.zeros((FONT_ROWS, w), bool)
        x = 0
        for c in text:
            g = self.glyph(c, bold)
            out[:, x:x + g.shape[1]] |= g
            x += g.shape[1] + spacing
        if scale > 1:
            out = np.repeat(np.repeat(out, scale, 0), scale, 1)
        return out

    def wrap(self, text: str, width, scale: int = 1, bold: bool = False) -> list[str]:
        """Words into lines no wider than width (px; or a function line_index -> px).
        Blank lines and explicit newlines are kept; a too-long word is split."""
        limit = width if callable(width) else (lambda i, w=width: w)
        lines: list[str] = []
        for para in str(text).translate(_ASCII).split("\n"):
            words = para.split()
            if not words:
                lines.append("")
                continue
            line = ""
            for word in words:
                trial = f"{line} {word}" if line else word
                if self.width(trial, scale, bold) <= limit(len(lines)):
                    line = trial
                    continue
                if line:
                    lines.append(line)
                line = word
                while self.width(line, scale, bold) > limit(len(lines)) and len(line) > 1:
                    k = len(line)
                    while k > 1 and self.width(line[:k], scale, bold) > limit(len(lines)):
                        k -= 1
                    lines.append(line[:k])
                    line = line[k:]
            lines.append(line)
        return lines

    def draw(self, img: np.ndarray, text: str, x: int, y: int, colour, scale: int = 1,
             bold: bool = False, outline=None, shadow=None, spacing: int = 1) -> None:
        """Draw at (x, y) = top-left of the cap height. outline: 1-scale-pixel ring colour;
        shadow: colour drawn one scale pixel down-right first."""
        m = self.mask(text, scale, bold, spacing)
        if outline is not None:
            k = max(1, scale // 2) if scale > 2 else 1
            ring = _dilate(np.pad(m, k), k)
            _blit(img, ring, x - k, y - k, outline)
        if shadow is not None:
            _blit(img, m, x + max(1, scale // 2), y + max(1, scale // 2), shadow)
        _blit(img, m, x, y, colour)


FONT = PixelFont()


def _blit(img: np.ndarray, mask: np.ndarray, x: int, y: int, colour) -> None:
    h, w = mask.shape
    H, W = img.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x0 >= x1 or y0 >= y1:
        return
    m = mask[y0 - y:y1 - y, x0 - x:x1 - x]
    region = img[y0:y1, x0:x1]
    region[m, :3] = _rgb(colour)
    region[m, 3] = 255


def font_atlas(scale: int = 1, bold: bool = False, outline: bool = False,
               colours=NUMBER_COLOURS) -> tuple[np.ndarray, dict]:
    """Every glyph in one RGBA atlas + BMFont-style metrics {char: (x, y, w, h, advance)}.
    outline: glyphs filled colours[0] ringed with colours[1] (stat numbers), else white."""
    pad = 1 if outline else 0
    chars = sorted(GLYPHS, key=ord)
    cell_h = FONT_ROWS * scale + 2 * pad
    per_row = 16
    widths = [FONT.glyph(c, bold).shape[1] * scale + 2 * pad for c in chars]
    cell_w = max(widths) + 1
    rows = math.ceil(len(chars) / per_row)
    atlas = np.zeros((rows * (cell_h + 1), per_row * cell_w, 4), np.uint8)
    metrics = {}
    for i, c in enumerate(chars):
        r, k = divmod(i, per_row)
        x, y = k * cell_w, r * (cell_h + 1)
        m = FONT.mask(c, scale, bold) if c != " " else np.zeros((FONT_ROWS * scale, 1), bool)
        if c == " ":
            m = np.zeros((FONT_ROWS * scale, (SPACE_W - 1) * scale), bool)
        if outline:
            ring = _dilate(np.pad(m, pad), pad)
            _blit(atlas, ring, x, y, colours[1])
            _blit(atlas, m, x + pad, y + pad, colours[0])
        else:
            _blit(atlas, m, x, y, "ffffff")
        adv = (FONT.glyph(c, bold).shape[1] + 1) * scale
        metrics[c] = (x, y, m.shape[1] + 2 * pad, m.shape[0] + 2 * pad, adv)
    return atlas, metrics


def write_bmfont(out_dir, name: str, scale: int = 1, bold: bool = False,
                 outline: bool = False) -> list[Path]:
    """<name>.fnt (BMFont text format: Godot imports it as a FontFile) + <name>.png."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    atlas, metrics = font_atlas(scale, bold, outline)
    png = out_dir / f"{name}.png"
    Image.fromarray(atlas, "RGBA").save(png)
    pad = 1 if outline else 0
    lines = [
        f'info face="PixelCard{" Bold" if bold else ""}" size={LINE_HEIGHT * scale} '
        f'bold={int(bold)} italic=0 charset="" unicode=1 stretchH=100 smooth=0 aa=1 '
        f'padding=0,0,0,0 spacing=1,1 outline={pad}',
        f"common lineHeight={LINE_HEIGHT * scale} base={FONT_CAP * scale} "
        f"scaleW={atlas.shape[1]} scaleH={atlas.shape[0]} pages=1 packed=0",
        f'page id=0 file="{png.name}"',
        f"chars count={len(metrics)}",
    ]
    for c, (x, y, w, h, adv) in metrics.items():
        lines.append(f"char id={ord(c)} x={x} y={y} width={w} height={h} xoffset={-pad} "
                     f"yoffset={-pad} xadvance={adv} page=0 chnl=15")
    fnt = out_dir / f"{name}.fnt"
    fnt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [fnt, png]


# --------------------------------------------------------------------------- drawing helpers

def _rgb(c) -> np.ndarray:
    if isinstance(c, str):
        c = c.lstrip("#")
        return np.array([int(c[i:i + 2], 16) for i in (0, 2, 4)], np.uint8)
    return np.asarray(c[:3], np.uint8)


def _ramp(hexes) -> list[np.ndarray]:
    return [_rgb(h) for h in hexes]


def _shift(m: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """out[y, x] = m[y - dy, x - dx] (False outside)."""
    H, W = m.shape
    out = np.zeros_like(m)
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    out[yd, xd] = m[ys, xs]
    return out


def _erode(m: np.ndarray, k: int = 1) -> np.ndarray:
    out = m.copy()
    for _ in range(k):
        out = out & _shift(out, 1, 0) & _shift(out, -1, 0) & _shift(out, 0, 1) & _shift(out, 0, -1)
    return out


def _dilate(m: np.ndarray, k: int = 1) -> np.ndarray:
    out = m.copy()
    for _ in range(k):
        n = out.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                n |= _shift(out, dy, dx)
        out = n
    return out


def _put(img: np.ndarray, mask: np.ndarray, colour) -> None:
    img[mask, :3] = _rgb(colour)
    img[mask, 3] = 255


def _grid(H: int, W: int):
    yy, xx = np.mgrid[0:H, 0:W]
    return yy + 0.5, xx + 0.5


def rrect_mask(H: int, W: int, x0: int, y0: int, x1: int, y1: int, r: float = 0) -> np.ndarray:
    """Rounded rectangle covering pixels x0..x1-1, y0..y1-1."""
    yy, xx = _grid(H, W)
    m = (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
    if r > 0:
        cx = np.clip(xx, x0 + r, x1 - r)
        cy = np.clip(yy, y0 + r, y1 - r)
        m &= (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r + 0.25
    return m


def circle_mask(H: int, W: int, cx: float, cy: float, r: float) -> np.ndarray:
    yy, xx = _grid(H, W)
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def poly_mask(H: int, W: int, pts) -> np.ndarray:
    import cv2
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(pts, np.float64)).astype(np.int32)], 1)
    return m > 0


def heart_mask(H: int, W: int, cx: float, cy: float, r: float) -> np.ndarray:
    yy, xx = _grid(H, W)
    x = (xx - cx) / (r * 1.05)
    y = -(yy - cy) / (r * 1.05) + 0.12
    return (x * x + y * y - 1) ** 3 - x * x * y ** 3 <= 0


def shield_mask(H: int, W: int, cx: float, cy: float, r: float) -> np.ndarray:
    top, mid, bot = cy - r * 0.95, cy + r * 0.2, cy + r * 1.02
    return poly_mask(H, W, [(cx - r, top), (cx + r, top), (cx + r, mid), (cx, bot),
                            (cx - r, mid)])


def paint_shape(img: np.ndarray, mask: np.ndarray, ramp, *, sunken: bool = False,
                bevel: int = 1, outline: bool = True, grain: float = 0.0, seed: int = 0) -> None:
    """Fill a mask as pixel art: 1px outline (ramp[0]) inside its edge, base ramp[3], bevel of
    `bevel` px lit top-left (ramp[4], corner shine ramp[5]) and shaded bottom-right (ramp[1]);
    sunken swaps them. grain = share of base pixels speckled lighter/darker (metal, paper)."""
    m = mask.astype(bool)
    if not m.any():
        return
    inner = _erode(m, 1) if outline else m
    if outline:
        _put(img, m & ~inner, ramp[0])
    _put(img, inner, ramp[3])
    if grain > 0:
        rng = np.random.default_rng(seed)
        r = rng.random(m.shape)
        _put(img, inner & (r < grain / 2), ramp[2])
        _put(img, inner & (r > 1 - grain / 2), ramp[4])
    hi, lo = (ramp[1], ramp[4]) if sunken else (ramp[4], ramp[1])
    for k in range(bevel, 0, -1):
        ul = inner & ~_shift(inner, k, k)
        dr = inner & ~_shift(inner, -k, -k)
        _put(img, dr & ~ul, lo)
        _put(img, ul & ~dr, hi)
        _put(img, ul & dr, ramp[2])
    if not sunken:
        corner = inner & ~_shift(inner, 1, 0) & ~_shift(inner, 0, 1)
        _put(img, corner, ramp[5])


def paint_gem(img: np.ndarray, mask: np.ndarray, ramp, cx: float, cy: float, r: float) -> None:
    """A rounded gem: outline, bands lit from the top-left, a highlight spot."""
    m = mask.astype(bool)
    inner = _erode(m, 1)
    _put(img, m & ~inner, ramp[0])
    yy, xx = _grid(*m.shape)
    v = -((xx - cx) + (yy - cy)) / (max(r, 1) * 1.414)       # +1 top-left .. -1 bottom-right
    idx = np.clip(np.floor((v + 1) / 2 * 3.999).astype(int) + 1, 1, 4)
    for i in range(1, 5):
        _put(img, inner & (idx == i), ramp[i])
    spot = circle_mask(*m.shape, cx - 0.38 * r, cy - 0.38 * r, max(0.8, 0.2 * r))
    _put(img, inner & spot, ramp[5])


# --------------------------------------------------------------------------- layout

def _u(size: str) -> float:
    if size not in SIZES:
        raise ValueError(f"Unknown card size '{size}' (use {', '.join(SIZES)})")
    return SIZES[size][0] / BASE_W


def _scaled(rect, u: float) -> dict:
    x0, y0, x1, y1 = rect
    X0, Y0, X1, Y1 = (round(v * u) for v in (x0, y0, x1, y1))
    return {"x": X0, "y": Y0, "w": X1 - X0, "h": Y1 - Y0}


def card_layout(size: str = "small", stats: str = "unit") -> dict:
    """Slot rects (x, y, w, h in card pixels) for a size and stats mode ("unit" = attack +
    health badges, "power" = one badge bottom centre, "none")."""
    if stats not in ("unit", "power", "none"):
        raise ValueError("stats must be unit, power or none")
    u = _u(size)
    W, H = SIZES[size]
    border = max(3, round(4 * u))
    gem_r = 12 * u
    badge_r = 11 * u
    slots = {
        "art": _scaled((9, 23, 117, 95), u),
        "name": _scaled((27, 8, 116, 20), u),
        "cost": _scaled((14 - 12, 14 - 12, 14 + 12, 14 + 12), u),
        "type": _scaled((13, 98, 102, 108), u),
        "rarity": _scaled((106, 98, 116, 108), u),
        "textbox": _scaled((10, 110, 116, 165), u),
        "text": _scaled((15, 114, 111, 151 if stats == "unit" else
                         (153 if stats == "power" else 161)), u),
        "attack": _scaled((14 - 11, 163 - 11, 14 + 11, 163 + 11), u) if stats == "unit" else None,
        "health": _scaled((112 - 11, 163 - 11, 112 + 11, 163 + 11), u) if stats == "unit" else None,
        "power": _scaled((63 - 10, 165 - 10, 63 + 10, 165 + 10), u) if stats == "power" else None,
        "banner": _scaled((18, 7, 120, 21), u),
        "typebar": _scaled((9, 97, 117, 109), u),
    }
    body = max(1, round(u + 0.01))     # rules text drops to 1x when it doesn't fit
    return {"size": size, "width": W, "height": H, "stats": stats, "border": border,
            "gem_radius": round(gem_r, 2), "badge_radius": round(badge_r, 2),
            "slots": slots,
            "font": {"text": body, "type": max(1, round(u + 0.01)), "name": max(1, round(u + 0.01)),
                     "number": max(1, int(((gem_r - max(2, round(2 * u))) * 2 * 0.72) // FONT_CAP)),
                     "line_height": LINE_HEIGHT, "cap": FONT_CAP}}


def _rect(slot: dict) -> tuple[int, int, int, int]:
    return slot["x"], slot["y"], slot["x"] + slot["w"], slot["y"] + slot["h"]


def _centre(slot: dict) -> tuple[float, float]:
    return slot["x"] + slot["w"] / 2, slot["y"] + slot["h"] / 2


# --------------------------------------------------------------------------- palette

def snap_image(rgba: np.ndarray, palette: np.ndarray | None) -> np.ndarray:
    """Every opaque colour -> the nearest palette colour (Lab). None = unchanged."""
    if palette is None or not len(palette):
        return rgba
    from .color import rgb_to_lab
    out = rgba.copy()
    on = out[..., 3] > 0
    cols, inv = np.unique(out[on][:, :3], axis=0, return_inverse=True)
    a = rgb_to_lab(cols.astype(np.float32))
    b = rgb_to_lab(np.asarray(palette, np.float32)[:, :3])
    idx = np.argmin(((a[:, None] - b[None]) ** 2).sum(-1), 1)
    out[on, :3] = np.asarray(palette, np.uint8)[idx][inv.reshape(-1)]
    return out


def _palette(project=None, style=None):
    from .effects import project_palette
    if style is None:
        style = (project or assets.current_project()).style
    return project_palette(style)


# --------------------------------------------------------------------------- frames

def render_frame(rarity: str = "common", faction: str = "neutral", size: str = "small",
                 stats: str = "unit", palette: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """The card frame (RGBA, art window transparent, gems and badges empty) + its layout."""
    if rarity not in METALS:
        raise ValueError(f"Unknown rarity '{rarity}' (use {', '.join(RARITIES)})")
    if faction not in FACTION_COLOURS:
        raise ValueError(f"Unknown faction '{faction}' (use {', '.join(FACTIONS)})")
    lay = card_layout(size, stats)
    W, H = lay["width"], lay["height"]
    u = W / BASE_W
    S = lay["slots"]
    metal = _ramp(METALS[rarity])
    fac = FACTION_COLOURS[faction]
    banner, parch, gem = _ramp(fac["banner"]), _ramp(fac["parchment"]), _ramp(fac["gem"])
    back = _ramp(fac["backdrop"])
    img = np.zeros((H, W, 4), np.uint8)
    b = lay["border"]
    bevel = max(1, round(u))
    seed = zlib.crc32(f"{rarity}{faction}{size}".encode())

    # card body: metal band, then the dark backdrop sunk into it
    body = rrect_mask(H, W, 0, 0, W, H, max(2, 4 * u))
    paint_shape(img, body, metal, bevel=bevel + (1 if rarity != "common" else 0),
                grain=0.035, seed=seed)
    inside = rrect_mask(H, W, b, b, W - b, H - b, max(1, 2 * u))
    paint_shape(img, inside, back, sunken=True, bevel=bevel)
    if rarity in ("epic", "legendary"):          # a second, inner metal line
        ring = rrect_mask(H, W, b + 1, b + 1, W - b - 1, H - b - 1, max(1, 2 * u))
        _put(img, ring & ~_erode(ring, 1), metal[4 if rarity == "legendary" else 3])

    # art window: metal bezel, then the hole
    ax0, ay0, ax1, ay1 = _rect(S["art"])
    bz = max(2, round(2 * u))
    paint_shape(img, rrect_mask(H, W, ax0 - bz, ay0 - bz, ax1 + bz, ay1 + bz, 0), metal,
                sunken=False, bevel=1)
    _put(img, rrect_mask(H, W, ax0 - 1, ay0 - 1, ax1 + 1, ay1 + 1), metal[0])
    img[ay0:ay1, ax0:ax1] = 0

    # name banner: a ribbon with a notched right end (its left end hides under the gem)
    bx0, by0, bx1, by1 = _rect(S["banner"])
    notch = max(2, round(4 * u))
    mid = (by0 + by1) / 2
    paint_shape(img, poly_mask(H, W, [(bx0, by0), (bx1, by0), (bx1 - notch, mid), (bx1, by1),
                                     (bx0, by1)]), banner, bevel=bevel)
    # type bar with the rarity gem at its right end
    tx0, ty0, tx1, ty1 = _rect(S["typebar"])
    paint_shape(img, rrect_mask(H, W, tx0, ty0, tx1, ty1, max(1, 2 * u)), banner, bevel=bevel)
    rc = _centre(S["rarity"])
    rr = S["rarity"]["h"] / 2
    paint_gem(img, poly_mask(H, W, [(rc[0], rc[1] - rr), (rc[0] + rr * 0.85, rc[1]),
                                    (rc[0], rc[1] + rr), (rc[0] - rr * 0.85, rc[1])]),
              _ramp(RARITY_GEMS[rarity]), rc[0], rc[1], rr)
    # rules text box: parchment
    x0, y0, x1, y1 = _rect(S["textbox"])
    paper = rrect_mask(H, W, x0, y0, x1, y1, max(1, 2 * u))
    paint_shape(img, paper, parch, sunken=True, bevel=bevel)

    # corner studs / rivets
    stud_r = max(1.0, 1.6 * u)
    for cx, cy in ((b / 2, H - b / 2), (W - b / 2, H - b / 2), (W - b / 2, b / 2)):
        m = circle_mask(H, W, cx, cy, stud_r + (0.6 if rarity != "common" else 0))
        if rarity in ("epic", "legendary"):
            paint_gem(img, m, _ramp(RARITY_GEMS[rarity]), cx, cy, stud_r)
        else:
            paint_gem(img, m, metal, cx, cy, stud_r)
    if rarity == "legendary":                    # crest on the bottom edge
        cx, cy = W / 2, H - b / 2
        paint_gem(img, poly_mask(H, W, [(cx, cy - 2.6 * u), (cx + 4 * u, cy), (cx, cy + 2.6 * u),
                                        (cx - 4 * u, cy)]), _ramp(RARITY_GEMS[rarity]), cx, cy,
                  3 * u)

    # stat badges
    if stats == "unit":
        c = _centre(S["attack"])
        paint_shape(img, shield_mask(H, W, c[0], c[1], lay["badge_radius"] * 0.92),
                    _ramp(ATTACK_RAMP), bevel=bevel)
        c = _centre(S["health"])
        paint_shape(img, heart_mask(H, W, c[0], c[1], lay["badge_radius"]),
                    _ramp(HEALTH_RAMP), bevel=bevel)
    elif stats == "power":
        c = _centre(S["power"])
        r = S["power"]["w"] / 2
        paint_shape(img, circle_mask(H, W, c[0], c[1], r), metal, bevel=bevel)
        paint_gem(img, circle_mask(H, W, c[0], c[1], r - max(2, round(2 * u))), gem,
                  c[0], c[1], r - 2)
    # cost gem: metal bezel + faction gem (over the art window's corner)
    c = _centre(S["cost"])
    r = lay["gem_radius"]
    paint_shape(img, circle_mask(H, W, c[0], c[1], r), metal, bevel=bevel)
    ir = r - max(2, round(2 * u))
    paint_gem(img, circle_mask(H, W, c[0], c[1], ir), gem, c[0], c[1], ir)

    return snap_image(img, palette), lay


def frame_colours(rarity: str, faction: str, palette=None) -> dict:
    """Text colours for a frame (hex), snapped like the frame."""
    fac = FACTION_COLOURS[faction]
    cols = {"name": fac["banner"][5], "name_shadow": fac["banner"][0], "type": fac["banner"][5],
            "type_shadow": fac["banner"][0], "ink": fac["ink"], "number": NUMBER_COLOURS[0],
            "number_outline": NUMBER_COLOURS[1]}
    if palette is not None and len(palette):
        arr = np.array([[_rgb(v) for v in cols.values()]], np.uint8)
        rgba = np.concatenate([arr, np.full(arr.shape[:2] + (1,), 255, np.uint8)], -1)
        snapped = snap_image(rgba, palette)[0, :, :3]
        cols = {k: "%02x%02x%02x" % tuple(int(c) for c in v) for k, v in zip(cols, snapped)}
    return cols


def nine_slices(rarity: str = "common", faction: str = "neutral", size: str = "small",
                palette=None) -> dict[str, tuple[np.ndarray, dict]]:
    """9-slice UI pieces in the card style: name -> (RGBA, margins {left, top, right, bottom}).
    border = the card's metal band (transparent middle), panel = parchment in metal,
    banner = the faction ribbon (stretch it sideways), dark = backdrop panel in metal."""
    u = _u(size)
    metal = _ramp(METALS[rarity])
    fac = FACTION_COLOURS[faction]
    b = max(3, round(4 * u))
    bevel = max(1, round(u))
    m = b + 2
    n = 3 * m
    out = {}

    def base():
        img = np.zeros((n, n, 4), np.uint8)
        paint_shape(img, rrect_mask(n, n, 0, 0, n, n, max(2, 4 * u)), metal, bevel=bevel)
        return img

    img = base()
    for cx, cy in ((b / 2, b / 2), (n - b / 2, b / 2), (b / 2, n - b / 2), (n - b / 2, n - b / 2)):
        paint_gem(img, circle_mask(n, n, cx, cy, max(1.0, 1.6 * u)), metal, cx, cy, 1.6 * u)
    img[b + 1:n - b - 1, b + 1:n - b - 1] = 0
    _put(img, rrect_mask(n, n, b, b, n - b, n - b) & ~rrect_mask(n, n, b + 1, b + 1, n - b - 1, n - b - 1),
         metal[0])
    out["border"] = (img, m)
    for name, ramp, grain in (("panel", _ramp(fac["parchment"]), 0.0),
                              ("dark", _ramp(fac["backdrop"]), 0.0)):
        img = base()
        paint_shape(img, rrect_mask(n, n, b, b, n - b, n - b, max(1, 2 * u)), ramp, sunken=True,
                    bevel=bevel, grain=grain)
        out[name] = (img, m)
    bh = max(10, round(14 * u))
    bw = 3 * bh
    img = np.zeros((bh, bw, 4), np.uint8)
    notch = max(2, round(4 * u))
    mid = bh / 2
    paint_shape(img, poly_mask(bh, bw, [(notch, 0), (bw - notch, 0), (bw - 1, 0), (bw - 1 - notch, mid),
                                        (bw - 1, bh), (0, bh), (notch, mid), (0, 0)]),
                _ramp(fac["banner"]), bevel=bevel)
    out["banner"] = (img, {"left": bh, "right": bh, "top": max(2, bh // 3),
                           "bottom": max(2, bh // 3)})
    res = {}
    for k, (im, mg) in out.items():
        margins = mg if isinstance(mg, dict) else {"left": mg, "top": mg, "right": mg, "bottom": mg}
        res[k] = (snap_image(im, palette), margins)
    return res


def _stylebox_tres(png_name: str, margins: dict) -> str:
    return "\n".join([
        '[gd_resource type="StyleBoxTexture" load_steps=2 format=3]', "",
        f'[ext_resource type="Texture2D" path="{png_name}" id="1_tex"]', "",
        "[resource]", 'texture = ExtResource("1_tex")',
        *[f"texture_margin_{s} = {float(margins[s])}" for s in ("left", "top", "right", "bottom")],
        ""])


def write_frame(out_dir, rarity: str = "common", faction: str = "neutral", size: str = "small",
                stats: str = "unit", palette=None, project=None, nine: bool = True) -> list[Path]:
    """frame.png + layout.json (slots, colours, font scales) + ui/ 9-slice pieces
    (<piece>.png, .json margins, .tres Godot StyleBoxTexture). palette: None = the project's."""
    if palette is None:
        palette = _palette(project)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    img, lay = render_frame(rarity, faction, size, stats, palette)
    lay = {**lay, "rarity": rarity, "faction": faction,
           "colours": frame_colours(rarity, faction, palette), "frame": "frame.png"}
    written = [out / "frame.png", out / "layout.json"]
    Image.fromarray(img, "RGBA").save(written[0])
    written[1].write_text(json.dumps(lay, indent=1), encoding="utf-8")
    if nine:
        ui = out / "ui"
        ui.mkdir(exist_ok=True)
        for name, (im, margins) in nine_slices(rarity, faction, size, palette).items():
            png = ui / f"{name}.png"
            Image.fromarray(im, "RGBA").save(png)
            js = ui / f"{name}.json"
            js.write_text(json.dumps({"image": png.name, "size": [im.shape[1], im.shape[0]],
                                      "patch_margin": margins}, indent=1), encoding="utf-8")
            tres = ui / f"{name}.tres"
            tres.write_text(_stylebox_tres(png.name, margins), encoding="utf-8")
            written += [png, js, tres]
    return written


# --------------------------------------------------------------------------- the card

@dataclass
class CardInfo:
    """What a card says. None stats = no badge."""
    name: str = "New card"
    cost: int | None = 1
    type: str = "Unit"
    text: str = ""
    attack: int | None = None
    health: int | None = None
    power: int | None = None
    rarity: str = "common"
    faction: str = "neutral"
    size: str = "small"

    @property
    def stats(self) -> str:
        if self.attack is not None or self.health is not None:
            return "unit"
        return "power" if self.power is not None else "none"

    @classmethod
    def from_dict(cls, d: dict) -> "CardInfo":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def _fit_scale(text: str, width: int, start: int, bold: bool) -> tuple[int, bool]:
    for s in range(max(1, start), 0, -1):
        for b in ((True, False) if bold else (False,)):
            if FONT.width(text, s, b) <= width:
                return s, b
    return 1, False


def _clip(text: str, width: int, scale: int, bold: bool) -> str:
    if FONT.width(text, scale, bold) <= width:
        return text
    while text and FONT.width(text + "..", scale, bold) > width:
        text = text[:-1]
    return text.rstrip() + ".."


def _rules_lines(text: str, lay: dict) -> tuple[list[str], int, bool]:
    """(lines, scale, overflow) of the rules text in its box: lines that reach down beside the
    stat badges are narrower; the scale drops if it doesn't fit."""
    S = lay["slots"]
    t, tb = S["text"], S["textbox"]
    a, h = S.get("attack"), S.get("health")
    overlap = 0
    if a is not None and h is not None:
        overlap = max(a["x"] + a["w"] - tb["x"], tb["x"] + tb["w"] - h["x"], 0)
    lines: list[str] = []
    for scale in range(lay["font"]["text"], 0, -1):
        lh = LINE_HEIGHT * scale
        rows = max(1, (t["h"] + (LINE_HEIGHT - FONT_CAP) * scale) // lh)
        if overlap:          # lines may run on between the badges, narrower
            rows = max(rows, (tb["y"] + tb["h"] - 2 * scale - t["y"] + 2 * scale) // lh)

        def width(i, scale=scale, lh=lh):
            bottom = t["y"] + i * lh + FONT_CAP * scale + 2 * scale
            if overlap and bottom > a["y"] + 2:
                return max(8, tb["w"] - 2 * overlap - 4 * scale)
            return t["w"]
        lines = FONT.wrap(text, width, scale) if text.strip() else []
        if len(lines) <= rows:
            return lines, scale, False
    lines = lines[:rows]
    lines[-1] = _clip(lines[-1] + "..", width(len(lines) - 1), 1, False)
    return lines, 1, True


def compose_card(info: CardInfo, art: np.ndarray | None = None, frame: np.ndarray | None = None,
                 layout: dict | None = None, palette=None) -> tuple[np.ndarray, dict]:
    """art (any size: cover-cropped to the window) under the frame, then the text. Returns
    (RGBA card, text info: lines and scales used)."""
    if frame is None or layout is None:
        frame, layout = render_frame(info.rarity, info.faction, info.size, info.stats, palette)
    lay = layout
    S = lay["slots"]
    W, H = lay["width"], lay["height"]
    out = np.zeros((H, W, 4), np.uint8)
    a = S["art"]
    if art is not None:
        out[a["y"]:a["y"] + a["h"], a["x"]:a["x"] + a["w"]] = fit_art(art, a["w"], a["h"])
    else:
        out[a["y"]:a["y"] + a["h"], a["x"]:a["x"] + a["w"]] = placeholder_art(
            a["w"], a["h"], info.faction)
    on = frame[..., 3] > 0
    out[on] = frame[on]
    cols = frame_colours(info.rarity, info.faction, palette)
    used = {}

    # name: biggest scale that fits, bold if it fits, else clipped
    n = S["name"]
    name = str(info.name or "")
    sc, bold = _fit_scale(name, n["w"], lay["font"]["name"], NAME_BOLD)
    name = _clip(name, n["w"], sc, bold)
    y = n["y"] + (n["h"] - FONT_CAP * sc) // 2
    FONT.draw(out, name, n["x"], y, cols["name"], sc, bold, shadow=cols["name_shadow"])
    used["name"] = {"scale": sc, "bold": bold, "text": name}
    # type line
    t = S["type"]
    typ = str(info.type or "")
    sc, bold = _fit_scale(typ, t["w"], lay["font"]["type"], False)
    typ = _clip(typ, t["w"], sc, False)
    FONT.draw(out, typ, t["x"], t["y"] + (t["h"] - FONT_CAP * sc) // 2, cols["type"], sc,
              shadow=cols["type_shadow"])
    used["type"] = {"scale": sc, "text": typ}
    # rules text
    lines, sc, overflow = _rules_lines(str(info.text or ""), lay)
    tx = S["text"]
    block = len(lines) * LINE_HEIGHT * sc - (LINE_HEIGHT - FONT_CAP) * sc
    top = tx["y"] + max(0, (tx["h"] - block) // 2) if block <= tx["h"] else tx["y"]
    for i, line in enumerate(lines):            # centred both ways, like most card games
        x = tx["x"] + (tx["w"] - FONT.width(line, sc)) // 2
        FONT.draw(out, line, x, top + i * LINE_HEIGHT * sc, cols["ink"], sc)
    used["text"] = {"scale": sc, "lines": lines, "overflow": overflow}
    # numbers
    num = lay["font"]["number"]
    for key, value, k in (("cost", info.cost, num), ("attack", info.attack, num),
                          ("health", info.health, num), ("power", info.power, num)):
        s = S.get(key)
        if s is None or value is None:
            continue
        txt = str(value)
        sp = 1
        room = (2 * (lay["gem_radius"] - max(2, round(2 * W / BASE_W))) + 2) if key == "cost"             else int(s["w"] * 0.8)
        while k > 1 and FONT.width(txt, k, spacing=sp) > room:
            k -= 1
        cx, cy = _centre(s)
        dy = -0.5 if key == "health" else (-1.0 * k / 2 if key == "attack" else 0)
        x = round(cx - FONT.width(txt, k, spacing=sp) / 2)
        y = round(cy - FONT_CAP * k / 2 + dy)
        FONT.draw(out, txt, x, y, cols["number"], k, outline=cols["number_outline"], spacing=sp)
        used[key] = {"scale": k}
    return out, used


def fit_art(art: np.ndarray, w: int, h: int, smooth: bool | None = None) -> np.ndarray:
    """Cover-crop art to w x h: crisp (nearest, whole-number) for pixel art that already has the
    size or a multiple, else smooth (Lanczos)."""
    art = np.asarray(art)
    if art.shape[2] == 3:
        art = np.concatenate([art, np.full(art.shape[:2] + (1,), 255, np.uint8)], -1)
    ah, aw = art.shape[:2]
    if (aw, ah) == (w, h):
        return art.copy()
    k = max(w / aw, h / ah)
    pixel = smooth is False or (smooth is None and k >= 1 and abs(k - round(k)) < 1e-6)
    im = Image.fromarray(art, "RGBA")
    nw, nh = max(w, round(aw * k)), max(h, round(ah * k))
    im = im.resize((nw, nh), Image.NEAREST if pixel else Image.LANCZOS)
    x0, y0 = (nw - w) // 2, (nh - h) // 3          # keep heads: crop more from the bottom
    out = np.asarray(im.crop((x0, y0, x0 + w, y0 + h))).copy()
    out[..., 3] = 255
    return out


def placeholder_art(w: int, h: int, faction: str = "neutral", picture=None) -> np.ndarray:
    """No art yet: the faction's sky gradient (+ the subject's picture if given)."""
    top, bot = (np.array(c, np.float32) for c in FACTION_COLOURS[faction]["sky"])
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    rgb = top * (1 - t) + bot * t
    rgb = np.broadcast_to(rgb, (h, w, 3))
    out = np.concatenate([np.rint(rgb).astype(np.uint8), np.full((h, w, 1), 255, np.uint8)], -1)
    if picture is not None:
        pic = np.asarray(picture)
        im = Image.fromarray(pic, "RGBA")
        box = im.getbbox()
        if box:
            im = im.crop(box)
        k = min(w * 0.9 / im.width, h * 0.9 / im.height)
        im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))),
                       Image.NEAREST if k >= 1 else Image.LANCZOS)
        base = Image.fromarray(out, "RGBA")
        base.alpha_composite(im, ((w - im.width) // 2, h - im.height - max(1, h // 20)))
        out = np.asarray(base).copy()
    return out


# --------------------------------------------------------------------------- sources

@dataclass
class Source:
    """Where a card's art comes from: kind character/item/prop/words, the asset folder (not for
    words), its name and the words describing it."""
    kind: str = "words"
    folder: str = ""
    name: str = ""
    prompt: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def source_from(kind: str, ref=None, words: str = "") -> Source:
    """kind "character" (library Character, folder or name), "item"/"prop" (Asset or folder),
    or "words" (ref ignored; the words are the subject)."""
    if kind not in SOURCE_KINDS:
        raise ValueError(f"Unknown source '{kind}' (use {', '.join(SOURCE_KINDS)})")
    if kind == "words":
        text = (words or str(ref or "")).strip()
        if not text:
            raise ValueError("Describe the card's picture in words, e.g. 'a fireball'.")
        return Source("words", "", text, text)
    if kind == "character":
        from .portraits import _character, character_prompt
        ch = _character(ref)
        return Source("character", str(ch.folder), ch.name, words or character_prompt(ch))
    folder = Path(getattr(ref, "folder", ref))
    meta = folder / "asset.json"
    if not meta.exists():
        raise ValueError(f"No {kind} asset in {folder}")
    d = json.loads(meta.read_text(encoding="utf-8"))
    return Source(kind, str(folder), d.get("name", folder.name),
                  words or d.get("extra", {}).get("prompt") or d.get("name", folder.name))


def source_picture(src: Source) -> np.ndarray | None:
    """The subject cut out (RGBA) from the source's own pictures, or None (words / nothing)."""
    from .pipeline import isolate, load_image
    from .settings import settings_for
    if src.kind == "words" or not src.folder:
        return None
    f = Path(src.folder)
    if src.kind == "character":
        candidates = [f / "picture.png"]
    elif src.kind == "item":
        try:
            choice = int(json.loads((f / "asset.json").read_text(encoding="utf-8"))
                         .get("extra", {}).get("icon_choice", 1))
        except (OSError, ValueError):
            choice = 1
        candidates = [f / f"picture_v{choice}.png", f / "picture_v1.png", f / "icon.png"]
    else:
        candidates = [f / "model.front.png", f / "drawn_v1.png", f / "sprite_v1.png", f / "thumb.png"]
    for p in candidates:
        if p.exists():
            rgba = load_image(p)
            return isolate(rgba, settings_for("object"))
    if src.kind == "character":
        from .portraits import _character, _front_sprite
        return _front_sprite(_character(f))
    return None


def default_info(src: Source) -> CardInfo:
    """Card fields to start from: units from characters (stats from the library's class),
    items and props as items/relics, words as a spell."""
    if src.kind == "character":
        from .portraits import _character
        try:
            hp, atk, *_ = _character(src.folder).stats()
        except Exception:  # noqa: BLE001 - a broken character still makes a card
            hp, atk = 28, 10
        attack, health = max(1, round(atk / 3)), max(1, round(hp / 7))
        return CardInfo(src.name, max(1, round((attack + health) / 2)), "Unit", "",
                        attack, health)
    if src.kind == "item":
        return CardInfo(src.name, 1, "Item", "")
    if src.kind == "prop":
        return CardInfo(src.name, 2, "Relic", "")
    return CardInfo(src.name.title()[:24], 2, "Spell", "")


# --------------------------------------------------------------------------- art (GPU)

ART_PROMPT = ("{subject}, {action}{scene}, fantasy trading card game illustration, dramatic "
              "composition, cinematic lighting, rim light, painterly, highly detailed digital "
              "painting, vivid colours, epic fantasy art")
PIXEL_ART_PROMPT = ("pixel art scene, {subject}, {action}{scene}, detailed background, fantasy "
                    "card game illustration, detailed 16-bit pixel art, crisp pixels, vivid "
                    "colours, dramatic lighting")
ART_NEGATIVE = ("text, letters, words, caption, watermark, signature, logo, border, frame, card "
                "frame, user interface, multiple panels, collage, split screen, blurry, lowres, "
                "jpeg artifacts, deformed, extra limbs, extra arms, bad hands, cropped head, "
                "out of frame, plain white background, two people, duplicate, clone, twins, crowd")
PAINTED_NEGATIVE = ART_NEGATIVE + ", pixel art, sprite, photo, photorealistic, 3d render"
PIXEL_NEGATIVE = (ART_NEGATIVE + ", photo, realistic, 3d render, smooth gradient, noise, white "
                  "background, plain background, simple background, isolated object, sprite, "
                  "game asset, icon, pedestal")
THING_NEGATIVE = "many objects, several bottles, clutter, collection, set, still life of many items"
SCENES = {
    "character": "on a windswept battlefield at sunset, banners and embers in the air",
    "item": "a single one standing alone as the centrepiece on an old wooden table, glowing "
            "magical light, dark softly blurred background, bokeh",
    "prop": "in a fantasy landscape, soft magical light",
    "words": "swirling magical energy, dark dramatic background",
}
ACTIONS = {"character": "dynamic action pose", "item": "", "prop": "", "words": ""}
# pictures: pass 1 paints the scene around the untouched subject (inpaint, SCENE_DENOISE), pass
# 2 repaints everything lightly (DENOISE) so the subject takes the scene's light and the style
SCENE_DENOISE = 1.0
DENOISE = {("character", "painted"): 0.42, ("character", "pixel"): 0.6,
           ("item", "painted"): 0.48, ("item", "pixel"): 0.55,
           ("prop", "painted"): 0.55, ("prop", "pixel"): 0.6}
PIXEL_OVER_DENOISE = 0.62   # words + pixel: the LoRA redraw over the clean scene
ART_ASPECT = 1.5          # the art window (108 x 72 at the base size)


def art_prompt(src: Source, style: str = "painted", scene: str | None = None,
               action: str | None = None, house=None) -> tuple[str, str]:
    """(positive, negative) for a card illustration of the source, with the project's house
    style (assets.with_style; house = a Style, None = the current project's) appended."""
    if style not in ART_STYLES:
        raise ValueError(f"Unknown art style '{style}' (use {', '.join(ART_STYLES)})")
    subject = re.sub(r"\s+", " ", src.prompt or src.name).strip().rstrip(".,; ")
    scene = SCENES[src.kind] if scene is None else scene.strip()
    action = ACTIONS[src.kind] if action is None else action.strip()
    template = ART_PROMPT if style == "painted" else PIXEL_ART_PROMPT
    text = template.format(subject=subject, action=f"{action}, " if action else "",
                           scene=scene or "dramatic background")
    negative = PAINTED_NEGATIVE if style == "painted" else PIXEL_NEGATIVE
    if src.kind in ("item", "prop"):
        negative += ", " + THING_NEGATIVE
    with_style = getattr(assets, "with_style", None)
    if with_style is not None:
        try:
            house = house or assets.project_style()
            text, negative = with_style(text, negative, house,
                                        "portrait" if style == "painted" else "sprite")
        except (ValueError, TypeError):
            pass
    return text, negative


def art_init(picture: np.ndarray, kind: str, faction: str = "neutral",
             size: tuple[int, int] = (1216, 832), with_mask: bool = False):
    """The subject placed on a scene-coloured backdrop, big and card-composed: characters from
    the head to the thighs, things centred and lit from behind. with_mask: also the subject's
    coverage (H, W) float 0..1 where it was placed."""
    W, H = size
    top, bot = (np.array(c, np.float32) for c in FACTION_COLOURS[faction]["sky"])
    t = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    rgb = top * (1 - t) + bot * t
    glow = np.exp(-(((xx - W / 2) / (W * 0.32)) ** 2 + ((yy - H * 0.45) / (H * 0.4)) ** 2))[..., None]
    rgb = rgb * (1 - 0.35 * glow) + 255 * 0.35 * glow
    base = Image.fromarray(np.clip(np.rint(rgb), 0, 255).astype(np.uint8), "RGB").convert("RGBA")
    im = Image.fromarray(np.asarray(picture), "RGBA")
    box = im.getbbox()
    if box:
        im = im.crop(box)
    if kind == "character":
        k = min(H * 1.5 / im.height, W * 0.95 / im.width)
        im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.LANCZOS)
        pos = ((W - im.width) // 2, round(H * 0.05))
    else:
        k = min(H * 0.76 / im.height, W * 0.55 / im.width)
        resample = Image.NEAREST if k >= 2 else Image.LANCZOS
        im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), resample)
        pos = ((W - im.width) // 2, (H - im.height) // 2 + round(H * 0.03))
    base.alpha_composite(im, pos)
    if not with_mask:
        return base.convert("RGB")
    cover = Image.new("L", (W, H), 0)
    cover.paste(im.getchannel("A"), pos)
    return base.convert("RGB"), np.asarray(cover, np.float32) / 255


def background_mask(cover: np.ndarray, grow: float = 0.012, feather: float = 0.01) -> np.ndarray:
    """Subject coverage -> inpaint mask (1 = paint): everything but the subject, eaten into its
    edge a little (grow, share of the height) and feathered so the scene meets it softly."""
    import cv2
    H = cover.shape[0]
    solid = (cover > 0.5).astype(np.uint8)
    g = max(1, round(grow * H))
    keep = cv2.erode(solid, np.ones((2 * g + 1, 2 * g + 1), np.uint8))
    f = max(1, round(feather * H)) * 2 + 1
    keep = cv2.GaussianBlur(keep.astype(np.float32), (f, f), 0)
    return np.clip(1.0 - keep, 0.0, 1.0)


def to_pixel_art(picture, w: int, h: int, style=None, colours: int | None = None,
                 palette: np.ndarray | None = None):
    """Any picture -> pixel-art Sprite exactly w x h (cover crop) in the project's colours
    (no GPU: the converter)."""
    from .pipeline import finish, load_image, prepare
    from .settings import settings_for
    st = style or assets.current_project().style
    s = settings_for("background", width=w, height=h, colors=colours or st.colours,
                     palette=st.palette, dither="none", fit="cover", anchor="center")
    rgba = load_image(picture)
    rgba[..., 3] = 255
    return finish(prepare(rgba, s), s, palette)


def generate_art(src: Source, style: str = "painted", *, count: int = VARIATIONS,
                 faction: str = "neutral", scene: str | None = None, action: str | None = None,
                 denoise: float | None = None, seed: int | None = None, client=None,
                 progress=None, **gen_opts) -> list[tuple[Image.Image, int]]:
    """Raw illustrations [(image, seed)] from the GPU: img2img over the subject's picture
    (art_init) when the source has one, else txt2img from the words."""
    from .generate import generate_images
    say = progress or (lambda t: None)
    positive, negative = art_prompt(src, style, scene, action)
    picture = source_picture(src)
    seed = int(seed if seed is not None else zlib.crc32((src.folder or src.prompt).encode())
               & 0x7FFFFFFF)
    gen_opts = dict(gen_opts)
    if style == "painted":
        gen_opts["lora"] = "none"          # the painted look: SDXL alone
    elif gen_opts.get("lora") == "none":
        gen_opts.pop("lora")               # pixel style always uses the pixel LoRA
    what = "painting" if style == "painted" else "drawing pixel art of"
    say(f"Card art: {what} {src.name} ({count} variation{'s' if count > 1 else ''}) on your GPU…")
    if picture is not None:
        init, cover = art_init(picture, src.kind, faction, with_mask=True)
        scene_mask = background_mask(cover)
        dn = denoise if denoise is not None else DENOISE.get((src.kind, style), 0.6)
        scene_opts = {**gen_opts, "lora": "none"}       # the scene pass is always clean
        out = []
        for i in range(count):
            say(f"Card art {i + 1}/{count}: the scene around {src.name}…")
            (scene, s), = generate_images(positive, negative, client=client, seed=seed + i,
                                          init_image=init, denoise=SCENE_DENOISE, mask=scene_mask,
                                          aspect=init.width / init.height, **scene_opts)
            say(f"Card art {i + 1}/{count}: painting it together…")
            (img, _), = generate_images(positive, negative, client=client, seed=seed + i,
                                        init_image=scene.convert("RGB"), denoise=dn,
                                        aspect=init.width / init.height, **gen_opts)
            out.append((img, s))
        return out
    if style == "painted":
        return generate_images(positive, negative, client=client, seed=seed, count=count,
                               aspect=ART_ASPECT, **gen_opts)
    # pixel from words: the pixel LoRA alone draws lone sprites on white, so the scene is
    # composed clean first and then redrawn as pixel art over it
    clean_pos, clean_neg = art_prompt(src, "painted", scene, action)
    out = []
    for i in range(count):
        say(f"Card art {i + 1}/{count}: the scene…")
        (scene_img, s), = generate_images(clean_pos, clean_neg, client=client, seed=seed + i,
                                          aspect=ART_ASPECT, **{**gen_opts, "lora": "none"})
        say(f"Card art {i + 1}/{count}: as pixel art…")
        (img, _), = generate_images(positive, negative, client=client, seed=seed + i,
                                    init_image=scene_img.convert("RGB"),
                                    denoise=denoise if denoise is not None else PIXEL_OVER_DENOISE,
                                    aspect=ART_ASPECT, **gen_opts)
        out.append((img, s))
    return out


# --------------------------------------------------------------------------- card assets

def _info_of(asset) -> CardInfo:
    return CardInfo.from_dict(asset.extra.get("card", {}))


def _art_path(asset) -> Path:
    return asset.folder / "art.png"


def new_card(info: CardInfo | dict | None = None, source: Source | None = None, *,
             art_style: str = "painted", scene: str | None = None, action: str | None = None,
             project=None) -> "assets.Asset":
    """A card asset (no art yet: a placeholder from the source's picture), composed and
    written. Make the art with make_art (GPU) or set_art."""
    register_kind()
    if isinstance(info, dict):
        info = CardInfo.from_dict(info)
    source = source or Source("words", "", (info.name if info else "card"),
                              info.name if info else "a magic card")
    info = info or default_info(source)
    _check(info)
    proj = project or assets.current_project()
    asset = assets.new_asset(KIND, info.name, proj, card=asdict(info), source=source.to_dict(),
                             art_style=art_style, scene=scene, action=action)
    render_card(asset, project=proj)
    return asset


def _check(info: CardInfo) -> None:
    if info.rarity not in METALS:
        raise ValueError(f"Unknown rarity '{info.rarity}' (use {', '.join(RARITIES)})")
    if info.faction not in FACTION_COLOURS:
        raise ValueError(f"Unknown faction '{info.faction}' (use {', '.join(FACTIONS)})")
    _u(info.size)
    for k in ("cost", "attack", "health", "power"):
        v = getattr(info, k)
        if v is not None and not isinstance(v, int):
            try:
                setattr(info, k, int(v))
            except (TypeError, ValueError):
                raise ValueError(f"{k} must be a whole number or empty") from None


def open_card(folder) -> "assets.Asset":
    folder = Path(getattr(folder, "folder", folder))
    d = json.loads((folder / "asset.json").read_text(encoding="utf-8"))
    if d.get("kind") != KIND:
        raise ValueError(f"{folder} is a {d.get('kind')} asset, not a card")
    return assets.Asset(KIND, d.get("name", folder.name), folder, d.get("tags", []),
                        d.get("collections", []), d.get("created", 0.0), d.get("extra", {}))


def list_cards(project=None) -> list:
    register_kind()
    return assets.list_assets(KIND, project=project)


def update_card(asset, project=None, **changes):
    """Change card fields (name, cost, type, text, attack, health, power, rarity, faction,
    size) and/or art_style/scene/action, then re-compose (art is re-fitted, no GPU)."""
    info = _info_of(asset)
    for k, v in changes.items():
        if k in ("art_style", "scene", "action"):
            asset.extra[k] = v
        elif hasattr(info, k):
            setattr(info, k, v)
        else:
            raise TypeError(f"Unknown card field '{k}'")
    _check(info)
    if info.name and info.name != asset.name:
        asset.name = info.name
    asset.extra["card"] = asdict(info)
    render_card(asset, project=project)
    return asset


def _art_window(info: CardInfo) -> tuple[int, int]:
    a = card_layout(info.size, info.stats)["slots"]["art"]
    return a["w"], a["h"]


def _finish_art(raw: Image.Image, style: str, w: int, h: int, st, palette) -> np.ndarray:
    """A raw illustration -> art at the window size (pixel style: converter, project colours)."""
    if style == "pixel":
        return to_pixel_art(raw, w, h, st, palette=palette).rgba
    return fit_art(np.asarray(raw.convert("RGBA")), w, h, smooth=True)


def make_art(asset, *, count: int = VARIATIONS, style: str | None = None, client=None,
             progress=None, project=None, seed: int | None = None, denoise: float | None = None,
             **gen_opts):
    """GPU: `count` illustrations for the card (art_raw_v*.png + art_v*.png at the window),
    the first chosen. Returns the asset."""
    say = progress or (lambda t: None)
    proj = project or assets.current_project()
    info = _info_of(asset)
    style = style or asset.extra.get("art_style", "painted")
    src = Source(**asset.extra.get("source", {}))
    results = generate_art(src, style, count=count, faction=info.faction,
                           scene=asset.extra.get("scene"), action=asset.extra.get("action"),
                           seed=seed, denoise=denoise, client=client, progress=say, **gen_opts)
    for old in asset.folder.glob("art_*v*.png"):
        old.unlink()
    for i, (img, _) in enumerate(results, 1):
        img.convert("RGB").save(asset.folder / f"art_raw_v{i}.png")
    asset.extra.update(art_style=style, seeds=[s for _, s in results], art_choice=1,
                       art_count=len(results))
    say("Fitting the art to the frame…")
    refit_art(asset, project=proj, compose=False)
    choose_art(asset, 1, project=proj)
    say(f"Card '{asset.name}': {len(results)} art variations")
    return asset


def refit_art(asset, project=None, compose: bool = True) -> list[Path]:
    """Re-make art_v*.png from art_raw_v*.png at the card's art window in its art style (after
    a size or style change, or a new project palette). No GPU."""
    proj = project or assets.current_project()
    st = proj.style
    info = _info_of(asset)
    w, h = _art_window(info)
    style = asset.extra.get("art_style", "painted")
    palette = _palette(style=st)
    out = []
    for raw in sorted(asset.folder.glob("art_raw_v*.png"), key=lambda p: int(re.sub(r"\D", "", p.stem))):
        i = int(re.sub(r"\D", "", raw.stem))
        art = _finish_art(Image.open(raw), style, w, h, st, palette)
        p = asset.folder / f"art_v{i}.png"
        Image.fromarray(art, "RGBA").save(p)
        out.append(p)
    if compose and out:
        choose_art(asset, int(asset.extra.get("art_choice", 1)), project=proj)
    return out


HD = 4      # art_hd.png = the chosen art at 4x the window (painted art for higher-res cards)


def _write_hd(asset) -> Path | None:
    """art_hd.png: the chosen raw art cover-cropped to HD x the art window (painted style;
    pixel style = the pixel art enlarged crisp)."""
    info = _info_of(asset)
    w, h = _art_window(info)
    i = int(asset.extra.get("art_choice", 1))
    raw = asset.folder / f"art_raw_v{i}.png"
    out = asset.folder / "art_hd.png"
    if asset.extra.get("art_style", "painted") == "pixel" and _art_path(asset).exists():
        art = np.asarray(Image.open(_art_path(asset)).convert("RGBA"))
        hd = np.repeat(np.repeat(art, HD, 0), HD, 1)
    elif raw.exists():
        hd = fit_art(np.asarray(Image.open(raw).convert("RGBA")), w * HD, h * HD, smooth=True)
    else:
        return None
    Image.fromarray(hd, "RGBA").save(out)
    return out


def art_variations(asset) -> list[Path]:
    return sorted(asset.folder.glob("art_v*.png"), key=lambda p: int(re.sub(r"\D", "", p.stem)))


def choose_art(asset, index: int, project=None) -> Path:
    src = asset.folder / f"art_v{index}.png"
    if not src.exists():
        raise ValueError(f"{asset.name} has no art variation {index}")
    shutil.copyfile(src, _art_path(asset))
    asset.extra["art_choice"] = index
    _write_hd(asset)
    render_card(asset, project=project)
    return _art_path(asset)


def set_art(asset, picture, style: str | None = None, project=None) -> Path:
    """Use your own picture as the card's art (fitted to the window; pixel style converts)."""
    from .pipeline import load_image
    proj = project or assets.current_project()
    rgba = load_image(picture)
    Image.fromarray(rgba, "RGBA").convert("RGB").save(asset.folder / "art_raw_v1.png")
    for old in asset.folder.glob("art_raw_v*.png"):
        if old.name != "art_raw_v1.png":
            old.unlink()
    for old in asset.folder.glob("art_v*.png"):
        old.unlink()
    if style:
        asset.extra["art_style"] = style
    asset.extra.update(art_choice=1, art_count=1, seeds=[])
    refit_art(asset, project=proj, compose=False)
    return choose_art(asset, 1, project=proj)


def render_card(asset, project=None) -> list[Path]:
    """frame.png, layout.json, ui/, card.png (+2x/4x), card.json and thumb.png from the
    asset's fields and art.png (placeholder from the source picture if there's none)."""
    proj = project or assets.current_project()
    palette = _palette(proj)
    info = _info_of(asset)
    f = asset.folder
    written = write_frame(f, info.rarity, info.faction, info.size, info.stats, palette)
    lay = json.loads((f / "layout.json").read_text(encoding="utf-8"))
    frame = np.asarray(Image.open(f / "frame.png").convert("RGBA"))
    art = None
    if _art_path(asset).exists():
        art = np.asarray(Image.open(_art_path(asset)).convert("RGBA"))
        a = lay["slots"]["art"]
        if art.shape[:2] != (a["h"], a["w"]):      # size changed: re-fit from the raw art
            if any(f.glob("art_raw_v*.png")):
                refit_art(asset, project=proj, compose=False)
                shutil.copyfile(f / f"art_v{int(asset.extra.get('art_choice', 1))}.png",
                                _art_path(asset))
                art = np.asarray(Image.open(_art_path(asset)).convert("RGBA"))
    if art is None:
        a = lay["slots"]["art"]
        try:
            pic = source_picture(Source(**asset.extra.get("source", {})))
        except Exception:  # noqa: BLE001 - a missing source still gets a card
            pic = None
        art = placeholder_art(a["w"], a["h"], info.faction, pic)
    card, used = compose_card(info, art, frame, lay, palette)
    Image.fromarray(card, "RGBA").save(f / "card.png")
    for k in (2, 4):
        Image.fromarray(card, "RGBA").resize((card.shape[1] * k, card.shape[0] * k),
                                             Image.NEAREST).save(f / f"card_{k}x.png")
    data = card_data(asset, info, lay, used)
    (f / "card.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    _thumb(card, f / "thumb.png")
    asset.extra["files"] = sorted(p.name for p in f.iterdir() if p.is_file())
    asset.save()
    tags = {"card", info.rarity, info.faction, info.type.split()[0].lower() if info.type else "card"}
    assets.set_tags(asset, sorted(tags | set(asset.tags)))
    return written + [f / "card.png", f / "card.json", f / "thumb.png"]


def card_data(asset, info: CardInfo, lay: dict, used: dict) -> dict:
    """card.json: everything a game needs to show the card (missing stats are left out)."""
    d = {"id": asset.folder.name, "name": info.name, "type": info.type, "text": info.text,
         "rarity": info.rarity, "faction": info.faction, "size": info.size,
         "art": "art.png" if _art_path(asset).exists() else None,
         "art_hd": "art_hd.png" if (asset.folder / "art_hd.png").exists() else None,
         "frame": "frame.png",
         "card": "card.png", "art_style": asset.extra.get("art_style", "painted"),
         "source": asset.extra.get("source", {}),
         "fonts": {"name": used["name"]["scale"], "name_bold": used["name"].get("bold", False),
                   "type": used["type"]["scale"], "text": used["text"]["scale"],
                   "number": lay["font"]["number"]},
         "layout": {"width": lay["width"], "height": lay["height"], "stats": lay["stats"],
                    "slots": {k: v for k, v in lay["slots"].items() if v is not None},
                    "colours": lay["colours"]}}
    for k in ("cost", "attack", "health", "power"):
        v = getattr(info, k)
        if v is not None:
            d[k] = int(v)
    return d


def _thumb(card: np.ndarray, out: Path, px: int = 96) -> Path:
    im = Image.fromarray(card, "RGBA")
    k = math.ceil(max(im.width, im.height) / px)
    if k > 1:
        im = im.reduce(k)
    canvas = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    canvas.alpha_composite(im, ((px - im.width) // 2, (px - im.height) // 2))
    canvas.save(out)
    return out


def load_card_image(asset, scale: int = 1) -> np.ndarray:
    p = asset.folder / ("card.png" if scale == 1 else f"card_{scale}x.png")
    return np.asarray(Image.open(p).convert("RGBA")).copy()


def make_card(source: Source, info: CardInfo | None = None, *, art_style: str = "painted",
              count: int = VARIATIONS, client=None, progress=None, project=None, scene=None,
              action=None, **gen_opts):
    """new_card + make_art in one go (GPU)."""
    asset = new_card(info or default_info(source), source, art_style=art_style, scene=scene,
                     action=action, project=project)
    return make_art(asset, count=count, client=client, progress=progress, project=project,
                    **gen_opts)


def contact_sheet(cards, scale: int = 2, gap: int = 6, background=(28, 26, 34)) -> np.ndarray:
    """Cards side by side (crisp enlarged), for looking at a set."""
    images = [load_card_image(c) if not isinstance(c, np.ndarray) else c for c in cards]
    images = [np.repeat(np.repeat(i, scale, 0), scale, 1) for i in images]
    h = max(i.shape[0] for i in images)
    w = sum(i.shape[1] for i in images) + gap * (len(images) + 1)
    out = np.zeros((h + 2 * gap, w, 4), np.uint8)
    out[..., :3] = background
    out[..., 3] = 255
    x = gap
    for i in images:
        a = i[..., 3:4].astype(np.float32) / 255
        region = out[gap:gap + i.shape[0], x:x + i.shape[1]]
        region[..., :3] = np.rint(i[..., :3] * a + region[..., :3] * (1 - a)).astype(np.uint8)
        x += i.shape[1] + gap
    return out


# --------------------------------------------------------------------------- engine export

GODOT_FONTS = (("regular", False, False), ("bold", True, False), ("number", False, True))


def _font_name(kind: str, scale: int) -> str:
    return f"font_{kind}_{scale}x"


def _fonts_needed(datas: list[dict]) -> dict[str, set]:
    need = {"regular": {1}, "bold": {1}, "number": {1}}
    for d in datas:
        f = d["fonts"]
        need["regular"] |= {f["text"], f["type"], f["name"]}
        need["bold"].add(f["name"])
        need["number"].add(f["number"])
    return need


def _copy_card_files(asset, dst: Path) -> list[Path]:
    dst.mkdir(parents=True, exist_ok=True)
    out = []
    for name in ("card.json", "card.png", "art.png", "art_hd.png", "frame.png", "layout.json"):
        p = asset.folder / name
        if p.exists():
            shutil.copyfile(p, dst / name)
            out.append(dst / name)
    if (asset.folder / "ui").exists():
        shutil.copytree(asset.folder / "ui", dst / "ui", dirs_exist_ok=True)
        out += sorted((dst / "ui").iterdir())
    return out


def export_cards(cards, out_dir, engine: str = "godot", res_path: str = "res://cards") -> list[Path]:
    """Cards (assets or folders) -> a folder to drop into a game project.
    godot: Card.tscn + Card.gd (set_card(dict) / load_card(json)), pixel fonts (.fnt BMFont),
           cards/<id>/ files, cards.json index, README.md. Copy the folder to res_path.
    unity: CardView.cs (builds a UGUI card from card.json at runtime), PNGs with .meta (point
           filter, no compression; 9-slice borders set), cards.json, README.md."""
    if engine not in ("godot", "unity"):
        raise ValueError("engine must be godot or unity")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    assets_ = [c if hasattr(c, "folder") else open_card(c) for c in cards]
    if not assets_:
        raise ValueError("No cards to export")
    written = []
    datas = []
    for a in assets_:
        files = _copy_card_files(a, out / "cards" / a.folder.name)
        written += files
        datas.append(json.loads((a.folder / "card.json").read_text(encoding="utf-8")))
    index = out / "cards.json"
    index.write_text(json.dumps({"cards": [f"cards/{d['id']}/card.json" for d in datas]},
                                indent=1), encoding="utf-8")
    written.append(index)
    if engine == "godot":
        written += _export_godot(out, datas, res_path.rstrip("/"))
    else:
        written += _export_unity(out, datas)
    return written


def _export_godot(out: Path, datas: list[dict], res: str) -> list[Path]:
    written = []
    fonts = out / "fonts"
    for kind, need in _fonts_needed(datas).items():
        for s in sorted(need):
            bold, outline = kind == "bold", kind == "number"
            written += write_bmfont(fonts, _font_name(kind, s), s, bold, outline)
    first = datas[0]
    (out / "Card.gd").write_text(CARD_GD.replace("@RES@", res), encoding="utf-8")
    (out / "Card.tscn").write_text(card_tscn(first, res), encoding="utf-8")
    (out / "README.md").write_text(GODOT_README.replace("@RES@", res), encoding="utf-8")
    return written + [out / "Card.gd", out / "Card.tscn", out / "README.md"]


def card_tscn(data: dict, res: str = "res://cards") -> str:
    """A Card scene laid out for one card's size: Art + Frame TextureRects, labels on top."""
    lay = data["layout"]
    S = lay["slots"]
    W, H = lay["width"], lay["height"]
    f = data["fonts"]
    card_dir = f"{res}/cards/{data['id']}"
    ext = [("Script", f"{res}/Card.gd", "1_script"),
           ("Texture2D", f"{card_dir}/art.png" if data.get("art") else f"{card_dir}/frame.png", "2_art"),
           ("Texture2D", f"{card_dir}/frame.png", "3_frame"),
           ("FontFile", f"{res}/fonts/{_font_name('bold' if f['name_bold'] else 'regular', f['name'])}.fnt", "4_name"),
           ("FontFile", f"{res}/fonts/{_font_name('regular', f['type'])}.fnt", "5_type"),
           ("FontFile", f"{res}/fonts/{_font_name('regular', f['text'])}.fnt", "6_text"),
           ("FontFile", f"{res}/fonts/{_font_name('number', f['number'])}.fnt", "7_num")]
    cols = lay["colours"]

    def color(hexs, a=1.0):
        c = _rgb(hexs)
        return f"Color({c[0] / 255:.4g}, {c[1] / 255:.4g}, {c[2] / 255:.4g}, {a})"

    lines = [f'[gd_scene load_steps={len(ext) + 1} format=3]', ""]
    for typ, path, rid in ext:
        lines.append(f'[ext_resource type="{typ}" path="{path}" id="{rid}"]')
    lines += ["", '[node name="Card" type="Control"]', "texture_filter = 1",
              f"custom_minimum_size = Vector2({W}, {H})", f"offset_right = {W}.0",
              f"offset_bottom = {H}.0", 'script = ExtResource("1_script")',
              f'card_json = "{card_dir}/card.json"', ""]

    def rect(name, typ, slot, extra=()):
        s = S.get(slot) if isinstance(slot, str) else slot
        x, y, w, h = (s["x"], s["y"], s["w"], s["h"]) if s else (0, 0, 0, 0)
        lines.extend([f'[node name="{name}" type="{typ}" parent="."]',
                      f"offset_left = {x}.0", f"offset_top = {y}.0",
                      f"offset_right = {x + w}.0", f"offset_bottom = {y + h}.0",
                      "mouse_filter = 2", *extra, ""])

    rect("Art", "TextureRect", "art", ['texture = ExtResource("2_art")', "expand_mode = 1",
                                       "stretch_mode = 6"])
    rect("Frame", "TextureRect", {"x": 0, "y": 0, "w": W, "h": H},
         ['texture = ExtResource("3_frame")', "expand_mode = 1"])

    def label(name, slot, font, size, colour, halign, valign, shadow=None, wrap=False, text=""):
        extra = [f'text = "{_gd_str(text)}"',
                 f'theme_override_fonts/font = ExtResource("{font}")',
                 f"theme_override_font_sizes/font_size = {size}",
                 f"theme_override_colors/font_color = {colour}",
                 f"horizontal_alignment = {halign}", f"vertical_alignment = {valign}",
                 "theme_override_constants/line_spacing = 0",
                 "clip_text = true" if not wrap else "autowrap_mode = 3"]
        if shadow:
            extra += [f"theme_override_colors/font_shadow_color = {shadow}",
                      "theme_override_constants/shadow_offset_x = 1",
                      "theme_override_constants/shadow_offset_y = 1",
                      "theme_override_constants/shadow_outline_size = 0"]
        rect(name, "Label", slot, extra)

    lh = LINE_HEIGHT
    label("Name", "name", "4_name", lh * f["name"], color(cols["name"]), 0, 1,
          color(cols["name_shadow"]), text=data["name"])
    label("Type", "type", "5_type", lh * f["type"], color(cols["type"]), 0, 1,
          color(cols["type_shadow"]), text=data["type"])
    label("Text", "text", "6_text", lh * f["text"], color(cols["ink"]), 1, 1, wrap=True,
          text=data["text"])
    for key in ("cost", "attack", "health", "power"):     # all of them: Card.gd hides unused
        label(key.capitalize(), key, "7_num", lh * f["number"], "Color(1, 1, 1, 1)", 1, 1,
              text=str(data.get(key, "")))
        if S.get(key) is None or key not in data:
            lines.insert(len(lines) - 1, "visible = false")
    return "\n".join(lines)


def _gd_str(s: str) -> str:
    return str(s).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


CARD_GD = '''extends Control
## A card made by pixel-pipeline. set_card(data) fills it from a card.json dictionary (fields,
## art/frame file names, layout rects, font scales); load_card(path) reads the file.

const FONT_DIR := "@RES@/fonts"

@export_file("*.json") var card_json := ""

var data := {}


func _ready() -> void:
	if card_json != "" and data.is_empty():     # set_card / load_card before _ready wins
		load_card(card_json)


func load_card(path: String) -> void:
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		push_error("Card: can't open " + path)
		return
	var parsed = JSON.parse_string(f.get_as_text())
	if typeof(parsed) != TYPE_DICTIONARY:
		push_error("Card: bad JSON in " + path)
		return
	set_card(parsed, path.get_base_dir())


func set_card(card: Dictionary, base_dir: String = "") -> void:
	data = card
	var lay: Dictionary = card.get("layout", {})
	if lay.has("width"):
		custom_minimum_size = Vector2(lay["width"], lay["height"])
		size = custom_minimum_size
	var slots: Dictionary = lay.get("slots", {})
	_place($Art, slots.get("art"))
	_place($Frame, {"x": 0, "y": 0, "w": lay.get("width", size.x), "h": lay.get("height", size.y)})
	if base_dir != "":
		if card.get("art") != null:
			$Art.texture = _texture(base_dir.path_join(str(card["art"])))
		$Frame.texture = _texture(base_dir.path_join(str(card.get("frame", "frame.png"))))
	var cols: Dictionary = lay.get("colours", {})
	_colour($Name, cols.get("name"), cols.get("name_shadow"))
	_colour($Type, cols.get("type"), cols.get("type_shadow"))
	_colour($Text, cols.get("ink"), null)
	var fonts: Dictionary = card.get("fonts", {})
	_text($Name, "name", str(card.get("name", "")), slots,
		_font("bold" if fonts.get("name_bold", false) else "regular", fonts.get("name", 1)))
	_text($Type, "type", str(card.get("type", "")), slots, _font("regular", fonts.get("type", 1)))
	_text($Text, "text", str(card.get("text", "")), slots, _font("regular", fonts.get("text", 1)))
	for key in ["cost", "attack", "health", "power"]:
		var node := get_node_or_null(key.capitalize())
		if node == null:
			continue
		var has: bool = card.has(key) and card[key] != null and slots.has(key)
		node.visible = has
		if has:
			_text(node, key, str(int(card[key])), slots, _font("number", fonts.get("number", 1)))
	_relayout.call_deferred()


func _relayout() -> void:
	# Labels are clamped to their minimum size, measured with whatever font was live when the
	# size was set (Godot's default before the pixel font is in use): place them again now.
	var slots: Dictionary = data.get("layout", {}).get("slots", {})
	for key in ["name", "type", "text", "cost", "attack", "health", "power"]:
		var node := get_node_or_null(key.capitalize())
		if node != null and slots.has(key):
			node.size = Vector2.ZERO
			_place(node, slots[key])


func _colour(node: Label, colour, shadow) -> void:
	if colour != null:
		node.add_theme_color_override("font_color", Color.html(str(colour)))
	if shadow != null:
		node.add_theme_color_override("font_shadow_color", Color.html(str(shadow)))


func _place(node: Control, slot) -> void:
	if slot == null:
		return
	node.position = Vector2(slot["x"], slot["y"])
	node.size = Vector2(slot["w"], slot["h"])


func _text(node: Label, key: String, value: String, slots: Dictionary, font: Font) -> void:
	# font first: a Label grows to its font's minimum size and doesn't shrink back by itself
	if font != null:
		node.add_theme_font_override("font", font)
		node.add_theme_font_size_override("font_size", font.fixed_size if font is FontFile and font.fixed_size > 0 else 10)
	node.text = value
	_place(node, slots.get(key))


func _font(kind: String, scale) -> Font:
	var path := "%s/font_%s_%dx.fnt" % [FONT_DIR, kind, int(scale)]
	if ResourceLoader.exists(path):
		return load(path)
	return null


func _texture(path: String) -> Texture2D:
	if ResourceLoader.exists(path):
		return load(path)
	var img := Image.load_from_file(path)
	return ImageTexture.create_from_image(img) if img != null else null
'''

GODOT_README = """# Cards for Godot 4 (made by pixel-pipeline)

Copy this folder into your project as `@RES@` (the paths inside expect that; export again
with another res_path to change it). Godot imports the PNGs and the BMFont `.fnt` pixel fonts.

- `Card.tscn` + `Card.gd`: a Control with `Art` and `Frame` TextureRects and `Name`, `Type`,
  `Text`, `Cost`, `Attack`, `Health`, `Power` Labels, positioned from the card's layout.
  Set `card_json` in the inspector, or from code:

      var card = preload("@RES@/Card.tscn").instantiate()
      add_child(card)
      card.load_card("@RES@/cards/fireball/card.json")
      # or card.set_card(dict, "@RES@/cards/fireball")

  Missing stats hide their label. Keep `texture_filter` Nearest (set on the root) and scale
  cards by whole numbers (Project Settings > Display > Window > Stretch Scale Mode: integer)
  so pixels stay crisp.
- `cards/<id>/`: `card.json` (all fields + layout rects + font scales), `art.png`,
  `frame.png` (transparent art window), `card.png` (everything baked: use it alone if you
  don't need live text), `layout.json`, `ui/` 9-slice pieces with `.tres` StyleBoxTextures
  (border, panel, dark, banner) for menus in the same style.
- `cards.json`: every card's card.json path.
- `fonts/`: the pixel font (regular, bold, outlined numbers) at the scales the cards use.
"""


def _export_unity(out: Path, datas: list[dict]) -> list[Path]:
    from .exports import _unity_write, unity_guid, unity_meta
    written = []
    for png in sorted(out.rglob("*.png")):
        size = Image.open(png).size
        rel = png.relative_to(out).as_posix()
        meta = unity_meta(unity_guid("cards/" + rel), size, None, ppu=100, pivot=(0.5, 0.5))
        js = png.with_suffix(".json")
        if png.parent.name == "ui" and js.exists():          # 9-slice: Unity border = L, B, R, T
            m = json.loads(js.read_text(encoding="utf-8"))["patch_margin"]
            meta = meta.replace("spriteBorder: {x: 0, y: 0, z: 0, w: 0}",
                                f"spriteBorder: {{x: {m['left']}, y: {m['bottom']}, "
                                f"z: {m['right']}, w: {m['top']}}}")
        p = png.with_name(png.name + ".meta")
        _unity_write(p, meta)
        written.append(p)
    cs = out / "CardView.cs"
    _unity_write(cs, CARD_VIEW_CS)
    readme = out / "README.md"
    readme.write_text(UNITY_README, encoding="utf-8")
    return written + [cs, readme]


CARD_VIEW_CS = r'''// A card made by pixel-pipeline, built as UGUI from its card.json at runtime.
// Put it on an empty GameObject under a Canvas, assign the card's card.json (TextAsset),
// art.png and frame.png (Sprites); Build() runs on Awake (or call SetCard from code).
using System;
using UnityEngine;
using UnityEngine.UI;

[Serializable] public class CardSlot { public int x, y, w, h; }

[Serializable] public class CardSlots {
    public CardSlot art, name, cost, type, rarity, textbox, text, attack, health, power;
}

[Serializable] public class CardColours {
    public string name, name_shadow, type, type_shadow, ink, number, number_outline;
}

[Serializable] public class CardLayoutData {
    public int width, height; public string stats; public CardSlots slots; public CardColours colours;
}

[Serializable] public class CardFonts { public int name = 1, type = 1, text = 1, number = 2; public bool name_bold; }

[Serializable] public class CardData {
    public string id, name, type, text, rarity, faction, size, art, frame, card;
    public int cost = -1, attack = -1, health = -1, power = -1;   // -1 = not on this card
    public CardFonts fonts; public CardLayoutData layout;
}

public class CardView : MonoBehaviour {
    public TextAsset cardJson;
    public Sprite art;
    public Sprite frame;
    public Font font;                 // any font; pixel fonts look best (default: LegacyRuntime)
    public int fontPixels = 10;       // font size for scale 1 (the pixel font's line height)
    public CardData data;

    void Awake() { if (cardJson != null) SetCard(cardJson.text); }

    public void SetCard(string json) { SetCard(JsonUtility.FromJson<CardData>(json)); }

    public void SetCard(CardData card) {
        data = card;
        for (int i = transform.childCount - 1; i >= 0; i--) {
            var child = transform.GetChild(i).gameObject;
            if (Application.isPlaying) Destroy(child); else DestroyImmediate(child);
        }
        var root = GetComponent<RectTransform>();
        if (root == null) root = gameObject.AddComponent<RectTransform>();
        var lay = card.layout;
        root.sizeDelta = new Vector2(lay.width, lay.height);
        if (font == null) font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        var c = lay.colours;
        AddImage("Art", lay.slots.art, art);
        AddImage("Frame", new CardSlot { x = 0, y = 0, w = lay.width, h = lay.height }, frame);
        AddText("Name", lay.slots.name, card.name, card.fonts.name, c.name, c.name_shadow,
                TextAnchor.MiddleLeft, card.fonts.name_bold);
        AddText("Type", lay.slots.type, card.type, card.fonts.type, c.type, c.type_shadow,
                TextAnchor.MiddleLeft, false);
        var t = AddText("Text", lay.slots.text, card.text, card.fonts.text, c.ink, null,
                        TextAnchor.UpperCenter, false);
        t.horizontalOverflow = HorizontalWrapMode.Wrap;
        AddNumber("Cost", lay.slots.cost, card.cost, card.fonts.number, c);
        AddNumber("Attack", lay.slots.attack, card.attack, card.fonts.number, c);
        AddNumber("Health", lay.slots.health, card.health, card.fonts.number, c);
        AddNumber("Power", lay.slots.power, card.power, card.fonts.number, c);
    }

    RectTransform Place(string name, CardSlot s) {
        var go = new GameObject(name, typeof(RectTransform));
        var rt = (RectTransform)go.transform;
        rt.SetParent(transform, false);
        rt.anchorMin = rt.anchorMax = new Vector2(0, 1);     // card json: top-left origin, y down
        rt.pivot = new Vector2(0, 1);
        rt.anchoredPosition = new Vector2(s.x, -s.y);
        rt.sizeDelta = new Vector2(s.w, s.h);
        return rt;
    }

    static bool Has(CardSlot s) { return s != null && s.w > 0 && s.h > 0; }

    void AddImage(string name, CardSlot s, Sprite sprite) {
        if (!Has(s) || sprite == null) return;
        var img = Place(name, s).gameObject.AddComponent<Image>();
        img.sprite = sprite;
        img.raycastTarget = false;
    }

    Text AddText(string name, CardSlot s, string value, int scale, string colour, string shadow,
                 TextAnchor anchor, bool bold) {
        if (!Has(s)) return null;
        var go = Place(name, s).gameObject;
        var t = go.AddComponent<Text>();
        t.font = font;
        t.fontSize = fontPixels * Mathf.Max(1, scale);
        t.fontStyle = bold ? FontStyle.Bold : FontStyle.Normal;
        t.text = value ?? "";
        t.alignment = anchor;
        t.color = Hex(colour);
        t.raycastTarget = false;
        t.horizontalOverflow = HorizontalWrapMode.Overflow;
        t.verticalOverflow = VerticalWrapMode.Overflow;
        if (!string.IsNullOrEmpty(shadow)) {
            var sh = go.AddComponent<Shadow>();
            sh.effectColor = Hex(shadow);
            sh.effectDistance = new Vector2(1, -1);
        }
        return t;
    }

    void AddNumber(string name, CardSlot s, int value, int scale, CardColours c) {
        if (!Has(s) || value < 0) return;
        var t = AddText(name, s, value.ToString(), scale, c.number, null, TextAnchor.MiddleCenter, true);
        var o = t.gameObject.AddComponent<Outline>();
        o.effectColor = Hex(c.number_outline);
        o.effectDistance = new Vector2(1, -1);
    }

    static Color Hex(string hex) {
        Color col;
        return ColorUtility.TryParseHtmlString("#" + (hex ?? "ffffff"), out col) ? col : Color.white;
    }
}
'''

UNITY_README = """# Cards for Unity (made by pixel-pipeline)

Needs the Unity UI package (com.unity.ugui, in every standard template; Unity 6 verified).
Copy this folder (with the .meta files) anywhere under `Assets/`. The PNGs import as Sprites
with Point filtering and no compression; the `ui/` pieces have their 9-slice borders set (use
them in an Image with Image Type: Sliced).

- `CardView.cs`: put it on an empty GameObject under a Canvas, then assign a card's
  `card.json` (TextAsset), `art.png` and `frame.png` (Sprites). It builds the card (Art and
  Frame Images, Name/Type/Text/Cost/Attack/Health/Power Texts) from the layout in card.json on
  Awake, or call `SetCard(jsonText)` / `SetCard(CardData)` from code. Missing stats are left
  out. Assign a pixel font to `font` for crisp text (default: Unity's LegacyRuntime font); set
  the Canvas Scaler to a whole-number scale so pixels stay square.
- `cards/<id>/card.png`: the whole card baked (art + frame + pixel-font text) if you don't
  need live text: use it as a single Sprite.
- `cards/<id>/layout.json` / `card.json`: slot rects in pixels, top-left origin, y down
  (CardView flips them for UGUI).
- `cards.json`: every card's card.json path.
"""

"""Tiles v3 (2026-10-15): ground tiles made the way a pixel artist would, at the tile's own
scale; the GPU gives material and colour, never a layout that then has to be squeezed.

    raw, pic = paint(structure("road", 32, 3), "road", client=c)   # SDXL over a 2x2 canvas
    tile = stone_tile(pic, 32, "road")            # its stones re-drawn as clean pixel shapes
    grass = drawn_tile("grass", 32, 3, picture=sdxl_swatch)       # code shapes, SDXL colours

Why (Kevin, 2026-10-15, robust_tiles_v2.png): a free SDXL swatch squeezed into a 32 px tile
lost its shapes (cobbles -> brick rows, dirt -> diagonal streaks, snow -> drifts on dark
ground), and the half-offset blends ghosted copies of features over each other. GPU A/B of
2026-10-15 (QA project): SDXL repainting a layout keeps the init's pixel steps at denoise
0.4-0.55 and lays its own rounded stones at 0.7; soft ground (grass, dirt, sand, snow, water) it
flattens into big blobs at every denoise and canvas scale. So, two methods (tilegen.METHODS):

paint (road, cliff): structure() draws an irregular layout at the tile size (mixed-size cobbles,
   boulders), init_canvas() scales it 16x, softens it and tiles it 2 x 2 (1024 px), SDXL
   repaints it at DENOISE (style LoRA at TILE_STYLE_STRENGTH), fold() cross-fades the four
   painted copies with sin^2 weights that are zero on the canvas border: the tile wraps exactly
   and nothing ghosts (all copies are paintings of one layout). stone_tile() then re-draws it:
   segment_stones() (watershed on the painting's evened lightness from one marker per stone, on
   a 3 x 3 wrap copy, ids by marker position so edge stones stay whole) -> each stone one shape
   in its own tone (rank of its painted lightness) and tint (warm / cool stones, 3 shared
   tints) on a hue-shifted ramp, lit rim upper left, shadow rim lower right, a glint on big
   stones, rounded corners, 1 px dark joints deepest under the stones. paint_variants() paints
   four other layouts in one more pass for the set's interiors.
drawn (grass, dirt, sand, snow, stone, water, fence, wall): SOFT painters draw the tile (blade
   tufts, pebbles, broken ripples, small drifts; flagstones, planks, bricks from neo_terrain)
   with evenly spread details (scatter: no clump marks the period) in a ramp mixed from the
   hand-built one and SDXL's swatch (picture_ramp; SWATCH_WEIGHT), so the user's words still
   set the colours; water = neo ripples (water.py) in the swatch's blue. drawn_variants() =
   other layouts in the same ramp.

finish_tile() is the generic tile finish (area downscale: no directional streaks, evened
large-scale light, contrast per terrain, hue-shifted ramp, no dither, lone pixels merged).
"""
from __future__ import annotations

import zlib

import numpy as np

# terrain -> neo_terrain painter for its layout
PAINT_SCALE = 512                   # SDXL pixels per game tile (a 2 x 2 canvas = 1024)
COPIES = {}                         # tile copies per canvas side (default 2)
# Calibrated 2026-10-15 (QA project, Pictures/PixelPipeline/tiles_quality_ab.png): at 0.4-0.55
# SDXL keeps the init's pixel steps (rows of square stones); at 0.7 it lays its own rounded
# stones at the init's scale. Soft ground (grass, dirt, sand, snow, water) it flattens at any
# denoise, so those are drawn by code in SDXL's colours (drawn_tile) instead.
DENOISE = {"road": 0.7, "cliff": 0.7, "stone": 0.7}
DENOISE_DEFAULT = 0.7
PAINTED = ("road", "cliff")         # SDXL repaints the layout, stone_tile re-draws it
# Lab L spread (std) the finished tile is stretched / squeezed toward, and its ramp length
CONTRAST = {"grass": (7.0, 11.0), "dirt": (6.5, 10.0), "road": (11.0, 17.0), "stone": (9.0, 15.0),
            "sand": (5.0, 8.5), "snow": (5.0, 8.0), "water": (7.0, 12.0), "cliff": (10.0, 16.0),
            "fence": (9.0, 15.0), "wall": (10.0, 16.0)}
RAMP_N = {"grass": 6, "dirt": 6, "road": 7, "stone": 7, "sand": 5, "snow": 5, "water": 6,
          "cliff": 7, "fence": 6, "wall": 7}
SPARKLE = {"snow": 4, "sand": 2, "water": 6}         # lone brightest pixels kept per 32 px tile
# warm lights / cool darks (Lab b shift at the ends of the ramp); snow + water stay cool
HUE_SHIFT = {"snow": -4.0, "water": 2.0}
HUE_SHIFT_DEFAULT = 7.0
# lightness band of the ramp (Lab L 0-100): snow never gets dark ground
L_RANGE = {"snow": (58.0, 99.0), "sand": (42.0, 92.0)}


def _rng(*key) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32("/".join(str(k) for k in key).encode()))


# --------------------------------------------------------------------------- structure

def scatter(t: int, n: int, rng, spacing: float) -> list[tuple[int, int]]:
    """Up to n points in a wrapping t x t tile, at least `spacing` px apart (dart throwing):
    details spread evenly, so no clump marks the tile's period."""
    pts: list[tuple[int, int]] = []
    for _ in range(n * 30):
        if len(pts) >= n:
            break
        y, x = int(rng.integers(t)), int(rng.integers(t))
        ok = True
        for py, px in pts:
            dy, dx = abs(py - y), abs(px - x)
            if np.hypot(min(dy, t - dy), min(dx, t - dx)) < spacing:
                ok = False
                break
        if ok:
            pts.append((y, x))
    return pts


def _put(idx, y, x, v, t, only=None):
    y, x = y % t, x % t
    if only is None or idx[y, x] in only:
        idx[y, x] = v


def _mottle(t: int, rng, cells: int, lo: float, hi: float) -> np.ndarray:
    """-1 / 0 / +1 small soft spots (high-frequency wrapping noise, no lone pixels): tone
    variation at a few pixels, never a tile-sized blotch."""
    from .neo_terrain import _majority, pnoise
    f = 0.7 * pnoise(t, t, cells, cells, rng) + 0.3 * pnoise(t, t, 2 * cells, 2 * cells, rng)
    return _majority(f > hi).astype(int) - _majority(f < lo).astype(int)


def _d(d):
    from .tilewords import PLAIN
    return PLAIN if d is None else d


def grass_idx(t: int, seed: int = 0, d=None) -> np.ndarray:
    """Lawn (6-step ramp): a mid green with small darker / lighter spots and evenly spread blade
    tufts lit from the upper left (light tips, mid blades, a dark root shadow). d (tilewords.
    Details): tufts x count, tall = taller blades, cracks = dry bare spots."""
    d = _d(d)
    rng = _rng("tgrass", t, seed)
    idx = 2 + _mottle(t, rng, max(3, t // 4), 0.2, 0.8)
    many = d.tufts * (1 + 0.3 * d.tall)
    n = max(4, t * t // 56) if many == 1.0 else max(3, int(t * t / 56 * many))
    sp = max(2.5, t / 7) if many == 1.0 else max(2.0, t / 7 / np.sqrt(max(many, 0.3)))
    for y, x in scatter(t, n, rng, sp):
        big = t >= 24 and rng.random() < 0.55 + 0.4 * d.tall
        b = int(idx[y, x])
        _put(idx, y + 1, x, max(1, b - 1), t)                      # a soft root shadow
        _put(idx, y, x - 1, min(5, b + 1), t)                      # blades
        _put(idx, y, x + 1, min(5, b + 1), t)
        _put(idx, y - 1, x - 1 if rng.random() < 0.5 else x + 1, min(4, b + 2), t)    # tip
        if big:
            _put(idx, y, x, min(5, b + 1), t)
            _put(idx, y - 1, x, min(5, b + 2), t)
            _put(idx, y - 2, x, min(5 if rng.random() < 0.2 else 4, b + 3), t)
            if d.tall > 0 and rng.random() < 0.4 + 0.5 * d.tall:     # tall grass: long blades
                lean = 1 if rng.random() < 0.5 else -1
                _put(idx, y - 3, x + (lean if rng.random() < 0.5 else 0), min(5, b + 3), t)
                if d.tall >= 1 and t >= 32:
                    _put(idx, y - 4, x + lean, min(5, b + 3), t)
                _put(idx, y + 1, x - lean, max(0, b - 2), t)          # deeper root shadow
    if d.cracks > 0:                                                  # dry: a few bare cracks
        from .tilewords import add_cracks
        idx = add_cracks(idx, rng, d.cracks * 0.5, dark=1, cell=max(6.0, t / 2.5))
    return idx


def dirt_idx(t: int, seed: int = 0, d=None) -> np.ndarray:
    """Packed earth (6 steps): small worn spots, evenly spread lit pebbles with a shadow on the
    lower right, a few dark pits and short cracks. d: pebbles x count (big_pebbles: up to 3 x 3),
    cracks = a dried-mud crack network, footprints = a trail of prints."""
    d = _d(d)
    rng = _rng("tdirt", t, seed)
    idx = 3 + _mottle(t, rng, max(3, t // 4), 0.27, 0.75)
    pb = d.pebbles
    n = max(3, t * t // 85) if pb == 1.0 else max(3, int(t * t / 85 * pb))
    for y, x in scatter(t, n, rng, max(3.0, t / 5) if pb == 1.0 else max(2.5, t / 5 / np.sqrt(pb))):
        w = 2 if rng.random() < 0.6 else 1
        h = 2 if (w == 2 and rng.random() < 0.5) else 1
        if d.big_pebbles and rng.random() < 0.5:
            w, h = 3, int(rng.integers(2, 4))
        for dy in range(h):
            for dx in range(w):
                _put(idx, y + dy, x + dx, 5 if dy == 0 and dx == 0 else 4, t)
        for dx in range(1, w + 1):
            _put(idx, y + h, x + dx, 1, t)                         # shadow below right
        _put(idx, y + h - 1, x + w, 1, t)
    for y, x in scatter(t, max(1, t // 10), rng, t / 4):           # pits
        _put(idx, y, x, 1, t)
        if rng.random() < 0.5:
            _put(idx, y, x + 1, 2, t)
    for _ in range(max(1, t // 32)):                                # a short crack
        y, x = int(rng.integers(t)), int(rng.integers(t))
        for _ in range(int(rng.integers(3, 5))):
            _put(idx, y, x, 1, t)
            x += 1
            y += int(rng.integers(-1, 2))
    if d.cracks > 0:
        from .tilewords import add_cracks
        idx = add_cracks(idx, rng, d.cracks, dark=0)
    if d.footprints:
        from .tilewords import add_footprints
        idx = add_footprints(idx, rng, deep=1, rim=4, mid=2)
    return idx


def sand_idx(t: int, seed: int = 0, d=None) -> np.ndarray:
    """Fine sand (5 steps): broken, gently curved wind ripples (a lit crest over a soft shadow),
    faint spots and a few grains. d: ripples x count and length (dunes: wide crest bands under
    them), pebbles x grains, cracks = dried crust, footprints."""
    d = _d(d)
    rng = _rng("tsand", t, seed)
    idx = 2 + _mottle(t, rng, max(3, t // 4), 0.3, 0.75)
    idx = np.clip(idx, 1, 3)
    if d.dunes:
        from .tilewords import dune_bands
        idx = np.clip(idx + dune_bands(t, rng), 1, 3)
    rip = d.ripples
    n = max(3, t * t // 120) if rip == 1.0 else max(3, int(t * t / 120 * rip))
    for y, x in scatter(t, n, rng, max(4.0, t / 4.5) if rip == 1.0 else max(3.0, t / 4.5 / np.sqrt(rip))):
        if rip == 1.0:
            L = int(rng.integers(max(3, t // 8), max(5, t // 4) + 1))
        else:
            L = min(t - 2, int(rng.integers(max(3, int(t / 8 * rip)), max(5, int(t / 4 * rip)) + 1)))
        yy = y
        for k in range(L):
            if k and rng.random() < 0.25:
                yy += int(rng.choice((-1, 1)))
            _put(idx, yy, x + k, 3, t)
            _put(idx, yy + 1, x + k, 1, t)
    pb = d.pebbles
    for y, x in scatter(t, max(1, t // 12) if pb == 1.0 else max(1, int(t / 12 * pb)), rng,
                        t / 4 if pb == 1.0 else t / 4 / np.sqrt(pb)):           # grains
        _put(idx, y, x, 4 if rng.random() < 0.6 else 1, t)
    if d.cracks > 0:
        from .tilewords import add_cracks
        idx = add_cracks(idx, rng, d.cracks, dark=1, lit=4)
    if d.footprints:
        from .tilewords import add_footprints
        idx = add_footprints(idx, rng, deep=1, rim=4, mid=2)
    return idx


def snow_idx(t: int, seed: int = 0, d=None) -> np.ndarray:
    """Fresh snow (5 steps, all light): a white body, small evenly spread drifts (a white crest,
    a soft blue shadow under it, deeper in its middle), tiny hollows, a few sparkles. d: fresh =
    fewer drifts + more sparkle, ripples = windswept (longer, more drifts), footprints."""
    d = _d(d)
    rng = _rng("tsnow", t, seed)
    body = 3
    idx = np.full((t, t), body)
    sp = _mottle(t, rng, max(3, t // 4), 0.36, 2.0)               # soft shaded hollows
    idx[sp < 0] = 2
    many = (0.45 if d.fresh else 1.0) * d.ripples ** 0.6
    n = max(3, t * t // 100) if many == 1.0 else max(2, int(t * t / 100 * many))
    for y, x in scatter(t, n, rng, max(4.0, t / 5) if many == 1.0 else max(3.5, t / 5 / np.sqrt(many))):
        if d.ripples == 1.0:
            L = int(rng.integers(max(3, t // 8), max(5, t // 4) + 1))
        else:
            L = min(t - 2, int(rng.integers(max(3, int(t / 8 * d.ripples)), max(5, int(t / 4 * d.ripples)) + 1)))
        for k in range(L):
            end = k in (0, L - 1)
            yy = y + (1 if end else 0)
            _put(idx, yy - 1, x + k, 4, t)                         # crest
            _put(idx, yy, x + k, 2 if end else 1, t)               # blue shadow under it
            if not end and L >= 5:
                _put(idx, yy + 1, x + k, 0 if 1 < k < L - 2 else 2, t)   # deepest in the middle
    for y, x in scatter(t, max(1, t // 10) if not d.fresh else max(2, int(t / 4)), rng,
                        t / 5 if not d.fresh else t / 7.5):          # sparkles
        _put(idx, y, x, 4, t, only=(body,))
    if d.footprints:
        from .tilewords import add_footprints
        idx = add_footprints(idx, rng, deep=0, rim=4, mid=1)
    return idx


SNOW_RAMP = "8299c9 a7bbdf c8d7ee e3ecf7 fafcff"      # body L ~93: drifts survive HD-2D light
SAND_RAMP = "a57e5e c9a473 dcbb85 ebd29c f7e9c0"
SOFT = {"grass": (grass_idx, "grass", 6), "dirt": (dirt_idx, "dirt", 6),
        "sand": (sand_idx, SAND_RAMP, 5), "snow": (snow_idx, SNOW_RAMP, 5)}


def neo_idx(fn, t: int, seed: int, n: int = 7, **kw) -> np.ndarray:
    """The ramp index map of a neo_terrain painter (drawn with an index-coded ramp)."""
    code = np.repeat(np.arange(n, dtype=np.uint8)[:, None], 3, 1)
    out = fn(t, seed=seed, ramp=code, **kw)
    return np.asarray(out[0] if isinstance(out, tuple) else out)[..., 0].astype(int)


def _flag_idx(t, seed, d=None):
    from .neo_terrain import flagstones
    idx = neo_idx(flagstones, t, seed)
    d = _d(d)
    if d.cracks > 0:
        from .tilewords import add_cracks
        idx = add_cracks(idx, _rng("flagcracks", t, seed), d.cracks * 0.6, dark=0,
                         cell=max(6.0, t / 2.5))
    return idx


def _plank_idx(t, seed, d=None):
    from .neo_terrain import planks
    return neo_idx(planks, t, seed)


def _brick_idx(t, seed, d=None):
    from .neo_terrain import rock_face
    code = np.repeat(np.arange(7, dtype=np.uint8)[:, None], 3, 1)
    return rock_face(t, t, code, seed, bricks=True, seams=(0,))[0][..., 0].astype(int)


SOFT.update({"stone": (_flag_idx, "stone", 7), "fence": (_plank_idx, "planks", 7),
             "wall": (_brick_idx, "brick", 7)})
# how much of a drawn tile's ramp comes from SDXL's swatch (the rest: the hand-built ramp)
SWATCH_WEIGHT = {"snow": 0.3, "water": 1.0}
SWATCH_WEIGHT_DEFAULT = 0.6


def soft_ramp(terrain: str) -> np.ndarray:
    from .neo_terrain import RAMP_KEYS, hex_ramp
    _, key, n = SOFT[terrain]
    return hex_ramp(RAMP_KEYS.get(key, key), n)


def soft_snow(t: int, seed: int = 0, ramp: np.ndarray | None = None):
    """Fresh snow in the snow ramp (RGBA, height)."""
    from .neo_terrain import _colour
    idx = snow_idx(t, seed)
    return _colour(soft_ramp("snow") if ramp is None else ramp, idx), idx / 4.0


def boulders(t: int, seed: int = 0, ramp: np.ndarray | None = None):
    """Rough rock seen from above: a few big lit boulders, cracks, dark gaps (cliff tops)."""
    from .neo_terrain import RAMPS, _cells, _majority, _scale, _shade, _colour, pnoise
    ramp = RAMPS["rock"] if ramp is None else ramp
    n = len(ramp)
    rng = _rng("boulders", t, seed)
    g = max(2, round(t / 11))
    ident, _, edge = _cells(t, t, g, g, rng, 0.8)
    gap = edge < 0.7
    dome = np.clip((edge - 0.5) / max(1.5, t / 10), 0, 1)
    rough = pnoise(t, t, max(2, t // 6), max(2, t // 6), rng)
    height = np.where(gap, 0.0, 0.35 + 0.45 * dome + 0.2 * rough)
    lit = _shade(height)
    tone = rng.integers(-1, 2, g * g)[ident]
    idx = _scale(n, 3.0) + tone + (lit > 0.12).astype(int) + (lit > 0.4).astype(int) \
        - (lit < -0.12).astype(int)
    idx = np.clip(idx, 1, n - 1)
    idx[gap] = 0
    idx[_majority(gap) & ~gap] = 1
    for _ in range(max(1, t // 12)):                     # hairline cracks
        y, x = int(rng.integers(t)), int(rng.integers(t))
        for _ in range(int(rng.integers(2, 5))):
            if not gap[y % t, x % t]:
                idx[y % t, x % t] = max(0, idx[y % t, x % t] - 2)
            x += int(rng.integers(0, 2))
            y += 1
    return _colour(ramp, idx), height


def cobbles(t: int, seed: int = 0, ramp: np.ndarray | None = None):
    """Irregular rounded cobbles of mixed sizes (dart-thrown centres, each with its own size: a
    weighted Voronoi, so no grid rows for SDXL to follow), lit domes in dark joints."""
    from .neo_terrain import RAMPS, _colour, _scale, _shade
    ramp = RAMPS["cobble"] if ramp is None else ramp
    n = len(ramp)
    rng = _rng("cobbles", t, seed)
    pts = scatter(t, max(4, int(t * t / 30)), rng, max(2.5, t / 6.5))
    rad = rng.uniform(0.75, 1.3, len(pts))
    yy, xx = np.mgrid[0:t, 0:t] + 0.5
    best = np.full((t, t), np.inf)
    second = np.full((t, t), np.inf)
    ident = np.zeros((t, t), int)
    for k, (py, px) in enumerate(pts):
        dy = np.abs(yy - py - 0.5)
        dx = np.abs(xx - px - 0.5)
        d = np.hypot(np.minimum(dy, t - dy), np.minimum(dx, t - dx)) / rad[k]
        closer = d < best
        second = np.where(closer, best, np.minimum(second, d))
        ident = np.where(closer, k, ident)
        best = np.where(closer, d, best)
    edge = (second - best) * 1.6
    mortar = edge < 0.75
    dome = np.clip((edge - 0.5) / max(1.2, t / 16 * 1.6), 0, 1)
    height = np.where(mortar, 0.0, 0.4 + 0.6 * dome)
    tone = rng.integers(-1, 2, len(pts))[ident]
    lit = _shade(height)
    idx = _scale(n, 3.4) + tone + (lit > 0.15).astype(int) + (lit > 0.5).astype(int)         - (lit < -0.15).astype(int)
    idx = np.clip(idx, 1, n - 1)
    idx[mortar] = 0
    return _colour(ramp, idx), height


def structure(terrain: str, t: int = 32, seed: int = 0) -> np.ndarray:
    """The tile's layout (RGBA, wraps) in neo 16-bit shapes at the tile's own scale."""
    from . import neo_terrain as nt
    if terrain in SOFT:
        from .neo_terrain import _colour
        return _colour(soft_ramp(terrain), SOFT[terrain][0](t, seed))
    if terrain == "cliff":
        return boulders(t, seed)[0]
    if terrain == "water":
        from .tilegen import neo_water_tile
        return neo_water_tile((40, 120, 200), t, seed)
    if terrain == "road":
        return cobbles(t, seed)[0]
    name = terrain
    if name in nt.TOPS:
        return nt.paint_top(name, t, seed)[0]
    from .maptiles import BUILTIN, builtin_texture
    return builtin_texture(terrain if terrain in BUILTIN else "grass", t, seed, even=True)


def _wrap_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    import cv2
    if sigma <= 0:
        return img
    p = int(np.ceil(3 * sigma))
    big = cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_WRAP)
    return cv2.GaussianBlur(big, (0, 0), sigma)[p:-p, p:-p]


def init_canvas(tile: np.ndarray, scale: int = PAINT_SCALE, n: int = 2,
                soften: float = 0.35) -> np.ndarray:
    """A tile -> (n*scale, n*scale) RGB canvas: n x n copies scaled up (nearest), the pixel
    steps softened (wrap-around blur of soften x one tile pixel) so SDXL paints shapes, not a
    grid. Periodic: its seams sit where SDXL can see across them."""
    import cv2
    rgb = np.ascontiguousarray(np.asarray(tile)[..., :3]).astype(np.uint8)
    t = rgb.shape[0]
    big = cv2.resize(rgb, (scale, scale), interpolation=cv2.INTER_NEAREST)
    big = _wrap_blur(big, soften * scale / t)
    return np.tile(big, (n, n, 1))


def fold(painted: np.ndarray, n: int = 2) -> np.ndarray:
    """n x n painted copies of one tile -> one tile that wraps exactly: every copy weighted by
    sin^2 across the whole canvas (zero on the canvas border), so where one copy meets its
    canvas edge another copy, painted continuously across that spot, takes over."""
    rgb = np.asarray(painted)[..., :3].astype(np.float64)
    h, w = rgb.shape[:2]
    sh, sw = h // n, w // n
    rgb = rgb[:sh * n, :sw * n]
    wy = np.sin(np.pi * (np.arange(sh * n) + 0.5) / (sh * n)) ** 2
    wx = np.sin(np.pi * (np.arange(sw * n) + 0.5) / (sw * n)) ** 2
    out = np.zeros((sh, sw, 3))
    tot = np.zeros((sh, sw, 1))
    for i in range(n):
        for j in range(n):
            wgt = (wy[i * sh:(i + 1) * sh, None] * wx[None, j * sw:(j + 1) * sw])[..., None]
            out += wgt * rgb[i * sh:(i + 1) * sh, j * sw:(j + 1) * sw]
            tot += wgt
    return np.clip(np.rint(out / np.maximum(tot, 1e-9)), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- the pixel finish

def _lab(rgb: np.ndarray) -> np.ndarray:
    from .color import rgb_to_lab
    return rgb_to_lab(np.asarray(rgb, np.float32))


def downscale_area(picture: np.ndarray, size: int) -> np.ndarray:
    """A wrapping picture -> size x size RGB by box averaging (isotropic: no streaks)."""
    import cv2
    rgb = np.ascontiguousarray(np.asarray(picture)[..., :3]).astype(np.uint8)
    h = rgb.shape[0]
    if h % size:
        k = max(1, int(round(h / size)))
        rgb = cv2.resize(rgb, (size * k, size * k), interpolation=cv2.INTER_LANCZOS4)
    k = rgb.shape[0] // size
    return rgb.reshape(size, k, size, k, 3).astype(np.float64).mean((1, 3))


def sharpen_wrap(lab: np.ndarray, amount: float = 0.6, sigma: float = 1.0) -> np.ndarray:
    """Unsharp mask on L (wrap-around): shapes get crisp rims before quantising."""
    L = lab[..., 0].astype(np.float32)
    blur = _wrap_blur(L, sigma)
    out = lab.copy()
    out[..., 0] = np.clip(L + amount * (L - blur), 0, 100)
    return out


def tile_ramp(lab: np.ndarray, n: int, terrain: str = "") -> np.ndarray:
    """(n, 3) Lab ramp dark -> light from a tile's own colours: n evenly stepped lightness levels
    across its 3rd-97th percentile (inside L_RANGE), each level's a/b = the mean of the pixels
    near that lightness, then hue-shifted (darks toward blue-purple, lights toward warm)."""
    px = lab.reshape(-1, 3).astype(np.float64)
    L = px[:, 0]
    lo, hi = np.percentile(L, 3), np.percentile(L, 97)
    rl = L_RANGE.get(terrain)
    if rl:
        lo, hi = max(lo, rl[0]), min(max(hi, lo + 12), rl[1])
    if hi - lo < 6 * (n - 1) * 0.5:                    # a flat tile still gets readable steps
        mid = (hi + lo) / 2
        lo, hi = mid - 1.5 * (n - 1), mid + 1.5 * (n - 1)
    levels = np.linspace(lo, hi, n)
    step = (hi - lo) / max(1, n - 1)
    out = np.zeros((n, 3))
    for i, lv in enumerate(levels):
        near = np.abs(L - lv) <= max(step, 3.0)
        pick = px[near] if near.sum() >= 4 else px[np.argsort(np.abs(L - lv))[:16]]
        out[i] = (lv, pick[:, 1].mean(), pick[:, 2].mean())
    s = np.linspace(-1, 1, n)
    shift = HUE_SHIFT.get(terrain, HUE_SHIFT_DEFAULT)
    out[:, 2] += shift * s                              # lights warmer (yellow), darks cooler
    out[:, 1] += -1.5 * s * (1 if shift > 0 else 0)     # darks a touch purple
    if shift < 0:                                       # snow: shadows blue, lights neutral white
        out[:, 2] = np.where(s < 0, out[:, 2] + shift * -s * 2.0, out[:, 2] * (1 - 0.6 * s))
    chroma = np.hypot(out[:, 1], out[:, 2])
    ease = 1 - 0.3 * s ** 2                             # less colourful at both ends
    k = np.where(chroma > 1e-6, ease, 1.0)
    out[:, 1] *= k
    out[:, 2] *= k
    return out


def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    from .color import lab_to_rgb
    return lab_to_rgb(np.asarray(lab, np.float32))


def despeckle(idx: np.ndarray, keep_top: int = 0, passes: int = 2) -> np.ndarray:
    """Lone pixels (no 4-neighbour of their own index, wrapping) take the median index of their
    8 neighbours; up to keep_top lone pixels of the brightest index stay (sparkles / glints)."""
    idx = idx.copy()
    top = idx.max()
    for _ in range(passes):
        nb4 = [np.roll(idx, s, a) for a in (0, 1) for s in (1, -1)]
        same = sum((b == idx).astype(int) for b in nb4)
        lone = same == 0
        if keep_top:
            ys, xs = np.nonzero(lone & (idx == top))
            for y, x in list(zip(ys, xs))[:keep_top]:
                lone[y, x] = False
        if not lone.any():
            break
        nb8 = np.stack([np.roll(np.roll(idx, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                        if dy or dx], -1)
        med = np.sort(nb8, -1)[..., 3:5].mean(-1)
        idx[lone] = np.rint(med[lone]).astype(idx.dtype)
    return idx


def finish_tile(picture, size: int, terrain: str = "", *, palette: np.ndarray | None = None,
                even: bool = True, ramp_n: int | None = None, sharpen: float = 0.5) -> np.ndarray:
    """A wrapping picture (any size) -> size x size RGBA pixel tile: area downscale, large-scale
    brightness evened, contrast set to the terrain's, a hue-shifted ramp fitted to its colours,
    no dither, lone pixels cleaned. palette: the project's locked colours (the ramp snaps to
    them, so the tile keeps <= ramp_n colours)."""
    from .maptiles import low_band
    small = downscale_area(picture, size)
    lab = _lab(np.clip(np.rint(small), 0, 255).astype(np.uint8)).astype(np.float64)
    if even:
        lab[..., 0] -= low_band(lab[..., 0], 1.5) * 0.85
    lo, hi = CONTRAST.get(terrain, (7.0, 14.0))
    sd = float(lab[..., 0].std())
    want = float(np.clip(sd, lo, hi))
    m = float(lab[..., 0].mean())
    lab[..., 0] = m + (lab[..., 0] - m) * (want / max(sd, 1e-3))
    if sharpen:
        lab = sharpen_wrap(lab, sharpen, max(0.7, size / 32))
    n = ramp_n or RAMP_N.get(terrain, 6)
    if size <= 16:
        n = max(4, n - 1)
    ramp = tile_ramp(lab, n, terrain)
    d = ((lab[..., None, 0] - ramp[None, None, :, 0]) ** 2
         + 0.45 * ((lab[..., None, 1:] - ramp[None, None, :, 1:]) ** 2).sum(-1))
    idx = d.argmin(-1)
    sparkle = SPARKLE.get(terrain, 0) * max(1, (size * size) // 1024)
    idx = despeckle(idx, keep_top=sparkle)
    cols = np.clip(_lab_to_rgb(ramp), 0, 255).astype(np.uint8)
    if palette is not None and len(palette):
        from .cards import snap_image
        cols = snap_image(np.dstack([cols[None], np.full((1, n), 255, np.uint8)]), palette)[0, :, :3]
    out = np.zeros((size, size, 4), np.uint8)
    out[..., :3] = cols[idx]
    out[..., 3] = 255
    return out


def neo_tile(terrain: str, size: int, seed: int = 0, palette: np.ndarray | None = None) -> np.ndarray:
    """No GPU (or its painting missed): the structure itself as the finished tile."""
    tex = structure(terrain, size, seed).copy()
    if palette is not None and len(palette):
        from .cards import snap_image
        tex = snap_image(tex, palette)
    tex[..., 3] = 255
    return tex


# --------------------------------------------------------------------------- the GPU paint

PAINT_PROMPT = ("{words}, top-down game ground texture seen from directly above, hand-painted "
                "16-bit pixel art tile, clean readable shapes, flat even lighting")
PAINT_NEGATIVE = ("perspective, horizon, sky, objects, characters, text, border, frame, vignette, "
                  "photo, 3d render, noise, grain, dithering, blurry")
TILE_STYLE_STRENGTH = 0.35          # the project's style LoRA, lowered for tiles (A/B 2026-10-15)


def paint_prompt(terrain: str, words: str = "", style=None) -> tuple[str, str]:
    """(positive, negative) for the paint pass: the terrain's material words lead, the user's
    words follow, the house style is appended (assets.with_style "tile")."""
    from . import assets
    from .tilegen import MATERIAL, TERRAINS
    text = (words or "").strip().rstrip(".,; ") or (TERRAINS[terrain][0] if terrain in TERRAINS
                                                     else terrain)
    material, avoid = MATERIAL.get(terrain, ("", ""))
    if material:
        text = f"{material}, {text}"
    asked = text.lower()
    neg = ", ".join([PAINT_NEGATIVE] + [w for w in (x.strip() for x in avoid.split(","))
                                        if w and w not in asked])
    return assets.with_style(PAINT_PROMPT.format(words=text), neg,
                             style or assets.project_style(), "tile")


def style_lora_opts(style, strength: float | None = TILE_STYLE_STRENGTH) -> dict:
    """gen options that use the style's own LoRA at the tile strength (None = generate's
    default: the current project's at its own strength; 0 = off)."""
    if strength is None:
        return {}
    lora = getattr(style, "style_lora", "") if style is not None else ""
    if not lora or strength <= 0:
        return {"style_lora": ""}
    return {"style_lora": lora, "style_lora_strength": float(strength)}


def paint(init_tile: np.ndarray, terrain: str, words: str = "", *, client=None, seed: int = 0,
          style=None, denoise: float | None = None, lora: float | None = TILE_STYLE_STRENGTH,
          n: int | None = None, soften: float = 0.5, **gen_opts) -> tuple[np.ndarray, np.ndarray]:
    """SDXL repaints an n x n canvas (1024 px) of init_tile: (painted canvas RGB, folded
    seamless picture RGB). n: COPIES (2 = 512 px per game tile, 4 = 256)."""
    from .generate import generate_images
    n = n or COPIES.get(terrain, 2)
    pos, neg = paint_prompt(terrain, words, style)
    canvas = init_canvas(init_tile, 1024 // n, n, soften)
    rgba = np.dstack([canvas, np.full(canvas.shape[:2], 255, np.uint8)])
    opts = {**style_lora_opts(style, lora), **gen_opts}
    (img, _), = generate_images(pos, neg, client=client, seed=seed, init_image=rgba,
                                denoise=denoise or DENOISE.get(terrain, DENOISE_DEFAULT),
                                aspect=1.0, **opts)
    from PIL import Image
    raw = np.asarray(img.convert("RGB").resize(canvas.shape[1::-1], Image.LANCZOS))
    return raw, fold_canvas(raw, n)


def fold_canvas(raw: np.ndarray, n: int) -> np.ndarray:
    """The seamless tile picture of an n x n painted canvas: n = 2 folds all four copies;
    larger n folds the inner 2 x 2 copies (away from the canvas border)."""
    if n <= 2:
        return fold(raw, n)
    s = raw.shape[0] // n
    return fold(raw[s:3 * s, s:3 * s], 2)


# --------------------------------------------------------------------------- drawn tiles

def picture_ramp(picture, n: int, terrain: str = "") -> np.ndarray:
    """(n, 3) RGB hue-shifted ramp from a picture's colours (SDXL's swatch: the material and
    colour the user's words asked for), evenly stepped, at least ~5 L apart."""
    small = downscale_area(picture, 64) if np.asarray(picture).shape[0] >= 64 else \
        np.asarray(picture)[..., :3].astype(np.float64)
    lab = _lab(np.clip(np.rint(small), 0, 255).astype(np.uint8)).astype(np.float64)
    ramp = tile_ramp(lab, n, terrain)
    lo, hi = CONTRAST.get(terrain, (7.0, 14.0))
    span = ramp[-1, 0] - ramp[0, 0]
    want = float(np.clip(span, 5.0 * (n - 1), 9.0 * (n - 1)))
    rl = L_RANGE.get(terrain, (12.0, 92.0))
    mid = float(np.clip(ramp[:, 0].mean(), rl[0] + want / 2, rl[1] - want / 2))
    ramp[:, 0] = np.linspace(mid - want / 2, mid + want / 2, n)
    return np.clip(_lab_to_rgb(ramp), 0, 255).astype(np.uint8)


def swatch_colour(picture, terrain: str = "water") -> np.ndarray:
    """The mean of a picture's saturated blue pixels (water), else a calm blue."""
    import cv2
    px = np.ascontiguousarray(np.asarray(picture)[..., :3]).reshape(-1, 3).astype(np.uint8)
    hsv = cv2.cvtColor(px[None], cv2.COLOR_RGB2HSV)[0].astype(int)
    blue = px[(hsv[:, 0] * 2 >= 170) & (hsv[:, 0] * 2 < 260) & (hsv[:, 1] > 60)]
    return blue.mean(0) if len(blue) >= 50 else np.array([40.0, 120.0, 200.0])


# 2026-10-17 (Kevin: tiles rebuilt at 64 px kept 32-px-sized details, tiny grass tufts): the soft
# painters' details are designed at DETAIL_PX tiles. A bigger tile is drawn at DETAIL_PX (same
# layout, same detail sizes relative to the tile) and brought up by a pixel-art upscaler (Scale2x:
# no blur, no new colours, diagonals stay smooth), then a fine 1-px grain is laid over its flat
# parts, so the ground has full-resolution texture with details the right size for the tile.
DETAIL_PX = 32
FINE_GRAIN = 0.06            # share of flat pixels that take a neighbouring ramp step


def detail_scale(size: int) -> int:
    """How many times bigger than DETAIL_PX a tile is (1 under 48 px; 64 px -> 2)."""
    return max(1, int(round(size / DETAIL_PX))) if size >= 48 else 1


def scale2x(a: np.ndarray) -> np.ndarray:
    """Scale2x / EPX on any (H, W) or (H, W, C) array (equality of whole pixels): wrapping, so a
    seamless tile stays seamless."""
    a = np.asarray(a)
    key = a if a.ndim == 2 else np.ascontiguousarray(a).view(
        np.dtype((np.void, a.dtype.itemsize * a.shape[2]))).reshape(a.shape[:2])
    B, D = np.roll(key, 1, 0), np.roll(key, 1, 1)      # above, left
    H_, F = np.roll(key, -1, 0), np.roll(key, -1, 1)   # below, right
    Ba, Da = np.roll(a, 1, 0), np.roll(a, 1, 1)
    Ha, Fa = np.roll(a, -1, 0), np.roll(a, -1, 1)
    ok = (B != H_) & (D != F)
    pick = lambda m, src: np.where((m & ok)[..., None] if a.ndim == 3 else (m & ok), src, a)
    e0 = pick(D == B, Da)
    e1 = pick(B == F, Fa)
    e2 = pick(D == H_, Da)
    e3 = pick(H_ == F, Fa)
    out = np.empty((a.shape[0] * 2, a.shape[1] * 2) + a.shape[2:], a.dtype)
    out[0::2, 0::2], out[0::2, 1::2], out[1::2, 0::2], out[1::2, 1::2] = e0, e1, e2, e3
    return out


def upscale_detail(tex: np.ndarray, k: int) -> np.ndarray:
    out = tex
    while k >= 2 and k % 2 == 0:
        out = scale2x(out)
        k //= 2
    if k > 1:
        out = np.repeat(np.repeat(out, k, 0), k, 1)
    return out


def fine_grain(tex: np.ndarray, ramp: np.ndarray, seed: int, share: float = FINE_GRAIN) -> np.ndarray:
    """Single ramp steps sprinkled over the FLAT parts of an upscaled tile (a pixel and its 4
    neighbours the same ramp colour), lighter on the upper-left side of nothing in particular:
    the 1-px texture a tile drawn at full size has."""
    t = tex.shape[0]
    rgb = tex[..., :3]
    ramp = np.asarray(ramp, np.uint8)
    hit = (rgb[:, :, None, :] == ramp[None, None]).all(-1)
    idx = np.where(hit.any(-1), hit.argmax(-1), -1)
    flat = idx >= 0
    for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
        flat &= np.roll(idx, sh, ax) == idx
    rng = _rng("grain", t, seed)
    r = rng.random((t, t))
    up = flat & (r < share / 2) & (idx < len(ramp) - 1)
    down = flat & (r > 1 - share / 2) & (idx > 0)
    out = tex.copy()
    out[up, :3] = ramp[idx[up] + 1]
    out[down, :3] = ramp[idx[down] - 1]
    return out


def drawn_tile(terrain: str, size: int, seed: int = 0, picture=None,
               palette: np.ndarray | None = None, ramp: np.ndarray | None = None,
               words: str = "") -> np.ndarray:
    """A soft terrain drawn by code (grass, dirt, sand, snow: SOFT painters; water: neo
    ripples) in colours taken from `picture` (SDXL's swatch) or `ramp`, else the hand-built
    ramp. Same seed + ramp = same tile; other seeds = other layouts in the same colours.
    words: the user's words also shape it (tilewords: "mossy" -> moss clumps, "cracked" ->
    cracks, "flower meadow" -> flowers, "rippled dune" -> long ripples ...)."""
    from .neo_terrain import _colour
    from .tilewords import details, stamp_details
    d = details(terrain, words)
    if terrain == "water":
        from .tilegen import neo_water_tile
        colour = swatch_colour(picture) if picture is not None else (40, 120, 200)
        if ramp is not None:
            colour = ramp[len(ramp) // 2]
        tex = neo_water_tile(colour, size, seed)
    else:
        fn, _, n = SOFT[terrain]
        if ramp is None:
            ramp = drawn_ramp(terrain, picture)
        ramp = np.asarray(ramp)
        k = detail_scale(size) if terrain in ("grass", "dirt", "sand", "snow") else 1
        if size % k:
            k = 1
        small = size // k
        idx = fn(small, seed, d) if not d.plain() else fn(small, seed)
        tex = _colour(ramp, idx)
        if not d.plain():
            tex = stamp_details(tex, np.clip(idx, 0, len(ramp) - 1), terrain, d, seed, ramp)
        if k > 1:                       # details the right size for a big tile (DETAIL_PX)
            tex = fine_grain(upscale_detail(tex, k), ramp, seed)
    if palette is not None and len(palette):
        from .cards import snap_image
        tex = snap_image(tex, palette)
    tex[..., 3] = 255
    return tex


def texture_ramp(tex: np.ndarray) -> np.ndarray:
    """A drawn tile's own colours dark -> light (its ramp)."""
    from .color import rgb_to_lab
    cols = np.unique(np.asarray(tex)[..., :3].reshape(-1, 3), axis=0)
    return cols[np.argsort(rgb_to_lab(cols.astype(np.float32))[:, 0])]


def drawn_variants(terrain: str, tex: np.ndarray, seed: int, count: int = 3,
                   ramp: np.ndarray | None = None, words: str = "") -> list[np.ndarray]:
    """More layouts of a drawn tile (other seeds, the same ramp and words): variety() sources.
    ramp None: the tile's own colours, used only when they are a full ramp (else []). Water:
    none (its ripples vary under variety's rolls)."""
    if terrain not in SOFT:
        return []
    if ramp is None:
        ramp = texture_ramp(tex)
        if len(ramp) != SOFT[terrain][2]:
            return []
    return [drawn_tile(terrain, tex.shape[0], seed + 101 * (k + 1), ramp=ramp, words=words)
            for k in range(count)]


def drawn_ramp(terrain: str, picture=None) -> np.ndarray | None:
    """The ramp drawn_tile uses for this picture (None for water / unknown terrains)."""
    if terrain not in SOFT:
        return None
    ramp = soft_ramp(terrain)
    if picture is None:
        return ramp
    n = SOFT[terrain][2]
    w = SWATCH_WEIGHT.get(terrain, SWATCH_WEIGHT_DEFAULT)
    mixed = (1 - w) * _lab(ramp) + w * _lab(picture_ramp(picture, n, terrain))
    return np.clip(_lab_to_rgb(mixed), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- stones re-rendered

STONY = ("road", "stone", "cliff")
GAP_SHARE = {"road": 0.24, "stone": 0.16, "cliff": 0.26}      # darkest share of a tile = joints
ROUND = {"road": True, "cliff": True, "stone": False}            # rounded corners
MARKER_GAP = {"road": 0.09, "cliff": 0.13, "stone": 0.12}     # stone centres apart (x tile)
STONE_L = {"road": (14.0, 84.0), "stone": (16.0, 86.0), "cliff": (12.0, 80.0)}   # ramp lightness
# stone shapes (2026-10-16, Kevin: "a few stones merge into odd shapes"): a stone smaller than
# min px joins the joint, one larger than max x tile area, more elongated than ASPECT_MAX or less
# convex than SOLIDITY_MIN (an L / T / dumbbell where two painted stones ran together) is split
# in two along a new 1 px joint, up to SPLIT_PASSES times.
STONE_AREA = {"road": (5, 0.13), "cliff": (7, 0.18), "stone": (6, 0.2)}
SPLIT_MIN_SHARE = 0.3                # each half of a split keeps >= this share of the stone
SOLIDITY_MIN = 0.78
ASPECT_MAX = 2.6
SPLIT_PASSES = 3


def _unwrap(ys: np.ndarray, xs: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """A wrapping stone's pixel coordinates made contiguous (a stone cut by the tile edge)."""
    ys, xs = ys.astype(float), xs.astype(float)
    for a in (ys, xs):
        if len(a) and a.min() == 0 and a.max() == size - 1:
            # the largest empty run of rows / columns is where the stone does not go: cut there
            occ = np.zeros(size, bool)
            occ[a.astype(int)] = True
            if occ.all():
                continue
            gaps = np.nonzero(~occ)[0]
            a[a < gaps[0]] += size
    return ys, xs


def stone_shape(ys: np.ndarray, xs: np.ndarray, size: int) -> tuple[int, float, float]:
    """(area px, solidity = area / convex hull area, aspect = long / short axis) of one stone."""
    n = len(ys)
    if n < 3:
        return n, 1.0, 1.0
    y, x = _unwrap(ys, xs, size)
    pts = np.concatenate([np.stack([y + dy, x + dx], 1) for dy in (0, 1) for dx in (0, 1)])
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts).volume
    except Exception:  # noqa: BLE001 - degenerate (a straight line of pixels)
        hull = float(n)
    cov = np.cov(np.stack([y, x])) + np.eye(2) / 12.0
    ev = np.sort(np.linalg.eigvalsh(cov))
    return n, float(min(1.0, n / max(hull, 1e-6))), float(np.sqrt(ev[1] / max(ev[0], 1e-6)))


def _halves(y: np.ndarray, x: np.ndarray, rng_key: int = 0) -> np.ndarray:
    """2-means of a stone's pixels (started from its two ends along the long axis): bool, True =
    the second half."""
    pts = np.stack([y, x], 1)
    c = pts.mean(0)
    w, v = np.linalg.eigh(np.cov(pts.T) + np.eye(2) / 12.0)
    proj = (pts - c) @ v[:, -1]
    cent = np.stack([pts[proj.argmin()], pts[proj.argmax()]]).astype(float)
    side = proj > 0
    for _ in range(8):
        d = ((pts[:, None] - cent[None]) ** 2).sum(-1)
        side = d[:, 1] < d[:, 0]
        if side.all() or (~side).all():
            break
        cent = np.stack([pts[~side].mean(0), pts[side].mean(0)])
    return side


def split_stones(stones: np.ndarray, gap: np.ndarray, size: int, terrain: str = "road",
                 passes: int = SPLIT_PASSES) -> tuple[np.ndarray, np.ndarray]:
    """Stones that are too big, too long or not convex (stone_shape) split in two by a 1 px joint
    (2-means of their pixels); crumbs below the minimum size join the joint. Returns new
    (stones relabelled 1..n, gap)."""
    stones, gap = stones.copy(), gap.copy()
    lo, hi = STONE_AREA.get(terrain, (4, 0.1))
    big = hi * size * size
    for _ in range(max(0, passes)):
        changed = False
        nxt = int(stones.max()) + 1
        for v in range(1, int(stones.max()) + 1):
            on = stones == v
            if not on.any():
                continue
            ys, xs = np.nonzero(on)
            area, solid, aspect = stone_shape(ys, xs, size)
            if area < 2 * lo + 2 or (area <= big and solid >= SOLIDITY_MIN and aspect <= ASPECT_MAX):
                continue
            y, x = _unwrap(ys, xs, size)
            side = _halves(y, x)
            least = max(lo + 2, SPLIT_MIN_SHARE * area)     # no crumbs: two real stones
            if side.sum() < least or (~side).sum() < least:
                continue
            b = np.zeros((size, size), bool)
            b[ys[side], xs[side]] = True
            a = on & ~b
            # the joint: pixels of the smaller half that touch the other half (4-neighbours)
            small, other = (b, a) if b.sum() <= a.sum() else (a, b)
            touch = small & (np.roll(other, 1, 0) | np.roll(other, -1, 0) |
                             np.roll(other, 1, 1) | np.roll(other, -1, 1))
            if (small & ~touch).sum() < lo:
                continue
            gap |= touch
            stones[touch] = 0
            stones[b & ~touch] = nxt
            nxt += 1
            changed = True
        if not changed:
            break
    # relabel by wrapped connectivity (a split can leave a crumb) and drop crumbs into the joint
    out = np.zeros_like(stones)
    count = 0
    for v in np.unique(stones[stones > 0]):
        part, k = _wrap_label(stones == v)
        for j in range(1, k + 1):
            m = part == j
            if m.sum() < lo:
                gap |= m
                continue
            count += 1
            out[m] = count
    out[gap] = 0
    return out, gap


def _wrap_label(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """4-connected components of a wrapping mask: (labels 1..n, 0 = off), n."""
    from scipy import ndimage as ndi
    lab, n = ndi.label(mask)
    parent = list(range(n + 1))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in list(zip(lab[0], lab[-1])) + list(zip(lab[:, 0], lab[:, -1])):
        if a and b:
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[ra] = rb
    roots = {r: i + 1 for i, r in enumerate(sorted({find(k) for k in range(1, n + 1)}))}
    remap = np.zeros(n + 1, int)
    for k in range(1, n + 1):
        remap[k] = roots[find(k)]
    return remap[lab], len(roots)


def segment_stones(picture, size: int, terrain: str = "road") -> tuple[np.ndarray, np.ndarray]:
    """(stone id per pixel, joint mask) of a wrapping stony picture at the tile size. Watershed
    on the painting's lightness (large-scale light evened out) from one marker per stone (local
    maxima at least ~1/10 tile apart), on a 3 x 3 wrap-around copy so stones run across the tile
    edges; ids are the markers' tile positions, so a stone cut by the edge stays one stone.
    Joints: the watershed lines at 2x, 1 px at the tile size."""
    import cv2
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    from .maptiles import low_band
    rgb = np.ascontiguousarray(np.asarray(picture)[..., :3]).astype(np.uint8)
    k2 = size * 2
    mid = cv2.resize(rgb, (k2, k2), interpolation=cv2.INTER_AREA)
    L = cv2.cvtColor(mid, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float64)
    L -= low_band(L, 2.0)
    Ls = _wrap_blur(L.astype(np.float32), max(1.0, k2 / 48))
    big = np.tile(Ls, (3, 3))
    md = max(2, int(round(k2 * MARKER_GAP.get(terrain, 0.1))))
    peaks = peak_local_max(big, min_distance=md, exclude_border=False)
    markers = np.zeros(big.shape, np.int32)
    ident = {}
    for y, x in peaks:
        key = (int(y) % k2, int(x) % k2)
        ident.setdefault(key, len(ident) + 1)
        markers[y, x] = ident[key]
    lab = watershed(-big, markers, watershed_line=True)[k2:2 * k2, k2:2 * k2]
    line = lab == 0
    joint = line.reshape(size, 2, size, 2).sum((1, 3)) >= 2
    ids = np.zeros((size, size), np.int32)
    blocks = lab.reshape(size, 2, size, 2).transpose(0, 2, 1, 3).reshape(size, size, 4)
    for y in range(size):
        for x in range(size):
            v = blocks[y, x][blocks[y, x] > 0]
            ids[y, x] = np.bincount(v).argmax() if len(v) else 0
    return ids, joint


def joints(picture, size: int, terrain: str = "road") -> np.ndarray:
    """The joints between stones of a wrapping picture at the tile size (segment_stones)."""
    return segment_stones(picture, size, terrain)[1]


def stone_tile(picture, size: int, terrain: str = "road", palette: np.ndarray | None = None,
               ramp_n: int | None = None) -> np.ndarray:
    """A stony picture (cobbles, boulders, slabs SDXL painted) re-drawn as pixel art: its joints
    found (joints), each stone between them one clean shape in its own tone of a hue-shifted
    ramp (stone-to-stone variety from the painting), a lit rim on its upper-left edge, a shadow
    rim on its lower right, a glint on big stones; joints dark, deepest under the stones."""
    small = downscale_area(picture, size)
    lab = _lab(np.clip(np.rint(small), 0, 255).astype(np.uint8)).astype(np.float64)
    n = ramp_n or RAMP_N.get(terrain, 7)
    lo, hi = CONTRAST.get(terrain, (10.0, 16.0))
    sd = float(lab[..., 0].std())
    m = float(lab[..., 0].mean())
    lab2 = lab.copy()
    lab2[..., 0] = m + (lab[..., 0] - m) * (float(np.clip(sd, lo, hi)) / max(sd, 1e-3))
    ramp = tile_ramp(lab2, n, terrain)
    lo_l, hi_l = STONE_L.get(terrain, (14.0, 80.0))
    shift = float(np.clip(lab[..., 0].mean() - 42.0, -10, 10))    # a pale / dark material stays so
    ramp[:, 0] = np.linspace(lo_l + shift, hi_l + shift, n)
    ids, gap = segment_stones(picture, size, terrain)
    gap = gap | (ids == 0)
    for _ in range(2):                      # dead-end joint spurs (lines into a stone) go
        nb = sum(np.roll(gap, s_, a).astype(int) for a in (0, 1) for s_ in (1, -1))
        gap &= nb >= 2
    # each stone = the pixels of one watershed id, split where a joint cuts it
    stones = np.zeros((size, size), int)
    count = 0
    for v in np.unique(ids[~gap]):
        part, k = _wrap_label((ids == v) & ~gap)
        for j in range(1, k + 1):
            on = part == j
            if on.sum() < max(3, size * size // 300):              # crumbs join the joint
                gap |= on
                continue
            count += 1
            stones[on] = count
    # stones that touch without a joint between them get one along their shared edge
    diff = ((stones != np.roll(stones, -1, 0)) & (np.roll(stones, -1, 0) > 0)) |         ((stones != np.roll(stones, -1, 1)) & (np.roll(stones, -1, 1) > 0))
    gap |= diff & (stones > 0)
    stones[gap] = 0
    stones, gap = split_stones(stones, gap, size, terrain)
    count = int(stones.max())
    if ROUND.get(terrain, True):                               # round the stones' corners
        same = lambda dy, dx: np.roll(np.roll(stones, dy, 0), dx, 1) == stones
        area = np.bincount(stones.ravel())[stones]
        corner = np.zeros_like(gap)
        for (ay, ax), (by, bx) in (((1, 0), (0, 1)), ((1, 0), (0, -1)), ((-1, 0), (0, 1)), ((-1, 0), (0, -1))):
            corner |= ~same(ay, ax) & ~same(by, bx)
        corner &= (stones > 0) & (area >= 7)
        gap |= corner
        stones[corner] = 0
    up = lambda a: np.roll(a, 1, 0)        # the pixel above / to the left / below / right
    left = lambda a: np.roll(a, 1, 1)
    down = lambda a: np.roll(a, -1, 0)
    right = lambda a: np.roll(a, -1, 1)
    lit_rim = ~gap & (up(gap) | left(gap))
    dark_rim = ~gap & (down(gap) | right(gap)) & ~lit_rim
    Ls = ramp[:, 0]
    from scipy import ndimage as ndi
    dist = ndi.distance_transform_edt(np.tile(~gap, (3, 3)))[size:2 * size, size:2 * size]
    # stone tints: each stone's own a/b (the painting's warm / cool stones), clustered to 3
    from .maptiles import low_band
    even = lab.copy()
    even[..., 0] -= low_band(lab[..., 0], 2.0)                 # the painting's light across the tile
    means = []
    for s_ in range(1, count + 1):
        on = stones == s_
        means.append(even[on].mean(0) if on.any() else even.reshape(-1, 3).mean(0))
    means = np.array(means) if means else np.zeros((0, 3))
    tints = _tints(means[:, 1:], 3) if len(means) else np.zeros((0, 2))
    out_lab = np.zeros((size, size, 3))
    ab0 = ramp[:, 1:]
    rank = np.argsort(np.argsort(means[:, 0])) if len(means) else np.zeros(0, int)
    span = max(1, n - 5)                                       # bodies 2 .. n-3: rims + glint fit
    body = [2 + int(round(span * r / max(1, len(rank) - 1))) for r in rank]

    def colour(level, tint):
        lv = int(np.clip(level, 0, n - 1))
        return np.array([Ls[lv], *(0.45 * ab0[lv] + 0.55 * tint)])
    for s_ in range(1, count + 1):
        on = stones == s_
        if not on.any():
            continue
        tint = tints[s_ - 1]
        level = body[s_ - 1]
        out_lab[on] = colour(level, tint)
        out_lab[on & dark_rim] = colour(level - 1, tint)
        out_lab[on & lit_rim] = colour(level + 1, tint)
        if on.sum() >= 16:                                     # glint: upper-left inside
            ys, xs = np.nonzero(on & (dist >= 1.5) & ~lit_rim)
            if len(ys):
                j = int(np.argmin(ys + xs))
                out_lab[ys[j], xs[j]] = colour(level + 2, tint)
                if on.sum() >= 24 and xs[j] + 1 < size and on[ys[j], xs[j] + 1]:
                    out_lab[ys[j], xs[j] + 1] = colour(level + 1, tint)
    out_lab[gap] = ramp[1]
    out_lab[gap & (up(~gap) | left(~gap))] = ramp[0]           # deepest below / right of a stone
    cols = np.clip(_lab_to_rgb(out_lab.reshape(-1, 3)), 0, 255).astype(np.uint8).reshape(size, size, 3)
    out = np.zeros((size, size, 4), np.uint8)
    out[..., :3] = cols
    out[..., 3] = 255
    if palette is not None and len(palette):
        from .cards import snap_image
        out = snap_image(out, palette)
    return out


def _tints(ab: np.ndarray, k: int) -> np.ndarray:
    """Each row's a/b snapped to one of k shared tints (1-D k-means along the warm-cool axis)."""
    if len(ab) <= k:
        return ab.copy()
    axis = ab[:, 1] + 0.3 * ab[:, 0]                            # warm (yellow / red) = +
    edges = np.quantile(axis, np.linspace(0, 1, k + 1))
    which = np.clip(np.searchsorted(edges[1:-1], axis), 0, k - 1)
    out = ab.copy()
    for c in range(k):
        if (which == c).any():
            out[which == c] = ab[which == c].mean(0)
    return out


def variant_canvas(terrain: str, t: int, seed: int, base: np.ndarray | None = None,
                   soften: float = 0.5) -> np.ndarray:
    """A 2 x 2 canvas (1024 px) of four OTHER layouts of the terrain (seeds seed+1..4): one
    SDXL pass paints four interiors for variety() (their edges are replaced by the base's band,
    so they needn't wrap)."""
    tiles = [structure(terrain, t, seed + 1 + k) for k in range(4)]
    big = [init_canvas(x, 512, 1, soften) for x in tiles]
    return np.concatenate([np.concatenate(big[:2], 1), np.concatenate(big[2:], 1)], 0)


def paint_variants(terrain: str, words: str = "", *, size: int, client=None, seed: int = 0,
                   style=None, denoise: float | None = None, lora: float | None = TILE_STYLE_STRENGTH,
                   palette: np.ndarray | None = None, **gen_opts) -> tuple[np.ndarray, list[np.ndarray]]:
    """(painted variant canvas RGB, [4 stone tiles]) - other stone layouts, same material."""
    from .generate import generate_images
    pos, neg = paint_prompt(terrain, words, style)
    canvas = variant_canvas(terrain, size, seed)
    rgba = np.dstack([canvas, np.full(canvas.shape[:2], 255, np.uint8)])
    opts = {**style_lora_opts(style, lora), **gen_opts}
    (img, _), = generate_images(pos, neg, client=client, seed=seed + 500, init_image=rgba,
                                denoise=denoise or DENOISE.get(terrain, DENOISE_DEFAULT),
                                aspect=1.0, **opts)
    from PIL import Image
    raw = np.asarray(img.convert("RGB").resize(canvas.shape[1::-1], Image.LANCZOS))
    return raw, canvas_tiles(raw, terrain, size, palette)


def canvas_tiles(raw: np.ndarray, terrain: str, size: int, palette=None) -> list[np.ndarray]:
    h = raw.shape[0] // 2
    return [stone_tile(raw[i * h:(i + 1) * h, j * h:(j + 1) * h], size, terrain, palette)
            for i in (0, 1) for j in (0, 1)]


# --------------------------------------------------------------------------- cliff faces

# Cliff / wall FACES drawn the tiles v3 way (2026-10-16, Kevin: "cliff faces still the old
# procedural blocks"): a clean boulder-and-strata layout at face scale (strata bands whose joints
# step, split into boulders by wobbly leaning cuts), each boulder re-drawn like stone_tile's stones
# (its own tone, one of 3 tints taken from the cliff tile's stones, lit top / left rim, shadow rim
# bottom / right, a glint, rounded corners, 1 px joints deepest under the stones), moss creeping
# along joints. Same signature as neo_terrain.rock_face, so CliffArt's strip and the HD-2D side
# walls call this instead (FACE_STYLE "boulders"; "strata" = the old neo blocks).
FACE_STYLE = "boulders"
FACE_BAND = (1 / 3.4, 1 / 2.1)            # strata height (x unit)
FACE_STONE_W = (0.38, 0.9)                # boulder width (x unit)


def stone_tints(tex: np.ndarray | None, k: int = 3) -> np.ndarray:
    """k (a, b) tint offsets from a stony tile's mid-tone pixels (warm / neutral / cool stones),
    or small neutral ones without a tile."""
    if tex is None:
        return np.array([[1.5, 4.0], [0.0, 0.0], [-0.5, -4.0]])[:k]
    lab = _lab(np.asarray(tex)[..., :3].reshape(-1, 3)).astype(np.float64)
    L = lab[:, 0]
    mid = lab[(L > np.percentile(L, 30)) & (L < np.percentile(L, 85))]
    if len(mid) < k:
        mid = lab
    axis = mid[:, 2] + 0.3 * mid[:, 1]
    edges = np.quantile(axis, np.linspace(0, 1, k + 1))
    which = np.clip(np.searchsorted(edges[1:-1], axis), 0, k - 1)
    base = mid[:, 1:].mean(0)
    return np.array([(mid[which == c, 1:].mean(0) if (which == c).any() else base) - base
                     for c in range(k)])


def face_layout(w: int, h: int, seed: int = 0, *, seams=(), unit: int | None = None,
                band: tuple[float, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(stone id per pixel, joint mask) of a rock face that wraps in x and y: strata bands (their
    top joints step up / down a pixel, except at `seams` rows, which stay straight so tiles cut
    there line up), each band split into boulders by cuts that lean and wobble."""
    from .neo_terrain import _steps
    rng = _rng("bface", w, h, seed)
    t = unit or w
    lo, hi = band or (t * FACE_BAND[0], t * FACE_BAND[1])
    lo, hi = max(3, int(round(lo))), max(3, int(round(hi)))
    seam_rows = sorted({int(s) % h for s in seams} | {0})
    tops, fixed = [], []
    for i, c0 in enumerate(seam_rows):
        c1 = seam_rows[i + 1] if i + 1 < len(seam_rows) else h
        y = c0
        while y < c1:
            tops.append(y)
            fixed.append(y == c0)
            step = int(rng.integers(lo, hi + 1))
            if c1 - (y + step) < lo:
                step = c1 - y
            y += step
    nb = len(tops)
    wob = np.stack([np.zeros(w, int) if f else _steps(w, 1, rng, (3, 8)) for f in fixed])
    yy, xx = np.mgrid[0:h, 0:w]
    edges = np.array(tops)[:, None] + wob
    band_of = (yy[None] >= edges[:, None, :]).sum(0) - 1
    band_of = np.where(band_of < 0, nb - 1, band_of)
    start = edges[band_of, xx]
    start = np.where(yy < start, start - h, start)
    nxt = np.where(band_of + 1 < nb, edges[np.minimum(band_of + 1, nb - 1), xx], edges[0, xx] + h)
    row_in = yy - start
    rows_left = nxt - yy
    ids = np.zeros((h, w), int)
    cutmask = np.zeros((h, w), bool)
    nid = 0
    for b in range(nb):
        cuts, x = [], float(rng.uniform(0, t * 0.5))
        while x < w:
            cuts.append(int(x))
            x += float(rng.uniform(t * FACE_STONE_W[0], t * FACE_STONE_W[1]))
        if len(cuts) > 1 and w - cuts[-1] + cuts[0] < t * FACE_STONE_W[0] * 0.7:
            cuts.pop()                                              # no sliver across the wrap
        lean = float(rng.choice((-0.25, -0.12, 0.0, 0.12, 0.25)))
        m = band_of == b
        cut_x = np.stack([c + lean * (row_in - hi / 2) + _steps(h, 1, rng, (4, 9))[yy] * 0.6
                          for c in cuts])
        rel = (xx[None] - np.rint(cut_x)) % w                       # px right of each cut
        k_idx = rel.argmin(0)
        ids[m] = (nid + k_idx)[m]
        cutmask |= (rel.min(0) < 1.0) & m
        nid += len(cuts)
    joint = (rows_left == 1) | cutmask
    return ids + 1, joint


def boulder_face(w: int, h: int, ramp: np.ndarray | None = None, seed: int = 0, *, seams=(),
                 band: tuple[float, float] | None = None, moss: np.ndarray | None = None,
                 unit: int | None = None, tints: np.ndarray | None = None, bricks: bool = False):
    """A rock face (RGBA, height) in the stone_tile look: face_layout's boulders, each one tone
    of the ramp (+ a shared tint), lit from the upper left, rounded, in dark joints; moss (a
    ramp) creeps along some joints. Wraps in x and y. bricks=True or FACE_STYLE "strata":
    neo_terrain.rock_face."""
    from scipy import ndimage as ndi
    from .neo_terrain import RAMPS, pnoise, rock_face
    if bricks or FACE_STYLE != "boulders":
        return rock_face(w, h, ramp, seed, bricks=bricks, seams=seams, band=band, moss=moss,
                         unit=unit)
    ramp = RAMPS["rock"] if ramp is None else np.asarray(ramp)
    n = len(ramp)
    rng = _rng("bface-col", w, h, seed)
    ids, gap = face_layout(w, h, seed, seams=seams, unit=unit, band=band)
    raw = np.where(gap, 0, ids)
    stones = np.zeros_like(raw)
    count = 0
    for v in np.unique(raw[raw > 0]):                 # split each id where joints cut it
        part, k = _wrap_label(raw == v)               # (wrapping: a stone across the edge is one)
        for j in range(1, k + 1):
            on = part == j
            if on.sum() < 4:                          # crumbs join the joint
                gap |= on
                continue
            count += 1
            stones[on] = count
    stones[gap] = 0
    # some big boulders split into two rows of smaller stones (mixed sizes, like the tops)
    from .neo_terrain import _steps
    for s_ in range(1, count + 1):
        on = stones == s_
        ys, xs = np.nonzero(on)
        if len(ys) < 60 or ys.max() - ys.min() > h // 2 or ys.max() - ys.min() < 9                 or rng.random() > 0.45:
            continue
        mid = int(np.median(ys)) + _steps(w, 1, rng, (3, 7))
        cut = on & (np.arange(h)[:, None] == mid[None, :])
        if cut.sum() < 4:
            continue
        gap |= cut
        lower = on & (np.arange(h)[:, None] > mid[None, :])
        count += 1
        stones[lower] = count
        stones[cut] = 0

    def same(dy, dx):
        return np.roll(np.roll(stones, dy, 0), dx, 1) == stones
    area = np.bincount(stones.ravel())[stones]
    corner = np.zeros_like(gap)
    for (ay, ax), (by, bx) in (((1, 0), (0, 1)), ((1, 0), (0, -1)), ((-1, 0), (0, 1)),
                               ((-1, 0), (0, -1))):
        corner |= ~same(ay, ax) & ~same(by, bx)
    corner &= (stones > 0) & (area >= 9)
    gap |= corner
    stones[corner] = 0
    # big boulders: round once more (a 2 px corner reads as a rounded stone, not a slab)
    area = np.bincount(stones.ravel())[stones]
    nb4 = sum(same(dy, dx).astype(int) for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)))
    corner2 = (stones > 0) & (area >= 40) & (nb4 <= 2)
    gap |= corner2
    stones[corner2] = 0
    up_g, left_g = np.roll(gap, 1, 0), np.roll(gap, 1, 1)
    down_g, right_g = np.roll(gap, -1, 0), np.roll(gap, -1, 1)
    lit_rim = ~gap & (up_g | left_g)
    dark_rim = ~gap & (down_g | right_g) & ~lit_rim
    ramp_lab = _lab(ramp.astype(np.uint8)).astype(np.float64)
    tints = stone_tints(None) if tints is None else np.asarray(tints, float)
    span = max(1, n - 5)
    out_lab = np.zeros((h, w, 3))
    height = np.zeros((h, w))
    tone = rng.permutation(count) if count else np.zeros(0, int)
    dist = ndi.distance_transform_edt(np.tile(~gap, (3, 3)))[h:2 * h, w:2 * w]

    def colour(level, tint):
        c = ramp_lab[int(np.clip(level, 0, n - 1))].copy()
        c[1:] += tint
        return c
    tone = rng.permutation(count) if count else np.zeros(0, int)
    # each boulder a dome: lit where it faces the upper left, shaded toward the lower right
    dome = np.clip(dist / 3.0, 0, 1) ** 0.7
    face_lit = np.roll(np.roll(dome, -1, 0), -1, 1) - np.roll(np.roll(dome, 1, 0), 1, 1)
    for s_ in range(1, count + 1):
        on = stones == s_
        tint = tints[int(rng.integers(len(tints)))]
        level = 2 + int(round(span * tone[s_ - 1] / max(1, count - 1)))
        out_lab[on] = colour(level, tint)
        out_lab[on & (face_lit > 0.3) & ~lit_rim] = colour(level + 1, tint)
        out_lab[on & (face_lit < -0.3) & ~dark_rim] = colour(level - 1, tint)
        out_lab[on & dark_rim] = colour(level - 1, tint)
        out_lab[on & lit_rim] = colour(level + 1, tint)
        height[on] = 0.6 + 0.4 * dome[on]
        if on.sum() >= 14:                            # glint: upper-left inside
            ys, xs = np.nonzero(on & (dist >= 1.5) & ~lit_rim)
            if len(ys):
                j = int(np.argmin(ys + xs))
                out_lab[ys[j], xs[j]] = colour(level + 2, tint)
    out_lab[gap] = ramp_lab[1]
    out_lab[gap & (np.roll(~gap, 1, 0) | np.roll(~gap, 1, 1))] = ramp_lab[0]
    rgb = np.clip(_lab_to_rgb(out_lab.reshape(-1, 3)), 0, 255).astype(np.uint8).reshape(h, w, 3)
    out = np.zeros((h, w, 4), np.uint8)
    out[..., :3] = rgb
    out[..., 3] = 255
    if moss is not None and len(moss):
        m = len(moss)
        f = pnoise(h, w, max(1, h // 8), max(1, w // 8), rng)
        mossy = gap & (f > 0.68)
        grow = mossy | (np.roll(mossy, -1, 0) & (f > 0.76))
        out[grow, :3] = np.asarray(moss)[np.clip((f[grow] * m).astype(int) - 1, 0, m - 1)]
        height = np.where(grow, np.maximum(height, 0.5), height)
    return out, height

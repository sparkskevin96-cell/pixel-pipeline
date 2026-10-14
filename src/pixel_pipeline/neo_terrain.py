"""Neo 16-bit terrain painting: the procedural ground, rock and wall textures the HD-2D board,
the iso look and the map editor fall back on, and the cliff faces tilegen draws.

    top, height = paint_top("cobble", 32, seed=0)        # RGBA + a height field (normals)
    face, _ = rock_face(32, 48, RAMPS["rock"], seed=1)   # strata, blocks, cracks; wraps in x and y
    lip = overhang(32, RAMPS["grass"], seed=2)           # grass hanging over a wall's top edge

Look rules (Kevin: "SNES as if 2D consoles never stopped"): hand-built hue-shifted ramps (dark
colours lean cool / purple, light ones warm / yellow), shapes before noise (every stone, board
and tuft is a readable cluster lit from the upper left: light rim, body, dark rim, mortar), no
dither, no single-pixel salt. Every texture wraps (periodic in x and y) so a map never shows a
seam. board_textures() also turns a project's saved tilesets (tilegen.MapLook) into the 3D
board's textures: tops = the set's own plain tiles (its variants), sides = rock faces in the
cliff set's colours, stone tiers = bricks in the wall set's colours, planks = the fence set's
wood; normal maps via normals.from_drawing so torch light rakes across them.
"""
from __future__ import annotations

import json
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# dark -> light; hue-shifted by hand (shadows cool, lights warm)
RAMP_KEYS = {
    "grass": "142a2c 1b4334 266238 3d8338 64a43c 9cc850 d4e57e",
    "dirt": "2a1a27 45282e 633a35 82523f 9f6d4b bc8c5d d6b07c",
    "sand": "4b3436 7a584c a57e5e c9a473 e2c48c f2e2b4",
    "snow": "30335c 4a5689 6e82b2 9ab0d6 c4d6ec ecf3fb ffffff",
    "stone": "201f2c 37384a 515668 6e7686 8e98a3 b2bbbf d8dcd6",
    "cobble": "1f1c29 36313e 524b57 72686e 938886 b5a99f d8cdbc",
    "rock": "19182a 2b2a3d 434459 5d5e70 7a7883 9a9490 bdb3a5",
    "brick": "1c1a27 33313e 4d4a56 6a656d 89827e aa9f92 cdc1ab",
    "planks": "28150f 482418 6b3a22 8b5430 aa6f40 c68f58 e0b47a",
    "moss": "16302a 22482f 37683a 5a8c3e 8cb24c",
}


def hex_ramp(keys: str, n: int | None = None) -> np.ndarray:
    """Key colours "rrggbb ..." -> (n, 3) uint8, interpolated in Lab (n None = the keys)."""
    from .color import lab_to_rgb, rgb_to_lab
    cols = np.array([[int(k[i:i + 2], 16) for i in (0, 2, 4)] for k in keys.split()], np.float32)
    if n is None or n == len(cols):
        return cols.astype(np.uint8)
    lab = rgb_to_lab(cols)
    pos = np.linspace(0, 1, len(lab))
    t = np.linspace(0, 1, n)
    out = np.stack([np.interp(t, pos, lab[:, k]) for k in range(3)], -1).astype(np.float32)
    return lab_to_rgb(out)


RAMPS = {k: hex_ramp(v) for k, v in RAMP_KEYS.items()}


def ramp_of(tex: np.ndarray, n: int = 7) -> np.ndarray:
    """A texture's own colours as an n-step dark -> light ramp (quantiles by Lab lightness)."""
    from .color import rgb_to_lab
    cols = tex[..., :3].reshape(-1, 3)
    if tex.shape[-1] == 4:
        cols = cols[tex[..., 3].reshape(-1) > 0]
    lum = rgb_to_lab(cols.astype(np.float32))[:, 0]
    order = np.argsort(lum, kind="stable")
    qs = np.linspace(0.02, 0.98, n)
    return cols[order[np.clip((qs * (len(order) - 1)).astype(int), 0, len(order) - 1)]].astype(np.uint8)


def neo_ramp(tex: np.ndarray, n: int = 7, lo: float = 14, hi: float = 82, tint: float = 0.7,
             grey: bool = True) -> np.ndarray:
    """A hue-shifted n-step ramp in a texture's tint: lightness lo..hi (Lab), the texture's mean
    a/b (grey: of its less colourful half, so moss on a rock doesn't turn the rock green) scaled
    by tint, darks leaning blue-purple and lights warm, chroma eased off at both ends."""
    from .color import lab_to_rgb, rgb_to_lab
    cols = tex[..., :3].reshape(-1, 3)
    if tex.shape[-1] == 4:
        cols = cols[tex[..., 3].reshape(-1) > 0]
    lab = rgb_to_lab(cols.astype(np.float32))
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    pick = lab[chroma <= np.median(chroma)] if grey else lab
    a, b = pick[:, 1].mean() * tint, pick[:, 2].mean() * tint
    s = np.linspace(-1, 1, n)
    ease = 1 - 0.45 * s ** 2
    out = np.stack([np.linspace(lo, hi, n), a * ease + 3 - s * 1.5, b * ease + 9 * s], -1)
    return lab_to_rgb(out.astype(np.float32))


def _rng(*key) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32("/".join(str(k) for k in key).encode()))


def pnoise(h: int, w: int, cy: int, cx: int, rng) -> np.ndarray:
    """Smooth value noise in [0, 1] that wraps (cy x cx bumps)."""
    grid = rng.random((cy, cx))

    def axis(n, c):
        t = (np.arange(n) + 0.5) * c / n
        i0 = np.floor(t).astype(int) % c
        f = t - np.floor(t)
        return i0, (i0 + 1) % c, f * f * (3 - 2 * f)
    y0, y1, fy = axis(h, cy)
    x0, x1, fx = axis(w, cx)
    top = grid[y0][:, x0] * (1 - fx) + grid[y0][:, x1] * fx
    bot = grid[y1][:, x0] * (1 - fx) + grid[y1][:, x1] * fx
    return top * (1 - fy)[:, None] + bot * fy[:, None]


def _majority(mask: np.ndarray, passes: int = 1) -> np.ndarray:
    """Wrap-around 3x3 majority: no lone pixels, blobs keep their shape."""
    m = mask.astype(np.int8)
    for _ in range(passes):
        s = sum(np.roll(np.roll(m, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1))
        m = (s >= 5).astype(np.int8)
    return m.astype(bool)


def _colour(ramp: np.ndarray, idx: np.ndarray) -> np.ndarray:
    idx = np.clip(np.asarray(idx), 0, len(ramp) - 1)
    out = np.zeros(idx.shape + (4,), np.uint8)
    out[..., :3] = ramp[idx]
    out[..., 3] = 255
    return out


def _scale(n_ramp: int, i: float) -> int:
    """Index i of a 7-step ramp mapped onto a ramp of n_ramp colours."""
    return int(round(i * (n_ramp - 1) / 6))


def _shade(height: np.ndarray) -> np.ndarray:
    """Light from the upper left on a wrapping height field: h[y+1, x+1] - h[y-1, x-1], so a
    slope facing up-left (rising toward the lower right... i.e. lit) is +."""
    return np.roll(np.roll(height, -1, 0), -1, 1) - np.roll(np.roll(height, 1, 0), 1, 1)


# --------------------------------------------------------------------------- stones

def _cells(h: int, w: int, gy: int, gx: int, rng, jitter: float = 0.7):
    """Jittered-grid Voronoi that wraps: (cell id, distance to own point, edge distance)."""
    pts = []
    for i in range(gy):
        for j in range(gx):
            pts.append(((i + 0.5 + jitter * (rng.random() - 0.5)) * h / gy,
                        (j + 0.5 + jitter * (rng.random() - 0.5)) * w / gx))
    pts = np.array(pts)
    yy, xx = np.mgrid[0:h, 0:w] + 0.5
    best = np.full((h, w), np.inf)
    second = np.full((h, w), np.inf)
    ident = np.zeros((h, w), int)
    for k, (py, px) in enumerate(pts):
        dy = np.abs(yy - py)
        dx = np.abs(xx - px)
        d = np.hypot(np.minimum(dy, h - dy), np.minimum(dx, w - dx) * 0.95)
        closer = d < best
        second = np.where(closer, best, np.minimum(second, d))
        ident = np.where(closer, k, ident)
        best = np.where(closer, d, best)
    return ident, best, (second - best) / 2


def cobble(t: int, seed: int = 0, ramp: np.ndarray | None = None):
    """Rounded paving stones: each a lit dome (light rim top-left, dark rim bottom-right) in dark
    mortar; tones vary a step per stone."""
    ramp = RAMPS["cobble"] if ramp is None else ramp
    n = len(ramp)
    rng = _rng("cobble", t, seed)
    g = max(2, round(t / 7))
    ident, _, edge = _cells(t, t, g, g, rng, 0.55)
    mortar = edge < 0.6
    dome = np.clip((edge - 0.4) / max(1.2, t / 16 * 1.6), 0, 1)     # flat top, 1-2 px bevel
    height = np.where(mortar, 0.0, 0.4 + 0.6 * dome)
    tone = rng.integers(-1, 2, g * g)[ident]
    lit = _shade(height)
    idx = _scale(n, 3.2) + tone
    idx = idx + (lit > 0.15).astype(int) + (lit > 0.5).astype(int) - (lit < -0.15).astype(int)
    idx = np.clip(idx, 1, n - 1)
    idx[mortar] = 0
    edge_m = mortar & ~_majority(mortar)          # mortar joints 1 px; lone mortar dots go
    idx[edge_m & (height == 0)] = 0
    return _colour(ramp, idx), height


def flagstones(t: int, seed: int = 0, ramp: np.ndarray | None = None):
    """Big square-cut slabs in running bond, bevelled, a crack in some."""
    ramp = RAMPS["stone"] if ramp is None else ramp
    n = len(ramp)
    rng = _rng("flag", t, seed)
    rows = max(2, t // 8)
    rh = t // rows
    idx = np.zeros((t, t), int)
    height = np.zeros((t, t))
    for r in range(rows):
        y0 = r * rh
        y1 = t if r == rows - 1 else y0 + rh
        cuts = sorted({int(c) for c in rng.choice(np.arange(0, t, max(2, t // 8)),
                                                  size=max(1, t // 12), replace=False)})
        off = int(rng.integers(t))
        cuts = sorted({(c + off) % t for c in cuts})
        for i, c0 in enumerate(cuts):
            c1 = cuts[(i + 1) % len(cuts)] + (t if i + 1 == len(cuts) else 0)
            if len(cuts) == 1:
                c1 = c0 + t
            xs = np.arange(c0, c1) % t
            tone = _scale(n, 3.0) + int(rng.integers(-1, 2))
            for y in range(y0, y1):
                for k, x in enumerate(xs):
                    top, left = y == y0, k == 0
                    bottom, right = y == y1 - 1, k == len(xs) - 1
                    if bottom or right:
                        v = 0 if (bottom and right) or bottom else 1
                        idx[y, x] = v
                        height[y, x] = 0.0
                    elif top or left:
                        idx[y, x] = min(n - 1, tone + 2)
                        height[y, x] = 0.9
                    elif y == y1 - 2 or k == len(xs) - 2:
                        idx[y, x] = max(1, tone - 1)
                        height[y, x] = 0.8
                    else:
                        idx[y, x] = tone
                        height[y, x] = 1.0
            if rng.random() < 0.35 and len(xs) > 4 and y1 - y0 > 4:      # a crack
                x = int(xs[len(xs) // 2])
                y = y0 + 1
                while y < y1 - 2:
                    idx[y, x % t] = max(0, idx[y, x % t] - 2)
                    y += 1
                    x += int(rng.integers(-1, 2))
    return _colour(ramp, idx), height


# --------------------------------------------------------------------------- soft ground

def _patches(t: int, rng, cells: int, levels=(0.38, 0.66)) -> np.ndarray:
    """0/1/2 big soft shapes (thresholded smooth noise, no lone pixels)."""
    f = 0.7 * pnoise(t, t, cells, cells, rng) + 0.3 * pnoise(t, t, cells * 2, cells * 2, rng)
    a = _majority(f > levels[0], 2)
    b = _majority(f > levels[1], 2)
    return a.astype(int) + b.astype(int)


def grass(t: int, seed: int = 0, ramp: np.ndarray | None = None, flowers: bool = False):
    """Lawn in 3 tones of big soft patches + little lit blade tufts (a light tip over a darker
    pair, shadow under) - no salt and pepper."""
    ramp = RAMPS["grass"] if ramp is None else ramp
    n = len(ramp)
    rng = _rng("grass", t, seed)
    base = _patches(t, rng, max(2, t // 10))
    idx = _scale(n, 2.4) + base
    height = base * 0.15
    tufts = max(3, t * t // 36)
    for _ in range(tufts):
        y, x = int(rng.integers(t)), int(rng.integers(t))
        b = idx[y, x]
        idx[y, x] = min(n - 1, b + 1 + (rng.random() < 0.3))      # tip
        idx[(y + 1) % t, (x - 1) % t] = min(n - 1, b + 1)
        idx[(y + 1) % t, (x + 1) % t] = min(n - 1, b + 1)
        idx[(y + 1) % t, x] = max(0, b - 1)                       # shade under the tip
        if rng.random() < 0.4:
            idx[(y + 2) % t, x] = max(0, b - 2)
        height[y, x] += 0.6
        height[(y + 1) % t, (x - 1) % t] += 0.35
        height[(y + 1) % t, (x + 1) % t] += 0.35
    out = _colour(ramp, idx)
    if flowers:
        for _ in range(max(1, t // 16)):
            y, x = int(rng.integers(t)), int(rng.integers(t))
            out[y, x, :3] = (250, 236, 150) if rng.random() < 0.5 else (240, 220, 250)
            out[(y + 1) % t, x, :3] = ramp[max(0, idx[(y + 1) % t, x] - 1)]
    return out, np.clip(height, 0, 1)


def dirt(t: int, seed: int = 0, ramp: np.ndarray | None = None):
    """Packed earth: big worn patches, a few lit pebbles with a shadow, a hairline crack."""
    ramp = RAMPS["dirt"] if ramp is None else ramp
    n = len(ramp)
    rng = _rng("dirt", t, seed)
    base = _patches(t, rng, max(2, t // 10), (0.42, 0.7))
    idx = _scale(n, 2.6) + base
    height = base * 0.2
    for _ in range(max(2, t * t // 70)):                          # pebbles
        y, x = int(rng.integers(t)), int(rng.integers(t))
        w = 2 if rng.random() < 0.6 else 1
        top = min(n - 1, idx[y, x] + 2)
        for dx in range(w):
            idx[y, (x + dx) % t] = top if dx == 0 else max(top - 1, 0)
            height[y, (x + dx) % t] = 0.9
        idx[(y + 1) % t, (x + 1) % t] = 1
        if w == 2:
            idx[(y + 1) % t, (x + 2) % t] = 1
    for _ in range(max(1, t // 24)):                              # cracks
        y, x = int(rng.integers(t)), int(rng.integers(t))
        for _ in range(int(rng.integers(2, 4))):
            idx[y % t, x % t] = 1
            x += 1
            y += int(rng.integers(-1, 2))
    return _colour(ramp, idx), np.clip(height, 0, 1)


def sand(t: int, seed: int = 0, ramp: np.ndarray | None = None):
    """Wind ripples (a lit crest over a soft shadow) and the odd shell / pebble."""
    ramp = RAMPS["sand"] if ramp is None else ramp
    n = len(ramp)
    rng = _rng("sand", t, seed)
    y, x = np.mgrid[0:t, 0:t]
    k = max(2, t // 8)
    wave = np.sin(2 * np.pi * (y * k / t + 0.18 * np.sin(2 * np.pi * x / t + rng.random() * 6)
                               + 0.08 * pnoise(t, t, 2, 2, rng)))
    idx = np.full((t, t), _scale(n, 3.0))
    crest = _majority(wave > 0.72)
    trough = _majority(wave < -0.6)
    idx[crest] += 1
    idx[np.roll(crest, 1, 0) & ~crest] -= 1
    idx[trough & (pnoise(t, t, 2, 2, rng) > 0.5)] -= 1
    for _ in range(max(1, t // 16)):
        py, px = int(rng.integers(t)), int(rng.integers(t))
        idx[py, px] = n - 1
        idx[(py + 1) % t, px] = 1
    return _colour(ramp, idx), (wave + 1) * 0.25


def snow(t: int, seed: int = 0, ramp: np.ndarray | None = None):
    """Soft drifts with cool blue shadows on their lower right, a few sparkles."""
    ramp = RAMPS["snow"] if ramp is None else ramp
    n = len(ramp)
    rng = _rng("snow", t, seed)
    c = max(2, t // 8)
    f = pnoise(t, t, c, c, rng) * 0.75 + pnoise(t, t, 2 * c, 2 * c, rng) * 0.25
    height = f
    lit = _shade(f) * (t / 2)
    idx = np.full((t, t), _scale(n, 4.4))
    idx[_majority(lit > 0.35)] += 1
    idx[_majority(lit < -0.3)] -= 1
    idx[_majority(lit < -0.7)] -= 1
    for _ in range(max(1, t // 12)):
        y, x = int(rng.integers(t)), int(rng.integers(t))
        idx[y, x] = n - 1
    return _colour(ramp, idx), height


def planks(t: int, seed: int = 0, ramp: np.ndarray | None = None, vertical: bool = False):
    """Boards with a lit top edge, dark gaps, grain streaks, butt joints and nail heads."""
    ramp = RAMPS["planks"] if ramp is None else ramp
    n = len(ramp)
    rng = _rng("planks", t, seed)
    bh = max(3, t // 4)
    boards = max(1, t // bh)
    bh = t // boards
    idx = np.zeros((t, t), int)
    height = np.zeros((t, t))
    for b in range(boards):
        y0, y1 = b * bh, (t if b == boards - 1 else (b + 1) * bh)
        tone = _scale(n, 3.0) + int(rng.integers(-1, 2))
        idx[y0:y1] = tone
        height[y0:y1] = 1
        idx[y0] = min(n - 1, tone + 1)                            # lit top edge
        idx[y1 - 1] = 0                                           # gap
        height[y1 - 1] = 0
        if y1 - y0 > 3:
            idx[y1 - 2] = max(1, tone - 1)
        for _ in range(int(rng.integers(1, 3))):                  # grain streaks
            gy = int(rng.integers(y0 + 1, max(y0 + 2, y1 - 2)))
            gx = int(rng.integers(t))
            for dx in range(int(rng.integers(t // 4, t // 2 + 1))):
                idx[gy, (gx + dx) % t] = max(1, tone - 1)
        jx = int(rng.integers(t))                                 # butt joint + nails
        idx[y0:y1 - 1, jx] = 1
        idx[y0:y1 - 1, (jx + 1) % t] = min(n - 1, tone + 1)
        if y1 - y0 >= 4:
            for nx in ((jx - 2) % t, (jx + 3) % t):
                idx[y0 + (y1 - y0) // 2 - 1, nx] = 1
    out = _colour(ramp, idx)
    if vertical:
        return np.ascontiguousarray(np.rot90(out)), np.ascontiguousarray(np.rot90(height))
    return out, height


# --------------------------------------------------------------------------- walls

def _steps(w: int, amp: int, rng, seg: tuple[int, int] = (3, 7)) -> np.ndarray:
    """A wrapping step line along x: runs of `seg` px at offsets -amp..amp (0 at x = 0)."""
    out = np.zeros(w, int)
    x = int(rng.integers(*seg))
    while x < w:
        n = int(rng.integers(*seg))
        out[x:x + n] = int(rng.integers(-amp, amp + 1))
        x += n
    return out


def rock_face(w: int, h: int, ramp: np.ndarray | None = None, seed: int = 0, *, bricks: bool = False,
              seams=(), band: tuple[float, float] | None = None, moss: np.ndarray | None = None,
              unit: int | None = None):
    """A vertical rock wall (or ashlar bricks) seen front-on, wrapping in x and y: strata bands
    (natural rock: their joints step up and down, blocks lean) split into blocks, each block lit
    (light top row and left column, darker bottom / right rim, its own tone and depth), 1 px dark
    joints, a few cracks; moss (a ramp) creeps along some joints. seams: rows that are always
    straight joints (tile edges, so tiles drawn from different pieces still line up). unit: the
    tile size blocks are sized by (default w). (RGBA, height)."""
    ramp = RAMPS["brick" if bricks else "rock"] if ramp is None else ramp
    n = len(ramp)
    rng = _rng("face", w, h, seed, bricks)
    t = unit or w
    lo, hi = band or ((t / 4, t / 4) if bricks else (t / 3.6, t / 2.2))
    lo, hi = max(3, int(round(lo))), max(3, int(round(hi)))
    seam_rows = sorted({int(s) % h for s in seams} | {0})
    tops, fixed = [], []                                          # band top rows
    for i, c0 in enumerate(seam_rows):
        c1 = seam_rows[i + 1] if i + 1 < len(seam_rows) else h
        y = c0
        while y < c1:
            tops.append(y)
            fixed.append(y == c0)
            step = int(rng.integers(lo, hi + 1))
            if c1 - (y + step) < lo:                              # don't leave a sliver
                step = c1 - y
            y += step
    nb = len(tops)
    amp = 0 if bricks else 1
    wob = np.stack([np.zeros(w, int) if f else _steps(w, amp, rng) for f in fixed])   # (nb, w)
    yy, xx = np.mgrid[0:h, 0:w]
    edges = np.array(tops)[:, None] + wob                         # band b starts at edges[b, x]
    # band index per pixel (wrapping: rows above the first edge belong to the last band)
    band_of = (yy[None] >= edges[:, None, :]).sum(0) - 1          # -1 = before band 0
    band_of = np.where(band_of < 0, nb - 1, band_of)
    start = edges[band_of, xx]
    nxt = np.where(band_of + 1 < nb, edges[np.minimum(band_of + 1, nb - 1), xx], edges[0, xx] + h)
    start = np.where(yy < start, start - h, start)
    row_in = yy - start
    rows_left = nxt - yy                                          # 1 = the band's last row
    # blocks: cut positions per band (bricks: running bond; rock: random widths, leaning)
    block = np.zeros((h, w), int)
    col_in = np.zeros((h, w), int)
    cols_left = np.zeros((h, w), int)
    tone_of, depth_of = [], []
    for b in range(nb):
        if bricks:
            bw = max(4, t // 2)
            off = (b % 2) * (bw // 2)
            cuts = sorted({(off + k * bw) % w for k in range(max(1, w // bw))})
        else:
            k = max(1, int(round(w / rng.uniform(t * 0.5, t * 1.0))))
            cuts = sorted({int(c) for c in rng.random(k) * w})
        lean = 0 if bricks or rng.random() < 0.5 else int(rng.choice((-1, 1)))
        m = band_of == b
        x_eff = (xx + lean * (row_in // 3)) % w
        cut = np.asarray(cuts)
        k_idx = np.searchsorted(cut, x_eff, side="right") - 1    # -1 = before the first cut
        k_idx = np.where(k_idx < 0, len(cut) - 1, k_idx)
        x0 = cut[k_idx]
        x1 = np.where(k_idx + 1 < len(cut), cut[np.minimum(k_idx + 1, len(cut) - 1)], cut[0] + w)
        x0 = np.where(x_eff < x0, x0 - w, x0)
        block[m] = (len(tone_of) + k_idx)[m]
        col_in[m] = (x_eff - x0)[m]
        cols_left[m] = (x1 - x_eff)[m]
        for _ in cuts:
            tone_of.append(_scale(n, 3.2 if bricks else 3.0) + int(rng.integers(-1, 2)))
            depth_of.append(0.0 if bricks else float(rng.uniform(-0.25, 0.15)))
    tone = np.asarray(tone_of)[block]
    depth = np.asarray(depth_of)[block]
    joint = (rows_left == 1) | (cols_left == 1)
    idx = tone.copy()
    height = 1.0 + depth
    first, left = row_in == 0, col_in == 0
    idx = np.where(first, tone + 2, np.where(left, tone + 1, idx))
    rim = ((rows_left == 2) | (cols_left == 2)) & ~first
    idx = np.where(rim, tone - 1, idx)
    height = np.where(first, 0.95 + depth, np.where(rim, 0.85 + depth, height))
    # chipped block corners: the joint takes the bottom-right inner corner of some blocks
    if not bricks:
        chip = (rows_left == 2) & (cols_left == 2) & (rng.random((h, w)) < 0.5)
        joint |= chip
    idx = np.clip(idx, 1, n - 1)
    idx[joint] = 0
    height = np.where(joint, 0.0, height)
    if not bricks:
        for _ in range(max(1, w * h // 140)):                     # cracks
            y, x = int(rng.integers(h)), int(rng.integers(w))
            for _ in range(int(rng.integers(2, max(3, t // 6) + 1))):
                if joint[y % h, x % w]:
                    break
                idx[y % h, x % w] = 0
                height[y % h, x % w] = 0.2
                y += 1
                x += int(rng.integers(-1, 2))
    out = _colour(ramp, idx)
    if moss is not None and len(moss):
        m = len(moss)
        f = pnoise(h, w, max(1, h // 8), max(1, w // 8), rng)
        mossy = joint & (f > 0.68)
        grow = mossy | (np.roll(mossy, -1, 0) & (f > 0.76))      # creeps onto the block above
        out[grow, :3] = moss[np.clip((f[grow] * m).astype(int) - 1, 0, m - 1)]
        height = np.where(grow, np.maximum(height, 0.5), height)
    return out, height


def overhang(t: int, ramp: np.ndarray, seed: int = 0, kind: str = "grass", rows: int | None = None):
    """The strip over a wall's top edge (RGBA, transparent below): a lit lip row, then ground
    hanging down in drips / tufts of the top's colours with a soft shadow under them. kind:
    grass (long tufts), soil (short lumps), snow (rounded cap), stone (just a lit bevel)."""
    rows = rows or max(4, t // 4 + 2)
    n = len(ramp)
    rng = _rng("lip", t, seed, kind)
    out = np.zeros((rows, t, 4), np.uint8)
    if kind == "stone":
        out[0, :, :3] = ramp[n - 1]
        out[1, :, :3] = ramp[max(0, n - 3)]
        out[:2, :, 3] = 255
        out[2, :, 3] = 90                                        # soft shadow under the bevel
        return out
    longest = {"grass": max(2, t // 4), "soil": max(1, t // 8), "snow": max(2, t // 6)}.get(kind, 2)
    f = pnoise(1, t, 1, max(2, t // 4), rng)[0]
    length = np.rint(f * longest).astype(int)
    if kind == "grass":                                           # a few long blades
        for x in rng.choice(t, size=max(1, t // 6), replace=False):
            length[x] = min(rows - 2, longest + int(rng.integers(1, 3)))
    elif kind == "snow":                                          # smooth, rounded
        length = np.rint(np.convolve(np.tile(length, 3), np.ones(3) / 3, "same")[t:2 * t]).astype(int)
    for x in range(t):
        L = int(length[x])
        out[0, x, :3] = ramp[n - 1]                               # lit lip
        out[0, x, 3] = 255
        for y in range(1, L + 1):
            v = n - 2 - min(n - 3, y * 3 // max(1, longest))     # darker toward the tip
            out[y, x, :3] = ramp[max(1, v)]
            out[y, x, 3] = 255
        out[L + 1, x, :3] = (0, 0, 0)                             # shadow on the rock
        out[L + 1, x, 3] = 110
        if L + 2 < rows:
            out[L + 2, x, 3] = 45
    return out


# --------------------------------------------------------------------------- per terrain

TOPS = {"grass": grass, "dirt": dirt, "sand": sand, "snow": snow, "stone": flagstones,
        "cobble": cobble, "planks": planks}
LIP_KIND = {"grass": "grass", "dirt": "soil", "sand": "soil", "snow": "snow", "stone": "stone",
            "cobble": "stone", "planks": "stone"}


def paint_top(name: str, t: int = 32, seed: int = 0, ramp: np.ndarray | None = None):
    """A terrain's top texture (RGBA, height). Unknown names / water: maptiles' built-in."""
    fn = TOPS.get(name)
    if fn is None:
        from .maptiles import builtin_texture
        tex = builtin_texture(name if name != "cobble" else "road", t, seed)
        return tex, tex[..., :3].mean(-1) / 255
    return fn(t, seed, ramp)


def paint_side(name: str, t: int = 32, seed: int = 0, ramp: np.ndarray | None = None):
    """A terrain block's side (RGBA, height), one tile = one unit tall: rock strata under
    soft ground, ashlar bricks under stone / cobble, boards under planks."""
    if name == "planks":
        return planks(t, seed + 3, ramp, vertical=True)
    if name in ("stone", "cobble"):
        return rock_face(t, t, ramp if ramp is not None else RAMPS["brick"], seed, bricks=True,
                         seams=(0,))
    if name == "sand":
        return rock_face(t, t, ramp if ramp is not None else hex_ramp(RAMP_KEYS["sand"]), seed,
                         band=(t / 3, t / 2), seams=(0,))
    if name == "water":
        top, _ = paint_top("water", t, seed)
        top[..., :3] = (top[..., :3].astype(np.float32) * 0.6).astype(np.uint8)
        return top, np.zeros((t, t))
    return rock_face(t, t, ramp, seed, seams=(0,), moss=RAMPS["moss"] if name == "grass" else None)


def tile_normals(rgba: np.ndarray, height: np.ndarray | None = None, relief: float = 0.9) -> np.ndarray:
    """A wrapping texture's normal map (normals.from_drawing on a 3x3 copy, centre kept, so the
    edges match); height (0..1) mixes into the brightness relief."""
    from .normals import from_drawing
    src = rgba.copy()
    src[..., 3] = 255
    if height is not None:
        lum = src[..., :3].astype(np.float32).mean(-1)
        mix = np.clip(0.45 * lum + 0.55 * 255 * np.clip(height, 0, 1), 0, 255).astype(np.uint8)
        src[..., :3] = mix[..., None]
    h, w = src.shape[:2]
    big = np.tile(src, (3, 3, 1))
    n = from_drawing(big, bevel=1.0, relief=relief)[h:2 * h, w:2 * w].copy()
    n[..., 3] = 255
    return n


# --------------------------------------------------------------------------- the 3D board's textures

@dataclass
class BoardTex:
    """One terrain on the HD-2D board."""
    tops: list                     # RGBA tiles (variants)
    weights: list
    side: np.ndarray
    lip: np.ndarray | None
    source: str = "built-in"       # or the tileset folder name the tops came from
    top_normals: list = field(default_factory=list)
    side_normal: np.ndarray | None = None


RELIEF = {"cobble": 1.6, "stone": 1.2, "planks": 1.0, "dirt": 0.9, "grass": 0.7, "sand": 0.6,
          "snow": 0.5, "water": 0.2}


def _sets_texture(look, key) -> np.ndarray | None:
    s = look.sets.get(key) if look is not None else None
    if s is None:
        return None
    p = Path(s.asset.folder) / "texture.png"
    if not p.exists():
        return None
    from PIL import Image
    return np.asarray(Image.open(p).convert("RGBA")).copy()


def board_textures(names, t: int = 32, seed: int = 0, look=None, variants: int = 4) -> dict:
    """{terrain: BoardTex} for the board. look (tilegen.MapLook, same tile size): terrains with
    their own saved set top with that set's plain tiles; walls are rock in the cliff set's
    colours (stone tiers: bricks in the wall set's), planks take the fence set's wood."""
    if look is not None and getattr(look, "tile", t) != t:
        look = None
    rock = _sets_texture(look, "cliff")
    wall = _sets_texture(look, "wall")
    wood = _sets_texture(look, "fence")
    rock_ramp = neo_ramp(rock) if rock is not None else None
    wall_ramp = neo_ramp(wall) if wall is not None else None
    out = {}
    for name in names:
        source = "built-in"
        tops, weights = [], []
        own = look.sets.get(name) if look is not None else None
        if own is not None and own.role != "cliff" and own.kind in ("corner", "blob"):
            tiles, w = look.plain(name)
            tops = [np.ascontiguousarray(x) for x in tiles]
            weights = list(w)
            source = own.name
        elif name == "planks" and wood is not None:
            from .tilegen import variety
            tops, weights = variety(wood, variants * 2, seed=seed)
            source = look.sets["fence"].name
        heights = []
        if not tops:
            for v in range(variants):
                tex, hgt = paint_top(name, t, seed * 31 + v)
                tops.append(tex)
                heights.append(hgt)
                weights.append(1.0)
        else:
            heights = [None] * len(tops)
        top_ramp = ramp_of(tops[0])
        if name == "planks":
            side, sh = planks(t, seed + 3, ramp_of(wood) if wood is not None else None, vertical=True)
        elif name in ("stone", "cobble"):
            side, sh = rock_face(t, t, wall_ramp if wall_ramp is not None else RAMPS["brick"], seed,
                                 bricks=True, seams=(0,))
        elif name == "water":
            side, sh = paint_side("water", t, seed)
        else:
            moss = ramp_of(tops[0], 5) if name == "grass" else None
            side, sh = rock_face(t, t, rock_ramp, seed + 7, seams=(0,), moss=moss,
                                 band=(t / 3, t / 2) if name == "sand" else None)
        lip = None if name == "water" else overhang(t, top_ramp if name not in ("stone", "cobble")
                                                    else ramp_of(side), seed, LIP_KIND.get(name, "soil"))
        rel = RELIEF.get(name, 0.8)
        out[name] = BoardTex(tops, weights, side, lip, source,
                             [tile_normals(x, h, rel) for x, h in zip(tops, heights)],
                             tile_normals(side, sh, 1.3))
    return out


def write_board_textures(names, folder, t: int = 32, seed: int = 0, look=None) -> dict:
    """map/tex files for board.gd: <name>_top.png (first variant: what older boards and the
    props read), <name>_tops.png + _tops_n.png (variants side by side + normals),
    <name>_side.png + _side_n.png, <name>_lip.png, and terrain.json (variants, weights,
    sources). Returns terrain.json's dict."""
    from PIL import Image
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    info = {"tile": t, "terrains": {}}
    for name, bt in board_textures(names, t, seed, look).items():
        Image.fromarray(bt.tops[0]).save(folder / f"{name}_top.png")
        Image.fromarray(np.concatenate(bt.tops, 1)).save(folder / f"{name}_tops.png")
        Image.fromarray(np.concatenate(bt.top_normals, 1)).save(folder / f"{name}_tops_n.png")
        Image.fromarray(bt.side).save(folder / f"{name}_side.png")
        Image.fromarray(bt.side_normal).save(folder / f"{name}_side_n.png")
        entry = {"variants": len(bt.tops), "weights": [round(float(w), 3) for w in bt.weights],
                 "source": bt.source}
        if bt.lip is not None:
            Image.fromarray(bt.lip).save(folder / f"{name}_lip.png")
            entry["lip"] = f"{name}_lip.png"
            entry["lip_height"] = bt.lip.shape[0] / t
        info["terrains"][name] = entry
    (folder / "terrain.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return info

"""The drawing engine of the premium effects (spellfx.py): hi-res heat fields -> crisp pixels.

Kevin's direction (2026-10-13): SNES-crisp shapes "as if 2D consoles never stopped", with BIG
hue-shifted palettes and great FX (Sea of Stars, Chained Echoes, Octopath spell effects).

How a frame is made: shapes (ellipses, tapered strokes, facet polygons, 4-point stars, rings,
noise-warped flame fields, physics particles) are drawn as heat fields (0 = a material ramp's
darkest colour, 1 = its white-hot top) at SS x the output size, on four materials drawn back
to front: back (ground light, main ramp), smoke, main, accent (sparks, embers). Each output
pixel = the topmost material covering at least half of it, its averaged heat banded onto that
material's hue-shifted ramp (no dither, no blur). Then the crisp finish: dim specks dropped,
pinholes filled, one-pixel colour islands merged (pixelclean), optional dark outline or bright
rim. Glow = each bright pixel's colour scaled by its emission, same layout, for bloom.

Timing is non-uniform: every effect is a timeline of phases (anticipation -> impact ->
dissipation) with their own frame durations (ms), and physics runs on real time, so spacing
follows the timing like an animator's. Effects may have several animations (tags), e.g.
fireball "fly" (loop) + "impact".

    finish(canvas, ramps) -> (rgba, glow rgba, colours used)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .color import lab_to_rgb, rgb_to_lab
from .palettes import hex_to_rgb

SS = 4                                   # supersampling of the hi-res heat fields
MATS = ("back", "smoke", "main", "accent")
MAT_RAMP = {"back": "main", "smoke": "smoke", "main": "main", "accent": "accent"}
EMIT = {"back": 0.7, "smoke": 0.0, "main": 1.0, "accent": 1.0}
SIZES = {"small": 64, "medium": 96, "large": 128, "huge": 192}

# --------------------------------------------------------------------------- palettes
# key colours dark -> bright per material; ramps are interpolated in Lab (hue shifts as it goes)
THEME_KEYS = {
    "fire": {"main": "2c0a2e 6a1236 a8202e d8401e f2741c fca72a ffd84e fff3a8 fffff0",
             "smoke": "1a1224 2e2034 46303e 624450 806068 a08480",
             "accent": "c2321c f8781c ffc23a fff09a fffff4"},
    "ice": {"main": "121640 1c2c6c 22509c 2c80c4 46b0e0 7ed6f0 b8eef8 e8fcff ffffff",
            "smoke": "26366a 3c5890 5c80b4 86a8d0 b2cee6 dcecf8",
            "accent": "4cb4e8 9ee6fa e4fcff ffffff"},
    "lightning": {"main": "160c44 2c157c 4826b8 5a4ce6 6c84f8 8cbcff b8e2ff e4f6ff ffffff",
                  "smoke": "1c1440 2e2266 463a8c 6458aa 8a7ec4",
                  "accent": "6a6cf0 a8d4ff fff4b0 ffffff"},
    "holy": {"main": "3a2448 6c3a3c a2582c d08a2c eeb63c fcd860 fff09c fffbd8 ffffff",
             "smoke": "4a3a5c 6e5a78 9a8494 c4b0a8 e6dac4",
             "accent": "f0a838 ffe070 fff8c8 ffffff"},
    "poison": {"main": "1a0c2a 341648 4a2a50 3e5a30 4e8a2c 7cba2a b4e03c e2f882 f8ffd4",
               "smoke": "1e1028 34183e 4e2656 6a3a6c 8a5a7c a8807e",
               "accent": "7ec82c c0f04c f4ffa8 ffffff"},
    "dark": {"main": "0c0618 220a34 3e1052 621a74 8a2a98 b246b8 d878d4 f2b4ec ffe8fc",
             "smoke": "0a0612 1a0e26 2c1a3a 42284e 5c3c62",
             "accent": "8a2aa8 d060e0 ffa8f4 fff0ff"},
    "arcane": {"main": "140c3e 241878 3a2cb4 5248e4 6c74f8 8aa8ff b4d8ff e4f4ff ffffff",
               "smoke": "1e1450 34287a 5040a0 7462c0 9c8cdc",
               "accent": "c048e0 f080f0 ffc8fc ffffff"},
    "nature": {"main": "0c2028 0e3a34 145e3a 248a38 4cb432 8cd83a c8f064 f0ffb0 ffffff",
               "smoke": "143024 1e4630 2e6040 4a7c52 6e9a6a",
               "accent": "e0f05c fff8a0 ffffe4 ffffff"},
    "water": {"main": "0c1240 12266e 14469e 1a70c8 2ca0e4 5ccaf0 9ce6f8 dcfaff ffffff",
              "smoke": "1e3260 30508a 4a74ac 6e9cca 9cc6e4 d0eaf8",
              "accent": "38b0f0 a0ecff e8fcff ffffff"},
    "steel": {"main": "1a1838 2a3062 3e5290 5a7cb8 86a8d8 b4d0ee dceeff f4faff ffffff",
              "smoke": "222440 383c60 545a80 7a80a0 a4a8c0",
              "accent": "8ab4f0 d0e8ff ffffff"},
}
SPELL_THEMES = tuple(THEME_KEYS) + ("project",)
RAMP_N = {"main": 14, "smoke": 9, "accent": 7}


def make_ramp(keys, n: int) -> np.ndarray:
    """Key colours (hex string / list) -> n colours interpolated in Lab, dark -> bright."""
    if isinstance(keys, str):
        keys = keys.split()
    lab = rgb_to_lab(hex_to_rgb(list(keys)).astype(np.float32))
    pos = np.linspace(0, 1, len(lab))
    t = np.linspace(0, 1, n)
    out = np.stack([np.interp(t, pos, lab[:, k]) for k in range(3)], -1).astype(np.float32)
    rgb = lab_to_rgb(out).reshape(-1, 3)
    keep = [0] + [i for i in range(1, n) if not np.array_equal(rgb[i], rgb[i - 1])]
    return rgb[keep]


def spell_ramps(theme: str = "fire", colours=None, palette: np.ndarray | None = None) -> dict:
    """{"main", "smoke", "accent"} ramps for a theme; colours = your key colours for main
    (smoke/accent then come from them); palette = snap every ramp to it (project mode)."""
    if colours:
        from .effects import _parse_colours
        rgb = _parse_colours(colours)
        if len(rgb) == 1:
            from .effects import ramp_from_colour
            rgb = ramp_from_colour(rgb[0], 7)
        order = np.argsort(rgb_to_lab(rgb.astype(np.float32))[:, 0], kind="stable")
        keys = ["%02x%02x%02x" % tuple(int(v) for v in c) for c in rgb[order]]
        spec = {"main": keys, "smoke": [keys[0], keys[min(1, len(keys) - 1)]] + ["5a5060", "8a8090"],
                "accent": keys[len(keys) // 2:] + ["ffffff"]}
    else:
        if theme not in THEME_KEYS:
            raise ValueError(f"Unknown theme '{theme}' (use {', '.join(SPELL_THEMES)})")
        spec = THEME_KEYS[theme]
    ramps = {k: make_ramp(v, RAMP_N[k]) for k, v in spec.items()}
    if palette is not None and len(palette):
        from .effects import snap_ramp
        ramps = {k: snap_ramp(v, palette) for k, v in ramps.items()}
    return ramps


# --------------------------------------------------------------------------- maths helpers

def clamp(v, lo=0.0, hi=1.0):
    return np.clip(v, lo, hi) if isinstance(v, np.ndarray) else max(lo, min(hi, v))


def smooth(e0, e1, v):
    v = clamp((v - e0) / (e1 - e0 + 1e-9))
    return v * v * (3 - 2 * v)


def ease_out(v, p=2.0):
    return 1 - (1 - clamp(v)) ** p


def ease_in(v, p=2.0):
    return clamp(v) ** p


def back_out(v, s=1.7):
    """Overshoot then settle (crystals snapping up)."""
    v = clamp(v) - 1
    return 1 + v * v * ((s + 1) * v + s)


def lerp(a, b, t):
    return a + (b - a) * t


class Noise:
    """Smooth periodic value noise in [0, 1) x [0, 1) (so loops wrap): an fbm texture baked
    once, then sampled bilinear with wrap-around (cv2.remap, fast)."""
    TEX = 256

    def __init__(self, rng, cells: int = 4, octaves: int = 3):
        from .effects import _Noise
        n = _Noise(rng, cells, octaves)
        g = (np.arange(self.TEX, dtype=np.float32) + 0.0) / self.TEX
        self.tex = n(g[None, :], g[:, None]).astype(np.float32)

    def __call__(self, u, v):
        u, v = np.broadcast_arrays(np.asarray(u, np.float32), np.asarray(v, np.float32))
        shape = u.shape
        mx = (np.mod(u, 1.0) * self.TEX).astype(np.float32).reshape(1, -1) if u.ndim < 2 else             (np.mod(u, 1.0) * self.TEX).astype(np.float32)
        my = (np.mod(v, 1.0) * self.TEX).astype(np.float32).reshape(mx.shape)
        out = cv2.remap(self.tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        return out.reshape(shape)


# --------------------------------------------------------------------------- the canvas

class Canvas:
    """Hi-res heat fields per material (-1 = empty), coordinates in output pixels."""

    def __init__(self, w: int, h: int, ss: int = SS):
        self.w, self.h, self.ss = w, h, ss
        self.W, self.H = w * ss, h * ss
        self.heat = {m: np.full((self.H, self.W), -1.0, np.float32) for m in MATS}

    def win(self, x0, y0, x1, y1):
        """Hi-res slices + coordinate grids (X row, Y column) of an output-px box, or None."""
        ss = self.ss
        i0, i1 = max(0, int(math.floor(x0 * ss))), min(self.W, int(math.ceil(x1 * ss)))
        j0, j1 = max(0, int(math.floor(y0 * ss))), min(self.H, int(math.ceil(y1 * ss)))
        if i0 >= i1 or j0 >= j1:
            return None
        X = ((np.arange(i0, i1, dtype=np.float32) + 0.5) / ss)[None, :]
        Y = ((np.arange(j0, j1, dtype=np.float32) + 0.5) / ss)[:, None]
        return (slice(j0, j1), slice(i0, i1)), X, Y

    def put(self, mat, sl, inside, heat, mode="max"):
        H = self.heat[mat]
        cur = H[sl]
        inside = np.broadcast_to(inside, cur.shape)
        heat = np.broadcast_to(np.asarray(heat, np.float32), cur.shape)
        if mode == "max":
            H[sl] = np.where(inside, np.maximum(cur, clamp(heat)), cur)
        elif mode == "over":
            H[sl] = np.where(inside, clamp(heat), cur)
        elif mode == "under":          # only where nothing is yet
            H[sl] = np.where(inside & (cur < 0), clamp(heat), cur)
        elif mode == "erase":
            H[sl] = np.where(inside, -1.0, cur)
        elif mode == "dim":            # darken what is there
            H[sl] = np.where(inside & (cur >= 0), clamp(cur - heat), cur)
        else:
            raise ValueError(mode)

    def field(self, mat, box, fn, mode="max"):
        """fn(X, Y) -> (inside, heat) over an output-px box (x0, y0, x1, y1)."""
        wn = self.win(*box)
        if wn is None:
            return
        sl, X, Y = wn
        inside, heat = fn(X, Y)
        self.put(mat, sl, inside, heat, mode)

    # -- primitives
    def ellipse(self, mat, cx, cy, rx, ry=None, heat=1.0, rot=0.0, mode="max", snap=False):
        """heat: number or fn(rho 0 centre..1 edge, X, Y)."""
        ry = rx if ry is None else ry
        if rx <= 0.05 or ry <= 0.05:
            return
        if snap:
            cx, cy = math.floor(cx) + 0.5, math.floor(cy) + 0.5
        R = max(rx, ry) + 1
        c, s = math.cos(rot), math.sin(rot)

        def fn(X, Y):
            dx, dy = X - cx, Y - cy
            u, v = (dx * c + dy * s) / rx, (-dx * s + dy * c) / ry
            rho = np.sqrt(u * u + v * v)
            return rho <= 1, heat(rho, X, Y) if callable(heat) else heat
        self.field(mat, (cx - R, cy - R, cx + R, cy + R), fn, mode)

    def stroke(self, mat, pts, widths, heats, mode="max", edge=0.0):
        """Tapered polyline: widths/heats per point (or one number); edge = how much heat
        drops from the centre line to the side."""
        pts = [(float(x), float(y)) for x, y in pts]
        n = len(pts)
        widths = [widths] * n if np.isscalar(widths) else list(widths)
        heats = [heats] * n if np.isscalar(heats) else list(heats)
        if n == 1:
            self.ellipse(mat, pts[0][0], pts[0][1], widths[0], heat=heats[0], mode=mode)
            return
        for k in range(n - 1):
            (x0, y0), (x1, y1) = pts[k], pts[k + 1]
            w0, w1, h0, h1 = widths[k], widths[k + 1], heats[k], heats[k + 1]
            if max(w0, w1) <= 0.05:
                continue
            m = max(w0, w1) + 1
            dx, dy = x1 - x0, y1 - y0
            L2 = dx * dx + dy * dy + 1e-9

            def fn(X, Y, x0=x0, y0=y0, dx=dx, dy=dy, L2=L2, w0=w0, w1=w1, h0=h0, h1=h1):
                t = np.clip(((X - x0) * dx + (Y - y0) * dy) / L2, 0, 1)
                d = np.hypot(X - (x0 + t * dx), Y - (y0 + t * dy))
                w = w0 + (w1 - w0) * t
                r = d / np.maximum(w, 1e-3)
                return (r <= 1) & (w > 0.05), (h0 + (h1 - h0) * t) - edge * r * r
            self.field(mat, (min(x0, x1) - m, min(y0, y1) - m, max(x0, x1) + m, max(y0, y1) + m),
                       fn, mode)

    def poly(self, mat, pts, heat, mode="max"):
        """Filled polygon (crystal facets, debris): flat heat."""
        pts = np.asarray(pts, np.float32)
        x0, y0 = pts.min(0) - 1
        x1, y1 = pts.max(0) + 1
        wn = self.win(x0, y0, x1, y1)
        if wn is None:
            return
        sl, X, Y = wn
        h, w = Y.shape[0], X.shape[1]
        off = np.array([sl[1].start, sl[0].start], np.float32)
        p = np.rint((pts * self.ss - off - 0.5) * 16).astype(np.int32)
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [p], 1, lineType=cv2.LINE_8, shift=4)
        self.put(mat, sl, m.astype(bool), heat, mode)

    def star(self, mat, cx, cy, size, heat=1.0, rot=0.0, thin=0.42, stretch=1.0, mode="max",
             snap=True):
        """4-point sparkle (an astroid): arms `size` px (x stretch), centre hottest."""
        if size <= 0.2:
            return
        if snap:
            cx, cy = math.floor(cx) + 0.5, math.floor(cy) + 0.5
        c, s = math.cos(rot), math.sin(rot)
        R = size * max(1.0, stretch) + 1

        def fn(X, Y):
            dx, dy = X - cx, Y - cy
            u, v = np.abs(dx * c + dy * s) / (size * stretch), np.abs(-dx * s + dy * c) / size
            f = u ** thin + v ** thin
            return f <= 1, heat * (1.0 - 0.45 * f)
        self.field(mat, (cx - R, cy - R, cx + R, cy + R), fn, mode)

    def ring(self, mat, cx, cy, r, width, heat=1.0, squash=1.0, keep=None, mode="max",
             inner_heat=None):
        """Ellipse outline r (x) / r*squash (y), `width` px thick; keep(angle) -> bool mask
        for arcs or broken rings; inner_heat = heat on the inner side (a gradient across)."""
        if r <= 0.2 or width <= 0.05:
            return
        R = r + width + 1

        def fn(X, Y):
            dx, dy = X - cx, (Y - cy) / max(squash, 1e-3)
            d = np.hypot(dx, dy) + 1e-6
            grad = np.sqrt((dx / d) ** 2 + (dy / (d * max(squash, 1e-3))) ** 2)
            off = (d - r) / grad                       # distance from the outline in real px
            inside = np.abs(off) <= width / 2
            if keep is not None:
                inside = inside & keep(np.arctan2(dy, dx))
            ht = heat
            if inner_heat is not None:
                k = np.clip(off / max(width, 1e-3) + 0.5, 0, 1)
                ht = inner_heat + (heat - inner_heat) * k
            return inside, ht
        self.field(mat, (cx - R, cy - R * squash - 1, cx + R, cy + R * squash + 1), fn, mode)

    # -- the finish
    def resolve(self):
        """Downsample: {mat: (coverage, mean heat)} at output size."""
        out = {}
        ss = self.ss
        for m in MATS:
            H = self.heat[m].reshape(self.h, ss, self.w, ss)
            on = H >= 0
            cnt = on.sum((1, 3))
            cov = cnt / (ss * ss)
            mean = np.where(on, H, 0).sum((1, 3)) / np.maximum(cnt, 1)
            out[m] = (cov.astype(np.float32), mean.astype(np.float32))
        return out


# --------------------------------------------------------------------------- finishing

_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _shift(a, dy, dx, fill):
    from .pixelclean import _shift as sh
    return sh(a, dy, dx, fill)


def finish(cv: Canvas, ramps: dict, outline: str = "none", cover: float = 0.5,
           light=(-0.6, -0.8)):
    """Canvas -> (rgba, glow rgba, used colours). outline: none | dark (shape edge one ramp
    step darker, so it reads on light ground) | rim (edge facing the light one step
    brighter)."""
    res = cv.resolve()
    h, w = cv.h, cv.w
    mat = np.full((h, w), -1, np.int8)
    lvl = np.zeros((h, w), np.int16)
    heat = np.zeros((h, w), np.float32)
    for k, m in enumerate(MATS):          # later (front) materials win
        cov, ht = res[m]
        n = len(ramps[MAT_RAMP[m]])
        on = cov >= cover
        mat = np.where(on, k, mat)
        lv = np.clip(np.floor(ht * n), 0, n - 1).astype(np.int16)
        lvl = np.where(on, lv, lvl)
        heat = np.where(on, ht, heat)
    vis = mat >= 0
    # dim specks: lone pixels (8-conn pieces of 1-2 px) that aren't bright
    num, labels, stats, _ = cv2.connectedComponentsWithStats(vis.astype(np.uint8), connectivity=8)
    if num > 1:
        small = np.nonzero(stats[1:, cv2.CC_STAT_AREA] <= 2)[0] + 1
        if len(small):
            peak = np.zeros(num, np.float32)
            np.maximum.at(peak, labels[vis], heat[vis])
            drop = small[peak[small] < 0.55]
            gone = np.isin(labels, drop)
            vis &= ~gone
            mat[gone] = -1
    # pinholes: a see-through pixel with all four neighbours set takes their look
    nb = [_shift(mat, dy, dx, -1) for dy, dx in _N4]
    hole = ~vis & np.all([b >= 0 for b in nb], 0)
    if hole.any():
        lv4 = [_shift(lvl, dy, dx, 0) for dy, dx in _N4]
        for y, x in zip(*np.nonzero(hole)):
            mat[y, x] = nb[0][y, x]
            lvl[y, x] = int(np.median([l[y, x] for l in lv4]))
        vis = mat >= 0
    # one-pixel colour islands one step off their neighbours -> the neighbours' colour
    for _ in range(2):
        same = np.zeros((h, w), np.int8)
        votes = []
        for dy, dx in _N4:
            m2, l2 = _shift(mat, dy, dx, -1), _shift(lvl, dy, dx, -999)
            same += ((m2 == mat) & (l2 == lvl)).astype(np.int8)
            votes.append(np.where(m2 == mat, l2, -999))
        V = np.stack(votes, -1)
        cnt_in = (V > -999).sum(-1)
        Vs = np.where(V > -999, V, 0)
        mean = Vs.sum(-1) / np.maximum(cnt_in, 1)
        near = np.rint(mean).astype(np.int16)
        lone = vis & (same == 0) & (cnt_in >= 3) & (np.abs(near - lvl) == 1)
        if not lone.any():
            break
        lvl = np.where(lone, near, lvl)
    # outline / rim on the shape's outer edge
    edge = vis & np.any([_shift(mat, dy, dx, -1) < 0 for dy, dx in _N4], 0)
    if outline == "dark":
        lvl = np.where(edge & (mat != MATS.index("smoke")), np.maximum(lvl - 2, 0), lvl)
    elif outline == "rim":
        lx, ly = light
        ox = (_shift(mat, 0, 1, -1) < 0).astype(np.int8) - (_shift(mat, 0, -1, -1) < 0)
        oy = (_shift(mat, 1, 0, -1) < 0).astype(np.int8) - (_shift(mat, -1, 0, -1) < 0)
        facing = edge & ((ox * lx + oy * ly) > 0.3)
        lvl = np.where(facing, lvl + 1, lvl)
    rgba = np.zeros((h, w, 4), np.uint8)
    glow = np.zeros((h, w, 4), np.uint8)
    used = []
    for k, m in enumerate(MATS):
        sel = vis & (mat == k)
        if not sel.any():
            continue
        ramp = ramps[MAT_RAMP[m]]
        li = np.clip(lvl[sel], 0, len(ramp) - 1)
        cols = ramp[li]
        rgba[sel, :3] = cols
        rgba[sel, 3] = 255
        used.append(np.unique(cols, axis=0))
        e = EMIT[m] * smooth(0.45, 0.95, (li + 0.5) / len(ramp))
        if EMIT[m] > 0:
            g = np.rint(cols.astype(np.float32) * e[:, None]).astype(np.uint8)
            on = e > 0.02
            idx = np.nonzero(sel)
            glow[idx[0][on], idx[1][on], :3] = g[on]
            glow[idx[0][on], idx[1][on], 3] = 255
    cols = np.unique(np.concatenate(used), axis=0) if used else np.zeros((0, 3), np.uint8)
    return rgba, glow, cols


def composite(rgba: np.ndarray, ground=(24, 22, 36), glow: np.ndarray | None = None,
              strength: float = 0.8) -> np.ndarray:
    """A frame on solid ground (opaque RGBA); glow = its emissive map, blurred and added like
    an engine's bloom (previews only: the sheets stay crisp)."""
    a = rgba[..., 3:4].astype(np.float32) / 255
    out = rgba[..., :3].astype(np.float32) * a + np.array(ground, np.float32) * (1 - a)
    if glow is not None:
        g = glow[..., :3].astype(np.float32) * (glow[..., 3:4] > 0)
        r = max(1.0, max(rgba.shape[:2]) / 40)
        bloom = cv2.GaussianBlur(g, (0, 0), r) * 0.9 + cv2.GaussianBlur(g, (0, 0), r * 3) * 0.8
        out = out + bloom * strength
    out = np.clip(out, 0, 255).astype(np.uint8)
    return np.dstack([out, np.full(out.shape[:2], 255, np.uint8)])


# --------------------------------------------------------------------------- timelines

@dataclass
class Beat:
    """One frame of a timeline."""
    i: int            # frame index in the whole sheet
    phase: str
    p: float          # progress through the phase, 0 (first frame) .. 1 (last)
    t: float          # seconds since the start of this animation, at the frame's middle
    ms: int
    k: int = 0        # frame index within its phase
    count: int = 1    # frames in the phase
    tag: str = ""


def timeline(phases, n: int | None = None, tag: str = "", start: int = 0) -> list[Beat]:
    """phases [(name, frames, ms_first, ms_last)] -> Beats; n rescales the frame counts
    (each phase keeps at least one frame) and the durations so the total time stays."""
    counts = [p[1] for p in phases]
    total0 = sum(counts)
    total_ms = sum(sum(np.linspace(p[2], p[3], p[1])) for p in phases)
    if n and n != total0:
        raw = [c * n / total0 for c in counts]
        counts = [max(1, int(round(r))) for r in raw]
        while sum(counts) > n:
            j = int(np.argmax([c - r if c > 1 else -9 for c, r in zip(counts, raw)]))
            counts[j] -= 1
        while sum(counts) < n:
            j = int(np.argmax([r - c for c, r in zip(counts, raw)]))
            counts[j] += 1
    beats, t, i = [], 0.0, start
    for (name, _, m0, m1), c in zip(phases, counts):
        mss = np.linspace(m0, m1, c) if c > 1 else [0.5 * (m0 + m1)]
        for k, ms in enumerate(mss):
            beats.append(Beat(i, name, k / max(1, c - 1) if c > 1 else 1.0, 0.0,
                              max(20, int(round(ms / 10.0)) * 10), k, c, tag))
            i += 1
    if n and n != total0:                 # keep the total time
        scale = total_ms / max(1, sum(b.ms for b in beats))
        for b in beats:
            b.ms = max(20, int(round(b.ms * scale / 10.0)) * 10)
    for b in beats:
        b.t = t + b.ms / 2000
        t += b.ms / 1000
    return beats


def loop_beats(n: int, ms: int, tag: str = "", start: int = 0) -> list[Beat]:
    return [Beat(start + k, "loop", k / n, k / n, ms, k, n, tag) for k in range(n)]


def phase_t(beats, name):
    """Start time (s) of a phase."""
    for b in beats:
        if b.phase == name:
            return b.t - b.ms / 2000
    return 0.0


# --------------------------------------------------------------------------- particles

class Particle:
    """Ballistic point with drag: x(t) = x0 + v (1 - e^-kt)/k + g t^2/2."""

    def __init__(self, x, y, vx, vy, t0=0.0, life=1.0, drag=2.0, gx=0.0, gy=0.0, size=1.0,
                 heat=1.0, **extra):
        self.x, self.y, self.vx, self.vy = x, y, vx, vy
        self.t0, self.life, self.drag, self.gx, self.gy = t0, life, drag, gx, gy
        self.size, self.heat = size, heat
        self.__dict__.update(extra)

    def age(self, t):
        return (t - self.t0) / self.life

    def pos(self, t):
        a = max(0.0, t - self.t0)
        k = self.drag
        f = (1 - math.exp(-k * a)) / k if k > 1e-6 else a
        return self.x + self.vx * f + 0.5 * self.gx * a * a, self.y + self.vy * f + 0.5 * self.gy * a * a


def draw_spark(cv, p: Particle, t, dt, mat="accent", streak=1.0, width=None, heat=None,
               min_len=0.0):
    """A particle as a short tapered streak from where it was dt ago (motion smear) to now."""
    a = p.age(t)
    if not 0 <= a <= 1:
        return
    x1, y1 = p.pos(t)
    x0, y0 = p.pos(max(p.t0, t - dt * streak))
    L = math.hypot(x1 - x0, y1 - y0)
    hw = (p.size if width is None else width) * (1 - 0.5 * a)
    ht = (p.heat if heat is None else heat) * (1 - 0.55 * a ** 1.5)
    if L < max(0.6, min_len) or hw < 0.75:
        if L < min_len and L > 1e-3:
            k = min_len / L
            x0, y0 = x1 - (x1 - x0) * k, y1 - (y1 - y0) * k
        elif hw < 0.9:                           # a 1 px dot, on a pixel centre
            cv.ellipse(mat, x1, y1, 0.55, heat=ht, snap=True)
            return
        else:
            cv.ellipse(mat, x1, y1, hw, heat=ht)
            return
    cv.stroke(mat, [(x0, y0), (x1, y1)], [hw * 0.35, hw], [ht * 0.8, ht])


# --------------------------------------------------------------------------- shapes

def flame_field(X, Y, bx, by, height, width, t, noise, lean=0.0, sway=0.0, turb=0.35,
                speed=1.2, scale=None):
    """A flame standing on (bx, by): density (>0 inside) and height share v (0 base..1 tip).
    The silhouette is domain-warped by noise scrolling upward (licks that rise and break)."""
    s = scale or max(width, 4) * 2.2
    n1 = noise(X / (s * 4), (Y + t * speed * s * 4) / (s * 4))
    n2 = noise(X / (s * 4) + 0.5, (Y + t * speed * s * 4) / (s * 4) + 0.37)
    v = (by - Y) / max(height, 1e-3)
    wx = (n1 - 0.5) * turb * width * 2.2 * (0.2 + v)
    u = (X + wx - bx - lean * height * np.clip(v, 0, 1) ** 1.5 - sway * np.sin(v * 3.2 + t * 7)
         * width * 0.4) / max(width, 1e-3)
    vv = v + (n2 - 0.5) * turb * 0.9 * np.clip(v, 0, 1)
    hw = np.clip(1 - vv, 0, 1) ** 0.75 * np.clip(0.35 + vv * 2.8, 0, 1)
    d = 1 - np.abs(u) / np.maximum(hw, 1e-3)
    d = np.where((vv > -0.12) & (vv < 1), d, -1)
    base = 1 - np.hypot(u * 1.0, np.minimum(vv, 0) * height / max(width, 1e-3))   # round bottom
    d = np.where(vv < 0, base * 0.9, d)
    return d, np.clip(vv, 0, 1)


def puff(cv, x, y, r, heat, noise, t, erosion=0.0, mat="smoke", lit=(0.35, 0.6), mode="max",
         squash=1.0, lobes=3, seed=0):
    """A smoke billow: a few round lobes, ragged by noise, lit from `lit` (dx, dy towards the
    light), eaten into wisps as erosion rises (0..1)."""
    if r < 0.6:
        return
    R = r * 1.6 + 1
    lx, ly = lit
    g = np.random.default_rng(seed)
    offs = [(0.0, 0.0, 1.0)] + [(math.cos(a) * 0.55, math.sin(a) * 0.45, g.uniform(0.55, 0.75))
                                for a in g.uniform(0, 2 * math.pi, max(0, lobes - 1))]

    def fn(X, Y):
        rho = None
        for ox, oy, k in offs:
            dx, dy = (X - x) / r - ox, (Y - y) / (r * squash) - oy
            q = np.sqrt(dx * dx + dy * dy) / k
            rho = q if rho is None else np.minimum(rho, q)
        dx, dy = (X - x) / r, (Y - y) / (r * squash)
        n = noise(X / (r * 5) + 0.13 * t, Y / (r * 5) - 0.2 * t)
        d = 1 - rho + (n - 0.5) * 0.5 - erosion * (0.25 + 0.9 * n)
        shade = heat + 0.3 * np.clip((dx * lx + dy * ly) / 1.4, -1, 1) + 0.1 * (n - 0.5)
        return d > 0, np.where(d < 0.12, shade - 0.1, shade)
    cv.field(mat, (x - R, y - R, x + R, y + R), fn, mode)


def bolt_points(rng, x0, y0, x1, y1, depth=5, spread=0.25, jag=0.55):
    """Midpoint-displaced lightning path."""
    pts = [(x0, y0), (x1, y1)]
    disp = math.hypot(x1 - x0, y1 - y0) * spread
    for _ in range(depth):
        out = [pts[0]]
        for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
            mx, my = (ax + bx) / 2, (ay + by) / 2
            L = math.hypot(bx - ax, by - ay) or 1
            nx, ny = -(by - ay) / L, (bx - ax) / L
            o = rng.uniform(-disp, disp)
            out += [(mx + nx * o, my + ny * o), (bx, by)]
        pts = out
        disp *= jag
    return pts


def draw_bolt(cv, pts, width, heat=1.0, part=1.0, glow=True, taper=0.5):
    """A lightning bolt in three nested strokes: wide dim glow, body, white-hot core."""
    n = len(pts)
    m = max(2, int(math.ceil(n * part)))
    pts = pts[:m]
    s = np.linspace(0, 1, len(pts))
    w = width * (1 - taper * s)
    if glow:
        cv.stroke("main", pts, w * 2.1, heat * 0.32)
    cv.stroke("main", pts, w * 1.5, heat * 0.62)
    cv.stroke("main", pts, np.maximum(w * 0.62, 0.5), heat * 0.98 + 0.02)


# --------------------------------------------------------------------------- effect context

class Ctx:
    def __init__(self, w, h, rng, opts):
        self.w, self.h, self.rng, self.opts = w, h, rng, opts
        self.S = min(w, h)

    def canvas(self):
        return Canvas(self.w, self.h)

    def noise(self, cells=4, octaves=3):
        return Noise(self.rng, cells, octaves)

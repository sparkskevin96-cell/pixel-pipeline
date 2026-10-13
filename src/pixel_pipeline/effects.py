"""Animated pixel-art effects, made procedurally on the CPU (no AI): slashes, hit sparks,
explosions, smoke, fire, magic circles, sparkles, heals, lightning, arrow trails, dust, splashes
and level-up glows, at the project's scale and in its palette.

Every effect is drawn as brightness fields (0..1) per frame, then banded onto a short colour
ramp (dark -> bright) with hard edges and an optional ordered dither at the band edges, so
the result is limited-palette pixel art rather than a blurred glow. Colours come from a theme
(fire = oranges, ice = blues, holy = golds, poison = greens, dark = purples, ...) or your own.

    make_effect("explosion")                      -> Asset in <project>/effects/explosion/
        sheet.png + sheet.json (Aseprite json-array, one tag) + sheet.aseprite +
        sheet.tres (Godot SpriteFrames) + preview.gif + thumb.png
    render_effect("fire", theme="poison")         -> Effect (frames in memory, for previews)

Same parameters + seed -> the same pixels. Looping effects (fire, magic_circle, sparkle,
arrow_trail, and heal / level_up with loop=True) are periodic: frame n would equal frame 0.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .color import lab_to_rgb, rgb_to_lab
from .palettes import hex_to_rgb, rgb_to_hex

# --------------------------------------------------------------------------- kinds + themes

# kind: (width, height) for a 64 px person, frames, frame ms, loops, default theme, label
SPECS = {
    "slash":        ((48, 48), 6, 50, False, "steel", "Slash (arc swipe)"),
    "hit":          ((32, 32), 6, 50, False, "holy", "Hit spark"),
    "explosion":    ((64, 64), 10, 70, False, "fire", "Explosion"),
    "smoke":        ((32, 32), 8, 80, False, "smoke", "Smoke puff"),
    "fire":         ((24, 32), 8, 90, True, "fire", "Fire (loop)"),
    "magic_circle": ((64, 48), 8, 100, True, "arcane", "Magic circle"),
    "sparkle":      ((32, 32), 8, 90, True, "arcane", "Spell sparkle"),
    "heal":         ((32, 48), 10, 80, False, "nature", "Heal"),
    "lightning":    ((24, 64), 6, 60, False, "electric", "Lightning bolt"),
    "arrow_trail":  ((40, 10), 6, 60, True, "steel", "Arrow trail"),
    "dust":         ((32, 16), 6, 70, False, "earth", "Dust (footstep)"),
    "splash":       ((32, 32), 8, 70, False, "water", "Splash (water)"),
    "level_up":     ((48, 64), 10, 80, False, "holy", "Level-up glow"),
}
EFFECT_KINDS = tuple(SPECS)

# dark -> bright
THEMES = {
    "fire":     ["4a1410", "8c2414", "d4461c", "f78c2a", "fcd04a", "fff8d0"],
    "ice":      ["1c2a5c", "2c5a9c", "3c96d8", "7cd0f0", "c4f0ff", "ffffff"],
    "holy":     ["5c3c14", "a06c1c", "dca42c", "f8d860", "fff0a8", "fffff0"],
    "poison":   ["1c3418", "2c6424", "4c9c30", "84d03c", "c4f070", "f0ffc8"],
    "dark":     ["1c0c2c", "3c1858", "6c2c94", "a050cc", "d494f0", "f8e0ff"],
    "arcane":   ["1c1c5c", "2c3ca0", "3c78e0", "5cc0f8", "b0f0ff", "ffffff"],
    "nature":   ["143c28", "1c6c3c", "30a450", "6cd870", "b8f8a0", "f4ffe8"],
    "electric": ["2c2470", "4c50c0", "7c9cf8", "b8dcff", "ecf8ff", "ffffff"],
    "steel":    ["2c3450", "4c608c", "8098c4", "b8d0ec", "e4f2ff", "ffffff"],
    "water":    ["14306c", "1c5ca8", "2c94dc", "64c8f0", "b4ecff", "ffffff"],
    "earth":    ["3c2c24", "6c5040", "9c7c5c", "c4a884", "e8d8b8"],
    "smoke":    ["34343f", "5c5c6a", "8a8a96", "b8b8c2", "e2e2e8"],
}
SOOT = ["241e24", "3e3438", "5e5254", "847a78", "a89c98"]   # explosion smoke


def _parse_colours(colours) -> np.ndarray:
    if isinstance(colours, str):
        colours = [c for c in colours.replace(",", " ").split() if c]
    out = []
    for c in colours:
        if isinstance(c, str):
            out.append(hex_to_rgb([c.strip().lstrip("#")])[0])
        else:
            out.append(np.array(c[:3], np.uint8))
    if not out:
        raise ValueError("No colours given")
    return np.array(out, np.uint8)


def ramp_from_colour(rgb, n: int = 6) -> np.ndarray:
    """One colour -> a dark-to-bright ramp of n with the same hue (highlights wash to white)."""
    lab = rgb_to_lab(np.asarray(rgb, np.float32).reshape(1, 3))[0]
    ls = np.linspace(20, 97, n)
    chroma = np.interp(np.linspace(0, 1, n), [0, 0.2, 0.55, 0.8, 1.0], [0.75, 1.0, 1.0, 0.55, 0.12])
    labs = np.stack([ls, lab[1] * chroma, lab[2] * chroma], -1).astype(np.float32)
    return lab_to_rgb(labs).reshape(-1, 3)


def theme_ramp(theme: str | None = None, colours=None) -> np.ndarray:
    """The colour ramp (N, 3) dark -> bright: your colours (1 = a ramp of its hue, 2+ = sorted
    by lightness), else the theme's."""
    if colours is not None and len(colours):
        rgb = _parse_colours(colours)
        if len(rgb) == 1:
            return ramp_from_colour(rgb[0])
        order = np.argsort(rgb_to_lab(rgb.astype(np.float32))[:, 0], kind="stable")
        return rgb[order]
    theme = theme or "fire"
    if theme not in THEMES:
        raise ValueError(f"Unknown theme '{theme}' (use {', '.join(THEMES)})")
    return hex_to_rgb(THEMES[theme])


def snap_ramp(ramp: np.ndarray, palette: np.ndarray | None) -> np.ndarray:
    """Each ramp colour -> the nearest palette colour (Lab), repeats dropped, order kept."""
    if palette is None or not len(palette):
        return ramp
    a = rgb_to_lab(ramp.astype(np.float32))
    b = rgb_to_lab(palette.astype(np.float32))
    idx = np.argmin(((a[:, None] - b[None]) ** 2).sum(-1), 1)
    seen, out = set(), []
    for i in idx:
        if int(i) not in seen:
            seen.add(int(i))
            out.append(palette[i])
    return np.array(out, np.uint8)


def project_palette(style) -> np.ndarray | None:
    """The project's fixed palette, or None when it's auto (or can't be loaded)."""
    spec = getattr(style, "palette", "auto") or "auto"
    if spec == "auto":
        return None
    try:
        from .palettes import load_palette
        return load_palette(spec)
    except Exception:  # noqa: BLE001 - an unreachable lospec palette shouldn't stop effects
        return None


# --------------------------------------------------------------------------- drawing helpers

BAYER4 = (np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]],
                   np.float32) + 0.5) / 16 - 0.5


def _clamp(v, lo=0.0, hi=1.0):
    return np.clip(v, lo, hi) if isinstance(v, np.ndarray) else max(lo, min(hi, v))


def _ease_out(v):
    v = _clamp(v)
    return 1 - (1 - v) ** 2


def _ease_in(v):
    return _clamp(v) ** 2


class _F:
    """Pixel-centre coordinates of a w x h frame."""

    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        self.x, self.y = xx + 0.5, yy + 0.5

    def zeros(self):
        return np.zeros((self.h, self.w), np.float32)


class _Noise:
    """Smooth value noise, periodic over [0, 1) in both u and v (so loops wrap exactly)."""

    def __init__(self, rng, cells: int = 4, octaves: int = 3):
        self.grids = [rng.random((cells * 2 ** k, cells * 2 ** k)).astype(np.float32)
                      for k in range(octaves)]

    @staticmethod
    def _sample(g, u, v):
        n = g.shape[0]
        x, y = (u % 1.0) * n, (v % 1.0) * n
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        fx, fy = x - x0, y - y0
        fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
        x0, y0 = x0 % n, y0 % n
        x1, y1 = (x0 + 1) % n, (y0 + 1) % n
        top = g[y0, x0] * (1 - fx) + g[y0, x1] * fx
        bot = g[y1, x0] * (1 - fx) + g[y1, x1] * fx
        return top * (1 - fy) + bot * fy

    def __call__(self, u, v):
        total, amp, norm = 0.0, 1.0, 0.0
        for g in self.grids:
            total = total + amp * self._sample(g, u, v)
            norm += amp
            amp *= 0.5
        return total / norm


def _seg_dist(F, x0, y0, x1, y1):
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy
    if L2 < 1e-9:
        return np.hypot(F.x - x0, F.y - y0)
    k = np.clip(((F.x - x0) * dx + (F.y - y0) * dy) / L2, 0, 1)
    return np.hypot(F.x - (x0 + k * dx), F.y - (y0 + k * dy))


def _poly_dist(F, pts):
    d = np.full((F.h, F.w), 1e9, np.float32)
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        d = np.minimum(d, _seg_dist(F, x0, y0, x1, y1))
    return d


def _star(I, F, cx, cy, size, value):
    """4-point twinkle: a plus with arms `size` px long fading out, bright centre."""
    if value <= 0:
        return I
    size = max(0.0, size)
    dx, dy = np.abs(F.x - (math.floor(cx) + 0.5)), np.abs(F.y - (math.floor(cy) + 0.5))
    arm = (np.minimum(dx, dy) < 0.5) & (np.maximum(dx, dy) <= size + 0.01)
    v = value * (1 - 0.55 * np.maximum(dx, dy) / max(size, 1))
    out = np.maximum(I, np.where(arm, v, 0))
    if size >= 2.5:   # a little diagonal halo round the centre of big ones
        diag = (dx > 0.5) & (dx < 1.5) & (dy > 0.5) & (dy < 1.5)
        out = np.maximum(out, np.where(diag, value * 0.45, 0))
    return out


def _dglow(F, coverage, value):
    """Ordered-dither transparency: `value` where coverage (0..1) beats the Bayer threshold, so a
    soft glow fades out as a pixel pattern instead of as dark colours."""
    thr = BAYER4[np.arange(F.h)[:, None] % 4, np.arange(F.w)[None] % 4] + 0.5
    return np.where(np.asarray(coverage) > thr, value, 0).astype(np.float32)


def _dot(I, F, cx, cy, value, r=0.5):
    """One pixel (r <= 0.5) or a small disc."""
    if r <= 0.5:
        hit = (F.x - 0.5 == math.floor(cx)) & (F.y - 0.5 == math.floor(cy))
    else:
        hit = np.hypot(F.x - cx, F.y - cy) <= r
    return np.maximum(I, np.where(hit, value, 0))


def _ring(F, cx, cy, rx, ry, width=0.6):
    """Distance (px, approx.) from the ellipse outline, for crisp 1 px rings."""
    X, Y = (F.x - cx) / max(rx, 1e-3), (F.y - cy) / max(ry, 1e-3)
    rho = np.sqrt(X * X + Y * Y) + 1e-6
    grad = np.sqrt((X / (rho * max(rx, 1e-3))) ** 2 + (Y / (rho * max(ry, 1e-3))) ** 2) + 1e-6
    return np.abs(rho - 1) / grad, rho


def _puffs(F, blobs):
    """Lit smoke balls: blobs [(cx, cy, r, depth)] -> (brightness, coverage 0..1).
    Light from the top left; the front-most ball (largest z) wins per pixel."""
    I, cov = F.zeros(), F.zeros()
    best = np.full((F.h, F.w), -1e9, np.float32)
    for bx, by, br, depth in blobs:
        if br <= 0.3:
            continue
        dx, dy = (F.x - bx) / br, (F.y - by) / br
        d2 = dx * dx + dy * dy
        inside = d2 < 1
        nz = np.sqrt(np.clip(1 - d2, 0, 1))
        z = nz * br + depth
        lit = np.clip(-0.5 * dx - 0.62 * dy + 0.6 * nz, 0, 1)
        sel = inside & (z > best)
        I = np.where(sel, 0.1 + 0.72 * lit, I)
        best = np.where(sel, z, best)
        cov = np.maximum(cov, np.clip(1 - np.sqrt(d2), 0, 1))
    return I, cov


def _bolt(rng, x0, y0, x1, y1, depth, spread):
    pts = [(x0, y0), (x1, y1)]
    disp = spread
    for _ in range(depth):
        out = [pts[0]]
        for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
            mx, my = (ax + bx) / 2, (ay + by) / 2
            L = math.hypot(bx - ax, by - ay) or 1
            nx, ny = -(by - ay) / L, (bx - ax) / L
            o = rng.uniform(-disp, disp)
            out += [(mx + nx * o, my + ny * o), (bx, by)]
        pts = out
        disp *= 0.55
    return pts


# --------------------------------------------------------------------------- the effects
# Each returns a list (one per frame) of layers [(brightness field, ramp name)], drawn in order.

def _times(n, loop):
    return [i / n for i in range(n)] if loop else [(i + 0.5) / n for i in range(n)]


def _slash(w, h, n, loop, rng):
    F = _F(w, h)
    cx, cy = w * 0.5, h * 0.6
    R = min(w, h) * 0.45
    T = max(2.5, R * 0.42)
    r = np.hypot(F.x - cx, F.y - cy)
    a0, span = -math.pi * 0.97, math.pi * 1.08
    rel = (np.arctan2(F.y - cy, F.x - cx) - a0) % (2 * math.pi)
    out = []
    for t in _times(n, loop):
        head = span * _ease_out(t / 0.45)
        tail = span * _ease_in((t - 0.08) / 0.92)
        seg = max(head - tail, 1e-3)
        u = np.clip((rel - tail) / seg, 0, 1)
        inside = (rel >= tail) & (rel <= head)
        shrink = 1 - 0.55 * _clamp((t - 0.45) / 0.55)
        thick = T * shrink * np.clip(np.sin(math.pi * u), 0, 1) ** 0.55 * (0.3 + 0.7 * u)
        d = R - r
        band = inside & (d >= -0.4) & (d <= thick)
        fade = 1 - 0.45 * _clamp((t - 0.5) / 0.5)
        I = np.where(band, (1 - np.clip(d, 0, None) / np.maximum(thick, 0.5) * 0.8)
                     * (0.5 + 0.5 * u) * fade + 0.08, 0)
        out.append([(I, "main")])
    return out


def _hit(w, h, n, loop, rng):
    F = _F(w, h)
    S = min(w, h)
    cx, cy = w / 2, h / 2
    dx, dy = F.x - cx, F.y - cy
    r, th = np.hypot(dx, dy), np.arctan2(dy, dx)
    sparks = [(rng.uniform(0, 2 * math.pi), rng.uniform(0.55, 1.0)) for _ in range(7)]
    out = []
    for k, t in enumerate(_times(n, loop)):
        A = S * 0.5 * math.sin(math.pi * _clamp(t / 0.6)) ** 0.6
        rot = 0.0 if k % 2 == 0 else math.pi / 4 * 0.25
        a = th + rot
        L = A * (0.18 + 0.82 * np.abs(np.cos(2 * a)) ** 12 + 0.4 * np.abs(np.sin(2 * a)) ** 12)
        I = np.clip(1.15 - r / np.maximum(L, 0.5), 0, 1) * (1 - 0.35 * t)
        core = S * 0.13 * (1 - t) ** 1.2
        I = np.where(r < core, 1.0, I)
        if 0.12 < t < 0.65:                                   # shock ring
            rr = S * 0.46 * _ease_out((t - 0.12) / 0.53)
            I = np.maximum(I, np.where(np.abs(r - rr) < 0.55, 0.55 * (1 - t), 0))
        if t > 0.15:                                          # flying sparks
            for ang, sp in sparks:
                p = _ease_out((t - 0.15) / 0.85)
                rp = S * 0.5 * sp * (0.35 + 0.65 * p)
                ln = S * 0.14 * (1 - p) + 0.6
                x1, y1 = cx + math.cos(ang) * rp, cy + math.sin(ang) * rp
                x0, y0 = cx + math.cos(ang) * (rp - ln), cy + math.sin(ang) * (rp - ln)
                d = _seg_dist(F, x0, y0, x1, y1)
                I = np.maximum(I, np.where(d < 0.6, 0.95 * (1 - p * 0.7), 0))
        out.append([(I, "main")])
    return out


def _explosion(w, h, n, loop, rng):
    F = _F(w, h)
    S = min(w, h)
    cx, cy = w / 2, h * 0.56
    noise = _Noise(rng, 4, 3)
    blobs = [(rng.uniform(0, 2 * math.pi), rng.uniform(0.05, 0.27) * S,
              rng.uniform(0.17, 0.27) * S, rng.uniform(0.0, 0.14) * S) for _ in range(8)]
    blobs.append((0.0, 0.0, 0.3 * S, 0.05 * S))
    debris = [(rng.uniform(-math.pi * 0.95, -math.pi * 0.05), rng.uniform(0.6, 1.0)) for _ in range(6)]
    out = []
    for t in _times(n, loop):
        grow = _ease_out(t / 0.32)
        shape = F.zeros()
        for a, rd, rb, rise in blobs:
            bx = cx + math.cos(a) * rd * (0.4 + 0.9 * grow)
            by = cy + math.sin(a) * rd * (0.4 + 0.7 * grow) - rise * t * 1.6 - S * 0.1 * t
            br = rb * (0.3 + 0.7 * grow) * (1 + 0.3 * t)
            shape = np.maximum(shape, 1 - np.hypot(F.x - bx, F.y - by) / br)
        nz = noise(F.x / w, F.y / h + t * 0.6)
        sh = shape + (nz - 0.5) * 0.55 * np.clip(shape * 4, 0, 1)
        heat = _clamp(1.15 - 1.45 * t)
        burn = 0.02 + 1.15 * _clamp((t - 0.22) / 0.6) ** 0.8
        fire = np.where(sh > burn, np.clip((sh - burn) / (1 - burn + 1e-3) * 0.7 + heat * 0.55, 0, 1), 0)
        if t < 0.16:
            fire = np.where(shape > 0.45, 1.0, fire)
        smoke_cut = 0.04 + 1.1 * _clamp((t - 0.5) / 0.5) ** 1.2
        smoke = np.where((sh > smoke_cut) & (t > 0.18),
                         np.clip(0.22 + 0.5 * sh - 0.45 * (F.y - cy) / S, 0.1, 0.85), 0)
        if 0.04 < t < 0.45:
            rr = S * 0.49 * _ease_out(t / 0.45)
            dist = np.abs(np.hypot(F.x - cx, (F.y - cy) * 1.15) - rr)
            fire = np.maximum(fire, np.where(dist < 0.55, 0.62 * (1 - t / 0.45) + 0.2, 0))
        for ang, sp in debris:
            if 0.1 < t < 0.85:
                tt = (t - 0.1) / 0.75
                px = cx + math.cos(ang) * S * 0.5 * sp * tt
                py = cy + math.sin(ang) * S * 0.5 * sp * tt + S * 0.45 * tt * tt
                fire = _dot(fire, F, px, py, 0.9 * (1 - tt) + 0.25)
        out.append([(smoke, "soot"), (fire, "main")])
    return out


def _smoke(w, h, n, loop, rng):
    F = _F(w, h)
    S = min(w, h)
    cx, base = w / 2, h * 0.78
    noise = _Noise(rng, 4, 2)
    blobs = [(rng.uniform(-0.22, 0.22) * w, rng.uniform(-0.12, 0.06) * h, rng.uniform(0.2, 0.3) * S,
              rng.uniform(0.25, 0.45) * h) for _ in range(5)]
    out = []
    for t in _times(n, loop):
        g = _ease_out(t / 0.55)
        balls = [(cx + ox * (0.3 + 0.9 * g), base + oy - rise * g - h * 0.05 * t,
                  r * (0.45 + 0.6 * g), -oy * 0.5) for ox, oy, r, rise in blobs]
        I, cov = _puffs(F, balls)
        cut = 0.02 + 1.05 * _clamp((t - 0.38) / 0.62)
        nz = noise(F.x / w, F.y / h - t * 0.5)
        vis = cov + (nz - 0.5) * 0.45 > cut
        out.append([(np.where(vis, I, 0), "main")])
    return out


def _fire(w, h, n, loop, rng):
    F = _F(w, h)
    noise = _Noise(rng, 3, 2)
    cx, base, H = w / 2, h - 1.0, h * 0.97
    v = (base - F.y) / H
    u = (F.x - cx) / (w * 0.47)
    embers = [(rng.uniform(-0.6, 0.6), rng.uniform(0, 1), rng.uniform(0, 1)) for _ in range(4)]
    out = []
    for t in [i / n for i in range(n)]:            # always periodic
        sway = 0.12 * np.clip(v, 0, 1) * np.sin(2 * math.pi * (t + v * 0.7))
        uu = u - sway
        hw = np.clip(1 - v, 0, 1) ** 0.85 * np.clip(0.55 + v * 3.2, 0, 1)
        shape = np.where((v >= 0) & (v <= 1), 1 - np.abs(uu) / np.maximum(hw, 1e-3), 0)
        nz = noise(F.x / w * 1.0, v * 0.8 - t)
        I = shape * (1.15 - 0.45 * v) - (nz - 0.3) * 0.95 * (0.25 + v)
        I = np.where(shape > 0, I, 0)
        for x0, ph, ph2 in embers:
            p = (ph + t) % 1.0
            ex = cx + x0 * w * 0.3 + math.sin(2 * math.pi * (p + ph2)) * 1.2
            ey = base - h * 0.35 - p * h * 0.75
            I = _dot(I, F, ex, ey, 0.75 * (1 - p) + 0.3)
        out.append([(np.clip(I, 0, 1), "main")])
    return out


def _magic_circle(w, h, n, loop, rng):
    F = _F(w, h)
    rx = w * 0.47
    ry = rx * 0.5
    cx, cy = w / 2, h - ry - 1.5
    motes = [(rng.uniform(-0.8, 0.8), rng.uniform(0, 1), rng.uniform(0, 1)) for _ in range(6)]
    out = []
    for t in [i / n for i in range(n)]:            # always periodic
        pulse = 0.88 + 0.12 * math.cos(2 * math.pi * t)
        I = F.zeros()
        for rad, val in ((1.0, 0.95), (0.82, 0.7), (0.36, 0.8)):
            d, _ = _ring(F, cx, cy, rx * rad, ry * rad)
            I = np.maximum(I, np.where(d < 0.62, val * pulse, 0))
        # runes between the outer rings, turning one rune-step per loop
        X, Y = (F.x - cx) / rx, (F.y - cy) / ry
        rho, phi = np.hypot(X, Y), np.arctan2(Y, X)
        k = 12
        step = 2 * math.pi / k
        rel = ((phi - t * step) % step) - step / 2
        tick = (rho > 0.86) & (rho < 0.96) & (np.abs(rel) * rho * rx < 0.75)
        I = np.maximum(I, np.where(tick, 0.85 * pulse, 0))
        dotr = (rho > 0.88) & (rho < 0.94) & (np.abs(np.abs(rel) - step / 2 + 0.0) * rho * rx < 0.6)
        I = np.maximum(I, np.where(dotr, 0.55 * pulse, 0))
        # hexagram turning the other way, 60 degrees per loop
        rot = -t * math.pi / 3
        pts = [(cx + math.cos(rot + j * math.pi / 3) * rx * 0.8,
                cy + math.sin(rot + j * math.pi / 3) * ry * 0.8) for j in range(6)]
        d = F.zeros() + 1e9
        for tri in ((0, 2, 4), (1, 3, 5)):
            a = [pts[j] for j in tri] + [pts[tri[0]]]
            d = np.minimum(d, _poly_dist(F, a))
        I = np.maximum(I, np.where(d < 0.55, 0.62 * pulse, 0))
        # inner glow, dithered
        _, rho2 = _ring(F, cx, cy, rx * 0.36, ry * 0.36)
        I = np.maximum(I, _dglow(F, np.where(rho2 < 1, (1 - rho2) * 0.9 * pulse, 0), 0.45))
        for mx, ph, size in motes:
            p = (ph + t) % 1.0
            px = cx + mx * rx * 0.8
            py = cy - p * (cy - 2)
            I = _star(I, F, px, py, 1 if size < 0.6 else 2, 0.5 + 0.45 * math.sin(math.pi * p))
        out.append([(I, "main")])
    return out


def _sparkle(w, h, n, loop, rng):
    F = _F(w, h)
    S = min(w, h)
    m = max(2, round(S * 0.12))
    twinkles = [(rng.uniform(m, w - m), rng.uniform(m, h - m), rng.uniform(0, 1),
                 rng.uniform(0.6, 1.0)) for _ in range(7)]
    twinkles[0] = (w / 2, h / 2, 0.0, 1.0)
    out = []
    for t in [i / n for i in range(n)]:            # always periodic
        I = F.zeros()
        for x, y, ph, big in twinkles:
            b = max(0.0, math.sin(2 * math.pi * (t + ph))) ** 1.5
            size = b * big * S * 0.22
            if b > 0.05:
                I = _star(I, F, x, y, max(0.0, round(size)), 0.45 + 0.55 * b)
        out.append([(I, "main")])
    return out


def _heal(w, h, n, loop, rng):
    F = _F(w, h)
    cx, base = w / 2, h - 3
    parts = [(rng.uniform(0.18, 0.82) * w, rng.uniform(0, 1), rng.uniform(0, 0.45),
              rng.random() < 0.45) for _ in range(9)]
    out = []
    for t in _times(n, loop):
        env = 0.8 if loop else math.sin(math.pi * _clamp(t / 0.95)) ** 0.7
        v = np.clip((base - F.y) / (h * 0.95), 0, 1)
        col = np.clip(1 - np.abs(F.x - cx) / (w * 0.3), 0, 1) ** 0.7 * (1 - v) ** 1.6
        I = _dglow(F, np.where(F.y <= base + 1, col * env * 1.4, 0), 0.5)
        if not loop and t < 0.6:
            d, _ = _ring(F, cx, base, w * 0.42 * _ease_out(t / 0.4), w * 0.42 * 0.3 * _ease_out(t / 0.4))
            I = np.maximum(I, np.where(d < 0.6, 0.75 * (1 - t / 0.6) + 0.15, 0))
        for x, ph, delay, plus in parts:
            p = (ph + t) % 1.0 if loop else (t * 1.35 - delay)
            if not 0 <= p <= 1:
                continue
            y = base - 2 - p * (h - 6)
            val = 0.55 + 0.45 * math.sin(math.pi * min(1.0, p * 1.6))
            if plus:
                I = _star(I, F, x, y, 2, val)
            else:
                I = _star(I, F, x, y, 1 if p < 0.6 else 0, val * 0.9)
        out.append([(I, "main")])
    return out


def _lightning(w, h, n, loop, rng):
    F = _F(w, h)
    top, bot = 0.0, h - 3.0
    bolts = []
    for _ in range(2):
        x0, x1 = w / 2 + rng.uniform(-0.15, 0.15) * w, w / 2 + rng.uniform(-0.12, 0.12) * w
        main = _bolt(rng, x0, top, x1, bot, 4, w * 0.32)
        j = rng.integers(len(main) // 4, len(main) // 2)
        bx, by = main[j]
        side = 1 if rng.random() < 0.5 else -1
        branch = _bolt(rng, bx, by, bx + side * w * 0.38, by + h * 0.25, 3, w * 0.12)
        bolts.append((_poly_dist(F, main), _poly_dist(F, branch), main[-1]))
    # (bolt, strength, how much of it, impact)
    plan = [(0, 0.55, 0.45, False), (0, 1.0, 1.0, True), (1, 1.0, 1.0, True),
            (0, 0.7, 1.0, True), (1, 0.4, 1.0, False), (None, 0.0, 0.0, False)]
    sparks = [(rng.uniform(-math.pi * 0.95, -math.pi * 0.05), rng.uniform(0.5, 1.0)) for _ in range(5)]
    out = []
    for i, t in enumerate(_times(n, loop)):
        b, s, part, impact = plan[min(len(plan) - 1, int(t * len(plan)))]
        I = F.zeros()
        if b is not None:
            dm, db, (ix, iy) = bolts[b]
            lim = F.y <= part * h
            I = np.where(lim & (dm < 0.75), s, I)
            I = np.where(lim & (dm >= 0.75) & (dm < 1.6), np.maximum(I, s * 0.55), I)
            I = np.where(lim & (dm >= 1.6) & (dm < 2.6), np.maximum(I, s * 0.25), I)
            if part >= 1:
                I = np.maximum(I, np.where(db < 0.6, s * 0.8, np.where(db < 1.5, s * 0.35, 0)))
            if impact:
                d, rho = _ring(F, ix, iy + 1, w * 0.38 * s, w * 0.12 * s)
                I = np.maximum(I, np.where(rho < 1, s * (0.95 - 0.5 * rho), 0))
        if t > 0.3:
            tt = (t - 0.3) / 0.7
            ix, iy = bolts[0][2]
            for ang, sp in sparks:
                px = ix + math.cos(ang) * w * 0.45 * sp * tt
                py = iy + math.sin(ang) * h * 0.2 * sp * tt + h * 0.15 * tt * tt
                I = _dot(I, F, px, py, 0.85 * (1 - tt) + 0.3)
        out.append([(I, "main")])
    return out


def _arrow_trail(w, h, n, loop, rng):
    F = _F(w, h)
    cy = h / 2
    q = F.x / w
    out = []
    for t in _times(n, loop):
        tt = t if loop else 0.0
        env = 1.0 if loop else 1 - 0.75 * t
        hw = 0.45 + h * 0.2 * q ** 1.4
        dy = np.abs(F.y - cy)
        stripe = 0.5 + 0.5 * np.cos(2 * math.pi * (3 * q + tt))
        body = (dy <= hw) & (q + 0.12 * stripe > 0.4)
        I = np.where(body, 0.3 + 0.45 * q * (0.6 + 0.4 * stripe), 0)
        I = np.maximum(I, np.where((dy <= hw) & (q > 0.75), 0.6 + 0.4 * q, 0))
        I = np.maximum(I, np.where((dy < 0.5) & (q > 0.15), 0.45 + 0.55 * q, 0))
        if not loop:   # one-shot: the tail gets eaten from the left
            I = np.where(q > t * 0.9, I, 0)
        for off, ph in ((-0.34, 0.25), (0.34, 0.6)):
            ly = cy + off * h
            dash = np.cos(2 * math.pi * (2 * q + 2 * tt + ph)) > 0.15
            line = (np.abs(F.y - ly) < 0.5) & dash & (q > 0.2) & (q < 0.92)
            I = np.maximum(I, np.where(line, 0.5 * q + 0.1, 0))
        out.append([(I * env, "main")])
    return out


def _dust(w, h, n, loop, rng):
    F = _F(w, h)
    cx, ground = w / 2, h - 1.5
    noise = _Noise(rng, 4, 2)
    puffs = []
    for side in (-1, 1):
        for k in range(2):
            puffs.append((side, rng.uniform(0.18, 0.42), rng.uniform(0.3, 0.42) * h,
                          rng.uniform(0.0, 0.25) * h))
    pebbles = [(rng.choice([-1, 1]) * rng.uniform(0.3, 0.9), rng.uniform(0.5, 1.0)) for _ in range(4)]
    out = []
    for t in _times(n, loop):
        g = _ease_out(t / 0.6)
        balls = [(cx + side * (2 + dist * w * g), ground - r * (0.5 + 0.4 * g) - rise * g,
                  r * (0.5 + 0.55 * g), -dist) for side, dist, r, rise in puffs]
        balls.append((cx, ground - h * 0.18, h * 0.32 * (1 - g * 0.6), 0.5))
        I, cov = _puffs(F, balls)
        cut = 0.02 + 1.05 * _clamp((t - 0.3) / 0.7)
        vis = (cov + (noise(F.x / w, F.y / h) - 0.5) * 0.4 > cut) & (F.y < ground + 1)
        I = np.where(vis, I, 0)
        for vx, vy in pebbles:
            px = cx + vx * w * 0.5 * t
            py = ground - 1 - vy * h * 0.9 * (t - t * t) * 3
            if t < 0.9:
                I = _dot(I, F, px, py, 0.35)
        out.append([(I, "main")])
    return out


def _splash(w, h, n, loop, rng):
    F = _F(w, h)
    cx, ground = w / 2, h - 3.5
    drops = [(rng.uniform(-0.45, 0.45) * w, rng.uniform(0.55, 1.0) * h, rng.integers(0, 2))
             for _ in range(9)]
    out = []
    for t in _times(n, loop):
        I = F.zeros()
        # central column up and back down
        hc = h * 0.4 * math.sin(math.pi * _clamp(t / 0.7)) ** 0.8
        if hc > 1:
            v = (ground - F.y) / hc
            half = 2.2 * (1 - np.clip(v, 0, 1)) ** 0.6 + 0.6
            col = (v >= 0) & (v <= 1) & (np.abs(F.x - cx) <= half)
            I = np.where(col, 0.55 + 0.45 * (1 - np.abs(F.x - cx - 0.6) / (half + 1)) * 0.9, I)
        # crown droplets on arcs
        for vx, vy, big in drops:
            p = t * 1.1
            px = cx + vx * p
            py = ground - (vy * 1.6 * p - h * 1.4 * p * p * 0.8)
            if p < 1.05 and py <= ground:
                I = _dot(I, F, px, py, 0.95, 1.0 if big else 0.5)
                I = _dot(I, F, px, py + 1, 0.55)
        # ripple on the water
        rr = w * (0.18 + 0.32 * _ease_out(t))
        d, rho = _ring(F, cx, ground, rr, rr * 0.28)
        I = np.maximum(I, np.where(d < 0.6, 0.85 * (1 - t) + 0.2, 0))
        if t > 0.35:
            d2, _ = _ring(F, cx, ground, rr * 0.6, rr * 0.6 * 0.28)
            I = np.maximum(I, np.where(d2 < 0.6, 0.5 * (1 - t) + 0.15, 0))
        out.append([(I, "main")])
    return out


def _level_up(w, h, n, loop, rng):
    F = _F(w, h)
    rx = w * 0.42
    ry = rx * 0.32
    cx, base = w / 2, h - ry - 1.5
    parts = [(rng.uniform(-0.8, 0.8) * rx, rng.uniform(0, 1), rng.uniform(0, 0.5), rng.random() < 0.4)
             for _ in range(8)]
    out = []
    for t in _times(n, loop):
        env = 1.0 if loop else math.sin(math.pi * _clamp(t / 0.92)) ** 0.5
        grow = 1.0 if loop else _ease_out(t / 0.3)
        Wp = w * 0.3 * env * (0.4 + 0.6 * grow)
        v = np.clip((base - F.y) / (base - 1), 0, 1)
        bands = 0.5 + 0.5 * np.cos(2 * math.pi * (3 * v - 2 * t))
        pil = np.clip(1 - np.abs(F.x - cx) / max(Wp, 0.5), 0, 1) ** 0.7 * (1 - v ** 1.4)
        I = np.where(F.y <= base, np.where(pil > 0.45, pil * (0.6 + 0.4 * bands), 0), 0)
        I = np.maximum(I, _dglow(F, np.where(F.y <= base, pil * 2.2 * (0.5 + 0.5 * bands), 0), 0.42))
        d, rho = _ring(F, cx, base, rx * grow, ry * grow)
        I = np.maximum(I, np.where(d < 0.62, 0.85 * env + 0.1, 0))
        I = np.maximum(I, _dglow(F, np.where(rho < 1, (1 - rho) * 1.3 * env, 0), 0.5))
        for x, ph, delay, big in parts:
            p = (ph + t) % 1.0 if loop else t * 1.4 - delay
            if 0 <= p <= 1:
                I = _star(I, F, cx + x, base - p * (base - 2), 2 if big else 1,
                          0.95 * math.sin(math.pi * p) ** 0.5)
        for k in range(2):                       # rising chevrons
            p = (t * 2 + k * 0.5) % 1.0 if loop else _clamp(t * 1.2 - k * 0.3)
            if 0 < p < 1:
                yc = base - 4 - p * (base - 8)
                a = max(2.0, w * 0.08)
                d = np.minimum(_seg_dist(F, cx - a, yc + a, cx, yc), _seg_dist(F, cx, yc, cx + a, yc + a))
                I = np.maximum(I, np.where(d < 0.6, 0.95 * math.sin(math.pi * p) ** 0.4, 0))
        out.append([(np.clip(I, 0, 1), "main")])
    return out


_MAKERS = {"slash": _slash, "hit": _hit, "explosion": _explosion, "smoke": _smoke, "fire": _fire,
           "magic_circle": _magic_circle, "sparkle": _sparkle, "heal": _heal,
           "lightning": _lightning, "arrow_trail": _arrow_trail, "dust": _dust,
           "splash": _splash, "level_up": _level_up}


# --------------------------------------------------------------------------- render

def _shade(I, ramp_len, dither: float, cutoff: float = 0.06):
    """Brightness -> ramp index per pixel (-1 = transparent)."""
    h, w = I.shape
    b = BAYER4[np.arange(h)[:, None] % 4, np.arange(w)[None] % 4]
    J = I + b * dither / ramp_len
    vis = (I > 0) & (J > cutoff)
    lvl = np.clip(np.floor((J - cutoff) / (1 - cutoff) * ramp_len), 0, ramp_len - 1).astype(int)
    return np.where(vis, lvl, -1)


@dataclass
class Effect:
    kind: str
    frames: list                     # RGBA uint8 frames, all (h, w, 4)
    palette: np.ndarray              # every colour used, (N, 3)
    frame_ms: int
    loop: bool
    theme: str | None
    ramp: np.ndarray = field(repr=False, default=None)

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.frames[0].shape[:2]
        return w, h


def effect_size(kind: str, size=None, style=None) -> tuple[int, int]:
    """(w, h): size = (w, h), or one number for the longer side; default from the Style
    (scaled from its person height)."""
    (bw, bh) = SPECS[kind][0]
    if isinstance(size, (tuple, list)):
        return max(4, int(size[0])), max(4, int(size[1]))
    if size:
        k = int(size) / max(bw, bh)
        return max(4, round(bw * k)), max(4, round(bh * k))
    ph = getattr(style, "person_height", 64) if style is not None else 64
    return max(4, round(bw * ph / 64)), max(4, round(bh * ph / 64))


def render_effect(kind: str, *, theme: str | None = None, colours=None, size=None,
                  frames: int | None = None, loop: bool | None = None, seed: int = 0,
                  dither: float = 0.5, style=None, palette: np.ndarray | None = None) -> Effect:
    """Draw an effect in memory. palette: snap colours to it (default: the style's fixed
    palette, if it has one). dither: 0 = flat bands, 0.5 = a little Bayer at band edges."""
    if kind not in SPECS:
        raise ValueError(f"Unknown effect '{kind}' (use {', '.join(EFFECT_KINDS)})")
    _, n0, ms, loop0, theme0, _ = SPECS[kind]
    n = int(frames or n0)
    if not 2 <= n <= 32:
        raise ValueError("frames must be between 2 and 32")
    loop = loop0 if loop is None else bool(loop)
    w, h = effect_size(kind, size, style)
    use_theme = None if colours else (theme or theme0)
    pal = palette if palette is not None else project_palette(style)
    ramps = {"main": snap_ramp(theme_ramp(use_theme, colours), pal),
             "soot": snap_ramp(hex_to_rgb(SOOT), pal)}
    rng = np.random.default_rng(seed)
    layers = _MAKERS[kind](w, h, n, loop, rng)
    out = []
    used = []
    for frame in layers:
        rgba = np.zeros((h, w, 4), np.uint8)
        for I, name in frame:
            ramp = ramps[name]
            idx = _shade(np.asarray(I, np.float32), len(ramp), dither)
            m = idx >= 0
            rgba[m, :3] = ramp[idx[m]]
            rgba[m, 3] = 255
            if m.any():
                used.append(name)
        out.append(rgba)
    colours_used = [ramps["main"]] + ([ramps["soot"]] if "soot" in used else [])
    all_cols = np.unique(np.concatenate(colours_used), axis=0)
    return Effect(kind, out, all_cols, ms, loop, use_theme, ramps["main"])


# --------------------------------------------------------------------------- files

def write_effect(effect: Effect, folder, name: str = "sheet", gif_scale: int | None = None) -> list[Path]:
    """sheet.png + .json (Aseprite json-array with one tag) + .aseprite + .tres + preview.gif
    + thumb.png in folder."""
    from .aseprite import write_aseprite
    from .export import pack_sheet, write_sheet
    from .godot import write_sprite_frames
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    png = folder / f"{name}.png"
    n = len(effect.frames)
    js = write_sheet(effect.frames, png, n, effect.frame_ms, effect.kind,
                     [f"{effect.kind} {i}" for i in range(n)])
    meta = json.loads(js.read_text(encoding="utf-8"))
    meta["meta"]["loop"] = effect.loop
    js.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    ase = folder / f"{name}.aseprite"
    write_aseprite(ase, effect.frames, effect.palette, durations=effect.frame_ms,
                   tags=[(effect.kind, 0, n - 1)], layer_name=effect.kind)
    _, rects = pack_sheet(effect.frames, n)
    tres = write_sprite_frames(folder / f"{name}.tres", png.name, rects,
                               animations={effect.kind: {"frames": list(range(n)),
                                                         "duration_ms": effect.frame_ms,
                                                         "loop": effect.loop}})
    gif = write_preview_gif(effect, folder / "preview.gif", gif_scale)
    thumb = write_thumb(effect, folder / "thumb.png")
    return [png, js, ase, tres, gif, thumb]


def write_preview_gif(effect: Effect, out, scale: int | None = None,
                      background=(28, 32, 46)) -> Path:
    w, h = effect.size
    k = scale or max(1, min(8, 192 // max(w, h)))
    images, durations = [], []
    for f in effect.frames:
        im = Image.new("RGBA", (w, h), background + (255,))
        im.alpha_composite(Image.fromarray(f, "RGBA"))
        images.append(im.convert("RGB").resize((w * k, h * k), Image.NEAREST))
        durations.append(effect.frame_ms)
    if not effect.loop:                          # a beat of nothing so one-shots read as one-shots
        images.append(Image.new("RGB", (w * k, h * k), background))
        durations.append(400)
    images[0].save(out, save_all=True, append_images=images[1:], duration=durations, loop=0)
    return Path(out)


def peak_frame(effect: Effect) -> np.ndarray:
    """The frame with the most visible pixels (what a thumbnail should show)."""
    return max(effect.frames, key=lambda f: int((f[..., 3] > 0).sum()))


def write_thumb(effect: Effect, out, px: int = 96) -> Path:
    im = Image.fromarray(peak_frame(effect), "RGBA")
    box = im.getbbox()
    if box:
        im = im.crop(box)
    k = max(1, int((px - 4) / max(im.size)))
    if k * max(im.size) > px:
        k = 1
    big = im.resize((im.width * k, im.height * k), Image.NEAREST)
    if max(big.size) > px:
        big.thumbnail((px, px), Image.NEAREST)
    canvas = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    canvas.alpha_composite(big, ((px - big.width) // 2, (px - big.height) // 2))
    canvas.save(out)
    return Path(out)


def make_effect(kind: str, name: str | None = None, *, theme: str | None = None, colours=None,
                size=None, frames: int | None = None, project=None, loop: bool | None = None,
                seed: int = 0, dither: float = 0.5, tags=None):
    """Make an effect and save it as an asset (kind "effect") in the project (default: the
    current one) at its scale and palette. Returns the Asset."""
    from .assets import current_project, new_asset, set_tags
    proj = project or current_project()
    style = proj.style
    fx = render_effect(kind, theme=theme, colours=colours, size=size, frames=frames, loop=loop,
                       seed=seed, dither=dither, style=style)
    label = name or (SPECS[kind][5].split(" (")[0] + (f" ({fx.theme})" if fx.theme and
                                                     fx.theme != SPECS[kind][4] else ""))
    w, h = fx.size
    asset = new_asset("effect", label, proj, effect=kind, theme=fx.theme,
                      colours=[rgb_to_hex(c) for c in fx.ramp], size=[w, h],
                      frames=len(fx.frames), frame_ms=fx.frame_ms, loop=fx.loop, seed=seed,
                      dither=dither, files=["sheet.png", "sheet.json", "sheet.aseprite",
                                            "sheet.tres", "preview.gif", "thumb.png"])
    write_effect(fx, asset.folder)
    set_tags(asset, sorted({"effect", kind, *([fx.theme] if fx.theme else []), *(tags or [])}))
    return asset

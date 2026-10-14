"""Parallax backgrounds: layered side-scroller skies and landscapes in the neo 16-bit look, at
any time of day, seamless left-right, for Godot (Parallax2D scene) and Unity (layers + json +
a scroll script).

    bg = render_background("mountains", tod="sunset")      # procedural, CPU
    bg.layers   # [Layer] far -> near: sky, clouds, far (mountains), mid (hills + forest), near
    bg = background_from_prompt("misty bamboo valley", tod="dawn", client=...)   # SDXL + depth
    asset = make_background("coast", tod="night")            # kind "background", every file

Procedural layers: the sky is horizontal bands of a Lab ramp from the zenith to the horizon
glow (hue-shifted: deep blue -> violet -> orange at sunset), one row of alternating pixels
between bands (the classic 16-bit blend), a banded sun / moon halo, stars at night. Clouds,
ridges, hills, tree lines and foliage are periodic in x (whole-number frequencies; anything
drawn near an edge is drawn again one width over), so every layer repeats seamlessly. Colours
per layer = the biome's colour lit by the time of day (TOD light / shade, like hd2d.LIGHTING)
and hazed toward the horizon colour by distance (aerial perspective), each layer a 4-6 colour
ramp: shadow side, body, lit side facing the sun, rim light on the near layer.

From a prompt: generate.generate_images draws a wide landscape with the house style
(assets.with_style kind "tile"), parallax.build_parallax splits it into depth layers (depth.py's
model when it is already on disk; else a no-download guess: rows + the sky's colour), then the
layers are graded to the time of day.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from . import assets

KIND = "background"


def register_kind() -> None:
    if KIND not in assets.KINDS:
        assets.KINDS = (*assets.KINDS, KIND)
    assets.FOLDERS.setdefault(KIND, "backgrounds")


register_kind()

TODS = ("dawn", "noon", "sunset", "night")

# sky + light per time of day (matching hd2d.LIGHTING: dawn violet/peach, noon blue, sunset
# purple/orange, night navy with a cool moon)
SKY = {
    "dawn": dict(keys="1c1e4a 3a3a78 6a5a9e a87aac e0a0a8 f8c8a8 ffe8c8", sun="fff4dc", halo="ffd0a0",
                 sun_pos=(0.72, 0.6), sun_r=10, light=(1.05, 0.9, 0.86), shade=(0.62, 0.6, 0.86),
                 cloud="8a6a9e c08aa8 f0b8b0 ffe0cc", horizon="e8b0a8", stars=0.15),
    "noon": dict(keys="1a3c9c 2452c0 3a72dc 5a94ec 86b8f4 b4d8fa dcf0ff", sun="ffffff", halo="fff8e0",
                 sun_pos=(0.3, 0.14), sun_r=9, light=(1.04, 1.02, 0.96), shade=(0.72, 0.8, 1.0),
                 cloud="9aaed8 c4d4f0 eef4ff ffffff", horizon="c8e4fa", stars=0.0),
    "sunset": dict(keys="140e36 2a1a5a 52286e 8a3a72 c8546a f08a52 ffc060", sun="fff0b0", halo="ffb050",
                   sun_pos=(0.66, 0.58), sun_r=13, light=(1.1, 0.78, 0.55), shade=(0.5, 0.4, 0.72),
                   cloud="4a2a5e 8a3e6a d8665a ffb070", horizon="f0905a", stars=0.05),
    "night": dict(keys="03040e 060a1e 0a1230 101c44 182858 243870 34487e", sun="eef4ff", halo="6a80c0",
                  sun_pos=(0.24, 0.2), sun_r=8, light=(0.55, 0.66, 0.95), shade=(0.24, 0.3, 0.52),
                  cloud="141a36 222c54 34406e 4c5a8a", horizon="2e3c70", stars=1.0),
}

# biome colours at noon (dark, body, lit) per layer
BIOMES = {
    "green": dict(far="5a6a8a 7e8eaa a8b4c8", snow="dce6f4", hill="2e5a34 4a8a3e 7ab84e", forest="183a28 2a5a32 4a8a3a",
                  near="0e2418 1c3c22 30602c"),
    "autumn": dict(far="6a5a7a 8e7e98 b4a8bc", snow="eee8f0", hill="5a3a1e 8e6a2a c49a3a", forest="4a1a14 8a3018 c8601e",
                   near="2a120e 4a1e12 7a3418"),
    "desert": dict(far="8a5a4a b07a5a d8a878", snow=None, hill="a87240 d0a060 f0cc8a", forest="7a5030 a06c3c c89050",
                   near="4a2a1a 6e4026 9a6034"),
    "snow": dict(far="7a88aa a4b2cc d0dcec", snow="ffffff", hill="a8b8d4 d0dcec f4f8ff", forest="1e3440 2e4e58 4a7072",
                 near="5a6a88 8a9ab8 c4d0e4"),
    "coast": dict(far="4a6a7a 6a8c9a 98b4bc", snow=None, hill="2e5a40 4a8a4a 7ab858", forest="1a3a2c 2c5a36 4c8a40",
                  near="14281e 22402c 36603a"),
}

# scene: biome + the shapes of the far / mid / near layers
SCENES = {
    "mountains": dict(biome="green", far="peaks", mid="hills", trees="round", near="bushes"),
    "forest": dict(biome="green", far="hills_far", mid="hills", trees="pines", near="trees"),
    "autumn": dict(biome="autumn", far="peaks", mid="hills", trees="round", near="bushes"),
    "coast": dict(biome="coast", far="islands", mid="sea", trees=None, near="bushes"),
    "desert": dict(biome="desert", far="mesas", mid="dunes", trees=None, near="rocks"),
    "snowfield": dict(biome="snow", far="peaks", mid="hills", trees="pines", near="drifts"),
}
LAYER_NAMES = ("sky", "clouds", "far", "mid", "near")
SCROLL = {"sky": 0.0, "clouds": 0.1, "far": 0.25, "mid": 0.5, "near": 0.85}
HAZE = {"far": 0.55, "sea": 0.36, "mid": 0.24, "near": 0.04}


@dataclass
class Layer:
    name: str
    rgba: np.ndarray
    scroll: float
    autoscroll: float = 0.0          # px/s (clouds drift)


@dataclass
class Background:
    scene: str
    tod: str | None
    layers: list
    meta: dict = field(default_factory=dict)

    @property
    def size(self):
        h, w = self.layers[0].rgba.shape[:2]
        return w, h

    def composite(self, offsets=None) -> np.ndarray:
        """Layers stacked (each shifted right by offsets[i] px, wrapping) -> RGBA."""
        out = np.zeros_like(self.layers[0].rgba, dtype=np.float32)
        for i, L in enumerate(self.layers):
            img = L.rgba if not offsets else np.roll(L.rgba, int(offsets[i]), axis=1)
            a = img[..., 3:4].astype(np.float32) / 255
            out = img.astype(np.float32) * a + out * (1 - a)
        out[..., 3] = 255
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)

    def at_scroll(self, x: float) -> np.ndarray:
        """What a camera that moved x px sees (each layer by its scroll factor)."""
        return self.composite([-round(x * L.scroll) for L in self.layers])


# --------------------------------------------------------------------------- colour helpers

def _hex(keys):
    from .palettes import hex_to_rgb
    return hex_to_rgb(keys.split() if isinstance(keys, str) else list(keys)).astype(np.float32)


def _lab_mix(a, b, t):
    from .color import lab_to_rgb, rgb_to_lab
    la, lb = rgb_to_lab(np.asarray(a, np.float32).reshape(-1, 3)), rgb_to_lab(np.asarray(b, np.float32).reshape(-1, 3))
    return lab_to_rgb(la * (1 - t) + lb * t).reshape(-1, 3).astype(np.float32)


def _ramp(keys, n):
    from .fxcore import make_ramp
    r = make_ramp(keys, n).astype(np.float32)
    while len(r) < n:
        r = np.concatenate([r, r[-1:]])
    return r


def layer_ramp(biome_keys: str, tod: str, layer: str, n: int = 5, palette=None) -> np.ndarray:
    """A layer's ramp: biome colours (dark, body, lit) lit by the time of day and hazed toward
    the horizon by distance."""
    sky = SKY[tod]
    base = _hex(biome_keys)
    light, shade = np.array(sky["light"], np.float32), np.array(sky["shade"], np.float32)
    k = np.linspace(0, 1, len(base))[:, None]
    lit = np.clip(base * (shade * (1 - k) + light * k), 0, 255)
    hz = _hex(sky["horizon"])[0]
    lit = _lab_mix(lit, np.repeat(hz[None], len(lit), 0), HAZE.get(layer, 0.0))
    keys = ["%02x%02x%02x" % tuple(int(v) for v in np.clip(c, 0, 255)) for c in lit]
    r = _ramp(keys, n)
    if palette is not None:
        from .weather import _snap_each
        r = _snap_each(r.astype(np.uint8), palette).astype(np.float32)
    return r


def _paint(rgba, mask, colour):
    rgba[mask, :3] = np.clip(colour, 0, 255).astype(np.uint8)
    rgba[mask, 3] = 255


# --------------------------------------------------------------------------- periodic 1D shapes

def _ridge(W, rng, kind: str, amp: float, base: float, seed_freqs=(1, 2, 3, 5, 8)):
    """Height above `base` (px, >= 0) per column, periodic over W."""
    x = (np.arange(W) + 0.5) / W
    h = np.zeros(W, np.float32)
    if kind == "peaks":
        for f, a in zip((2, 3, 5, 9, 15), (1.0, 0.7, 0.45, 0.25, 0.12)):
            ph = rng.uniform(0, math.pi)
            h += a * (1 - np.abs(np.sin(math.pi * f * x + ph))) ** 1.6
        h = h / h.max()
    elif kind == "mesas":
        for f, a in zip((1, 2, 3, 7), (1.0, 0.7, 0.4, 0.12)):
            h += a * np.sin(2 * math.pi * f * x + rng.uniform(0, 6.3))
        h = (h - h.min()) / (np.ptp(h) + 1e-6)
        top = np.clip((h - 0.5) * 6, 0, 1)                     # wide flat-topped mesas, 2 tiers
        h = np.where(top > 0.35, 0.55 + 0.25 * (np.rint(top * 2) / 2), 0.15 * h + 0.9 * top)
        h = h * 0.75
    else:                                       # rolling hills / dunes / islands
        freqs = {"hills": (1, 2, 3, 5), "hills_far": (2, 3, 5, 7), "dunes": (2, 3, 4, 7),
                 "islands": (3, 4, 7, 11)}.get(kind, (1, 2, 3, 5))
        for f, a in zip(freqs, (1.0, 0.5, 0.3, 0.15)):
            h += a * np.sin(2 * math.pi * f * x + rng.uniform(0, 6.3))
        h = (h - h.min()) / (np.ptp(h) + 1e-6)
        if kind == "islands":
            h = np.clip((h - 0.55) * 2.4, 0, 1) ** 0.7
        if kind == "dunes":
            h = h ** 1.4
    return h * amp


def _wrapped_xs(x, r, W):
    return [x] + ([x - W] if x + r > W else []) + ([x + W] if x - r < 0 else [])


def _sun_dir(sky) -> float:
    """-1 = the sun is to the left, +1 right (which slopes are lit)."""
    return 1.0 if sky["sun_pos"][0] > 0.5 else -1.0


# --------------------------------------------------------------------------- layers

def _sky(W, H, tod, rng, palette=None):
    sky = SKY[tod]
    ramp = _ramp(sky["keys"], 12)
    if palette is not None:
        from .weather import _snap_each
        ramp = _snap_each(ramp.astype(np.uint8), palette).astype(np.float32)
    hor = int(H * 0.66)
    y = np.arange(H)
    t = np.clip(y / hor, 0, 1) ** 0.85                       # bands get closer near the horizon
    band = np.minimum((t * (len(ramp) - 1)).astype(int), len(ramp) - 1)
    img = np.zeros((H, W, 4), np.uint8)
    img[..., 3] = 255
    for yy in range(H):
        img[yy, :, :3] = ramp[band[yy]].astype(np.uint8)
    # the 16-bit blend: one row of alternating pixels where a band starts
    for yy in range(1, H):
        if band[yy] != band[yy - 1]:
            xs = np.arange(W)
            sel = (xs + yy) % 2 == 0
            img[yy, sel, :3] = ramp[band[yy - 1]].astype(np.uint8)
    # sun / moon with banded halo rings
    sx, sy = sky["sun_pos"][0] * W, sky["sun_pos"][1] * H
    r = max(4.0, sky["sun_r"] * H / 270)
    yy, xx = np.mgrid[0:H, 0:W]
    halo = _hex(sky["halo"])[0]
    for ring, k in ((r * 3.2, 0.14), (r * 2.2, 0.26), (r * 1.5, 0.42)):
        for cx in _wrapped_xs(sx, ring, W):
            m = (xx + 0.5 - cx) ** 2 + (yy + 0.5 - sy) ** 2 <= ring * ring
            img[m, :3] = np.clip(img[m, :3].astype(np.float32) * (1 - k) + halo * k, 0, 255).astype(np.uint8)
    for cx in _wrapped_xs(sx, r, W):
        d2 = (xx + 0.5 - cx) ** 2 + (yy + 0.5 - sy) ** 2
        disc = d2 <= r * r
        img[disc, :3] = _hex(sky["sun"])[0].astype(np.uint8)
        if tod == "night":                                   # moon: a crescent shadow + maria
            sh = (xx + 0.5 - cx - r * 0.45) ** 2 + (yy + 0.5 - sy + r * 0.2) ** 2 <= (r * 0.95) ** 2
            img[disc & sh, :3] = (_hex(sky["sun"])[0] * np.array([0.55, 0.6, 0.8])).astype(np.uint8)
            for ox, oy, rr in ((-0.35, -0.2, 0.22), (-0.1, 0.35, 0.16)):
                m = (xx + 0.5 - cx - ox * r) ** 2 + (yy + 0.5 - sy - oy * r) ** 2 <= (rr * r) ** 2
                img[disc & m & ~sh, :3] = (_hex(sky["sun"])[0] * 0.84).astype(np.uint8)
        else:
            rim = (d2 <= r * r) & (d2 > (r - 1.2) ** 2)
            img[rim, :3] = np.clip(_hex(sky["sun"])[0] * 0.92 + halo * 0.08, 0, 255).astype(np.uint8)
    # stars
    if sky["stars"] > 0:
        n = int(sky["stars"] * W * H / 220)
        for _ in range(n):
            x, y0 = int(rng.integers(W)), int(rng.integers(int(hor * 0.85)))
            b = rng.random()
            c = np.array([200, 210, 255]) * (0.5 + 0.5 * b) * (1 - y0 / hor * 0.6)
            img[y0, x, :3] = np.clip(np.maximum(img[y0, x, :3], c), 0, 255).astype(np.uint8)
            if b > 0.93 and 1 <= y0 < H - 1:
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    img[y0 + dy, (x + dx) % W, :3] = np.clip(np.maximum(img[y0 + dy, (x + dx) % W, :3], c * 0.6), 0, 255).astype(np.uint8)
    return img


def _clouds(W, H, tod, rng, amount=1.0, palette=None):
    """Cumulus rows: clusters of round lobes with flat bottoms, banded: shaded underside, body,
    lit top toward the sun, a bright rim."""
    sky = SKY[tod]
    ramp = _ramp(sky["cloud"], 5)
    if palette is not None:
        from .weather import _snap_each
        ramp = _snap_each(ramp.astype(np.uint8), palette).astype(np.float32)
    img = np.zeros((H, W, 4), np.uint8)
    yy, xx = np.mgrid[0:H, 0:W] + 0.5
    lx = _sun_dir(sky) * 0.6
    count = max(2, int(5 * amount * W / 480))
    for i in range(count):
        cx, cy = rng.uniform(0, W), rng.uniform(0.12, 0.5) * H
        width = rng.uniform(40, 90) * (0.6 + 0.6 * (cy / H)) * H / 270
        lobes = [(rng.uniform(-0.5, 0.5) * width, rng.uniform(-0.05, 0.25) * width * 0.3,
                  rng.uniform(0.18, 0.32) * width) for _ in range(int(rng.integers(4, 8)))]
        bottom = cy + width * 0.08
        for ox in _wrapped_xs(cx, width, W):
            inside = np.zeros((H, W), bool)
            shade = np.full((H, W), -9.0, np.float32)
            for dx, dy, r in lobes:
                px, py = ox + dx, cy - dy
                d = ((xx - px) ** 2 + ((yy - py) * 1.25) ** 2) / (r * r)
                m = d <= 1
                inside |= m
                # light from above + the sun's side
                nx, ny = (xx - px) / r, (yy - py) / r
                s = -ny * 0.9 + nx * lx
                shade = np.where(m, np.maximum(shade, s), shade)
            inside &= yy <= bottom
            if not inside.any():
                continue
            lvl = np.clip(np.floor((shade + 0.9) / 1.8 * 4), 0, 3).astype(int)
            lvl = np.where(yy > bottom - 2.5, 0, lvl)                  # the flat shaded base
            sel = inside & (img[..., 3] == 0)
            img[sel, :3] = ramp[lvl[sel]].astype(np.uint8)
            img[sel, 3] = 255
            # rim: top edge pixels one step brighter
            up = np.zeros_like(inside)
            up[1:] = inside[:-1]
            rim = inside & ~up
            img[rim & sel, :3] = ramp[4].astype(np.uint8)
    return img


def _ridge_layer(W, H, kind, biome, tod, rng, layer, top, amp, palette=None, snow=None, trees=None,
                 tree_keys=None):
    """A filled silhouette from its ridge line down to the bottom, lit by slope, with snow caps
    (peaks), haze bands toward its foot and an optional tree line on top."""
    ramp = layer_ramp(biome, tod, layer, 6, palette)
    sky = SKY[tod]
    h = _ridge(W, rng, kind, amp, top)
    line = np.rint(top + amp - h).astype(int)                 # y of the top per column
    img = np.zeros((H, W, 4), np.uint8)
    yy = np.arange(H)[:, None]
    body = yy >= line[None, :]
    k = max(3, W // 60)
    hs = np.convolve(np.concatenate([h[-k:], h, h[:k]]), np.ones(2 * k + 1) / (2 * k + 1), "same")[k:-k]
    slope = (np.roll(hs, -2) - np.roll(hs, 2)) / 2           # >0 rising to the right (smoothed: big facets)
    lit = (slope * _sun_dir(sky)) < -0.15                     # faces the sun
    dark = (slope * _sun_dir(sky)) > 0.5
    lv = np.full((H, W), 2)
    lv = np.where(lit[None, :], 3, lv)
    lv = np.where(dark[None, :], 1, lv)
    # crisp facets: shading bands follow the slope pattern down a few px, then flatten
    depth = yy - line[None, :]
    lv = np.where(depth > amp * 0.55, 1, lv)
    lv = np.where(depth > amp * 0.85 + 6, 0, lv)
    lv = np.where((depth >= 0) & (depth < 1) & lit[None, :], 4, lv)   # sunlit ridge rim
    if kind == "mesas":                                       # sandstone strata
        lv = np.where((depth > 2) & ((depth // 2) % 4 == 0) & (lv > 0), lv - 1, lv)
    img[body, :3] = ramp[lv[body]].astype(np.uint8)
    img[body, 3] = 255
    if snow is not None and kind == "peaks":
        sc = _ramp(snow, 3)
        snow_line = top + amp * 0.32
        capped = body & (yy < snow_line + (np.sin(np.arange(W) * 0.7) * 2)[None, :]) & (depth < amp * 0.4)
        sl = np.where(lit[None, :], 2, np.where(dark[None, :], 0, 1))
        sl = np.broadcast_to(sl, (H, W))
        cols = sc[sl[capped]] * np.array(sky["light"], np.float32) * 0.98
        img[capped, :3] = np.clip(cols, 0, 255).astype(np.uint8)
    if trees:
        _tree_line(img, line, W, H, trees, layer_ramp(tree_keys or biome, tod, layer, 6, palette), rng, sky)
    return img, line


def _tree_line(img, line, W, H, trees, ramp, rng, sky):
    """Crowns along a ridge: round (deciduous) or pines, each with a lit side."""
    x = 0.0
    k = max(0.5, H / 270)
    sd = _sun_dir(sky)
    yy, xx = np.mgrid[0:H, 0:W] + 0.5
    while x < W:
        r = rng.uniform(3.5, 7.0) * k
        cx = x
        base = line[int(cx) % W] + 1
        for ox in _wrapped_xs(cx, r * 2, W):
            if trees == "pines":
                hgt = r * rng.uniform(2.6, 3.6)
                top = base - hgt
                m = (yy >= top) & (yy <= base + 2) & (np.abs(xx - ox) <= (yy - top) / hgt * r * 1.05)
                # stepped branches: notches every few px
                m &= ~(((yy - top).astype(int) % 5 == 0) & (np.abs(xx - ox) > (yy - top) / hgt * r * 0.7))
            else:
                cy = base - r * 0.9
                m = (xx - ox) ** 2 + ((yy - cy) * 1.1) ** 2 <= r * r
                m |= (np.abs(xx - ox) < 1.0) & (yy > cy) & (yy < base + 2)
            if not m.any():
                continue
            lv = np.where((xx - ox) * sd > r * 0.15, 3, 2)
            lv = np.where((xx - ox) * sd < -r * 0.45, 1, lv)
            img[m, :3] = ramp[lv[m]].astype(np.uint8)
            img[m, 3] = 255
        x += r * rng.uniform(1.1, 1.7)


def _sea(W, H, top, tod, rng, palette=None):
    """A sea band from the horizon down: depth bands, glint lines under the sun."""
    sky = SKY[tod]
    from .water import THEMES
    keys = THEMES["night" if tod == "night" else "ocean"].split()[1:7]
    ramp = layer_ramp(" ".join(keys[:3]), tod, "sea", 6, palette)
    img = np.zeros((H, W, 4), np.uint8)
    yy, xx = np.mgrid[0:H, 0:W]
    sea = yy >= top
    lv = np.clip(((yy - top) / max(1, H - top) * 4).astype(int), 0, 3)
    lv = 3 - lv
    img[sea, :3] = ramp[lv[sea] + 1].astype(np.uint8)
    img[sea, 3] = 255
    # horizontal glint dashes, denser under the sun
    sx = sky["sun_pos"][0] * W
    for _ in range(int(W * (H - top) / 60)):
        y = int(rng.integers(top + 1, H))
        x = int(rng.normal(sx, W * 0.08)) if rng.random() < 0.6 else int(rng.integers(W))
        ln = int(rng.integers(2, 7))
        cols = (np.arange(ln) + x) % W
        near_sun = abs(((x - sx + W / 2) % W) - W / 2) < W * 0.12
        c = _hex(sky["halo"])[0] if near_sun and tod != "noon" else ramp[5] * np.array(sky["light"])
        img[y, cols, :3] = np.clip(c, 0, 255).astype(np.uint8)
    img[top, :, :3] = ramp[5].astype(np.uint8)
    return img


def _near(W, H, kind, biome, tod, rng, palette=None):
    """The foreground: dark clumps at the bottom with a rim of light on the sun's side."""
    ramp = layer_ramp(biome, tod, "near", 5, palette)
    sky = SKY[tod]
    sd = _sun_dir(sky)
    img = np.zeros((H, W, 4), np.uint8)
    yy, xx = np.mgrid[0:H, 0:W] + 0.5
    ground = H - int(H * 0.06)
    k = max(0.45, H / 270)
    img[ground:, :, :3] = ramp[0].astype(np.uint8)
    img[ground:, :, 3] = 255
    x = 0.0
    while x < W:
        if kind in ("bushes", "drifts", "trees"):
            r = (rng.uniform(7, 16) if kind != "trees" else rng.uniform(10, 20)) * k
            cx, cy = x, ground - r * (0.35 if kind != "trees" else rng.uniform(1.6, 3.2))
            for ox in _wrapped_xs(cx, r * 1.6, W):
                m = np.zeros((H, W), bool)
                for dx, dy, k in ((0, 0, 1.0), (-0.6, 0.25, 0.7), (0.6, 0.3, 0.75), (0.1, -0.45, 0.6)):
                    m |= (xx - ox - dx * r) ** 2 + (yy - cy - dy * r) ** 2 <= (k * r) ** 2
                if kind == "trees":
                    m |= (np.abs(xx - ox) < r * 0.14) & (yy > cy) & (yy < ground + 1)
                m &= yy < H
                sh = (xx - ox) * sd / r - (yy - cy) / r * 0.6
                lv = np.where(sh > 0.55, 3, np.where(sh > 0.1, 2, 1))
                img[m, :3] = ramp[lv[m]].astype(np.uint8)
                img[m, 3] = 255
            x += r * rng.uniform(1.0, 1.8)
        elif kind == "rocks":
            r = rng.uniform(5, 12) * k
            for ox in _wrapped_xs(x, r * 2, W):
                m = (np.abs(xx - ox) / (r * 1.3) + np.abs(yy - ground) / r) <= 1
                m &= yy <= ground + 2
                lv = np.where((xx - ox) * sd > 0, 3, 1)
                lv = np.where(yy < ground - r * 0.6, 4, lv)
                img[m, :3] = ramp[lv[m]].astype(np.uint8)
                img[m, 3] = 255
            x += r * rng.uniform(3.0, 7.0)
        else:                                               # cliffs (coast): a rocky foreground
            r = rng.uniform(14, 26) * k
            for ox in _wrapped_xs(x, r * 2, W):
                top = ground - r * 1.2
                m = (yy >= top + np.abs(xx - ox) * 0.6) & (np.abs(xx - ox) < r * 1.4)
                lv = np.where((xx - ox) * sd > 0, 2, 1)
                lv = np.where((yy - top - np.abs(xx - ox) * 0.6) < 2, 4, lv)
                img[m, :3] = ramp[lv[m]].astype(np.uint8)
                img[m, 3] = 255
            x += r * rng.uniform(2.0, 4.0)
    # grass / tufts along the top edge of everything (1-2 px blades)
    if kind in ("bushes", "trees"):
        alpha = img[..., 3] > 0
        top = alpha & ~np.vstack([np.zeros((1, W), bool), alpha[:-1]])
        ys, xs = np.nonzero(top)
        for y, x in zip(ys, xs):
            if rng.random() < 0.35 and y > 1:
                img[y - 1, x, :3] = ramp[2].astype(np.uint8)
                img[y - 1, x, 3] = 255
    return img


def render_background(scene: str = "mountains", tod: str = "sunset", *, width: int = 480, height: int = 270,
                      seed: int = 0, clouds: float = 1.0, style=None, palette=None) -> Background:
    """Procedural parallax layers (see the module doc). scene: SCENES; tod: dawn | noon | sunset
    | night; width x height in game pixels (480x270 = 1080p at 4x)."""
    from .effects import project_palette
    if scene not in SCENES:
        raise ValueError(f"Unknown scene '{scene}' (use {', '.join(SCENES)})")
    if tod not in SKY:
        raise ValueError(f"Unknown time of day '{tod}' (use {', '.join(TODS)})")
    W, H = int(width), int(height)
    if W < 64 or H < 48:
        raise ValueError("backgrounds need at least 64 x 48 pixels")
    pal = palette if palette is not None else project_palette(style)
    rng = np.random.default_rng(seed)
    sc = SCENES[scene]
    bio = BIOMES[sc["biome"]]
    layers = [Layer("sky", _sky(W, H, tod, rng, pal), SCROLL["sky"]),
              Layer("clouds", _clouds(W, H, tod, rng, clouds, pal), SCROLL["clouds"], autoscroll=-4.0)]
    hor = int(H * 0.66)
    far, _ = _ridge_layer(W, H, sc["far"], bio["far"], tod, rng, "far", hor - H * 0.36, H * 0.36, pal,
                          snow=bio["snow"])
    layers.append(Layer("far", far, SCROLL["far"]))
    if sc["mid"] == "sea":
        mid = _sea(W, H, hor + 2, tod, rng, pal)
    else:
        mid, _ = _ridge_layer(W, H, sc["mid"], bio["hill"], tod, rng, "mid", hor - H * 0.06,
                              H * 0.16, pal, trees=sc["trees"], tree_keys=bio["forest"])
    layers.append(Layer("mid", mid, SCROLL["mid"]))
    layers.append(Layer("near", _near(W, H, sc["near"], bio["near"], tod, rng, pal), SCROLL["near"]))
    return Background(scene, tod, layers, {"seed": seed, "size": [W, H], "source": "procedural"})


# --------------------------------------------------------------------------- from a prompt

def depth_available() -> bool:
    """Is depth.py's model already on disk (no download)?"""
    import os
    from . import depth
    if os.getenv(depth.ENV_MODEL):
        return Path(os.getenv(depth.ENV_MODEL)).exists()
    return any((depth.CACHE_DIR / s.file).exists() for s in depth.SOURCES)


def guess_depth(rgb: np.ndarray) -> np.ndarray:
    """A no-download depth guess for a landscape: lower rows are nearer; pixels coloured like
    the sky (the top rows) are farthest."""
    import cv2
    from .color import rgb_to_lab
    h, w = rgb.shape[:2]
    lab = rgb_to_lab(rgb.reshape(-1, 3).astype(np.float32)).reshape(h, w, 3)
    sky = np.median(lab[: max(1, h // 10)].reshape(-1, 3), 0)
    dist = np.linalg.norm(lab - sky, axis=-1)
    skyish = dist < max(12.0, np.percentile(dist, 25))
    y = (np.arange(h, dtype=np.float32) / max(1, h - 1))[:, None]
    d = np.broadcast_to(y ** 1.3, (h, w)).copy()
    d[skyish & (y < 0.75)] = 0.0
    return cv2.GaussianBlur(d, (0, 0), max(1.0, h / 120))


def background_prompt(words: str, tod: str, style=None) -> tuple[str, str]:
    tw = {"dawn": "at dawn, pink and violet sky", "noon": "midday, clear blue sky",
          "sunset": "at sunset, orange and purple sky", "night": "at night, moonlit, dark blue sky"}[tod]
    pos = f"{words}, wide side-scroller parallax background landscape, {tw}, layered distant scenery"
    neg = "characters, people, text, ui, border, frame, close-up"
    from .assets import with_style
    return with_style(pos, neg, style if style is not None else None, "tile")


def background_from_prompt(words: str, tod: str = "sunset", *, width: int = 480, height: int = 270,
                           layers: int = 3, client=None, seed: int | None = None, depth=None,
                           style=None, colours: int = 32, **gen_opts) -> Background:
    """SDXL draws a wide landscape in the house style -> seamless across -> split into depth
    layers (parallax.build_parallax) -> graded to the time of day. depth: a depth map to use;
    None = depth.py's model when it is on disk, else guess_depth (no downloads here)."""
    from .generate import generate_images
    from .parallax import build_parallax, seamless_x
    from .weather import _tint
    st = style if style is not None else assets.project_style()
    pos, neg = background_prompt(words, tod, st)
    (img, s), = generate_images(pos, neg, client=client, seed=seed, count=1, aspect=width / height, **gen_opts)
    rgb = np.asarray(img.convert("RGB"))
    if depth is None:
        if depth_available():
            from .depth import estimate_depth
            depth = estimate_depth(rgb)
        else:
            depth = guess_depth(rgb)
    res = build_parallax(rgb, layers, width, height, colors=colours, dither="none", depth=depth)
    out = [Layer("sky" if i == 0 else n, sp.rgba, [0.1, 0.4, 0.85][min(i, 2)] if layers == 3 else
                 float(np.linspace(0.1, 0.9, layers)[i])) for i, (sp, n) in enumerate(zip(res.sprites, res.names))]
    lo, hi = {"dawn": "dawn", "noon": "noon", "sunset": "sunset", "night": "night"}[tod], None
    del hi
    for L in out:                              # grade: lit top of each layer's ramp, shade below
        a = L.rgba[..., 3] > 0
        if a.any():
            px = L.rgba[a, :3]
            lum = px.astype(np.float32).mean(1)
            order = np.argsort(np.argsort(lum)) / max(1, len(lum) - 1)
            graded = np.empty_like(px)
            for q, sel in ((0, order < 0.5), (1, order >= 0.5)):
                if sel.any():
                    graded[sel] = _tint(px[sel], lo, 0.7)[...] if q == 0 else _tint(px[sel][::-1], lo, 0.7)[::-1]
            L.rgba[a, :3] = graded
    return Background("prompt", tod, out, {"prompt": pos, "seed": s, "size": [width, height],
                                           "source": "sdxl", "depth": "model" if depth_available() else "guess"})


# --------------------------------------------------------------------------- files

UNITY_CS = """using UnityEngine;

// pixel-pipeline parallax: put on an empty GameObject; it builds a SpriteRenderer per layer from
// parallax.json (sprites imported with these .meta files) and scrolls them with the camera.
public class PixelParallax : MonoBehaviour
{
    [System.Serializable] public class LayerInfo { public string name; public string file; public float scroll; public float autoscroll; }
    [System.Serializable] public class Info { public LayerInfo[] layers; public int width; public int height; public float ppu; }
    public TextAsset parallaxJson;
    public Sprite[] sprites;              // same order as parallax.json layers
    public Transform cam;
    Info info;
    Transform[,] tiles;
    Vector3 start;
    float t;

    void Start()
    {
        info = JsonUtility.FromJson<Info>(parallaxJson.text);
        if (cam == null && Camera.main != null) cam = Camera.main.transform;
        start = cam != null ? cam.position : Vector3.zero;
        tiles = new Transform[info.layers.Length, 3];
        for (int i = 0; i < info.layers.Length; i++)
            for (int k = 0; k < 3; k++)
            {
                var go = new GameObject(info.layers[i].name + "_" + k);
                go.transform.SetParent(transform, false);
                var sr = go.AddComponent<SpriteRenderer>();
                sr.sprite = sprites[i];
                sr.sortingOrder = -100 + i;
                tiles[i, k] = go.transform;
            }
    }

    void LateUpdate()
    {
        if (cam == null) return;
        t += Time.deltaTime;
        float w = info.width / info.ppu;
        for (int i = 0; i < info.layers.Length; i++)
        {
            var L = info.layers[i];
            float x = (cam.position.x - start.x) * (1f - L.scroll) + L.autoscroll * t / info.ppu;
            float off = Mathf.Repeat(cam.position.x - x, w);
            for (int k = 0; k < 3; k++)
                tiles[i, k].position = new Vector3(cam.position.x - off + (k - 1) * w, cam.position.y, 10f - i);
        }
    }
}
"""


def write_background(bg: Background, folder, name: str = "background", ppu: float = 1.0) -> list[Path]:
    """Layer PNGs (<name>_<i>_<layer>.png), a preview, Godot Parallax2D scene (clouds
    autoscroll), parallax.json + Unity metas + PixelParallax.cs, a scrolling preview GIF."""
    from .exports import unity_guid, unity_meta
    from .parallax import write_godot_parallax
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    pngs, written = [], []
    for i, L in enumerate(bg.layers):
        p = folder / f"{name}_{i}_{L.name}.png"
        Image.fromarray(L.rgba, "RGBA").save(p)
        pngs.append(p)
    written += pngs
    comp = bg.composite()
    prev = folder / "preview.png"
    Image.fromarray(comp, "RGBA").save(prev)
    th = Image.fromarray(comp, "RGBA")
    th.thumbnail((192, 108), Image.NEAREST)
    th.save(folder / "thumb.png")
    written += [prev, folder / "thumb.png"]
    tscn = write_godot_parallax(pngs, folder / f"{name}_parallax.tscn", [L.scroll for L in bg.layers],
                                repeat=True, root_name=f"{name}_parallax")
    text = tscn.read_text(encoding="utf-8")
    from .parallax import layer_names
    roles = layer_names(len(bg.layers))
    for L, role in zip(bg.layers, roles):
        if L.autoscroll:
            node = f'[node name="{role.capitalize()}" type="Parallax2D" parent="."]\n'
            text = text.replace(node, node + f"autoscroll = Vector2({L.autoscroll:g}, 0)\n")
    tscn.write_text(text, encoding="utf-8")
    written.append(tscn)
    W, H = bg.size
    info = {"scene": bg.scene, "tod": bg.tod, "width": W, "height": H, "ppu": ppu,
            "layers": [{"name": L.name, "file": p.name, "scroll": L.scroll, "autoscroll": L.autoscroll}
                       for L, p in zip(bg.layers, pngs)], "meta": bg.meta}
    js = folder / "parallax.json"
    js.write_text(json.dumps(info, indent=1), encoding="utf-8")
    written.append(js)
    udir = folder / "unity"
    udir.mkdir(exist_ok=True)
    for p in pngs + [js]:
        dst = udir / p.name
        dst.write_bytes(p.read_bytes())
        guid = unity_guid(f"pixel-pipeline/bg/{folder.name}/{p.name}")
        if p.suffix == ".png":
            meta = unity_meta(guid, (W, H), None, ppu=ppu)
        else:
            meta = f"fileFormatVersion: 2\nguid: {guid}\nTextScriptImporter:\n  externalObjects: {{}}\n  userData: \n  assetBundleName: \n  assetBundleVariant: \n"
        (udir / (p.name + ".meta")).write_text(meta, encoding="utf-8")
        written += [dst, udir / (p.name + ".meta")]
    cs = udir / "PixelParallax.cs"
    cs.write_text(UNITY_CS, encoding="utf-8")
    written.append(cs)
    gif = folder / "preview.gif"
    frames = [Image.fromarray(bg.at_scroll(x)[..., :3]) for x in range(0, W, max(4, W // 60))]
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=60, loop=0)
    written.append(gif)
    return written


def make_background(scene: str = "mountains", tod: str = "sunset", name: str | None = None, *, project=None,
                    width: int | None = None, height: int | None = None, seed: int = 0, prompt: str | None = None,
                    client=None, tags=None, **opts):
    """Make a background (procedural, or from words with prompt=) and save it as kind
    "background". Size default: 480x270 (Kevin's 1080p base). Returns the Asset."""
    register_kind()
    proj = project or assets.current_project()
    st = proj.style
    W, H = int(width or 480), int(height or 270)
    if prompt:
        bg = background_from_prompt(prompt, tod, width=W, height=H, client=client, seed=seed, style=st, **opts)
    else:
        bg = render_background(scene, tod, width=W, height=H, seed=seed, style=st, **opts)
    label = name or (f"{(prompt or scene).strip().title()[:40]} ({tod})")
    a = assets.new_asset(KIND, label, proj, scene=bg.scene, tod=tod, size=[W, H], seed=seed, prompt=prompt,
                         layers=[L.name for L in bg.layers])
    files = write_background(bg, a.folder)
    a.extra["files"] = [p.relative_to(a.folder).as_posix() for p in files]
    a.save()
    assets.set_tags(a, sorted({"background", bg.scene, tod, *(tags or [])}))
    return a

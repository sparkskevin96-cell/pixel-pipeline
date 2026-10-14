"""Weather & atmosphere assets: rain, snow, fog, god rays, leaves, petals, embers, fireflies,
dust motes, sandstorm, lightning and cloud shadows, drawn procedurally (CPU, no AI) in the neo
16-bit look ("SNES as if 2D consoles never stopped": crisp pixel shapes, big hue-shifted ramps,
banded - never blurry - transparency, glow maps for what shines).

    w = render_weather("rain", tod="night")      # -> Weather
    w.fx                 # effects.Effect: the looping full-screen overlay tile (frames, glow, ms)
    w.particles          # {"drop_near": [frames], "splash": [...], ...} particle textures
    w.emitters           # [Emitter] how a particle system throws them (one per depth layer)
    asset = make_weather("rain", tod="night")    # saved as kind "weather" with every file

Every overlay is a seamless LOOP that also TILES (both ways, or across only for god rays):
everything moves a whole number of tiles per loop (rain near layer 4 tiles down + 1 across,
snow sways a whole number of times...), so the last frame runs into the first and a 128 px
tile covers any screen. Layers by depth: far (small, slow, dim), mid, near (big, fast,
bright). Shapes are drawn as heat fields at 4x (fxcore.Canvas) and banded onto Lab ramps
tinted for the time of day (TOD_TINT: dawn / noon / sunset / night, matching hd2d.LIGHTING);
each band has its own alpha, so fog and light shafts are see-through in crisp steps.

Files (write_weather): sheet.png/.json/.aseprite/.tres (+ sheet_glow.png) like every effect, so
the Library's exports (Godot 2D, Unity, GameMaker...) work as is; particles/<name>.png strips
(+ _glow); weather.json (emitters in pixels AND in Unity units / metres); godot/: <name>_overlay
.tscn (a CanvasLayer that tiles the loop over any screen: 2D games), <name>_particles2d.tscn
(GPUParticles2D per depth layer), <name>_particles3d.tscn (GPUParticles3D billboards for 3D /
HD-2D boards), weather_overlay.gdshader + weather_fx.gd.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from . import assets
from .fxcore import Canvas, Noise, bolt_points, draw_bolt, finish, make_ramp, smooth

KIND = "weather"


def register_kind() -> None:
    """Make sure "weather" is an asset kind (additive; assets.py owns the list)."""
    if KIND not in assets.KINDS:
        assets.KINDS = (*assets.KINDS, KIND)
    assets.FOLDERS.setdefault(KIND, "weather")


register_kind()

MATS = ("back", "smoke", "main", "accent")
TODS = ("dawn", "noon", "sunset", "night")

# multipliers for a ramp's dark end (lo) and bright end (hi) per time of day (from hd2d.LIGHTING:
# ambient for the shadows, the sun / moon for the lights)
TOD_TINT = {
    None: ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
    "dawn": ((0.80, 0.74, 0.96), (1.04, 0.9, 0.86)),
    "noon": ((0.96, 1.0, 1.06), (1.02, 1.01, 0.98)),
    "sunset": ((0.78, 0.6, 0.86), (1.08, 0.82, 0.62)),
    "night": ((0.42, 0.5, 0.84), (0.7, 0.8, 1.02)),
}


# --------------------------------------------------------------------------- data

@dataclass
class Emitter:
    """One particle layer, engine-neutral. Pixels and seconds, 2D y down (3D: y up, /ppu)."""
    name: str
    texture: str                  # particles/<texture>.png (a strip of `frames` frames)
    frames: int = 1
    amount: int = 60
    lifetime: float = 1.0
    velocity: tuple = (0.0, 100.0)   # px/s
    speed_var: float = 0.15          # +- share of the speed
    spread: float = 4.0              # degrees
    gravity: tuple = (0.0, 0.0)      # px/s^2
    scale: tuple = (1.0, 1.0)
    sway: float = 0.0                # 0..1: turbulence (snow, leaves, motes)
    spin: float = 0.0                # deg/s (random +-)
    fade: tuple = (0.1, 0.2)         # share of life fading in, fading out
    anim: str = "none"               # none | life (plays once over its life) | loop
    anim_fps: float = 10.0
    area: str = "sky"                # sky (falls in from above) | fill (the whole view) |
    #                                  ground (stays where born) | below (rises from below) |
    #                                  side (blows in from the left/right)
    blend: str = "mix"               # mix | add
    glow: bool = False
    depth: float = 0.5               # 0 far .. 1 near (z order, 3D spread)
    align: bool = False              # turn the texture along the velocity (rain streaks)
    alpha: float = 1.0


@dataclass
class Weather:
    kind: str
    tod: str | None
    fx: object                        # effects.Effect (the looping overlay tile)
    particles: dict                   # name -> [RGBA frames]
    particle_glow: dict               # name -> [RGBA frames] (glowing ones only)
    emitters: list
    meta: dict = field(default_factory=dict)   # wrap, scroll, flash, screen, label...

    @property
    def size(self):
        return self.fx.size


# --------------------------------------------------------------------------- colour

def _tint(ramp: np.ndarray, tod, strength: float = 1.0) -> np.ndarray:
    lo, hi = (np.array(v, np.float32) for v in TOD_TINT.get(tod, TOD_TINT[None]))
    n = len(ramp)
    t = np.linspace(0, 1, n)[:, None] if n > 1 else np.ones((1, 1))
    mul = lo * (1 - t) + hi * t
    mul = 1 + (mul - 1) * strength
    return np.clip(ramp.astype(np.float32) * mul, 0, 255).astype(np.uint8)


def _snap_each(ramp: np.ndarray, palette) -> np.ndarray:
    if palette is None or not len(palette):
        return ramp
    from .color import rgb_to_lab
    a = rgb_to_lab(ramp.astype(np.float32))
    b = rgb_to_lab(np.asarray(palette, np.uint8).astype(np.float32))
    return np.asarray(palette, np.uint8)[np.argmin(((a[:, None] - b[None]) ** 2).sum(-1), 1)]


def build_look(spec: dict, tod=None, palette=None, colours=None) -> dict:
    """{mat: {"keys", "n", "a": (alpha dark, alpha bright), "e": (emit from, to), "tint"}} ->
    {mat: (ramp (n,3), alpha (n,), emit (n,))}. colours: your key colours replace "main"'s."""
    out = {}
    for m, d in spec.items():
        keys = d["keys"]
        if colours and m in ("main", "accent"):
            from .effects import _parse_colours
            rgb = _parse_colours(colours)
            keys = ["%02x%02x%02x" % tuple(int(v) for v in c) for c in rgb]
            if len(keys) == 1:
                keys = ["%02x%02x%02x" % tuple(int(v * k) for v in rgb[0]) for k in (0.35, 0.7)] + keys + ["ffffff"]
        ramp = make_ramp(keys, d.get("n", 6))
        ramp = _tint(ramp, tod, d.get("tint", 1.0))
        ramp = _snap_each(ramp, palette)
        n = len(ramp)
        a0, a1 = d.get("a", (255, 255))
        alpha = np.rint(np.linspace(a0, a1, n) ** 1.0).astype(np.uint8)
        e0, e1 = d.get("e", (0.0, 0.0))
        emit = np.linspace(e0, e1, n).astype(np.float32)
        out[m] = (ramp, alpha, emit)
    return out


def wfinish(cv: Canvas, look: dict, cover: float = 0.45):
    """Canvas -> (rgba, glow rgba): per pixel the topmost material covering `cover` of it, its
    mean heat banded onto the material's ramp, with that band's alpha (crisp see-through
    steps). Glow = colour x the band's emission where the frontmost material emits."""
    res = cv.resolve()
    h, w = cv.h, cv.w
    rgba = np.zeros((h, w, 4), np.uint8)
    glow = np.zeros((h, w, 4), np.uint8)
    for m in MATS:
        if m not in look:
            continue
        cov, ht = res[m]
        ramp, alpha, emit = look[m]
        n = len(ramp)
        on = cov >= cover
        if not on.any():
            continue
        lv = np.clip(np.floor(ht * n), 0, n - 1).astype(np.int16)[on]
        rgba[on, :3] = ramp[lv]
        rgba[on, 3] = alpha[lv]
        e = emit[lv]
        g = np.zeros((int(on.sum()), 4), np.uint8)
        lit = e > 0.01
        g[lit, :3] = np.clip(ramp[lv][lit].astype(np.float32) * e[lit, None], 0, 255).astype(np.uint8)
        g[lit, 3] = 255
        glow[on] = g
    return rgba, glow


# --------------------------------------------------------------------------- helpers

class _Ctx:
    def __init__(self, kind, w, h, n, rng, opts, scale):
        self.kind, self.W, self.H, self.n, self.rng, self.opts, self.k = kind, w, h, n, rng, opts, scale

    ss = 4

    def canvas(self, w=None, h=None):
        return Canvas(w or self.W, h or self.H, self.ss)

    def noise(self, cells=4, octaves=3):
        return Noise(self.rng, cells, octaves)


def _wraps(x, y, r, W, H, wx=True, wy=True):
    """(x, y) and its copies across the tile edges a shape of radius r overlaps."""
    x = x % W if wx else x
    y = y % H if wy else y
    xs = [x] + ([x - W] if wx and x + r > W else []) + ([x + W] if wx and x - r < 0 else [])
    ys = [y] + ([y - H] if wy and y + r > H else []) + ([y + H] if wy and y - r < 0 else [])
    return [(a, b) for a in xs for b in ys]


def _streak(cv, mat, x, y, dx, dy, length, width, heat, tail=0.35, W=None, H=None, wy=True):
    """A tapered streak whose head is at (x, y), coming from direction (dx, dy), wrapped."""
    L = math.hypot(dx, dy) or 1.0
    ux, uy = dx / L, dy / L
    for px, py in _wraps(x, y, length + 2, W, H, True, wy):
        cv.stroke(mat, [(px - ux * length, py - uy * length), (px, py)], [width * 0.55, width],
                  [heat * tail, heat])


def _dot(cv, mat, x, y, r, heat, W, H, wy=True, snap=True):
    for px, py in _wraps(x, y, r + 1, W, H, True, wy):
        cv.ellipse(mat, px, py, r, heat=heat, snap=snap)


def _lens(cx, cy, L, Wd, ang, n=10):
    """A pointed leaf / petal outline: length L, width Wd, turned by ang."""
    t = np.linspace(0, math.pi, n)
    pts_r = [(math.cos(a) * L / 2, math.sin(a) * Wd / 2) for a in t]
    pts = pts_r + [(x, -y) for x, y in reversed(pts_r[1:-1])]
    c, s = math.cos(ang), math.sin(ang)
    return [(cx + x * c - y * s, cy + x * s + y * c) for x, y in pts]


def _strip(frames):
    """Frames side by side (a particle texture with h_frames)."""
    return np.concatenate(frames, axis=1)


def _sprites(w, h, n, draw, look, cover=0.45):
    """n frames of a small particle sprite: draw(cv, k) on a w x h canvas."""
    frames, glows = [], []
    for k in range(n):
        cv = Canvas(w, h)
        draw(cv, k)
        f, g = wfinish(cv, look, cover)
        frames.append(f)
        glows.append(g)
    return frames, glows


# =========================================================================== the weathers
# maker(c) -> dict(draw=fn(cv, k, p), look=spec, cover=, particles={name: (w, h, n, draw(cv,k))},
#               emitters=[Emitter], meta={...});  p = loop phase 0..1 of frame k

def _rain(c: _Ctx):
    """Three depth layers of slanted streaks (far thin and dim, near long with a bright head),
    crown splashes and puddle rings on the ground (ground=False for side views)."""
    W, H, rng, s = c.W, c.H, c.rng, c.k
    heavy = float(c.opts.get("intensity", 1.0))
    wind = float(c.opts.get("wind", 0.35))
    ground = bool(c.opts.get("ground", True))
    layers = []
    for name, cnt, ky, ln, wd, heat, mat in (("far", 70, 2, 4.5, 0.5, 0.42, "back"),
                                             ("mid", 34, 3, 8.5, 0.55, 0.62, "main"),
                                             ("near", 12, 4, 15.0, 0.85, 0.95, "main")):
        kx = int(round(wind * ky * H / W)) if wind else 0
        for _ in range(max(1, int(cnt * heavy * (W * H) / (128 * 128)))):
            layers.append(dict(x=rng.uniform(0, W), y=rng.uniform(0, H), kx=kx, ky=ky,
                               ln=ln * s * rng.uniform(0.8, 1.2), wd=wd * max(0.8, s),
                               heat=heat * rng.uniform(0.85, 1.05), mat=mat))
    n = c.n
    splashes = [dict(x=rng.uniform(0, W), y=rng.uniform(0, H), f0=int(rng.integers(n)),
                     big=rng.random() < 0.35) for _ in range(int(9 * heavy * W * H / 16384))] if ground else []
    rings = [dict(x=rng.uniform(0, W), y=rng.uniform(0, H), f0=int(rng.integers(n)))
             for _ in range(int(6 * heavy * W * H / 16384))] if ground else []

    def splash(cv, x, y, j, big, mat="accent"):
        r = (1.6 if big else 1.0) * s
        if j == 0:
            cv.ellipse(mat, x, y, 0.6 * r + 0.3, 0.45 * r + 0.2, heat=0.95)
        elif j == 1:
            cv.ring(mat, x, y, 1.4 * r, 0.8, heat=0.85, squash=0.45)
            for sx in (-1, 1):
                cv.ellipse(mat, x + sx * 1.3 * r, y - 1.6 * r, 0.5, heat=0.9, snap=True)
        elif j == 2:
            cv.ring(mat, x, y, 2.2 * r, 0.7, heat=0.6, squash=0.4,
                    keep=lambda a: np.abs(np.sin(a)) > 0.25)
            for sx in (-1, 1):
                cv.ellipse(mat, x + sx * 2.2 * r, y - 1.4 * r, 0.5, heat=0.72, snap=True)
        elif j == 3:
            for sx in (-1, 1):
                cv.ellipse(mat, x + sx * 2.8 * r, y - 0.3 * r, 0.5, heat=0.5, snap=True)

    def ripple(cv, x, y, j):
        if j > 6:
            return
        r = (1.4 + j * 1.05) * s
        cv.ring("smoke", x, y, r, 0.75, heat=0.85 - j * 0.11, squash=0.42,
                keep=(lambda a: np.abs(np.sin(a * 0.5 + 0.7)) > 0.2) if j > 3 else None)

    def draw(cv, k, p):
        for d in layers:
            x = d["x"] + d["kx"] * W * p
            y = d["y"] + d["ky"] * H * p
            _streak(cv, d["mat"], x, y, d["kx"] * W, d["ky"] * H, d["ln"], d["wd"], d["heat"],
                    W=W, H=H)
        for d in rings:
            j = (k - d["f0"]) % n
            for px, py in _wraps(d["x"], d["y"], 9 * s, W, H):
                ripple(cv, px, py, j)
        for d in splashes:
            j = (k - d["f0"]) % n
            if j < 4:
                for px, py in _wraps(d["x"], d["y"], 5 * s, W, H):
                    splash(cv, px, py, j, d["big"])

    look = {"back": dict(keys="27304e 3c5070 5a7896 86a6c2", n=4, a=(70, 120)),
            "smoke": dict(keys="34507a 5a80aa 8ab0d4 c4e0f4", n=4, a=(60, 150)),
            "main": dict(keys="2c3c62 4a6890 7aa0c8 b0d4ee e6f6ff", n=6, a=(90, 225)),
            "accent": dict(keys="6a90ba a6cce8 dcf2ff ffffff", n=4, a=(150, 240))}
    v = 4 * H / (c.n * c.ms / 1000)
    vx = wind * v
    parts = {
        "drop_far": (1, 6, 1, lambda cv, k: cv.stroke("back", [(0.5, 0.2), (0.5, 5.8)], [0.3, 0.5], [0.3, 0.6])),
        "drop_mid": (2, 10, 1, lambda cv, k: cv.stroke("main", [(1.0, 0.3), (1.0, 9.6)], [0.35, 0.55], [0.3, 0.7])),
        "drop_near": (3, 16, 1, lambda cv, k: cv.stroke("main", [(1.5, 0.3), (1.5, 15.4)], [0.4, 0.85], [0.3, 0.98])),
        "splash": (10, 7, 4, lambda cv, k: splash(cv, 5, 5.5, k, True)),
        "ripple": (18, 9, 7, lambda cv, k: ripple(cv, 9, 4.5, k)),
    }
    ems = [Emitter("far", "drop_far", amount=int(160 * heavy), lifetime=1.1, velocity=(vx * 0.5, v * 0.5),
                   area="sky", depth=0.1, align=True, alpha=0.7, fade=(0.0, 0.05)),
           Emitter("mid", "drop_mid", amount=int(90 * heavy), lifetime=0.8, velocity=(vx * 0.75, v * 0.75),
                   area="sky", depth=0.5, align=True, fade=(0.0, 0.05)),
           Emitter("near", "drop_near", amount=int(36 * heavy), lifetime=0.55, velocity=(vx, v),
                   area="sky", depth=0.9, align=True, fade=(0.0, 0.05))]
    if ground:
        ems += [Emitter("splash", "splash", frames=4, amount=int(40 * heavy), lifetime=0.16,
                        velocity=(0, 0), area="ground", anim="life", depth=0.6, fade=(0, 0)),
                Emitter("ripple", "ripple", frames=7, amount=int(26 * heavy), lifetime=0.42,
                        velocity=(0, 0), area="ground", anim="life", depth=0.55, fade=(0, 0.2))]
    return dict(draw=draw, look=look, cover=0.4, particles=parts, emitters=ems,
                meta={"wet": True, "darken": 0.72})


def _snow(c: _Ctx):
    """Flakes in three depths: far single pixels, mid crosses, near six-armed flakes and soft
    out-of-focus discs, all swaying a whole number of times per loop."""
    W, H, rng, s = c.W, c.H, c.rng, c.k
    heavy = float(c.opts.get("intensity", 1.0))
    wind = float(c.opts.get("wind", 0.15))
    flakes = []
    for name, cnt, ky, size in (("far", 60, 1, 0), ("mid", 26, 1, 1), ("near", 9, 2, 2), ("bokeh", 3, 3, 3)):
        for _ in range(max(1, int(cnt * heavy * W * H / 16384))):
            flakes.append(dict(x=rng.uniform(0, W), y=rng.uniform(0, H), ky=ky,
                               kx=int(round(wind * ky)), size=size, amp=rng.uniform(1.5, 4.5) * s * (1 + size * 0.4),
                               m=int(rng.integers(1, 3)), ph=rng.uniform(0, 2 * math.pi),
                               rot=rng.uniform(0, math.pi), spin=int(rng.choice([-1, 1]))))

    def flake(cv, x, y, size, rot, heat=1.0):
        if size == 0:
            cv.ellipse("back", x, y, 0.55, heat=0.75 * heat, snap=True)
        elif size == 1:
            cv.star("main", x, y, 1.45 * s, heat=0.8 * heat, rot=0.0, thin=0.9)
        elif size == 2:
            r = 2.6 * s
            for a in range(3):
                ang = rot + a * math.pi / 3
                dx, dy = math.cos(ang) * r, math.sin(ang) * r
                cv.stroke("main", [(x - dx, y - dy), (x + dx, y + dy)], 0.48, [0.75, 0.75])
            cv.ellipse("accent", x, y, 0.75 * s, heat=1.0, snap=True)
        else:
            cv.ellipse("smoke", x, y, 3.2 * s, heat=lambda rho, X, Y: 0.85 - 0.5 * rho)

    def draw(cv, k, p):
        for f in flakes:
            x = f["x"] + f["kx"] * W * p + f["amp"] * math.sin(2 * math.pi * f["m"] * p + f["ph"])
            y = f["y"] + f["ky"] * H * p
            for px, py in _wraps(x, y, 5 * s, W, H):
                flake(cv, px, py, f["size"], f["rot"] + f["spin"] * 2 * math.pi * p / 3)

    look = {"back": dict(keys="8a96c0 b4c2e2 dce6f8", n=3, a=(150, 220)),
            "smoke": dict(keys="b8c8ea dce8fa f4f8ff", n=3, a=(40, 110)),
            "main": dict(keys="7484b4 a0b2dc cddcf4 eef4ff ffffff", n=5, a=(200, 250)),
            "accent": dict(keys="eef6ff ffffff", n=2, a=(255, 255))}
    v = H / (c.n * c.ms / 1000)
    parts = {
        "flake_small": (2, 2, 1, lambda cv, k: cv.ellipse("main", 1, 1, 0.9, heat=0.85)),
        "flake": (5, 5, 1, lambda cv, k: flake(cv, 2.5, 2.5, 1, 0)),
        "flake_big": (9, 9, 1, lambda cv, k: flake(cv, 4.5, 4.5, 2, 0.3)),
        "bokeh": (9, 9, 1, lambda cv, k: flake(cv, 4.5, 4.5, 3, 0)),
    }
    ems = [Emitter("far", "flake_small", amount=int(140 * heavy), lifetime=5.0, velocity=(wind * v, v * 0.6),
                   spread=12, sway=0.25, area="sky", depth=0.1, alpha=0.8),
           Emitter("mid", "flake", amount=int(70 * heavy), lifetime=4.0, velocity=(wind * v, v),
                   spread=12, sway=0.4, spin=40, area="sky", depth=0.5),
           Emitter("near", "flake_big", amount=int(22 * heavy), lifetime=2.6, velocity=(wind * v * 2, v * 2),
                   spread=12, sway=0.55, spin=90, area="sky", depth=0.85),
           Emitter("bokeh", "bokeh", amount=int(6 * heavy), lifetime=1.6, velocity=(wind * v * 3, v * 3),
                   spread=10, sway=0.3, scale=(1.0, 1.6), area="sky", depth=1.0, alpha=0.6)]
    return dict(draw=draw, look=look, cover=0.4, particles=parts, emitters=ems,
                meta={"snow": True})


def _fog_field(noise_a, noise_b, X, Y, W, H, p, thick=1.0, drift=1):
    """Fog density in [0, 1]: two periodic noises drifting at whole tiles per loop, against
    each other, banded on the cv side. Ground-hugging: denser toward the tile's middle band."""
    a = noise_a(X / W - drift * p, Y / H * 2)
    b = noise_b(X / W + 2 * drift * p + 0.31, Y / H * 2 + 0.17)
    d = 0.62 * a + 0.38 * b
    return np.clip((d - 0.58 + 0.1 * thick) * 2.6, 0, 1)


def _fog(c: _Ctx):
    """Drifting fog banks: two noise layers sliding past each other (a whole tile per loop),
    cut into 5 crisp density bands; the top edge of each bank catches the light."""
    W, H = c.W, c.H
    thick = float(c.opts.get("intensity", 1.0))
    na, nb = c.noise(2, 3), c.noise(4, 2)

    def draw(cv, k, p):
        def fn(X, Y):
            d = _fog_field(na, nb, X, Y, W, H, p, thick)
            up = _fog_field(na, nb, X, Y - 2.0, W, H, p, thick)
            lit = np.clip((d - up) * 3.0, 0, 0.35)          # bank tops face the sky: lighter
            return d > 0.08, np.clip(d * 0.85 + lit, 0, 1)
        cv.field("smoke", (0, 0, W, H), fn)

    look = {"smoke": dict(keys="8894b0 a2aec6 bec8da d8e0ec f0f4fa", n=5, a=(30, 132))}

    def puff(cv, k, w=64, h=32):
        g = Noise(np.random.default_rng(7 + k), 3, 3)

        def fn(X, Y):
            rx, ry = (X - w / 2) / (w / 2), (Y - h / 2) / (h / 2)
            body = 1 - np.sqrt(rx * rx + ry * ry * 1.2)
            d = body * 1.4 + (g(X / w, Y / h) - 0.5) * 0.9
            return d > 0.05, np.clip(d, 0, 1) * 0.85
        cv.field("smoke", (0, 0, w, h), fn)

    parts = {"fog_puff": (64, 32, 1, lambda cv, k: puff(cv, 0)),
             "fog_wisp": (96, 24, 1, lambda cv, k: puff(cv, 1, 96, 24))}
    v = W / (c.n * c.ms / 1000)
    ems = [Emitter("banks", "fog_puff", amount=18, lifetime=14.0, velocity=(v, 0), spread=2,
                   scale=(1.5, 3.0), sway=0.05, area="fill", depth=0.3, fade=(0.3, 0.35), alpha=0.7),
           Emitter("wisps", "fog_wisp", amount=10, lifetime=10.0, velocity=(v * 1.6, 0), spread=2,
                   scale=(1.2, 2.2), area="fill", depth=0.8, fade=(0.3, 0.35), alpha=0.5)]
    return dict(draw=draw, look=look, cover=0.5, particles=parts, emitters=ems,
                meta={"fog": True, "fog_density": 0.035 * thick})


def _god_rays(c: _Ctx):
    """Slanted shafts of light from the top that breathe (a whole number of pulses per loop),
    with dust motes drifting through them, catching the light only inside a shaft."""
    W, H, rng, s = c.W, c.H, c.rng, c.k
    slant = float(c.opts.get("slant", 0.42))
    shafts = []
    x = 0.0
    while x < W:
        wd = rng.uniform(4, 12) * s
        shafts.append(dict(x=x + wd / 2, w=wd, heat=rng.uniform(0.45, 0.95), L=H * rng.uniform(0.55, 1.0),
                           m=int(rng.integers(1, 3)), ph=rng.uniform(0, 2 * math.pi)))
        x += wd + rng.uniform(14, 46) * s
    motes = [dict(x=rng.uniform(0, W), y=rng.uniform(0, H), ax=rng.uniform(2, 6), ay=rng.uniform(2, 5),
                  m=int(rng.integers(1, 3)), ph=rng.uniform(0, 6.3)) for _ in range(int(26 * W * H / 27648))]

    def shaft_heat(X, Y, p):
        out = np.full(np.broadcast(X, Y).shape, -1.0, np.float32)
        for d in shafts:
            u = (X - slant * Y - d["x"]) % W
            u = np.where(u > W / 2, u - W, u)
            half = d["w"] / 2 * (1 + 0.5 * Y / H)
            pulse = 0.78 + 0.22 * math.sin(2 * math.pi * d["m"] * p + d["ph"])
            fall = np.clip(1 - Y / d["L"], 0, 1) ** 1.3
            core = 1 - (np.abs(u) / np.maximum(half, 1e-3)) ** 2
            h = d["heat"] * pulse * fall * (0.55 + 0.45 * core)
            out = np.where((np.abs(u) <= half) & (fall > 0.02), np.maximum(out, h), out)
        return out

    def draw(cv, k, p):
        def fn(X, Y):
            h = shaft_heat(X, Y, p)
            return h > 0.06, h
        cv.field("main", (0, 0, W, H), fn)
        for m in motes:
            x = m["x"] + m["ax"] * math.sin(2 * math.pi * m["m"] * p + m["ph"])
            y = m["y"] + m["ay"] * math.cos(2 * math.pi * p + m["ph"])
            x, y = x % W, min(max(y, 0.5), H - 0.5)
            lit = float(shaft_heat(np.float32(x), np.float32(y), p))
            if lit > 0.15:
                cv.ellipse("accent", x, y, 0.55, heat=0.5 + 0.5 * lit, snap=True)

    look = {"main": dict(keys="a07448 d4a466 f0cc8c fce8bc fffaec", n=5, a=(12, 92), e=(0.1, 0.5),
                         tint=0.8),
            "accent": dict(keys="ffe8b0 fff8e8 ffffff", n=3, a=(200, 255), e=(0.6, 1.0), tint=0.5)}

    def ray(cv, k, w=24, h=96):
        def fn(X, Y):
            u = (X - w / 2) / (w / 2)
            v = Y / h
            hh = (1 - u * u) * (1 - v) ** 1.2
            return (np.abs(u) < 1) & (hh > 0.05), hh
        cv.field("main", (0, 0, w, h), fn)

    parts = {"ray": (24, 96, 1, ray),
             "mote": (3, 3, 1, lambda cv, k: cv.star("accent", 1.5, 1.5, 1.4, heat=0.9, thin=0.9))}
    ems = [Emitter("shafts", "ray", amount=5, lifetime=6.0, velocity=(0, 0), area="fill", depth=0.3,
                   blend="add", glow=True, scale=(1.0, 2.5), fade=(0.4, 0.4), alpha=0.3),
           Emitter("motes", "mote", amount=40, lifetime=5.0, velocity=(4, -2), spread=180, sway=0.3,
                   area="fill", depth=0.6, blend="add", glow=True, fade=(0.3, 0.3))]
    return dict(draw=draw, look=look, cover=0.45, particles=parts, emitters=ems,
                meta={"wrap": "x", "volumetric": 0.03})


def _leaves(c: _Ctx, petals: bool = False):
    """Leaves (autumn reds, oranges, yellows) or cherry petals tumbling down: each one turns a
    whole number of times and flips over (narrow, then its darker underside) as it sways."""
    W, H, rng, s = c.W, c.H, c.rng, c.k
    heavy = float(c.opts.get("intensity", 1.0))
    wind = float(c.opts.get("wind", 0.5))
    items = []
    for _ in range(max(2, int((30 if petals else 16) * heavy * W * H / 25600))):
        depth = rng.random()
        items.append(dict(x=rng.uniform(0, W), y=rng.uniform(0, H), ky=1 if depth < 0.6 else 2,
                          kx=int(round(wind * (1 if depth < 0.6 else 2))),
                          amp=rng.uniform(3, 9) * s, m=int(rng.integers(1, 3)), ph=rng.uniform(0, 6.3),
                          rot=rng.uniform(0, 6.3), r=int(rng.choice([-1, 1])), f=int(rng.integers(1, 4)),
                          L=(rng.uniform(4.5, 6.0) if petals else rng.uniform(7.0, 10.0)) * s * (0.65 + 0.5 * depth),
                          mat="main" if rng.random() < 0.6 else "accent", far=depth < 0.25))

    def leaf(cv, x, y, L, ang, flip, mat):
        wd = L * (0.62 if petals else 0.5) * max(0.22, abs(flip))
        heat = 0.85 if flip > 0 else 0.45
        cv.poly(mat, _lens(x, y, L, wd, ang, 12), heat)
        if not petals and abs(flip) > 0.45 and L > 4:              # the vein: one shade darker
            c_, s_ = math.cos(ang), math.sin(ang)
            cv.stroke(mat, [(x - c_ * L * 0.45, y - s_ * L * 0.45), (x + c_ * L * 0.3, y + s_ * L * 0.3)],
                      0.3, heat - 0.3, mode="over")

    def draw(cv, k, p):
        for d in items:
            x = d["x"] + d["kx"] * W * p + d["amp"] * math.sin(2 * math.pi * d["m"] * p + d["ph"])
            y = d["y"] + d["ky"] * H * p
            ang = d["rot"] + d["r"] * 2 * math.pi * p
            flip = math.cos(2 * math.pi * d["f"] * p + d["ph"])
            mat = "back" if d["far"] else d["mat"]
            for px, py in _wraps(x, y, d["L"] + 1, W, H):
                leaf(cv, px, py, d["L"], ang, flip, mat)

    if petals:
        look = {"back": dict(keys="8a5a8a b484b0 d8b0d4", n=3, a=(150, 200)),
                "main": dict(keys="a04a7a d06a9a f096b8 ffc4d8 fff0f4", n=5, a=(255, 255)),
                "accent": dict(keys="b0587e e888a8 ffd0e0 ffffff", n=4, a=(255, 255))}
    else:
        look = {"back": dict(keys="5a3048 7e4a48 a06a50", n=3, a=(170, 210)),
                "main": dict(keys="5a1a2a 8e2a24 c44a1c ec7a22 ffb040", n=5, a=(255, 255)),
                "accent": dict(keys="6a4a1a a07a1c d4b02a f4e05a fff6a8", n=5, a=(255, 255))}
    name = "petal" if petals else "leaf"
    L0 = (5.5 if petals else 8.5) * s
    sz = int(math.ceil(L0 + 3))

    def part(cv, k, mat="main"):
        p = k / 8
        leaf(cv, sz / 2, sz / 2, L0, 2 * math.pi * p, math.cos(2 * math.pi * p), mat)
    v = H / (c.n * c.ms / 1000)
    parts = {name: (sz, sz, 8, part), f"{name}_b": (sz, sz, 8, lambda cv, k: part(cv, k, "accent"))}
    ems = [Emitter("far", name, frames=8, amount=int(12 * heavy), lifetime=6, velocity=(wind * v * 0.7, v * 0.7),
                   spread=20, sway=0.5, anim="loop", anim_fps=8, area="sky", depth=0.2, scale=(0.6, 0.8), alpha=0.75),
           Emitter("near", name, frames=8, amount=int(10 * heavy), lifetime=4, velocity=(wind * v * 1.4, v * 1.4),
                   spread=20, sway=0.6, anim="loop", anim_fps=10, area="sky", depth=0.8),
           Emitter("near_b", f"{name}_b", frames=8, amount=int(8 * heavy), lifetime=4,
                   velocity=(wind * v * 1.2, v * 1.2), spread=25, sway=0.6, anim="loop", anim_fps=9,
                   area="sky", depth=0.7)]
    return dict(draw=draw, look=look, cover=0.45, particles=parts, emitters=ems, meta={})


def _embers(c: _Ctx):
    """Embers rising off a fire below: flickering a whole number of times, swaying, short
    hot trails; a few big ones; all glow."""
    W, H, rng, s = c.W, c.H, c.rng, c.k
    heavy = float(c.opts.get("intensity", 1.0))
    items = [dict(x=rng.uniform(0, W), y=rng.uniform(0, H), ky=int(rng.integers(1, 3)),
                  amp=rng.uniform(2, 7) * s, m=int(rng.integers(1, 3)), ph=rng.uniform(0, 6.3),
                  fl=int(rng.integers(2, 6)), big=rng.random() < 0.15)
             for _ in range(max(3, int(26 * heavy * W * H / 16384)))]

    def draw(cv, k, p):
        dp = 1.0 / c.n
        for d in items:
            def pos(q):
                return (d["x"] + d["amp"] * math.sin(2 * math.pi * d["m"] * q + d["ph"]),
                        d["y"] - d["ky"] * H * q)
            x, y = pos(p)
            x0, y0 = pos(p - dp * 0.7)
            heat = 0.6 + 0.4 * math.sin(2 * math.pi * d["fl"] * p + d["ph"])
            for (px, py), (qx, qy) in zip(_wraps(x, y, 8, W, H), _wraps(x0, y0, 8, W, H)):
                if abs(px - qx) > W / 2 or abs(py - qy) > H / 2:
                    continue
                cv.stroke("main", [(qx, qy), (px, py)], [0.35, 0.6 if not d["big"] else 1.0],
                          [heat * 0.45, heat])
                if d["big"]:
                    cv.ellipse("accent", px, py, 0.6, heat=heat, snap=True)

    look = {"main": dict(keys="5a1420 a0281c dc5418 fc9424 ffd04a fff4b0", n=6, a=(160, 255),
                         e=(0.3, 1.0), tint=0.25),
            "accent": dict(keys="ffe070 fffbe0 ffffff", n=3, a=(255, 255), e=(1.0, 1.0), tint=0.1)}

    def ember(cv, k):
        h = (0.55, 0.85, 1.0, 0.75)[k]
        cv.ellipse("main", 1.5, 1.5, 1.1, heat=lambda rho, X, Y: h * (1 - 0.4 * rho))
    v = H / (c.n * c.ms / 1000)
    parts = {"ember": (3, 3, 4, ember),
             "spark": (1, 4, 1, lambda cv, k: cv.stroke("main", [(0.5, 3.8), (0.5, 0.2)], [0.3, 0.5], [0.4, 1.0]))}
    ems = [Emitter("embers", "ember", frames=4, amount=int(40 * heavy), lifetime=3.0, velocity=(0, -v * 1.5),
                   spread=20, sway=0.45, anim="loop", anim_fps=12, area="below", blend="add", glow=True,
                   depth=0.6, fade=(0.05, 0.4)),
           Emitter("sparks", "spark", amount=int(24 * heavy), lifetime=1.6, velocity=(0, -v * 2.5), spread=25,
                   sway=0.2, area="below", blend="add", glow=True, align=True, depth=0.8, fade=(0.0, 0.5))]
    return dict(draw=draw, look=look, cover=0.4, particles=parts, emitters=ems, meta={"light": [1.0, 0.55, 0.25]})


def _fireflies(c: _Ctx):
    """Fireflies wandering on looping paths, each blinking on and off (a whole number of
    blinks per loop): a bright core and a dim halo when lit."""
    W, H, rng, s = c.W, c.H, c.rng, c.k
    items = [dict(x=rng.uniform(0, W), y=rng.uniform(0, H), ax=rng.uniform(4, 14) * s, ay=rng.uniform(3, 9) * s,
                  a=int(rng.integers(1, 3)), b=int(rng.integers(1, 3)), p1=rng.uniform(0, 6.3),
                  p2=rng.uniform(0, 6.3), bl=int(rng.integers(1, 4)), p3=rng.uniform(0, 6.3))
             for _ in range(max(3, int(int(c.opts.get("count", 18)) * W * H / 25600)))]

    def bug(cv, x, y, on):
        if on <= 0.05:
            cv.ellipse("main", x, y, 0.55, heat=0.25, snap=True)
            return
        cv.ellipse("back", x, y, 1.5 + 3.0 * on * s, heat=lambda rho, X, Y: on * (0.95 - 0.6 * rho))
        cv.ellipse("main", x, y, 0.7 + 0.7 * on, heat=0.55 + 0.45 * on, snap=True)
        if on > 0.7:
            cv.star("accent", x, y, 1.6 * s, heat=1.0, thin=0.6)

    def draw(cv, k, p):
        for d in items:
            x = d["x"] + d["ax"] * math.sin(2 * math.pi * d["a"] * p + d["p1"])
            y = d["y"] + d["ay"] * math.sin(2 * math.pi * d["b"] * p + d["p2"])
            on = max(0.0, math.sin(2 * math.pi * d["bl"] * p + d["p3"])) ** 2
            for px, py in _wraps(x, y, 5 * s, W, H):
                bug(cv, px, py, on)

    look = {"back": dict(keys="2e5a2a 4a8a30 7ab838", n=3, a=(60, 150), e=(0.3, 0.6), tint=0.2),
            "main": dict(keys="3a5a24 7aa02a bce03a ecff7a", n=4, a=(180, 255), e=(0.2, 1.0), tint=0.15),
            "accent": dict(keys="f4ffb0 ffffff", n=2, a=(255, 255), e=(1.0, 1.0), tint=0.0)}

    def part(cv, k):                     # engines bloom it: no baked halo, just the lit body
        on = max(0.0, math.sin(math.pi * k / 5)) ** 1.5
        cv.ellipse("main", 2.5, 2.5, 0.7 + 0.8 * on, heat=0.4 + 0.6 * on, snap=True)
        if on > 0.7:
            cv.star("accent", 2.5, 2.5, 1.7, heat=1.0, thin=0.6)
    parts = {"firefly": (5, 5, 6, part)}
    ems = [Emitter("fireflies", "firefly", frames=6, amount=int(c.opts.get("count", 12)) * 2, lifetime=4.0,
                   velocity=(6, -3), spread=180, speed_var=0.6, sway=0.6, anim="loop", anim_fps=4, area="fill",
                   blend="add", glow=True, depth=0.6, fade=(0.2, 0.2))]
    return dict(draw=draw, look=look, cover=0.4, particles=parts, emitters=ems,
                meta={"light": [0.7, 1.0, 0.4]})


def _dust(c: _Ctx):
    """Dust motes hanging in the air: tiny warm specks drifting slowly, a few twinkling."""
    W, H, rng, s = c.W, c.H, c.rng, c.k
    items = [dict(x=rng.uniform(0, W), y=rng.uniform(0, H), kx=int(rng.choice([0, 1])), ax=rng.uniform(1, 4),
                  ay=rng.uniform(1, 4), m=int(rng.integers(1, 3)), ph=rng.uniform(0, 6.3),
                  tw=int(rng.integers(1, 4)), big=rng.random() < 0.2)
             for _ in range(max(4, int(40 * float(c.opts.get("intensity", 1.0)) * W * H / 16384)))]

    def draw(cv, k, p):
        for d in items:
            x = d["x"] + d["kx"] * W * p + d["ax"] * math.sin(2 * math.pi * d["m"] * p + d["ph"])
            y = d["y"] + d["ay"] * math.cos(2 * math.pi * d["m"] * p + d["ph"] * 1.3)
            tw = 0.5 + 0.5 * math.sin(2 * math.pi * d["tw"] * p + d["ph"])
            for px, py in _wraps(x, y, 3, W, H):
                if d["big"] and tw > 0.8:
                    cv.star("accent", px, py, 1.6 * s, heat=0.9, thin=0.7)
                else:
                    cv.ellipse("main", px, py, 0.55 if not d["big"] else 0.9, heat=0.4 + 0.5 * tw, snap=True)

    look = {"main": dict(keys="7a6a5a a89070 d0b890 f0e0c0", n=4, a=(90, 200), e=(0.0, 0.3), tint=0.6),
            "accent": dict(keys="fff0c8 ffffff", n=2, a=(220, 255), e=(0.6, 0.9), tint=0.4)}
    parts = {"mote": (2, 2, 1, lambda cv, k: cv.ellipse("main", 1, 1, 0.9, heat=0.8)),
             "glint": (3, 3, 1, lambda cv, k: cv.star("accent", 1.5, 1.5, 1.4, heat=0.9, thin=0.8))}
    ems = [Emitter("motes", "mote", amount=60, lifetime=6.0, velocity=(5, -1), spread=180, speed_var=0.8,
                   sway=0.35, area="fill", depth=0.5, fade=(0.3, 0.3), alpha=0.8),
           Emitter("glints", "glint", amount=10, lifetime=3.0, velocity=(3, 0), spread=180, sway=0.3,
                   area="fill", depth=0.7, blend="add", glow=True, fade=(0.4, 0.4))]
    return dict(draw=draw, look=look, cover=0.4, particles=parts, emitters=ems, meta={})


def _sandstorm(c: _Ctx):
    """Sand blown sideways: hazy bands of dust racing across (two noise layers, 2 and 3 tiles
    per loop), fast grain streaks, the odd tumbling pebble."""
    W, H, rng, s = c.W, c.H, c.rng, c.k
    heavy = float(c.opts.get("intensity", 1.0))
    na, nb = c.noise(3, 3), c.noise(5, 2)
    grains = [dict(x=rng.uniform(0, W), y=rng.uniform(0, H), kx=int(rng.integers(3, 6)), ky=int(rng.choice([0, 0, 1])),
                   ln=rng.uniform(5, 13) * s, near=rng.random() < 0.3, ph=rng.uniform(0, 6.3))
              for _ in range(max(4, int(48 * heavy * W * H / 24576)))]
    pebbles = [dict(x=rng.uniform(0, W), y=rng.uniform(0, H), ph=rng.uniform(0, 6.3))
               for _ in range(max(1, int(3 * heavy * W * H / 24576)))]

    def draw(cv, k, p):
        def haze(X, Y):
            a = na(X / W - 2 * p, Y / H * 2)
            b = nb(X / W - 3 * p + 0.4, Y / H * 2 + 0.2)
            v = np.clip(((0.6 * a + 0.4 * b) - 0.5 + 0.1 * heavy) * 2.4, 0, 1)
            return v > 0.05, v
        cv.field("smoke", (0, 0, W, H), haze)
        for d in grains:
            x = d["x"] + d["kx"] * W * p
            y = d["y"] + d["ky"] * H * p + 1.5 * math.sin(2 * math.pi * p * 2 + d["ph"])
            _streak(cv, "main" if d["near"] else "back", x, y, d["kx"] * W, d["ky"] * H, d["ln"],
                    0.55 if d["near"] else 0.45, 0.9 if d["near"] else 0.6, W=W, H=H)
        for d in pebbles:
            x = d["x"] + 6 * W * p
            y = d["y"] + 3 * math.sin(2 * math.pi * 3 * p + d["ph"])
            for px, py in _wraps(x, y, 3, W, H):
                cv.ellipse("accent", px, py, 1.1 * s, 0.8 * s, heat=0.7, rot=2 * math.pi * 4 * p)

    look = {"smoke": dict(keys="b08860 c8a074 dcbc90 f0dab0", n=4, a=(26, 120), tint=0.7),
            "back": dict(keys="a07a50 c8a070 e4c898", n=3, a=(110, 180), tint=0.7),
            "main": dict(keys="b0804c d8a868 f4d494 fff0c8", n=4, a=(190, 240), tint=0.7),
            "accent": dict(keys="4a3024 7a5034 a87448", n=3, a=(255, 255), tint=0.7)}
    v = 4 * W / (c.n * c.ms / 1000)
    parts = {"grain": (2, 10, 1, lambda cv, k: cv.stroke("main", [(1.0, 0.3), (1.0, 9.7)], [0.3, 0.55], [0.4, 0.9])),
             "haze": (64, 24, 1, lambda cv, k: cv.ellipse("smoke", 32, 12, 31, 11,
                                                          heat=lambda rho, X, Y: 0.85 * (1 - rho))),
             "pebble": (3, 3, 1, lambda cv, k: cv.ellipse("accent", 1.5, 1.5, 1.2, heat=0.7))}
    ems = [Emitter("haze", "haze", amount=24, lifetime=3.0, velocity=(v * 0.6, 0), spread=3, scale=(1.5, 3.5),
                   area="side", depth=0.3, fade=(0.25, 0.3), alpha=0.6),
           Emitter("grains", "grain", amount=int(120 * heavy), lifetime=1.2, velocity=(v, 8), spread=4,
                   area="side", align=True, depth=0.6, fade=(0.05, 0.1)),
           Emitter("pebbles", "pebble", amount=int(6 * heavy), lifetime=1.0, velocity=(v * 1.3, -10), spread=8,
                   gravity=(0, 60), spin=360, area="side", depth=0.9)]
    return dict(draw=draw, look=look, cover=0.45, particles=parts, emitters=ems,
                meta={"fog": True, "fog_density": 0.03, "fog_colour": [0.78, 0.6, 0.4]})


LIGHTNING_MS = (50, 40, 40, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 200)
LIGHTNING_FLASH = (0.08, 0.25, 0.9, 0.15, 0.75, 0.5, 0.3, 0.18, 0.1, 0.06, 0.03, 0.0, 0.0, 0.0)


def _lightning(c: _Ctx):
    """A strike, not a loop: a faint leader, the bolt with branches and a ground flash, a dark
    blink, the restrike, then it fades as the branches die. meta flash = how bright the screen
    flashes each frame (the overlay and the HD-2D sky light use it)."""
    W, H, rng = c.W, c.H, c.rng
    x0 = W * rng.uniform(0.4, 0.6)
    main = bolt_points(rng, x0, -2, W * rng.uniform(0.35, 0.65), H - 6, depth=6, spread=0.18)
    branches = []
    for _ in range(4):
        i = int(rng.integers(len(main) // 6, len(main) * 3 // 4))
        bx, by = main[i]
        ang = rng.uniform(0.35, 1.0) * rng.choice([-1, 1])
        L = H * rng.uniform(0.18, 0.35)
        branches.append(bolt_points(rng, bx, by, bx + math.sin(ang) * L, by + math.cos(ang) * L, depth=4,
                                    spread=0.25))

    def draw(cv, k, p):
        if k == 0:
            draw_bolt(cv, main, 0.8, heat=0.55, part=0.35, glow=False)
            return
        if k == 1:
            draw_bolt(cv, main, 1.0, heat=0.75, part=0.75, glow=False)
            return
        if k == 3:                                        # the dark blink between strikes
            draw_bolt(cv, main, 0.6, heat=0.35, glow=False)
            return
        fade = {2: 1.0, 4: 0.95, 5: 0.8, 6: 0.65, 7: 0.5, 8: 0.38, 9: 0.28, 10: 0.2}.get(k, 0.0)
        if fade <= 0:
            if k < 13:
                cv.ellipse("smoke", main[-1][0], H - 4, 6 - k * 0.3, 2, heat=0.5)
            return
        draw_bolt(cv, main, 1.6 * (0.6 + 0.4 * fade), heat=fade)
        for b in branches[: 4 if fade > 0.6 else 2]:
            draw_bolt(cv, b, 0.9 * fade + 0.3, heat=fade * 0.8, glow=fade > 0.5)
        gx = main[-1][0]
        cv.ellipse("back", gx, H - 4, W * 0.32 * fade + 3, 4 * fade + 1.5,
                   heat=lambda rho, X, Y: 0.25 + 0.6 * fade * (1 - rho))
        if fade > 0.6:
            cv.star("accent", gx, H - 6, 6 * fade, heat=1.0, stretch=1.8)

    from .fxcore import spell_ramps
    ramps = spell_ramps("lightning")
    return dict(draw=draw, ramps=ramps, particles={}, emitters=[], durations=list(LIGHTNING_MS),
                meta={"wrap": "none", "flash": list(LIGHTNING_FLASH), "light": [0.7, 0.8, 1.0],
                      "oneshot": True})


def _cloud_shadows(c: _Ctx):
    """Soft-edged cloud shadows (3 crisp alpha steps) drifting across the ground, a whole tile
    per loop; for top-down and HD-2D maps (the 3D board projects them as a decal)."""
    W, H = c.W, c.H
    na = c.noise(3, 4)
    cover_amt = float(c.opts.get("intensity", 1.0))
    cvs = c.canvas()

    def fn(X, Y):
        d = na(X / W, Y / H)
        v = np.clip((d - 0.6 + 0.08 * cover_amt) * 4.0, 0, 1)
        return v > 0.04, v
    cvs.field("smoke", (0, 0, W, H), fn)
    look = {"smoke": dict(keys="141030 1c1840 241e4c", n=3, a=(72, 150), tint=0.5)}
    base, _ = wfinish(cvs, build_look(look, c.opts.get("_tod"), c.opts.get("_palette")), 0.5)

    def frame(k, p):
        return np.roll(base, int(round(W * p)), axis=1)
    parts = {"cloud_shadow": (W, H, 1, None)}
    return dict(frame=frame, base=base, look=look, particles=parts, emitters=[],
                meta={"scroll": [W / (c.n * c.ms / 1000), 0.0], "shadow": True})


# kind: (label, (w, h) tile at scale 1, frames, ms per frame, blend, maker)
KINDS = {
    "rain":          ("Rain (streaks, splashes, ripples)", (128, 128), 16, 40, "mix", _rain),
    "snow":          ("Snow (flakes by depth)", (128, 128), 24, 70, "mix", _snow),
    "fog":           ("Fog / mist banks", (192, 128), 32, 110, "mix", _fog),
    "god_rays":      ("God rays / light shafts", (192, 144), 24, 90, "add", _god_rays),
    "leaves":        ("Drifting autumn leaves", (160, 160), 24, 80, "mix", _leaves),
    "petals":        ("Drifting petals", (160, 160), 24, 80, "mix", lambda c: _leaves(c, True)),
    "embers":        ("Embers rising", (128, 128), 16, 70, "add", _embers),
    "fireflies":     ("Fireflies", (160, 160), 32, 90, "add", _fireflies),
    "dust":          ("Dust motes", (128, 128), 24, 110, "add", _dust),
    "sandstorm":     ("Sandstorm", (192, 128), 16, 50, "mix", _sandstorm),
    "lightning":     ("Lightning strike + flash", (96, 160), 14, 60, "add", _lightning),
    "cloud_shadows": ("Cloud shadows", (192, 192), 24, 160, "mix", _cloud_shadows),
}
WEATHER_KINDS = tuple(KINDS)


def weather_label(kind: str) -> str:
    return KINDS[kind][0]


def _scale(style, scale) -> float:
    if scale:
        return float(scale)
    ph = getattr(style, "person_height", 64) if style is not None else 64
    return float(min(1.5, max(0.5, ph / 64)))


def render_weather(kind: str, *, tod: str | None = None, intensity: float = 1.0, wind: float | None = None,
                   seed: int = 0, scale: float | None = None, frames: int | None = None,
                   colours=None, style=None, palette=None, ground: bool | None = None,
                   **opts) -> Weather:
    """Draw a weather / atmosphere asset in memory (see the module doc). tod: dawn | noon |
    sunset | night | None (neutral colours). intensity: 0.3 (light) .. 2 (heavy). wind: -1..1
    (rain, snow, leaves). scale: pixel size of drops / flakes (default from the project's
    person height). ground: rain splashes + puddle rings (default: off for side-scrollers).
    palette / style: snap to the project's palette (default: the style's fixed one)."""
    from .effects import Effect, project_palette
    if kind not in KINDS:
        raise ValueError(f"Unknown weather '{kind}' (use {', '.join(KINDS)})")
    if tod is not None and tod not in TODS:
        raise ValueError(f"Unknown time of day '{tod}' (use {', '.join(TODS)})")
    label, (w0, h0), n0, ms, blend, maker = KINDS[kind]
    k = _scale(style, scale)
    W, H = max(16, int(round(w0 * min(k, 1.25)))), max(16, int(round(h0 * min(k, 1.25))))
    W -= W % 2
    H -= H % 2
    n = int(frames) if frames and kind != "lightning" else n0
    if not 4 <= n <= 64:
        raise ValueError("frames must be between 4 and 64")
    pal = palette if palette is not None else project_palette(style)
    if ground is None:
        ground = getattr(style, "tile_style", "") != "side" if style is not None else True
    o = dict(opts, intensity=float(intensity), ground=bool(ground), _tod=tod, _palette=pal)
    if wind is not None:
        o["wind"] = float(wind)
    rng = np.random.default_rng(seed)
    c = _Ctx(kind, W, H, n, rng, o, k)
    c.ms = ms
    if kind in ("fog", "god_rays", "sandstorm"):     # smooth fields: 2x supersampling is plenty
        c.ss = 2
    spec = maker(c)
    outs, glows = [], []
    if "ramps" in spec:                                   # lightning: the spell engine's ramps
        ramps = spec["ramps"]
        if pal is not None:
            from .effects import snap_ramp
            ramps = {m: snap_ramp(r, pal) for m, r in ramps.items()}
        for i in range(n):
            cv = c.canvas()
            spec["draw"](cv, i, i / n)
            f, g, _ = finish(cv, ramps, "none")
            outs.append(f)
            glows.append(g)
        look = None
    elif "frame" in spec:                                 # cloud shadows: one picture, scrolled
        for i in range(n):
            outs.append(spec["frame"](i, i / n))
            glows.append(np.zeros_like(outs[-1]))
        look = build_look(spec["look"], tod, pal, colours)
    else:
        look = build_look(spec["look"], tod, pal, colours)
        for i in range(n):
            cv = c.canvas()
            spec["draw"](cv, i, i / n)
            f, g = wfinish(cv, look, spec.get("cover", 0.45))
            outs.append(f)
            glows.append(g)
    # particle textures
    parts, pglow = {}, {}
    for name, (pw, ph, pn, pdraw) in spec.get("particles", {}).items():
        if pdraw is None:
            parts[name] = [spec["base"]]
            continue
        fr, gl = _sprites(pw, ph, pn, pdraw, look, 0.4)
        parts[name] = fr
        if any(g[..., 3].any() for g in gl):
            pglow[name] = gl
    has_glow = any(g[..., 3].any() for g in glows)
    stack = np.concatenate([f[f[..., 3] > 0][:, :3] for f in outs if f[..., 3].any()] or
                           [np.zeros((1, 3), np.uint8)])
    used = np.unique(stack, axis=0)
    durs = spec.get("durations") or [ms] * n
    loop = not spec.get("meta", {}).get("oneshot", False)
    main = look["main"][0] if look and "main" in look else (look["smoke"][0] if look else spec["ramps"]["main"])
    fx = Effect(kind, outs, used, durs[0], loop, tod, main)
    fx.durations, fx.glow, fx.blend = list(durs), (glows if has_glow else None), blend
    fx.tags = [(kind if kind != "lightning" else "strike", 0, n - 1, loop)]
    meta = {"label": label, "wrap": "xy", "tile": [W, H], "loop_ms": int(sum(durs)), "intensity": float(intensity),
            "seed": seed, "scale": k, "screen": [480, 270]}
    meta.update(spec.get("meta", {}))
    if "scroll" not in meta:
        meta["scroll"] = [0.0, 0.0]
    return Weather(kind, tod, fx, parts, pglow, spec.get("emitters", []), meta)


# =========================================================================== files

def _emitter_json(e: Emitter, ppu: float) -> dict:
    d = asdict(e)
    d["velocity"], d["gravity"] = list(e.velocity), list(e.gravity)
    d["scale"], d["fade"] = list(e.scale), list(e.fade)
    vx, vy = e.velocity
    gx, gy = e.gravity
    d["units"] = {"ppu": ppu, "velocity": [round(vx / ppu, 4), round(-vy / ppu, 4)],
                  "gravity": [round(gx / ppu, 4), round(-gy / ppu, 4)],
                  "note": "Unity units / metres, y up (2D y down px / ppu, flipped)"}
    return d


def write_weather(w: Weather, folder, name: str = "sheet", ppu: float = 42.0) -> list[Path]:
    """Everything for one weather asset into folder (see the module doc)."""
    from .effects import write_effect
    folder = Path(folder)
    written = write_effect(w.fx, folder, name)
    pdir = folder / "particles"
    pdir.mkdir(exist_ok=True)
    pinfo = {}
    for pname, frames in w.particles.items():
        p = pdir / f"{pname}.png"
        Image.fromarray(_strip(frames), "RGBA").save(p)
        written.append(p)
        pinfo[pname] = {"file": f"particles/{p.name}", "frames": len(frames),
                        "size": [int(frames[0].shape[1]), int(frames[0].shape[0])]}
        if pname in w.particle_glow:
            g = pdir / f"{pname}_glow.png"
            Image.fromarray(_strip(w.particle_glow[pname]), "RGBA").save(g)
            written.append(g)
            pinfo[pname]["glow"] = f"particles/{g.name}"
    info = {"kind": w.kind, "tod": w.tod, "sheet": f"{name}.png", "frames": len(w.fx.frames),
            "frame_ms": w.fx.frame_durations(), "loop": w.fx.loop, "blend": w.fx.blend,
            "glow": f"{name}_glow.png" if w.fx.glow is not None else None, "meta": w.meta,
            "particles": pinfo, "emitters": [_emitter_json(e, ppu) for e in w.emitters]}
    js = folder / "weather.json"
    js.write_text(json.dumps(info, indent=1), encoding="utf-8")
    written.append(js)
    from .weather_godot import write_godot_weather
    written += write_godot_weather(w, folder, name, ppu=ppu)
    return written


def make_weather(kind: str, name: str | None = None, *, project=None, tod: str | None = None,
                 intensity: float = 1.0, wind: float | None = None, seed: int = 0,
                 colours=None, tags=None, **opts):
    """Make a weather asset and save it in the project (kind "weather"). Returns the Asset."""
    from .assets import current_project, new_asset, set_tags
    register_kind()
    proj = project or current_project()
    st = proj.style
    w = render_weather(kind, tod=tod, intensity=intensity, wind=wind, seed=seed, colours=colours,
                       style=st, **opts)
    label = name or (weather_label(kind).split(" (")[0] + (f" ({tod})" if tod else ""))
    asset = new_asset(KIND, label, proj, weather=kind, tod=tod, intensity=float(intensity),
                      wind=wind, seed=seed, size=list(w.size), frames=len(w.fx.frames),
                      durations=w.fx.frame_durations(), loop=w.fx.loop, blend=w.fx.blend,
                      colours=list(colours) if isinstance(colours, (list, tuple)) else colours,
                      meta=w.meta, options={k: v for k, v in opts.items() if not k.startswith("_")})
    files = write_weather(w, asset.folder, ppu=st.ppu)
    asset.extra["files"] = [p.relative_to(asset.folder).as_posix() for p in files]
    asset.save()
    set_tags(asset, sorted({"weather", kind, *([tod] if tod else []), *(tags or [])}))
    return asset


def overlay_frame(w: Weather, i: int, screen=(240, 135), offset=(0, 0)) -> np.ndarray:
    """Frame i tiled over a screen (what the CanvasLayer overlay shows), RGBA."""
    f = w.fx.frames[i % len(w.fx.frames)]
    h, wd = f.shape[:2]
    sw, sh = screen
    wrap = w.meta.get("wrap", "xy")
    ox, oy = int(offset[0]), int(offset[1])
    if wrap == "none":
        out = np.zeros((sh, sw, 4), np.uint8)
        x0, y0 = (sw - wd) // 2 + ox, max(0, sh - h) + oy
        xs, ys = max(0, x0), max(0, y0)
        xe, ye = min(sw, x0 + wd), min(sh, y0 + h)
        if xe > xs and ye > ys:
            out[ys:ye, xs:xe] = f[ys - y0:ye - y0, xs - x0:xe - x0]
        return out
    reps = (-(-sh // h) + 1, -(-sw // wd) + 1, 1)
    big = np.tile(f, reps)
    big = np.roll(big, (oy % h, ox % wd), axis=(0, 1))
    out = big[:sh, :sw].copy()
    if wrap == "x":
        out[h:] = 0
    return out


def preview_over(w: Weather, background: np.ndarray, i: int, glow: bool = True, offset=(0, 0)) -> np.ndarray:
    """Frame i over a background picture (RGB/RGBA), alpha-blended (additive for add-blend
    kinds), glow bloomed like an engine; lightning also flashes the picture. -> RGB uint8."""
    import cv2
    bg = background[..., :3].astype(np.float32)
    sh, sw = bg.shape[:2]
    scroll = w.meta.get("scroll", [0, 0])
    loop_s = sum(w.fx.frame_durations()) / 1000
    t = sum(w.fx.frame_durations()[:i % len(w.fx.frames)]) / 1000
    off = (offset[0] + scroll[0] * t if w.kind != "cloud_shadows" else offset[0], offset[1] + scroll[1] * t)
    del loop_s
    ov = overlay_frame(w, i, (sw, sh), off).astype(np.float32)
    a = ov[..., 3:4] / 255
    if w.kind == "lightning":
        fl = w.meta["flash"][i % len(w.meta["flash"])]
        bg = bg * (1 - fl * 0.55) + np.array([200, 215, 255], np.float32) * fl * 0.55
    if w.fx.blend == "add":
        out = bg + ov[..., :3] * a
    else:
        out = bg * (1 - a) + ov[..., :3] * a
    if glow and w.fx.glow is not None:
        gw = Weather(w.kind, w.tod, type(w.fx)(w.kind, w.fx.glow, w.fx.palette, w.fx.frame_ms, w.fx.loop, None),
                     {}, {}, [], w.meta)
        g = overlay_frame(gw, i, (sw, sh), off).astype(np.float32)
        g = g[..., :3] * (g[..., 3:4] > 0)
        r = max(1.0, max(sh, sw) / 160)
        out = out + cv2.GaussianBlur(g, (0, 0), r) * 0.8 + cv2.GaussianBlur(g, (0, 0), r * 3) * 0.6
    return np.clip(out, 0, 255).astype(np.uint8)

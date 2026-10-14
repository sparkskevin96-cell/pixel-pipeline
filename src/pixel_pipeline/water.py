"""Animated water tiles: seamless looping water with specular glints and shoreline foam, for
top-down, side-scroller and isometric maps (CPU procedural, neo 16-bit look).

    wt = render_water(tile=32, theme="ocean", tod="sunset")
    wt.frames        # the open-water tile, n frames: tiles both ways AND loops in time
    wt.shore         # {mask: [frames]}: the 47 blob shore tiles (tilegen's geometry), foam that
                     # washes in and out, wet sand where the tide was
    wt.side_surface / wt.side_body   # side view: wave crest + body (body tiles downward)
    wt.iso           # isometric diamond (W x W/2), sampled from the same water
    asset = make_water(...)          # kind "water": Godot animated TileSet, Tiled .tsx, strips

How it loops and tiles: the surface is a sum of waves with whole-number frequencies across the
tile (so it wraps) moving a whole number of periods per loop (so frame n = frame 0); a
periodic noise warp bends them. Banded onto a Lab ramp (deep -> light, hue-shifted, tinted for
the time of day like weather.py); crest lines are the tops of a second, faster wave field;
glints are 2-frame stars at fixed spots (glow map, so engines bloom them).

Shore tiles use tilegen.blob_region with the same 47 masks, margins and edge jitter as tilegen
water sets, so every shared edge matches; the margin breathes +-1-2 px over the loop (the tide)
and the foam line rides the waterline. make_water(tileset=...) takes the colours and the shore
ground of a saved tilegen water set (default: the project's newest one), so the animated set
matches the static one it replaces. The HD-2D board's matching shader is hd2d.WATER_SHADER_PIXEL
(make_hd2d(water="pixel")).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from . import assets

KIND = "water"
VIEWS = ("topdown", "side", "iso")


def register_kind() -> None:
    if KIND not in assets.KINDS:
        assets.KINDS = (*assets.KINDS, KIND)
    assets.FOLDERS.setdefault(KIND, "water")


register_kind()

THEMES = {
    "ocean": "0a1440 102a6c 14469a 1a68c0 2c90dc 58b8ee 9adcf8 dcf8ff ffffff",
    "lagoon": "0a2a44 0c4a62 0e6e7c 18948e 34b8a2 6cd8b6 aaf0d4 e4fff4 ffffff",
    "river": "142440 1c3c5e 285a7c 3a7c98 5a9eb4 88c2d0 bce0e8 eefaff ffffff",
    "swamp": "141e16 1c3020 26442a 365a32 4c723a 6c8c44 98aa5a cccc86 f0eebc",
    "lava": "2a0810 5a0e14 961a12 cc3410 f06218 ff9a2a ffd050 fff4a0 ffffff",
    "night": "040818 081430 0c2048 143066 1e4884 34689e 5a8cba 9cc0e0 e8f4ff",
}
GLOWING = ("lava",)


@dataclass
class WaterTiles:
    tile: int
    frames: list                 # open water, (t, t, 4) x n
    glow: list                   # emissive glints, same layout
    shore: dict                  # mask -> [frames]; -1 = the plain shore ground
    side_surface: list
    side_body: list
    iso: list
    ms: int
    ramp: np.ndarray
    theme: str
    tod: str | None
    base: str = "sand"
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.frames)


def _ramp(theme, colours, tod, palette):
    from .fxcore import make_ramp
    from .weather import _snap_each, _tint
    keys = THEMES.get(theme, THEMES["ocean"]).split()
    if colours is not None and len(colours):
        keys = list(colours)
    ramp = make_ramp(keys, 9)
    while len(ramp) < 9:
        ramp = np.concatenate([ramp, ramp[-1:]])
    ramp = _tint(ramp, tod, 0.0 if theme in GLOWING else 0.9)
    return _snap_each(ramp, palette)


def tileset_colours(asset) -> list[str]:
    """Key colours (dark -> light hex) of a saved tilegen water set's texture.png."""
    from .color import rgb_to_lab
    p = Path(asset.folder) / "texture.png"
    px = np.asarray(Image.open(p).convert("RGB")).reshape(-1, 3)
    L = rgb_to_lab(px.astype(np.float32))[:, 0]
    order = np.argsort(L)
    q = [px[order[int(f * (len(order) - 1))]] for f in (0.02, 0.2, 0.4, 0.6, 0.8, 0.98)]
    keys = ["%02x%02x%02x" % tuple(int(v) for v in c) for c in q]
    hi = np.clip(q[-1].astype(int) + 60, 0, 255)
    return ["%02x%02x%02x" % tuple(int(v * 0.55) for v in q[0])] + keys + ["%02x%02x%02x" % tuple(hi), "ffffff"]


def newest_water_set(project=None):
    """The project's newest top-down tilegen water set, or None."""
    try:
        for a in assets.list_assets("tileset", project=project):
            if a.extra.get("role") == "water" and a.extra.get("camera", "topdown") == "topdown" and \
                    (Path(a.folder) / "texture.png").exists():
                return a
    except Exception:  # noqa: BLE001 - no project
        return None
    return None


# --------------------------------------------------------------------------- the surface

def _fields(t: int, rng, calm: float = 1.0):
    """Static parts: the noise warp and wave table."""
    from .fxcore import Noise
    warp = Noise(rng, 3, 3)
    g = (np.arange(t, dtype=np.float32) + 0.5) / t
    U, V = np.meshgrid(g, g)
    wp = warp(U, V)
    wp2 = Noise(rng, 4, 2)(U, V)
    waves = [(0, 1, 1, 0.5), (1, 2, -1, 0.3), (-1, 3, 1, 0.2), (1, 4, 2, 0.1)]
    phases = rng.uniform(0, 2 * math.pi, len(waves))
    return U, V, wp, wp2, waves, phases, warp


def _surface(t, p, F, calm=1.0):
    """Height in [0, 1] and crest mask of the open water at loop phase p."""
    U, V, wp, wp2, waves, ph, warp = F
    h = np.zeros_like(U)
    norm = 0.0
    for (kx, ky, m, a), f in zip(waves, ph):
        h += a * np.sin(2 * math.pi * (kx * U + ky * V + 0.35 * wp) - 2 * math.pi * m * p + f)
        norm += a
    blob = warp(U + p, V * 2 - p) * 0.5 + warp(U * 2 - p + 0.3, V + 0.5) * 0.5   # drifting swell
    h = 0.68 * (h / norm + 1) / 2 + 0.32 * np.clip((blob - 0.5) * 2.0 + 0.5, 0, 1)
    r = np.sin(2 * math.pi * (1 * U + 4 * V + 0.55 * wp2) - 2 * math.pi * 2 * p) * \
        np.sin(2 * math.pi * (2 * U - 1 * V + 0.3 * wp) + 2 * math.pi * 1 * p)
    crest = (r > 0.7 + 0.1 * (1 - calm)) & (h > 0.5)
    hi = (r > 0.86) & (h > 0.6)
    return h, crest, hi


def _open_frame(t, p, F, ramp, calm=1.0):
    h, crest, hi = _surface(t, p, F, calm)
    lv = 1 + np.clip(np.floor(h * 4.0), 0, 3).astype(int)          # body bands 1..4
    lv = np.where(crest, 5, lv)
    lv = np.where(hi, 6, lv)
    out = np.zeros((t, t, 4), np.uint8)
    out[..., :3] = ramp[lv]
    out[..., 3] = 255
    return out, lv


def _glints(t, n, rng, count):
    return [(int(rng.integers(t)), int(rng.integers(t)), int(rng.integers(n))) for _ in range(count)]


def _put_glints(img, glow, gl, k, n, ramp):
    t = img.shape[0]
    for x, y, f0 in gl:
        j = (k - f0) % n
        if j > 1:
            continue
        pts = [(0, 0, 8)] if j == 0 else [(0, 0, 8), (1, 0, 7), (-1, 0, 7), (0, 1, 7), (0, -1, 7)]
        for dx, dy, li in pts:
            xx, yy = (x + dx) % t, (y + dy) % t
            img[yy, xx, :3] = ramp[li]
            glow[yy, xx, :3] = ramp[li]
            glow[yy, xx, 3] = 255


# --------------------------------------------------------------------------- shores

def _touch(mask, other):
    p = np.pad(other, 1, mode="edge")
    return mask & (p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:])


def _shore_tile(mask, base, water, lv, k, n, geo, ramp, tide):
    from .tilegen import blob_region
    m, jx, jy = geo
    t = base.shape[0]
    d = int(round(tide * math.sin(2 * math.pi * k / n)))
    mk = {key: max(1, v + d) for key, v in m.items()}
    region = blob_region(t, mask, mk, jx, jy)
    lowest = blob_region(t, mask, {key: max(1, v - tide) for key, v in m.items()}, jx, jy)
    out = base.copy()
    wet = lowest & ~region
    out[wet, :3] = np.clip(out[wet, :3].astype(np.float32) * 0.72, 0, 255).astype(np.uint8)
    out[region] = water[region]
    if region.all() or not region.any():
        return out
    shore = _touch(region, ~region)
    out[shore, :3] = ramp[7]
    inner = _touch(region & ~shore, shore)
    yy, xx = np.mgrid[0:t, 0:t]
    froth = inner & (((xx + yy + k) % 3) != 0)
    out[froth, :3] = ramp[6]
    second = _touch(region & ~shore & ~inner, inner) & (((xx * 2 + yy + 2 * k) % 5) < 2)
    out[second, :3] = ramp[5]
    edge = _touch(~region, region) & ~wet
    out[edge, :3] = np.clip(out[edge, :3].astype(np.float32) * 0.8, 0, 255).astype(np.uint8)
    del lv
    return out


# --------------------------------------------------------------------------- side + iso

def _side(t, p, F, ramp, k, n):
    """(surface, body) frames: the body is the open water seen through the side, bands by
    depth, moving light lines; the surface tile adds the crest over see-through air."""
    U, V, wp, wp2 = F[:4]
    y = (np.arange(t)[:, None] + 0.5) / t
    x = U
    caust = np.sin(2 * math.pi * (2 * x + 0.4 * wp) - 2 * math.pi * p) * np.sin(2 * math.pi * (1 * V + 0.3 * wp2) + 2 * math.pi * p)
    lv = np.full((t, t), 2)
    lv = np.where(caust > 0.55, 3, lv)
    lv = np.where(caust > 0.85, 4, lv)
    body = np.zeros((t, t, 4), np.uint8)
    body[..., :3] = ramp[lv]
    body[..., 3] = 235
    # surface: wave line in whole pixels
    xs = (np.arange(t) + 0.5) / t
    A = max(1.0, t / 16)
    line = np.rint(t * 0.3 + A * np.sin(2 * math.pi * (xs - p)) + 0.6 * A * np.sin(2 * math.pi * (2 * xs + p) + 1.3)).astype(int)
    surf = body.copy()
    yy = np.arange(t)[:, None]
    above = yy < line[None, :]
    surf[above] = 0
    crest = yy == line[None, :]
    under = yy == line[None, :] + 1
    deep_top = (yy > line[None, :] + 1) & (yy <= line[None, :] + max(2, t // 8))
    surf[crest, :3], surf[crest, 3] = ramp[7], 255
    surf[under, :3], surf[under, 3] = ramp[6], 245
    surf[deep_top & (lv[..., ] < 4), :3] = ramp[4]
    # spray dots above the crest, a couple per frame
    rng = np.random.default_rng(k)
    for _ in range(max(1, t // 16)):
        sx = int(rng.integers(t))
        sy = int(line[sx]) - 1 - int(rng.integers(2))
        if 0 <= sy < t and (sx + k) % 3 == 0:
            surf[sy, sx, :3], surf[sy, sx, 3] = ramp[8], 255
    del y
    return surf, body


def _iso(frame: np.ndarray) -> np.ndarray:
    """A top-down tile seen as an iso diamond W x W/2 (same sampling as tilesets.py)."""
    t = frame.shape[0]
    W, H = t, t // 2
    out = np.zeros((H, W, 4), np.uint8)
    yy, xx = np.mgrid[0:H, 0:W] + 0.5
    a = (xx - W / 2) / W + yy / H
    b = yy / H - (xx - W / 2) / W
    inside = (np.abs(xx - W / 2) / (W / 2) + np.abs(yy - H / 2) / (H / 2)) <= 1.0
    sx = np.clip((a * t).astype(int) % t, 0, t - 1)
    sy = np.clip((b * t).astype(int) % t, 0, t - 1)
    out[inside] = frame[sy[inside], sx[inside]]
    return out


# --------------------------------------------------------------------------- render

def render_water(tile: int | None = None, *, theme: str = "ocean", tod: str | None = None, frames: int = 8,
                 ms: int = 140, seed: int = 0, colours=None, base="sand", style=None, palette=None,
                 project=None, calm: float = 1.0, glints: int | None = None, tide: int | None = None) -> WaterTiles:
    """Every water tile, in memory. tile: px (default the project's). theme: ocean | lagoon |
    river | swamp | lava | night (or colours: key colours dark -> light). base: the shore ground
    (a terrain name, see tilegen.base_texture, or an RGBA tile)."""
    from .effects import project_palette
    from .tilegen import BLOB_MASKS, _blob_geometry, base_texture
    st = style or (project.style if project is not None else None)
    t = int(tile or getattr(st, "tile", 32) or 32)
    if t < 8 or t % 2:
        raise ValueError("tile must be an even number of pixels, 8 or more")
    if theme not in THEMES and not colours:
        raise ValueError(f"Unknown water theme '{theme}' (use {', '.join(THEMES)})")
    if not 2 <= frames <= 32:
        raise ValueError("frames must be between 2 and 32")
    pal = palette if palette is not None else project_palette(st)
    ramp = _ramp(theme, colours, tod, pal)
    rng = np.random.default_rng(seed)
    F = _fields(t, rng, calm)
    gl = _glints(t, frames, rng, glints if glints is not None else max(1, t * t // 500))
    base_tex = base if isinstance(base, np.ndarray) else base_texture(base, t, project, seed)
    geo = _blob_geometry(t, "water", seed)
    tide = int(tide if tide is not None else max(1, t // 16))
    outs, glows, sides_s, sides_b, isos = [], [], [], [], []
    shore: dict = {m: [] for m in BLOB_MASKS}
    shore[-1] = []
    for k in range(frames):
        p = k / frames
        f, lv = _open_frame(t, p, F, ramp, calm)
        g = np.zeros_like(f)
        _put_glints(f, g, gl, k, frames, ramp)
        if theme in GLOWING:                     # lava glows by brightness
            hot = lv >= 3
            g[hot, :3] = (ramp[lv[hot]].astype(np.float32) * (0.25 + 0.2 * (lv[hot, None] - 3))).astype(np.uint8)
            g[hot, 3] = 255
        outs.append(f)
        glows.append(g)
        for m in BLOB_MASKS:
            shore[m].append(_shore_tile(m, base_tex, f, lv, k, frames, geo, ramp, tide))
        shore[-1].append(base_tex.copy())
        s, b = _side(t, p, F, ramp, k, frames)
        sides_s.append(s)
        sides_b.append(b)
        isos.append(_iso(f))
    return WaterTiles(t, outs, glows, shore, sides_s, sides_b, isos, int(ms), ramp,
                      theme if not colours else "custom", tod, base if isinstance(base, str) else "ground",
                      {"seed": seed, "tide": tide, "calm": calm})


# --------------------------------------------------------------------------- files

def _strip(frames):
    return np.concatenate(frames, 1)


def shore_layout(wt: WaterTiles, cols: int = 8):
    """[(mask, col, row)]: tile 0 = the plain ground, then the 47 blob masks (tilegen order)."""
    from .tilegen import BLOB_MASKS
    keys = [-1] + list(BLOB_MASKS)
    return [(m, i % cols, i // cols) for i, m in enumerate(keys)]


def shore_atlas(wt: WaterTiles, cols: int = 8) -> np.ndarray:
    """Atlas for Godot animated tiles: tile (c, r) at atlas coords (c * n, r), its frames to the
    right of it."""
    t, n = wt.tile, wt.n
    lay = shore_layout(wt, cols)
    rows = max(r for _, _, r in lay) + 1
    atlas = np.zeros((rows * t, cols * n * t, 4), np.uint8)
    for m, c, r in lay:
        for k, f in enumerate(wt.shore[m]):
            x = (c * n + k) * t
            atlas[r * t:(r + 1) * t, x:x + t] = f
    return atlas


def write_godot_shore(wt: WaterTiles, out_tres, png_name: str, cols: int = 8) -> Path:
    """Animated Godot 4 TileSet: terrain set 0 Match Corners and Sides, terrain 0 = shore
    ground, 1 = water (tilegen's peering bits), every tile n frames."""
    from .tilegen import BITS8, _border_alternatives, canonical
    t, n = wt.tile, wt.n
    dur = wt.ms / 1000
    lines = ['[gd_resource type="TileSet" load_steps=3 format=3]', "",
             f'[ext_resource type="Texture2D" path="{png_name}" id="1_atlas"]', "",
             '[sub_resource type="TileSetAtlasSource" id="TileSetAtlasSource_water"]',
             'texture = ExtResource("1_atlas")', f"texture_region_size = Vector2i({t}, {t})"]
    for m, c, r in shore_layout(wt, cols):
        key = f"{c * n}:{r}"
        lines += [f"{key}/animation_frames_count = {n}"]
        lines += [f"{key}/animation_frame_{k}/duration = {dur:g}" for k in range(n)]
        lines += [f"{key}/0 = 0", f"{key}/0/terrain_set = 0", f"{key}/0/terrain = {0 if m < 0 else 1}"]
        for b, name in BITS8:
            on = 0 if m < 0 else (1 if canonical(m) & b else 0)
            lines.append(f"{key}/0/terrains_peering_bit/{name} = {on}")
    alts = _border_alternatives()
    lines.append(f"0:0/next_alternative_id = {len(alts) + 1}")
    for alt, empty in enumerate(alts, start=1):
        k0 = f"0:0/{alt}"
        lines += [f"{k0} = {alt}", f"{k0}/terrain_set = 0", f"{k0}/terrain = 0"]
        lines += [f"{k0}/terrains_peering_bit/{nm} = 0" for _, nm in BITS8 if nm not in empty]
    c0, c1 = wt.frames[0][..., :3].reshape(-1, 3).mean(0), wt.shore[-1][0][..., :3].reshape(-1, 3).mean(0)
    lines += ["", "[resource]", f"tile_size = Vector2i({t}, {t})", "terrain_set_0/mode = 0",
              f'terrain_set_0/terrain_0/name = "{wt.base}"',
              "terrain_set_0/terrain_0/color = Color(%.3f, %.3f, %.3f, 1)" % tuple(c1 / 255),
              'terrain_set_0/terrain_1/name = "water"',
              "terrain_set_0/terrain_1/color = Color(%.3f, %.3f, %.3f, 1)" % tuple(c0 / 255),
              'sources/0 = SubResource("TileSetAtlasSource_water")', ""]
    out_tres = Path(out_tres)
    out_tres.write_text("\n".join(lines), encoding="utf-8")
    return out_tres


def write_tiled_shore(wt: WaterTiles, out_tsx, png_name: str, cols: int = 8) -> Path:
    """Tiled tileset with an <animation> per tile (frames to the right of each tile)."""
    t, n = wt.tile, wt.n
    atlas_cols = cols * n
    rows = max(r for _, _, r in shore_layout(wt, cols)) + 1
    out = [f'<?xml version="1.0" encoding="UTF-8"?>',
           f'<tileset version="1.10" name="water" tilewidth="{t}" tileheight="{t}" tilecount="{atlas_cols * rows}" columns="{atlas_cols}">',
           f' <image source="{png_name}" width="{atlas_cols * t}" height="{rows * t}"/>']
    for m, c, r in shore_layout(wt, cols):
        tid = r * atlas_cols + c * n
        out.append(f' <tile id="{tid}"><properties><property name="mask" type="int" value="{m}"/></properties><animation>')
        out += [f'  <frame tileid="{tid + k}" duration="{wt.ms}"/>' for k in range(n)]
        out.append(" </animation></tile>")
    out.append("</tileset>")
    p = Path(out_tsx)
    p.write_text("\n".join(out) + "\n", encoding="utf-8")
    return p


def example_map(wt: WaterTiles, k: int, cols: int = 14, rows: int = 9, seed: int = 0) -> np.ndarray:
    """A little lake drawn with the shore set at frame k (previews, the test)."""
    from .tilegen import canonical, example_blob, neighbour_mask
    on = example_blob(cols, rows, seed)
    t = wt.tile
    img = np.zeros((rows * t, cols * t, 4), np.uint8)
    for y in range(rows):
        for x in range(cols):
            m = canonical(neighbour_mask(on, y, x)) if on[y, x] else -1
            tile = wt.shore.get(m, wt.shore[255])[k]
            img[y * t:(y + 1) * t, x * t:(x + 1) * t] = tile
    return img


def write_water(wt: WaterTiles, folder) -> list[Path]:
    from .effects import Effect, write_effect
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fx = Effect("water", wt.frames, np.unique(wt.ramp, axis=0), wt.ms, True, wt.theme, wt.ramp)
    fx.durations, fx.glow, fx.blend, fx.tags = [wt.ms] * wt.n, wt.glow, "mix", [("water", 0, wt.n - 1, True)]
    written = write_effect(fx, folder, "sheet")
    files = {"water.png": _strip(wt.frames), "water_glow.png": _strip(wt.glow),
             "side_surface.png": _strip(wt.side_surface), "side_body.png": _strip(wt.side_body),
             "iso_water.png": _strip(wt.iso), "shore_tileset.png": shore_atlas(wt)}
    for name, img in files.items():
        p = folder / name
        Image.fromarray(img, "RGBA").save(p)
        written.append(p)
    written.append(write_godot_shore(wt, folder / "shore_tileset.tres", "shore_tileset.png"))
    written.append(write_tiled_shore(wt, folder / "shore_tileset.tsx", "shore_tileset.png"))
    ex = [example_map(wt, k) for k in range(wt.n)]
    Image.fromarray(ex[0], "RGBA").save(folder / "example.png")
    ims = [Image.fromarray(e, "RGBA").convert("RGB").resize((e.shape[1] * 2, e.shape[0] * 2), Image.NEAREST) for e in ex]
    ims[0].save(folder / "example.gif", save_all=True, append_images=ims[1:], duration=wt.ms, loop=0)
    written += [folder / "example.png", folder / "example.gif"]
    info = {"tile": wt.tile, "frames": wt.n, "frame_ms": wt.ms, "theme": wt.theme, "tod": wt.tod,
            "base": wt.base, "strips": {k: k for k in files if k != "shore_tileset.png"},
            "shore": {"atlas": "shore_tileset.png", "layout": "tile (c, r) at atlas (c * frames, r); frames go right",
                      "tiles": [{"mask": m, "col": c, "row": r} for m, c, r in shore_layout(wt)]},
            "meta": wt.meta}
    js = folder / "water.json"
    js.write_text(json.dumps(info, indent=1), encoding="utf-8")
    written.append(js)
    return written


def make_water(name: str | None = None, *, project=None, theme: str = "ocean", tod: str | None = None,
               tileset=None, frames: int = 8, ms: int = 140, seed: int = 0, base: str | None = None,
               tags=None, **opts):
    """Make animated water and save it in the project (kind "water"). tileset: a saved tilegen
    water set to match (its colours, tile size and shore ground); None = the project's newest
    one if it has any, False = none. Returns the Asset."""
    register_kind()
    proj = project or assets.current_project()
    st = proj.style
    colours = opts.pop("colours", None)
    src = newest_water_set(proj) if tileset is None else (tileset or None)
    tile = opts.pop("tile", None)
    if src is not None:
        colours = colours or tileset_colours(src)
        tile = tile or src.extra.get("tile")
        base = base or src.extra.get("base", "sand")
    wt = render_water(tile, theme=theme, tod=tod, frames=frames, ms=ms, seed=seed, colours=colours,
                      base=base or "sand", style=st, project=proj, **opts)
    label = name or (f"{wt.theme.title()} water" + (f" ({tod})" if tod else ""))
    a = assets.new_asset(KIND, label, proj, theme=wt.theme, tod=tod, tile=wt.tile, frames=wt.n, frame_ms=ms,
                         seed=seed, base=wt.base, from_tileset=(src.folder.name if src is not None else None))
    files = write_water(wt, a.folder)
    a.extra["files"] = [p.relative_to(a.folder).as_posix() for p in files]
    a.save()
    assets.set_tags(a, sorted({"water", wt.theme, *([tod] if tod else []), *(tags or [])}))
    return a

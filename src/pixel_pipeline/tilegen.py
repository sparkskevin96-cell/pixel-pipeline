"""Tile auto-variations: paths, cliffs, water edges and fences drawn by the GPU in the project's
house style, built into proper tilesets.

    sw = make_swatches("water", "clear lagoon water", count=3)   # SDXL pictures -> seamless tiles
    ts = build_tileset("water", sw[0].texture, picture=sw[0].picture)   # 47 shore tiles + variants
    asset = save_tileset(ts, "Lagoon")       # Projects/<p>/tilesets/lagoon/ (kind "tileset")

1. Swatch: SDXL + Pixel Art XL (generate.generate_images) draws a texture of the terrain at the
   project's camera; the prompt goes through assets.with_style ("tile"), and generate adds the
   project's trained style LoRA. The picture is made seamless (half-offset copies cross-faded,
   or the offset copy's centre cross repainted by an inpaint pass: its edges wrap), its large-scale
   brightness evened out (no tile-sized blotch to mark the period), then pixelized tileable at the
   project's tile size and snapped to the project palette.
2. Set (no GPU), by what the terrain is:
   - ground (dirt path, road, sand, snow, stone over the base, usually grass): 16 corner tiles
     (maptiles, Godot Match Corners).
   - water: 47 "blob" tiles (Godot Match Corners and Sides): shore with a foam line.
   - cliff: 47 blob tiles: raised ground, a darker rock face row on open south sides.
   - fence / wall: 16 line tiles by connected sides (post, ends, straight, corners, T, cross),
     transparent, with collision; the base ground follows in the last row.
   Side-scroller projects get tilesets.build_side_set (ground + top), isometric ones floors +
   blocks. Fewer visible repeats: every terrain gets ~8 interior variants of its full tile (other
   pieces of the picture, rolls and the flips / turns its lighting allows, a detail stamp or two),
   all inside the tile's 1-2 px border band so every edge still matches; they share the
   probability evenly (Godot probability, Tiled tile probability, the example map).
3. Files: tileset.png + .tres (Godot TileSet, terrain peering bits), .tsx (Tiled wang set) +
   example.tmj, tileset.json (layout), sheet.png/.json (Unity export via exports.py), texture.png
   (the seamless tile: maplib's godot-tilemap export paints with it), example.png, thumb.png.
4. Maps: MapLook draws a map (maplib MapDoc) with the project's saved top-down sets: plain cells
   from the variants, corner sets toward their neighbours, shores, cliffs where the ground drops
   2+ levels, fence/wall props as line tiles; the editor, its preview and the exports share it.

The asset kind "tileset" is in assets.KINDS; register_kind() re-adds it if another module
rebuilt the tuple.
"""
from __future__ import annotations

import json
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from . import assets

KIND = "tileset"


def register_kind() -> None:
    """Make sure "tileset" is an asset kind (additive; assets.py owns the list)."""
    if KIND not in assets.KINDS:
        assets.KINDS = (*assets.KINDS, KIND)
    assets.FOLDERS.setdefault(KIND, "tilesets")


register_kind()

# --------------------------------------------------------------------------- terrains

# name: (default words, role, built-in maptiles look used without the GPU)
TERRAINS = {
    "grass": ("lush green grass, small leaves and clover", "ground", "grass"),
    "dirt": ("dirt path, packed brown earth, small pebbles", "ground", "dirt"),
    "road": ("cobblestone road, rounded paving stones", "ground", "road"),
    "stone": ("stone floor, worn flagstones", "ground", "stone"),
    "sand": ("fine beach sand, small ripples", "ground", "sand"),
    "snow": ("fresh snow, soft drifts", "ground", "snow"),
    "water": ("clear blue water, gentle ripples", "water", "water"),
    "cliff": ("rough rocky cliff face, layered grey stone", "cliff", "stone"),
    "fence": ("weathered wooden planks, wood grain", "line", "road"),
    "wall": ("stone brick wall, mortar lines", "line", "stone"),
}
ROLES = ("ground", "water", "cliff", "line")
CAMERAS = ("topdown", "side", "iso")
# Map editor terrains (diorama.TERRAINS) a generated tileset stands in for on export
MAP_TERRAIN = {"grass": "grass", "dirt": "dirt", "road": "cobble", "stone": "stone", "sand": "sand",
               "snow": "snow", "water": "water"}

PROMPTS = {
    "topdown": ("seamless tileable texture of {words}, seen from directly above, top-down game map "
                "tile, flat even lighting, no perspective, fills the whole image, pixel art"),
    "side": ("seamless tileable texture of {words}, side view cross-section, 2D platformer ground "
             "tile, flat even lighting, fills the whole image, pixel art"),
    "iso": ("seamless tileable texture of {words}, seen from directly above, isometric game floor "
            "tile, flat even lighting, fills the whole image, pixel art"),
}
NEGATIVE = ("perspective, horizon, sky, objects, characters, people, text, border, frame, vignette, "
            "shadows, 3d render, photo")


def role_of(terrain: str) -> str:
    return TERRAINS[terrain][1] if terrain in TERRAINS else "ground"


def camera_for(style: assets.Style | None = None) -> str:
    """The project's tile camera: side-scroller -> side, isometric -> iso, else top-down."""
    st = style or assets.project_style()
    if st.tile_style == "side" or st.project_type == "side-scroller":
        return "side"
    if st.tile_style == "iso" or st.project_type == "isometric":
        return "iso"
    return "topdown"


def swatch_prompt(terrain: str, words: str = "", camera: str = "topdown",
                  style: assets.Style | None = None) -> tuple[str, str]:
    """(positive, negative) for a texture swatch, with the house style appended."""
    if camera not in CAMERAS:
        raise ValueError(f"camera must be one of {', '.join(CAMERAS)}")
    text = (words or "").strip().rstrip(".,; ") or (TERRAINS[terrain][0] if terrain in TERRAINS
                                                     else terrain)
    return assets.with_style(PROMPTS[camera].format(words=text), NEGATIVE,
                             style or assets.project_style(), "tile")


# --------------------------------------------------------------------------- seamless pictures

def _rgb(picture) -> np.ndarray:
    from .pipeline import load_image
    a = load_image(picture) if not isinstance(picture, np.ndarray) else picture
    return np.ascontiguousarray(a[..., :3]).astype(np.uint8)


def seam_cross(h: int, w: int, width: float = 0.14) -> np.ndarray:
    """Soft mask (0..1) of the centre cross where a half-offset copy has its seams."""
    y = np.abs(np.arange(h) + 0.5 - h / 2) / (h * width / 2)
    x = np.abs(np.arange(w) + 0.5 - w / 2) / (w * width / 2)
    fy = np.clip(1.5 - y, 0, 1)[:, None]
    fx = np.clip(1.5 - x, 0, 1)[None, :]
    return np.maximum(fy, fx)


def seamless(picture, *, size: int = 512, client=None, prompt: str = "", negative: str = "",
             seed: int = 0, inpaint: bool = False, denoise: float = 0.8) -> np.ndarray:
    """A picture -> an RGB picture that wraps. CPU: four half-offset copies cross-faded so each
    is invisible along its own seams. inpaint=True: the half-offset copy (which wraps: its edges
    are the picture's middle) gets its centre cross repainted on the GPU; the edges stay."""
    rgb = np.asarray(Image.fromarray(_rgb(picture)).resize((size, size), Image.LANCZOS))
    rolled = np.roll(rgb, (size // 2, size // 2), axis=(0, 1))
    cross = seam_cross(size, size)
    if inpaint:
        from .generate import generate_images
        rgba = np.dstack([rolled, np.full((size, size), 255, np.uint8)])
        (img, _), = generate_images(prompt, negative, client=client, seed=seed, init_image=rgba,
                                    mask=cross, denoise=denoise)
        fixed = np.asarray(img.convert("RGB").resize((size, size), Image.LANCZOS))
        return np.clip(fixed * cross[..., None] + rolled * (1 - cross[..., None]), 0, 255).astype(np.uint8)
    # four half-offset copies, each weighted to zero along its own seams (sin^2 + cos^2 = 1)
    s = np.sin(np.pi * (np.arange(size) + 0.5) / size) ** 2
    wx, wy = (s, 1 - s), (s, 1 - s)
    out = np.zeros(rgb.shape, np.float32)
    for iy in (0, 1):
        for ix in (0, 1):
            w = (wy[iy][:, None] * wx[ix][None, :])[..., None]
            out += w * np.roll(rgb, (iy * size // 2, ix * size // 2), axis=(0, 1))
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _tile_settings(size: int, style: assets.Style, colours: int | None = None):
    from .settings import Settings
    return Settings(subject="background", width=size, height=size, fit="cover", anchor="center",
                    remove_background=False, crop_to_subject=False, tileable=True, patch=4, detail=0,
                    colors=colours or min(16, style.colours), dither="none", cleanup=False,
                    outline=False)


def project_palette(style: assets.Style | None) -> np.ndarray | None:
    from .effects import project_palette as pp
    return pp(style or assets.project_style())


def pixel_texture(picture, size: int, style: assets.Style | None = None,
                  palette: np.ndarray | None = None) -> np.ndarray:
    """A seamless picture -> a seamless size x size pixel tile (RGBA) in the project's colours
    (palette: these colours exactly, else the project's locked palette, else auto)."""
    from .pipeline import finish, prepare
    st = style or assets.project_style()
    pal = palette if palette is not None else project_palette(st)
    rgb = _rgb(picture)
    rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])
    s = _tile_settings(size, st)
    out = finish(prepare(rgba, s), s, pal).rgba
    out[..., 3] = 255
    return out


def edge_colour(tex: np.ndarray) -> tuple[int, int, int]:
    """The darkest tenth, a bit darker: the line where the terrain meets another."""
    cols = tex[..., :3].reshape(-1, 3).astype(int)
    lum = cols.sum(1)
    dark = cols[lum <= np.percentile(lum, 10)].mean(0)
    return tuple(int(v) for v in np.clip(dark * 0.85, 0, 255))


def interior_variants(texture: np.ndarray, count: int = 3, source: np.ndarray | None = None,
                      style: assets.Style | None = None, seed: int = 0) -> list[np.ndarray]:
    """[texture, variant, ...] the 2026-10-13 way (build_tileset(mix=False); variety() is
    today's): each variant has the texture's border band (so it sits next to every edge tile)
    and a different interior: a shifted piece of the same seamless source, in the texture's own
    colours. The shift with the least mismatch at the band is kept."""
    t = texture.shape[0]
    out = [texture]
    if count <= 1:
        return out
    rng = np.random.default_rng(seed + 7)
    band = max(2, t // 8)
    y, x = np.mgrid[0:t, 0:t]
    dist = np.minimum(np.minimum(x, t - 1 - x), np.minimum(y, t - 1 - y)).astype(np.float32)
    from .maptiles import _periodic_noise
    inner = dist + 1.5 * (_periodic_noise(t, max(2, t // 8), rng) - 0.5) >= band
    pal = np.unique(texture[..., :3].reshape(-1, 3), axis=0)
    boundary = inner & ~(np.roll(inner, 1, 0) & np.roll(inner, -1, 0) & np.roll(inner, 1, 1)
                         & np.roll(inner, -1, 1))
    for _ in range(count - 1):
        best, best_err = None, None
        for _ in range(4 if source is not None else 10):     # converter calls are the slow part
            if source is not None:
                h, w = source.shape[:2]
                shifted = np.roll(source, (int(rng.integers(h)), int(rng.integers(w))), axis=(0, 1))
                cand = pixel_texture(shifted, t, style, pal)
            else:
                cand = np.roll(texture, (int(rng.integers(1, t)), int(rng.integers(1, t))), axis=(0, 1))
            err = np.abs(cand[boundary, :3].astype(int) - texture[boundary, :3].astype(int)).sum()
            if best_err is None or err < best_err:
                best, best_err = cand, err
        v = texture.copy()
        v[inner] = best[inner]
        out.append(v)
    return out


# --------------------------------------------------------------------------- fewer visible repeats

def _lum(img: np.ndarray) -> np.ndarray:
    return img[..., :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114])


def even_out(picture, cycles: float = 2.0, strength: float = 0.9) -> np.ndarray:
    """A seamless RGB picture with its large-scale brightness flattened (its lowest wrapping
    frequencies, ~`cycles` per side): the tile made from it has no blotch that marks the period."""
    from .maptiles import low_band
    rgb = _rgb(picture).astype(np.float64)
    delta = low_band(_lum(rgb), cycles) * strength
    return np.clip(np.rint(rgb - delta[..., None]), 0, 255).astype(np.uint8)


TRANSFORMS = {"flip_h": lambda a: a[:, ::-1], "flip_v": lambda a: a[::-1],
              "turn_180": lambda a: a[::-1, ::-1], "transpose": lambda a: a.transpose(1, 0, 2),
              "turn_90": lambda a: np.rot90(a, 1), "turn_270": lambda a: np.rot90(a, 3),
              "anti_transpose": lambda a: np.rot90(a, 2).transpose(1, 0, 2)}


def _grain(tex: np.ndarray) -> np.ndarray:
    """(light skew x, light skew y, log x/y contrast): how the look depends on direction."""
    lum = _lum(tex)
    out = []
    for d in (np.roll(lum, -1, 1) - lum, np.roll(lum, -1, 0) - lum):
        sd = d.std()
        out.append(float((d ** 3).mean() / sd ** 3) if sd > 1e-9 else 0.0)
    ax = np.abs(np.roll(lum, -1, 1) - lum).mean()
    ay = np.abs(np.roll(lum, -1, 0) - lum).mean()
    out.append(float(np.log((ax + 1e-6) / (ay + 1e-6))))
    return np.array(out)


def symmetries(tex: np.ndarray, tol: float = 0.35) -> list[str]:
    """The flips / turns a texture allows: those that keep its light direction (gradient skew)
    and grain (x/y contrast). Stones lit from the upper left don't flip; plain grass does."""
    g = _grain(tex)
    return [n for n, f in TRANSFORMS.items() if np.abs(_grain(np.ascontiguousarray(f(tex))) - g).max() <= tol]


def _inner(t: int, band: int, seed: int) -> np.ndarray:
    from .maptiles import _periodic_noise
    rng = np.random.default_rng(seed + 7)
    y, x = np.mgrid[0:t, 0:t]
    dist = np.minimum(np.minimum(x, t - 1 - x), np.minimum(y, t - 1 - y)).astype(np.float32)
    return dist + 1.5 * (_periodic_noise(t, max(2, t // 8), rng) - 0.5) >= band


def variety(texture: np.ndarray, count: int = 8, sources=(), *, seed: int = 0, band: int | None = None,
            stamps: bool = True, max_corr: float = 0.3) -> tuple[list[np.ndarray], list[float]]:
    """([texture, variant, ...], weights). Interiors come from every roll and every allowed flip /
    turn (symmetries) of the texture and of `sources` (other tiles of the same picture), inside
    the texture's own border band (1 px at 16, 2 at 32) so every edge still matches. The ones
    that join the band best win, each correlating at most max_corr with those already in; a
    detail stamp on the last one or two (weight 0.5)."""
    t = texture.shape[0]
    if count <= 1:
        return [texture], [1.0]
    band = band or max(1, t // 12)
    inner = _inner(t, band, seed)
    edge = inner & ~(np.roll(inner, 1, 0) & np.roll(inner, -1, 0) & np.roll(inner, 1, 1) & np.roll(inner, -1, 1))
    py, px = np.nonzero(edge)
    a = texture[py, px, :3].astype(np.int32)
    sh = np.arange(t)
    ys, xs = (py[None, :] - sh[:, None]) % t, (px[None, :] - sh[:, None]) % t     # (shift, n)
    names = ["id", *symmetries(texture)]
    cands = []                                        # (seam error, source, transform, dy, dx)
    for si, src in enumerate([texture, *sources]):
        for name in names:
            b = np.ascontiguousarray(TRANSFORMS[name](src) if name != "id" else src)[..., :3].astype(np.int32)
            err = np.abs(b[ys[:, None, :], xs[None, :, :]] - a).sum((-1, -2))   # (dy, dx)
            for (dy, dx), e in np.ndenumerate(err):
                if not (si == 0 and name == "id" and dy == 0 and dx == 0):
                    cands.append((int(e), si, name, dy, dx))
    cands.sort()
    srcs = [texture, *sources]
    def norm(v):
        v = v - v.mean()
        return v / max(1e-9, float(np.sqrt((v * v).sum())))
    n_stamp = max(1, count // 4) if stamps and count >= 4 else 0
    out, lums = [texture], [norm(_lum(texture)[inner])]
    for _, si, name, dy, dx in cands[:4000]:
        src = srcs[si]
        b = np.ascontiguousarray(TRANSFORMS[name](src) if name != "id" else src)
        c = np.roll(b, (dy, dx), (0, 1))
        cl = norm(_lum(c)[inner])
        if max(float(cl @ o) for o in lums) > max_corr:      # too like one already in
            continue
        v = texture.copy()
        v[inner] = c[inner]
        out.append(v)
        lums.append(cl)
        if len(out) >= count:
            break
    weights = [1.0] * len(out)
    rng = np.random.default_rng(seed + 11)
    for i in range(min(n_stamp, len(out) - 1)):          # the last few get a detail, rarer
        out[-1 - i] = _stamp(out[-1 - i], inner, rng)
        weights[-1 - i] = 0.5
    return out, weights


def _stamp(tile: np.ndarray, inner: np.ndarray, rng) -> np.ndarray:
    """A few small clusters in the tile's own lightest / darkest colours, inside the band."""
    t = tile.shape[0]
    out = tile.copy()
    cols = tile[inner][:, :3]
    lum = _lum(cols)
    accents = (cols[lum.argmax()], cols[lum.argmin()])
    ys, xs = np.nonzero(inner)
    for _ in range(max(1, t // 10)):
        i = rng.integers(len(ys))
        c = accents[rng.integers(2)]
        for dy, dx in ((0, 0), (0, 1), (1, 0), (-1, 0), (0, -1))[: rng.integers(1, 4)]:
            y, x = ys[i] + dy, xs[i] + dx
            if 0 <= y < t and 0 <= x < t and inner[y, x]:
                out[y, x, :3] = c
    return out


def repeat_score(img: np.ndarray, t: int) -> float:
    """How exactly a picture repeats every tile: normalised autocorrelation of its brightness at
    the lattice lags (t, 0), (0, t), (t, t), (t, -t). 1 = one tile copied; noise ~0."""
    lum = _lum(img)
    lum = lum - lum.mean()
    h, w = lum.shape
    f = np.fft.rfft2(lum, (2 * h, 2 * w))
    ac = np.fft.irfft2(np.abs(f) ** 2, (2 * h, 2 * w))

    def at(dy, dx):
        return ac[dy % (2 * h), dx % (2 * w)] / ((h - abs(dy)) * (w - abs(dx)))
    return float(np.mean([at(t, 0), at(0, t), at(t, t), at(t, -t)]) / max(at(0, 0), 1e-9))


def repeat_contrast(img: np.ndarray, t: int, blur: float | None = None) -> float:
    """The contrast (grey levels, std) of what every tile has in common: the picture folded onto
    one tile and averaged, after a blur of t/4 px (the map seen from a step back)."""
    lum = _lum(img)
    s = t / 4 if blur is None else blur
    if s > 0:
        h, w = lum.shape
        fy, fx = np.fft.fftfreq(h)[:, None], np.fft.fftfreq(w)[None, :]
        lum = np.real(np.fft.ifft2(np.fft.fft2(lum) * np.exp(-2 * (np.pi * s) ** 2 * (fy ** 2 + fx ** 2))))
    h, w = lum.shape
    return float(lum[:h // t * t, :w // t * t].reshape(h // t, t, w // t, t).mean((0, 2)).std())


def cell_noise(rows: int, cols: int, seed: int = 0, salt: str = "") -> np.ndarray:
    """Uniform [0, 1) per cell from (seed, salt, col, row) only: repainting one cell never
    reshuffles the variants of the others."""
    r, c = np.mgrid[0:rows, 0:cols]
    h = (c.astype(np.uint64) * np.uint64(0x9E3779B1) + r.astype(np.uint64) * np.uint64(0x85EBCA77)
         + np.uint64(zlib.crc32(f"{seed}/{salt}".encode())))
    for k, m in ((15, 0x2C1B3C6D), (12, 0x297A2D39), (15, 0x9E3779B1)):
        h = (h ^ (h >> np.uint64(k))) * np.uint64(m) & np.uint64(0xFFFFFFFF)
    return (h & np.uint64(0xFFFFFF)).astype(np.float64) / float(1 << 24)


def pick(options: list, weights, u: float):
    """The option a uniform number u lands on, by weight."""
    c = np.cumsum(np.asarray(weights, np.float64))
    return options[min(len(options) - 1, int(np.searchsorted(c / c[-1], u, side="right")))]


def fill_area(tiles: list[np.ndarray], weights, cells: int = 32, seed: int = 0) -> np.ndarray:
    """cells x cells tiles of one terrain placed by weight (the metric's test area)."""
    t = tiles[0].shape[0]
    u = cell_noise(cells, cells, seed, "area")
    img = np.zeros((cells * t, cells * t, 4), np.uint8)
    for (y, x), v in np.ndenumerate(u):
        img[y * t:(y + 1) * t, x * t:(x + 1) * t] = pick(tiles, weights, v)
    return img


@dataclass
class Swatch:
    terrain: str
    texture: np.ndarray          # tile x tile RGBA, seamless, project colours
    picture: np.ndarray | None   # the seamless source picture (RGB), None for built-ins
    seed: int = 0
    prompt: str = ""
    raw: np.ndarray | None = None   # what SDXL drew


def builtin_swatch(terrain: str, size: int = 32, seed: int = 0,
                   style: assets.Style | None = None, even: bool = True) -> Swatch:
    """No GPU: the maptiles pattern for the terrain (large-scale brightness evened out), snapped
    to the project palette."""
    from .maptiles import builtin_texture
    look = TERRAINS[terrain][2] if terrain in TERRAINS else terrain
    tex = builtin_texture(look, size, seed, even=even)
    pal = project_palette(style)
    if pal is not None:
        from .cards import snap_image
        tex = snap_image(tex, pal)
    return Swatch(terrain, tex, None, seed, "built-in")


def make_swatches(terrain: str, words: str = "", *, count: int = 3, camera: str | None = None,
                  size: int | None = None, client=None, seed: int | None = None,
                  style: assets.Style | None = None, inpaint_seams: bool = False, even: bool = True,
                  say=None, **gen_opts) -> list[Swatch]:
    """`count` texture variations from the GPU (seeds seed, seed+1, ...), each made seamless,
    evened out (even_out) and pixelized at the project's tile size."""
    from .generate import generate_images
    st = style or assets.project_style()
    camera = camera or camera_for(st)
    size = size or st.tile
    pos, neg = swatch_prompt(terrain, words, camera, st)
    say = say or (lambda text: None)
    say(f"Drawing {count} {terrain} texture(s) on the GPU…")
    pics = generate_images(pos, neg, client=client, seed=seed, count=count, aspect=1.0, **gen_opts)
    out = []
    for i, (img, s) in enumerate(pics):
        say(f"Making texture {i + 1} of {len(pics)} seamless…")
        raw = np.asarray(img.convert("RGB"))
        pic = seamless(raw, client=client, prompt=pos, negative=neg, seed=s, inpaint=inpaint_seams)
        if even:
            pic = even_out(pic)
        out.append(Swatch(terrain, pixel_texture(pic, size, st), pic, s, pos, raw))
    return out


# --------------------------------------------------------------------------- shared helpers

def _shade(rgb, k: float) -> np.ndarray:
    return np.clip(np.asarray(rgb, np.float32) * k, 0, 255).astype(np.uint8)


def _touch(mask: np.ndarray, other: np.ndarray) -> np.ndarray:
    """mask pixels with a 4-neighbour in `other` (outside the tile = edge-replicated)."""
    p = np.pad(other, 1, mode="edge")
    return mask & (p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:])


def _tapered_jitter(t: int, amp: float, rng) -> np.ndarray:
    """A wobble along a tile edge that is 0 at both ends (so neighbouring tiles agree)."""
    from .maptiles import _periodic_noise
    n = _periodic_noise(t, max(2, t // 8), rng)[0] - 0.5
    return amp * 2 * n * np.sin(np.pi * (np.arange(t) + 0.5) / t)


# --------------------------------------------------------------------------- 47-tile blob sets

N, NE, E, SE, S, SW, W, NW = 1, 2, 4, 8, 16, 32, 64, 128
BITS8 = ((N, "top_side"), (NE, "top_right_corner"), (E, "right_side"), (SE, "bottom_right_corner"),
         (S, "bottom_side"), (SW, "bottom_left_corner"), (W, "left_side"), (NW, "top_left_corner"))
_CORNER_SIDES = {NE: (N, E), SE: (S, E), SW: (S, W), NW: (N, W)}


def canonical(mask: int) -> int:
    """A corner only counts when both its sides are the terrain too (47 distinct tiles)."""
    for c, (a, b) in _CORNER_SIDES.items():
        if mask & c and not (mask & a and mask & b):
            mask &= ~c
    return mask


BLOB_MASKS = sorted({canonical(m) for m in range(256)})          # 47, the last is 255 (full)


def blob_region(t: int, mask: int, margins: dict, jx: np.ndarray, jy: np.ndarray) -> np.ndarray:
    """True where the terrain is in a blob tile. margins {N, E, S, W: px} toward open sides;
    inner corners are elliptic notches with the two sides' margins, so every tile edge meets
    its neighbour's exactly."""
    y = np.arange(t)[:, None] + 0.5
    x = np.arange(t)[None, :] + 0.5
    mn, me, ms, mw = margins[N], margins[E], margins[S], margins[W]
    inside = np.ones((t, t), bool)
    if not mask & N:
        inside &= y > mn + jx[None, :]
    if not mask & S:
        inside &= (t - y) > ms + jx[None, :]
    if not mask & W:
        inside &= x > mw + jy[:, None]
    if not mask & E:
        inside &= (t - x) > me + jy[:, None]
    for c, (vs, hs) in _CORNER_SIDES.items():
        cx = t if hs == E else 0
        cy = t if vs == S else 0
        rx, ry = margins[hs], margins[vs]
        if mask & vs and mask & hs and not mask & c and rx > 0 and ry > 0:
            inside &= ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 >= 1
        if not mask & vs and not mask & hs and rx > 0 and ry > 0:   # round outer corners
            r = min(rx, ry) * 0.8
            ox = (t - rx - r) if hs == E else (rx + r)
            oy = (t - ry - r) if vs == S else (ry + r)
            box = ((x > ox) if hs == E else (x < ox)) & ((y > oy) if vs == S else (y < oy))
            inside &= ~(box & (np.hypot(x - ox, y - oy) > r))
    return inside


def water_tile(base: np.ndarray, water: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Base with water where region is: a light foam line at the shore, shallow water behind
    it, wet (darker) ground in front."""
    out = base.copy()
    out[region] = water[region]
    if region.all() or not region.any():
        return out
    cols = water[..., :3].reshape(-1, 3).astype(np.float32)
    light = cols[cols.sum(1).argmax()]
    foam = np.clip(light * 0.4 + 255 * 0.6, 0, 255).astype(np.uint8)
    shore = _touch(region, ~region)
    out[shore, :3] = foam
    shallow = _touch(region & ~shore, shore)
    out[shallow, :3] = np.clip(out[shallow, :3] * 0.5 + light * 0.5, 0, 255).astype(np.uint8)
    wet = _touch(~region, region)
    out[wet, :3] = _shade(out[wet, :3], 0.72)
    return out


def cliff_tile(base: np.ndarray, rock: np.ndarray, region: np.ndarray, face: np.ndarray,
               edge: tuple[int, int, int]) -> np.ndarray:
    """Raised ground (the base, a little lighter) where region is; the rock face (darker, with
    strata) where face is; rims, outline and a shadow on the low ground."""
    t = base.shape[0]
    out = base.copy()
    out[region, :3] = _shade(base[region, :3], 1.15)
    if face.any():
        y = np.arange(t)[:, None] + np.zeros((1, t), int)
        f = _shade(rock[..., :3], 0.62)
        strata = (y % 4 == 3)
        f[strata] = _shade(f[strata], 0.8)
        out[face, :3] = f[face]
        out[_touch(face, region), :3] = edge                       # lip under the top
        out[face & (y == t - 1), :3] = edge                        # foot of the face
        out[_touch(region, face), :3] = _shade(base[_touch(region, face), :3], 1.25)
    low = ~region & ~face
    rim = _touch(region, low)
    out[rim, :3] = edge
    shadow = _touch(low, region | face)
    out[shadow, :3] = _shade(out[shadow, :3], 0.7)
    return out


def _blob_geometry(t: int, role: str, seed: int):
    rng = np.random.default_rng(zlib.crc32(f"blob/{t}/{seed}".encode()))
    if role == "cliff":
        m = {N: max(2, round(t * 0.12)), E: max(2, round(t * 0.12)), W: max(2, round(t * 0.12)),
             S: max(3, round(t * 0.45))}
    else:
        m = {k: max(2, round(t * 0.25)) for k in (N, E, S, W)}
    jx = _tapered_jitter(t, t * 0.07, rng)
    jy = _tapered_jitter(t, t * 0.07, rng)
    return m, jx, jy


def blob_tile(role: str, mask: int, base: np.ndarray, over: np.ndarray, edge, geo) -> np.ndarray:
    m, jx, jy = geo
    t = base.shape[0]
    region = blob_region(t, mask, m, jx, jy)
    if role == "cliff":
        no_face = dict(m)
        no_face[S] = 0
        face = blob_region(t, mask, no_face, jx, jy) & ~region
        return cliff_tile(base, over, region, face, edge)
    return water_tile(base, over, region)


# --------------------------------------------------------------------------- 16-tile line sets

LN, LE, LS, LW = 1, 2, 4, 8                    # connected sides (fence / wall autotiling)
SIDES4 = ((LN, "top_side"), (LE, "right_side"), (LS, "bottom_side"), (LW, "left_side"))


def _boxes_fence(t: int, mask: int) -> list[tuple[str, int, int, int, int]]:
    """[(part, x0, y0, x1, y1)] of a fence tile, drawn in order."""
    pw = max(2, t // 6)
    px0 = t // 2 - pw // 2
    py0, py1 = round(t * 0.25), round(t * 0.72)
    rt = max(2, t // 10)
    r1, r2 = round(t * 0.36), round(t * 0.55)
    vb = max(2, pw - 1)
    vx0 = t // 2 - vb // 2
    out = []
    if mask & LN:
        out.append(("rail_v", vx0, 0, vx0 + vb, t // 2))
    if mask & LS:
        out.append(("rail_v", vx0, t // 2, vx0 + vb, t))
    for r in (r1, r2):
        if mask & LW:
            out.append(("rail", 0, r, t // 2, r + rt))
        if mask & LE:
            out.append(("rail", t // 2, r, t, r + rt))
    out.append(("post", px0, py0, px0 + pw, py1))
    return out


def _boxes_wall(t: int, mask: int) -> list[tuple[str, int, int, int, int]]:
    a, b = round(t * 0.22), t - round(t * 0.22)
    ty0, ty1, fy1 = round(t * 0.12), round(t * 0.5), round(t * 0.8)
    out = []
    if mask & LW:
        out.append(("front", 0, ty1, t // 2, fy1))
    if mask & LE:
        out.append(("front", t // 2, ty1, t, fy1))
    if not mask & LS:
        out.append(("front", a, ty1, b, fy1))
    if mask & LW:
        out.append(("top", 0, ty0, t // 2, ty1))
    if mask & LE:
        out.append(("top", t // 2, ty0, t, ty1))
    out.append(("top", a, 0 if mask & LN else ty0, b, t if mask & LS else ty1))
    return out


def line_boxes(kind: str, t: int, mask: int):
    return (_boxes_wall if kind == "wall" else _boxes_fence)(t, mask)


def line_tile(kind: str, mask: int, material: np.ndarray, edge) -> np.ndarray:
    """A transparent fence/wall tile for one set of connected sides, coloured from the
    material texture; outlined, open only where it continues into the next tile."""
    t = material.shape[0]
    out = np.zeros((t, t, 4), np.uint8)
    mat = material[..., :3]
    y = np.arange(t)[:, None] + np.zeros((1, t), int)
    for part, x0, y0, x1, y1 in line_boxes(kind, t, mask):
        sl = (slice(y0, y1), slice(x0, x1))
        if part == "front":
            px = _shade(mat[sl], 0.68)
            px[(y[sl] - y0) % 4 == 3] = _shade(px[(y[sl] - y0) % 4 == 3], 0.75)
        elif part == "top":
            px = _shade(mat[sl], 1.12)
        elif part == "post":
            px = mat[sl].copy()
            px[:1] = _shade(px[:1], 1.3)                            # lit top of the post
            px[:, -1:] = _shade(px[:, -1:], 0.75)
        else:
            px = _shade(mat[sl], 1.05)
            px[-1:] = _shade(px[-1:], 0.78)
        out[sl][..., :3] = px
        out[sl][..., 3] = 255
    solid = out[..., 3] > 0
    p = np.pad(solid, 1)
    line = ~solid & (p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:])
    out[line, :3] = edge
    out[line, 3] = 255
    return out


# --------------------------------------------------------------------------- the tileset

@dataclass
class GenTileset:
    """A generated tileset: atlas + what each tile is, ready to write for Godot / Tiled / Unity."""
    terrain: str
    kind: str                    # corner | blob | line | side | iso
    camera: str
    tile: int
    atlas: np.ndarray            # RGBA (iso: floors; blocks in extra_atlas)
    tiles: list                  # [{"col", "row", "terrain", "bits": {godot peering name: id},
    #                               "mask"?, "variant"?, "probability"?}]
    names: list                  # terrain names in the Godot terrain set (0 = base)
    colours: list
    texture: np.ndarray          # the terrain's seamless tile
    edge: tuple
    base: str = "grass"
    example: np.ndarray | None = None      # (rows, cols, 2) atlas coords per cell; -1 = empty
    under: np.ndarray | None = None        # line sets: example ground cells (rows, cols, 2)
    built: object = None                   # maptiles.TerrainSet / tilesets.SideSet / IsoSet
    extra_atlas: np.ndarray | None = None  # iso blocks
    meta: dict = field(default_factory=dict)

    @property
    def columns(self) -> int:
        return self.atlas.shape[1] // self.tile

    def lookup(self) -> dict:
        """mask (blob/line) or corners (corner) -> [(col, row), ...]"""
        out: dict = {}
        for t in self.tiles:
            key = t.get("mask", t.get("corners"))
            if key is not None:
                out.setdefault(key if not isinstance(key, list) else tuple(key), []).append(
                    (t["col"], t["row"]))
        return out

    def choices(self) -> dict:
        """lookup() with weights: key -> ([(col, row), ...], [probability, ...])."""
        return tile_choices(self.tiles)


def tile_choices(tiles: list) -> dict:
    """tileset.json "tiles" -> {mask / corners / "ground": ([(col, row), ...], [weight, ...])}."""
    out: dict = {}
    for t in tiles:
        key = "ground" if t.get("ground") else t.get("mask", t.get("corners"))
        if key is None:
            continue
        xy, w = out.setdefault(tuple(key) if isinstance(key, list) else key, ([], []))
        xy.append((t["col"], t["row"]))
        w.append(float(t.get("probability", 1.0)))
    return out


def _colour(tex) -> tuple[int, int, int]:
    return tuple(int(c) for c in tex[..., :3].reshape(-1, 3).mean(0))


def _grid_atlas(tiles_img: list[np.ndarray], cols: int, t: int, h: int | None = None) -> np.ndarray:
    h = h or t
    rows = -(-len(tiles_img) // cols)
    atlas = np.zeros((rows * h, cols * t, 4), np.uint8)
    for i, im in enumerate(tiles_img):
        r, c = divmod(i, cols)
        atlas[r * h:(r + 1) * h, c * t:(c + 1) * t] = im
    return atlas


def base_texture(name: str = "grass", size: int = 32, project: assets.Project | None = None,
                 seed: int = 0, even: bool = True) -> np.ndarray:
    """The ground a set blends into: the project's newest saved tileset of that terrain (its
    texture.png at this size), else the built-in look in the project's colours."""
    try:
        for a in assets.list_assets(KIND, project=project):
            p = Path(a.folder) / "texture.png"
            if a.extra.get("terrain") == name and a.extra.get("tile") == size and p.exists():
                return np.asarray(Image.open(p).convert("RGBA")).copy()
    except Exception:  # noqa: BLE001 - no project: fall back to the built-in look
        pass
    style = project.style if project is not None else None
    return builtin_swatch(name, size, seed, style, even).texture


def _snap(img: np.ndarray, style) -> np.ndarray:
    pal = project_palette(style)
    if pal is None:
        return img
    from .cards import snap_image
    return snap_image(img, pal)


def build_tileset(terrain: str, texture: np.ndarray, *, picture: np.ndarray | None = None,
                  base: str | np.ndarray = "grass", camera: str | None = None, variants: int = 3,
                  seed: int = 0, style: assets.Style | None = None,
                  project: assets.Project | None = None, mix: bool = True) -> GenTileset:
    """The terrain's tileset (no GPU). texture: its seamless tile (Swatch.texture); picture:
    the seamless picture it came from (variants come from other pieces of it); base: the
    terrain it blends into (a name -> base_texture, or a texture). variants (1-4): pieces of
    the picture; mix: up to 8 interior tiles per terrain sharing the probability (variety),
    False = the 2026-10-13 way (1-4 shifted interiors, each a third as likely as the plain tile)."""
    st = style or (project.style if project is not None else assets.project_style())
    camera = camera or camera_for(st)
    if camera not in CAMERAS:
        raise ValueError(f"camera must be one of {', '.join(CAMERAS)}")
    t = texture.shape[0]
    variants = max(1, min(4, int(variants)))
    base_name = base if isinstance(base, str) else "ground"
    base_tex = base_texture(base, t, project, seed, mix) if isinstance(base, str) else base
    if base_tex.shape[0] != t:
        raise ValueError(f"The base texture is {base_tex.shape[0]} px; the terrain is {t} px")
    role = role_of(terrain)
    edge = edge_colour(texture)
    if mix:
        count = min(8, 2 * variants + 2)
        crops = _crops(picture, texture, variants - 1, st, seed) if picture is not None else []
        over_vars, over_w = variety(texture, count, crops, seed=seed)
        base_vars, base_w = variety(base_tex, count, seed=seed + 1)
    else:
        over_vars = interior_variants(texture, variants, picture, st, seed)
        base_vars = interior_variants(base_tex, variants, None, st, seed + 1)
        over_w = [1.0] + [0.35] * (len(over_vars) - 1)
        base_w = [1.0] + [0.35] * (len(base_vars) - 1)
    if camera != "topdown":                          # side / iso sets keep 1-4 picture variants
        over_vars, over_w = over_vars[:variants], over_w[:variants]
    if role == "line":
        if camera == "iso":
            raise ValueError("Fences and walls are made for top-down and side-scroller maps; in an "
                             "isometric project make them as props.")
        ts = _line_set(terrain, texture, base_vars[:4], edge, camera, base_name, seed, base_w[:4])
    elif camera == "side":
        ts = _side_set(terrain, role, over_vars, base_tex, base_name, edge, seed)
    elif camera == "iso":
        ts = _iso_set(terrain, over_vars, base_tex, base_name, edge)
    elif role == "ground":
        ts = _corner_set(terrain, over_vars, base_vars, base_name, edge, seed, over_w, base_w)
    else:
        ts = _blob_set(terrain, role, over_vars, base_vars, base_name, edge, seed, over_w, base_w)
    ts.atlas = _snap(ts.atlas, st)
    if ts.extra_atlas is not None:
        ts.extra_atlas = _snap(ts.extra_atlas, st)
    ts.meta.update(role=role, variants=variants, seed=seed, mix=mix)
    return ts


def _crops(picture, texture, n: int, style, seed: int) -> list[np.ndarray]:
    """n more tiles from other pieces of the seamless picture, in the texture's colours."""
    rng = np.random.default_rng(seed + 5)
    pal = np.unique(texture[..., :3].reshape(-1, 3), axis=0)
    h, w = picture.shape[:2]
    return [pixel_texture(np.roll(picture, (int(rng.integers(h)), int(rng.integers(w))), axis=(0, 1)),
                          texture.shape[0], style, pal) for _ in range(max(0, n))]


def _prob(w: float) -> dict:
    return {} if w == 1 else {"probability": round(float(w), 3)}


def _corner_set(terrain, over_vars, base_vars, base_name, edge, seed, over_w=None,
                base_w=None) -> GenTileset:
    from .maptiles import CORNERS, Terrain, build_terrain_set
    t = over_vars[0].shape[0]
    base_t = Terrain(base_name, base_vars[0], edge_colour(base_vars[0]))
    over_t = Terrain(terrain, over_vars[0], edge)
    ts = build_terrain_set([base_t, over_t], variants=len(base_vars), seed=seed)
    over_w = over_w or [1.0] + [0.35] * (len(over_vars) - 1)
    base_w = base_w or [1.0] + [0.35] * (len(base_vars) - 1)
    for v, tile in enumerate(base_vars):
        ts.atlas[:t, v * t:(v + 1) * t] = tile
        ts.tiles[v]["probability"] = base_w[v]
    extra = over_vars[1:]
    if extra:
        cols = max(ts.atlas.shape[1] // t, 15 + len(extra))
        wide = np.zeros((ts.atlas.shape[0], cols * t, 4), np.uint8)
        wide[:, :ts.atlas.shape[1]] = ts.atlas
        for i, tile in enumerate(extra):
            wide[t:2 * t, (15 + i) * t:(16 + i) * t] = tile
            ts.tiles.append({"col": 15 + i, "row": 1, "terrain": 1, "corners": (1, 1, 1, 1),
                             "probability": over_w[i + 1]})
        ts.atlas = wide
    tiles = []
    for d in ts.tiles:
        tiles.append({"col": d["col"], "row": d["row"], "terrain": d["terrain"],
                      "corners": tuple(d["corners"]), **_prob(d.get("probability", 1)),
                      "bits": {n: int(i) for n, i in zip(CORNERS, d["corners"])}})
    from .maptiles import example_vertex_grid
    vg = example_vertex_grid(16, 10, 4, seed)                    # its lake + its road
    gs = GenTileset(terrain, "corner", "topdown", t, ts.atlas, tiles, [base_name, terrain],
                    ts.colours, over_vars[0], edge, base_name, built=ts)
    gs.example = corner_cells(np.where((vg == 1) | (vg == 3), 1, 0), gs, seed)
    return gs


def _blob_set(terrain, role, over_vars, base_vars, base_name, edge, seed, over_w=None,
              base_w=None) -> GenTileset:
    t = over_vars[0].shape[0]
    over_w = over_w or [1.0] + [0.35] * (len(over_vars) - 1)
    base_w = base_w or [1.0] + [0.35] * (len(base_vars) - 1)
    geo = _blob_geometry(t, role, seed)
    imgs, tiles = [], []
    cols = 8
    for v, tile in enumerate(base_vars):            # row 0: plain base (+ variants)
        imgs.append(tile)
        tiles.append({"col": v, "row": 0, "terrain": 0, "mask": -1, "variant": v,
                      "bits": {n: 0 for _, n in BITS8}, **_prob(base_w[v])})
    imgs += [np.zeros((t, t, 4), np.uint8)] * (cols - len(base_vars))
    for i, m in enumerate(BLOB_MASKS):              # rows 1-6: the 47 blob tiles
        img = blob_tile(role, m, base_vars[0], over_vars[0], edge, geo)
        imgs.append(img)
        r, c = divmod(cols + i, cols)
        tiles.append({"col": c, "row": r, "terrain": 1, "mask": m,
                      "bits": {n: (1 if canonical(m) & b else 0) for b, n in BITS8}})
    # then full-tile variants (a plateau's full tile is its raised base: the base's variants)
    full, weights = (base_vars, base_w) if role == "cliff" else (over_vars, over_w)
    for v, tile in enumerate(full[1:], start=1):
        img = blob_tile(role, 255, tile if role == "cliff" else base_vars[0], tile, edge, geo)
        imgs.append(img)
        r, c = divmod(len(imgs) - 1, cols)
        tiles.append({"col": c, "row": r, "terrain": 1, "mask": 255, "variant": v,
                      **_prob(weights[v]), "bits": {n: 1 for _, n in BITS8}})
    atlas = _grid_atlas(imgs, cols, t)
    gs = GenTileset(terrain, "blob", "topdown", t, atlas, tiles, [base_name, terrain],
                    [_colour(base_vars[0]), _colour(over_vars[0])], over_vars[0], edge, base_name)
    gs.example = blob_cells(example_blob(16, 10, seed), gs, seed)
    return gs


def _line_set(terrain, material, base_vars, edge, camera, base_name, seed, base_w=None) -> GenTileset:
    t = material.shape[0]
    kind = "wall" if terrain == "wall" else "fence"
    imgs, tiles = [], []
    for m in range(16):
        imgs.append(line_tile(kind, m, material, edge))
        tiles.append({"col": m % 4, "row": m // 4, "terrain": 0, "mask": m,
                      "bits": {n: 0 for b, n in SIDES4 if m & b},
                      "boxes": [(x0, y0, x1, y1) for _, x0, y0, x1, y1 in line_boxes(kind, t, m)]})
    for v, tile in enumerate(base_vars):            # row 4: the ground it stands on
        imgs.append(tile)
        tiles.append({"col": v, "row": 4, "ground": True, "variant": v,
                      **_prob((base_w or [1.0] * 4)[v])})
    atlas = _grid_atlas(imgs[:16], 4, t)
    ground = _grid_atlas(base_vars, 4, t)
    atlas = np.concatenate([atlas, ground], 0)
    gs = GenTileset(terrain, "line", camera, t, atlas, tiles, [terrain], [_colour(material)],
                    material, edge, base_name)
    gs.example, gs.under = line_cells(example_line(12, 9), gs, seed)
    return gs


def _side_set(terrain, role, over_vars, base_tex, base_name, edge, seed) -> GenTileset:
    from .maptiles import Terrain
    from .tilesets import build_side_set, example_level, side_masks
    t = over_vars[0].shape[0]
    if role == "water":
        return _side_water(terrain, over_vars, edge)
    if terrain in ("grass", "snow"):     # grows on top of the ground (the base, e.g. dirt)
        ground_t, top_t = Terrain(base_name, base_tex, edge_colour(base_tex)), Terrain(terrain, over_vars[0], edge)
        variants = []
    else:
        ground_t, top_t = Terrain(terrain, over_vars[0], edge), Terrain(base_name, base_tex, edge_colour(base_tex))
        variants = over_vars[1:]
    ss = build_side_set(ground_t, top_t, seed)
    atlas = ss.atlas
    tiles = []
    for m, (c, r) in ss.masks.items():
        tiles.append({"col": c, "row": r, "terrain": 0, "mask": m, "bits": {}})
    if variants:                                     # row 5: inside-ground variants
        row = np.zeros((t, atlas.shape[1], 4), np.uint8)
        for i, v in enumerate(variants[:4]):
            row[:, i * t:(i + 1) * t] = v
            tiles.append({"col": i, "row": 5, "terrain": 0, "mask": 15, "variant": i + 1,
                          "probability": 0.35, "bits": {}})
        atlas = np.concatenate([atlas, row], 0)
    solid, plats = example_level(seed=seed)
    masks = side_masks(solid)
    cells = np.full(solid.shape + (2,), -1, int)
    for (y, x), m in np.ndenumerate(masks):
        if m >= 0:
            cells[y, x] = ss.masks[int(m)]
    for x, y, k in plats:
        cells[y, x] = ss.platforms[k]
    return GenTileset(terrain, "side", "side", t, atlas, tiles, [ground_t.name],
                      [_colour(ground_t.texture)], over_vars[0], edge, base_name, cells,
                      built=ss, meta={"variants_side": [(d["col"], d["row"]) for d in tiles
                                                        if d.get("variant")]})


def _side_water(terrain, over_vars, edge) -> GenTileset:
    """Side view water: a surface tile (foam line, see-through air above) + body tiles."""
    t = over_vars[0].shape[0]
    tex = over_vars[0]
    surface = tex.copy()
    lip = max(2, t // 4)
    surface[:lip, :, 3] = 0
    cols = tex[..., :3].reshape(-1, 3).astype(np.float32)
    light = cols[cols.sum(1).argmax()]
    surface[lip, :, :3] = np.clip(light * 0.4 + 255 * 0.6, 0, 255).astype(np.uint8)
    surface[lip + 1, :, :3] = np.clip(surface[lip + 1, :, :3] * 0.5 + light * 0.5, 0, 255).astype(np.uint8)
    surface[lip:, :, 3] = 200
    imgs = [surface] + [v.copy() for v in over_vars]
    for im in imgs[1:]:
        im[..., 3] = 200
    tiles = [{"col": 0, "row": 0, "terrain": 0, "mask": 0,
              "bits": {"left_side": 0, "right_side": 0, "bottom_side": 0}}]
    for i in range(len(over_vars)):
        tiles.append({"col": i + 1, "row": 0, "terrain": 0, "mask": 1, "variant": i,
                      "bits": {"top_side": 0, "left_side": 0, "right_side": 0, "bottom_side": 0},
                      **({"probability": 0.35} if i else {})})
    atlas = _grid_atlas(imgs, len(imgs), t)
    cells = np.full((6, 12, 2), -1, int)
    cells[3, :] = (0, 0)
    for y in (4, 5):
        cells[y, :] = (1, 0)
    return GenTileset(terrain, "side-water", "side", t, atlas, tiles, [terrain], [_colour(tex)],
                      tex, edge, "air", cells)


def _iso_set(terrain, over_vars, base_tex, base_name, edge) -> GenTileset:
    from .maptiles import Terrain
    from .tilesets import build_iso_set
    t = over_vars[0].shape[0]
    width = max(4, t - t % 4)
    terrains = [Terrain(base_name, base_tex, edge_colour(base_tex))]
    terrains += [Terrain(terrain if i == 0 else f"{terrain}_{i + 1}", v, edge)
                 for i, v in enumerate(over_vars)]
    iso = build_iso_set(terrains, width)
    tiles = [{"col": i, "row": 0, "terrain": i, "bits": {}} for i in range(len(terrains))]
    return GenTileset(terrain, "iso", "iso", width, iso.floors, tiles, iso.names,
                      [_colour(x.texture) for x in terrains], over_vars[0], edge, base_name,
                      built=iso, extra_atlas=iso.blocks)


# --------------------------------------------------------------------------- example maps

def example_blob(cols: int = 16, rows: int = 10, seed: int = 0) -> np.ndarray:
    """(rows, cols) bool: a lake / plateau with a bay and a small island of base."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:rows, 0:cols]
    d = ((xx - cols * 0.55) / (cols * 0.32)) ** 2 + ((yy - rows * 0.5) / (rows * 0.36)) ** 2
    on = d + 0.25 * rng.random((rows, cols)) < 1.0
    on[rows // 2, int(cols * 0.55)] = False
    on[:, :2] = False
    return on


def neighbour_mask(on: np.ndarray, y: int, x: int) -> int:
    rows, cols = on.shape
    m = 0
    for bit, (dy, dx) in ((N, (-1, 0)), (NE, (-1, 1)), (E, (0, 1)), (SE, (1, 1)), (S, (1, 0)),
                          (SW, (1, -1)), (W, (0, -1)), (NW, (-1, -1))):
        yy, xx = y + dy, x + dx
        if 0 <= yy < rows and 0 <= xx < cols and on[yy, xx]:
            m |= bit
        elif not (0 <= yy < rows and 0 <= xx < cols):
            m |= bit                                    # past the map edge: carry on
    return canonical(m)


def blob_cells(on: np.ndarray, gs: GenTileset, seed: int = 0) -> np.ndarray:
    table = gs.choices()
    u = cell_noise(*on.shape, seed, "blob")
    out = np.zeros(on.shape + (2,), int)
    for (y, x), v in np.ndenumerate(on):
        out[y, x] = pick(*table[neighbour_mask(on, y, x) if v else -1], u[y, x])
    return out


def corner_cells(on: np.ndarray, gs: GenTileset, seed: int = 0) -> np.ndarray:
    """(rows, cols, 2) atlas coords for a (rows+1, cols+1) corner grid of 0/1 (base / terrain)."""
    table = gs.choices()
    rows, cols = on.shape[0] - 1, on.shape[1] - 1
    u = cell_noise(rows, cols, seed, "corner")
    out = np.zeros((rows, cols, 2), int)
    for y in range(rows):
        for x in range(cols):
            key = (int(on[y, x]), int(on[y, x + 1]), int(on[y + 1, x]), int(on[y + 1, x + 1]))
            out[y, x] = pick(*table.get(key, table[(0, 0, 0, 0)]), u[y, x])
    return out


def example_line(cols: int = 12, rows: int = 9) -> np.ndarray:
    """(rows, cols) bool: a fenced yard with a gate gap, a branch and a cross."""
    on = np.zeros((rows, cols), bool)
    on[1, 1:9] = on[6, 1:9] = True
    on[1:7, 1] = on[1:7, 8] = True
    on[6, 4] = False                                   # the gate
    on[3, 8:11] = True                                 # a branch (T)
    on[2:6, 5] = True
    on[3, 4:7] = True                                  # a cross inside
    on[8, 2:5] = True                                  # a loose run (ends)
    on[8, 10] = True                                   # a lone post
    return on


def line_mask(on: np.ndarray, y: int, x: int) -> int:
    rows, cols = on.shape
    m = 0
    for bit, (dy, dx) in ((LN, (-1, 0)), (LE, (0, 1)), (LS, (1, 0)), (LW, (0, -1))):
        yy, xx = y + dy, x + dx
        if 0 <= yy < rows and 0 <= xx < cols and on[yy, xx]:
            m |= bit
    return m


def line_cells(on: np.ndarray, gs: GenTileset, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    out = np.full(on.shape + (2,), -1, int)
    for (y, x), v in np.ndenumerate(on):
        if v:
            m = line_mask(on, y, x)
            out[y, x] = (m % 4, m // 4)
    ground = gs.choices()["ground"]
    u = cell_noise(*on.shape, seed, "ground")
    under = np.zeros(on.shape + (2,), int)
    for (y, x), v in np.ndenumerate(u):
        under[y, x] = pick(*ground, v)
    return out, under


def render_cells(atlas: np.ndarray, cells: np.ndarray, t: int, under: np.ndarray | None = None) -> np.ndarray:
    rows, cols = cells.shape[:2]
    img = Image.new("RGBA", (cols * t, rows * t), (0, 0, 0, 0))
    for layer in ([under] if under is not None else []) + [cells]:
        for (y, x) in np.ndindex(rows, cols):
            c, r = layer[y, x]
            if c < 0:
                continue
            tile = Image.fromarray(np.ascontiguousarray(atlas[r * t:(r + 1) * t, c * t:(c + 1) * t]))
            img.alpha_composite(tile, (x * t, y * t))
    return np.asarray(img)


def preview(gs: GenTileset) -> np.ndarray:
    """A small sample map built from the set (RGBA)."""
    if gs.kind == "iso":
        from .tilesets import example_iso, render_iso
        floor, blocks = example_iso(8, 8, n=len(gs.names))
        return render_iso(gs.built, floor, blocks)
    return render_cells(gs.atlas, gs.example, gs.tile, gs.under)


# --------------------------------------------------------------------------- writers

def _godot_header(png: str, t: int, h: int | None = None, res_id: str = "TileSetAtlasSource_gen") -> list[str]:
    return ['[gd_resource type="TileSet" load_steps=3 format=3]', "",
            f'[ext_resource type="Texture2D" path="{png}" id="1_atlas"]', "",
            f'[sub_resource type="TileSetAtlasSource" id="{res_id}"]',
            'texture = ExtResource("1_atlas")', f"texture_region_size = Vector2i({t}, {h or t})"]


def _terrain_lines(gs: GenTileset, mode: int) -> list[str]:
    lines = [f"terrain_set_0/mode = {mode}"]
    for i, (name, (r, g, b)) in enumerate(zip(gs.names, gs.colours)):
        lines += [f'terrain_set_0/terrain_{i}/name = "{name}"',
                  f"terrain_set_0/terrain_{i}/color = Color({r / 255:.3f}, {g / 255:.3f}, {b / 255:.3f}, 1)"]
    return lines


# Map-border patterns for the plain base tile: Godot counts cells past the map's edge as empty,
# so base at the edge needs tiles whose outer bits are empty (like maptiles' alternatives).
_BORDERS = (("top_left_corner", "top_side", "top_right_corner"),
            ("bottom_left_corner", "bottom_side", "bottom_right_corner"),
            ("top_left_corner", "left_side", "bottom_left_corner"),
            ("top_right_corner", "right_side", "bottom_right_corner"))


def _border_alternatives() -> list[set]:
    out = [set(b) for b in _BORDERS]
    for a, b in ((0, 2), (0, 3), (1, 2), (1, 3)):                   # map corners
        out.append(set(_BORDERS[a]) | set(_BORDERS[b]))
    return out


def write_godot(gs: GenTileset, out_tres, png_name: str) -> Path:
    """Godot 4 TileSet .tres: terrain set + peering bits for the kind (corner: Match Corners,
    blob: Match Corners and Sides, line/side: Match Sides; iso: floors + blocks)."""
    out_tres = Path(out_tres)
    if gs.kind == "corner":
        from .maptiles import write_godot_terrain_tileset
        return write_godot_terrain_tileset(gs.built, out_tres, png_name)
    if gs.kind == "iso":
        from .tilesets import write_godot_iso_tileset
        blocks = Path(png_name).stem + "_blocks.png"
        return write_godot_iso_tileset(gs.built, out_tres, png_name, blocks)
    if gs.kind == "side":
        from .tilesets import write_godot_side_tileset
        write_godot_side_tileset(gs.built, out_tres, png_name)
        text = out_tres.read_text(encoding="utf-8")
        c15, r15 = gs.built.masks[15]
        src = [ln for ln in text.splitlines() if ln.startswith(f"{c15}:{r15}/0")]
        add = []
        for c, r in gs.meta.get("variants_side", []):
            add += [f"{c}:{r}/0" + ln[len(f"{c15}:{r15}/0"):] for ln in src]
            add.append(f"{c}:{r}/0/probability = 0.35")
        if add:
            text = text.replace("\n\n[resource]", "\n" + "\n".join(add) + "\n\n[resource]", 1)
            out_tres.write_text(text, encoding="utf-8")
        return out_tres
    t, h = gs.tile, gs.tile / 2
    lines = _godot_header(png_name, t)
    for d in gs.tiles:
        if d.get("ground"):
            lines.append(f"{d['col']}:{d['row']}/0 = 0")
            continue
        key = f"{d['col']}:{d['row']}/0"
        lines += [f"{key} = 0", f"{key}/terrain_set = 0", f"{key}/terrain = {d['terrain']}"]
        if "probability" in d:
            lines.append(f"{key}/probability = {d['probability']}")
        for name, tid in d["bits"].items():
            lines.append(f"{key}/terrains_peering_bit/{name} = {tid}")
        for i, (x0, y0, x1, y1) in enumerate(d.get("boxes", [])):
            pts = f"PackedVector2Array({x0 - h:g}, {y0 - h:g}, {x1 - h:g}, {y0 - h:g}, {x1 - h:g}, {y1 - h:g}, {x0 - h:g}, {y1 - h:g})"
            lines.append(f"{key}/physics_layer_0/polygon_{i}/points = {pts}")
    if gs.kind == "blob":
        alts = _border_alternatives()
        lines.append(f"0:0/next_alternative_id = {len(alts) + 1}")
        for alt, empty in enumerate(alts, start=1):
            key = f"0:0/{alt}"
            lines += [f"{key} = {alt}", f"{key}/terrain_set = 0", f"{key}/terrain = 0"]
            lines += [f"{key}/terrains_peering_bit/{n} = 0" for _, n in BITS8 if n not in empty]
    lines += ["", "[resource]", f"tile_size = Vector2i({t}, {t})"]
    if gs.kind == "line":
        lines.append("physics_layer_0/collision_layer = 1")
    lines += _terrain_lines(gs, {"blob": 0, "line": 2, "side-water": 2}[gs.kind])
    lines += ['sources/0 = SubResource("TileSetAtlasSource_gen")', ""]
    out_tres.write_text("\n".join(lines), encoding="utf-8")
    return out_tres


def _wangid(gs: GenTileset, d: dict) -> str | None:
    """Tiled wangid: top, top-right, right, bottom-right, bottom, bottom-left, left, top-left;
    colours 1-based, 0 = none."""
    if gs.kind == "corner":
        tl, tr, bl, br = d["corners"]
        return f"0,{tr + 1},0,{br + 1},0,{bl + 1},0,{tl + 1}"
    if gs.kind == "blob":
        m = 0 if d["mask"] == -1 else d["mask"]
        order = (N, NE, E, SE, S, SW, W, NW)
        return ",".join("2" if (d["mask"] != -1 and m & b) else "1" for b in order)
    if gs.kind in ("line", "side", "side-water") and "mask" in d:
        m = d["mask"]
        if gs.kind == "side-water":
            m = 15 if m else 14
        bits = (LN, LE, LS, LW) if gs.kind == "line" else (1, 2, 4, 8)
        vals = ["1" if m & b else "0" for b in bits]
        return f"{vals[0]},0,{vals[1]},0,{vals[2]},0,{vals[3]},0"
    return None


def write_tsx(gs: GenTileset, out_tsx, png_name: str) -> Path:
    """Tiled tileset with a wang set (corner / mixed / edge) so Tiled's terrain brush paints it."""
    from xml.sax.saxutils import quoteattr
    out_tsx = Path(out_tsx)
    h_px, w_px = gs.atlas.shape[:2]
    th = gs.tile // 2 if gs.kind == "iso" else gs.tile
    cols = w_px // gs.tile
    count = cols * (h_px // th)
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             f'<tileset version="1.10" tiledversion="1.11.0" name={quoteattr(gs.terrain)} '
             f'tilewidth="{gs.tile}" tileheight="{th}" tilecount="{count}" columns="{cols}">',
             f' <image source={quoteattr(png_name)} width="{w_px}" height="{h_px}"/>']
    for d in gs.tiles:                               # variants: Tiled's terrain fill picks by weight
        if d.get("probability", 1) != 1:
            lines.append(f' <tile id="{d["row"] * cols + d["col"]}" probability="{d["probability"]:g}"/>')
    wtype = {"corner": "corner", "blob": "mixed"}.get(gs.kind, "edge")
    tiles = [(d, _wangid(gs, d)) for d in gs.tiles]
    tiles = [(d, w) for d, w in tiles if w]
    if tiles:
        names = gs.names if gs.kind in ("corner", "blob") else [gs.names[0]]
        lines.append(f' <wangsets>\n  <wangset name={quoteattr(gs.terrain)} type="{wtype}" tile="-1">')
        for name, (r, g, b) in zip(names, gs.colours):
            lines.append(f'   <wangcolor name={quoteattr(name)} color="#{r:02x}{g:02x}{b:02x}" '
                         f'tile="-1" probability="1"/>')
        for d, w in tiles:
            lines.append(f'   <wangtile tileid="{d["row"] * cols + d["col"]}" wangid="{w}"/>')
        lines.append("  </wangset>\n </wangsets>")
    lines.append("</tileset>")
    out_tsx.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_tsx


def write_tmj(gs: GenTileset, out_tmj, tsx_name: str) -> Path | None:
    """The example map as a Tiled map (orthogonal sets)."""
    if gs.example is None:
        return None
    cols = gs.columns
    rows_, cols_ = gs.example.shape[:2]

    def data(cells):
        return [0 if c < 0 else 1 + r * cols + c for c, r in cells.reshape(-1, 2).tolist()]
    layers = []
    if gs.under is not None:
        layers.append(("Ground", gs.under))
    layers.append(("Terrain" if gs.kind != "line" else gs.terrain.title(), gs.example))
    doc = {"type": "map", "version": "1.10", "tiledversion": "1.11.0", "orientation": "orthogonal",
           "renderorder": "right-down", "width": cols_, "height": rows_, "tilewidth": gs.tile,
           "tileheight": gs.tile, "infinite": False, "nextlayerid": len(layers) + 1,
           "nextobjectid": 1, "tilesets": [{"firstgid": 1, "source": tsx_name}],
           "layers": [{"type": "tilelayer", "id": i + 1, "name": n, "width": cols_, "height": rows_,
                       "x": 0, "y": 0, "opacity": 1, "visible": True, "data": data(c)}
                      for i, (n, c) in enumerate(layers)]}
    out_tmj = Path(out_tmj)
    out_tmj.write_text(json.dumps(doc), encoding="utf-8")
    return out_tmj


def _sheet(gs: GenTileset) -> tuple[np.ndarray, dict]:
    """sheet.png + sheet.json (frames hash) for exports.py (Unity slices it per tile)."""
    t = gs.tile
    frames = {}
    if gs.kind == "iso":
        img = np.concatenate([gs.atlas, gs.extra_atlas], 0)
        h, bh = gs.atlas.shape[0], gs.extra_atlas.shape[0]
        for i, name in enumerate(gs.names):
            frames[f"{name}_floor"] = {"frame": {"x": i * t, "y": 0, "w": t, "h": h}, "duration": 100}
            frames[f"{name}_block"] = {"frame": {"x": i * t, "y": h, "w": t, "h": bh}, "duration": 100}
    else:
        img = gs.atlas
        rows, cols = img.shape[0] // t, img.shape[1] // t
        for r in range(rows):
            for c in range(cols):
                if img[r * t:(r + 1) * t, c * t:(c + 1) * t, 3].any():
                    frames[f"{gs.terrain}_r{r}_c{c}"] = {"frame": {"x": c * t, "y": r * t, "w": t, "h": t},
                                                         "duration": 100}
    return img, {"frames": frames, "meta": {"image": "sheet.png", "size": {"w": img.shape[1],
                                                                           "h": img.shape[0]}}}


def layout_json(gs: GenTileset) -> dict:
    def clean(d):
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}
    return {"terrain": gs.terrain, "kind": gs.kind, "camera": gs.camera, "tile": gs.tile,
            "base": gs.base, "terrains": gs.names, "edge": list(gs.edge),
            "columns": gs.columns, "tiles": [clean(d) for d in gs.tiles],
            "masks": {"corner": "corners tl, tr, bl, br = terrain ids",
                      "blob": "N 1, NE 2, E 4, SE 8, S 16, SW 32, W 64, NW 128 (corners only "
                              "with both sides); -1 = plain base",
                      "line": "connected sides N 1, E 2, S 4, W 8 (atlas col m % 4, row m // 4)",
                      "side": "ground neighbours top 1, right 2, bottom 4, left 8"}.get(gs.kind, ""),
            **({"example": gs.example.tolist()} if gs.example is not None else {})}


def write_tileset(gs: GenTileset, folder, stem: str = "tileset") -> list[Path]:
    """Every file of a set into folder (see the module docstring)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    png = folder / f"{stem}.png"
    Image.fromarray(gs.atlas, "RGBA").save(png)
    written = [png]
    if gs.extra_atlas is not None:
        bp = folder / f"{stem}_blocks.png"
        Image.fromarray(gs.extra_atlas, "RGBA").save(bp)
        written.append(bp)
    written.append(write_godot(gs, folder / f"{stem}.tres", png.name))
    tsx = write_tsx(gs, folder / f"{stem}.tsx", png.name)
    written.append(tsx)
    tmj = write_tmj(gs, folder / "example.tmj", tsx.name)
    if tmj:
        written.append(tmj)
    j = folder / f"{stem}.json"
    j.write_text(json.dumps(layout_json(gs), indent=1), encoding="utf-8")
    written.append(j)
    tex = folder / "texture.png"
    Image.fromarray(gs.texture, "RGBA").save(tex)
    ex = folder / "example.png"
    pv = preview(gs)
    Image.fromarray(pv, "RGBA").save(ex)
    sheet, meta = _sheet(gs)
    Image.fromarray(sheet, "RGBA").save(folder / "sheet.png")
    (folder / "sheet.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    written += [tex, ex, folder / "sheet.png", folder / "sheet.json"]
    return written


def save_tileset(gs: GenTileset, name: str | None = None, *, project: assets.Project | None = None,
                 swatch: Swatch | None = None, words: str = "") -> assets.Asset:
    """Store the set as a "tileset" asset in the project."""
    register_kind()
    name = name or f"{gs.terrain.title()} tiles"
    a = assets.new_asset(KIND, name, project, terrain=gs.terrain, base=gs.base, camera=gs.camera,
                         set=gs.kind, role=gs.meta.get("role", role_of(gs.terrain)), tile=gs.tile,
                         edge=list(gs.edge), words=words,
                         prompt=(swatch.prompt if swatch else ""), seed=(swatch.seed if swatch else 0),
                         variants=gs.meta.get("variants", 1))
    write_tileset(gs, a.folder)
    if swatch is not None and swatch.raw is not None:
        Image.fromarray(swatch.raw).save(Path(a.folder) / "swatch.png")
    pv = preview(gs)
    th = pv[:min(pv.shape[0], 8 * gs.tile), :min(pv.shape[1], 8 * gs.tile)]
    Image.fromarray(th, "RGBA").save(a.thumb)
    a.tags = sorted(set(a.tags) | {gs.terrain, gs.camera})
    a.save()
    return a


def make(terrain: str, words: str = "", *, project: assets.Project | None = None, name: str | None = None,
         gpu: bool = True, pick: int = 0, count: int = 3, variants: int = 3, base: str = "grass",
         client=None, seed: int | None = None, **gen_opts) -> assets.Asset:
    """One call: swatches (GPU, or the built-in look) -> pick one -> set -> saved asset."""
    st = project.style if project is not None else assets.project_style()
    if gpu:
        sw = make_swatches(terrain, words, count=count, client=client, seed=seed, style=st, **gen_opts)
        chosen = sw[min(pick, len(sw) - 1)]
    else:
        chosen = builtin_swatch(terrain, st.tile, seed or 0, st)
    gs = build_tileset(terrain, chosen.texture, picture=chosen.picture, base=base, variants=variants,
                       seed=chosen.seed, style=st, project=project)
    return save_tileset(gs, name, project=project, swatch=chosen, words=words)


# --------------------------------------------------------------------------- map editor exports

def project_textures(project: assets.Project | None = None, size: int = 32) -> dict:
    """{map terrain (maplib/diorama name): (texture RGBA, edge colour)} from the project's
    saved top-down ground/water tilesets at this tile size; the newest one wins."""
    out: dict = {}
    try:
        items = assets.list_assets(KIND, project=project)
    except Exception:  # noqa: BLE001
        return out
    for a in items:                                    # newest first
        ex = a.extra
        name = MAP_TERRAIN.get(ex.get("terrain", ""))
        p = Path(a.folder) / "texture.png"
        if (not name or name in out or ex.get("camera", "topdown") != "topdown"
                or ex.get("role") not in ("ground", "water") or not p.exists()):
            continue
        tex = np.asarray(Image.open(p).convert("RGBA")).copy()
        if tex.shape[0] != size:
            continue
        out[name] = (tex, tuple(ex.get("edge") or edge_colour(tex)))
    return out


def project_tilesets(project: assets.Project | None = None) -> list[assets.Asset]:
    try:
        return assets.list_assets(KIND, project=project)
    except Exception:  # noqa: BLE001
        return []


def rebuild_tileset(asset: assets.Asset, *, save: bool = False, mix: bool = True,
                    keep_texture: bool = False) -> GenTileset:
    """A saved set built again on the CPU with today's code: from its swatch.png (the GPU
    picture, made seamless and evened out again) or the built-in look. keep_texture: the saved
    texture.png stays the tile (a GPU seam repaint can't be redone on the CPU), the picture
    only gives interiors. save=True rewrites the asset's files (and asset.json's variants/mix)."""
    ex = asset.extra
    folder = Path(asset.folder)
    proj = assets.Project(folder.parent.parent.name, folder.parent.parent)
    st = proj.style
    t = int(ex.get("tile") or st.tile)
    terrain = ex.get("terrain", "grass")
    raw = folder / "swatch.png"
    if raw.exists():
        pic = seamless(np.asarray(Image.open(raw).convert("RGB")))
        pic = even_out(pic) if mix else pic
        tex = pixel_texture(pic, t, st)
        if keep_texture and (folder / "texture.png").exists():
            tex = np.asarray(Image.open(folder / "texture.png").convert("RGBA")).copy()
    else:
        pic, tex = None, builtin_swatch(terrain, t, int(ex.get("seed", 0)), st, even=mix).texture
    gs = build_tileset(terrain, tex, picture=pic, base=ex.get("base", "grass"),
                       camera=ex.get("camera", "topdown"), variants=int(ex.get("variants", 3)),
                       seed=int(ex.get("seed", 0)), style=st, project=proj, mix=mix)
    if save:
        write_tileset(gs, folder)
        asset.extra.update(variants=gs.meta.get("variants", 3), mix=mix)
        asset.save()
    return gs


# --------------------------------------------------------------------------- the map look

LINE_KEYS = ("fence", "wall")


def look_key(extra: dict) -> str | None:
    """What a saved set draws on a map: a map terrain (diorama name), "cliff" or a prop kind."""
    terrain = extra.get("terrain", "")
    role = extra.get("role") or role_of(terrain)
    if role in ("ground", "water"):
        return MAP_TERRAIN.get(terrain)
    if role == "cliff":
        return "cliff"
    return terrain if role == "line" and terrain in LINE_KEYS else None


def look_options(project: assets.Project | None = None, tile: int | None = None) -> dict:
    """{look key: [saved top-down sets for it, newest first]} at the project's tile size."""
    if tile is None:
        tile = (project.style if project is not None else assets.project_style()).tile
    out: dict = {}
    for a in project_tilesets(project):
        ex = a.extra
        key = look_key(ex)
        if (key and ex.get("camera", "topdown") == "topdown" and int(ex.get("tile", 0)) == tile
                and (a.folder / "tileset.json").exists() and (a.folder / "tileset.png").exists()):
            out.setdefault(key, []).append(a)
    return out


@dataclass
class LookSet:
    """A saved set loaded for drawing maps."""
    key: str
    asset: assets.Asset
    kind: str                    # corner | blob | line
    role: str
    tile: int
    base: str                    # the map terrain its plain base is
    atlas: np.ndarray
    overlay: np.ndarray          # drawn over other tiles: base pixels clear, shading as black
    choices: dict                # tile_choices

    @property
    def name(self) -> str:
        return self.asset.folder.name


def _over_base(src: np.ndarray, base: np.ndarray, role: str) -> np.ndarray:
    """A transition tile minus its plain base: same pixels clear, darkened ones black at that
    strength (they darken whatever is under them), a plateau's lightening dropped."""
    s, b = src[..., :3].astype(np.float32), base[..., :3].astype(np.float32)
    same = (src[..., :3] == base[..., :3]).all(-1)
    k = s / np.maximum(b, 1)
    even = (k.max(-1) - k.min(-1) < 0.08) & (b.min(-1) >= 12)
    km = k.mean(-1)
    shade = ~same & even & (km > 0.3) & (km < 0.97)
    out = src.copy()
    out[same, 3] = 0
    out[shade, :3] = 0
    out[shade, 3] = np.round((1 - km[shade]) * 255).astype(np.uint8)
    if role == "cliff":
        out[~same & even & (km > 1.03) & (km < 1.6), 3] = 0
    return out


def overlay_atlas(atlas: np.ndarray, choices: dict, t: int, kind: str, role: str) -> np.ndarray:
    """The set as drawn over a map's ground: transition tiles via _over_base, full tiles whole
    (not a cliff's: its plateau is the ground itself), plain base tiles gone; line tiles are
    see-through already (their ground row goes)."""
    out = np.zeros_like(atlas)

    def cell(c, r):
        return slice(r * t, (r + 1) * t), slice(c * t, (c + 1) * t)
    if kind == "line":
        for key, (xys, _) in choices.items():
            for c, r in xys if key != "ground" else ():
                out[cell(c, r)] = atlas[cell(c, r)]
        return out
    plain, full = (-1, 255) if kind == "blob" else ((0, 0, 0, 0), (1, 1, 1, 1))
    if plain not in choices:
        return atlas.copy()
    base = atlas[cell(*choices[plain][0][0])]
    for key, (xys, _) in choices.items():
        for c, r in xys if key != plain else ():
            if key == full:
                if role != "cliff":
                    out[cell(c, r)] = atlas[cell(c, r)]
            else:
                out[cell(c, r)] = _over_base(atlas[cell(c, r)], base, role)
    return out


def load_look_set(asset: assets.Asset, key: str | None = None) -> LookSet | None:
    folder = Path(asset.folder)
    try:
        layout = json.loads((folder / "tileset.json").read_text(encoding="utf-8"))
        atlas = np.asarray(Image.open(folder / "tileset.png").convert("RGBA")).copy()
    except (OSError, ValueError):
        return None
    kind = layout.get("kind")
    if kind not in ("corner", "blob", "line"):
        return None
    role = asset.extra.get("role") or role_of(layout.get("terrain", ""))
    t = int(layout["tile"])
    choices = tile_choices(layout["tiles"])
    base = layout.get("base", "grass")
    return LookSet(key or look_key(asset.extra), asset, kind, role, t, MAP_TERRAIN.get(base, base),
                   atlas, overlay_atlas(atlas, choices, t, kind, role), choices)


@dataclass
class LookLayer:
    key: str                     # "ground" or the set's look key
    atlas: np.ndarray            # what the layer draws with
    cells: np.ndarray            # (rows, cols, 2) atlas col, row; -1 = nothing
    set: LookSet | None = None


def _tiles_of(atlas: np.ndarray, t: int, xys) -> list[np.ndarray]:
    return [atlas[r * t:(r + 1) * t, c * t:(c + 1) * t] for c, r in xys]


_DIRS8 = ((N, (-1, 0)), (NE, (-1, 1)), (E, (0, 1)), (SE, (1, 1)), (S, (1, 0)), (SW, (1, -1)),
          (W, (0, -1)), (NW, (-1, -1)))


class MapLook:
    """A map drawn with the project's saved top-down tilesets (one per key: a map terrain,
    "cliff", "fence", "wall"); built-in textures for terrains none covers. layers() is what the
    editor shows and what godot-tilemap / tiled write, cell for cell:
      ground   every cell's plain tiles by weight (a shore cell: the terrain around it),
      corner   path / road / sand sets reach into the cells around theirs (Godot Match Corners),
      blob     shores inside the water cells, cliffs where the ground drops 2+ levels,
      line     fence / wall props by connected sides."""

    def __init__(self, sets: dict, tile: int):
        from .diorama import TERRAINS as MAP
        self.sets = sets
        self.tile = tile
        self.terrains = list(MAP)
        self._plain = {name: self.plain(name) for name in self.terrains}
        n = max(len(v[0]) for v in self._plain.values())
        self.ground_atlas = np.zeros((len(self.terrains) * tile, n * tile, 4), np.uint8)
        for i, name in enumerate(self.terrains):
            for j, tl in enumerate(self._plain[name][0]):
                self.ground_atlas[i * tile:(i + 1) * tile, j * tile:(j + 1) * tile] = tl

    @classmethod
    def for_project(cls, project: assets.Project | None = None, choices: dict | None = None,
                    tile: int | None = None) -> "MapLook | None":
        """choices {key: set folder name, "" = built-in}; a key left out (or a set that is
        gone) = its newest set. None when nothing applies."""
        if tile is None:
            tile = (project.style if project is not None else assets.project_style()).tile
        sets = {}
        for key, items in look_options(project, tile).items():
            want = (choices or {}).get(key)
            if want == "":
                continue
            ls = load_look_set(next((a for a in items if a.folder.name == want), items[0]), key)
            if ls is not None and ls.tile == tile:
                sets[key] = ls
        return cls(sets, tile) if sets else None

    @property
    def props(self) -> set:
        """Prop kinds the look draws as tiles (their icons stay out)."""
        return {k for k in LINE_KEYS if k in self.sets}

    def plain(self, terrain: str) -> tuple[list, list]:
        """(tiles, weights) a plain cell of this terrain is drawn with: its own set's full
        tiles, else the plain base of the newest set over it, else the built-in texture."""
        own = self.sets.get(terrain)
        if own is not None and own.kind in ("corner", "blob") and own.role != "cliff":
            xys, w = own.choices[(1, 1, 1, 1) if own.kind == "corner" else 255]
            return _tiles_of(own.atlas, own.tile, xys), w
        for s in sorted(self.sets.values(), key=lambda s: (s.kind == "line", -s.asset.created)):
            key = {"corner": (0, 0, 0, 0), "blob": -1, "line": "ground"}[s.kind]
            if s.base == terrain and key in s.choices:
                xys, w = s.choices[key]
                return _tiles_of(s.atlas, s.tile, xys), w
        from .maplib import terrain_textures
        return [terrain_textures(self.tile)[terrain][0]], [1.0]

    def layers(self, d, seed: int = 0) -> list[LookLayer]:
        """The map (a diorama.Diorama) as tile layers, bottom first."""
        rows, cols = d.rows, d.cols
        terr = np.asarray(d.terrain, object)
        hgt = np.asarray(d.height, int)
        under = terr.copy()
        sets = self.sets
        blob_keys = [k for k, s in sets.items() if s.kind == "blob" and s.role != "cliff"]
        corner_keys = sorted((k for k, s in sets.items() if s.kind == "corner"),
                             key=lambda k: -int((terr == k).sum()))
        out, shores = [], []
        for k in blob_keys:                                   # shores (+ what shows under them)
            s, on = sets[k], terr == k
            cells = np.full((rows, cols, 2), -1, int)
            u = cell_noise(rows, cols, seed, k)
            for y, x in np.argwhere(on):
                m = neighbour_mask(on, y, x)
                if m != 255 and m in s.choices:
                    cells[y, x] = pick(*s.choices[m], u[y, x])
                    under[y, x] = _neighbour_terrain(terr, y, x, k, s.base, self.terrains)
            shores += [LookLayer(k, s.overlay, cells, s)] if (cells[..., 0] >= 0).any() else []
        u = cell_noise(rows, cols, seed, "ground")
        ground = np.zeros((rows, cols, 2), int)
        for (y, x), name in np.ndenumerate(under):
            name = name if name in self._plain else "grass"
            w = self._plain[name][1]
            ground[y, x] = (pick(list(range(len(w))), w, u[y, x]), self.terrains.index(name))
        out.append(LookLayer("ground", self.ground_atlas, ground))
        wet = set(blob_keys) | {"water"}
        free = ~np.isin(terr, list(wet | set(corner_keys)))
        for k in corner_keys:                                 # paths reach into the cells around
            s, p = sets[k], np.pad(terr == k, 1)
            v = (p[:-1, :-1] | p[:-1, 1:] | p[1:, :-1] | p[1:, 1:]).astype(int)
            cells = np.full((rows, cols, 2), -1, int)
            u = cell_noise(rows, cols, seed, k)
            for y, x in np.argwhere(free):
                key = (v[y, x], v[y, x + 1], v[y + 1, x], v[y + 1, x + 1])
                if any(key) and key in s.choices:
                    cells[y, x] = pick(*s.choices[key], u[y, x])
            out += [LookLayer(k, s.overlay, cells, s)] if (cells[..., 0] >= 0).any() else []
        out += shores
        if "cliff" in sets:                                   # drops of 2+ levels
            s = sets["cliff"]
            cells = np.full((rows, cols, 2), -1, int)
            for y, x in np.argwhere(~np.isin(terr, list(wet))):
                m = 0
                for bit, (dy, dx) in _DIRS8:
                    yy, xx = y + dy, x + dx
                    if not (0 <= yy < rows and 0 <= xx < cols) or hgt[yy, xx] >= hgt[y, x] - 1:
                        m |= bit
                m = canonical(m)
                if m != 255 and m in s.choices:
                    cells[y, x] = s.choices[m][0][0]
            out += [LookLayer("cliff", s.overlay, cells, s)] if (cells[..., 0] >= 0).any() else []
        for k in LINE_KEYS:                                   # fences / walls
            if k not in sets:
                continue
            on = np.zeros((rows, cols), bool)
            for kind, c, r in d.props:
                if kind == k and 0 <= c < cols and 0 <= r < rows:
                    on[r, c] = True
            cells = np.full((rows, cols, 2), -1, int)
            for y, x in np.argwhere(on):
                m = line_mask(on, y, x)
                if m in sets[k].choices:
                    cells[y, x] = sets[k].choices[m][0][0]
            out += [LookLayer(k, sets[k].overlay, cells, sets[k])] if (cells[..., 0] >= 0).any() else []
        return out

    def render(self, d, seed: int = 0, layers: list | None = None) -> np.ndarray:
        """The layers composited: (rows*tile, cols*tile, 4) RGBA."""
        t = self.tile
        img = np.zeros((d.rows * t, d.cols * t, 4), np.float32)
        for layer in layers if layers is not None else self.layers(d, seed):
            for y, x in np.argwhere(layer.cells[..., 0] >= 0):
                c, r = layer.cells[y, x]
                src = layer.atlas[r * t:(r + 1) * t, c * t:(c + 1) * t].astype(np.float32)
                a = src[..., 3:4] / 255
                dst = img[y * t:(y + 1) * t, x * t:(x + 1) * t]
                dst[..., :3] = src[..., :3] * a + dst[..., :3] * (1 - a)
                dst[..., 3:] = np.maximum(dst[..., 3:], src[..., 3:])
        return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def _neighbour_terrain(terr: np.ndarray, y: int, x: int, skip: str, default: str, known) -> str:
    """The most common other terrain around a cell (what shows at the edge of its shore)."""
    from collections import Counter
    rows, cols = terr.shape
    n = Counter(terr[yy, xx] for yy in range(max(0, y - 1), min(rows, y + 2))
                for xx in range(max(0, x - 1), min(cols, x + 2)) if terr[yy, xx] != skip)
    for name, _ in n.most_common():
        if name in known:
            return name
    return default if default in known else "grass"

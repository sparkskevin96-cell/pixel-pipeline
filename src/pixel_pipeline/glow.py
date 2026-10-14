"""Emissive (glow) masks: which pixels of a sprite give off light, so a game's bloom picks out
fire, crystals, lanterns, magic orbs and glowing eyes and nothing else.

    auto_glow(rgba)                  -> (H, W, 4) mask for one sprite
    write_glow_sheet("sheet.png")    -> sheet_glow.png next to it (None when nothing glows)

`<sheet>_glow.png` sits next to `<sheet>.png` (and `<sheet>_n.png`) and is laid out exactly like
the sheet. Ours are white where the sprite glows with alpha = how strongly; engines read any
glow map as alpha x max(r, g, b), so a grey or coloured mask (effects.py writes its own with the
same name) means the same thing. The glowing colour is always the sprite's own pixel colour.

What counts as glowing (auto_glow): bright, saturated pixels that stand out from what is around
them (a flame on a dark torch, an orb in a hand, a red eye), the near-white cores right next to
them, in small clusters: a large bright area is cloth or armour, not a light. Warm hues (skin,
wood, gold trim) need to be more saturated than cool ones. Effects (fire, magic) are light
through and through: effect=True makes every opaque pixel glow by its brightness.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def glow_sheet_path(sheet_png) -> Path:
    p = Path(sheet_png)
    return p.with_name(p.stem + "_glow.png")


def mask_strength(glow: np.ndarray) -> np.ndarray:
    """A glow map (any convention: white + alpha, grey, coloured) -> 0..1 strength per pixel."""
    g = glow.astype(np.float32) / 255.0
    if g.ndim == 2:
        return g
    a = g[..., 3] if g.shape[-1] == 4 else 1.0
    return a * g[..., :3].max(-1)


def _hsv(rgb: np.ndarray):
    mx = rgb.max(-1)
    mn = rgb.min(-1)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    hsv = cv2.cvtColor(np.ascontiguousarray((rgb * 255).astype(np.uint8)), cv2.COLOR_RGB2HSV_FULL)
    hue = hsv[..., 0].astype(np.float32) * (360.0 / 256.0)
    return hue, sat, mx


def auto_glow(rgba: np.ndarray, *, min_value: float = 0.78, min_sat: float = 0.45,
              warm_sat: float = 0.62, contrast: float = 0.08, max_share: float = 0.08,
              effect: bool = False) -> np.ndarray:
    """(H, W, 4) uint8 glow mask for one sprite: white, alpha = strength. Nothing glows ->
    all zeros. effect=True: the whole sprite glows by brightness (fire, spells)."""
    h, w = rgba.shape[:2]
    out = np.zeros((h, w, 4), np.uint8)
    opaque = rgba[..., 3] >= 128
    if not opaque.any():
        return out
    rgb = rgba[..., :3].astype(np.float32) / 255.0
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    if effect:
        k = np.clip(0.25 + lum * 0.95, 0, 1)
        out[opaque, :3] = 255
        out[opaque, 3] = (k[opaque] * 255).astype(np.uint8)
        return out
    hue, sat, val = _hsv(rgb)
    warm = (hue < 55) | (hue > 340)                   # reds, oranges, yellows: skin and wood too
    need = np.where(warm, warm_sat, min_sat)
    # brightness of the surroundings (opaque pixels only, 7x7): a light is brighter than its holder
    o = opaque.astype(np.float32)
    around = cv2.blur(lum * o, (7, 7)) / np.maximum(cv2.blur(o, (7, 7)), 1e-6)
    stands_out = (lum - around) >= contrast
    hot = opaque & (sat >= need) & ((val >= 0.86) | ((val >= min_value) & stands_out))
    # near-white cores touching a hot pixel (the middle of a flame, a crystal's highlight)
    near = cv2.dilate(hot.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2) > 0
    core = opaque & near & (val >= 0.9)
    m = hot | core
    if not m.any():
        return out
    # clusters: small ones are lights (eyes, gems, a lantern); a big one must also be much
    # brighter than the pixels around it (a flame on a dark torch), else it is a bright cape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    small = max(4, int(max_share * opaque.sum()))
    keep = np.zeros(n, bool)
    ring_k = np.ones((3, 3), np.uint8)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        comp = labels == i
        if val[comp].max() < 0.9:
            continue
        if area <= small:
            keep[i] = True
            continue
        ring = (cv2.dilate(comp.astype(np.uint8), ring_k, iterations=2) > 0) & opaque & ~m
        keep[i] = area <= 0.45 * opaque.sum() and (not ring.any() or lum[comp].mean() - lum[ring].mean() >= 0.25)
    m = keep[labels] & m
    strength = np.clip((val - min_value) / max(1e-3, 1 - min_value), 0.0, 1.0) * 0.6 + 0.4
    strength = np.where(core, 1.0, strength)
    out[m, :3] = 255
    out[m, 3] = (strength[m] * 255).astype(np.uint8)
    return out


def _frame_rects(sheet_png: Path, size: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    js = sheet_png.with_suffix(".json")
    if js.exists():
        try:
            raw = json.loads(js.read_text(encoding="utf-8")).get("frames", [])
            items = raw.values() if isinstance(raw, dict) else raw
            rects = [(int(f["frame"]["x"]), int(f["frame"]["y"]), int(f["frame"]["w"]), int(f["frame"]["h"]))
                     for f in items]
            if rects:
                return rects
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return [(0, 0, size[0], size[1])]


def glow_sheet(sheet: np.ndarray, rects, **opts) -> np.ndarray:
    """The glow map of a whole sheet, frame by frame (so "small cluster" means small for that frame)."""
    out = np.zeros(sheet.shape[:2] + (4,), np.uint8)
    for x, y, w, h in rects:
        out[y:y + h, x:x + w] = auto_glow(sheet[y:y + h, x:x + w], **opts)
    return out


def write_glow_sheet(sheet_png, *, effect: bool = False, force: bool = False, out=None,
                     **opts) -> Path | None:
    """Write <sheet>_glow.png next to the sheet (or to `out`) from auto_glow, frame by frame.
    An existing glow map is kept (it may be hand-made or from effects.py) unless force.
    Returns the path, or None when nothing in the sheet glows."""
    sheet_png = Path(sheet_png)
    path = Path(out) if out is not None else glow_sheet_path(sheet_png)
    if path.exists() and not force:
        return path
    sheet = np.asarray(Image.open(sheet_png).convert("RGBA"))
    g = glow_sheet(sheet, _frame_rects(sheet_png, (sheet.shape[1], sheet.shape[0])), effect=effect, **opts)
    if not g[..., 3].any():
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(g, "RGBA").save(path)
    return path


def glow_colour(sheet_png, glow_png=None) -> tuple[float, float, float] | None:
    """Mean colour (0..1, brightest channel = 1) of the glowing pixels: the colour of the light
    the sprite casts (a torch's orange, a crystal's cyan). None if nothing glows."""
    sheet_png = Path(sheet_png)
    gp = Path(glow_png) if glow_png else glow_sheet_path(sheet_png)
    if not gp.exists():
        return None
    rgb = np.asarray(Image.open(sheet_png).convert("RGBA")).astype(np.float32)[..., :3]
    k = mask_strength(np.asarray(Image.open(gp).convert("RGBA")))
    if k.shape != rgb.shape[:2] or k.sum() <= 0:
        return None
    c = (rgb * k[..., None]).sum((0, 1)) / k.sum()
    c = c / max(1e-6, c.max())
    return tuple(round(float(v), 3) for v in c)

"""Photo -> sprite pipeline.

    load_image -> isolate (cut out subject) -> prepare (crop, fit, pixelize) -> finish (palette, dither, outline)

`prepare` is the expensive, palette-independent half, so a batch can share one
palette across every frame (consistent colours for animations / sprite sets).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image, ImageOps

from .color import adjust, fill_transparent, luminance
from .cutout import remove_background
from .finishing import outline_ring, remove_orphans, selout_colors
from .palettes import auto_palette, hex_to_rgb, load_palette
from .pixelize import make_tileable, pixelize
from .quantize import quantize
from .settings import Settings

ImageSource = Union[str, Path, Image.Image, np.ndarray]


def load_image(src: ImageSource) -> np.ndarray:
    """Any image (path / PIL / array) -> RGBA uint8, phone-photo rotation applied."""
    if isinstance(src, np.ndarray):
        arr = src.astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, -1)
        if arr.shape[2] == 3:
            arr = np.concatenate([arr, np.full(arr.shape[:2] + (1,), 255, np.uint8)], -1)
        return arr.copy()
    img = src if isinstance(src, Image.Image) else Image.open(src)
    img = ImageOps.exif_transpose(img)
    return np.asarray(img.convert("RGBA")).copy()


def isolate(rgba: np.ndarray, settings: Settings) -> np.ndarray:
    """Make the background transparent. Images that already have transparency are kept as-is."""
    if not settings.remove_background or (rgba[..., 3] < 255).any():
        return rgba
    out = rgba.copy()
    out[..., 3] = remove_background(np.ascontiguousarray(rgba[..., :3]), settings.cutout_model)
    return out


@dataclass
class Prepared:
    rgb: np.ndarray   # (H, W, 3) downscaled colours, not yet snapped to a palette
    mask: np.ndarray  # (H, W) bool, True = opaque


@dataclass
class Sprite:
    rgba: np.ndarray     # (H, W, 4) final pixels
    indices: np.ndarray  # (H, W) palette index, -1 = transparent
    palette: np.ndarray  # (N, 3) RGB

    @property
    def size(self) -> tuple[int, int]:
        return self.rgba.shape[1], self.rgba.shape[0]

    def image(self, scale: int = 1) -> Image.Image:
        img = Image.fromarray(self.rgba, "RGBA")
        if scale > 1:
            img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
        return img

    def save(self, path, scale: int = 1) -> None:
        self.image(scale).save(path)


def _crop_to_subject(rgba: np.ndarray, s: Settings) -> np.ndarray:
    alpha = rgba[..., 3]
    if alpha.min() == 255:
        return rgba
    ys, xs = np.nonzero(alpha >= s.alpha_threshold)
    if len(xs) == 0:
        return rgba
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    m = int(round(max(x1 - x0, y1 - y0) * s.margin))
    return rgba[max(0, y0 - m):y1 + m, max(0, x0 - m):x1 + m]


def _fit(rgba: np.ndarray, out_w: int, out_h: int, inset: int, s: Settings) -> np.ndarray:
    """Scale the subject onto a (out_h*patch, out_w*patch) RGBA canvas."""
    p = s.patch
    cw, ch = out_w * p, out_h * p
    iw, ih = max(1, out_w - 2 * inset) * p, max(1, out_h - 2 * inset) * p
    sh, sw = rgba.shape[:2]
    if s.fit == "stretch":
        nw, nh = iw, ih
    elif s.fit in ("contain", "cover"):
        scale = (min if s.fit == "contain" else max)(iw / sw, ih / sh)
        nw, nh = max(1, round(sw * scale)), max(1, round(sh * scale))
    else:
        raise ValueError(f"Unknown fit '{s.fit}' (use contain, cover, stretch)")

    # Resize with premultiplied alpha so background colours don't bleed into edges.
    alpha = rgba[..., 3].astype(np.float32) / 255
    premul = rgba[..., :3].astype(np.float32) * alpha[..., None]
    interp = cv2.INTER_AREA if nw < sw else cv2.INTER_CUBIC
    a = np.clip(cv2.resize(alpha, (nw, nh), interpolation=interp), 0, 1).reshape(nh, nw)
    rgb = cv2.resize(premul, (nw, nh), interpolation=interp).reshape(nh, nw, 3)
    rgb = np.clip(rgb / np.maximum(a, 1e-6)[..., None], 0, 255)

    x = inset * p + (iw - nw) // 2
    bottom = s.anchor == "bottom" and s.fit == "contain"
    y = inset * p + ((ih - nh) if bottom else (ih - nh) // 2)
    sx, sy, dx, dy = max(0, -x), max(0, -y), max(0, x), max(0, y)
    w, h = min(nw - sx, cw - dx), min(nh - sy, ch - dy)
    canvas = np.zeros((ch, cw, 4), np.float32)
    canvas[dy:dy + h, dx:dx + w, :3] = rgb[sy:sy + h, sx:sx + w]
    canvas[dy:dy + h, dx:dx + w, 3] = a[sy:sy + h, sx:sx + w] * 255
    return np.rint(canvas).astype(np.uint8)


def prepare(rgba: np.ndarray, s: Settings) -> Prepared:
    """Isolated RGBA photo -> downscaled colours + opacity mask (no palette yet)."""
    if s.crop_to_subject:
        rgba = _crop_to_subject(rgba, s)
    out_w, out_h = s.resolve_size(rgba.shape[1], rgba.shape[0])
    inset = 1 if s.outline and min(out_w, out_h) > 4 else 0  # leave room for the outline
    canvas = _fit(rgba, out_w, out_h, inset, s)
    rgb, alpha = canvas[..., :3], canvas[..., 3]
    rgb = fill_transparent(rgb, alpha, s.alpha_threshold)
    if s.tileable:
        rgb = make_tileable(rgb)
    rgb = adjust(rgb, s.brightness, s.contrast, s.saturation)
    small, mask = pixelize(rgb, alpha, s.patch, s.detail, s.downscale, s.alpha_threshold,
                           wrap=s.tileable)
    return Prepared(small, mask)


def resolve_palette(s: Settings, pixels: np.ndarray) -> np.ndarray:
    if s.system not in (None, "off"):
        from .systems import system_palette
        return system_palette(s.system, pixels, s.colors, s.subject)
    pal = load_palette(s.palette)
    return auto_palette(pixels, s.colors) if pal is None else pal


def _outline_index(palette: np.ndarray, color: str) -> tuple[np.ndarray, int]:
    if color == "auto":
        return palette, int(np.argmin(luminance(palette)))
    hexcode = color.lstrip("#")
    if not re.fullmatch(r"[0-9a-fA-F]{6}", hexcode):
        raise ValueError(f"Outline colour must be 'auto' or #rrggbb, got '{color}'")
    rgb = hex_to_rgb([hexcode])[0]
    hit = np.nonzero((palette == rgb).all(1))[0]
    if len(hit):
        return palette, int(hit[0])
    return np.vstack([palette, rgb]), len(palette)


def finish(prep: Prepared, s: Settings, palette: np.ndarray | None = None) -> Sprite:
    """Snap to the palette, dither, clean up, add the outline."""
    mask = prep.mask.copy()
    pixels = prep.rgb[mask] if mask.any() else prep.rgb.reshape(-1, 3)
    pal = palette if palette is not None else resolve_palette(s, pixels)
    idx = quantize(prep.rgb, mask, pal, s.dither, s.dither_strength)
    if s.cleanup and s.dither == "none":  # dither patterns are deliberately "lonely" pixels
        idx = remove_orphans(idx, pal)
    if s.outline:
        ring = outline_ring(mask)
        if s.outline_style == "selout":
            idx[ring] = selout_colors(idx, ring, pal)[ring]
        elif s.outline_style == "solid":
            pal, oi = _outline_index(pal, s.outline_color)
            idx[ring] = oi
        else:
            raise ValueError(f"Unknown outline style '{s.outline_style}' (use solid, selout)")
        mask |= ring
    return sprite_from_indices(idx, pal)


def sprite_from_indices(idx: np.ndarray, palette: np.ndarray) -> Sprite:
    mask = idx >= 0
    rgba = np.zeros(idx.shape + (4,), np.uint8)
    rgba[mask, :3] = palette[idx[mask]]
    rgba[mask, 3] = 255
    return Sprite(rgba, idx, palette)


def convert(src: ImageSource, settings: Settings, palette: np.ndarray | None = None) -> Sprite:
    """One photo -> one sprite."""
    return finish(prepare(isolate(load_image(src), settings), settings), settings, palette)


def sprites_from_rgba(images: list[np.ndarray], settings: Settings,
                      shared_palette: bool = True) -> list[Sprite]:
    """Already-isolated RGBA images -> sprites (optionally sharing one palette)."""
    preps = [prepare(rgba, settings) for rgba in images]
    palette = None
    if shared_palette and preps:
        pixels = np.concatenate([p.rgb[p.mask] if p.mask.any() else p.rgb.reshape(-1, 3)
                                 for p in preps])
        palette = resolve_palette(settings, pixels)
    return [finish(p, settings, palette) for p in preps]


def convert_batch(sources: list[ImageSource], settings: Settings,
                  shared_palette: bool = True) -> list[Sprite]:
    """Many photos -> sprites. shared_palette keeps colours identical across the set."""
    return sprites_from_rgba([isolate(load_image(src), settings) for src in sources],
                             settings, shared_palette)

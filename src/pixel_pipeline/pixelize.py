"""Edge-preserving downscale: the step that turns a photo into a pixel grid.

The contrast-aware outline expansion and contrast-based downscale are adapted
from PixelOE by Shih-Ying Yeh (KohakuBlueLeaf), Apache-2.0,
https://github.com/KohakuBlueleaf/PixelOE -- reimplemented here with numpy +
OpenCV only so the core install doesn't need PyTorch.
"""
from __future__ import annotations

import cv2
import numpy as np

from .color import match_color

KERNEL_SQUARE = np.ones((3, 3), np.uint8)
KERNEL_CROSS = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

DOWNSCALE_MODES = ("contrast", "median", "center", "area")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def expansion_weight(rgb: np.ndarray, k: int, avg_scale: float = 9, dist_scale: float = 4) -> np.ndarray:
    """Per-pixel weight: ~1 where dark detail sits on a bright area, ~0 for the reverse."""
    lum = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)[..., 0]
    k = max(2, k)
    avg = cv2.medianBlur(lum, 2 * k + 1).astype(np.float32) / 255
    lumf = lum.astype(np.float32) / 255
    window = np.ones((k, k), np.uint8)
    hi, lo = cv2.dilate(lumf, window), cv2.erode(lumf, window)
    weight = _sigmoid((avg - 0.5) * avg_scale - ((hi - avg) - (avg - lo)) * dist_scale)
    stride = max(1, (k // 4) * 2)
    h, w = weight.shape
    if stride > 1 and h // stride and w // stride:
        weight = cv2.resize(weight, (w // stride, h // stride), interpolation=cv2.INTER_LINEAR)
        weight = cv2.resize(weight, (w, h), interpolation=cv2.INTER_LINEAR)
    span = weight.max() - weight.min()
    return (weight - weight.min()) / span if span > 1e-6 else np.full_like(weight, 0.5)


def outline_expansion(rgb: np.ndarray, thickness: int, k: int, wrap: bool = False) -> np.ndarray:
    """Thicken important edges so they survive downscaling as clean pixel lines."""
    if thickness <= 0:
        return rgb
    pad = 0
    if wrap:  # keep tileable images seamless
        pad = 2 * k + 2 * thickness
        rgb = cv2.copyMakeBorder(rgb, pad, pad, pad, pad, cv2.BORDER_WRAP)
    weight = expansion_weight(rgb, k)[..., None]
    keep = _sigmoid((weight - 0.5) * 5) * 0.25
    eroded = cv2.erode(rgb, KERNEL_SQUARE, iterations=thickness).astype(np.float32)
    dilated = cv2.dilate(rgb, KERNEL_SQUARE, iterations=thickness).astype(np.float32)
    out = eroded * weight + dilated * (1 - weight)
    out = out * (1 - keep) + rgb.astype(np.float32) * keep
    out = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    out = cv2.erode(out, KERNEL_CROSS, iterations=thickness)
    out = cv2.dilate(out, KERNEL_CROSS, iterations=thickness * 2)
    out = cv2.erode(out, KERNEL_CROSS, iterations=thickness)
    if pad:
        out = out[pad:-pad, pad:-pad]
    return out


def _blocks(arr: np.ndarray, p: int) -> np.ndarray:
    """(H*p, W*p, C) -> (H, W, p*p, C)"""
    h, w = arr.shape[0] // p, arr.shape[1] // p
    c = arr.shape[2] if arr.ndim == 3 else 1
    b = arr[:h * p, :w * p].reshape(h, p, w, p, c).transpose(0, 2, 1, 3, 4)
    return b.reshape(h, w, p * p, c)


def downscale(rgb: np.ndarray, patch: int, mode: str = "contrast") -> np.ndarray:
    """Collapse each patch x patch block to one pixel."""
    h, w = rgb.shape[0] // patch, rgb.shape[1] // patch
    if mode == "area":
        return cv2.resize(rgb[:h * patch, :w * patch], (w, h), interpolation=cv2.INTER_AREA)
    if mode == "center":
        c = patch // 2
        return np.ascontiguousarray(rgb[c:h * patch:patch, c:w * patch:patch])
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab).astype(np.float32)
    blocks = _blocks(lab, patch)
    out = np.median(blocks, axis=2)
    if mode == "contrast":
        lum = blocks[..., 0]
        med, mean = np.median(lum, -1), lum.mean(-1)
        hi, lo = lum.max(-1), lum.min(-1)
        pick = lum[..., (patch // 2) * patch + patch // 2].copy()
        dark = (med < mean) & (hi - med > med - lo)
        light = (med > mean) & (hi - med < med - lo)
        pick[dark], pick[light] = lo[dark], hi[light]
        out[..., 0] = pick
    elif mode != "median":
        raise ValueError(f"Unknown downscale mode '{mode}' (use {', '.join(DOWNSCALE_MODES)})")
    return cv2.cvtColor(np.clip(np.rint(out), 0, 255).astype(np.uint8), cv2.COLOR_Lab2RGB)


def downscale_alpha(alpha: np.ndarray, patch: int, threshold: int = 128) -> np.ndarray:
    return _blocks(alpha[..., None], patch)[..., 0].mean(-1) >= threshold


def pixelize(rgb: np.ndarray, alpha: np.ndarray, patch: int, thickness: int,
             mode: str = "contrast", threshold: int = 128, wrap: bool = False):
    """Canvas (H*patch, W*patch) -> (rgb (H, W, 3), opaque mask (H, W))."""
    opaque = alpha >= threshold
    expanded = outline_expansion(rgb, thickness, patch, wrap=wrap)
    if thickness > 0:
        expanded = match_color(expanded, rgb, opaque)
    return downscale(expanded, patch, mode), downscale_alpha(alpha, patch, threshold)


def make_tileable(rgb: np.ndarray, blend: float = 0.25) -> np.ndarray:
    """Cross-fade with a half-offset copy so the image wraps seamlessly."""
    h, w = rgb.shape[:2]
    shifted = np.roll(rgb, (h // 2, w // 2), axis=(0, 1)).astype(np.float32)
    ys, xs = np.arange(h), np.arange(w)
    wy = np.clip(np.minimum(ys, h - 1 - ys) / max(1.0, blend * h), 0, 1)
    wx = np.clip(np.minimum(xs, w - 1 - xs) / max(1.0, blend * w), 0, 1)
    weight = (wy[:, None] * wx[None, :])[..., None]
    out = rgb.astype(np.float32) * weight + shifted * (1 - weight)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)

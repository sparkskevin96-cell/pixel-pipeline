"""Colour-space helpers. All perceptual maths happens in CIE Lab (L 0..100)."""
from __future__ import annotations

import cv2
import numpy as np


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """RGB (uint8 or float 0..255, any leading shape) -> float32 Lab."""
    shape = rgb.shape
    arr = np.clip(rgb.reshape(-1, 1, 3).astype(np.float32), 0, 255) / 255.0
    return cv2.cvtColor(arr, cv2.COLOR_RGB2Lab).reshape(shape)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """float Lab -> uint8 RGB, same leading shape."""
    shape = lab.shape
    arr = cv2.cvtColor(lab.reshape(-1, 1, 3).astype(np.float32), cv2.COLOR_Lab2RGB)
    return np.clip(np.rint(arr * 255), 0, 255).astype(np.uint8).reshape(shape)


def luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb_to_lab(rgb)[..., 0]


def adjust(rgb: np.ndarray, brightness: float = 0.0, contrast: float = 1.0,
           saturation: float = 1.0) -> np.ndarray:
    """Brightness (-1..1), contrast and saturation (1 = unchanged) in Lab."""
    if brightness == 0 and contrast == 1 and saturation == 1:
        return rgb
    lab = rgb_to_lab(rgb)
    lab[..., 0] = (lab[..., 0] - 50.0) * contrast + 50.0 + brightness * 50.0
    lab[..., 1:] *= saturation
    lab[..., 0] = np.clip(lab[..., 0], 0, 100)
    return lab_to_rgb(lab)


def match_color(img: np.ndarray, ref: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Shift img's per-channel Lab mean/std to ref's (over mask, if given)."""
    src, dst = rgb_to_lab(img), rgb_to_lab(ref)
    sel = mask if mask is not None and mask.any() else np.ones(img.shape[:2], bool)
    out = src.copy()
    for c in range(3):
        s, d = src[..., c][sel], dst[..., c][sel]
        s_std = s.std()
        scale = d.std() / s_std if s_std > 1e-6 else 1.0
        out[..., c] = (src[..., c] - s.mean()) * scale + d.mean()
    return lab_to_rgb(out)


def fill_transparent(rgb: np.ndarray, alpha: np.ndarray, threshold: int = 128) -> np.ndarray:
    """Replace colours under transparent pixels with nearby opaque colours.

    Push-pull fill: stops background colours (or black padding) bleeding into
    the subject's edge pixels during filtering and downscaling.
    """
    w = (alpha >= threshold).astype(np.float32)
    if w.all() or not w.any():
        return rgb
    color = rgb.astype(np.float32) * w[..., None]
    levels = [(color, w)]
    while min(levels[-1][1].shape) > 1:
        c, ws = levels[-1]
        size = (max(1, ws.shape[1] // 2), max(1, ws.shape[0] // 2))
        shape = (size[1], size[0])
        levels.append((cv2.resize(c, size, interpolation=cv2.INTER_AREA).reshape(shape + (3,)),
                       cv2.resize(ws, size, interpolation=cv2.INTER_AREA).reshape(shape)))
    c, ws = levels[-1]
    filled = c / np.maximum(ws, 1e-6).reshape(ws.shape + (1,))
    for c, ws in reversed(levels[:-1]):
        up = cv2.resize(filled, (ws.shape[1], ws.shape[0]), interpolation=cv2.INTER_LINEAR)
        up = up.reshape(ws.shape + (3,))
        filled = c + up * (1.0 - ws)[..., None]
    out = rgb.copy()
    hole = w == 0
    out[hole] = np.clip(np.rint(filled[hole]), 0, 255).astype(np.uint8)
    return out

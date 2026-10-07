"""Map pixels onto a palette, with optional dithering."""
from __future__ import annotations

import numpy as np

from .color import rgb_to_lab

DITHER_MODES = ("none", "floyd", "bayer2", "bayer4", "bayer8")


def bayer_matrix(n: int) -> np.ndarray:
    """n x n ordered-dither thresholds centred on 0, range (-0.5, 0.5)."""
    m = np.array([[0, 2], [3, 1]], np.float32)
    while m.shape[0] < n:
        m = np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
    return (m + 0.5) / (n * n) - 0.5


def palette_spread(palette: np.ndarray) -> float:
    """Typical gap between neighbouring palette colours, per RGB channel."""
    if len(palette) < 2:
        return 32.0
    p = palette.astype(np.float32)
    d = np.sqrt(((p[:, None] - p[None]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(1)) / np.sqrt(3))


def nearest(lab: np.ndarray, pal_lab: np.ndarray, chunk: int = 65536) -> np.ndarray:
    flat = lab.reshape(-1, 3)
    out = np.empty(len(flat), np.int16)
    for i in range(0, len(flat), chunk):
        d = ((flat[i:i + chunk, None, :] - pal_lab[None]) ** 2).sum(-1)
        out[i:i + chunk] = d.argmin(1)
    return out.reshape(lab.shape[:-1])


def quantize(rgb: np.ndarray, mask: np.ndarray, palette: np.ndarray,
             method: str = "none", strength: float = 1.0) -> np.ndarray:
    """Return palette indices (H, W); -1 where mask is False (transparent)."""
    pal_lab = rgb_to_lab(palette)
    h, w = mask.shape
    if method == "none":
        idx = nearest(rgb_to_lab(rgb), pal_lab)
    elif method.startswith("bayer"):
        n = int(method[5:])
        m = np.tile(bayer_matrix(n), (h // n + 1, w // n + 1))[:h, :w]
        shifted = rgb.astype(np.float32) + (m * palette_spread(palette) * strength)[..., None]
        idx = nearest(rgb_to_lab(shifted), pal_lab)
    elif method == "floyd":
        idx = _floyd(rgb_to_lab(rgb).astype(np.float64), mask, pal_lab.astype(np.float64), strength)
    else:
        raise ValueError(f"Unknown dither '{method}' (use {', '.join(DITHER_MODES)})")
    idx = idx.astype(np.int16)
    idx[~mask] = -1
    return idx


def _floyd(lab: np.ndarray, mask: np.ndarray, pal: np.ndarray, strength: float) -> np.ndarray:
    h, w = mask.shape
    idx = np.full((h, w), -1, np.int16)
    spread = ((1, 0, 7 / 16), (-1, 1, 3 / 16), (0, 1, 5 / 16), (1, 1, 1 / 16))
    for y in range(h):
        for x in range(w):
            if not mask[y, x]:
                continue
            old = lab[y, x]
            i = int(((pal - old) ** 2).sum(1).argmin())
            idx[y, x] = i
            err = (old - pal[i]) * strength
            for dx, dy, f in spread:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and ny < h and mask[ny, nx]:
                    lab[ny, nx] += err * f
    return idx

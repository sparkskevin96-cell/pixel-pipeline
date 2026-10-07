"""Hand-made touches on palette-index maps: stray-pixel cleanup and outlines."""
from __future__ import annotations

import cv2
import numpy as np

from .color import rgb_to_lab
from .pixelize import KERNEL_CROSS

_N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors(idx: np.ndarray, offsets) -> np.ndarray:
    """(H, W, len(offsets)) neighbour indices; -1 outside the image."""
    padded = np.pad(idx, 1, constant_values=-1)
    h, w = idx.shape
    return np.stack([padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy, dx in offsets], -1)


def remove_orphans(idx: np.ndarray, palette: np.ndarray, max_delta: float = 25.0,
                   passes: int = 2) -> np.ndarray:
    """Replace single pixels that match none of their 8 neighbours with the
    neighbours' most common colour -- but only when the change is subtle
    (Lab distance < max_delta), so deliberate highlights like eye glints stay."""
    idx = idx.copy()
    lab = rgb_to_lab(palette)
    for _ in range(passes):
        nb = _neighbors(idx, _N8)
        lonely = (idx >= 0) & ~(nb == idx[..., None]).any(-1) & ((nb >= 0).sum(-1) >= 5)
        changed = 0
        for y, x in zip(*np.nonzero(lonely)):
            vals = nb[y, x][nb[y, x] >= 0]
            best = np.bincount(vals).argmax()
            if np.linalg.norm(lab[best] - lab[idx[y, x]]) < max_delta:
                idx[y, x] = best
                changed += 1
        if not changed:
            break
    return idx


def outline_ring(mask: np.ndarray) -> np.ndarray:
    """Transparent pixels 4-adjacent to the silhouette."""
    return cv2.dilate(mask.astype(np.uint8), KERNEL_CROSS).astype(bool) & ~mask


def selout_colors(idx: np.ndarray, ring: np.ndarray, palette: np.ndarray,
                  darken: float = 0.55) -> np.ndarray:
    """For each ring pixel, a palette index that is a darker shade of the colour it touches."""
    lab = rgb_to_lab(palette)
    nb = _neighbors(idx, _N4)
    out = np.full(idx.shape, -1, np.int16)
    darkest = int(np.argmin(lab[:, 0]))
    for y, x in zip(*np.nonzero(ring)):
        vals = nb[y, x][nb[y, x] >= 0]
        if not len(vals):
            out[y, x] = darkest
            continue
        src = lab[np.bincount(vals).argmax()]
        target = np.array([src[0] * darken, src[1] * 0.85, src[2] * 0.85])
        darker = np.nonzero(lab[:, 0] < src[0] - 4)[0]
        if not len(darker):
            out[y, x] = darkest
            continue
        d = ((lab[darker] - target) ** 2).sum(1)
        out[y, x] = darker[d.argmin()]
    return out

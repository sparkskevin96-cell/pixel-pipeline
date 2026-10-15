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
    ys, xs = np.nonzero(ring)
    if not len(ys):
        return out
    # the most common 4-neighbour colour of every ring pixel (ties: the lowest index, as
    # bincount().argmax() picks), then one shade per distinct colour
    v = nb[ys, xs].astype(np.int64)                                     # (n, 4)
    ok = v >= 0
    count = ((v[:, :, None] == v[:, None, :]) & ok[:, None, :]).sum(-1)
    key = np.where(ok, count * (1 << 32) - v, np.iinfo(np.int64).min)
    src_i = v[np.arange(len(v)), key.argmax(1)]
    shade: dict[int, int] = {}
    for s_i in np.unique(src_i[ok.any(1)]).tolist():
        src = lab[s_i]
        target = np.array([src[0] * darken, src[1] * 0.85, src[2] * 0.85])
        darker = np.nonzero(lab[:, 0] < src[0] - 4)[0]
        if not len(darker):
            shade[s_i] = darkest
            continue
        d = ((lab[darker] - target) ** 2).sum(1)
        shade[s_i] = int(darker[d.argmin()])
    out[ys, xs] = [shade[int(s_i)] if any_ else darkest
                   for s_i, any_ in zip(src_i.tolist(), ok.any(1).tolist())]
    return out

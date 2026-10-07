"""Parallax backgrounds: one photo -> far / mid / near pixel-art layers + a Godot 4 scene.

    photo -> depth (Depth Anything V2 Small) -> split into layers by depth -> fill what
    nearer layers hide -> pixelize every layer on the same canvas with ONE palette
    -> <name>_layer0_far.png ... + <name>_preview.png + <name>_parallax.tscn (Parallax2D)

What looks right in a side-scroller, and how `split_layers` gets there:

* Cuts follow on-screen motion. The model outputs relative inverse depth (disparity), and
  in a real camera pan image motion is proportional to disparity, so grouping pixels by
  disparity (1-D k-means, cuts snapped to histogram valleys = depth discontinuities)
  groups things that should scroll together.
* The far layer is fully opaque, so whatever the layers do there is never a hole.
* Layers keep scrolling against each other for the whole level, so every part a nearer
  layer hides WILL be seen. Each hidden pixel goes to the farther layer that continues
  behind it along its row (a tree trunk in front of hills -> the hills layer; a canopy
  against the sky -> the far layer): layer boundaries such as a forest line carry on
  horizontally behind the occluder. Filling the whole hidden area into *every* farther
  layer instead leaves "cardboard" patches of invented mid-layer stuff sliding over the
  sky.
* Hidden areas are filled along rows too, by mirroring the texture on either side into
  the gap and cross-fading (fill="mirror"; wraps around when the layer repeats). Side-
  scroller scenery is horizontal bands (sky, horizon, hill line, field), so sideways
  continuation with real texture looks like more of the same. "rows" interpolates edge
  colours (smooth, can streak); "telea" / "ns" are OpenCV inpainting (flat facets on big
  gaps).
* No halos: depth is guided-filtered by the photo so near edges follow colour edges,
  blurred near->far transitions are snapped to one side (else a thin false "mid" outline
  appears around everything), and a thin fringe of farther layers next to nearer ones is
  re-filled, so near-object edge colours don't stay behind as a ghost outline when the
  near layer slides away.
* Specks and pinholes are removed (majority filter + small-island removal) so a layer is
  a few clean shapes, not confetti.
"""
from __future__ import annotations

import dataclasses
import os
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .pipeline import (
    ImageSource,
    Prepared,
    Sprite,
    finish,
    load_image,
    prepare,
    resolve_palette,
)
from .settings import Settings, settings_for

FILL_MODES = ("mirror", "rows", "telea", "ns")
SPLIT_METHODS = ("kmeans", "quantile")
DEFAULT_SCROLL = {1: [0.5], 2: [0.3, 0.8], 3: [0.2, 0.5, 0.85]}
REF_LINES = 180  # without `px`, assume the layers end up this many pixels tall


# --- small helpers ---------------------------------------------------------------------

def layer_names(n: int) -> list[str]:
    """far, mid, near (2 layers: far, near; 4+: far, mid1, mid2, ..., near)."""
    if n == 1:
        return ["far"]
    if n == 2:
        return ["far", "near"]
    if n == 3:
        return ["far", "mid", "near"]
    return ["far"] + [f"mid{i}" for i in range(1, n - 1)] + ["near"]


def default_scroll_scales(n: int) -> list[float]:
    """Godot Parallax2D scroll_scale.x per layer, far -> near (1 = moves with the camera)."""
    return list(DEFAULT_SCROLL.get(n) or np.round(np.linspace(0.15, 0.9, n), 2).tolist())


def _odd(x: float) -> int:
    return max(1, round(x) // 2 * 2 + 1)


def _pad_x(a: np.ndarray, p: int) -> np.ndarray:
    return np.concatenate([a[:, -p:], a, a[:, :p]], axis=1)


def _xwrap(fn, a: np.ndarray, pad: int, wrap: bool) -> np.ndarray:
    """Apply a spatial filter; with wrap, the left and right edges see each other."""
    pad = min(int(pad), a.shape[1])
    if not wrap or pad <= 0:
        return fn(a)
    return np.ascontiguousarray(fn(_pad_x(a, pad))[:, pad:-pad])


def _resize(a: np.ndarray, w: int, h: int, interp=cv2.INTER_LINEAR) -> np.ndarray:
    if a.shape[1] == w and a.shape[0] == h:
        return a.copy()
    out = cv2.resize(a, (w, h), interpolation=interp)
    return out.reshape((h, w) + a.shape[2:])


def seamless_x(img: np.ndarray, overlap: int) -> np.ndarray:
    """Make an image wrap left-to-right: cross-fade its last `overlap` columns into its
    first ones. Returns an image `overlap` columns narrower whose right edge continues
    straight into its left edge. Unlike pixelize.make_tileable (half-offset copy, both
    axes) nothing is duplicated and the top and bottom are left alone -- side-scroller
    layers only repeat horizontally.
    """
    w, b = img.shape[1], int(overlap)
    if b <= 0:
        return img.copy()
    if b >= w:
        raise ValueError(f"overlap {b} must be smaller than the width {w}")
    t = ((np.arange(b) + 0.5) / b).reshape((1, b) + (1,) * (img.ndim - 2))
    head, tail = img[:, :b].astype(np.float32), img[:, w - b:].astype(np.float32)
    blend = head * t + tail * (1 - t)
    out = img[:, :w - b].copy()
    if np.issubdtype(img.dtype, np.integer):
        info = np.iinfo(img.dtype)
        blend = np.clip(np.rint(blend), info.min, info.max)
    out[:, :b] = blend.astype(img.dtype)
    return out


def _cover(img: np.ndarray, w: int, h: int, interp=cv2.INTER_AREA) -> np.ndarray:
    """Centre-crop to w:h, then resize to exactly (w, h) (same as fit='cover')."""
    sh, sw = img.shape[:2]
    if sw * h > w * sh:      # too wide: trim the sides
        cw = max(1, round(sh * w / h))
        x = (sw - cw) // 2
        img = img[:, x:x + cw]
    else:                    # too tall: trim top and bottom
        ch = max(1, round(sw * h / w))
        y = (sh - ch) // 2
        img = img[y:y + ch]
    up = img.shape[1] < w
    return _resize(img, w, h, cv2.INTER_CUBIC if up and interp == cv2.INTER_AREA else interp)


# --- depth -> layer labels -------------------------------------------------------------

def guided_filter(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Colour guided filter (He et al.): smooths `src` while snapping its edges to the
    guide's colour edges. guide float32 (H, W, 3) 0..1, src float32 (H, W)."""
    def box(a):
        return cv2.blur(a, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)

    g = guide.astype(np.float32)
    p = src.astype(np.float32)
    mean_g, mean_p = box(g), box(p)
    cov_gp = box(g * p[..., None]) - mean_g * mean_p[..., None]
    sigma = np.empty(g.shape[:2] + (3, 3), np.float32)
    for i in range(3):
        for j in range(i, 3):
            v = box(g[..., i] * g[..., j]) - mean_g[..., i] * mean_g[..., j]
            sigma[..., i, j] = sigma[..., j, i] = v
    sigma += eps * np.eye(3, dtype=np.float32)
    a = np.linalg.solve(sigma, cov_gp[..., None])[..., 0]
    b = mean_p - (a * mean_g).sum(-1)
    return (box(a) * g).sum(-1) + box(b)


def depth_thresholds(depth: np.ndarray, layers: int = 3, method: str = "kmeans") -> np.ndarray:
    """Cut points (layers - 1, ascending) splitting a 0..1 depth map (1 = near) into layers.

    kmeans: 1-D k-means on the depth histogram, each cut then moved to the emptiest
    histogram bin near it, i.e. onto a depth discontinuity, so objects aren't sliced.
    quantile: equal pixel counts per layer.
    """
    if layers < 2:
        return np.zeros(0, np.float32)
    values = np.clip(np.asarray(depth, np.float32).ravel(), 0, 1)
    if method == "quantile":
        cuts = np.quantile(values, np.arange(1, layers) / layers)
    elif method == "kmeans":
        bins = 256
        hist, edges = np.histogram(values, bins=bins, range=(0, 1))
        hist = hist.astype(np.float64)
        mids = (edges[:-1] + edges[1:]) / 2
        c = np.quantile(values, (np.arange(layers) + 0.5) / layers).astype(np.float64)
        for _ in range(200):
            assign = np.abs(mids[:, None] - c[None]).argmin(1)
            new = c.copy()
            for j in range(layers):
                wsum = hist[assign == j].sum()
                if wsum > 0:
                    new[j] = (hist[assign == j] * mids[assign == j]).sum() / wsum
            if np.allclose(new, c, atol=1e-7):
                break
            c = new
        c = np.sort(c)
        smooth = cv2.GaussianBlur(hist.reshape(1, -1), (0, 0), 2.0,
                                  borderType=cv2.BORDER_REPLICATE)[0]
        cuts = []
        for k in range(layers - 1):
            t = (c[k] + c[k + 1]) / 2
            lo, hi = t - 0.5 * (t - c[k]), t + 0.5 * (c[k + 1] - t)
            sel = np.nonzero((mids >= lo) & (mids <= hi))[0]
            if len(sel):
                best = sel[smooth[sel] <= smooth[sel].min() * 1.02 + 1e-9]
                t = mids[best[np.abs(mids[best] - t).argmin()]]
            cuts.append(t)
        cuts = np.array(cuts)
    else:
        raise ValueError(f"Unknown split method '{method}' (use {', '.join(SPLIT_METHODS)})")
    cuts = np.clip(cuts, 1e-4, 1 - 1e-4)
    for k in range(1, len(cuts)):  # strictly increasing
        cuts[k] = max(cuts[k], cuts[k - 1] + 1e-4)
    return cuts.astype(np.float32)


def _fill_nearest(labels: np.ndarray, unknown: np.ndarray, n: int, wrap: bool) -> np.ndarray:
    """Give every unknown pixel the label of the nearest known pixel."""
    known = ~unknown
    if not unknown.any() or not known.any():
        return labels
    pad = labels.shape[1] // 2

    def dist_to(i):
        src = np.where(known & (labels == i), 0, 255).astype(np.uint8)
        if (src == 255).all():
            return np.full(labels.shape, np.inf, np.float32)
        return _xwrap(lambda a: cv2.distanceTransform(a, cv2.DIST_L2, 5), src, pad, wrap)

    present = [i for i in range(n) if (known & (labels == i)).any()]
    dists = np.stack([dist_to(i) for i in present])
    out = labels.copy()
    out[unknown] = np.asarray(present, labels.dtype)[dists.argmin(0)][unknown]
    return out


def snap_edges(depth: np.ndarray, radius: int, gate: float = 0.12,
               wrap: bool = False) -> np.ndarray:
    """Toggle filter at depth discontinuities: where the local depth range exceeds `gate`,
    snap each value to the local min or max, whichever is closer. A blurred near->far
    edge otherwise passes through mid depths and leaves a thin false 'mid' outline (and
    leaf gaps in a tree become 'mid'). Smooth slopes (ground, sky) are left alone."""
    if radius < 1:
        return depth
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    hi = _xwrap(lambda a: cv2.dilate(a, k), depth, radius + 1, wrap)
    lo = _xwrap(lambda a: cv2.erode(a, k), depth, radius + 1, wrap)
    snapped = np.where(hi - depth < depth - lo, hi, lo)
    return np.where(hi - lo > gate, snapped, depth).astype(np.float32)


def _mode_filter(labels: np.ndarray, n: int, k: int, wrap: bool) -> np.ndarray:
    """Majority vote in a k x k window (a median would invent 'mid' between near and far)."""
    votes = np.stack([_xwrap(lambda a: cv2.blur(a, (k, k)), (labels == i).astype(np.float32),
                             k, wrap) for i in range(n)])
    return votes.argmax(0).astype(labels.dtype)


def _clean_labels(labels: np.ndarray, n: int, smooth: int, min_area: int, wrap: bool) -> np.ndarray:
    """Majority-filter the label map, then dissolve islands smaller than min_area (specks
    AND small holes) into whatever surrounds them."""
    if smooth > 1 and n > 1:
        labels = _mode_filter(labels, n, _odd(smooth), wrap)
    if min_area <= 1 or n < 2:
        return labels
    pad = min(labels.shape[1] // 2, max(8, int(np.sqrt(min_area)) * 3))
    unknown = np.zeros(labels.shape, bool)
    for i in range(n):
        def small_islands(m):
            _, cc, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            small = np.nonzero(stats[1:, cv2.CC_STAT_AREA] < min_area)[0] + 1
            return np.isin(cc, small)
        unknown |= _xwrap(small_islands, (labels == i).astype(np.uint8), pad, wrap)
    if unknown.all():
        return labels
    return _fill_nearest(labels, unknown, n, wrap)


def label_layers(rgb: np.ndarray, depth: np.ndarray, layers: int = 3, *, method: str = "kmeans",
                 thresholds=None, refine: bool = True, smooth: float = 1.0,
                 min_island: float = 40, px: float | None = None, wrap: bool = False):
    """Depth -> cleaned label map (0 = far ... layers-1 = near) at the image's resolution.
    Returns (labels uint8 (H, W), cuts). Sizes are in output pixels (see split_layers)."""
    h, w = depth.shape[:2]
    px = px or min(h, w) / REF_LINES
    d = np.clip(np.asarray(depth, np.float32), 0, 1)
    if refine:
        r = max(1, round(1.5 * px))
        guide = rgb[..., :3].astype(np.float32) / 255
        d = _xwrap(lambda a: guided_filter(a[..., :3], a[..., 3], r, 2e-3),
                   np.dstack([guide, d]), 3 * r, wrap)
        d = snap_edges(d, max(1, round(1.5 * px)), wrap=wrap)
    cuts = (np.sort(np.asarray(thresholds, np.float32)) if thresholds is not None
            else depth_thresholds(d, layers, method))
    if len(cuts) != layers - 1:
        raise ValueError(f"{layers} layers need {layers - 1} thresholds, got {len(cuts)}")
    labels = np.digitize(d, cuts).astype(np.uint8)
    labels = _clean_labels(labels, layers, smooth * px, int(min_island * px * px), wrap)
    return labels, cuts


# --- filling what nearer layers hide ---------------------------------------------------

def _blur_known(rgb: np.ndarray, known: np.ndarray, sigma: float, wrap: bool) -> np.ndarray:
    """Normalised-convolution blur: averages only known pixels (no colour from holes)."""
    if sigma <= 0:
        return rgb.astype(np.float32)
    k = known.astype(np.float32)
    blur = lambda a: cv2.GaussianBlur(a, (0, 0), sigma)
    pad = int(3 * sigma) + 1
    num = _xwrap(blur, rgb.astype(np.float32) * k[..., None], pad, wrap)
    den = _xwrap(blur, k, pad, wrap)
    return np.where(den[..., None] > 1e-4, num / np.maximum(den, 1e-6)[..., None], 0)


def _row_neighbours(known: np.ndarray, wrap: bool):
    """For every pixel: column of the nearest known pixel to its left / right in the same
    row, in a (3x tiled when wrap) coordinate system. Missing = -1 / width."""
    h, w = known.shape
    reps = 3 if wrap else 1
    k = np.tile(known, (1, reps))
    cols = np.arange(w * reps)[None].repeat(h, 0)
    left = np.maximum.accumulate(np.where(k, cols, -1), axis=1)
    right = np.minimum.accumulate(np.where(k, cols, w * reps)[:, ::-1], axis=1)[:, ::-1]
    if wrap:
        left, right, cols = left[:, w:2 * w], right[:, w:2 * w], cols[:, w:2 * w]
    return left, right, cols, w * reps


def _nearest_rows(row_ok: np.ndarray) -> np.ndarray:
    good = np.nonzero(row_ok)[0]
    return good[np.abs(np.arange(len(row_ok))[:, None] - good[None]).argmin(1)]


def assign_hidden(labels: np.ndarray, hidden: np.ndarray, wrap: bool = False) -> np.ndarray:
    """Label each hidden pixel with the nearest visible label in its ROW (left or right,
    around the seam when wrap): layer boundaries carry on horizontally behind an
    occluder, the way horizons and hill lines do. Fully hidden rows copy the nearest row."""
    known = ~hidden
    if not hidden.any() or not known.any():
        return labels
    h = labels.shape[0]
    left, right, cols, n = _row_neighbours(known, wrap)
    lab = np.tile(labels, (1, 3 if wrap else 1))
    rows = np.arange(h)[:, None]
    ll = lab[rows, np.clip(left, 0, n - 1)]
    rl = lab[rows, np.clip(right, 0, n - 1)]
    use_left = (left >= 0) & ((right >= n) | (cols - left <= right - cols))
    out = labels.copy()
    out[hidden] = np.where(use_left, ll, rl)[hidden]
    row_ok = known.any(1)
    if not row_ok.all():
        src = _nearest_rows(row_ok)
        out[~row_ok] = out[src[~row_ok]]
    return out


def fill_rows(rgb: np.ndarray, known: np.ndarray, wrap: bool = False, sigma: float = 1.0,
              mirror: bool = True) -> np.ndarray:
    """Fill unknown pixels row by row from the known pixels to their left and right
    (around the seam when wrap). mirror=True reflects the texture on each side into the
    gap and cross-fades the two reflections (keeps clouds/foliage textured); False
    interpolates between the two edge colours (smooth, can streak). Rows with nothing
    known reflect the nearest known rows. Returns float32 (H, W, 3)."""
    h, w = known.shape
    out = rgb.astype(np.float32).copy()
    if known.all() or not known.any():
        return out if known.all() else np.zeros_like(out)
    left, right, cols, n = _row_neighbours(known, wrap)
    reps = n // w
    k = np.tile(known, (1, reps))
    edge = np.tile(_blur_known(rgb, known, sigma, wrap), (1, reps, 1))
    tex = np.tile(rgb.astype(np.float32), (1, reps, 1))
    rows = np.arange(h)[:, None]
    has_l, has_r = left >= 0, right < n
    li, ri = np.clip(left, 0, n - 1), np.clip(right, 0, n - 1)
    cl, cr = edge[rows, li], edge[rows, ri]
    flat = np.ones(known.shape, bool)  # pixels left with a plain edge colour (can streak)
    if mirror:
        ml = np.clip(2 * li + 1 - cols, 0, n - 1)   # reflect about the left edge
        mr = np.clip(2 * ri - 1 - cols, 0, n - 1)   # ... and the right edge
        okl, okr = k[rows, ml] & has_l, k[rows, mr] & has_r
        tl, tr = tex[rows, ml], tex[rows, mr]
        # a side whose reflection lands on a hole borrows the other side's texture
        cl = np.where(okl[..., None], tl, np.where(okr[..., None], tr, cl))
        cr = np.where(okr[..., None], tr, np.where(okl[..., None], tl, cr))
        flat = ~(okl | okr)
    span = np.maximum(ri - li, 1)
    t = np.clip((cols - li) / span, 0, 1)[..., None]
    t = t * t * (3 - 2 * t)  # smoothstep: short cross-fade in the middle
    fill = np.where((has_l & has_r)[..., None], cl * (1 - t) + cr * t,
                    np.where(has_l[..., None], cl, cr))
    hole = ~known
    out[hole] = fill[hole]
    row_ok = known.any(1)
    if not row_ok.all():  # reflect known rows into fully unknown ones
        good = np.nonzero(row_ok)[0]
        y = np.arange(h)
        src = _nearest_rows(row_ok)
        refl = np.clip(2 * src - y, good.min(), good.max())
        refl = np.where(row_ok[refl], refl, src)
        out[~row_ok] = out[refl[~row_ok]]
    soft_px = hole & flat  # soften row-to-row streaks where no texture could be copied
    if soft_px.any():
        soft = cv2.GaussianBlur(out, (1, 0), sigmaX=0.1, sigmaY=max(0.5, 2 * sigma))
        out[soft_px] = soft[soft_px]
    return out


def _inpaint(rgb: np.ndarray, known: np.ndarray, mode: str, radius: float,
             wrap: bool) -> np.ndarray:
    if mode in ("mirror", "rows"):
        return fill_rows(rgb, known, wrap, sigma=radius / 2, mirror=mode == "mirror")
    flag = {"telea": cv2.INPAINT_TELEA, "ns": cv2.INPAINT_NS}.get(mode)
    if flag is None:
        raise ValueError(f"Unknown fill '{mode}' (use {', '.join(FILL_MODES)})")
    hole = (~known).astype(np.uint8) * 255
    stacked = np.dstack([rgb.astype(np.uint8), hole])
    res = _xwrap(lambda a: cv2.inpaint(np.ascontiguousarray(a[..., :3]),
                                       np.ascontiguousarray(a[..., 3]), radius, flag),
                 stacked, rgb.shape[1] // 4, wrap)
    return res.astype(np.float32)


def split_layers(rgb: np.ndarray, depth: np.ndarray, layers: int = 3, *, method: str = "kmeans",
                 thresholds=None, fill: str = "mirror", refine: bool = True, smooth: float = 1.0,
                 min_island: float = 40, fringe: float = 1.0, feather: float = 0.35,
                 px: float | None = None, wrap: bool = False, work_px: float = 3.0,
                 return_labels: bool = False):
    """Split a photo into `layers` RGBA layers by depth, far -> near, all on the photo's
    canvas (H, W). depth: float (H, W) 0..1, 1 = nearest.

    - layer 0 (far) is fully opaque;
    - each layer is opaque where it is visible, plus the part of what nearer layers hide
      that it continues into along the row (filled, see the module docstring);
      transparent where a farther layer belongs;
    - edges are soft (feathered alpha); prepare() turns them into hard pixel edges.

    px: photo pixels per final output pixel (make_parallax passes the patch size); the
    size knobs smooth (majority-vote window), min_island (area, px^2), fringe (re-filled band
    next to nearer layers) and feather (alpha blur) are in OUTPUT pixels. Default px
    assumes a 180-line result. Work happens at ~work_px pixels per output pixel.
    method / thresholds: see depth_thresholds (thresholds = explicit 0..1 cuts).
    fill: mirror (default) | rows | telea | ns. wrap: the layer repeats horizontally.
    return_labels: also return the cleaned label map and the cuts.
    """
    rgb = np.ascontiguousarray(np.asarray(rgb)[..., :3].astype(np.uint8))
    h, w = rgb.shape[:2]
    depth = np.asarray(depth, np.float32)
    if depth.shape[:2] != (h, w):
        depth = _resize(depth, w, h)
    layers = int(layers)
    if layers < 1:
        raise ValueError("layers must be >= 1")
    px = float(px or min(h, w) / REF_LINES)
    f = min(1.0, work_px / px)
    ww, wh = max(8, round(w * f)), max(8, round(h * f))
    u = px * ww / w                                   # work pixels per output pixel
    small = _resize(rgb, ww, wh, cv2.INTER_AREA)
    d_small = _resize(depth, ww, wh, cv2.INTER_AREA)
    labels, cuts = label_layers(small, d_small, layers, method=method, thresholds=thresholds,
                                refine=refine, smooth=smooth, min_island=min_island, px=u,
                                wrap=wrap)

    fringe_k = _odd(2 * fringe * u) if fringe > 0 else 0
    out = []
    for i in range(layers):
        own = labels == i
        nearer = labels > i
        if i == 0:
            region = np.ones_like(own)
        else:  # hidden pixels go to the farther layer that continues behind them
            region = assign_hidden(labels, nearer, wrap) == i
        source = own.copy()
        if fringe_k > 1 and nearer.any():
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fringe_k, fringe_k))
            grown = _xwrap(lambda a, k=kernel: cv2.dilate(a, k), nearer.astype(np.uint8),
                           fringe_k, wrap).astype(bool)
            trimmed = own & ~grown
            source = trimmed if trimmed.sum() >= 0.25 * own.sum() else own
        if not source.any():
            source = own if own.any() else np.ones_like(own)
        filled = small.astype(np.float32)
        if not source.all():
            # copy texture only from solid areas: slivers (sky seen through leaves) would
            # seed the fill with streaks and stray leaf colours
            ko = _odd(2 * u)
            ell = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ko, ko))
            solid = _xwrap(lambda a, e=ell: cv2.morphologyEx(a, cv2.MORPH_OPEN, e),
                           source.astype(np.uint8), ko, wrap).astype(bool)
            seed = solid if solid.sum() >= 0.5 * source.sum() else source
            filled = _inpaint(small, seed, fill, max(1.0, 1.5 * u), wrap)
        # back to full resolution: real photo pixels where the layer is itself, fill elsewhere
        weight = _resize(source.astype(np.float32), w, h)
        fill_full = _resize(filled, w, h, cv2.INTER_CUBIC)
        layer_rgb = rgb.astype(np.float32) * weight[..., None] + fill_full * (1 - weight[..., None])
        if i == 0:
            alpha = np.ones((h, w), np.float32)
        else:
            a = region.astype(np.float32)
            if feather > 0:
                sig = feather * u
                a = _xwrap(lambda x, s=sig: cv2.GaussianBlur(x, (0, 0), s), a, int(3 * sig) + 2,
                           wrap)
            alpha = np.clip(_resize(a, w, h), 0, 1)
        rgba = np.dstack([np.clip(np.rint(layer_rgb), 0, 255).astype(np.uint8),
                          np.rint(alpha * 255).astype(np.uint8)])
        out.append(rgba)
    if return_labels:
        return out, labels, cuts
    return out


# --- photo -> pixel-art layers ---------------------------------------------------------

@dataclass
class ParallaxLayers:
    sprites: list[Sprite]      # far -> near, same size, one shared palette
    names: list[str]           # far, mid, near ...
    palette: np.ndarray        # (N, 3)
    depth: np.ndarray          # (H, W) float 0..1 at output size (1 = near)
    labels: np.ndarray         # (h, w) label map at work resolution (0 = far)
    cuts: np.ndarray           # depth thresholds used


def _load_depth(depth, h: int, w: int) -> np.ndarray:
    """A depth map given as an array or an image file (white = near)."""
    if isinstance(depth, (str, Path)):
        img = Image.open(depth)
        arr = np.asarray(img)
        if arr.ndim == 3:
            arr = np.asarray(img.convert("L"))
        maxv = 65535.0 if arr.dtype == np.uint16 or arr.max() > 255 else 255.0
        d = arr.astype(np.float32) / maxv
    else:
        d = np.asarray(depth, np.float32)
        if d.ndim == 3:
            d = d[..., :3].mean(-1)
        if d.max() > 1.0:
            d = d / (65535.0 if d.max() > 255 else 255.0)
    return np.clip(_resize(d, w, h), 0, 1)


def _prepare_wrapped(rgba: np.ndarray, s: Settings, pad: int) -> Prepared:
    """prepare() a horizontally repeating layer: wrap-pad it by `pad` output pixels each
    side so edge-aware pixelization sees the far edge as neighbour, then trim."""
    if pad <= 0:
        return prepare(rgba, s)
    padded = _pad_x(rgba, pad * s.patch)
    prep = prepare(padded, dataclasses.replace(s, width=s.width + 2 * pad))
    return Prepared(np.ascontiguousarray(prep.rgb[:, pad:-pad]),
                    np.ascontiguousarray(prep.mask[:, pad:-pad]))


def _visible_pixels(preps: list[Prepared]) -> np.ndarray:
    """Colours that can ever be on screen: each layer's opaque pixels, minus rows that
    nearer layers cover edge to edge (those stay hidden at any scroll position), so
    never-seen fill doesn't use up palette slots."""
    picks = []
    covered = np.zeros(preps[0].mask.shape, bool)
    for pr in reversed(preps):  # near -> far
        full_rows = covered.all(1)
        picks.append(pr.rgb[pr.mask & ~full_rows[:, None]])
        covered |= pr.mask
    pixels = np.concatenate(picks)
    return pixels if len(pixels) else preps[0].rgb.reshape(-1, 3)


def parallax_settings(width: int = 320, height: int = 180, palette: str = "auto", colors: int = 32,
                      dither: str = "bayer4", **overrides) -> Settings:
    """The background preset, forced onto one fixed canvas so all layers line up."""
    forced = dict(fit="cover", crop_to_subject=False, outline=False, remove_background=False,
                  tileable=False)  # tileable: wrap is handled here, x only
    for key in forced:
        overrides.pop(key, None)
    s = settings_for("background", width=width, height=height, palette=palette, colors=colors,
                     dither=dither, **overrides)
    return dataclasses.replace(s, **forced)


def build_parallax(photo: ImageSource, layers: int = 3, width: int = 320, height: int | None = 180,
                   palette: str = "auto", colors: int = 32, dither: str = "bayer4",
                   wrap: bool = True, *, depth=None, method: str = "kmeans", thresholds=None,
                   fill: str = "mirror", blend: float = 0.125,
                   **setting_overrides) -> ParallaxLayers:
    """Photo -> pixel-art parallax layers in memory (no files). See make_parallax."""
    rgba = load_image(photo)
    rgb = np.ascontiguousarray(rgba[..., :3])
    if depth is None:
        from .depth import estimate_depth
        h, w = rgb.shape[:2]
        f = min(1.0, 1036 / max(h, w))  # the network sees <= 1036 px anyway
        small = _resize(rgb, max(1, round(w * f)), max(1, round(h * f)), cv2.INTER_AREA)
        d = _resize(estimate_depth(small), w, h)
    else:
        d = _load_depth(depth, rgb.shape[0], rgb.shape[1])
    if not height:
        height = max(1, round(width * rgb.shape[0] / rgb.shape[1]))
    s = parallax_settings(width, height, palette, colors, dither, **setting_overrides)
    p = s.patch
    cw, ch = s.width * p, s.height * p
    extra = round(blend * cw) if wrap else 0
    rgb_c = _cover(rgb, cw + extra, ch)
    d_c = _cover(d, cw + extra, ch)
    if extra:
        rgb_c, d_c = seamless_x(rgb_c, extra), seamless_x(d_c, extra)
    parts, labels, cuts = split_layers(rgb_c, d_c, layers, method=method, thresholds=thresholds,
                                       fill=fill, px=p, wrap=wrap, return_labels=True)
    pad = min(4, s.width) if wrap else 0
    preps = [_prepare_wrapped(part, s, pad) for part in parts]
    pal = resolve_palette(s, _visible_pixels(preps))
    sprites = [finish(pr, s, pal) for pr in preps]
    depth_out = _resize(d_c, s.width, s.height, cv2.INTER_AREA)
    return ParallaxLayers(sprites, layer_names(layers), sprites[0].palette, depth_out, labels, cuts)


def composite(sprites: list[Sprite], offsets: list[int] | None = None) -> np.ndarray:
    """Stack layers far -> near (optionally each shifted right by offsets[i], wrapping)."""
    out = np.zeros_like(sprites[0].rgba)
    for i, sp in enumerate(sprites):
        layer = sp.rgba if not offsets else np.roll(sp.rgba, int(offsets[i]), axis=1)
        a = layer[..., 3:4].astype(np.float32) / 255
        out = np.rint(layer * a + out * (1 - a)).astype(np.uint8)
    return out


def make_parallax(photo: ImageSource, out_dir, name: str | None = None, layers: int = 3,
                  width: int = 320, height: int | None = 180, palette: str = "auto",
                  colors: int = 32, dither: str = "bayer4", wrap: bool = True,
                  scroll_scales: list[float] | None = None, *, depth=None, method: str = "kmeans",
                  thresholds=None, fill: str = "mirror", godot: bool = True, scale: int = 1,
                  save_depth: bool = False, **setting_overrides) -> dict:
    """Photo -> `layers` pixel-art parallax layers sharing ONE palette, plus a preview and
    a Godot 4 scene.

    Writes to out_dir:
      <name>_layer0_far.png, <name>_layer1_mid.png, <name>_layer2_near.png  (1x, RGBA)
      <name>_preview.png        all layers stacked (what the camera sees at rest)
      <name>_parallax.tscn      Parallax2D scene (godot=True)
      <name>_depth.png          depth at output size, white = near (save_depth=True)
    scale > 1 also writes nearest-upscaled copies of the preview (<name>_preview@Nx.png).

    wrap: layers repeat seamlessly left-right (x only). depth: skip the model and use this
    depth map (array or image, white = near; handy for a hand-painted one). method /
    thresholds / fill: see split_layers. Other keyword args are Settings fields (patch,
    detail, dither_strength, saturation, ...). Returns {"layers": [...], "preview": path,
    "godot": path, "depth": path, "scroll_scales": [...], "cuts": [...]}.
    """
    src_name = Path(photo).stem if isinstance(photo, (str, Path)) else "parallax"
    name = name or src_name
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = build_parallax(photo, layers, width, height, palette, colors, dither, wrap, depth=depth,
                         method=method, thresholds=thresholds, fill=fill, **setting_overrides)
    paths = []
    for i, (sp, role) in enumerate(zip(res.sprites, res.names)):
        path = out_dir / f"{name}_layer{i}_{role}.png"
        sp.save(path)
        paths.append(path)
    preview = out_dir / f"{name}_preview.png"
    comp = composite(res.sprites)
    Image.fromarray(comp, "RGBA").save(preview)
    result = {"layers": paths, "preview": preview}
    if scale > 1:
        big = out_dir / f"{name}_preview@{scale}x.png"
        h, w = comp.shape[:2]
        Image.fromarray(comp, "RGBA").resize((w * scale, h * scale), Image.NEAREST).save(big)
        result["preview_scaled"] = big
    scales = list(scroll_scales) if scroll_scales else default_scroll_scales(layers)
    if len(scales) != layers:
        raise ValueError(f"{layers} layers need {layers} scroll scales, got {len(scales)}")
    if godot:
        result["godot"] = write_godot_parallax(paths, out_dir / f"{name}_parallax.tscn", scales,
                                               repeat=wrap, root_name=f"{name}_parallax")
    if save_depth:
        dpath = out_dir / f"{name}_depth.png"
        Image.fromarray(np.rint(res.depth * 255).astype(np.uint8), "L").save(dpath)
        result["depth"] = dpath
    result["scroll_scales"] = scales
    result["cuts"] = [round(float(c), 4) for c in res.cuts]
    return result


# --- Godot 4 ---------------------------------------------------------------------------

def _node_name(text: str) -> str:
    name = re.sub(r'[.:@/"%\s]+', "_", text).strip("_")
    return name or "Parallax"


def write_godot_parallax(layer_pngs, out_tscn, scroll_scales: list[float] | None = None,
                         repeat: bool = True, *, repeat_times: int = 2, res_dir: str | None = None,
                         root_name: str | None = None, z_index: int | None = -10) -> Path:
    """Godot 4.3+ scene: a Node2D with one Parallax2D per layer (far first, so nearer layers
    draw on top), each holding a Sprite2D (top-left at 0,0, nearest filtering).

    scroll_scale = Vector2(s, 1): s < 1 scrolls slower than the camera (further away);
    vertical follows the camera 1:1. repeat: repeat_size = (layer width, 0) so the layer
    tiles horizontally forever; repeat_times copies each side cover wide / zoomed-out views.
    Texture paths are relative to the .tscn (keep the PNGs next to it, anywhere in the
    project), or res_dir + file name if given (e.g. "res://art/bg/").
    z_index: put on the root so the background draws behind default-z level nodes
    wherever the scene is instanced (None = leave at 0).
    """
    pngs = [Path(p) for p in layer_pngs]
    out_tscn = Path(out_tscn)
    n = len(pngs)
    if n == 0:
        raise ValueError("no layers")
    scales = list(scroll_scales) if scroll_scales else default_scroll_scales(n)
    if len(scales) != n:
        raise ValueError(f"{n} layers need {n} scroll scales, got {len(scales)}")
    roles = layer_names(n)
    out_tscn.parent.mkdir(parents=True, exist_ok=True)
    lines = ["[gd_scene format=3]", ""]
    for i, (png, role) in enumerate(zip(pngs, roles)):
        if res_dir:
            path = res_dir.rstrip("/") + "/" + png.name
        else:
            path = Path(os.path.relpath(png.resolve(), out_tscn.parent.resolve())).as_posix()
        lines.append(f'[ext_resource type="Texture2D" path="{path}" id="{i + 1}_{role}"]')
    lines += ["", f'[node name="{_node_name(root_name or out_tscn.stem)}" type="Node2D"]']
    if z_index:
        lines.append(f"z_index = {int(z_index)}")
    for i, (png, role, s) in enumerate(zip(pngs, roles, scales)):
        with Image.open(png) as im:
            w, _ = im.size
        node = role.capitalize()
        lines += ["", f'[node name="{node}" type="Parallax2D" parent="."]',
                  f"scroll_scale = Vector2({float(s):g}, 1)"]
        if repeat:
            lines += [f"repeat_size = Vector2({w}, 0)", f"repeat_times = {int(repeat_times)}"]
        lines += ["", f'[node name="Sprite2D" type="Sprite2D" parent="{node}"]',
                  "texture_filter = 1", f'texture = ExtResource("{i + 1}_{role}")',
                  "centered = false"]
    out_tscn.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_tscn

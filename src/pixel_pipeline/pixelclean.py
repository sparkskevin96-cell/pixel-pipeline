"""Pixel clean-up for rendered sprites, so 3D renders look placed by hand.

A splat render shrunk to 64 px leaves the marks of the machine: lone pixels floating off the
silhouette, one- and two-pixel islands of a shade where the lighting bands meet, doubled
"L" corners where the outline steps along a slope. clean_sprites() fixes those on the
palette indices, the same way for every frame of a sheet:

1. specks: tiny pieces apart from the body (a lone pixel and its outline) are removed
2. pinholes: single see-through pixels inside the body are filled (no-outline sprites)
3. colour islands: same-colour patches of 1-2 pixels inside the body take the colour around
   them, unless they stand out strongly (an eye, a glint stays)
4. pixel-perfect outline: one-pixel nubs on a flat edge become outline, and outline pixels
   no body pixel needs (a doubled "L" corner whose two outline neighbours already touch
   diagonally, a stub) are removed. finish()'s 4-connected ring has no doubled corners by
   itself; they appear after the nubs go, or in sprites outlined some other way
5. the selout outline is shaded again from the body colours it now touches, so it stays a
   consistent 1 px darker edge
6. optional `palette`: the sprites' shared palette is snapped to a project palette first,
   one mapping for all frames, so an animation can't flicker between colours

Normal maps (sprite.normal) keep their alignment: removed pixels are removed there too and
filled pixels take their neighbours' normal.
"""
from __future__ import annotations

import cv2
import numpy as np

from .color import rgb_to_lab
from .finishing import selout_colors
from .pipeline import Sprite

_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_K3 = np.ones((3, 3), np.uint8)


def _shift(a: np.ndarray, dy: int, dx: int, fill) -> np.ndarray:
    """a moved so out[y, x] = a[y + dy, x + dx]; `fill` outside."""
    out = np.full_like(a, fill)
    h, w = a.shape
    ys, yd = (slice(dy, h), slice(0, h - dy)) if dy >= 0 else (slice(0, h + dy), slice(-dy, h))
    xs, xd = (slice(dx, w), slice(0, w - dx)) if dx >= 0 else (slice(0, w + dx), slice(-dx, w))
    out[yd, xd] = a[ys, xs]
    return out


def boundary(mask: np.ndarray) -> np.ndarray:
    """Opaque pixels with a see-through (or off-image) 4-neighbour: the outline ring of a
    finished sprite."""
    out = np.zeros_like(mask)
    for dy, dx in _N4:
        out |= ~_shift(mask, dy, dx, False)
    return out & mask


def snap_palette(palette: np.ndarray, target) -> tuple[np.ndarray, np.ndarray]:
    """(new palette, lut old index -> new index): every colour to its nearest target colour
    (Lab), duplicates merged."""
    if isinstance(target, str):
        from .palettes import load_palette
        loaded = load_palette(target)
        if loaded is None:
            return palette, np.arange(len(palette))
        target = loaded
    target = np.asarray(target, np.uint8).reshape(-1, 3)
    lab, tl = rgb_to_lab(palette.astype(np.float32)), rgb_to_lab(target.astype(np.float32))
    pick = ((lab[:, None] - tl[None]) ** 2).sum(-1).argmin(1)
    used, lut = np.unique(pick, return_inverse=True)
    return target[used], lut.reshape(-1)


def _remove_specks(idx: np.ndarray, speck: int) -> np.ndarray:
    """Pieces (8-connected) of at most `speck` pixels apart from the biggest one."""
    mask = (idx >= 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 2:
        return np.zeros(idx.shape, bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    main = 1 + int(np.argmax(areas))
    small = [i for i in range(1, n) if i != main and stats[i, cv2.CC_STAT_AREA] <= speck]
    return np.isin(labels, small) if small else np.zeros(idx.shape, bool)


def _fill_pinholes(idx: np.ndarray) -> np.ndarray:
    """See-through pixels with all four neighbours opaque -> the most common neighbour.
    Returns the filled pixels."""
    mask = idx >= 0
    nbs = np.stack([_shift(idx, dy, dx, -1) for dy, dx in _N4], -1)
    hole = ~mask & (nbs >= 0).all(-1)
    for y, x in zip(*np.nonzero(hole)):
        idx[y, x] = np.bincount(nbs[y, x]).argmax()
    return hole


def _merge_islands(idx: np.ndarray, body: np.ndarray, lab: np.ndarray, size: int,
                   delta: float, passes: int = 2) -> None:
    """Same-colour patches (4-connected) of at most `size` body pixels take the colour they
    share the most border with, when that is within `delta` (Lab) of theirs. In place."""
    h, w = idx.shape
    for _ in range(passes):
        changed = False
        for v in np.unique(idx[body]):
            m = ((idx == v) & body).astype(np.uint8)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
            for i in range(1, n):
                x, y, bw, bh, area = stats[i]
                if area > size:
                    continue
                y0, y1, x0, x1 = max(0, y - 1), min(h, y + bh + 1), max(0, x - 1), min(w, x + bw + 1)
                comp = labels[y0:y1, x0:x1] == i
                ring = cv2.dilate(comp.astype(np.uint8), _K3).astype(bool) & ~comp
                ring &= body[y0:y1, x0:x1]
                around = idx[y0:y1, x0:x1][ring]
                around = around[around != v]
                if not len(around):
                    continue
                best = int(np.bincount(around).argmax())
                if np.linalg.norm(lab[best] - lab[v]) < delta:
                    idx[y0:y1, x0:x1][comp] = best
                    changed = True
        if not changed:
            break


def _flatten_spurs(idx: np.ndarray, body: np.ndarray) -> None:
    """One-pixel nubs on a flat body edge (render noise) become outline: the body pixel
    with one body neighbour, whose base pixel has body on both sides across. In place."""
    h, w = idx.shape
    for y, x in zip(*np.nonzero(body)):
        nbs = [(y + dy, x + dx, dy, dx) for dy, dx in _N4
               if 0 <= y + dy < h and 0 <= x + dx < w and body[y + dy, x + dx]]
        if len(nbs) != 1:
            continue
        by, bx, dy, dx = nbs[0]
        across = [(by + dx, bx + dy), (by - dx, bx - dy)]
        if not all(0 <= a < h and 0 <= b < w and body[a, b] for a, b in across):
            continue
        ring = [idx[y + ey, x + ex] for ey, ex in _N4
                if 0 <= y + ey < h and 0 <= x + ex < w and idx[y + ey, x + ex] >= 0
                and not body[y + ey, x + ex]]
        if ring:
            idx[y, x] = ring[0]
            body[y, x] = False


def _pixel_perfect(idx: np.ndarray, body: np.ndarray) -> np.ndarray:
    """Remove outline pixels no body pixel needs: a corner that only doubles an L (its two
    outline neighbours already touch diagonally) or a stub sticking off the outline.
    Returns the removed pixels."""
    h, w = idx.shape
    mask = idx >= 0
    removed = np.zeros_like(mask)

    def at(y, x, grid):
        return 0 <= y < h and 0 <= x < w and grid[y, x]
    changed = True
    while changed:
        changed = False
        for y, x in zip(*np.nonzero(mask & ~body)):
            if any(at(y + dy, x + dx, body) for dy, dx in _N4):
                continue                              # it keeps a body pixel enclosed
            horiz = [at(y, x + d, mask) for d in (-1, 1)]
            vert = [at(y + d, x, mask) for d in (-1, 1)]
            corner = sum(horiz) == 1 and sum(vert) == 1
            stub = sum(horiz) + sum(vert) == 1
            if not (corner or stub):
                continue
            mask[y, x] = False
            removed[y, x] = True
            changed = True
    idx[removed] = -1
    return removed


def _fill_normals(normal: np.ndarray, want: np.ndarray, have: np.ndarray) -> None:
    """Give `want` pixels the mean normal of their `have` 8-neighbours (in place)."""
    from .normals import decode, encode
    vec = decode(normal[..., :3]) * have[..., None]
    k = np.ones((3, 3), np.float32)
    s = np.stack([cv2.filter2D(vec[..., c].astype(np.float32), -1, k, borderType=cv2.BORDER_CONSTANT)
                  for c in range(3)], -1)
    s[np.linalg.norm(s, axis=-1) < 1e-6] = (0, 0, 1)
    s /= np.linalg.norm(s, axis=-1, keepdims=True)
    normal[want, :3] = encode(s[want])
    normal[want, 3] = 255


def clean_sprite(sprite: Sprite, outline: str = "selout", *, speck: int = 5, islands: int = 2,
                 island_delta: float = 30.0, corners: bool = True, holes: bool = True,
                 lut: np.ndarray | None = None, palette: np.ndarray | None = None) -> Sprite:
    """One sprite cleaned (see the module notes). outline: the sprite's outline style
    (selout, solid, none). islands: biggest patch merged (0 = off, e.g. for dithered art).
    lut/palette: from snap_palette (index remap + the new palette)."""
    idx = sprite.indices.astype(np.int32).copy()
    pal = sprite.palette if palette is None else palette
    if lut is not None:
        idx = np.where(idx >= 0, lut[np.maximum(idx, 0)], -1)
    old_mask = sprite.rgba[..., 3] > 0
    lab = rgb_to_lab(pal.astype(np.float32))
    gone = _remove_specks(idx, speck) if speck else np.zeros(idx.shape, bool)
    idx[gone] = -1
    if holes and outline == "none":
        _fill_pinholes(idx)
    has_outline = outline in ("selout", "solid")
    mask = idx >= 0
    body = mask & ~boundary(mask) if has_outline else mask
    if islands:
        _merge_islands(idx, body, lab, islands, island_delta)
    if corners and has_outline:          # without an outline these would be body pixels
        _flatten_spurs(idx, body)
        gone |= _pixel_perfect(idx, body)
    mask = idx >= 0
    if outline == "selout":
        ring = mask & ~body
        inner = np.where(ring, -1, idx)
        shade = selout_colors(inner, ring, pal)
        idx[ring] = shade[ring]
    rgba = np.zeros(idx.shape + (4,), np.uint8)
    rgba[mask, :3] = pal[idx[mask]]
    rgba[mask, 3] = 255
    normal = sprite.normal
    if normal is not None:
        normal = normal.copy()
        normal[~mask] = 0
        add = mask & ~old_mask
        if add.any():
            _fill_normals(normal, add, mask & old_mask)
    return Sprite(rgba, idx.astype(sprite.indices.dtype), pal, normal)


def clean_sprites(sprites: list[Sprite], outline: str = "selout", palette=None,
                  **options) -> list[Sprite]:
    """Every sprite cleaned the same way. palette: a project palette (array or palette spec
    like 'pico8', 'lospec:<slug>', a .gpl file) to snap the shared palette to first."""
    cache: dict[int, tuple] = {}
    out = []
    for s in sprites:
        lut = pal = None
        if palette is not None:
            key = id(s.palette)
            if key not in cache:
                pal, lut = snap_palette(s.palette, palette)
                cache[key] = (pal, lut)
            pal, lut = cache[key]
        out.append(clean_sprite(s, outline, lut=lut, palette=pal, **options))
    return out


def outline_style(settings) -> str:
    """The outline style a Settings renders with: selout, solid or none."""
    return settings.outline_style if settings.outline else "none"

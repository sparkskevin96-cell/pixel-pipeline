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

from .color import lab_to_rgb, rgb_to_lab
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
    share the most border with, when that is within `delta` (Lab) of theirs. In place.

    Colours are visited in index order and each colour's patches in raster order of their
    first pixel (the order connected-component labels number them), so earlier merges feed
    later ones exactly as labelling every colour would; only pixels that can be in a small
    patch are looked at (at most size-1 same-colour neighbours, or just merged into that
    colour), the rest of the picture is never labelled."""
    if size < 1:
        return
    if size > 2:
        _merge_islands_labelled(idx, body, lab, size, delta, passes)
        return
    h, w = idx.shape
    flat_idx = idx.reshape(-1)
    flat_body = body.reshape(-1)
    for _ in range(passes):
        vals = np.unique(idx[body])
        links = [(_shift(idx, dy, dx, -1) == idx) & _shift(body, dy, dx, False) & body
                 for dy, dx in _N4]
        same = np.sum(links, 0)
        small = body & (same == 0)                       # one-pixel patches
        if size == 2:                                    # pairs: both ends have one link
            for (dy, dx), ln in zip(_N4, links):
                small |= ln & (same == 1) & (_shift(same, dy, dx, -1) == 1)
        cand = np.flatnonzero(small.reshape(-1))
        by_v: dict[int, list[int]] = {}
        for q, v in zip(cand.tolist(), flat_idx[cand].tolist()):
            by_v.setdefault(v, []).append(q)
        turned: dict[int, list[int]] = {}
        changed = False
        for v in vals.tolist():
            seeds = by_v.get(v, []) + turned.get(v, [])
            if not seeds:
                continue
            comps, done = [], set()
            for s0 in seeds:
                if s0 in done or flat_idx[s0] != v:
                    continue
                comp, stack, seen, big = [s0], [s0], {s0}, False
                while stack and not big:
                    q = stack.pop()
                    qy, qx = divmod(q, w)
                    for dy, dx in _N4:
                        ny, nx = qy + dy, qx + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            r = ny * w + nx
                            if r not in seen and flat_body[r] and flat_idx[r] == v:
                                seen.add(r)
                                comp.append(r)
                                stack.append(r)
                                if len(comp) > size:
                                    big = True
                                    break
                done |= seen
                if not big:
                    comps.append(sorted(comp))
            comps.sort(key=lambda c: c[0])
            for comp in comps:
                ring = set()
                for q in comp:
                    qy, qx = divmod(q, w)
                    for ny in (qy - 1, qy, qy + 1):
                        for nx in (qx - 1, qx, qx + 1):
                            if 0 <= ny < h and 0 <= nx < w:
                                ring.add(ny * w + nx)
                ring.difference_update(comp)
                around = [a for a in (int(flat_idx[r]) for r in ring if flat_body[r]) if a != v]
                if not around:
                    continue
                best = int(np.bincount(around).argmax())
                if np.linalg.norm(lab[best] - lab[v]) < delta:
                    flat_idx[comp] = best
                    turned.setdefault(best, []).extend(comp)
                    changed = True
        if not changed:
            break


def _merge_islands_labelled(idx: np.ndarray, body: np.ndarray, lab: np.ndarray, size: int,
                            delta: float, passes: int = 2) -> None:
    """_merge_islands by labelling every colour's patches (any `size`)."""
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
    with one body neighbour, whose base pixel has body on both sides across. In place.
    Decided in raster order on the picture as earlier decisions left it; only pixels with one
    body neighbour, or next to a pixel just turned, can change, so only those are visited."""
    import heapq
    h, w = idx.shape
    cnt = np.zeros((h, w), np.int32)
    for dy, dx in _N4:
        cnt += _shift(body, dy, dx, False)
    heap = np.flatnonzero((body & (cnt == 1)).reshape(-1)).tolist()   # sorted = a heap
    done = set()
    while heap:
        p = heapq.heappop(heap)
        if p in done:
            continue
        done.add(p)
        y, x = divmod(p, w)
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
            for ny, nx in ((y, x + 1), (y + 1, x - 1), (y + 1, x), (y + 1, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and body[ny, nx]:    # later pixels it touches
                    heapq.heappush(heap, ny * w + nx)


def _pixel_perfect(idx: np.ndarray, body: np.ndarray) -> np.ndarray:
    """Remove outline pixels no body pixel needs: a corner that only doubles an L (its two
    outline neighbours already touch diagonally) or a stub sticking off the outline.
    Returns the removed pixels. Each pass decides the outline pixels in raster order on the
    mask as earlier removals left it; only pixels that qualify when the pass starts, or sit
    next to one just removed, can qualify, so only those are visited."""
    import heapq
    h, w = idx.shape
    mask = idx >= 0
    removed = np.zeros_like(mask)
    near_body = np.zeros_like(mask)
    for dy, dx in _N4:
        near_body |= _shift(body, dy, dx, False)

    def at(y, x, grid):
        return 0 <= y < h and 0 <= x < w and grid[y, x]
    changed = True
    while changed:
        changed = False
        listed = mask & ~body
        hz = _shift(mask, 0, -1, False).astype(np.int32) + _shift(mask, 0, 1, False)
        vt = _shift(mask, -1, 0, False).astype(np.int32) + _shift(mask, 1, 0, False)
        qualify = listed & ~near_body & (((hz == 1) & (vt == 1)) | (hz + vt == 1))
        heap = np.flatnonzero(qualify.reshape(-1)).tolist()
        done = set()
        while heap:
            p = heapq.heappop(heap)
            if p in done:
                continue
            done.add(p)
            y, x = divmod(p, w)
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
            for ny, nx in ((y, x + 1), (y + 1, x)):    # later pixels that see this one
                if 0 <= ny < h and 0 <= nx < w and listed[ny, nx] and mask[ny, nx]:
                    heapq.heappush(heap, ny * w + nx)
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
                 lut: np.ndarray | None = None, palette: np.ndarray | None = None,
                 reshade: bool = True) -> Sprite:
    """One sprite cleaned (see the module notes). outline: the sprite's outline style
    (selout, solid, none). islands: biggest patch merged (0 = off, e.g. for dithered art).
    lut/palette: from snap_palette (index remap + the new palette). reshade: shade a selout
    outline again from the body colours it touches (False: the caller colours it)."""
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
    if outline == "selout" and reshade:
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


# --------------------------------------------------------------------------- hand-pixelled

# Pixel artists shift hue along a ramp: highlights toward warm yellow-orange, shadows toward
# cool blue-violet (Lab a/b directions of those hues).
_WARM = np.array([np.cos(np.radians(70.0)), np.sin(np.radians(70.0))], np.float32)
_COOL = np.array([np.cos(np.radians(285.0)), np.sin(np.radians(285.0))], np.float32)
LIGHT_DIR = np.array([-0.45, 0.6, 0.66], np.float32) / np.float32(np.linalg.norm([-0.45, 0.6, 0.66]))


def material_groups(lab: np.ndarray, counts: np.ndarray, max_groups: int = 6,
                    merge: float = 11.0, l_weight: float = 0.22) -> np.ndarray:
    """Palette colours (Lab) grouped into materials (stone, thatch, wood, glass...): colours
    close in hue and saturation join even when their lightness differs, the way one ramp
    holds a material's light and shadow. counts: pixels per colour (0 = unused). Returns a
    group number per colour (-1 for unused ones)."""
    used = [int(i) for i in np.nonzero(counts > 0)[0]]
    label = np.full(len(lab), -1, np.int32)
    if not used:
        return label
    # hue and saturation relative to lightness: a thatch roof's dark shadows are greyer than
    # its lights in Lab, but still lean the same way
    k = 65.0 / (lab[:, 0:1] + 15.0)
    feat = np.c_[lab[:, 0] * l_weight, lab[:, 1:] * k].astype(np.float64)
    groups = [[i] for i in used]

    def centre(g):
        w = counts[g].astype(np.float64)
        return (feat[g] * w[:, None]).sum(0) / w.sum()
    centres = [centre(g) for g in groups]
    while len(groups) > 1:
        c = np.array(centres)
        d = np.sqrt(((c[:, None] - c[None]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        i, j = np.unravel_index(int(np.argmin(d)), d.shape)
        if d[i, j] > merge and len(groups) <= max_groups:
            break
        i, j = min(i, j), max(i, j)
        groups[i] += groups.pop(j)
        centres.pop(j)
        centres[i] = centre(groups[i])
    for n, g in enumerate(groups):
        label[g] = n
    return label


def _levels(values: np.ndarray, steps: int, gap: float) -> np.ndarray:
    """1-D k-means of lightness into up to `steps` levels about `gap` apart."""
    if not len(values):
        return np.array([50.0])
    lo, hi = float(np.percentile(values, 2)), float(values.max())   # glints are rare: keep them
    n = int(np.clip(round((hi - lo) / gap) + 1, 1, steps))
    lev = np.linspace(lo, hi, n) if n > 1 else np.array([float(np.median(values))])
    for _ in range(12):
        k = np.abs(values[:, None] - lev[None]).argmin(1)
        for j in range(n):
            if (k == j).any():
                lev[j] = float(values[k == j].mean())
    lev = np.sort(lev)
    keep = [lev[0]]
    for v in lev[1:]:                      # levels that ran together become one
        if v - keep[-1] >= gap * 0.6:
            keep.append(v)
    return np.array(keep)


def _ramp(levels: np.ndarray, ab: np.ndarray, mid: float, shift: float) -> np.ndarray:
    """Lab colours of a material's ramp: lightness levels with the hue shifted warm in the
    lights and cool in the shadows, slightly richer shadows and paler lights."""
    span = max(float(np.abs(levels - mid).max()), 1e-6)
    # a grey (stone, iron) takes only a hint of the shift, or its lights turn tan
    shift = shift * float(np.clip(np.hypot(*ab) / 14.0, 0.3, 1.0))
    out = []
    for L in levels:
        t = float(np.clip((L - mid) / span, -1, 1))
        c = ab * (1.0 - 0.18 * t) + shift * (t * _WARM if t > 0 else -t * _COOL)
        out.append((L, c[0], c[1]))
    return np.array(out, np.float32)


def _face_bins(n: np.ndarray) -> np.ndarray:
    """Which way each (camera-space) normal faces: 0 = up (a roof, a lid), 1..6 = sideways
    sectors around the view axis."""
    up = n[..., 1] > 0.72
    az = np.arctan2(n[..., 0], n[..., 2] + 1e-6)
    sector = 1 + np.clip(((az + np.pi) / (2 * np.pi) * 6).astype(int), 0, 5)
    return np.where(up, 0, sector)


def _rgba(idx: np.ndarray, pal: np.ndarray) -> np.ndarray:
    mask = idx >= 0
    rgba = np.zeros(idx.shape + (4,), np.uint8)
    rgba[mask, :3] = pal[idx[mask]]
    rgba[mask, 3] = 255
    return rgba


def _material_outline(sp: Sprite, owner: np.ndarray, outline_of: list[int],
                      fallback: int) -> Sprite:
    """Outline pixels take the outline shade of the material they border most.
    owner: material per palette index (-1 = not a ramp colour)."""
    idx = sp.indices.astype(np.int32).copy()
    mask = idx >= 0
    body = mask & ~boundary(mask)
    ring = mask & ~body
    mats = np.where(body, owner[np.maximum(idx, 0)], -1)
    nbs = np.stack([_shift(mats, dy, dx, -1) for dy, dx in _N4], -1)
    for y, x in zip(*np.nonzero(ring)):
        v = nbs[y, x][nbs[y, x] >= 0]
        idx[y, x] = outline_of[int(np.bincount(v).argmax())] if len(v) else fallback
    return Sprite(_rgba(idx, sp.palette), idx.astype(np.int16), sp.palette, sp.normal)


def _majority_material(mat: np.ndarray, L: np.ndarray, ngroups: int, r: int = 1,
                       dark: float = 50.0) -> np.ndarray:
    """A shadow pixel (lightness under `dark`) whose material is rare around it (a
    grey-looking shadow in a thatch roof) joins the material most of its (2r+1)^2 window
    has. Light ones stay: they are glints and highlights, not mistaken shadows."""
    k = np.ones((2 * r + 1, 2 * r + 1), np.float32)
    votes = np.stack([cv2.filter2D((mat == m).astype(np.float32), -1, k,
                                   borderType=cv2.BORDER_CONSTANT) for m in range(ngroups)], 0)
    total = votes.sum(0)
    top = votes.argmax(0)
    own = np.take_along_axis(votes, np.maximum(mat, 0)[None], 0)[0]
    switch = (mat >= 0) & (L < dark) & (own < 0.25 * total) & (votes.max(0) >= 0.5 * total)
    return np.where(switch, top, mat).astype(mat.dtype)


def _smooth_within(L: np.ndarray, mat: np.ndarray, tol: float, r: int = 1) -> np.ndarray:
    """Render noise out, details kept: a pixel within `tol` of the median lightness of its
    same-material neighbours (a (2r+1)^2 window) takes that median; one that stands out more
    (a hoop, a window frame, a glint) stays."""
    h, w = L.shape
    vals, same = [], []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            vals.append(_shift(L, dy, dx, np.nan))
            same.append(_shift(mat, dy, dx, -2) == mat)
    stack = np.where(np.stack(same), np.stack(vals), np.nan)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(stack, 0)
    ok = (mat >= 0) & np.isfinite(med) & (np.abs(L - med) < tol)
    return np.where(ok, med, L).astype(np.float32)


def hand_pixel(sprites: list[Sprite], outline: str = "selout", *, max_materials: int = 6,
               steps: int = 5, gap: float = 10.0, hue_shift: float = 9.0,
               equalise: float = 0.0, relight: float = 12.0, islands: int = 2,
               smooth: float = 1.0, chroma: float = 1.15, merge: float = 11.0,
               dither: bool = False, clean: bool = True, palette=None) -> list[Sprite]:
    """Re-colour rendered sprites the way they'd be pixelled by hand, the same way for every
    frame (they must share one palette, as turnaround sprites do):

    1. the colours are grouped into materials (material_groups, Lab; `merge` = how far apart
       two materials must be, at most `max_materials`); a pixel whose material is rare around
       it joins its neighbours' (a grey-looking shadow pixel inside a thatch roof)
    2. with a normal map: relight adds clean light from the normals (faces turned to the
       light, top left, lighter; Lab L per unit of n.l). equalise (0..1) first takes out the
       lightness differences the source baked into each face direction of a material; for 3D
       models objects.equalise_faces does that better, on the model, before rendering
    3. smooth: on sprites 64 px and up, lightness noise under smooth x gap is evened out
       within a material, details that stand out more stay
    4. each material gets a ramp of up to `steps` lightness levels about `gap` apart (Lab L),
       hue shifted warm in the lights and cool in the shadows (hue_shift, Lab units; greys
       get a hint of it), chroma x `chroma`; every pixel snaps to its material's nearest
       level: flat, crisp bands
    5. clean-up (clean_sprite: specks, islands of up to `islands` pixels, pixel-perfect
       outline) and a selout outline in each material's own dark shade
    dither: a checker between two levels where a pixel falls half way (off by default).
    palette: a project palette (array or spec, see snap_palette) the colours are snapped to."""
    if not sprites:
        return []
    pal = sprites[0].palette
    lab_pal = rgb_to_lab(pal.astype(np.float32))
    has_outline = outline in ("selout", "solid")
    frames = []
    counts = np.zeros(len(pal), np.int64)
    for s in sprites:
        idx = s.indices.astype(np.int32)
        mask = idx >= 0
        body = mask & ~boundary(mask) if has_outline else mask
        if not body.any():
            body = mask
        counts += np.bincount(idx[body], minlength=len(pal))[:len(pal)]
        frames.append((idx, mask, body))
    group = material_groups(lab_pal, counts, max_materials, merge)
    ngroups = int(group.max()) + 1 if (group >= 0).any() else 0

    data = []                          # [lightness, material, normals] per frame
    for s, (idx, mask, body) in zip(sprites, frames):
        L = lab_pal[np.maximum(idx, 0), 0].astype(np.float32)
        mat = np.where(body, group[np.maximum(idx, 0)], -1)
        n = None
        if s.normal is not None and (equalise or relight):
            from .normals import decode
            n = decode(s.normal[..., :3])
            n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)
        data.append([L, mat, n])

    if ngroups > 1:
        for d in data:                 # big sprites only: in a small one a dark pixel in the
            if min(d[1].shape) >= 96:  # wood is a hoop or a gap, not a mistaken shadow
                d[1] = _majority_material(d[1], d[0], ngroups, 2 if min(d[1].shape) >= 160 else 1)
    lit = [d for d in data if d[2] is not None]
    if lit and equalise:
        bins = [_face_bins(n) for _, _, n in lit]
        for m in range(ngroups):
            total = sum(int((mat == m).sum()) for _, mat, _ in lit)
            if total < 20:
                continue
            mean_m = float(sum(float(L[mat == m].sum()) for L, mat, _ in lit) / total)
            for b in range(7):
                sel = [(mat == m) & (bb == b) for (_, mat, _), bb in zip(lit, bins)]
                cnt = sum(int(q.sum()) for q in sel)
                if cnt < max(12, 0.01 * total):
                    continue
                off = (sum(float(d[0][q].sum()) for d, q in zip(lit, sel)) / cnt - mean_m)
                for d, q in zip(lit, sel):
                    d[0][q] -= off * equalise
    if lit and relight:
        lams = [np.clip(n @ LIGHT_DIR, 0, 1) for _, _, n in lit]
        for m in range(ngroups):
            ls = np.concatenate([lam[mat == m] for (_, mat, _), lam in zip(lit, lams)])
            if not len(ls):
                continue
            mid = float(np.median(ls))
            for d, lam in zip(lit, lams):
                q = d[1] == m
                d[0][q] += relight * (lam[q] - mid)

    if smooth:
        for d in data:                 # small sprites: every pixel is a detail, leave them
            if min(d[0].shape) >= 64:
                d[0] = _smooth_within(d[0], d[1], gap * smooth, 2 if min(d[0].shape) >= 160 else 1)

    new_lab, ramp_of, outline_of = [], [], []
    for m in range(ngroups):
        vals = np.clip(np.concatenate([L[mat == m] for L, mat, _ in data]), 3, 97)
        cols = np.nonzero(group == m)[0]
        w = counts[cols].astype(np.float64)
        ab = ((lab_pal[cols, 1:] * w[:, None]).sum(0) / max(w.sum(), 1)).astype(np.float32)
        lev = _levels(vals, steps, gap)
        mid = float(np.median(vals)) if len(vals) else 50.0
        ramp = _ramp(lev, ab * chroma, mid, hue_shift)
        ramp_of.append((len(new_lab), lev))
        new_lab.extend(ramp)
        cool = ab * 1.05 + hue_shift * 1.2 * _COOL
        outline_of.append(len(new_lab))
        new_lab.append((max(4.0, float(ramp[0][0]) * 0.55), float(cool[0]), float(cool[1])))
    new_lab.append((8.0, 0.0, 0.0))                       # outline with no material beside it
    fallback = len(new_lab) - 1
    new_pal = lab_to_rgb(np.array(new_lab, np.float32).reshape(-1, 1, 3)).reshape(-1, 3)
    if outline == "solid":
        ring_cols = [s.palette[f[0][f[1] & ~f[2]]] for s, f in zip(sprites, frames)]
        ring_cols = [c[0] for c in ring_cols if len(c)]
        if ring_cols:
            new_pal = np.vstack([new_pal, ring_cols[0][None]])
            fallback = len(new_pal) - 1
    owner = np.full(len(new_pal), -1, np.int32)
    for m, (start, lev) in enumerate(ramp_of):
        owner[start:start + len(lev)] = m

    out = []
    for s, (idx, mask, body), (L, mat, _) in zip(sprites, frames, data):
        new = np.full(idx.shape, -1, np.int32)
        h, w = idx.shape
        yy, xx = np.mgrid[0:h, 0:w]
        for m in range(ngroups):
            q = mat == m
            if not q.any():
                continue
            start, lev = ramp_of[m]
            d = np.abs(L[q][:, None] - lev[None])
            k = d.argmin(1)
            if dither and len(lev) > 1:
                order = np.argsort(d, 1)
                srt = np.take_along_axis(d, order, 1)
                half = (srt[:, 1] - srt[:, 0]) < 0.2 * (srt[:, 0] + srt[:, 1])
                k = np.where(half & ((yy[q] + xx[q]) % 2 == 1), order[:, 1], k)
            new[q] = start + k
        new[mask & ~body] = fallback
        sp = Sprite(_rgba(new, new_pal), new.astype(np.int16), new_pal, s.normal)
        if clean:
            sp = clean_sprite(sp, outline, islands=0 if dither else islands, island_delta=gap * 1.7,
                              reshade=False)
        if outline == "selout":
            sp = _material_outline(sp, owner, outline_of, fallback)
        out.append(sp)
    if palette is not None:            # a project palette: the ramps land on its colours
        snapped, lut = snap_palette(out[0].palette, palette)
        out = [clean_sprite(o, outline, speck=0, islands=0, corners=False, lut=lut,
                            palette=snapped, reshade=False) for o in out]
    return out


# --------------------------------------------------------------------------- art-directed clean-up
# Kevin (2026-10-17): "why do the skins look so pixely... like lots of artifacts". A drawing
# downsampled to 64-128 px and bent by a skeleton keeps the marks of both: near-identical
# colours side by side (two greens a step apart that read as noise), single pixels of another
# shade sprinkled through a flat area, 1-3 px islands, crumbs off the silhouette. art_clean()
# fixes those on the KEY drawings (every bent frame copies its pixels from them, so cleaning
# the keys cleans every frame), the same way for all directions (one shared palette):
#   1. ramps: palette colours closer than `merge_de` (Lab) merge (their pixels take the more
#      used one): fewer, clearer steps per material
#   2. cluster smoothing: a body pixel whose 8 neighbours mostly (>= `majority`) share ONE
#      other colour of the same material (within `smooth_de`) takes it - salt-and-pepper inside
#      a shade goes, while edges between shades (neighbours split) and real details (a far
#      colour: an eye, a gem, a strap) stay
#   3. islands up to `islands` px merge into the colour around them (within island_delta)
#   4. crumbs: pieces up to `speck` px (or speck_share of the body) apart from the body go
#   5. the 1 px outline is shaded again from the body colours it touches (selout), so it is a
#      consistent dark edge
ART = {"merge_de": 7.0, "smooth_de": 26.0, "majority": 4, "islands": 3, "island_delta": 26.0,
       "speck": 6, "speck_share": 0.015, "passes": 3}


def merge_palette(sprites: list[Sprite], merge_de: float = 5.0) -> list[Sprite]:
    """Colours of the shared palette closer than merge_de (Lab) become one (the most used)."""
    if not sprites:
        return sprites
    pal = sprites[0].palette
    lab = rgb_to_lab(pal.astype(np.float32))
    counts = np.zeros(len(pal), np.int64)
    for sp in sprites:
        v = sp.indices[sp.indices >= 0]
        counts += np.bincount(v, minlength=len(pal))[:len(pal)]
    order = np.argsort(-counts, kind="stable")
    lut = np.arange(len(pal))
    kept = []
    for i in order:
        if counts[i] == 0:
            continue
        hit = next((k for k in kept if np.linalg.norm(lab[k] - lab[i]) < merge_de), None)
        if hit is None:
            kept.append(i)
        else:
            lut[i] = hit
    out = []
    for sp in sprites:
        idx = np.where(sp.indices >= 0, lut[np.maximum(sp.indices, 0)], -1).astype(sp.indices.dtype)
        rgba = sp.rgba.copy()
        on = idx >= 0
        rgba[on, :3] = pal[idx[on]]
        out.append(Sprite(rgba, idx, pal, sp.normal))
    return out


def _smooth_clusters(idx: np.ndarray, body: np.ndarray, lab: np.ndarray, smooth_de: float,
                     majority: int) -> int:
    """Step 2 of art_clean, in place; returns how many pixels changed."""
    h, w = idx.shape
    nb = np.stack([_shift(idx, dy, dx, -1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                   if (dy, dx) != (0, 0)], -1)                         # (h, w, 8)
    nbody = np.stack([_shift(body, dy, dx, False) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                      if (dy, dx) != (0, 0)], -1)
    nb = np.where(nbody, nb, -1)
    ys, xs = np.nonzero(body)
    if not len(ys):
        return 0
    v = nb[ys, xs].astype(np.int64)                                    # (n, 8)
    ok = v >= 0
    same = (v[:, :, None] == v[:, None, :]) & ok[:, None, :]
    count = same.sum(-1)
    count[~ok] = 0
    best = v[np.arange(len(v)), count.argmax(1)]
    bestn = count.max(1)
    cur = idx[ys, xs].astype(np.int64)
    cand = (bestn >= majority) & (best != cur) & (best >= 0)
    # the pixel's own colour must be rare around it (an isolated shade, not an edge)
    own = (v == cur[:, None]).sum(1)
    cand &= own <= 1
    if not cand.any():
        return 0
    d = np.linalg.norm(lab[np.maximum(best, 0)] - lab[cur], axis=1)
    cand &= d < smooth_de
    idx[ys[cand], xs[cand]] = best[cand]
    return int(cand.sum())


def art_clean(sprites: list[Sprite], outline: str = "selout", **options) -> list[Sprite]:
    """The art-directed clean-up of drawn keys (see above); one shared palette in and out."""
    o = {**ART, **options}
    if not sprites:
        return sprites
    sprites = merge_palette(sprites, o["merge_de"]) if o["merge_de"] > 0 else list(sprites)
    pal = sprites[0].palette
    lab = rgb_to_lab(pal.astype(np.float32))
    has_outline = outline in ("selout", "solid")
    out = []
    for sp in sprites:
        idx = sp.indices.astype(np.int32).copy()
        for _ in range(int(o["passes"])):
            mask = idx >= 0
            body = mask & ~boundary(mask) if has_outline else mask
            n = _smooth_clusters(idx, body, lab, o["smooth_de"], int(o["majority"]))
            if not n:
                break
        tmp = Sprite(sp.rgba, idx.astype(sp.indices.dtype), pal, sp.normal)
        speck = max(int(o["speck"]), int(o["speck_share"] * (idx >= 0).sum()))   # crumbs scale with size
        tmp = clean_sprite(tmp, outline, speck=speck, islands=int(o["islands"]),
                           island_delta=float(o["island_delta"]))
        out.append(tmp)
    return out


def tone_match(sprites: list[Sprite], ref: int, *, strength: float = 0.6, most: float = 14.0,
               dark: float = 6.0) -> list[Sprite]:
    """Views much darker than the reference drawing (a muddy back view: armour + cloak drawn as
    one dark mass) brought part way toward the reference's lightness: each view's body lightness
    quantiles are matched to the reference's (`strength` of the way, at most `most` Lab L), and
    every colour it uses moves to the nearest colour of the SHARED palette at its new lightness
    (same hue): no new colours, the shading order kept. Views within `dark` L of the reference
    are left alone."""
    if not sprites:
        return sprites
    pal = sprites[0].palette
    lab = rgb_to_lab(pal.astype(np.float32))
    r = sprites[ref]
    rl = lab[r.indices[r.indices >= 0], 0]
    if not len(rl):
        return sprites
    qs = np.linspace(5, 95, 19)
    rq = np.percentile(rl, qs)
    out = []
    for i, sp in enumerate(sprites):
        on = sp.indices >= 0
        if i == ref or not on.any():
            out.append(sp)
            continue
        vl = lab[sp.indices[on], 0]
        if np.median(vl) > np.median(rl) - dark:
            out.append(sp)
            continue
        vq = np.percentile(vl, qs)
        used = np.unique(sp.indices[on])
        lut = np.arange(len(pal))
        for u in used:
            L = lab[u, 0]
            shift = np.clip((np.interp(L, vq, rq) - L) * strength, 0.0, most)
            if shift < 1.0:
                continue
            target = np.array([L + shift, lab[u, 1], lab[u, 2]], np.float32)
            d = ((lab - target) ** 2 * np.array([1.0, 2.0, 2.0], np.float32)).sum(1)
            lut[u] = int(np.argmin(d))
        idx = np.where(on, lut[np.maximum(sp.indices, 0)], -1).astype(sp.indices.dtype)
        rgba = sp.rgba.copy()
        rgba[on, :3] = pal[idx[on]]
        out.append(Sprite(rgba, idx, pal, sp.normal))
    return out

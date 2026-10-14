"""UV unwrapping, atlas packing and texture baking for game meshes (numpy only).

    unwrap      charts grown over neighbouring faces within a normal cone, each projected on its
                own mean-normal plane; a chart whose projection folds over itself is split by
                depth; charts turned to their smallest bounding box and shelf-packed into a
                square atlas at ONE texel density (so pixel textures have even pixels).
    rasterize   every texel/pixel centre inside every triangle, vectorised (bounding boxes ->
                barycentrics); shared by the baker and the preview renderer.
    bake        colour from the heavy painted model (closest point on the same side), normals
                from the cleaned high-poly into the low-poly's tangent space (OpenGL, +Y up, as
                glTF wants), gutters padded so mipmaps don't bleed.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np


from .gamemesh import Mesh, Surface, face_areas, face_normals, vertex_normals


# --------------------------------------------------------------------------- rasterizing

def rasterize(tri2d: np.ndarray, width: int, height: int, chunk: int = 4_000_000):
    """Pixel centres (x + 0.5, y + 0.5) inside 2D triangles (F, 3, 2) in pixel units.
    Returns (pixel index y * width + x, face index, barycentrics (n, 3)). Edges count as inside
    for both neighbours (a texel on a seam belongs to either)."""
    tri2d = np.asarray(tri2d, np.float64)
    lo = np.floor(tri2d.min(1) - 0.5).astype(np.int64) + 1
    hi = np.floor(tri2d.max(1) - 0.5).astype(np.int64)
    lo = np.maximum(lo, 0)
    hi[:, 0] = np.minimum(hi[:, 0], width - 1)
    hi[:, 1] = np.minimum(hi[:, 1], height - 1)
    w = np.maximum(hi[:, 0] - lo[:, 0] + 1, 0)
    h = np.maximum(hi[:, 1] - lo[:, 1] + 1, 0)
    count = w * h
    faces = np.nonzero(count > 0)[0]
    out_p, out_f, out_b = [], [], []
    start = 0
    csum = np.cumsum(count[faces])
    while start < len(faces):
        base = csum[start - 1] if start else 0
        end = int(np.searchsorted(csum, base + chunk, side="right"))
        end = max(end, start + 1)
        fs = faces[start:end]
        n = count[fs]
        fi = np.repeat(fs, n)
        k = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
        px = lo[fi, 0] + k % w[fi]
        py = lo[fi, 1] + k // w[fi]
        t = tri2d[fi]
        x, y = px + 0.5, py + 0.5
        x0, y0 = t[:, 0, 0], t[:, 0, 1]
        x1, y1 = t[:, 1, 0], t[:, 1, 1]
        x2, y2 = t[:, 2, 0], t[:, 2, 1]
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        ok = np.abs(den) > 1e-14
        den = np.where(ok, den, 1.0)
        b0 = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / den
        b1 = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / den
        b2 = 1 - b0 - b1
        eps = -1e-9
        inside = ok & (b0 >= eps) & (b1 >= eps) & (b2 >= eps)
        out_p.append((py * width + px)[inside])
        out_f.append(fi[inside])
        out_b.append(np.stack([b0, b1, b2], 1)[inside])
        start = end
    if not out_p:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 3))
    return np.concatenate(out_p), np.concatenate(out_f), np.concatenate(out_b)


# --------------------------------------------------------------------------- charts

def face_adjacency_lists(f: np.ndarray) -> list:
    """Faces sharing an edge, per face (manifold edges only)."""
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    owner = np.tile(np.arange(len(f)), 3)
    key = np.sort(e, 1)
    k = key[:, 0] * (int(f.max()) + 1) + key[:, 1]
    order = np.argsort(k, kind="stable")
    ks, os_ = k[order], owner[order]
    same = np.nonzero(ks[1:] == ks[:-1])[0]
    nbr = [[] for _ in range(len(f))]
    for i in same:
        a, b = int(os_[i]), int(os_[i + 1])
        if a != b:
            nbr[a].append(b)
            nbr[b].append(a)
    return nbr


def grow_charts(m: Mesh, cone: float = 50.0) -> np.ndarray:
    """Chart id per face: flood over neighbours whose normal stays within `cone` degrees of
    the chart's running mean normal. Biggest flat areas first."""
    fn = face_normals(m.v, m.f)
    area = face_areas(m.v, m.f)
    nbr = face_adjacency_lists(m.f)
    chart = np.full(len(m.f), -1, np.int64)
    cos_c = math.cos(math.radians(cone))
    # seeds: faces whose neighbours agree most (flat places) first
    flat = np.array([np.mean([fn[i] @ fn[j] for j in nb]) if nb else 0 for i, nb in enumerate(nbr)])
    order = np.argsort(-(flat + 1e-3 * area / max(area.max(), 1e-20)))
    cid = 0
    for seed in order:
        if chart[seed] >= 0:
            continue
        acc = fn[seed] * area[seed]
        mean = fn[seed]
        chart[seed] = cid
        q = deque([seed])
        while q:
            i = q.popleft()
            for j in nbr[i]:
                if chart[j] < 0 and fn[j] @ mean >= cos_c:
                    chart[j] = cid
                    acc = acc + fn[j] * area[j]
                    mean = acc / max(np.linalg.norm(acc), 1e-20)
                    q.append(j)
        cid += 1
    return chart


def merge_charts(m: Mesh, chart: np.ndarray, fn, area, nbr, cone: float,
                 max_faces: int | None = None) -> np.ndarray:
    """Greedily join neighbouring charts while every face stays within `cone` degrees of the
    joint mean normal (longest shared borders first). max_faces: only charts that small may
    join a neighbour."""
    cos_c = math.cos(math.radians(cone))
    chart = chart.copy()
    for _ in range(6):
        ids, chart = np.unique(chart, return_inverse=True)
        k = len(ids)
        members = [[] for _ in range(k)]
        for i, c in enumerate(chart):
            members[c].append(i)
        members = [np.array(x) for x in members]
        acc = np.zeros((k, 3))
        np.add.at(acc, chart, fn * area[:, None])
        border = {}
        for i, nb in enumerate(nbr):
            ci = chart[i]
            for j in nb:
                cj = chart[j]
                if ci < cj:
                    border[(ci, cj)] = border.get((ci, cj), 0) + 1
        alive = np.ones(k, bool)
        parent = np.arange(k)
        merged = 0
        for (a, b), _n in sorted(border.items(), key=lambda kv: -kv[1]):
            while parent[a] != a:
                a = parent[a]
            while parent[b] != b:
                b = parent[b]
            if a == b:
                continue
            if max_faces is not None and min(len(members[a]), len(members[b])) > max_faces:
                continue
            tot = acc[a] + acc[b]
            nrm = np.linalg.norm(tot)
            if nrm < 1e-20:
                continue
            mean = tot / nrm
            faces = np.concatenate([members[a], members[b]])
            if (fn[faces] @ mean).min() < cos_c:
                continue
            if len(members[b]) > len(members[a]):
                a, b = b, a
            parent[b] = a
            members[a] = faces
            members[b] = np.zeros(0, np.int64)
            acc[a] = tot
            merged += 1
        root = np.arange(k)
        for c in range(k):
            r = c
            while parent[r] != r:
                r = parent[r]
            root[c] = r
        chart = root[chart]
        if merged == 0:
            break
    return np.unique(chart, return_inverse=True)[1]


def _frame(n: np.ndarray):
    """Two unit axes perpendicular to n."""
    a = np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(a, n)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def _overlaps(uv: np.ndarray, faces: np.ndarray, res: int = 96) -> float:
    """Share of a chart's projected area covered twice (fold-over)."""
    lo, hi = uv.min(0), uv.max(0)
    size = float((hi - lo).max())
    if size <= 0:
        return 0.0
    t = (uv[faces] - lo) / size * (res - 2) + 1
    px, fi, _ = rasterize(t, res, res)
    if len(px) == 0:
        return 0.0
    cnt = np.bincount(px, minlength=res * res)
    # pixels exactly on shared edges count twice: ignore pixels hit by 2 adjacent faces only
    return float((cnt > 2).sum() + 0.5 * (cnt == 2).sum()) / max(1, (cnt > 0).sum()) \
        if (cnt > 1).sum() > 0.02 * (cnt > 0).sum() else 0.0


def _flipped(uv, faces) -> np.ndarray:
    t = uv[faces]
    return ((t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1])
            - (t[:, 2, 0] - t[:, 0, 0]) * (t[:, 1, 1] - t[:, 0, 1])) <= 0


def _components(faces_idx: np.ndarray, nbr: list) -> list:
    """Edge-connected groups among the given faces."""
    inset = set(int(i) for i in faces_idx)
    seen, groups = set(), []
    for s in faces_idx:
        s = int(s)
        if s in seen:
            continue
        g, q = [], deque([s])
        seen.add(s)
        while q:
            i = q.popleft()
            g.append(i)
            for j in nbr[i]:
                if j in inset and j not in seen:
                    seen.add(j)
                    q.append(j)
        groups.append(np.array(g, np.int64))
    return groups


@dataclass
class Chart:
    faces: np.ndarray            # face indices of the mesh
    verts: np.ndarray            # mesh vertex indices used (local order)
    uv: np.ndarray               # (len(verts), 2) in world units, turned to the min bbox
    size: np.ndarray             # (w, h) world units
    offset: np.ndarray = field(default_factory=lambda: np.zeros(2))   # placed (texels)


def _project(m: Mesh, faces: np.ndarray, fn, area):
    n = (fn[faces] * area[faces, None]).sum(0)
    n /= max(np.linalg.norm(n), 1e-20)
    u, w = _frame(n)
    verts, local = np.unique(m.f[faces], return_inverse=True)
    p = m.v[verts]
    uv = np.stack([p @ u, p @ w], 1)
    return verts, local.reshape(-1, 3), uv, n


def _min_box(uv: np.ndarray) -> np.ndarray:
    """uv turned so its bounding box is smallest (tries the hull edge directions)."""
    if len(uv) < 3:
        return uv - uv.min(0)
    try:
        from scipy.spatial import ConvexHull
        hull = uv[ConvexHull(uv).vertices]
    except Exception:
        hull = uv
    edges = np.roll(hull, -1, 0) - hull
    ang = np.unique(np.round(np.arctan2(edges[:, 1], edges[:, 0]) % (np.pi / 2), 6))
    best, best_a = None, 0.0
    for a in ang:
        c, s = math.cos(-a), math.sin(-a)
        r = hull @ np.array([[c, s], [-s, c]])
        ext = r.max(0) - r.min(0)
        area = ext[0] * ext[1]
        if best is None or area < best:
            best, best_a = area, a
    c, s = math.cos(-best_a), math.sin(-best_a)
    r = uv @ np.array([[c, s], [-s, c]])
    ext = r.max(0) - r.min(0)
    if ext[1] > ext[0]:                       # lie down: wide charts pack better on shelves
        r = r[:, ::-1] * np.array([1.0, -1.0])
    return r - r.min(0)


def lscm(v: np.ndarray, faces: np.ndarray) -> np.ndarray | None:
    """Least-squares conformal map of a disc-like chart (Levy et al. 2002): local vertex
    positions v (n, 3), faces (m, 3) local indices -> uv (n, 2) at the chart's 3D scale, or
    None if the solve fails."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import lsqr, spsolve
    n = len(v)
    p0, p1, p2 = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    e1 = p1 - p0
    e2 = p2 - p0
    nrm = np.cross(e1, e2)
    dbl = np.linalg.norm(nrm, axis=1)
    ok = dbl > 1e-14
    if ok.sum() < 1:
        return None
    x_ax = e1 / np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-20)
    y_ax = np.cross(nrm / np.maximum(dbl[:, None], 1e-20), x_ax)
    loc = np.zeros((len(faces), 3, 2))
    loc[:, 1, 0] = np.linalg.norm(e1, axis=1)
    loc[:, 2, 0] = (e2 * x_ax).sum(1)
    loc[:, 2, 1] = (e2 * y_ax).sum(1)
    W = np.stack([loc[:, 2] - loc[:, 1], loc[:, 0] - loc[:, 2], loc[:, 1] - loc[:, 0]], 1)
    s_ = 1.0 / np.sqrt(np.maximum(dbl, 1e-20))
    rows, cols, vals = [], [], []
    m = len(faces)
    for j in range(3):
        vj = faces[:, j]
        wr, wi = W[:, j, 0] * s_, W[:, j, 1] * s_
        # real row: wr*u - wi*v ; imag row: wi*u + wr*v   (u at col vj, v at col n + vj)
        rows += [np.arange(m), np.arange(m), m + np.arange(m), m + np.arange(m)]
        cols += [vj, n + vj, vj, n + vj]
        vals += [wr, -wi, wi, wr]
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(2 * m, 2 * n)).tocsc()
    # pin the two vertices farthest apart
    i0 = int(np.argmax(np.linalg.norm(v - v.mean(0), axis=1)))
    i1 = int(np.argmax(np.linalg.norm(v - v[i0], axis=1)))
    if i0 == i1:
        return None
    d = float(np.linalg.norm(v[i1] - v[i0]))
    pinned = np.array([i0, n + i0, i1, n + i1])
    pv = np.array([0.0, 0.0, d, 0.0])
    free = np.setdiff1d(np.arange(2 * n), pinned)
    b = -(A[:, pinned] @ pv)
    Af = A[:, free]
    try:
        sol = spsolve((Af.T @ Af).tocsc(), Af.T @ b)
    except Exception:
        sol = lsqr(Af, b, atol=1e-10, btol=1e-10, iter_lim=20 * n + 200)[0]
    x = np.zeros(2 * n)
    x[free] = sol
    x[pinned] = pv
    uv = np.stack([x[:n], x[n:]], 1)
    if not np.isfinite(uv).all():
        return None
    t = uv[faces]
    a2 = 0.5 * np.abs((t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1])
                      - (t[:, 2, 0] - t[:, 0, 0]) * (t[:, 1, 1] - t[:, 0, 1])).sum()
    a3 = 0.5 * dbl.sum()
    if a2 <= 1e-20:
        return None
    return uv * math.sqrt(a3 / a2)


def _stretch(uv, local, v_local) -> float:
    """Worst-to-median ratio of uv area to 3D area over a chart's faces (1 = even)."""
    t = uv[local]
    a2 = 0.5 * np.abs((t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1])
                      - (t[:, 2, 0] - t[:, 0, 0]) * (t[:, 1, 1] - t[:, 0, 1]))
    p = v_local[local]
    a3 = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    r = a2 / np.maximum(a3, 1e-20)
    med = np.median(r)
    big = a3 > 0.2 * np.median(a3)
    if not big.any() or med <= 0:
        return 1.0
    return float(max(med / max(r[big].min(), 1e-12), r[big].max() / med))


def _split(m: Mesh, faces: np.ndarray, fn, n, flip) -> list:
    if flip is not None and flip.any() and not flip.all():
        return [faces[~flip], faces[flip]]
    c = m.v[m.f[faces]].mean(1)                 # cut across the chart's longest extent:
    c = c - c.mean(0)                           # two discs, and rings get opened
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    key = c @ vt[0]
    med = np.median(key)
    parts = [faces[key <= med], faces[key > med]]
    if min(len(p) for p in parts) == 0:
        half = len(faces) // 2
        parts = [faces[:half], faces[half:]]
    return parts


def make_charts(m: Mesh, cone: float = 72.0, min_faces: int = 3, conformal: bool = True,
                max_stretch: float = 4.0) -> list[Chart]:
    """Charts: grown within `cone` degrees, flattened by LSCM (conformal=True) or projected on
    their mean plane; folded / overlapping / too stretched charts are split in two."""
    fn = face_normals(m.v, m.f)
    area = face_areas(m.v, m.f)
    nbr = face_adjacency_lists(m.f)
    chart = merge_charts(m, grow_charts(m, cone), fn, area, nbr, cone)
    # leftover crumbs join the neighbour that faces most alike (checked for folds below)
    chart = merge_charts(m, chart, fn, area, nbr, min(cone + 25, 89.0), max_faces=min_faces)
    out: list[Chart] = []
    todo = [np.nonzero(chart == c)[0] for c in np.unique(chart)]
    while todo:
        faces = todo.pop()
        if len(faces) == 0:
            continue
        ch, flip, n = _flatten(m, faces, fn, area, conformal, max_stretch)
        if ch is None and len(faces) > 1:
            for p in _split(m, faces, fn, n, flip):
                todo.extend(_components(p, nbr))
            continue
        if ch is None:                           # a lone face always flattens
            verts, local, uv, n = _project(m, faces, fn, area)
            uv = _min_box(uv)
            ch = Chart(faces, verts, uv, uv.max(0))
        out.append(ch)
    return _absorb_crumbs(m, out, fn, area, nbr, conformal, max_stretch, min_faces)


def _flatten(m, faces, fn, area, conformal, max_stretch):
    """(Chart or None if it folds / overlaps / stretches, flipped faces, mean normal)."""
    verts, local, uv, n = _project(m, faces, fn, area)
    flip = _flipped(uv, local)
    if flip.all():
        uv = uv * np.array([-1.0, 1.0])
        flip = ~flip
    if conformal and len(faces) > 2:
        u2 = lscm(m.v[verts], local)
        if u2 is not None:
            f2 = _flipped(u2, local)
            if f2.all():
                u2, f2 = u2 * np.array([-1.0, 1.0]), ~f2
            if not f2.any():
                uv, flip = u2, f2
    bad = flip.any() or (len(faces) > 4 and _overlaps(uv, local) > 0)
    if not bad and len(faces) > 8 and _stretch(uv, local, m.v[verts]) > max_stretch:
        bad = True
    if bad:
        return None, flip, n
    uv = _min_box(uv)
    return Chart(faces, verts, uv, uv.max(0)), flip, n


def _absorb_crumbs(m, charts, fn, area, nbr, conformal, max_stretch, small):
    """Charts of <= `small` faces join the neighbour they share most edges with, when the joined
    chart still flattens cleanly (fewer charts = less padding = more texels)."""
    owner = np.empty(len(m.f), np.int64)
    for i, c in enumerate(charts):
        owner[c.faces] = i
    want: dict[int, list[int]] = {}
    for i, c in enumerate(charts):
        if len(c.faces) > small:
            continue
        votes: dict[int, int] = {}
        for fi in c.faces:
            for j in nbr[fi]:
                o = int(owner[j])
                if o != i and len(charts[o].faces) > small:
                    votes[o] = votes.get(o, 0) + 1
        if votes:
            want.setdefault(max(votes, key=votes.get), []).append(i)
    gone = set()
    for target, crumbs in want.items():
        groups = [crumbs] if len(crumbs) > 1 else []
        groups += [[c] for c in crumbs[:12]]
        base = charts[target].faces
        for g in groups:
            g = [c for c in g if c not in gone]
            if not g:
                continue
            faces = np.concatenate([base] + [charts[c].faces for c in g])
            ch, _, _ = _flatten(m, faces, fn, area, conformal, max_stretch)
            if ch is not None:
                charts[target] = ch
                base = faces
                gone.update(g)
    return [c for i, c in enumerate(charts) if i not in gone]


def _shelf(sizes: np.ndarray, width: float) -> tuple[np.ndarray, float]:
    """Shelf packing of (w, h) boxes sorted by height: (offsets, used height)."""
    order = np.argsort(-sizes[:, 1], kind="stable")
    offs = np.zeros_like(sizes)
    x = y = shelf_h = 0.0
    for i in order:
        w, h = sizes[i]
        if x + w > width and x > 0:
            y += shelf_h
            x, shelf_h = 0.0, 0.0
        offs[i] = (x, y)
        x += w
        shelf_h = max(shelf_h, h)
    return offs, y + shelf_h


@dataclass
class Atlas:
    """A mesh with split UV seams: positions, faces, uv (0..1, v down), and for every new
    vertex the index of the source vertex."""
    v: np.ndarray
    f: np.ndarray
    uv: np.ndarray
    source: np.ndarray
    size: int
    density: float               # texels per world unit
    charts: int


def _chart_mask(c: Chart, scale: float, cell: float, pad_cells: int, local) -> np.ndarray:
    """Cells a chart covers at `scale` texels per unit, grown by the padding."""
    import cv2
    pts = c.uv * scale / cell + pad_cells
    w = int(math.ceil(pts[:, 0].max() + pad_cells)) + 1
    h = int(math.ceil(pts[:, 1].max() + pad_cells)) + 1
    mask = np.zeros((h, w), np.uint8)
    tri = np.round(pts[local] * 4).astype(np.int32)
    cv2.fillPoly(mask, list(tri), 1, lineType=cv2.LINE_8, shift=2)
    for t in tri:                                  # thin slivers still claim their cells
        cv2.polylines(mask, [t], True, 1, 1, cv2.LINE_8, shift=2)
    return mask


def _turn(c: Chart) -> np.ndarray:
    """The chart's uv turned 90 degrees (a turn, not a mirror: faces keep their winding)."""
    return np.stack([c.uv[:, 1], c.size[0] - c.uv[:, 0]], 1)


def _place(occ: np.ndarray, mask: np.ndarray):
    """Lowest-then-leftmost free spot for mask in occupancy (or None). Looks in a band below
    the first row with room first, then everywhere."""
    G = occ.shape[0]
    h, w = mask.shape
    if h > G or w > G:
        return None
    rows = np.nonzero(~occ.all(1))[0]
    if len(rows) == 0:
        return None
    top = max(0, int(rows[0]) - 1)
    for bottom in (min(G, top + h + 96), G):
        spot = _search(occ[top:bottom], mask)
        if spot is not None:
            return spot[0] + top, spot[1]
        if bottom == G:
            break
    return None


def _search(sub_occ: np.ndarray, mask: np.ndarray):
    from scipy.signal import fftconvolve
    h, w = mask.shape
    if sub_occ.shape[0] < h or sub_occ.shape[1] < w:
        return None
    sub = sub_occ.astype(np.float32)
    if h * w <= 400:                       # small: its whole box must be free (summed areas)
        sat = np.zeros((sub.shape[0] + 1, sub.shape[1] + 1), np.float32)
        sat[1:, 1:] = sub.cumsum(0).cumsum(1)
        hit = sat[h:, w:] - sat[:-h, w:] - sat[h:, :-w] + sat[:-h, :-w]
    else:
        hit = fftconvolve(sub, mask[::-1, ::-1].astype(np.float32), mode="valid")
    free = np.flatnonzero(hit.ravel() < 0.5)
    if len(free) == 0:
        return None
    y, x = divmod(int(free[0]), hit.shape[1])
    return y, x


def pack(m: Mesh, charts: list[Chart], size: int = 1024, padding: int = 2,
         fill: float = 0.72) -> Atlas:
    """Place charts at one texel density so they fill a size x size atlas: each chart's real
    footprint (not its box) goes into the lowest free spot of a cell grid, biggest first,
    turned 90 degrees when that sits lower."""
    import cv2
    cell = max(1.0, size / 512.0)
    G = int(size / cell)
    pad_cells = max(1, int(math.ceil(padding / cell)))
    kernel = np.ones((2 * pad_cells + 1,) * 2, np.uint8)
    locals_ = []
    for c in charts:
        remap = np.full(int(c.verts.max()) + 1, -1, np.int64)
        remap[c.verts] = np.arange(len(c.verts))
        locals_.append(remap[m.f[c.faces]])
    areas = []
    for c, lf in zip(charts, locals_):
        t = c.uv[lf]
        areas.append(0.5 * np.abs((t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1])
                                  - (t[:, 2, 0] - t[:, 0, 0]) * (t[:, 1, 1] - t[:, 0, 1])).sum())
    total = max(float(np.sum(areas)), 1e-20)
    scale = math.sqrt(fill * size * size / total)
    order = np.argsort([-(c.size[0] * c.size[1]) for c in charts], kind="stable")
    for _attempt in range(40):
        occ = np.zeros((G, G), bool)
        offs = [None] * len(charts)
        turned = [False] * len(charts)
        ok = True
        for i in order:
            c = charts[i]
            best = None
            for turn in (False, True):
                cc = Chart(c.faces, c.verts, _turn(c) if turn else c.uv,
                           c.size[::-1] if turn else c.size)
                mk = _chart_mask(cc, scale, cell, pad_cells, locals_[i])
                probe = cv2.dilate(mk, kernel) if pad_cells else mk     # keep a gap of pad
                spot = _place(occ, probe)
                if spot is not None and (best is None or spot < best[0]):
                    best = (spot, turn, mk)
                if c.size[0] == c.size[1]:
                    break
            if best is None:
                ok = False
                break
            (y, x), turn, mk = best
            occ[y:y + mk.shape[0], x:x + mk.shape[1]] |= mk.astype(bool)
            offs[i] = np.array([x, y], np.float64) * cell
            turned[i] = turn
        if ok:
            break
        scale *= 0.94

    vs, fs, uvs, src = [], [], [], []
    base = 0
    for c, o, t, lf in zip(charts, offs, turned, locals_):
        cuv = _turn(c) if t else c.uv
        uv = (cuv * scale + o + pad_cells * cell) / size
        vs.append(m.v[c.verts])
        uvs.append(uv)
        src.append(c.verts)
        fs.append(lf + base)
        base += len(c.verts)
    v = np.concatenate(vs)
    f = np.concatenate(fs).astype(np.int64)
    uv = np.concatenate(uvs)
    # keep the source face order (LOD meshes keep their winding and order)
    order = np.argsort(np.concatenate([c.faces for c in charts]), kind="stable")
    return Atlas(v, f[order], uv, np.concatenate(src), size, scale, len(charts))


def unwrap(m: Mesh, size: int = 1024, padding: int = 2, cone: float = 72.0) -> Atlas:
    return pack(m, make_charts(m, cone), size, padding)


# --------------------------------------------------------------------------- tangents

def tangents(v, f, uv, n) -> np.ndarray:
    """(n, 4) per-vertex tangents (xyz + handedness w) from the uv layout. uv with v DOWN
    (glTF): the bitangent glTF expects is cross(normal, tangent.xyz) * w and points to +v up
    in texture space... computed so that shaders' TBN maps a +Y-up normal map correctly."""
    p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    t0, t1, t2 = uv[f[:, 0]], uv[f[:, 1]], uv[f[:, 2]]
    e1, e2 = p1 - p0, p2 - p0
    d1, d2 = t1 - t0, t2 - t0
    r = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    r = np.where(np.abs(r) < 1e-20, 1e-20, r)
    sdir = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) / r[:, None]      # d position / d u
    tdir = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) / r[:, None]      # d position / d v (down)
    tan = np.zeros_like(v)
    bit = np.zeros_like(v)
    for k in range(3):
        np.add.at(tan, f[:, k], sdir)
        np.add.at(bit, f[:, k], tdir)
    t = tan - n * (n * tan).sum(1, keepdims=True)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-20)
    bad = ~np.isfinite(t).all(1) | (np.linalg.norm(t, axis=1) < 0.5)
    if bad.any():
        alt = np.cross(n[bad], np.array([0.0, 1.0, 0.0]))
        alt[np.linalg.norm(alt, axis=1) < 1e-6] = (1.0, 0.0, 0.0)
        t[bad] = alt / np.linalg.norm(alt, axis=1, keepdims=True)
    # glTF bitangent = cross(N, T) * w must point along texture "up" (= -v when v is down)
    up = -bit
    w = np.where((np.cross(n, t) * up).sum(1) < 0, -1.0, 1.0)
    return np.concatenate([t, w[:, None]], 1)


# --------------------------------------------------------------------------- baking

@dataclass
class Bake:
    colour: np.ndarray           # (S, S, 4) uint8 sRGB
    normal: np.ndarray | None    # (S, S, 3) uint8 tangent space (OpenGL)
    mask: np.ndarray             # (S, S) bool, texels inside triangles
    vertex_normals: np.ndarray   # per atlas vertex
    tangents: np.ndarray         # per atlas vertex (n, 4)


def _same_side(surface: Surface, p, n, k: int = 8, reach: float | None = None):
    """Closest points on `surface` with a face facing like n (falls back to plain closest)."""
    from .gamemesh import closest_on_triangles
    _, idx = surface.tree.query(p, k=k, workers=-1)
    idx = idx.reshape(len(p), -1)
    best = np.full(len(p), np.inf)
    bf = np.zeros(len(p), np.int64)
    bb = np.zeros((len(p), 3))
    for j in range(idx.shape[1]):
        fi = surface.face[idx[:, j]]
        tri = surface.f[fi]
        q, b = closest_on_triangles(p, surface.v[tri[:, 0]], surface.v[tri[:, 1]],
                                    surface.v[tri[:, 2]])
        d = np.linalg.norm(p - q, axis=1)
        facing = (surface.fn[fi] * n).sum(1)
        score = d + np.where(facing < 0.0, 10.0 * (reach or 1.0), 0.0)
        better = score < best
        best[better], bf[better], bb[better] = score[better], fi[better], b[better]
    return bf, bb


def pad_gutters(img: np.ndarray, mask: np.ndarray, rounds: int = 8) -> np.ndarray:
    """Spread edge texels outwards into the empty atlas (mipmaps then don't pull in black)."""
    import cv2
    out = img.astype(np.float32).copy()
    m = mask.astype(np.float32)
    k = np.ones((3, 3), np.float32)
    for _ in range(rounds):
        if m.min() >= 1:
            break
        s = cv2.filter2D(out * m[..., None], -1, k, borderType=cv2.BORDER_CONSTANT)
        c = cv2.filter2D(m, -1, k, borderType=cv2.BORDER_CONSTANT)
        grow = (m < 1) & (c > 0)
        out[grow] = s[grow] / c[grow][:, None]
        m = np.where(grow, 1.0, m)
    if m.min() < 1:                                   # far corners: the mean colour
        out[m < 1] = out[m >= 1].mean(0) if (m >= 1).any() else 0
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def bake(atlas: Atlas, low_normals_src: np.ndarray, colour_src, normal_src: Mesh | None = None,
         supersample: int = 1, gutter: int = 8) -> Bake:
    """Bake into atlas.size^2 texels.

    low_normals_src: smooth vertex normals of the low mesh BEFORE uv splitting (indexed by
    atlas.source), so shading is continuous across seams. colour_src: (Surface, vertex colours
    (n, 3) sRGB 0..255) of the painted heavy model. normal_src: the cleaned high-poly for the
    normal map (None = no normal map)."""
    S = atlas.size * supersample
    n_low = low_normals_src[atlas.source]
    tan = tangents(atlas.v, atlas.f, atlas.uv, n_low)
    px, fi, b = rasterize(atlas.uv[atlas.f] * S, S, S)
    # one sample per texel (the last face wins on seams; both sides are the same surface)
    _, keep = np.unique(px[::-1], return_index=True)
    keep = len(px) - 1 - keep
    px, fi, b = px[keep], fi[keep], b[keep]
    tri = atlas.f[fi]
    P = (atlas.v[tri] * b[..., None]).sum(1)
    N = (n_low[tri] * b[..., None]).sum(1)
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-20)
    surf, vcol = colour_src
    reach = float((atlas.v.max(0) - atlas.v.min(0)).max())
    cf, cb = _same_side(surf, P, N, reach=reach)
    col = (vcol[surf.f[cf]] * cb[..., None]).sum(1)
    colour = np.zeros((S * S, 4), np.float32)
    colour[px, :3] = col
    colour[px, 3] = 255
    mask = np.zeros(S * S, bool)
    mask[px] = True
    normal_img = None
    if normal_src is not None:
        hs = normal_src if isinstance(normal_src, Surface) else Surface(normal_src.v, normal_src.f)
        hn = vertex_normals(hs.v, hs.f)
        nf, nb = _same_side(hs, P, N, reach=reach)
        Hn = (hn[hs.f[nf]] * nb[..., None]).sum(1)
        Hn /= np.maximum(np.linalg.norm(Hn, axis=1, keepdims=True), 1e-20)
        T = (tan[tri, :3] * b[..., None]).sum(1)
        T = T - N * (T * N).sum(1, keepdims=True)
        T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-20)
        w = np.sign((tan[tri, 3] * b).sum(1))
        w[w == 0] = 1
        B = np.cross(N, T) * w[:, None]
        ts = np.stack([(Hn * T).sum(1), (Hn * B).sum(1), (Hn * N).sum(1)], 1)
        ts[:, 2] = np.maximum(ts[:, 2], 0.05)
        ts /= np.linalg.norm(ts, axis=1, keepdims=True)
        nimg = np.zeros((S * S, 3), np.float32)
        nimg[:] = (128, 128, 255)
        nimg[px] = (ts * 0.5 + 0.5) * 255
        normal_img = nimg.reshape(S, S, 3)
    colour = colour.reshape(S, S, 4)
    mask = mask.reshape(S, S)
    if supersample > 1:
        colour, mask, normal_img = _downsample(colour, mask, normal_img, supersample)
    rgb = pad_gutters(colour[..., :3], mask, gutter)
    rgba = np.concatenate([rgb, np.full(mask.shape + (1,), 255, np.uint8)], 2)
    if normal_img is not None:
        nm = normal_img.astype(np.float32) / 127.5 - 1
        nm /= np.maximum(np.linalg.norm(nm, axis=2, keepdims=True), 1e-6)
        normal_img = pad_gutters((nm * 0.5 + 0.5) * 255, mask, gutter)
    return Bake(rgba, normal_img, mask, n_low, tan)


def _downsample(colour, mask, normal, k):
    S = colour.shape[0] // k
    m = mask.reshape(S, k, S, k).astype(np.float32)
    cnt = m.sum((1, 3))
    c = (colour[..., :3].reshape(S, k, S, k, 3) * m[..., None]).sum((1, 3))
    c = c / np.maximum(cnt, 1)[..., None]
    out_c = np.concatenate([c, (cnt > 0)[..., None] * 255.0], 2)
    out_n = None
    if normal is not None:
        n = (normal.reshape(S, k, S, k, 3) * m[..., None]).sum((1, 3)) / np.maximum(cnt, 1)[..., None]
        out_n = np.where((cnt > 0)[..., None], n, np.array([128, 128, 255.0]))
    return out_c, cnt > 0, out_n


# --------------------------------------------------------------------------- pixel look

def quantize_texture(rgba: np.ndarray, mask: np.ndarray, colours: int = 32,
                     palette: np.ndarray | None = None) -> np.ndarray:
    """Snap the texels to a palette (given (k, 3) 0..255, or k-means in Lab over the used
    texels)."""
    import cv2
    from .color import lab_to_rgb, rgb_to_lab
    used = rgba[mask][:, :3]
    if len(used) == 0:
        return rgba
    lab_all = rgb_to_lab(rgba[..., :3].reshape(-1, 3)).astype(np.float64)
    if palette is None:
        lab = lab_all.reshape(rgba.shape[:2] + (3,))[mask].astype(np.float32)
        k = int(min(colours, max(2, len(np.unique(used, axis=0)))))
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2)
        cv2.setRNGSeed(0)
        _, _, centres = cv2.kmeans(lab, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        rgb_pal = lab_to_rgb(centres.astype(np.float32)).astype(np.float64)
    else:
        rgb_pal = np.asarray(palette, np.float64)[:, :3]
    pal_lab = rgb_to_lab(rgb_pal).astype(np.float64)
    idx = np.empty(len(lab_all), np.int64)
    for s0 in range(0, len(lab_all), 262144):
        d = ((lab_all[s0:s0 + 262144, None, :] - pal_lab[None]) ** 2).sum(2)
        idx[s0:s0 + 262144] = d.argmin(1)
    out = rgba.copy()
    out[..., :3] = np.clip(np.rint(rgb_pal[idx]), 0, 255).reshape(rgba.shape[:2] + (3,))
    return out

"""Game-ready meshes from the app's heavy 3D models (Hunyuan3D marching-cubes output).

    cleanup + remesh  surface -> narrow-band signed distance on a voxel grid -> marching cubes
                      (skimage). One closed, manifold surface: no floaters, no inner shells,
                      holes smaller than a few voxels closed, thin sheets kept (the surface is
                      taken half a voxel out, so a blade or a cape is at least a voxel thick).
    symmetrize        optional: every vertex whose mirror image lies on the surface moves to the
                      average of itself and that image (a sword in one hand stays where it is).
    smooth            Taubin (lambda/mu, no shrinking), damped on creases so silhouettes and
                      sharp edges stay.
    decimate          quadric error metric, collapsed in parallel batches (an independent set
                      of cheapest edges per round, link condition + fold-over checks), numpy only.

Everything is numpy / scipy / scikit-image; no mesh library beyond trimesh for loading.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy import ndimage
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)

BUDGETS = {"character": 12_000, "animal": 12_000, "prop": 4_000, "item": 2_500,
           "building": 6_000}
LOD_SHARES = (1.0, 0.5, 0.25)


@dataclass
class Mesh:
    """Vertices (n, 3) float64, faces (m, 3) int64."""
    v: np.ndarray
    f: np.ndarray

    @property
    def tris(self) -> int:
        return len(self.f)

    def copy(self) -> "Mesh":
        return Mesh(self.v.copy(), self.f.copy())


# --------------------------------------------------------------------------- geometry helpers

def face_normals(v: np.ndarray, f: np.ndarray, unit: bool = True) -> np.ndarray:
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    if unit:
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)
    return n


def vertex_normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals."""
    fn = face_normals(v, f, unit=False)
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)


def face_areas(v, f) -> np.ndarray:
    return 0.5 * np.linalg.norm(face_normals(v, f, unit=False), axis=1)


def adjacency(f: np.ndarray, n: int) -> sp.csr_matrix:
    """Binary vertex adjacency."""
    r = np.concatenate([f[:, 0], f[:, 1], f[:, 2], f[:, 1], f[:, 2], f[:, 0]])
    c = np.concatenate([f[:, 1], f[:, 2], f[:, 0], f[:, 0], f[:, 1], f[:, 2]])
    a = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(n, n))
    a.data[:] = 1.0
    return a


def edges_of(f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unique sorted edges (e, 2) and how many faces use each."""
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), 1)
    key = e[:, 0] * (int(f.max()) + 1) + e[:, 1]
    uk, first, cnt = np.unique(key, return_index=True, return_counts=True)
    return e[first], cnt


def closest_on_triangles(p, a, b, c):
    """Closest points on triangles (a, b, c) to points p, all (n, 3): (points, barycentrics)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp = p - b
    d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    cp = p - c
    d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    n = len(p)
    bary = np.zeros((n, 3))
    done = np.zeros(n, bool)

    def put(mask, w):
        nonlocal done
        m = mask & ~done
        bary[m] = w[m] if w.ndim == 2 else w
        done |= m

    one = np.ones(n)
    put((d1 <= 0) & (d2 <= 0), np.array([1.0, 0, 0]))
    put((d3 >= 0) & (d4 <= d3), np.array([0, 1.0, 0]))
    put((d6 >= 0) & (d5 <= d6), np.array([0, 0, 1.0]))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = d1 / (d1 - d3)
        put((vc <= 0) & (d1 >= 0) & (d3 <= 0), np.stack([1 - t, t, 0 * one], 1))
        t = d2 / (d2 - d6)
        put((vb <= 0) & (d2 >= 0) & (d6 <= 0), np.stack([1 - t, 0 * one, t], 1))
        t = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        put((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), np.stack([0 * one, 1 - t, t], 1))
        den = va + vb + vc
        den = np.where(np.abs(den) < 1e-30, 1e-30, den)
        w1, w2 = vb / den, vc / den
        put(np.ones(n, bool), np.stack([1 - w1 - w2, w1, w2], 1))
    bary = np.nan_to_num(bary, nan=1 / 3)
    q = a * bary[:, :1] + b * bary[:, 1:2] + c * bary[:, 2:]
    return q, bary


def sample_surface(v, f, spacing: float, seed: int = 0, minimum: int = 1):
    """Points on every face, about one per spacing^2 of area: (points, face index)."""
    rng = np.random.default_rng(seed)
    area = face_areas(v, f)
    count = np.maximum(minimum, np.ceil(area / spacing ** 2).astype(np.int64))
    face = np.repeat(np.arange(len(f)), count)
    r1, r2 = rng.random(len(face)), rng.random(len(face))
    s = np.sqrt(r1)
    w = np.stack([1 - s, s * (1 - r2), s * r2], 1)
    tri = v[f[face]]
    return (tri * w[..., None]).sum(1), face


class Surface:
    """Closest-point queries on a triangle mesh (KD-tree of dense samples + exact refinement
    on the nearest samples' faces and their neighbours)."""

    def __init__(self, v, f, spacing: float | None = None):
        self.v, self.f = np.asarray(v, np.float64), np.asarray(f, np.int64)
        if spacing is None:
            edge = np.linalg.norm(self.v[self.f[:, 0]] - self.v[self.f[:, 1]], axis=1)
            spacing = float(np.median(edge)) * 0.6
        self.pts, self.face = sample_surface(self.v, self.f, spacing)
        self.tree = cKDTree(self.pts)
        self.fn = face_normals(self.v, self.f)

    def closest(self, p, k: int = 4):
        """(points, faces, barycentrics, distances) of the closest surface points."""
        p = np.asarray(p, np.float64)
        _, idx = self.tree.query(p, k=k, workers=-1)
        idx = idx.reshape(len(p), -1)
        best_d = np.full(len(p), np.inf)
        best_q = np.zeros_like(p)
        best_f = np.zeros(len(p), np.int64)
        best_b = np.zeros((len(p), 3))
        for j in range(idx.shape[1]):
            fi = self.face[idx[:, j]]
            tri = self.f[fi]
            q, b = closest_on_triangles(p, self.v[tri[:, 0]], self.v[tri[:, 1]], self.v[tri[:, 2]])
            d = np.linalg.norm(p - q, axis=1)
            better = d < best_d
            best_d[better], best_q[better], best_f[better], best_b[better] = \
                d[better], q[better], fi[better], b[better]
        return best_q, best_f, best_b, best_d


# --------------------------------------------------------------------------- cleanup + remesh

def weld(v, f, tol: float = 1e-7) -> Mesh:
    """Merge coincident vertices, drop degenerate and repeated faces and unused vertices."""
    key = np.round(v / max(tol, 1e-12)).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    f = inv.reshape(-1)[f]
    ok = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 2] != f[:, 0])
    f = f[ok]
    _, keep = np.unique(np.sort(f, 1), axis=0, return_index=True)
    f = f[np.sort(keep)]
    return compact(Mesh(v[first], f))


def compact(m: Mesh) -> Mesh:
    used = np.zeros(len(m.v), bool)
    used[m.f.reshape(-1)] = True
    remap = np.cumsum(used) - 1
    return Mesh(m.v[used], remap[m.f])


def components(m: Mesh) -> tuple[int, np.ndarray]:
    """Vertex-connected pieces: (count, label per vertex)."""
    from scipy.sparse.csgraph import connected_components
    return connected_components(adjacency(m.f, len(m.v)), directed=False)


def drop_small_pieces(m: Mesh, share: float = 0.002) -> Mesh:
    """Remove pieces with less than `share` of the surface area (floaters)."""
    k, lab = components(m)
    if k <= 1:
        return m
    area = face_areas(m.v, m.f)
    fa = np.bincount(lab[m.f[:, 0]], area, k)
    keep = fa >= share * fa.sum()
    keep[np.argmax(fa)] = True
    return compact(Mesh(m.v, m.f[keep[lab[m.f[:, 0]]]]))


@dataclass
class Grid:
    origin: np.ndarray
    step: float
    shape: tuple

    def points(self, idx) -> np.ndarray:
        return self.origin + np.asarray(idx, np.float64) * self.step


def remesh(m: Mesh, voxels: int = 256, offset: float = 0.45, close: int = 1,
           min_share: float = 0.002) -> tuple[Mesh, Grid]:
    """A clean closed manifold surface around the model by marching cubes on a narrow-band
    signed distance. voxels = cells along the longest side; offset = how far out (in voxels)
    the surface is taken (keeps thin sheets); close = voxels of morphological closing for the
    inside/outside fill (closes holes up to ~2*close voxels wide)."""
    from skimage.measure import marching_cubes
    t0 = time.perf_counter()
    lo, hi = m.v.min(0), m.v.max(0)
    step = float((hi - lo).max()) / voxels
    pad = 3 + close
    origin = lo - pad * step
    shape = tuple(int(x) for x in np.ceil((hi - lo) / step).astype(int) + 2 * pad + 1)
    grid = Grid(origin, step, shape)
    pts, face = sample_surface(m.v, m.f, step * 0.45)
    cell = np.floor((pts - origin) / step + 0.5).astype(np.int64)      # nearest grid node
    shell = np.zeros(shape, bool)
    shell[cell[:, 0], cell[:, 1], cell[:, 2]] = True
    # inside = what the (closed) shell encloses
    struct = ndimage.generate_binary_structure(3, 3)
    closed = ndimage.binary_dilation(shell, struct, iterations=close) if close else shell
    filled = ndimage.binary_fill_holes(closed)
    inside = filled & ~closed
    if close:
        inside = ndimage.binary_dilation(inside, struct, iterations=close) & filled
    # narrow band: unsigned distance to the true surface, sign from inside/outside, normals near
    band = ndimage.binary_dilation(shell, struct, iterations=3)
    bi = np.argwhere(band)
    bp = grid.points(bi)
    surf = Surface(m.v, m.f, spacing=step * 0.45)
    q, fi, _, d = surf.closest(bp, k=3)
    side = ((bp - q) * surf.fn[fi]).sum(1)
    far = np.full(shape, 4.0 * step, np.float32)
    sdf = np.where(inside, -far, far)
    sign = np.where(inside[bi[:, 0], bi[:, 1], bi[:, 2]], -1.0, 1.0)
    near = d < 1.5 * step                       # trust the face's side close to the surface
    sign[near] = np.where(side[near] < 0, -1.0, 1.0)
    sdf[bi[:, 0], bi[:, 1], bi[:, 2]] = sign * d
    sdf = ndimage.gaussian_filter(sdf, 0.5)
    verts, faces, _, _ = marching_cubes(sdf, level=offset * step, spacing=(step,) * 3,
                                        allow_degenerate=False)
    out = Mesh(verts.astype(np.float64) + origin, faces.astype(np.int64)[:, ::-1].copy())
    out = weld(out.v, out.f, tol=step * 1e-4)
    # marching cubes winding: outward normals point to growing sdf
    fn = face_normals(out.v, out.f)
    c = out.v[out.f].mean(1)
    probe = np.floor((c + fn * step - origin) / step + 0.5).astype(int)
    probe = np.clip(probe, 0, np.array(shape) - 1)
    if np.mean(sdf[probe[:, 0], probe[:, 1], probe[:, 2]] > offset * step) < 0.5:
        out.f = out.f[:, ::-1].copy()
    out = drop_small_pieces(out, min_share)
    log.info("remesh: %d -> %d tris on %s grid in %.1fs", m.tris, out.tris, shape,
             time.perf_counter() - t0)
    return out, grid


# --------------------------------------------------------------------------- symmetry + smoothing

def symmetry_score(m: Mesh, axis: int = 0, centre: float | None = None, tol: float = 0.012):
    """(share of the surface whose mirror image lies on the surface within tol x size, centre)."""
    size = float((m.v.max(0) - m.v.min(0)).max())
    if centre is None:
        a = face_areas(m.v, m.f)
        centre = float((m.v[m.f].mean(1)[:, axis] * a).sum() / a.sum())
    mir = m.v.copy()
    mir[:, axis] = 2 * centre - mir[:, axis]
    d, _ = cKDTree(m.v).query(mir, workers=-1)
    return float(np.mean(d < tol * size)), centre


def symmetrize(m: Mesh, axis: int = 0, centre: float | None = None, tol: float = 0.02,
               rounds: int = 3) -> tuple[Mesh, float]:
    """Pull mirror-matching parts into exact symmetry; parts with no mirror twin stay.
    Returns (mesh, share of vertices that were symmetrized)."""
    share, centre = symmetry_score(m, axis, centre)
    size = float((m.v.max(0) - m.v.min(0)).max())
    v = m.v.copy()
    w = np.zeros(len(v))
    for _ in range(rounds):
        tree = cKDTree(v)
        mir = v.copy()
        mir[:, axis] = 2 * centre - mir[:, axis]
        d, j = tree.query(mir, workers=-1)
        w = np.clip(1.0 - d / (tol * size), 0.0, 1.0) ** 2
        twin = v[j].copy()
        twin[:, axis] = 2 * centre - twin[:, axis]
        target = 0.5 * (v + twin)
        target[:, axis] = np.where(np.abs(v[:, axis] - centre) < 0.25 * tol * size,
                                   target[:, axis], target[:, axis])
        v = v + w[:, None] * (target - v)
    return Mesh(v, m.f.copy()), float(np.mean(w > 0.5))


def crease_weight(m: Mesh, angle: float = 35.0) -> np.ndarray:
    """Per-vertex 0 (on a crease sharper than `angle` degrees) .. 1 (smooth)."""
    fn = face_normals(m.v, m.f)
    vn = vertex_normals(m.v, m.f)
    dev = np.zeros(len(m.v))
    for k in range(3):
        cosang = np.clip((fn * vn[m.f[:, k]]).sum(1), -1, 1)
        np.maximum.at(dev, m.f[:, k], np.degrees(np.arccos(cosang)))
    return np.clip(1.0 - (dev - angle) / angle, 0.15, 1.0)


def taubin(m: Mesh, rounds: int = 10, lam: float = 0.5, mu: float = -0.53,
           crease: float | None = 35.0, keep=None) -> Mesh:
    """Taubin smoothing (no shrinking), damped on creases (crease degrees, None = off).
    keep: per-vertex 0..1 extra damping (1 = don't move)."""
    a = adjacency(m.f, len(m.v))
    deg = np.asarray(a.sum(1)).ravel()
    a = sp.diags(1.0 / np.maximum(deg, 1)) @ a
    v = m.v.copy()
    # creases measured after a few plain rounds: noise is gone then, real edges stay sharp
    w = crease_weight(taubin(m, 6, lam, mu, crease=None), crease) if crease else np.ones(len(v))
    if keep is not None:
        w = w * (1.0 - np.clip(keep, 0, 1))
    w = w[:, None]
    for _ in range(rounds):
        v = v + lam * w * (a @ v - v)
        v = v + mu * w * (a @ v - v)
    return Mesh(v, m.f.copy())


# --------------------------------------------------------------------------- decimation (QEM)

def _plane_quadrics(v, f) -> np.ndarray:
    """(n, 4, 4) area-weighted sum of face plane quadrics per vertex."""
    fn = face_normals(v, f, unit=False)
    area = np.linalg.norm(fn, axis=1) * 0.5
    n = fn / np.maximum(2 * area[:, None], 1e-20)
    d = -(n * v[f[:, 0]]).sum(1)
    p = np.concatenate([n, d[:, None]], 1)
    kq = p[:, :, None] * p[:, None, :] * area[:, None, None]
    q = np.zeros((len(v), 4, 4))
    for k in range(3):
        np.add.at(q, f[:, k], kq)
    return q


def _boundary_quadrics(v, f, weight: float) -> np.ndarray:
    """Planes through boundary edges, perpendicular to their face: keeps open borders."""
    q = np.zeros((len(v), 4, 4))
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    owner = np.tile(np.arange(len(f)), 3)
    key = np.sort(e, 1)
    k = key[:, 0] * (len(v) + 1) + key[:, 1]
    _, inv, cnt = np.unique(k, return_inverse=True, return_counts=True)
    b = cnt[inv] == 1
    if not b.any():
        return q
    e, owner = e[b], owner[b]
    fn = face_normals(v, f)[owner]
    dvec = v[e[:, 1]] - v[e[:, 0]]
    length = np.linalg.norm(dvec, axis=1)
    n = np.cross(dvec, fn)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)
    p = np.concatenate([n, -(n * v[e[:, 0]]).sum(1)[:, None]], 1)
    kq = p[:, :, None] * p[:, None, :] * (weight * length ** 2)[:, None, None]
    for j in range(2):
        np.add.at(q, e[:, j], kq)
    return q


def decimate(m: Mesh, target: int, *, boundary_weight: float = 100.0, max_rounds: int = 800,
             importance=None, quiet: bool = False, debug: bool = False,
             regularize: float = 0.05, strict: bool = False) -> Mesh:
    """Quadric-error edge collapse down to `target` triangles.

    Parallel: each round collapses an independent set of the cheapest edges (no two share a
    face), so it is all array maths. importance: optional per-vertex multiplier (>1 keeps more
    triangles there)."""
    t0 = time.perf_counter()
    v, f = m.v.copy(), m.f.copy()
    nv = len(v)
    Q = _plane_quadrics(v, f) + _boundary_quadrics(v, f, boundary_weight)
    scale = float((v.max(0) - v.min(0)).max())
    # a weak pull to each vertex's own place: keeps solves stable on flat / creased parts
    # (an in-plane move costs nothing to the planes alone, and spikes grew from that)
    va = np.zeros(nv)
    fa = face_areas(v, f)
    for k in range(3):
        np.add.at(va, f[:, k], fa / 3)
    eps = regularize * va
    K = np.zeros((nv, 4, 4))
    K[:, 0, 0] = K[:, 1, 1] = K[:, 2, 2] = 1.0
    K[:, :3, 3] = -v
    K[:, 3, :3] = -v
    K[:, 3, 3] = (v * v).sum(1)
    Q += eps[:, None, None] * K
    imp = np.ones(nv) if importance is None else np.asarray(importance, np.float64)
    blocked_keys = np.zeros(0, np.int64)          # rejected edges wait till their area changes
    alive_faces = f
    empty = 0
    relax = False                                 # thin sheets: allow non-manifold collapses
    for rnd in range(max_rounds):
        nf = len(alive_faces)
        if nf <= target:
            break
        f = alive_faces
        e, efaces = edges_of(f)
        a, b = e[:, 0], e[:, 1]
        Qe = Q[a] + Q[b]
        A = Qe[:, :3, :3]
        rhs = -Qe[:, :3, 3]
        det = np.linalg.det(A)
        mid = 0.5 * (v[a] + v[b])
        elen = np.linalg.norm(v[a] - v[b], axis=1)
        x = mid.copy()
        good = np.abs(det) > 1e-30
        if good.any():
            sol = np.linalg.solve(A[good], rhs[good][..., None])[..., 0]
            ok = np.linalg.norm(sol - mid[good], axis=1) < 1.5 * elen[good]
            gi = np.nonzero(good)[0]
            x[gi[ok]] = sol[ok]

        def cost_at(p):
            h = np.concatenate([p, np.ones((len(p), 1))], 1)
            return np.einsum("ni,nij,nj->n", h, Qe, h)

        c = cost_at(x)
        for alt in (v[a], v[b], mid):             # an endpoint can beat a bad solve
            ca = cost_at(alt)
            better = ca < c - 1e-18
            x[better], c[better] = alt[better], ca[better]
        c = np.maximum(c, 0.0) + 1e-6 * (elen / scale) ** 2 * scale ** 2 * 1e-4
        c = c * np.maximum(imp[a], imp[b])
        ekey = a.astype(np.int64) * nv + b
        if len(blocked_keys):
            c = np.where(np.isin(ekey, blocked_keys), np.inf, c)
        # unique costs (tie break by index) for the independent set
        order = np.argsort(c, kind="stable")
        rank = np.empty(len(c), np.int64)
        rank[order] = np.arange(len(c))
        need = (nf - target) // 2 + 1
        cand = (rank < max(need, len(c) // 6)) & np.isfinite(c)   # the cheaper edges only
        big = np.int64(len(c) + 10)
        r = np.where(cand, rank, big)
        pv = np.full(nv, big, np.int64)
        np.minimum.at(pv, a, r)
        np.minimum.at(pv, b, r)
        fmin = pv[f].min(1)
        qv = np.full(nv, big, np.int64)
        for k in range(3):
            np.minimum.at(qv, f[:, k], fmin)
        sel = cand & (qv[a] == r) & (qv[b] == r)
        si = np.nonzero(sel)[0]
        if len(si) == 0:
            break
        # link condition: common neighbours == faces on the edge
        adj = adjacency(f, nv)
        common = np.asarray(adj[a[si]].multiply(adj[b[si]]).sum(1)).ravel()
        okl = common == efaces[si]
        if relax:
            okl[:] = True
        # don't pinch: an inner edge between two border vertices
        border = np.zeros(nv, bool)
        ee, ec = e, efaces
        border[ee[ec == 1].ravel()] = True
        okl &= ~((efaces[si] == 2) & border[a[si]] & border[b[si]])
        nlink = int((~okl).sum())
        lrej = si[~okl]
        si = si[okl]
        # fold-over + sliver check on every face around the moved vertices
        eid = np.full(nv, -1, np.int64)
        eid[a[si]] = si
        eid[b[si]] = si
        fe = eid[f]                                   # (nf, 3)
        touched = (fe >= 0).any(1)
        tf = np.nonzero(touched)[0]
        fet = fe[tf]
        which = np.max(fet, 1)
        n_on = (fet == which[:, None]).sum(1)
        keep_face = n_on == 1                         # faces with both ends vanish
        tf, fet, which = tf[keep_face], fet[keep_face], which[keep_face]
        tri = v[f[tf]].copy()
        old_n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        moved = fet >= 0
        tri[moved] = x[which][np.nonzero(moved)[0]]
        new_n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        on = np.linalg.norm(old_n, axis=1)
        nn = np.linalg.norm(new_n, axis=1)
        cosang = (old_n * new_n).sum(1) / np.maximum(on * nn, 1e-30)
        l2 = sum(np.square(np.linalg.norm(tri[:, i] - tri[:, (i + 1) % 3], axis=1)) for i in range(3))
        quality = 2 * np.sqrt(3) * nn / np.maximum(l2, 1e-30)
        bad_face = (cosang < 0.25) | (quality < 0.04)
        bad = np.zeros(len(e), bool)
        bad[which[bad_face]] = True
        rejected = np.concatenate([si[bad[si]], lrej])
        si = si[~bad[si]]
        if debug:
            print(rnd, nf, 'sel', int(sel.sum()), 'link-', nlink, 'fold-', int(bad[si].sum() if False else len(rejected) - nlink), 'ok', len(si))
        if len(rejected):
            blocked_keys = np.union1d(blocked_keys, ekey[rejected])
        # nothing passes but the topology rule (thin capes, blades): let sheets touch
        relax = (not strict) and len(si) < 0.1 * max(1, int(sel.sum())) and nlink > len(rejected) // 2
        if len(si) == 0:
            empty += 1
            if empty >= 8:
                if not quiet:
                    log.info("decimate: stuck at %d tris", nf)
                break
            continue
        empty = 0
        # collapse b -> a
        aa, bb = a[si], b[si]
        v[aa] = x[si]
        Q[aa] += Q[bb]
        imp[aa] = np.maximum(imp[aa], imp[bb])
        remap = np.arange(nv)
        remap[bb] = aa
        touched = np.zeros(nv, bool)
        moved_v = np.zeros(nv, bool)
        moved_v[aa] = moved_v[bb] = True
        touched[f[moved_v[f].any(1)].ravel()] = True
        f = remap[f]
        okf = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 2] != f[:, 0])
        f = f[okf]
        if relax:                                  # same face twice, same way round: once
            rot = np.argmin(f, 1)
            canon = np.stack([f[np.arange(len(f)), rot], f[np.arange(len(f)), (rot + 1) % 3],
                              f[np.arange(len(f)), (rot + 2) % 3]], 1)
            _, first = np.unique(canon, axis=0, return_index=True)
            f = f[np.sort(first)]
        alive_faces = f
        if len(blocked_keys):
            ka, kb = blocked_keys // nv, blocked_keys % nv
            blocked_keys = blocked_keys[~(touched[ka] | touched[kb])]
    out = compact(Mesh(v, alive_faces))
    if not quiet:
        log.info("decimate: %d -> %d tris in %.1fs", m.tris, out.tris, time.perf_counter() - t0)
    return out


def fast_available() -> bool:
    """Is fast-simplification (Forstmann's quadric simplifier, MIT, C++) installed?"""
    try:
        import fast_simplification  # noqa: F401
        return True
    except ImportError:
        return False


def decimate_fast(m: Mesh, target: int, *, agg: float = 7.0, finish: float = 0.0,
                  quiet: bool = False, **numpy_options) -> Mesh:
    """decimate() with fast-simplification (Forstmann's quadric simplifier in C++, ~50x
    faster on a 400k-triangle remesh) doing the bulk: down to finish x target triangles, then
    the numpy decimate() takes the last step, so the final edges pass its manifold, fold-over
    and sliver checks (finish 0 = fast-simplification all the way). Falls back to decimate()
    when the package is missing."""
    try:
        import fast_simplification
    except ImportError:
        return decimate(m, target, quiet=quiet, **numpy_options)
    t0 = time.perf_counter()
    if m.tris <= target:
        return m.copy()
    goal = int(target * finish) if finish >= 1 else int(target)
    out = m
    if m.tris > goal:
        v, f = fast_simplification.simplify(np.ascontiguousarray(m.v, np.float64),
                                            np.ascontiguousarray(m.f, np.int32),
                                            target_count=goal, agg=agg)
        f = np.asarray(f, np.int64)
        ok = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 2] != f[:, 0])
        out = compact(Mesh(np.asarray(v, np.float64), f[ok]))
    if out.tris > target:
        out = decimate(out, target, quiet=True, **numpy_options)
    if not quiet:
        log.info("decimate (fast): %d -> %d tris in %.2fs", m.tris, out.tris,
                 time.perf_counter() - t0)
    return out


# --------------------------------------------------------------------------- one call

@dataclass
class Options:
    budget: int = 12_000                    # LOD0 triangles
    lods: tuple = LOD_SHARES                # shares of the budget per LOD
    voxels: int = 256                       # remesh resolution (cells on the longest side)
    smooth: int = 8                         # Taubin rounds (0 = off)
    crease: float = 35.0                    # degrees kept sharp while smoothing
    symmetry: str = "auto"                  # auto | on | off (mirror across x = centre)
    symmetry_axis: int = 0
    min_share: float = 0.002                # pieces smaller than this share of area go
    importance: object = None               # callable(v) -> per-vertex multiplier
    plane: object = None                    # (point, normal) mirror plane (rig's left/right)
    decimator: str = "auto"                 # auto (fast-simplification when installed, else
                                            # numpy) | fast | numpy
    fast_agg: float = 3.0                   # fast-simplification aggressiveness (lower = closer)


@dataclass
class GameMeshes:
    high: Mesh                              # cleaned, remeshed, smoothed: the bake source
    lods: list                              # [Mesh] LOD0, LOD1, LOD2
    notes: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    symmetric: float = 0.0


def build(v, f, options: Options | None = None) -> GameMeshes:
    """Heavy mesh -> cleaned high-poly + LODs at the budget."""
    o = options or Options()
    notes, times = [], {}
    t = time.perf_counter()
    src = weld(np.asarray(v, np.float64), np.asarray(f, np.int64))
    k, _ = components(src)
    src = drop_small_pieces(src, o.min_share)
    k2, _ = components(src)
    if k2 < k:
        notes.append(f"dropped {k - k2} floating pieces")
    high, _ = remesh(src, o.voxels, min_share=o.min_share)
    times["remesh"] = time.perf_counter() - t
    sym = 0.0
    if o.symmetry != "off":
        t = time.perf_counter()
        R, origin = np.eye(3), np.zeros(3)
        if o.plane is not None:                 # turn the mirror plane onto x = 0
            origin = np.asarray(o.plane[0], np.float64)
            nx = np.asarray(o.plane[1], np.float64)
            nx = nx / np.linalg.norm(nx)
            up = np.array([0.0, 1.0, 0.0])
            nz = np.cross(nx, up)
            if np.linalg.norm(nz) < 1e-6:
                nz = np.cross(nx, [0.0, 0.0, 1.0])
            nz /= np.linalg.norm(nz)
            ny = np.cross(nz, nx)
            R = np.stack([nx, ny, nz])
        local = Mesh((high.v - origin) @ R.T, high.f)
        axis = 0 if o.plane is not None else o.symmetry_axis
        score, centre = symmetry_score(local, axis, 0.0 if o.plane is not None else None)
        if o.symmetry == "on" or score > 0.55:
            local, sym = symmetrize(local, axis, centre)
            high = Mesh(local.v @ R + origin, high.f)
            notes.append(f"symmetrized {sym:.0%} of the surface (mirror score {score:.0%})")
        else:
            notes.append(f"not symmetric enough to mirror ({score:.0%})")
        times["symmetry"] = time.perf_counter() - t
    if o.smooth:
        t = time.perf_counter()
        high = taubin(high, o.smooth, crease=o.crease)
        times["smooth"] = time.perf_counter() - t
    t = time.perf_counter()
    imp = o.importance(high.v) if callable(o.importance) else None
    fast = (o.decimator == "fast" or (o.decimator == "auto" and fast_available())) and imp is None
    lods = []
    cur = high
    for share in o.lods:
        target = max(64, int(o.budget * share))
        if fast:                  # each LOD from the one before (closer than from the high)
            cur = decimate_fast(cur, target, agg=o.fast_agg)
        else:
            cur = decimate(cur, target, importance=None if imp is None or cur is not high else imp)
        lods.append(cur)
    times["decimate"] = time.perf_counter() - t
    if fast:
        notes.append("decimated with fast-simplification")
    return GameMeshes(high, lods, notes, times, sym)

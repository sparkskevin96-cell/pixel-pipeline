"""A small software renderer for previews of game meshes: z-buffered triangles (uvbake.rasterize),
vertex colours or a texture (nearest or bilinear), optional tangent-space normal map, Lambert +
ambient + rim, optional wireframe. Orthographic camera turned by yaw / tilted by elevation."""
from __future__ import annotations

import math

import numpy as np

from .uvbake import rasterize


def _rot(yaw: float, elev: float) -> np.ndarray:
    y, e = math.radians(yaw), math.radians(elev)
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rx = np.array([[1, 0, 0], [0, math.cos(e), -math.sin(e)], [0, math.sin(e), math.cos(e)]])
    return rx @ ry


def _sample(img: np.ndarray, uv: np.ndarray, nearest: bool) -> np.ndarray:
    h, w = img.shape[:2]
    x = np.mod(uv[:, 0], 1.0) * w - 0.5
    y = np.mod(uv[:, 1], 1.0) * h - 0.5
    if nearest:
        xi = np.clip(np.floor(x + 0.5).astype(int), 0, w - 1)
        yi = np.clip(np.floor(y + 0.5).astype(int), 0, h - 1)
        return img[yi, xi].astype(np.float64)
    x0 = np.clip(np.floor(x).astype(int), 0, w - 1)
    y0 = np.clip(np.floor(y).astype(int), 0, h - 1)
    x1, y1 = np.clip(x0 + 1, 0, w - 1), np.clip(y0 + 1, 0, h - 1)
    fx, fy = (x - np.floor(x))[:, None], (y - np.floor(y))[:, None]
    f = img.astype(np.float64)
    return (f[y0, x0] * (1 - fx) * (1 - fy) + f[y0, x1] * fx * (1 - fy)
            + f[y1, x0] * (1 - fx) * fy + f[y1, x1] * fx * fy)


def render(v, f, *, size: int = 512, yaw: float = 0.0, elev: float = 10.0, colours=None,
           uv=None, texture=None, normal_map=None, tangents=None, normals=None,
           nearest: bool = False, wire: bool = False, frame=None, background=(0, 0, 0, 0),
           light=(-0.45, 0.65, 0.6), wire_colour=(20, 20, 26)) -> np.ndarray:
    """RGBA uint8 image. frame = (centre (3,), half extent) to share framing between renders.
    colours: per-vertex (n, 3) 0..255 sRGB; or uv + texture (H, W, 3/4) uint8."""
    from .gamemesh import vertex_normals
    v = np.asarray(v, np.float64)
    f = np.asarray(f, np.int64)
    R = _rot(yaw, elev)
    if frame is None:
        lo, hi = v.min(0), v.max(0)
        centre = (lo + hi) / 2
        half = float(np.linalg.norm(hi - lo)) / 2 * 0.92
    else:
        centre, half = frame
    p = (v - centre) @ R.T
    scale = size / (2 * half)
    sx = p[:, 0] * scale + size / 2
    sy = -p[:, 1] * scale + size / 2
    depth = -p[:, 2]                               # camera looks down -z; nearer = smaller
    tri = np.stack([sx, sy], 1)[f]
    # back faces away
    d1, d2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    facing = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) < 0
    fidx = np.nonzero(facing)[0]
    px, fi, b = rasterize(tri[fidx], size, size)
    fi = fidx[fi]
    z = (depth[f[fi]] * b).sum(1)
    order = np.lexsort((z, px))
    px, fi, b, z = px[order], fi[order], b[order], z[order]
    first = np.ones(len(px), bool)
    first[1:] = px[1:] != px[:-1]
    px, fi, b = px[first], fi[first], b[first]
    if normals is None:
        normals = vertex_normals(v, f)
    n = (normals[f[fi]] * b[..., None]).sum(1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    if uv is not None and texture is not None:
        t_uv = (uv[f[fi]] * b[..., None]).sum(1)
        base = _sample(texture[..., :3], t_uv, nearest)
        if normal_map is not None and tangents is not None:
            tn = _sample(normal_map[..., :3], t_uv, nearest) / 127.5 - 1
            T = (tangents[f[fi], :3] * b[..., None]).sum(1)
            T -= n * (T * n).sum(1, keepdims=True)
            T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
            w = np.sign((tangents[f[fi], 3] * b).sum(1))
            w[w == 0] = 1
            B = np.cross(n, T) * w[:, None]
            n = T * tn[:, :1] + B * tn[:, 1:2] + n * tn[:, 2:3]
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    elif colours is not None:
        base = (np.asarray(colours, np.float64)[f[fi], :3] * b[..., None]).sum(1)
    else:
        base = np.full((len(fi), 3), 190.0)
    nc = n @ R.T                                   # camera space
    L = np.asarray(light, np.float64)
    L /= np.linalg.norm(L)
    lam = np.clip(nc @ L, 0, 1)
    rim = np.clip(1 - np.abs(nc[:, 2]), 0, 1) ** 3 * 0.15
    shade = 0.42 + 0.68 * lam + rim
    rgb = np.clip(base * shade[:, None], 0, 255)
    img = np.zeros((size * size, 4), np.float64)
    img[:] = background
    img[px, :3] = rgb
    img[px, 3] = 255
    img = img.reshape(size, size, 4)
    if wire:
        _wire(img, tri[fidx], f[fidx], z_ok=None, colour=wire_colour, depth=depth, size=size,
              zbuf=_zbuffer(px, (depth[f[fi]] * b).sum(1), size))
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def _zbuffer(px, z, size):
    zb = np.full(size * size, np.inf)
    zb[px] = z
    return zb.reshape(size, size)


def _wire(img, tri, faces, z_ok, colour, depth, size, zbuf):
    """Draw visible triangle edges (depth-tested against the z-buffer)."""
    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    ez = np.concatenate([depth[faces][:, [0, 1]], depth[faces][:, [1, 2]], depth[faces][:, [2, 0]]])
    length = np.linalg.norm(e[:, 1] - e[:, 0], axis=1)
    steps = np.maximum(2, np.ceil(length * 1.5).astype(int))
    idx = np.repeat(np.arange(len(e)), steps)
    k = np.arange(int(steps.sum())) - np.repeat(np.cumsum(steps) - steps, steps)
    t = (k / (steps[idx] - 1))[:, None]
    pts = e[idx, 0] * (1 - t) + e[idx, 1] * t
    zz = ez[idx, 0] * (1 - t[:, 0]) + ez[idx, 1] * t[:, 0]
    x = np.clip(pts[:, 0].astype(int), 0, size - 1)
    y = np.clip(pts[:, 1].astype(int), 0, size - 1)
    span = float(np.nanmax(zbuf[np.isfinite(zbuf)]) - np.nanmin(zbuf[np.isfinite(zbuf)])) \
        if np.isfinite(zbuf).any() else 1.0
    vis = zz <= zbuf[y, x] + 0.004 * span
    a = 0.75
    img[y[vis], x[vis], :3] = img[y[vis], x[vis], :3] * (1 - a) + np.asarray(colour) * a
    img[y[vis], x[vis], 3] = 255


def turntable(v, f, frames: int = 8, **kw) -> list[np.ndarray]:
    """Renders all the way round (shared framing)."""
    v = np.asarray(v, np.float64)
    lo, hi = v.min(0), v.max(0)
    frame = kw.pop("frame", None) or ((lo + hi) / 2, float(np.linalg.norm(hi - lo)) / 2 * 0.92)
    return [render(v, f, yaw=360.0 * i / frames, frame=frame, **kw) for i in range(frames)]

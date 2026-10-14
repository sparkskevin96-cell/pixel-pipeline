"""Drawn characters on a skeleton (hybrid): the GPU draws the character as pixel art from every
direction, the rigged 3D model only guides it, and the skeleton bends the DRAWING for every
animation frame (like Spine / paper-doll animation) so the pixels stay crisp.

    1. Key drawings. The key pose (first frame of idle: most frames are near it, so drawings
       bend least; the rig's rest pose if there is no idle) is rendered from each direction with
       the library camera (Style.elevation), already on the frame's pixel grid (blocky: each
       frame pixel a k x k block, hard edges) and SDXL redraws it as a pixel-art sprite
       (img2img ~0.6, the project's house style + style LoRA, one seed for all directions). The
       drawing is cut out and brought back to the frame grid at EXACTLY the camera's scale and
       position (the init picture is a known crop of the frame, k px per frame pixel), so it
       lines up with the 3D model pixel for pixel. One palette for all directions. Drawings
       whose silhouette strays too far from the model's (IoU) are drawn again with another seed.
    2. Correspondence. The rest pose seen from the same camera at frame size: per pixel the
       front surface's 3D position, normal and skin weights. Every drawn pixel takes those of
       its own pixel (or of the nearest model pixel, offset in the screen plane, for pixels the
       drawing added), so each drawn pixel is a little square glued to the skeleton.
    3. Animation. Per frame the squares are skinned with the rig's joint matrices (the same
       linear blend skinning anim3d uses), projected with the same camera and splatted at 4x4
       samples per pixel; per output pixel the nearest-to-centre sample of the front surface
       wins (nearest-neighbour sampling of the bent drawing: no new colours, no blur).
       Parts hidden behind others at rest that come into view (an arm swinging out from behind
       the body) are filled from the model's surface points, coloured from the drawing of the
       nearest direction that sees them (palette indices, so no new colours either).
    4. Normal maps: the skinned 3D normal of each drawn pixel, mixed with normals.from_drawing
       relief, so HD-2D lights and glow keep working. Output = anim3d.AnimatedSet: the usual
       sheet.png/json/aseprite/tres/_n/_glow writers.

Identity: the key pose reproduces the key drawing exactly (tests/test_drawnrig.py).
Limits: a part the drawing puts somewhere else than the model (SDXL moved the sword, added a
shield) follows the bones under it; big whole-body turns (death lying down) foreshorten the
drawing like a paper cut-out.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .anim3d import (Action, AnimatedSet, Clip, Rig, clip_times, load_rig, sample_surface,
                     select_clips, surface_pose, write_animated)
from .pipeline import Sprite
from .turnaround import direction_names

SS = 4                          # samples per pixel side when bending a drawing
DEPTH_EPS = 2.0                 # px: samples this close behind the nearest are the same surface
TIGHT_EPS = 0.6                 # px: the points a rest-view pixel takes its 3D data from
OCCLUDE_EPS = 3.0               # px: a hidden part must be this much nearer to cover the drawing
MIN_COVER = 5                   # of SS*SS samples: a pixel needs this many to be drawn

# what each facing looks like, in words the image model knows (screen: E = facing right)
DIRECTION_WORDS = {
    "E": ("side view, profile facing right", "front view, back view, facing left"),
    "SE": ("three-quarter view, body turned toward the bottom right, facing the viewer and right",
           "back view, facing left"),
    "S": ("front view, facing the viewer", "back view, side view"),
    "SW": ("three-quarter view, body turned toward the bottom left, facing the viewer and left",
           "back view, facing right"),
    "W": ("side view, profile facing left", "front view, back view, facing right"),
    "NW": ("three-quarter back view, seen from behind, turned away toward the top left",
           "face, front view, facing the viewer"),
    "N": ("back view, seen from behind, facing away from the viewer",
          "face, front view, facing the viewer"),
    "NE": ("three-quarter back view, seen from behind, turned away toward the top right",
           "face, front view, facing the viewer"),
}
KEY_PROMPT = ("pixel art sprite of {prompt}, single character, full body, standing, {view}, seen "
              "from slightly above, centered, plain white background, game asset, clean outline, "
              "crisp readable pixel shapes")
KEY_NEGATIVE = ("multiple characters, multiple views, sprite sheet, cropped, out of frame, cut off, "
                "busy background, scenery, cast shadow, ground, frame, border, text, watermark, "
                "3d render, photo, blurry, smooth gradient")


def direction_words(name: str) -> tuple[str, str]:
    """(positive, negative) view words for a direction name (16-direction names use the nearest
    of the 8 main ones)."""
    if name in DIRECTION_WORDS:
        return DIRECTION_WORDS[name]
    names16 = direction_names(16)
    k = names16.index(name)
    main = direction_names(8)[(k + 1) // 2 % 8] if k % 2 else direction_names(8)[k // 2]
    return DIRECTION_WORDS[main]


# --------------------------------------------------------------------------- camera

@dataclass
class Camera:
    """turnaround._splat's orthographic camera for one facing (screen x right, y down; pixel i
    covers [i, i+1))."""
    angle: float          # facing on screen, degrees clockwise from E (Godot Vector2.angle)
    elevation: float
    scale: float          # px per world unit
    ox: float
    oy: float
    width: int
    height: int
    yaw: float = 0.0

    def _cs(self):
        t = math.radians(90.0 - self.angle + self.yaw)
        e = math.radians(self.elevation)
        return math.cos(t), math.sin(t), math.cos(e), math.sin(e)

    def project(self, p: np.ndarray):
        """World points (centred) -> (sx, sy, depth): float pixel coords and depth in px
        (smaller = nearer)."""
        ct, st, ce, se = self._cs()
        x = p[..., 0] * ct + p[..., 2] * st
        z = -p[..., 0] * st + p[..., 2] * ct
        y = p[..., 1]
        sx = self.ox + x * self.scale
        sy = self.oy - (y * ce - z * se) * self.scale
        return sx, sy, (-y * se - z * ce) * self.scale

    def axes(self):
        """World vectors of one pixel right, one pixel down, and the unit vector to the camera."""
        ct, st, ce, se = self._cs()
        right = np.array([ct, 0.0, st]) / self.scale
        down = -np.array([st * se, ce, -ct * se]) / self.scale
        toward = np.array([-st * ce, se, ct * ce])
        return right, down, toward

    def to_camera(self, n: np.ndarray) -> np.ndarray:
        """World normals -> camera space (x right, y up, z toward the camera)."""
        ct, st, ce, se = self._cs()
        up = np.array([st * se, ce, -ct * se])
        right = np.array([ct, 0.0, st])
        toward = np.array([-st * ce, se, ct * ce])
        return np.stack([n @ right, n @ up, n @ toward], -1)

    def region(self, x0: float, y0: float, size: float, px: int) -> "Camera":
        """The same view, the frame square [x0, x0+size) x [y0, y0+size) blown up to px x px."""
        k = px / size
        return Camera(self.angle, self.elevation, self.scale * k, (self.ox - x0) * k,
                      (self.oy - y0) * k, px, px, self.yaw)


# --------------------------------------------------------------------------- skinned body

class Body:
    """Points spread over the rig's surface, skinned with exact linear blend skinning: a point
    is bind-space position + dense weights over every bone matrix (skin joints and the nodes of
    unskinned meshes), so a frame is one (P, B) @ (B, 16) product."""

    def __init__(self, rig: Rig, points: int = 400_000, seed: int = 0):
        self.rig = rig
        self.slots: list[tuple[str, int]] = []         # ("skin", s) blocks then ("node", n)
        offset, skin_at, node_at = 0, {}, {}
        for p in rig.primitives:
            if p.skin is not None and p.skin not in skin_at:
                skin_at[p.skin] = offset
                offset += len(rig.skins[p.skin][0])
        for p in rig.primitives:
            if p.skin is None and p.node not in node_at:
                node_at[p.node] = offset
                offset += 1
        self.skin_at, self.node_at, self.bones = skin_at, node_at, offset
        surface = sample_surface(rig, points, seed)
        binds, normals, weights = [], [], []
        for p, (face, bary) in zip(rig.primitives, surface.prim):
            if not len(face):
                continue
            corner = p.faces[face]
            pos = (p.positions[corner] * bary[..., None]).sum(1)
            tri = p.positions[p.faces]
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])[face]
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
            w = np.zeros((len(face), self.bones), np.float32)
            if p.skin is None:
                w[:, node_at[p.node]] = 1
            else:
                rows = np.arange(len(face))
                for c in range(3):
                    for k in range(p.joints.shape[1]):
                        np.add.at(w, (rows, skin_at[p.skin] + p.joints[corner[:, c], k]),
                                  bary[:, c] * p.weights[corner[:, c], k])
            binds.append(pos)
            normals.append(n)
            weights.append(w)
        self.bind = np.concatenate(binds).astype(np.float64)
        self.bind_n = np.concatenate(normals)
        self.weights = np.concatenate(weights)
        self.weights /= np.maximum(self.weights.sum(1, keepdims=True), 1e-12)
        self.colours = surface.colours
        rest = self.pose(rig.worlds())[0]
        lo, hi = rest.min(0), rest.max(0)
        self.centre = np.array([(lo[0] + hi[0]) / 2, 0.0, (lo[2] + hi[2]) / 2])

    def mats(self, worlds: np.ndarray) -> np.ndarray:
        """(B, 4, 4) bone matrices of a pose."""
        out = np.zeros((self.bones, 4, 4))
        for s, at in self.skin_at.items():
            joints, ibm = self.rig.skins[s]
            out[at:at + len(joints)] = worlds[joints] @ ibm
        for n, at in self.node_at.items():
            out[at] = worlds[n]
        return out

    @staticmethod
    def blend(weights: np.ndarray, mats: np.ndarray) -> np.ndarray:
        return (weights.astype(np.float64) @ mats.reshape(len(mats), 16)).reshape(-1, 4, 4)

    def pose(self, worlds: np.ndarray, idx=None):
        """(points, normals) in world space (not centred)."""
        b = self.blend(self.weights if idx is None else self.weights[idx], self.mats(worlds))
        v = self.bind if idx is None else self.bind[idx]
        n = self.bind_n if idx is None else self.bind_n[idx]
        pts = np.einsum("pij,pj->pi", b[:, :3, :3], v) + b[:, :3, 3]
        nrm = np.einsum("pij,pj->pi", b[:, :3, :3], n)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        return pts, nrm


@dataclass
class Pose:
    """One animation frame: bone matrices + the offset that keeps a walk on the spot."""
    mats: np.ndarray
    shift: np.ndarray      # subtracted from world positions (centre + drift)
    worlds: np.ndarray = field(default=None, repr=False)   # node world matrices


def clip_poses(body: Body, clips: list[Clip], frames: int, in_place: bool = True
               ) -> list[list[Pose]]:
    rig = body.rig
    out = []
    for clip in clips:
        drift = np.zeros(3)
        if in_place and clip.duration > 0:
            moved = rig.centroid(clip, clip.end) - rig.centroid(clip, clip.start)
            drift = moved * np.array([1.0, 0.0, 1.0]) / clip.duration
        row = []
        for t in clip_times(clip, frames):
            w = rig.worlds(clip, t)
            row.append(Pose(body.mats(w), body.centre + drift * (t - clip.start), w))
        out.append(row)
    return out


def rest_pose(body: Body) -> Pose:
    w = body.rig.worlds()
    return Pose(body.mats(w), body.centre.copy(), w)


def framing(body: Body, poses: list[Pose], names: list[str], elevation: float, width: int,
            height: int, margin: int = 2, sample: int = 30_000) -> list[Camera]:
    """One scale and one ground row for every direction and frame (like render_points): the
    union of everything fits the frame."""
    rng = np.random.default_rng(1)
    idx = rng.choice(len(body.bind), min(sample, len(body.bind)), replace=False)
    unit = [Camera(k * 360.0 / len(names), elevation, 1.0, 0.0, 0.0, width, height)
            for k in range(len(names))]
    lo, hi = np.full(2, np.inf), np.full(2, -np.inf)
    for p in poses:
        b = Body.blend(body.weights[idx], p.mats)
        pts = np.einsum("pij,pj->pi", b[:, :3, :3], body.bind[idx]) + b[:, :3, 3] - p.shift
        for cam in unit:
            sx, sy, _ = cam.project(pts)
            lo = np.minimum(lo, [sx.min(), sy.min()])
            hi = np.maximum(hi, [sx.max(), sy.max()])
    scale = min((width - 2 * margin) / max(hi[0] - lo[0], 1e-9),
                (height - 2 * margin) / max(hi[1] - lo[1], 1e-9))
    ox = width / 2 - scale * (lo[0] + hi[0]) / 2
    oy = height - margin - scale * hi[1]
    return [Camera(c.angle, elevation, scale, ox, oy, width, height) for c in unit]


# --------------------------------------------------------------------------- rest view

@dataclass
class RestView:
    """The rest pose seen by one camera at frame size: the front surface per pixel."""
    mask: np.ndarray       # (H, W) bool
    pos: np.ndarray        # (H, W, 3) world position (not centred)
    normal: np.ndarray     # (H, W, 3) world normal
    weights: np.ndarray    # (H, W, B)
    depth: np.ndarray      # (H, W) px
    point_pix: np.ndarray  # (P,) pixel of every body point (-1 = off the frame)
    point_front: np.ndarray  # (P,) the point is on the front surface


def rest_view(body: Body, cam: Camera, rest: Pose, pts=None, nrm=None, smooth: int = 1) -> RestView:
    """Front surface per pixel. Attributes come from the points within TIGHT_EPS of the nearest
    (one surface: an arm in front of a cape doesn't mix their weights, which would shred the
    drawing when they part); weights are then smoothed between neighbours that already move
    alike, and pixels whose main bone differs from all around them take their neighbours'."""
    if pts is None:
        pts, nrm = body.pose(rest.worlds)
    W, H = cam.width, cam.height
    sx, sy, depth = cam.project(pts - rest.shift)
    ix, iy = np.floor(sx).astype(np.int64), np.floor(sy).astype(np.int64)
    ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    lin = np.where(ok, iy * W + ix, -1)
    zmin = np.full(W * H, np.inf)
    np.minimum.at(zmin, lin[ok], depth[ok])
    near = zmin[np.maximum(lin, 0)]
    front = ok & (depth <= near + DEPTH_EPS)
    count = np.bincount(lin[front], minlength=W * H)
    mask = count >= 2
    tight = ok & (depth <= near + TIGHT_EPS)
    li = lin[tight]
    c = np.maximum(np.bincount(li, minlength=W * H), 1).astype(np.float64)

    def avg(values):
        return np.stack([np.bincount(li, values[tight, k], W * H) / c
                         for k in range(values.shape[1])], -1)
    pos = avg(pts).reshape(H, W, 3)
    normal = avg(nrm)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    weights = avg(body.weights).reshape(H, W, -1)
    dmap = np.bincount(li, depth[tight], W * H) / c
    view = RestView(mask.reshape(H, W), pos, normal.reshape(H, W, 3), weights,
                    dmap.reshape(H, W), lin, front)
    for _ in range(2):
        view.weights = _bone_islands(view)
    for _ in range(smooth):
        view.weights = _smooth_weights(view)
    return view


def _shifts(H, W):
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                yield ((slice(max(0, -dy), H + min(0, -dy)), slice(max(0, -dx), W + min(0, -dx))),
                       (slice(max(0, dy), H + min(0, dy)), slice(max(0, dx), W + min(0, dx))))


def _bone_islands(v: RestView) -> np.ndarray:
    """Pixels whose main bone at most one neighbour shares take the weights of the neighbours'
    most common main bone (stray single pixels would fly off on their own)."""
    H, W = v.mask.shape
    bone = np.where(v.mask, v.weights.argmax(-1), -1)
    nb = v.weights.shape[-1]
    same = np.zeros((H, W), np.int32)
    votes = np.zeros((H, W, nb), np.int32)
    for dst, src in _shifts(H, W):
        same[dst] += (bone[src] == bone[dst]) & (bone[src] >= 0)
        b = bone[src]
        yy, xx = np.nonzero(b >= 0)
        votes[dst][yy, xx, b[yy, xx]] += 1
    lonely = v.mask & (same <= 1) & (votes.max(-1) >= 3)
    if not lonely.any():
        return v.weights
    want = votes.argmax(-1)
    acc = np.zeros_like(v.weights)
    n = np.zeros((H, W))
    for dst, src in _shifts(H, W):
        use = (bone[src] == want[dst])
        acc[dst] += np.where(use[..., None], v.weights[src], 0)
        n[dst] += use
    out = v.weights.copy()
    out[lonely] = acc[lonely] / np.maximum(n[lonely], 1)[:, None]
    return out


def _smooth_weights(v: RestView) -> np.ndarray:
    """3x3 average among neighbours on the same surface whose weights are already close."""
    H, W = v.mask.shape
    acc = v.weights.copy()
    n = np.ones((H, W))
    for dst, src in _shifts(H, W):
        same = (v.mask[dst] & v.mask[src]
                & (np.abs(v.depth[dst] - v.depth[src]) < DEPTH_EPS)
                & (np.abs(v.weights[dst] - v.weights[src]).sum(-1) < 0.6))
        acc[dst] += np.where(same[..., None], v.weights[src], 0)
        n[dst] += same
    w = acc / n[..., None]
    return np.where(v.mask[..., None], w, v.weights)


# --------------------------------------------------------------------------- drawn layer

@dataclass
class Drawn:
    """A key drawing glued to the skeleton: per drawn pixel the bind-space centre, the bind-space
    vectors of one pixel right / down, the bind-space normal and the skin weights."""
    cam: Camera
    indices: np.ndarray        # (H, W) key drawing palette indices, -1 = empty
    palette: np.ndarray
    ys: np.ndarray
    xs: np.ndarray
    weights: np.ndarray        # (N, B)
    centre: np.ndarray         # (N, 3) bind space
    right: np.ndarray          # (N, 3)
    down: np.ndarray           # (N, 3)
    normal: np.ndarray         # (N, 3)
    hidden: np.ndarray = field(default=None, repr=False)   # body points for disocclusion
    hidden_idx: np.ndarray = field(default=None, repr=False)  # their palette indices


def glue(body: Body, view: RestView, cam: Camera, rest: Pose, indices: np.ndarray,
         palette: np.ndarray) -> Drawn:
    """Each drawn pixel takes the 3D point, normal and weights of its rest-view pixel; pixels
    the drawing added outside the model take the nearest model pixel's, moved in the screen
    plane to their own place (so the rest pose lands every pixel exactly where it was drawn)."""
    from scipy.ndimage import distance_transform_edt
    ys, xs = np.nonzero(indices >= 0)
    if not view.mask.any():
        raise ValueError("The model is not in the frame")
    _, (ny, nx) = distance_transform_edt(~view.mask, return_indices=True)
    ny, nx = ny[ys, xs], nx[ys, xs]
    P = view.pos[ny, nx]
    sx, sy, _ = cam.project(P - rest.shift)
    right, down, _ = cam.axes()
    P = P + (xs + 0.5 - sx)[:, None] * right + (ys + 0.5 - sy)[:, None] * down
    weights = view.weights[ny, nx]
    b = Body.blend(weights, rest.mats)
    inv = np.linalg.inv(b)
    centre = np.einsum("pij,pj->pi", inv[:, :3, :3], P) + inv[:, :3, 3]
    return Drawn(cam, indices, palette, ys, xs, weights.astype(np.float32), centre,
                 inv[:, :3, :3] @ right, inv[:, :3, :3] @ down,
                 np.einsum("pij,pj->pi", inv[:, :3, :3], view.normal[ny, nx]))


def add_hidden(drawn: Drawn, body: Body, view: RestView, rest: Pose, point_idx: np.ndarray,
               pts: np.ndarray | None = None, max_points: int = 160_000, seed: int = 0) -> None:
    """Body points behind the drawing at rest (inside its silhouette, behind its surface), with
    palette indices (point_idx, -1 = unknown): they fill parts that come out from behind."""
    H, W = drawn.indices.shape
    pix = view.point_pix
    flat_idx = drawn.indices.reshape(-1)
    inside = (pix >= 0) & (flat_idx[np.maximum(pix, 0)] >= 0)
    drawn_depth = np.full(H * W, np.inf)
    _, _, d = drawn.cam.project(_pose_centres(drawn, rest))
    drawn_depth[drawn.ys * W + drawn.xs] = d
    if pts is None:
        pts, _ = body.pose(rest.worlds)
    _, _, depth = drawn.cam.project(pts - rest.shift)
    hidden = inside & (depth > drawn_depth[np.maximum(pix, 0)] + DEPTH_EPS) & (point_idx >= 0)
    idx = np.nonzero(hidden)[0]
    if len(idx) > max_points:
        idx = np.sort(np.random.default_rng(seed).choice(idx, max_points, replace=False))
    drawn.hidden = idx
    drawn.hidden_idx = point_idx[idx]


def _pose_centres(drawn: Drawn, pose: Pose) -> np.ndarray:
    b = Body.blend(drawn.weights, pose.mats)
    return np.einsum("pij,pj->pi", b[:, :3, :3], drawn.centre) + b[:, :3, 3] - pose.shift


def _first_per(lin: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Index (into lin) of the smallest key per distinct lin value."""
    order = np.lexsort((key, lin))
    first = np.ones(len(order), bool)
    first[1:] = lin[order][1:] != lin[order][:-1]
    return order[first]


def bend(drawn: Drawn, body: Body, pose: Pose, *, fill: bool = True):
    """The key drawing in one pose -> (indices (H, W), camera-space normals (H, W, 3))."""
    cam = drawn.cam
    W, H = cam.width, cam.height
    b = Body.blend(drawn.weights, pose.mats)
    R = b[:, :3, :3]
    c = np.einsum("pij,pj->pi", R, drawn.centre) + b[:, :3, 3] - pose.shift
    r = np.einsum("pij,pj->pi", R, drawn.right)
    d = np.einsum("pij,pj->pi", R, drawn.down)
    off = (np.arange(SS) + 0.5) / SS - 0.5
    ou, ov = np.meshgrid(off, off)
    ou, ov = ou.reshape(-1), ov.reshape(-1)
    samples = c[:, None] + ou[None, :, None] * r[:, None] + ov[None, :, None] * d[:, None]
    sx, sy, depth = cam.project(samples.reshape(-1, 3))
    src = np.repeat(np.arange(len(c)), SS * SS)
    ix, iy = np.floor(sx).astype(np.int64), np.floor(sy).astype(np.int64)
    ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    sx, sy, depth, src, ix, iy = sx[ok], sy[ok], depth[ok], src[ok], ix[ok], iy[ok]
    lin = iy * W + ix
    zmin = np.full(W * H, np.inf)
    np.minimum.at(zmin, lin, depth)
    front = depth <= zmin[lin] + DEPTH_EPS
    lin_f, src_f = lin[front], src[front]
    dist = (sx[front] - ix[front] - 0.5) ** 2 + (sy[front] - iy[front] - 0.5) ** 2
    count = np.bincount(lin_f, minlength=W * H)
    pick = _first_per(lin_f, dist)
    out = np.full(W * H, -1, np.int32)
    chosen = np.full(W * H, -1, np.int64)
    chosen[lin_f[pick]] = src_f[pick]
    on = count >= MIN_COVER
    # thin spots of a stretched part: some samples, most 4-neighbours drawn
    some = (count > 0) & ~on
    if some.any():
        grid = on.reshape(H, W)
        nb = np.zeros((H, W), np.int32)
        nb[1:] += grid[:-1]
        nb[:-1] += grid[1:]
        nb[:, 1:] += grid[:, :-1]
        nb[:, :-1] += grid[:, 1:]
        on |= some & (nb.reshape(-1) >= 3)
    keep = on & (chosen >= 0)
    flat_src = drawn.indices[drawn.ys, drawn.xs]
    out[keep] = flat_src[chosen[keep]]
    zd = np.where(keep, zmin, np.inf)
    nrm = np.zeros((W * H, 3))
    n_all = np.einsum("pij,pj->pi", R, drawn.normal)
    nrm[keep] = n_all[chosen[keep]]
    if fill and drawn.hidden is not None and len(drawn.hidden):
        pts, pn = _pose_points(body, pose.mats, drawn.hidden)
        fx, fy, fz = cam.project(pts - pose.shift)
        jx, jy = np.floor(fx).astype(np.int64), np.floor(fy).astype(np.int64)
        okf = (jx >= 0) & (jx < W) & (jy >= 0) & (jy < H)
        fl = jy[okf] * W + jx[okf]
        fz, fi, pn = fz[okf], drawn.hidden_idx[okf], pn[okf]
        fmin = np.full(W * H, np.inf)
        np.minimum.at(fmin, fl, fz)
        ff = fz <= fmin[fl] + DEPTH_EPS
        fl, fi, pn = fl[ff], fi[ff], pn[ff]
        fcount = np.bincount(fl, minlength=W * H)
        k = int(len(drawn.palette)) + 1
        votes = np.bincount(fl * k + fi, minlength=W * H * k).reshape(W * H, k)
        fidx = votes.argmax(1)
        use = (fcount >= 2) & ((out < 0) | (fmin < zd - OCCLUDE_EPS))
        out[use] = fidx[use]
        if use.any():
            nsum = np.stack([np.bincount(fl, pn[:, j], W * H) for j in range(3)], -1)
            nrm[use] = nsum[use]
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    return out.reshape(H, W), cam.to_camera(nrm).reshape(H, W, 3)


def _pose_points(body: Body, mats: np.ndarray, idx: np.ndarray):
    b = Body.blend(body.weights[idx], mats)
    pts = np.einsum("pij,pj->pi", b[:, :3, :3], body.bind[idx]) + b[:, :3, 3]
    nrm = np.einsum("pij,pj->pi", b[:, :3, :3], body.bind_n[idx])
    return pts, nrm


# --------------------------------------------------------------------------- key drawings

def init_picture(rig: Rig, cam: Camera, region: tuple[float, float, float], pose: Pose,
                 px: int = 384,
                 points: int = 2_000_000, light: float = 0.25, seed: int = 0,
                 out: int = 1024, blocky: bool = False, cloud=None) -> Image.Image:
    """The rest pose big and nearly flat, as the SDXL init: the frame square `region` (x0, y0,
    size in frame px) at out x out, white background. blocky: already on the frame's pixel grid
    (each frame pixel an out/size block, hard edges), so SDXL draws pixels on that grid and the
    drawing comes back to the frame without resampling (crisper than a smooth init)."""
    from .turnaround import _splat
    pts, nrm, cols = cloud if cloud is not None else surface_cloud(rig, pose, points, seed)
    big = cam.region(*region, px)
    centre = pose.shift
    ct, st, ce, se = big._cs()
    light_dir = np.array([-0.45, 0.6, 0.66])
    light_dir /= np.linalg.norm(light_dir)
    rgba = _splat((pts - centre).astype(np.float32), nrm.astype(np.float32),
                  cols.astype(np.float32), ct, st, ce, se, big.scale, big.ox, big.oy,
                  px, light, light_dir)
    if blocky:
        import cv2
        size = int(region[2])
        a = rgba[..., 3:].astype(np.float32) / 255
        pre = cv2.resize(rgba[..., :3].astype(np.float32) * a, (size, size),
                         interpolation=cv2.INTER_AREA)
        cov = cv2.resize(a[..., 0], (size, size), interpolation=cv2.INTER_AREA)
        rgb = pre / np.maximum(cov, 1e-6)[..., None]
        small = np.dstack([np.clip(rgb, 0, 255), np.where(cov >= 0.5, 255, 0)]).astype(np.uint8)
        img = Image.fromarray(small, "RGBA").resize((out, out), Image.NEAREST)
    else:
        img = Image.fromarray(rgba, "RGBA").resize((out, out), Image.LANCZOS)
    canvas = Image.new("RGBA", (out, out), (255, 255, 255, 255))
    canvas.alpha_composite(img)
    return canvas.convert("RGB")


def surface_cloud(rig: Rig, pose: Pose, points: int = 2_000_000, seed: int = 0):
    """(points, normals, colours) of the model in a pose, dense enough for the init pictures."""
    surface = sample_surface(rig, points, seed)
    pts, nrm = surface_pose(rig, surface, pose.worlds)
    return pts, nrm, surface.colours


def key_region(view: RestView, pad: int = 6) -> tuple[int, int, int]:
    """A square of the frame around the rest pose (x0, y0, size) for the key drawing."""
    ys, xs = np.nonzero(view.mask)
    size = int(max(xs.max() - xs.min(), ys.max() - ys.min())) + 1 + 2 * pad
    cx, cy = (xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2
    return int(round(cx - size / 2)), int(round(cy - size / 2)), size


def grid_region(view: RestView, target: int = 1024, pad: int = 6) -> tuple[tuple[int, int, int], int]:
    """key_region grown so the drawing is a whole number k of pixels per frame pixel and a
    multiple of 8 (SDXL sizes): ((x0, y0, size), k)."""
    x0, y0, size = key_region(view, pad)
    k = max(1, round(target / size))
    while (size * k) % 8:
        size += 1
    return (x0, y0, size), k


def drawing_to_frame(drawing: Image.Image, region, frame: tuple[int, int], settings,
                     patch: int | None = None):
    """A drawn picture of the region -> (prepared RGB, mask) on the frame grid, cut out."""
    from .model3d import cut_out
    from .objects import main_subject
    from .pipeline import Prepared, prepare
    from .settings import Settings
    rgba = main_subject(cut_out(drawing))
    x0, y0, size = region
    s = Settings(**{**settings.to_dict(), "width": size, "height": size, "fit": "stretch",
                    "crop_to_subject": False, "outline": False, "remove_background": False,
                    # on the drawing's own grid (median/no-thickening tried: softer, less crisp)
                    **({"patch": patch} if patch else {})})
    prep = prepare(rgba, s)
    W, H = frame
    rgb = np.zeros((H, W, 3), np.uint8)
    mask = np.zeros((H, W), bool)
    ys, xs = np.mgrid[0:size, 0:size]
    fy, fx = ys + y0, xs + x0
    ok = (fy >= 0) & (fy < H) & (fx >= 0) & (fx < W)
    rgb[fy[ok], fx[ok]] = prep.rgb[ys[ok], xs[ok]]
    mask[fy[ok], fx[ok]] = prep.mask[ys[ok], xs[ok]]
    return Prepared(rgb, mask)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum()) / max(1, int((a | b).sum()))


def _say(progress):
    return progress or (lambda t: None)


def _gpu_temperature() -> int | None:
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=10)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001 - no NVIDIA tools: no pacing
        return None


def pace(limit: int = 80, abort: int = 87, rest: float = 4.0, say=None) -> None:
    """Wait for the GPU to cool below `limit` C between drawings; stop above `abort`."""
    t = _gpu_temperature()
    if t is None:
        return
    if t > abort:
        raise RuntimeError(f"GPU at {t} C (over {abort} C): stopped to let the laptop cool")
    while t is not None and t > limit:
        (say or (lambda m: None))(f"GPU at {t} C, resting…")
        time.sleep(rest)
        t = _gpu_temperature()


@dataclass
class KeySet:
    names: list[str]
    sprites: list[Sprite]          # key drawings on the frame grid, one palette
    ious: list[float]
    seeds: list[int]
    mirrored: list[bool]


def draw_keys(rig: Rig, body: Body, cams: list[Camera], views: list[RestView], rest: Pose,
              prompt: str, settings, *, seed: int = 1234, denoise: float = 0.6,
              style=None, client=None, folder: Path | None = None, reuse: bool = True,
              min_iou: float = 0.72, retries: int = 1, mirror: bool | None = None,
              blocky: bool = True, progress=None) -> KeySet:
    """One pixel-art drawing per direction over the model's rest pose (see the module notes).
    folder: keeps init_<d>.png / drawn_<d>.png; reuse = draw only directions not there yet.
    mirror: None = mirror W-side drawings from the E side when the model looks the same
    mirrored (silhouette IoU >= 0.93), True/False = always/never."""
    from .assets import project_style, with_style
    from .generate import generate_images
    from .pipeline import finish, resolve_palette
    from .pixelclean import clean_sprite
    from .settings import Settings
    say = _say(progress)
    names = [direction_names(len(cams))[k] for k in range(len(cams))]
    W, H = cams[0].width, cams[0].height
    st = style if style is not None else project_style()
    folder = Path(folder) if folder else None
    if folder:
        folder.mkdir(parents=True, exist_ok=True)
    preps, ious, seeds, mirrored = [None] * len(names), [0.0] * len(names), [seed] * len(names), \
        [False] * len(names)
    n = len(names)

    def twin(k):                          # the E-side direction this W-side one mirrors
        a = (180.0 - cams[k].angle) % 360.0
        j = int(round(a / (360.0 / n))) % n
        return j if j != k else None
    cloud = None
    order = sorted(range(n), key=lambda k: (cams[k].angle > 90.0 and cams[k].angle < 270.0, k))
    for k in order:
        d = names[k]
        region, kpx = grid_region(views[k]) if blocky else (key_region(views[k]), None)
        gen = region[2] * kpx if blocky else 1024
        j = twin(k)
        if (90.0 < cams[k].angle < 270.0) and j is not None and preps[j] is not None:
            flip = np.fliplr(views[j].mask)
            same = iou(flip, views[k].mask)
            if mirror or (mirror is None and same >= 0.93):
                rgb = np.fliplr(preps[j].rgb).copy()
                m = np.fliplr(preps[j].mask).copy()
                # mirrored about the frame centre line; move it onto this view's axis
                shift = int(round(2 * cams[k].ox - W))
                rgb, m = np.roll(rgb, shift, 1), np.roll(m, shift, 1)
                from .pipeline import Prepared
                preps[k] = Prepared(rgb, m)
                ious[k] = iou(m, views[k].mask)
                mirrored[k] = True
                say(f"{d}: mirrored from {names[j]} (IoU {ious[k]:.2f})")
                continue
        best = None
        for attempt in range(retries + 1):
            s_k = seed + 1000 * attempt
            path = folder / (f"drawn_{d}.png" if attempt == 0 else f"drawn_{d}_{attempt}.png") \
                if folder else None
            if path is not None and reuse and path.exists():
                drawing = Image.open(path).convert("RGB")
            else:
                if cloud is None:
                    cloud = surface_cloud(rig, rest)
                init = init_picture(rig, cams[k], region, rest, out=gen, blocky=blocky,
                                    cloud=cloud)
                if folder:
                    init.save(folder / f"init_{d}.png")
                view, view_neg = direction_words(d)
                text, neg = with_style(KEY_PROMPT.format(prompt=prompt, view=view),
                                       f"{KEY_NEGATIVE}, {view_neg}", st, "character")
                if client is None:             # the real GPU: let it cool between drawings
                    pace(say=say)
                say(f"Drawing the knight facing {d} (seed {s_k})…".replace("the knight", "it"))
                (drawing, _), = generate_images(text, neg, client=client, seed=s_k, count=1,
                                                init_image=init, denoise=denoise, width=gen,
                                                height=gen)
                if path is not None:
                    drawing.save(path)
            prep = drawing_to_frame(drawing, region, (W, H), settings, kpx)
            score = iou(prep.mask, views[k].mask)
            say(f"{d}: silhouette IoU {score:.2f}")
            if best is None or score > best[0]:
                best = (score, prep, s_k)
            if score >= min_iou:
                break
        ious[k], preps[k], seeds[k] = best
    s = Settings(**{**settings.to_dict(), "outline": False})
    pixels = np.concatenate([p.rgb[p.mask] for p in preps if p.mask.any()])
    palette = resolve_palette(s, pixels)
    outline = settings.outline_style if settings.outline else "none"
    sprites = []
    for p in preps:
        sp = finish(p, s, palette)
        sp = clean_sprite(sp, outline)
        sprites.append(sp)
    return KeySet(names, sprites, ious, seeds, mirrored)


# --------------------------------------------------------------------------- whole character

def _point_colours(body: Body, views: list[RestView], keys: KeySet, key_pose: Pose
                   ) -> list[np.ndarray]:
    """Per direction: a palette index for every body point, from the drawing of the nearest
    direction that sees it (else from the nearest point that has one)."""
    from scipy.spatial import cKDTree
    n = len(views)
    seen = []
    for v, sp in zip(views, keys.sprites):
        flat = sp.indices.reshape(-1)
        idx = np.full(len(body.bind), -1, np.int32)
        ok = v.point_front & (v.point_pix >= 0)
        idx[ok] = flat[v.point_pix[ok]]
        seen.append(idx)
    rest_pts, _ = body.pose(key_pose.worlds)
    out = []
    for k in range(n):
        idx = seen[k].copy()
        for step in sorted(range(1, n), key=lambda s_: (min(s_, n - s_), s_)):
            j = (k + step) % n
            need = idx < 0
            idx[need] = seen[j][need]
        missing = idx < 0
        if missing.any() and (~missing).any():
            tree = cKDTree(rest_pts[~missing])
            _, near = tree.query(rest_pts[missing])
            idx[missing] = idx[~missing][near]
        out.append(idx)
    return out


def drawn_animation(rigged, prompt: str, settings, *, directions: int = 8, frames: int = 8,
                    animations: list[str] | None = None, elevation: float = 30.0,
                    seed: int = 1234, denoise: float = 0.6, style=None, client=None,
                    folder=None, reuse: bool = True, points: int = 400_000, clean: bool = True,
                    normals: bool = True, mirror: bool | None = None, key_pose: str = "idle",
                    blocky: bool = True, retries: int = 1, min_iou: float = 0.72,
                    progress=None) -> tuple[AnimatedSet, dict]:
    """A rigged character -> AnimatedSet whose every frame is a key drawing bent by the
    skeleton. settings: converter settings at the frame size (width x height, colours,
    outline). Returns (set, info: keys, ious, seeds, seconds)."""
    from .normals import decode, encode, face_camera, from_drawing
    from .pixelclean import clean_sprite
    say = _say(progress)
    t0 = time.time()
    rig = load_rig([rigged])
    clips = select_clips(rig, animations)
    names = direction_names(directions)
    W, H = settings.width, settings.height
    say("Reading the skeleton…")
    body = Body(rig, points)
    poses = clip_poses(body, clips, frames)
    rest = rest_pose(body)
    names_c = [c.name for c in clips]
    if key_pose in names_c:                    # draw the first frame of e.g. idle: most frames
        rest = poses[names_c.index(key_pose)][0]   # are near it, so the drawing bends least
    cams = None
    saved = Path(folder) / "keys.json" if folder else None
    if reuse and saved is not None and saved.exists():   # keep the scale the keys were drawn at
        try:
            cam = json.loads(saved.read_text(encoding="utf-8")).get("camera") or {}
        except ValueError:
            cam = {}
        if (cam.get("width"), cam.get("height"), cam.get("directions"), cam.get("elevation"),
                cam.get("key_pose")) == (W, H, len(names), elevation, key_pose):
            cams = [Camera(k * 360.0 / len(names), elevation, cam["scale"], cam["ox"], cam["oy"],
                           W, H) for k in range(len(names))]
    if cams is None:
        cams = framing(body, [p for row in poses for p in row] + [rest], names, elevation, W, H)
    pts, nrm = body.pose(rest.worlds)
    views = [rest_view(body, c, rest, pts, nrm) for c in cams]
    t_keys = time.time()
    keys = draw_keys(rig, body, cams, views, rest, prompt, settings, seed=seed, denoise=denoise,
                     style=style, client=client, folder=folder, reuse=reuse, mirror=mirror,
                     blocky=blocky, retries=retries, min_iou=min_iou, progress=say)
    t_bend = time.time()
    if folder:
        for d, sp in zip(keys.names, keys.sprites):
            Image.fromarray(sp.rgba).save(Path(folder) / f"key_{d}.png")
    colours = _point_colours(body, views, keys, rest)
    palette = keys.sprites[0].palette
    outline = settings.outline_style if settings.outline else "none"
    say("Bending the drawings for every frame…")
    sprites = [[[None] * frames for _ in names] for _ in clips]
    for k, cam in enumerate(cams):
        drawn = glue(body, views[k], cam, rest, keys.sprites[k].indices, palette)
        add_hidden(drawn, body, views[k], rest, colours[k], pts)
        for a, row in enumerate(poses):
            for f, pose in enumerate(row):
                idx, ncam = bend(drawn, body, pose)
                if clean:
                    idx = drop_fragments(idx)
                sp = _sprite(idx, palette, ncam)
                if clean:
                    sp = clean_sprite(sp, outline)
                    gone = (drop_fragments(sp.indices, fill_holes=False) < 0) & (sp.indices >= 0)
                    if gone.any():                 # clean-up can cut a thin bridge
                        sp.indices[gone] = -1
                        sp.rgba[gone] = 0
                        if sp.normal is not None:
                            sp.normal[gone] = 0
                if normals:
                    nd = decode(from_drawing(sp.rgba)[..., :3])
                    n3 = face_camera(decode(sp.normal[..., :3]) if sp.normal is not None
                                     else ncam)
                    mix = 0.65 * n3 + 0.35 * nd
                    mix /= np.maximum(np.linalg.norm(mix, axis=-1, keepdims=True), 1e-12)
                    nm = np.zeros(idx.shape + (4,), np.uint8)
                    on = sp.indices >= 0
                    nm[..., :3] = encode(face_camera(mix))
                    nm[..., 3] = np.where(on, 255, 0)
                    nm[~on] = 0
                    sp.normal = nm
                sprites[a][k][f] = sp
    actions = [Action(c.name, c.loop, max(16, int(round(1000 * c.duration
                                                         / max(1, frames if c.loop else frames - 1)))))
               for c in clips]
    info = {"keys": keys, "seconds": {"setup": t_keys - t0, "drawings": t_bend - t_keys,
                                      "bending": time.time() - t_bend},
            "ious": dict(zip(keys.names, keys.ious)), "seeds": dict(zip(keys.names, keys.seeds)),
            "mirrored": [n for n, m in zip(keys.names, keys.mirrored) if m],
            "camera": {"scale": cams[0].scale, "ox": cams[0].ox, "oy": cams[0].oy, "width": W,
                       "height": H, "directions": len(names), "elevation": elevation,
                       "key_pose": key_pose}}
    return AnimatedSet(actions, names, sprites), info


def drop_fragments(idx: np.ndarray, share: float = 0.004, least: int = 6,
                   fill_holes: bool = True) -> np.ndarray:
    """Remove bits torn off the body by bending (pieces smaller than `share` of the sprite or
    `least` px, 8-connected) and fill single-pixel holes from their neighbours."""
    import cv2
    on = (idx >= 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(on, connectivity=8)
    if n > 2:
        small = stats[:, cv2.CC_STAT_AREA] < max(least, share * on.sum())
        small[0] = False
        idx = np.where(small[labels], -1, idx)
    H, W = idx.shape
    pad = np.pad(idx, 1, constant_values=-1)
    nb = [pad[1 + dy:1 + dy + H, 1 + dx:1 + dx + W] for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))]
    hole = (idx < 0) & np.all([b >= 0 for b in nb], 0) & fill_holes
    if hole.any():
        stack = np.stack(nb, -1)[hole]
        idx = idx.copy()
        idx[hole] = [np.bincount(r).argmax() for r in stack]
    return idx


def _sprite(idx: np.ndarray, palette: np.ndarray, ncam: np.ndarray | None = None) -> Sprite:
    from .normals import encode, face_camera
    on = idx >= 0
    rgba = np.zeros(idx.shape + (4,), np.uint8)
    rgba[on, :3] = palette[idx[on]]
    rgba[on, 3] = 255
    normal = None
    if ncam is not None:
        normal = np.zeros(idx.shape + (4,), np.uint8)
        normal[..., :3] = encode(face_camera(ncam))
        normal[..., 3] = np.where(on, 255, 0)
        normal[~on] = 0
    return Sprite(rgba, idx.astype(np.int32), palette, normal)


def drawn_sheets(character_or_rigged, prompt: str | None = None, *, out_png=None,
                 directions: int | None = None, clips: list[str] | None = None, frames: int = 8,
                 seed: int = 1234, denoise: float = 0.6, kind: str | None = None,
                 blocky: bool = True,
                 size: str | None = None, client=None, keys_folder=None, reuse: bool = True,
                 progress=None) -> tuple[AnimatedSet, dict, list[Path]]:
    """A library Character (or a rigged .glb + prompt) -> drawn sheets at the library scale of
    the CURRENT project (frame size, colours, outline, elevation, directions), written with
    anim3d.write_animated (sheet.png/json/aseprite/tres/_facing.gd + _n + _glow) to out_png
    (default: the character's sheet.png). keys_folder: where the key drawings are kept
    (default: <character>/keys)."""
    from .library import Character, frame_size, style as lib_style
    from .settings import settings_for
    st = lib_style()
    if isinstance(character_or_rigged, Character):
        ch = character_or_rigged
        rigged, prompt = ch.rigged, prompt or ch.prompt
        kind, size = kind or ch.kind, size or ch.size
        out_png = out_png or ch.sheet
        keys_folder = keys_folder or ch.folder / "keys"
    else:
        rigged = Path(character_or_rigged)
        kind, size = kind or "person", size or "normal"
        out_png = out_png or rigged.with_name("sheet.png")
        keys_folder = keys_folder or rigged.parent / "keys"
    if not prompt:
        raise ValueError("Drawn sheets need the character's words (prompt).")
    w, h = frame_size(kind, size, st)
    colours = st.colours if kind == "person" else max(8, round(st.colours * 2 / 3))
    s = settings_for(kind if kind == "animal" else "person", width=w, height=h, colors=colours)
    if st.outline == "none":
        s.outline = False
    elif st.outline in ("selout", "solid"):
        s.outline_style = st.outline
    if st.palette != "auto":
        s.palette = st.palette
    anim, info = drawn_animation(rigged, prompt, s, directions=directions or st.directions,
                                 frames=frames, animations=clips, elevation=st.elevation,
                                 seed=seed, denoise=denoise, style=st, client=client,
                                 folder=keys_folder, reuse=reuse, blocky=blocky,
                                 progress=progress)
    written = write_animated(anim, out_png, aseprite=True, godot=True, normals=True)
    meta = {k: v for k, v in info.items() if k != "keys"}
    Path(keys_folder, "keys.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return anim, info, written

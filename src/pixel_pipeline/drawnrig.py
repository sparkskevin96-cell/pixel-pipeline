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
# reference pairing: the drawn reference on the left, the new view painted on the right
PAIR_PROMPT = ("character turnaround sheet, {n} views of the same character side by side, identical "
               "design, {same}, same colours, same proportions, pixel art sprite of {prompt}, "
               "full body, standing, {views}, seen from slightly above, plain white background, "
               "game asset, clean outline, crisp readable pixel shapes")
PAIR_NEGATIVE = ("different characters, different outfit, different colours, mismatched design, "
                 "cropped, out of frame, cut off, busy background, scenery, cast shadow, ground, "
                 "frame, border, text, watermark, 3d render, photo, blurry, smooth gradient")
PANEL_WORDS = {"S": "front view", "E": "side view facing right", "W": "side view facing left",
               "N": "back view"}
SAME_WORDS = {"person": "same outfit and armour", "animal": "same fur and markings"}
# what the image model likes to add whatever the model holds: kept out unless the model has it
SHIELD_WORDS = ("shield", "buckler")
ANIMAL_ABSENT = ("saddle", "rider", "harness", "collar", "clothes")
# things a character holds or wears that the image model happily adds when the words name them;
# the init picture already shows the ones the 3D model has
HELD = ("shield", "buckler", "sword", "greatsword", "bow", "crossbow", "quiver", "staff", "wand",
        "axe", "spear", "dagger", "knife", "hammer", "mace", "lance", "halberd", "book", "tome",
        "lantern", "torch", "scythe", "whip", "flail", "club", "orb")


def model_words(prompt: str, held=HELD) -> str:
    """The prompt without held things ("..., sword and shield" -> "..."): the drawing takes those
    from the model's picture, so words can't add a shield the model doesn't have."""
    import re
    nouns = "|".join(held)
    pat = re.compile(r"(\s*,\s*|\s+and\s+|\s+with\s+|\s+holding\s+|\s+wielding\s+)"
                     r"(an?\s+|the\s+|his\s+|her\s+)?([\w-]+\s+)?(?:" + nouns + r")s?\b", re.I)
    out = prompt
    while True:
        new = pat.sub("", out)
        if new == out:
            break
        out = new
    return re.sub(r"\s+", " ", out).strip(" ,")


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
    colour: np.ndarray = field(default=None, repr=False)   # (H, W, 3) the model's own colours


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
                    dmap.reshape(H, W), lin, front,
                    avg(body.colours.astype(np.float64)).reshape(H, W, 3))
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


def bone_names(body: Body) -> list[str]:
    """The joint (or node) name of every bone slot."""
    out = [""] * body.bones
    for s, at in body.skin_at.items():
        for i, j in enumerate(body.rig.skins[s][0]):
            out[at + i] = body.rig.names[j]
    for nd, at in body.node_at.items():
        out[at] = body.rig.names[nd]
    return out


def colour_segments(rgb: np.ndarray, mask: np.ndarray, step: float = 14.0) -> np.ndarray:
    """Labels of the drawing's colour regions: 4-connected pixels whose Lab colours differ by
    less than `step` (a pixel-art sword blade is one region, its outline cuts it from the leg
    behind it). -1 = empty."""
    from .color import rgb_to_lab
    H, W = mask.shape
    lab = rgb_to_lab(rgb.astype(np.float32))
    lbl = np.where(mask, np.arange(H * W).reshape(H, W), -1)
    right = mask[:, 1:] & mask[:, :-1] & (np.linalg.norm(lab[:, 1:] - lab[:, :-1], axis=-1) < step)
    down = mask[1:] & mask[:-1] & (np.linalg.norm(lab[1:] - lab[:-1], axis=-1) < step)
    big = H * W + 1
    for _ in range(H * W):
        cur = np.where(lbl >= 0, lbl, big)
        new = cur.copy()
        new[:, 1:] = np.where(right, np.minimum(new[:, 1:], cur[:, :-1]), new[:, 1:])
        new[:, :-1] = np.where(right, np.minimum(new[:, :-1], cur[:, 1:]), new[:, :-1])
        new[1:] = np.where(down, np.minimum(new[1:], cur[:-1]), new[1:])
        new[:-1] = np.where(down, np.minimum(new[:-1], cur[1:]), new[:-1])
        new = np.where(mask, new, -1)
        if np.array_equal(new, lbl):
            break
        lbl = new
    return lbl


def source_pixels(view: RestView, rgb: np.ndarray | None, mask: np.ndarray, held=(),
                  arm=(), reach: int = 3, held_reach: int = 4, colour_gap: float = 20.0):
    """Which model pixel each drawn pixel takes its 3D point and skin weights from: (ny, nx).
    Plain: its own pixel (the nearest model pixel for pixels drawn outside the model).
    With the drawing's colours (rgb):
      - colour: a pixel whose colour is far from the model's under it takes the best-matching
        model pixel within `reach` px (SDXL moved an edge a pixel or two: the cape stays cape);
      - held things: the drawing's colour regions (colour_segments) lying mostly on the model's
        hand-bone pixels (`held` bone slots: a sword, a staff) take hand pixels for the rest of
        the region within `held_reach` px of the hand part, or all of it when the region is thin
        (a blade drawn a few px off the model's, over the legs, still swings with the hand).
        Pixels on the same arm (`arm` slots) keep their own."""
    from scipy.ndimage import distance_transform_edt, maximum as nd_max
    H, W = mask.shape
    _, (ny, nx) = distance_transform_edt(~view.mask, return_indices=True)
    if rgb is None or view.colour is None:
        return ny, nx
    from .color import rgb_to_lab
    yy, xx = np.mgrid[0:H, 0:W]
    lab_d = rgb_to_lab(rgb.astype(np.float32))
    lab_m = rgb_to_lab(np.nan_to_num(view.colour).clip(0, 255).astype(np.float32))
    own_de = np.linalg.norm(lab_d - lab_m[ny, nx], axis=-1)
    own_cost = own_de / 10 + 0.3 * np.hypot(ny - yy, nx - xx)
    best, by, bx = own_cost.copy(), ny.copy(), nx.copy()
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            if dy * dy + dx * dx > reach * reach or not (dy or dx):
                continue
            sy, sx = np.clip(yy + dy, 0, H - 1), np.clip(xx + dx, 0, W - 1)
            cost = np.linalg.norm(lab_d - lab_m[sy, sx], axis=-1) / 10 + 0.3 * math.hypot(dy, dx)
            better = view.mask[sy, sx] & (cost < best)
            best = np.where(better, cost, best)
            by, bx = np.where(better, sy, by), np.where(better, sx, bx)
    switch = mask & (own_de > colour_gap) & (best < own_cost - 1.0)
    bone = np.where(view.mask, view.weights.argmax(-1), -1)
    if switch.any():                    # a switch only where neighbours agree (no lone pixel
        new = np.where(switch, bone[by, bx], bone[ny, nx])   # on another bone: it would tear)
        same = np.zeros((H, W), np.int32)
        for dst, src in _shifts(H, W):
            same[dst] += mask[src] & (new[src] == new[dst])
        switch &= same >= 2
    ny, nx = np.where(switch, by, ny), np.where(switch, bx, nx)
    if len(held):
        bone = np.where(view.mask, view.weights.argmax(-1), -1)
        hand = view.mask & np.isin(bone, list(held))
        if hand.any():
            far, (hy, hx) = distance_transform_edt(~hand, return_indices=True)
            seg = colour_segments(rgb, mask)
            src_bone = bone[ny, nx]
            on_hand = mask & np.isin(src_bone, list(held))
            keep = on_hand | np.isin(src_bone, list(arm))
            ids, inv = np.unique(seg[mask], return_inverse=True)
            share = np.bincount(inv, on_hand[mask]) / np.maximum(np.bincount(inv), 1)
            size = np.bincount(inv)
            edge = np.zeros((H, W), bool)
            for dst, src in _shifts(H, W):
                edge[dst] |= seg[src] != seg[dst]
            inner = distance_transform_edt(~(edge | ~mask))
            thick = np.asarray(nd_max(inner, seg, ids)) if len(ids) else np.zeros(0)
            grab = (share >= 0.35) & (size >= 3)
            seg_grab = np.zeros((H, W), bool)
            seg_thin = np.zeros((H, W), bool)
            seg_grab[mask] = grab[inv]
            seg_thin[mask] = (thick <= 1.0)[inv]
            move = seg_grab & ~keep & ((far <= held_reach) | seg_thin)
            ny, nx = np.where(move, hy, ny), np.where(move, hx, nx)
    return ny, nx


def glue(body: Body, view: RestView, cam: Camera, rest: Pose, indices: np.ndarray,
         palette: np.ndarray, rgb: np.ndarray | None = None, held=(), arm=()) -> Drawn:
    """Each drawn pixel takes the 3D point, normal and weights of a model pixel (source_pixels:
    its own; with the drawing's rgb, colour- and held-part-matched), moved in the screen plane to
    its own place (so the rest pose lands every pixel exactly where it was drawn)."""
    ys, xs = np.nonzero(indices >= 0)
    if not view.mask.any():
        raise ValueError("The model is not in the frame")
    ny, nx = source_pixels(view, rgb, indices >= 0, held, arm)
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


def bend(drawn: Drawn, body: Body, pose: Pose, *, fill: bool = True, cam: Camera | None = None):
    """The key drawing in one pose -> (indices (H, W), camera-space normals (H, W, 3)).
    cam: see it with another direction's camera (a body turned toward that camera, e.g. lying
    down, shows this drawing's face to it; its pixel squares are the same size)."""
    cam = cam or drawn.cam
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
                 out: int = 1024, blocky: bool = False, cloud=None,
                 view: "RestView | None" = None, outline: bool = False,
                 crisp: float = 0.0) -> Image.Image:
    """The rest pose big and nearly flat, as the SDXL init: the frame square `region` (x0, y0,
    size in frame px) at out x out, white background. blocky: already on the frame's pixel grid
    (each frame pixel an out/size block, hard edges), so SDXL draws pixels on that grid and the
    drawing comes back to the frame without resampling (crisper than a smooth init).
    crisp: local contrast on the frame grid (unsharp mask amount): the model's far side is
    painted by a blurred projection, so back views start soft without it."""
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
        if crisp > 0:
            on = (cov >= 0.5)[..., None]
            blur = cv2.GaussianBlur(np.where(on, rgb, 0), (0, 0), 1.0)
            wsum = cv2.GaussianBlur(on.astype(np.float32)[..., 0], (0, 0), 1.0)[..., None]
            rgb = rgb + crisp * (rgb - blur / np.maximum(wsum, 1e-6))
        small = np.dstack([np.clip(rgb, 0, 255), np.where(cov >= 0.5, 255, 0)]).astype(np.uint8)
        if outline:
            small = _outlined(small, view, region)
        img = Image.fromarray(small, "RGBA").resize((out, out), Image.NEAREST)
    else:
        img = Image.fromarray(rgba, "RGBA").resize((out, out), Image.LANCZOS)
    canvas = Image.new("RGBA", (out, out), (255, 255, 255, 255))
    canvas.alpha_composite(img)
    return canvas.convert("RGB")


def _outlined(small: np.ndarray, view: "RestView | None", region) -> np.ndarray:
    """A pixel-art style init: dark edge pixels on the silhouette and where the depth jumps
    (an arm in front of the body), so SDXL reads clean separate shapes."""
    out = small.copy()
    on = out[..., 3] > 0
    pad = np.pad(on, 1)
    edge = on & ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
    if view is not None:
        x0, y0, size = (int(v) for v in region)
        H, W = view.mask.shape
        d = np.full((size, size), np.inf)
        ys, xs = np.mgrid[0:size, 0:size]
        fy, fx = ys + y0, xs + x0
        ok = (fy >= 0) & (fy < H) & (fx >= 0) & (fx < W)
        d[ys[ok], xs[ok]] = np.where(view.mask[fy[ok], fx[ok]], view.depth[fy[ok], fx[ok]], np.inf)
        far = np.zeros_like(on)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = np.roll(d, (dy, dx), (0, 1))
            both = np.isfinite(d) & np.isfinite(nb)
            far |= both & (np.where(both, d - np.where(both, nb, 0), 0) > 4.0)   # behind a nearer part
        edge |= far & on
    rgb = out[..., :3].astype(np.float32)
    rgb[edge] = rgb[edge] * 0.3 + np.array([20, 16, 28]) * 0.7
    out[..., :3] = rgb.astype(np.uint8)
    return out


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


def thin_cover(drawn: np.ndarray, model: np.ndarray) -> float:
    """Share of the model's thin parts (a staff, a blade, a tail tip: what a 3x3 opening takes
    off its silhouette) that the drawing covers within a pixel; 1 when it has none. IoU barely
    notices a staff left out, this does."""
    import cv2
    m = model.astype(np.uint8)
    thin = (m > 0) & (cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) == 0)
    if thin.sum() < 6:
        return 1.0
    near = cv2.dilate(drawn.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return float((thin & near).sum()) / float(thin.sum())


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum()) / max(1, int((a | b).sum()))


def colour_histogram(rgb: np.ndarray, mask: np.ndarray, bins: int = 8) -> np.ndarray:
    """Normalised Lab histogram of the masked pixels (L 0-100, a/b -48..48)."""
    from .color import rgb_to_lab
    if not mask.any():
        return np.zeros(bins ** 3)
    lab = rgb_to_lab(rgb[mask].astype(np.float32))
    lab = np.clip(lab, (0, -47.9, -47.9), (99.9, 47.9, 47.9))     # saturated colours: edge bins
    h, _ = np.histogramdd(lab, bins=(bins, bins, bins),
                          range=((0, 100), (-48, 48), (-48, 48)))
    return h.reshape(-1) / max(1.0, h.sum())


def colour_distance(a_rgb, a_mask, b_rgb, b_mask) -> float:
    """0 = same colours in the same amounts, 1 = nothing in common (total variation of the Lab
    histograms): how far a key drawing's colours drift from the reference's."""
    return 0.5 * float(np.abs(colour_histogram(a_rgb, a_mask) - colour_histogram(b_rgb, b_mask)).sum())


def sharpness(rgb: np.ndarray, mask: np.ndarray) -> float:
    """Mean lightness step between neighbouring pixels inside the sprite: crisp, readable
    shapes score high, soft muddy drawings low."""
    from .color import rgb_to_lab
    L = rgb_to_lab(rgb.astype(np.float32))[..., 0]
    sx = mask[:, 1:] & mask[:, :-1]
    sy = mask[1:] & mask[:-1]
    steps = np.concatenate([np.abs(np.diff(L, axis=1))[sx], np.abs(np.diff(L, axis=0))[sy]])
    return float(steps.mean()) if len(steps) else 0.0


def trim_outside(prep, model_mask: np.ndarray, reach: int = 2):
    """Remove drawn pixels more than `reach` px outside the model's silhouette (a shield or a
    plume the model doesn't have would hang off nothing). Returns (prep, share removed)."""
    import cv2
    near = cv2.dilate(model_mask.astype(np.uint8), np.ones((2 * reach + 1,) * 2, np.uint8)) > 0
    out = prep.mask & ~near
    share = float(out.sum()) / max(1, int(prep.mask.sum()))
    prep.mask = prep.mask & near
    return prep, share


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
    report: dict = field(default_factory=dict)   # per direction: chosen candidate + scores


REFERENCE = "S"                    # drawn first; every other direction copies its design
# (init: o = outlined / k = outlined + local contrast, denoise[, seed offset]) tried per view;
# on the knight the outlined init won every direction over a plain one (sharper). Back views
# (the model's far side is a blurred projection of the picture) get a third, crisper try.
CANDIDATES = (("o", 0.6), ("o", 0.72))
BACK_CANDIDATES = (("k", 0.6, 1), ("o", 0.6), ("o", 0.72))
# a try this good is kept without drawing the others (IoU, colour distance, sharpness vs the
# reference: back views are softer by nature): ~half the GPU time of trying every candidate
GOOD = {("person", "front"): (0.88, 0.36, 0.85), ("person", "back"): (0.88, 0.30, 0.60),
        # animals: small frames (a wolf is ~30 px tall) cost IoU at every edge
        ("animal", "front"): (0.80, 0.42, 0.80), ("animal", "back"): (0.78, 0.42, 0.60)}
# the view the others copy: people from the front; animals from the side (a wolf's front view
# at sprite size is a blob; its profile carries the design)
REFERENCES = {"person": "S", "animal": "E"}
SHARP_WEIGHT = 0.8                 # score = IoU + w * sharpness vs reference - colour - off-model


class Cancelled(Exception):
    """The user stopped drawing; drawings made so far stay in the keys folder (resume)."""


def available() -> bool:
    """Can this PC draw characters (a local ComfyUI is installed or answering)? PIXEL_DRAWN=0/1
    forces it (tests: 0)."""
    import os
    env = os.environ.get("PIXEL_DRAWN")
    if env is not None:
        return env.strip().lower() not in ("0", "false", "no", "off", "")
    try:
        from .generate import ComfyClient, find_comfyui
        if find_comfyui() is not None:
            return True
        ComfyClient().ping()
        return True
    except Exception:  # noqa: BLE001 - not installed, not running
        return False


def _panel(sprite_rgba: np.ndarray, mask: np.ndarray, y0: int, size: int, k: int) -> Image.Image:
    """A reference key drawing (frame pixels) as a size x size panel centred on it, at the
    target's rows (same baseline), each pixel a k x k block on white."""
    ys, xs = np.nonzero(mask)
    cx = (xs.min() + xs.max() + 1) / 2
    x0 = int(round(cx - size / 2))
    H, W = mask.shape
    out = np.full((size, size, 3), 255, np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    fy, fx = yy + y0, xx + x0
    ok = (fy >= 0) & (fy < H) & (fx >= 0) & (fx < W)
    on = np.zeros((size, size), bool)
    on[yy[ok], xx[ok]] = mask[fy[ok], fx[ok]]
    out[on] = sprite_rgba[(yy + y0)[on], (xx + x0)[on], :3]
    return Image.fromarray(out).resize((size * k, size * k), Image.NEAREST)


def _side_reference(angle: float) -> str:
    """For a back-facing view: the side view nearest it (E for the right half, W for the left)."""
    return "E" if (angle >= 270.0 or angle < 90.0) else "W"


def _is_back(angle: float) -> bool:
    return 180.0 < angle < 360.0


def mirror_frame(arr: np.ndarray, ox: float) -> np.ndarray:
    """A frame-sized array mirrored about the camera's vertical axis (screen x = ox): what the
    twin direction (angle -> 180 - angle) sees of a left/right-symmetric model."""
    W = arr.shape[1]
    return np.roll(np.fliplr(arr), int(round(2 * ox - W)), 1)


def twin_index(cams: list[Camera], k: int) -> int | None:
    """The direction that sees the model mirrored (E <-> W, SE <-> SW, NE <-> NW...)."""
    n = len(cams)
    a = (180.0 - cams[k].angle) % 360.0
    j = int(round(a / (360.0 / n))) % n
    return j if j != k else None


def symmetric_pair(views: list[RestView], cams: list[Camera], k: int, j: int,
                   min_iou: float = 0.93, max_colour: float = 0.12) -> bool:
    """Does direction k look like direction j mirrored (silhouette and the model's colours)?
    A sword in one hand or a cape over one shoulder fails it, so only symmetric models mirror."""
    a, b = views[k], views[j]
    flip = mirror_frame(b.mask, cams[k].ox)
    if iou(flip, a.mask) < min_iou:
        return False
    col = mirror_frame(np.nan_to_num(b.colour), cams[k].ox).clip(0, 255).astype(np.uint8)
    return colour_distance(np.nan_to_num(a.colour).clip(0, 255).astype(np.uint8), a.mask,
                           col, flip) <= max_colour


def neutral_tint(rgb: np.ndarray, mask: np.ndarray, view: RestView, *, chroma: float = 8.0,
                 reach: float = 20.0, least: float = 0.08, keep: float = 0.2):
    """Take out the colour cast the image model puts on neutral things (steel drawn lavender by
    "hue-shifted shading"): the median a/b of the drawing where the MODEL is grey, minus the
    model's own, is removed from the drawing's near-grey pixels (fading out by chroma 'reach'),
    so shading's relative hue shifts and saturated parts (a red cape) stay as drawn. `keep` of
    the cast stays (the house style's cool shadows); the weights are smoothed over neighbours so
    one shaded surface shifts as one (per-pixel weights split ramps into speckle).
    Returns (rgb, (da, db) removed)."""
    from .color import lab_to_rgb, rgb_to_lab
    if view.colour is None:
        return rgb, (0.0, 0.0)
    m = mask & view.mask
    if m.sum() < 20:
        return rgb, (0.0, 0.0)
    lm = rgb_to_lab(np.nan_to_num(view.colour[m]).astype(np.float32))
    neu = np.hypot(lm[:, 1], lm[:, 2]) < chroma
    if neu.sum() < max(20, least * m.sum()):
        return rgb, (0.0, 0.0)
    ld = rgb_to_lab(rgb[m].astype(np.float32))
    # a near-grey model is grey: half its own faint tint (red bleeding off a cape) is the target
    tint = (1 - keep) * (np.median(ld[neu, 1:], 0) - 0.5 * np.median(lm[neu, 1:], 0))
    if float(np.hypot(*tint)) < 2.0:
        return rgb, (0.0, 0.0)
    import cv2
    lab = rgb_to_lab(rgb.astype(np.float32))
    ab = lab[..., 1:] - tint
    w = np.where(mask, np.clip((reach - np.hypot(ab[..., 0], ab[..., 1])) / (reach / 2), 0, 1), 0)
    on = (w > 0).astype(np.float32)                    # saturated pixels stay exactly as drawn
    wsum = cv2.GaussianBlur(on, (0, 0), 1.0)
    w = np.where(w > 0, cv2.GaussianBlur(w.astype(np.float32), (0, 0), 1.0)
                 / np.maximum(wsum, 1e-6), 0)
    lab[..., 1:] -= tint[None, None] * w[..., None]
    out = rgb.copy()
    out[mask] = lab_to_rgb(lab[mask])
    return out, (float(tint[0]), float(tint[1]))


def _gen_name(d: str, cand) -> str:
    tag, den = cand[0], cand[1]
    off = cand[2] if len(cand) > 2 else 0
    return f"drawn_{d}_{tag}{int(round(den * 100))}" + (f"_s{off}" if off else "") + ".png"


def draw_order(names: list[str], cams: list[Camera], reference: str = REFERENCE) -> list[int]:
    """Reference first, then the sides, the back, the front diagonals, the back diagonals: each
    view is drawn after the views it copies."""
    ref = reference if reference in names else names[0]

    def key(k):
        d, a = names[k], cams[k].angle
        if d == ref:
            return (0, k)
        if d in ("E", "W"):
            return (1, k)
        if d == "N":
            return (2, k)
        return (4 if _is_back(a) else 3, k)
    return sorted(range(len(names)), key=key)


def view_refs(d: str, angle: float, done, reference: str = REFERENCE) -> list[str]:
    """The drawn views a direction is painted beside: front views beside the reference (S);
    back views beside the back (N; N itself beside S) and the nearest side view."""
    if d == reference:
        return []
    if not _is_back(angle):
        return [reference]
    refs = ["N" if d != "N" and "N" in done else reference]
    side = _side_reference(angle)
    if side in refs:                          # the reference is that side: the other one
        side = "W" if side == "E" else "E"
    if side in done and side != d and side not in refs:
        refs.append(side)
    return refs


def draw_keys_paired(rig: Rig, body: Body, cams: list[Camera], views: list[RestView], rest: Pose,
                     prompt: str, settings, *, seed: int = 1234, style=None, client=None,
                     folder: Path | None = None, reuse: bool = True, candidates=CANDIDATES,
                     back_candidates=BACK_CANDIDATES, reference: str | None = None,
                     drop_held: bool = True, absent=(), kind: str = "person",
                     seeds: dict | None = None, mirror: bool | None = None, never_mirror=(),
                     tint: bool = True, palette=None, cancel=None, gpu=None,
                     progress=None) -> KeySet:
    """Key drawings that agree with each other: the reference direction (S) is drawn first over
    the model (img2img on the frame grid); every other direction is painted on a canvas
    [reference(s) | this direction's model render] where only the right panel is repainted
    (inpaint), so the left panels act as the design sheet (front views beside S, the back beside
    S + a side, back diagonals beside the back + a side).
    Each direction tries `candidates` (back views `back_candidates`: (init, denoise[, seed
    offset])) and keeps the best by silhouette IoU, sharpness, colour distance to its first
    reference and pixels drawn off the model. Neutral colours are corrected toward the model's
    (neutral_tint). mirror: None = a direction whose twin (E <-> W) is drawn already is mirrored
    from it when the model looks the same mirrored (symmetric_pair); True/False = always/never.
    absent: things the model doesn't have (e.g. "shield"): kept out by the negative prompt.
    seeds: per direction seed (a redrawn direction gets a new one). cancel(): True = stop
    (raises Cancelled; finished drawings stay for a resume). gpu(what): a context manager held
    around each drawing (e.g. a lock shared with other GPU users). palette: keep this one (a
    redrawn direction is snapped to the palette the others were made with, so they stay as
    they were)."""
    from contextlib import nullcontext

    from .assets import project_style, with_style
    from .generate import generate_images
    from .pipeline import Prepared, finish, resolve_palette
    from .pixelclean import clean_sprite
    from .settings import Settings
    say = _say(progress)
    n = len(cams)
    names = direction_names(n)
    W, H = cams[0].width, cams[0].height
    st = style if style is not None else project_style()
    # held things the model lacks leave the words (a "sword and shield" knight without one);
    # the ones it has stay: a thin staff only in the init picture got dropped by SDXL
    words = model_words(prompt, tuple(absent) or HELD) if drop_held else prompt
    seeds = dict(seeds or {})
    folder = Path(folder) if folder else None
    if folder:
        folder.mkdir(parents=True, exist_ok=True)
    gpu = gpu or (lambda what: nullcontext())
    s_plain = Settings(**{**settings.to_dict(), "outline": False})
    outline_style = settings.outline_style if settings.outline else "none"
    preps: dict = {}
    keyrgba: dict = {}
    report: dict = {}
    cloud = None
    reference = reference or REFERENCES.get(kind, REFERENCE)
    ref = reference if reference in names else names[0]
    order = draw_order(names, cams, ref)
    same = SAME_WORDS.get(kind, SAME_WORDS["person"])

    def cands(k):
        return back_candidates if _is_back(cams[k].angle) and back_candidates else candidates

    # how many drawings are left (for the ETA): every candidate file not made yet
    left = sum(1 for k in order for c in cands(k)
               if not (folder and reuse and (folder / _gen_name(names[k], c)).exists()))
    made, took = 0, []                 # seconds per drawing (median: the first loads models)
    for k in order:
        d = names[k]
        a = cams[k].angle
        j = twin_index(cams, k)
        if (j is not None and names[j] in preps and mirror is not False
                and d not in never_mirror and (mirror or symmetric_pair(views, cams, k, j))):
            src = preps[names[j]]
            prep = Prepared(mirror_frame(src.rgb, cams[k].ox).copy(),
                            mirror_frame(src.mask, cams[k].ox).copy())
            preps[d] = prep
            report[d] = dict(candidate=f"mirror of {names[j]}", iou=iou(prep.mask, views[k].mask),
                             sharp=sharpness(prep.rgb, prep.mask), colour_distance=0.0,
                             off_model=0.0, score=0.0, refs=[names[j]], mirrored=True,
                             seed=report[names[j]].get("seed", seed))
            keyrgba[d] = clean_sprite(finish(prep, s_plain, None), outline_style).rgba
            left -= sum(1 for c in cands(k)
                        if not (folder and reuse and (folder / _gen_name(d, c)).exists()))
            say(f"{d}: mirrored from {names[j]} (the model is the same both ways)")
            continue
        refs = view_refs(d, a, keyrgba, ref)
        panels = len(refs) + 1
        target = 1024 if panels == 1 else (768 if panels == 2 else 640)
        region, kpx = grid_region(views[k], target)
        P = region[2] * kpx
        view_words, view_neg = direction_words(d)
        if panels == 1:
            text, neg = with_style(KEY_PROMPT.format(prompt=words, view=view_words),
                                   f"{KEY_NEGATIVE}, {view_neg}", st, "character")
        else:
            left_words = ", ".join(f"{'left' if i == 0 else 'middle'}: {PANEL_WORDS.get(r, r)}"
                                   for i, r in enumerate(refs))
            text, neg = with_style(
                PAIR_PROMPT.format(n="two" if panels == 2 else "three", prompt=words, same=same,
                                   views=f"{left_words}, right: {view_words}"),
                f"{PAIR_NEGATIVE}, right: {view_neg}", st, "character")
        if absent:
            neg = f"{neg}, {', '.join(absent)}"
        s_d = int(seeds.get(d, seed))
        # colours compared with the view it copies most; the back with the side (S would reward
        # a back view drawn as a second front)
        cref = refs[-1] if d == "N" and len(refs) > 1 else (refs[0] if refs else None)
        good = GOOD[(kind if kind in ("animal",) else "person",
                     "back" if _is_back(a) else "front")]
        best = None
        for cand in cands(k):
            tag, den = cand[0], cand[1]
            s_c = s_d + (cand[2] if len(cand) > 2 else 0)
            path = folder / _gen_name(d, cand) if folder else None
            if path is not None and reuse and path.exists():
                drawing = Image.open(path).convert("RGB")
            else:
                if cancel is not None and cancel():
                    raise Cancelled(f"Stopped before drawing {d}; {len(preps)} directions done "
                                    "(they are kept: drawing again carries on from here)")
                if cloud is None:
                    cloud = surface_cloud(rig, rest)
                init = init_picture(rig, cams[k], region, rest, out=P, blocky=True, cloud=cloud,
                                    view=views[k], outline=tag in ("o", "k"),
                                    crisp=0.8 if tag == "k" else 0.0)
                canvas = Image.new("RGB", (P * panels, P), "white")
                for i, r in enumerate(refs):
                    canvas.paste(_panel(keyrgba[r], keyrgba[r][..., 3] > 0, region[1], region[2],
                                        kpx), (i * P, 0))
                canvas.paste(init, ((panels - 1) * P, 0))
                if folder:
                    canvas.save(folder / f"init_{d}_{tag}.png")
                mask = None
                if panels > 1:
                    mask = np.zeros((P, P * panels), np.float32)
                    mask[:, (panels - 1) * P:] = 1
                if client is None:
                    pace(say=say)
                eta = ""
                if made:
                    eta = f", about {max(1, round(float(np.median(took)) * left / 60))} min left"
                say(f"Drawing it facing {d} ({'beside ' + ' + '.join(refs) if refs else 'reference'}"
                    f", try {tag}{den}; {made + 1} of {made + left}{eta})…")
                t1 = time.time()
                with gpu(f"hybrid: {d} {tag}{den}"):
                    (drawing, _), = generate_images(text, neg, client=client, seed=s_c, count=1,
                                                    init_image=canvas, denoise=den, mask=mask,
                                                    width=P * panels, height=P)
                took.append(time.time() - t1)
                made += 1
                left -= 1
                drawing = drawing.crop(((panels - 1) * P, 0, panels * P, P))
                if path is not None:
                    drawing.save(path)
            prep = drawing_to_frame(drawing, region, (W, H), settings, kpx)
            prep, off = trim_outside(prep, views[k].mask)
            cast = (0.0, 0.0)
            if tint:
                prep.rgb, cast = neutral_tint(prep.rgb, prep.mask, views[k])
            sc_iou = iou(prep.mask, views[k].mask)
            thin = thin_cover(prep.mask, views[k].mask)
            sharp = sharpness(prep.rgb, prep.mask)
            dist = (colour_distance(prep.rgb, prep.mask, preps[cref].rgb, preps[cref].mask)
                    if cref in preps else 0.0)
            ref_sharp = sharpness(preps[ref].rgb, preps[ref].mask) if ref in preps else sharp
            score = (sc_iou + SHARP_WEIGHT * min(sharp / max(ref_sharp, 1e-6), 1.2) - dist
                     - 2 * off - (1.0 - thin))
            say(f"{d} {tag}{den}: IoU {sc_iou:.2f}, sharp {sharp:.1f}, colour distance "
                f"{dist:.2f}, off-model {off:.1%}, thin parts {thin:.0%} -> {score:.2f}")
            if best is None or score > best[0]:
                best = (score, prep, dict(candidate=f"{tag}{den}" + (f"+{cand[2]}" if len(cand) > 2
                                                                     else ""),
                                          iou=sc_iou, sharp=sharp, colour_distance=dist,
                                          thin=thin,
                                          off_model=off, score=score, refs=refs, seed=s_d,
                                          tint=[round(cast[0], 2), round(cast[1], 2)]))
            if (sc_iou >= good[0] and dist <= good[1] and sharp >= good[2] * ref_sharp
                    and off < 0.02 and thin >= 0.7 and cand is not cands(k)[-1]):
                say(f"{d}: {tag}{den} is good, keeping it")
                left -= sum(1 for c in cands(k)[cands(k).index(cand) + 1:]
                            if not (folder and reuse and (folder / _gen_name(d, c)).exists()))
                break
        preps[d] = best[1]
        report[d] = best[2]
        keyrgba[d] = clean_sprite(finish(preps[d], s_plain, None), outline_style).rgba
    order_n = list(names)
    if palette is None:
        pixels = np.concatenate([preps[d].rgb[preps[d].mask] for d in order_n
                                 if preps[d].mask.any()])
        palette = resolve_palette(s_plain, pixels)
    palette = np.asarray(palette, np.uint8)
    sprites = [clean_sprite(finish(preps[d], s_plain, palette), outline_style) for d in order_n]
    refsp = sprites[order_n.index(ref)]
    for d, sp in zip(order_n, sprites):             # final numbers, on the shared palette
        report[d]["colour_distance_final"] = colour_distance(
            sp.rgba[..., :3], sp.indices >= 0, refsp.rgba[..., :3], refsp.indices >= 0)
    return KeySet(order_n, sprites, [report[d]["iou"] for d in order_n],
                  [int(report[d].get("seed", seed)) for d in order_n],
                  [bool(report[d].get("mirrored")) for d in order_n], report)


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
    direction that sees it (else from the nearest point that has one). Other directions' edge
    pixels (the outline) come last: a surface seen only edge-on there (a belly, the soles) would
    come out outline-dark when it turns into view (lying down)."""
    from scipy.spatial import cKDTree
    n = len(views)
    seen, inner = [], []
    for v, sp in zip(views, keys.sprites):
        on = sp.indices >= 0
        pad = np.pad(on, 1)
        edge = on & ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
        flat = sp.indices.reshape(-1)
        idx = np.full(len(body.bind), -1, np.int32)
        ok = v.point_front & (v.point_pix >= 0)
        idx[ok] = flat[v.point_pix[ok]]
        seen.append(idx)
        inside = idx.copy()
        inside[ok & edge.reshape(-1)[np.maximum(v.point_pix, 0)]] = -1
        inner.append(inside)
    rest_pts, _ = body.pose(key_pose.worlds)
    out = []
    steps = sorted(range(1, n), key=lambda s_: (min(s_, n - s_), s_))
    for k in range(n):
        idx = seen[k].copy()
        for pool in (inner, seen):
            for step in steps:
                j = (k + step) % n
                need = idx < 0
                idx[need] = pool[j][need]
        missing = idx < 0
        if missing.any() and (~missing).any():
            tree = cKDTree(rest_pts[~missing])
            _, near = tree.query(rest_pts[missing])
            idx[missing] = idx[~missing][near]
        out.append(idx)
    return out


def has_shield(body: Body, pose: Pose, least: float = 0.09) -> bool:
    """Does the model hold a shield? The forearm + hand part seen from the front and the side
    (front surface, 128 px per body height): a shield is a broad plate there (widest inscribed
    circle >= `least` of the height; a bare or gauntleted arm, a sword or a staff are under ~0.07,
    the library paladin's shield 0.12)."""
    from scipy.ndimage import distance_transform_edt
    names = bone_names(body)
    arm = [i for i, n in enumerate(names) if n.startswith(("forearm", "hand"))]
    if not arm:
        return False
    pts, _ = body.pose(pose.worlds)
    tall = float(np.ptp(pts[:, 1])) or 1.0
    sel = np.isin(body.weights.argmax(1), arm)
    size = 128
    best = 0.0
    for ang in (90.0, 0.0, 180.0):
        cam = Camera(ang, 0.0, size / tall, size / 2, 0.0, size, size + 16)
        sx, sy, d = cam.project(pts - pose.shift)
        sy = sy + 8 + float(pts[:, 1].max()) * cam.scale
        ix, iy = np.floor(sx).astype(np.int64), np.floor(sy).astype(np.int64)
        ok = (ix >= 0) & (ix < size) & (iy >= 0) & (iy < size + 16)
        lin = iy * size + ix
        zmin = np.full(size * (size + 16), np.inf)
        np.minimum.at(zmin, lin[ok], d[ok])
        front = ok & (d <= zmin[np.where(ok, lin, 0)] + 0.02 * cam.scale) & sel
        m = np.zeros(size * (size + 16), bool)
        m[lin[front]] = True
        best = max(best, float(distance_transform_edt(m.reshape(size + 16, size)).max()) / size)
    return best >= least


def auto_absent(prompt: str, kind: str, body: Body, pose: Pose) -> tuple[str, ...]:
    """Things the image model would add that this model doesn't have, for the negative prompt.
    People: a shield unless the model holds one (whatever the words say: the picture the model
    was built from may have none). Animals: saddles, riders, collars... unless the words ask."""
    words = prompt.lower()
    if kind == "animal":
        return tuple(w for w in ANIMAL_ABSENT if w not in words)
    return () if has_shield(body, pose) else SHIELD_WORDS


def facing_choice(key: Pose, pose: Pose, cams: list[Camera], k: int, node: int):
    """For a pose whose body has turned (lying down, spinning): the direction j whose drawing
    now faces camera k best -> (j, cosine). Drawing j shows the side of the body that faced
    camera j at the key pose; after the torso's turn R that side faces R @ toward_j."""
    def rot(w):
        m = w[node][:3, :3]
        return m / np.maximum(np.linalg.norm(m, axis=0, keepdims=True), 1e-12)
    R = rot(pose.worlds) @ rot(key.worlds).T
    toward = cams[k].axes()[2]
    scores = [float(toward @ (R @ c.axes()[2])) for c in cams]
    j = int(np.argmax(scores))
    return j, scores[j]


def _held_slots(body: Body):
    """(held, arm) bone slots: the end of each arm (forearm + hand: auto-rigged weapon arms
    often run the hand out along the blade, so the blade is mostly forearm) and the upper arms."""
    names = bone_names(body)
    held = [i for i, n in enumerate(names) if n.startswith(("hand", "forearm"))]
    arm = [i for i, n in enumerate(names) if n.startswith("upper_arm")]
    return held, arm


def recolour_palette(palette: np.ndarray, pairs) -> np.ndarray:
    """A re-skin's colour swaps ([[old hex, new hex], ...]) applied to the drawings' palette
    (each palette colour within 4 Lab units of an old colour becomes its new one)."""
    if not pairs:
        return palette
    from .color import rgb_to_lab
    old = np.array([[int(o[i:i + 2], 16) for i in (0, 2, 4)] for o, _ in pairs], np.float32)
    new = np.array([[int(nw[i:i + 2], 16) for i in (0, 2, 4)] for _, nw in pairs], np.uint8)
    lp, lo = rgb_to_lab(palette.astype(np.float32)), rgb_to_lab(old)
    d = np.linalg.norm(lp[:, None] - lo[None], axis=-1)
    near = d.argmin(1)
    out = palette.copy()
    hit = d[np.arange(len(palette)), near] < 4.0
    out[hit] = new[near[hit]]
    return out


def _check(cancel, what: str):
    if cancel is not None and cancel():
        raise Cancelled(what)


def drawn_animation(rigged, prompt: str, settings, *, directions: int = 8, frames: int = 8,
                    animations: list[str] | None = None, elevation: float = 30.0,
                    seed: int = 1234, denoise: float = 0.6, style=None, client=None,
                    folder=None, reuse: bool = True, points: int = 400_000, clean: bool = True,
                    normals: bool = True, mirror: bool | None = None, key_pose: str = "idle",
                    blocky: bool = True, retries: int = 1, min_iou: float = 0.72,
                    paired: bool = True, candidates=CANDIDATES, back_candidates=BACK_CANDIDATES,
                    max_turn: float = 45.0, min_face: float = 0.55, absent=None,
                    kind: str = "person", seeds: dict | None = None, never_mirror=(),
                    recolour=None, palette=None, cancel=None, gpu=None, progress=None
                    ) -> tuple[AnimatedSet, dict]:
    """A rigged character -> AnimatedSet whose every frame is a key drawing bent by the
    skeleton. settings: converter settings at the frame size (width x height, colours,
    outline). absent: None = auto_absent (from the model and the words). Poses whose torso
    turned more than max_turn degrees out of a drawing (lying down, spins) bend the drawing that
    faces the camera then (facing_choice); under min_face (cosine) the model's own surface is
    used, coloured from the drawings (model_frame). recolour: a re-skin's colour swaps.
    Returns (set, info: keys, ious, seeds, seconds)."""
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
    old = {}
    if saved is not None and saved.exists():
        try:
            old = json.loads(saved.read_text(encoding="utf-8"))
        except ValueError:
            old = {}
    if reuse:                                  # keep the scale the keys were drawn at
        cam = old.get("camera") or {}
        if (cam.get("width"), cam.get("height"), cam.get("directions"), cam.get("elevation"),
                cam.get("key_pose")) == (W, H, len(names), elevation, key_pose):
            cams = [Camera(k * 360.0 / len(names), elevation, cam["scale"], cam["ox"], cam["oy"],
                           W, H) for k in range(len(names))]
    if cams is None:
        cams = framing(body, [p for row in poses for p in row] + [rest], names, elevation, W, H)
    camera = {"scale": cams[0].scale, "ox": cams[0].ox, "oy": cams[0].oy, "width": W,
              "height": H, "directions": len(names), "elevation": elevation, "key_pose": key_pose}
    if saved is not None and not saved.exists():   # a cancelled run resumes at the same scale
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_text(json.dumps({"camera": camera, "partial": True}), encoding="utf-8")
    if recolour is None:
        recolour = old.get("recolour")
    pts, nrm = body.pose(rest.worlds)
    views = [rest_view(body, c, rest, pts, nrm) for c in cams]
    if absent is None:
        absent = auto_absent(prompt, kind, body, rest)
        if absent:
            say(f"Keeping out what the model doesn't have: {', '.join(absent)}")
    t_keys = time.time()
    if paired:
        keys = draw_keys_paired(rig, body, cams, views, rest, prompt, settings, seed=seed,
                                style=style, client=client, folder=folder, reuse=reuse,
                                candidates=candidates, back_candidates=back_candidates,
                                absent=absent, kind=kind, seeds=seeds, mirror=mirror,
                                never_mirror=never_mirror, palette=palette, cancel=cancel,
                                gpu=gpu, progress=say)
    else:
        keys = draw_keys(rig, body, cams, views, rest, prompt, settings, seed=seed,
                         denoise=denoise, style=style, client=client, folder=folder, reuse=reuse,
                         mirror=mirror, blocky=blocky, retries=retries, min_iou=min_iou,
                         progress=say)
    t_bend = time.time()
    drawn_palette = [f"{r:02x}{g:02x}{b:02x}" for r, g, b in keys.sprites[0].palette.tolist()]
    palette = recolour_palette(keys.sprites[0].palette, recolour)
    for sp in keys.sprites:
        sp.palette = palette
        on = sp.indices >= 0
        sp.rgba[on, :3] = palette[sp.indices[on]]
    if folder:
        for d, sp in zip(keys.names, keys.sprites):
            Image.fromarray(sp.rgba).save(Path(folder) / f"key_{d}.png")
    colours = _point_colours(body, views, keys, rest)
    outline = settings.outline_style if settings.outline else "none"
    say("Bending the drawings for every frame…")
    sprites = [[[None] * frames for _ in names] for _ in clips]
    torso = _torso_node(rig)
    held, arm = _held_slots(body)
    fallback, refaced = 0, 0
    drawns = []
    for k, cam in enumerate(cams):
        sp = keys.sprites[k]
        drawn = glue(body, views[k], cam, rest, sp.indices, palette, rgb=sp.rgba[..., :3],
                     held=held, arm=arm)
        add_hidden(drawn, body, views[k], rest, colours[k], pts)
        drawns.append(drawn)
    total = len(cams) * sum(len(r) for r in poses)
    done, t_b = 0, time.time()
    for k, cam in enumerate(cams):
        _check(cancel, "Stopped while bending the drawings (the key drawings are kept)")
        if k and done:
            eta = (time.time() - t_b) / done * (total - done)
            say(f"Bending the drawings: facing {names[k]} ({k + 1} of {len(cams)}, "
                f"about {max(1, round(eta))} s left)…")
        for a, row in enumerate(poses):
            for f, pose in enumerate(row):
                done += 1
                j = k
                if torso is not None and turned_away(rest, pose, cam, torso) > max_turn:
                    j, face = facing_choice(rest, pose, cams, k, torso)
                    if face < min_face:
                        j = None
                if j is None:
                    idx, ncam = model_frame(body, pose, cam, colours[k])   # nothing faces it
                    fallback += 1
                else:
                    idx, ncam = bend(drawns[j], body, pose, cam=cam)
                    refaced += j != k
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
            "keys_report": keys.report, "model_frames": fallback, "refaced_frames": refaced,
            "absent": list(absent or ()), "camera": camera, "palette": drawn_palette}
    if recolour:
        info["recolour"] = recolour
    return AnimatedSet(actions, names, sprites), info


def _torso_node(rig: Rig) -> int | None:
    for name in ("chest", "spine", "hips"):
        if name in rig.names:
            return rig.names.index(name)
    return None


def turned_away(key: Pose, pose: Pose, cam: Camera, node: int) -> float:
    """Degrees the torso has turned out of the drawing's plane since the key pose: the larger of
    how far the camera-facing direction and the body's up direction have tipped (a body lying
    down ~90; turning on the spot about the vertical, or a lean in the picture plane, is fine
    for the drawing up to that)."""
    def rot(w):
        m = w[node][:3, :3]
        return m / np.maximum(np.linalg.norm(m, axis=0, keepdims=True), 1e-12)
    R = rot(pose.worlds) @ rot(key.worlds).T
    toward = cam.axes()[2]
    up = np.array([0.0, 1.0, 0.0])
    tilt = math.degrees(math.acos(np.clip(float(toward @ (R @ toward)), -1, 1)))
    lie = math.degrees(math.acos(np.clip(float(up @ (R @ up)), -1, 1)))
    return max(tilt, lie)


def model_frame(body: Body, pose: Pose, cam: Camera, point_idx: np.ndarray, sample: int = 250_000):
    """A frame from the model's own surface coloured with the drawings' palette indices (for
    poses the drawing can't bend into, e.g. lying down): front surface per pixel, majority
    colour, no new colours."""
    W, H = cam.width, cam.height
    sel = np.nonzero(point_idx >= 0)[0]
    if len(sel) > sample:
        sel = sel[:: max(1, len(sel) // sample)]
    pts, nrm = _pose_points(body, pose.mats, sel)
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    sx, sy, z = cam.project(pts - pose.shift)
    ix, iy = np.floor(sx).astype(np.int64), np.floor(sy).astype(np.int64)
    ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    lin = iy[ok] * W + ix[ok]
    z, ci, nrm = z[ok], point_idx[sel][ok], nrm[ok]
    zmin = np.full(W * H, np.inf)
    np.minimum.at(zmin, lin, z)
    front = z <= zmin[lin] + DEPTH_EPS
    lin, ci, nrm = lin[front], ci[front], nrm[front]
    count = np.bincount(lin, minlength=W * H)
    k = int(point_idx.max()) + 2
    votes = np.bincount(lin * k + ci, minlength=W * H * k).reshape(W * H, k)
    out = majority(np.where(count >= 2, votes.argmax(1), -1).astype(np.int32).reshape(H, W),
                   k).reshape(-1)
    nsum = np.stack([np.bincount(lin, nrm[:, j], W * H) for j in range(3)], -1)
    nsum /= np.maximum(np.linalg.norm(nsum, axis=1, keepdims=True), 1e-12)
    return out.reshape(H, W), cam.to_camera(nsum).reshape(H, W, 3)


def majority(idx: np.ndarray, k: int | None = None) -> np.ndarray:
    """Each drawn pixel takes the most common colour of its 3x3 neighbourhood when that colour
    holds at least 4 of the 9 (calms the speckle of colours sampled off a point cloud)."""
    H, W = idx.shape
    k = k or int(idx.max()) + 2
    votes = np.zeros((H, W, k), np.int16)
    pad = np.pad(idx, 1, constant_values=-1)
    yy, xx = np.mgrid[0:H, 0:W]
    for dy in range(3):
        for dx in range(3):
            v = pad[dy:dy + H, dx:dx + W]
            on = v >= 0
            np.add.at(votes, (yy[on], xx[on], v[on]), 1)
    best = votes.argmax(-1)
    keep = (idx >= 0) & (votes.max(-1) >= 4)
    return np.where(keep, best, idx).astype(np.int32)


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


def drawn_settings(kind: str, size: str, st):
    """Converter settings for a library character's frames (the same ones render_sheets uses)."""
    from .library import frame_size
    from .settings import settings_for
    w, h = frame_size(kind, size, st)
    colours = st.colours if kind == "person" else max(8, round(st.colours * 2 / 3))
    s = settings_for(kind if kind == "animal" else "person", width=w, height=h, colors=colours)
    if st.outline == "none":
        s.outline = False
    elif st.outline in ("selout", "solid"):
        s.outline_style = st.outline
    if st.palette != "auto":
        s.palette = st.palette
    return s


def drawn_sheets(character_or_rigged, prompt: str | None = None, *, out_png=None,
                 directions: int | None = None, clips: list[str] | None = None, frames: int = 8,
                 seed: int = 1234, denoise: float = 0.6, kind: str | None = None,
                 blocky: bool = True, candidates=CANDIDATES, back_candidates=BACK_CANDIDATES,
                 absent=None, size: str | None = None, client=None, keys_folder=None,
                 reuse: bool = True, redraw=(), name: str = "", cancel=None, gpu=None,
                 progress=None) -> tuple[AnimatedSet, dict, list[Path]]:
    """A library Character (or a rigged .glb + prompt) -> drawn sheets at the library scale of
    the CURRENT project (frame size, colours, outline, elevation, directions), written with
    anim3d.write_animated (sheet.png/json/aseprite/tres/_facing.gd + _n + _glow) to out_png
    (default: the character's sheet.png). keys_folder: where the key drawings are kept
    (default: <character>/keys); drawings already there are reused (resume after a cancel).
    redraw: directions to draw again with a new seed (one bad direction); reuse=False draws
    every direction again (a new house style). No words: the character's name / kind."""
    from .library import Character, style as lib_style
    st = lib_style()
    if isinstance(character_or_rigged, Character):
        ch = character_or_rigged
        rigged, prompt = ch.rigged, prompt or ch.prompt
        kind, size = kind or ch.kind, size or ch.size
        name = name or ch.name
        out_png = out_png or ch.sheet
        keys_folder = keys_folder or ch.folder / "keys"
    else:
        rigged = Path(character_or_rigged)
        kind, size = kind or "person", size or ("medium" if kind == "animal" else "normal")
        out_png = out_png or rigged.with_name("sheet.png")
        keys_folder = keys_folder or rigged.parent / "keys"
    keys_folder = Path(keys_folder)
    saved = keys_folder / "keys.json"
    old = {}
    if saved.exists():
        try:
            old = json.loads(saved.read_text(encoding="utf-8"))
        except ValueError:
            old = {}
    if not prompt:                        # the words it was drawn from, else its name
        prompt = old.get("prompt") or (f"a {name.lower()}" if name else
                                       ("an animal" if kind == "animal" else "a character"))
    seeds = {d: int(v) for d, v in (old.get("seeds") or {}).items()} if reuse else {}
    bin_ = keys_folder / "old"
    for d in (redraw if reuse else ("",)):   # old drawings put aside (a candidate not tried
        if reuse:                            # this time must not be picked up later)
            seeds[d] = int(seeds.get(d, seed)) + 7919    # a new seed
        if keys_folder.exists():
            bin_.mkdir(exist_ok=True)
            for f in keys_folder.glob(f"drawn_{d}_*.png" if d else "drawn_*.png"):
                f.replace(bin_ / f.name)
    s = drawn_settings(kind, size, st)
    anim, info = drawn_animation(rigged, prompt, s, directions=directions or st.directions,
                                 frames=frames, animations=clips, elevation=st.elevation,
                                 seed=seed, denoise=denoise, style=st, client=client,
                                 folder=keys_folder, reuse=reuse, blocky=blocky,
                                 candidates=candidates, back_candidates=back_candidates,
                                 absent=absent, kind=kind, seeds=seeds, never_mirror=tuple(redraw),
                                 recolour=old.get("recolour"), cancel=cancel, gpu=gpu,
                                 palette=_hex_palette(old.get("palette")) if redraw else None,
                                 progress=progress)
    written = write_animated(anim, out_png, aseprite=True, godot=True, normals=True)
    meta = {k: v for k, v in info.items() if k != "keys"}
    meta["prompt"] = prompt
    saved.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return anim, info, written


def _hex_palette(hexes):
    if not hexes:
        return None
    return np.array([[int(h[i:i + 2], 16) for i in (0, 2, 4)] for h in hexes], np.uint8)
